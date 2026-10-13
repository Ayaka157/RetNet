"""Auto-scored capability benchmark for RetNet HF checkpoints.

Replaces the hand-scored 0-5 battery in scripts/eval_cap_hf.py.  Every item
carries a deterministic checker, so the same checkpoint + seed always yields
the same score, and "does not qualify" becomes a measurable statement.

    python scripts/bench_auto.py ^
        --models v10=checkpoints/retnet_272m_v10,rp=checkpoints/retnet_272m_rp ^
        --tokenizer data_cache/chat_pack_v10/bpe_tokenizer.json ^
        --seeds 42,43,44 --out logs/bench

Scoring: each item is 0 or 1.  Category score = mean over items x seeds.
Total = weighted mean over categories (weights in CATEGORIES).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402
from retnet.chat import encode_prompt  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402

# ---------------------------------------------------------------- checkers
# Each checker gets the decoded reply (and the raw reply for think detection)
# and returns True/False.  Keep them dumb and readable.


def c_any(*vals):
    return lambda r, raw: any(v in r for v in vals)


def c_regex(pat):
    rx = re.compile(pat)
    return lambda r, raw: bool(rx.search(r))


def c_len(minlen=0, maxlen=10 ** 9):
    return lambda r, raw: minlen <= len(r.strip()) <= maxlen


def c_hangul(*vals):
    return lambda r, raw: c_any(*vals)(r, raw) and c_len(40)(r, raw)


def c_not_think():
    # reply must not contain an unterminated <think>
    return lambda r, raw: not ("<think>" in raw and "</think>" not in raw)


def c_numbered(n=3):
    def f(r, raw):
        hits = set()
        for line in r.splitlines():
            m = re.match(r"\s*([0-9１-９])[.、)）]", line)
            if m:
                hits.add(m.group(1))
        return len(hits) >= n
    return f


def c_json_keys(*keys):
    def f(r, raw):
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return False
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return False
        return isinstance(obj, dict) and all(k in obj for k in keys)
    return f


def c_ascii_ratio(minratio=0.8, minlen=40):
    def f(r, raw):
        s = r.strip()
        if len(s) < minlen:
            return False
        a = sum(1 for ch in s if ord(ch) < 128)
        return a / max(len(s), 1) >= minratio
    return f


def c_distinct_ngram(minratio=0.65, n=4, minlen=40):
    """Guards against the degenerate repetition loop the 100M models fall into."""
    def f(r, raw):
        s = r.strip()
        if len(s) < minlen:
            return False
        grams = [s[i:i + n] for i in range(len(s) - n + 1)]
        if not grams:
            return False
        return len(set(grams)) / len(grams) >= minratio
    return f


def c_and(*fns):
    return lambda r, raw: all(fn(r, raw) for fn in fns)


# ---------------------------------------------------------------- battery
# (mode, category_weight, id, prompt, system, max_new, checker)
BATTERY = [
    # ---- 续写：没有唯一答案，判"像不像人话"：够长、不循环、不泄 think
    ("续写", 0.8, "cont_cn_sci", "人工智能是一种让机器", None, 110,
     c_and(c_len(40), c_distinct_ngram(), c_not_think())),
    ("续写", 0.8, "cont_cn_narr", "那是一个寒冷的冬夜，他独自走在", None, 110,
     c_and(c_len(20), c_distinct_ngram(0.6), c_not_think())),
    ("续写", 0.8, "cont_en_hist", "The history of the English language begins", None, 110,
     c_and(c_ascii_ratio(0.8, 40), c_distinct_ngram(), c_not_think())),
    ("续写", 0.8, "cont_poem", "床前明月光，疑是地上霜。举头望明月，", None, 60,
     c_and(c_len(10), c_not_think())),

    # ---- 填空：唯一答案，最好判
    ("填空", 1.2, "cloze_capital_cn", "中国的首都是", None, 24, c_any("北京")),
    ("填空", 1.2, "cloze_formula", "水的化学式是", None, 24, c_regex(r"H\s*2\s*O|H₂O")),
    ("填空", 1.2, "cloze_author", "《红楼梦》的作者是", None, 24, c_any("曹雪芹")),
    ("填空", 1.2, "cloze_capital_en", "The capital of France is", None, 24, c_any("Paris")),
    ("填空", 1.2, "cloze_geo", "珠穆朗玛峰位于", None, 24, c_any("喜马拉雅", "中国", "尼泊尔")),

    # ---- 问答
    ("问答", 1.2, "qa_fact", "北京有哪些著名的景点？", None, 80,
     c_any("故宫", "长城", "颐和园", "天安门", "天坛")),
    ("问答", 1.2, "qa_recipe", "番茄炒蛋怎么做？", None, 80, c_any("鸡蛋", "番茄", "炒")),
    ("问答", 1.5, "qa_math", "17 加 25 等于多少？", None, 60, c_regex(r"(?<!\d)42(?!\d)")),
    ("问答", 1.5, "qa_logic", "如果今天是星期三，那么三天后是星期几？", None, 60, c_any("星期六")),
    ("问答", 0.8, "qa_chitchat", "你真好看", None, 60, c_and(c_len(2, 200), c_not_think())),

    # ---- 角色扮演：项目主线，权重最高
    ("角色扮演", 2.0, "rp_linghu", "你是谁？",
     "你是令狐冲（出自笑傲江湖）。你性格豪爽不羁，不在意规矩，重情重义，说话直来直去。", 80,
     c_and(c_any("令狐冲"), c_not_think())),
    ("角色扮演", 2.0, "rp_catgirl", "你在干什么呀？",
     "你是猫娘小白，说话可爱亲切，**每句话结尾都要加“喵”**，不要忘记这个口癖。", 80, c_any("喵")),
    ("角色扮演", 2.0, "rp_doctor", "我最近老是失眠怎么办？",
     "你是一位经验丰富的中医大夫，说话温和稳重，喜欢讲道理。", 80,
     c_and(c_len(20), c_any("失眠", "睡眠", "调理", "中医", "作息"), c_not_think())),
    ("角色扮演", 2.0, "rp_tsundere", "我今天生病了",
     "你性格高冷傲娇，说话简短带刺，嘴上不承认关心别人，**常用“哼”“……”**。", 80,
     c_any("哼", "……")),

    # ---- 结构化 / 指令遵循
    ("结构化", 1.2, "st_list", "列出三种中国名茶，用 1. 2. 3. 的格式。", None, 80, c_numbered(3)),
    ("结构化", 1.2, "st_json", "请用 JSON 格式回答，包含 name、age、hobby 三个字段。", None, 100,
     c_json_keys("name", "age", "hobby")),
    ("结构化", 1.2, "st_steps", "请分三步说明如何煮方便面，每步一行，用步骤一/步骤二/步骤三开头。", None, 100,
     c_and(c_any("步骤一"), c_any("步骤二"), c_any("步骤三"))),
    ("结构化", 1.0, "st_translate", "把“今天天气很好”翻译成英文。", None, 60,
     c_and(c_regex(r"(?i)weather"), c_regex(r"(?i)nice|good|great|fine|beautiful"))),
]

MULTITURN = [
    ("多轮", 2.0, "mt_name",
     [("我叫小明，很高兴认识你", None), ("我叫什么名字？", None)], 70,
     c_any("小明")),
]

CATEGORIES = {}
for mode, w, *_ in BATTERY + MULTITURN:
    CATEGORIES[mode] = w


def cut(text: str) -> str:
    return text.split("用户：")[0].split("助手：")[0].strip()


def run_one(model, tok, seed, item, eos_id):
    mode = item[0]
    torch.manual_seed(seed)
    if mode == "多轮":
        # 多轮项没有 system 槽：(mode, weight, id, convo, max_new, checker)
        _, weight, iid, convo, max_new, checker = item
        turns = []
        reply = ""
        for user, _ in convo:
            ids = encode_prompt(tok, turns, user)
            x = torch.tensor([ids], device="cuda")
            out = model.generate(x, max_new_tokens=max_new, temperature=ARGS.temperature,
                                 top_k=ARGS.top_k, top_p=ARGS.top_p,
                                 repetition_penalty=ARGS.repetition_penalty,
                                 no_repeat_ngram_size=4, eos_id=eos_id)
            reply = cut(tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True))
            turns.append((user, reply))
    else:
        _, weight, iid, prompt, system, max_new, checker = item
        ids = encode_prompt(tok, [], prompt, system=system)
        x = torch.tensor([ids], device="cuda")
        out = model.generate(x, max_new_tokens=max_new, temperature=ARGS.temperature,
                             top_k=ARGS.top_k, top_p=ARGS.top_p,
                             repetition_penalty=ARGS.repetition_penalty,
                             no_repeat_ngram_size=4, eos_id=eos_id)
        reply = cut(tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True))
    ok = bool(checker(reply, reply))
    leak = ("<think>" in reply and "</think>" not in reply)
    return {"id": iid, "mode": mode, "weight": weight, "seed": seed,
            "pass": ok, "think_leak": leak, "reply": reply[:400]}


def main():
    global ARGS
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="comma-separated name=path pairs")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--out", default="logs/bench")
    ARGS = ap.parse_args()

    seeds = [int(s) for s in ARGS.seeds.split(",")]
    tok = BPETokenizer.load(ARGS.tokenizer)
    outdir = Path(ARGS.out)
    outdir.mkdir(parents=True, exist_ok=True)

    specs = []
    for pair in ARGS.models.split(","):
        name, path = pair.split("=", 1)
        specs.append((name.strip(), path.strip()))

    summary = {}
    for name, path in specs:
        print(f"\n{'=' * 70}\n[{name}] loading {path}\n{'=' * 70}", flush=True)
        model = RetNetForCausalLM.from_pretrained(path)
        n_param = sum(p.numel() for p in model.parameters())
        model = model.cuda().eval()
        print(f"[{name}] params={n_param/1e6:.2f}M", flush=True)

        records, t0 = [], time.time()
        for seed in seeds:
            for item in BATTERY + MULTITURN:
                records.append(run_one(model, tok, seed, item, tok.eos_id))
            done = len(records)
            print(f"  [{name}] seed {seed} done  ({done} items, "
                  f"{time.time()-t0:.0f}s)", flush=True)

        # aggregate
        by_mode = {}
        for r in records:
            m = by_mode.setdefault(r["mode"], {"n": 0, "pass": 0})
            m["n"] += 1
            m["pass"] += int(r["pass"])
        for m in by_mode.values():
            m["score"] = m["pass"] / m["n"]

        total_w = sum(CATEGORIES[k] for k in by_mode)
        total = sum(by_mode[k]["score"] * CATEGORIES[k] for k in by_mode) / total_w
        leak_rate = sum(r["think_leak"] for r in records) / len(records)

        summary[name] = {
            "path": path, "params_M": n_param / 1e6,
            "seeds": seeds, "n_items": len(records),
            "total_weighted": total, "think_leak_rate": leak_rate,
            "by_mode": by_mode,
            "records": records,
        }
        (outdir / f"{name}.json").write_text(
            json.dumps(summary[name], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{name}] weighted total = {total:.3f} | think-leak {leak_rate*100:.1f}%",
              flush=True)

        del model
        torch.cuda.empty_cache()

    # ---- combined markdown
    names = [n for n, _ in specs]
    modes = [it[0] for it in BATTERY + MULTITURN]
    seen, ordered = set(), []
    for m in modes:
        if m not in seen:
            seen.add(m)
            ordered.append(m)

    lines = ["# 自动判分基准（bench_auto.py）", "",
             f"- tokenizer: `{ARGS.tokenizer}`",
             f"- seeds: {seeds} · T={ARGS.temperature} top_k={ARGS.top_k} "
             f"top_p={ARGS.top_p} rep={ARGS.repetition_penalty}",
             f"- 每项 0/1 分；类别分 = 项均分；总分 = 类别加权平均", "",
             "| 模型 | 参数 | " + " | ".join(ordered) + " | **加权总分** | think 泄漏 |",
             "| --- | --- | " + " | ".join(["---"] * len(ordered)) + " | --- | --- |"]
    for n in names:
        s = summary[n]
        cells = []
        for m in ordered:
            bm = s["by_mode"].get(m)
            cells.append(f"{bm['score']:.2f}" if bm else "—")
        lines.append(f"| {n} | {s['params_M']:.1f}M | " + " | ".join(cells) +
                     f" | **{s['total_weighted']:.3f}** | {s['think_leak_rate']*100:.1f}% |")
    lines += ["", "类别权重：" + ", ".join(f"{k}={v}" for k, v in CATEGORIES.items())]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nsaved -> {outdir}")


if __name__ == "__main__":
    main()
