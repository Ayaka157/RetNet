"""Capability validation across usage modes (CPU inference, no GPU conflict).

Modes: 续写 completion / 填空 cloze / 问答-闲聊 qa / 角色扮演 roleplay /
       结构化输出 structured / 多轮 multi-turn.

Outputs raw results to stdout (and optionally a file) for scoring by hand.

    python scripts/capability_eval.py \
        --checkpoint checkpoints/retnet_100m_rp_v2/model.pt \
        --tokenizer data_cache/chat_pack_v5/bpe_tokenizer.json \
        --tag rp_v2 --out logs/capability_eval/rp_v2.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.chat import encode_prompt  # noqa: E402
from retnet.model import RetNetLM, strip_optimized_prefix  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402

# (mode, [(label, prompt_or_persona_prompt, max_new_tokens)])
BATTERY = [
    ("续写 completion",
     [("中文科普", "人工智能是一种让机器", 110),
      ("中文叙事", "那是一个寒冷的冬夜，他独自走在", 110),
      ("英文历史", "The history of the English language begins", 110),
      ("古诗", "床前明月光，疑是地上霜。举头望明月，", 60)]),
    ("填空 cloze",
     [("常识-首都", "中国的首都是", 24),
      ("科学-化学式", "水的化学式是", 24),
      ("文学-作者", "《红楼梦》的作者是", 24),
      ("英文-首都", "The capital of France is", 24),
      ("地理", "珠穆朗玛峰位于", 24)]),
    ("问答 qa",
     [("事实", "北京有哪些著名的景点？", 80),
      ("菜谱", "番茄炒蛋怎么做？", 80),
      ("数学", "17 加 25 等于多少？", 60),
      ("推理", "如果今天是星期三，那么三天后是星期几？", 60),
      ("闲聊", "你真好看", 60)]),
    ("角色扮演 roleplay",
     [("令狐冲", ("你是令狐冲（出自笑傲江湖）。你性格豪爽不羁，不在意规矩，重情重义，说话直来直去。", "你是谁？"), 80),
      ("猫娘", ("你是猫娘小白，说话可爱亲切，**每句话结尾都要加“喵”**，不要忘记这个口癖。", "你在干什么呀？"), 80),
      ("中医", ("你是一位经验丰富的中医大夫，说话温和稳重，喜欢讲道理。", "我最近老是失眠怎么办？"), 80),
      ("傲娇", ("你性格高冷傲娇，说话简短带刺，嘴上不承认关心别人，**常用“哼”“……”**。", "我今天生病了"), 80)]),
    ("结构化 structured",
     [("列表-茶叶", "列出三种中国名茶，用 1. 2. 3. 的格式。", 80),
      ("JSON", "请用 JSON 格式回答，包含 name、age、hobby 三个字段。", 100),
      ("步骤", "请分三步说明如何煮方便面，每步一行，用步骤一/步骤二/步骤三开头。", 100),
      ("翻译", "把“今天天气很好”翻译成英文。", 60)]),
    ("多轮 multi-turn",
     [("记忆测试", [("我叫小明，很高兴认识你", None), ("我叫什么名字？", None)], 70)]),
]


def fix_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def cut(text: str) -> str:
    return text.split("用户：")[0].split("助手：")[0].strip()


def main():
    fix_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tag", default="cap")
    ap.add_argument("--out", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RetNetLM(ck["config"]).to(torch.device("cpu"))
    model.load_state_dict(strip_optimized_prefix(ck.get("model_state", ck)))
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)
    step = ck.get("step", "?")

    lines = [f"# 能力验证：{args.tag}（step={step}）", "",
             f"- checkpoint: `{args.checkpoint}`",
             f"- tokenizer: `{args.tokenizer}`",
             f"- 采样: T={args.temperature} top_k={args.top_k} top_p={args.top_p} "
             f"rep={args.repetition_penalty} seed={args.seed}", ""]

    def emit(text):
        print(text, flush=True)
        lines.append(text)

    t0 = time.time()
    for mode, items in BATTERY:
        emit(f"## {mode}")
        emit("")
        for item in items:
            if mode.startswith("多轮"):
                label, convo, mn = item
                turns = []
                for user, _ in convo:
                    ids = encode_prompt(tok, turns, user)
                    x = torch.tensor([ids], device="cpu")
                    with torch.inference_mode():
                        out = model.generate(x, max_new_tokens=mn,
                                             temperature=args.temperature,
                                             top_k=args.top_k, top_p=args.top_p,
                                             repetition_penalty=args.repetition_penalty,
                                             no_repeat_ngram_size=4, eos_id=tok.eos_id)
                    r = cut(tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True))
                    emit(f"**{label}**")
                    emit(f"- 用户：{user}")
                    emit(f"- 助手：{r}")
                    turns.append((user, r))
                emit("")
                continue
            label, prompt, mn = item
            if mode.startswith("角色扮演"):
                persona, prompt_text = prompt
                ids = encode_prompt(tok, [], prompt_text, system=persona)
                emit(f"**{label}**")
                emit(f"- 设定：{persona}")
                emit(f"- 用户：{prompt_text}")
            else:
                ids = encode_prompt(tok, [], prompt)
                emit(f"**{label}**")
                emit(f"- 输入：{prompt}")
            x = torch.tensor([ids], device="cpu")
            with torch.inference_mode():
                out = model.generate(x, max_new_tokens=mn,
                                     temperature=args.temperature, top_k=args.top_k,
                                     top_p=args.top_p,
                                     repetition_penalty=args.repetition_penalty,
                                     no_repeat_ngram_size=4, eos_id=tok.eos_id)
            r = tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True)
            r = cut(r)
            emit(f"- 输出：{r}")
            emit("")
    emit(f"_(生成耗时 {time.time() - t0:.0f}s)_")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
