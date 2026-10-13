"""Synthesize role-play samples whose 设定 carries an explicit STYLE constraint.

Why: the RP eval showed the model learns *who* a character is (它会自称"我韦小宝")
but ignores *how* the 设定 tells it to speak — "每句话结尾加喵" produced no 喵.
Reason: every persona in the training data is a personality *description*; none is a
format/verbal-tic *instruction*, so the model never learned that meta-skill.

This script rewrites existing dialogue with rule-based style transforms and writes a
matching persona that states the rule, teaching "obey the style written in 设定".

Output: data_cache/roleplay/style_rp.jsonl
    {"system_prompt": "...", "conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}

    python scripts/build_style_rp.py --max_samples 200000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.packing import cjk_ratio, clean, iter_jsonl, iter_lccc, messages_to_pairs  # noqa: E402

Pairs = List[Tuple[str, str]]

SENT_SPLIT = re.compile(r"(?<=[。！？!?…])")
NAMES_F = ["小白", "小樱", "铃铛", "阿狸", "初雪", "小满", "月月", "糖糖", "雪球", "咪咪"]
NAMES_M = ["阿澈", "小虎", "黑豆", "阿岚", "石头", "大牛", "小北", "青云"]


def sentences(text: str) -> List[str]:
    parts = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    return parts or [text.strip()]


# ------------------------------------------------------------------ transforms
def t_tail_word(reply: str, rng: random.Random):
    """句尾语气词（喵/呢/哦/啦/呀/嘛）。"""
    word = rng.choice(["喵", "呢", "哦", "啦", "呀", "嘛", "哒"])
    out = []
    for s in sentences(reply):
        if s and s[-1] in "。！？!?…":
            out.append(s[:-1] + word + s[-1])
        else:
            out.append(s + word)
    role = rng.choice(NAMES_F + NAMES_M)
    persona = (f"你是{role}，说话可爱亲切，"
               f"**每句话结尾都要加“{word}”**，不要忘记这个口癖。")
    return persona, "".join(out)


def t_third_person(reply: str, rng: random.Random):
    """用第三人称自称。"""
    role = rng.choice(NAMES_F)
    out = reply.replace("我们", "大家").replace("我", role)
    persona = (f"你是{role}，性格天真，"
               f"**永远用第三人称称呼自己（说“{role}”而不是“我”）**。")
    return persona, out


def t_short(reply: str, rng: random.Random):
    """极简短句：硬截断到 20 字以内，保证约束严格成立。"""
    first = sentences(reply)[0]
    if len(first) > 18:
        cut = first[:18]
        last_comma = max(cut.rfind("，"), cut.rfind(","))
        if last_comma >= 8:
            first = cut[:last_comma] + "。"
        else:
            first = cut.rstrip("，,、") + "。"
    persona = "你说话极其简短，**每次只回一句话，不超过 20 个字**，绝不长篇大论。"
    return persona, first


def t_cold(reply: str, rng: random.Random):
    """高冷傲娇：短句 + 哼/……（先去掉原文的笑声，否则与人设矛盾）。"""
    reply = re.sub(r"^(哈+|呵+|嘻+|嘿+|啊+)[，,。！!~]*", "", reply).strip()
    first = sentences(reply)[0] if reply else "……"
    if len(first) > 30:
        first = first[:30].rstrip("，,、") + "……"
    head = rng.choice(["哼，", "……", "哼。", ""])
    tail = rng.choice(["……", "，别误会。", "，我才不在乎。", "。"])
    persona = ("你性格高冷傲娇，说话简短带刺，嘴上不承认关心别人，"
               "**常用“哼”“……”，绝不说甜言蜜语**。")
    return persona, f"{head}{first.rstrip('。！？…?!')}{tail}"


def t_action(reply: str, rng: random.Random):
    """动作神态描写。"""
    act = rng.choice(["【轻轻一笑】", "【抬起头看你】", "【叹了口气】", "【眼睛一亮】",
                      "【挠挠头】", "【别过脸去】", "【往前凑了凑】", "【眉头微皱】"])
    persona = "你在回复时**先用【】写出当下的动作或神态，再说话**，让人有画面感。"
    return persona, f"{act}{reply}"


def t_catchphrase(reply: str, rng: random.Random):
    """固定口头禅开场。"""
    phrase = rng.choice(["本大爷", "在下", "老夫", "本小姐", "咱家", "吾"])
    persona = (f"你自称“{phrase}”，**每次开口都要先说“{phrase}”**，语气自信。")
    first = sentences(reply)[0]
    return persona, f"{phrase}告诉你，{first}"


def t_polite_ancient(reply: str, rng: random.Random):
    """文言/古风腔。"""
    out = (reply.replace("你", "阁下").replace("很", "甚")
                .replace("非常", "极为").replace("我", "在下"))
    persona = "你说话用古风文言腔，称对方为“阁下”，自称“在下”，用词文雅。"
    return persona, out


def t_verbose(reply: str, rng: random.Random):
    """详细展开（只挑本来就长的回复）。"""
    persona = "你说话详细耐心，**喜欢把道理展开解释清楚**，通常回答三句以上。"
    return persona, reply


TRANSFORMS: List[Tuple[str, Callable, int]] = [
    ("tail_word", t_tail_word, 3),
    ("third_person", t_third_person, 2),
    ("short", t_short, 2),
    ("cold", t_cold, 2),
    ("action", t_action, 2),
    ("catchphrase", t_catchphrase, 1),
    ("ancient", t_polite_ancient, 1),
    ("verbose", t_verbose, 1),
]
WEIGHTED = [name for name, _, w in TRANSFORMS for _ in range(w)]
BY_NAME = {name: fn for name, fn, _ in TRANSFORMS}


# --------------------------------------------------------------------- sources
def source_pairs(args) -> Iterator[Pairs]:
    """Dialogue pairs to rewrite: LCCC casual + ChatHaruhi + zh GPT4 role-play."""
    rp = Path(args.rp_dir)
    corpus = Path(args.corpus_dir)

    h54 = rp / "haruhi_54k.jsonl"
    if h54.exists():
        for obj in iter_jsonl(h54):
            q = clean(obj.get("user_question"), 200)
            a = clean(obj.get("agent_response"), 300)
            if q and a and cjk_ratio(a) > 0.5:
                yield [(q.strip("「」"), a.strip("「」"))]

    zh4 = rp / "roleplay_zh_gpt4.jsonl"
    if zh4.exists():
        for obj in iter_jsonl(zh4):
            pairs = messages_to_pairs(obj.get("conversations") or [], 300)
            for p in pairs:
                yield [p]

    lccc = corpus / "lccc_base_train.jsonl"
    if lccc.exists():
        for pairs in iter_lccc(lccc, args.lccc_stride, 200, min_turn_chars=10):
            yield pairs[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_cache/roleplay/style_rp.jsonl")
    ap.add_argument("--rp_dir", default="data_cache/roleplay")
    ap.add_argument("--corpus_dir", default="data_cache/chat_corpus")
    ap.add_argument("--max_samples", type=int, default=250_000)
    ap.add_argument("--lccc_stride", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in BY_NAME}
    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for pairs in source_pairs(args):
            if written >= args.max_samples:
                break
            if not pairs:
                continue
            name = rng.choice(WEIGHTED)
            fn = BY_NAME[name]
            # 「verbose」只用长回复，否则名不副实
            if name == "verbose" and len(pairs[0][1]) < 80:
                name, fn = "action", BY_NAME["action"]
            try:
                persona, new_reply = fn(pairs[0][1], rng)
            except Exception:
                continue
            new_reply = new_reply.strip()
            if not new_reply or len(new_reply) < 2:
                continue
            convo = [{"from": "human", "value": pairs[0][0]},
                     {"from": "gpt", "value": new_reply}]
            # 多轮时后续轮次也套同一风格，强化「设定持续生效」
            for u, a in pairs[1:2]:
                try:
                    _, a2 = fn(a, rng)
                except Exception:
                    a2 = a
                if a2.strip():
                    convo.append({"from": "human", "value": u})
                    convo.append({"from": "gpt", "value": a2.strip()})
            f.write(json.dumps({"system_prompt": persona, "conversations": convo},
                               ensure_ascii=False) + "\n")
            counts[name] += 1
            written += 1
            if written % 50000 == 0:
                print(f"  {written:,} samples", flush=True)

    print(f"saved {written:,} style-constrained samples -> {out_path}", flush=True)
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14} {n:>8,}", flush=True)


if __name__ == "__main__":
    main()
