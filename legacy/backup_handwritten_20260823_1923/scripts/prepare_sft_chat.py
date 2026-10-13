"""Convert downloaded extra SFT JSONL files into 用户/助手 chat training ids.

Input : data_cache/extra_sft/*.jsonl
Output: data_cache/chat_sft_mix/{train.ids.npy,val.ids.npy,bpe_tokenizer.json}

Handles common HF formats:
- messages: [{role: user/assistant, content: ...}]
- conversations: [{from: human/gpt, value: ...}] or [{role:..., content:...}]
- instruction / input / output
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.tokenizer import BPETokenizer


def clean(s):
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ")
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def role_map(role):
    r = str(role).lower()
    if r in {"user", "human", "客户", "用户"}:
        return "用户"
    if r in {"assistant", "gpt", "bot", "ai", "助手"}:
        return "助手"
    return None


def messages_to_text(messages):
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = role_map(m.get("role", m.get("from", "")))
        content = clean(m.get("content", m.get("value", "")))
        if role and content:
            parts.append(f"{role}：{content}")
    return "\n".join(parts)


def parse_belle_turns(text):
    # Belle's instruction field already contains Human:/Assistant: turns.
    parts = []
    for m in re.finditer(r"(Human|Assistant)\s*[:：]\s*(.*?)(?=(?:Human|Assistant)\s*[:：]|$)", text, flags=re.S):
        role = "用户" if m.group(1) == "Human" else "助手"
        val = clean(m.group(2))
        if val:
            parts.append(f"{role}：{val}")
    return "\n".join(parts)


def example_to_text(obj):
    if isinstance(obj.get("messages"), list):
        return messages_to_text(obj["messages"])
    if isinstance(obj.get("conversations"), list):
        return messages_to_text(obj["conversations"])
    if "instruction" in obj:
        ins = clean(obj.get("instruction"))
        if re.search(r"Human\s*[:：]", ins):
            text = parse_belle_turns(ins)
        else:
            text = f"用户：{ins}"
        inp = clean(obj.get("input"))
        if inp:
            text += f"\n用户：{inp}"
        out = clean(obj.get("output"))
        if out:
            text += f"\n助手：{out}"
        return text
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data_cache/extra_sft")
    ap.add_argument("--out_dir", default="data_cache/chat_sft_mix")
    ap.add_argument("--tokenizer", default="data_cache/chat_recovery_mix/bpe_tokenizer.json")
    ap.add_argument("--max_per_file", type=int, default=30000)
    ap.add_argument("--min_tokens", type=int, default=24)
    ap.add_argument("--max_tokens", type=int, default=1024)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_chunks = []
    val_chunks = []
    train_docs = val_docs = 0

    for path in sorted(Path(args.input_dir).glob("*.jsonl")):
        kept = 0
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if kept >= args.max_per_file:
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = example_to_text(obj)
                if not text or "用户：" not in text or "助手：" not in text:
                    continue
                enc = tok.encode(text)
                if not (args.min_tokens <= len(enc) <= args.max_tokens):
                    continue
                ids = [tok.bos_id] + enc + [tok.eos_id]
                arr = np.asarray(ids, dtype=np.int64)
                if i % 50 == 0:
                    val_chunks.append(arr)
                    val_docs += 1
                else:
                    train_chunks.append(arr)
                    train_docs += 1
                kept += 1
        print(f"{path.name}: kept={kept} train_docs={train_docs} val_docs={val_docs}", flush=True)

    if train_chunks:
        np.save(str(out / "train.ids.npy"), np.concatenate(train_chunks))
    if val_chunks:
        np.save(str(out / "val.ids.npy"), np.concatenate(val_chunks))
    import shutil
    shutil.copyfile(args.tokenizer, out / "bpe_tokenizer.json")
    total = sum(len(x) for x in train_chunks)
    print(f"saved {out}: train_tokens={total:,}, val_tokens={sum(len(x) for x in val_chunks):,}")


if __name__ == "__main__":
    main()
