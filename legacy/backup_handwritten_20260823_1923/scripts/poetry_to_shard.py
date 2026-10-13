"""Convert chinese-poetry JSON into a pretrain int16 shard.

唐/宋/元/诗经/楚辞/论语 → flat token stream (BOS/EOS per poem), written as an
int16 .npy shard that next pretraining runs can pick up via --shard_dirs.

    python scripts/poetry_to_shard.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.tokenizer import BPETokenizer  # noqa: E402

SOURCES = [
    ("全唐诗", "poet.tang.*.json"),
    ("全唐诗", "poet.song.*.json"),
    ("五代诗词", "poet.wudai.*.json"),
    ("元曲", "yuanqu.json"),
    ("诗经", "shijing.json"),
    ("楚辞", "chuci.json"),
    ("论语", "lunyu.json"),
    ("宋词", "ci.song.*.json"),
    ("宋词", "ci.wu.*.json"),
    ("曹操诗集", "caocao.json"),
    ("纳兰性德", "nalanxingde.json"),
]


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    poetry_dir = ROOT / "data_cache/nonhf/poetry/chinese-poetry"
    tok = BPETokenizer.load(ROOT / "data_cache/chat_pack_v4/bpe_tokenizer.json")
    out_dir = ROOT / "data_cache/pretrain_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    for sub, pattern in SOURCES:
        d = poetry_dir / sub
        if not d.exists():
            continue
        files = sorted(d.glob(pattern))
        n = 0
        for p in files:
            try:
                arr = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            for poem in arr if isinstance(arr, list) else []:
                paras = poem.get("paragraphs") or poem.get("content") or []
                if isinstance(paras, str):
                    paras = paras.split("\n")
                body = "\n".join(x.strip() for x in paras if x and x.strip())
                if len(body) >= 20:
                    texts.append(body)
                    n += 1
        print(f"{sub}/{pattern}: {n} works", flush=True)

    print(f"encoding {len(texts)} works ...", flush=True)
    flat: list[int] = []
    for i in range(0, len(texts), 512):
        encs = tok.encode_batch(texts[i:i + 512])
        for enc in encs:
            flat.append(tok.bos_id)
            flat.extend(enc)
            flat.append(tok.eos_id)
    arr = np.asarray(flat, dtype=np.int16)
    path = out_dir / "poetry_zh.npy"
    np.save(str(path), arr)
    print(f"saved {len(flat):,} tokens -> {path} ({path.stat().st_size / 1048576:.0f} MB)",
          flush=True)


if __name__ == "__main__":
    main()
