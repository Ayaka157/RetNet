"""04b - fetch a wikitext-103-raw slice, so the "original recipe" control corpus is
not missing its English component (legacy/prepare_chat_corpus.py sampled 200k
wikitext103 *rows* alongside the Chinese sources; we mirror that: row-level, not
article-level, with the same 2000-char clean cap).

Writes out/corpus/wikitext103.txt (one row per line).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT  # noqa: E402

OUTDIR = OUT / "out" / "corpus"
N_ROWS = 12000


def clean_text(s: str, max_chars: int = 2000) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars].rstrip()


def main() -> None:
    os.environ.setdefault("HF_HOME", str(OUT / "out" / "hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(OUT / "out" / "hf" / "datasets"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dst = OUTDIR / "wikitext103.txt"
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print("already have", dst, dst.stat().st_size)
        return
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    kept, rows, chars = 0, 0, 0
    with dst.open("w", encoding="utf-8") as f:
        for row in ds:
            t = row.get("text", "")
            if not t.strip():
                continue
            if t.lstrip().startswith("=") and t.rstrip().endswith("="):
                continue                      # heading row, not body text
            body = clean_text(t)
            if not body:
                continue
            f.write(body + "\n")
            rows += 1
            chars += len(body)
            if rows >= N_ROWS:
                break
    print(f"wrote {dst} rows={rows} chars={chars:,}")


if __name__ == "__main__":
    main()
