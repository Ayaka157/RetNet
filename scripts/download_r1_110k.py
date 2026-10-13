"""Download Chinese-DeepSeek-R1-Distill-data-110k (A-grade, math/reasoning).

Streams from HF with an example cap; saves JSONL to data_cache/a_grade/.

    python scripts/download_r1_110k.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets"))

REPO = "Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT"
CAP = 110_000
MB_CAP = 4000


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    from datasets import load_dataset
    out = Path("data_cache/a_grade")
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "r1_110k_sft.jsonl"
    if dst.exists():
        print(f"skip existing {dst} ({dst.stat().st_size/1048576:.0f}MB)", flush=True)
        return
    print(f"streaming {REPO} ...", flush=True)
    tmp = dst.with_suffix(".part")
    try:
        ds = load_dataset(REPO, split="train", streaming=True)
        count = 0
        with open(tmp, "w", encoding="utf-8") as f:
            for ex in ds:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1
                if count % 20000 == 0:
                    print(f"  {count:,} / {tmp.stat().st_size/1048576:.0f}MB", flush=True)
                if count >= CAP or tmp.stat().st_size / 1048576 >= MB_CAP:
                    break
        tmp.replace(dst)
        print(f"saved r1_110k_sft: {count:,} examples, {dst.stat().st_size/1048576:.0f}MB", flush=True)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        tmp.unlink(missing_ok=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
