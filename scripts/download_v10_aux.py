"""Download v10 auxiliary datasets: I_Wonder_Why (facts) + Better-Ruozhiba.

    python scripts/download_v10_aux.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets"))

# label: (repo, split, cap, mb_cap)
DATASETS = [
    ("i_wonder_why", "Mxode/I_Wonder_Why-Chinese", "train", 300_000, 2000),
    ("better_ruozhiba", "FunnySaltyFish/Better-Ruozhiba", "train", 50_000, 200),
]


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    from datasets import load_dataset
    out = Path("data_cache/a_grade")
    out.mkdir(parents=True, exist_ok=True)
    for label, repo, split, cap, mb_cap in DATASETS:
        dst = out / f"{label}.jsonl"
        if dst.exists():
            print(f"skip existing {dst} ({dst.stat().st_size/1048576:.0f}MB)", flush=True)
            continue
        print(f"streaming {repo} split={split} ...", flush=True)
        tmp = dst.with_suffix(".part")
        try:
            ds = load_dataset(repo, split=split, streaming=True)
            count = 0
            with open(tmp, "w", encoding="utf-8") as f:
                for ex in ds:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 20000 == 0:
                        print(f"  {label}: {count:,} / {tmp.stat().st_size/1048576:.0f}MB", flush=True)
                    if count >= cap or tmp.stat().st_size / 1048576 >= mb_cap:
                        break
            tmp.replace(dst)
            print(f"saved {label}: {count:,} examples, {dst.stat().st_size/1048576:.0f}MB", flush=True)
        except Exception as exc:
            print(f"FAILED {label} ({repo}): {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
