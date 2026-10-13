"""Download literature / screenplay / dialogue corpora to drive E.

Goals: give the RP model (1) narrative & description voice from different
regions (classical Chinese lit, English classics, web novels already on disk),
and (2) direct dialogue pairs from movies (Cornell) that can be mined into
role-play samples.

    python scripts/download_literature.py --all
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets"))

# (label, repo, config, split, max_examples, max_mb)
DATASETS = [
    ("cornell_movie_dialog", "cornell-movie-dialog/cornell_movie_dialog", None, "train",
     400_000, 200),
    ("wikisource_zh", "wikimedia/wikisource", "20231201.zh", "train", 150_000, 600),
    ("wikisource_en", "wikimedia/wikisource", "20231201.en", "train", 150_000, 600),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None, choices=[d[0] for d in DATASETS])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out_dir", default="data_cache/literature")
    args = ap.parse_args()

    selected = [d for d in DATASETS if args.all or d[0] in (args.datasets or [])]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    for label, repo, config, split, cap, mb_cap in selected:
        dst = out / f"{label}.jsonl"
        if dst.exists():
            print(f"skip existing {dst}", flush=True)
            continue
        print(f"streaming {repo} config={config} cap={cap} ...", flush=True)
        tmp = dst.with_suffix(".part")
        try:
            ds = load_dataset(repo, config, split=split, streaming=True)
            if hasattr(ds, "keys"):
                first = next(iter(ds.keys()))
                ds = ds[first]
            count = 0
            with open(tmp, "w", encoding="utf-8") as f:
                for ex in ds:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 20000 == 0:
                        print(f"  {label}: {count:,} examples, "
                              f"{tmp.stat().st_size / 1048576:.0f} MB", flush=True)
                    if count >= cap:
                        break
                    if tmp.stat().st_size / 1048576 >= mb_cap:
                        print(f"  {label}: hit {mb_cap} MB cap", flush=True)
                        break
            tmp.replace(dst)
            print(f"saved {label}: {count:,} examples, "
                  f"{dst.stat().st_size / 1048576:.0f} MB -> {dst}", flush=True)
        except Exception as exc:
            print(f"FAILED {label}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            tmp.unlink(missing_ok=True)
            continue


if __name__ == "__main__":
    main()
