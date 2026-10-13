"""Download A-grade datasets (from docs/07_数据集总表_v2.csv) for the next SFT
round.  Streaming download with per-dataset caps so big corpora don't eat disk.

    python scripts/download_a_grade.py --all
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets"))

# label: (repo, config, split, max_examples, max_mb)
DATASETS = [
    ("nana_catgirl", "cyberlangke/Nana-catgirl-dataset-110k", None, None, 120_000, 600),
    ("hundredcv_chat", "taoyang168/HundredCV-Chat", None, None, 30_000, 100),
    ("roleplay_singleturn", "LooksJuicy/Chinese-Roleplay-SingleTurn", None, None, 20_000, 40),
    ("meow_reasoning", "Mxode/Meow-Reasoning-100K", None, None, 100_000, 600),
    ("refined_anime", "iosscu/Refined-Anime-Text", None, None, 1_000_000, 2500),
    ("novelist_cot", "Dxniz/Novelist-CoT", None, None, 200_000, 300),
    ("moegirlpedia", "YCWTG/MoeGirlPedia_zh_cleaned_latest", None, None, 300_000, 1500),
    ("scholar_novels", "y1yang0/scholar-novels-curated", None, None, 50_000, 1500),
    ("chinese_novel_cc0", "asd567557275/chinese_novel", None, None, 20_000, 10),
    ("classical_corpus", "gujilab/chinese-classical-corpus", None, None, 200_000, 1500),
    ("lightnovel5000", "bh2821/LightNovel5000", None, None, 200_000, 2000),
    ("sys_novel", "Orion-zhen/sys-novel-cleaned", None, None, 200_000, 2000),
    ("longevorolebench", "IAAR-Shanghai/LongEvoRoleBench", None, None, 200_000, 2000),
]


def stream_one(repo, config, split):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split, streaming=True)
    if hasattr(ds, "keys"):
        first = next(iter(ds.keys()))
        ds = ds[first]
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None,
                    choices=[d[0] for d in DATASETS])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out_dir", default="data_cache/a_grade")
    args = ap.parse_args()

    selected = [d for d in DATASETS if args.all or d[0] in (args.datasets or [])]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for label, repo, config, split, cap, mb_cap in selected:
        dst = out / f"{label}.jsonl"
        if dst.exists():
            print(f"skip existing {dst}", flush=True)
            continue
        print(f"streaming {repo} cap={cap} mb_cap={mb_cap} ...", flush=True)
        tmp = dst.with_suffix(".part")
        try:
            ds = stream_one(repo, config, split)
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
            print(f"FAILED {label} ({repo}): {type(exc).__name__}: {str(exc)[:150]}", flush=True)
            tmp.unlink(missing_ok=True)
            continue


if __name__ == "__main__":
    main()
