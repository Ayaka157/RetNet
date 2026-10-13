"""Download role-play / character-dialogue datasets (raw JSONL) to drive E.

Only repos verified reachable by ``scripts/probe_datasets.py`` are listed here.
Everything is streamed and written as JSONL; parsing into 用户/助手 training
samples happens later in ``scripts/build_chat_mix_v4.py`` so we never have to
re-download to fix a parser.

    python scripts/download_roleplay_data.py --all
    python scripts/download_roleplay_data.py --datasets haruhi_54k coser
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets"))

# label: (repo, config, split, max_examples)
DATASETS = {
    # ---- Chinese role-play ----
    "haruhi_54k": ("silk-road/ChatHaruhi-54K-Role-Playing-Dialogue", None, None, 60_000),
    "haruhi_expand": ("silk-road/ChatHaruhi-Expand-118K", None, None, 130_000),
    "haruhi_zero_pippa": ("silk-road/Haruhi-Zero-RolePlaying-movie-PIPPA", None, None, 60_000),
    "rolebench": ("ZenMoore/RoleBench", None, None, 200_000),
    "coser": ("Neph0s/CoSER", None, None, 60_000),
    "roleplay_zh_gpt4": ("shibing624/roleplay-zh-sharegpt-gpt4-data",
                         "sharegpt_formatted_data-evol-gpt4", None, 60_000),
    # ---- English role-play (style/structure transfer) ----
    "limarp_aug": ("grimulkan/LimaRP-augmented", None, None, 30_000),
    "claude_rp_30k": ("Norquinal/claude_multiround_chat_30k", None, None, 30_000),
    # ---- large Chinese chat / instruct ----
    "belle_3_5m": ("BelleGroup/train_3.5M_CN", None, None, 400_000),
    "sharegpt_zh_90k": ("shareAI/ShareGPT-Chinese-English-90k", None, None, 100_000),
    "moss_003": ("YeungNLP/moss-003-sft-data", None, None, 200_000),
    "smoltalk_zh_full": ("opencsg/smoltalk-chinese", None, None, 300_000),
    "firefly_full": ("YeungNLP/firefly-train-1.1M", None, None, 1_100_000),
    # ---- Chinese novels: narration + dialogue style for role-play ----
    "webnovel_cn": ("zxbsmk/webnovel_cn", None, None, 300_000),
}


def stream_one(repo, config, split):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split, streaming=True)
    if hasattr(ds, "keys"):
        first = next(iter(ds.keys()))
        print(f"  using split: {first}", flush=True)
        ds = ds[first]
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None, choices=list(DATASETS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out_dir", default="data_cache/roleplay")
    ap.add_argument("--max_examples", type=int, default=None,
                    help="override the per-dataset cap")
    args = ap.parse_args()

    selected = list(DATASETS) if args.all else (args.datasets or [])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name in selected:
        repo, config, split, cap = DATASETS[name]
        if args.max_examples:
            cap = args.max_examples
        dst = out / f"{name}.jsonl"
        if dst.exists():
            print(f"skip existing {dst}", flush=True)
            continue
        print(f"streaming {repo} config={config} cap={cap} ...", flush=True)
        tmp = dst.with_suffix(".part")
        try:
            ds = stream_one(repo, config, split)
            count = 0
            with open(tmp, "w", encoding="utf-8") as f:
                for ex in ds:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 20000 == 0:
                        print(f"  {name}: {count:,}", flush=True)
                    if count >= cap:
                        break
            tmp.replace(dst)
            print(f"saved {name}: {count:,} examples -> {dst} "
                  f"({dst.stat().st_size / 1048576:.1f} MB)", flush=True)
        except Exception as exc:
            print(f"FAILED {name} ({repo}): {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            tmp.unlink(missing_ok=True)
            continue


if __name__ == "__main__":
    main()
