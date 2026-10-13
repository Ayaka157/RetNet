"""Download recommended small/medium chat and SFT datasets to drive E.

Streaming download: only the first ``--max_examples`` examples per dataset are
written to JSONL, so it does not fill the C drive.  All outputs go to
``data_cache/extra_sft`` on E by default.

Examples:
    python scripts/download_extra_data.py --datasets coig ultrachat smoltalk
    python scripts/download_extra_data.py --all --max_examples 50000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets"))

# COIG-CQIA 是一个数据集但包含多个 config，脚本会为每个 config 单独存一个 JSONL。
COIG_CONFIGS = ["zhihu", "douban", "ruozhiba", "xhs", "coig_pc"]

# (repo_id, split, [configs...])；configs 为 None 表示单数据集单文件。
DATASETS = {
    "coig": ("m-a-p/COIG-CQIA", "train", COIG_CONFIGS),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "train_sft", None),
    "smoltalk": ("HuggingFaceTB/smoltalk", "train", ["everyday-conversations"]),
    "smoltalk_chinese": ("opencsg/smoltalk-chinese", None, None),
    "belle": ("BelleGroup/multiturn_chat_0.8M", "train", None),
    "sharegpt4": ("shibing624/sharegpt_gpt4", "train", None),
    "ruozhiba": ("hfl/ruozhiba_gpt4", "train", None),
    # 追加：中文短答指令 + 中文长答知识，用于扩容 SFT 混合数据
    "firefly": ("YeungNLP/firefly-train-1.1M", "train", None),
    "moss_zh": ("Aratako/Moss-003-SFT-Chinese", "train", None),
    "alpaca_zh": ("llamafactory/alpaca_zh_demo", "train", None),
    "zhihu_kol": ("wangrui6/Zhihu-KOL", "train", None),
}


def _save_examples(ds, dst, max_examples, label):
    with open(dst, "w", encoding="utf-8") as f:
        count = 0
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
            if count % 5000 == 0:
                print(f"  {label}: {count}", flush=True)
            if count >= max_examples:
                break
    print(f"saved {label}: {count} examples -> {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None, choices=list(DATASETS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max_examples", type=int, default=100000)
    ap.add_argument("--out_dir", type=str, default="data_cache/extra_sft")
    ap.add_argument("--coig_configs", nargs="*", default=None,
                    help="覆盖 coig 默认下载的 config 列表")
    args = ap.parse_args()

    selected = list(DATASETS) if args.all else (args.datasets or [])
    coig_configs = args.coig_configs if args.coig_configs is not None else COIG_CONFIGS
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    def stream_one(repo, config, split):
        ds = load_dataset(repo, config, split=split, streaming=True)
        if hasattr(ds, "keys"):
            first_split = next(iter(ds.keys()))
            print(f"  using first split: {first_split}", flush=True)
            ds = ds[first_split]
        return ds

    for name in selected:
        repo, split, configs = DATASETS[name]
        if name == "coig":
            configs = coig_configs
        try:
            if configs:
                for config in configs:
                    dst = out / f"{name}_{config}.jsonl"
                    if dst.exists():
                        print(f"skip existing {dst}", flush=True)
                        continue
                    print(f"streaming {repo} config={config} split={split} ...", flush=True)
                    ds = stream_one(repo, config, split)
                    _save_examples(ds, dst, args.max_examples, f"{name}_{config}")
            else:
                dst = out / f"{name}.jsonl"
                if dst.exists():
                    print(f"skip existing {dst}", flush=True)
                    continue
                print(f"streaming {repo} (split={split or 'auto'}) ...", flush=True)
                ds = stream_one(repo, None, split)
                _save_examples(ds, dst, args.max_examples, name)
        except Exception as exc:
            # One unavailable/renamed repo must not abort the whole download run.
            print(f"FAILED {name} ({repo}): {type(exc).__name__}: {exc}", flush=True)
            continue


if __name__ == "__main__":
    main()
