"""Merge token shards into one memory-mapped train.ids.npy / val.ids.npy."""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np


def count_tokens(paths):
    n = 0
    for p in paths:
        n += np.load(p, mmap_mode="r").shape[0]
    return n


def merge(paths, out_path):
    total = count_tokens(paths)
    print(f"merging {len(paths)} shards, {total:,} tokens -> {out_path}")
    arr = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.int64, shape=(total,))
    pos = 0
    for p in paths:
        shard = np.load(p, mmap_mode="r")
        arr[pos:pos + shard.shape[0]] = shard
        pos += shard.shape[0]
    arr.flush()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", default="data_cache/pretrain_shards")
    ap.add_argument("--out_dir", default="data_cache/pretrain_merged")
    ap.add_argument("--pattern", default="*_train_*.npy")
    ap.add_argument("--val_pattern", default="*_val.ids.npy")
    ap.add_argument("--tokenizer", default="data_cache/chat_mix_v20k/bpe_tokenizer.json")
    args = ap.parse_args()
    shard_dir = Path(args.shard_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_paths = sorted(glob.glob(str(shard_dir / args.pattern)))
    val_paths = sorted(glob.glob(str(shard_dir / args.val_pattern)))
    if not train_paths:
        raise SystemExit("no train shards found")
    merge(train_paths, out / "train.ids.npy")
    if val_paths:
        merge(val_paths, out / "val.ids.npy")
    if Path(args.tokenizer).exists():
        shutil.copyfile(args.tokenizer, out / "bpe_tokenizer.json")
        print("copied tokenizer")
    print("DONE")


if __name__ == "__main__":
    main()
