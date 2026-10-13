"""Build a dialogue-heavy recovery mix for chat fine-tuning after pretraining.

Mix = full chat_mix_v20k dialogue data + an evenly sampled slice of the large
pretrain corpus (default 25M tokens).  This restores 用户/助手 chat behaviour
without completely forgetting the general knowledge learned during the long
pretraining run.

Example:
    python scripts/build_recovery_mix.py \
        --out_dir data_cache/chat_recovery_mix \
        --pretrain_sample_tokens 25000000
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogue_dir", default=str(ROOT / "data_cache/chat_mix_v20k"))
    ap.add_argument("--pretrain_dir", default=str(ROOT / "data_cache/pretrain_merged"))
    ap.add_argument("--out_dir", default=str(ROOT / "data_cache/chat_recovery_mix"))
    ap.add_argument("--pretrain_sample_tokens", type=int, default=25_000_000)
    args = ap.parse_args()

    dialogue = Path(args.dialogue_dir)
    pretrain = Path(args.pretrain_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dlg = np.load(dialogue / "train.ids.npy", mmap_mode="r")
    src = np.load(pretrain / "train.ids.npy", mmap_mode="r")
    n_sample = min(args.pretrain_sample_tokens, src.shape[0])
    step = max(1, src.shape[0] // n_sample)
    indices = np.arange(0, step * n_sample, step, dtype=np.int64)[:n_sample]
    sample = src[indices]

    total = dlg.shape[0] + n_sample
    dst = np.lib.format.open_memmap(str(out / "train.ids.npy"), mode="w+",
                                    dtype=np.int64, shape=(total,))
    dst[:dlg.shape[0]] = dlg
    dst[dlg.shape[0]:] = sample
    dst.flush()

    # Validation stays dialogue-heavy: this is what we care about for chat.
    dlg_val = np.load(dialogue / "val.ids.npy", mmap_mode="r")
    np.save(str(out / "val.ids.npy"), dlg_val)
    shutil.copyfile(dialogue / "bpe_tokenizer.json", out / "bpe_tokenizer.json")

    print(f"dialogue tokens : {dlg.shape[0]:,}")
    print(f"pretrain sample : {n_sample:,} (stride={step})")
    print(f"recovery total  : {total:,}")
    print(f"val tokens      : {dlg_val.shape[0]:,}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
