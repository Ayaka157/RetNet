"""Validate tokenized pretraining shards / merged npy before training.

Checks:
- token id range and invalid ids
- unknown-token ratio in samples
- BOS/EOS markers exist
- decode a few random segments to verify text is not garbage
- embedding shape / forward sanity on a tiny RetNet instance

Usage:
    python scripts/validate_pretrain_data.py --data_dir data_cache/pretrain_merged
    python scripts/validate_pretrain_data.py --data_dir data_cache/pretrain_shards --pattern '*_train_*.npy'
"""
from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import numpy as np
import torch
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.model import RetNetConfig, RetNetLM
from retnet.tokenizer import BPETokenizer


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_cache/pretrain_merged")
    ap.add_argument("--tokenizer", default="data_cache/pretrain_merged/bpe_tokenizer.json")
    ap.add_argument("--pattern", default=None,
                    help="if given, validate all matching shard files instead of train.ids.npy")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--sample_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    vocab = tok.vocab_size
    unk = tok.unk_id
    bos = tok.bos_id
    eos = tok.eos_id
    print(f"tokenizer: vocab={vocab}, unk={unk}, bos={bos}, eos={eos}")

    data_dir = Path(args.data_dir)
    if args.pattern:
        files = sorted(glob.glob(str(data_dir / args.pattern)))
        if not files:
            raise SystemExit("no files match pattern")
        print(f"validating {len(files)} shard files: {files[0]} ... {files[-1]}")
        total_tokens = 0
        total_bos = total_eos = total_unk = 0
        invalid_ids = 0
        random.seed(args.seed)
        for fi, path in enumerate(files):
            arr = np.load(path, mmap_mode="r")
            total_tokens += arr.shape[0]
            for _ in range(max(1, args.samples // len(files))):
                if arr.shape[0] <= args.sample_len:
                    start = 0
                else:
                    start = random.randrange(0, arr.shape[0] - args.sample_len)
                seg = arr[start:start + args.sample_len]
                total_bos += int((seg == bos).sum())
                total_eos += int((seg == eos).sum())
                total_unk += int((seg == unk).sum())
                invalid_ids += int(((seg < 0) | (seg >= vocab)).sum())
            # decode first sample of first file only
            if fi == 0:
                print("\n--- sample decode from first shard ---")
                _print_decode(tok, arr[:args.sample_len * 2])
        print(f"\nchecked {total_tokens:,} tokens from {len(files)} shards")
    else:
        train_path = data_dir / "train.ids.npy"
        if not train_path.exists():
            raise SystemExit(f"{train_path} not found")
        arr = np.load(train_path, mmap_mode="r")
        total_tokens = arr.shape[0]
        print(f"train.ids.npy: {total_tokens:,} tokens, dtype={arr.dtype}, file={train_path}")

        random.seed(args.seed)
        total_bos = total_eos = total_unk = 0
        invalid_ids = 0
        for _ in range(args.samples):
            start = random.randrange(0, total_tokens - args.sample_len)
            seg = arr[start:start + args.sample_len]
            total_bos += int((seg == bos).sum())
            total_eos += int((seg == eos).sum())
            total_unk += int((seg == unk).sum())
            invalid_ids += int(((seg < 0) | (seg >= vocab)).sum())

        print("\n--- sample decode (random segments) ---")
        for _ in range(3):
            start = random.randrange(0, total_tokens - 512)
            _print_decode(tok, arr[start:start + 512])

    print("\n=== checks ===")
    sampled_tokens = args.samples * args.sample_len if not args.pattern else None
    print(f"sampled special BOS={total_bos}, EOS={total_eos}, UNK={total_unk}")
    print(f"sampled invalid ids = {invalid_ids}")
    if invalid_ids:
        print("ERROR: invalid token ids found!")
    elif total_unk == 0:
        print("OK: no unknown token in samples")
    else:
        print("WARNING: unknown tokens found; check if it is expected")
    if total_bos == 0 or total_eos == 0:
        print("WARNING: BOS/EOS not found in samples; documents may not be separated")
    else:
        print("OK: BOS/EOS markers present")

    # embedding sanity
    print("\n=== embedding sanity ===")
    cfg = RetNetConfig(vocab_size=vocab, d_model=128, num_layers=1, num_heads=4)
    model = RetNetLM(cfg).eval()
    sample = torch.randint(0, vocab, (2, 16))
    with torch.inference_mode():
        logits = model(sample, mode="parallel")
    print(f"RetNetLM accepts vocab={vocab}: logits shape={tuple(logits.shape)}")
    print("PASS: embedding / forward shape OK")


def _print_decode(tok, arr):
    ids = arr[: min(len(arr), 1024)].tolist()
    text = tok.decode(ids, skip_special_tokens=True)
    print(text[:600].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
