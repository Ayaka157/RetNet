"""Robust segmented training loop for long unattended runs.

It repeatedly calls ``train.py`` in segments and automatically resumes from
``last_model.pt``.  If a segment crashes (OOM, Windows restart, etc.), it
retries the same segment with exponential backoff.

Example:
    python scripts/train_loop.py --total_steps 200000 --segment_steps 2000 \\
        --save_dir checkpoints/retnet_100m_chat --log_dir logs/loop -- \\
        --prepared_data_dir data_cache/chat_mix_v20k --vocab_size 20000 \\
        --d_model 768 --num_layers 12 --num_heads 12 --seq_len 256 \\
        --batch_size 16 --lr 3e-4 --warmup_steps 2000 --min_lr 3e-5 \\
        --mode parallel --amp bf16 --device cuda --chat_format
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # torch.load checkpoint 需要 retnet 模块

TRAIN_PY = ROOT / "train.py"


def checkpoint_step(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ckpt.get("step", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_steps", type=int, required=True)
    ap.add_argument("--segment_steps", type=int, default=2000)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--log_dir", default="logs/train_loop")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max_retries", type=int, default=5)
    args, extra = ap.parse_known_args()
    # argparse on Windows sometimes keeps the literal "--" separator in extras.
    if extra and extra[0] == "--":
        extra = extra[1:]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    last_model = save_dir / "last_model.pt"
    best_model = save_dir / "best_model.pt"
    if not last_model.exists() and best_model.exists():
        print("last_model.pt not found; falling back to best_model.pt")
        last_model = best_model

    start = checkpoint_step(last_model) if last_model.exists() else 0
    print(f"start step = {start}, total = {args.total_steps}")
    retries = 0

    while start < args.total_steps:
        end = min(start + args.segment_steps, args.total_steps)
        cmd = [args.python, str(TRAIN_PY), *extra,
               "--lr_total_steps", str(args.total_steps),
               "--max_steps", str(end), "--save_dir", str(save_dir)]
        if last_model.exists():
            cmd += ["--resume_from", str(last_model)]
        log = log_dir / f"segment_{start:06d}_{end:06d}.log"
        print(f"running segment {start}->{end}, log={log}")
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n===== command: {' '.join(cmd)} =====\n")
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode == 2:
            print("argument error (rc=2), stopping. Inspect:", log)
            sys.exit(2)
        if proc.returncode == 0:
            new_step = checkpoint_step(last_model) if last_model.exists() else end
            if new_step < end:
                print(f"WARNING: segment ended at {new_step}, expected {end}; retrying")
                time.sleep(30)
                retries += 1
                continue
            start = end
            retries = 0
        else:
            retries += 1
            print(f"segment failed rc={proc.returncode}, retry {retries}/{args.max_retries}")
            if retries >= args.max_retries:
                print("too many failures, stopping. Inspect:", log)
                sys.exit(1)
            time.sleep(60 * min(retries, 10))
    print("ALL DONE")


if __name__ == "__main__":
    main()
