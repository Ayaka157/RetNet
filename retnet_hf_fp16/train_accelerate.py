"""Accelerate bare-loop training micro-benchmark (vs Trainer overhead check).

    python retnet_hf/train_accelerate.py --model_dir checkpoints/retnet_143m_hf \
        --data_dir data_cache/chat_pack_v8 --steps 200 --batch_size 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accelerate import Accelerator  # noqa: E402
from retnet_hf import RetNetForCausalLM, PackedChatDataset, ChatDataCollator  # noqa: E402


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1.1e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    args = ap.parse_args()

    model = RetNetForCausalLM.from_pretrained(args.model_dir)
    probe = np.load(Path(args.data_dir) / "train.ids.npy", mmap_mode="r")
    seq_len = probe.shape[1] - 1
    del probe
    ds = PackedChatDataset(str(Path(args.data_dir) / "train.ids.npy"),
                           str(Path(args.data_dir) / "train.wcode.npy"), seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        pin_memory=True, num_workers=0)

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1)
    model, loader = accelerator.prepare(model, loader)
    if args.compile:
        model = torch.compile(model, backend="inductor")
        print("torch.compile applied", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    opt = accelerator.prepare(opt)

    data_iter = iter(loader)
    tok_total = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        with accelerator.accumulate(model):
            out = model(batch["input_ids"], labels=batch["labels"], wcodes=batch["wcodes"])
            accelerator.backward(out.loss)
            opt.step()
            opt.zero_grad()
        tok_total += batch["input_ids"].numel()
        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step} | {tok_total / el / 1000:.1f}k tok/s | loss {out.loss.item():.3f}",
                  flush=True)
    el = time.time() - t0
    print(f"FINAL: {args.steps} steps in {el:.1f}s -> "
          f"{tok_total / el / 1000:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
