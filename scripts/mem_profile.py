"""Exact GPU memory profile for one fp16 8K training step (remote).

Prints torch.cuda.memory.summary() categories + peak.  Run on remote:
    python scripts/mem_profile.py --seq 8192 --batch 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf_fp16.modeling_retnet import RetNetForCausalLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_cache/chat_pack_v11")
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()  # reset BEFORE anything loads
    torch.cuda.empty_cache()

    model = RetNetForCausalLM.from_pretrained("checkpoints/retnet_272m_rp").cuda().train()
    model.config.mode = "chunkwise"
    model.config.chunk_size = 64
    # enable per-block gradient checkpointing like Trainer does
    for b in model.blocks:
        b.gradient_checkpointing = True
    after_model = torch.cuda.memory_allocated() / 1e9
    print(f"after model load : allocated={after_model:.2f} GB")

    ids_np = np.load(Path(args.data) / "val.ids.npy", mmap_mode="r")[0][: args.seq + 1]
    codes_np = np.load(Path(args.data) / "val.wcode.npy", mmap_mode="r")[0][1: args.seq + 1]
    x = torch.tensor(ids_np[:-1].astype(np.int64), device="cuda").unsqueeze(0).repeat(args.batch, 1)
    wc = torch.tensor(codes_np.astype(np.int64), device="cuda").unsqueeze(0).repeat(args.batch, 1)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    after_opt = torch.cuda.memory_allocated() / 1e9
    print(f"after optimizer  : allocated={after_opt:.2f} GB (model+grad+Adam)")
    model.zero_grad()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x, labels=x, wcodes=wc)
        loss = out.loss
    loss.backward()
    print(f"loss={loss.item():.4f} (nan={torch.isnan(loss).item()})  step_time={time.time()-t0:.1f}s")
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"max_memory_allocated = {peak:.2f} GB  (含模型+优化器+训练)")
    print(f"memory_reserved     = {reserved:.2f} GB")
    print(f"memory_reserved     = {reserved:.2f} GB")
    print("\n--- torch.cuda.memory_summary ---")
    print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
