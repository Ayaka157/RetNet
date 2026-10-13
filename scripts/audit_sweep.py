"""Batch sweep: measured peak VRAM per batch size, with and without the
logits.float() copy.  Authoritative numbers, nothing hand-waved."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402


def fmt(n: float) -> str:
    return f"{n / 1e9:.3f} GB"


def run_step(model, opt, B, L, V, amp_dtype, use_float):
    x = torch.randint(0, V, (B, L), device="cuda")
    labels = torch.randint(0, V, (B, L), device="cuda")
    wc = torch.ones((B, L), device="cuda", dtype=torch.long)

    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()

    with torch.autocast("cuda", dtype=amp_dtype):
        logits = model(input_ids=x).logits  # raw logits, no loss yet
        if use_float:
            logits_f = logits.float()
            token_loss = F.cross_entropy(logits_f.reshape(-1, V), labels.reshape(-1),
                                         reduction="none")
        else:
            token_loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                                         reduction="none")
        w = torch.ones((B, L), device="cuda", dtype=torch.float32)
        loss = (token_loss * w.reshape(-1)).sum() / w.sum().clamp_min(1.0)

    loss.backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return peak, reserved, loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    amp = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    model = RetNetForCausalLM.from_pretrained("checkpoints/retnet_272m_rp").cuda().train()
    model.config.mode = "chunkwise"
    model.config.chunk_size = 64
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=1.1e-4, weight_decay=0.1)
    V = model.config.vocab_size
    L = args.seq

    print(f"{'batch':>5} | {'logits.float()':>14} | {'peak_alloc':>10} | "
          f"{'reserved':>9} | {'loss':>9} | {'vs 8GiB':>10}")
    print("-" * 70)
    for B in (1, 2, 3, 4):
        peak, reserved, loss = run_step(model, opt, B, L, V, amp, use_float=True)
        print(f"{B:5d} | {'YES':>14} | {fmt(peak):>10} | {fmt(reserved):>9} | "
              f"{loss:9.4f} | {fmt(8192 * 1024**2 - peak):>10}")
    # fix test: drop logits.float()
    for B in (2, 4):
        peak, reserved, loss = run_step(model, opt, B, L, V, amp, use_float=False)
        print(f"{B:5d} | {'NO (fix)':>14} | {fmt(peak):>10} | {fmt(reserved):>9} | "
              f"{loss:9.4f} | {fmt(8192 * 1024**2 - peak):>10}")


if __name__ == "__main__":
    main()
