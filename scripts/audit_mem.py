"""Complete memory + parameter audit for RetNet 272M training.

Loads the SAME model class that train.py actually imports (retnet_hf, the
ORIGINAL — NOT the fp16 fork), enumerates every parameter tensor, computes the
AdamW optimizer state, and MEASURES a real training step under bf16 autocast
with chunkwise retention.  Peak memory is read from
torch.cuda.max_memory_allocated() after resetting BEFORE model load, so it
includes the fixed zone (params + grad + Adam) AND the transient activations.

Usage (local 3070 8GB == remote 4070 Laptop 8GB, same VRAM size):
    python scripts/audit_mem.py --batch 4 --seq 4096 --dtype bf16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# train.py imports RetNetForCausalLM from `retnet_hf` (the ORIGINAL module),
# NOT retnet_hf_fp16.  Measure that exact code path.
from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402


def fmt(nbytes: float) -> str:
    return f"{nbytes / 1e9:.3f} GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("=" * 70)
    print("CONFIG (checkpoints/retnet_272m_rp/config.json)")
    print("=" * 70)
    cfg = RetNetHFConfig.from_pretrained("checkpoints/retnet_272m_rp")
    for k in ("vocab_size", "d_model", "num_layers", "num_heads", "value_dim",
              "ffn_dim", "tie_weights", "use_gla", "use_growth_ln",
              "growth_old_dim", "use_swiglu", "dropout"):
        print(f"  {k:16s} = {getattr(cfg, k)}")
    key_dim = cfg.d_model // cfg.num_heads
    vhead = cfg.value_dim // cfg.num_heads
    print(f"  key_dim (derived) = {key_dim}, value_head_dim = {vhead}")

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = RetNetForCausalLM.from_pretrained("checkpoints/retnet_272m_rp").cuda().train()
    model.config.mode = "chunkwise"
    model.config.chunk_size = 64

    # ---- test: does HF gradient_checkpointing_enable actually set the block flag? ----
    model.gradient_checkpointing_enable()
    gc_on = [b.gradient_checkpointing for b in model.blocks]
    print(f"\ngradient_checkpointing_enable() -> block flags: "
          f"{sum(gc_on)}/{len(gc_on)} blocks set to True")
    print("  (if 0/20, the ORIGINAL model silently IGNORES grad_checkpoint=True)")
    # force the flag on for the measurement, matching what Trainer INTENDS
    for b in model.blocks:
        b.gradient_checkpointing = True

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel params = {n_params:,}  ({n_params / 1e6:.2f}M)")

    # ---- parameter breakdown by module ----
    print("\n" + "=" * 70)
    print("PARAMETER BREAKDOWN (fp32 master = 4 bytes each)")
    print("=" * 70)
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if n >= 1_000_000 or "gamma" in name:
            print(f"  {name:55s} {str(tuple(p.shape)):28s} {n:>12,}  {fmt(n * 4)}")
    print(f"  {'TOTAL':55s} {'':28s} {total:>12,}  {fmt(total * 4)}")
    # optimizer state
    print(f"\n  fp32 params            = {fmt(total * 4)}")
    print(f"  AdamW exp_avg (fp32)   = {fmt(total * 4)}")
    print(f"  AdamW exp_avg_sq (fp32)= {fmt(total * 4)}")
    print(f"  gradients (fp32)       = {fmt(total * 4)}")
    print(f"  FIXED ZONE total       = {fmt(total * 4 * 4)}")

    after_model = torch.cuda.memory_allocated()
    print(f"\n[measured] after model load      : {fmt(after_model)}")

    opt = torch.optim.AdamW(model.parameters(), lr=1.1e-4, weight_decay=0.1)
    after_opt = torch.cuda.memory_allocated()
    print(f"[measured] after AdamW (no grad) : {fmt(after_opt)}")

    B, L, V = args.batch, args.seq, cfg.vocab_size
    x = torch.randint(0, V, (B, L), device="cuda")
    labels = torch.randint(0, V, (B, L), device="cuda")
    wc = torch.ones((B, L), device="cuda", dtype=torch.long)

    # ---- analytical activation/buffer sizes (B*L*V is the elephant) ----
    print("\n" + "=" * 70)
    print(f"TRANSIENT BUFFERS  (batch={B}, seq={L}, vocab={V}, B*L={B*L:,})")
    print("=" * 70)
    logits_elems = B * L * V
    print(f"  logits [{B},{L},{V}] {args.dtype}      = {fmt(logits_elems * 2)}")
    print(f"  logits.float()  fp32                    = {fmt(logits_elems * 4)}")
    print(f"  CE log_softmax  fp32 (internal)         = {fmt(logits_elems * 4)}")
    print(f"  embedding out   fp32 [{B},{L},{cfg.d_model}] = {fmt(B * L * cfg.d_model * 4)}")
    print(f"  per-block saved x (GC) bf16 [{B},{L},{cfg.d_model}] x{cfg.num_layers}")

    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()  # peak measured from THIS point (training only)

    with torch.autocast("cuda", dtype=amp_dtype):
        out = model(input_ids=x, labels=labels, wcodes=wc)
        loss = out.loss
    after_fwd = torch.cuda.memory_allocated()
    print(f"\n[measured] after forward (allocated)   : {fmt(after_fwd)}")
    print(f"[measured] loss = {loss.item():.4f}  nan={torch.isnan(loss).item()}")

    loss.backward()
    after_bwd = torch.cuda.memory_allocated()
    print(f"[measured] after backward (allocated)  : {fmt(after_bwd)}")

    opt.step()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"\n[measured] PEAK max_memory_allocated    : {fmt(peak)}")
    print(f"[measured] memory_reserved              : {fmt(reserved)}")
    print(f"[measured] GPU total 8GiB, usable ~{fmt(8192 * 1024**2 - 477 * 1024**2)}")

    headroom = 8192 * 1024**2 - peak
    print(f"[result] headroom vs 8GiB               : {fmt(headroom)}")


if __name__ == "__main__":
    main()
