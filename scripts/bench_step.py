"""Measure one training step: time (fwd/bwd/optim), tokens/s, peak VRAM.

This is the measuring stick for the compute-path optimisation work.  It runs the
*real* model on *real* packed data with the same loss as ``retnet_hf/train.py``,
so a number here is directly comparable to the training log.

    python scripts/bench_step.py --model_dir checkpoints/retnet_272m_rp \
        --data_dir data_cache/chat_pack_v13 --batch 4 --chunk_size 64 \
        --grad_checkpoint

    # where does the time go
    python scripts/bench_step.py ... --profile

    # sweep
    python scripts/bench_step.py ... --sweep chunk_size=32,64,128,256
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.data import PackedChatDataset, ChatDataCollator  # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM              # noqa: E402


def load(args):
    model = RetNetForCausalLM.from_pretrained(args.model_dir)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    probe = np.load(Path(args.data_dir) / "train.ids.npy", mmap_mode="r")
    seq_len = probe.shape[1] - 1
    n_rows = probe.shape[0]
    del probe
    ds = PackedChatDataset(str(Path(args.data_dir) / "train.ids.npy"),
                           str(Path(args.data_dir) / "train.wcode.npy"), seq_len)
    return model, ds, seq_len, n_rows


def make_batches(ds, collator, batch, n_batches, device):
    out = []
    for i in range(n_batches):
        rows = [ds[(i * batch + j) % len(ds)] for j in range(batch)]
        b = collator(rows)
        out.append({k: v.to(device, non_blocking=True) for k, v in b.items()})
    return out


def weighted_loss(model, out_logits, labels, wcodes):
    """Exactly the loss retnet_hf/modeling_retnet.py computes."""
    logits_f = out_logits.float()
    token_loss = F.cross_entropy(logits_f.reshape(-1, logits_f.shape[-1]),
                                 labels.reshape(-1), reduction="none")
    w = torch.zeros_like(wcodes, dtype=torch.float32)
    w = torch.where(wcodes == 1, torch.ones_like(w), w)
    w = torch.where(wcodes == 2, torch.full_like(w, model.config.prompt_loss_weight), w)
    w = torch.where(wcodes == 3, torch.ones_like(w), w)
    return (token_loss * w.reshape(-1)).sum() / w.sum().clamp_min(1.0)


def run_once(model, batch, mode, chunk_size, args, autocast_ctx):
    """Call through the public ``forward`` so ``torch.compile`` actually applies.

    (Calling ``model._forward_hidden`` bypasses ``OptimizedModule.__call__`` and
    silently measures eager — that mistake invalidated an earlier A/B run.)
    """
    cfg = model.config if not hasattr(model, "_orig_mod") else model._orig_mod.config
    cfg.mode = mode
    cfg.chunk_size = chunk_size
    t0 = time.perf_counter()
    with autocast_ctx():
        # 故意**不**传 mode/chunk_size 关键字：HF Trainer 就是这么调的
        # （它只传 input_ids/labels/wcodes，其余从 config 读）。
        # 传不传关键字会让 Dynamo 走不同的特化路径 —— 这正是
        # bench_step 测出 1.65x 而真实训练倒挂 0.61x 的原因。
        out = model(input_ids=batch["input_ids"], labels=batch["labels"],
                    wcodes=batch["wcodes"])
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    out.loss.backward()
    torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t0
    return t_fwd, 0.0, t_bwd, out.loss.detach()


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--mode", default="chunkwise")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--grad_checkpoint", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--sweep", default=None, help="key=v1,v2,...")
    ap.add_argument("--dropout", type=float, default=None,
                    help="override every Dropout p (e.g. 0.0); default keeps the config")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    device = "cuda"
    model, ds, seq_len, n_rows = load(args)
    if args.dropout is not None:
        import torch.nn as nn
        n_drop = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = args.dropout
                n_drop += 1
        model.config.dropout = args.dropout
        print(f"dropout overridden to {args.dropout} on {n_drop} modules", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    model.to(device).train()
    collator = ChatDataCollator()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]

    def autocast_ctx():
        if amp_dtype is None:
            from contextlib import nullcontext
            return nullcontext()
        return torch.autocast("cuda", dtype=amp_dtype)

    print(f"model {n_params/1e6:.2f}M  seq_len={seq_len}  rows={n_rows}  "
          f"mode={args.mode}  chunk={args.chunk_size}  batch={args.batch}  "
          f"precision={args.precision}  grad_ckpt={args.grad_checkpoint}  "
          f"compile={args.compile}")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    if args.compile:
        model = torch.compile(model, backend="inductor")

    tok_per_step = args.batch * seq_len
    batches = make_batches(ds, collator, args.batch, args.steps + args.warmup + 2, device)

    # warmup
    for i in range(args.warmup):
        run_once(model, batches[i], args.mode, args.chunk_size, args, autocast_ctx)
        opt.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    fwds, losses, bwds, totals = [], [], [], []
    for i in range(args.warmup, args.warmup + args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f, ls, b, loss = run_once(model, batches[i], args.mode, args.chunk_size, args, autocast_ctx)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tot = time.perf_counter() - t0
        fwds.append(f); losses.append(ls); bwds.append(b); totals.append(tot)

    peak = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30
    med = statistics.median(totals)
    print(f"\n  step_time  median {med*1000:8.1f} ms   "
          f"(min {min(totals)*1000:.0f} / max {max(totals)*1000:.0f})")
    print(f"  fwd        median {statistics.median(fwds)*1000:8.1f} ms  "
          f"({statistics.median(fwds)/med*100:4.1f}%)")
    print(f"  loss       median {statistics.median(losses)*1000:8.1f} ms  "
          f"({statistics.median(losses)/med*100:4.1f}%)")
    print(f"  bwd        median {statistics.median(bwds)*1000:8.1f} ms  "
          f"({statistics.median(bwds)/med*100:4.1f}%)")
    print(f"  throughput {tok_per_step/med:8.0f} tok/s")
    print(f"  TFLOPs     {6*n_params*tok_per_step/med/1e12:8.2f}  (6N rule, model only)")
    print(f"  VRAM       alloc {peak:.2f} GiB / reserved {peak_res:.2f} GiB")
    print(f"  loss       {loss.item():.4f}")

    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        run_once(model, batches[0], args.mode, args.chunk_size, args, autocast_ctx)
        opt.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_once(model, batches[0], args.mode, args.chunk_size, args, autocast_ctx)
            opt.zero_grad(set_to_none=True)
        print("\n=== 按 CUDA 时间排序的算子（前 20）===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "model_dir": args.model_dir, "params": n_params, "seq_len": seq_len,
            "batch": args.batch, "chunk_size": args.chunk_size, "mode": args.mode,
            "precision": args.precision, "grad_checkpoint": args.grad_checkpoint,
            "compile": args.compile,
            "step_ms": med * 1000, "fwd_ms": statistics.median(fwds) * 1000,
            "bwd_ms": statistics.median(bwds) * 1000,
            "tok_per_s": tok_per_step / med, "tflops": 6 * n_params * tok_per_step / med / 1e12,
            "peak_alloc_gib": peak, "peak_reserved_gib": peak_res,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
