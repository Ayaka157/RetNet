"""验证 fused_linear_cross_entropy 与标准路径数值一致，并量出显存差。

    python scripts/test_fused_ce.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.fused_ce import fused_linear_cross_entropy  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
dev = "cuda"
torch.manual_seed(0)

N, D, V = 2048, 1024, 20000
for scale in (0.02, 1.0):
    h = (torch.randn(N, D, device=dev, dtype=torch.float32) * scale).requires_grad_(True)
    W = (torch.randn(V, D, device=dev, dtype=torch.float32) * 0.02).requires_grad_(True)
    tgt = torch.randint(0, V, (N,), device=dev)
    w = torch.rand(N, device=dev) ** 2          # 模拟权重码（含 0）

    # 参考：与 modeling_retnet.py 原路径完全一致
    h1 = h.clone().detach().requires_grad_(True)
    W1 = W.clone().detach().requires_grad_(True)
    lg = h1 @ W1.t()
    lg_f = lg.float()
    ce = F.cross_entropy(lg_f, tgt, reduction="none")
    loss_ref = (ce * w).sum() / w.sum().clamp_min(1.0)
    loss_ref.backward()

    # 融合
    h2 = h.clone().detach().requires_grad_(True)
    W2 = W.clone().detach().requires_grad_(True)
    loss_new = fused_linear_cross_entropy(h2, W2, tgt, w)
    loss_new.backward()

    def rel(a, b):
        return ((a - b).abs().max() / b.abs().clamp_min(1e-9).max()).item()

    print(f"--- hidden 尺度 {scale} ---")
    print(f"  loss   ref {loss_ref.item():.6f}   fused {loss_new.item():.6f}   "
          f"|差| {abs(loss_ref.item()-loss_new.item()):.2e}")
    print(f"  grad_h 相对最大差 {rel(h2.grad, h1.grad):.2e}")
    print(f"  grad_W 相对最大差 {rel(W2.grad, W1.grad):.2e}")
    ok = (abs(loss_ref.item() - loss_new.item()) < 1e-3
          and rel(h2.grad, h1.grad) < 1e-4 and rel(W2.grad, W1.grad) < 1e-4)
    print(f"  => {'等价 ✓' if ok else '不等价 ✗'}")
    del h, W, h1, W1, h2, W2, lg, lg_f, ce
    torch.cuda.empty_cache()

print("\n=== 峰值显存对比（只算损失那一段）===")
for bs in (1, 2):
    n = bs * 2048
    def peak(fn):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        h = torch.randn(n, D, device=dev, dtype=torch.float32, requires_grad=True)
        W = torch.randn(V, D, device=dev, dtype=torch.float32, requires_grad=True)
        tgt = torch.randint(0, V, (n,), device=dev)
        w = torch.rand(n, device=dev)
        fn(h, W, tgt, w).backward()
        p = torch.cuda.max_memory_allocated() / 2**20
        del h, W, tgt, w
        torch.cuda.empty_cache()
        return p

    def ref_fn(h, W, tgt, w):
        lg_f = (h @ W.t()).float()
        ce = F.cross_entropy(lg_f, tgt, reduction="none")
        return (ce * w).sum() / w.sum().clamp_min(1.0)

    pr = peak(ref_fn)
    pf = peak(lambda h, W, tgt, w: fused_linear_cross_entropy(h, W, tgt, w))
    print(f"  batch {bs} (N={n}): 参考 {pr:7.1f} MiB   融合 {pf:7.1f} MiB   "
          f"省 {pr-pf:6.1f} MiB")
