"""验证两件事：

A. 旧 `_init_weights` 把 `group_norm.weight` 也置零了 —— 这会让 retention 支路
   输出**恒为 0**，于是「快速路径 vs 参考路径」的模型级等价测试**空过**（两边都是 0）。
   判据：把 group_norm.weight 手动置 0，看两条路径是否变成逐位相同。

B. `group_norm.weight` 正常（=1）时，bf16 autocast 下的 1e-2 相对差到底是不是
   单纯的低精度噪声 —— 判据：同样的比较在 fp32 下应当小几个数量级。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import retnet_hf.modeling_retnet as MR                                # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig, MultiScaleRetention  # noqa: E402
from retnet_hf.retention_fast import chunkwise_batched                 # noqa: E402

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

mcfg = RetNetHFConfig(vocab_size=2000, d_model=256, num_layers=3, num_heads=4,
                      value_dim=512, ffn_dim=512, dropout=0.0)
torch.manual_seed(1)
model = RetNetForCausalLM(mcfg).to("cpu").eval()
ids = torch.randint(0, 2000, (2, 512))

gn = [(n, p) for n, p in model.named_parameters() if n.endswith("group_norm.weight")]
print("修复后 group_norm.weight 的值：")
for n, p in gn:
    print(f"  {n}: 全为1={bool((p == 1).all())}  全为0={bool((p == 0).all())}  "
          f"min={float(p.detach().min()):.6g} max={float(p.detach().max()):.6g}")


def run(patch, amp):
    orig = MultiScaleRetention._chunkwise_retention if patch else None

    def patched(self, qq, kk, vv, state, chunk_size):
        return chunkwise_batched(qq, kk, vv, self.gamma.detach().to(torch.float32), state, chunk_size)
    if patch:
        MultiScaleRetention._chunkwise_retention = patched
    # ⚠️ 关键：不关掉这个开关，「参考」就是被测函数本身，比较恒为 0（我第一版就栽在这）。
    saved = MR.FAST_CHUNKWISE
    MR.FAST_CHUNKWISE = False
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            return model._forward_hidden(ids, mode="chunkwise", return_state=True,
                                         chunk_size=128)[0].float()
    finally:
        MR.FAST_CHUNKWISE = saved
        if patch:
            MultiScaleRetention._chunkwise_retention = orig


def cmp(tag, amp):
    ref, new = run(False, amp), run(True, amp)
    d = (new - ref).abs()
    rmean = (d.mean() / ref.abs().mean().clamp_min(1e-9)).item()
    rmax = (d.max() / ref.abs().max().clamp_min(1e-30)).item()
    print(f"  {tag}: rel_mean={rmean:.3e}  rel_max={rmax:.3e}  "
          f"ref 是否恒为0={bool((ref == 0).all())}")
    return rmean


print("\nA. group_norm.weight 全置 0（复刻旧的坏初始化）")
for n, p in gn:
    with torch.no_grad():
        p.zero_()
cmp("  bf16 autocast", True)
print("  → 若 rel_mean ≈ 0 且 ref 恒为 0，说明旧代码下这个测试是**空过**的")

print("\nB. group_norm.weight 恢复成 1（正确初值）")
for n, p in gn:
    with torch.no_grad():
        p.fill_(1.0)
cmp("  bf16 autocast", True)
cmp("  fp32（关掉 autocast）", False)
print("  → 若 fp32 比 bf16 小两三个数量级，则 1e-2 是 bf16 精度噪声，不是内核不等价")
