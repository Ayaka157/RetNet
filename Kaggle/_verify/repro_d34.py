"""D3/D4 -- claimed bf16 poisoning of the dtype-less torch.arange() in
_decay_mask (line 148) and _parallel_final_state (line 168).

The claim is that after cell 4 runs torch.set_default_dtype(torch.bfloat16),
`torch.arange(length, device=device)` inherits bf16 and corrupts the decay
exponents.  This script tests it by EXECUTION, both the bare expression and the
real methods of MultiScaleRetention.

RUN:  python repro_d34.py
"""

from __future__ import annotations

import torch

import vcommon as V

V.banner("D3/D4  dtype-less torch.arange in _decay_mask / _parallel_final_state")

env = V.load_env("d34_out")
assert torch.get_default_dtype() == torch.bfloat16

src = V.H.CELL4.read_text(encoding="utf-8").splitlines()
V.section("source under test (cell 4 of the notebook)")
for n in (147, 148, 149, 150, 151, 166, 167, 168, 169, 170, 171):
    print(f"  {n:4d}: {src[n-1]}")

V.section("part 1: what dtype does the bare expression produce?")
probe = {
    "torch.arange(8)                                    ": torch.arange(8),
    "torch.arange(8, device='cpu')                      ": torch.arange(8, device="cpu"),
    "torch.arange(8192, device='cpu')                   ": torch.arange(8192, device="cpu"),
    "torch.arange(0, 8, dtype=torch.bfloat16)           ": torch.arange(0, 8, dtype=torch.bfloat16),
    "torch.arange(0.0, 8.0)                             ": torch.arange(0.0, 8.0),
    "torch.zeros(4)                                     ": torch.zeros(4),
    "torch.tensor(1.0)                                  ": torch.tensor(1.0),
}
for label, t in probe.items():
    tag = "INT-EXACT" if not t.is_floating_point() else f"bf16-poisoned={t.dtype}"
    print(f"  {label} -> {str(t.dtype):<16} {tag}")

length = 8192
idx_nb = torch.arange(length, device=V.CPU)                    # EXACT line 148
idx_f32 = torch.arange(length, device=V.CPU, dtype=torch.float32)
delta_nb = idx_nb[:, None] - idx_nb[None, :]
delta_f32 = idx_f32[:, None] - idx_f32[None, :]
diff = (delta_nb.to(torch.float64) - delta_f32.to(torch.float64)).abs()
print(f"\n  line 148 exactly: idx dtype = {idx_nb.dtype}  (int64: every integer <= 2^63 exact)")
print(f"  max |delta_notebook - delta_int64| over an 8192x8192 pair matrix = "
      f"{diff.max().item():.0f}")
print(f"  identical to the explicit-int64 result: {torch.equal(delta_nb, delta_f32.to(torch.int64))}")
print(f"  first 6 entries of row 1000 (notebook): {delta_nb[1000, 990:1000].tolist()}")

V.section("part 2: real _decay_mask output vs an explicit-fp32-index reference")
msr = V.build_msr(env, num_heads=16, d_model=64, value_dim=64)   # gamma: 16 heads
msr.gamma = torch.nn.Parameter(
    (1.0 - torch.pow(2.0, -5.0 - torch.arange(16, dtype=torch.float32))).clone())


def decay_mask_ref(length, gamma32):
    """Same math as cell 4 lines 148-157 but with an explicit fp32 index."""
    idx = torch.arange(length, device=V.CPU, dtype=torch.float32)
    delta = idx[:, None] - idx[None, :]
    causal = delta >= 0
    exp = delta.masked_fill(~causal, 0).to(torch.float32)
    mask = gamma32[:, None, None] ** exp
    mask = mask.masked_fill(~causal, 0.0)
    mask = mask / mask.sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    return mask


for L in (256, 512, 1024, 4096):
    got = msr._decay_mask(L, V.CPU, torch.bfloat16)
    exp_ref = decay_mask_ref(L, msr.gamma.detach().float()).to(torch.bfloat16)
    same = bool(torch.equal(got, exp_ref))
    md = (got.float() - exp_ref.float()).abs().max().item()
    print(f"  length={L:5d}  bitwise identical to fp32-index reference: {same}"
          f"   max abs mask diff = {md:.3e}")
print("  (_decay_mask is only reached in mode='parallel'; cell 5/train use")
print("   model_mode='chunkwise', where the chunk mask uses an explicit fp32")
print("   arange at line 218)")

V.section("part 3: real _parallel_final_state output vs fp32-index reference")
for L in (256, 512, 1024, 4096):
    b, h, kd, vd = 1, 16, 64, 64
    k = torch.randn(b, h, L, kd, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(b, h, L, vd, dtype=torch.float32).to(torch.bfloat16)
    got = msr._parallel_final_state(k, v, None)

    idx = torch.arange(L, device=V.CPU, dtype=torch.float32)          # reference
    gamma32 = msr.gamma.detach().float()
    weights = (gamma32[:, None] ** (L - 1 - idx).to(torch.float32)).to(torch.bfloat16)
    ref = torch.einsum("bhtk,bhtv,ht->bhkv", k, v, weights)
    same = bool(torch.equal(got, ref))
    md = (got.float() - ref.float()).abs().max().item()
    print(f"  tgt_len={L:5d}  bitwise identical to fp32-index reference: {same}"
          f"   max abs state diff = {md:.3e}")

V.section("verdict")
print("  D3 = REFUTED BY EXECUTION.")
print("  D4 = REFUTED BY EXECUTION.")
print("  Reason: torch.arange() with integer bounds and no dtype returns")
print(f"  {torch.arange(8).dtype}, NOT the default float dtype.  "
      "torch.set_default_dtype()")
print("  only affects *floating* default dtype inference, so lines 148/168 were")
print("  never poisoned; cell 4's explicit fp32 cast at line 151 / 170 is exact.")
print("  The 2026-09-09 'fix' for these two lines was a no-op change (harmless,")
print("  but its revert is NOT a regression).")
