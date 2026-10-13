"""D2 -- MultiScaleRetention._rotations casts the *positions* to the compute
dtype (bf16), so integer positions > 256 are quantized and the RoPE phases are
destroyed at seq_len=8192.  Also: self.inv_freq is registered as bf16 and drifts
from the exact fp32 rotary frequencies.

RUN:  python repro_d2.py
"""

from __future__ import annotations

import torch

import vcommon as V

V.banner("D2  bf16 rotary positions + bf16 inv_freq")

env = V.load_env("d2_out")
assert torch.get_default_dtype() == torch.bfloat16, "cell 4 must have poisoned the default dtype"
msr = V.build_msr(env, num_heads=1, d_model=64, value_dim=64)   # key_dim = 64 = 1024/16
KEY_DIM = msr.key_dim
SEQ = 8192

src = V.H.CELL4.read_text(encoding="utf-8").splitlines()
V.section("source under test (cell 4 of the notebook)")
for n in (95, 96, 97, 118, 119, 120, 121, 122, 123):
    print(f"  {n:4d}: {src[n-1]}")
print(f"  key_dim used = {KEY_DIM} (real model: d_model 1024 / 16 heads = 64)")

# ---------------------------------------------------------------- positions
V.section("part 1: the position tensor built by _rotations (line 120)")
bf16_pos = torch.arange(0, SEQ + 1, device=V.CPU, dtype=torch.bfloat16)
exact_pos = torch.arange(0, SEQ + 1, device=V.CPU, dtype=torch.float64)
err = (bf16_pos.to(torch.float64) - exact_pos).abs()
print(f"  torch.arange(0, 8193, dtype=torch.bfloat16)  dtype={bf16_pos.dtype}")
print(f"  max |position error| over 0..8192 = {err.max().item():.0f} positions")
print(f"  number of distinct wrong positions = {(err > 0).sum().item()} / {SEQ + 1}")
print("  first index where error != 0     :", int((err > 0).nonzero()[0].item()))
print("  (dtype-less torch.arange() is NOT affected: it returns "
      f"{torch.arange(5).dtype}, so only this explicit-dtype call is poisoned)")
print()
print("  position | exact | bf16 value | abs err")
for p in (256, 257, 512, 1024, 4096, 8191, 8192):
    v = torch.tensor([p], dtype=torch.bfloat16).item()
    print(f"  {p:9d} | {p:6d} | {v:10.1f} | {abs(v - p):7.1f}")

# ---------------------------------------------------------------- inv_freq
V.section("part 2: self.inv_freq drift (lines 95-97: registered as bf16)")
inv_bf16 = msr.inv_freq.to(torch.float64)                     # notebook buffer
inv_exact = 1.0 / (10000.0 ** torch.linspace(0.0, 1.0, KEY_DIM // 2,
                                             dtype=torch.float64))
inv_exact = inv_exact.repeat_interleave(2)
rel = (inv_bf16 - inv_exact).abs() / inv_exact
print(f"  inv_freq dtype in the notebook = {msr.inv_freq.dtype} "
      f"(persistent={('inv_freq' in msr.state_dict())})")
print(f"  max relative inv_freq error = {rel.max().item():.3e} "
      f"at index {int(rel.argmax().item())}")
print(f"  inv_freq[0] (highest freq) = {inv_bf16[0].item():.10f} "
      f"(exact 1.0, rel err {rel[0].item():.3e})")
print(f"  inv_freq[-1] (lowest freq) = {inv_bf16[-1].item():.10e} "
      f"(exact {inv_exact[-1].item():.10e}, rel err {rel[-1].item():.3e})")
phase_drift = (inv_bf16 - inv_exact).abs() * SEQ
print(f"  resulting phase drift at position 8192: max {phase_drift.max().item():.4f} rad, "
      f"mean {phase_drift.mean().item():.4f} rad")
print(f"  #frequencies with phase drift > 0.1 rad = "
      f"{int((phase_drift > 0.1).sum().item())} / {KEY_DIM}")

# ---------------------------------------------------------------- sin/cos
V.section("part 3: sin/cos produced by the notebook path vs exact fp32 math")
sin_nb, cos_nb = msr._rotations(SEQ, 0, V.CPU, torch.bfloat16)      # notebook path
freqs_ref = torch.outer(torch.arange(SEQ, dtype=torch.float64), inv_exact)
sin_ref, cos_ref = torch.sin(freqs_ref), torch.cos(freqs_ref)

d_sin = (sin_nb.to(torch.float64) - sin_ref).abs()
d_cos = (cos_nb.to(torch.float64) - cos_ref).abs()
print(f"  sin dtype={sin_nb.dtype} shape={tuple(sin_nb.shape)}")
print(f"  max |sin - sin_exact| = {d_sin.max().item():.4f}   "
      f"max |cos - cos_exact| = {d_cos.max().item():.4f}")
print(f"  mean |sin - sin_exact| = {d_sin.mean().item():.4f}")
tot = d_sin.numel()
print(f"  entries with |sin err| > 0.1 : {int((d_sin > 0.1).sum().item())} / {tot} "
      f"({100.0 * (d_sin > 0.1).sum().item() / tot:.2f}%)")
print(f"  entries with |sin err| > 0.5 : {int((d_sin > 0.5).sum().item())} / {tot} "
      f"({100.0 * (d_sin > 0.5).sum().item() / tot:.2f}%)")
print("\n  phase at the highest rotary frequency (inv_freq = 1.0), sin(position):")
print("  position | notebook sin | exact sin | abs err | built-in fp32 sin")
for p in (256, 512, 4096, 8191, 8192):
    if p >= SEQ:
        continue
    b = sin_nb[p, 0].item()
    rf = sin_ref[p, 0].item()
    print(f"  {p:9d} | {b:12.4f} | {rf:9.4f} | {abs(b - rf):7.4f} | "
          f"{torch.sin(torch.tensor(float(p))).item():9.4f}")
print("  (the last column is what the phase *should* be: fp32 positions give")
print("   sin(position) exactly to fp32 accuracy)")

# ---------------------------------------------------------------- effect on q
V.section("part 4: effect on theta_shift(q) -- relative deviation of q_rot")
torch.manual_seed(0)
H, D = 16, KEY_DIM
q = torch.randn(2, H, SEQ, D, dtype=torch.float32).to(torch.bfloat16)
sin_r, cos_r = torch.sin(freqs_ref).to(torch.bfloat16), torch.cos(freqs_ref).to(torch.bfloat16)
rot_nb = env["theta_shift"](q, sin_nb, cos_nb)
rot_ref = env["theta_shift"](q, sin_r, cos_r)
rel = ((rot_nb.float() - rot_ref.float()).norm(dim=-1)
       / rot_ref.float().norm(dim=-1).clamp_min(1e-6))
print(f"  q shape {tuple(q.shape)}, both rotations bf16")
print(f"  per-(batch,head,token) relative L2 deviation of q_rot:")
print(f"    max  = {rel.max().item():.4f}")
print(f"    mean = {rel.mean().item():.4f}")
print(f"    #tokens with deviation > 10% : {int((rel > 0.10).sum().item())} / {rel.numel()}")
print(f"    first token with deviation > 10% : "
      f"{int((rel.max(dim=0).values.max(dim=0).values > 0.10).nonzero()[0].item())}")

V.section("verdict")
print("  D2 rotary-position part = PROVEN BY EXECUTION (hard numerical damage).")
print("  D2 inv_freq-drift part  = PROVEN BY EXECUTION (bf16 buffer, but the")
print("       damage is dominated by the position quantization).")
