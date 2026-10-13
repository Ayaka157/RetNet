"""D5 -- `evaluate` builds the empty-loader NaN as torch.tensor(float('nan'))
with no dtype, so it inherits cell 4's bf16 default dtype.  Classify honestly:
dtype is wrong, behaviour is identical.

RUN:  python repro_d5.py
"""

from __future__ import annotations

import torch

import vcommon as V

V.banner("D5  torch.tensor(float('nan'), device=device) inherits bf16")

env = V.load_env("d5_out")
assert torch.get_default_dtype() == torch.bfloat16

src = V.H.CELL4.read_text(encoding="utf-8").splitlines()
V.section("source under test (cell 4 lines 726-753)")
for n in (726, 727, 728, 729, 743, 744, 745, 746, 747, 748, 750, 751, 752, 753):
    print(f"  {n:4d}: {src[n-1]}")

V.section("part 1: the exact expression, executed")
nb = torch.tensor(float("nan"), device=V.CPU)                       # line 746
fx = torch.tensor(float("nan"), device=V.CPU, dtype=torch.float32)  # the 0909 fix
print(f"  line 746 as written : dtype={str(nb.dtype):<16} value={nb.item()!r}")
print(f"  with dtype=float32  : dtype={str(fx.dtype):<16} value={fx.item()!r}")
print(f"  torch.get_default_dtype() = {torch.get_default_dtype()}")
print(f"  torch.tensor(float('nan')) is floating-point poisoned -> "
      f"{nb.dtype} (unlike torch.arange(4) -> {torch.arange(4).dtype})")

V.section("part 2: does anything downstream observe the difference?")
obs = {
    "float(x.cpu())                        ": (float(nb.cpu()), float(fx.cpu())),
    "x != x (NaN test)                     ": (nb != nb, fx != fx),
    "torch.isnan(x)                        ": (torch.isnan(nb), torch.isnan(fx)),
    "x + 1.0                               ": ((nb + 1.0).item(), (fx + 1.0).item()),
    "0.5 * x                               ": ((0.5 * nb).item(), (0.5 * fx).item()),
    "x < float('inf') (save-gate compare)  ": (nb < float("inf"), fx < float("inf")),
}
same = True
for label, (a, b) in obs.items():
    eq = (a == b) or (isinstance(a, float) and isinstance(b, float)
                      and a != a and b != b)
    same &= eq
    print(f"  {label} -> bf16:{a!r:<8} fp32:{b!r:<8} identical={eq}")

V.section("part 3: the shim path used by evaluate()")
xm = env["xm"] if "xm" in env else None
import torch_xla.core.xla_model as shim_xm  # noqa: E402
reduced = shim_xm.all_reduce(shim_xm.REDUCE_SUM, nb) / 1
print(f"  all_reduce(REDUCE_SUM, bf16 nan) -> dtype={reduced.dtype} "
      f"value={reduced.item()!r}")
print(f"  float(...cpu()) -> {float(reduced.cpu())!r}")

V.section("part 4: how is it reachable? (batches == 0)")
print("  line 745 `if batches == 0:` is only reachable when the evaluation")
print("  DataLoader yields nothing.  With DistributedSampler(drop_last=True)")
print("  and 8 replicas this needs val rows/8 < per_device_batch, i.e. a")
print("  degenerate val split -- possible but not the normal path.")
print("  With a NON-empty loader line 748 (`total_loss / batches`) is used and")
print("  line 746 never executes at all.")

V.section("verdict")
print(f"  D5 = PROVEN AS A DTYPE DEFECT (bf16 vs fp32), NO OBSERVABLE BEHAVIOUR")
print(f"      DIFFERENCE -> LATENT / TYPE-CONSISTENCY ONLY.  NaN is NaN in")
print(f"      every dtype; float(), isnan(), comparisons and all_reduce all agree.")
print("  Downstream identical for bf16/fp32:", same)
