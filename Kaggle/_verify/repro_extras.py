"""Extra checks backing the 'checked and NOT a defect' list in FINDINGS_V1.md.

RUN:  python repro_extras.py
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path

import torch

import vcommon as V

V.banner("EXTRAS  init-weights safety + checkpoint storage hazards")

env = V.load_env("extras_out")
ds, md = V.H.setup_tiny_fixtures()

V.section("E1: does MultiScaleRetention._init_weights() clobber the decay gate?")
msr = V.build_msr(env, num_heads=4, d_model=64, value_dim=64, use_gla=True)
with torch.no_grad():
    msr.gamma.fill_(0.5)
before = msr.gamma.detach().clone()
gn_w = msr.group_norm.weight.detach().clone()
gn_b = msr.group_norm.bias.detach().clone()
msr._init_weights()
print(f"  gamma before={before.tolist()} after={msr.gamma.detach().tolist()}")
print(f"  gamma untouched: {torch.equal(before, msr.gamma.detach())}")
print(f"  group_norm.weight/bias untouched: "
      f"{torch.equal(gn_w, msr.group_norm.weight.detach()) and torch.equal(gn_b, msr.group_norm.bias.detach())}")
q_before = msr.q_proj.weight.detach().clone()
msr._init_weights()
print(f"  q_proj.weight was re-initialised: "
      f"{not torch.equal(q_before, msr.q_proj.weight.detach())}")

V.section("E2: model-level _init_weights_local() keeps the gate")
model_cls = env["RetNetForCausalLM"]
torch.set_default_dtype(torch.float32)
m = model_cls(V.tiny_config(env, use_gla=True))
torch.set_default_dtype(torch.bfloat16)
g = m.blocks[0].msr.gamma.detach().clone()
expected = 1.0 - torch.pow(2.0, -5.0 - torch.arange(4, dtype=torch.float32))
print(f"  gamma after construction     : {g.tolist()}")
print(f"  expected 1-2**(-5-i), i<4    : {expected.tolist()}")
print(f"  bitwise equal: {torch.equal(g, expected)}")

V.section("E3: _save_checkpoint pruning assumes every checkpoint-* dir name is numeric")
print("  cell 4 line 680-683:")
lines = V.H.CELL4_SRC.splitlines()
for n in (680, 681, 682, 683):
    print(f"  {n:4d}: {lines[n-1]}")
for name in ("checkpoint-500", "checkpoint-tmp", "checkpoint-", "checkpoint-1-final"):
    try:
        val = int(name.split("-")[-1])
        print(f"    int({name!r}.split('-')[-1]) = {val}   OK")
    except BaseException as exc:  # noqa: BLE001
        print(f"    int({name!r}.split('-')[-1]) -> {type(exc).__name__}: {exc}")
print("  -> if any directory matching `checkpoint-*` has a non-numeric suffix")
print("     (a manual copy, a partial/temp dir), the pruning loop raises")
print("     ValueError before it can delete anything.  Combined with D1 (the")
print("     same glob) any such dir also crashes resume earlier.")

V.section("E4: pruning deletes files only -- a subdirectory inside a checkpoint "
          "dir cannot be unlinked")
tmp = V.H.SANDBOX / "extras_out" / "checkpoint-99"
if tmp.exists():
    shutil.rmtree(tmp)
(tmp / "nested").mkdir(parents=True)
(tmp / "training.pt").write_text("x")
try:
    for child in tmp.iterdir():
        child.unlink()
    print("  UNEXPECTED SUCCESS")
except BaseException as exc:  # noqa: BLE001
    print(f"  {type(exc).__name__}: {exc}")
print("  (POSIX raises IsADirectoryError here; this Windows host reports")
print("   PermissionError [WinError 5] -- both are the same failure mode.)")
print("  -> `_save_checkpoint` lines 686-687 fail the same way if a checkpoint")
print("     dir ever contains a subdirectory (reasoned; the notebook itself never")
print("     writes one).")

V.section("E5: _latest_checkpoint returns (None, None) on an empty dir -> handled")
out = V.H.SANDBOX / "extras_out" / "empty_out"
out.mkdir(parents=True, exist_ok=True)
print(f"  _latest_checkpoint(empty dir) = {env['_latest_checkpoint'](str(out))}")
print("  train_worker guards with `if latest_dir is not None:` (line 875) -> OK")

V.section("E6: bf16 arange round-trip probe (mechanism evidence for D2)")
a = torch.arange(4096, 4192, dtype=torch.bfloat16)
c = torch.tensor([float(v) for v in range(4096, 4192)], dtype=torch.bfloat16)
print(f"  arange 4096..4128 : {[float(v) for v in a[:33]]}")
print(f"  per-element cast  : {[float(v) for v in c[:33]]}")
print(f"  identical: {torch.equal(a, c)}   -> arange lands on the coarse grid")
print(f"  torch.tensor(4127., dtype=torch.bfloat16) = "
      f"{float(torch.tensor(4127., dtype=torch.bfloat16))} (round-to-nearest)")

V.section("verdict")
print("  E1/E2: no gate clobbering -- the 'checked and not a defect' list holds.")
print("  E3/E4: two extra storage-path hazards (only reachable with unexpected")
print("         filesystem content); both are reasoned + partially executed.")
print("  E5:    the None branch is handled correctly.")
print("  E6:    the D2 quantization is not simple round-to-nearest.")
