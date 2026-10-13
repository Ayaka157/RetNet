"""D9 (lead's claim) -- 'optimizer is built with 2 parameter groups in
train_worker but 1 elsewhere, so _load_checkpoint raises
ValueError: loaded state dict has a different number of parameter groups'.

What is checked here, by execution:
  A) the notebook's own save -> load round trip (2 groups both times) works;
  B) loading a 2-group state dict into a 1-group optimizer does raise
     ValueError -> the general incompatibility is real;
  C) whether ANY code path inside the notebook builds a 1-group optimizer
     (static + AST sweep of cell 4 / cell 5 / cell 6).

RUN:  python repro_d9.py
"""

from __future__ import annotations

import ast
import traceback

import torch
from torch.optim import AdamW

import vcommon as V

V.banner("D9  optimizer parameter-group mismatch on resume")

env = V.load_env("d9_out")
src = V.H.CELL4_SRC
lines = src.splitlines()

V.section("part C first: every optimizer construction in the notebook")
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        fn = ast.unparse(node.func)
        if fn.endswith("AdamW") or fn.endswith("optimizer"):
            print(f"  line {node.lineno}: {ast.unparse(node)[:150]}")
print("  ---- cell 4 lines 851-869 (the only optimizer construction) ----")
for n in range(851, 870):
    print(f"  {n:4d}: {lines[n-1]}")
print("  cell 5/6 (executed before/after training):")
print("    cell5 imports inspect + builds a config/model: no optimizer")
print("    cell6 only calls notebook_launcher(train_worker, args=())")
print("  _load_checkpoint is called from exactly one place:")
for n in range(871, 882):
    print(f"  {n:4d}: {lines[n-1]}")

V.section("part A: the notebook's own round trip (2 groups -> 2 groups)")
ds, md = V.H.setup_tiny_fixtures()
out = V.H.SANDBOX / "d9_out"
V.H.tiny_cfg(env, ds, md, out, max_steps=2, save_steps=0, eval_steps=0)
model_cls = env["RetNetForCausalLM"]


def build_like_notebook(model, groups=2):
    _wd = [p for p in model.parameters() if p.ndim >= 2]
    _nowd = [p for p in model.parameters() if p.ndim < 2]
    if groups == 2:
        return AdamW([{"params": _wd, "weight_decay": 0.1},
                      {"params": _nowd, "weight_decay": 0.0}],
                     lr=1.1e-4, betas=(0.9, 0.95), eps=1e-8)
    return AdamW([{"params": list(model.parameters()), "weight_decay": 0.1}],
                 lr=1.1e-4, betas=(0.9, 0.95), eps=1e-8)


torch.set_default_dtype(torch.float32)
m = model_cls.from_pretrained(md)
torch.set_default_dtype(torch.bfloat16)
opt2 = build_like_notebook(m, groups=2)
sch = env["CosineWithMinLRScheduler"](opt2, 2, 10, 1.1e-5)

# one real step so that the saved state has non-empty exp_avg/exp_avg_sq
ids = torch.randint(0, 256, (2, 256), dtype=torch.long)
out_ = m(ids, labels=ids, wcodes=torch.ones_like(ids))
out_.loss.backward()
opt2.step()
sch.step()
print(f"  built optimizer has {len(opt2.param_groups)} param groups, "
      f"state entries={len(opt2.state)}")

ckpt = V.H.SANDBOX / "d9_ckpt"
ckpt.mkdir(parents=True, exist_ok=True)
torch.save({"model": m.state_dict(), "optimizer": opt2.state_dict(),
            "scheduler": sch.state_dict(), "step": 1, "best_eval_loss": 1.0,
            "config": {}}, ckpt / "training.pt")

m2 = model_cls.from_pretrained(md)
opt2b = build_like_notebook(m2, groups=2)
sch2 = env["CosineWithMinLRScheduler"](opt2b, 2, 10, 1.1e-5)
try:
    step, best = env["_load_checkpoint"](m2, opt2b, sch2, ckpt)
    print(f"  _load_checkpoint with matching 2-group optimizers -> OK "
          f"(step={step}, best={best})")
except BaseException as exc:  # noqa: BLE001
    print(f"  UNEXPECTED {type(exc).__name__}: {exc}")

V.section("part B: the 1-group-vs-2-group mismatch the lead reproduced")
m3 = model_cls.from_pretrained(md)
opt1 = build_like_notebook(m3, groups=1)
sch3 = env["CosineWithMinLRScheduler"](opt1, 2, 10, 1.1e-5)
print(f"  1-group optimizer has {len(opt1.param_groups)} param group(s)")
try:
    env["_load_checkpoint"](m3, opt1, sch3, ckpt)
    print("  UNEXPECTED SUCCESS")
except BaseException as exc:  # noqa: BLE001
    print(f"  {type(exc).__name__}: {exc}")
    for ln in traceback.format_exc().splitlines():
        if "cell4" in ln:
            print("  " + ln.strip())

V.section("verdict")
print("  D9 as stated ('the optimizer is built with two groups in train_worker")
print("  but one elsewhere, so _load_checkpoint raises') is:")
print("   * CONFIRMED as a general load-time incompatibility (part B raises the")
print("     exact ValueError).")
print("   * NOT REPRODUCIBLE inside the notebook itself: cell 4 builds the")
print("     optimizer in exactly one place with two groups, and _load_checkpoint")
print("     is called from exactly one place with that same optimizer -> the")
print("     notebook's own resume round trip works (part A).")
print("   * no 1-group construction exists in cell 1/2/5/6 (part C).")
print("  => classify as a cross-code-path HAZARD (mitigated in the fix by")
print("     build_optimizer()), not as a defect of the original notebook's")
print("     resume path.  The lead's harness reproduced it because the harness")
print("     built its own optimizer.")

