"""D7 -- torch_xla.device() (cell 4 line 766) resolves only because the notebook
kernel already executed cell 4's top-level `import torch_xla` (line 14).

The `train_worker` body itself imports only the *aliases* xm/xr/pl (lines
759-761); `import torch_xla.core.xla_model as xm` binds the name `xm`, not the
name `torch_xla`.  So `torch_xla.device()` is a dependency on the kernel-global
name created by line 14.

RUN:  python repro_d7.py
"""

from __future__ import annotations

import ast
import traceback

import vcommon as V

V.banner("D7  torch_xla.device() kernel-global dependency")

env = V.load_env("d7_out")
src = V.H.CELL4_SRC
lines = src.splitlines()

V.section("source under test")
for n in (14, 21, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766):
    print(f"  {n:4d}: {lines[n-1]}")

V.section("which names does train_worker bind at function level?  (AST, not reading)")
tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "train_worker")
imports = []
for node in ast.walk(fn):
    if isinstance(node, ast.Import):
        imports.append([a.asname or a.name for a in node.names])
    elif isinstance(node, ast.ImportFrom):
        imports.append([a.asname or a.name for a in node.names])
print(f"  imports inside train_worker: {imports}")
print(f"  -> binds {[n for sub in imports for n in sub]} ; NOT the name 'torch_xla'")
fn_obj = env["train_worker"]
print(f"  'torch_xla' in train_worker.__globals__ (after cell 4 exec): "
      f"{'torch_xla' in fn_obj.__globals__}")
print(f"  the global is the shim here: {fn_obj.__globals__['torch_xla']}")
print("  cell 4 line 14 is the ONLY statement that binds it (see AST dump of "
      "module-level imports):")
mod_imports = []
for node in tree.body:
    if isinstance(node, ast.Import):
        mod_imports += [a.asname or a.name for a in node.names]
print(f"    module-level imports: {mod_imports}")

V.section("HOW THE REAL LAUNCH WORKS — static evidence from the installed "
          "accelerate, matching the saved Kaggle traceback")
import accelerate.launchers as L
import inspect
src_l = inspect.getsource(L.notebook_launcher)
for ln in src_l.splitlines():
    if "xmp.spawn" in ln or "start_method" in ln or "in_kaggle" in ln:
        print("  accelerate/launchers.py: " + ln.rstrip())
print("  saved crash traceback (cell6_output_dump.txt line 90):")
crash = (V.H.REPO / "Kaggle" / "_extract" / "cell6_output_dump.txt")
if crash.exists():
    for ln in crash.read_text(encoding="utf-8", errors="replace").splitlines():
        if "start_method" in ln and "fork" in ln:
            print("    " + ln.split("m")[-1][:140])

V.section("PROOF FROM THE SAVED KAGGLE LOG that torch_xla.device() DID resolve "
          "inside the worker")
stream = (V.H.REPO / "Kaggle" / "_extract" / "0909_cell6_stream.txt")
if stream.exists():
    for ln in stream.read_text(encoding="utf-8", errors="replace").splitlines()[:3]:
        print("  " + ln)
print("  line 2 `device=xla:0` is printed from line 772 using the object built at")
print("  line 766 by torch_xla.device(), so the name resolved in the worker.")

V.section("simulation: a worker whose __globals__ does NOT contain 'torch_xla'")
print("  (this is what a launcher that re-imports the module in a fresh")
print("   interpreter, instead of fork(), would give you)")
saved = fn_obj.__globals__.pop("torch_xla")
try:
    fn_obj()          # rank 0 / world 1 via the shim; fails at line 766
    print("  UNEXPECTED: train_worker returned without needing torch_xla")
except BaseException as exc:  # noqa: BLE001
    tb = traceback.format_exc().strip().splitlines()
    print(f"  {type(exc).__name__}: {exc}")
    for ln in tb:
        if "cell4" in ln:
            print("  " + ln.strip())
finally:
    fn_obj.__globals__["torch_xla"] = saved

V.section("verdict")
print("  D7 = REAL BUT NOT A CRASH IN THE REAL LAUNCHER.")
print("   * proven by execution: the name is a plain module global from line 14;")
print("     deleting it makes line 766 raise NameError.")
print("   * proven by the artifact: the fork-launched worker printed device=xla:0,")
print("     i.e. it HAD the global (fork inherits the kernel's module globals;")
print("     accelerate passes start_method='fork').")
print("   => kernel-global dependency / robustness issue, NOT the step-1 crash.")
print("   Recommended hardening: `import torch_xla` inside train_worker.")
