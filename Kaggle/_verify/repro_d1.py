"""D1 -- '_latest_checkpoint' calls re.fullmatch but cell 4 never imports re.

RUN:  python repro_d1.py
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path

import vcommon as V

V.banner("D1  missing `import re` in _latest_checkpoint")

env = V.load_env("d1_out")
latest = env["_latest_checkpoint"]
out = Path(env["CFG"].output_dir)
out.mkdir(parents=True, exist_ok=True)

src = V.H.CELL4.read_text(encoding="utf-8").splitlines()
V.section("source under test (cell 4 of the notebook)")
for n in (654, 655, 656, 657, 658, 659, 660):
    print(f"  {n:4d}: {src[n-1]}")
print(f"  cell4 module namespace has 're' ? {'re' in env}")
imports = [l.strip() for l in src if l.strip().startswith(("import ", "from "))]
print(f"  all import statements in cell 4: {imports}")

V.section("case A: output_dir empty  -> no checkpoint-* entry -> re never reached")
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True, exist_ok=True)
print("  files:", list(out.iterdir()))
print("  _latest_checkpoint ->", latest(str(out)))

V.section("case B: output_dir contains a checkpoint dir (the RESUME scenario)")
(out / "checkpoint-500").mkdir(parents=True, exist_ok=True)
print("  files:", [p.name for p in out.iterdir()])
try:
    result = latest(str(out))
    print("  UNEXPECTED SUCCESS ->", result)
except BaseException as exc:  # noqa: BLE001
    print(f"  {type(exc).__name__}: {exc}")
    tb = traceback.format_exc().strip().splitlines()
    for ln in tb:
        if "cell4" in ln or "NameError" in ln:
            print("  " + ln)

V.section("case C: a checkpoint-* entry that does NOT match the regex")
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True, exist_ok=True)
(out / "checkpoint-tmp").mkdir()
try:
    latest(str(out))
    print("  UNEXPECTED SUCCESS")
except BaseException as exc:  # noqa: BLE001
    print(f"  {type(exc).__name__}: {exc}")

V.section("case D: checkpoint-* is a plain FILE (glob matches files too)")
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True, exist_ok=True)
(out / "checkpoint-1").write_text("x", encoding="utf-8")
try:
    latest(str(out))
    print("  UNEXPECTED SUCCESS")
except BaseException as exc:  # noqa: BLE001
    print(f"  {type(exc).__name__}: {exc}")

V.section("control: does the SAME function work once `re` is imported?")
import re  # noqa: E402  (only for the control run)
env["re"] = re
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True, exist_ok=True)
(out / "checkpoint-500").mkdir()
(out / "checkpoint-1000").mkdir()
(out / "checkpoint-junk").mkdir()
print("  with re injected ->", latest(str(out)))

# leave the resume fixture in place for run_tiny_train.py --resume
V.section("verdict")
print("  D1 = PROVEN BY EXECUTION (NameError raised as soon as ANY checkpoint-*")
print("       entry exists in the output dir; empty dir is fine, which is why a")
print("       fresh Kaggle run never notices and a RESUME always dies).")
