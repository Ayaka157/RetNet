"""Run every T1 repro script and tee the output into _verify/logs/.

RUN:  python run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

VERIFY = Path(__file__).resolve().parent
LOGS = VERIFY / "logs"
PY = sys.executable

SCRIPTS = [
    ("harness_selftest", ["harness.py", "--demo-dtypes"]),
    ("d1_latest_checkpoint", ["repro_d1.py"]),
    ("d2_rotations_bf16", ["repro_d2.py"]),
    ("d3_d4_arange_refuted", ["repro_d34.py"]),
    ("d5_evaluate_nan", ["repro_d5.py"]),
    ("d6_gamma_restore", ["repro_d6.py"]),
    ("d7_torch_xla_device", ["repro_d7.py"]),
    ("d8_world_size_refuted", ["repro_d8.py"]),
    ("d9_optimizer_groups", ["repro_d9.py"]),
    ("sweep_constructors", ["repro_sweep.py"]),
    ("extras_init_and_storage", ["repro_extras.py"]),
    ("e2e_fresh", ["run_tiny_train.py", "fresh"]),
    ("e2e_resume", ["run_tiny_train.py", "resume"]),
    ("e2e_resume_re", ["run_tiny_train.py", "resume-re"]),
    ("crash_forensics", ["repro_crash_forensics.py"]),
]

LOGS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    results = []
    for name, cmd in SCRIPTS:
        t0 = time.perf_counter()
        env = dict(os.environ)
        if name == "e2e_resume_re":
            pass
        proc = subprocess.run([PY, *cmd], cwd=str(VERIFY), capture_output=True,
                              text=True, errors="replace", env=env)
        out = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr.strip() else "")
        (LOGS / f"{name}.txt").write_text(out, encoding="utf-8")
        dt = time.perf_counter() - t0
        results.append((name, proc.returncode, dt, len(out.splitlines())))
        print(f"  {name:<26} exit={proc.returncode} {dt:6.1f}s  lines={len(out.splitlines())}")
    print("\n  logs: " + str(LOGS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
