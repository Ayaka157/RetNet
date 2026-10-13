"""Wait for the TPU session slot, then run the FIXED notebook on TPU v5e-8.

Kaggle allows only one concurrent batch TPU session, and the CLI (2.2.x) has no
cancel command, so an earlier submission is holding the slot.  This polls until
that session leaves QUEUED/RUNNING, pushes the finished runner, and then polls
the new version to completion, downloading logs and output.

The notebook this pushes is generated from `202610011026_fix.ipynb` by
build_kaggle_runner.py (MAX_STEPS=20 smoke run, crash diagnostics enabled).

Run:  python watch_and_run_tpu.py [--poll-seconds 90] [--max-wait-min 720]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

KERNEL = "ayaka157/retnet-kaggle-train-8k-fix"
RUN_DIR = Path(r"E:\Workspaces\RetNet\Kaggle\_kaggle_run\smoke")
LOG = Path(r"E:\Workspaces\RetNet\Kaggle\_kaggle_run\tpu_run.log")
BUSY = ("QUEUED", "RUNNING")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return (p.stdout or "") + (p.stderr or "")


def status() -> str:
    out = run(["kaggle", "kernels", "status", KERNEL])
    for line in out.splitlines():
        if "has status" in line:
            s = line.split('"')[1]
            return s.split(".")[-1]
    return "UNKNOWN"


def main() -> int:
    poll = 90
    max_wait_min = 720
    if "--poll-seconds" in sys.argv:
        poll = int(sys.argv[sys.argv.index("--poll-seconds") + 1])
    if "--max-wait-min" in sys.argv:
        max_wait_min = int(sys.argv[sys.argv.index("--max-wait-min") + 1])

    deadline = time.time() + max_wait_min * 60
    log(f"watching {KERNEL}; waiting for a free TPU session slot")

    # ---- phase 1: wait for the slot ----
    pushed = False
    while time.time() < deadline:
        st = status()
        log(f"status={st}")
        if st not in BUSY:
            log("slot appears free; pushing the FIXED runner (v(new))")
            out = run(["kaggle", "kernels", "push"], cwd=RUN_DIR)
            log("push: " + out.strip().replace("\n", " | "))
            if "successfully pushed" in out:
                pushed = True
                break
            if "Maximum batch TPU session" in out:
                log("slot still occupied; will retry")
            else:
                log("push failed with an unexpected error; will retry")
        time.sleep(poll)

    if not pushed:
        log("gave up waiting for a TPU slot")
        return 1

    # ---- phase 2: wait for completion ----
    last = None
    while time.time() < deadline:
        st = status()
        if st != last:
            log(f"status={st}")
            last = st
        if st not in BUSY:
            break
        time.sleep(poll)

    log(f"final status={status()}")
    logs = run(["kaggle", "kernels", "output", KERNEL, "-p", str(RUN_DIR)])
    log("output download: " + (logs.strip() or "(no files)"))
    page = run(["kaggle", "kernels", "logs", KERNEL])
    (RUN_DIR / "kernel_log.txt").write_text(page, encoding="utf-8")
    log(f"kernel log saved to {RUN_DIR / 'kernel_log.txt'} ({len(page)} chars)")
    for crit in ("[train] step", "[eval] step", "[gates]", "[cast]", "[resume]",
                 "Traceback", "Error", "error", "nan"):
        hits = [ln for ln in page.splitlines() if crit in ln]
        if hits:
            log(f"  {crit!r}: {len(hits)} line(s); first -> {hits[0][:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
