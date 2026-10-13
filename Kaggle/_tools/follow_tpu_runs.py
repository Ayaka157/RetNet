"""Follow the smoke run, then auto-launch the full 30k-step training run.

Chain (one TPU session at a time, so this is strictly sequential):
  1. poll the smoke kernel until it stops;
  2. pull its logs and output, print the lines that matter;
  3. decide pass/fail from the smoke output;
  4. on pass, push the `full` target and follow it too;
  5. keep every downloaded artifact under _kaggle_run/.

Pass criteria for the smoke (all must hold):
  * `[train] step=     1` present  -> the first XLA sync succeeded (the historic
    crash point)
  * no `nan` in the train loss line -> the GAMMA_MAX clamp did its job
  * `[done] saved Hugging Face model` present -> checkpoint + export work
  * no `TRAIN WORKER FAILED` / `BrokenProcessPool` in the log

Run:  python follow_tpu_runs.py [--poll-seconds 60] [--max-wait-min 700]
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

KAGGLE = r"C:\Tools\Miniconda3\Scripts\kaggle.exe"
RUN_ROOT = Path(r"E:\Workspaces\RetNet\Kaggle\_kaggle_run")
SMOKE = ("ayaka157/retnet-kaggle-train-8k-fix", RUN_ROOT / "smoke")
FULL = ("ayaka157/retnet-kaggle-train-8k-full", RUN_ROOT / "full")
LOG = RUN_ROOT / "follow.log"
BUSY = ("QUEUED", "RUNNING")


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return (p.stdout or "") + (p.stderr or "")


def status(ref: str) -> str:
    out = run([KAGGLE, "kernels", "status", ref])
    for line in out.splitlines():
        if "has status" in line:
            return line.split('"')[1].split(".")[-1]
    return "UNKNOWN"


def follow(ref: str, workdir: Path, poll: int, deadline: float) -> str:
    last = None
    while time.time() < deadline:
        st = status(ref)
        if st != last:
            log(f"{ref} status={st}")
            last = st
        if st not in BUSY:
            break
        time.sleep(poll)
    return status(ref)


def collect(ref: str, workdir: Path) -> str:
    log(f"downloading output for {ref}")
    out = run([KAGGLE, "kernels", "output", ref, "-p", str(workdir)])
    log("output: " + (out.strip().replace("\n", " | ")[:400] or "(none)"))
    page = run([KAGGLE, "kernels", "logs", ref])
    (workdir / "kernel_log.txt").write_text(page, encoding="utf-8")
    log(f"log saved: {workdir / 'kernel_log.txt'} ({len(page)} chars)")
    return page


def verdict(page: str) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if "[train] step=" in page:
        steps = re.findall(r"\[train\] step=\s*(\d+) loss=\s*([0-9.]+|nan)", page)
        notes.append(f"train lines: {len(steps)}; first={steps[0] if steps else None}")
        if steps and steps[0][1] == "nan":
            ok = False
            notes.append("FAIL: step-1 loss is nan -> the gate clamp did not take effect")
        elif steps:
            first_loss = float(steps[0][1])
            if not (0.0 < first_loss < 20.0):
                ok = False
                notes.append(f"FAIL: implausible step-1 loss {first_loss}")
            else:
                notes.append(f"OK: step-1 loss {first_loss} (expect ~ln(20000)=9.9)")
    else:
        ok = False
        notes.append("FAIL: no `[train] step=` line -> the first XLA sync never completed")

    if "[gates]" in page:
        notes.append("gates: " + next(ln.strip() for ln in page.splitlines() if "[gates]" in ln))
    if "clamped_at_load=1" in page:
        notes.append("OK: the >1 decay gate was clamped at load")
    if "[cast]" in page:
        notes.append("cast: " + next(ln.strip() for ln in page.splitlines() if "[cast]" in ln))
    if "[done] saved Hugging Face model" in page:
        notes.append("OK: final HF export succeeded")
    else:
        ok = False
        notes.append("FAIL: no final HF export line")
    if "TRAIN WORKER FAILED" in page or "BrokenProcessPool" in page:
        ok = False
        notes.append("FAIL: worker failure / BrokenProcessPool in the log")
    if re.search(r"\bnan\b", page):
        notes.append("WARNING: the word 'nan' appears somewhere in the log")
    return ok, notes


def main() -> int:
    poll, max_wait = 60, 700
    if "--poll-seconds" in sys.argv:
        poll = int(sys.argv[sys.argv.index("--poll-seconds") + 1])
    if "--max-wait-min" in sys.argv:
        max_wait = int(sys.argv[sys.argv.index("--max-wait-min") + 1])
    deadline = time.time() + max_wait * 60

    log("=== phase 1: smoke run ===")
    st = follow(SMOKE[0], SMOKE[1], poll, deadline)
    log(f"smoke finished with status={st}")
    page = collect(SMOKE[0], SMOKE[1])
    ok, notes = verdict(page)
    for n in notes:
        log("  " + n)

    if not ok:
        log("=== smoke did NOT pass; not launching the full run ===")
        return 1

    log("=== smoke PASSED; launching full 30k-step run ===")
    push = run([KAGGLE, "kernels", "push"], cwd=FULL[1])
    log("push: " + push.strip().replace("\n", " | ")[:300])
    if "successfully pushed" not in push:
        log("full push failed (possibly the TPU slot is still held); stopping")
        return 1

    log("=== phase 2: full run ===")
    st = follow(FULL[0], FULL[1], poll, deadline)
    log(f"full finished with status={st}")
    page = collect(FULL[0], FULL[1])
    for n in verdict(page)[1]:
        log("  " + n)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
