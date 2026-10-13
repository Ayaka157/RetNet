"""Wait for the next checkpoint-N dir under retnet_272m_rp and dump log_history loss curve.

Usage:
    python scripts/watch_272m_training.py [--ckpt N] [--timeout 3600]

If --ckpt is given, wait for that specific checkpoint; otherwise wait for ANY
checkpoint-N directory newer than now.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

OUT = Path("checkpoints/retnet_272m_rp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    deadline = time.time() + args.timeout
    target = None
    if args.ckpt is not None:
        target = OUT / f"checkpoint-{args.ckpt}"

    while time.time() < deadline:
        if target is not None:
            if target.exists():
                break
        else:
            for p in sorted(OUT.glob("checkpoint-*")):
                if p.is_dir() and (p / "trainer_state.json").exists():
                    target = p
                    break
        time.sleep(10)

    if target is None or not (target / "trainer_state.json").exists():
        print(f"TIMEOUT: no trainer_state.json under {OUT}", flush=True)
        sys.exit(1)

    state = json.loads((target / "trainer_state.json").read_text(encoding="utf-8"))
    hist = state.get("log_history", [])
    print(f"==== {target.name}  global_step={state.get('global_step')}  entries={len(hist)} ====", flush=True)
    for e in hist:
        step = e.get("step", "?")
        loss = e.get("loss")
        eval_loss = e.get("eval_loss")
        lr = e.get("learning_rate")
        grad = e.get("grad_norm")
        line = f"step={step}"
        if loss is not None:
            line += f"  loss={loss:.4f}"
        if grad is not None:
            line += f"  grad_norm={grad:.4f}"
        if lr is not None:
            line += f"  lr={lr:.2e}"
        if eval_loss is not None:
            line += f"  eval_loss={eval_loss:.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
