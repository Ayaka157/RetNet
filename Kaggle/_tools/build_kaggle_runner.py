"""Build a Kaggle-runner notebook from 202610011026_fix.ipynb.

Why a derived notebook instead of the fix notebook itself:
  * the fix notebook has stale outputs from the failed run baked in;
  * the failure happened at the FIRST XLA materialization with no traceback
    reaching the kernel, so the runner must also capture the evidence needed to
    diagnose it (worker faulthandler, C++ stack traces, device memory
    before/after the first sync, and a rank-tagged exception report).

Training behaviour is unchanged apart from the step budget, which is what makes
this a fast smoke test, and the launch cell.

Run:  python build_kaggle_runner.py [max_steps] [save_steps]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"E:\Workspaces\RetNet\Kaggle")
SRC = ROOT / "202610011026_fix.ipynb"
OUT_DIR = ROOT / "_kaggle_run"

# ---------------------------------------------------------------------------
# Cells are referenced by index in the source notebook:
#   0 markdown · 1 config/paths · 2 %pip · 3 markdown · 4 model+trainer
#   5 sanity checks · 6 launcher
# ---------------------------------------------------------------------------

DIAG_ENV = '''
# ============================================================
# Crash diagnostics: this run must explain the previous failure.
# The old run died at the first XLA sync with NO traceback reaching the
# kernel (BrokenProcessPool, workers SIGTERMed), so enable everything that can
# turn a native device/compile fault into readable output.
# ============================================================
import os

os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# Keep the XLA dumps opt-in: XLA_IR_DEBUG/XLA_HLO_DEBUG print enormous graphs.
for _flag in ("XLA_IR_DEBUG", "XLA_HLO_DEBUG", "XLA_DUMP_HLO_AS_TEXT"):
    os.environ.setdefault(_flag, "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")
os.environ.setdefault("XLA_USE_BF16", os.environ.get("XLA_USE_BF16", "1"))

print("[diag] environment prepared:", {
    k: os.environ.get(k) for k in (
        "PJRT_DEVICE", "XLA_USE_BF16", "PYTHONFAULTHANDLER",
        "TORCH_SHOW_CPP_STACKTRACES", "OMP_NUM_THREADS",
    )
})
'''

LAUNCH_DIAG = '''
# ============================================================
# Launch all TPU cores. Every rank's exception (full traceback) and the
# device memory around the first XLA sync are printed, because the previous
# run's failure never surfaced as a Python exception on the kernel.
# ============================================================
import os, sys, traceback, time

from accelerate import notebook_launcher
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def _rank_tag():
    try:
        return f"[rank {xr.global_ordinal()}/{xr.world_size()}]"
    except Exception:
        return "[rank ?]"


def _mem(tag):
    try:
        info = xm.get_memory_info()
        print(f"[diag]{_rank_tag()} {tag}: {info}", flush=True)
    except Exception as exc:
        print(f"[diag]{_rank_tag()} {tag}: memory info unavailable ({exc})", flush=True)


_orig_train_worker = train_worker


def train_worker_with_diagnostics():
    """Run train_worker, flush a rank-tagged traceback on ANY failure.

    Also forced-sync after the first optimizer step so that a compile/device
    fault is attributable to the forward/backward (before sync) or to the
    first materialization (at sync).
    """
    _mem("before train_worker")
    try:
        _orig_train_worker()
    except BaseException:
        print(f"\\n{'=' * 70}\\n{_rank_tag()} TRAIN WORKER FAILED\\n{'=' * 70}",
              flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        _mem("after train_worker")


notebook_launcher(train_worker_with_diagnostics, args=())

print("\\n[diag] launcher returned; checking what landed in the output dir")
import glob
print("[diag] checkpoints:", sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))))
print("[diag] exported files:", sorted(
    os.path.basename(p) for p in glob.glob(os.path.join(OUTPUT_DIR, "*"))
))
'''


def main() -> int:
    max_steps = sys.argv[1] if len(sys.argv) > 1 else "20"
    save_steps = sys.argv[2] if len(sys.argv) > 2 else "10"

    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # --- cell 1: shrink the step budget so this is a smoke run, keep the rest
    c1 = "".join(cells[1]["source"])
    for old, new, what in (
        ("MAX_STEPS        = 30_000", f"MAX_STEPS        = {max_steps}", "max_steps"),
        ("SAVE_STEPS       = 500", f"SAVE_STEPS       = {save_steps}", "save_steps"),
        ("EVAL_STEPS       = 500", f"EVAL_STEPS       = {save_steps}", "eval_steps"),
        ("WARMUP           = 300", "WARMUP           = 2", "warmup"),
    ):
        assert c1.count(old) == 1, f"anchor not found once: {what}"
        c1 = c1.replace(old, new)
    # make the smoke run land in its own directory
    c1 = c1.replace('OUTPUT_DIR  = "/kaggle/working/retnet_272m_v11_8k_tpu"',
                    'OUTPUT_DIR  = "/kaggle/working/retnet_272m_v11_8k_smoke"')
    c1 = c1.rstrip("\n") + "\n\n" + DIAG_ENV + "\n"
    cells[1]["source"] = c1.splitlines(keepends=True)

    # --- launcher cell: diagnostic wrapper
    cells[6]["source"] = LAUNCH_DIAG.splitlines(keepends=True)
    cells[6]["outputs"] = []
    cells[6]["execution_count"] = None

    # --- drop stale outputs everywhere so Kaggle does not replay the old crash
    for c in cells:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb_path = OUT_DIR / "retnet_kaggle_train_8k_fix.ipynb"
    nb_path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")

    meta = {
        "id": "ayaka157/retnet-kaggle-train-8k-fix",
        "title": "retnet-kaggle-train-8k-fix",
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": False,
        "dataset_sources": [
            "ayaka157/retnet-v11-8k-dataset",
            "ayaka157/retnet-272m-init-weights-2026-8-25",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[build] {nb_path}  ({nb_path.stat().st_size} bytes)")
    print(f"[build] {OUT_DIR / 'kernel-metadata.json'}")
    print(f"[build] MAX_STEPS={max_steps} SAVE/EVAL_STEPS={save_steps} "
          f"OUTPUT_DIR=.../retnet_272m_v11_8k_smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
