"""Pre-stage BOTH Kaggle runs from the fixed notebook.

  --target smoke   MAX_STEPS=20,  save/eval every 10, separate output dir,
                   crash diagnostics on   (proves the first XLA sync works)
  --target full    MAX_STEPS=30000, save/eval every 500, production output dir

Both write their own directory under `_kaggle_run/<target>/` with a
`kernel-metadata.json` whose kernel id is distinct, so either can be pushed
without touching the other.

Kaggle permits only ONE concurrent batch TPU session, so these are pushed one
at a time; run `kaggle kernels push` from the target directory.

Run:  python build_kaggle_runs.py             (stages both)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"E:\Workspaces\RetNet\Kaggle")
SRC = ROOT / "202610011026_fix.ipynb"
RUN_ROOT = ROOT / "_kaggle_run"

DIAG_ENV = '''
# ============================================================
# Crash diagnostics — this run must also explain the previous failure.
# The old run died at the first XLA sync with NO traceback reaching the
# kernel (BrokenProcessPool, workers SIGTERMed), so turn on everything that
# can make a native device/compile fault readable, and record device memory
# around the first sync.
# ============================================================
import os

os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
for _flag in ("XLA_IR_DEBUG", "XLA_HLO_DEBUG", "XLA_DUMP_HLO_AS_TEXT"):
    os.environ.setdefault(_flag, "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")

print("[diag] environment:", {
    k: os.environ.get(k) for k in (
        "PJRT_DEVICE", "XLA_USE_BF16", "PYTHONFAULTHANDLER",
        "TORCH_SHOW_CPP_STACKTRACES", "OMP_NUM_THREADS",
    )
})
'''

LAUNCH_DIAG = '''
# ============================================================
# Launch all 8 TPU cores. Every rank's exception (full traceback) and the
# device memory around the first XLA sync are printed, because the previous
# failure never surfaced as a Python exception on the kernel.
# ============================================================
import glob, os, sys, traceback

from accelerate import notebook_launcher
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def _rank():
    try:
        return f"[rank {xr.global_ordinal()}/{xr.world_size()}]"
    except Exception:
        return "[rank ?]"


def _mem(tag):
    try:
        print(f"[diag]{_rank()} {tag}: {xm.get_memory_info()}", flush=True)
    except Exception as exc:
        print(f"[diag]{_rank()} {tag}: memory info unavailable ({exc})", flush=True)


_orig = train_worker


def train_worker_diag():
    _mem("before train_worker")
    try:
        _orig()
    except BaseException:
        print(f"\\n{'=' * 70}\\n{_rank()} TRAIN WORKER FAILED\\n{'=' * 70}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        _mem("after train_worker")


notebook_launcher(train_worker_diag, args=())

print("\\n[diag] launcher returned; checking the output dir", flush=True)
print("[diag] checkpoints:", sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))))
print("[diag] files:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUTPUT_DIR, "*"))))
'''

TARGETS = {
    "smoke": {
        "id": "ayaka157/retnet-kaggle-train-8k-fix",
        "title": "retnet-kaggle-train-8k-fix",
        "max_steps": "20",
        "save_steps": "10",
        "warmup": "2",
        "out_dir": "/kaggle/working/retnet_272m_v11_8k_smoke",
        "diagnostics": True,
    },
    "full": {
        "id": "ayaka157/retnet-kaggle-train-8k-full",
        "title": "retnet-kaggle-train-8k-full",
        "max_steps": "30_000",
        "save_steps": "500",
        "warmup": "300",
        "out_dir": "/kaggle/working/retnet_272m_v11_8k_tpu",
        "diagnostics": True,
    },
}


def stage(name: str, spec: dict) -> Path:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    c1 = "".join(cells[1]["source"])
    for old, new, what in (
        ("MAX_STEPS        = 30_000", f"MAX_STEPS        = {spec['max_steps']}", "max_steps"),
        ("SAVE_STEPS       = 500", f"SAVE_STEPS       = {spec['save_steps']}", "save_steps"),
        ("EVAL_STEPS       = 500", f"EVAL_STEPS       = {spec['save_steps']}", "eval_steps"),
        ("WARMUP           = 300", f"WARMUP           = {spec['warmup']}", "warmup"),
        ('OUTPUT_DIR  = "/kaggle/working/retnet_272m_v11_8k_tpu"',
         f'OUTPUT_DIR  = "{spec["out_dir"]}"', "output_dir"),
    ):
        assert c1.count(old) == 1, f"{name}: anchor not unique: {what}"
        c1 = c1.replace(old, new)
    if spec["diagnostics"]:
        c1 = c1.rstrip("\n") + "\n\n" + DIAG_ENV + "\n"
    cells[1]["source"] = c1.splitlines(keepends=True)

    if spec["diagnostics"]:
        cells[6]["source"] = LAUNCH_DIAG.splitlines(keepends=True)
    for c in cells:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None

    out = RUN_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    nb_name = f"retnet_kaggle_train_8k_{name}.ipynb"
    (out / nb_name).write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": spec["id"],
        "title": spec["title"],
        "code_file": nb_name,
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
    (out / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage] {name}: {out}  MAX_STEPS={spec['max_steps']} "
          f"SAVE/EVAL={spec['save_steps']} OUTPUT={spec['out_dir']}")
    return out


def main() -> int:
    for name, spec in TARGETS.items():
        stage(name, spec)
    print("\npush with:  cd <dir>; kaggle kernels push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
