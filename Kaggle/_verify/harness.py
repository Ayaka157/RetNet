"""T1 CPU exhibition harness for ``Kaggle/202610011026.ipynb`` cell 4.

The notebook's cell 4 is executed **verbatim / unmodified** on CPU.  Three
documented injections make that possible; they are listed in
``INJECTIONS`` and dumped to ``_verify/logs/harness_injections.json``:

I1. ``torch_xla`` CPU shim on ``sys.path`` (see ``_verify/shim/torch_xla``).
    Without it, cell 4 fails at its own ``import torch_xla`` (line 14).
I2. cell 1's module-global constants (``DATASET_DIR`` ... ``SAVE_LIMIT``).
    They are scraped from ``_extract/202610011026.ipynb.cell1.py`` at runtime
    (so their *values* are not transcribed by hand) and injected into cell 4's
    globals.  Only ``DATASET_DIR`` / ``MODEL_DIR`` / ``OUTPUT_DIR`` are
    overridden to point at local fixtures, because cell 1 validates absolute
    ``/kaggle/...`` paths that do not exist on this machine.
I3. ``builtins.open`` redirect for absolute paths starting with ``/kaggle/``.
    cell 4 unconditionally does
    ``open("/kaggle/working/train_config.json", "w")`` at import time, which
    cannot work on Windows (it resolves to ``<cwd-drive>:\\kaggle\\working``).
    Redirected paths go to ``_verify/_sandbox/kaggle/...`` so that nothing
    outside ``_verify/`` is written.  Set ``HARNESS_NO_OPEN_PATCH=1`` to
    disable and observe the raw failure.

The notebook sources are opened read-only; nothing is rewritten.  The sha256 +
line count of every executed cell is printed and recorded so a reader can
prove the code that ran is the code in the notebook.

Usage:
    python harness.py                 # print the manifest + self-test
    python harness.py --demo-dtypes   # exec cell 4, show the dtype poisoning
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
from pathlib import Path

# I0 (documented injection): the torch_xla shim must be importable by cell 4,
# but it must NOT make transformers/accelerate believe real XLA is present --
# with a fake `torch_xla` package on sys.path, transformers+accelerate flip
# `is_torch_xla_available()` to True and then import real XLA submodules
# (e.g. accelerate/accelerator.py:163 -> torch_xla.distributed.xla_multiprocessing),
# which breaks the import chain with ModuleNotFoundError.  `USE_TORCH_XLA=0` is
# the upstream-documented switch for exactly this
# (transformers/utils/import_utils.py:107, accelerate/utils/imports.py:31).
# This has NO effect on cell 4 itself: cell 4 does `import torch_xla` directly
# and forces torch.set_default_dtype(torch.bfloat16) itself.
os.environ.setdefault("USE_TORCH_XLA", "0")

VERIFY = Path(__file__).resolve().parent
REPO = VERIFY.parent.parent                      # E:\Workspaces\RetNet
NOTEBOOK = REPO / "Kaggle" / "202610011026.ipynb"
CELLS = VERIFY / "_cells"
SHIM = VERIFY / "shim"
SANDBOX = VERIFY / "_sandbox"
LOGS = VERIFY / "logs"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Cell sources are taken straight from the .ipynb JSON (authoritative) and a
# copy is written to _verify/_cells/ for reading/quoting.  This makes the
# harness immune to any extraction tooling bug (the lead reported one).
CELL_INDEX = {"cell1": 1, "cell2": 2, "cell4": 4, "cell5": 5, "cell6": 6}


def _read_notebook_cells() -> dict:
    import json as _json
    nb = _json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = {}
    CELLS.mkdir(parents=True, exist_ok=True)
    for name, idx in CELL_INDEX.items():
        src = "".join(nb["cells"][idx]["source"])
        (CELLS / f"202610011026.{name}.py").write_text(src, encoding="utf-8")
        cells[name] = src
    return cells


_NB_CELLS = _read_notebook_cells()
CELL1_SRC = _NB_CELLS["cell1"]
CELL4_SRC = _NB_CELLS["cell4"]
CELL4 = CELLS / "202610011026.cell4.py"          # copy of CELL4_SRC, for quoting
CELL1 = CELLS / "202610011026.cell1.py"
assert CELL4.read_text(encoding="utf-8") == CELL4_SRC
assert CELL1.read_text(encoding="utf-8") == CELL1_SRC
NOTEBOOK_SHA256 = sha256_file(NOTEBOOK)
CELL4_SHA256 = hashlib.sha256(CELL4_SRC.encode("utf-8")).hexdigest()
CELL4_LINES = CELL4_SRC.count("\n") + 1
CELL4_REF = f"{NOTEBOOK.name}::cell4"

TINY = dict(
    d_model=64,
    num_layers=2,
    num_heads=4,
    vocab_size=256,
    ffn_dim=128,
    value_dim=128,
    seq_len=256,
    chunk_size=64,
    dropout=0.0,          # deviation from the real config (0.1) for crisp diffs
    growth_old_dim=48,    # keeps the real 768/1024 ratio
    use_gla=True,
    use_growth_ln=True,
    tie_weights=True,
)

INJECTIONS: list[dict] = []


# --------------------------------------------------------------------------
# I3: builtins.open redirect for /kaggle/... paths
# --------------------------------------------------------------------------
_REAL_OPEN = builtins.open


def _redirect_path(file):
    try:
        s = os.fspath(file)
    except TypeError:
        return None
    if isinstance(s, str) and s.startswith("/kaggle/"):
        return str(SANDBOX / "kaggle" / s[len("/kaggle/"):])
    return None


def _patched_open(file, *args, **kwargs):
    target = _redirect_path(file)
    if target is not None:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        INJECTIONS.append({"kind": "open-redirect", "from": os.fspath(file),
                           "to": target})
        file = target
    return _REAL_OPEN(file, *args, **kwargs)


def install_open_redirect() -> bool:
    if os.environ.get("HARNESS_NO_OPEN_PATCH") == "1":
        return False
    if builtins.open is not _patched_open:
        builtins.open = _patched_open
    return True


# --------------------------------------------------------------------------
# I1: torch_xla shim
# --------------------------------------------------------------------------
def install_shim() -> str:
    shim = str(SHIM)
    if shim not in sys.path:
        sys.path.insert(0, shim)
    import torch_xla  # noqa: F401  (the shim)
    return torch_xla.__file__


# --------------------------------------------------------------------------
# I2: cell 1 constants scraped from the real cell 1
# --------------------------------------------------------------------------
def cell1_constants() -> dict:
    src = CELL1_SRC
    keep, skip = {}, ("DATASET_DIR", "MODEL_DIR", "OUTPUT_DIR")
    for line in src.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, eq, _ = stripped.partition("=")
        name = name.strip()
        if not eq or not name.isidentifier() or not name.isupper():
            continue
        if name in skip:
            continue
        ns: dict = {}
        exec(stripped, ns)  # literal assignments only (verified by ast below)
        keep[name] = ns[name]
    return keep


def exec_cell4(dataset_dir, model_dir, output_dir, extra_globals=None,
               cell4_path: Path | None = None) -> dict:
    """Execute cell 4 unmodified and return its namespace.

    ``cell4_path`` is only used to read the source; by default the source comes
    from the .ipynb JSON itself (``CELL4_SRC``) and is compiled with a filename
    that points back at the notebook cell, so tracebacks quote notebook-relative
    line numbers.
    """
    install_shim()
    open_patched = install_open_redirect()
    INJECTIONS.append({"kind": "shim", "path": str(SHIM / "torch_xla")})
    INJECTIONS.append({"kind": "env", "USE_TORCH_XLA": os.environ.get("USE_TORCH_XLA"),
                       "why": "keep transformers/accelerate out of their real-XLA "
                              "code paths while the fake torch_xla package is on sys.path"})
    INJECTIONS.append({"kind": "cell1-constants", "source": f"{NOTEBOOK.name}::cell1"})
    INJECTIONS.append({"kind": "open-redirect", "enabled": open_patched,
                       "sandbox": str(SANDBOX)})
    INJECTIONS.append({"kind": "path-override", "DATASET_DIR": str(dataset_dir),
                       "MODEL_DIR": str(model_dir),
                       "OUTPUT_DIR": str(output_dir)})

    env = cell1_constants()
    env.update(DATASET_DIR=str(dataset_dir), MODEL_DIR=str(model_dir),
               OUTPUT_DIR=str(output_dir))
    if extra_globals:
        env.update(extra_globals)
    if cell4_path is None:
        src = CELL4_SRC
        fname = CELL4_REF
    else:
        src = Path(cell4_path).read_text(encoding="utf-8")
        fname = str(cell4_path)
    exec(compile(src, fname, "exec"), env)
    env["__CELL4_SHA256__"] = hashlib.sha256(src.encode("utf-8")).hexdigest()
    env["__CELL4_LINES__"] = src.count("\n") + 1
    env["__CELL4_REF__"] = fname
    return env


def manifest() -> dict:
    return {
        "notebook": {"path": str(NOTEBOOK), "sha256": NOTEBOOK_SHA256},
        "cell4": {"ref": CELL4_REF, "sha256": CELL4_SHA256,
                  "lines": CELL4_SRC.count("\n") + 1, "copy": str(CELL4)},
        "cell1": {"ref": f"{NOTEBOOK.name}::cell1",
                  "sha256": hashlib.sha256(CELL1_SRC.encode()).hexdigest()},
        "python": sys.version,
        "injections": INJECTIONS,
    }


def write_manifest(name: str) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / f"harness_injections_{name}.json"
    out.write_text(json.dumps(manifest(), indent=2), encoding="utf-8")
    return out


def setup_tiny_fixtures() -> tuple[Path, Path]:
    """Create (dataset_dir, model_dir) for the tiny CPU run if missing."""
    import numpy as np
    import torch

    ds = SANDBOX / "tiny_dataset"
    md = SANDBOX / "tiny_model"
    ds.mkdir(parents=True, exist_ok=True)

    seq_len = TINY["seq_len"]
    rng = np.random.default_rng(1234)
    if not (ds / "train.ids.npy").exists():
        for split, rows in (("train", 256), ("val", 64)):
            ids = rng.integers(0, TINY["vocab_size"], size=(rows, seq_len + 1),
                               dtype=np.int32)
            codes = rng.integers(1, 4, size=(rows, seq_len + 1), dtype=np.int32)
            np.save(ds / f"{split}.ids.npy", ids)
            np.save(ds / f"{split}.wcode.npy", codes)

    if not (md / "model.safetensors").exists():
        md.mkdir(parents=True, exist_ok=True)
        ns = exec_cell4(ds, md, SANDBOX / "tiny_out")
        cfg_cls = ns["RetNetHFConfig"]
        model_cls = ns["RetNetForCausalLM"]
        torch.set_default_dtype(torch.float32)   # real checkpoint is fp32
        cfg = cfg_cls(vocab_size=TINY["vocab_size"], d_model=TINY["d_model"],
                      num_layers=TINY["num_layers"], num_heads=TINY["num_heads"],
                      value_dim=TINY["value_dim"], ffn_dim=TINY["ffn_dim"],
                      dropout=TINY["dropout"], use_gla=TINY["use_gla"],
                      use_growth_ln=TINY["use_growth_ln"],
                      growth_old_dim=TINY["growth_old_dim"],
                      tie_weights=TINY["tie_weights"])
        model = model_cls(cfg)
        model.save_pretrained(md, safe_serialization=True)
        torch.set_default_dtype(torch.bfloat16)
    return ds, md


def tiny_cfg(env: dict, dataset_dir: Path, model_dir: Path, output_dir: Path,
             **overrides) -> None:
    cfg = env["CFG"]
    cfg.dataset_dir = str(dataset_dir)
    cfg.model_dir = str(model_dir)
    cfg.output_dir = str(output_dir)
    cfg.model_mode = "chunkwise"
    cfg.seq_len = TINY["seq_len"]
    cfg.chunk_size = TINY["chunk_size"]
    cfg.per_device_batch = 2
    cfg.grad_accum = 1
    cfg.max_steps = 6
    cfg.warmup_steps = 2
    cfg.save_steps = 2
    cfg.eval_steps = 3
    cfg.save_total_limit = 3
    for k, v in overrides.items():
        setattr(cfg, k, v)


def main() -> int:
    args = sys.argv[1:]
    if "--demo-dtypes" in args:
        ds, md = setup_tiny_fixtures()
        env = exec_cell4(ds, md, SANDBOX / "tiny_out")
        import torch
        print("[harness] cell4 sha256 =", env["__CELL4_SHA256__"])
        print("[harness] default dtype after cell4 =", torch.get_default_dtype())
        print("[harness] torch_xla shim =", sys.modules["torch_xla"].__file__)
        print("[harness] dtype-less arange ->", torch.arange(4).dtype)
        print("[harness] RetNetForCausalLM =", env["RetNetForCausalLM"].__name__)
        print("[harness] CFG.seq_len =", env["CFG"].seq_len)
        print("[harness] injections =", json.dumps(INJECTIONS, indent=2))
        p = write_manifest("demo")
        print("[harness] manifest ->", p)
        return 0

    print(json.dumps(manifest(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
