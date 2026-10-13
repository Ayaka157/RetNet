# `_verify` — CPU verification harness for `Kaggle/202610011026.ipynb` (task T1)

Owner: `cpu-verifier`. Read `FINDINGS_V1.md` first — it is the report. This file is the quick key.

## What this is

Cell 4 of the notebook (model + training loop, 992 lines) is executed **unmodified** on CPU:

* the cell source is read straight from `Kaggle/202610011026.ipynb` (JSON), so no extraction tool
  can change the line numbers or the code;
* a CPU stand-in for `torch_xla` lives in `shim/` and is put on `sys.path` (see the docstring in
  `shim/torch_xla/__init__.py` for exactly what it substitutes);
* `harness.py` records every injection into `logs/harness_injections_*.json`.

Injection summary (all documented, none hidden):

| id | injection | why |
|---|---|---|
| I0 | `USE_TORCH_XLA=0` in the environment | keep transformers/accelerate out of real-XLA code paths while the fake `torch_xla` package is importable |
| I1 | `shim/torch_xla/**` on `sys.path` | cell 4 does `import torch_xla` (line 14) and `from torch_xla.utils.checkpoint import ...` (line 21) |
| I2 | cell 1's constants scraped from cell 1 and injected; `DATASET_DIR/MODEL_DIR/OUTPUT_DIR` point at `_sandbox/` | cell 1 asserts absolute `/kaggle/...` paths |
| I3 | `builtins.open` redirect for paths starting with `/kaggle/` → `_sandbox/kaggle/` | cell 4 unconditionally writes `/kaggle/working/train_config.json` at import (line 645); disable with `HARNESS_NO_OPEN_PATCH=1` |

## Run it

```powershell
cd E:\Workspaces\RetNet\Kaggle\_verify
$py = "C:\Tools\Miniconda3\envs\AI\python.exe"
& $py harness.py --demo-dtypes    # prove cell 4 imports + the bf16 default dtype is active
& $py run_all.py                  # every repro, output tee'd into logs/
& $py run_tiny_train.py fresh     # real train_worker() end to end on CPU
& $py repro_d2.py                 # one defect at a time
```

`run_all.py` currently finishes all 15 scripts with exit 0 (the `e2e_resume` run is *expected* to
report `NameError` — that is D1).

## Script map

| script | claim |
|---|---|
| `repro_d1.py`, `run_tiny_train.py resume` | missing `import re` → resume crashes |
| `repro_d2.py` | bf16 rotary positions + bf16 `inv_freq` |
| `repro_d34.py` | dtype-less `arange` claims (refuted) |
| `repro_d5.py` | bf16 NaN in `evaluate` (cosmetic) |
| `repro_d6.py` | fp32 gate restore: buffer branch + value loss |
| `repro_d7.py` | `torch_xla.device()` kernel-global dependency |
| `repro_d8.py` | `world_size` NameError claim (refuted) |
| `repro_d9.py` | optimizer param-group mismatch on resume |
| `repro_sweep.py` | every tensor constructor in the cells |
| `repro_extras.py` | init-weights safety + checkpoint storage hazards |
| `repro_crash_forensics.py` | what the saved Kaggle crash log proves / does not prove |

## Limits

No TPU, no real `torch_xla`, no XLA compilation, no real collectives, no HBM limits. The shim's
`all_reduce` is an identity, `rendezvous`/`mark_step` are no-ops, and `checkpoint` wraps
`torch.utils.checkpoint`. Anything XLA-specific is marked *reasoned only* in `FINDINGS_V1.md`.
