# FINDINGS_V1 — independent CPU verification of `Kaggle/202610011026.ipynb` cell 4

Author: **cpu-verifier** (shared task `task-1`, T1). Status: complete for everything except the
TPU-only questions called out below.

**Evidence base.** Cell 4 is read **directly from the `.ipynb` JSON** and executed **unmodified** on
CPU (`compile()` + `exec()` of the cell source, no rewrite) with a documented `torch_xla` shim.

| artifact | sha256 |
|---|---|
| `E:\Workspaces\RetNet\Kaggle\202610011026.ipynb` | `39c42904d3ddaf85feceb815...` (identical prefix to the original attachment) |
| `202610011026.ipynb::cell4` (992 lines) | `0988426a50978413b501373e...` |
| `202610011026.ipynb::cell1` | `c1726efc85a53ff610916ba7...` |

> Note on extraction: `_extract\202610011026.ipynb.cellN.py` is **byte-identical** to the notebook
> JSON except for one trailing newline added by the extractor (verified by comparison, see
> `logs/`), and the paths the lead quoted later (`_extract\202610011026.cell4.py`) do not exist.
> `harness.py` therefore extracts cells itself from the notebook, so my line numbers cannot be
> affected by any extraction tool. The fix cell 4 (`202610011026_fix.ipynb::cell4`) is 1020 lines,
> sha256 `25286425cb07c735f1051b10f20e7f60`, and matches `_extract_fix\fix.cell4.py` exactly.

Every quoted output below is reproduced from the archived run logs in `_verify/logs/*.txt`.

---

## HARNESS STATUS

**What the harness can execute today (proven by `run_all.py`: all 14 scripts exit 0 except the
deliberate failure run):**

* cell 4 **unmodified**: class definitions, `RetNetHFConfig`, `MultiScaleRetention`,
  `GrowthLayerNorm`, `RetNetBlock`, `RetNetForCausalLM`, `PackedChatDataset`, `ChatDataCollator`,
  `TrainConfig`/`CFG`, `_latest_checkpoint`, `_save_checkpoint`, `_load_checkpoint`,
  `CosineWithMinLRScheduler`, `evaluate`, `train_worker`.
* the **whole `train_worker()` end to end** on CPU with a tiny fixture model/dataset
  (`_verify/_sandbox`): chunkwise forward + backward, block-level gradient checkpointing
  (`_xla_checkpoint` shim, both non-reentrant and reentrant), `clip_grad_norm_`,
  `xm.optimizer_step`, LR schedule, eval, checkpoint save/prune/load, resume, and the final
  `save_pretrained` HF export with tied weights.
* `torch.set_default_dtype(torch.bfloat16)` (cell 4 line 23) is really in effect during every run,
  so dtype defects reproduce exactly as on the TPU.

**What it cannot execute / cannot prove locally:**

* no TPU and no real `torch_xla`: **no** XLA compilation, no lazy-graph behaviour, no HBM limits,
  no real collectives (the shim's `all_reduce` is an identity), no `xla` device, no PJRT.
* no XLA-version-specific failure modes (compile crashes, layout pinning errors, libtpu aborts).
* the shim's `checkpoint` wraps `torch.utils.checkpoint`; it does not reproduce torch_xla's
  `optimization_barrier_`/RNG handling.
* `USE_TORCH_XLA=0` is exported by the harness (documented injection I0) so that the fake
  `torch_xla` package does not drag transformers/accelerate into their real-XLA code paths
  (without it, `import transformers` dies with
  `ModuleNotFoundError: No module named 'torch_xla.distributed.xla_multiprocessing'`).
* cell 1 is **not** executed verbatim: its constants are scraped from cell 1's source and injected,
  and the three `/kaggle/...` paths are overridden to local fixtures (cell 1 asserts absolute
  Kaggle paths). Cell 4's unconditional
  `open("/kaggle/working/train_config.json", "w")` is redirected to
  `_verify/_sandbox/kaggle/working/` because on Windows `/kaggle/...` resolves to
  `<drive>:\kaggle\...`; set `HARNESS_NO_OPEN_PATCH=1` to see the raw failure.

**Commands (all from `E:\Workspaces\RetNet\Kaggle\_verify`, python =
`C:\Tools\Miniconda3\envs\AI\python.exe`):**

```powershell
cd E:\Workspaces\RetNet\Kaggle\_verify
& C:\Tools\Miniconda3\envs\AI\python.exe harness.py --demo-dtypes      # manifest + injection log
& C:\Tools\Miniconda3\envs\AI\python.exe repro_d1.py                   # D1
& C:\Tools\Miniconda3\envs\AI\python.exe repro_d2.py                   # D2
& C:\Tools\Miniconda3\envs\AI\python.exe repro_d34.py                  # D3/D4 (refuted)
& C:\Tools\Miniconda3\envs\AI\python.exe repro_d5.py                   # D5
& C:\Tools\Miniconda3\envs\AI\python.exe repro_d6.py                   # D6
& C:\Tools\Miniconda3\envs\AI\python.exe repro_d7.py                   # D7
& C:\Tools\Miniconda3\envs\AI\python.exe repro_d8.py                   # D8 (refuted)
& C:\Tools\Miniconda3\envs\AI\python.exe repro_d9.py                   # D9 (optimizer groups)
& C:\Tools\Miniconda3\envs\AI\python.exe repro_sweep.py                # constructor sweep
& C:\Tools\Miniconda3\envs\AI\python.exe run_tiny_train.py fresh       # e2e train + export
& C:\Tools\Miniconda3\envs\AI\python.exe run_tiny_train.py resume      # e2e resume -> D1
& C:\Tools\Miniconda3\envs\AI\python.exe run_tiny_train.py resume-re   # resume with `re` injected
& C:\Tools\Miniconda3\envs\AI\python.exe repro_crash_forensics.py      # BrokenProcessPool forensics
& C:\Tools\Miniconda3\envs\AI\python.exe run_all.py                    # all of the above -> logs/
```

---

## 1. Claim table

| id | claim | location (notebook cell 4) | method | status |
|---|---|---|---|---|
| D1 | `_latest_checkpoint` calls `re.fullmatch`, `re` never imported | 654–660 (`re.fullmatch` at **657**) | proven by execution | **CONFIRMED** — trigger is *resume*, not a fresh run |
| D2a | rotary positions cast to bf16 → phases destroyed at seq 8192 | 118–123 (**120**) | proven by execution | **CONFIRMED** (severe) |
| D2b | `self.inv_freq` registered as bf16 → drifts from fp32 rotary freqs | 95–97 | proven by execution | **CONFIRMED** (secondary) |
| D3 | dtype-less `torch.arange` in `_decay_mask` inherits bf16 | 148 | proven by execution | **REFUTED** (int64, bitwise identical output) |
| D4 | dtype-less `torch.arange` in `_parallel_final_state` inherits bf16 | 168 | proven by execution | **REFUTED** (int64, bitwise identical output) |
| D5 | `torch.tensor(float("nan"))` inherits bf16 in `evaluate` | 746 | proven by execution | **CONFIRMED as latent/cosmetic only** |
| D6a | fp32 gate restore skips the `use_gla=False` **buffer** branch | 839–842 vs 93–94 | proven by execution | **CONFIRMED** (branch not exercised by the shipped config) |
| D6b | the restore fixes the dtype but **not the values** | 834–842 | proven by execution | **CONFIRMED (new)** — stronger than the brief; relevant to the shipped config |
| D6c | `model.to(bf16)` collapses 12/16 heads to exactly 1.0 | 83, 834 | proven by execution | **PARTIALLY CONFIRMED**: true for the code's init formula, **false for the shipped checkpoint's trained gates** |
| D7 | `torch_xla.device()` depends on the kernel-global `import torch_xla` | 14 vs 759–766 | proven by execution + artifact | **CONFIRMED as a kernel-global dependency; it does NOT crash under the real fork launcher** |
| D8 | `world_size` unbound on ranks 1..7 → NameError at step 1 | 764 vs 934 | proven by execution + AST + crash log | **REFUTED** |
| D9 | 2-group optimizer vs 1-group loader → `ValueError` on resume | 701–702 vs 853–863 | proven by execution both ways | **PARTIALLY CONFIRMED**: real cross-code-path hazard; **not reachable** from the notebook itself |
| D10 | constructor sweep (proactive) | see §4 | proven by execution | 4 poisoned sites, 2 benign, 1 reasoned-only |

---

## 2. Confirmed defects — verbatim evidence

### D1 — `_latest_checkpoint` crashes on resume (`import re` missing) — **PROVEN**

Location: cell 4 lines 654–660; the `re` reference is line 657. Cell 4's complete import list
contains no `re` (AST-extracted). `E:\...\_verify\logs\d1_latest_checkpoint.txt`:

```
  cell4 module namespace has 're' ? False
--- case B: output_dir contains a checkpoint dir (the RESUME scenario) --------
  files: ['checkpoint-500']
  NameError: name 're' is not defined
    File "202610011026.ipynb::cell4", line 657, in _latest_checkpoint
  NameError: name 're' is not defined. Did you forget to import 're'?
```

Control run with `re` injected by the harness returns `(1000, .../checkpoint-1000)`.
End-to-end via the real `train_worker()` (`logs/e2e_resume.txt`):

```
!!! train_worker() FAILED: NameError: name 're' is not defined
   File "202610011026.ipynb::cell4", line 874, in train_worker
   File "202610011026.ipynb::cell4", line 657, in _latest_checkpoint
   NameError: name 're' is not defined. Did you forget to import 're'?
```

Empty output dir → `(None, None)` and **no** error (`logs/d1_latest_checkpoint.txt`, case A), which
is why a fresh Kaggle session never reports it. Any `checkpoint-*` **file or dir** triggers it,
even one that does not match the regex (cases C and D) — the name lookup happens before the match.
Severity: **A for resume**, not the cause of the current step-1 crash.

### D2 — bf16 rotary positions and bf16 `inv_freq` — **PROVEN**

Location: lines 118–123 (`position = torch.arange(offset, offset+length, device=device, dtype=dtype)`
at 120 with `dtype = q.dtype = bf16`) and lines 95–97. `logs/d2_rotations_bf16.txt`:

```
  torch.arange(0, 8193, dtype=torch.bfloat16)  dtype=torch.bfloat16
  max |position error| over 0..8192 = 31 positions
  number of distinct wrong positions = 7296 / 8193
  first index where error != 0     : 257
  position | exact | bf16 value | abs err
        256 |    256 |      256.0 |     0.0
        257 |    257 |      256.0 |     1.0
       8191 |   8191 |     8192.0 |     1.0
  max |sin - sin_exact| = 2.0000   max |cos - cos_exact| = 2.0000
  mean |sin - sin_exact| = 0.2331
  entries with |sin err| > 0.1 : 177900 / 524288 (33.93%)
  entries with |sin err| > 0.5 : 86086 / 524288 (16.42%)
  per-(batch,head,token) relative L2 deviation of q_rot:
    max  = 1.5200   mean = 0.6509
    #tokens with deviation > 10% : 253241 / 262144
```

`inv_freq` (lines 95–97) is registered **bf16 and non-persistent** (so it is not in the checkpoint
and never restored):

```
  inv_freq dtype in the notebook = torch.bfloat16 (persistent=False)
  max relative inv_freq error = 2.200e-03 at index 4
  resulting phase drift at position 8192: max 9.9466 rad, mean 0.7226 rad
  #frequencies with phase drift > 0.1 rad = 24 / 64
```

Both halves match the lead's independently measured numbers (7296 errors, max 31, max |sin| error
2.0). Note (measured, mechanism not explained): in the tail `torch.arange(..., dtype=torch.bfloat16)`
rounds onto the coarse grid (e.g. `4096..4127 → 4096.0`), so the error reaches a full bf16 spacing
(31) rather than half of it (16); per-element casts of the same integers differ from `arange`'s
output (probe output in the log).

### D5 — `torch.tensor(float("nan"), device=device)` is bf16 — **PROVEN DTYPE, NO IMPACT**

`logs/d5_evaluate_nan.txt`:

```
  line 746 as written : dtype=torch.bfloat16   value=nan
  with dtype=float32  : dtype=torch.float32    value=nan
  float(x.cpu()) -> bf16:nan fp32:nan identical=True
  x != x         -> bf16:tensor(True) fp32:tensor(True) identical=True
  torch.isnan(x) -> bf16:tensor(True) fp32:tensor(True) identical=True
  x < float('inf')-> bf16:tensor(False) fp32:tensor(False) identical=True
  all_reduce(REDUCE_SUM, bf16 nan) -> dtype=torch.bfloat16 value=nan
```

Classification: **latent / type-consistency only**. It is also only reachable when the eval loader
yields zero batches (`if batches == 0:`, line 745); with a non-empty loader line 748 is used.
The old `dtype=torch.float32` "fix" was correct hygiene, nothing more.

### D6 — the fp32 decay-gate restore — **CONFIRMED, but the magnitude claim needs correcting**

Location: lines 83–97 (gate creation) and 834–842 (restore loop). `logs/d6_gamma_restore.txt`:

**(a) the buffer branch is never restored** (`use_gla=False`, gamma is a *persistent buffer*):

```
  gamma dtype=torch.float32  isinstance(nn.Parameter)=False  persistent in state_dict=True
  after model.to(bfloat16) + the notebook restore loop:
    gamma dtype=torch.bfloat16 (NOT restored)  #==1.0: 12/16
    gamma is an optimizer parameter? False
```

**(b) the Parameter branch's "fp32 master weight" restore does not restore values** — proven on
the notebook's own path (`from_pretrained` → `.to(device, dtype=torch.bfloat16)` → lines 839–842),
with gates initialised to the code's own formula:

```
  [init-formula gates] 1) after from_pretrained: gamma dtype=torch.float32, #==1.0: 0/16
                       2) after model.to(dtype=torch.bfloat16): dtype=bfloat16, #==1.0: 12/16
                       3) after the notebook restore loop: dtype=torch.float32, #==1.0: 12/16
        values bitwise equal to the pre-.to() fp32 master? False
        max |gamma_after - gamma_input| = 1.953e-03
```

Per-head table (excerpt): heads `i>=4` (i.e. `1-2**(-5-i)`, d = 2**-9 …) round to exactly `1.0`,
and `gamma**64 = 1.0` for those heads → **chunk_decay = 1.0, i.e. no cross-chunk decay at all**.
12/16 = heads `[4..15]`. My earlier arithmetic (11) was wrong; the executed result agrees with the
briefing's 12.

**(c) magnitude — the shipped checkpoint is different.** `blocks.*.msr.gamma` in
`retnet_272m_init_weights` is **already trained** (fp32, 0.786 … 0.9923, max 0.9922732), not the
init formula:

```
   heads == exactly 1.0 after a bf16 cast: 1/320   -> the 12/16 collapse does NOT happen
   bf16 rounding error: max rel 3.860e-03, mean rel 1.025e-03
   block0 heads 12..15 fp32 : [0.8862733, 0.8862768, 0.8862786, 0.8862795]
   block0 heads 12..15 bf16 : [0.88671875, 0.88671875, 0.88671875, 0.88671875]
   relative error of chunk_decay = gamma**64 after the bf16 cast: max 2.797e-01
```

So for **this** checkpoint: no head is pinned to 1.0, but the gates are quantized (up to 0.39 %
relative, 28 % error in `gamma**64`) and 4 heads are merged into one identical value. The 12/16
collapse does happen if the gates are at their initialisation values (fresh-init path / any run
that does not load these trained gates). Impact measured on chunkwise logits (init-formula case):

```
  ||logits(12 dead gates) - logits(exact fp32 gates)|| / ||exact|| = 0.0127
  max |logit difference| = 0.019485
```

Also worth noting: `gamma` comes out of `from_pretrained` **fp32 even though the default dtype is
bf16**, because line 83 constructs it with an explicit `dtype=torch.float32`; every other weight is
created in bf16 and the fp32 checkpoint values are rounded at load. That is the intended
global-bf16 strategy — but it means the only genuinely-fp32 master weight in the model loses its
precision at line 834 and is never recovered.

### D7 — `torch_xla.device()` is a kernel-global dependency — **CONFIRMED (not a crash)**

`logs/d7_torch_xla_device.txt`:

```
  imports inside train_worker: [['xm'], ['xr'], ['pl']]
  -> binds ['xm', 'xr', 'pl'] ; NOT the name 'torch_xla'
  'torch_xla' in train_worker.__globals__ (after cell 4 exec): True
  module-level imports: ['json', 'os', 'np', 'torch', 'nn', 'F', 'torch_xla', 'math', 'random', 'time', 'torch']
```

Deleting the global and calling `train_worker()` reproduces the dependency:

```
  NameError: name 'torch_xla' is not defined
    File "202610011026.ipynb::cell4", line 766, in train_worker
```

Why it does **not** crash in the real run: `accelerate/launchers.py` TPU branch does
`xmp.spawn(launcher, args=args, start_method="fork")` (also visible in the saved Kaggle traceback),
and fork inherits the notebook kernel's module globals. Independent artifact proof: the stream's
second line is `[XLA] world_size=8 device=xla:0`, printed at line 772 from the object built at line
766 — so the name resolved inside a worker. Classification: robustness issue,
**not** the step-1 crash. Hardening: `import torch_xla` inside `train_worker`.

### D8 — `world_size` on non-master ranks — **REFUTED**

`logs/d8_world_size_refuted.txt`: AST proof places the binding at line 764, column 4, in the
function body with no enclosing `if`, while line 770 opens the master guard:

```
   763:     rank = xr.global_ordinal()
   764:     world_size = xr.world_size()
   765:     # device = xm.xla_device()
   766:     device = torch_xla.device()
   767:
   768:     _seed_everything(CFG.seed + rank)
   769:
   770:     if xm.is_master_ordinal():
```

Execution with the shim as **rank 1 of 8** (`XLA_SHIM_WORLD_SIZE=8 XLA_SHIM_ORDINAL=1`, and the
shim's `is_master_ordinal()` forced True on that rank so the logging block is observable):

```
[XLA] world_size=8 device=cpu model_mode=chunkwise batch/device=2
[train] step=     1 loss=0.69352 lr=1.100e-04 elapsed=0.0m
[train] step=     2 loss=0.69423 lr=1.100e-04 elapsed=0.0m
  train_worker() COMPLETED as rank 1/8 -> no NameError
```

Independent artifact proof: `0909_cell6_stream.txt` contains the deprecated
`xm.mark_step()` warning at `109047568.py:935` **exactly 8 times**, and line 935 sits three lines
after the `/ world_size` use at line 934. The current notebook has `xm.mark_step()` at
`[751, 935, 975]`; the older `202609092100` revision has it at `[759, 946, 986]` — so the crashing
run was the **current** notebook, and all 8 ranks got past line 934. `world_size` is not the cause.

### D9 — optimizer parameter-group mismatch — **CONFIRMED as a hazard, NOT reachable in-notebook**

`logs/d9_optimizer_groups.txt`:

```
--- part A: the notebook's own round trip (2 groups -> 2 groups) --------
  built optimizer has 2 param groups, state entries=41
  _load_checkpoint with matching 2-group optimizers -> OK (step=1, best=1.0)
--- part B: the 1-group-vs-2-group mismatch the lead reproduced --------
  1-group optimizer has 1 param group(s)
  ValueError: loaded state dict has a different number of parameter groups
  File "202610011026.ipynb::cell4", line 702, in _load_checkpoint
```

Part C (AST sweep of cell 4, plus reading cells 1/2/5/6): cell 4 constructs `AdamW` exactly once
(line 855, two groups) and `_load_checkpoint` is called exactly once (line 876) with that same
optimizer, so the notebook's own resume path does not hit it. The error only appears when the
checkpoint was written by code with a different grouping — e.g. the lead's acceptance harness.
`build_optimizer()` in the fix removes the hazard; it does not fix a live notebook failure.

---

## 3. Refuted claims — verbatim evidence

### D3/D4 — dtype-less `torch.arange` in `_decay_mask` / `_parallel_final_state` — **REFUTED**

`logs/d3_d4_arange_refuted.txt`:

```
  torch.arange(8, device='cpu')          -> torch.int64      INT-EXACT
  torch.arange(8192, device='cpu')       -> torch.int64      INT-EXACT
  torch.arange(0, 8, dtype=torch.bfloat16)-> torch.bfloat16   bf16-poisoned=torch.bfloat16
  torch.zeros(4)                         -> torch.bfloat16   bf16-poisoned=torch.bfloat16
  line 148 exactly: idx dtype = torch.int64
  max |delta_notebook - delta_int64| over an 8192x8192 pair matrix = 0
  identical to the explicit-int64 result: True
  length=  256  bitwise identical to fp32-index reference: True   max abs mask diff = 0.000e+00
  length=  512  bitwise identical to fp32-index reference: True   max abs mask diff = 0.000e+00
  length= 1024  bitwise identical to fp32-index reference: True   max abs mask diff = 0.000e+00
  length= 4096  bitwise identical to fp32-index reference: True   max abs mask diff = 0.000e+00
  tgt_len=4096  bitwise identical to fp32-index reference: True   max abs state diff = 0.000e+00
```

`torch.set_default_dtype()` changes only the **floating** default dtype; integer-argument
`arange` returns `int64`, so lines 148/168 are exact and the explicit `.to(torch.float32)` at lines
151/170 costs nothing. The 2026-09-09 change to these two lines was a no-op, and reverting it is
**not** a regression. (Contrast with D2 line 120, which *does* pass the compute dtype explicitly.)

### D8 — see above. **REFUTED** (AST + rank-1 execution + the 8× line-935 artifact).

---

## 4. Proactive sweep — every constructor in cells 1/2/4/5/6

Measured dtype-inference facts with cell 4's bf16 default active (`logs/sweep_constructors.txt`):
`arange(int)`, `tensor(int)`, `randint` → **int64 (immune)**; `zeros/ones/full/empty/eye/rand/randn`,
`tensor(float)`, `linspace`, `arange(float args)` → **bf16**.

Complete list of constructor sites in cell 4 (`_like` forms inherit their input dtype):

| site | verdict |
|---|---|
| 83 `torch.pow(2.0, -5.0 - torch.arange(num_heads, dtype=torch.float32))` | explicit fp32, OK |
| 95–97 `torch.linspace(..., dtype=torch.float32)` then `.to(torch.bfloat16)` | explicit, but **D2b** |
| 120 `torch.arange(offset, offset+length, dtype=dtype)` | **POISONED (D2a)** |
| 148 `torch.arange(length, device=device)` | int64, exact (D3 refuted) |
| 168 `torch.arange(tgt_len, device=k.device)` | int64, exact (D4 refuted) |
| 181 `torch.zeros(..., dtype=q.dtype)` | explicit, OK |
| 207–209 `torch.zeros(..., dtype=torch.bfloat16)` | explicit; hard-coded bf16 (see §4.3) |
| 218 `torch.arange(chunk_size, ..., dtype=torch.float32)` | explicit fp32, OK |
| 304–305 `torch.full((n,), gamma_new)` / `(n,), beta_new)` | default dtype bf16, **benign** (1.0/0.0 exact) |
| 469 `torch.zeros_like(wcodes, dtype=torch.float32)` | explicit fp32, OK |
| 470/472 `torch.ones_like(w)`, 471 `torch.full_like(w, ...)` | inherit fp32 `w`, OK |
| 531 `torch.zeros_like(probs)` | inherits bf16 probs, OK |
| 544/550 `torch.tensor(..., dtype=torch.long)` | explicit, OK |
| 729/904 `torch.zeros((), device=device, dtype=torch.float32)` | explicit fp32, OK |
| 746 `torch.tensor(float("nan"), device=device)` | **POISONED (D5), no impact** |

Cells 1/2/5/6 contain no tensor constructors (cell 5 only builds a config + model and prints
`state_dict()` keys).

### 4.1 New finding — `GrowthLayerNorm` parameters are created in bf16 (benign)

Lines 304–305 have no `dtype`; executed result is `torch.bfloat16` for both `gamma_new` and
`beta_new`. Benign because the initialisation values are exactly `1.0`/`0.0` (bf16-exact), and the
forward path upcasts to fp32 explicitly (line 315). Same class of thing for every
`nn.LayerNorm`/`GroupNorm`/`Linear` weight initialised to 1.0/0.0. The poisoning only bites values
with more than 8 significand bits — which is exactly why D2a/D2b and D6b matter.

### 4.2 New finding — hard-coded `torch.bfloat16` in the chunkwise path (reasoned only)

Lines 209, 211 (state) and 222–226 (`inner_mask`, `cross_decay`, `value_decay`, `chunk_decay`) cast
to `torch.bfloat16` unconditionally, ignoring `q.dtype`. In the real run everything is bf16, so this
cannot fail; under any fp32 execution (or an fp32 XLA run without `XLA_USE_BF16`) the recurrence
silently degrades to bf16 masks. **Reasoned only** — no crash reproduced.

### 4.3 Checked and *not* defects (so nobody re-fixes them)

`logs/extras_init_and_storage.txt` backs E1/E2/E5 below by execution:

* `MultiScaleRetention._init_weights()` does **not** clobber the gate: with `gamma` forced to 0.5,
  `_init_weights()` leaves `gamma` and `group_norm.weight/bias` bitwise unchanged while
  re-initialising `q_proj.weight` (`gamma untouched: True`, `q_proj.weight was re-initialised: True`).
  Model construction also produces exactly `1-2**(-5-i)` (`bitwise equal: True`).
* `_latest_checkpoint` on an empty dir returns `(None, None)` and `train_worker` guards it with
  `if latest_dir is not None:` (line 875) — the `None` handling is correct.
* `_save_checkpoint` passes a **nested** dict to `xm.save`. torch_xla 2.9.0's
  `save()` → `_maybe_convert_to_cpu` → `ToXlaTensorArena` walks nested containers, so nested XLA
  tensors are converted. Source-verified, not assumed.
* The final `save_pretrained(state_dict=..., safe_serialization=True)` with tied weights works: the
  tiny run's export has 41 tensors, `lm_head.weight` deduplicated, and
  `from_pretrained` reloads with `lm_head.weight is embedding.weight → True`
  (`logs/e2e_fresh.txt`).
* Gradient checkpointing (`_xla_checkpoint` on `_block_forward` returning a `(y, state)` tuple with
  non-tensor args) executes and backprops correctly in both non-reentrant and reentrant modes.
* `clip_grad_norm_` over bf16 params, `xm.optimizer_step`, eval with `all_reduce`, checkpoint
  pruning (`int(p.name.split("-")[-1])`) and resume from a pruned set all execute cleanly.

### 4.4 New finding — two storage-path hazards in the checkpoint code (low severity)

`logs/extras_init_and_storage.txt`:

```
  int('checkpoint-tmp'.split('-')[-1])   -> ValueError: invalid literal for int() with base 10: 'tmp'
  int('checkpoint-'.split('-')[-1])      -> ValueError: invalid literal for int() with base 10: ''
  int('checkpoint-1-final'.split('-')[-1]) -> ValueError: invalid literal for int() with base 10: 'final'
```

`_save_checkpoint` lines 681–682 glob `checkpoint-*` and sort by
`int(p.name.split("-")[-1])`, so **any** matching directory with a non-numeric suffix (a manual
copy, an interrupted write) makes the pruning loop raise `ValueError` before it deletes anything.
The same glob makes D1 fire earlier on any such entry. Separately, pruning removes children with
`child.unlink()` (lines 686–687), which cannot remove a directory:

```
  PermissionError: [WinError 5] ... 'checkpoint-99\nested'
  (POSIX raises IsADirectoryError here; this Windows host reports PermissionError [WinError 5].)
```

Both require unexpected filesystem content; neither is the step-1 crash. **Proven by execution**
for the expressions, **reasoned** for reaching them from the notebook.

---

## 5. Crash forensics — `BrokenProcessPool` (**UNDETERMINED**, reasoned)

`logs/crash_forensics.txt`. What the saved artifacts prove:

1. **The crashing run was the current notebook.** The deprecated `xm.mark_step()` warning is at
   `109047568.py:935`; the current cell 4 has `xm.mark_step()` at `[751, 935, 975]`, the older
   revision at `[759, 946, 986]`.
2. **All 8 workers reached line 935** (`':935'` appears 8 times, `[XLA]` banner once,
   `[train] step=` **zero** times, `Traceback` **zero** times) — hence all 8 evaluated
   `... / world_size` at line 934 (D8 refuted a second time, from the artifact).
3. **The failure was native, not a Python exception.** One worker (PID 169) reports
   `Signal 15 raised at PC ... while already in FailureSignalHandler!`, i.e. it was already dying of
   a fault when the kernel (PID 14) SIGTERMed the six others. A Python exception in a child would
   come back through `ProcessPoolExecutor` as a `RemoteTraceback`; the dump shows only
   `BrokenProcessPool`.
4. **It happened at the first XLA materialization.** torch_xla 2.9.0's
   `optimizer_step(barrier=False)` (line 923) does **not** sync — it only calls `torch_xla.sync()`
   when `barrier=True` — so the first sync of step 1 is line 935. The master's `[train] step=1`
   print sits after it and never appeared.

Candidates (all **reasoned only**, none provable locally): HBM exhaustion while materializing the
whole step-1 graph (20 blocks × 128 unrolled chunks, 272M bf16 params + grads + 2 Adam states);
an XLA compile failure of that graph; or the documented `pin_layout=True` compile risk
(torch_xla docstring: *"Layout pining ... might cause some xla compilation to fail. Unpin the
layout when you see error message like 'HloModule has a mix of layout constrained'"* — cell 4 uses
the default `pin_layout=True` at lines 750 and 932).

What the next TPU run must capture: the **dying worker's stderr to a file** (the kernel stream lost
the fatal message); `PYTHONFAULTHANDLER=1`, `TORCH_SHOW_CPP_STACKTRACES=1`, `XLA_IR_DEBUG=1`,
`XLA_HLO_DEBUG=1`; `xm.get_memory_info()` (`bytes_used`/`bytes_limit`/`peak_bytes_used`) before and
after the first sync; a bisect with `xm.mark_step()` after `model.to(device)` / after the first
forward / after backward; a size-shrink sweep (`max_steps=1`, `per_device_batch=1`, `seq_len` 1024
or 2048, gradient checkpointing on/off) to separate OOM (size-dependent) from a compiler bug
(size-independent); `pin_layout=False` at lines 750/932; and one run without `XLA_USE_BF16=1`.

---

## 6. Where I disagree with the lead's positions

| lead's position | my verdict |
|---|---|
| D1 real, trigger = resume | **agree** (proven end-to-end) |
| D2 real, 7296/8192 positions, max err 31, max sin err 2.0 | **agree exactly** (independently measured) |
| D5 cosmetic | **agree** (proven no behavioural difference) |
| D6/F2 "restore only covers `nn.Parameter`" | **agree, and it is worse than stated**: on the `use_gla=True` branch the restore cannot recover the values either (D6b). Also the "12 of 16 heads collapse" refers to the init formula, **not** to the shipped checkpoint's trained gates (1/320 there). |
| D3/D4 refuted | **agree** (bitwise-identical outputs) |
| D8 refuted | **agree** — plus the artifact's 8× line-935 warnings independently prove all 8 ranks passed line 934 |
| D9 = real defect (2 vs 1 param groups) | **partially disagree**: the `ValueError` is real and reproducible, but the notebook builds the 2-group optimizer exactly once (line 855) and calls `_load_checkpoint` once (line 876) with that same object, so the **notebook itself cannot hit it**. It is a cross-code-path hazard (e.g. an external harness writing the checkpoint). |
| crash = UNDETERMINED | **agree**; §5 adds the revision identification, the "native fault, not a Python exception" evidence, and the first-sync location |

Not reproduced / could not determine: the `BrokenProcessPool` root cause (no TPU); any
XLA-specific defect (compile crash, collective behaviour, HBM limits); whether the TPU run's
`_save_checkpoint` at step 500 and `xm.rendezvous` behave as intended (source-verified only).

---

## 7. Files written (all inside `E:\Workspaces\RetNet\Kaggle\_verify\`)

```
harness.py                 # executes cell 4 unmodified; documented injections; manifest
vcommon.py                 # shared helpers for the repro scripts
shim\torch_xla\...         # CPU torch_xla shim (device/core.xla_model/runtime/utils.checkpoint/
                           #   distributed.parallel_loader/distributed.xla_multiprocessing)
repro_d1.py repro_d2.py repro_d34.py repro_d5.py repro_d6.py repro_d7.py repro_d8.py
repro_d9.py repro_sweep.py repro_crash_forensics.py
run_tiny_train.py          # fresh | resume | resume-re
run_all.py                 # runs everything, tees to logs/
_cells\202610011026.cell{0..6}.py   # extraction straight from the notebook JSON
_sandbox\...               # tiny dataset, tiny fp32 model, run outputs (throwaway)
logs\*.txt                 # verbatim outputs quoted above
FINDINGS_V1.md             # this file
```

Nothing outside `_verify\` was written or modified; the notebook, the extracted cells and the fix
notebook were read-only inputs.
