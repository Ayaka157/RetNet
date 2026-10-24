# FIX LOG — 20261024_Submit_pre-audit

Scope: every implementation defect in `_audit_fixed/AUDIT_REPORT.md` (6 × P0, 9 × P1, 11 × P2).
Method: read the report, then read the code before changing it; for each item re-check the
report's claim against the actual source, fix the **shared** root cause, and add a test that
fails if the fix is reverted. Documentation claims that were false were regenerated from the
code or rewritten against it. All checks below were re-run at the end from this directory.

Environment: Python 3.13.7, torch 2.10.0+cu128, transformers 4.57.6 (`C:\tools\miniconda3\envs\AI`),
CPU-only for the numeric/structure checks. No data/checkpoint claims are asserted here — those
are marked UNSURE where they depend on files not present in the snapshot.

Final state: `python verify.py` → 9/9 checks pass; `cd code && python -m pytest tests -q` →
**82 passed, 4 skipped**; both script-style tests print `all pass`.

---

## P0 findings

### P0-1 — E22 “chunked CE” did not chunk at the documented shape
- **Verdict**: CONFIRMED, exactly as stated.
- **Root cause**: `chunk_tokens` was a fixed *token count* (262 144). It is independent of
  `V` (the factor that makes the fp32 copy big) and larger than the production token count
  (`B·L = 32 768`), so the loop ran once and the 2.6 GB `[N,V]` fp32 copy remained.
- **Fix**: derive the chunk from a byte budget —
  `ce_chunk_tokens(V, chunk_bytes) = chunk_bytes // (4V)`; default 64 MB ⇒ 838 tokens/chunk
  at `V=20000`, 40 chunks at the production shape. Exposed as
  `RetNetHFConfig.ce_chunk_bytes` (persisted in `config.json`) and as an explicit
  `chunk_tokens` override for tests. Docstring contradiction removed.
- **Files**: `code/retnet_hf/modeling_retnet.py`.
- **Verification**: `tests/test_regressions_extra.py::test_ce_chunk_size_is_derived_from_a_byte_budget`
  (838 tokens/chunk at V=20000) and `::test_chunked_ce_splits_at_production_token_count`
  (spies on `F.cross_entropy` at `n=32768`, asserts ≥2 chunks, each ≤ budget, all tokens covered).
  The old test (`chunk_tokens=7`) still passes but is no longer the only guard.
- **Residual risk**: the budget bounds the fp32 logits slice only; CE internals
  (`log_softmax`) need roughly another copy. Peak is O(budget), not exactly 64 MB.

### P0-2 — `_preflight` made the documented rescue plan unstartable
- **Verdict**: CONFIRMED. `ratio < 0.95 → SystemExit` with no bypass; the plan starts from a
  33%-live checkpoint.
- **Root cause**: the guard conflated “fresh model with broken growth/init” (must refuse) with
  “deliberately pruned/grown checkpoint” (known state). It also had no baseline, so it could
  not tell whether *this* run made things worse.
- **Fix**: pure decision function `check_live_ratio(ratio, threshold, allow_dead, baseline)`:
  pass if above threshold, or explicit `--allow_dead_params`, or within 2% of a recorded
  baseline; otherwise refuse with both remedies in the message. `_preflight` writes
  `live_params.json` after a successful start and reads it from `model_dir`/`output_dir`;
  the file stores `ratio_min` (historical minimum) so repeated sub-threshold erosion cannot
  re-base the baseline forever. `docs/02_resume_plan_220cny.md` §2 now tells the operator to pass the flag.
- **Files**: `code/retnet_hf/train.py`, `docs/02_resume_plan_220cny.md`, `docs/00F1_fix_matrix.md`.
- **Verification**: `tests/test_preflight_live.py` — four boundary cases of the pure function;
  `test_preflight_refuses_broken_fresh_model`; `test_preflight_allows_documented_pruned_start`
  (33% + flag); `test_preflight_uses_recorded_baseline`; `test_baseline_tracks_the_historical_minimum`.
- **Residual risk**: a checkpoint whose live ratio is already below the baseline but within 2%
  per run will still be allowed; the minimum is monotone now, so the erosion cannot continue
  indefinitely without tripping the 2% window eventually — but a one-off 1.9% loss is accepted.

### P0-3 — `require_eos` escape hatch was inert
- **Verdict**: CONFIRMED (mechanism). The data content is UNSURE (packs absent).
- **Root cause**: the key existed in `_FlatStage` and in the error message but was dropped by
  `Stage` (no field), `parse_plan` (not parsed), and `train.py`'s stage-dict builder.
- **Fix**: `Stage.require_eos: bool = True`; parsed in `parse_plan`; carried by
  `train.py::plan_stage_specs` into `StagePackedDataset`; included in hot-reload
  (`Stage.signature`, `update_stages`).
- **Files**: `code/retnet_hf/plan.py`, `code/retnet_hf/train.py`, `code/retnet_hf/data.py`.
- **Verification**: `tests/test_plan_eos.py` goes JSON → `parse_plan` → `plan_stage_specs` →
  `stage_ready` on an eos-free shard: `require_eos:false` opens, absent key refuses with the
  correct remedy text, hot reload keeps the value.
- **Residual risk**: whether the v22 mixtures actually contain eos is still UNSURE; the code
  path is now correct but the data fact needs a data-machine check.

### P0-4 — E07 comment asserted a gradient audit that did not exist
- **Verdict**: CONFIRMED. `grep grad.abs` found it only in `scripts/grow_500m.py`.
- **Root cause**: a comment was used in place of a check; the previous “fix” replaced one
  rationalising comment with a claim of a check that did not exist.
- **Fix**: implemented `GradCoverageCallback`: registers a backward hook on every
  `requires_grad` parameter (reading `p.grad` in `on_step_end` is useless — HF calls
  `zero_grad()` before callbacks), records `max|grad|`, and at step 1 aborts when the
  element fraction of dead parameters exceeds `--dead_grad_tol` (default 2%). Stale comment
  deleted; `live_param_ratio` no longer doubles as the only “liveness” evidence.
- **Files**: `code/retnet_hf/train.py`, `code/scripts/_gen_code_doc.py`, `docs/00F1_fix_matrix.md`.
- **Verification**: `tests/test_grad_coverage.py` — healthy model has no dead params; freezing
  `blocks.1.msr.q_proj.weight` makes it dead; `on_step_end` raises `SystemExit`; hooks are
  removed afterwards; the check waits for the configured step.
- **Residual risk**: with `find_unused_parameters=False` under DDP, a parameter that is unused
  on some ranks may be reported dead; that is the intended meaning (it receives no gradient on
  this rank). Hook overhead is one Python call per parameter per backward until step 1 only.

### P0-5 — status table contradicted README, verifier certified it
- **Verdict**: CONFIRMED (doc 01 claimed not-fixed for E10/E11/E12/E15/E18/E19/E21/E22/E23).
- **Root cause**: three hand-maintained status locations and a verifier that never read them.
- **Fix**: `code/scripts/fix_status.py` is the single source (`FIX_STATUS`, `FIX_NOTE`);
  `_gen_code_doc.py` and `_gen_fix_doc.py` consume it; `verify.py` G-check compares
  `fix_status` vs docs/01_function_reference.md table vs docs/00F1_fix_matrix.md `**Status**` vs README table. All docs regenerated.
- **Files**: `code/scripts/fix_status.py` (new), `_gen_code_doc.py`, `_gen_fix_doc.py`,
  `verify.py`, `README.md`, `docs/01_function_reference.md`, `docs/00F1_fix_matrix.md`.
- **Verification**: `verify.py` prints
  `[G] status tables: fix_status 27, 01 27, README 27, 00F1 27`
- **Residual risk**: the README table is still *written* by hand (verify enforces equality,
  it does not generate the file). A README generator would remove the last manual step.

### P0-6 — E13 GC guard could never be satisfied by the producer
- **Verdict**: CONFIRMED by inspection and by running the producer.
- **Root cause**: the guard demanded `rebuild_cmd`/`source_ids`; `build_flat_corpus.py` wrote
  neither. The only passing test hand-wrote a fixture no shipped tool creates.
- **Fix**: producer now writes `argv`, `rebuild_cmd`, and `source_ids` into `meta.json`;
  `gc_stage` accepts `rebuild_cmd`, `source_ids`, or legacy `sources + text_fields`
  (`_proves_rebuildable`); added the missing “do not delete the validation set” check via
  `protect()` + `norm_key` (the old docstring claimed it, the body did not implement it).
  `tests/test_producer_gc.py` trains a tiny tokenizer, runs the real producer, then GCs the
  produced directory.
- **Files**: `code/scripts/build_flat_corpus.py`, `code/retnet_hf/data.py`,
  `code/retnet_hf/train.py` (protect call).
- **Verification**: producer test — `freed > 0` and shards gone; legacy meta accepted; meta
  without proof refused; protected dir refused.
- **Residual risk**: `_rebuild_cmd` records the argv, not the input data’s continued
  existence; the proof means “reproducible if inputs still exist”, which is what the metadata
  can witness. Source existence is not checked.

---

## P1 findings

### P1-1 — documented test command / “56 passed” did not reproduce
- **Verdict**: CONFIRMED (exit 4 from the root; 5 collection errors from `code/`).
- **Fix**: README command is `cd code`; true counts measured (**82 passed, 4 skipped**);
  `tests/conftest.py` excludes five files that import modules outside this snapshot
  (`retnet/`, `bench/`, `play_v2`), with a per-file reason table; the 4 skips (missing
  tokenizer, gitignored) are documented. `00F1_fix_matrix.md` no longer claims “41”.
- **Files**: `README.md`, `code/tests/conftest.py`, `docs/00_error_table.md`, `docs/00F1_fix_matrix.md`.
- **Verification**: `pytest tests -q` → 82 passed, 4 skipped; both script tests pass.
- **Residual risk**: the five excluded files are still shipped; if the reviewer expects them
  to run, they must be vendored with their dependencies (out of scope for this snapshot).

### P1-2 — E17 half-fixed; launcher default still won
- **Verdict**: CONFIRMED (`DEFAULTS["save_total_limit"] = 3`).
- **Root cause**: the same field had two independent defaults, and the test only regexed `train.py`.
- **Fix**: `retnet_hf/defaults.py` defines `DEFAULT_SAVE_TOTAL_LIMIT = 4` and
  `DEFAULT_NO_DECAY_1D = True`; `train.py` and `run_task.py` import them; both launcher
  branches pass the merged value explicitly.
- **Verification**: `tests/test_train_config_guards.py::test_save_total_limit_is_one_shared_constant`
  and `::test_run_task_passes_save_total_limit_in_both_branches` (plan and non-plan, values
  4 and 6).
- **Residual risk**: none known for this field; other launcher/train defaults still differ by
  design (documented in docs/01_function_reference.md) and would need the same treatment if they ever must agree.

### P1-3 — E06 wiring untested, `--decay_1d` unreachable, gamma check dead for `use_gla=False`
- **Verdict**: CONFIRMED, all four sub-claims.
- **Fix**: extracted `build_parser()` (pure) and test `--decay_1d → no_decay_1d=False`;
  `run_task` emits exactly one of `--no_decay_1d`/`--decay_1d`; `_preflight` reads gamma from
  `named_parameters()` **and** `named_buffers()`; removed the false “step 100/1000 ratio”
  promise and replaced it with `DecayGuardCallback`, which inspects the real optimizer groups
  at `on_train_begin` and aborts if any 1-D parameter sits in a `weight_decay>0` group.
- **Verification**: `test_decay_1d_flag_parses_to_false`, `test_run_task_emits_both_directions`,
  `test_trainer_receives_parsed_flag`, `test_preflight_checks_gamma_even_when_it_is_a_buffer`,
  plus `tests/test_no_decay_1d.py`.
- **Residual risk**: the decay-group assertion runs only when the optimizer exists at
  `on_train_begin`; if a future HF version builds it later, the callback logs a warning and
  skips (by design, no silent pass — it prints).

### P1-4 — E15 had three mutually exclusive specs
- **Verdict**: CONFIRMED (code off / 00F1 fixed / docs 02 “keep it”).
- **Fix**: chose the code’s policy (no in-training best-model selection) and rewrote
  `docs/02_resume_plan_220cny.md` §2/§5.4: train saves by step, selection is offline over checkpoints with
  `eval_holdout.py`. `docs/00_error_table.md`, `00F1_fix_matrix.md`, `01` aligned; `FIX_STATUS` marks E15 Fixed.
- **Files**: `docs/02_resume_plan_220cny.md`, `docs/00_error_table.md`, `docs/00F1_fix_matrix.md`, `docs/01_function_reference.md`, generator text.
- **Verification**: `test_best_model_selection_is_disabled` still pins the code; docs/02_resume_plan_220cny.md now
  describes the exact offline commands. No code change was needed.
- **Residual risk**: there is no one-command selection script; it is a documented `foreach`
  loop. Noted in docs/02_resume_plan_220cny.md §9 and here.

### P1-5 — E14 guard was path-equality only; test vacuous
- **Verdict**: CONFIRMED (the “alternative spelling” test used the identical absolute path).
- **Root cause**: the guard guarded a property (path equality) different from the failure mode
  (data equality); content-level contamination was invisible.
- **Fix**: new `retnet_hf/provenance.py`: (1) `norm_key = normcase(resolve)` catches relative
  paths, trailing separators and case variants; (2) declared sources from `meta.json`
  (`source_ids`, legacy `sources+text_fields`); (3) row-level 64-bit FNV hashes, val side reads
  `val.ids.npy` (the old code skipped every file containing “val”, so the val set was always
  empty). Content scan is capped (200k rows ≈ 0.8 GB) and prints when truncated.
  The validation set is also registered with `StagePackedDataset.protect`.
- **Verification**: `tests/test_val_disjoint.py` — relpath, trailing separator, case variation;
  shared `source_ids`; two rows copied from val into train; assertion that val hashes come from
  `val.ids.npy`.
- **Residual risk**: content detection beyond 200k rows is sampling; declared-source detection
  requires producer metadata. Whether the current v22 packs overlap is UNSURE (packs absent).

### P1-6 — `eval_holdout.py` lacked both probes
- **Verdict**: CONFIRMED (`--by-position`, `--ctx-ablation` absent).
- **Fix**: implemented both per docs/02_resume_plan_220cny.md §5.3: 8 position buckets with per-bucket loss, and
  `Δ_ctx = mean L(256) − L(2048)` over the last 256 positions; `--ctx` and `--pos-buckets`
  exposed; `ctx`/`seq_len` must be multiples of `chunk_size` (chunkwise kernel constraint) and
  are validated before the run. Also fixed `_token_ce` to chunk the fp32 conversion.
- **Verification**: `tests/test_eval_probes.py` — end-to-end on a tiny model: 8 buckets with
  the expected ranges, Δ_ctx identity and token count, misaligned ctx refused up front.
- **Residual risk**: the short-window probe restarts xPos positions at 0 by definition of the
  documented probe; it measures “context usefulness”, not equivalent inference.

### P1-7 — `reinit_gamma.py` silently truncated sharded checkpoints
- **Verdict**: CONFIRMED by inspection; whether the v21 export is sharded is UNSURE.
- **Fix**: `load_checkpoint_state()` merges shards via `model.safetensors.index.json`, rejects
  duplicate keys, asserts every `weight_map` key was read, refuses multi-file input without an
  index (instead of reading `files[0]`), writes a single file, removes a stale index, and
  asserts the written key set and gamma values.
- **Verification**: `tests/test_reinit_gamma_shards.py` — two shards + index → merged output
  with all keys, both gammas replaced, index not copied; multi-file/no-index raises.
- **Residual risk**: output is always single-file; very large checkpoints will produce one big
  safetensors instead of preserving sharding (fine for this size, noted).

### P1-8 — `grow_500m.py` header contradicted its `grow_norm`
- **Verdict**: CONFIRMED. Implementation wrote `full(n_new, gate)`; the header claimed “source
  prefix copied, new dims zero”.
- **Root cause**: the header described an intended mapping that was never implemented.
- **Fix**: header rewritten to the implemented mapping (second segment **as a whole** set to
  `BLOCK_GATE=1.0` in blocks / `FINAL_GATE=0.0` at final), with the reason the source’s
  learned second-segment gates are *not* copied (v21 block gates are 0 → copying keeps new
  dims dead; a non-zero final gate lets the re-randomised embedding columns leak into logits —
  measured max|Δ|=0.529). Added `_check_norm_layout` before the trainability probe, asserting
  lengths, uniform gate and zero beta. Fixed `TARGET_PARAMS=0` making success return 1.
- **Verification**: `tests/test_growth_trainable.py` — success now `rc == 0`; two-stage growth
  (48→72→96) passes function preservation (2.6e-3 < 1e-2) and asserts the gate layout; an
  out-of-recipe source (block 1.3 / final 0.4) is **refused** with `rc == 2` instead of silently
  producing an incomparable model.
- **Residual risk / correction to the report**: the report said function preservation holds
  “only because final_norm’s gate is 0”. That is incomplete: it also requires the source’s
  block second-segment gates to be 1.0 (the recipe’s invariant). A source trained away from
  that cannot be re-grown function-preservingly while also reviving dead dims; the script now
  fails loudly rather than pretending. This is documented in the header as a precondition.

### P1-9 — seven false technical claims in docs/01_function_reference.md
- **Verdict**: CONFIRMED for all seven.
- **Fix** (prose in `_gen_code_doc.py`, then regenerated):
  1. 9.2% direction corrected (“9.2% slower”, self-contradiction removed).
  2. checkpoint-layer semantics corrected (“disabled at layer index >= threshold”, i.e. layers below threshold
     are checkpointed).
  3. `_sync_resume_state` now described as comparing exactly four fields
     (`train_batch_size/logging_steps/eval_steps/save_steps`).
  4. “bitwise equal” → “equal up to floating-point rounding, not bitwise”, with the measured deltas and the test’s explicit
     non-bitwise assertion.
  5. `gc_stage` now actually protects the validation dir (`protect()`), and the text says how.
  6. `DEFAULTS` claim rewritten: it centralises **launcher** defaults; `train.py` keeps its own
     for direct use; the seven differing fields are listed and explained; only the two fields
     that must agree (`save_total_limit`, `no_decay_1d`) are shared constants.
  7. `gc_stage` “rebuild within two hours” replaced with the actual proof semantics (no time
     bound; `rebuild_cmd`/`source_ids`/legacy fields).
- **Verification**: `verify.py` A/C re-check code blocks and text length; `H` checks the key
  literals; the claims are now consistent with the cited lines.
- **Residual risk**: prose not covered by a machine check can drift again; only the greppable
  invariants are covered by `verify.py` H.

---

## P2 findings

| Report | Verdict | Fix | Verification / residual risk |
|---|---|---|---|
| P2-1 `--eval_batches` dead | CONFIRMED | `cap_eval_rows` caps eval rows; default **0 = unlimited** to keep historical `eval_loss` comparable; `run_task` passes `eval_batches` (default 0) | `test_eval_batches_defaults_to_unlimited_and_is_passed_through`; residual: setting N changes the metric basis, documented |
| P2-2 flat `weight` dropped / hot reload drops keys | CONFIRMED | `_FlatStage` receives `weight`; `Stage.signature()` covers name/dir/steps/mode/gc/min_shards/exclude/weight/require_eos; hot reload passes weight+require_eos | `test_plan_eos.py::test_flat_stage_weight_reaches_wcodes` and `::test_hot_reload_does_not_drop_require_eos`; residual: none known |
| P2-3 `retention_fast` claims/path | CONFIRMED | 5 batched matmuls documented; path → `tests/test_fast_chunkwise.py`; “bitwise” → “fp-rounding equivalent” | doc-only; code unchanged |
| P2-4 citation inside code fence | CONFIRMED | `_add_citations._safe_insert` counts fence state and moves the marker before the fence; docs/02_resume_plan_220cny.md rewritten without the broken fence | `tests/test_doc_tools.py` (3 cases); residual: only this inserter is fixed, other generators use anchors |
| P2-5 docs/04 dropout placement | CONFIRMED | equation now shows `Dropout` inside MSR before `out_proj`, residual add has none; FFN dropout before the add; explicit note | doc-only; code unchanged |
| P2-6 docs/00_error_table.md header arithmetic | CONFIRMED | docs/00_error_table.md fully rewritten; no per-file line total is claimed | doc-only |
| P2-7 docs/03 “fixed chunkwise” | CONFIRMED | text now says chunkwise holds on the launcher path (`run_task`), direct `train.py` defaults to parallel; mode is visible in the log | doc-only |
| P2-8 tests that cannot fail | CONFIRMED | `rc in (0,1)` → `rc == 0` (script no longer returns 1 when `TARGET_PARAMS=0`); E21 asserts the fast path is taken (`torch.stack` poisoned); E22 asserts chunk *structure* at n=32768; E14 uses real alternate spellings; E12 goes through the plan chain | listed tests; residual: `test_eos_present` still tests `_FlatStage` directly, but the chain is covered by `test_plan_eos` |
| P2-9 verify.py blind spots | CONFIRMED | added G (status table ×4), H (15 literal contracts, positive and negative), I (MANIFEST hashes/missing/unlisted); scope statement in docstring + README; did **not** add `plan.py`/`control.py` to the per-function FILES (documented scope instead, as the report allowed) | `verify.py` run; residual: prose claims beyond H remain unverified |
| P2-10 `_INDEX_CACHE` unbounded | CONFIRMED | `OrderedDict` with `maxsize=8`, LRU eviction; still not thread-safe (GIL-atomic get/set/move) | `test_fast_chunkwise` exercises the cache; no dedicated test (minor) |
| P2-11 fused CE extra work | CONFIRMED | `_forward_hidden(need_state=False)` skips `_parallel_final_state`; head fp32 conversion hoisted once per forward and once per backward | fused path exercised indirectly; residual: no dedicated fused-CE unit test in the snapshot |

---

## Where I believe the report is inaccurate or incomplete

1. **P1-8’s causal explanation is incomplete.** The report says preservation holds “only
   because `final_norm`’s gate is 0”. It also depends on the source’s block second-segment
   gates being exactly 1.0. With block gate 1.3 / final 0.4 the two-stage growth changes
   logits by ~0.5. I documented this as a precondition and made `--verify` refuse
   out-of-recipe sources; the header now states both conditions.
2. **P0-2’s proposed framing “fresh vs resume”** does not actually separate the cases: the
   rescue plan is a *fresh* run from a pruned checkpoint (no `--resume`). The operative
   distinction is “expected dead mass” (explicit flag or recorded baseline), not
   resume-vs-fresh. I implemented the latter.
3. **P1-3’s suggestion to implement a “step-100/1000 gamma ratio” assertion** is a weak
   guard: gamma legitimately changes with learning, so a ratio cannot distinguish weight
   decay from optimisation. I replaced it with a structural check on the optimizer groups,
   which is the actual property E06 is about.
4. **P2-1’s “wire it into an eval-batch cap or delete it”** — I wired it, but chose
   default `0 = unlimited` rather than keeping the previously hard-coded 40, because making
   40 suddenly effective would silently shrink the eval set (and change `eval_loss` for
   every historical comparison). The launcher still passes the configured value explicitly.
5. **P0-5’s “derive the table from a single source”** is implemented for docs/01_function_reference.md and 00F1;
   README is still hand-written but `verify.py` G makes divergence impossible to land
   silently. A full README generator is a reasonable follow-up, not a correctness gap.

## UNSURE

- Whether `retnet_500m_v21_export` is sharded (P1-7’s impact) — the checkpoint is not in the
  snapshot; the code now handles both cases and fails loudly on ambiguity.
- Whether the v22 mixtures / `val_pack` contain eos and overlap — packs are not in the
  snapshot. The guards are correct; the data facts need a data-machine run.
- All checkpoint-dependent numbers quoted in docs (67% zeros, gamma medians, live counts).
  They are labelled as historical measurements in docs/02_resume_plan_220cny.md §8, not re-asserted here.
- GPU performance claims (throughput, memory peaks, compile mode) — not reproduced on CPU.

## Verification run (from this directory)

```
python verify.py
  [A] parsed 103 function sections in 01
  [B] 103 functions in code, 103 documented
  [D] 00 has 27 ids, 01 cites 19, 0 dangling
  [E] 00F1 parsed 27 entries, 44 code blocks
  [F] 00F1 covers 27/27 ids
  [G] status tables: fix_status 27, 01 27, README 27, 00F1 27
  [H] literal contracts: 15
  [I] MANIFEST 69 entries, 0 missing, 0 hash mismatches, 0 unlisted; 69 files in tree
  PASS

cd code && python -m pytest tests -q   → 82 passed, 4 skipped
python tests/test_plan_gc.py           → all pass
python tests/test_plan_pipeline.py     → all pass
```

## Still open

1. Data/checkpoint facts listed under UNSURE need a data-machine run with the current code
   (`_preflight` + `eval_holdout.py`).
2. Validation provenance is exact only where `meta.json` declares sources; content comparison
   is capped at 200k rows.
3. Offline model selection is a documented loop, not a script.
4. `verify.py` covers 7 files at function level; `plan.py`, `control.py`, `provenance.py`,
   `defaults.py`, `scripts/`, `tests/` rely on docstrings and tests (scope stated in README).
5. GPU performance numbers were not reproduced; no performance claim in this log is new.
