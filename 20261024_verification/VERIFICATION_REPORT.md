# Verification gate — 20261024_Submit_pre-audit

Scope: the 26 audit findings (P0-1…P0-6, P1-1…P1-9, P2-1…P2-11) claimed fixed in
`20261024_Submit_pre-audit/FIX_LOG.md`, plus the 27-item status table it publishes.
Target tree was treated as read-only. All executions were done on a copy at
`%TEMP%\preaudit_verify` with `C:\tools\miniconda3\envs\AI\python.exe`
(Python 3.13.7, torch 2.10.0+cu128, transformers 4.57.6).

## Evidence run

| Command | Result |
|---|---|
| `python verify.py` (in the submission) | PASS, 9/9 checks; MANIFEST 69 entries, 69 files |
| `python -m pytest tests -q` (from `code`) | **82 passed, 4 skipped**, exit 0 |
| `python tests/test_plan_gc.py` | all pass, exit 0 |
| `python tests/test_plan_pipeline.py` | all pass, exit 0 |
| `run_task.py --config <temp plan config> --dry` | plan branch emits `--eval_batch_mult 2`, `--save_total_limit 6`, `--decay_1d`, `--allow_dead_params` |
| `_preflight`/`check_live_ratio` erosion simulation | 12 runs at −1.9 pp each all accepted (0.330 → 0.102) |
| Open-stage hot-reload probe | `require_eos` false→true and `weight` 1→5 on an already-open `_FlatStage` are **ignored** |
| Partial-dead-gradient probe | zeroed growth columns are invisible to `GradCoverageCallback` (dead fraction 0.14 %, only `gamma_new`/`beta_new`) |
| Fused-CE gradient probe | fused gradient differs from `F.cross_entropy` by up to 0.00145 (0.8 % of max |grad|) because `p.scatter_(..., -1.0)` replaces `softmax_target` instead of subtracting 1 |
| Real data check | `pretrain_v2` shards (the rescue-mixture sources) contain eos in the first per-shard sample of all 34/37/25 selected shards; `pretrain_v2/meta.json` does **not** exist |

## Verdict summary

| Fix | Verdict | Short reason |
|---|---|---|
| P0-1 E22 | PASS | byte-budget chunking real; 838 tok/chunk at V=20000; ≥2 chunks at n=32768 tested |
| P0-2 E08 | CONDITIONAL | plan startable; baseline ratchets downward indefinitely, contrary to its own doc |
| P0-3 E12 | CONDITIONAL | chain works and rescue shards have eos; hot reload of an open stage ignores the new value |
| P0-4 E07 | CONDITIONAL | check exists and fires on frozen tensors; partial (row/column) dead mass is not counted |
| P0-5 | PASS | four status tables agree; verifier G enforces it; README still hand-written |
| P0-6 E13 | CONDITIONAL | producer proof works; existing `pretrain_v2` has no meta, so the historical gc stage still frees 0 |
| P1-1 | PASS | 82/4 reproducer; 5 excluded files documented |
| P1-2 E17 | PASS | one constant; both launcher branches pass the value |
| P1-3 E06 | PASS | `--decay_1d` reachable; parser/trainer wiring asserted; gamma buffer checked |
| P1-4 E15 | PASS | docs/02 now matches code (offline selection) |
| P1-5 E14 | PASS | three layers implemented and tested; 200 k-row prefix cap remains |
| P1-6 | PASS | both probes implemented and tested |
| P1-7 N01 | PASS | shard merge, key-set/duplicate checks, written-key assertion |
| P1-8 | PASS | header matches implementation; layout and two-growth tests; out-of-recipe source refused |
| P1-9 | PASS | seven specific prose errors corrected; new count errors exist in doc 00 (P2-6) |
| P2-1 | PASS | `--eval_batches` is a real row cap, default 0=unlimited |
| P2-2 | **FAIL** | factory weight fixed, but hot reload of an already-open stage still ignores weight/require_eos |
| P2-3 | PASS | 5 matmuls and `tests/` path corrected |
| P2-4 | CONDITIONAL | fence-aware inserter fixed/tested; `_add_citations.py` still points at pre-rename Chinese doc filenames |
| P2-5 | **FAIL** | docs/04 equation still puts Dropout **after** W_O; prose note says before |
| P2-6 | CONDITIONAL | old “2,240” total gone; rewritten doc now says “thirteen P0s” (table has 18), “six rewrites” (seven listed), “6 new test files” (seven) |
| P2-7 | CONDITIONAL | docs/03 caveat fixed; docs/01 still says “Training uses chunkwise” unconditionally |
| P2-8 | PASS | growth `rc==0`, stack poison, production-shape CE, real path spellings, plan chain |
| P2-9 | PASS | G/H/I added, scope stated; H is 15 literals and does not cover prose |
| P2-10 | PASS | LRU max 8 eviction; thread-safety gap acknowledged |
| P2-11 | CONDITIONAL | need_state/hoist correct, but no test and the fused path’s gradient is numerically wrong |

Overall: **HOLD**. Core P0 path is real, but P2-2/P2-5 remain false and four P0 guards have named
predicates that do not pass yet.

---

## P0 findings

### P0-1 — E22 chunked CE
**Claim.** `_chunked_weighted_ce` chunks by a 64 MB byte budget derived from V; config field
`ce_chunk_bytes`; tests at the production token count. **Mechanism.** `ce_chunk_tokens(V,
chunk_bytes) = chunk_bytes // (4·V)` → 838 tokens/chunk at V=20000; the call site passes
`config.ce_chunk_bytes`; each chunk is `.float()`ed before `F.cross_entropy`. **Evidence.**
`tests/test_regressions_extra.py::test_ce_chunk_size_is_derived_from_a_byte_budget` asserts 838;
`::test_chunked_ce_splits_at_production_token_count` spies on `F.cross_entropy` at n=32768 and
requires ≥2 chunks, each ≤ budget, all tokens covered; pytest 82/4 passed; verify H has the
64 MB constant. **Residual.** The budget bounds only the fp32 logits slice; CE internals need
another copy. `ce_chunk_bytes<=0` is documented as “whole segment”, but the call site falls back
to 64 MB and the function clamps to 1 token — a minor doc/implementation mismatch. **Verdict:
PASS.**

### P0-2 — E08 preflight live ratio
**Claim.** The rescue plan starts from a 33 %-live checkpoint: `--allow_dead_params`, optionally a
`live_params.json` baseline; `ratio_min` is meant to stop slow erosion. **Mechanism.**
`check_live_ratio` passes if ratio ≥95 %, or explicit flag, or `ratio >= baseline - 0.02`;
`_preflight` reads the baseline from `model_dir`/`output_dir`, writes `live_params.json` after a
successful run. **Evidence.** The four boundary tests pass; `docs/02` §2 and `v22_a1.json` set
`allow_dead_params`; the plan branch dry-run shows it. **But** `write_live_baseline` stores
`ratio_min = min(previous, ratio)`, so every admitted 1.9 pp drop becomes the new baseline.
Simulation: 0.330 → 0.311 → … → 0.102 after 12 runs; `check_live_ratio` still returns True.
The doc claim in `docs/01` L263 (“ratio_min … so that slow erosion … does not slip past the
guard”) is false. A stale `live_params.json` in `output_dir` can also exempt a broken fresh model
within 2 %. **Residual/condition.** Baseline must not be lowered by the run it just admitted (or
the tolerance must be cumulative), and a 12-run −1.9 pp sequence must eventually be refused.
**Verdict: CONDITIONAL.**

### P0-3 — E12 `require_eos`
**Claim.** `Stage.require_eos` is parsed, carried through `plan_stage_specs`, `StagePackedDataset`
and hot reload; a JSON→dataset test covers the chain. **Mechanism.** `Stage` field, `parse_plan`
bool, `plan_stage_specs`, `_FlatStage` check, `Stage.signature`, `update_stages` spec dict.
**Evidence.** `test_plan_eos` passes (false opens, absent key refuses with the correct key name,
default true). Real-data check: the 34/37/25 shards selected for `rescue_m1/2/3` all contain eos
in the first per-shard sample, so the plan’s `require_eos:true` is satisfiable. **But** an
already-open stage is reused by `(name, data_dir)` in `update_stages`; changing `require_eos`
false→true on that stage leaves `_FlatStage.require_eos=False` and `stage_ready` still True
(probe). The test only flips an unopened stage. **Residual/condition.** The hot-reload claim is
false for the active stage; `update_stages` must update/reopen it, and a test must open first.
**Verdict: CONDITIONAL.**

### P0-4 — E07 gradient coverage
**Claim.** `GradCoverageCallback` registers backward hooks, records max|grad| per parameter,
aborts after step 1 if the element share of dead parameters exceeds `--dead_grad_tol` (2 %).
**Mechanism.** Hooks on `requires_grad` params; `on_step_end` at `global_step >= check_step`;
`dead_numel/total` over parameters whose max grad is 0; hooks removed after check; wired with
`trainer.add_callback`. **Evidence.** `test_grad_coverage` covers healthy, frozen and wait-for-step
cases; H asserts the `add_callback` line and the old comment is gone. **Residual.** The statistic
is per **parameter**, not per element: a tensor with zeroed rows/columns but any live element is
counted alive. Probe: a pruned growth model gives 0.14 % dead, only `gamma_new/beta_new`; the
v21-style zero rows/columns are invisible. The docs’ “element share” is therefore a lower bound.
**Residual/condition.** Count per-element (or per-tensor zero fraction) and add a partial-zero
test, or state the limitation. **Verdict: CONDITIONAL.**

### P0-5 — status tables
**Claim.** `fix_status.py` is the single source; `verify.py` G compares it with docs/01, README
and 00F1. **Mechanism.** `_gen_code_doc.py`/`_gen_fix_doc.py` import `FIX_STATUS`; G parses all
four tables. **Evidence.** `verify.py` → `[G] status tables: fix_status 27, 01 27, README 27,
00F1 27`; docs regenerated. **Residual.** README and docs/00 are still hand-written; G checks
agreement, not truth. E23’s “every fix has a test that goes red if reverted” is contradicted by
the FIX_LOG’s own P2-10/P2-11 gaps (no dedicated tests) and by doc-only fixes. **Verdict: PASS**
(consistency fixed; the truth of every “Fixed” status is a separate residual).

### P0-6 — E13 GC proof
**Claim.** The producer writes `argv`, `rebuild_cmd`, `source_ids`; GC accepts those or legacy
`sources+text_fields`; validation dir is protected. **Mechanism.** `build_flat_corpus` meta;
`_proves_rebuildable`; `protect()`/`norm_key`; `gc_stage` three checks. **Evidence.**
`test_producer_gc` runs the real producer and GCs its output (freed>0), legacy meta accepted,
missing proof refused, protected dir refused; pytest passed. **Residual.** The real
`data_cache/pretrain_v2` has **no `meta.json`**, so the historical `plan_ddp2` stage
`pretrain_v2_noskypile` (`gc:true`) still refuses and frees 0; the fix’s test only covers
`build_flat_corpus` output. `_rebuild_cmd` is not shell-quoted and source existence is not
checked (acknowledged). **Condition.** GC on a copy of `pretrain_v2` must free >0, or the plan
must set `gc:false` and document that only producer outputs are GC-able. **Verdict:
CONDITIONAL.**

## P1 findings

### P1-1 — test counts
**Claim.** README says `cd code`, 82 passed / 4 skipped; five files excluded with reasons.
**Evidence.** Actual run: 82 passed, 4 skipped; both script tests pass; conftest lists the five
files plus the two script tests; no stale “41” claim. **Residual.** The five files still ship but
cannot run here; script tests are manual. **Verdict: PASS.**

### P1-2 — E17 save_total_limit
**Claim.** `DEFAULT_SAVE_TOTAL_LIMIT=4` shared; both launcher branches pass it. **Evidence.**
`defaults.py`; `run_task.py` DEFAULTS and both branches; `test_save_total_limit...` and
`test_run_task...`; plan dry-run emitted `--save_total_limit 6` when configured. **Verdict:
PASS.**

### P1-3 — E06 no-decay / decay_1d
**Claim.** Parser pure; `--decay_1d` reachable; both launcher directions; gamma buffer check;
`DecayGuardCallback`. **Evidence.** Dry-run with `no_decay_1d:false` emitted `--decay_1d`;
`build_parser` tests; `test_no_decay_1d`; `test_preflight_checks_gamma_even_when_it_is_a_buffer`;
HF 4.57.6 creates the optimizer before `on_train_begin` (inspected), so the guard sees it. Guard
logs and skips if absent. **Verdict: PASS.**

### P1-4 — E15 policy
**Claim.** Code has `load_best_model_at_end=False`; docs/02 now says offline selection.
**Evidence.** `docs/02` §2/§5.4; code/test; docs/00/00F1 aligned. **Residual.** No one-shot
selection script; documented as a `foreach` loop. **Verdict: PASS.**

### P1-5 — E14 validation provenance
**Claim.** Path + declared source + row hashes; genuine alternative spellings; val side reads
`val.ids.npy`. **Evidence.** `test_val_disjoint` passes all cases (relpath, trailing separator,
case, source overlap, content overlap, val.ids); `assert_val_disjoint` raises before training.
**Residual.** Content scan is a 200 k-row prefix, not a sample across the stream; declared source
requires producer meta; the real packs are absent. **Verdict: PASS.**

### P1-6 — eval probes
**Claim.** `--by-position` (8 buckets) and `--ctx-ablation` (Δ_ctx over last ctx positions), with
chunk alignment validation. **Evidence.** `test_eval_probes` end-to-end: 8 buckets, Δ_ctx identity,
misaligned ctx refused; docs/02 commands now run as written. **Residual.** Short window restarts
positions by definition; full logits are still materialized. **Verdict: PASS.**

### P1-7 — reinit_gamma shards
**Claim.** Merge shards via index; duplicate/missing key checks; reject multi-file/no-index;
write single file; assert key set and gamma. **Evidence.** `test_reinit_gamma_shards` passes;
code inspected. **Residual.** Output always single-file; extra shard keys not in the index are
not rejected; whether v21_export is sharded remains UNSURE. **Verdict: PASS.**

### P1-8 — grow_500m header/layout
**Claim.** Header now states whole-segment gate (block 1.0, final 0.0), layout asserted, two-stage
growth tested, out-of-recipe source refused. **Evidence.** `test_growth_trainable` (rc=0,
revert→rc=2, two-stage layout, out-of-recipe->rc=2); `_check_norm_layout`; header lines. **Residual.**
A source whose block gates drifted from 1.0 can no longer be grown with `--verify` (it now
refuses); the real v13/v21 export is absent. **Verdict: PASS.**

### P1-9 — seven doc claims
**Claim.** Seven specific prose errors corrected. **Evidence.** Spot-checked all seven: 9.2 %
direction, checkpoint-layer semantics, four resume fields, non-bitwise equivalence, val protection,
DEFAULTS split, no “two hours”. **Residual.** New count errors in doc 00 (P2-6); docs/01’s
“Training uses chunkwise” remains (P2-7). **Verdict: PASS** for the seven claims.

## P2 findings

### P2-1 — eval_batches
**Claim.** Real row cap; default 0 unlimited. **Evidence.** `cap_eval_rows`, `Subset`, run_task
pass-through; test. **Residual.** Setting N changes the metric basis (documented). **Verdict:
PASS.**

### P2-2 — flat weight / hot reload — **FAIL**
**Claim.** `_stage` passes `weight`; `update_stages` passes `weight`/`require_eos`. **Mechanism.**
`_stage` does pass weight at first open. But `update_stages` reuses cached stages keyed only by
`(name, data_dir)` and never mutates/reopens them. **Evidence.** Probe: an open flat stage with
`require_eos=False` stays ready after the plan flips it to true; its `weight` stays 1 and
`wcodes` stay 1 after the plan sets weight 5. The shipped test flips an **unopened** stage, so it
passes. **Residual.** The active stage’s weight/require_eos/mode changes are silently ignored —
the exact hot-reload defect P2-2 claimed to fix. **Verdict: FAIL.**

### P2-3 — retention_fast docs
**Claim.** 5 matmuls, path `tests/`, “rounding not bitwise”. **Evidence.** docstring matches the
5 matmuls in `chunkwise_batched`. **Verdict: PASS.**

### P2-4 — citation insertion
**Claim.** `_safe_insert` is fence-aware; docs/02 fence fixed. **Evidence.** `test_doc_tools`
passes; docs/02 has 12 fence lines (even). **Residual.** `_add_citations.py` still points
`DOC00`/`DOC02` at `docs/00_错误总表.md` / `docs/02_续训恢复计划_220元.md`, which no longer exist;
`add_00`/`add_02` exit “找不到小节” on the current tree, and the test only covers the helper.
**Condition.** Update the two constants and add an entry-point test. **Verdict: CONDITIONAL.**

### P2-5 — docs/04 dropout — **FAIL**
**Claim.** “Equation now shows Dropout inside MSR before out_proj; residual add has none.”
**Evidence.** `docs/04` L72 still writes `u = Dropout(W_O(SiLU(g) ⊙ GroupNorm(Ret)))`, i.e.
dropout **after** `W_O`; the prose note below correctly says “before out_proj”. The equation and
note contradict each other; the claimed redraw did not happen. **Residual.** The note is right,
but the displayed formula is wrong. **Condition.** Rewrite as `u = W_O(Dropout(SiLU(g) ⊙
GroupNorm(Ret)))` (dropout before W_O) and remove the “older version drew this wrongly” wording
or align it. **Verdict: FAIL.**

### P2-6 — docs/00 arithmetic
**Claim.** The seven-files/two-files header arithmetic is gone. **Evidence.** No “2,240”; but the
rewritten opening says “the thirteen P0s” while the table has 18 P0 rows; the P1-9 row says “six
rewrites” and then lists seven; E23 says “6 new test files” while seven were added. **Condition.**
Correct those counts or remove them. **Verdict: CONDITIONAL.**

### P2-7 — chunkwise caveat
**Claim.** docs/03 now says chunkwise holds on the launcher path and direct train.py defaults to
parallel. **Evidence.** docs/03 §4 correct; but docs/01 L1470 still says “Training uses
chunkwise” without the launcher caveat. **Condition.** Add “on the launcher path” (or the
`--mode` caveat) to docs/01. **Verdict: CONDITIONAL.**

### P2-8 — tests that cannot fail
**Claim.** rc==0 for growth; stack-poison for E21; chunk structure at n=32768; real path
spellings; plan chain. **Evidence.** Tests pass and are constructed to go red when reverted;
`test_plan_gc` is no longer the only E13 guard. **Residual.** `test_eos_present` still tests
`_FlatStage` directly (acknowledged; chain covered by test_plan_eos). **Verdict: PASS.**

### P2-9 — verify blind spots
**Claim.** G status, H literal contracts, I manifest, scope stated. **Evidence.** verify passes
G/H/I; 15 contracts; manifest 69/69. **Residual.** H covers only 15 literals and no prose; docs 00
and 04 are outside A/B. **Verdict: PASS.**

### P2-10 — index cache
**Claim.** OrderedDict max 8 with LRU eviction. **Evidence.** `_INDEX_CACHE_MAX=8`, eviction after
insert; test_fast_chunkwise exercises. **Residual.** Not thread-safe (GIL-atomic only), no
dedicated eviction test. **Verdict: PASS.**

### P2-11 — fused CE work — CONDITIONAL
**Claim.** `need_state=False` skips the state einsum; head fp32 cast hoisted once per forward and
once per backward; no dedicated test. **Evidence.** Code does both; fused path exercised
indirectly is **false** (no test sets `RETNET_FUSED_CE=1`). A direct probe shows the fused
gradient differs from `F.cross_entropy` by 0.00145 (0.8 % of max |grad|) because
`_fused_backward` does `p.scatter_(..., -1.0)` instead of `p[target] -= 1`; the loss matches, the
gradient does not. `docs/02` E22 recommends `RETNET_FUSED_CE=1` as the memory fallback.
**Condition.** Fix the scatter and add a fused-vs-reference gradient test, or remove the fallback
recommendation. **Verdict: CONDITIONAL.**

## Status table (27 items)

`verify.py` G proves the four published copies agree; it does not prove each status. All 27 are
“Fixed” except N02/N03 (“Doc corrected”). E23’s universal test claim is the one status whose
content is overstated (see P0-5). No other status contradicted the code in the files inspected.

## Overall release decision

**HOLD.** The main training path is materially repaired (P0-1, P0-5, all P1s, and most P2s hold),
but the submission cannot yet be released as “all 27 fixed”: P2-2’s hot reload still silently
drops `weight`/`require_eos` on the active stage, P2-5 still contains the wrong dropout equation
it claimed to fix, and the P0-2/P0-3/P0-4/P0-6 guards have named predicates that do not pass.
These are narrow and fixable; the core direction is sound.
