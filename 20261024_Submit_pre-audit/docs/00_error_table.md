# 00 Error Table (Current State)

> Audit scope: the 7 files on the training path (`retnet_hf/*`, `run_task.py`, `scripts/grow_500m.py`)
> and the data / plan / pack-building pipelines they depend on.
> Line-number convention: the legacy line numbers cited in this table come from the 2026-10-20 commit (audit report `_audit_fixed/AUDIT_REPORT.md`);
> for the complete post-fix code excerpts and the item-by-item notes see `00F1_fix_matrix.md`, and for function-level notes see `01`.
> Status source: `code/scripts/fix_status.py` (README, 01 and 00F1 are cross-checked by `verify.py`).

## 0. In One Sentence

The common pattern across the thirteen P0s is not "some line was written wrong" but that **the guard and the property it claims to guard are not the same thing**:
counting parameters without reading their values, checking a path without checking the data, writing a comment instead of writing a check, writing a status table instead of generating one.
The principle for this round of fixes is that every guard must be provably able to fail, and its failure point must land on the property it claims to guard.

---

## 1. Defect Index

| ID | Severity | Original defect | Current disposition | Location |
|---|---|---|---|---|
| E01 | P0 | New matrices zero-initialized wholesale during growth, new rows and columns all 0 | Fixed | `grow_500m.fill()` |
| E02 | P0 | New norm gates `gamma_new`/`beta_new` all 0 | Fixed | `grow_500m.grow_norm()`: whole-segment gating, blocks 1.0 / final 0.0 |
| E03 | P0 | New embedding columns all 0 | Fixed | `grow_500m.main` embedding section |
| E04 | P0 | Growth validation compared only logits, not gradients | Fixed | 5-step backward + zero-gradient refusal to persist; layout assertions |
| E05 | P0 | `GrowthLayerNorm` new channels 0 on both sides | Fixed | class default `gamma_new=1.0` |
| E06 | P0 | `--no_decay_1d` off by default, gamma crushed by weight decay | Fixed | on by default + `--decay_1d` + optimizer-group assertion |
| E07 | P0 | A comment rationalized the missing check with "the gradient exists, it is just 0" | Fixed | `GradCoverageCallback` (backward hook + stop at step 1) |
| E08 | P0 | Parameter count read only from config, liveness never verified | Fixed | live parameter ratio + `--allow_dead_params` + `live_params.json` |
| E09 | P0 | `--add-eos` defaulted to False | Fixed | default True + `--no-add-eos` |
| E10 | P0 | Pack-building calls did not pass `--add-eos`, the `auto` field leaked into metadata | Fixed | explicit `--add-eos` in the call string + body field + fail fast |
| E11 | P0 | Built 16K packs knowing there were no boundaries | Fixed | any source with `add_eos=false` is rejected |
| E12 | P0 | Flat row slicing had no document boundaries, the `require_eos` pass-through was a broken chain | Fixed | `Stage.require_eos` through the whole chain + spot checks |
| E13 | P0 | GC deleted the only copy without checking whether it was rebuildable | Fixed | producer writes the proof + `gc_stage` three-layer check + validation set protection |
| E14 | P0 | `val_dir` overlapped the training data | Fixed | three-layer check: path + declared provenance + row content |
| E15 | P0 | Best checkpoint chosen on a contaminated `eval_loss` | Fixed | training-time selection disabled; offline `eval_holdout.py` |
| E16 | P0 | parallel did two extra normalizations, not equivalent to chunkwise | Fixed | removed the two private normalizations |
| E17 | P1 | `save_total_limit` default 3 vs 4 in the template | Fixed | single constant in `defaults.py`; passed explicitly in both branches |
| E18 | P1 | Hot-editing the total step count raised the LR | Fixed | `should_accept_reschedule` rejects by default |
| E19 | P1 | `refresh` compared counts only, same-size replacement was undetectable | Fixed | signature snapshot taken at scan time |
| E20 | P1 | plan branch did not pass `eval_batch_mult` | Fixed | passed in both branches |
| E21 | P1 | recurrent looped token by token | Fixed | `tgt_len==1` fast path (including a test that it really took the fast path) |
| E22 | P1 | `logits.float()` materialized `[B,L,V]` | Fixed | chunking by byte budget + chunk-count assertion at the production shape |
| E23 | P2 | No test checked whether parameters receive gradients | Fixed | 6 new test files; every guard has a backward case |
| N01 | P0 | `reinit_gamma` worked per layer instead of per head | Fixed | per-head `build_gamma` + shard merge + key-set assertion |
| N02 | P1 | gamma median computed with dead heads mixed in | Documentation correction | the report now gives both "all heads" and "live heads" figures |
| N03 | P1 | Layer width for layers 18/19 measured on the wrong basis | Documentation correction | per-tensor live row/column statistics |
| N04 | P0 | Evaluation forced parallel, so it did not test the training function | Fixed | pinned chunkwise + `--chunk-size` |

---

## 2. Item by Item (Root Cause and Current Disposition)

### E01–E04 Growth Script

The old `fill()` built new matrices with `torch.zeros`, the old `grow_norm()` wrote the new gates as 0, and the new embedding columns were 0 as well.
Together these three left the new parameters with a gradient of exactly 0 from step 0 onward; and because validation compared logits only, a "perfectly zero side branch" was bound to pass.
Now: new columns are set to 0 (preserves the function), new rows are random (preserves gradients), intermediate-layer gates are 1.0 (preserves gradients),
the final gate is 0.0 (preserves the output), and embedding `[grow_old_dim:]` is re-randomized (revives dead columns);
a new 5-step backward trainability check and a segmented norm layout assertion were added, and any failure refuses to persist (rc=2).
Two successive growths (48→72→96) run end to end in the tests, with function fidelity max|Δ| < 1e-2.

### E05 GrowthLayerNorm

The class default for `gamma_new` used to be 0, so multiplying the new channels gave a constant 0. The default is now 1.0 and `beta_new` stays 0;
the growth script overrides them explicitly instead of relying on "the default happening to be correct".

### E06 1-D Parameters and weight decay

`msr.gamma` is a bare `nn.Parameter`: it matched neither the bias rule nor the LayerNorm rule, so weight decay kept shrinking it all the way down.
Now `--no_decay_1d` is the default, and `--decay_1d` is kept to reproduce the old behavior; the parsing and wiring in `train.py` are covered by
`build_parser()` unit tests (the old test assigned `_no_decay_1d` by hand, so deleting the wiring still left it green).
A new `DecayGuardCallback` checks the optimizer parameter groups directly at the start of training and stops if a 1-D parameter appears in a
group with `weight_decay>0`. The gamma check in `_preflight` now walks both parameters and
buffers — with `use_gla=False`, gamma is a buffer, so that check was dead in the old implementation.

### E07 Gradient Coverage

The old implementation had only a comment. The new `GradCoverageCallback` registers a backward hook on every `requires_grad` parameter
at construction time and records `grad.abs().max()` (it cannot wait for `on_step_end` and read `p.grad`: HF has already called zero_grad),
summarizes the dead parameters at the end of step 1, and raises SystemExit when the element fraction exceeds `--dead_grad_tol` (default 2%).
`tests/test_grad_coverage.py` proves it goes red by freezing one parameter.

### E08 Live Parameters

`live_param_ratio` counts non-zero elements; `check_live_ratio` is a pure function that distinguishes four cases:
pass / explicit `--allow_dead_params` / consistent with the historical baseline / reject.
The historical baseline is written to `live_params.json` by `write_live_baseline` after training has started successfully.
This way "the new model is broken" is still rejected, "resuming from a deliberately pruned checkpoint" can proceed, and both are logged.

### E09–E11 Pack-Building Pipeline

`--add-eos` defaults to True; both callers pass it explicitly; the body field must be given explicitly, and when the field is absent entirely the
run exits on the first record; `build_ctx_mix` rejects any source with `add_eos=false` outright (letting it through requires an explicit `--allow-boundaryless`).
In an environment without a tokenizer these three are explicit skips, not fake passes (see the test section of the README).

### E12 Broken require_eos Chain

The `_FlatStage` error message told operators to write `require_eos: false`, but `Stage` had no such field,
`parse_plan` dropped it, and `train.py` did not pass it when building the stage dict — so the error could never be fixed.
Now `Stage.require_eos` is a first-class field, `parse_plan` parses it, `plan_stage_specs` passes it through,
and the hot-update signature includes it; `tests/test_plan_eos.py` walks from JSON to `stage_ready` to verify it works.

### E13 GC and the Rebuildability Proof

The first fix required `rebuild_cmd`/`source_ids`, but the producer never wrote either key — so "deleting the wrong thing"
became "never deleting anything", and the 6 `gc:true` stages in the plan freed 0 bytes.
Now the producer writes argv plus `rebuild_cmd` and `source_ids` into meta; GC accepts three kinds of proof:
`rebuild_cmd`, `source_ids`, and the legacy producer's `sources+text_fields`.
`protect()` registers the validation set directory, and `gc_stage` refuses to delete it using the normalized path (the old docstring said so, the function body did not).
`tests/test_producer_gc.py` runs the real producer loop end to end instead of using hand-written fixtures.

### E14 Same-Source Validation Set

The old guard compared `Path.resolve()` strings only, and the "alternative spelling" in the test was the same absolute path.
Now `retnet_hf/provenance.py` works in three layers: normalized paths (resolve + normcase),
the provenance declared in meta.json, and a content fingerprint of per-line 64-bit hashes. The content layer is capped at 200,000 lines by default,
and any truncation is reported faithfully. The tests trigger each layer separately with genuinely different spellings, genuinely same-source declarations, and genuinely duplicated lines.

### E15 Selection Policy

Three documents used to contradict each other (the code disabled it / 00F1 called it Fixed / docs/02 said it had to be kept). They are now unified:
`load_best_model_at_end=False` during training, saving by step count only; after training, `eval_holdout.py`
reports loss on the same basis for each candidate checkpoint and the choice is made from that. `docs/02` has been rewritten accordingly.
Residual limitation: there is no "one-click selection" script; the selection process is "run eval_holdout once per checkpoint, then compare".

### E16 parallel Normalization

Removed `mask/mask.sum().sqrt()` from `_decay_mask` and `/scores.abs().sum()` from `_parallel_retention`. The three paths (parallel/chunkwise/recurrent) now differ only in how they are implemented.
Evaluation is pinned to chunkwise, matching training.

### E17 Default Values

`DEFAULT_SAVE_TOTAL_LIMIT = 4` lives in `retnet_hf/defaults.py`, and both `train.py` and
`run_task.py` import it; the plan and non-plan branches both pass it explicitly.
The tests assert on argv for both branches and both values, not on the source via regex.

### E18 Hot-Editing the Step Count

`should_accept_reschedule` is a pure function: it accepts when the total step count is unchanged, when `allow_reschedule` is passed explicitly,
or when the new LR is at most 102% of the old LR; otherwise it rejects the whole change and prints both LRs.
The test builds two `Plan` objects directly and does not start a Trainer.

### E19 refresh Signatures

`_scan` records `(filename, mtime_ns, size)` at the moment of scanning; `refresh` compares the saved values against a fresh scan.
The test disables mmap, writes new same-size content in place, pushes mtime into the future, and asserts True the first time and False the second.

### E20 eval_batch_mult

The plan branch of `run_task.build_cmd` now passes the parameter; the test asserts that the value in argv equals the configured value.

### E21 recurrent Fast Path

When `tgt_len==1` it does a single state update and outer product directly, building no list and calling no `torch.stack`.
A new test proves that the fast path is really taken: `torch.stack` is replaced with something that raises, and the fast path still completes.

### E22 Cross-Entropy Chunking

The chunk size is derived backwards from a byte budget (`ce_chunk_tokens(V, 64MB)`): 838 tokens per chunk at V=20000,
and 40 chunks for the production shape B=16/L=2048. The test has two layers: numerical equivalence, plus recording the slice shapes handed to
`F.cross_entropy` at n=32768 and asserting that there are ≥2 chunks and that no chunk exceeds the budget.

### E23 Tests

Six new test files were added (preflight, gradient coverage, plan eos, producer GC, shard re-initialization, evaluation probes),
and several old tests that "could not fail for the reason they advertised" were fixed:
`rc in (0,1)` became `rc == 0` (the script no longer always returns 1 when `TARGET_PARAMS=0`),
E21 gained branch counting, E22 gained structural assertions, and E14 was switched to genuinely different spellings.

### N01 reinit_gamma

`gamma` is written by head index (`build_gamma`), and a duplicate value within the same layer is rejected; for sharded checkpoints all shards are merged via
`model.safetensors.index.json`, and after writing a single file the code asserts that the key set matches the source and that
gamma really was replaced; multiple shards without an index are rejected outright (the old implementation read only the first shard).

### N02 / N03 Reporting-Basis Corrections

N02: the gamma median must be given on both bases, "all heads" and "live heads"; fixed values in dead heads skew the median.
N03: layer width comes from per-tensor live-row/live-column statistics and must not be inferred from d_model in the config. Neither item involved a code change.

### N04 Evaluation Path

`eval_holdout.py` pins `mode="chunkwise"` and the same `chunk_size` as training;
it adds two probes, `--by-position` (bucketing by position) and `--ctx-ablation` (Δ_ctx),
and validates that `ctx` is an integer multiple of the chunk.

---

## 3. Audit-Report P0/P1/P2 Items and Their Disposition

Some items in the audit report carry no E/N number (documentation completeness, test trustworthiness, performance items);
their current disposition is given item by item below, so that "only E/N were fixed" does not pass as done.

| Report item | Content | Current disposition |
|---|---|---|
| P0-1 | E22 chunking produced only one chunk at the production shape | Fixed: byte budget + chunk-count assertion (see E22) |
| P0-2 | The E08 guard made the rescue plan unable to start | Fixed: `--allow_dead_params` + baseline (see E08) |
| P0-3 | E12's escape hatch did not take effect | Fixed: `require_eos` through the whole chain (see E12) |
| P0-4 | The check the E07 comment claimed did not exist | Fixed: a real callback (see E07) |
| P0-5 | The documentation status table contradicted the README, and verify did not check it | Fixed: `fix_status.py` as the single source + verify item G |
| P0-6 | The GC guard could not be satisfied by the producer | Fixed: producer writes the proof + legacy-format compatibility + a real-producer test (see E13) |
| P1-1 | The README test command and "56 passed" were not reproducible | Fixed: command changed to `cd code`; real count 82 passed / 4 skipped; the 5 out-of-scope files are explicitly excluded in conftest with an explanation |
| P1-2 | E17 was only half fixed, run_task still used 3 | Fixed: single constant + argv tests for both branches (see E17) |
| P1-3 | E06 wiring untested, `--decay_1d` could not be passed in, gamma buffer check absent, step-ratio assertion did not exist | Fixed: `build_parser` unit tests + the flag in both directions + buffer check + optimizer-group assertion; the false "step 100/1000" promise was removed |
| P1-4 | E15's three specifications were mutually exclusive | Fixed: unified as "training-time selection off + offline eval_holdout"; docs/02 rewritten |
| P1-5 | E14 compared paths only and the test passed vacuously | Fixed: three-layer check + a test with genuinely different spellings (see E14) |
| P1-6 | Two probes were missing from `eval_holdout.py` | Fixed: `--by-position` / `--ctx-ablation` + end-to-end tests (see N04) |
| P1-7 | `reinit_gamma` silently truncated shards | Fixed: shard merge + key-set assertion + rejection of multiple files with no index (see N01) |
| P1-8 | The `grow_500m` header contradicted the implementation | Fixed: header rewritten as a whole-segment gate mapping + layout assertions + a two-growth test; the premise that "the source's old gates are not inherited" is stated explicitly |
| P1-9 | Seven technical errors in docs/01 | Fixed: six rewrites (9.2% direction, checkpoint layer semantics, resume field, bitwise equivalence, GC validation set, the two sets of DEFAULTS values, two-hour rebuild); rewritten uniformly by the generator |
| P2-1 | `--eval_batches` was a dead parameter | Fixed: it is now a real cap on the number of evaluation rows |
| P2-2 | flat `weight` lost, hot update dropped keys | Fixed: `_stage` passes weight; `Stage.signature` covers all fields; hot update passes `weight`/`require_eos` |
| P2-3 | `retention_fast`'s matmul count and path were outdated | Fixed: 5 matmuls; points to `tests/test_fast_chunkwise.py`; reworded to "equivalent up to rounding, not bitwise" |
| P2-4 | Citation markers were inserted into code fences | Fixed: `_add_citations._safe_insert` tracks fence state and moves them before the fence |
| P2-5 | The dropout position in docs/04 was drawn wrong | Fixed: dropout sits inside MSR, before out_proj; the residual add has no dropout |
| P2-6 | The line-count arithmetic in the docs/00 header | Fixed: this file was rewritten and no longer gives a total where "the sum of six files poses as seven" |
| P2-7 | docs/03's "training is pinned to chunkwise" holds only for the task configuration | Fixed: it now states that it holds only when the launcher passes `mode=chunkwise`; running `train.py` directly defaults to parallel |
| P2-8 | Several tests could not fail for the reason they claimed | Fixed: each was converted to behavioral/structural assertions (E21/E22/E14/E12/E23) |
| P2-9 | Blind spots in verify.py | Fixed: three new items — status table, literal contract, MANIFEST; the scope is written into the docstring and the README |
| P2-10 | `_INDEX_CACHE` was unbounded | Fixed: OrderedDict LRU, capped at 8 |
| P2-11 | fused CE computed the state for nothing and rebuilt the head on every chunk | Fixed: `need_state=False` skips the state einsum; the head's fp32 cast is done once per forward pass |

---

## 4. Items Still Open

1. **Facts about the data/checkpoints cannot be verified inside the snapshot** (`retnet_500m_v21_export`, `hf_pack_*`,
   `val_pack` are not in the directory): the 67% zero parameters, the live-head gamma median, and whether the v22 mixture contains eos
   all need one more run of `_preflight` / `eval_holdout.py` with the current code on the data machine before they can be settled.
2. **The content layer of the validation set check is a sampled check**: the default cap is 200,000 lines and anything beyond it is not scanned; the declared-provenance layer depends on meta.json.
3. **Offline selection has no one-click script**: a person has to run `eval_holdout.py` on each candidate checkpoint in turn.
4. **The performance claims were not reproduced**: this snapshot verifies numerics and structure only, with no GPU benchmark.
