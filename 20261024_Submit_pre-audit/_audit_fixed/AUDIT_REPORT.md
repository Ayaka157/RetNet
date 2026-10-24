# Adversarial implementation audit — `20261020_Submit`

Audit target: `E:\Workspaces\RetNet_Submissions_Review\20261020_Submit` (read-only; nothing in the submission was modified).
Environment used for live checks: `C:\tools\miniconda3\envs\AI\python.exe` — Python 3.13.7, torch 2.10.0+cu128, transformers 4.57.6, triton 3.7.1 (the stack the README claims).
Method: docs and code read as two competing specifications, through three lenses (theoretical correctness, implementation fidelity, engineering risk). Every finding below was either executed or traced to a cited line; anything I could not settle is marked UNSURE.

Severity legend: **P0** = blocks the documented workflow or silently defeats a claimed safety property; **P1** = the claim is materially false or the guard only half-works; **P2** = correctness/robustness debt that will bite the next maintainer.

---

## What the submission claims, and what reproduces

| Claim | Reproduces? | Evidence |
|---|---|---|
| `python verify.py` → six checks, exit 0 | **Yes** | `[A] 83`, `[B] 83/83`, `[D] 27 ids / 19 refs / 0 dangling`, `[E] 27 / 34 blocks`, `[F] 27/27`, `✅`, exit 0 |
| `MANIFEST.sha256` covers every file | **Yes** | 55 entries, 55/55 match, 0 missing, 0 unlisted (56 files incl. the manifest) |
| `pytest tests -q` → 56 passed | **No** | from `20261020_Submit`: exit 4 (`tests` not found); from `code/`: 36 collected, **32 passed, 4 skipped, 5 collection errors** |
| `test_plan_gc.py` / `test_plan_pipeline.py` pass | **Yes** | both print `全部通过`, exit 0 |
| 27 fixes landed, each with a test/check | **No** | 9 of them are listed as 未修 by the submission's own `docs/01`; see P0-5 |

---

## P0 findings

### P0-1 · The E22 "chunked CE" fix does not chunk at the documented shape, and is worse at larger batches
- **Location**: `code/retnet_hf/modeling_retnet.py:373-411` (`chunk_tokens: int = 262_144`, loop at `:402`); call site `:576-581`; repeated in `docs/00F1:830-841`, `docs/01:1165-1166`, `docs/04:341-349`.
- **Lens**: theoretical correctness + engineering risk.
- **Evidence (executed)**: patched `F.cross_entropy` to record input shapes.
  - `n = 32768` (the documented `B=16 × L=2048`): **1 call**, shape `[32768, V]` → at `V=20000` that is `32768×20000×4 = 2.62 GB` fp32, exactly the copy the fix claims to remove.
  - `n = 600000`: chunks `[262144, 262144, 75712]` → at `V=20000`, **20.97 GB per chunk**.
- **Reasoning**: the loop step is the chunk size, so whenever `B·L ≤ 262144` there is a single chunk and the "chunked" path is byte-for-byte the old allocation. The docstring even contains the contradiction inline: "`262144×20000×4 ≈ 21 GB？——不，chunk 是 token 数，实际按 N=32768 切`" — no such `N=32768` logic exists. `docs/04:348` then asserts "只是不物化 `[B,L,V]` 的 fp32 副本", which is false at the target shape.
- **Fix**: derive the chunk from a token budget, e.g. `chunk_tokens = min(n, max(1, 2**25 // V))` (≈1638 tokens at V=20000, ≈64 MB), expose it, and add a test that asserts **≥2 chunks at the production shape** (the current test passes `chunk_tokens=7` on 37 tokens and would also pass if the argument were ignored).
- **Verdict**: CONFIRMED.

### P0-2 · The E08 `_preflight` guard makes the submission's own rescue plan unstartable
- **Location**: `code/retnet_hf/train.py:226-231` (`if ratio < 0.95: raise SystemExit`), called unconditionally at `:352`; `docs/02:36,53-55`; `docs/04:405`.
- **Lens**: engineering risk (integration of two "fixed" items).
- **Evidence (executed)**: zeroing ~2/3 of a tiny model gives
  `[preflight] live params 9,346/27,204 = 34.36%` → `SystemExit: ... 模型里有大片恒零参数，先修生长/初始化，别拿它训练。`
  The plan's starting point is `checkpoints/retnet_500m_v21_export`, documented as **338,120,704 / 504,179,896 = 67.06% zero** (`docs/00:137`, `docs/03:101`), i.e. 33% live. `docs/02` §1 says "在 `checkpoints/retnet_500m_v21_export` 上继续训，不缩壳、不重训、不生长".
- **Reasoning**: `_preflight` has no bypass flag, no `--resume` special case and no threshold argument. The first command of the ¥187 plan therefore aborts before step 0. `docs/02` §2 E08 labels the guard "硬前提（便宜）" and only debates whether live is 166M or higher — even 300M would be 59% < 95%.
- **Fix**: separate the two cases the guard conflates: fresh-start models must be ≥95% live; resuming an existing (possibly pruned) checkpoint must compare against that checkpoint's own live ratio, or require an explicit `--allow_dead_params` that logs the ratio and continues. Update `docs/02` §1/§2 accordingly.
- **Verdict**: CONFIRMED.

### P0-3 · E12's documented escape hatch is inert, and plan mode cannot open the rescue mixtures
- **Location**: `code/retnet_hf/data.py:97-98,120-149,270,293-297`; `code/retnet_hf/plan.py:66-75` (`Stage`), `:149-168` (`parse_plan`), `:517-520`; `code/retnet_hf/train.py:358-361`; error text/`docs/00F1:381`, `docs/01:1836,1877`.
- **Lens**: implementation fidelity + engineering risk.
- **Evidence (executed)**:
  - `parse_plan({"stages":[{"...","require_eos": False}]})` → `Stage` fields = `data_dir, exclude, gc, min_shards, mode, name, steps, weight` — **no `require_eos`**.
  - Building `StagePackedDataset` exactly as `train.py` does for plan mode on an eos-free shard → `stage_ready(0) → (False, "ValueError: ... 没有任何 <eos> ... 在计划里给该阶段写 require_eos: false")`, and `len(ds)` raises the same `ValueError`.
- **Reasoning**: the guard's own error message tells the operator to write a key that `plan.py` drops, so the advertised remedy does nothing. `docs/02:169-171` plans `rescue_m1/m2/m3` from CFE/wiki/poetry/fineweb/cosmo/math; `docs/02:124` states only `pretrain_v2` has eos ("`pretrain_v2` 不缺"), and none of the v22 mixtures contains `pretrain_v2`. The failure is a startup crash, not the "wait for data" path.
- **Fix**: add `require_eos: bool = True` to `Stage`, copy it in `parse_plan`, and include it in **all three** stage-dict builders (`train.py:359-361`, `plan.py:517-520`); add a test that goes JSON → `parse_plan` → `StagePackedDataset.stage_ready` (the current test only exercises `_FlatStage(require_eos=...)` directly).
- **Verdict**: CONFIRMED (mechanism). The on-disk mixtures' eos content is UNSURE — the packs are not in the snapshot.

### P0-4 · The E07 comment asserts a gradient audit that does not exist — and the submission's own doc says so
- **Location**: `code/retnet_hf/train.py:425-430`; `docs/00F1:208-221`; `docs/01:17,134,626`; `docs/00:482` (§4 item 1).
- **Lens**: implementation fidelity.
- **Evidence**: the shipped comment says "**另加**一次梯度覆盖检查：第 1 步之后统计 `grad.abs().max()==0` 的参数，超过阈值直接停机。" `grep grad.abs` over `code/` finds it only in `scripts/grow_500m.py:282`. `docs/01:626` states "梯度覆盖检查（00 §4 第 1 条）**尚未落地**，是下一项待补的断言"; `docs/01:17` marks E07 "部分修".
- **Reasoning**: E07's original defect was a comment that rationalized away missing detection. The fix replaces it with a comment that claims detection exists. This is the same failure mode, one level harder to catch, because the word "另加" implies code elsewhere. `docs/00` §4 item 1 ("step 1 之后…必须为 0") is still unmet.
- **Fix**: delete the sentence, or implement it as a `TrainerCallback` that at `global_step == 1` collects `p.grad is None or p.grad.abs().max()==0` and aborts above a threshold; add a test that a deliberately frozen parameter turns it red.
- **Verdict**: CONFIRMED.

### P0-5 · The submission's own per-function document contradicts its README on 9 of 27 fixes, and the verifier certifies it anyway
- **Location**: `docs/01:9-33` (status table) vs `README.md:28-39` vs `docs/00F1` statuses; verifier `verify.py`.
- **Lens**: implementation fidelity / documentation integrity.
- **Evidence**: `docs/01` says **未修** for E10, E11, E12, E15, E18, E19, E21, E22, E23, and **部分修** for E07. The code implements them: E10 `_data_supervisor.py:190` + `_prep_ext.py:84`; E11 `build_ctx_mix.py:82-91`; E12 `data.py:120-149`; E18 `plan.py:249-269`; E19 `data.py:185-201`; E21 `modeling_retnet.py:210-214`; E22 `modeling_retnet.py:373-411`; E23 `tests/test_growth_trainable.py:86-98`; E15 `train.py:412`. `docs/01` also contradicts **itself**: `:25` says E15 未修 while `:463` shows `load_best_model_at_end=False` and `:626` says "已关闭".
- **Reasoning**: `verify.py` checks code-block byte-identity, heading coverage, explanation length and ID references — never the status table. So the strongest-looking artifact in the submission (`✅ 全部通过`) coexists with a table that tells a reviewer nine fixes are absent. This is the cleanest demonstration of the audit thesis: the verification is sound but scoped to the properties that happen to hold.
- **Fix**: derive the status table from a single machine-readable source (e.g. a `FIX_STATUS` dict consumed by both `_gen_code_doc.py` and README generation), and add a `verify.py` check that the table equals `00F1`'s statuses. Regenerate doc 01.
- **Verdict**: CONFIRMED.

### P0-6 · E13's GC guard can never be satisfied by the tool that produces the data
- **Location**: `code/retnet_hf/data.py:377-380` (`m.get("rebuild_cmd") or m.get("source_ids")`); producer `code/scripts/build_flat_corpus.py:319-333`; test fixture `code/tests/test_plan_gc.py:4b`.
- **Lens**: engineering risk.
- **Evidence**: the producer's `meta.json` keys are `tokenizer, tokenizer_vocab, sources, text_fields, sep, add_eos, records_seen, records_errors, total_tokens, shards, dtype, built_at` — neither `rebuild_cmd` nor `source_ids` is ever written. The passing 4b test hand-writes a `{"rebuild_cmd": ...}` file that no shipped tool creates.
- **Reasoning**: the fix converts "delete without proof" into "never delete". With `plan_ddp2.json`'s six `gc: true` stages, every `gc_stage` call prints a refusal and frees 0 bytes; the disk headroom the plan depends on never materializes. The test cannot catch it because its fixture is not a producer output.
- **Fix**: write `rebuild_cmd` (the exact argv is already available in `_data_supervisor`) and `source_ids` into the producer's meta; accept `sources + text_fields` as a legacy proof; make the test run the real producer on two tiny records.
- **Verdict**: CONFIRMED.

---

## P1 findings

### P1-1 · The documented test command and the "56 passed" claim do not reproduce
- **Location**: `README.md:61-73`; `docs/00F1:910` ("41 项通过"); `code/tests/`.
- **Lens**: implementation fidelity.
- **Evidence (executed)**: `cd 20261020_Submit; pytest tests -q` → `ERROR: file or directory not found: tests` (exit 4; the directory is `code/tests`). From `code/`: `36 tests collected, 5 errors`; with `--continue-on-collection-errors`: **32 passed, 4 skipped, 5 errors**. The 5 erroring files import packages absent from the submission (`bench`, `retnet`, `play_v2`): `test_checkers` (15 tests), `test_model` (5), `test_bench_equivalence`, `test_play_v2_prompt`, `test_runner_lockstep`. `def test_` totals exactly 56, so the number is arithmetically right but unreachable here; `00F1` claims 41 (exactly 56−15).
- **Reasoning**: 20 of 56 tests live in files that cannot be imported from this tree. The README discloses only the 4 tokenizer skips, not the five dead files.
- **Fix**: delete or vendor the 5 files (or add them to `collect_ignore_glob` with a reason), fix the README command to `cd code`, and state the true counts; reconcile 41 vs 56.
- **Verdict**: CONFIRMED.

### P1-2 · E17 is half-fixed: the launcher's default still wins, and the guard checks the wrong file
- **Location**: `code/run_task.py:69` (`DEFAULTS["save_total_limit"] = 3`), `:158` (`cfg.get(..., 3)`); `code/retnet_hf/train.py:303-306` (default 4); `tests/test_train_config_guards.py:28-32`; `docs/00F1:596-604` ("两处默认值必须同时改").
- **Lens**: implementation fidelity.
- **Evidence (executed)**: `python code/run_task.py --config <plan config without save_total_limit> --dry` → `--save_total_limit 3`. Because `load_config` merges `DEFAULTS`, even the plan branch's fallback 4 (`run_task.py:127`) is unreachable.
- **Reasoning**: E17's stated defect was exactly "配置写 4、实际跑 3". The test only regexes `train.py`, so it is green while the launcher still forces 3. `00F1` asserts both branches pass it down; the non-plan branch passes 3.
- **Fix**: remove the key from `DEFAULTS` (or set 4), use one literal (`cfg["save_total_limit"]`), and add a `run_task` argv test for both branches.
- **Verdict**: CONFIRMED.

### P1-3 · E06's wiring is untested, cannot be disabled through the launcher, and its gamma preflight is dead for `use_gla=False`; the promised step-ratio assertion does not exist
- **Location**: `code/retnet_hf/train.py:457` (only wiring), `:314-320` (defaults), `:233-240` (gamma check), `:58-61` (stale docstring "默认关闭"); `run_task.py:186-187`; `tests/test_no_decay_1d.py:27-31`; `docs/01:71,285`; `docs/00F1:196`.
- **Lens**: implementation fidelity.
- **Evidence (executed/inspected)**:
  - `--enable` path: `test_no_decay_1d` builds the trainer with `CompileSafeTrainer.__new__` and **hand-assigns** `t._no_decay_1d`; deleting `train.py:457` restores the original bug with all tests green.
  - `run_task` only appends `--no_decay_1d` when truthy; `no_decay_1d: false` in JSON is silently ignored (dry-run shows no decay flag; train.py's default is True).
  - `_preflight` collects `n for n,p in model.named_parameters() if n.endswith("msr.gamma")`; with `use_gla=False` gamma is a **buffer** → probe: 0 names, so no print and no `<=0` rejection.
  - No step-100/1000 gamma ratio assertion exists (`grep gamma_init|print.*gamma` in `train.py` finds only the preflight).
- **Reasoning**: E06 is the most expensive item in the plan ("这是本轮最贵的一条"). Three of its four advertised guards are either absent or unreachable, and the code's own docstring still says the switch is "默认关闭".
- **Fix**: pass `--decay_1d` when the JSON says false; test the argparse→trainer wiring (refactor the flag into a pure function); read gamma from `named_parameters()` **and** `named_buffers()`; implement the step-ratio callback or delete the claim from `_preflight`'s docstring, `docs/01` and `00F1`.
- **Verdict**: CONFIRMED.

### P1-4 · E15 has three mutually exclusive specs
- **Location**: `code/retnet_hf/train.py:407-414` (`load_best_model_at_end=False`, test enforces it); `docs/00F1:526-544` ("已修"); `docs/01:25` ("未修"); `docs/02:61` ("**保留** `load_best_model_at_end` … 不要关掉它，关掉等于放弃选优… 硬前提").
- **Lens**: implementation fidelity.
- **Evidence**: quoted above; `tests/test_train_config_guards.py:21-25` locks the code's choice in.
- **Reasoning**: the operational plan's hard prerequisite is the negation of the shipped code. A reader following `docs/02` will look for a best-checkpoint selection that will never happen.
- **Fix**: choose one policy. If the code's choice stands, rewrite `docs/02` §2 E15 to "offline selection after training" and point at the (missing) selection script; otherwise re-enable with a clean `val_pack` and update README/doc 01.
- **Verdict**: CONFIRMED.

### P1-5 · E14's guard is path-equality only, and its third test case is vacuous
- **Location**: `code/retnet_hf/train.py:198-210`; `tests/test_val_disjoint.py:34-39`; `docs/00F1:520`.
- **Lens**: theoretical correctness + implementation fidelity.
- **Evidence**: `rel = Path(str(d))` is the identical absolute path — not "同一路径的另一种写法" as the comment and `00F1` claim ("相对路径、大小写、尾斜杠…"). The guard compares `Path(...).resolve()` strings, so content-level contamination is invisible: `plan_ddp2.json` keeps `val_dir: chat_pack_v13` while `sft_mix` (40% of it, per `docs/00:328`) is materialized in `hf_pack_sft` (`plan_ddp2.json:94-98`) — different directories, same records.
- **Reasoning**: the documented failure was data-level, the guard is path-level; `docs/02` replaces `val_dir` with `val_pack` (clean), so the specific new plan is fine, but the guard does not enforce the property it is named for.
- **Fix**: record pack ids/row hashes in each stage's `meta.json` and in the val pack, and assert disjoint provenance; fix the test to use a genuinely different spelling (relative path via `os.path.relpath`, trailing separator, case variation).
- **Verdict**: CONFIRMED (guard/test); whether the v22 data still overlaps is UNSURE.

### P1-6 · `eval_holdout.py` lacks the two probes the plan's gates require
- **Location**: `code/scripts/eval_holdout.py:31-43` (args), used by `docs/02:205-209,336`.
- **Lens**: engineering risk.
- **Evidence (executed)**: `--by-position`: ABSENT, `--ctx-ablation`: ABSENT, `--chunk-size`: PRESENT. `docs/02:336` instructs `eval_holdout.py … --by-position --ctx-ablation`; `docs/02:5` claims "本文只写能开机执行的东西".
- **Reasoning**: P1's acceptance criterion and G1c's second gate depend on `Δ_ctx`, which no shipped script computes; the command fails at argparse.
- **Fix**: implement the two flags (the doc estimates ~40 lines) or remove them from the plan and replace the gate with something measurable now.
- **Verdict**: CONFIRMED.

### P1-7 · `reinit_gamma.py` silently truncates sharded checkpoints
- **Location**: `code/scripts/reinit_gamma.py:106-111`.
- **Lens**: engineering risk.
- **Evidence**: `files = sorted(src.glob("*.safetensors")); sd = load_file(str(files[0]))`, then `save_file(sd, dst/"model.safetensors")` and only `.json/.txt` are copied. For a checkpoint split into `model-00001-of-00002.safetensors` + `…-00002…`, the output contains only shard 1's tensors, with no key-count or completeness check.
- **Reasoning**: `docs/02`'s A2 stage plans to run this on `retnet_500m_v21_export`; a sharded source would produce a model that loads with missing keys (or fails), silently.
- **Fix**: require exactly one file or merge shards via `model.safetensors.index.json`; assert the written key set equals the source key set.
- **Verdict**: CONFIRMED by inspection; whether the export is sharded is UNSURE (checkpoint not in snapshot).

### P1-8 · `grow_500m.py`'s header contradicts its own `grow_norm` about preserving the source gate
- **Location**: `code/scripts/grow_500m.py:24-26` vs `:167-191`; `docs/00F1:47-83`; `docs/01:15` ("生长脚本不再覆写为 0").
- **Lens**: theoretical correctness / fidelity.
- **Evidence**: the header says the mapping is "`ln_old(768)` + `gamma_new[0:256]`（源值）+ `gamma_new[256:640]`（**零**）" and calls the result a zero side branch. The code writes `torch.full((640,), gate)` with `gate = 1.0` for block norms and `0.0` for `final_norm` — the source's learned `gamma_new[:256]` is discarded and the new dims are **not** a zero branch. Function preservation at the logits still holds, but only because `final_norm`'s gate is 0; the text's mechanism is wrong. `docs/01:15` compounds it by saying the script "不再覆写为 0" when `final_norm` is exactly 0 by design.
- **Reasoning**: this is the recipe a future growth will follow; the stated mapping is not the implemented one, and the "zero side branch" justification would mislead a re-derivation (e.g. someone setting `BLOCK_GATE=1` "because the doc says the branch is zero" would be surprised twice).
- **Fix**: rewrite the header to the implemented mapping (old segment copied, new segment gated 1.0 in blocks / 0.0 at final), and assert the layout after growth (`gamma_new[:640] == 1` for blocks, `== 0` for final).
- **Verdict**: CONFIRMED.

### P1-9 · Seven concrete false technical claims in `docs/01` (prose, not code blocks)
- **Location / evidence** (each verified against the cited code):
  1. `docs/01:842` — "把归一化挪到 permute 之前会**快 9.2%**…收益为负". The code comment (`modeling_retnet.py:157-164`) measured the opposite: "**实测反而慢 9.2%**（190 vs 174 ms/微步…）". The sentence is also self-contradictory.
  2. `docs/01:1149` — "本层编号**达到**检查点阈值时才启用". Code (`modeling_retnet.py:363-365`) disables at/after the threshold: `if 0 <= checkpoint_layers <= layer_idx: do_ckpt = False` → layers **below** the threshold are checkpointed.
  3. `docs/01:210` — `_sync_resume_state` "把…`grad_accum`、`max_steps` 等字段…逐项比较". Code (`train.py:162-167`) compares only `train_batch_size, logging_steps, eval_steps, save_steps`.
  4. `docs/01:1677` — "数学上与 `_chunkwise_retention` **逐位等价**，有测试保证". Probe: max|Δ| = 1.9e-6 (γ=0.5), 1.5e-5 (γ=0.99), 6.8e-3 (γ=1.05); and `test_fast_chunkwise.py:188-193` explicitly fails if the two are bit-identical ("比较退化为空过").
  5. `docs/01:2222` — `gc_stage` lists "验证集目录跳过" as a safety constraint. The body (`data.py:342-397`) has no val-dir check at all (the code docstring repeats the claim).
  6. `docs/01:2413` — `DEFAULTS` "把所有默认值集中在一处，避免同一个字段在 `train.py` 与 `run_task.py` 里各有一套默认值". The two files disagree on at least **eight** fields: `mode` (chunkwise/parallel), `precision` (fp32/bf16), `batch_size` (6/10), `max_steps` (65000/45000), `warmup` (500/1000), `save_total_limit` (3/4), `grad_checkpoint` (True/False), `compile` (False/True).
  7. `docs/01:1964` — E13 "要求在删除前先确认源数据仍在、并且存在一条能在**两小时内**重建它们的…命令". `gc_stage` only checks that `meta.json` contains one of two keys; there is no source-existence check and no time bound ("两小时" exists only in the generator text `_gen_code_doc.py:477`).
- **Lens**: implementation fidelity.
- **Reasoning**: `verify.py` cannot see any of this: prose is not compared, and the code blocks around it are byte-identical. 70 of 83 sections are accurate; these seven (plus P0-5's table) are not.
- **Fix**: move these sentences into a claims table checked against code (`grep`-able invariants: defaults, thresholds, tolerances), or regenerate the prose from docstrings that were themselves reviewed.
- **Verdict**: CONFIRMED for each.

---

## P2 findings

### P2-1 · `--eval_batches` is a dead argument
`train.py:263` defines it; nothing reads it. `run_task.py:123,147` pass `40` in both branches. Exactly the "死字段" class the docs are built around.
**Fix**: wire it into an eval-batch cap or delete it and the two argv entries. CONFIRMED.

### P2-2 · Flat-stage `weight` is silently dropped, and plan hot-reload drops `weight`/`require_eos`
`StagePackedDataset._stage` (`data.py:295-297`) constructs `_FlatStage` without `weight`; probe: plan `weight=3` → `wcodes` = `[1]`, while `_FlatStage(weight=3)` directly yields `[3]`. `plan.py:517-520` also omits both keys on update.
**Fix**: pass both through all construction sites; test the factory (not the class). CONFIRMED.

### P2-3 · Overstated equivalence and a stale reference in `retention_fast.py`
Module docstring says "3 matmuls total" — the function does 5 (`U`, `S_all`, `qk`, `inner@v`, `cross`). `modeling_retnet.py:25` points at `scripts/test_fast_chunkwise.py` (it lives in `tests/`). `docs/00:433` says "与参考实现逐位一致" — false per P1-9.4.
**Fix**: correct the counts/paths; say "equivalent to fp rounding, not bitwise". CONFIRMED.

### P2-4 · A citation marker was inserted inside a fenced code block in `docs/02`
`docs/02:223-225`: the python fence opens, then blank line, then `（参 [03]）`, then `# S1 ...`. The pseudo-code block is syntactically broken and renders as prose. The citation inserter (`_add_citations.py`) is not fence-aware.
**Fix**: make the inserter skip fenced regions; regenerate 02. CONFIRMED.

### P2-5 · `docs/04` §1 places dropout on the wrong side of `out_proj`
The equation says `a = MSR(h)`, `h' = h + Dropout(a)`. Code (`modeling_retnet.py:277-280`) applies `dropout` **inside** MSR after gating and **before** `out_proj`; the residual add has no dropout. The FFN line matches the code.
**Fix**: draw dropout before `W_O` (or state that it is pre-output-projection). CONFIRMED.

### P2-6 · `docs/00` header arithmetic
"7 个文件 / 2,240 行" — 2,240 is exactly the sum of the **six** per-file counts in §3 (506+621+180+352+345+236); `fused_ce.py` is the seventh and is omitted.
**Fix**: correct the total or drop the per-file total. CONFIRMED.

### P2-7 · `docs/03` "training is fixed to chunkwise / closed-form scan" is launch-path dependent
`train.py:289-290` defaults `--mode parallel`; `FAST_CHUNKWISE` defaults to `"0"` (`modeling_retnet.py:26`) and is only enabled by `run_task`'s `fast_chunkwise` env. The statement is true for `tasks/v21_resume.json` (`mode=chunkwise`, `fast_chunkwise=true`), false for a direct `train.py` invocation. `docs/01:1017` repeats it without the caveat.
**Fix**: say "the shipped task config runs chunkwise"; change the argparse default or print the effective mode prominently (train.py already prints the kernel, but not `mode`). CONFIRMED.

### P2-8 · Tests that cannot fail for the reason they advertise
- `test_growth_trainable.py:82`: `assert rc in (0,1)`, but `_patch_dims` sets `TARGET_PARAMS=0`, so `ok_n` is always False and success always returns **1** — the `0` branch is dead; deleting the whole trainability block still passes this line (only `:97` protects E04).
- `test_eos_present.py:30-45`: all three cases pass `require_eos` explicitly; flipping the default at `data.py:98` to False breaks nothing.
- `test_regressions_extra.py:89-106` (E21): `manual_ns`/`manual_out` re-type the implementation's two lines; E21 is a performance defect, and deleting the `tgt_len == 1` branch leaves the values identical.
- `test_regressions_extra.py:110-136` (E22): numerics only, `chunk_tokens=7`; it passes if the whole-tensor `logits.float()` returns (P0-1).
- `test_plan_gc.py` (E13's only guard) is excluded from pytest by `tests/conftest.py:18-21`; 7 other files have zero `def test_` and silently collect nothing.
**Fix**: add behavior assertions (chunk count at production shape; default `require_eos`; branch-taken/perf counters; key-set equality for growth output). CONFIRMED.

### P2-9 · `verify.py` is sound but blind exactly where the defects live
It covers 7 files (no `plan.py`, no `control.py` — where E18 lives), checks code blocks/heading coverage/length/ID refs, and never compares prose, defaults, wiring or the status table. It also does not verify `MANIFEST.sha256`. `docs/04:34,423` discloses the plan.py omission, so this is a scope limitation rather than a lie — but the README calls it "独立校验脚本" and prints "✅ 全部通过" immediately above a doc table that says nine fixes are missing.
**Fix**: add (a) a status-table cross-check, (b) a small "claims" check for identifier/default/threshold strings, (c) manifest verification, (d) include `plan.py`/`control.py` in `FILES` or state the scope in the README.

### P2-10 · `_INDEX_CACHE` is an unbounded module-level cache
`retention_fast.py:30,39-47`: keyed by `(chunk_size, str(device))`, never evicted; holds `[C,C]` tensors. With many chunk sizes/devices in one process (sweeps, tests) it grows monotonically; it is also not thread-safe and can pin tensors on a device that is being emptied.
**Fix**: `functools.lru_cache(maxsize=8)` or key by device index and bound it. CONFIRMED (minor).

### P2-11 · The fused-CE path does unnecessary work and re-converts the head every chunk
`modeling_retnet.py:553-557` passes `return_state=True` for the whole forward, so every layer computes `_parallel_final_state` (an `einsum` over `[B,H,L,kd,vd]`) even though `new_states` is discarded. `fused_ce.py:41` recomputes `weight.float().t()` per chunk (112 MB for the 0.5B head) instead of hoisting it.
**Fix**: add a `need_state` flag for the fused path; hoist the fp32 head conversion once per `forward`. CONFIRMED (performance only).

---

## Checked and found clean

- **`MANIFEST.sha256`**: 55/55 hashes match, no missing entries, no unlisted files (the 56th file is the manifest itself).
- **`verify.py`'s six checks** reproduce exactly, including 83/83 function coverage, 34 byte-identical `00F1` blocks, 27/27 IDs and exit 0.
- **E16's fix is real**: on the shipped stack, `max|parallel − chunkwise| = 5.4e-7` and `max|chunk64 − chunk32| = 5.4e-7` (fp32, tiny model). The deleted normalizations are gone; `generate`'s parallel prefill now targets the same function.
- **The fast chunkwise math**: matches the reference exactly at γ=0 (the `clamp_min(tiny)` + `where` handling correctly reproduces `0**0 = 1`, `0**k = 0`), 1.9e-6 at γ=0.5, 1.5e-5 at γ=0.99. The closed-form scan `T[h,i,j]=γ^{i-1-j}` and the state/cross-decay formulas match `docs/04` §4.4–4.5 and the reference loop.
- **Fused CE algebra**: `loss = Σ w·CE / Σw`, `dL/dlogits = (softmax − onehot)·w/Σw`, `grad_h = (dL/dlogits) @ W`, `grad_w = (dL/dlogits)^T @ h` — all correct on inspection (with the usual `ignore_index=-100` caveat, which the data pipeline never produces).
- **Gradient checkpointing**: `gradient_checkpointing_enable()` under transformers 4.57.6 **does** reach the blocks (`[True, True]`), and `set_checkpoint_layers(n)` correctly enables layers `< n`. My initial suspicion that the flag was inert was refuted by the probe — no finding.
- **Tied-weight growth pattern**: loading a model's **own** `state_dict` after replacing `embedding.weight` preserves the replacement (the two state-dict entries alias the same storage; `lm_head` comes last but copies the already-updated storage). No hidden corruption there.
- **`_chunked_weighted_ce` numerics**: 2.4e-7 from the naive implementation with `chunk_tokens=16` (the arithmetic is right; the default size is the problem, P0-1).
- **`run_task --dry`**: the plan branch passes `--eval_batch_mult` (E20's fix works when the JSON sets it), and `--save_total_limit 4` is passed when the task JSON says 4.
- **The 4 `test_build_scripts` skips are genuine** (module-level `skipif` on the gitignored tokenizer), and `def test_` really totals 56.
- **`control.py` stop semantics** are consistent with HF's `on_step_end` → `_maybe_log_save_evaluate` → stop ordering (inspected; not executed under a live Trainer).
- **`configs/tasks/plan_ddp2.json` arithmetic**: stage steps sum to 159,857; 65,536 tokens/step follows from `nproc=2 × batch 16 × 2048` in `v21_resume.json`; the per-1B-token price and token/parameter ratios recompute correctly (except the ¥2.58/1,000-step figure, which is off by ~1.5%: 2.543 vs 2.58).

## Could not verify (UNSURE)

- **All checkpoint- and data-dependent claims**: `retnet_500m_v21_export`, `hf_pack_*`, `pretrain_v2`, the v22 mixtures and `val_pack` are not in the snapshot. So 338,120,704/67.06%, the γ medians (0.5656/0.693/0.848), layer-18/19 widths, the 2.2e-4 growth Δ, the 41 zero `gamma_new` tensors, and whether the v22 mixtures contain any eos are all **UNSURE** (the last one is the load-bearing assumption behind P0-3's impact).
- **GPU-only claims**: 1.69 it/s, 57k–61k tok/s, the 6.7% compile-mode gain, the 8 GiB memory peaks, and the 2.6 GB/21 GB figures are arithmetic or cited measurements, not reproduced here (no matching GPU workload run).
- **Doc 01's remaining prose**: all 83 sections were read by a delegated lane and I verified its 12 most consequential quotes directly; the other 70 sections were not independently re-read line-by-line by me.
- **HF internals**: `trainer.py` line numbers quoted in comments (2298, 3523, 2753-2772) were not re-read in the installed transformers source; the two behaviors that matter (checkpointing, step/save ordering) were probed or reasoned about directly.
- **Citations/page numbers**: `Papers/` is not shipped, so the GB/T 7714 page/section claims in docs 03/04 cannot be checked.
- **`v21_export` sharding**: unknown, so P1-7's impact is conditional.
