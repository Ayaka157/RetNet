# 02 Resume Plan (¥220 hard cap / ¥196 actual-spend ceiling)

> First draft: 2026-10-20. **This version: 2026-10-24, rewritten against the fixed code and the audit report.**
> The budget, step counts and thresholds are still the historical estimates counted in the first draft (§8 marks their sources); every passage that involves
> **code behavior** has been rewritten against the current implementation, so the commands can be copied and run directly.
>
> What changed in this version (differences from the first draft):
> 1. **E15 became "selection off during training + offline selection"**. The first draft said "keep `load_best_model_at_end`,
>    it is a hard prerequisite", which is the opposite of the code; the code now has it off, and selection is done offline with `eval_holdout.py`.
> 2. **E08 gained `--allow_dead_params`**: this plan starts from a pruned checkpoint with 33% live parameters, and
>    without this switch `_preflight` refuses to start training (this is exactly audit item P0-2).
> 3. **E12's `require_eos: false` now really takes effect** (throughout the `Stage`/`parse_plan`/stage-dict chain).
> 4. **Both context probes are implemented**; see §5.3 for the commands and output fields. No more "about 40 lines still to be written".
> 5. **Test counts follow what this snapshot measured**: `cd code && python -m pytest tests -q` → 82 passed / 4 skipped.
> 6. Removed the quotation markers inside the §6 code block from the first draft (they used to slice the pseudo-code into prose).

---

## 0. Starting point and verified/unverified facts

| Fact | Status | Basis |
|---|---|---|
| Starting point is `checkpoints/retnet_500m_v21_export`: continue training, no shell shrink, no retraining, no growth | Plan premise | Local checkpoint, not included in the snapshot |
| About 67% of the nominal 504,179,896 parameters are zero | **Not re-verified in this snapshot** | First-draft spot check; re-running the current `_preflight` once on the data machine settles it |
| Layers 18/19 are d1024, not d1408 (mixed width 18×d768 + 2×d1024) | **Verified (first-draft audit note)** | Per-tensor live rows/live columns; this snapshot contains no weights, so the conclusion carries over |
| Median live-head gamma: ep3 0.848 / v21 0.693 (half-life 4.21 / 1.89 tokens) | **Not re-verified in this snapshot** | First-draft safetensors scan; print the gamma distribution from `_preflight` before using `check_holdout` |
| The previous round's held-out numbers were measured on `mode=parallel`, which is not the same function as the chunkwise training path | Fixed (E16/N04) | `eval_holdout.py` is now pinned to chunkwise; the old numbers can only serve as a delta baseline |
| `val_dir` used to overlap the training data (chat_pack_v13 was 40% of sft_mix) | Fixed (E14) | Three-layer same-source check; this plan switches to a clean `data_cache/val_pack` |
| Local unique Chinese tokens 1,247,202,821; total excluding skypile 2,307,246,540 | First-draft measurement, not re-verified in this snapshot | Sum of 62 npy headers |

---

## 1. Conclusion

**Continue training on `checkpoints/retnet_500m_v21_export`; no shell shrink, no retraining, no growth.**
**A budget of ¥187 (13.4 GPU hours, about 4.65B tokens, including 0.1B of chat rehabilitation) pushes the in-domain held-out
(5-file weighted) from 3.5023 down to 3.15–3.30; I estimate a 60% chance of clearing the 3.2659 criterion.**
**Training a 166M model from scratch does not fit inside ¥200**: local Chinese is only 1.247B, a 4.65B budget requires more than 2.7 passes over Chinese,
and no veto test can stop the loss early within the first 0.25B.

---

## 2. Code prerequisites before training starts (all in place; commands match the current implementation)

| ID | Current behavior | How this plan uses it | What happens if unmet |
|---|---|---|---|
| E06 | `--no_decay_1d` is **on by default**; `--decay_1d` is the inverse switch; `DecayGuardCallback` asserts at training start that 1-D parameters are not in a `weight_decay>0` group | Pass no switch at all (default on) | Within 500 steps the live-head gamma keeps being pushed down by wd, and ¥150 buys back v21's end point |
| E07 | `GradCoverageCallback` counts zero-gradient parameters after step 1 and stops when the element share exceeds `--dead_grad_tol` (default 2%) | Default threshold; check the dead-parameter list in the log first | Dead parameters run silently through the whole run |
| E08 | `_preflight` refuses when the live-parameter ratio is <95%, unless `--allow_dead_params` is given or it matches the `live_params.json` baseline | **Must pass `--allow_dead_params`** (the starting point has 33% live parameters) | The plan's very first command exits with SystemExit |
| E12 | A plan stage only lets a boundary-free flat stream through when it writes `require_eos: false`; the default is True | This plan's `pretrain_v2` has eos, so **do not write false**; any future boundary-free pack must write it explicitly | A boundary-free stream is trained silently (or a misconfigured false lets it through) |
| E13 | `gc_stage` requires that the meta can prove rebuildability (`rebuild_cmd`/`source_ids`/`sources+text_fields`), and refuses to delete the validation-set directory | **`gc:false`** on every stage of this plan (there is only one copy of the data) | The only copy is deleted after the first stage finishes and the later stages wait forever for data |
| E14 | Three-layer same-source check for the validation set (path / declared source / row hashes) | Use `data_cache/val_pack` for `val_dir`; startup prints the number of scanned rows | eval_loss is contaminated, so both selection and the criterion are invalid |
| E15 | **No selection** during training (`load_best_model_at_end=False`); checkpoints are saved purely by step count | Compare candidate checkpoints offline with the §5.4 commands after training | The contaminated eval_loss picks the wrong checkpoint |
| E16/N04 | Evaluation is pinned to `mode=chunkwise` with the same `chunk_size` as training | Re-measure the P1 baseline with the current code; old numbers are only a delta reference | Training and evaluation are not the same function |
| E17 | `save_total_limit` comes from the single constant 4 in `retnet_hf/defaults.py` | Write 4 explicitly in the task JSON; the plan branch also passes it | The best checkpoint is deleted by mtime rotation |
| E18 | Hot-changing the total step count is refused by default (when it would make the current LR jump up >2%) | To extend the budget, write `"allow_reschedule": true` in the JSON and accept one controlled hot restart | An unapproved LR jump |
| E20 | The plan branch also passes `eval_batch_mult` | Write `eval_batch_mult: 2` in the task JSON | Evaluation overhead is 6.6% instead of 2.9% |
| E22 | Cross-entropy is chunked by a byte budget (64 MB, 838 tokens/chunk at V=20000) | Defaults are fine; if GPU memory is still tight, use `RETNET_FUSED_CE=1` for the fused path | About 2.6 GB extra materialized per step |
| P2-1 | `--eval_batches N` is a real cap on evaluation rows; default 0 = no limit | Unlimited by default; write N in the task JSON when evaluation cost must be controlled | Evaluation runs over the whole val set and wastes money |

`run_task.py` translates the switches above into argv; `--dry` lets you check this directly:

```powershell
cd code
python run_task.py --config ..\configs\tasks\v22_a1.json --dry
```

> ⚠️ `--dry` prints the full command line. Confirm that it contains `--allow_dead_params`, `--save_total_limit 4`,
> `--eval_batch_mult 2`, and that it does **not** contain `--decay_1d`.

---

## 3. Data plan (reusing the first draft's measured numbers)

### 3.1 How many tokens per pack

| Pack | train tokens | Shards | Chinese | Use in this plan |
|---|---:|---:|---|---|
| `chinese_fineweb_edu_v2` | 1,013,826,215 | 20 | ✅ main domain | All of M1/M2/M3 |
| `wikipedia_zh` | 202,958,320 | 4 | ✅ | All |
| `poetry_zh` | 30,418,286 | 1 | ✅ | All |
| `fineweb_edu` | 504,765,133 | 10 | ❌ English | M1 uses 5 shards, M2 uses the other 5 (no repeats) |
| `cosmopedia` | 403,355,751 | 8 | ❌ English (synthetic textbooks) | M1 uses 3 shards, M2 uses the other 5 |
| `open_web_math` | 151,922,835 | 3 | Mixed | M1 uses 1 shard, M2 uses 2 shards |
| `skypile` | 504,097,083 | 10 | ✅ but **excluded** | License + it is one of the held-out sets |
| **Total (excluding skypile)** | **2,307,246,540** | 46 | 1,247,202,821 Chinese (54%) | |
| val (6 shards, not used in training) | 5,470,698 | 6 | | 5 of them form the criterion `val_pack` (4,463,629) |

### 3.2 The three mixture directories

Build them with NTFS hard links (no extra space on the same volume); `_FlatStage` only globs top-level `*.npy`.

| Directory | File composition | Unique tokens | Chinese share |
|---|---|---:|---:|
| `data_cache/rescue_m1` | CFE×20 + wiki×4 + poetry×1 + fineweb×5 (0–4) + cosmo×3 (0–2) + math×1 (0) | 1,700,932,530 | 73.32% |
| `data_cache/rescue_m2` | CFE×20 + wiki×4 + poetry×1 + fineweb×5 (5–9) + cosmo×5 (3–7) + math×2 (1–2) | 1,853,516,831 | 67.29% |
| `data_cache/rescue_m3` | CFE×20 + wiki×4 + poetry×1 (pure Chinese anneal) | 1,247,202,821 | 100% |

Usage: `A1+A2+B1 = 2.75B` goes through M1 (1.62 passes); `B2 = 1.3B` goes through M2 (0.70 passes); `B3 = 0.5B` goes through M3 (0.40 passes).

### 3.3 Repetition and eos

- Of the 4.55B pretraining part of this budget, Chinese is about 3.391B, and local unique Chinese is 1.247B → Chinese goes **2.72 passes**.
- `pretrain_v2` **has eos** (measured 0.071% in the first draft, one boundary every 1,409 tokens). This plan therefore
  does not need `require_eos: false`; if a boundary-free pack is used in the future, false must be written explicitly and the reason noted in the experiment log.
- For the train/validation gap stop rules see §6 S3/S4; `val_pack` is assembled in §3.4 and never enters training.

### 3.4 How `val_pack` is assembled (the current scripts have no generator; done manually once)

1. Concatenate the 5 `*val*.npy` files (excluding `skypile_val`) in order into a 1-D int16 array;
2. Cut it into 2,178 rows × 2049 (`seq_len=2048`);
3. Write `val.ids.npy` as 2-D `(2178, 2049)`; `val.wcode.npy` is all 1s (uint8);
4. Put a `meta.json` in the directory containing `source_ids: ["pretrain_v2/*val*"]` —
   E14's declared-source layer uses it; if it is missing, that layer only prints "declared layer skipped".

> This step has no script and is a **known gap** (see §9). After assembly, use
> `python -c "import numpy as np; print(np.load('data_cache/val_pack/val.ids.npy', mmap_mode='r').shape)"`
> to check the shape.

### 3.5 The hard gap in code corpora

There is no code pack left locally (`hf_pack_code*` was deleted by the previous round's GC, E13). This round feeds no code,
so it can only hedge with weight inheritance and the 26.7% (M1) / 32.7% (M2) of English textbooks/math in the mixture.
**The release notes must not claim "code ability preserved"**; if that claim is needed, spend ¥3 to run the code-subset evaluation.

---

## 4. Staged budget table

Unit price (first-draft derivation): `¥13.97/h ÷ 3600 ÷ 110,000 tok/s = ¥35.28 / 1B token`;
including 10% for evaluation/checkpointing/restarts → **¥38.81 / 1B token**; 1 step = 2 GPUs × 16 × 2048 = **65,536 tokens**;
1,000 steps = **¥2.58** (overhead included).

| Stage | Purpose | steps | tokens | LR | Data | GPU h | ¥ | Acceptance threshold |
|---|---|---:|---:|---|---:|---:|---:|---|
| **P1** baseline re-measurement | Re-measure with the current code + both probes | — | 4.46M(eval) | — | `val_pack` | 0.10 | 1.4 | Baseline saved to disk; difference from 3.5023 ≤0.15; `Δ_ctx` on record |
| **A1** stop the bleeding + re-feed Chinese | no_decay_1d + 73% Chinese | 7,629 | 0.50B | 4e-5→1.2e-5, warm 100 | M1 | 1.39 | 19.4 | G1a/G1b/G1c |
| **A2** gamma restoration | Live-head gamma ← ep3 values | 3,815 | 0.25B | 3e-5→1e-5, warm 50 | M1 | 0.69 | 9.7 | G2 |
| **B1** main repair | Pull Chinese back | 30,518 | 2.00B | 3e-5→cosine | M1 | 5.56 | 77.6 | Global step 41,962: CFE ≤3.55 and avg ≤3.32 |
| **B2** stabilize domains + cool down | English/math do not regress | 19,836 | 1.30B | →1.2e-5 | M2 | 3.61 | 50.4 | avg ≤3.30; fineweb/cosmo degradation vs P1 ≤0.10 |
| **B3** in-domain anneal | Pure Chinese, LR down to the floor | 7,629 | 0.50B | 1.2e-5→2e-6 | M3 | 1.39 | 19.4 | avg ≤3.27 |
| **P5** wrap-up | Full-domain evaluation + offline selection + export verification | — | eval | — | — | 0.20 | 2.8 | `verify_export` key names clean; loss not NaN; parameter count and live parameters printed together |
| **P6** chat rehabilitation | Preserve the output format | 1,526 | 0.10B | 5e-6 constant | `chat_pack_v13` (packed, val excluded) | 0.28 | 4.0 | Two exports: `heldout/` and `chat/`; re-measure held-out after P6 and report it separately |
| **Restarts/compile** | One warmup per each of 4 runs | — | — | — | — | 0.15 | 2.1 | — |
| **Total** | | **70,953** | **4.65B** | | | **13.4** | **¥187** | Hard cap ¥196 |

**Balance**: ¥220 − ¥187 = ¥33 (of which ¥20 is buffer, so movable slack = ¥13).

**Plan file (`tasks/plan_v22_rescue.json`)** — fields aligned with the current schema in `retnet_hf/plan.py`:

```json
{
  "plan_version": 1,
  "seq_len": 2048,
  "stages": [
    {"name": "rescue_m1", "data_dir": "data_cache/rescue_m1", "mode": "flat",
     "steps": 30518, "gc": false, "min_shards": 34, "require_eos": true},
    {"name": "rescue_m2", "data_dir": "data_cache/rescue_m2", "mode": "flat",
     "steps": 19836, "gc": false, "min_shards": 37},
    {"name": "rescue_m3", "data_dir": "data_cache/rescue_m3", "mode": "flat",
     "steps": 7629,  "gc": false, "min_shards": 25}
  ],
  "lr": {"peak": 3e-5, "min": 2e-6, "warmup": 150},
  "save_steps": 2000,
  "eval_steps": 2000,
  "val_dir": "data_cache/val_pack",
  "min_free_gb": 20,
  "notes": "v22 rescue: 4.65B/13.4h/¥187 (including the 0.1B SFT of P6)."
}
```

Key fields of the task JSON (`tasks/v22_a1.json`):

```json
{
  "name": "v22_a1",
  "model_dir": "checkpoints/retnet_500m_v21_export",
  "output_dir": "checkpoints/retnet_500m_v22_a1",
  "plan": "tasks/plan_v22_rescue.json",
  "batch_size": 16, "mode": "chunkwise", "chunk_size": 64,
  "precision": "bf16", "grad_checkpoint": true, "compile": false,
  "save_total_limit": 4, "eval_batch_mult": 2,
  "no_decay_1d": true, "allow_dead_params": true,
  "resume": false, "auto_restart_max": 3, "restart_wait_min": 10,
  "seed": 1234
}
```

---

## 5. The cheap veto test (¥30.7, ≤¥50)

**The question to answer**: can Chinese held-out still be saved at all? If the slope over the first 0.25B tokens is not enough,
the remaining 4B will not reach the criterion either, and we must stop on the spot.

### 5.1 Where the thresholds come from

- The criterion requires a 5-file weighted 3.2659; the starting point is 3.5023 → a gap of 0.2364.
- CFE carries 42.66% of the weight; assuming the other four domains each improve by 0.08–0.18, CFE must be ≤3.43.
  From a starting point of 3.8605 → CFE has to drop 0.43.
- 0.43 spread over 4.55B = 0.095 nat/B on average; diminishing returns mean the slope over the first 0.5B must be at least 3× the average
  (0.28/B). G1's threshold comes from this arithmetic, not from guesswork.

### 5.2 The four gates (stop if any one fails)

| Gate | When measured | What is measured | Threshold | If it fails |
|---|---|---|---|---|
| **G1a** | step 3,815 (0.25B, ≈¥11 cumulative) | CFE val | ≤3.79 (slope ≥0.28/B) | Stop. Take §7-A |
| **G1b** | step 7,629 (0.5B, ≈¥21 cumulative) | CFE val + 5-file avg | CFE ≤3.72 and avg ≤3.42 | Stop. Take §7-A |
| **G1c** | Same as G1b | (1) median live-head gamma (2) long-range probe `Δ_ctx` | (1) ≥0.72 (starting point 0.693) (2) `Δ_ctx ≥1.15×baseline` or ≥0.10 absolute | (1) fails → E06 did not take effect, stop immediately (a code fault, not tuning) (2) fails → §5.5 |
| **G2** | End of A2 (0.75B, ≈¥31 cumulative) | 5-file avg | ≤3.39 goes straight into B; 3.39–3.46 goes into B only if A2 improves ≥0.03 over the end of A1; >3.46 stop | Stop. Take §7-B |

### 5.3 The two probes (implemented; the commands run as-is)

```powershell
cd code
python scripts/eval_holdout.py --model checkpoints/retnet_500m_v22_a1 --tag a1_end `
  --val-glob "data_cache/val_pack/*.npy" --seq-len 2048 --chunk-size 64 `
  --by-position --ctx-ablation --ctx 256
```

- `--by-position`: splits the sequence into 8 buckets by position (0–255 … 1792–2047) and reports the loss per bucket.
  Use it to see whether the model gets cheaper further along, and whether it flattens out after 256 (flat = receptive field <256).
- `--ctx-ablation`: for each window, computes `L(256)` once more (feeding only the last 256 tokens) and subtracts it from the
  `L(2048)` loss at the **last 256 positions**, `Δ_ctx = mean(L(256) − L(2048))`.
  Positive = long-range context helps; 0 = unused.
- Both switches require `ctx` and `seq_len` to be integer multiples of `chunk_size` (a chunkwise kernel constraint); otherwise
  the script errors out before it starts.
- Fields in the output JSONL: `by_position: [{range, tokens, loss}]`,
  `ctx_ablation: {ctx, seq_len, loss_short, loss_full_tail, delta_ctx, tokens}`.

### 5.4 Offline selection (replaces the first draft's keep-best)

After training ends, run one identical-protocol evaluation per candidate checkpoint, then compare:

```powershell
cd code
foreach ($ck in Get-ChildItem checkpoints\retnet_500m_v22_b1\checkpoint-*) {
  python scripts/eval_holdout.py --model $ck.FullName --tag $ck.Name `
    --val-glob "data_cache/val_pack/*.npy" --seq-len 2048 --chunk-size 64 `
    --out logs/holdout_eval.jsonl
}
```

`logs/holdout_eval.jsonl` holds one candidate per line; pick the smallest, weighted by `loss` and `tokens`.
**There is no one-shot script**, which is a known gap (§9); this is more honest than letting the training loop auto-select on a contaminated eval_loss.

### 5.5 When G1c's item (2) fails

Do not overturn the conclusion; change what A2 does: skip "restore ep3 gamma" and instead do only 0.25B of same-distribution continued training,
and add the ¥9.7 saved to B1. Rationale: CFE is dropping, which shows "data + LR" is working; touching gamma at that point only introduces a distribution jump.
**A2's budget is not cancelled, only repurposed.**

---

## 6. Stop rules

Run `check()` every 200 steps (≈2 min); `spend` follows cloud billing, not estimates.

```python
# S1 gamma is being decayed (E06 failed) -- top priority, because it would make the whole remaining 4B wasted
if step >= 500 and median(gamma_live) < 0.95 * gamma_live_at_step0:
    abort("E06 not in effect: gamma is still being pulled down by weight decay")
    # action: do not submit a new task; fix the code; rollback target = none (weights are intact)

# S2 held-out regressing (two consecutive evals)
if n_evals >= 2 and avg_now > best_avg + 0.05 and avg_prev > best_avg + 0.05:
    stop_graceful(); restore(best_ckpt)          # best_ckpt is chosen offline per §5.4

# S3 loss abnormally low (a sign of duplicated data)
if cfe_val < 3.00 and step < 30000:
    stop_graceful(); restore(best_ckpt); audit("suspected duplication/leak: check whether M1 has duplicate shards and whether val entered the training set")
if train_loss_200 < cfe_val - 0.60:
    stop_graceful(); restore(best_ckpt)

# S4 training below validation (leak)
if train_loss_200 < val_avg - 0.25:
    stop_graceful(); audit("val entered the training set: run E14's three-layer assertions")

# S5 out of GPU memory: money is decided by tokens, not by steps
if OOM and batch == 16:  batch = 12; steps *= 16/12; restart_from(last_ckpt)
elif OOM and batch == 12: batch = 8;  steps *= 12/8;  restart_from(last_ckpt)
elif OOM and batch == 8:  stop_graceful(); restore(best_ckpt)   # do not lower seq_len

# S6 bad data (once at stage start)
if eos_rate(sample_1000_rows) == 0:  abort("the whole pack has no boundaries (E09/E10 recurrence)")
if shard_count(stage) < min_shards and waited_steps > 200:  stop_graceful()

# S7 timeout/over budget
if spend >= 196.0:  ctrl("next-checkpoint")      # exit gracefully after the save, no taskkill
if wall_clock_h >= 15.5:  ctrl("next-checkpoint")

# S8 plan corrupted
if plan_file_parse_error or total_steps decreased mid-run:  stop_graceful()
```

| Rule | Trigger | Action | Roll back to |
|---|---|---|---|
| S1 gamma | At step 500 the live-head median is <0.95× the starting point | Stop immediately, fix the code | No rollback (the weights are intact) |
| S2 regression | Two consecutive evals more than 0.05 worse than best | Stop gracefully | `best_ckpt` (chosen offline per §5.4) |
| S3 abnormally low | CFE<3.00 and step<30k; or train 0.60 below val | Stop gracefully + audit the data | `best_ckpt` |
| S4 leak | train 0.25 below val_avg | Stop gracefully + run E14 assertions | `best_ckpt` |
| S5 OOM | Step the batch down 16→12→8 | Change batch and recompute steps by tokens | Most recent checkpoint |
| S6 data | eos rate 0 / shards short after 200 steps | Stop or wait for data | Not applicable |
| S7 budget | spend ≥¥196 or ≥15.5h | `ctrl next-checkpoint` | Not applicable |
| S8 plan | Parse failure / total steps lowered | Stop gracefully | Not applicable |

> S1 now also has an automatic guard: `DecayGuardCallback` inspects the optimizer parameter groups directly at training start,
> and exits with SystemExit as soon as a 1-D parameter appears in a `weight_decay>0` group (no need to wait for step 500).
> So S1 is the second line of defense, not the only one.

---

## 7. Failure budget (three branches; there is no "add more money")

| Where we stop | Spent | Left | What to do next |
|---|---:|---:|---|
| **G1a** (0.25B) | ¥12.8 | ¥207 | **A. Stop training, shut the machine down.** Do not start a second route: training 166M from scratch to 3.3 needs ≥6B tokens, local Chinese is only 1.247B, and no veto test can stop the loss early. |
| **G1b** (0.5B) | ¥20.8 | ¥199 | Same as A. One extra step: archive this round's fix tests and documentation (this snapshot is it) and start the next round from it. |
| **G2** (0.75B) | ¥30.5 | ¥189 | **B. Skip gamma restoration, cut the B segment from 3.8B to 3.2B** (M1 2.0B + M2 1.2B, no B3): 3.2B × ¥38.81 = ¥124.2, plus A's ¥30.5, P1/P5/restarts ¥5.5, P6 ¥4.0 → total spend ≈¥164, leaving ¥56 untouched. The criterion becomes "release as long as it is below 3.5023 and not below 3.35". |

**Minimum deliverable** (holds no matter which branch we stop on):
1. `checkpoints/retnet_500m_v21_export` (3.5023) — the best exportable weights that exist;
2. this snapshot: code and documentation that fix E01–E23/N01–N04, **each with a test that can fail**;
3. a measured slope table: ¥/1B, nat/0.5B, the `Δ_ctx` baseline. The next round continues the arithmetic from this table.

---

## 8. Source of every number

| Number | Value | Source | Formula/basis |
|---|---:|---|---|
| Unit price | ¥13.97/h | Given in the first draft | — |
| Steady-state throughput | 110,000 tok/s | First-draft measurement | 1.69 it/s × 65,536 |
| Tokens per step | 65,536 | `tasks/v21_resume.json` | 2 × 16 × 2048 |
| Cost per 1B tokens | ¥35.28 | Derived | 13.97 ÷ 3600 ÷ 110,000 × 1e9 |
| Including 10% overhead | ¥38.81/1B | Derived | 35.28 × 1.10 |
| Per 1,000 steps | ¥2.58 | Derived | 1000 × 65,536 ÷ 110,000 ÷ 3600 × 13.97 × 1.10 |
| Local corpus | 2,811,343,623 tokens | First-draft measurement | Sum of 62 npy headers |
| Unique Chinese tokens | 1,247,202,821 | First-draft measurement | CFE + wiki + poetry |
| M1 / M2 unique tokens | 1,700,932,530 / 1,853,516,831 | First-draft measurement | Summed shard header by shard header |
| val (5 shards for the criterion) | 4,463,629 | First-draft measurement | Sum of 5 shards |
| eos density | 0.071% | First-draft measurement | 35,480 / 50,000,000 |
| Live-head gamma (ep3 / v21) | 0.848 / 0.693 | First-draft measurement | First 12 heads per layer, median of 240 values over 20 layers |
| Dead parameters | 338,120,704 (67.06%) | First-draft spot check | Re-running the current `_preflight` once settles it |
| held-out baseline | 3.5023 | `logs/holdout_eval.jsonl` `v21_final` | Weighted over 4,454,400 tokens |
| Criterion | 3.2659 | Given in the first draft | — |
| Required delta | −0.2364 | Derived | 3.2659 − 3.5023 |
| Chinese repetition passes | 2.72 | Derived | 3.391 / 1.2472 |
| G1a threshold 3.79 | Slope 0.28/B | Derived | (3.8605−3.79)/0.25B |
| Total plan price | ¥187 | Derived | Sum of the 9 rows in the staged table |
| Cap | ¥196 | Derived | ¥200 − 2% error margin; hitting the line triggers the S7 stop |

---

## 9. Known gaps (do not pretend they do not exist)

1. **No `val_pack` generator script**: §3.4 is a manual procedure and row widths are easy to get wrong. The next round should add a
   `scripts/build_val_pack.py` and self-check it with E14's three-layer assertions.
2. **No one-shot offline-selection script**: §5.4 is a `foreach` loop plus manual comparison of JSONL.
3. **No code corpus locally**: §3.5; code ability can only come from weight inheritance.
4. **Data-related numbers were not re-verified in this snapshot**: the live-parameter ratio, gamma distribution and eos density all need one
   re-run on the data machine with the current code; this snapshot only guarantees code behavior.

---

## Appendix A: Startup SOP

0. **First check whether this machine has torch**: if not, only build links / assemble val / change code, and leave torch-dependent verification to the cloud machine.
1. **Build the mixtures (local, ¥0)**: hard-link to `data_cache/rescue_m1|m2|m3/` per §3.2 and
   count the shards: 34 / 37 / 25 `.npy` files.
2. **Assemble `val_pack`**: §3.4, 5 shards → 2,178 rows × 2049; `val.wcode.npy` all 1s; write `meta.json`.
3. **Run the tests (local CPU is enough)**: `cd code && python -m pytest tests -q` → 82 passed / 4 skipped;
   then run `python tests/test_plan_gc.py` and `test_plan_pipeline.py` manually.
4. **First thing on the machine (5 minutes)**: `nvidia-smi`; confirm the unit price is still ≤¥14.5/h;
   load from `v21_export`, run 3 steps, and print `params` and `live=(p!=0).sum()`.
   **Add `--allow_dead_params`**, otherwise `_preflight` refuses as expected.
5. **P1 baseline**: `python scripts/eval_holdout.py --model checkpoints/retnet_500m_v21_export --tag v22_baseline --by-position --ctx-ablation`
   (the current code is pinned to chunkwise).
6. **A1 → A2 → B → P6**: four runs of `run_task.py --config tasks/v22_*.json`.
   To stop, **always** use `python run_task.py --ctrl next-checkpoint`, never `taskkill`.
7. **Optional (balance ≥¥25)**: `--tag code` runs the code-subset evaluation (¥3).

## Appendix B: Three questions answerable within 1 hour of starting training

1. **Will A2's gamma restoration cause a 0.3 nat spike**: if within 0.05B tokens after the restoration `train_loss`
   is >0.5 above the pre-restoration value and has not come back down after 200 steps → roll back to the checkpoint at the end of A1 and skip A2.
2. **Is the `Δ_ctx` baseline 0**: if it is 0, the 2048 window is really only using a few dozen tokens,
   the B-segment ceiling is 3.3–3.4, and the criterion very likely will not be met (follow §7-B and do not add money on the fly).
3. **Can 1.69 it/s be reproduced on the new data directories**: after the first 1,000 steps, recompute the remaining budget with the measured rate;
   if it is <1.55 it/s, cut B2 from 1.3B to 1.0B.

---

## References

> The numbers correspond one-to-one with the file prefixes in the `Papers/` directory.

| ID | Original title | Local file | Link |
|---|---|---|---|
| [01] | Retentive Network: A Successor to Transformer for Large Language Models | `Papers/01_Retentive-Network_2307.08621.pdf` | <https://arxiv.org/abs/2307.08621> |
| [02] | A Length-Extrapolatable Transformer | `Papers/02_Length-Extrapolatable-Transformer_2212.10554.pdf` | <https://arxiv.org/abs/2212.10554> |
| [03] | Gated Linear Attention Transformers with Hardware-Efficient Training | `Papers/03_Gated-Linear-Attention_2312.06635.pdf` | <https://arxiv.org/abs/2312.06635> |
| [04] | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention | `Papers/04_Transformers-are-RNNs_2006.16236.pdf` | <https://arxiv.org/abs/2006.16236> |
| [05] | Group Normalization | `Papers/05_Group-Normalization_1803.08494.pdf` | <https://arxiv.org/abs/1803.08494> |
| [06] | GLU Variants Improve Transformer | `Papers/06_GLU-Variants_2002.05202.pdf` | <https://arxiv.org/abs/2002.05202> |
| [07] | Net2Net: Accelerating Learning via Knowledge Transfer | `Papers/07_Net2Net_1511.05641.pdf` | <https://arxiv.org/abs/1511.05641> |
| [08] | Network Morphism | `Papers/08_Network-Morphism_1603.01670.pdf` | <https://arxiv.org/abs/1603.01670> |
| [09] | bert2BERT: Towards Reusable Pretrained Language Models | `Papers/09_bert2BERT_2110.07143.pdf` | <https://arxiv.org/abs/2110.07143> |
| [10] | Efficient Training of BERT by Progressively Stacking | `Papers/10_StackBERT_PMLR-v97-gong19a.pdf` | <https://proceedings.mlr.press/v97/gong19a.html> |
| [11] | Progressively Stacking 2.0 | `Papers/11_Progressively-Stacking-2.0_2011.13635.pdf` | <https://arxiv.org/abs/2011.13635> |
| [12] | Learning to Grow Pretrained Models for Efficient Transformer Training | `Papers/12_LiGO_2303.00980.pdf` | <https://arxiv.org/abs/2303.00980> |
| [13] | ReZero is All You Need: Fast Convergence at Large Depth | `Papers/13_ReZero_2003.04887.pdf` | <https://arxiv.org/abs/2003.04887> |
| [14] | Fixup Initialization: Residual Learning Without Normalization | `Papers/14_Fixup_1901.09321.pdf` | <https://arxiv.org/abs/1901.09321> |
| [15] | Training Compute-Optimal Large Language Models | `Papers/15_Chinchilla_2203.15556.pdf` | <https://arxiv.org/abs/2203.15556> |
| [16] | The FineWeb Datasets | `Papers/16_FineWeb_2406.17557.pdf` | <https://arxiv.org/abs/2406.17557> |
| [17] | Skywork: A More Open Bilingual Foundation Model | `Papers/17_Skywork_2310.19341.pdf` | <https://arxiv.org/abs/2310.19341> |
