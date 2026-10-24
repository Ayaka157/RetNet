# CANONICAL — what is authoritative in this workspace

Written 2026-10-24 as part of plan `reorg-20261024` (item T1.4).
This file exists because authority in this repository was previously recoverable only
by cross-reading `README.md`, `RUNBOOK.md` and a gitignored audit under `logs/`.
Where something is still unknown it is marked **UNSURE** rather than guessed.

---

## 0. Development mainline

Designated 2026-10-24. Two directories are the **mainline of record**:

| Directory | Role |
| --- | --- |
| `20261024_Submit_pre-audit/` | The audited source tree: `code/` (retnet_hf, run_task, scripts, tests), `docs/` (six documents, English), `_audit_fixed/AUDIT_REPORT.md`, `FIX_LOG.md`, `verify.py`, `MANIFEST.sha256`. |
| `20261024_verification/` | The independent audit reports that gate it: `VERIFICATION_REPORT.md` (the 26 findings) and `MATH_AND_THEORETICAL_CONSISTENCY_AUDIT.md` (the 04 document against the papers). |

Development continues from `20261024_Submit_pre-audit/code/`, and any change is expected to
keep `python 20261024_Submit_pre-audit/verify.py` green. That gate was last run on
2026-10-24 and passed all nine checks (A–I): 103 documented functions, 27 ids, consistent
status tables in four places, and a complete 69-entry MANIFEST with zero hash mismatches.

**These two directories are treated as immutable snapshots.** Their contents are deliberately
not edited in place — the verification gate hashes all 69 files, so any edit would invalidate
it. Corrections go into a new dated snapshot, not into this one.

The rest of the repository remains the **working tree**: `retnet/`, `retnet_hf/`, `scripts/`,
`tasks/`, `bench/`, `tests/`, `serve/` are what actually run, and they are not byte-identical
to the snapshot. `config.json:auto_map` on Hugging Face still points at the working-tree
module. When the two diverge, the snapshot is the reference for *what was audited*, and the
working tree is the reference for *what executes*.

---

## 1. Current dataset

| | |
| --- | --- |
| Path | `data_cache/chat_pack_v13/` |
| Shape | `(269561, 2049)` int16 + `uint8` wcode |
| Tokens | 552,330,489 |
| Tokenizer | `data_cache/chat_pack_v13/bpe_tokenizer.json`, 20,000 vocab (md5 prefix `afcfce861f`) |
| Built by | `scripts/build_chat_mix_v13.py` from `data_cache/chat_pack_v9` |
| Why it is current | it is the only pack referenced by the live training path and the 0.5B run |

**Do not confuse with** the eleven other `chat_pack_*` directories. They form a
single lineage of re-packings, not eleven corpora. See `docs/21b_数据集盘点_20261017.md`.

Retention classes: `ROOT` (raw corpora, never derivable here), `CURRENT`,
`BUILD INPUT` (ancestor of a kept pack), `DERIVABLE` (archived), `SUPERSEDED`,
`INCOMPATIBLE` (wrong tokenizer), `NO MASK / CORRUPT`.

## 2. Current weights

Seven `model.safetensors` files exist under `checkpoints/`, each 1923.3 MB, and
**all seven have distinct SHA-256 hashes**. There is no duplicate weight file and
no de-duplication opportunity.

| Checkpoint | Role |
| --- | --- |
| `checkpoints/retnet_500m_v15b_ep3/` | **serving default** (README serve command) |
| `checkpoints/retnet_500m_v21_export/` | HF export of the v21 line |
| `checkpoints/retnet_500m_v15b_ep2/` | epoch 2 leg |
| `checkpoints/retnet_500m_v15b/` | epoch 1 leg, plus `checkpoint-67390/` with optimizer parts |
| `checkpoints/retnet_500m/`, `probe_500m/` | earlier 500M runs, distinct |

Published weights live on Hugging Face, not in this repository:

| Repository | Contents | Status |
| --- | --- | --- |
| `Ayaka157/RetNet-0.5B` | epoch 3, `v15b_ep3` | usable |
| `Ayaka157/RetNet-0.5B-ep2` | `v15b_ep2` | usable |
| `Ayaka157/RetNet-0.5B-ep1` | `v15b` | usable |
| `Ayaka157/RetNet-100M` | 401 MB, no local copy | usable |
| `Ayaka157/RetNet-0.5B-ep1-ckpt67390` | `v15b/checkpoint-67390` + `trainer_state.json` | usable; the resume point for epoch 2 |
| `Ayaka157/RetNet-FAILED-checkpoints` | `01_v21_export` only | **failed — do not use** |

**Publication policy (2026-10-24):** only checkpoints produced by training are published.
Two artifacts are deliberately excluded and stay local-only:

- `checkpoints/probe_500m/` — random weights; its own config `_note` says it is for
  VRAM/step-time measurement and "not usable for evaluation".
- `checkpoints/retnet_500m/` — the width-growth *product*, not a training output; it has no
  `trainer_state.json`. This is where the 338,120,704 dead parameters were created.

Failed *trained* models go to `RetNet-FAILED-checkpoints` with an explicit do-not-use banner,
never into a normal model repository.

## 3. Entry points

### Training (authoritative)

```powershell
python run_task.py --config tasks/v22_ctx16k.json     # newest config
python run_task.py --config tasks/<name>.json --resume
python run_task.py --config tasks/<name>.json --status
```

`tasks/` holds 26 JSON configs plus 8 launcher scripts. One JSON per training run.
The model code is split two ways and both halves are live:
`retnet/` is the handwritten core (recurrence, packing, chat encoding);
`retnet_hf/` wraps the same math for the HuggingFace `Trainer`.

### Interactive inference (two scripts, both live, different jobs)

| Script | Reads | Status |
| --- | --- | --- |
| `play.py` | handwritten `model.pt` only (the 100M series) | legacy, still the only reader for `.pt` checkpoints |
| `play_v2.py` | `retnet_hf` safetensors directories | **current** — and a hard dependency of serving |

This resolves the "which one is canonical" question **by evidence, not preference**:
`serve/retnet_play_server.py` line 48 does `from play_v2 import encode_prompt_ids`,
and `tests/test_play_v2_prompt.py` imports the same function. `play.py` cannot load
any current checkpoint because the v10-and-later checkpoints are safetensors.

Neither script is renamed or removed. They serve different checkpoint formats.

### Serving

```powershell
python serve/retnet_play_server.py --model_dir checkpoints/retnet_500m_v15b_ep3 --port 8077
```

## 4. Documented dead routes — do not retry

Recorded in `RUNBOOK.md` §5 and the technical report. Each one cost real time.

| Route | Why it is dead |
| --- | --- |
| **fp16** | two seeds hit `loss 0.0 / grad_norm NaN` at steps 20 and 40; bf16 was fine at the same steps. Code retained under `archive/retnet_hf_fp16/` |
| **v11 8K / 4K re-pack** | `build_chat_mix_v11.py` re-slices rows that were already truncated at 512 in v10; the truncation is baked into the tokens and re-cutting cannot recover it. Both runs were interrupted with zero output |
| **Re-slicing R1 for more CoT** | `r1_110k_sft` is already 100% inside v13 and repeated five times (740M exposures). The reasoning score is limited by token budget, not CoT supply |
| **Chasing general capability at 272M** | v10's 65k steps beat the 143-item baseline by +0.026 at McNemar p=0.084 — not significant — for 10h46m and a 26.5% think-leak |
| **`build_chat_mix_v9.py --seq_len 2048`** | the long-text reader grows to 13 GB RSS and is OOM-killed. Wide windows must go through `build_chat_mix_v13.py`, which only re-cuts |

## 5. Git baseline

`main` at the time of writing carries the reorganization commits. The one
recoverable pre-purge marker is the tag `archive/pre-purge-20261019`.
`archive/MANIFEST.baseline.sha256` is the pre-move hash baseline for every path outside
`.git/`, `data_cache/`, `checkpoints/` and `hf_cache/`.

Three directories under `Kaggle/_audit/checks/` (`tmpa3758029`, `tmpadjufmpf`,
`tmpr957s6fh`) grant only SYSTEM, Administrators and OWNER RIGHTS with inheritance
disabled, so the normal baseline pass could not enumerate them. They were read once
under an elevated (UAC) read-only pass on 2026-10-24: they hold **six leftover test
fixtures** from the checkpoint-resume harness, nothing else. Their hashes are
appended to `archive/MANIFEST.baseline.sha256` under a provenance comment. No `takeown` and
no `icacls` change was made, so the directories remain as they were.

## 6. Still UNSURE

| Question | What would settle it |
| --- | --- |
| Are the 16 `v21_ddp_1019_*.out` files true byte-prefixes of their successor? | monotonic sizes 6.6→10.2 MB make it near-certain, but a two-file `fc /b` diff would prove it |
| Does anything *outside* this workspace read `logs/_3090_recover/`? | search your external scripts and notebooks for the literal path |
| Why are `_archive_from_instance/` filenames mojibake? | content is hash-identical to live docs, so only the extraction locale is unexplained |
| Are `scripts/fp16_*.py` and `scripts/audit_mem.py` still run? | one execution; they import `retnet_hf_fp16`, which now lives under `archive/` |

## 7. Where the record of each relocation lives

`archive/INDEX.md` is the ledger: for every relocated tree it records the origin,
destination, date, reason, how many of its files were byte-identical to live, how
many were unique, and the name of the undo script under `archive/_undo/`.
