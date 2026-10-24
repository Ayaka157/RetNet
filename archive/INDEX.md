# archive/ — relocation ledger

Every path below was **moved, never deleted**. Each row names the origin, the
destination, the date, why it moved, how much of it was already duplicated
elsewhere, and the script that puts it back.

Plan: `PLAN.reorg-20261024.md`. Execution log: `PLAN.reorg-20261024.execution-log.md`.
Baseline hash manifest: `MANIFEST.baseline.sha256` (taken before any move).

**Undo convention.** Every script lives in `archive/_undo/` and takes no arguments.
Each refuses to run if its destination already exists, and verifies SHA-256 after
restoring. `Move-Item` scripts are for untracked payload; `git mv` scripts are for
tracked trees, so the reversal is recorded as a rename rather than a delete.

---

## Relocations

### T1.3 — dataset audit promoted out of `logs/`

| | |
| --- | --- |
| From | `logs/_agent_dataset_audit.md` |
| To | `docs/21b_数据集盘点_20261017.md` |
| Date | 2026-10-24 |
| Reason | The only document explaining which chat pack is current lived in a **gitignored** directory — one `git clean` from death. |
| Duplicated elsewhere | no (gitignored, so it had no copy in git) |
| Verified | SHA-256 `91A24E11…` identical before and after |
| Undo | `Move-Item docs\21b_数据集盘点_20261017.md logs\_agent_dataset_audit.md` |

### T2.1 — source code evicted from the run-products directory

| | |
| --- | --- |
| From | `logs/_3090_recover/` |
| To | `archive/20261019_3090_recover/` |
| Date | 2026-10-24 |
| Files | 366 (33.4 MB) |
| Reason | `.gitignore` declares `logs/` to mean run products, but this tree held `retnet/`, `retnet_hf/`, `scripts/`, `tests/`, `bench/`, `play_v2.py`, `run_task.py`. |
| Duplicated elsewhere | **311 of 366** byte-identical to live source (measured pre-move) |
| Unique payload | **55 files, 5.9 MB** — these exist nowhere else |
| Verified | 366/366 hashes matched at destination |
| Undo | `archive/_undo/T2_1.undo.ps1` |

### T2.2 — instance pull consolidated

| | |
| --- | --- |
| From | `_archive_from_instance/` |
| To | `archive/20261019_instance_pull/` |
| Date | 2026-10-24 |
| Files | 437 (165.5 MB) |
| Reason | Archive pulled back from a rented instance; its filenames are GBK-mojibake while its content is hash-identical to live docs. |
| Duplicated elsewhere | **273 of 437** byte-identical to live (measured pre-move) |
| Unique payload | **164 files, 157.9 MB** — includes the only non-git copy of `docs/INDEX.md` |
| Precondition | `docs/archive/INDEX_deleted_20261018.md` was copied out **before** the move |
| Verified | 437/437 hashes matched at destination |
| Undo | `archive/_undo/T2_2.undo.ps1` (does not reverse the precondition copy) |

### T2.3 — prefix dumps grouped inside the archive

| | |
| --- | --- |
| From | `archive/20261019_instance_pull/logs/archive/` |
| To | `archive/20261019_instance_pull/prefixes/` |
| Date | 2026-10-24 |
| Files | 16 (133.5 MB); `v21_ddp_1019_1712.out` deliberately stayed put |
| Reason | Progressive dumps of one training log. Every file is a byte-prefix of its successor — **16/16 adjacent pairs verified by hashing the shorter file and comparing it to the prefix of the longer**, not inferred from file sizes. |
| Duplicated elsewhere | none as complete files; by construction each is contained in the next |
| Note | `v21_ddp_1019_1712.out` is byte-identical to `archive/20261019_instance_pull/logs/linux/v21_ddp.out`. There is **no** live `logs/linux/v21_ddp.out`. |
| Verified | 16/16 hashes matched at destination |
| Undo | `archive/_undo/T2_3.undo.ps1` |

### T2.4 — superseded submission snapshot

| | |
| --- | --- |
| From | `20261020_Submit/` |
| To | `archive/20261020_Submit/` |
| Date | 2026-10-24 |
| Files | 56 (tracked in git) |
| Reason | Superseded by `20261024_Submit_pre-audit/`, which is now the deliverable of record. |
| Duplicated elsewhere | 5 documents byte-identical to live `docs/`; the other 51 not measured against live |
| Verified | `git mv` recorded as renames (history preserved) |
| Undo | `archive/_undo/T2_4.undo.ps1` |

### T2.7 — abandoned fp16 branch

| | |
| --- | --- |
| From | `retnet_hf_fp16/` |
| To | `archive/retnet_hf_fp16/` |
| Date | 2026-10-24 |
| Files | 5 (tracked) |
| Reason | Numeric-hardening fork written to chase the fp16 NaN; `RUNBOOK.md` §5 records fp16 as a rejected route. The owner confirmed it is early development and no longer in use. |
| Duplicated elsewhere | no |
| Live consumers | **seven diagnostic scripts** import it — `fp16_chunk_state`, `fp16_grad_first`, `fp16_loc_8k`, `fp16_locator`, `fp16_seq_scan`, `mem_profile`, `validate_fp16_fork` — and `tasks/v11_bf16_4k.json` / `tasks/v11_fp16.json` set `train_script` to it. No training or serving entry point imports it, and no test does. Those scripts raise `ModuleNotFoundError` loudly while it is archived. |
| Verified | 5/5 hashes matched at destination |
| Undo | `archive/_undo/T2_7.undo.ps1` |

### D2 — restored notes moved into their own archive

| | |
| --- | --- |
| From | `docs/` — the 21 files restored by T2.6 |
| To | `docs/archive/restored-2026-10-24/` |
| Date | 2026-10-24 |
| Files | 21, **each stored as its own file** under its original name |
| Reason | Restoring them to `docs/` created duplicate numbering (`02` twice, `04` twice) and made RUNBOOK's bare `docs/02` and `docs/04` references ambiguous. The owner directed that an archive be created and the files kept separately. |
| Effect | live `docs/` is 7 files with no numbering collision; every reference in `README.md`, `RUNBOOK.md` and `README.zh-CN.md` was rewritten to the new paths |
| Verified | 21/21 hashes preserved |
| Undo | `git mv` each file back into `docs/`, then `git checkout -- README.md RUNBOOK.md README.zh-CN.md` |

### T2.8 — Kaggle notebook archive

| | |
| --- | --- |
| From | `Kaggle/archived/` |
| To | `archive/kaggle/` |
| Date | 2026-10-24 |
| Files | 33 (2.78 MB, tracked) |
| Reason | `(1)`..`(8)` download-duplicate naming mixed loose notes with notebook revisions at the repo root. |
| Duplicated elsewhere | 3 files byte-identical to each other; the remainder are distinct revisions with distinct hashes |
| Ordering preserved | `archive/kaggle/RENAME_MAP.md` records every original name, size, mtime and hash. **UNSURE:** what each Nth revision changed is not recoverable from the filenames. |
| Verified | 33/33 hashes matched at destination |
| Undo | `archive/_undo/T2_8.undo.ps1` (moves `RENAME_MAP.md` to `archive/_undo/` first) |

### T2.9 — derivable chat packs

| | |
| --- | --- |
| From | `data_cache/chat_pack_v11/`, `chat_pack_v11_4k/`, `chat_pack_v12/` |
| To | `archive/data/` (same three names) |
| Date | 2026-10-24 |
| Files | 18 (3.81 GB) |
| Reason | All three are pure transforms of `chat_pack_v10`, which was kept. `v11` and `v11_4k` say `"source": "v10 re-pack to 8K"` in their own `meta.json`; `v12` records `strip_think: true` with `base_v9`. |
| Duplicated elsewhere | not duplicated — *derivable*. Rebuild cost is minutes from `v10` via `scripts/build_chat_mix_v11.py` and the v10 strip-think path. |
| Precondition | `data_cache/index.json` recorded each pack's `meta.json` verbatim plus all 18 SHA-256 values **before** the move |
| Live consumers that will now raise `FileNotFoundError` | `scripts/fp16_loc_8k.py`, `scripts/fp16_seq_scan.py`, `scripts/mem_profile.py`, `scripts/make_kaggle_zips.py`, and the two dead-route configs `tasks/v11_bf16_4k.json`, `tasks/v12_272m.json`. All fail loudly, none silently; all are re-runnable after `build_chat_mix_v11.py`. |
| Verified | 18/18 hashes matched at destination (16.4 s) |
| Undo | `archive/_undo/T2_9.undo.ps1` |

---

## Not relocations (recorded for completeness)

- **T2.5** `docs/_report_0p5b/` — was gitignored, now committed. No path change.
- **T2.6** 21 docs restored from git history (`e11ece8^`, `6355e93^`). Additions, not
  moves — but they were subsequently relocated by **D2** above.
- **T1.2** `20261024_Submit_pre-audit/` — was untracked, now committed. No path change.
- **D2 / FEATURES.md** — new top-level `FEATURES.md` describes current capabilities only,
  deliberately carrying no run state, losses or timestamps.

## Open, deliberately not done

- **T2.1b — HELD at the owner's request.** 28 `.py` files remain under `logs/`. Only 3 are redundant; **25 are the only copy on disk**, and several are operational tooling (release scrubbers, HF upload helpers, PDF bounds checkers, CPU sizing probes). A wholesale move into `archive/` would bury live tools to satisfy an acceptance criterion this plan invented, so nothing was moved. Awaiting a decision between: retire the criterion, promote the reusable tools into `scripts/`, or archive them wholesale.
- **Tier 3** (`pretrain_merged` int64→int16, `chat_pack_v3..v10`, `chat_recovery_mix`) and **Tier 4** (forbidden) were never in scope for this execution.
