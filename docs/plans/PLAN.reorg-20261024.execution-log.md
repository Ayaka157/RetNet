# Execution log — plan reorg-20261024

Companion to `docs/plans/PLAN.reorg-20261024.md`. This file records **every deviation** from
the approved plan, so "followed the plan exactly" is auditable rather than asserted.

Executed 2026-10-24 (Asia/Shanghai). Starting HEAD `52c27cd`. All work is committed;
`git status --porcelain` is empty at the end.

## Commit sequence

| # | Commit | Scope |
| --- | --- | --- |
| 1 | `9bbd22e` | plan file itself |
| 2 | `ce09e3a` | T1.2 track the untracked deliverable |
| 3 | `d7339d1` | T1.3 promote the dataset audit out of `logs/` |
| 4 | `5abc126` | T1.1 baseline manifest, T1.4 `CANONICAL.md`, T1.5 ignore rules |
| 5 | `4ab99b6` | T2.9 precondition: `data_cache/index.json` |
| 6 | `3704db0` | Group A — T2.1 |
| 7 | `2786c93` | Group B — T2.2, T2.3 |
| 8 | `83e1396` | Group C — T2.4, T2.5 |
| 9 | `5115f72` | Group D — T2.6 |
| 10 | `224e6ae` | Group E — T2.8 (T2.7 withheld) |
| 11 | `3f08c1e` | Group F — T2.9, ledger, ACL blind-spot closure |
| 12 | *(this commit)* | execution log + post-execution manifest |

## Deviations and amendments

### A0 — baseline has an ACL blind spot (found at T1.1)
`Get-ChildItem -Recurse` could not descend into three directories under
`Kaggle/_audit/checks/`. My first survey had silently skipped them because it used
`-ErrorAction SilentlyContinue`; the manifest pass was changed to capture the error
variable so the gap is recorded instead of invisible. Closed later under A6.

### A1 — `archive/` ignore pattern had to be root-anchored and item-specific
The plan said "add `archive/`". A bare `archive/` pattern matches at **any depth**, so
it would also have ignored `docs/archive/`, and it would have blocked `git mv` for the
tracked trees in T2.4/T2.7/T2.8. Implemented instead as:

```
/archive/20261019_3090_recover/
/archive/20261019_instance_pull/
/archive/data/
```

Verified with `git check-ignore`: payload ignored, `docs/archive/`, `archive/20261020_Submit/`,
`archive/retnet_hf_fp16/`, `archive/kaggle/` all remain trackable.

### A2 — decision D3 resolved by evidence, and no rename performed
The plan forbade renaming until D3 was answered. Rather than guess, the question was
settled by reading the code: `serve/retnet_play_server.py:48` does
`from play_v2 import encode_prompt_ids`, and `play.py` can only load handwritten
`model.pt`, not any current safetensors checkpoint. Both scripts are live and serve
different formats. Recorded in `CANONICAL.md` §3. **No file was renamed.**

### A3 — added `scripts/build_data_index.py` (not in the plan)
The plan required `data_cache/index.json` but not a generator. Without one the index
is not reproducible, so a 200-line generator was added. Additive and reversible with
`git rm`. It reads `.npy` headers by hand and never touches array payloads.

### A4 — T2.9 blast radius was under-counted in the plan
The plan said "only dead routes reference them" (assumed 3 references). The index
measured **12** live references for `chat_pack_v11`. Inspection showed 1 is the index
script itself, 2 are Kaggle *remote* paths (`/kaggle/input/...`), and 1 is the builder
that regenerates the pack. The genuine local readers are four diagnostic/packaging
scripts and two dead-route configs. The plan's substance held; the ledger now lists
exactly which scripts will raise `FileNotFoundError`.

### A5 — T2.3's stated justification was factually wrong
The plan claimed the last `v21_ddp` snapshot is "byte-identical to live
`logs/linux/v21_ddp.out`". **There is no live `logs/linux/v21_ddp.out`.** The identical
file is `_archive_from_instance/logs/linux/v21_ddp.out`, i.e. a sibling already inside
the archive. The action was still correct; the reason in `archive/INDEX.md` is now
accurate. Separately, the assumed "prefix" property was *proved* rather than inferred:
all **16/16** adjacent pairs were verified by hashing the shorter file and comparing it
to the prefix of the longer.

### A6 — UAC elevation used (approved by the user, read-only)
Three directories granted only `SYSTEM`, `Administrators` and `OWNER RIGHTS` with
inheritance disabled; the session runs at Medium integrity with a deny-only
Administrators token. Two elevated `pwsh` runs were used, each approved via a UAC
prompt the user accepted:

1. enumerate the directories (found 6 leftover checkpoint-resume test fixtures);
2. hash those 6 files.

**No `takeown`, no `icacls` change, no file created or deleted in those directories.**
The 6 hashes are appended to `archive/MANIFEST.baseline.sha256` under a provenance comment.

### A7 — "restore" interpreted as restore-to-original-paths, and one side effect surfaced
D1 was answered "Restore, then turn to archive". Executed the restore half literally:
**18** files from `6355e93^` plus **3** (`docs/02_训练管线.md`, `docs/24_…`, `docs/26_…`)
from `e11ece8^`, because the latter three were deleted by a different commit. Total 21.

Side effect, **reported not resolved**: `docs/` now has duplicate numbering —
`02_训练管线.md` beside `02_续训恢复计划_220元.md`, and `04_路线与扩展研究.md` beside
`04_模型数学与损失.md`. Running `RUNBOOK`'s bare references `docs/02` and `docs/04` are
now ambiguous. The "then turn to archive" half was **not** executed, because where the
restored files should live (and whether README/RUNBOOK links get rewritten to point at
`docs/archive/`) is a decision, not a mechanic.

### A8 — T2.7 withheld, per the user
The user asked for information before archiving `retnet_hf_fp16`. It was **not moved**.
Evidence gathered is in the final report: 7 diagnostic scripts import it, 2 dead-route
task configs point `train_script` at it, and no training/serving entry point imports it.

### A9 — the undo rehearsal caught a real bug
The first rehearsal failed: `Import-Csv -Delimiter ' '` cannot split a manifest that
uses **two** spaces, so `RelPath` came back empty and the script tried to hash a
directory. Template rewritten to split on `'  '` explicitly, then re-rehearsed:
round-trip byte-identical, idempotency guard correctly refused a second run. The `git mv`
undo variant was separately rehearsed in a throwaway clone.

## Verification results

| Check | Result |
| --- | --- |
| Baseline manifest entries | 2,244 (2,238 unelevated + 6 elevated) |
| Post-execution entries | 2,294 |
| Paths removed | 899 |
| — byte-preserved at destination | **893** |
| — ACL-blind, unchanged, unreadable unelevated | 6 |
| — **unexplained** | **0** |
| Content changed in place | 1 (`.gitignore`, from T1.5) |
| Per-move hash verification | T2.1 366/366 · T2.2 437/437 · T2.3 16/16 · T2.8 33/33 · T2.9 18/18 |
| Prefix proof | 16/16 adjacent pairs |

## Acceptance criteria — measured outcome

| # | Criterion | Result |
| --- | --- | --- |
| 1 | Baseline manifest exists; post differs only on approved IDs | **PASS** — 899 removals, 893 byte-preserved, 6 ACL-blind, **0 unexplained** |
| 2 | `git status --porcelain` clean or only intended renames | **PASS** — clean |
| 3 | Every Tier 2 item has an undo script; pattern rehearsed | **PASS** — 8 scripts (`T2_1,2,3,4,5,6,8,9`), both move patterns rehearsed |
| 4 | `archive/INDEX.md` ledgers every relocation | **PASS** — origin, destination, date, reason, duplication, undo |
| 5 | `data_cache/` and `checkpoints/` only changed per T2.9 | **PASS** — `checkpoints` 29 files / 14,405.7 MB identical; `data_cache` 26→23 dirs, the 3 all verified in `archive/data/` |
| 6 | README and RUNBOOK link checks return zero missing | **PARTIAL** — README 0 missing; RUNBOOK 4 |
| 7 | No source file remains under `logs/` | **FAIL** — 28 `.py` files remain |

### AC6 detail — why RUNBOOK still has 4

- `checkpoints/retnet_272m_rp/` and `checkpoints/retnet_272m_v10/` — the 272M weights were
  deleted from disk (recorded in commit `f66bd5c`, "Record what is on HF and what was
  deleted locally"). They are **not recoverable from git**, so no restore is possible.
- `docs/02` and `docs/04` — bare prefix references that were unambiguous before the T2.6
  restore and are ambiguous now, because `docs/` contains two files numbered `02` and two
  numbered `04` (see A7). This is a direct consequence of restoring to original paths.

### AC7 detail — `logs/` still holds source

T2.1 only relocated `logs/_3090_recover/`. A wider check after the fact found **28 loose
scripts** at the `logs/` root (`_hf_login_from_file.py`, `_scrub_scan.py`, `_shot_ui.py`,
`_try_prompts.py`, …) plus `logs/_space/fetch_model.py`, `logs/linux/_fetch_corpora.py`
and `logs/autodl_20261012/scripts/linux/env_check.py`. They are gitignored scratch
helpers, they were outside T2.1's stated scope, and relocating them is a **new action
that was never approved**, so they were left in place. Proposed follow-up: **T2.1b**,
same pattern as T2.1, pending approval.

Note: `logs/_hf_login_from_file.py` is the script that reads `HF Access Token.txt`, which
is why Tier 4 forbids moving that token file.

---

# Round 2 — owner decisions applied

The owner answered the four open questions. Three were executed; one was held.

| # | Owner answer | Action taken |
| --- | --- | --- |
| T2.7 | archive it ("early development and now no longer in use") | `retnet_hf_fp16/` -> `archive/retnet_hf_fp16/`, 5/5 hashes verified, undo `T2_7.undo.ps1` |
| 2 | create an archive and store them separately, then write a new current-features document | 21 restored notes -> `docs/archive/restored-2026-10-24/` (each its own file, 21/21 hashes preserved); all references rewritten; `FEATURES.md` written |
| 3 | **Hold — need information** | **no action taken**; inventory supplied instead |
| 4 | mark them gone | RUNBOOK §0 annotated: both 272M checkpoints marked deleted locally, weights on HF; snapshot caveat added |

### Clarification on the T2.7 instruction
The answer read "ACHIEVE retnet_hf_fp16". Read literally, "achieve" is not an operation this
plan defines. It was interpreted as **archive**, because the stated reason — "early
development and now no longer in use" — is a reason to retire, and the owner's answer to
question 1 was "as mentioned above". If the intent was something else, the reversal is
`archive/_undo/T2_7.undo.ps1`, one command.

### D1 final shape
The "then turn to archive" half is now done. `docs/` holds 7 live files with no numbering
collision; the 21 restored notes live under `docs/archive/restored-2026-10-24/` as separate
files. References were rewritten in `README.md`, `README.zh-CN.md` and `RUNBOOK.md`, and the
four bare `docs/02`, `docs/04`, `docs/09` references were expanded to full paths.

### New document
`FEATURES.md` describes the current capability set — architecture, training, data, serving,
evaluation, tests, repository map, and the deliberate scope limits. It carries **no** losses,
throughput, training progress, dates or "currently" statements, per the instruction that it
represent current features and not real-time status.

### Superseding acceptance status

| # | Criterion | Previous | Now |
| --- | --- | --- | --- |
| 1 | manifest diff only on approved IDs | PASS | **PASS** — 904 removals, 898 byte-preserved, 6 ACL-blind, **0 unexplained** |
| 2 | working tree clean | PASS | **PASS** |
| 3 | undo script per item | PASS | **PASS** — 9 scripts |
| 4 | relocation ledger complete | PASS | **PASS** — 9 ledger sections, incl. T2.7 and D2 |
| 5 | data/weights untouched beyond T2.9 | PASS | **PASS** — `checkpoints` 29 files / 14,405.7 MB unchanged |
| 6 | README and RUNBOOK zero missing links | PARTIAL | **PASS** — 0 and 0 |
| 7 | no source left under `logs/` | FAIL | **FAIL — held by owner decision 3**, no action taken |

In-place content changes across the whole execution: `.gitignore`, `README.md`,
`README.zh-CN.md`, `RUNBOOK.md`. Everything else that changed location is byte-identical.

---

# Round 3 — T2.1b executed as *changed* option 2

The owner answered AC7 with "Do as your suggestion", then clarified: "i mean change it
(the option 2)" — i.e. selective promotion, adapted for the safety problem found during
pre-flight.

### The finding that changed the split
Before promoting anything into a **tracked** directory, all 28 candidates were scanned for
secret-shaped content. Two hit:

- `_scrub_apply.py` — contains a real internal network address (one occurrence)
- `_pdf_scrub_check.py` — contains the same address (one occurrence)

These are the very scripts written to strip that class of identifier before publishing, so
promoting them would have committed exactly what the project's own `.gitignore` preamble
forbids ("credentials and private information never enter the repository"). The split was
therefore changed from the original suggestion.

### What was done
- **13 promoted into `scripts/`** and are now tracked: `_cpu_footprint`, `_cpu_threads`,
  `_cpu_timing`, `_scrub_context`, `_scrub_scan`, `_verify_pdf`, `_pagecheck`,
  `_crosscheck2`, `_crossdiff`, `_hf_login_from_file`, `_hf_stage`, `_hf_upload`,
  `_fetch_corpora`.
- **15 archived into `archive/logs_scratch/`**, which is gitignored: the 5 screenshot/demo
  scripts, 2 plan-pipeline tests, 4 bench utilities, `fetch_model.py`, the redundant
  `env_check.py`, and the 2 address-bearing scrubbers.
- `/archive/logs_scratch/` added to `.gitignore` with a root-anchored rule.

### Two deviations worth recording
1. **The suggested split was amended for one file.** `_fetch_corpora.py` was originally in
   my "instance helpers -> archive" group, but `scripts/linux/_supervise_fetch.sh:16`
   already calls `scripts/_fetch_corpora.py` — a path that did not exist. Promoting it
   repaired a pre-existing broken reference instead of creating a new one, so it moved to
   the promote side.
2. **I created and then caught a leak of my own.** The first version of the `.gitignore`
   comment explained the exclusion by writing the address out literally — which would have
   committed it. The staged-blob secret sweep caught it before commit; the comment now
   describes the address without reproducing it.

### Result
`logs/` contains **0** `.py` files. No staged blob contains the address or a token literal.

| # | Criterion | Status |
| --- | --- | --- |
| 7 | no source file remains under `logs/` | **PASS** (previously FAIL) |
