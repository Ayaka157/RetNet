# RetNet Workspace Reorganization Plan

- **Plan ID:** reorg-20261024
- **Written:** 2026-10-24 (Asia/Shanghai)
- **Repository HEAD at authoring:** `52c27cd` on `main`
- **Status:** approved for execution — Tier 1 and Tier 2 only
- **Execution log:** `PLAN.reorg-20261024.execution-log.md`

This plan is the output of a read-only archaeological survey of the workspace.
No file was moved, renamed or deleted while it was being written. The survey
manifest was written outside the workspace, to `%TEMP%\retnet_survey\manifest.csv`.

---

## 0. Operating constraints

| Constraint | How the plan honours it |
| --- | --- |
| NEVER delete | Nothing leaves except by relocation into `archive/` with a recorded reason. No forward script contains `Remove-Item` against a data path. |
| NEVER move before approval | Every action has an ID. An ID is executed only after explicit approval. |
| ALWAYS guarantee reversibility | T1.1 must land before any Tier 2 item. Every Tier 2 item ships a paired undo script and a before/after hash manifest. |
| Dry-run by default | Forward scripts accept `-WhatIf` and are run with it before the real move. |

## 1. Evidence base (measured, not assumed)

- 2,237 files hashed with SHA-256 (463.7 MB), 0 unreadable. Scope excluded `.git`,
  `data_cache/`, `checkpoints/`, `hf_cache/`.
- Workspace total ~59 GB: `data_cache` 43.3 GB / 2,532 files, `checkpoints` 14.1 GB /
  29 files, `hf_cache` 1.0 GB.
- Git healthy: clean HEAD `52c27cd` on `main`, 712 tracked files, no stash, no
  worktrees, **zero dangling or unreachable objects**, reflog intact to `140a2f2`,
  one tag `archive/pre-purge-20261019`. The only dirt was one untracked directory.
- 463 exact-duplicate hash groups; 63.7 MB redundancy (0.1% of the tree).
- All 7 `checkpoints/**/model.safetensors` have distinct hashes — 13.5 GB, no
  de-duplication is possible.
- Nothing running at survey time: 0 python processes, 0 listeners on port 8077,
  lock/flag files 5.8–61.9 days stale.
- Reference sweep over 1,030 text files: every one of the 25 `data_cache`
  subdirectories has at least 3 referencing files. There are no data orphans.

## 2. Structural findings

1. **Source code lives inside `logs/`.** `logs/_3090_recover/` holds `retnet/`,
   `retnet_hf/`, `scripts/`, `tests/`, `bench/`, `play_v2.py`, `run_task.py`.
   311 of its 366 files are byte-identical to live source. This violates the
   workspace's own `.gitignore` rule that `logs/` means run products.
2. **Four recovery mirrors with drift.** `legacy/`, `logs/_3090_recover/`,
   `_archive_from_instance/`, and two submission snapshots. Files byte-identical
   to live: 311/366, 273/437, 42/60. Files **unique to the mirrors**: 55, 164, 18.
   `_archive_from_instance` uniquely holds `docs/INDEX.md` — the exact file the
   README links to and which is absent from live `docs/`.
3. **Documentation link rot.** Commit `6355e93` ("Drop the narrative docs")
   deleted 18 files / 5,708 lines deliberately and left the links behind.
   README: 4 dead local links. RUNBOOK: 8 dead paths.
4. **An untracked deliverable.** `20261024_Submit_pre-audit/` (70 files) is the
   current audited submission and is not in git. One `git clean -fdx` destroys it.
5. **Data duplication is documented but not actionable in bulk.** The in-repo
   audit measures 7.2x duplication across 12 chat packs (~552 M unique tokens of
   3,976 M). Most of that is ROOT, BUILD INPUT, or cheap historical control.

## 3. Target architecture

```
RetNet/
  retnet/ retnet_hf/ bench/ tests/ tasks/ serve/ scripts/ figures/ Papers/
  research/ legacy/ Kaggle/
  docs/
   00-04, 00F1, 21b          live notes
   archive/                  restored deleted notes + preserved mojibake INDEX
 20261024_Submit_pre-audit/ tracked deliverable
  archive/                   NEW: relocation ledger + relocated trees
   INDEX.md                   the ledger
   _undo/                     per-item undo scripts + before-manifests
   20261019_3090_recover/     T2.1
   20261019_instance_pull/    T2.2 + T2.3 prefixes/
   20261020_Submit/           T2.4
   retnet_hf_fp16/            T2.7
   kaggle/                    T2.8
   data/                      T2.9
  data_cache/                unchanged paths + INDEX.md + index.json
  checkpoints/ hf_cache/     untouched
  logs/                      run products only, no source
  CANONICAL.md               NEW: the authoritative map
  MANIFEST.baseline.sha256   NEW: pre-move baseline
```

## 4. Tier 1 — act now

### T1.1 Baseline manifest
Create `MANIFEST.baseline.sha256` at the repo root: SHA-256 + relative path for
every file outside `.git/`, `data_cache/`, `checkpoints/`, `hf_cache/`.
Precondition for reversibility. Mirrors the existing `20261020_Submit/MANIFEST.sha256`
convention. **Undo:** delete the file.

### T1.2 Track the untracked deliverable
`git add 20261024_Submit_pre-audit/ && git commit`. **Undo:** `git reset --soft HEAD~1`.

### T1.3 Make the data knowledge durable
Move `logs/_agent_dataset_audit.md` to `docs/21b_数据集盘点_20261017.md` and commit.
Rationale: the best document on which chat pack is current lives in a gitignored
directory. **Undo:** `git rm --cached` then move the file back.

### T1.4 Write `CANONICAL.md`
One authoritative answer for: current dataset, current serving weights, current
training entry point, the documented dead routes, and any unresolved ambiguity.
Purely additive. **Undo:** delete the file.

### T1.5 Prepare the ignore rules
Add root-anchored ignores so relocated payload cannot enter the index, while
tracked trees can still be moved with `git mv`. See amendment A1 in the execution
log: a bare `archive/` pattern would also match `docs/archive/`, so the ignores
must be root-anchored and item-specific.

Required patterns:

```
/archive/20261019_3090_recover/
/archive/20261019_instance_pull/
/archive/data/
```

Plus neutralise the `docs/_report_0p5b/` ignore line so T2.5 can commit it.
**Undo:** `git checkout -- .gitignore`.

## 5. Tier 2 — propose, apply only on approval

Each group is independently approvable. One commit per group. Every forward
command is run with `-WhatIf` first.

### Group A — restore the workspace's own layout rule

**T2.1** `logs/_3090_recover/` (366 files, 33.4 MB) -> `archive/20261019_3090_recover/`.
Reason: source code inside a run-products directory; 311/366 files byte-identical
to live. Preserve the 55 unique files (5.9 MB).
Undo: `Move-Item archive/20261019_3090_recover logs/_3090_recover`.

### Group B — consolidate instance archives

**T2.2** `_archive_from_instance/` (437 files, 165.5 MB) -> `archive/20261019_instance_pull/`.
Reason: 273/437 duplicated; the tree is a GBK-mojibake extraction whose content is
proven identical by hash while only the names are corrupted.
**Mandatory precondition:** copy `_archive_from_instance/docs/INDEX.md` to
`docs/archive/INDEX_deleted_20261018.md` first — it is the only copy of that file
outside git history.
Undo: `Move-Item archive/20261019_instance_pull _archive_from_instance`.
Do not reverse the precondition copy; it is redundant by design.

**T2.3** The 16 `v21_ddp_1019_*.out` snapshots (~148 MB) ->
`archive/20261019_instance_pull/prefixes/`, keeping `v21_ddp_1019_1712.out` in place.
Reason: monotonic sizes 6.6 -> 10.2 MB; each file is a growing prefix of its
successor. UNSURE: inferred from monotonic size, not diffed.
Undo: `Move-Item` the 16 files back.

### Group C — reconcile the submission snapshots

**T2.4** `20261020_Submit/` (56 files, tracked) -> `archive/20261020_Submit/` via
`git mv`. Reason: superseded by T1.2's snapshot; 5 documents byte-identical to live
`docs/`. Undo: `git mv` back.

**T2.5** `docs/_report_0p5b/` (6 files, gitignored) -> committed. Reason: it is the
working scaffold behind committed `docs/03` and `docs/04`; the source is ignored
while its output is tracked. Undo: `git rm --cached -r`.

### Group D — repair documentation integrity

**T2.6** Resolve 12 dead links. Decision D1 taken as **restore** (the plan's
recommendation): recover from `6355e93^` into `docs/`:
`docs/12_训练状态盘点.md`, `docs/24_AutoDL上机手册.md`,
`docs/26_RetNet-0.5B_技术报告.md`, `docs/INDEX.md`.
Undo: `git checkout -- README.md RUNBOOK.md docs/`.

### Group E — retire rejected and duplicated branches

**T2.7** `retnet_hf_fp16/` (5 files, tracked) -> `archive/retnet_hf_fp16/` via
`git mv`. Reason: RUNBOOK §5 records fp16 as a rejected route (loss 0.0 /
grad_norm NaN at steps 20 and 40). Residual risk R7: `scripts/audit_mem.py` and
`scripts/fp16_*.py` import it. Undo: `git mv` back.

**T2.8** `Kaggle/archived/` -> `archive/kaggle/` via `git mv`, with a rename map
recording every file's original name, size, mtime and hash. Reason: `(1)`..`(8)`
download-duplicate naming; three files byte-identical. These are version groups,
never equivalents. Undo: `git mv` back.

### Group F — the first data slice

**T2.9** `data_cache/chat_pack_v11` (1.24 GB) + `chat_pack_v11_4k` (1.24 GB) +
`chat_pack_v12` (1.32 GB) = 3.8 GB -> `archive/data/`.
Reason: all three are pure transforms of `chat_pack_v10`, which is kept. `v11` and
`v11_4k` are `row_len` repacks (`meta.json` states `"source": "v10 re-pack to 8K"`);
`v12` is `v10` with `<think>` stripped (prompt and plain token counts identical,
only reply differs). Their only config references drive routes RUNBOOK §5 records
as dead and zero-yield.
**Precondition:** `data_cache/index.json` must first record each pack's `meta.json`
verbatim plus per-file SHA-256 for the files being moved.
Undo: `Move-Item archive/data/chat_pack_vNN data_cache/chat_pack_vNN`.

## 6. Data retention taxonomy

| Class | Meaning | Action |
| --- | --- | --- |
| ROOT | Raw corpora, not derivable here (14.8 GB) | Never move |
| CURRENT | Read by training/serving now | Never move |
| BUILD INPUT | Ancestor a kept artifact is built from | Never move while descendant kept |
| DERIVABLE | Regenerable from a kept input in minutes | The only archivable class |
| SUPERSEDED | Historical control, not cheaply regenerable | Keep |
| INCOMPATIBLE | Wrong tokenizer (16k/8k vs 20k) | Keep, never mix |
| NO MASK / CORRUPT | Documented unusable, provenance unclear | Keep, never delete |

Lineage: raw -> v3 -> v4 -> {v5..v9}; v9 -> v10 -> {v11, v11_4k, v12}; v9 -> v13.
`pretrain_v2` is CURRENT. `pretrain_merged` is SUPERSEDED with a 7.9 GB int64 waste.

## 7. Tier 3 — deferred, with the evidence that would decide each

- **P1 `pretrain_merged` int64 -> int16** (saves 7.9 GB). Gate: verify the overlap
  with `pretrain_v2`, which the in-repo audit flags as never checked.
- **P2 `chat_pack_v3..v10`** (6.2 GB historical controls). Gate: confirmation that
  no cross-window comparison is planned.
- **P3 `chat_recovery_mix` / `chat_sft_mix`.** Gate: provenance evidence.

## 8. Tier 4 — do not do

- Never move `data_cache/` or `checkpoints/` wholesale (57 GB behind literal paths
  in 26 task configs).
- Never hardlink-deduplicate identical files (edits would silently alias across
  versions that must diverge).
- Never move `HF Access Token.txt` (breaks `scripts/_hf_login_from_file.py`).
- Never delete the crash dump, stale locks, or `__pycache__`.
- Never rewrite git history or drop the archive tag.

## 9. Execution protocol

```
Gate 0  approve item IDs
Gate 1  T1.1 manifest written and verified
Gate 2  T1.2-T1.5 landed; git clean; .gitignore amended
Gate 3  per Tier 2 group: dry-run -> review -> apply -> verify -> next group
Gate 4  final manifest diff: only approved paths changed
```

Rules: one group per commit; dry-run default; verify immediately after each move;
abort on any hash mismatch, any path outside the approved set changing, or any
unexpected untracked addition.

## 10. Reversibility design

For each Tier 2 item, before the move:
1. `archive/_undo/T2_n.before.sha256` — SHA-256, size, mtime of every touched path.
2. `archive/_undo/T2_n.undo.ps1` — idempotent, refuses if the destination exists,
   verifies hashes after restoring.
3. After the move: re-hash at destination, compare to the before-manifest, and
   assert `git status --porcelain` shows only intended entries.

Two traps this design exists to prevent:
- **Relocating an ignored tree un-ignores it.** `logs/` and `_archive_from_instance/`
  are ignored by name; after relocation into `archive/` the payload would surface as
  untracked (165 MB) and could be swept in by a later `git add -A`. Hence T1.5 first.
- **Relocating a tracked tree with `Move-Item` records a deletion.** Use `git mv`.

## 11. Acceptance criteria

1. `MANIFEST.baseline.sha256` exists; a post-execution manifest differs only on the
   approved IDs.
2. `git status --porcelain` is clean, or shows only intended rename entries.
3. Every Tier 2 item has a paired undo script; the undo pattern has been rehearsed
   once on a throwaway tree.
4. `archive/INDEX.md` records for every relocated path: origin, destination, date,
   reason, unique-file count, undo script name.
5. No `data_cache/` or `checkpoints/` path has changed except the approved T2.9 set.
6. README and RUNBOOK link checks return zero missing local paths.
7. No source file remains under `logs/`.
