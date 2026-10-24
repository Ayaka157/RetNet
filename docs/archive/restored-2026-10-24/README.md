# restored-2026-10-24 — historical engineering notes

These 21 files were recovered from git history on 2026-10-24 and then moved here, out of
`docs/`, so that the live `docs/` directory keeps a single document per number.

| Source commit | Files |
| --- | --- |
| `6355e93^` | 18 notes, including `INDEX.md` |
| `e11ece8^` | 3 notes: `02_训练管线`, `24_AutoDL上机手册`, `26_RetNet-0.5B_技术报告` |

Each file is stored separately under its original name, and all 21 are byte-identical to
the version they were restored from. Their hashes are in `MANIFEST.baseline.sha256` and in
`archive/_undo/POST.sha256`.

## These are historical documents, not current guidance

They describe the 272M era and the surrounding decisions. **Do not treat them as current
status.** For the current capability set see [`/FEATURES.md`](../../../FEATURES.md), for
which artifact is authoritative see [`/CANONICAL.md`](../../../CANONICAL.md), and for
operations see [`/RUNBOOK.md`](../../../RUNBOOK.md).

## Known: their relative links do not all resolve

The bodies were deliberately **not** rewritten. Keeping them byte-identical to what was
restored was judged more valuable than repairing links inside historical records, and it
keeps the provenance verifiable by hash. The consequence is measured and recorded here
rather than hidden:

- **29 links** point at notes deleted in `e11ece8` that were **not** restored and do not
  exist anywhere: `03_评测与基准`, `10_远程部署笔记`, `11b_TileKernels可用性评估`,
  `14_基准测试v3`, `15_计算路径优化`, `16_低显存参数效率`, `19_RetNet-272M_技术报告`,
  `20_分词器调研结论`, `22_0.5B_提速研究`, `25_一步计算构成与压缩空间`, plus
  `./archive/` and `./paper/`.
- **17 links** now resolve to the wrong depth because of the move: `../RUNBOOK.md`,
  `../bench/CONTRACT.md`, `figures/v21_training_status.png`, and siblings such as
  `07_数据集总表_v2.csv` that still live in `docs/` or `figures/`.

Resolving the first group is a decision, not a mechanic: it means restoring the remaining
`e11ece8` deletions as well. If that is wanted, restoring them **into this same directory**
keeps them together and costs one `git checkout` per file.

`docs/archive/INDEX_deleted_20261018.md` is a separate artifact — the preserved copy of the
old `docs/INDEX.md`, captured before `_archive_from_instance/` was relocated. Its sibling
links are likewise historical.
