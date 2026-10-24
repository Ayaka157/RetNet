# Kaggle archived notebooks — rename map

Recorded 2026-10-24 by plan reorg-20261024 (T2.8) immediately before the
`Kaggle/archived` -> `archive/kaggle` relocation. Filenames are byte-for-byte
unchanged. This file exists so the ordering information carried by the `(1)`..`(8)`
suffixes is not lost, and so every file stays verifiable by hash.

## The `(N)` suffixes are NOT equivalents

A `(N)` suffix means a browser re-download of a file that already existed: it is a
superseded revision, never a duplicate to be collapsed. Three files are byte-identical
(`retnet-kaggle-train-8k (8).ipynb`, its `.json` twin, and the `FixThis/` copy); all
other revisions differ.

**UNSURE:** what each Nth revision changed is not recoverable from the filenames.
The mtime column is the only ordering evidence that survives this move.

| file | bytes | mtime | sha256 (first 16) |
|---|---:|---|---|
| `旧的报错，别看！！！.txt` | 49,591 | 2026-08-25 23:48 | `DB871133CDD3CE3E` |
| `FixThis/.workbuddy/memory/2026-08-27.md` | 6,941 | 2026-08-27 22:31 | `AFD7B4E7DE4964F3` |
| `FixThis/.workbuddy/memory/2026-08-28.md` | 1,943 | 2026-08-28 13:27 | `30CB6E682A1992AF` |
| `FixThis/previous_analysis/autocast_vs_scan_conflict.md` | 20,509 | 2026-08-27 18:42 | `2C9AE410DD94A265` |
| `FixThis/previous_analysis/retnet_root_cause.md` | 8,536 | 2026-08-27 18:42 | `0EB47715E9C382A4` |
| `FixThis/RetNet_TPU_修复分析报告.md` | 17,799 | 2026-08-28 13:26 | `D6E292C703529C1F` |
| `FixThis/retnet-kaggle-train-8k (8).ipynb` | 87,666 | 2026-08-27 14:29 | `B23B3077ADE9866C` |
| `FixThis/retnet-kaggle-train-8k (fixed).ipynb` | 95,658 | 2026-08-27 19:10 | `06C40A53C63A0C40` |
| `FixThis/retnet-kaggle-train-8k-final-run1.ipynb` | 161,286 | 2026-08-27 21:50 | `4A6F29514B7A7804` |
| `FixThis/retnet-kaggle-train-8k-final.ipynb` | 99,192 | 2026-08-27 19:37 | `50530253C92142AF` |
| `FixThis/retnet-kaggle-train-8k-final2-run1.ipynb` | 71,564 | 2026-08-28 13:12 | `EB19E31719449876` |
| `FixThis/retnet-kaggle-train-8k-final2.ipynb` | 98,505 | 2026-08-27 22:17 | `CD6E60ED4A77163C` |
| `FixThis/retnet-kaggle-train-8k-final3.ipynb` | 98,940 | 2026-08-28 13:24 | `250B912E71C6BB0B` |
| `retnet_kaggle_train_8k-202608252225.ipynb` | 42,002 | 2026-08-25 22:12 | `20D50D7EBBDA8D7F` |
| `retnet-kaggle-tpu-v5e8-global-bf16 (1).ipynb` | 157,939 | 2026-08-27 12:28 | `B748EA9F3D58C554` |
| `retnet-kaggle-tpu-v5e8-global-bf16.ipynb` | 166,359 | 2026-08-27 12:23 | `A984DBF9A5B986F2` |
| `retnet-kaggle-tpu-v5e8-scan-dtype-fixed (1).ipynb` | 103,303 | 2026-08-27 12:17 | `80150CACE9B052EE` |
| `retnet-kaggle-tpu-v5e8-scan-dtype-fixed.ipynb` | 165,794 | 2026-08-27 12:15 | `F301959170346CF8` |
| `retnet-kaggle-tpu-v5e8-torchxla-scan-fixed-launch.ipynb` | 54,556 | 2026-08-27 12:06 | `61AF0E9573641B26` |
| `retnet-kaggle-tpu-v5e8-torchxla-scan.ipynb` | 54,603 | 2026-08-27 12:03 | `1317EB36549F75C7` |
| `retnet-kaggle-train-8k (1).ipynb` | 45,627 | 2026-08-27 11:52 | `37E0CE26E3889E7B` |
| `retnet-kaggle-train-8k (2).ipynb` | 44,317 | 2026-08-27 11:56 | `5695F7B45B0AA120` |
| `retnet-kaggle-train-8k (3).ipynb` | 156,325 | 2026-08-27 12:12 | `10087BBBD00F31DE` |
| `retnet-kaggle-train-8k (4).ipynb` | 159,750 | 2026-08-27 13:18 | `D4275C3A140A6720` |
| `retnet-kaggle-train-8k (5).ipynb` | 158,939 | 2026-08-27 13:25 | `EC8CD764783EF290` |
| `retnet-kaggle-train-8k (6).ipynb` | 104,380 | 2026-08-27 13:34 | `49D46FED479F04CF` |
| `retnet-kaggle-train-8k (7).ipynb` | 102,037 | 2026-08-27 14:18 | `A27EC03E078D2048` |
| `retnet-kaggle-train-8k (8).ipynb` | 87,666 | 2026-08-27 14:29 | `B23B3077ADE9866C` |
| `retnet-kaggle-train-8k (8).json` | 87,666 | 2026-08-27 14:29 | `B23B3077ADE9866C` |
| `retnet-kaggle-train-8k-202608260217.ipynb` | 95,490 | 2026-08-26 02:17 | `C6C867EFD5C89059` |
| `retnet-kaggle-train-8k.ipynb` | 44,811 | 2026-08-25 22:28 | `D4F53BD0DE615ECC` |
| `the-ultimate-pytorch-tpu-tutorial-jigsaw-xlm-r.ipynb` | 225,457 | 2026-08-26 02:36 | `9C9FE19EE71A6F7C` |
| `v5e-training.md.md` | 38,358 | 2026-08-25 23:51 | `11AB6C2DC43435EE` |

Full hashes: `archive/_undo/T2_8.before.sha256`. Undo: `archive/_undo/T2_8.undo.ps1`.
