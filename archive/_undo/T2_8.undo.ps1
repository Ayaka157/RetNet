# T2_8 undo: move the Kaggle notebook archive back to Kaggle/archived.
# `git mv` is used so the reversal is recorded as a rename, not a delete+add.
$ErrorActionPreference='Stop'
Set-Location 'E:\Workspaces\RetNet'
if (-not (Test-Path 'archive\kaggle')) { throw "Nothing to undo: archive/kaggle absent" }
if (Test-Path 'Kaggle\archived')       { throw "Refusing: Kaggle/archived already exists" }
# RENAME_MAP.md was added after the move and has no counterpart at the original path.
if (Test-Path 'archive\kaggle\RENAME_MAP.md') { Move-Item 'archive\kaggle\RENAME_MAP.md' 'archive\_undo\T2_8.RENAME_MAP.md' }
git mv archive/kaggle Kaggle/archived
"undo done: Kaggle/archived restored (RENAME_MAP preserved at archive/_undo/T2_8.RENAME_MAP.md). Commit it to record the reversal."
