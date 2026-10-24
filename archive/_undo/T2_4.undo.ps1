# T2_4 undo: move the superseded submission snapshot back to the repository root.
# History is preserved because both the original move and this undo use `git mv`.
$ErrorActionPreference='Stop'
Set-Location 'E:\Workspaces\RetNet'
if (-not (Test-Path 'archive\20261020_Submit')) { throw "Nothing to undo: archive/20261020_Submit absent" }
if (Test-Path '20261020_Submit')               { throw "Refusing: 20261020_Submit already exists" }
git mv archive/20261020_Submit 20261020_Submit
"undo done: 20261020_Submit restored. Commit it to record the reversal."
