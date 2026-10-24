# T2_5 undo: stop tracking docs/_report_0p5b (files stay on disk).
$ErrorActionPreference='Stop'
Set-Location 'E:\Workspaces\RetNet'
git rm -r --cached docs/_report_0p5b | Out-Null
"undo done: docs/_report_0p5b untracked again (files still present). Commit to record it."
