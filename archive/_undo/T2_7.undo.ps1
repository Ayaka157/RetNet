# T2_7 undo: move the abandoned fp16 branch back to the repository root.
# `git mv` so the reversal is recorded as a rename. Note that seven diagnostic
# scripts under scripts/ import this module; they fail loudly while it is archived.
$ErrorActionPreference='Stop'
Set-Location 'E:\Workspaces\RetNet'
if (-not (Test-Path 'archive\retnet_hf_fp16')) { throw "Nothing to undo: archive/retnet_hf_fp16 absent" }
if (Test-Path 'retnet_hf_fp16')               { throw "Refusing: retnet_hf_fp16 already exists" }
git mv archive/retnet_hf_fp16 retnet_hf_fp16
"undo done: retnet_hf_fp16 restored. Commit it to record the reversal."
