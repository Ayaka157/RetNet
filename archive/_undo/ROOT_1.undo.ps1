# ROOT_1 undo: reverse the 2026-10-24 root rearrangement (42 -> 35 items).
# Four items are tracked (git mv, so history is preserved as renames); three are
# gitignored binaries (Move-Item). Reference edits in six files are reverted with
# `git checkout` afterwards, because they were made in the same commit.
$ErrorActionPreference='Stop'
Set-Location 'E:\Workspaces\RetNet'

# tracked
if (Test-Path 'docs\plans\PLAN.reorg-20261024.md')               { git mv docs/plans/PLAN.reorg-20261024.md PLAN.reorg-20261024.md }
if (Test-Path 'docs\plans\PLAN.reorg-20261024.execution-log.md') { git mv docs/plans/PLAN.reorg-20261024.execution-log.md PLAN.reorg-20261024.execution-log.md }
if (Test-Path 'archive\MANIFEST.baseline.sha256')                { git mv archive/MANIFEST.baseline.sha256 MANIFEST.baseline.sha256 }
if (Test-Path 'scripts\_probe_bench_cats.py')                    { git mv scripts/_probe_bench_cats.py _probe.py }

# ignored binaries
if (Test-Path 'archive\logs_bundle.tgz')                  { Move-Item archive\logs_bundle.tgz logs_bundle.tgz }
if (Test-Path 'archive\crash\101126-13906-01.dmp')        { Move-Item archive\crash\101126-13906-01.dmp 101126-13906-01.dmp }
if (Test-Path 'docs\paper\2609.19969.pdf')                { Move-Item docs\paper\2609.19969.pdf 2609.19969.pdf }

# reference edits made in the same commit
git checkout -- CANONICAL.md archive/INDEX.md docs/archive/restored-2026-10-24/README.md scripts/build_data_index.py 2>$null

# empty helper dirs
foreach ($d in 'docs\plans','archive\crash') {
  if ((Test-Path $d) -and -not (Get-ChildItem $d -Force)) { Remove-Item $d -Force }
}
"undo done: root restored to 42 items. Commit to record the reversal."
