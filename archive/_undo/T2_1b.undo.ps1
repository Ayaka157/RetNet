# T2_1b undo: put the 28 helper scripts back under logs/ exactly where they came from.
# Reverses both halves: the 13 promoted into scripts/ (tracked) and the 15 archived
# into archive/logs_scratch/ (untracked by design - two of them embed a real internal
# network address and must never be committed).
$ErrorActionPreference='Stop'
Set-Location 'E:\Workspaces\RetNet'

$promoted = @{
  '_cpu_footprint.py'='logs'; '_cpu_threads.py'='logs'; '_cpu_timing.py'='logs'
  '_scrub_context.py'='logs'; '_scrub_scan.py'='logs'
  '_verify_pdf.py'='logs'; '_pagecheck.py'='logs'
  '_crosscheck2.py'='logs'; '_crossdiff.py'='logs'
  '_hf_login_from_file.py'='logs'; '_hf_stage.py'='logs'; '_hf_upload.py'='logs'
  '_fetch_corpora.py'='logs\linux'
}
foreach ($k in $promoted.Keys) {
  $from = Join-Path 'scripts' $k
  $to   = Join-Path $promoted[$k] $k
  if (-not (Test-Path $from)) { Write-Warning "absent: $from"; continue }
  if (Test-Path $to)           { Write-Warning "refusing, exists: $to"; continue }
  Move-Item -LiteralPath $from -Destination $to
}
git rm --cached --quiet -- scripts/_cpu_footprint.py scripts/_cpu_threads.py scripts/_cpu_timing.py `
  scripts/_scrub_context.py scripts/_scrub_scan.py scripts/_verify_pdf.py scripts/_pagecheck.py `
  scripts/_crosscheck2.py scripts/_crossdiff.py scripts/_hf_login_from_file.py scripts/_hf_stage.py `
  scripts/_hf_upload.py scripts/_fetch_corpora.py 2>$null

$archived = @{
  '_shot_chat.py'='logs'; '_shot_ui.py'='logs'; '_shot_ui2.py'='logs'
  '_try_prompts.py'='logs'; '_try_prompts2.py'='logs'
  '_test_plan_gc.py'='logs'; '_test_plan_pipeline.py'='logs'
  '_bench_probe.py'='logs'; '_dump_bench.py'='logs'; '_drop_phase2_codemath.py'='logs'
  '_overlap_check.py'='logs'; 'fetch_model.py'='logs\_space'
  'env_check.py'='logs\autodl_20261012\scripts\linux'
  '_scrub_apply.py'='logs'; '_pdf_scrub_check.py'='logs'
}
foreach ($k in $archived.Keys) {
  $from = Join-Path 'archive\logs_scratch' $k
  $to   = Join-Path $archived[$k] $k
  if (-not (Test-Path $from)) { Write-Warning "absent: $from"; continue }
  if (Test-Path $to)           { Write-Warning "refusing, exists: $to"; continue }
  Move-Item -LiteralPath $from -Destination $to
}
$n = (Get-ChildItem logs -Recurse -File -Filter *.py).Count
"undo done: {0} .py files now under logs/ (expected 28). Commit to record the reversal." -f $n
