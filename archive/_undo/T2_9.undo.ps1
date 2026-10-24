# T2_9 undo: move the three derivable chat packs back into data_cache/.
# Verifies every file against the hashes captured before the move.
param([switch]$SkipVerify)
$ErrorActionPreference='Stop'
$Archive ='E:\Workspaces\RetNet\archive\data'
$Data    ='E:\Workspaces\RetNet\data_cache'
$Manifest='E:\Workspaces\RetNet\archive\_undo\T2_9.before.sha256'
$packs = @('chat_pack_v11','chat_pack_v11_4k','chat_pack_v12')
foreach ($p in $packs) {
  $from = Join-Path $Archive $p
  $to   = Join-Path $Data $p
  if (-not (Test-Path $from)) { throw "Nothing to undo: $from absent" }
  if (Test-Path $to)          { throw "Refusing: $to already exists" }
  Move-Item -LiteralPath $from -Destination $to
}
$entries = Get-Content -LiteralPath $Manifest | Where-Object { $_ -match '^[0-9A-Fa-f]{64}  ' } | ForEach-Object { $x=$_ -split '  ',2; [pscustomobject]@{Hash=$x[0];Rel=$x[1]} }
$bad = @()
foreach ($e in $entries) {
  $p = Join-Path 'E:\Workspaces\RetNet' ($e.Rel -replace '/','\')
  if (-not (Test-Path -LiteralPath $p)) { $bad += "MISSING: $($e.Rel)"; continue }
  if ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash -ne $e.Hash) { $bad += "MISMATCH: $($e.Rel)" }
}
if ($bad) { $bad | ForEach-Object { $_ }; exit 1 }
"undo verified: {0} files restored to data_cache/ (3 packs)" -f $entries.Count
