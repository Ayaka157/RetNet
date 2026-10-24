# T2_3 undo: move the 16 v21_ddp prefix dumps back from prefixes/ to logs/archive/
# The 17th file (v21_ddp_1019_1712.out) never moved and needs no undo.
param([switch]$SkipVerify)
$ErrorActionPreference='Stop'
$Prefixes='E:\Workspaces\RetNet\archive\20261019_instance_pull\prefixes'
$Target  ='E:\Workspaces\RetNet\archive\20261019_instance_pull\logs\archive'
$Manifest='E:\Workspaces\RetNet\archive\_undo\T2_3.before.sha256'
if (-not (Test-Path $Prefixes)) { throw "Nothing to undo: $Prefixes absent" }
$entries = Get-Content -LiteralPath $Manifest | Where-Object { $_ -match '^[0-9A-Fa-f]{64}  ' } | ForEach-Object { $p=$_ -split '  ',2; [pscustomobject]@{Hash=$p[0];RelPath=$p[1]} }
foreach ($e in $entries) {
  $from = Join-Path $Prefixes $e.RelPath
  $to   = Join-Path $Target   $e.RelPath
  if (-not (Test-Path $from)) { throw "Nothing to undo: $from absent" }
  if (Test-Path $to)          { throw "Refusing: $to already exists" }
  Move-Item -LiteralPath $from -Destination $to
}
$bad = $entries | Where-Object { (Get-FileHash -LiteralPath (Join-Path $Target $_.RelPath) -Algorithm SHA256).Hash -ne $_.Hash }
if ($bad) { $bad.RelPath | ForEach-Object { "MISMATCH: $_" }; exit 1 }
"undo verified: {0} prefix dumps restored" -f $entries.Count
