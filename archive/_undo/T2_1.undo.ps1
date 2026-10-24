# T2_1 undo: archive/20261019_3090_recover -> logs/_3090_recover
# Restores logs/_3090_recover exactly as it was before plan reorg-20261024.
param([switch]$SkipVerify)
$ErrorActionPreference='Stop'
$From='E:\Workspaces\RetNet\archive\20261019_3090_recover'
$To  ='E:\Workspaces\RetNet\logs\_3090_recover'
$Manifest='E:\Workspaces\RetNet\archive\_undo\T2_1.before.sha256'
if (-not (Test-Path $From)) { throw "Nothing to undo: $From absent" }
if (Test-Path $To)         { throw "Refusing: $To already exists" }
Move-Item -LiteralPath $From -Destination $To
$entries = Get-Content -LiteralPath $Manifest | Where-Object { $_ -match '^[0-9A-Fa-f]{64}  ' } | ForEach-Object { $p = $_ -split '  ', 2; [pscustomobject]@{Hash=$p[0];RelPath=$p[1]} }
$bad = $entries | Where-Object { (Get-FileHash -LiteralPath (Join-Path $To $_.RelPath) -Algorithm SHA256).Hash -ne $_.Hash }
if ($bad) { $bad.RelPath | ForEach-Object { "MISMATCH: $_" }; exit 1 }
"undo verified: {0} files restored to logs/_3090_recover" -f $entries.Count
