# T2_2 undo: archive/20261019_instance_pull -> _archive_from_instance
# Does NOT undo the INDEX.md precondition copy; that copy is redundant by design.
param([switch]$SkipVerify)
$ErrorActionPreference='Stop'
$From='E:\Workspaces\RetNet\archive\20261019_instance_pull'
$To  ='E:\Workspaces\RetNet\_archive_from_instance'
$Manifest='E:\Workspaces\RetNet\archive\_undo\T2_2.before.sha256'
if (-not (Test-Path $From)) { throw "Nothing to undo: $From absent" }
if (Test-Path $To)         { throw "Refusing: $To already exists" }
Move-Item -LiteralPath $From -Destination $To
$entries = Get-Content -LiteralPath $Manifest | Where-Object { $_ -match '^[0-9A-Fa-f]{64}  ' } | ForEach-Object { $p=$_ -split '  ',2; [pscustomobject]@{Hash=$p[0];RelPath=$p[1]} }
$bad = $entries | Where-Object { (Get-FileHash -LiteralPath (Join-Path $To $_.RelPath) -Algorithm SHA256).Hash -ne $_.Hash }
if ($bad) { $bad.RelPath | ForEach-Object { "MISMATCH: $_" }; exit 1 }
"undo verified: {0} files restored to _archive_from_instance" -f $entries.Count
