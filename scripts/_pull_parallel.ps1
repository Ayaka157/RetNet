<#
并行分块拉取大文件 —— 为高延迟链路准备。

## 为什么要它

实测这条链路（本地 ↔ AutoDL 内蒙B区，RTT **261 ms**）：
  sftp/scp        77 KB/s
  单路 ssh 管道   117~276 KB/s（波动大）
  4 路并行        700 KB/s（聚合）
TCP 单流的窗口/丢包在高 RTT 下严重限制吞吐，多路并行能直接叠加。所以大文件切成 N 块
并行拉，再拼接，最后**用 sha256 校验**（并行分块必须校验，否则错了也不知道）。

## 用法

  pwsh -File scripts/_pull_parallel.ps1 -Remote /root/gpu-scratch/x/model.safetensors `
       -Local checkpoints/x/model.safetensors -Streams 8

## 注意

- `copy /b` 是字节安全的（PowerShell 的 `>` 会把二进制当文本，**不能用**）。
- 最后一块不设 `count`，直接读到 EOF，所以总大小不必被块大小整除。
- 校验失败时**保留分块**便于排查，不会静默成功。
#>
param(
  [Parameter(Mandatory = $true)][string]$Remote,
  [Parameter(Mandatory = $true)][string]$Local,
  [int]$Streams = 8,
  [string]$Expect = "",
  [string]$HostAlias = "gpu-host"
)
$ErrorActionPreference = "Stop"

$dir = Split-Path -Parent $Local
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

$sizeStr = (& ssh -o BatchMode=yes $HostAlias "stat -c %s '$Remote'").Trim()
$size = [int64]$sizeStr
if ($size -le 0) { throw "拿不到远端文件大小: $Remote" }
Write-Host ("远端 {0}" -f $Remote)
Write-Host ("大小 {0:N0} 字节 = {1:N1} MB；{2} 路并行" -f $size, ($size / 1MB), $Streams)

$chunkBytes = [int64][math]::Ceiling($size / $Streams / 1MB) * 1MB
$parts = @()
$jobs = @()
$sw = [Diagnostics.Stopwatch]::StartNew()
for ($i = 0; $i -lt $Streams; $i++) {
  $skipMB = [int]($i * $chunkBytes / 1MB)
  if ($skipMB * 1MB -ge $size) { break }
  $part = "$Local.part$i"
  $parts += $part
  if (Test-Path $part) { Remove-Item $part -Force }
  $end = ($i + 1) * $chunkBytes
  $countArg = ""
  if ($end -lt $size) { $countArg = " count=$([int]($chunkBytes / 1MB))" }
  $remoteCmd = "dd if='$Remote' bs=1M skip=$skipMB$countArg 2>/dev/null"
  $arg = "/c ssh -o BatchMode=yes -o ServerAliveInterval=30 $HostAlias `"$remoteCmd`" > `"$part`""
  $jobs += Start-Process -FilePath cmd.exe -ArgumentList $arg -PassThru -WindowStyle Hidden
}
$jobs | Wait-Process
$sw.Stop()
$got = ($parts | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum
Write-Host ("分块下载 {0:N0} 字节 / {1:N1} s = {2:N0} KB/s" -f $got, $sw.Elapsed.TotalSeconds, ($got / $sw.Elapsed.TotalSeconds / 1KB))

if (Test-Path $Local) { Remove-Item $Local -Force }
$list = ($parts | ForEach-Object { "`"$_`"" }) -join "+"
cmd /c "copy /b $list `"$Local`"" | Out-Null

$localSize = (Get-Item $Local).Length
$ok = ($localSize -eq $size)
Write-Host ("拼接后 {0:N0} 字节（远端 {1:N0}）→ {2}" -f $localSize, $size, $(if ($ok) { "大小一致" } else { "大小不一致" }))

if ($Expect) {
  $h = (Get-FileHash -Algorithm SHA256 -Path $Local).Hash.ToLower()
  if ($h -eq $Expect.ToLower()) { Write-Host "sha256 校验通过 $h" }
  else { Write-Host "sha256 校验失败：实得 $h / 期望 $($Expect.ToLower())"; $ok = $false }
}

if ($ok) { $parts | ForEach-Object { Remove-Item $_ -Force -ErrorAction SilentlyContinue } }
else { Write-Host "保留分块以便排查：$($parts -join ', ')" }
if (-not $ok) { exit 1 }
