<#
并行分块上传大文件 —— `_pull_parallel.ps1` 的反方向。

## 为什么要它

下载方向实测（本地 ↔ AutoDL 内蒙B，RTT 261 ms）：
  sftp/scp        77 KB/s
  单路 ssh 管道   117~276 KB/s
  4 路并行        700 KB/s（聚合）
  8 路并行        526 KB/s（实测，拉 model.safetensors 得到）
单条 TCP 在高 RTT 下被窗口/丢包卡住，多路并行能直接叠加。上传方向单路约 1.5 MB/s，
按同样思路切块并行，往往还能再快几倍。

## 用法

  pwsh -File scripts/_push_parallel.ps1 -Local <本地文件> -Remote <远端绝对路径> -Streams 8

## 关键点

- 分块名用 `part00..partNN` **零填充**，远端 `cat part*` 的字典序才等于数字序。
- 上传完**必须校验**：本地 sha256 vs 远端 sha256，不一致就报错并保留分块。
- 只传文件，不传目录（目录请先本地打包）。
#>
param(
  [Parameter(Mandatory = $true)][string]$Local,
  [Parameter(Mandatory = $true)][string]$Remote,
  [int]$Streams = 8,
  [string]$Expect = "",
  [string]$HostAlias = "gpu-host-a"
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Local)) { throw "本地文件不存在: $Local" }
$size = (Get-Item $Local).Length
Write-Host ("本地 {0} = {1:N0} 字节 = {2:N1} MB；{3} 路并行" -f $Local, $size, ($size / 1MB), $Streams)

$remoteDir = ($Remote -replace '/[^/]+$', '')
& ssh -o BatchMode=yes $HostAlias "mkdir -p '$remoteDir' /root/_push" | Out-Null

$chunk = [int64][math]::Ceiling($size / $Streams / 1MB) * 1MB
$parts = @()
$src = [IO.File]::OpenRead($Local)
try {
  for ($i = 0; $i -lt $Streams; $i++) {
    $off = [int64]($i * $chunk)
    if ($off -ge $size) { break }
    $len = [Math]::Min($chunk, $size - $off)
    $part = Join-Path ([IO.Path]::GetTempPath()) ("push_{0}_{1:d2}.part" -f ([IO.Path]::GetFileName($Local)), $i)
    $parts += $part
    $dst = [IO.File]::Create($part)
    try {
      $src.Position = $off
      $buf = New-Object byte[] (4MB)
      $left = $len
      while ($left -gt 0) {
        $n = $src.Read($buf, 0, [int][Math]::Min($buf.Length, $left))
        if ($n -le 0) { break }
        $dst.Write($buf, 0, $n)
        $left -= $n
      }
    } finally { $dst.Dispose() }
  }
} finally { $src.Dispose() }
Write-Host ("切成 {0} 块，最大 {1:N1} MB" -f $parts.Count, ($chunk / 1MB))

$sw = [Diagnostics.Stopwatch]::StartNew()
$jobs = @()
for ($i = 0; $i -lt $parts.Count; $i++) {
  $name = "part{0:d2}" -f $i
  $jobs += Start-Process -FilePath "scp" `
    -ArgumentList @("-q", "-o", "BatchMode=yes", $parts[$i], "${HostAlias}:/root/_push/$name") `
    -PassThru -WindowStyle Hidden
}
$jobs | Wait-Process
$sw.Stop()
$bad = ($jobs | Where-Object { $_.ExitCode -ne 0 }).Count
if ($bad -gt 0) { throw "有 $bad 个分块上传失败" }
Write-Host ("分块上传完成 {0:N1} s = {1:N0} KB/s（聚合）" -f $sw.Elapsed.TotalSeconds, ($size / $sw.Elapsed.TotalSeconds / 1KB))

& ssh -o BatchMode=yes $HostAlias "cat /root/_push/part* > '$Remote' && rm -f /root/_push/part* && stat -c %s '$Remote'" | ForEach-Object { Write-Host "远端拼接后 $_ 字节（本地 $size）" }

$localHash = (Get-FileHash -Algorithm SHA256 -Path $Local).Hash.ToLower()
$remoteHash = (& ssh -o BatchMode=yes $HostAlias "sha256sum '$Remote'").Split()[0].ToLower()
Write-Host "本地 sha256 $localHash"
Write-Host "远端 sha256 $remoteHash"
if ($localHash -ne $remoteHash) { throw "sha256 不一致 —— 上传损坏，分块保留在 /root/_push 供排查" }
if ($Expect -and $localHash -ne $Expect.ToLower()) { throw "与期望值不符（期望 $($Expect.ToLower())）" }
Write-Host "✅ sha256 一致"

$parts | ForEach-Object { Remove-Item $_ -Force -ErrorAction SilentlyContinue }
