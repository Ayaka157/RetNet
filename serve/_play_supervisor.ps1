<#
试玩服务的看护进程（由 start_play.ps1 派生，不要直接手跑）

职责：盯住三件事，一旦成立就**收尾清理**并把状态文件删掉——
  1) 启动器（终端 / 本会话）已经没了 → 说明用户按了 Ctrl+C、关了终端、或机器在关机
  2) 后端（uvicorn）意外退出
  3) 连接器（cloudflared）意外退出
任何一种情况都意味着"现在是个半死不活的状态"，看护进程会把剩下的都收掉，
这样下次启动不会踩到孤儿进程，也不会出现"网页 502 但进程还在"的迷惑现场。

参数：
  -BackendPid     后端 pid（必填）
  -ConnectorPid   连接器 pid（0 = 没有）
  -QuickTunnelPid 临时隧道 pid（0 = 没有）
  -LauncherPid    启动器 pid（0 = 不盯启动器，即 detached 模式）
  -StateFile      状态文件路径（收尾时删掉）
  -LogFile        看护日志
#>
param(
  [Parameter(Mandatory = $true)][int]$BackendPid,
  [int]$ConnectorPid = 0,
  [int]$QuickTunnelPid = 0,
  [int]$LauncherPid = 0,
  [Parameter(Mandatory = $true)][string]$StateFile,
  [Parameter(Mandatory = $true)][string]$LogFile
)

function Log([string]$m) {
  $line = '[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $m
  Add-Content -Path $LogFile -Value $line -Encoding UTF8 -ErrorAction SilentlyContinue
}
function Alive([int]$p) {
  if ($p -le 0) { return $false }
  return [bool](Get-Process -Id $p -ErrorAction SilentlyContinue)
}
function KillIt([int]$p, [string]$name) {
  if ($p -gt 0 -and (Alive $p)) {
    try { Stop-Process -Id $p -Force -ErrorAction Stop; Log ("收尾：停掉 {0} pid {1}" -f $name, $p) }
    catch { Log ("收尾：停 {0} pid {1} 失败（{2}）" -f $name, $p, $_.Exception.Message) }
  }
}

Log ("看护启动：backend={0} connector={1} quick={2} launcher={3}" -f $BackendPid, $ConnectorPid, $QuickTunnelPid, $LauncherPid)

$reason = ''
while ($true) {
  Start-Sleep -Seconds 3
  if ($LauncherPid -gt 0 -and -not (Alive $LauncherPid)) { $reason = '启动器/终端已退出'; break }
  if (-not (Alive $BackendPid)) { $reason = '后端意外退出'; break }
  if ($ConnectorPid -gt 0 -and -not (Alive $ConnectorPid)) { $reason = '连接器意外退出'; break }
}

Log ("触发收尾：{0}" -f $reason)
KillIt $BackendPid '后端'
KillIt $ConnectorPid '连接器'
KillIt $QuickTunnelPid '临时隧道'
if (Test-Path $StateFile) { Remove-Item $StateFile -Force -ErrorAction SilentlyContinue }
Log '收尾完成（状态文件已删）'
