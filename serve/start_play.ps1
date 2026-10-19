<#
RetNet 试玩服务 · 一键启停（后端 + Cloudflare 连接器 + 可选临时隧道 + 看护进程）

  .\serve\start_play.ps1                  # 启动（后台常驻），写状态文件，派生看护进程，打印地址后退出
  .\serve\start_play.ps1 -Attach          # 同上，但本脚本留在前台当“主人”：Ctrl+C / 关终端 → 自动收尾
  .\serve\start_play.ps1 -Status          # 看现在谁在跑（读状态文件 + 实测端口/健康）
  .\serve\start_play.ps1 -Stop            # 收尾：先出汇总，再停后端/连接器/临时隧道/看护，删状态文件
  .\serve\start_play.ps1 -Clean           # 强制清理（状态文件丢了/上次被硬杀时用）；启动时也会自动跑一遍
  .\serve\start_play.ps1 -Foreground      # 前台跑后端（调试用，Ctrl+C 直接停）
  .\serve\start_play.ps1 -QuickTunnel     # 额外起一条 trycloudflare 临时隧道
  .\serve\start_play.ps1 -NoTunnel        # 只起后端
  .\serve\start_play.ps1 -Token "口令"     # 开分享口令

可选：-Port 8077  -Model checkpoints/retnet_500m_v21_export  -Dtype bf16  -RootPath /play  -Workers 1
      -TunnelTokenFile <路径>  -PublicHost <域名>

cloudflared 走 PATH 查找（`Get-Command`），拿不到就退到当前目录里的同名文件。
`-PublicHost` 只用于打印对外地址，留空则不打印（隧道本身照常工作，路由在 Cloudflare 后台配）。

## 意外终止怎么收尾（这是本脚本的重点）

| 场景 | 谁负责收尾 |
|---|---|
| `-Attach` 下按 Ctrl+C | 本脚本 `finally` → 收尾；看护进程也会兜底 |
| 直接关掉终端窗口 | 看护进程发现“启动器 pid 没了” → 收尾（后端/连接器/临时隧道全停） |
| 后端或连接器自己崩了 | 看护进程发现 → 把剩下的停掉（避免“网页 502 但进程还挂着”的半死状态） |
| 机器关机/断电 | 进程随系统一起消失；**残留的状态文件**由下次启动的自动 `-Clean` 收掉 |
| 用任务管理器硬杀 | 同上一行；下次启动自动清理 |

状态文件：`logs\play_state.json`（记录各进程 pid，供 `-Status` / `-Stop` / `-Clean` 精确定位）。
只停“认得出来是自己的”进程：后端要求 pid 的 cmdline 含 `retnet_play_server.py`；
连接器要求含 `--no-autoupdate`；临时隧道要求含 `--url http://127.0.0.1:<port>`。
你那条 `cloudflared tunnel run --token eyJhIjoi…`（服务隧道 other-service-tunnel）不在任何一类里，永远不会被这个脚本碰。
#>
param(
  [int]$Port = 8077,
  [string]$Model = "checkpoints/retnet_500m_v21_export",
  [string]$Dtype = "bf16",
  [string]$RootPath = "/play",
  [string]$Token = "",
  [int]$Workers = 1,
  [string]$TunnelTokenFile = "C:\Tools\secrets\cf-tunnel-token.txt",
  [string]$PublicHost = "",
  [switch]$Stop,
  [switch]$Status,
  [switch]$Clean,
  [switch]$Attach,
  [switch]$Foreground,
  [switch]$QuickTunnel,
  [switch]$NoTunnel
)
$ErrorActionPreference = "Stop"

$root    = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$logDir  = Join-Path $root "logs"
$outLog  = Join-Path $logDir "serve.out"
$errLog  = Join-Path $logDir "serve.err"
$tunOut  = Join-Path $logDir "tunnel-llm.out"
$tunErr  = Join-Path $logDir "tunnel-llm.err"
$qtOut   = Join-Path $logDir "quicktunnel.out"
$qtErr   = Join-Path $logDir "quicktunnel.err"
$supLog  = Join-Path $logDir "supervisor.out"
$stateF  = Join-Path $logDir "play_state.json"
$supPath = Join-Path $root "serve\_play_supervisor.ps1"
$srvPath = Join-Path $root "serve\retnet_play_server.py"

# cloudflared 解析成**绝对路径**。
# ⚠️ 不能只写 "cloudflared" 然后 Test-Path：Test-Path 不查 PATH，只按相对当前目录解析，
#    必然判否（这个坑踩过一次：脚本报"找不到 cloudflared"而它其实就在 C:\Tools 下）。
#    Get-Command 才走 PATH，拿回 .Source 之后后面的 Test-Path 才有意义。
$cfExe = "cloudflared"
$cfCmd = Get-Command $cfExe -ErrorAction SilentlyContinue
if ($cfCmd) { $cfExe = $cfCmd.Source }

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# ------------------------------------------------------------------ 小工具

function Find-Python {
  foreach ($c in @("C:\Tools\Miniconda3\envs\AI\python.exe",
                   "$env:USERPROFILE\miniconda3\envs\AI\python.exe",
                   "$env:USERPROFILE\anaconda3\envs\AI\python.exe")) {
    if (Test-Path $c) { return $c }
  }
  throw "找不到 AI 环境的 python（需要 torch + transformers + fastapi）。"
}
function Get-PortPids([int]$p) {
  (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique
}
function Alive([int]$p) { if ($p -le 0) { return $false }; return [bool](Get-Process -Id $p -ErrorAction SilentlyContinue) }
function Load-State {
  if (-not (Test-Path $stateF)) { return $null }
  try { return (Get-Content $stateF -Raw | ConvertFrom-Json) } catch { return $null }
}
function Save-State($obj) { $obj | ConvertTo-Json | Set-Content -Path $stateF -Encoding UTF8 }
function Remove-State { if (Test-Path $stateF) { Remove-Item $stateF -Force -ErrorAction SilentlyContinue } }

function CmdLine([int]$p) {
  $x = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $p) -ErrorAction SilentlyContinue
  if ($x) { return [string]$x.CommandLine }
  return ""
}
# 只杀“认得出来是自己的”：后端看脚本名，连接器看 --no-autoupdate，临时隧道看 --url 本端口
function Stop-Ours([int]$p, [string]$kind) {
  if ($p -le 0 -or -not (Alive $p)) { return $false }
  $c = CmdLine $p
  $mine = $false
  switch ($kind) {
    'backend'   { $mine = ($c -match 'retnet_play_server\.py') }
    'connector' { $mine = ($c -match 'cloudflared' -and $c -match '--no-autoupdate' -and $c -notmatch '--url') }
    'quick'     { $mine = ($c -match 'cloudflared' -and $c -match ("--url\s+http://127\.0\.0\.1:{0}" -f $Port)) }
    'supervisor'{ $mine = ($c -match '_play_supervisor\.ps1') }
    default     { $mine = $false }
  }
  if (-not $mine) { Write-Host ("  跳过 pid {0}（cmdline 不像本项目的 {1}）" -f $p, $kind); return $false }
  try { Stop-Process -Id $p -Force -ErrorAction Stop; Write-Host ("  停掉 {0} pid {1}" -f $kind, $p); return $true }
  catch { Write-Host ("  停 {0} pid {1} 失败：{2}" -f $kind, $p, $_.Exception.Message); return $false }
}
function Get-Connectors {
  Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match '--no-autoupdate' -and $_.CommandLine -notmatch '--url' }
}
function Get-QuickTunnels([int]$p) {
  Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match '--url' -and $_.CommandLine -match ("127\.0\.0\.1:{0}" -f $p) }
}
function Get-Health {
  try { return (Invoke-RestMethod -Uri ("http://127.0.0.1:{0}{1}/api/health" -f $Port, $RootPath) -TimeoutSec 4) }
  catch { return $null }
}

# 清理残留：状态文件里的 pid + 端口占用者 + 任何本项目的连接器/临时隧道
function Invoke-Clean([switch]$Quiet) {
  $st = Load-State
  if ($st) {
    if (-not $Quiet) { Write-Host "  按状态文件清理（{0}）" -f $st.startedAt }
    Stop-Ours ([int]$st.supervisor) 'supervisor' | Out-Null
    Stop-Ours ([int]$st.backend)    'backend'    | Out-Null
    Stop-Ours ([int]$st.connector)  'connector'  | Out-Null
    Stop-Ours ([int]$st.quickTunnel)'quick'      | Out-Null
  }
  foreach ($p in (Get-PortPids $Port)) { Stop-Ours $p 'backend' | Out-Null }
  foreach ($x in (Get-Connectors)) { Stop-Ours ([int]$x.ProcessId) 'connector' | Out-Null }
  foreach ($x in (Get-QuickTunnels $Port)) { Stop-Ours ([int]$x.ProcessId) 'quick' | Out-Null }
  Remove-State
}
function Read-TunnelToken([string]$path) {
  if (-not (Test-Path $path)) { return $null }
  $t = ((Get-Content $path -Raw) -replace '\s','').Trim()
  if ($t.Length -lt 40) { return $null }
  $b64 = $t.Replace('-','+').Replace('_','/'); while ($b64.Length % 4 -ne 0) { $b64 += '=' }
  try {
    $o = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($b64)) | ConvertFrom-Json
    if (-not $o.t) { return $null }
    return [pscustomobject]@{ Token = $t; TunnelId = $o.t }
  } catch { return $null }
}
function Write-Summary([string]$title) {
  $h = Get-Health
  Write-Host ""
  Write-Host ("== {0} ==" -f $title) -ForegroundColor Cyan
  if ($h) {
    Write-Host ("  已服务 {0} 次 · 拒绝 {1} · 取消 {2} · 最近 {3} tok/s · 运行 {4}s" -f `
      $h.served, $h.rejected, $h.cancelled, $h.last_tok_per_s, $h.uptime_s)
    Write-Host ("  模型 {0}（{1:N1}M，{2}/{3}）" -f $h.model, ($h.params/1e6), $h.device, $h.dtype)
  } else {
    Write-Host "  （后端当前无响应，拿不到统计）"
  }
}

# ------------------------------------------------------------------ -Status

if ($Status) {
  Write-Host "== 状态 =="
  $st = Load-State
  if ($st) {
    Write-Host ("  状态文件 {0}（启动于 {1}）" -f $stateF, $st.startedAt)
    foreach ($k in @('backend','connector','quickTunnel','supervisor','launcher')) {
      $v = [int]$st.$k
      $mark = '—'
      if ($v -gt 0) { if (Alive $v) { $mark = 'alive' } else { $mark = 'DEAD' } }
      Write-Host ("    {0,-11} {1,-7} {2}" -f $k, $v, $mark)
    }
    $ph = if ($st.publicHost) { "公网 https://{0}{1}/" -f $st.publicHost, $st.rootPath } else { "公网 （未设 -PublicHost）" }
    Write-Host ("  端口 {0} / 路径 {1} / {2}" -f $st.port, $st.rootPath, $ph)
  } else {
    Write-Host "  没有状态文件（服务可能没起，或上次被硬杀后没清理）"
  }
  $lp = Get-PortPids $Port
  Write-Host ("  端口 {0} 监听: {1}" -f $Port, $(if ($lp) { ($lp -join ',') } else { '无' }))
  Write-Host ("  本项目连接器: {0}" -f (@(Get-Connectors).Count))
  Write-Host ("  临时隧道: {0}" -f (@(Get-QuickTunnels $Port).Count))
  $h = Get-Health
  Write-Host ("  健康检查: {0}" -f $(if ($h) { 'ok' } else { '无响应' }))
  return
}

# ------------------------------------------------------------------ -Stop

if ($Stop) {
  Write-Summary "收尾汇总"
  Write-Host ""
  Write-Host "开始收尾："
  $st = Load-State
  if ($st) {
    # 先停看护，免得它在我们收尾过程中又插一脚
    Stop-Ours ([int]$st.supervisor) 'supervisor' | Out-Null
  }
  Invoke-Clean -Quiet
  Write-Host "  已停止：后端 / 连接器 / 临时隧道 / 看护进程；状态文件已删"
  Write-Host ("  你的服务隧道（other-service-tunnel）与本脚本无关，未触碰。")
  return
}

# ------------------------------------------------------------------ -Clean

if ($Clean -and -not $Attach -and -not $Foreground) {
  Write-Host "强制清理残留："
  Invoke-Clean
  Write-Host "  完成"
  return
}

# ------------------------------------------------------------------ 启动

if (-not (Test-Path (Join-Path $root $Model))) { throw ("找不到模型目录：{0}（相对仓库根）" -f $Model) }
if (-not (Test-Path $srvPath)) { throw ("找不到后端脚本：{0}" -f $srvPath) }
$py = Find-Python

Write-Host "先清理上次残留（含被硬杀/关机后留下的）："
Invoke-Clean -Quiet
if (Get-PortPids $Port) { Write-Host ("  端口 {0} 仍被占用，再等 2 秒"); Start-Sleep -Seconds 2 }

$srvArgs = @($srvPath, "--model_dir", $Model, "--port", "$Port",
             "--dtype", $Dtype, "--root_path", $RootPath, "--workers", "$Workers")
if ($Token) { $srvArgs += @("--token", $Token) }

if ($Foreground) {
  Write-Host "前台运行后端（Ctrl+C 停止；此模式不启隧道）…"
  try { & $py @srvArgs } finally { Write-Host "前台后端已退出" }
  return
}

# ---- 后端 ----
Write-Host ("启动后端：{0}" -f $py)
$backend = Start-Process -FilePath $py -ArgumentList $srvArgs -WorkingDirectory $root `
  -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -WindowStyle Hidden

$health = $null
for ($i = 0; $i -lt 60; $i++) {
  Start-Sleep -Seconds 2
  if (-not (Alive $backend.Id)) { break }
  $health = Get-Health
  if ($health) { break }
}
if (-not $health) {
  Write-Host "❌ 后端没起来，日志尾部：" -ForegroundColor Red
  Get-Content $errLog -Tail 15 -ErrorAction SilentlyContinue
  Stop-Ours $backend.Id 'backend' | Out-Null
  return
}
Write-Host ""
Write-Host "✅ 后端就绪" -ForegroundColor Green
Write-Host ("   模型 {0}（{1:N1}M 参数，{2}/{3}）" -f $health.model, ($health.params/1e6), $health.device, $health.dtype)
Write-Host ("   本机 http://127.0.0.1:{0}{1}/" -f $Port, $RootPath)

# ---- 连接器 ----
$connectorId = 0
if (-not $NoTunnel) {
  $exist = @(Get-Connectors)
  if ($exist.Count -gt 0) {
    $connectorId = [int]$exist[0].ProcessId
    Write-Host ("   连接器已在运行 pid {0}，复用" -f $connectorId)
  } elseif (-not (Test-Path $cfExe)) {
    Write-Host ("⚠️ 找不到 {0}，跳过连接器" -f $cfExe) -ForegroundColor Yellow
  } else {
    $info = Read-TunnelToken $TunnelTokenFile
    if (-not $info) {
      Write-Host ("⚠️ 读不到有效隧道 token（{0}），跳过连接器" -f $TunnelTokenFile) -ForegroundColor Yellow
    } else {
      $env:TUNNEL_TOKEN = $info.Token                     # 走环境变量，不进命令行/日志
      $connector = Start-Process -FilePath $cfExe -ArgumentList @("tunnel","--no-autoupdate","run") `
        -WorkingDirectory $root -RedirectStandardOutput $tunOut -RedirectStandardError $tunErr -PassThru -WindowStyle Hidden
      Remove-Item Env:TUNNEL_TOKEN -ErrorAction SilentlyContinue
      $connectorId = $connector.Id
      $okc = $false
      for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 2
        if (Select-String -Path $tunOut, $tunErr -Pattern 'Registered tunnel connection' -ErrorAction SilentlyContinue) { $okc = $true; break }
      }
      Write-Host ("   连接器 pid {0}（隧道 id {1}）{2}" -f $connectorId, $info.TunnelId, $(if ($okc) { '已注册' } else { '⚠️ 未见注册成功' }))
      if ($okc) {
        Write-Host ""
        if ($PublicHost) {
          Write-Host ("🌐 公网地址：https://{0}{1}/" -f $PublicHost, $RootPath) -ForegroundColor Cyan
          if ($Token) { Write-Host ("   （带口令： https://{0}{1}/?t={2}）" -f $PublicHost, $RootPath, $Token) }
        } else {
          Write-Host "🌐 连接器已注册；对外地址由 Cloudflare 后台的 Public Hostname 决定（本脚本未设 -PublicHost）" -ForegroundColor Cyan
        }
      }
    }
  }
}

# ---- 临时隧道（可选）----
$quickId = 0
if ($QuickTunnel -and (Test-Path $cfExe)) {
  $qt = Start-Process -FilePath $cfExe -ArgumentList @("tunnel","--url","http://127.0.0.1:$Port","--no-autoupdate") `
    -WorkingDirectory $root -RedirectStandardOutput $qtOut -RedirectStandardError $qtErr -PassThru -WindowStyle Hidden
  $quickId = $qt.Id
  $url = $null
  for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    $hit = Select-String -Path $qtOut, $qtErr -Pattern "https://[a-z0-9\-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { $url = ([regex]::Match($hit.Line, "https://[a-z0-9\-]+\.trycloudflare\.com")).Value; break }
  }
  if ($url) {
    $link = "$url$RootPath/"
    if ($Token) { $link += "?t=$Token" }
    Write-Host ""
    Write-Host ("🌐 临时分享：{0}" -f $link) -ForegroundColor Cyan
  } else { Write-Host "⚠️ 临时隧道没解析出地址，看 logs/quicktunnel.out|err" -ForegroundColor Yellow }
}

# ---- 状态文件 + 看护进程 ----
$supArgs = @("-NoProfile","-ExecutionPolicy","Bypass","-File",$supPath,
             "-BackendPid","$($backend.Id)","-ConnectorPid","$connectorId","-QuickTunnelPid","$quickId",
             "-StateFile",$stateF,"-LogFile",$supLog)
if ($Attach) { $supArgs += @("-LauncherPid","$PID") }      # 盯住本终端：关掉它就收尾
$sup = Start-Process -FilePath "pwsh" -ArgumentList $supArgs -WorkingDirectory $root -PassThru -WindowStyle Hidden

Save-State ([pscustomobject]@{
  startedAt   = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  port        = $Port
  rootPath    = $RootPath
  publicHost  = $PublicHost
  model       = $Model
  backend     = $backend.Id
  connector   = $connectorId
  quickTunnel = $quickId
  supervisor  = $sup.Id
  launcher    = $(if ($Attach) { $PID } else { 0 })
})

Write-Host ""
Write-Host ("看护进程 pid {0}（日志 logs/supervisor.out）" -f $sup.Id)

if (-not $Attach) {
  Write-Host "已在后台常驻。查看/停止： .\serve\start_play.ps1 -Status | -Stop"
  return
}

# ---- Attach：本脚本留在前台当主人 ----
Write-Host "前台看守中（Ctrl+C 或关闭本窗口 → 自动收尾）…" -ForegroundColor Yellow
try {
  while ($true) {
    Start-Sleep -Seconds 5
    if (-not (Alive $backend.Id)) { Write-Host "后端已退出，收尾"; break }
    if ($connectorId -gt 0 -and -not (Alive $connectorId)) { Write-Host "连接器已退出，收尾"; break }
  }
} finally {
  Write-Host ""
  Write-Summary "收尾汇总"
  Stop-Ours $sup.Id 'supervisor' | Out-Null
  Stop-Ours $backend.Id 'backend' | Out-Null
  Stop-Ours $connectorId 'connector' | Out-Null
  Stop-Ours $quickId 'quick' | Out-Null
  Remove-State
  Write-Host "已收尾（后端 / 连接器 / 临时隧道 / 看护进程）"
}
