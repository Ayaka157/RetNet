# 把崩溃证据从系统目录收集到工作区（**需要以管理员身份运行**）。
#
#   pwsh -NoProfile -File scripts\collect_crash_dumps.ps1
#
# 只做「复制出来」，不改动系统目录、不改 ACL、不删除任何东西。
# 目标：E:\Workspaces\RetNet\logs\crash\
#
# 背景（2026-10-11）：桌面在 07:18 前后蓝屏一次（BugCheck 0x3b，有 minidump），
# 08:11 前后整机冻结一次（只能强制重启，**没有 dump**）。
# 另外 10/10 傍晚有两次 VIDEO_ENGINE_TIMEOUT_DETECTED（0x141）的 WER 报告。
# minidump 与 WER 报告在系统目录下，普通权限读不了，所以需要这一步。

$ErrorActionPreference = 'Continue'
$dst = Join-Path (Split-Path $PSScriptRoot -Parent) 'logs\crash'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Write-Host "收集到: $dst"

# 前置检查：是否管理员
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
  ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "!! 当前不是管理员，Minidump 与 WER 都读不到。请用管理员身份重跑。" -ForegroundColor Red
    Write-Host "   右键 PowerShell -> 以管理员身份运行，再执行本脚本。" -ForegroundColor Red
}

# ---------- 1. minidump ----------
Write-Host "`n=== 1. Minidump ==="
$md = 'C:\Windows\Minidump'
if (Test-Path $md) {
    Get-ChildItem $md -Filter *.dmp -ErrorAction SilentlyContinue | ForEach-Object {
        $target = Join-Path $dst $_.Name
        Copy-Item $_.FullName $target -Force -ErrorAction SilentlyContinue
        if (Test-Path $target) {
            Write-Host ("  ✅ {0}  ({1:N0} 字节, 崩溃于 {2})" -f $_.Name, $_.Length, $_.LastWriteTime)
        } else {
            Write-Host ("  ❌ 复制失败: {0}" -f $_.Name) -ForegroundColor Red
        }
    }
} else { Write-Host "  （没有 C:\Windows\Minidump）" }

# ---------- 2. 完整转储（若存在，可能很大）----------
Write-Host "`n=== 2. MEMORY.DMP ==="
if (Test-Path 'C:\Windows\MEMORY.DMP') {
    $sz = (Get-Item 'C:\Windows\MEMORY.DMP').Length
    if ($sz -gt 2GB) {
        Write-Host ("  存在，但 {0:N1} GB，跳过（如需要请手动复制）" -f ($sz / 1GB)) -ForegroundColor Yellow
    } else {
        Copy-Item 'C:\Windows\MEMORY.DMP' (Join-Path $dst 'MEMORY.DMP') -Force
        Write-Host ("  ✅ 已复制 ({0:N0} 字节)" -f $sz)
    }
} else { Write-Host "  （不存在——所以冻结那次没有转储）" }

# ---------- 3. WER Kernel_* 报告（**最有价值**：Report.wer 里常直接写出错模块）----------
Write-Host "`n=== 3. WER 报告 ==="
foreach ($root in @('C:\ProgramData\Microsoft\Windows\WER\ReportArchive',
                    'C:\ProgramData\Microsoft\Windows\WER\ReportQueue')) {
    if (-not (Test-Path $root)) { continue }
    $dirs = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^Kernel_|BlueScreen|LiveKernelEvent' }
    foreach ($d in $dirs) {
        $tag = ($d.Name -replace '[^\w\-]', '_')
        $out = Join-Path $dst $tag
        New-Item -ItemType Directory -Force -Path $out | Out-Null
        # 只要文本报告 → 体积小、可读；.cab/.dmp 太大就别拽了
        Copy-Item (Join-Path $d.FullName 'Report.wer') $out -Force -ErrorAction SilentlyContinue
        Get-ChildItem $d.FullName -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -in '.txt', '.xml', '.wer' } |
            ForEach-Object { Copy-Item $_.FullName $out -Force -ErrorAction SilentlyContinue }
        $n = (Get-ChildItem $out -File -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host ("  {0}  ->  {1} 个文件  ({2})" -f $d.Name.Substring(0, [Math]::Min(48, $d.Name.Length)), $n, $d.LastWriteTime)
    }
}

# ---------- 4. LiveKernelReports（显卡 TDR 转储通常在这里）----------
Write-Host "`n=== 4. LiveKernelReports ==="
$lk = 'C:\Windows\LiveKernelReports'
$any = Get-ChildItem $lk -Recurse -File -ErrorAction SilentlyContinue
if ($any) { $any | ForEach-Object { Write-Host "  $($_.FullName)  ($($_.Length) 字节)" } }
else { Write-Host "  （空——所以 08:11 那次冻结没留下看门狗转储）" }

# ---------- 5. 事件日志导出（文本，便于我直接读）----------
Write-Host "`n=== 5. 导出事件日志 ==="
try {
    Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddDays(-3)} -ErrorAction SilentlyContinue |
        Where-Object { $_.ProviderName -in 'nvlddmkm','Microsoft-Windows-WER-SystemErrorReporting','Microsoft-Windows-Kernel-Power','EventLog','Microsoft-Windows-Kernel-General','volmgr','Application Popup' } |
        Select-Object TimeCreated, Id, ProviderName, LevelDisplayName, Message |
        Export-Csv (Join-Path $dst 'system_events.csv') -NoTypeInformation -Encoding UTF8
    Write-Host "  ✅ system_events.csv"
} catch { Write-Host ("  ❌ 导出失败: {0}" -f $_.Exception.Message) -ForegroundColor Red }

Write-Host "`n=== 汇总 ==="
Get-ChildItem $dst -Recurse | Select-Object @{n='rel';e={$_.FullName.Replace($dst,'')}},Length |
  Format-Table -AutoSize
Write-Host "把整个 logs\crash 目录留给我即可（或在对话里告诉我已就绪）。"
