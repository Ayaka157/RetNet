# 验证 _run_scale_probe.ps1 新增的两条加固机制（不碰 GPU）。
#
# ⚠️ 第一版这个测试**自己有 bug**：用 `-ArgumentList @('-c','代码里有空格…')` 时
#    Start-Process 会把数组元素用空格拼起来而**不加引号**，python 立刻报错退出，
#    于是 (A) 读回 0 行、(B) 进程秒退导致超时路径**根本没被测到**。
#    改成把代码写进 .py 文件、只传路径，彻底绕开引号问题。
$ErrorActionPreference = 'Continue'
$py  = 'C:\Tools\Miniconda3\envs\AI\python.exe'
$tmp = "$env:TEMP\probe_driver_test"
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$script:fails = 0
function Check($cond, $msg) {
    if ($cond) { Write-Host "  ✅ $msg" } else { Write-Host "  ❌ $msg"; $script:fails++ }
}

# --- 与驱动里逐字相同的解析器 ---
function Parse-Result {
    param([string[]]$Lines)
    $r = [ordered]@{}
    foreach ($ln in $Lines) {
        if (-not $ln) { continue }
        if ($ln -match '(\d+)\s+ms/(\d+)\s+token')  { $r.opt_ms = $Matches[1]; $r.tok_step = $Matches[2] }
        if ($ln -match '(\d+)\s+tok/s')             { if (-not $r.tok_s) { $r.tok_s = $Matches[1] } }
        if ($ln -match 'out of memory|OutOfMemory|CUDA error') { $r.oom = 'YES' }
    }
    return $r
}

# 把脚本写成文件，args 里就只剩路径（和真实驱动一样：无空格）
$okPy = "$tmp\ok.py"
[IO.File]::WriteAllText($okPy, "print('  opt step        999 ms/8192 token')`nprint('  throughput  12345 tok/s')`n")
$slowPy = "$tmp\slow.py"
[IO.File]::WriteAllText($slowPy, "import time`nprint('start', flush=True)`ntime.sleep(30)`nprint('never')`n")
$oomPy = "$tmp\oom.py"
[IO.File]::WriteAllText($oomPy, "import sys`nprint('CUDA out of memory!')`nsys.exit(1)`n")

Write-Host "=== A. 正常命令：捕获 + 解析 ==="
$tmpOut = "$tmp\a.log"
$p = Start-Process -FilePath $py -ArgumentList @($okPy) -NoNewWindow -PassThru `
        -RedirectStandardOutput $tmpOut -RedirectStandardError "$tmp\a.err"
$p | Wait-Process -Timeout 60 -ErrorAction Stop
$o = @(); if (Test-Path $tmpOut) { $o += Get-Content $tmpOut -Encoding utf8 }
$r = Parse-Result $o
Check ($o.Count -ge 2) "重定向文件读回 $($o.Count) 行"
Check ($r.opt_ms -eq '999') "解析到 opt_ms=999（实际 '$($r.opt_ms)'）"
Check ($r.tok_s -eq '12345') "解析到 tok_s=12345（实际 '$($r.tok_s)'）"

Write-Host "=== B. 超时：Wait-Process -Timeout 必须抛，且进程被杀 ==="
$p2 = Start-Process -FilePath $py -ArgumentList @($slowPy) -NoNewWindow -PassThru `
         -RedirectStandardOutput "$tmp\b.log" -RedirectStandardError "$tmp\b.err"
$timedOut = $false
$sw = [Diagnostics.Stopwatch]::StartNew()
try { $p2 | Wait-Process -Timeout 5 -ErrorAction Stop }
catch { $timedOut = $true; Stop-Process -Id $p2.Id -Force -ErrorAction SilentlyContinue }
$sw.Stop()
$el = [math]::Round($sw.Elapsed.TotalSeconds,1)
Check $timedOut "超时抛异常被 catch 捕获（实际等了 ${el}s，期望 ≈5s）"
Check ($sw.Elapsed.TotalSeconds -ge 4) "确实等满了超时窗口（${el}s >= 4s）"
Start-Sleep -Seconds 2
Check (-not [bool](Get-Process -Id $p2.Id -ErrorAction SilentlyContinue)) "超时后子进程已终止"
# 被杀掉的进程必须**没有**产出可解析结果 → 才能靠这一点判断"不该记 completed"
$ob = @(); if (Test-Path "$tmp\b.log") { $ob += Get-Content "$tmp\b.log" -Encoding utf8 }
Check (-not (Parse-Result $ob).opt_ms) "被杀的进程解析不出 opt_ms（不会误记 completed）"

Write-Host "=== C. OOM 也能被识别（应记 completed，因为重试也不会有结果）==="
$p3 = Start-Process -FilePath $py -ArgumentList @($oomPy) -NoNewWindow -PassThru `
         -RedirectStandardOutput "$tmp\c.log" -RedirectStandardError "$tmp\c.err"
$p3 | Wait-Process -Timeout 30 -ErrorAction SilentlyContinue
$oc = @(); if (Test-Path "$tmp\c.log") { $oc += Get-Content "$tmp\c.log" -Encoding utf8 }
if (Test-Path "$tmp\c.err") { $oc += Get-Content "$tmp\c.err" -Encoding utf8 }
Check ((Parse-Result $oc).oom -eq 'YES') "OOM 被解析出来"

Write-Host "=== D. completed.txt 跳过逻辑 ==="
$cf = "$tmp\completed.txt"
foreach ($l in @('E1a','E1b','E3a')) { Add-Content -Path $cf -Value $l -Encoding utf8 }
$done = Get-Content $cf -Encoding utf8
Check ($done -contains 'E1b') "E1b 判为已完成 → 会跳过"
Check (-not ($done -contains 'E3b')) "E3b 未完成 → 会重跑"

Write-Host ""
if ($script:fails -eq 0) { Write-Host "✅ 全部验证通过" } else { Write-Host "❌ $($script:fails) 项失败" }
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
exit $script:fails
