# 等满预算基准跑完后，自动对**每一份** .jsonl 出「零分类裁决表」。
#
#   启动：pwsh -NoProfile -File scripts\_run_zero_analysis.ps1
#   输出：logs\bench_v3_full\zero_verdict.txt
#
# 为什么单独挂一个：`docs/23` §8.4 的决策依赖那张表，而
# **数学 / 编码 / 长上下文** 三类（`docs/17` 判定真正容量相关的那几类）
# 在基准里是**最后**才跑的（数据集列序如此）。等它们出来时驱动已经在抢 GPU 了。
#
# 安全性：
#   * 本脚本**只读文件、只用 CPU**，不碰 GPU，所以不会与 scale_probe 驱动竞争
#   * 不杀任何进程、不写任何被别的作业写的文件
$ErrorActionPreference = 'Continue'
$py   = 'C:\Tools\Miniconda3\envs\AI\python.exe'
$root = 'E:\Workspaces\RetNet'
$out  = "$root\logs\bench_v3_full"
$verdict = "$out\zero_verdict.txt"
Set-Location $root

# ⚠️ **必须设这两个**。第一次跑漏了，结果整份裁决表没落盘：
#    python 打印 ⚠ / ❌ 等非 GBK 字符，pwsh 用控制台默认编码（GBK）解码子进程输出 →
#    `UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0'`，
#    `$txt` 抓不到「已解析 N 条」→ 正则匹配空数组 → 只写了表头就结束。
#    驱动脚本里有这两行，裁决链当时忘了抄。
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

function Say($m) {
    $line = ('[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $m)
    Write-Host $line
    Add-Content -Path $verdict -Value $line -Encoding utf8
}

"script start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Set-Content -Path $verdict -Encoding utf8

$pidFile = "$out\.pid"
$benchPid = if (Test-Path $pidFile) { [int](Get-Content $pidFile -Raw).Trim() } else { -1 }
Say "等待基准结束：PID=$benchPid，最长 12 小时"

$deadline = (Get-Date).AddHours(12)
while ((Get-Date) -lt $deadline) {
    $alive = $false
    if ($benchPid -gt 0) { $alive = [bool](Get-Process -Id $benchPid -ErrorAction SilentlyContinue) }
    if (-not $alive) { break }
    Start-Sleep -Seconds 60
}
Say "基准进程已退出（或等待超时）。开始出裁决表。"

$files = Get-ChildItem "$out\*.jsonl" -ErrorAction SilentlyContinue
if (-not $files) { Say "!! 没找到任何 .jsonl，退出。"; exit 1 }

foreach ($f in $files) {
    Say ("========== " + $f.Name + "  (" + [math]::Round($f.Length/1MB,2) + " MB) ==========")
    $txt = & $py "scripts\analyze_zero_categories.py" $f.FullName 2>&1
    $txt | ForEach-Object { Add-Content -Path $verdict -Value $_ -Encoding utf8 }
    # 文件写的时刻可能还在增长；把「已解析条数」也记下来便于判断是否完整
    $n = ($txt | Select-String '已解析 (\d+) 条').Matches.Groups[1].Value
    Say ("  （本次解析 " + $n + " 条）")
}

Say "全部完成。"
exit 0
