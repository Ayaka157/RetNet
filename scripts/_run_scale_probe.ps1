# 无人值守实验驱动：等满预算基准释放 GPU，然后依次跑 E1–E4，把推算换成实测。
#   - 解析只用 ASCII 正则（与 python 输出编码、控制台代码页彻底解耦）
#   - 启动时先做解析器自检，自检不过直接退出（不留 5 小时白等）
#   - 每个实验前检查显存是否真的空闲（防上一个 OOM 的残留进程占卡）
$ErrorActionPreference = 'Continue'
$py   = 'C:\Tools\Miniconda3\envs\AI\python.exe'
$root = 'E:\Workspaces\RetNet'
$out  = "$root\logs\scale_probe"
$txt  = "$out\results.txt"
$DATA = 'data_cache\chat_pack_v13'
Set-Location $root
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Say($m) {
    $line = ('[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $m)
    Write-Host $line
    Add-Content -Path $txt -Value $line -Encoding utf8
}

# ---------- 解析器（纯 ASCII 正则）----------
function Parse-Result {
    param([string[]]$Lines)
    $r = [ordered]@{}
    foreach ($ln in $Lines) {
        if (-not $ln) { continue }
        if ($ln -match 'steps=(\d+)')                 { $r.steps = $Matches[1] }
        if ($ln -match '(\d+)\s+ms/(\d+)\s+token')    { $r.opt_ms = $Matches[1]; $r.tok_step = $Matches[2] }
        if ($ln -match '(\d+)\s+tok/s')               { if (-not $r.tok_s) { $r.tok_s = $Matches[1] } }
        if ($ln -match 'alloc\s+([\d.]+)\s+GiB')      { $r.peak_alloc = $Matches[1] }
        if ($ln -match 'reserved\s+([\d.]+)\s+GiB')   { $r.peak_reserved = $Matches[1] }
        if ($ln -match '([\d.]+)\s*->\s*([\d.]+)')    { $r.loss_first = $Matches[1]; $r.loss_last = $Matches[2] }
        if ($ln -match 'out of memory|OutOfMemory|CUDA error') { $r.oom = 'YES' }
    }
    return $r
}

# ---------- 解析器自检（用 bench_trainer.py 的 f-string 逐字还原）----------
function Self-Test {
    $fake = @(
        '  steps=40 wall=30.1s   batch=1 x accum=4',
        '  micro median      197 ms/token',
        '  opt step        788 ms/8192 token',
        '  throughput        10396 tok/s   (10.40k)',
        '  per token 0.0962 ms',
        '  peak mem alloc  6.20 GiB / reserved  6.61 GiB   (card 8.00 GiB)',
        '  mem headroom  1.39 GiB',
        '  loss first/last: 2.4000 -> 2.3900  (40 records)'
    )
    # 真机上这两行是中文，但 ASCII 数字与单位相同；用中文原样再测一遍
    $fake2 = @(
        '  优化步        788 ms/8192 token',
        '  吞吐        10396 tok/s   （10.40k）',
        '  峰值显存 alloc  6.20 GiB / reserved  6.61 GiB   （卡总容量 8.00 GiB）',
        '  loss 首/末: 2.4000 -> 2.3900  （40 条记录，非 NaN 说明训练真的在跑）'
    )
    $a = Parse-Result $fake
    $b = Parse-Result $fake2
    $ok = ($a.opt_ms -eq '788') -and ($a.tok_s -eq '10396') -and ($a.peak_reserved -eq '6.61') -and
          ($b.opt_ms -eq '788') -and ($b.tok_s -eq '10396') -and ($b.peak_alloc -eq '6.20') -and
          ($b.peak_reserved -eq '6.61') -and ($b.loss_first -eq '2.4000') -and ($b.loss_last -eq '2.3900')
    return @{ ok = $ok; ascii = $a; chinese = $b }
}

Say ("Console.OutputEncoding = " + [Console]::OutputEncoding.WebName + " ; PowerShell " + $PSVersionTable.PSVersion)
$st = Self-Test
if (-not $st.ok) {
    Say "!! 解析器自检失败，直接退出，不浪费等待时间。"
    Say ("   ascii:   " + ($st.ascii | ConvertTo-Json -Compress))
    Say ("   chinese: " + ($st.chinese | ConvertTo-Json -Compress))
    exit 3
}
Say ("解析器自检通过（ASCII 与中文两种输入都能解析）")

# ---------- 等 GPU 空闲 ----------
$benchPidFile = "$root\logs\bench_v3_full\.pid"
$benchPid = if (Test-Path $benchPidFile) { [int](Get-Content $benchPidFile -Raw).Trim() } else { -1 }
Say "等待 GPU：基准 PID=$benchPid，阈值 1500 MiB，最长等 10 小时"
$deadline = (Get-Date).AddHours(10)
$waited = $false
while ((Get-Date) -lt $deadline) {
    $alive = $false
    if ($benchPid -gt 0) { $alive = [bool](Get-Process -Id $benchPid -ErrorAction SilentlyContinue) }
    $raw = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader) -join ''
    $used = [int](($raw -replace '[^0-9]', ''))
    if (-not $alive -and $used -lt 1500) { $waited = $true; break }
    Start-Sleep -Seconds 60
}
if (-not $waited) { Say "等了 10 小时 GPU 仍不空闲，放弃。"; exit 2 }
Say "GPU 空闲。开始实验。"

# ---------- 跑一个配置 ----------
# 加固（本轮新增，因为驱动可能在无人值守下跑几小时）：
#   1. 每个实验有**超时**：挂死最多浪费 TimeoutSec，不会吃掉整晚
#   2. **可续跑**：完成的实验记进 completed.txt，重跑驱动会跳过它们
$completedFile = "$out\completed.txt"
$TIMEOUT_SEC = 1800        # 单实验 30 分钟（正常约 5~10 分钟，含 compile）

function Invoke-Bench {
    param([string]$Label, [string]$Model, [string[]]$Extra, [int]$Steps = 40, [int]$WarmSkip = 15)

    # --- 可续跑：已完成的跳过 ---
    if (Test-Path $completedFile) {
        $done = Get-Content $completedFile -Encoding utf8
        if ($done -contains $Label) { Say "--- SKIP $Label（已在 completed.txt 里）"; return }
    }

    for ($k = 0; $k -lt 20; $k++) {
        $raw = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader) -join ''
        $used = [int](($raw -replace '[^0-9]', ''))
        if ($used -lt 1500) { break }
        Say "  显存仍有 $used MiB，等 30s"
        Start-Sleep -Seconds 30
    }
    Say "--- $Label"
    $env:RETNET_FAST_CHUNKWISE = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $argv = @('scripts\bench_trainer.py', '--model_dir', $Model, '--data_dir', $DATA,
              '--max_steps', "$Steps", '--batch_size', '1', '--grad_accum', '4',
              '--optim', 'adamw_bnb_8bit', '--warm_skip', "$WarmSkip")
    $argv += $Extra

    # --- 超时保护：写到临时文件再读，超时就杀 ---
    $tmpOut = "$out\_last.log"
    $tmpErr = "$out\_last.err"
    Remove-Item $tmpOut, $tmpErr -Force -ErrorAction SilentlyContinue
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $timedOut = $false
    $proc = Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -PassThru `
                -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr
    try {
        $proc | Wait-Process -Timeout $TIMEOUT_SEC -ErrorAction Stop
    } catch {
        $timedOut = $true
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    $sw.Stop()
    $o = @()
    if (Test-Path $tmpOut) { $o += Get-Content $tmpOut -Encoding utf8 }
    if (Test-Path $tmpErr) { $o += Get-Content $tmpErr -Encoding utf8 }

    Add-Content -Path $txt -Value ("### " + $Label + "   [" + ($Extra -join ' ') + "]") -Encoding utf8
    if ($timedOut) {
        Add-Content -Path $txt -Value ("  TIMEOUT > {0}s（已杀进程；不记入 completed，下次重跑会重试）" -f $TIMEOUT_SEC) -Encoding utf8
        Say "  TIMEOUT > $TIMEOUT_SEC s，已杀"
        return
    }
    $r = Parse-Result $o
    if ($r.oom) {
        Add-Content -Path $txt -Value "  OOM" -Encoding utf8
        Say "  OOM"
    } else {
        $line = ("  opt_ms={0}  tok/s={1}  peak_alloc={2}GiB  peak_reserved={3}GiB  loss {4}->{5}" -f `
                 $r.opt_ms, $r.tok_s, $r.peak_alloc, $r.peak_reserved, $r.loss_first, $r.loss_last)
        Add-Content -Path $txt -Value $line -Encoding utf8
        if (-not $r.opt_ms) {
            Add-Content -Path $txt -Value "  (解析不到结果，原始输出末 25 行如下)" -Encoding utf8
            $o | Select-Object -Last 25 | ForEach-Object { Add-Content -Path $txt -Value ("    " + $_) -Encoding utf8 }
        }
        Say $line
    }
    Add-Content -Path $txt -Value ("  wall={0:N0}s" -f $sw.Elapsed.TotalSeconds) -Encoding utf8
    # 成功/OOM 都算「跑过了」；TIMEOUT 与未知失败不算
    if ($r.oom -or $r.opt_ms) { Add-Content -Path $completedFile -Value $Label -Encoding utf8 }
}

$CK = 'checkpoints\retnet_272m_v13'
$P5 = 'checkpoints\probe_500m'
$P6 = 'checkpoints\probe_600m'

# E1：检查点省多少显存（同模型，只差开关）—— 验证 docs/16 §3.4 的 1.28x 与 0.65 GiB
Invoke-Bench "E1a 272M no-ckpt" $CK @('--chunk_size','128','--compile','--no_grad_checkpoint')
Invoke-Bench "E1b 272M full-ckpt" $CK @('--chunk_size','128','--compile')

# E3：目标尺寸能不能装下、跑多快（全量检查点 = 最省显存、最保守）
Invoke-Bench "E3a 500M full-ckpt chunk128" $P5 @('--chunk_size','128','--compile') -Steps 30 -WarmSkip 12
Invoke-Bench "E3b 600M full-ckpt chunk128" $P6 @('--chunk_size','128','--compile') -Steps 30 -WarmSkip 12

# E2（本轮升为最高优先）：分层检查点扫描 —— **在目标尺寸上**，
#   而不是 272M（docs/15 §6 已在 272M 上测过，那里"全不检查点"更优所以无用武之地；
#   0.5B/0.6B 装不下，它就成了唯一能把余量换成速度的手段）。
#   顺序从「检查点多、最安全」到「检查点少、可能颠簸」，逐档记录；
#   颠簸的失败模式是 6x 而不是 1.1x（docs/15 §6 记 K=6/20 -> 4457 ms）。
foreach ($n in 16, 12, 10, 8) {
    Invoke-Bench "E2a 500M ckpt-layers=$n/20" $P5 @('--chunk_size','128','--compile','--checkpoint_layers',"$n") -Steps 30 -WarmSkip 12
}
foreach ($n in 20, 16, 12) {
    Invoke-Bench "E2b 600M ckpt-layers=$n/24" $P6 @('--chunk_size','128','--compile','--checkpoint_layers',"$n") -Steps 30 -WarmSkip 12
}

# E4：chunk 256 的显存增量（验证 §2.4 的 C^2 预测；512/1024 已被 docs/15 §6 否定，不再跑）
Invoke-Bench "E4a 500M ckpt chunk256" $P5 @('--chunk_size','256','--compile') -Steps 30 -WarmSkip 12
Invoke-Bench "E4b 600M ckpt chunk256" $P6 @('--chunk_size','256','--compile') -Steps 30 -WarmSkip 12

Say "全部实验完成。结果见 $txt"
