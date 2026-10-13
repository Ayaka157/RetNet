# 第二轮无人值守实验：在**修好初始化之后重建的**探针上复测，并把 E5（分页优化器）补上。
#
# 为什么必须重测：旧探针的 gamma 被 _init_weights 置零 → 批量 chunkwise 前向恒为 NaN。
# NaN 不改变 GEMM 的形状与次数，所以**步时/显存理论上仍然有效**，但那是推理、不是实测。
# 这里把它换成实测：同一批配置在健康探针上再跑一遍，看数字是否重现。
#
# E5 的动机：600M 的 reserved 8.05~8.18 GiB 已经越过 8 GiB 卡，实测慢 5.8~8.9 倍。
# 8-bit Adam 的状态是 2.004 B/参数（600M ≈ 1.15 GiB），换成分页优化器把这块搬到
# 主机内存，是唯一可能把 600M 拉回卡内的现成手段。
#
#   V1a/V1b/V2a = 复现 E3a/E3b/E4b
#   V2b         = 复现 E2a 的 ckpt-layers=16（「分层检查点被支配」这一结论的复现）
#   E5a/E5b/E5c = 分页 8-bit Adam
$ErrorActionPreference = 'Continue'
$py   = 'C:\Tools\Miniconda3\envs\AI\python.exe'
$root = 'E:\Workspaces\RetNet'
$out  = "$root\logs\scale_probe2"
$txt  = "$out\results.txt"
$DATA = 'data_cache\chat_pack_v13'
Set-Location $root
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Say($m) {
    $line = ('[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $m)
    Write-Host $line
    Add-Content -Path $txt -Value $line -Encoding utf8
}

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

function Self-Test {
    $fake = @(
        '  steps=120 wall=293.7s   batch=1 x accum=4 = 等效 4',
        '  优化步      10933 ms/8192 token',
        '  吞吐          749 tok/s   （0.75k）',
        '  峰值显存 alloc  7.49 GiB / reserved  8.05 GiB   （卡总容量 8.00 GiB）',
        '  loss 首/末: 16.2488 -> 16.1000  （1 条记录）'
    )
    $a = Parse-Result $fake
    return (($a.opt_ms -eq '10933') -and ($a.peak_reserved -eq '8.05') -and
            ($a.loss_first -eq '16.2488') -and ($a.steps -eq '120'))
}

Say ("Console.OutputEncoding = " + [Console]::OutputEncoding.WebName + " ; PowerShell " + $PSVersionTable.PSVersion)
if (-not (Self-Test)) { Say "!! 解析器自检失败，直接退出"; exit 3 }
Say "解析器自检通过"

$completedFile = "$out\completed.txt"
$TIMEOUT_SEC = 2400

function Invoke-Bench {
    param([string]$Label, [string]$Model, [string[]]$Extra,
          [int]$Steps = 30, [int]$WarmSkip = 12, [string]$Optim = 'adamw_bnb_8bit')

    if (Test-Path $completedFile) {
        $done = Get-Content $completedFile -Encoding utf8
        if ($done -contains $Label) { Say "--- SKIP $Label（已完成）"; return }
    }
    for ($k = 0; $k -lt 20; $k++) {
        $raw = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader) -join ''
        $used = [int](($raw -replace '[^0-9]', ''))
        if ($used -lt 1500) { break }
        Say "  显存仍有 $used MiB，等 30s"
        Start-Sleep -Seconds 30
    }
    Say "--- $Label   [optim=$Optim]"
    $env:RETNET_FAST_CHUNKWISE = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $argv = @('scripts\bench_trainer.py', '--model_dir', $Model, '--data_dir', $DATA,
              '--max_steps', "$Steps", '--batch_size', '1', '--grad_accum', '4',
              '--optim', $Optim, '--warm_skip', "$WarmSkip")
    $argv += $Extra

    # 每个实验留独立日志（旧驱动只留最后一份，事后无法回看颠簸过程）
    $safe = ($Label -replace '[^\w\-]', '_')
    $tmpOut = "$out\$safe.log"
    $tmpErr = "$out\$safe.err"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $timedOut = $false
    $proc = Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -PassThru `
                -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr
    try { $proc | Wait-Process -Timeout $TIMEOUT_SEC -ErrorAction Stop }
    catch {
        $timedOut = $true
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    $sw.Stop()
    $o = @()
    if (Test-Path $tmpOut) { $o += Get-Content $tmpOut -Encoding utf8 }
    if (Test-Path $tmpErr) { $o += Get-Content $tmpErr -Encoding utf8 }

    Add-Content -Path $txt -Value ("### " + $Label + "  [optim=" + $Optim + "]  [" + ($Extra -join ' ') + "]") -Encoding utf8
    if ($timedOut) {
        Add-Content -Path $txt -Value ("  TIMEOUT > {0}s" -f $TIMEOUT_SEC) -Encoding utf8
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
            Add-Content -Path $txt -Value "  (解析不到结果，原始输出末 25 行)" -Encoding utf8
            $o | Select-Object -Last 25 | ForEach-Object { Add-Content -Path $txt -Value ("    " + $_) -Encoding utf8 }
        }
        Say $line
    }
    Add-Content -Path $txt -Value ("  wall={0:N0}s" -f $sw.Elapsed.TotalSeconds) -Encoding utf8
    if ($r.oom -or $r.opt_ms) { Add-Content -Path $completedFile -Value $Label -Encoding utf8 }
}

$P5 = 'checkpoints\probe_500m'
$P6 = 'checkpoints\probe_600m'

# --- 第一步：在健康探针上复现上一轮的关键数字 ---------------------------
Invoke-Bench "V1a 500M ckpt chunk128"        $P5 @('--chunk_size','128','--compile')
Invoke-Bench "V1b 600M ckpt chunk128"        $P6 @('--chunk_size','128','--compile')
Invoke-Bench "V2a 600M ckpt chunk256"        $P6 @('--chunk_size','256','--compile')
Invoke-Bench "V2b 500M ckpt-layers=16/20"    $P5 @('--chunk_size','128','--compile','--checkpoint_layers','16')

# --- 第二步：分页优化器能不能把 600M 拉回卡内 ---------------------------
Invoke-Bench "E5a 500M paged chunk128"       $P5 @('--chunk_size','128','--compile') -Optim 'paged_adamw_8bit'

# ⛔⛔ 2026-10-11 08:10 —— **E5b / E5c 已禁用，不要解除注释** ⛔⛔
#
# E5b（600M + paged_adamw_8bit）**两次把整机弄崩**：
#   · 第 1 次 07:33:06 起跑 → 07:35:18 记到 nvlddmkm 事件 153（显示驱动故障）
#   · 第 2 次 08:08:06 起跑 → 训练本身 118 s 正常跑完（结果有效），
#     但 **08:10:42.527 再次记到 nvlddmkm 153 —— 这是系统冻结前最后一条系统事件**，
#     随后画面彻底卡死，只能强制重启（下次开机 08:15:15）。
#
# 机制（推断，方向明确）：bnb 分页优化器把优化器状态放在 **CUDA managed / 统一内存**；
# 600M 在这块卡上 reserved 已 7.4~8.2 GiB，统一内存超额后驱动不停在 host↔device 搬页，
# WDDM 显示路径被拖住 → nvlddmkm 故障 → 桌面冻结。
# 本机 10 个实验里**只有这个组合**出过 nvlddmkm 事件，其余一个都没有。
#
# 想再要 600M + 分页的数据，必须换安全做法（先在笔记本上试，或改用
# adamw_bnb_8bit + 其它省显存手段），**不要在这台机器上直接重跑**。
#
# E5b 的结果仍然有效，已留档：2,960 ms / reserved 7.39 GiB / 余量 0.61 GiB
# （对照 600M 非分页：19,948 ms / 8.18 GiB）——即 0.6B 靠分页优化器确实装得下，
# 代价是步时 4.15×（vs 713）。详见 docs/23 §9.2。
#
# Invoke-Bench "E5b 600M paged chunk128"       $P6 @('--chunk_size','128','--compile') -Optim 'paged_adamw_8bit'
# Invoke-Bench "E5c 600M paged chunk256"       $P6 @('--chunk_size','256','--compile') -Optim 'paged_adamw_8bit'

Say "全部实验完成（E5b/E5c 已被安全禁用，见脚本内注释）。结果见 $txt"
