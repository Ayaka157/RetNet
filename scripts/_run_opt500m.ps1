# 0.5B 训练速度优化：本机短探针驱动。
#
# 硬规则（用户 2026-10-11 定）：
#   · **本机只做短时、低占用任务** → 每个探针超时 300 秒（5 分钟），到点强杀
#   · **严格不超卡** → 每个探针起跑前检查显存空闲；候选里不放任何超出预算的配置
#     （batch2 需要约 3.3 GiB 激活、总额 ~8.5 GiB > 8 GiB 卡，因此**不试**）
#   · **一次一个** → 串行，每个探针之间留间隔
#
# 已排除的维度（不要重复试）：
#   · optim=paged_adamw_8bit —— 500M 上零速度收益（E5a：1,832 vs 1,831），且统一内存
#     是 600M 那次事故的开关；速度优化用不上它
#   · chunk=256 —— 已实测更慢（1,944 vs 1,891）
#   · batch>1 / accum 重排 —— 激活超预算
#
#   pwsh -NoProfile -File scripts\_run_opt500m.ps1
$ErrorActionPreference = 'Continue'
$py   = 'C:\Tools\Miniconda3\envs\AI\python.exe'
$root = 'E:\Workspaces\RetNet'
$out  = "$root\logs\opt500m"
$txt  = "$out\results.txt"
$DATA = 'data_cache\chat_pack_v13'
$MODEL = 'checkpoints\probe_500m'          # 健康探针（形状决定步时；已修 gamma 缺陷）
Set-Location $root
New-Item -ItemType Directory -Force -Path $out | Out-Null

$TIMEOUT_SEC = 300
$GAP_SEC     = 15

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
        if ($ln -match '(\d+)\s+ms/(\d+)\s+token')  { $r.opt_ms = $Matches[1]; $r.tok_step = $Matches[2] }
        if ($ln -match '(\d+)\s+tok/s')             { if (-not $r.tok_s) { $r.tok_s = $Matches[1] } }
        if ($ln -match '微步中位\s+(\d+)\s+ms/步')    { $r.micro_ms = $Matches[1] }
        if ($ln -match 'alloc\s+([\d.]+)\s+GiB')    { $r.peak_alloc = $Matches[1] }
        if ($ln -match 'reserved\s+([\d.]+)\s+GiB') { $r.peak_reserved = $Matches[1] }
        if ($ln -match 'out of memory|OutOfMemory|CUDA error') { $r.oom = 'YES' }
    }
    return $r
}

function Self-Test {
    $fake = @(
        '  微步中位      458 ms/步  （min 400 / max 600）',
        '  优化步       1831 ms/8192 token',
        '  吞吐         4473 tok/s   （4.47k）',
        '  峰值显存 alloc  6.42 GiB / reserved  6.91 GiB   （卡总容量 8.00 GiB）'
    )
    $a = Parse-Result $fake
    return (($a.opt_ms -eq '1831') -and ($a.micro_ms -eq '458') -and
            ($a.peak_reserved -eq '6.91') -and ($a.tok_s -eq '4473'))
}

Say ("parser self-test = " + (Self-Test))
if (-not (Self-Test)) { Say '!! 解析器自检失败，退出'; exit 3 }

$completedFile = "$out\completed.txt"

function Invoke-Probe {
    param([string]$Label, [string[]]$Extra)
    if ((Test-Path $completedFile) -and ((Get-Content $completedFile -Encoding utf8) -contains $Label)) {
        Say "--- SKIP $Label"; return
    }
    # 严格起跑前置：显存必须接近空闲，否则说明上一个探针没退干净
    for ($k = 0; $k -lt 10; $k++) {
        $raw = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader) -join ''
        $used = [int](($raw -replace '[^0-9]', ''))
        if ($used -lt 1200) { break }
        Say "  显存仍有 $used MiB，等 20s（不放行）"
        Start-Sleep -Seconds 20
    }
    Say "--- $Label   [$($Extra -join ' ')]"
    $env:RETNET_FAST_CHUNKWISE = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $argv = @('scripts\bench_trainer.py', '--model_dir', $MODEL, '--data_dir', $DATA,
              '--max_steps', '30', '--batch_size', '1', '--grad_accum', '4',
              '--optim', 'adamw_bnb_8bit', '--warm_skip', '12') + $Extra
    $safe = ($Label -replace '[^\w\-]', '_')
    $o1 = "$out\$safe.log"; $o2 = "$out\$safe.err"
    Remove-Item $o1, $o2 -Force -ErrorAction SilentlyContinue
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $py -ArgumentList $argv -WindowStyle Hidden -PassThru `
           -RedirectStandardOutput $o1 -RedirectStandardError $o2
    $timedOut = $false
    try { $p | Wait-Process -Timeout $TIMEOUT_SEC -ErrorAction Stop }
    catch { $timedOut = $true; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    $sw.Stop()
    $o = @()
    if (Test-Path $o1) { $o += Get-Content $o1 -Encoding utf8 }
    if (Test-Path $o2) { $o += Get-Content $o2 -Encoding utf8 }
    Add-Content -Path $txt -Value ("### " + $Label + "  [" + ($Extra -join ' ') + "]") -Encoding utf8
    if ($timedOut) {
        Add-Content -Path $txt -Value ("  TIMEOUT > {0}s（已杀；不记完成，下次重试）" -f $TIMEOUT_SEC) -Encoding utf8
        Say "  TIMEOUT"
    } else {
        $r = Parse-Result $o
        if ($r.oom) { Add-Content -Path $txt -Value '  OOM' -Encoding utf8; Say '  OOM' }
        else {
            $line = ("  opt_ms={0}  micro_ms={1}  tok/s={2}  alloc={3}GiB  reserved={4}GiB" -f `
                     $r.opt_ms, $r.micro_ms, $r.tok_s, $r.peak_alloc, $r.peak_reserved)
            Add-Content -Path $txt -Value $line -Encoding utf8
            Say $line
        }
        if ($r.oom -or $r.opt_ms) { Add-Content -Path $completedFile -Value $Label -Encoding utf8 }
    }
    Add-Content -Path $txt -Value ("  wall={0:N0}s" -f $sw.Elapsed.TotalSeconds) -Encoding utf8
    Start-Sleep -Seconds $GAP_SEC
}

# --- 探针清单（每个都远在预算内，且 <=5 分钟）------------------------------
Invoke-Probe 'O1 基线 chunk128'        @('--chunk_size','128','--compile')
Invoke-Probe 'O2 reduce-overhead'      @('--chunk_size','128','--compile','--compile_mode','reduce-overhead')
Invoke-Probe 'O3 chunk64'              @('--chunk_size','64','--compile')
Invoke-Probe 'O4 数据加载 workers=4'    @('--chunk_size','128','--compile','--num_workers','4','--prefetch','2')

# ★ 本轮重点：把「重算」换成「卸载到主机内存」。
#   梯度检查点实测让 272M 的步时变成 1.965×（E1a/E1b），反解出重算约 927 ms/优化步；
#   而 save_on_cpu 要搬的只有层边界激活（20×2048×1408×2 B ≈ 115 MB/微步），
#   PCIe 单向 ~5 ms。若成立，步时理论上可从 1,887 ms 掉到 ~1,000 ms（1.85×）。
#
#   ⚠️ O5（save_on_cpu + compile）**已失败**：kernel 里报
#   `AssertionError: expected size 22==22, stride 131072==64 at dim=1` ——
#   saved-tensor hook 打乱了编译图的 fake tensor 传播（错误提示就是
#   "incorrect fake (aka meta) kernel for a custom op"）。所以下面拆成两条，
#   把「save_on_cpu 本身能不能用」与「不编译要付多少代价」分开测。
Invoke-Probe 'O5 save_on_cpu 替代检查点' @('--chunk_size','128','--compile','--save_on_cpu')
Invoke-Probe 'O6 save_on_cpu 不编译'      @('--chunk_size','128','--save_on_cpu')
Invoke-Probe 'O7 基线不编译（对照）'        @('--chunk_size','128')

Say '第一轮探针完成。'
