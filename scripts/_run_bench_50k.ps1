# v14 50000 步终版（retnet_v14_50k）的完整基准，**在笔记本上跑**。
#
# ⛔ 不要在桌面本机跑这个（它是 ~1.5 小时的长占用 GPU 作业）。
#    用户规则：**训练与长作业跑笔记本，本机只做短时低占用任务**。
#    而且本机的推理基准**确实触发过 nvlddmkm 故障**：10/10 09:08:54 与 19:48:39
#    两次 153 事件都与 bench 的 jsonl 写入同一秒（见 docs/23 §9.2 的对账表）。
#
# 可比性口径完全照抄 scripts\_run_bench_full.ps1（v13 / v14c10k 那一轮）：
#   同一题库 bench_v3.jsonl、同一 tokenizer、同一解码参数、同一种子、同 budget=1024。
#   唯一差异是模型，以及它是**另一次会话**跑的 → 步时/吞吐不可与上一轮直接比，
#   分数可比（同题库 + 同种子 + 同解码参数，分数是模型的属性）。
#   tag=b1024_50k，不覆盖上一轮的 summary.b1024.md。
#
# ── 关键工程点：远程长作业怎么才不被杀 ───────────────────────────────
# Windows OpenSSH 会把会话放进一个 job object，**会话结束时会连带杀掉所有子进程**。
# 所以 `Start-Process`（无论加不加 -NoNewWindow）起的作业都会在 ssh 断开后消失 ——
# 实测：打印完 1020 项表头后进程就没了，err 里连异常都没有。
# 正确做法是用 **WMI 创建进程**（Invoke-CimMethod Win32_Process Create）：
# 它由 WmiPrvSE 托管、不在 ssh 的 job object 里，断开后继续跑。
# 重定向交给 `cmd /c "... > log 2> err"`，因为 WMI 不支持 RedirectStandardOutput。
#
#     pwsh -NoProfile -File scripts\_run_bench_50k.ps1          # 从本机驱动笔记本
$ErrorActionPreference = 'Continue'
$sshKey = "$env:USERPROFILE\.ssh\id_ed25519"
$remote = 'Ayaka@192.168.1.100'

$remoteScript = @'
$root = 'C:\Users\Ayaka\RetNet'
$out  = "$root\logs\bench_v3_full"
$py   = 'C:\Users\Ayaka\miniconda3\envs\AI\python.exe'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$raw = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader) -join ''
$used = [int](($raw -replace '[^0-9]',''))
if ($used -gt 1500) { "!! 笔记本显存已占用 $used MiB，拒绝起跑"; exit 2 }

# 参数刻意不含空格 → 整条命令行不需要内层引号，cmd 解析最稳
$argstr = @(
  'scripts\bench_v3.py',
  '--models','v14_50k=checkpoints/retnet_272m_v14/checkpoint-50000',
  '--tokenizer','data_cache/chat_pack_v13/bpe_tokenizer.json',
  '--dataset','bench/data/bench_v3.jsonl',
  '--seeds','42','--temperature','0.7','--top_k','40','--top_p','0.9',
  '--repetition_penalty','1.1','--no_repeat_ngram_size','4',
  '--batch','1','--lc_prefill','parallel','--budget','1024',
  '--out',$out,'--tag','b1024_50k','--no-resume'
) -join ' '
$cl = 'cmd.exe /c "cd /d ' + $root + ' && set PYTHONIOENCODING=utf-8 && ' +
      $py + ' ' + $argstr + ' > ' + $out + '\run_50k.log 2> ' + $out + '\run_50k.err"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $cl; CurrentDirectory = $root }
Set-Content -Path "$out\.pid_50k" -Value $r.ProcessId -Encoding ascii
"started via WMI  ReturnValue=$($r.ReturnValue)  cmdPid=$($r.ProcessId)"
'@

$enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteScript))
& ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -i $sshKey $remote "powershell -NoProfile -EncodedCommand $enc"
Write-Host "监控：ssh $remote 后看 logs\bench_v3_full\run_50k.log"
