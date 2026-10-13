# 满预算基准：带 PID 记录的启动器。
# 上次用 `& python ...` 起作业，job_kill 只杀了 pwsh 外壳，python 变成孤儿还在占 GPU。
# 这次显式记录子进程 PID，并把 stdout 重定向到文件便于监控。
$ErrorActionPreference = 'Continue'
$py   = 'C:\Tools\Miniconda3\envs\AI\python.exe'
$root = 'E:\Workspaces\RetNet'
$out  = "$root\logs\bench_v3_full"

New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $root
$env:PYTHONIOENCODING = 'utf-8'

$argv = @(
    'scripts\bench_v3.py',
    '--models', 'v13=checkpoints/retnet_272m_v13,v14c10k=checkpoints/retnet_272m_v14/checkpoint-10000',
    '--tokenizer', 'data_cache/chat_pack_v13/bpe_tokenizer.json',
    '--dataset', 'bench/data/bench_v3.jsonl',
    '--seeds', '42',
    '--temperature', '0.7', '--top_k', '40', '--top_p', '0.9',
    '--repetition_penalty', '1.1', '--no_repeat_ngram_size', '4',
    '--batch', '1', '--lc_prefill', 'parallel',
    '--budget', '1024',
    '--out', $out,
    '--tag', 'b1024',
    '--no-resume'
)

$p = Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -PassThru `
       -RedirectStandardOutput "$out\run.log" -RedirectStandardError "$out\run.err"
Set-Content -Path "$out\.pid" -Value $p.Id -Encoding ascii
Write-Host ("started pid={0}  log={1}" -f $p.Id, "$out\run.log")
$p.WaitForExit()
Write-Host ("exited rc={0}" -f $p.ExitCode)
