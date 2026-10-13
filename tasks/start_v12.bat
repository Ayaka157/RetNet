@echo off
REM ============================================================
REM  v12_272m : chat_pack_v12 (R1 <think> stripped) from retnet_272m_rp
REM  bf16, chunkwise, batch 4, seq 512, 60000 steps
REM  Launched detached by the workstation via SSH.
REM ============================================================
cd /d C:\Users\Ayaka\RetNet
if not exist logs mkdir logs
echo ==== v12_272m start %DATE% %TIME% ==== >> logs\v12_272m.log
C:\Users\Ayaka\miniconda3\envs\AI\python.exe run_task.py --config tasks\v12_272m.json >> logs\v12_272m.log 2>&1
echo ==== v12_272m exit rc=%ERRORLEVEL% %DATE% %TIME% ==== >> logs\v12_272m.log
