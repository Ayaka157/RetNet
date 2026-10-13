@echo off
REM ============================================================
REM  v13_272m_2k : chat_pack_v13 (seq 2048, R1 <think> KEPT intact)
REM  from retnet_272m_rp, bf16, chunkwise, batch 4, 12000 steps
REM  Measured on this laptop: 2.283 s/step @ seq 2048 bf16 batch 4
REM ============================================================
cd /d C:\Users\Ayaka\RetNet
if not exist logs mkdir logs
echo ==== v13_272m_2k start %DATE% %TIME% ==== >> logs\v13_272m_2k.log
C:\Users\Ayaka\miniconda3\envs\AI\python.exe -u run_task.py --config tasks\v13_272m_2k.json >> logs\v13_272m_2k.log 2>&1
echo ==== v13_272m_2k exit rc=%ERRORLEVEL% %DATE% %TIME% ==== >> logs\v13_272m_2k.log
