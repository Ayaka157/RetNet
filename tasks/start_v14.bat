@echo off
REM ============================================================
REM v14: continue training from the v13 final export.
REM Fresh cosine schedule, peak LR lowered to 4e-5.
REM
REM Launch this via schtasks. Do NOT use Start-Process: its
REM children belong to the SSH session and are killed the moment
REM SSH disconnects.
REM
REM Keep this file pure ASCII. cmd.exe reads .bat in the OEM code
REM page (GBK here); UTF-8 comments get mangled into stray
REM commands and the script dies on line 2.
REM ============================================================
cd /d C:\Users\Ayaka\RetNet
if not exist logs mkdir logs
echo ==== v14 start %DATE% %TIME% ==== >> logs\v14_272m_cont.log
set "PYTHONIOENCODING=utf-8"
set "RETNET_FAST_CHUNKWISE=1"
C:\Users\Ayaka\miniconda3\envs\AI\python.exe -u run_task.py --config tasks\v14_272m_2k_cont.json >> logs\v14_272m_cont.log 2>&1
echo ==== v14 exit rc=%ERRORLEVEL% %DATE% %TIME% ==== >> logs\v14_272m_cont.log
