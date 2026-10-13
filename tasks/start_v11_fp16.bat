@echo off
REM ============================================================
REM  RetNet v11 8K fp16 training launcher (remote 192.168.1.100)
REM  Usage (in cmd):  start_v11_fp16.bat
REM    start_v11_fp16.bat --status     show latest loss
REM    start_v11_fp16.bat --monitor    live watch loss
REM    start_v11_fp16.bat --resume     resume from last checkpoint
REM  Logs to logs\v11_fp16.log  (cmd redirect, no Tee-Object)
REM ============================================================
cd /d C:\Users\Ayaka\RetNet
if "%1"=="--status" (
    C:\Users\Ayaka\miniconda3\envs\AI\python.exe run_task.py --config tasks\v11_fp16.json --status
    goto :eof
)
if "%1"=="--monitor" (
    C:\Users\Ayaka\miniconda3\envs\AI\python.exe run_task.py --config tasks\v11_fp16.json --monitor
    goto :eof
)
if "%1"=="--resume" (
    echo Resuming from last checkpoint...
    C:\Users\Ayaka\miniconda3\envs\AI\python.exe run_task.py --config tasks\v11_fp16.json --resume
    goto :eof
)
echo ============================================================
echo  RetNet v11 8K fp16 training start
echo  model: checkpoints\retnet_272m_rp
echo  data : data_cache\chat_pack_v11 (8K window, 54015 rows)
echo  out  : checkpoints\retnet_272m_v11_fp16
echo  log  : logs\v11_fp16.log
echo  vram : batch2 ~5.7GB / 8GB safe
echo ============================================================
C:\Users\Ayaka\miniconda3\envs\AI\python.exe run_task.py --config tasks\v11_fp16.json > logs\v11_fp16.log 2>&1
echo.
echo Training finished, exit code: %ERRORLEVEL%
pause
