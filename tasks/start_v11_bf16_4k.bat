@echo off
REM ============================================================
REM  RetNet v11 4K bf16 training launcher (remote 192.168.1.100)
REM  Usage (in cmd):  start_v11_bf16_4k.bat
REM    --status   show latest loss
REM    --monitor  live watch loss
REM    --resume   resume from last checkpoint
REM  Logs to logs\v11_bf16_4k.log
REM ============================================================
cd /d C:\Users\Ayaka\RetNet
if "%1"=="--status" (
    C:\Users\Ayaka\miniconda3\envs\AI\python.exe run_task.py --config tasks\v11_bf16_4k.json --status
    goto :eof
)
if "%1"=="--monitor" (
    C:\Users\Ayaka\miniconda3\envs\AI\python.exe run_task.py --config tasks\v11_bf16_4k.json --monitor
    goto :eof
)
if "%1"=="--resume" (
    echo Resuming from last checkpoint...
    C:\Users\Ayaka\miniconda3\envs\AI\python.exe run_task.py --config tasks\v11_bf16_4k.json --resume
    goto :eof
)
echo ============================================================
echo  RetNet v11 4K bf16 training start
echo  model: checkpoints\retnet_272m_rp
echo  data : data_cache\chat_pack_v11_4k (4K window, 108030 rows)
echo  out  : checkpoints\retnet_272m_v11_bf16
echo  log  : logs\v11_bf16_4k.log
echo  vram : batch4 ~5.5GB / 8GB safe
echo ============================================================
C:\Users\Ayaka\miniconda3\envs\AI\python.exe run_task.py --config tasks\v11_bf16_4k.json > logs\v11_bf16_4k.log 2>&1
echo.
echo Training finished, exit code: %ERRORLEVEL%
pause
