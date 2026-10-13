@echo off
REM ============================================================
REM v14 continuation, segment 2: resume from checkpoint-10000
REM using the repaired / optimized code.
REM
REM Differences vs start_v14.bat:
REM   * config tasks\v14_272m_2k_resume.json
REM       resume=true            -> continue from checkpoint-10000
REM       chunk_size=128         -> now actually takes effect
REM       eval_batch_mult=1      -> first eval no longer costs +128 s
REM   * writes logs\v14_272m_resume.log instead of appending to the
REM     old log, so the two segments stay separable.
REM
REM STOP IT GRACEFULLY (do NOT taskkill):
REM       python run_task.py --ctrl next-checkpoint
REM   That drops logs\ctrl.txt; the training process reads it at step
REM   boundaries and exits with rc=0 AFTER the next checkpoint is on
REM   disk, so run_task.py will not auto-restart.
REM
REM Launch via schtasks, not Start-Process: the latter's children
REM belong to the SSH session and die when it disconnects.
REM
REM Keep this file pure ASCII. cmd.exe reads .bat in the OEM code page
REM (GBK here); UTF-8 comments get mangled into stray commands and the
REM script dies on line 2 -- that is exactly what happened to the
REM first version of start_v14.bat.
REM ============================================================
cd /d C:\Users\Ayaka\RetNet
if not exist logs mkdir logs
echo ==== v14 resume start %DATE% %TIME% ==== >> logs\v14_272m_resume.log
set "PYTHONIOENCODING=utf-8"
set "RETNET_FAST_CHUNKWISE=1"
C:\Users\Ayaka\miniconda3\envs\AI\python.exe -u run_task.py --config tasks\v14_272m_2k_resume.json >> logs\v14_272m_resume.log 2>&1
echo ==== v14 resume exit rc=%ERRORLEVEL% %DATE% %TIME% ==== >> logs\v14_272m_resume.log
