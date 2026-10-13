#!/usr/bin/env bash
# 到点优雅收尾 —— 先存检查点，再让训练在步边界正常退出，最后可选自动关机。
#
#   setsid nohup bash scripts/linux/_graceful_stop.sh 10.5 240 45 \
#     > logs/linux/tripwire.out 2>&1 &
#
# 参数：<小时数> [存档后等待秒数=240] [训练退出后延时关机分钟数=0（不关）]
#
# ## 为什么不用 `timeout` 硬杀
#
# `timeout` 发 SIGTERM，落点随机。写一个 checkpoint 约 60 秒，每 5000 步才存一次
# （约 51 分钟），所以硬杀有约 2% 的概率正好落在写盘窗口里，留下半个 checkpoint 目录，
# `--resume` 可能读不了。控制旗标走的是 HF 正常路径：
#
#   on_step_end 置 should_training_stop -> 本步 _maybe_log_save_evaluate 照常执行
#   -> trainer.train() 正常返回 -> trainer.save_model() + verify_export()
#   -> 打印 TRAINING DONE -> rc=0
#
# 即「先落盘、再退出、最终模型照常导出」，run_train.sh 的两道校验都判成功。
#
# ## 为什么是两步
#
# `save` 是一次性指令，让训练在下一个步边界立刻存一次；隔几分钟再发 `stop`，
# 保证那次存档写完再退。顺序反了会丢掉最后一段训练。
set -uo pipefail
HOURS="${1:?用法: _graceful_stop.sh <小时数> [存档后等待秒=240] [退出后关机分钟=0]}"
WAIT_AFTER_SAVE="${2:-240}"
SHUTDOWN_AFTER_MIN="${3:-0}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-/root/miniconda3/bin/python}"

SECS="$(awk -v h="$HOURS" 'BEGIN{printf "%d", h*3600}')"
echo "[tripwire] 现在 $(date '+%H:%M:%S')；将在 ${HOURS} 小时后（$(date -d "+${SECS} seconds" '+%H:%M:%S')）优雅收尾"
sleep "$SECS"

echo "[tripwire] $(date '+%H:%M:%S') 到点，请求立刻存一次检查点（训练在下一步边界执行）"
"$PY" run_task.py --ctrl save || echo "[tripwire] !! save 指令写入失败"

sleep "$WAIT_AFTER_SAVE"

echo "[tripwire] $(date '+%H:%M:%S') 请求 stop：步边界优雅退出，最终模型照常导出"
"$PY" run_task.py --ctrl stop || echo "[tripwire] !! stop 指令写入失败"

# 等训练进程退出（最多 30 分钟），这样后面的汇总拿到的是终态
for _ in $(seq 1 180); do
  pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1 || break
  sleep 10
done
echo "[tripwire] $(date '+%H:%M:%S') 训练进程已退出，开始汇总"

LOG="$(ls -1t logs/linux/train_*.log 2>/dev/null | head -1)"
if [[ -n "$LOG" ]]; then
  echo "=== 终态日志: $LOG ==="
  tr '\r' '\n' < "$LOG" | grep -aE "\[ctrl\]|TRAINING DONE|训练进程退出码|已确认|eval_loss|out of memory|Traceback" | tail -12
fi
echo "=== 检查点 ==="
ls -1dt checkpoints/retnet_500m_v15b/checkpoint-* 2>/dev/null | head -6 || echo "（无）"
echo "=== 显存 ==="
nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader 2>/dev/null

# ★ 自动跑完整收尾检查（含 --resume 恢复验证）。
# 为什么要在这里跑：目标要求「启动并**守住**一次可恢复的训练」，而恢复路径此前
# 从未被真正跑过。收尾后只剩约 20 分钟窗口，如果只靠人守着，一旦没赶上就白跑。
# 放在哨兵里，验证就不依赖任何人在场。
echo "[tripwire] $(date '+%H:%M:%S') 开始训后收尾检查（含恢复验证）"
if [[ "${SKIP_FINALIZE:-0}" == "1" ]]; then
  # 兜底哨兵用这个：它只负责「到点优雅停」，收尾交给看门狗，免得两边同时跑
  # `_finalize.sh`（那会两个进程同时抢 GPU，恢复验证必然失败）。
  echo "[tripwire] SKIP_FINALIZE=1，收尾检查交给看门狗"
else
  bash scripts/linux/_finalize.sh 2>&1 | tee logs/linux/finalize.out
  echo "[tripwire] $(date '+%H:%M:%S') 收尾检查退出码 ${PIPESTATUS[0]}"
fi

if [[ "$SHUTDOWN_AFTER_MIN" -gt 0 ]]; then
  echo "[tripwire] $(date '+%H:%M:%S') ${SHUTDOWN_AFTER_MIN} 分钟后自动关机（留出训后评测窗口）"
  sleep $(( SHUTDOWN_AFTER_MIN * 60 ))
  sync
  echo "[tripwire] $(date '+%H:%M:%S') 执行关机"
  /usr/bin/shutdown || echo "[tripwire] !! shutdown 调用失败，请手动关机"
fi
