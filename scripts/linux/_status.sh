#!/usr/bin/env bash
# 一眼看训练状态 —— 每次查看只用一次 ssh，省得反复登。
#
#   bash scripts/linux/_status.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "=== 训练进程 ==="
pgrep -af 'train[.]py --model_dir' | head -1 || echo "（没有训练进程）"

echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,power.draw,utilization.gpu,temperature.gpu \
  --format=csv,noheader 2>/dev/null || echo "（nvidia-smi 不可用）"

LOG="$(ls -1t logs/linux/train_*.log 2>/dev/null | head -1)"
echo "=== 最新训练日志: $LOG ==="
if [[ -n "$LOG" ]]; then
  echo "-- 进度 --"
  tr '\r' '\n' < "$LOG" | grep -aoE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+it/s\]" | tail -2
  echo "-- 最新 loss --"
  tr '\r' '\n' < "$LOG" | grep -aE "\{'loss'" | tail -3
  echo "-- 评测 --"
  tr '\r' '\n' < "$LOG" | grep -aE "eval_loss" | tail -3
  echo "-- 收尾/异常 --"
  tr '\r' '\n' < "$LOG" | grep -aE "\[ctrl\]|TRAINING DONE|训练进程退出码|Traceback|out of memory|Error" | tail -5
fi

echo "=== 检查点（最新的 6 个）==="
ls -1dt checkpoints/*/checkpoint-* 2>/dev/null | head -6 || echo "（无）"

echo "=== 磁盘 / 旗标 ==="
df -h /root/gpu-scratch | tail -1
echo "ctrl: $(cat logs/ctrl.txt 2>/dev/null | tr '\n' ' ')"
