#!/usr/bin/env bash
# 一键启动正式 0.5B 训练：峰值采样 + 训练 + 到点优雅收尾，三者互不依赖。
#
#   bash scripts/linux/_launch_real.sh 10.5
#
# 参数是「多少小时后优雅收尾」。为什么按小时数而不是按 --max-hours：
# 控制旗标能让训练**先存检查点再正常退出**（rc=0、最终模型照常导出），
# 而 `timeout` 的 SIGTERM 有一定概率落在写盘窗口里，留下半个 checkpoint。
# --max-hours 仍作为硬背板保留。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
R="$PWD"
HOURS="${1:-10.5}"
TRAIN_LOG="$R/logs/linux/v15b_real.out"

# 清掉可能残留的控制指令（残留的 stop/next-checkpoint 会让训练一开始就退出）
rm -f "$R/logs/ctrl.txt"
echo "[launch] 已清空控制旗标（等价于 run）"

setsid nohup bash "$R/scripts/linux/_peak_watch.sh" "$R/logs/linux/_peak_v15b.txt" 5 \
  >/dev/null 2>&1 </dev/null &
echo "[launch] 峰值采样已启动"

setsid nohup bash "$R/scripts/linux/_graceful_stop.sh" "$HOURS" 240 20 \
  >"$R/logs/linux/tripwire.out" 2>&1 </dev/null &
echo "[launch] 收尾哨兵已启动：${HOURS} 小时后存档并优雅退出，再 20 分钟后自动关机"

setsid nohup bash "$R/scripts/linux/run_train.sh" \
  --config tasks/v15b_500m_24g.json \
  --max-hours 10.9 \
  --no-shutdown \
  >"$TRAIN_LOG" 2>&1 </dev/null &
echo "[launch] 训练已启动 -> $TRAIN_LOG"

# ★ 起跑后必须**验证训练进程真的在**。2026-10-11 踩过一次：`--max-hours 10.9` 进了
# bash 的 $(( ))（只吃整数）→ 算术报错 → 脚本继续跑到 timeout 才失败 → 日志里看着
# 正常，但训练进程一次都没起来。所以这里做一道会失败的检查。
for _ in $(seq 1 24); do
  if pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1; then
    echo "[launch] ✅ 已确认训练进程在跑（PID $(pgrep -f 'train[.]py --model_dir' | head -1)）"
    exit 0
  fi
  sleep 5
done
echo "[launch] ❌ 120 秒内没有训练进程 —— 启动失败。日志末尾："
tail -20 "$TRAIN_LOG"
exit 1
