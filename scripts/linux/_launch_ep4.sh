#!/usr/bin/env bash
# ep4 启动器 v2 —— 修正上一版的致命错误：_watchdog.sh 的参数必须是**整数**。
#
# v1 的教训（2026-10-17 13:10 实例被误关）：
#   我把 watchdog 的上限传成 `9.0`，而 _watchdog.sh 第 32 行是
#       if [[ "$waited" -ge $(( MAX_WAIT_H * 3600 )) ]]; then
#   bash 算术不支持浮点 → `9.0: syntax error: invalid arithmetic operator`
#   → 循环异常退出 → 脚本判"训练进程已退出（等待 30s）" → 跑 finalize → 关机。
#   训练其实好好在跑，存活到 13:10:32 才被关机杀掉，白丢 13 分钟。
#   现在传 `9`（整数）。_graceful_stop.sh 用 awk 换算，浮点没问题，可以传 8.0。
#
# 时间轴（13:17 起算，max_steps=50500 @1.77it/s = 7.93h）：
#   13:17  训练启动
#   21:10  跑到 50500 步 → TRAINING DONE → 导出 model.safetensors → 退出
#   21:10  _watchdog.sh 发现退出 → _finalize.sh → 10 分钟后关机
#   21:17  _graceful_stop.sh 8.0h 兜底（若训练因故没跑完就优雅停）
#   21:50  死手开关：无条件关机
set -uo pipefail
cd /root/autodl-tmp/RetNet
PY=/root/miniconda3/bin/python
CFG=tasks/v18_ep4_500m_pro6000.json
LOG=logs/linux/v18_ep4.out
mkdir -p logs/linux

NOW=$(date +%s)
DEADLINE=$(date -d 'today 21:50' +%s)
SECS=$((DEADLINE - NOW))
echo "[launch] $(date '+%H:%M:%S') 死手开关 ${SECS}s 后（$(date -d @$DEADLINE '+%m-%d %H:%M:%S')）无条件关机"
setsid nohup bash -c "sleep $SECS; sync; /usr/bin/shutdown; sleep 60; /usr/bin/shutdown" \
  > logs/linux/deadman_ep4.out 2>&1 &

rm -f logs/ctrl.txt
# 上一轮被杀时可能留下半个输出目录，清掉重来
rm -rf checkpoints/retnet_500m_v15b_ep4

SKIP_FINALIZE=1 setsid nohup bash scripts/linux/_graceful_stop.sh 8.0 240 0 \
  > logs/linux/tripwire_ep4.out 2>&1 &
echo "[launch] 优雅收尾兜底：8.0 h（21:17）"

setsid nohup timeout -k 120 32400 "$PY" run_task.py --config "$CFG" > "$LOG" 2>&1 &
echo "[launch] 训练已启动，日志 $LOG"

sleep 150
setsid nohup bash scripts/linux/_peak_watch.sh logs/linux/_peak_ep4.txt 5 > /dev/null 2>&1 &
# ⚠️ 第一个参数是整数小时，不能写 9.0
setsid nohup bash scripts/linux/_watchdog.sh 9 10 > logs/linux/watchdog_ep4.out 2>&1 &
echo "[launch] $(date '+%H:%M:%S') 守卫已挂：peak + watchdog(9 小时整数上限 / 退出后 10 分钟关机)"
echo "[launch] 预期 21:10 完成 50500 步 → finalize → ~21:25 关机"
