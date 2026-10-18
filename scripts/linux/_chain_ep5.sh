#!/usr/bin/env bash
# ep5 接力：等 ep4 把余弦跑完 → 校验 → 释放 ep4 的中间 checkpoint → 起 ep5 → 挂 ep5 的守卫
#
# 由 _arm_ep5.sh 在 16:05 左右拉起。它自己要活到 21:17 之后，所以用 setsid nohup。
# 任何一步校验失败都**立即安排 12 分钟后关机**，避免机器空转烧钱。
set -uo pipefail
cd /root/autodl-tmp/RetNet
PY=/root/miniconda3/bin/python
LOG4=logs/linux/v18_ep4.out
OUT4=checkpoints/retnet_500m_v15b_ep4
CFG5=tasks/v19_ep5_500m_pro6000.json
LOG5=logs/linux/v19_ep5.out

bail() {
  echo "[chain] !! $1"
  echo "[chain] 12 分钟后关机（不在坏状态上继续烧钱）"
  setsid nohup bash -c 'sleep 720; sync; /usr/bin/shutdown' > /dev/null 2>&1 &
  exit "$2"
}

echo "[chain] $(date '+%H:%M:%S') 等 ep4 的 train.py 退出"
while pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1; do
  sleep 30
done
echo "[chain] $(date '+%H:%M:%S') ep4 进程已退出"

grep -aq "TRAINING DONE" "$LOG4" || bail "$LOG4 里没有 TRAINING DONE" 3
[[ -f "$OUT4/model.safetensors" ]] || bail "$OUT4/model.safetensors 不存在（最终模型没导出）" 4

echo "[chain] ep4 收尾确认：TRAINING DONE + model.safetensors 都在"
ls -la "$OUT4/model.safetensors"

# 释放磁盘：ep5 从 ep4 的 model.safetensors 起跑，中间 checkpoint 不再需要
# （不删的话 ep4+ep5 的 checkpoint 会叠到 ~39 GB，逼近 50 GB 上限）
echo "[chain] 释放 ep4 的中间 checkpoint："
du -sh "$OUT4"/checkpoint-* 2>/dev/null
rm -rf "$OUT4"/checkpoint-*
df -h /root/autodl-tmp | tail -1

sleep 30
rm -f logs/ctrl.txt
echo "[chain] $(date '+%H:%M:%S') 启动 ep5（$CFG5）"
setsid nohup timeout -k 120 30600 "$PY" run_task.py --config "$CFG5" > "$LOG5" 2>&1 &

sleep 150
setsid nohup bash scripts/linux/_peak_watch.sh logs/linux/_peak_ep5.txt 5 > /dev/null 2>&1 &
SKIP_FINALIZE=1 setsid nohup bash scripts/linux/_graceful_stop.sh 7.9 240 0 \
  > logs/linux/tripwire_ep5.out 2>&1 &
setsid nohup bash scripts/linux/_watchdog.sh 8 10 > logs/linux/watchdog_ep5.out 2>&1 &
echo "[chain] $(date '+%H:%M:%S') ep5 守卫已挂：tripwire 7.9h / watchdog 8h / 死手开关仍是绝对 05:35"
echo "[chain] ep5 日志 $LOG5；预期 48000 步 ≈ 7.6h → ~04:45 完成，~04:55 关机"
