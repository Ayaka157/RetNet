#!/usr/bin/env bash
# v21 启动器：2 卡 DDP + 计划模式 + 三重守卫。
#
# 依据实测：DDP 稳态 1.30 it/s（65,536 token/步，85,197 tok/s）。
# 计划 60,955 步 → 13.0 h → ¥182（余额 ¥198.5）。
set -uo pipefail
cd /root/autodl-tmp/RetNet
PY=/root/miniconda3/bin/python
CFG=tasks/v21_plan_ddp2.json
LOG=logs/linux/v21_ddp.out
OUT=checkpoints/retnet_500m_v21

echo "=== 0) 前置检查 ==="
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  GPU 数: $NGPU（应为 2）"
[[ "$NGPU" -ge 2 ]] || { echo "  !! 少于 2 张卡，DDP 无法启动"; exit 2; }
[[ -f checkpoints/retnet_500m_v15b_ep3/model.safetensors ]] || { echo "  !! 找不到起点模型 ep3"; exit 3; }
NS=$(ls data_cache/pretrain_v2/*.npy 2>/dev/null | wc -l)
echo "  pretrain_v2 分片: $NS（应为 62）"
[[ "$NS" -ge 60 ]] || { echo "  !! 语料不完整"; exit 4; }
# ⚠️ 昨天残留的 stop 旗标会让训练在第 1 步退出（冒烟时踩过）
rm -f logs/ctrl.txt
echo "  ctrl.txt 已清"

mkdir -p logs/linux
echo "=== 1) 腾磁盘 ==="
rm -rf "$OUT" checkpoints/_smoke_ddp
df -h /root/autodl-tmp | tail -1

echo "=== 2) 死手开关（13.8 h 后无条件关机）==="
DEADMAN=$(( 13*3600 + 48*60 ))
echo "  现在 $(date '+%H:%M:%S')；死手在 $(date -d "+${DEADMAN} seconds" '+%m-%d %H:%M:%S')"
setsid nohup bash -c "sleep $DEADMAN; sync; /usr/bin/shutdown; sleep 60; /usr/bin/shutdown" \
  > logs/linux/deadman_v21.out 2>&1 &

echo "=== 3) 启动 DDP 训练 ==="
setsid nohup timeout -k 300 51000 "$PY" run_task.py --config "$CFG" > "$LOG" 2>&1 &
echo "  日志: $LOG"

echo "=== 4) 等它真正跑起来（DDP + 编译预热约 3-5 分钟）==="
for i in $(seq 1 40); do
  sleep 15
  if grep -aq "it/s" "$LOG" 2>/dev/null; then
    echo "  第 $((i*15))s 已出现进度条"
    break
  fi
  if grep -aqiE "Traceback|Error" "$LOG" 2>/dev/null; then
    echo "  !! 启动报错，看 $LOG"
    grep -aiE "Traceback|Error" "$LOG" | head -5
    exit 5
  fi
done
grep -aE "\[plan\]|find_unused|DDP" "$LOG" | tail -6

echo "=== 5) 挂守卫 ==="
setsid nohup bash scripts/linux/_peak_watch.sh logs/linux/_peak_v21.txt 5 > /dev/null 2>&1 &
SKIP_FINALIZE=1 setsid nohup bash scripts/linux/_graceful_stop.sh 13.2 240 0 \
  > logs/linux/tripwire_v21.out 2>&1 &
# ⚠️ 第一个参数必须是整数小时
setsid nohup bash scripts/linux/_watchdog.sh 14 10 > logs/linux/watchdog_v21.out 2>&1 &
echo "  peak + tripwire(13.2h) + watchdog(14h) 已挂"

echo "=== 6) 重启语料下载（断点续传，与训练并行）==="
if [[ -f logs/_dl_manifest.tsv ]]; then
  setsid nohup bash scripts/linux/_curl_download.sh > logs/linux/curl_dl.out 2>&1 &
  echo "  下载器已启动"
else
  echo "  !! 找不到 logs/_dl_manifest.tsv，下载未启动"
fi

echo "=== 7) 当前状态 ==="
sleep 20
tr '\r' '\n' < "$LOG" | grep -a "it/s" | tail -1
df -h /root/autodl-tmp | tail -1
echo "=== 完成 ==="
