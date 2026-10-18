#!/usr/bin/env bash
# v20 启动器：计划模式训练（新语料 + 对话退火），三重兜底。
#
# 前提：实例必须是**有卡**模式；无卡模式下 nvidia-smi 不通，本脚本第 0 步就会退出。
# 预算：61,000 步 @1.75 it/s ≈ 9.7h ≈ ¥68。守卫上限给到 10.2h，留 ~30 min 收尾。
set -uo pipefail
cd /root/autodl-tmp/RetNet
PY=/root/miniconda3/bin/python
CFG=tasks/v20_fineweb_plan.json
LOG=logs/linux/v20_fineweb.out
OUT=checkpoints/retnet_500m_v20_fineweb

echo "=== 0) 前置检查 ==="
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "!! 当前没有 GPU（无卡模式）。请先在控制台关机、再用有卡模式开机，然后重跑本脚本。"
  exit 2
fi
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[[ -f checkpoints/retnet_500m_v15b_ep3/model.safetensors ]] || { echo "!! 找不到起点模型 ep3"; exit 3; }
[[ -f tasks/plan.json ]] || { echo "!! 找不到 tasks/plan.json"; exit 4; }
NP=$(ls data_cache/pretrain_v2/*.npy 2>/dev/null | wc -l)
echo "  pretrain_v2 分片数: $NP（应为 62）"
[[ "$NP" -ge 60 ]] || { echo "!! 新语料不完整"; exit 5; }
$PY -c "import sys;sys.path.insert(0,'.');from retnet_hf.plan import PlanWatcher;w=PlanWatcher('tasks/plan.json',verbose=False);p=w.load(force=True);print('  计划: %d 段, total %d 步, seq %d' % (len(p.stages),p.total_steps,p.seq_len))" || { echo "!! 计划解析失败"; exit 6; }

echo "=== 1) 腾磁盘 ==="
rm -f checkpoints/retnet_500m_v15b/model.safetensors \
      checkpoints/retnet_500m_v15b_ep2/model.safetensors
rm -rf checkpoints/retnet_500m_v15b_ep4/checkpoint-*
rm -rf "$OUT"
echo "  已删 ep1/ep2 冗余模型（本地 + HF 都有）与 ep4 中间 checkpoint"
df -h /root/autodl-tmp | tail -1

echo "=== 2) 死手开关（10.2 h 后无条件关机）==="
DEADMAN=$(( 10*3600 + 12*60 ))
echo "  现在 $(date '+%H:%M:%S')；死手在 $(date -d "+${DEADMAN} seconds" '+%m-%d %H:%M:%S')"
setsid nohup bash -c "sleep $DEADMAN; sync; /usr/bin/shutdown; sleep 60; /usr/bin/shutdown" \
  > logs/linux/deadman_v20.out 2>&1 &

echo "=== 3) 启动训练（计划模式）==="
rm -f logs/ctrl.txt
setsid nohup timeout -k 120 $(( 12*3600 )) "$PY" run_task.py --config "$CFG" > "$LOG" 2>&1 &
echo "  日志: $LOG"

echo "=== 4) 挂守卫 ==="
sleep 180
setsid nohup bash scripts/linux/_peak_watch.sh logs/linux/_peak_v20.txt 5 > /dev/null 2>&1 &
SKIP_FINALIZE=1 setsid nohup bash scripts/linux/_graceful_stop.sh 9.9 240 0 \
  > logs/linux/tripwire_v20.out 2>&1 &
# ⚠️ 第一个参数是整数小时
setsid nohup bash scripts/linux/_watchdog.sh 11 10 > logs/linux/watchdog_v20.out 2>&1 &
echo "  peak + tripwire(9.9h) + watchdog(11h 整数) 已挂"

echo "=== 5) 起来了吗 ==="
sleep 60
pgrep -f 'train[.]py --model_dir' >/dev/null && echo "  train.py 在跑 ✓" || echo "  ⚠️ train.py 不在，看 $LOG"
tr '\r' '\n' < "$LOG" | grep -aE '\[plan\]|\[stage\]|rows=|Traceback|Error' | tail -8
tail -c 400 "$LOG" | tr '\r' '\n' | grep -a 'it/s' | tail -1
echo "=== 完成。改 tasks/plan.json 即改训练计划，无需重启。 ==="
