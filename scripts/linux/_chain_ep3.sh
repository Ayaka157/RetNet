#!/usr/bin/env bash
# 链式启动 epoch 3 —— 无人值守接力。
#
#   setsid nohup bash scripts/linux/_chain_ep3.sh > logs/linux/chain_ep3.out 2>&1 &
#
# ## 为什么需要它
#
# 用户去睡觉，epoch 2 在 max_steps 到顶后自己退出（约 00:41）。要接着跑第三轮，
# 就必须有人"在那一刻"动手。人不在场，所以把"那一刻"交给本脚本。
#
# ## 为什么先杀掉旧看门狗
#
# `_watchdog.sh` 判定的是「train.py 还在不在」——epoch 2 一退出它就会跑收尾并关机，
# 正好会把刚起来的 epoch 3 一起带走。所以**外层先杀掉旧看门狗**，本脚本在
# epoch 3 起来之后再挂一份新的（见末尾）。顺序不能反。
#
# ## 失败时的行为（钱优先）
#
# * epoch 2 没有 `TRAINING DONE`（崩了 / 被截断）→ **不启动第三轮**，只挂 12 分钟后的关机。
# * epoch 2 的最终模型没导出 → 同上。
# * 任何一步异常 → 脚本退出，死手开关（本脚本开头挂的那个）仍在，最多多烧到 ~03:52。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PYTHON:-/root/miniconda3/bin/python}"
LOG2="logs/linux/v16_ep2.out"
LOG3="logs/linux/v17_ep3.out"
OUT2="checkpoints/retnet_500m_v15b_ep2"
CFG="tasks/v17_ep3_500m_pro6000.json"

echo "[chain] 现在 $(date '+%m-%d %H:%M:%S')；等 epoch 2 的 train.py 退出"

# 死手开关：无条件关机（比看门狗晚，是"看门狗卡死/被杀"时的兜底）
# 12000 s = 3.33 h；余额在 03:59:59 会再扣一小时，所以必须早于那一刻。
setsid nohup bash -c 'sleep 12000; sync; /usr/bin/shutdown; sleep 60; /usr/bin/shutdown' \
  > logs/linux/deadman_ep3.out 2>&1 &
echo "[chain] 死手开关已挂：预计 $(date -d '+12000 seconds' '+%m-%d %H:%M:%S') 关机"

while pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1; do
  sleep 20
done
echo "[chain] $(date '+%H:%M:%S') epoch 2 进程已退出"

if ! grep -aq "TRAINING DONE" "$LOG2"; then
  echo "[chain] !! $LOG2 里没有 TRAINING DONE —— 不启动第三轮（不在坏状态上继续烧钱）"
  echo "[chain] 12 分钟后关机"
  setsid nohup bash -c 'sleep 720; sync; /usr/bin/shutdown' > /dev/null 2>&1 &
  exit 3
fi

if [[ ! -f "$OUT2/model.safetensors" ]]; then
  echo "[chain] !! $OUT2/model.safetensors 不存在（最终模型没导出）—— 不启动第三轮"
  setsid nohup bash -c 'sleep 720; sync; /usr/bin/shutdown' > /dev/null 2>&1 &
  exit 4
fi
echo "[chain] epoch 2 收尾确认：TRAINING DONE + $OUT2/model.safetensors 都在"

sleep 30                      # 等显存真正释放
rm -f logs/ctrl.txt           # 清掉可能残留的 stop/save 旗标
echo "[chain] $(date '+%H:%M:%S') 启动 epoch 3（$CFG）"
setsid nohup timeout -k 120 18000 "$PY" run_task.py --config "$CFG" > "$LOG3" 2>&1 &

sleep 90                      # 等新进程真的起来，再挂守卫（避免守卫误判"已退出"）
setsid nohup bash scripts/linux/_peak_watch.sh logs/linux/_peak_ep3.txt 5 > /dev/null 2>&1 &
# 最后一个参数改成 10 分钟：epoch 3 结束时余额只剩个位数，不能再像 epoch 2 那样留 40 分钟
setsid nohup bash scripts/linux/_watchdog.sh 4 10 > logs/linux/watchdog_ep3.out 2>&1 &
echo "[chain] $(date '+%H:%M:%S') 守卫已重挂：peak=_peak_ep3.txt，watchdog=4h 上限/退出后 10 分钟关机"
echo "[chain] 本轮预期 2.64 h → 约 03:21 结束；日志 $LOG3"
