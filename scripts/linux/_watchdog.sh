#!/usr/bin/env bash
# 训练看门狗：**等训练进程退出**（不管是跑完、崩了还是被停），然后跑收尾检查并关机。
#
#   setsid nohup bash scripts/linux/_watchdog.sh 13 10 > logs/linux/watchdog.out 2>&1 &
#
# ## 为什么不用「定时哨兵」了
#
# `_graceful_stop.sh` 适合「余额只够跑一部分」的场景 —— 到点主动收尾。
# 但余额充足、目标是**跑完整个 epoch** 时，定时哨兵反而有害：训练在 max_steps 到顶时
# 会自己打 `TRAINING DONE` 并正常退出，定时哨兵会把它截断在收尾时刻。
# 所以这里改成「等它自己结束」：`trainer.train()` 跑完 max_steps → save_model +
# verify_export → TRAINING DONE → 进程退出 → 本脚本收尾 + 关机。
#
# 参数：<最多等几小时=13，**必须是整数**> <收尾后几分钟关机=10>
#
# ## 2026-10-17 的事故（已修）
#
# 把小时数传成 `9.0`，后面的 `$(( MAX_WAIT_H * 3600 ))` 直接报
#   `9.0: syntax error: invalid arithmetic operator (error token is ".0")`
# —— bash 算术不支持浮点。循环因此异常退出，脚本立刻判「训练进程已退出（等待 30s）」，
# 跑了 finalize 并把实例关掉，而训练其实好好在跑，白丢 13 分钟。
# 两道防线：参数入口做整数校验（下面），退出判定改成连续 3 次看不到进程才算数。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MAX_WAIT_H="${1:-13}"
SHUTDOWN_AFTER_MIN="${2:-10}"

if ! [[ "$MAX_WAIT_H" =~ ^[0-9]+$ ]]; then
  echo "[watchdog] !! 第一个参数（最多等几小时）必须是非负整数，收到 '$MAX_WAIT_H'" >&2
  exit 2
fi
if ! [[ "$SHUTDOWN_AFTER_MIN" =~ ^[0-9]+$ ]]; then
  echo "[watchdog] !! 第二个参数（几分钟后关机）必须是非负整数，收到 '$SHUTDOWN_AFTER_MIN'" >&2
  exit 2
fi

MB=30                      # 每轮采样间隔（秒）
CONFIRM_MISSES=3           # 连续这么多次看不到进程才认定它真的退出了

echo "[watchdog] 现在 $(date '+%H:%M:%S')，等训练进程退出（最多 ${MAX_WAIT_H} 小时）"

# 先等它出现（最多 5 分钟），免得刚启动还没起来就误判成"已退出"
for _ in $(seq 1 60); do
  pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1 && break
  sleep 5
done

waited=0
misses=0
while true; do
  if pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1; then
    misses=0
  else
    misses=$(( misses + 1 ))
    if [[ "$misses" -ge "$CONFIRM_MISSES" ]]; then
      break
    fi
  fi
  sleep "$MB"
  waited=$(( waited + MB ))
  if [[ "$waited" -ge $(( MAX_WAIT_H * 3600 )) ]]; then
    echo "[watchdog] 已等 ${MAX_WAIT_H} 小时仍未退出，强制进入收尾"
    break
  fi
done

echo "[watchdog] $(date '+%H:%M:%S') 训练进程已退出（等待 ${waited}s，连续 ${CONFIRM_MISSES} 次采样未见），开始收尾检查"
bash scripts/linux/_finalize.sh 2>&1 | tee logs/linux/finalize.out
echo "[watchdog] $(date '+%H:%M:%S') 收尾检查结束，${SHUTDOWN_AFTER_MIN} 分钟后自动关机"
sleep $(( SHUTDOWN_AFTER_MIN * 60 ))
sync
echo "[watchdog] $(date '+%H:%M:%S') 执行关机（数据保留，15 天内可恢复）"
/usr/bin/shutdown || echo "[watchdog] !! shutdown 调用失败，请手动关机"
