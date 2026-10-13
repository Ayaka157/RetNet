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
# 参数：<最多等几小时=13> <收尾后几分钟关机=10>
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MAX_WAIT_H="${1:-13}"
SHUTDOWN_AFTER_MIN="${2:-10}"

echo "[watchdog] 现在 $(date '+%H:%M:%S')，等训练进程退出（最多 ${MAX_WAIT_H} 小时）"

# 先等它出现（最多 5 分钟），免得刚启动还没起来就误判成"已退出"
for _ in $(seq 1 60); do
  pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1 && break
  sleep 5
done

waited=0
while pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1; do
  sleep 30
  waited=$(( waited + 30 ))
  if [[ "$waited" -ge $(( MAX_WAIT_H * 3600 )) ]]; then
    echo "[watchdog] 已等 ${MAX_WAIT_H} 小时仍未退出，强制进入收尾"
    break
  fi
done

echo "[watchdog] $(date '+%H:%M:%S') 训练进程已退出（等待 ${waited}s），开始收尾检查"
bash scripts/linux/_finalize.sh 2>&1 | tee logs/linux/finalize.out
echo "[watchdog] $(date '+%H:%M:%S') 收尾检查结束，${SHUTDOWN_AFTER_MIN} 分钟后自动关机"
sleep $(( SHUTDOWN_AFTER_MIN * 60 ))
sync
echo "[watchdog] $(date '+%H:%M:%S') 执行关机（数据保留，15 天内可恢复）"
/usr/bin/shutdown || echo "[watchdog] !! shutdown 调用失败，请手动关机"
