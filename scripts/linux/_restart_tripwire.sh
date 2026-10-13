#!/usr/bin/env bash
# 重启收尾哨兵，**保持原来的收尾时刻不变**（默认次日 05:01:16）。
#
#   bash scripts/linux/_restart_tripwire.sh            # 目标 = 次日 05:01:16
#   bash scripts/linux/_restart_tripwire.sh "tomorrow 04:30:00"
#
# 为什么需要"保持时刻"：哨兵是 `sleep N` 实现的一次性定时器，改脚本后必须重启进程，
# 而重启会把 10.5 小时从头算起 —— 那就比余额耗尽点（05:38）还晚，等于白设。
# 所以这里按**绝对时刻**反推剩余小时数（awk 算，支持小数）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TARGET_SPEC="${1:-tomorrow 05:01:16}"
NOW="$(date +%s)"
TARGET="$(date -d "$TARGET_SPEC" +%s)"
H="$(awk -v a="$NOW" -v b="$TARGET" 'BEGIN{printf "%.4f", (b-a)/3600}')"

echo "[restart] 现在 $(date '+%m-%d %H:%M:%S')；目标收尾 $(date -d "$TARGET_SPEC" '+%m-%d %H:%M:%S')；剩余 ${H} 小时"
if ! awk -v h="$H" 'BEGIN{exit !(h > 0.05)}'; then
  echo "!! 剩余时间不足 3 分钟，放弃重启（避免设成一个已经过去的时刻）"; exit 2
fi

if pkill -f '_graceful[_]stop'; then
  echo "[restart] 已停掉旧哨兵"
else
  echo "[restart] 没有旧哨兵在跑（或已退出）"
fi
sleep 2

# 清掉残留指令：残留的 stop / next-checkpoint 会让训练在下一步就退出
rm -f logs/ctrl.txt
echo "[restart] 已清空控制旗标"

setsid nohup bash scripts/linux/_graceful_stop.sh "$H" 240 10 \
  > logs/linux/tripwire.out 2>&1 </dev/null &
sleep 2
if pgrep -f '_graceful[_]stop' >/dev/null 2>&1; then
  echo "[restart] ✅ 新哨兵已启动：$(pgrep -f '_graceful[_]stop' | head -1)"
  head -2 logs/linux/tripwire.out
else
  echo "[restart] ❌ 新哨兵没起来"; tail -5 logs/linux/tripwire.out 2>/dev/null; exit 1
fi
