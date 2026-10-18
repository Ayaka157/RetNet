#!/usr/bin/env bash
# 把 ep4 的旧守卫全部换掉，改挂 ep5 的接力链 + 新死手开关。
#
# 必须杀掉旧守卫的三个原因：
#   1) _graceful_stop.sh 8.0 会在 21:14 到点，截断 ep4 最后的几步并留下 ctrl 旗标干扰 ep5
#   2) _watchdog.sh 9 10 会在 ep4 退出后跑 finalize 并关机，接力链根本轮不到
#   3) 旧死手开关是绝对 21:50，到点无条件关机
set -uo pipefail
cd /root/autodl-tmp/RetNet

echo "=== 1) 停掉旧守卫（按命令行模式匹配，不用 PID，避免 PID 复用误杀）==="
pkill -f '_graceful_stop.sh 8.0'  && echo "  已停 tripwire 脚本"  || echo "  tripwire 脚本未找到"
pkill -f '_watchdog.sh 9 10'      && echo "  已停 watchdog"       || echo "  watchdog 未找到"
pkill -f 'sleep 28800'            && echo "  已停 tripwire 的 sleep"  || echo "  sleep 28800 未找到"
pkill -f 'sleep 30918'            && echo "  已停旧死手开关"      || echo "  sleep 30918 未找到"
sleep 2

echo "=== 2) 复核残留 ==="
if pgrep -af '_graceful_stop|_watchdog' | grep -q .; then
  pgrep -af '_graceful_stop|_watchdog' | sed 's/\(.\{60\}\).*/\1.../'
else
  echo "  tripwire / watchdog 均已清空"
fi

echo "=== 3) 挂新的绝对时间死手开关：明天 05:35 ==="
DEADLINE=$(date -d 'tomorrow 05:35' +%s)
NOW=$(date +%s)
SECS=$((DEADLINE - NOW))
echo "  现在 $(date '+%m-%d %H:%M:%S')；死手在 $(date -d @$DEADLINE '+%m-%d %H:%M:%S')（${SECS}s 后）"
setsid nohup bash -c "sleep $SECS; sync; /usr/bin/shutdown; sleep 60; /usr/bin/shutdown" \
  > logs/linux/deadman_ep5.out 2>&1 &

echo "=== 4) 拉起 ep5 接力链 ==="
setsid nohup bash scripts/linux/_chain_ep5.sh > logs/linux/chain_ep5.out 2>&1 &
sleep 3
echo "  接力链进程："
pgrep -af '_chain_ep5' | sed 's/\(.\{70\}\).*/\1.../'

echo "=== 5) 当前 ep4 仍在跑，确认没被误伤 ==="
pgrep -f 'train[.]py --model_dir' >/dev/null && echo "  ep4 train.py 仍在运行 ✓" || echo "  ⚠️ ep4 train.py 不见了！"
tail -c 400 logs/linux/v18_ep4.out | tr "\r" "\n" | grep -a "it/s" | tail -1
