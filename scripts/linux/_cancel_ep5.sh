#!/usr/bin/env bash
# 撤销 ep5 接力：把「今晚 21:17 自动再练 48,000 步」的计划取消。
# ep4 本身不动（它已经跑了 1/3，杀掉等于把今天的钱全扔了），但改成：
#   ep4 跑完 → watchdog 收尾 → 关机，不再起新的一轮。
set -uo pipefail
cd /root/autodl-tmp/RetNet

echo "=== 1) 停掉 ep5 接力链（防止 21:17 自动开跑）==="
pkill -f '_chain_ep5.sh' && echo "  已停接力链" || echo "  接力链未找到"

echo "=== 2) 停掉为 ep5 挂的远期死手开关（明天 05:35）==="
pkill -f 'sleep 48768' && echo "  已停 48768s 死手" || echo "  未找到（可能 PID 不同，下面按模式再扫）"
for p in $(pgrep -f 'usr/bin/shutdown'); do
  # 只杀 sleep 在 3 万秒以上的远期死手，别碰别的
  echo "  发现 shutdown 包装进程 $p"
done

echo "=== 3) 重挂：ep4 退出后 finalize + 关机；外加今天 21:40 的绝对死手 ==="
DEADLINE=$(date -d 'today 21:40' +%s)
NOW=$(date +%s)
SECS=$((DEADLINE - NOW))
echo "  现在 $(date '+%H:%M:%S')；新死手在 $(date -d @$DEADLINE '+%m-%d %H:%M:%S')（${SECS}s 后）"
setsid nohup bash -c "sleep $SECS; sync; /usr/bin/shutdown; sleep 60; /usr/bin/shutdown" \
  > logs/linux/deadman_ep4_final.out 2>&1 &

setsid nohup bash scripts/linux/_watchdog.sh 6 10 > logs/linux/watchdog_ep4_final.out 2>&1 &
echo "  watchdog 已挂（ep4 退出 → finalize → 10 分钟后关机）"

echo "=== 4) 复核 ==="
echo "  ep4 train.py：$(pgrep -f 'train[.]py --model_dir' | wc -l) 个"
echo "  接力链：$(pgrep -f '_chain_ep5' | wc -l) 个（应为 0）"
echo "  watchdog：$(pgrep -f '_watchdog.sh 6 10' | wc -l) 个（应为 1）"
pgrep -af 'sleep [0-9]{4,}' | sed 's/\(.\{55\}\).*/\1.../'
