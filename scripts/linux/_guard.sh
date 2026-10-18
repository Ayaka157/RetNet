#!/usr/bin/env bash
# 守卫的安全起停。**存在的理由就是一次事故**：
#
# 2026-10-18 19:37，为了清理守卫残留的 sleep 进程，执行了
#     pkill -f "^sleep [0-9]{5}$"
# 而死手开关的形态是
#     bash -c "sleep 69840; sync; /usr/bin/shutdown; sleep 60; /usr/bin/shutdown"
# 把它的 sleep 杀掉后，**bash 立刻往下执行 shutdown**，实例当场关机，
# 训到 33,544 步的训练丢了 1,044 步（约 ¥4 和 17 分钟），并且需要人工开机。
#
# 教训：守护进程要以**进程组**为单位起停，绝不能单独杀它内部的 sleep——
# 那等于把 shell 往前推了一条命令。
#
# 用法：
#   bash scripts/linux/_guard.sh status
#   bash scripts/linux/_guard.sh stop  all
#   bash scripts/linux/_guard.sh start deadman 20h18m
#   bash scripts/linux/_guard.sh start tripwire 19h54m
#   bash scripts/linux/_guard.sh start watchdog 22
set -uo pipefail
cd /root/autodl-tmp/RetNet
LOG=logs/linux

# 把 18h30m / 90m / 3600 这类写法化成秒
to_seconds() {
  local s=$1
  if [[ "$s" =~ ^([0-9]+)h([0-9]+)m$ ]]; then echo $(( BASH_REMATCH[1]*3600 + BASH_REMATCH[2]*60 ));
  elif [[ "$s" =~ ^([0-9]+)h$ ]]; then echo $(( BASH_REMATCH[1]*3600 ));
  elif [[ "$s" =~ ^([0-9]+)m$ ]]; then echo $(( BASH_REMATCH[1]*60 ));
  else echo "$s"; fi
}

# 按**完整命令行关键字**找出进程组，整组杀掉。
# setsid 起的守卫各自是一个组，杀组 = 连它的子 sleep 一起杀，
# 父 bash 同时消失，所以不会出现"杀掉 sleep 反倒推进到 shutdown"的情况。
stop_one() {
  local pat=$1
  local pids
  pids=$(pgrep -f "$pat" | grep -v "^$$\$" || true)
  [[ -z "$pids" ]] && { echo "  [$pat] 没有在跑"; return 0; }
  for p in $pids; do
    local pg
    pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    if [[ -n "$pg" && "$pg" != "1" && "$pg" != "$$" ]]; then
      kill -TERM -- "-$pg" 2>/dev/null || kill -TERM "$p" 2>/dev/null
    else
      kill -TERM "$p" 2>/dev/null
    fi
  done
  sleep 1
  echo "  [$pat] 已停"
}

status() {
  echo "=== 守卫现状 ==="
  ps -eo pid,args | grep -E "(_graceful_stop|_watchdog|_disk_guard|shutdown)" \
    | grep -v "grep\|bash -c" | sed 's/^/  /'
  echo "=== 死手（bash -c 里的 sleep + shutdown）==="
  ps -eo pid,args | grep "sleep [0-9]*; sync" | grep -v grep | sed 's/^/  /'
  echo "=== 训练 ==="
  ps -eo pid,args | grep "[r]etnet_hf/train.py" | cut -c1-90 | sed 's/^/  /'
}

case "${1:-status}" in
  status) status ;;
  stop)
    case "${2:-all}" in
      deadman) stop_one "sync; /usr/bin/shutdown" ;;
      tripwire) stop_one "_graceful_stop[.]sh" ;;
      watchdog) stop_one "_watchdog[.]sh" ;;
      all)
        stop_one "_graceful_stop[.]sh"
        stop_one "_watchdog[.]sh"
        stop_one "sync; /usr/bin/shutdown"
        ;;
      *) echo "未知目标: $2（deadman|tripwire|watchdog|all）"; exit 2 ;;
    esac
    ;;
  start)
    case "${2:-}" in
      deadman)
        local_s=$(to_seconds "${3:?需要时长，如 20h18m}")
        setsid nohup bash -c "sleep $local_s; sync; /usr/bin/shutdown; sleep 60; /usr/bin/shutdown" \
          > "$LOG/deadman.out" 2>&1 < /dev/null &
        echo "  [deadman] 已挂 $((local_s/3600))h$(( local_s%3600/60 ))m -> $(date -d "+${local_s} seconds" '+%m-%d %H:%M')"
        ;;
      tripwire)
        # 实例上没有 python3（只有 /root/miniconda3/bin/python），用 awk 换算。
        # 原先写成 `python3 ... || echo 10`，在实例上会静默退化成 10 小时。
        h=$(awk -v s="${3:?需要时长秒数}" 'BEGIN{printf "%.4f", s/3600}')
        SKIP_FINALIZE=1 setsid nohup bash scripts/linux/_graceful_stop.sh "$h" 240 0 \
          > "$LOG/tripwire.out" 2>&1 < /dev/null &
        echo "  [tripwire] 已挂 ${3}s -> $(date -d "+${3} seconds" '+%m-%d %H:%M')"
        ;;
      watchdog)
        setsid nohup bash scripts/linux/_watchdog.sh "${3:?需要整点小时数}" 10 \
          > "$LOG/watchdog.out" 2>&1 < /dev/null &
        echo "  [watchdog] 已挂 ${3}h"
        ;;
      *) echo "用法: $0 start {deadman <时长>|tripwire <秒>|watchdog <整点小时>}"; exit 2 ;;
    esac
    ;;
  *) echo "用法: $0 {status|stop <目标>|start <目标> <参数>}"; exit 2 ;;
esac
