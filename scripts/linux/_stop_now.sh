#!/usr/bin/env bash
# 立即优雅停止正在跑的训练：先 save（让检查点落在当前步），再 stop（步边界退出）。
#
#   bash scripts/linux/_stop_now.sh
#
# 用途：需要**改启动参数重启**时（例如加了余额、要把 --max-hours 放宽），
# 先用它把最近的训练进度存成一个检查点，再从那里 --resume，比硬杀少丢几步。
#
# 与 _graceful_stop.sh 的区别：那个是「定时到点才动手」，这个是「立刻动手」。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-/root/miniconda3/bin/python}"

if ! pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1; then
  echo "[stop] 没有正在跑的训练进程"; exit 0
fi

echo "[stop] $(date '+%H:%M:%S') 请求 save（下一次步边界存检查点）"
"$PY" run_task.py --ctrl save || echo "[stop] !! save 写入失败"

sleep 90

echo "[stop] $(date '+%H:%M:%S') 请求 stop（步边界优雅退出）"
"$PY" run_task.py --ctrl stop || echo "[stop] !! stop 写入失败"

for _ in $(seq 1 60); do
  pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1 || break
  sleep 10
done

if pgrep -f 'train[.]py --model_dir' >/dev/null 2>&1; then
  echo "[stop] ❌ 10 分钟后训练进程仍在 —— stop 没生效（这是唯一没在实跑里验过的分支）"
  exit 1
fi
echo "[stop] ✅ $(date '+%H:%M:%S') 训练进程已退出"
