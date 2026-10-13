#!/usr/bin/env bash
# 采样 GPU 显存/功耗，直到训练进程退出 —— 用来抓「峰值」而不是碰运气看一眼。
#
#   bash scripts/linux/_peak_watch.sh /tmp/_peak.txt 5
#
# 正式训练跑到第 10 小时才发生的内存高峰（存档、评测、load_best）在短测里只有
# 采到才算证据。这里每 5 秒记一行 memory.used,power.draw，最后附一行最大值。
set -uo pipefail
OUT="${1:-/tmp/_peak.txt}"
INT="${2:-5}"
: > "$OUT"

# ⚠️ 第一版直接进 while 判断，而训练进程通常比采样器晚几秒才起来 —— 于是它
# 立刻判定「进程不在」就退出了，峰值文件是空的。先等进程出现（最多 300 秒）。
waited=0
until pgrep -f 'train.py --model_dir' >/dev/null 2>&1; do
  sleep 2; waited=$((waited + 2))
  if [[ $waited -ge 300 ]]; then
    echo "等不到训练进程（300 秒），退出" >> "$OUT"; exit 1
  fi
done

while pgrep -f 'train.py --model_dir' >/dev/null 2>&1; do
  nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits >> "$OUT" 2>/dev/null
  sleep "$INT"
done
if [[ -s "$OUT" ]]; then
  mx="$(cut -d, -f1 "$OUT" | tr -d ' ' | sort -n | tail -1)"
  # ⚠️ 不要硬编码"可用显存"。第一版这里写死了「可用 24119 MiB」，而那个数是我
  # 自己减出来的，没有任何测量来源 —— 与 docs/22 §0 第 21 条同一类错误。
  # 现在运行时从 nvidia-smi 读总量，余量当场算。
  total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
  echo "峰值显存 = ${mx} MiB / 总量 ${total:-?} MiB（余量 $(( ${total:-0} - mx )) MiB）" >> "$OUT"
fi
