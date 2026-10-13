#!/usr/bin/env bash
# 训练日志摘要：从 logs/linux/ 里找最新（或指定）的训练日志，把 HF 的时间指标和
# 最后一条进度抽出来。tqdm 用 \r 刷新，所以必须先把 CR 换成 LF 再 grep。
#
#   bash scripts/linux/_tail_training.sh              # 最新一份
#   bash scripts/linux/_tail_training.sh <文件名>      # 指定
#
# 输出刻意保持"能直接看成结论"的形态：步时、剩余时间、loss。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2

logdir="logs/linux"
[[ -d "$logdir" ]] || { echo "没有 $logdir"; exit 2; }

if [[ $# -ge 1 ]]; then
  log="$logdir/$1"
else
  log="$(ls -1t "$logdir" 2>/dev/null | head -1)"
  log="$logdir/$log"
fi
[[ -f "$log" ]] || { echo "找不到日志：$log"; exit 2; }

echo "=== 日志: $log ==="
echo "--- 文件大小 / 最后写入 ---"
ls -l "$log" | awk '{print "  " $5 " 字节   " $6 " " $7 " " $8}'

# CR -> LF 后再处理，否则 tqdm 的整行都在一条"行"里
norm="$(mktemp)"
tr '\r' '\n' < "$log" > "$norm"

echo "--- 关键行（内核/优化器/退出）---"
grep -aE 'chunkwise kernel|optimizer:|TRAINING DONE|训练进程退出码|出错了|Error|error:' "$norm" | tail -6 | sed 's/^/  /'

echo "--- HF 时间指标 ---"
grep -a -oE "'train_runtime': [0-9.]+|'train_steps_per_second': [0-9.]+|'train_loss': [0-9.]+|'eval_loss': [0-9.]+" "$norm" | tail -8 | sed 's/^/  /'

echo "--- 最后 3 条进度（含步时）---"
grep -aE '[0-9]+/[0-9]+ \[' "$norm" | tail -3 | sed 's/^/  /'

echo "--- 最后 3 条 loss 记录 ---"
grep -aE "\{'loss'" "$norm" | tail -3 | sed 's/^/  /'

if [[ "${1:-}" == "--full" || "${2:-}" == "--full" ]]; then
  echo "--- 全文（CR 已转 LF，最后 60 行）---"
  tail -60 "$norm" | sed 's/^/  /'
fi

rm -f "$norm"
