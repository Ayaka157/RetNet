#!/usr/bin/env bash
# 训后一键收尾：终态核对 → 导出键检查 → loss/评测曲线 → 显存峰值 → 恢复验证 →（可选）关机
#
#   bash scripts/linux/_finalize.sh              # 只做检查
#   bash scripts/linux/_finalize.sh --shutdown   # 检查完关机
#
# 为什么要它：正式训练由 `_graceful_stop.sh` 在到点时优雅收尾，收尾后只剩约 20 分钟
# （之后哨兵会关机、余额也会在 05:38 耗尽）。这 20 分钟里要做的事全部固定，
# 与其临时敲命令，不如一条脚本跑完并留下可粘贴的汇总。
#
# 退出码：0 = 全部检查通过；1 = 有检查未过（恢复验证失败也会让它非零）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
R="$PWD"
OUT="checkpoints/retnet_500m_v15b"
SHUTDOWN=0
SKIP_RESUME=0
for a in "$@"; do
  case "$a" in
    --shutdown)    SHUTDOWN=1 ;;
    --skip-resume) SKIP_RESUME=1 ;;   # 训练还在跑时先验 1~6 段的解析，别碰 GPU
    *) echo "未知参数: $a"; exit 2 ;;
  esac
done

bad=0
# 选主训练日志：**按内容选，不按 mtime**。
# ⚠️ 2026-10-12 踩到：`_resume_test.sh` 也会通过 run_train.sh 生成一份 `train_*.log`，
# 于是 `ls -1t | head -1` 选中的是那次只跑 2 步的测试日志，第 4/5/9 段全读成 0 ——
# 报告"静默读了错文件"，比读不到更危险。现在选**步计数最大**的那份。
pick_log() {
  local best="" beststep=-1 f s
  for f in logs/linux/train_*.log; do
    [[ -f "$f" ]] || continue
    s="$(tr '\r' '\n' < "$f" | grep -aoE "[0-9]+/67390" | tail -1 | cut -d/ -f1)"
    [[ -n "${s:-}" ]] || continue
    if [[ "$s" -gt "$beststep" ]]; then beststep="$s"; best="$f"; fi
  done
  echo "$best"
}
LOG="$(pick_log)"
if [[ -n "$LOG" ]]; then
  echo "（主日志按步计数选中：$LOG）"
fi

echo "=========== 1. 终态日志 ==========="
if [[ -n "$LOG" ]]; then
  echo "日志: $LOG"
  tr '\r' '\n' < "$LOG" | grep -aE "\[ctrl\]|TRAINING DONE|训练进程退出码|已确认|Traceback|out of memory" | tail -10
  grep -aq "TRAINING DONE" "$LOG" && echo "  ✅ TRAINING DONE 在" || { echo "  ❌ 没有 TRAINING DONE"; bad=1; }
else
  echo "  ❌ 找不到训练日志"; bad=1
fi

echo
echo "=========== 2. 检查点 ==========="
ls -1d "$OUT"/checkpoint-* 2>/dev/null | sed 's#.*/##' | sort -t- -k2 -n | tr '\n' ' '; echo
n_ckpt="$(ls -1d "$OUT"/checkpoint-* 2>/dev/null | wc -l)"
echo "  共 $n_ckpt 个"

echo
echo "=========== 3. 最终模型导出 ==========="
ls -1 "$OUT"/*.safetensors "$OUT"/config.json 2>/dev/null | sed 's#.*/##' || true
if [[ -n "$LOG" ]]; then
  tr '\r' '\n' < "$LOG" | grep -a "\[verify\]" | tail -3
fi

echo
echo "=========== 4. 评测曲线（val loss）==========="
if [[ -n "$LOG" ]]; then
  tr '\r' '\n' < "$LOG" | grep -aoE "'eval_loss': [0-9.]+" | tail -14
  echo "  评测次数: $(tr '\r' '\n' < "$LOG" | grep -ac "eval_loss")"
fi

echo
echo "=========== 5. 训练 loss（每 20 步一条，抽头/中/尾）==========="
if [[ -n "$LOG" ]]; then
  tr '\r' '\n' < "$LOG" | grep -aoE "\{'loss': [0-9.]+" | awk 'NR%1==0' > /tmp/_losses.txt
  total="$(wc -l < /tmp/_losses.txt)"
  echo "  记录数: $total"
  if [[ "$total" -gt 0 ]]; then
    sed -n "1p;$(( total / 2 ))p;${total}p" /tmp/_losses.txt | sed 's/^/    /'
  fi
  tr '\r' '\n' < "$LOG" | grep -aoE "[0-9]+/67390" | tail -1 | sed 's/^/  末步数: /'
fi

echo
echo "=========== 6. 显存峰值（每 5 秒采样）==========="
if [[ -s logs/linux/_peak_v15b.txt ]]; then
  PEAK="$(cut -d, -f1 logs/linux/_peak_v15b.txt | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1)"
  TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
  echo "  采样点数: $(grep -c ',' logs/linux/_peak_v15b.txt)"
  echo "  峰值 memory.used = ${PEAK} MiB / 总量 ${TOTAL:-?} MiB（余量 $(( ${TOTAL:-0} - PEAK )) MiB）"
  if [[ -n "${TOTAL:-}" && "$PEAK" -ge "$TOTAL" ]]; then
    echo "  ❌ 峰值达到总量 —— 危险（可能已 OOM）"; bad=1
  fi
else
  echo "  ❌ 峰值文件为空"; bad=1
fi

echo
echo "=========== 7. 存档恢复验证（--resume）==========="
if [[ "$SKIP_RESUME" == "1" ]]; then
  echo "  ⏭  已按 --skip-resume 跳过（训练还在跑时不要碰 GPU）"
else
  bash scripts/linux/_resume_test.sh
  rv=$?
  [[ "$rv" == "0" ]] || bad=1
fi

echo
echo "=========== 8. 磁盘 ==========="
df -h /root/gpu-scratch | tail -1

echo
echo "=========== 9. 训练完整性与 token 口径 ==========="
# 目标文件：1 个 epoch = 67390 个优化步 = 269561 行 × 2048 token = 552,330,489 token
TARGET=67390
if [[ -n "$LOG" ]]; then
  LASTSTEP="$(tr '\r' '\n' < "$LOG" | grep -aoE "[0-9]+/${TARGET}" | tail -1 | cut -d/ -f1)"
  if [[ -n "$LASTSTEP" ]]; then
    TOK=$(( LASTSTEP * 8192 ))
    echo "  末步数 = ${LASTSTEP} / ${TARGET}"
    echo "  已训 token = ${TOK}（= ${LASTSTEP} 步 × 8192）"
    echo "  token/参数 = $(awk -v t="$TOK" 'BEGIN{printf "%.3f", t/504179896}')（参数量 504,179,896）"
    if [[ "$LASTSTEP" -ge "$TARGET" ]]; then
      echo "  ✅ 跑满 1 个 epoch（552,330,489 token，= 1.095 token/参数）"
    else
      echo "  ⚠️ 未跑满：只到 $(( LASTSTEP * 100 / TARGET ))% 个 epoch（用 --resume 可续）"
      bad=1
    fi
  else
    echo "  ❌ 日志里读不到步计数"; bad=1
  fi
fi

echo
if [[ "$bad" == "0" ]]; then
  echo "✅ 收尾检查全部通过"
else
  echo "❌ 有检查未过（见上面标 ❌ / ⚠️ 的行）"
fi

if [[ "$SHUTDOWN" == "1" ]]; then
  echo "→ 按 --shutdown 执行关机（数据保留 15 天）"
  sync
  /usr/bin/shutdown || echo "!! shutdown 调用失败，请手动关机"
fi
exit $(( bad > 0 ? 1 : 0 ))
