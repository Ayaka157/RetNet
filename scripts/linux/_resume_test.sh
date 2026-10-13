#!/usr/bin/env bash
# 存档恢复验证 —— 目标第 (4) 项要求「含存档与 --resume」的冒烟验证。
#
# 为什么必须补：翻遍实例上 12 个训练日志，**没有任何一条 `[resume]` 行**。
# 也就是说 `--resume` 这条路径此前从未被真正跑过，第 (4) 项只有「存档」那一半
# 有证据。而正式训练是可恢复性唯一的兜底，不验证等于没兜底。
#
# 做法：把正式训练最后一个 checkpoint 复制到独立目录，从它 `--resume` 跑 2 个优化步。
# `--save-steps 1` 会同时把 eval_steps 设成 1，于是这 2 步里会做 2 次评测 + 1 次存档，
# **且存档目录名与 trainer_state.json 的 global_step 会直接暴露步计数是否续上**。
#
# 四条判据（全过才算通过；每一条都能失败）：
#   1. 日志出现 train.py 自己打的「[resume] 从 .../checkpoint-N 续训」
#   2. 新存档是 checkpoint-(N+2)，且它的 trainer_state.json 里 global_step == N+2
#      —— 若没续上，global_step 会是 2
#   3. 日志出现 TRAINING DONE
#   4. 进程退出码 0
#
# 另有一条**天然**判据：若真的没 resume，它要从 step 1 跑 N+2 ≈ 6.7 万步、
# 花 11 小时，脚本不可能在几分钟内回来。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
R="$PWD"
SRC="checkpoints/retnet_500m_v15b"
DEST="checkpoints/_resume_test"
LOG="$R/logs/linux/_resume_test.out"
PY="${PYTHON:-/root/miniconda3/bin/python}"

# 删除前先确认路径就是我们要的那个。
# ⚠️ 2026-10-12 踩到：第一版只写了 `*/checkpoints/_resume_test`，而 $DEST 是**相对路径**
# `checkpoints/_resume_test`（前面没有斜杠），于是 case **不匹配 → 自己把自己拦下**，
# 恢复验证从来没跑过。防呆写太窄就变成了故障源本身。
# 现在同时接受相对/绝对两种写法，并把基准目录也钉死。
DEST_ABS="$PWD/$DEST"
case "$DEST" in
  checkpoints/_resume_test|*/checkpoints/_resume_test) ;;
  *) echo "❌ 拒绝操作可疑路径: $DEST"; exit 2 ;;
esac
case "$DEST_ABS" in
  "$R"/checkpoints/_resume_test) ;;
  *) echo "❌ 拒绝操作可疑绝对路径: $DEST_ABS"; exit 2 ;;
esac

LAST="$(ls -1d "$SRC"/checkpoint-* 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)"
if [[ -z "${LAST:-}" ]]; then
  echo "❌ 找不到 $SRC/checkpoint-*，无法验证恢复"; exit 2
fi
NEXT=$(( LAST + 2 ))
echo "=== 恢复验证：源 checkpoint-$LAST → --resume 跑到 max_steps=$NEXT ==="

rm -rf "$DEST"; mkdir -p "$DEST"
cp -r "$SRC/checkpoint-$LAST" "$DEST/" || { echo "❌ 复制检查点失败"; exit 2; }

# 记下跑之前已有的 train_*.log，跑完把**新增的那份**改名。
# 否则这份 2 步的测试日志会成为"最新日志"，把 _finalize.sh 的选择带偏
# （它现在按步计数选，但双重保险：污染源直接改名）。
BEFORE="$(ls -1 logs/linux/train_*.log 2>/dev/null | sort)"

bash scripts/linux/run_train.sh --config tasks/v15b_500m_24g.json --resume \
  --max-steps "$NEXT" --save-steps 1 --output-dir "$DEST" --no-shutdown > "$LOG" 2>&1
rc=$?

AFTER="$(ls -1 logs/linux/train_*.log 2>/dev/null | sort)"
for f in $(comm -13 <(echo "$BEFORE") <(echo "$AFTER") 2>/dev/null); do
  mv "$f" logs/linux/_resume_test_run.log && echo "（恢复验证日志已改名为 _resume_test_run.log，避免污染 finalize 的选择）"
done

pass=0; fail=0
say_ok() { echo "  ✅ $1"; pass=$(( pass + 1 )); }
say_no() { echo "  ❌ $1"; fail=$(( fail + 1 )); }

# 判据 1
if grep -aq "\[resume\] 从 .*checkpoint-$LAST 续训" "$LOG"; then
  say_ok "train.py 报告从 checkpoint-$LAST 续训"
else
  say_no "日志里没有「[resume] 从 .../checkpoint-$LAST 续训」"
fi

# 判据 2：新存档 + global_step
NEWCKPT="$DEST/checkpoint-$NEXT"
if [[ -f "$NEWCKPT/trainer_state.json" ]]; then
  got="$("$PY" - "$NEWCKPT/trainer_state.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("global_step"))
PY
)"
  if [[ "$got" == "$NEXT" ]]; then
    say_ok "新存档 checkpoint-$NEXT 的 global_step = $got（步计数确实续上了）"
  else
    say_no "新存档存在但 global_step = $got，期望 $NEXT"
  fi
else
  say_no "没有生成 $NEWCKPT/trainer_state.json，无法确认步计数"
fi

# 判据 3
if grep -aq "TRAINING DONE" "$LOG"; then
  say_ok "打印了 TRAINING DONE"
else
  say_no "没有 TRAINING DONE"
fi

# 判据 4
if [[ "$rc" == "0" ]]; then
  say_ok "退出码 0"
else
  say_no "退出码 $rc（期望 0）"
fi

echo
if [[ "$fail" -eq 0 ]]; then
  echo "✅ 恢复验证通过（$pass/4）—— 交付的 checkpoint 确实可用 --resume 续训"
else
  echo "❌ 恢复验证失败（$fail 项未过，$pass 项通过）；日志末尾："
  tr '\r' '\n' < "$LOG" | tail -25
fi
exit $(( fail > 0 ? 1 : 0 ))
