#!/usr/bin/env bash
# 控制实验：run_train.sh 必须能分辨成功与三种失败。
#
# ## 为什么需要它
#
# 2026-10-11 实跑 100 步测速时发现两处「不可能失败的检查」：
#   1. run_task.py 把子进程的 rc=1 吞掉，自己以 0 退出 → run_train.sh 看到
#      「训练进程退出码: 0」，于是「训练崩了」被记成「训完了」，还会接着自动关机。
#   2. run_train.sh 的干跑闸门写成 `if ! $PY --dry | tee` —— 判定的是 tee 的退出码，
#      从来没拦住过任何东西。
#
# ## 四个用例（每个都能失败，不是摆设）
#
#   ok     伪造成「跑完」          → 期望 rc=0 且日志里有 TRAINING DONE
#   fail   伪造成「rc=1 崩溃」     → 期望 rc!=0
#   silent 伪造成「rc=0 但没跑完」 → 期望 rc!=0 且提示找不到 TRAINING DONE
#   drybad 伪造成「配置校验不过」  → 期望 rc=2（干跑闸门拦住）
#
# 前置检查（nvidia-smi / 数据集 / 模型）照常跑，所以要在**实例上**执行；
# 但它不碰 GPU、不训练，几秒结束，不产生任何算力开销。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export REAL_PY="${REAL_PY:-/root/miniconda3/bin/python}"
STUB="$PWD/scripts/linux/_fake_python.sh"
chmod +x "$STUB"

NAMES=(ok fail silent drybad)
FAKE_MODES=(ok fail silent ok)
FAKE_DRYS=(ok ok ok bad)
EXPECT_RC=(0 1 1 2)
EXPECT_TXT=("已确认 TRAINING DONE" "训练进程退出码" "找不到 TRAINING DONE" "配置校验失败")

pass=0
fail=0
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  out="$( FAKE_MODE="${FAKE_MODES[$i]}" FAKE_DRY="${FAKE_DRYS[$i]}" \
          bash scripts/linux/run_train.sh \
            --config tasks/v15b_500m_24g.json \
            --python "$STUB" \
            --output-dir /tmp/_fake_out \
            --no-shutdown 2>&1 )"
  rc=$?

  ok_rc=1
  [[ "$rc" == "${EXPECT_RC[$i]}" ]] && ok_rc=0
  ok_txt=1
  grep -qF "${EXPECT_TXT[$i]}" <<<"$out" && ok_txt=0

  if [[ $ok_rc -eq 0 && $ok_txt -eq 0 ]]; then
    echo "  ✅ $name: rc=$rc（期望 ${EXPECT_RC[$i]}），日志含「${EXPECT_TXT[$i]}」"
    pass=$((pass+1))
  else
    if [[ $ok_rc -eq 0 ]]; then rc_note="对"; else rc_note="错"; fi
    if [[ $ok_txt -eq 0 ]]; then txt_note="含"; else txt_note="缺"; fi
    echo "  ❌ $name: rc=$rc（期望 ${EXPECT_RC[$i]}，$rc_note）"
    echo "        日志${txt_note}「${EXPECT_TXT[$i]}」"
    echo "        ---- 末尾 12 行 ----"
    tail -12 <<<"$out" | sed 's/^/        /'
    fail=$((fail+1))
  fi
done

echo
if [[ $fail -eq 0 ]]; then
  echo "✅ $pass/4 全部通过 —— run_train.sh 能分辨真成功与三种失败"
  exit 0
else
  echo "❌ $fail/4 用例失败"
  exit 1
fi
