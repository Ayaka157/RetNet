#!/usr/bin/env bash
# v21 -> v22 自动接力：等 v21 训完 → 收尾评测 → gamma 重初始化 → 起 8k 长上下文扩展。
#
#   setsid nohup bash scripts/linux/_chain_v22.sh > logs/linux/chain_v22.out 2>&1 &
#
# 设计要点：
#   1. 用 `ps` 轮询训练进程消失，而不是读日志里某一行——日志格式改过一次。
#   2. gamma 重初始化是 **v22 的前置条件**：v21 里 gamma 被 weight decay 拉到了 0.543
#      （半衰期约 1 token），不重置就直接拉长窗口等于白训。
#   3. v22 用 `--no_decay_1d`，否则重初始化完还会被拉下去。
#   4. 两个阶段各自挂守卫，不要一个守卫跨两个 run。
set -uo pipefail
cd /root/autodl-tmp/RetNet
PY=/root/miniconda3/bin/python
LOGD=logs/linux

echo "=== 等 v21 训练进程结束 ==="
while pgrep -f "retnet_hf/train[.]py" > /dev/null; do
  sleep 60
done
echo "  $(date '+%m-%d %H:%M:%S') v21 训练进程已退出"

# ⚠️ 必须**立刻**停掉 v21 的守卫，不能等 v22 启动之后再停：
#    v21 的 tripwire 是按预计完成时间挂的，很可能正好落在
#    "v21 已退出 → v22 还没起" 这段收尾窗口里；那时它会去停一个已经不存在的训练进程。
echo "=== 0) 停掉 v21 的守卫 ==="
bash scripts/linux/_guard.sh stop all

echo "=== 1) v21 收尾（导出 + 域内评测）==="
bash scripts/linux/_finalize_v21.sh 2>&1 | tail -30

echo "=== 2) v21 权重做 gamma 重初始化 ==="
SRC=checkpoints/retnet_500m_v21
if [ ! -f "$SRC/model.safetensors" ]; then
  CK=$(ls -d "$SRC"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  echo "  顶层没有导出，用 $CK"
  SRC="$CK"
fi
$PY -u scripts/reinit_gamma.py --model "$SRC" \
    --out checkpoints/retnet_500m_v22_init --formula logspace --tau-max 32000 2>&1 | tail -8

echo "=== 3) 启动 v22（8k 上下文扩展）==="
source /etc/network_turbo >/dev/null 2>&1
rm -f logs/ctrl.txt
setsid nohup $PY run_task.py --config tasks/v22_ctx16k.json \
  > "$LOGD/v22_ctx16k.out" 2>&1 < /dev/null &
echo "  日志: $LOGD/v22_ctx16k.out"

echo "=== 4) 等它真正开始（编译预热约 8-10 分钟）==="
for i in $(seq 1 80); do
  sleep 15
  if grep -aq "it/s" "$LOGD/v22_ctx16k.out" 2>/dev/null; then
    echo "  $(($i*15))s 后出现进度条"
    break
  fi
  if grep -aqiE "Traceback|Error" "$LOGD/v22_ctx16k.out" 2>/dev/null; then
    echo "  !! v22 启动报错"
    grep -aiE "Traceback|Error" "$LOGD/v22_ctx16k.out" | head -6
    exit 1
  fi
done
grep -aE "no_decay|\[plan\]|DDP:" "$LOGD/v22_ctx16k.out" | tail -5

echo "=== 5) v22 守卫 ==="
# 时间窗要**贴紧 v22 的实际时长**（约 1.25 h），否则训练跑完后实例会空转到守卫到点：
# 4 小时的 watchdog 意味着 ¥39 的空转账单，而那时余额只剩个位数。
bash scripts/linux/_guard.sh stop all
bash scripts/linux/_guard.sh start deadman 1h50m
bash scripts/linux/_guard.sh start tripwire 5700     # 1h35m
bash scripts/linux/_guard.sh start watchdog 2

echo "=== 6) v22 跑完后收尾 ==="
while pgrep -f "retnet_hf/train[.]py" > /dev/null; do
  sleep 60
done
echo "  v22 训练进程已退出，做最后核对"
$PY -u scripts/eval_holdout.py --model checkpoints/retnet_500m_v22 \
    --tag v22_ctx16k --batch 2 2>&1 | tail -8
echo "  --- 与 ep3 基线对比 ---"
PYBIN=$PY
$PYBIN - <<'PYEOF'
import json
recs = {}
for line in open("logs/holdout_eval.jsonl", encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        recs[r["tag"]] = r
base = recs.get("ep3_full", {}).get("loss")
for tag in ("v21_final", "v22_ctx16k"):
    r = recs.get(tag)
    if r and base:
        drop = (base - r["loss"]) / base * 100
        print("  %-14s %.4f   降 %.1f%%" % (tag, r["loss"], drop))
PYEOF
df -h /root/autodl-tmp | tail -1
echo "=== 接力链完成 ==="
