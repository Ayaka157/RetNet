#!/usr/bin/env bash
# v21 续训：误关机后从最近 checkpoint 恢复，并把守卫、数据监督器一起拉起来。
#
#   bash scripts/linux/_resume_v21.sh
#
# 设计要点（都是踩过的坑）：
#   1. **先验证 checkpoint 可加载**再起训练，否则会白编译 8 分钟后才崩。
#   2. 守卫用 `_guard.sh` 挂（按进程组），**绝不单独杀 sleep** ——
#      2026-10-18 就是 `pkill -f "^sleep …"` 把死手的 sleep 杀掉，
#      导致 `bash -c "sleep …; shutdown"` 立刻执行 shutdown，实例当场关机。
#   3. 训练起后再挂守卫，避免守卫先到点。
#   4. code_ling2 段需要 hf_pack_code2，若还没分词完，同时把监督器拉起来。
set -uo pipefail
cd /root/autodl-tmp/RetNet
PY=/root/miniconda3/bin/python
OUT=checkpoints/retnet_500m_v21
LOG=logs/linux/v21_ddp.out

echo "=== 0) 前置检查 ==="
NVIDIA=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "  GPU: $NVIDIA 张"
[[ "$NVIDIA" -ge 2 ]] || { echo "  !! 不足 2 张卡"; exit 2; }
df -h /root/autodl-tmp | tail -1 | sed 's/^/  /'

CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[[ -n "$CKPT" ]] || { echo "  !! 没有 checkpoint，无法续训"; exit 3; }
echo "  最近 checkpoint: $CKPT"

echo "=== 1) 验证 checkpoint 可加载（不跑完整前向，只查键）==="
CKPT="$CKPT" "$PY" - <<PYCHECK
import glob, os
from safetensors.torch import load_file
p = os.environ["CKPT"]
f = glob.glob(os.path.join(p, "*.safetensors"))
if not f:
    raise SystemExit("  !! checkpoint 里没有 safetensors")
d = load_file(f[0])
bad = sum(1 for k in d if k.startswith("_orig_mod."))
print(f"  {len(d)} 个键，带 _orig_mod. 前缀 {bad} 个")
need_opt = os.path.exists(os.path.join(p, "optimizer.pt"))
need_sch = os.path.exists(os.path.join(p, "scheduler.pt"))
print(f"  optimizer.pt={need_opt}  scheduler.pt={need_sch}")
if bad or not need_opt:
    raise SystemExit("  !! checkpoint 不完整")
PYCHECK

echo "=== 2) 清理旧进程（按进程组）==="
bash scripts/linux/_guard.sh stop all 2>/dev/null || true
for p in $(pgrep -f "retnet_hf/train[.]py"); do kill $p 2>/dev/null; done
sleep 2

echo "=== 3) 启动续训 ==="
rm -f logs/ctrl.txt
setsid nohup "$PY" run_task.py --config tasks/v21_resume.json > "$LOG" 2>&1 < /dev/null &
echo "  日志: $LOG"

echo "=== 4) 等它真正开始（编译预热约 8 分钟）==="
for i in $(seq 1 60); do
  sleep 15
  if grep -aq "it/s" "$LOG" 2>/dev/null; then
    echo "  $(($i*15))s 后已出现进度条"
    break
  fi
  if grep -aqiE "Traceback|Error" "$LOG" 2>/dev/null; then
    echo "  !! 启动报错"
    grep -aiE "Traceback|Error" "$LOG" | head -5
    exit 4
  fi
done
grep -aE "\[resume\]|\[plan\]|DDP:" "$LOG" | tail -6

echo "=== 5) 挂守卫（按进程组）==="
# 计划：148,000 步，已到约 32,500，剩 115,500 步 / 1.607 it/s ≈ 20.0 h
bash scripts/linux/_guard.sh start deadman 20h40m
bash scripts/linux/_guard.sh start tripwire 71928     # 19h58m
bash scripts/linux/_guard.sh start watchdog 24

echo "=== 6) 数据监督器（code_ling2 还需要 hf_pack_code2）==="
if [ ! -d data_cache/hf_pack_code2 ]; then
  source /etc/network_turbo >/dev/null 2>&1
  setsid nohup "$PY" -u scripts/_data_supervisor.py > logs/linux/data_supervisor.out 2>&1 < /dev/null &
  echo "  已拉起监督器（准备 code_ling2）"
else
  echo "  hf_pack_code2 已存在"
fi

echo "=== 7) 状态 ==="
sleep 20
tr '\r' '\n' < "$LOG" | grep -a '| [0-9]*/' | tail -1
bash scripts/linux/_guard.sh status
