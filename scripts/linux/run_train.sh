#!/usr/bin/env bash
# AutoDL / Linux 上的训练启动器 —— scripts\_run_*.ps1 的等价物。
#
# 它做的事，就是让「付费的 GPU 时段」不浪费在环境问题上：
#   1. 起跑前检查：GPU 是否可见/空闲、数据集与模型是否就位
#   2. 可选覆盖 save_steps / resume（**不改** tasks/ 下的正式配置，只生成临时副本）
#   3. 全程 tee 到带时间戳的日志
#   4. 可选 --max-hours：到点强杀（防止忘记关机烧钱）
#   5. 默认**训练结束后自动 /usr/bin/shutdown**（AutoDL 省钱绝招之一）
#
#   bash scripts/linux/run_train.sh --config tasks/v15_500m.json
#   bash scripts/linux/run_train.sh --resume --save-steps 1000 --max-hours 12
#   bash scripts/linux/run_train.sh --no-shutdown        # 调试时别关机
#
# 退出码：0 成功；2 前置检查不过；其它 = 训练进程的退出码

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="tasks/v15_500m.json"
RESUME=""
SAVE_STEPS=""
MAX_STEPS=""
OUT_DIR_OVERRIDE=""
MAX_HOURS="0"
SHUTDOWN="1"

# 找 python：AutoDL 的非交互 shell 里 `python3` **不在 PATH 上**（实测），
# 但解释器存在于 /root/miniconda3/bin/python。这里做一次兜底探测，
# 免得每条命令都得手写全路径。
find_python() {
  local c
  for c in "${PYTHON:-}" python3 python /root/miniconda3/bin/python /opt/conda/bin/python; do
    [[ -z "$c" ]] && continue
    if command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]]; then echo "$c"; return 0; fi
  done
  return 1
}
PYTHON="$(find_python)" || { echo "!! 找不到 python（试过 python3/python//root/miniconda3/bin/python）"; exit 2; }
export PATH="$(dirname "$(command -v "$PYTHON" || echo "$PYTHON")"):$PATH"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2"; shift 2 ;;
    --resume)     RESUME="--resume"; shift ;;
    --save-steps) SAVE_STEPS="$2"; shift 2 ;;
    --max-steps)  MAX_STEPS="$2"; shift 2 ;;
    --output-dir) OUT_DIR_OVERRIDE="$2"; shift 2 ;;
    --max-hours)  MAX_HOURS="$2"; shift 2 ;;
    --no-shutdown) SHUTDOWN="0"; shift ;;
    --python)     PYTHON="$2"; shift 2 ;;
    -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

# --max-hours 允许小数（bash 算术不允许，见第 4 步的注释），这里只校验它是正数
if [[ "$MAX_HOURS" != "0" ]] && ! awk -v h="$MAX_HOURS" 'BEGIN{exit !(h+0>0)}'; then
  echo "!! --max-hours 需要正数（可带小数，如 10.5），收到: $MAX_HOURS"; exit 2
fi

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
# RETNET_FAST_CHUNKWISE 由 run_task.py 依据配置里的 fast_chunkwise 自行设置，这里不碰

LOG_DIR="$REPO_ROOT/logs/linux"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/train_${STAMP}.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== run_train.sh 启动 ==="
say "repo    : $REPO_ROOT"
say "config  : $CONFIG"
say "python  : $($PYTHON -c 'import sys;print(sys.version.split()[0])' 2>&1)"
say "log     : $LOG"

# ---------- 1. 前置检查 ----------
fail=0
if ! command -v nvidia-smi >/dev/null 2>&1; then
  say "!! 没有 nvidia-smi：你是不是用【无卡模式】开机的？切回正常模式再跑。"; fail=1
else
  GPU_LINE="$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -1)"
  say "GPU     : $GPU_LINE"
  USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
  if [[ "${USED_MIB:-99999}" -gt 1500 ]]; then
    say "!! 显存已占用 ${USED_MIB} MiB —— 有别的进程占着卡，先关掉（同一时刻只跑一个作业）"; fail=1
  fi
fi

[[ -f "$CONFIG" ]] || { say "!! 配置不存在: $CONFIG"; fail=1; }
[[ -f run_task.py ]] || { say "!! 找不到 run_task.py（工作目录不对？）"; fail=1; }

DATA_DIR="$($PYTHON -c "import json;print(json.load(open('$CONFIG'))['data_dir'])" 2>/dev/null)"
MODEL_DIR="$($PYTHON -c "import json;print(json.load(open('$CONFIG'))['model_dir'])" 2>/dev/null)"
OUT_DIR="$($PYTHON -c "import json;print(json.load(open('$CONFIG'))['output_dir'])" 2>/dev/null)"
say "model   : $MODEL_DIR"
say "data    : $DATA_DIR"
say "out     : $OUT_DIR"

[[ -e "$DATA_DIR/train.ids.npy" ]] || { say "!! 数据集缺失: $DATA_DIR/train.ids.npy"; fail=1; }
if [[ ! -e "$MODEL_DIR" ]]; then say "!! 模型目录缺失: $MODEL_DIR"; fail=1; fi
[[ $fail -eq 0 ]] || { say "❌ 前置检查不过，退出。"; exit 2; }

# ---------- 2. 可选覆盖：只生成临时配置，不动 tasks/ ----------
RUN_CONFIG="$CONFIG"
if [[ -n "$RESUME" || -n "$SAVE_STEPS" || -n "$MAX_STEPS" || -n "$OUT_DIR_OVERRIDE" ]]; then
  RUN_CONFIG="$LOG_DIR/_run_config_${STAMP}.json"
  OVERRIDE_RESUME="$([[ -n "$RESUME" ]] && echo true || echo false)"
  OVERRIDE_SAVE="${SAVE_STEPS:-}"
  OVERRIDE_STEPS="${MAX_STEPS:-}"
  OVERRIDE_OUT="${OUT_DIR_OVERRIDE:-}"
  $PYTHON - "$CONFIG" "$RUN_CONFIG" "$OVERRIDE_RESUME" "$OVERRIDE_SAVE" "$OVERRIDE_STEPS" "$OVERRIDE_OUT" <<'PY'
import json, sys
src, dst, resume, save, steps, outdir = sys.argv[1:7]
cfg = json.load(open(src, encoding="utf-8"))
if resume == "true":
    cfg["resume"] = True
if save:
    cfg["save_steps"] = int(save)
    cfg["eval_steps"] = int(save)
if steps:
    cfg["max_steps"] = int(steps)
if outdir:
    cfg["output_dir"] = outdir
json.dump(cfg, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[override] resume={cfg.get('resume')} save_steps={cfg.get('save_steps')} "
      f"max_steps={cfg.get('max_steps')} output_dir={cfg.get('output_dir')}")
PY
  OUT_DIR="$( $PYTHON -c "import json;print(json.load(open('$RUN_CONFIG'))['output_dir'])" )"
  say "临时配置: $RUN_CONFIG（原配置未改动）"
fi

# ---------- 3. 干跑一遍，把将要执行的命令行打印出来 ----------
# ⚠️ 2026-10-11 修复：原来写的是 `if ! $PYTHON run_task.py --dry 2>&1 | tee -a "$LOG"`，
# 管道让 `if` 判定的是 **tee** 的退出码 —— tee 几乎永远返回 0，所以这道闸门从来没
# 拦住过任何东西（又一处「不可能失败的检查」）。现在取 PIPESTATUS[0]。
say "--- 干跑校验配置 ---"
$PYTHON run_task.py --config "$RUN_CONFIG" --dry 2>&1 | tee -a "$LOG"
if [[ "${PIPESTATUS[0]}" != "0" ]]; then
  say "!! 配置校验失败（死字段 / 未知 flag），退出"; exit 2
fi

# ---------- 4. 正式训练 ----------
CMD=("$PYTHON" run_task.py --config "$RUN_CONFIG")
if [[ -n "$RESUME" ]]; then CMD+=(--resume); fi

if [[ "$MAX_HOURS" != "0" ]]; then
  # ⚠️ 2026-10-11 修复：以前写 `SECS=$(( MAX_HOURS * 3600 ))`。bash 的算术只吃整数，
  # 传 `--max-hours 10.9` 会得到 `syntax error: invalid arithmetic operator (error
  # token is ".9")`。因为脚本是 `set -uo pipefail`（没有 -e），错误只让 SECS 变空、
  # 脚本继续往下跑，最后 timeout 拿到空值 —— **训练进程一次都没启动，日志里却看不出**
  # 「没跑」。实测就踩到了。现在用 awk 支持小数，并在入口直接拒绝非法值。
  SECS="$(awk -v h="$MAX_HOURS" 'BEGIN{printf "%d", h*3600}')"
  if [[ -z "$SECS" || "$SECS" -le 0 ]]; then
    say "!! --max-hours 解析失败（收到 $MAX_HOURS），退出"; exit 2
  fi
  say "--- 开始训练（硬上限 ${MAX_HOURS} 小时，到点强杀）---"
  timeout -k 120 "$SECS" "${CMD[@]}" 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  if [[ "$RC" == "124" || "$RC" == "137" ]]; then
    say "⏰ 到达 ${MAX_HOURS} 小时上限，已强杀。检查点已按 save_steps 保存，可 --resume 续训。"
  else
    say "训练进程退出码: $RC"
  fi
else
  say "--- 开始训练（无时长上限）---"
  "${CMD[@]}" 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  say "训练进程退出码: $RC"
fi

# ---------- 4b. 正向复核：rc=0 不等于训完了 ----------
# 2026-10-11 实测过一例：train.py 崩在 TrainingArguments 校验上，日志里有 traceback
# 和 `[exit] rc=1`，而 run_task.py 仍以 0 退出（子进程退出码没往外传），于是
# 「崩了」被记成「训完了」。那次是我自己引入配置错误撞出来的，但换任何别的崩溃
# 一样会被吞。所以这里不信 rc，只认 train.py 自己打的完成标记。
if [[ "$RC" == "0" ]]; then
  if grep -aq "TRAINING DONE" "$LOG"; then
    say "✅ 已确认 TRAINING DONE —— 训练完整跑完（train + save_model + verify_export）"
  else
    say "!! rc=0 但日志里找不到 TRAINING DONE —— 按失败处理，不自动关机"
    RC=1
  fi
fi

# ---------- 5. 打印最后的检查点，然后关机 ----------
say "--- 已保存的检查点 ---"
ls -1dt "$OUT_DIR"/checkpoint-* 2>/dev/null | head -5 | tee -a "$LOG" || say "（还没有检查点）"

if [[ "$SHUTDOWN" == "1" ]]; then
  say "训练结束 —— 执行 /usr/bin/shutdown 自动关机（省钱；数据保留 15 天）"
  sync
  /usr/bin/shutdown || say "!! shutdown 调用失败，请手动关机"
else
  say "--no-shutdown 已指定，保持开机（记得手动关机；单价=${PRICE_PER_HOUR:-未设置，看实例计费}）"
fi

exit "$RC"
