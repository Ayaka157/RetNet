#!/usr/bin/env bash
# 在租来的 24 GB 卡上量「免检查点到底快多少」——把 docs 里那 1.96× 的**推算**换成实测。
#
# 为什么必须去云上量：本机 8 GB 卡连「免检查点」都开不起来（直接 OOM），
# 所以这个数一直是外推的。24 GB 卡才能把它测出来。
#
#   bash scripts/linux/speed_probe.sh
#   bash scripts/linux/speed_probe.sh --model checkpoints/retnet_500m --timeout 600
#
# 设计要点（照搬本机 scripts\_run_opt500m.ps1 的加固）：
#   · 每个探针**独立超时**，到点强杀（不让一个挂死的配置吃掉整段租金）
#   · **可续跑**：跑完的记进 completed.txt，重跑自动跳过
#   · 每个探针前查显存是否真的空出来（防上一个进程没退干净）
#   · 结果与日志分开存放，事后可回溯
#
# 探针清单为什么是这几个：
#   ckpt         —— 带检查点的基线，口径与本机 1,887 ms 一致（但**跨机不可比**，只作参照）
#   nockpt       —— 免检查点，batch1×accum4。**这是核心测量**
#   nockpt_b2a2  —— 免检查点，batch2×accum2。每步 token 数与上一条相同（8192），
#                   但微步 M=4096，GEMM 效率更高；24 GB 装得下
#   不含 batch>=4：batch4 的激活约 24 GiB，加上 state 4.81 GiB 必然 OOM（算过，不试）

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="checkpoints/retnet_500m"
DATA="data_cache/chat_pack_v13"
OUT="logs/opt_linux"
TIMEOUT=600
STEPS=30
WARM=12
PRICE_PER_HOUR="${PRICE_PER_HOUR:-1.32}"     # AutoDL RTX 3090 按量价，¥/小时

# 找 python：AutoDL 的非交互 shell 里 `python3` **不在 PATH 上**（实测），
# 但解释器确实存在于 /root/miniconda3/bin/python。所以这里做一次兜底探测。
find_python() {
  local c
  for c in "${PYTHON:-}" python3 python /root/miniconda3/bin/python /opt/conda/bin/python; do
    [[ -z "$c" ]] && continue
    if command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]]; then echo "$c"; return 0; fi
  done
  return 1
}
PYTHON="$(find_python)" || { echo "!! 找不到 python 解释器，试过 python3/python//root/miniconda3/bin/python"; exit 2; }
# 让子进程也能找到（bench_trainer.py 里可能 spawn 子进程）
export PATH="$(dirname "$(command -v "$PYTHON" || echo "$PYTHON")"):$PATH"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --data)    DATA="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --price)   PRICE_PER_HOUR="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

export PYTHONIOENCODING=utf-8
# ★ 关键：bench_trainer.py 不会自己设这个开关（只有 run_task.py 会依据配置设）。
#   漏掉它就会跑**参考 per-chunk 循环**：那是 Python 里的 16 次循环，
#   叠上 torch.compile 的反复重编译，实测 600 秒连一步都跑不完（本机同配置只要 94 秒）。
#   这不是"慢一点"，是数量级的差别 —— 所以下面还加了一道内核断言。
export RETNET_FAST_CHUNKWISE=1
mkdir -p "$OUT"
RESULTS="$OUT/results.txt"
DONE="$OUT/completed.txt"
touch "$DONE"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RESULTS"; }

free_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

# 从 bench_trainer 的输出里取指标（这些标签是本机实测过的原文）
parse() {
  local log="$1"
  OPT_MS=$(grep -oE '[0-9]+ ms/[0-9]+ token' "$log" | head -1 | grep -oE '^[0-9]+')
  TOK_S=$(grep -oE '[0-9]+ tok/s' "$log" | head -1 | grep -oE '^[0-9]+')
  RES_G=$(grep -oE 'reserved +[0-9.]+ GiB' "$log" | head -1 | grep -oE '[0-9.]+')
}

run_probe() {
  local label="$1"; shift
  if grep -qxF "$label" "$DONE" 2>/dev/null; then say "--- SKIP $label（已完成）"; return; fi

  for _ in $(seq 1 10); do
    local u; u=$(free_mib)
    [[ "${u:-99999}" -lt 1200 ]] && break
    say "  显存仍有 ${u} MiB，等 20s（不放行）"; sleep 20
  done

  say "--- $label   [$*]"
  local log="$OUT/$label.log"
  : > "$log"
  timeout -k 30 "$TIMEOUT" "$PYTHON" scripts/bench_trainer.py \
      --model_dir "$MODEL" --data_dir "$DATA" \
      --max_steps "$STEPS" --warm_skip "$WARM" \
      --optim adamw_bnb_8bit "$@" >>"$log" 2>&1
  local rc=$?

  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    say "  TIMEOUT > ${TIMEOUT}s（已杀；不记完成，下次会重试）"; return
  fi
  if grep -qE 'out of memory|OutOfMemoryError|CUDA error' "$log"; then
    say "  OOM"; echo "### $label [$*]" >>"$RESULTS"; echo "  OOM" >>"$RESULTS"
    echo "$label" >>"$DONE"; return
  fi

  # ★ 内核断言：测出来的数字只有在**跑的是批量 chunkwise** 时才可比。
  #   参考循环与批量版的步时差一个数量级，混在一起就是错的表。
  #   这道检查是有意做成"能失败"的 —— 它本该在第一次就抓住我漏 export 的那个 bug。
  if ! grep -q 'chunkwise kernel: batched' "$log"; then
    local kernel_line
    kernel_line=$(grep -m1 'chunkwise kernel:' "$log" || echo '(日志里没有内核信息)')
    say "  ❌ 内核断言失败：$kernel_line"
    say "     本次数字作废（未记入 completed，修好后重跑）"
    echo "### $label [$*]" >>"$RESULTS"
    echo "  INVALID: $kernel_line（不是批量 chunkwise，数字不可比）" >>"$RESULTS"
    return
  fi

  parse "$log"
  echo "### $label [$*]" >>"$RESULTS"
  if [[ -n "${OPT_MS:-}" ]]; then
    say "  opt_ms=${OPT_MS}  tok/s=${TOK_S}  reserved=${RES_G}GiB"
    echo "  opt_ms=${OPT_MS}  tok/s=${TOK_S}  reserved=${RES_G}GiB" >>"$RESULTS"
    echo "$label" >>"$DONE"
  else
    say "  ⚠️ 解析不到指标，见 $log"
    echo "  (解析不到结果，见 $log)" >>"$RESULTS"
  fi
  sleep 10
}

say "=== 0.5B 免检查点探针 ==="
say "model=$MODEL  data=$DATA  timeout=${TIMEOUT}s  单价=¥${PRICE_PER_HOUR}/h"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  say "!! 没有 nvidia-smi：是不是无卡模式？切回正常模式再跑。"; exit 2
fi
say "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

run_probe ckpt          --chunk_size 128 --compile
run_probe nockpt        --chunk_size 128 --compile --no_grad_checkpoint
run_probe nockpt_b2a2   --chunk_size 128 --compile --no_grad_checkpoint --batch_size 2 --grad_accum 2

# ---------- 汇总：把推算换成实测，并直接算出完整训练的钱 ----------
say ""
say "=== 汇总 ==="
val() { grep -A1 "^### $1 " "$RESULTS" 2>/dev/null | grep -oE 'opt_ms=[0-9]+' | head -1 | cut -d= -f2; }
CK=$(val ckpt); NK=$(val nockpt); NK2=$(val nockpt_b2a2)

printf '%s\n' "ckpt=${CK:-?}ms  nockpt=${NK:-?}ms  nockpt_b2a2=${NK2:-?}ms" | tee -a "$RESULTS"

if [[ -n "${CK:-}" && -n "${NK:-}" ]]; then
  # 单 epoch = 67390 个优化步（batch1×accum4 口径，8192 token/步）
  STEPS_PER_EPOCH=67390
  ratio=$(awk -v a="$CK" -v b="$NK" 'BEGIN{printf "%.3f", a/b}')
  say "**免检查点的实测倍数 = ${CK}/${NK} = ${ratio}×**（推算值 1.96×）"
  for pair in "nockpt:$NK" "nockpt_b2a2:$NK2"; do
    lbl="${pair%%:*}"; ms="${pair##*:}"
    [[ -z "$ms" || "$ms" == "$NK2" && "$lbl" == "nockpt_b2a2" && -z "${NK2:-}" ]] && continue
    awk -v ms="$ms" -v n="$STEPS_PER_EPOCH" -v p="$PRICE_PER_HOUR" -v lbl="$lbl" \
      'BEGIN{h=ms*n/3600000.0; printf "  %-12s 1 epoch = %.1f 小时 = ¥%.1f；20×Chinchilla(10.08B token) = %.1f 天 = ¥%.0f\n", lbl, h, h*p, h*18.26/24, h*18.26*p}' | tee -a "$RESULTS"
  done
else
  say "数据不全，无法算倍数（可能有的探针超时了，重跑本脚本会自动续跑）"
fi

say ""
say "原始日志在 $OUT/  ；汇总在 $RESULTS"
say "把 $RESULTS 贴回来即可（不需要贴 GPU 日志全文）"
