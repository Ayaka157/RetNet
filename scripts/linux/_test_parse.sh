#!/usr/bin/env bash
# speed_probe.sh 里解析逻辑的自测 —— 用**真实的 bench_trainer 日志**做正向对照。
#
# 为什么需要它：探针驱动一旦解析不出来，就会把「已经付了钱的测量」当成失败，
# 而日志本身是好的。这类 bug 在云端代价高，所以在本地先证明解析能工作。
#
#   bash scripts/linux/_test_parse.sh
#
# 退出码 0 = 全部符合预期；非 0 = 解析逻辑坏了，别带着它上云。

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

fails=0

# —— 与 speed_probe.sh 完全一致的解析逻辑（故意复制，测的就是这份）——
parse() {
  local log="$1"
  OPT_MS=$(grep -oE '[0-9]+ ms/[0-9]+ token' "$log" | head -1 | grep -oE '^[0-9]+')
  TOK_S=$(grep -oE '[0-9]+ tok/s' "$log" | head -1 | grep -oE '^[0-9]+')
  RES_G=$(grep -oE 'reserved +[0-9.]+ GiB' "$log" | head -1 | grep -oE '[0-9.]+')
}

check_case() {
  local log="$1" want_ms="$2" want_res="$3" want_tok="$4"
  if [[ ! -f "$log" ]]; then
    echo "  [SKIP] 没有这个日志: $log"; return
  fi
  parse "$log"
  local ok=1
  [[ "$OPT_MS" == "$want_ms" ]]  || ok=0
  [[ "$RES_G"  == "$want_res" ]] || ok=0
  [[ -z "$want_tok" || "$TOK_S" == "$want_tok" ]] || ok=0
  if [[ $ok -eq 1 ]]; then
    echo "  [PASS] $(basename "$log")  opt_ms=$OPT_MS tok/s=$TOK_S reserved=${RES_G}GiB"
  else
    echo "  [FAIL] $(basename "$log")"
    echo "         期望 opt_ms=$want_ms tok/s=${want_tok:-*} reserved=$want_res"
    echo "         实得 opt_ms=${OPT_MS:-空} tok/s=${TOK_S:-空} reserved=${RES_G:-空}GiB"
    fails=$((fails+1))
  fi
}

echo "=== 解析逻辑自测（用真实日志）==="
check_case logs/scale_probe2/V1a_500M_ckpt_chunk128.log     1831 6.91 4473
check_case logs/scale_probe2/V1b_600M_ckpt_chunk128.log    19948 8.18  411
check_case logs/scale_probe2/E5a_500M_paged_chunk128.log    1832 6.25 4472
check_case logs/scale_probe2/V2a_600M_ckpt_chunk256.log    13881 8.07  590

# —— 反向对照：一个必然不该被解析出数字的文件 ——
echo ""
echo "=== 反向对照：OOM / 报错日志不应解析出指标 ==="
tmp="$(mktemp)"
printf 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 22.00 MiB\n' > "$tmp"
parse "$tmp"
if [[ -z "${OPT_MS:-}" && -z "${TOK_S:-}" && -z "${RES_G:-}" ]]; then
  echo "  [PASS] OOM 日志正确地没有产出任何指标"
else
  echo "  [FAIL] OOM 日志竟然解析出 opt_ms=$OPT_MS tok_s=$TOK_S res=$RES_G —— 会把失败误记成成功"
  fails=$((fails+1))
fi
rm -f "$tmp"

# —— 内核断言：这道的存在，是因为我漏了 export RETNET_FAST_CHUNKWISE，
#    结果探针跑了慢速参考循环、600 秒连一步都没完成。断言必须能抓住它。 ——
echo ""
echo "=== 内核断言（chunkwise kernel 必须是 batched）==="
kbad="$(mktemp)"; kgood="$(mktemp)"
printf 'chunkwise kernel: reference loop\noptimizer: adamw_bnb_8bit\n' > "$kbad"
printf 'chunkwise kernel: batched\noptimizer: adamw_bnb_8bit\n' > "$kgood"
if grep -q 'chunkwise kernel: batched' "$kbad"; then
  echo "  [FAIL] reference loop 竟然通过了断言 —— 这道检查抓不住真 bug"
  fails=$((fails+1))
else
  echo "  [PASS] reference loop 被正确拒绝（这正是 2026-10-11 实测踩到的坑）"
fi
if grep -q 'chunkwise kernel: batched' "$kgood"; then
  echo "  [PASS] batched 正常通过"
else
  echo "  [FAIL] 连 batched 都不通过，断言写反了"
  fails=$((fails+1))
fi
# 顺带验真实日志：本机所有 bench 日志都应该是 batched
real_bad=0
for f in logs/scale_probe2/V1a_500M_ckpt_chunk128.log logs/scale_probe2/E5a_500M_paged_chunk128.log; do
  [[ -f "$f" ]] || continue
  if grep -q 'chunkwise kernel: batched' "$f"; then
    echo "  [PASS] 真实日志 $(basename "$f") 是 batched"
  else
    echo "  [FAIL] 真实日志 $(basename "$f") 不是 batched"
    real_bad=$((real_bad+1))
  fi
done
fails=$((fails+real_bad))
rm -f "$kbad" "$kgood"

echo ""
if [[ $fails -eq 0 ]]; then
  echo "✅ 解析逻辑全部通过"
  exit 0
else
  echo "❌ $fails 项失败 —— 修好再上云"
  exit 1
fi
