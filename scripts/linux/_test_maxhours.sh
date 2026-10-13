#!/usr/bin/env bash
# 控制实验：--max-hours 的小数换算与非法值拒绝。
#
# 起因：2026-10-11 我用 `--max-hours 10.9` 起正式训练，bash 的 $(( )) 只吃整数，
# 报 `syntax error: invalid arithmetic operator (error token is ".9")`；脚本是
# `set -uo pipefail`（没 -e），于是错误只让 SECS 变空、脚本继续跑到 timeout 才失败，
# 训练进程一次都没启动。所以现在用 awk 换算，并在入口拒绝非法值。
set -uo pipefail

echo "=== 合法值应换算出正确秒数 ==="
for h in 10.5 0.25 11 0.1; do
  SECS="$(awk -v h="$h" 'BEGIN{printf "%d", h*3600}')"
  printf '  h=%-6s -> SECS=%s\n' "$h" "$SECS"
  [[ -n "$SECS" && "$SECS" -gt 0 ]] || echo "    ❌ 换算失败"
done

echo "=== 非法值应被拒绝（退出码非 0）==="
for h in abc -1 0 "" 1e3; do
  if awk -v h="$h" 'BEGIN{exit !(h+0>0)}'; then
    printf '  h=%-6s 被判合法 ❌（应当拒绝）\n' "'$h'"
  else
    printf '  h=%-6s 被拒绝 ✅\n' "'$h'"
  fi
done

echo "=== 旧的整数写法对 10.9 确实会失败（负向对照）==="
if ( MAX_HOURS=10.9; SECS=$(( MAX_HOURS * 3600 )) ) 2>/dev/null; then
  echo "  ❌ 整数写法竟然没报错"
else
  echo "  ✅ 整数写法报错（这就是当初静默失败的原因）"
fi
