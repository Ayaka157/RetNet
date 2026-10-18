#!/usr/bin/env bash
# 用 curl 直接下 HF 文件（断点续传 + 重试），绕过 huggingface_hub 在抖动代理下的 HEAD/etag 卡死。
#
# 输入 logs/_dl_manifest.tsv：每行  url \t 本地子目录 \t 文件名 \t 期望字节数
# 依赖 /etc/network_turbo 提供的代理（huggingface.co 直连不通）。
set -uo pipefail
cd /root/autodl-tmp/RetNet
source /etc/network_turbo >/dev/null 2>&1
export https_proxy="${https_proxy:-http://10.37.1.23:12798}"
export http_proxy="${http_proxy:-$https_proxy}"

MANIFEST=logs/_dl_manifest.tsv
[[ -f "$MANIFEST" ]] || { echo "!! 找不到 $MANIFEST"; exit 2; }

total=$(wc -l < "$MANIFEST")
i=0
ok=0
skip=0
fail=0
while IFS=$'\t' read -r url local fname want; do
  [[ -z "${url:-}" ]] && continue
  # 清单可能来自 Windows（CRLF），必须剥掉 \r 否则 [[ -eq ]] 会报算术错误
  want="${want%$'\r'}"
  fname="${fname%$'\r'}"
  i=$((i + 1))
  out="data/hf/$local/$fname"
  mkdir -p "$(dirname "$out")"
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [[ "$have" -eq "$want" ]]; then
    skip=$((skip + 1))
    continue
  fi
  echo "[$i/$total] $local/$fname  have=${have} want=${want}"
  for try in 1 2 3 4 5 6 7 8; do
    if curl -L -C - --retry 3 --retry-delay 5 --connect-timeout 20 --max-time 1800 \
            --speed-limit 1024 --speed-time 120 -sS -o "$out" "$url"; then
      break
    fi
    echo "    retry $try (have=$(stat -c %s "$out" 2>/dev/null || echo 0))"
    sleep 5
  done
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [[ "$have" -eq "$want" ]]; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    echo "    ⚠️ 不完整: have=$have want=$want"
  fi
done < "$MANIFEST"

echo "===== 完成: ok=$ok skip=$skip fail=$fail / 共 $total ====="
du -sh data/hf/* 2>/dev/null
df -h /root/autodl-tmp | tail -1
