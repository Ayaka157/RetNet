#!/usr/bin/env bash
# 下载看护：source 学术代理 → 跑 fetch → 失败自动拉起，直到成功或 12 轮。
# 用 setsid + nohup + </dev/null 三层保证 SSH 断开后仍存活。
set -uo pipefail
cd /root/autodl-tmp/RetNet
source /etc/network_turbo >/dev/null 2>&1

export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60

LOG=logs/linux/fetch_corpora.out
PY=/root/miniconda3/bin/python

for i in $(seq 1 12); do
  echo "########## supervisor 第 $i 轮 $(date '+%H:%M:%S') ##########"
  $PY -u scripts/_fetch_corpora.py
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "########## 全部下载完成 $(date '+%H:%M:%S') ##########"
    break
  fi
  echo "########## 第 $i 轮 rc=$rc，30s 后重试 ##########"
  sleep 30
done
echo "########## watchdog 结束 $(date '+%H:%M:%S') ##########"
du -sh data/hf/* 2>/dev/null
