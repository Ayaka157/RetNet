#!/usr/bin/env bash
# DDP 冒烟：两步走
#   1) 20 步、不编译：只验 DDP 能否建成（unused parameter / 通信 / rank 守卫）
#   2) 150 步、开编译：测真实吞吐，决定这轮跑多长
set -uo pipefail
cd /root/autodl-tmp/RetNet
PY=/root/miniconda3/bin/python
PORT=${PORT:-29511}

cat > logs/_plan_smoke_ddp.json <<'JSON'
{
  "plan_version": 99,
  "seq_len": 2048,
  "stages": [
    {"name": "smoke", "data_dir": "data_cache/pretrain_v2", "mode": "flat",
     "exclude": ["skypile_"], "steps": 150}
  ],
  "lr": {"peak": 1.1e-4, "min": 1.1e-5, "warmup": 20},
  "save_steps": 100000,
  "eval_steps": 100000,
  "val_dir": "data_cache/chat_pack_v13",
  "notes": "DDP 冒烟专用，不产出模型"
}
JSON

run() {   # run <steps> <batch> <extra-args...>
  local steps=$1 batch=$2; shift 2
  rm -rf checkpoints/_smoke_ddp
  timeout 1200 $PY -m torch.distributed.run --nproc_per_node 2 \
        --master_port "$PORT" \
        retnet_hf/train.py \
        --model_dir checkpoints/retnet_500m_v15b_ep3 \
        --output_dir checkpoints/_smoke_ddp \
        --plan logs/_plan_smoke_ddp.json \
        --batch_size "$batch" --mode chunkwise --chunk_size 128 \
        --precision bf16 --eval_batches 4 --save_total_limit 1 "$@"
  return $?
}

echo "########## 1) DDP 建成性验证：20 步 / 不编译 ##########"
if run 20 16 --no-compile 2>&1 | tee /tmp/ddp_smoke1.log | tail -25; then
  echo "  DDP 建成 OK"
else
  echo "  !! 失败，检查 /tmp/ddp_smoke1.log"
  grep -aiE "unused|DDP|error|Traceback" /tmp/ddp_smoke1.log | head -10
  exit 1
fi

echo
echo "########## 2) 真实吞吐：150 步 / 开编译 ##########"
run 150 16 2>&1 | tee /tmp/ddp_smoke2.log | grep -aE "it/s|train_runtime|TRAINING DONE|真实 LR" | tail -8

echo
echo "########## 汇总 ##########"
rate=$(tr '\r' '\n' < /tmp/ddp_smoke2.log | grep -aoE "[0-9.]+it/s" | tail -1)
run_s=$(grep -aoE "'train_runtime': [0-9.]+" /tmp/ddp_smoke2.log | tail -1 | grep -oE "[0-9.]+")
echo "  末段速率: ${rate:-未取到}"
echo "  train_runtime: ${run_s:-未取到} s（含编译预热，仅供参考）"
du -sh checkpoints/_smoke_ddp 2>/dev/null
