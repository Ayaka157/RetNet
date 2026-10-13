"""Build the Kaggle training notebook (retnet_kaggle_train_8k.ipynb).

Reads the RetNet model classes verbatim from retnet_hf/modeling_retnet.py so the
state_dict keys match the checkpoint byte-for-byte (no transcription risk), then
assembles a 5-cell notebook:
  1. config + path validation
  2. deps
  3. GPU / precision detection (bf16 if Ampere+, else fp32 for T4)
  4. %%writefile train_retnet.py  (model + data + HF Trainer, DDP-ready)
  5. launch via torch.distributed.run (2x T4)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- read model classes verbatim from the ORIGINAL (state_dict keys must match) ---
model_src = (ROOT / "retnet_hf" / "modeling_retnet.py").read_text(encoding="utf-8")
idx = model_src.index("class RetNetHFConfig")
MODEL_CLASSES = model_src[idx:].rstrip() + "\n"

# ---------------------------------------------------------------------------
# Cell 1: config + validation
# ---------------------------------------------------------------------------
CELL_CONFIG = '''# ============================================================
# 【第 1 个 cell】配置区 —— 所有路径与超参数都在这里，改完直接运行
# ============================================================
import os, json

# ---- 路径配置（按你上传到 Kaggle 的 Dataset 名称修改）----
# Kaggle 会把上传的 zip 自动解压到 /kaggle/input/<dataset名>/
DATASET_DIR = "/kaggle/input/retnet-v11-8k/chat_pack_v11"      # 数据集
MODEL_DIR   = "/kaggle/input/retnet-272m-init/retnet_272m_rp"  # 初始权重
OUTPUT_DIR  = "/kaggle/working/retnet_272m_v11_8k"             # 输出(checkpoint + 最终权重)

# ---- 训练超参数 ----
SEQ_LEN          = 8192   # 上下文长度（数据行宽 = SEQ_LEN + 1）
PER_DEVICE_BATCH = 2      # 每张卡 batch；2 卡 DDP 总 batch = 4
GRAD_ACCUM       = 1      # 梯度累积；有效 batch = PER_DEVICE_BATCH * GPU数 * GRAD_ACCUM
MAX_STEPS        = 30000
LR               = 1.1e-4
MIN_LR           = 1.1e-5
WARMUP           = 300
CHUNK_SIZE       = 64
SAVE_STEPS       = 400    # 400×~23s≈2.6h 落盘一次；12h 会话能存 ~4 个（勿设超过 ~1800，否则会话被杀前存不下）
EVAL_STEPS       = 2500
SAVE_LIMIT       = 3
WEIGHT_DECAY     = 0.1
MAX_GRAD_NORM    = 1.0
USE_COMPILE      = False  # torch.compile 提速（首次编译慢，T4 上慎开，失败自动回退 eager）

# ---- 显存参考（fp32 每卡）----
# 固定区(参数+梯度+Adam fp32) ~4.36GB + logits/CE 各 b*0.655GB + GC 保存输入 b*0.67GB + 反传重算 ~2GB
#   b=2 -> ~10.3GB（安全）  b=3 -> ~12.3GB  b=4 -> ~14.3GB（贴近 16GB，可能溢出）
# OOM 就把 PER_DEVICE_BATCH 往下调。

# ---- 路径合法性校验（不合法立刻抛错）----
def _must_dir(p, name):
    if not os.path.isdir(p):
        raise FileNotFoundError(
            f"[配置错误] {name} 目录不存在: {p}\\n"
            f"请确认 Kaggle Dataset 名称与上面路径一致。")
def _must_file(p, name):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"[配置错误] {name} 文件不存在: {p}")

_must_dir(DATASET_DIR, "DATASET_DIR")
_must_dir(MODEL_DIR, "MODEL_DIR")
for f in ("train.ids.npy", "train.wcode.npy", "val.ids.npy", "val.wcode.npy"):
    _must_file(os.path.join(DATASET_DIR, f), f"数据集/{f}")
for f in ("config.json", "model.safetensors"):
    _must_file(os.path.join(MODEL_DIR, f), f"权重/{f}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = dict(
    dataset_dir=DATASET_DIR, model_dir=MODEL_DIR, output_dir=OUTPUT_DIR,
    seq_len=SEQ_LEN, per_device_batch=PER_DEVICE_BATCH, grad_accum=GRAD_ACCUM,
    max_steps=MAX_STEPS, lr=LR, min_lr=MIN_LR, warmup=WARMUP, chunk_size=CHUNK_SIZE,
    save_steps=SAVE_STEPS, eval_steps=EVAL_STEPS, save_limit=SAVE_LIMIT,
    weight_decay=WEIGHT_DECAY, max_grad_norm=MAX_GRAD_NORM, use_compile=USE_COMPILE,
)
with open("/kaggle/working/config.json", "w", encoding="utf-8") as f:
    json.dump(CFG, f, ensure_ascii=False, indent=2)

print("[OK] 路径校验通过，配置已写入 /kaggle/working/config.json")
print(f"  数据集 : {DATASET_DIR}")
print(f"  权重   : {MODEL_DIR}")
print(f"  输出   : {OUTPUT_DIR}")
print(f"  seq={SEQ_LEN}  per_device_batch={PER_DEVICE_BATCH}  grad_accum={GRAD_ACCUM}  max_steps={MAX_STEPS}")
'''

# ---------------------------------------------------------------------------
# Cell 2: deps
# ---------------------------------------------------------------------------
CELL_DEPS = '''# ============================================================
# 【第 2 个 cell】安装依赖
# ============================================================
import subprocess, sys
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# torch 在 Kaggle 已预装；补装/对齐训练栈
_pip("transformers<5", "accelerate", "safetensors")

import torch
print("torch        :", torch.__version__, "| cuda:", torch.version.cuda)
print("transformers :", __import__("transformers").__version__)
print("accelerate   :", __import__("accelerate").__version__)
'''

# ---------------------------------------------------------------------------
# Cell 3: GPU / precision detection
# ---------------------------------------------------------------------------
CELL_GPU = '''# ============================================================
# 【第 3 个 cell】检测 GPU 与精度（T4 无 bf16 -> 回退 fp32）
# ============================================================
import torch

N_GPUS = torch.cuda.device_count()
assert N_GPUS >= 2, f"需要 2 张 GPU，当前只有 {N_GPUS} 张"
for i in range(N_GPUS):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i}: {p.name}  {p.total_memory / 2**30:.1f} GiB  (cap {torch.cuda.get_device_capability(i)})")

cap = torch.cuda.get_device_capability(0)
PRECISION = "bf16" if cap[0] >= 8 else "fp32"
print(f"precision = {PRECISION}" + ("  (Ampere+ 原生 bf16)" if PRECISION == "bf16"
                                    else "  (Turing 无 bf16 TensorCore，按约定回退 fp32)"))
'''

# ---------------------------------------------------------------------------
# Cell 4: writefile the training script (model + data + Trainer)
# ---------------------------------------------------------------------------
TRAIN_SCRIPT = '''"""RetNet 272M 8K DDP 训练脚本（由 notebook 生成）。"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (PreTrainedModel, PretrainedConfig, TrainingArguments,
                          Trainer)

# ==================== 模型定义（与 checkpoint 状态字典键逐字一致）====================
''' + MODEL_CLASSES + '''

# 补丁：确保梯度检查点在各 transformers 版本下都可靠开启
def _set_gradient_checkpointing(self, module, value=False, **kwargs):
    if isinstance(module, RetNetBlock):
        module.gradient_checkpointing = value
if not hasattr(RetNetForCausalLM, "_set_gradient_checkpointing"):
    RetNetForCausalLM._set_gradient_checkpointing = _set_gradient_checkpointing
RetNetForCausalLM._supports_gradient_checkpointing = True


# ==================== 数据 ====================
class PackedChatDataset(Dataset):
    """行布局 (seq_len+1)：input_ids=row[:-1], labels=row[1:], wcodes=codes[1:]"""
    def __init__(self, ids_path, wcode_path, seq_len):
        self.ids = np.load(ids_path, mmap_mode="r")
        self.codes = np.load(wcode_path, mmap_mode="r")
        assert self.ids.ndim == 2 and self.ids.shape == self.codes.shape
        assert self.ids.shape[1] == seq_len + 1, \\
            f"rows={self.ids.shape[1]} != seq_len+1={seq_len + 1}"
        self.seq_len = seq_len

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        row = self.ids[idx]
        return {
            "input_ids": torch.from_numpy(row[:-1].astype(np.int64)),
            "labels": torch.from_numpy(row[1:].astype(np.int64)),
            "wcodes": torch.from_numpy(self.codes[idx][1:].astype(np.int64)),
        }


class ChatDataCollator:
    def __call__(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "wcodes": torch.stack([b["wcodes"] for b in batch]),
        }


# ==================== 主流程 ====================
def main():
    cfg = json.load(open("/kaggle/working/config.json", encoding="utf-8"))

    cap = torch.cuda.get_device_capability(0)
    precision = "bf16" if cap[0] >= 8 else "fp32"

    model = RetNetForCausalLM.from_pretrained(cfg["model_dir"])
    model.config.mode = "chunkwise"
    model.config.chunk_size = cfg["chunk_size"]
    # 显式打开梯度检查点（不依赖 Trainer 内部行为）
    model.gradient_checkpointing_enable()
    for b in model.blocks:
        b.gradient_checkpointing = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ds = cfg["dataset_dir"]
    train_ds = PackedChatDataset(os.path.join(ds, "train.ids.npy"),
                                 os.path.join(ds, "train.wcode.npy"), cfg["seq_len"])
    val_ds = PackedChatDataset(os.path.join(ds, "val.ids.npy"),
                               os.path.join(ds, "val.wcode.npy"), cfg["seq_len"])

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=False,     # 不覆盖：跨 12h 会话 resume 时保留旧 checkpoint
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_batch"],
        per_device_eval_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": cfg["min_lr"]},
        warmup_steps=cfg["warmup"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        bf16=(precision == "bf16"),
        fp16=False,                      # 约定：bf16 或 fp32，绝不用 fp16（8K 会 NaN）
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=1234,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,  # 模型无未用参数，关掉去掉每步额外图遍历
        include_inputs_for_metrics=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=ChatDataCollator(),
    )

    if cfg["use_compile"]:
        try:
            trainer._set_signature_columns_if_needed()
            trainer.model = torch.compile(trainer.model, backend="inductor")
            print("[info] torch.compile(inductor) applied", flush=True)
        except Exception as e:
            print(f"[warn] torch.compile failed ({e}); continuing eager", flush=True)

    print(f"[info] precision={precision}  rank={os.environ.get('LOCAL_RANK')}  "
          f"train_rows={len(train_ds)}  val_rows={len(val_ds)}", flush=True)

    # 自动续训：从 output_dir 最新 checkpoint 继续（跨 12h 会话）
    from pathlib import Path as _P
    _ckpts = sorted(_P(cfg["output_dir"]).glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1]))
    resume = str(_ckpts[-1]) if _ckpts else None
    if resume and os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[info] resume from {resume}", flush=True)

    trainer.train(resume_from_checkpoint=resume)

    if trainer.is_world_process_zero():
        trainer.save_model(cfg["output_dir"])
        print("[done] final weights saved to", cfg["output_dir"], flush=True)


if __name__ == "__main__":
    main()
'''

CELL_WRITEFILE = "%%writefile /kaggle/working/train_retnet.py\n" + TRAIN_SCRIPT

# ---------------------------------------------------------------------------
# Cell 5: launch
# ---------------------------------------------------------------------------
CELL_LAUNCH = '''# ============================================================
# 【第 5 个 cell】启动 DDP 训练（2 卡 torch.distributed.run）
# ============================================================
import subprocess, sys

cmd = [
    sys.executable, "-m", "torch.distributed.run",
    "--nproc_per_node", str(N_GPUS),
    "--master_port", "29500",
    "/kaggle/working/train_retnet.py",
]
print("launching:", " ".join(cmd), flush=True)
subprocess.check_call(cmd)
print("[done] training finished; weights in /kaggle/working/retnet_272m_v11_8k/")
'''

# ---------------------------------------------------------------------------
# assemble notebook
# ---------------------------------------------------------------------------
def cell(source: str, kind: str = "code"):
    return {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }

MARKDOWN_INTRO = """# RetNet 272M · 8K 上下文 · Kaggle T4×2 训练

三个交付物：本 notebook + 数据集 zip + 初始权重 zip。

流程：① 改第 1 个 cell 的路径配置 → ② 依次运行所有 cell。

精度自动选择：有 bf16 用 bf16，否则（T4 无 bf16 TensorCore）回退 fp32。
"""

nb = {
    "cells": [
        cell(MARKDOWN_INTRO, "markdown"),
        cell(CELL_CONFIG),
        cell(CELL_DEPS),
        cell(CELL_GPU),
        cell(CELL_WRITEFILE),
        cell(CELL_LAUNCH),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "Kaggle" / "retnet_kaggle_train_8k.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] wrote {out}  ({len(nb['cells'])} cells)")
print("cells:", [c["cell_type"] for c in nb["cells"]])
