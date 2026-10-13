"""Build the Kaggle TPU v5e-8 training notebook (retnet_kaggle_train_8k_tpu.ipynb).

TPU differences vs the T4 notebook:
  * bf16 is NATIVE on TPU (fixed, no fp32 fallback)
  * launcher: accelerate launch --tpu --num_processes=8 (not torchrun)
  * gradient checkpointing OFF (XLA unsupported + 128GB HBM doesn't need it)
  * dataloader num_workers=0 + pin_memory=False
  * model classes read verbatim from retnet_hf/modeling_retnet.py (same as T4)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

model_src = (ROOT / "retnet_hf" / "modeling_retnet.py").read_text(encoding="utf-8")
idx = model_src.index("class RetNetHFConfig")
MODEL_CLASSES = model_src[idx:].rstrip() + "\n"

# --- TPU：把 _chunkwise_retention 的 Python for 循环改成 torch_xla.scan（XLA while）---
# Python 循环会被 XLA 展开成 128 份子图导致编译爆炸；scan 降级为 while 算子只编译一次。
_OLD_CHUNK = '''    def _chunkwise_retention(self, q, k, v, state, chunk_size):
        """Equation (7): inner-chunk parallel + cross-chunk recurrent state."""
        bsz, _, tgt_len, key_dim = q.shape
        if tgt_len % chunk_size != 0:
            raise ValueError(f"chunkwise mode requires seq_len ({tgt_len}) to be a "
                             f"multiple of chunk_size ({chunk_size})")
        if state is None:
            state = torch.zeros(bsz, self.num_heads, key_dim, v.shape[-1],
                                device=q.device, dtype=q.dtype)
        gamma = self.gamma.to(device=q.device, dtype=q.dtype)
        outputs = []
        rel = torch.arange(chunk_size, device=q.device)
        delta = rel[:, None] - rel[None, :]
        causal = delta >= 0
        inner_mask = torch.where(causal, gamma[:, None, None] ** delta.masked_fill(~causal, 0),
                                 torch.zeros_like(delta, dtype=torch.float32))
        cross_decay = gamma[:, None] ** (rel + 1)[None, :]
        value_decay = gamma[:, None] ** (chunk_size - 1 - rel)[None, :]
        chunk_decay = gamma ** chunk_size

        for i in range(tgt_len // chunk_size):
            sl = slice(i * chunk_size, (i + 1) * chunk_size)
            qi, ki, vi = q[:, :, sl], k[:, :, sl], v[:, :, sl]
            inner_scores = torch.matmul(qi, ki.transpose(-1, -2)) * inner_mask
            inner = torch.matmul(inner_scores, vi)
            cross = torch.matmul(qi * cross_decay[None, :, :, None], state)
            outputs.append(inner + cross)
            weighted_v = vi * value_decay[None, :, :, None]
            state = chunk_decay.view(1, -1, 1, 1) * state + \\
                torch.matmul(ki.transpose(-1, -2), weighted_v)

        output = torch.cat(outputs, dim=2)
        return output, state'''

_NEW_CHUNK = '''    def _chunkwise_retention(self, q, k, v, state, chunk_size):
        """Equation (7): inner-chunk parallel + cross-chunk recurrent state.

        TPU 上把块循环改写成 torch_xla.experimental.scan（XLA while），避免 Python
        for 循环被 XLA 展开成 128 份子图导致编译爆炸 / 内存溢出。
        """
        bsz, _, tgt_len, key_dim = q.shape
        if tgt_len % chunk_size != 0:
            raise ValueError(f"chunkwise mode requires seq_len ({tgt_len}) to be a "
                             f"multiple of chunk_size ({chunk_size})")
        n_chunks = tgt_len // chunk_size
        if state is None:
            state = torch.zeros(bsz, self.num_heads, key_dim, v.shape[-1],
                                device=q.device, dtype=q.dtype)
        gamma = self.gamma.to(device=q.device, dtype=q.dtype)
        rel = torch.arange(chunk_size, device=q.device)
        delta = rel[:, None] - rel[None, :]
        causal = delta >= 0
        inner_mask = torch.where(causal, gamma[:, None, None] ** delta.masked_fill(~causal, 0),
                                 torch.zeros_like(delta, dtype=torch.float32))
        cross_decay = gamma[:, None] ** (rel + 1)[None, :]
        value_decay = gamma[:, None] ** (chunk_size - 1 - rel)[None, :]
        chunk_decay = gamma ** chunk_size

        try:
            from torch_xla.experimental.scan import scan as _scan
        except ImportError:
            _scan = None

        if _scan is not None:
            # q/k/v: [bsz, heads, tgt_len, d] -> [n_chunks, bsz, heads, chunk_size, d]
            q_c = q.reshape(bsz, self.num_heads, n_chunks, chunk_size, key_dim).permute(2, 0, 1, 3, 4)
            k_c = k.reshape(bsz, self.num_heads, n_chunks, chunk_size, key_dim).permute(2, 0, 1, 3, 4)
            v_c = v.reshape(bsz, self.num_heads, n_chunks, chunk_size, v.shape[-1]).permute(2, 0, 1, 3, 4)

            def _chunk_fn(carry, xs):
                qi, ki, vi = xs
                inner = torch.matmul(torch.matmul(qi, ki.transpose(-1, -2)) * inner_mask, vi)
                cross = torch.matmul(qi * cross_decay[None, :, :, None], carry)
                y = inner + cross
                weighted_v = vi * value_decay[None, :, :, None]
                new_carry = chunk_decay.view(1, -1, 1, 1) * carry + \\
                    torch.matmul(ki.transpose(-1, -2), weighted_v)
                return new_carry, y

            final_state, ys = _scan(_chunk_fn, state, (q_c, k_c, v_c))
            output = ys.permute(1, 2, 0, 3, 4).reshape(bsz, self.num_heads, tgt_len, -1)
            return output, final_state

        # 回退：Python 循环（无 torch_xla scan 的环境，如 GPU/CPU）
        outputs = []
        for i in range(n_chunks):
            sl = slice(i * chunk_size, (i + 1) * chunk_size)
            qi, ki, vi = q[:, :, sl], k[:, :, sl], v[:, :, sl]
            inner_scores = torch.matmul(qi, ki.transpose(-1, -2)) * inner_mask
            inner = torch.matmul(inner_scores, vi)
            cross = torch.matmul(qi * cross_decay[None, :, :, None], state)
            outputs.append(inner + cross)
            weighted_v = vi * value_decay[None, :, :, None]
            state = chunk_decay.view(1, -1, 1, 1) * state + \\
                torch.matmul(ki.transpose(-1, -2), weighted_v)
        output = torch.cat(outputs, dim=2)
        return output, state'''

assert _OLD_CHUNK in MODEL_CLASSES, "_chunkwise_retention 原方法定位失败"
MODEL_CLASSES = MODEL_CLASSES.replace(_OLD_CHUNK, _NEW_CHUNK)

CELL_CONFIG = '''# ============================================================
# 【第 1 个 cell】配置区 —— 所有路径与超参数都在这里，改完直接运行
# ============================================================
import os, json

# ★TPU 环境补丁（必须最先执行，在任何 import torch_xla 之前）★
# 删掉 Kaggle 预置的 TPU_PROCESS_ADDRESSES / CLOUD_TPU_TASK_ID，
# 否则 PJRT 误判成 "local 单worker"，报 "Expected 8 worker addresses, got 1"
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("CLOUD_TPU_TASK_ID", None)
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
os.environ["XLA_USE_BF16"] = "1"   # 显式启用 bf16（TPU 原生主力精度）
# 禁用 transformers 的 TF/JAX 支持（TPU 环境 tensorflow 可能损坏，只走 PyTorch/XLA）
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

# ---- 路径配置（按你上传到 Kaggle 的 Dataset 名称修改）----
DATASET_DIR = "/kaggle/input/retnet-v11-8k/chat_pack_v11"      # 数据集
MODEL_DIR   = "/kaggle/input/retnet-272m-init/retnet_272m_rp"  # 初始权重
OUTPUT_DIR  = "/kaggle/working/retnet_272m_v11_8k_tpu"         # 输出

# ---- 训练超参数 ----
SEQ_LEN          = 8192   # 上下文长度（数据行宽 = SEQ_LEN + 1）
PER_DEVICE_BATCH = 4      # 每 TPU 核 batch；8 核 data-parallel 全局 batch = 32
GRAD_ACCUM       = 1      # 梯度累积
MAX_STEPS        = 30000
LR               = 1.1e-4
MIN_LR           = 1.1e-5
WARMUP           = 300
CHUNK_SIZE       = 64
SAVE_STEPS       = 400    # TPU 快，但 20h/周额度 + 会话上限，仍建议小步保存
EVAL_STEPS       = 2500
SAVE_LIMIT       = 3
WEIGHT_DECAY     = 0.1
MAX_GRAD_NORM    = 1.0

# ---- 显存参考（TPU bf16，每核 16GB）----
# 模型副本(bf16参数0.55G + fp32梯度1.09G + fp32 Adam 2.18G) ~3.8G/核
# + logits [B,8192,20000] bf16 = B*0.33G。PER_DEVICE_BATCH=4 时每核 ~6-7G，很宽裕，可试 8。

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
    weight_decay=WEIGHT_DECAY, max_grad_norm=MAX_GRAD_NORM,
)
with open("/kaggle/working/config.json", "w", encoding="utf-8") as f:
    json.dump(CFG, f, ensure_ascii=False, indent=2)

print("[OK] 路径校验通过，配置已写入 /kaggle/working/config.json")
print(f"  数据集 : {DATASET_DIR}")
print(f"  权重   : {MODEL_DIR}")
print(f"  输出   : {OUTPUT_DIR}")
print(f"  seq={SEQ_LEN}  per_device_batch={PER_DEVICE_BATCH}  max_steps={MAX_STEPS}")
'''

CELL_DEPS = '''# ============================================================
# 【第 2 个 cell】安装依赖（TPU 环境：不要重装 torch，也不要动 tensorflow/libtpu）
# ============================================================
import subprocess, sys
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

_pip("transformers<5", "accelerate>=0.20", "safetensors")

import torch, torch_xla
print("torch     :", torch.__version__)
print("torch_xla :", torch_xla.__version__)
print("transformers :", __import__("transformers").__version__)
print("accelerate   :", __import__("accelerate").__version__)
# 注：torch_xla 会警告 tensorflow 可能冲突，那是「潜在」警告不是错误——只要不 import tensorflow 就没事。
# libtpu 是 torch_xla 的运行时库，绝对不能卸载。
'''

CELL_TPU = '''# ============================================================
# 【第 3 个 cell】检测 TPU 核数
# ============================================================
import os, torch_xla.core.xla_model as xm

try:
    N_TPU = xm.xrt_world_size()
except Exception:
    N_TPU = int(os.environ.get("TPU_NUM_DEVICES", "0") or 0)
if N_TPU <= 0:
    N_TPU = 8  # v5e-8
print(f"TPU v5e cores = {N_TPU}")
assert N_TPU >= 8, f"期望 8 核（v5e-8），当前 {N_TPU}"
'''

TRAIN_SCRIPT = '''"""RetNet 272M 8K TPU 训练脚本（由 notebook 生成）。"""
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
def main(index=None, model=None):
    cfg = json.load(open("/kaggle/working/config.json", encoding="utf-8"))

    # 模型可传入（notebook_launcher fork 前只加载一次，8 进程共享，避免低内存 SIGSEGV）
    if model is None:
        model = RetNetForCausalLM.from_pretrained(cfg["model_dir"])
    model.config.mode = "chunkwise"
    model.config.chunk_size = cfg["chunk_size"]
    # TPU：不用梯度检查点（XLA 不支持，且 128GB HBM 不需要）

    # 设备信息（确认在 TPU 而非 CPU）
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    print(f"[debug] rank={xr.global_ordinal()} device={xm.xla_device()} "
          f"world_size={xr.world_size()} model_device={next(model.parameters()).device}",
          flush=True)

    ds = cfg["dataset_dir"]
    train_ds = PackedChatDataset(os.path.join(ds, "train.ids.npy"),
                                 os.path.join(ds, "train.wcode.npy"), cfg["seq_len"])
    val_ds = PackedChatDataset(os.path.join(ds, "val.ids.npy"),
                               os.path.join(ds, "val.wcode.npy"), cfg["seq_len"])

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=False,     # 跨会话 resume 不删旧 checkpoint
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
        bf16=True,                      # TPU 原生 bf16
        fp16=False,
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
        dataloader_num_workers=0,       # TPU 必须为 0
        dataloader_pin_memory=False,    # TPU 用 XLA 的 MpDeviceLoader
        seed=1234,
        gradient_checkpointing=False,   # TPU 关闭
        remove_unused_columns=False,    # TPU 上避免 XLA 签名检测删列
        include_inputs_for_metrics=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=ChatDataCollator(),
    )

    print(f"[info] bf16=True  mode=chunkwise  train_rows={len(train_ds)}  "
          f"val_rows={len(val_ds)}", flush=True)

    # 自动续训：从 output_dir 最新 checkpoint 继续
    from pathlib import Path as _P
    _ckpts = sorted(_P(cfg["output_dir"]).glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1]))
    resume = str(_ckpts[-1]) if _ckpts else None

    print(f"[debug] rank={xr.global_ordinal()} 开始训练；首次 forward 触发 XLA 编译"
          f"（chunkwise 128 次循环展开，可能几分钟，CPU 打满/RAM 高属正常）", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    print(f"[debug] rank={xr.global_ordinal()} 训练循环结束", flush=True)

    if trainer.is_world_process_zero():
        trainer.save_model(cfg["output_dir"])
        print("[done] final weights saved to", cfg["output_dir"], flush=True)


if __name__ == "__main__":
    main()
'''

CELL_WRITEFILE = "%%writefile /kaggle/working/train_retnet_tpu.py\n" + TRAIN_SCRIPT

CELL_LAUNCH = '''# ============================================================
# 【第 5 个 cell】启动 TPU 训练（notebook_launcher 官方 notebook 方式）
# ============================================================
import importlib.util, sys, json
from accelerate import notebook_launcher

# 加载训练脚本（拿到模型类 + main，不触发 __main__）
spec = importlib.util.spec_from_file_location("train_retnet_tpu", "/kaggle/working/train_retnet_tpu.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["train_retnet_tpu"] = mod
spec.loader.exec_module(mod)

# 模型只加载一次（fork 前），8 个进程共享，避免 Kaggle 低内存 SIGSEGV
cfg = json.load(open("/kaggle/working/config.json", encoding="utf-8"))
model = mod.RetNetForCausalLM.from_pretrained(cfg["model_dir"])
model.config.mode = "chunkwise"
model.config.chunk_size = cfg["chunk_size"]

def training_function(model):
    mod.main(0, model)

# notebook_launcher：notebook 上跑 TPU 的唯一正确方式（fork，非 spawn）
notebook_launcher(training_function, args=(model,), num_processes=N_TPU,
                  mixed_precision="bf16")
print("[done] training finished; weights in /kaggle/working/retnet_272m_v11_8k_tpu/")
'''

MARKDOWN_INTRO = """# RetNet 272M · 8K 上下文 · Kaggle TPU v5e-8 训练

三个交付物：本 notebook + 数据集 zip + 初始权重 zip（与 T4 版共用）。

TPU v5e-8：bf16 原生、128GB HBM、20h/周额度。bf16 是主力精度，272M+bf16+8K 全速跑。

流程：① 改第 1 个 cell 路径 → ② 依次运行所有 cell（Accelerator 选 TPU v5e-8）。

注意：chunkwise 的 Python 循环会被 XLA 展开，首次编译可能较慢，属正常。
"""


def cell(source: str, kind: str = "code"):
    return {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


nb = {
    "cells": [
        cell(MARKDOWN_INTRO, "markdown"),
        cell(CELL_CONFIG),
        cell(CELL_DEPS),
        cell(CELL_TPU),
        cell(CELL_WRITEFILE),
        cell(CELL_LAUNCH),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "TPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "Kaggle" / "retnet_kaggle_train_8k_tpu.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] wrote {out}  ({len(nb['cells'])} cells)")
