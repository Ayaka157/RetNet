"""梯度累积在这个模型上是否正确？—— 只读源码与标志位，不跑前向。"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

m = RetNetForCausalLM.from_pretrained(str(ROOT / "checkpoints/retnet_272m_rp"))
sig = inspect.signature(m.forward)
print("forward 签名:", sig)
print("有 **kwargs :", any(p.kind == inspect.Parameter.VAR_KEYWORD
                           for p in sig.parameters.values()))

ta = TrainingArguments(output_dir=str(ROOT / "checkpoints" / "_probe_accum"),
                       max_steps=1, per_device_train_batch_size=1,
                       gradient_accumulation_steps=4, bf16=True,
                       report_to=[], save_strategy="no", eval_strategy="no")
tr = Trainer(model=m, args=ta)
print(f"\nTrainer.model_accepts_loss_kwargs = {tr.model_accepts_loss_kwargs}")

src = inspect.getsource(Trainer.training_step)
print("\n=== Trainer.training_step 里与梯度累积/归一化相关的行 ===")
for line in src.splitlines():
    t = line.strip()
    if re.search(r"accumulation|num_items_in_batch|loss\s*=|loss\s*/|/\s*self", t):
        print("   ", t[:118])

src2 = inspect.getsource(Trainer.compute_loss)
print("\n=== Trainer.compute_loss 相关行 ===")
for line in src2.splitlines():
    t = line.strip()
    if re.search(r"num_items_in_batch|loss|kwargs", t):
        print("   ", t[:118])
