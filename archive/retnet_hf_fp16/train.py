"""Role-play SFT with the mature HF stack (Trainer + Accelerate + safetensors).

Replaces the handwritten train.py for Stage 2+.  Same hyper-parameters as the
handwritten loop: 45k steps, batch 10, seq 512, cosine 1.1e-4 -> 1.1e-5,
prompt_loss_weight 0.1, bf16, torch.compile(inductor).

    python retnet_hf/train.py \
        --model_dir checkpoints/retnet_100m_pretrain_v4_hf \
        --data_dir data_cache/chat_pack_v8 \
        --output_dir checkpoints/retnet_100m_rp_v3_hf \
        --max_steps 45000 --batch_size 10 --lr 1.1e-4 --warmup 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf import RetNetForCausalLM, PackedChatDataset, ChatDataCollator  # noqa: E402
from retnet_hf.modeling_retnet import RetNetHFConfig  # noqa: E402


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="migrated HF checkpoint dir")
    ap.add_argument("--data_dir", required=True, help="packed chat data dir")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=45000)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1.1e-4)
    ap.add_argument("--min_lr", type=float, default=1.1e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_batches", type=int, default=40)
    ap.add_argument("--compile", action="store_true", default=True,
                    help="torch.compile with inductor backend")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--grad_checkpoint", action="store_true",
                    help="gradient checkpointing (saves VRAM, ~25% slower)")
    ap.add_argument("--mode", default="parallel", choices=["parallel", "chunkwise"],
                    help="retention forward mode used for training")
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--resume", action="store_true", help="resume from output_dir")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="autocast precision (fp16/fp32 for T4/P100 on Kaggle/Colab)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    model = RetNetForCausalLM.from_pretrained(str(model_dir))
    model.config.mode = args.mode
    model.config.chunk_size = args.chunk_size
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded model from {model_dir} (params={n_params / 1e6:.2f}M)", flush=True)

    # packed rows are seq_len+1 wide
    seq_len = int(Path(data_dir / "train.ids.npy").stat().st_size /
                  (438647 * 2)) if False else None  # placeholder
    import numpy as np
    probe = np.load(data_dir / "train.ids.npy", mmap_mode="r")
    seq_len = probe.shape[1] - 1
    del probe
    print(f"seq_len={seq_len}", flush=True)

    train_ds = PackedChatDataset(str(data_dir / "train.ids.npy"),
                                 str(data_dir / "train.wcode.npy"), seq_len)
    val_ds = PackedChatDataset(str(data_dir / "val.ids.npy"),
                               str(data_dir / "val.wcode.npy"), seq_len)
    collator = ChatDataCollator()
    print(f"train rows={len(train_ds)} val rows={len(val_ds)}", flush=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=not args.resume,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.min_lr},
        warmup_steps=args.warmup,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),   # fp32: 两者皆 False = 全精度 fp32 训练
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=2500,
        save_strategy="steps",
        save_steps=2500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=args.seed,
        gradient_checkpointing=args.grad_checkpoint,
        include_inputs_for_metrics=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    # torch.compile 必须在 Trainer 构造之后：transformers 4.57 的 RemoveColumnsCollator
    # 惰性地按 model.forward 签名过滤数据列，compile 包装后签名丢失会导致 input_ids 被滤掉。
    # 因此先强制缓存未编译模型的签名，再替换 model。
    trainer._set_signature_columns_if_needed()
    if args.compile:
        try:
            trainer.model = torch.compile(trainer.model, backend="inductor")
            print("torch.compile(inductor) applied to trainer.model", flush=True)
        except Exception as exc:
            print(f"torch.compile failed ({exc}); continuing eager", flush=True)
    # 恢复中断：从 output_dir 最后一个 checkpoint 继续
    trainer.train(resume_from_checkpoint=args.output_dir if args.resume else False)
    trainer.save_model(args.output_dir)
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
