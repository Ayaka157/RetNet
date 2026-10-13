"""Time real ``Trainer`` training steps — the path ``run_task.py`` actually runs.

``scripts/bench_step.py`` calls the model directly and reports 1,023 ms/step with
``torch.compile``.  The real training run reports 3,651 ms/step with the same
settings.  This harness reproduces the Trainer path so the two can be compared
apples to apples: same model construction, same ``TrainingArguments``, same
dataset objects, same collator.

    python scripts/bench_trainer.py --model_dir checkpoints/retnet_272m_rp ^
        --data_dir data_cache/chat_pack_v13 --max_steps 60 --chunk_size 128 --compile
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.data import PackedChatDataset, ChatDataCollator        # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM, FAST_CHUNKWISE  # noqa: E402


class StepTimer(Trainer):
    """Trainer that records wall time per optimisation step.

    ``use_save_on_cpu``：把「重算」换成「卸载到主机内存」——
    用 ``torch.autograd.graph.save_on_cpu`` **替换**梯度检查点：激活不在 GPU 上重算，
    而是存进 pinned CPU 内存再取回。卸载体量只有层边界激活
    （20 层 × 2048 × 1408 × 2 B ≈ 115 MB/微步），PCIe 单向约 5 ms；
    而实测重算成本约 232 ms/微步（由检查点代价 1.965× 反解）。
    这是唯一能直接抹掉检查点那 ~49% 步时的路。
    """

    use_save_on_cpu = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.step_times: list[float] = []
        self.eval_times: list[float] = []
        self._t0 = None

    def training_step(self, *a, **kw):
        self._t0 = time.perf_counter()
        if StepTimer.use_save_on_cpu:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                out = super().training_step(*a, **kw)
        else:
            out = super().training_step(*a, **kw)
        torch.cuda.synchronize()
        self.step_times.append(time.perf_counter() - self._t0)
        return out

    def evaluate(self, *a, **kw):
        # 评测不进 training_step，得单独计时，否则它会被算进某一步的步时里
        # （v14 那个「16 步花了 292 秒」的现象就是这么来的）。
        t0 = time.perf_counter()
        out = super().evaluate(*a, **kw)
        torch.cuda.synchronize()
        self.eval_times.append(time.perf_counter() - t0)
        return out


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=128)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no_grad_checkpoint", action="store_true")
    ap.add_argument("--warm_skip", type=int, default=20,
                    help="丢弃前 N 步（inductor 首次编译）再统计")
    ap.add_argument("--optim", default=None,
                    help="TrainingArguments.optim，如 adamw_torch_fused / adamw_bnb_8bit")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="梯度累积步数；batch 1 + accum 4 = 等效 batch 4")
    ap.add_argument("--compile_mode", default=None,
                    help="torch.compile 模式；reduce-overhead 会启用 CUDA Graphs")
    ap.add_argument("--checkpoint_layers", type=int, default=None,
                    help="只对前 N 层做梯度检查点（配合 --grad_checkpoint 使用）")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="dataloader 工作进程数；0 = 主线程同步取数（会阻塞训练循环）")
    ap.add_argument("--prefetch", type=int, default=2,
                    help="每个 worker 的预取批数")
    ap.add_argument("--eval_steps", type=int, default=0,
                    help=">0 则开启评测，每 N 个优化步一次（复刻 run_task.py 的边界）")
    ap.add_argument("--eval_batch_mult", type=int, default=4,
                    help="评测 batch = 训练 batch × 这个数（run_task.py 的 eval_batch_mult）")
    ap.add_argument("--save_on_cpu", action="store_true",
                    help="用 torch.autograd.graph.save_on_cpu **替换**梯度检查点："
                         "激活存到 pinned CPU 内存再取回，不重算（探针 E6）")
    args = ap.parse_args()

    if args.save_on_cpu:
        StepTimer.use_save_on_cpu = True
        print("save_on_cpu: 开启（并自动关闭梯度检查点，两者是互斥的两条路）")

    dev = "cuda"
    model = RetNetForCausalLM.from_pretrained(args.model_dir)
    print(f"chunkwise kernel: {'batched' if FAST_CHUNKWISE else 'reference loop'}")

    dd = Path(args.data_dir)
    probe = np.load(dd / "train.ids.npy", mmap_mode="r")
    seq_len = probe.shape[1] - 1
    del probe
    train_ds = PackedChatDataset(str(dd / "train.ids.npy"), str(dd / "train.wcode.npy"), seq_len)
    val_ds = PackedChatDataset(str(dd / "val.ids.npy"), str(dd / "val.wcode.npy"), seq_len)

    ta = TrainingArguments(
        output_dir=str(ROOT / "checkpoints" / "_bench_trainer"),
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size * args.eval_batch_mult,
        learning_rate=1.1e-4,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1.1e-5},
        warmup_steps=5,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        eval_strategy=("steps" if args.eval_steps else "no"),
        eval_steps=(args.eval_steps or None),
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=(args.prefetch if args.num_workers > 0 else None),
        dataloader_pin_memory=True,
        seed=1234,
        gradient_checkpointing=(not args.no_grad_checkpoint) and (not args.save_on_cpu),
        include_inputs_for_metrics=False,
    )
    if args.optim:
        ta.optim = args.optim
        print(f"optimizer: {args.optim}")
    model.config.mode = "chunkwise"
    model.config.chunk_size = args.chunk_size

    tr = StepTimer(model=model, args=ta, train_dataset=train_ds, eval_dataset=val_ds,
                   data_collator=ChatDataCollator())
    tr._set_signature_columns_if_needed()
    if args.checkpoint_layers is not None:
        n = tr.model.set_checkpoint_layers(args.checkpoint_layers)
        total = len(tr.model.blocks)
        print(f"分层检查点：前 {n}/{total} 层检查点（其余层保留激活）")
    if args.compile:
        kw = {"backend": "inductor"}
        if args.compile_mode:
            kw["mode"] = args.compile_mode
            print(f"torch.compile mode={args.compile_mode}")
        compiled = torch.compile(tr.model, **kw)
        tr.model = compiled
        # Trainer.__init__ 把**原始**模型存进了 self.model_wrapped，而 train()
        # 走的是 self._wrap_model(self.model_wrapped)。只替换 tr.model 的话，
        # 编译后的模型根本不会被调用 —— 这正是 dynamo counters 为空的原因。
        tr.model_wrapped = compiled

    t0 = time.perf_counter()
    tr.train()
    wall = time.perf_counter() - t0

    # --- 诊断：编译到底有没有被用上 ---
    print("\n  --- 诊断 ---")
    m = tr.model
    chain = []
    for _ in range(6):
        chain.append(type(m).__name__)
        m = getattr(m, "module", None) or getattr(m, "_orig_mod", None)
        if m is None:
            break
    print(f"  tr.model 包装链: {' -> '.join(chain)}")
    try:
        from torch._dynamo.utils import counters
        cc = {k: v for k, v in counters.items() if v}
        print(f"  dynamo counters: {cc if cc else '{}  ← 什么都没编译！'}")
    except Exception as exc:
        print(f"  counters 不可用: {exc}")
    try:
        import torch._dynamo as _d
        print(f"  dynamo 统计: {dict(list(_d.utils.counters.items())[:6])}")
    except Exception:
        pass

    ts = tr.step_times
    tail = ts[args.warm_skip:] if len(ts) > args.warm_skip else ts
    med = statistics.median(tail)          # 每微步
    opt_ms = med * args.grad_accum         # 每个优化步
    eff_batch = args.batch_size * args.grad_accum
    tok_step = eff_batch * seq_len
    peak_a = torch.cuda.max_memory_allocated() / 2**30
    peak_r = torch.cuda.max_memory_reserved() / 2**30
    print(f"\n  steps={len(ts)} 墙钟={wall:.1f}s   "
          f"batch={args.batch_size} × accum={args.grad_accum} = 等效 {eff_batch}")
    print(f"  微步中位 {med*1000:8.0f} ms/步  "
          f"（min {min(tail)*1000:.0f} / max {max(tail)*1000:.0f}）"
          f"   跳过前 {args.warm_skip}")
    print(f"  优化步   {opt_ms*1000:8.0f} ms/{tok_step} token")
    print(f"  吞吐     {tok_step/opt_ms:8.0f} tok/s   （{tok_step/opt_ms/1000:.2f}k）")
    print(f"  每 token {opt_ms/tok_step*1000:.4f} ms")
    print(f"  峰值显存 alloc {peak_a:5.2f} GiB / reserved {peak_r:5.2f} GiB"
          f"   （卡总容量 8.00 GiB）")
    print(f"  显存余量 {8.0 - peak_r:5.2f} GiB")
    if len(ts) > 3:
        print(f"  前 3 步：{['%.0f' % (x*1000) for x in ts[:3]]} ms  ← 含编译")
        print(f"  末 3 步：{['%.0f' % (x*1000) for x in ts[-3:]]} ms")
    if tr.eval_times:
        n_val = len(tr.eval_dataset)
        eb = args.batch_size * args.eval_batch_mult
        print(f"\n  评测 {len(tr.eval_times)} 次（val {n_val} 条 · eval batch {eb}）")
        for k, t in enumerate(tr.eval_times):
            print(f"    第{k + 1}次 {t:7.2f}s → {n_val * seq_len / t:9.0f} tok/s"
                  f"   （只有前向；训练含反向是 {tok_step / opt_ms:.0f} tok/s）")
        med = statistics.median(tr.eval_times)
        print(f"    中位 {med:.2f}s，按 eval_steps=2500、50000 步算 → "
              f"{med * 20 / 60:.1f} 分钟，占 "
              f"{med * 20 / (med * 20 + 50000 * opt_ms):.1%} 的运行时间")
    hist = [e for e in tr.state.log_history if "loss" in e]
    if hist:
        print(f"  loss 首/末: {hist[0]['loss']:.4f} -> {hist[-1]['loss']:.4f}  "
              f"（{len(hist)} 条记录，非 NaN 说明训练真的在跑）")
        bad = [e for e in hist if not (e["loss"] == e["loss"])]
        print(f"  NaN 记录数: {len(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
