"""按 **CUDA 内核**（不是算子归因）拆解一步训练的钱花在哪。

文档 §1 已经证明 profiler 的**算子归因会误导**：它把 `NativeDropoutBackward0`
记成 25%，实测关掉 dropout 只省 3.9%。原因是算子时间把子调用叠加计入，
而真正决定墙钟的是下面那一层内核。

这个脚本走 chrome trace 路径：导出后直接读 `cat == "kernel"` 的原始事件，
按内核名聚合，得到「谁真的占了 GPU 时间」和「有多少个内核、平均多长」。

平均内核时长是关键判据：
  * 平均只有几微秒 → 发射受限，融合或 CUDA Graphs 才有用
  * 平均几十微秒以上 → GPU 受限，只能少算

    python scripts/profile_kernels.py --batch 1 --grad_accum 4 --no_grad_checkpoint ^
        --optim adamw_bnb_8bit --compile --steps 8 --active 3
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.data import PackedChatDataset, ChatDataCollator        # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM, FAST_CHUNKWISE  # noqa: E402


def classify(name: str) -> str:
    n = name.lower()
    if "gemm" in n or "nvjet" in n or "cutlass" in n or "sm90" in n or "sm80" in n:
        return "GEMM"
    if "triton" in n:
        if "_red_" in n or "red" in n.split("_")[1:2]:
            return "Triton 归约"
        if "persistent" in n:
            return "Triton persistent"
        return "Triton 逐元素"
    if "elementwise" in n or "vectorized" in n:
        return "elementwise"
    if "norm" in n:
        return "归一化"
    if "copy" in n or "cat" in n or "slice" in n or "gather" in n:
        return "拷贝/索引"
    if "adam" in n or "multi_tensor" in n:
        return "优化器"
    if "flash" in n or "attention" in n or "fmha" in n:
        return "注意力"
    if "reduce" in n or "scan" in n:
        return "归约"
    if "act" in n or "silu" in n or "gelu" in n or "exp" in n or "softmax" in n:
        return "激活/数学"
    return "其它"


class ProfTimer(Trainer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.step_times: list[float] = []
        self._t0 = None
        self.prof = None

    def training_step(self, *a, **kw):
        self._t0 = time.perf_counter()
        out = super().training_step(*a, **kw)
        torch.cuda.synchronize()
        self.step_times.append(time.perf_counter() - self._t0)
        if self.prof is not None:
            self.prof.step()
        return out


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="checkpoints/retnet_272m_rp")
    ap.add_argument("--data_dir", default="data_cache/chat_pack_v13")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--active", type=int, default=3, help="真正采样的优化步数")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=128)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no_grad_checkpoint", action="store_true")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--out", default=None, help="chrome trace 输出路径")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--ops", action="store_true",
                    help="额外打印算子级 Top（含输入形状）。⚠️ §1 的归因警告仍然成立："
                         "算子时间会把子调用叠加计入，**不能拿来判占比**，但可以用来"
                         "**定位**某个内核是谁发的")
    args = ap.parse_args()

    model = RetNetForCausalLM.from_pretrained(args.model_dir)
    print(f"chunkwise kernel: {'batched' if FAST_CHUNKWISE else 'reference loop'}")
    dd = Path(args.data_dir)
    probe = np.load(dd / "train.ids.npy", mmap_mode="r")
    seq_len = probe.shape[1] - 1
    del probe
    train_ds = PackedChatDataset(str(dd / "train.ids.npy"), str(dd / "train.wcode.npy"), seq_len)
    val_ds = PackedChatDataset(str(dd / "val.ids.npy"), str(dd / "val.wcode.npy"), seq_len)

    ta = TrainingArguments(
        output_dir=str(ROOT / "checkpoints" / "_profile_kernels"),
        overwrite_output_dir=True,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size * 4,
        learning_rate=1.1e-4, lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1.1e-5}, warmup_steps=5, weight_decay=0.1,
        max_grad_norm=1.0, bf16=True, logging_steps=1, eval_strategy="no",
        save_strategy="no", report_to=[], dataloader_num_workers=0,
        dataloader_pin_memory=True, seed=1234,
        gradient_checkpointing=not args.no_grad_checkpoint,
    )
    ta.optim = args.optim
    model.config.mode = "chunkwise"
    model.config.chunk_size = args.chunk_size

    tr = ProfTimer(model=model, args=ta, train_dataset=train_ds, eval_dataset=val_ds,
                   data_collator=ChatDataCollator())
    tr._set_signature_columns_if_needed()
    if args.compile:
        compiled = torch.compile(tr.model, backend="inductor")
        tr.model = compiled
        tr.model_wrapped = compiled

    trace_path = args.out or os.path.join(tempfile.gettempdir(), "retnet_kernels.json")

    # 先跑几步让 inductor 编译完，再开 profiler 采样
    warmup_steps = max(args.steps - args.active, 1)
    ta.max_steps = warmup_steps
    tr.train()

    from torch.profiler import profile, ProfilerActivity, schedule
    tr.step_times.clear()
    tr.args.max_steps = tr.state.global_step + args.active
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=args.ops,
                 schedule=schedule(wait=0, warmup=0, active=args.active, repeat=1)) as prof:
        tr.prof = prof
        tr.train()
        tr.prof = None
    torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    size_mb = os.path.getsize(trace_path) / 2**20
    print(f"\ntrace: {trace_path}  ({size_mb:.1f} MB)")
    print(f"采样优化步时: {['%.0f' % (t*1000) for t in tr.step_times]} ms")

    with open(trace_path, "r", encoding="utf-8") as fh:
        trace = json.load(fh)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    del trace

    total = collections.Counter()
    count = collections.Counter()
    durs = collections.defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") != "kernel":
            continue
        d = ev.get("dur", 0)          # 微秒
        if d <= 0:
            continue
        nm = ev.get("name", "?")
        total[nm] += d
        count[nm] += 1
        durs[nm].append(d)
    del events

    if not total:
        print("没有采集到内核事件")
        return 1

    busy_us = sum(total.values())
    n_kern = sum(count.values())
    # 注意：Trainer.training_step 是**每个微批**调用一次，而 prof.step() 就写在
    # 那里，所以采样窗口是 active 个**微步**，不是优化步。
    n_micro = max(args.active, 1)
    n_opt = max(len(tr.step_times) / max(args.grad_accum, 1), 1)
    del n_opt
    print(f"\n内核总数 {n_kern}（{n_kern / n_micro:.0f}/微步，"
          f"{n_kern / n_micro * args.grad_accum:.0f}/优化步）  "
          f"累计忙时 {busy_us / 1000:.0f} ms（{busy_us / 1000 / n_micro:.0f} ms/微步）")
    print(f"平均内核时长 {busy_us / n_kern:.1f} µs"
          f"   中位 {statistics.median([d for v in durs.values() for d in v]):.1f} µs")

    print(f"\n  {'类别':<14}{'忙时(ms)':>10}{'占比':>8}{'内核数':>10}{'平均µs':>9}")
    cat_t = collections.Counter()
    cat_n = collections.Counter()
    for nm, t in total.items():
        c = classify(nm)
        cat_t[c] += t
        cat_n[c] += count[nm]
    for c, t in cat_t.most_common():
        print(f"  {c:<14}{t / 1000:>10.0f}{t / busy_us:>7.1%}{cat_n[c]:>10}"
              f"{t / cat_n[c]:>9.1f}")

    print(f"\n  单个内核 Top {args.top}（按累计忙时）")
    print(f"  {'忙时(ms)':>9}{'占比':>8}{'次数':>8}{'平均µs':>9}  内核名")
    for nm, t in total.most_common(args.top):
        print(f"  {t / 1000:>9.1f}{t / busy_us:>7.1%}{count[nm]:>8}"
              f"{t / count[nm]:>9.1f}  {nm[:96]}")

    if args.ops:
        print("\n  算子级 Top 30（含输入形状）")
        print("  ⚠️ 只用于**定位某个内核是谁发的**；右边的 ms 会把子调用叠加计入，"
              "不能当占比看（§1 的教训）")
        rows = [e for e in prof.key_averages(group_by_input_shape=True)
                if e.self_device_time_total > 0]
        rows.sort(key=lambda e: -e.self_device_time_total)
        for e in rows[:30]:
            print(f"  {e.self_device_time_total / 1000:>9.1f}ms {e.count:>6}x  "
                  f"{e.key[:64]:<64}{str(e.input_shapes)[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
