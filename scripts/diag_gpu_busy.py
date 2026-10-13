"""这一步到底是 GPU 受限，还是内核发射/CPU 供给受限？

决定性的判据：把**同一次运行**里的
  * 墙钟时间
  * GPU 上所有内核的累计忙时（profiler 的 CUDA total）
  * 启动的内核数
放在一起比。

若 CUDA 忙时 ≈ 墙钟 → GPU 受限，优化只能靠「少算」或「算得更快」。
若 CUDA 忙时明显小于墙钟 → 有 GPU 空闲，瓶颈在发射/CPU，靠融合或 CUDA Graphs
才可能赢；这种情况下摘掉某个内核**不会**改变墙钟，正好解释前三次微优化全 0%。

    python scripts/diag_gpu_busy.py --batch 1 --grad_accum 4 --no_grad_checkpoint ^
        --optim adamw_bnb_8bit --compile
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.data import PackedChatDataset, ChatDataCollator          # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM, FAST_CHUNKWISE  # noqa: E402


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="checkpoints/retnet_272m_rp")
    ap.add_argument("--data_dir", default="data_cache/chat_pack_v13")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=128)
    ap.add_argument("--no_grad_checkpoint", action="store_true")
    ap.add_argument("--optim", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    dev = "cuda"
    model = RetNetForCausalLM.from_pretrained(args.model_dir).to(dev).train()
    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable()
    model.config.mode = "chunkwise"
    model.config.chunk_size = args.chunk_size
    dd = Path(args.data_dir)
    ds = PackedChatDataset(str(dd / "train.ids.npy"), str(dd / "train.wcode.npy"), 2048)
    col = ChatDataCollator()
    b = {k: v.to(dev) for k, v in col([ds[i] for i in range(args.batch)]).items()}
    if args.optim == "adamw_bnb_8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-6)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    if args.compile:
        model = torch.compile(model, backend="inductor")

    ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    A = max(1, args.grad_accum)

    def step():
        for _ in range(A):
            with ctx:
                out = model(input_ids=b["input_ids"], labels=b["labels"],
                            wcodes=b["wcodes"])
            (out.loss / A).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    print(f"batch={args.batch}×accum={A} chunk={args.chunk_size} "
          f"ckpt={not args.no_grad_checkpoint} compile={args.compile} "
          f"optim={args.optim or 'adamw_torch'}  FAST={FAST_CHUNKWISE}")

    for _ in range(4):
        step()
    torch.cuda.synchronize()

    # 墙钟
    ts = []
    for _ in range(args.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    wall = statistics.median(ts)

    # GPU 忙时与内核数
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
    torch.cuda.synchronize()
    evs = prof.key_averages()
    cuda_us = sum(getattr(e, "self_device_time_total", 0.0) for e in evs)
    n_kernels = sum(e.count for e in evs if getattr(e, "self_device_time_total", 0.0) > 0)
    n_unique = len([e for e in evs if getattr(e, "self_device_time_total", 0.0) > 0])

    print(f"\n  墙钟        {wall*1000:8.1f} ms")
    print(f"  GPU 累计忙时 {cuda_us/1000:8.1f} ms   →  占墙钟 {cuda_us/1000/(wall*1000)*100:.1f}%")
    print(f"  GPU 空闲     {(wall*1000-cuda_us/1000):8.1f} ms   →  {(wall*1000-cuda_us/1000)/(wall*1000)*100:.1f}%")
    print(f"  内核启动数   {n_kernels:8d} 次（{n_unique} 种），"
          f"平均 {cuda_us/max(n_kernels,1):.1f} µs/次")
    print()
    if cuda_us / 1000 / (wall * 1000) > 0.90:
        print("  判定：**GPU 受限**。摘掉任何单个内核都不会改变墙钟，"
              "只能靠减少总工作量或提高单核效率。")
    elif cuda_us / 1000 / (wall * 1000) > 0.75:
        print("  判定：**接近 GPU 受限**，但有可观空隙 —— 融合/减少启动仍有余地。")
    else:
        print("  判定：**发射/CPU 受限**。这就是前几次微优化全部 0% 的原因；"
              "真正的杠杆是减少内核数（融合）或 CUDA Graphs。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
