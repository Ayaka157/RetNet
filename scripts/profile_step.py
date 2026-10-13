"""把一步训练的 CUDA 时间按类别汇总，找出 GEMM 之外的钱花在哪。

bench_step.py --profile 只打前 20 行，且按算子名分组，看不出类别占比。
这里按 (GEMM / 融合逐元素 / 拷贝 / 归一化 / 其它) 归并。

    python scripts/profile_step.py --chunk_size 128 --compile
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.data import PackedChatDataset, ChatDataCollator  # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM, FAST_CHUNKWISE  # noqa: E402


def classify(name: str) -> str:
    n = name.lower()
    if "compiledfunction" in n or "autograd" in n:
        return "autograd 包装（与子项重叠）"
    if n.startswith("aten::mm") or "gemm" in n or n.startswith("aten::addmm"):
        return "GEMM (mm)"
    if n.startswith("aten::bmm") or "bmm" in n:
        return "GEMM (bmm)"
    if "triton_" in n:
        if "_red_" in n:
            return "Triton 归约融合"
        if "_persistent_" in n:
            return "Triton persistent"
        return "Triton 逐元素融合"
    if n.startswith("aten::copy_") or n.startswith("aten::clone") or n.startswith("aten::cat"):
        return "拷贝"
    if any(k in n for k in ("layer_norm", "group_norm", "native_layer", "native_group")):
        return "归一化"
    if n.startswith("aten::to") or "_to_copy" in n:
        return "dtype 转换"
    if any(k in n for k in ("mul", "add", "silu", "gelu", "softmax", "exp", "log", "pow",
                            "elementwise", "vectorized")):
        return "逐元素"
    if "dropout" in n:
        return "dropout"
    return "其它"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="checkpoints/retnet_272m_rp")
    ap.add_argument("--data_dir", default="data_cache/chat_pack_v13")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=128)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no_grad_checkpoint", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--optim", default=None)
    ap.add_argument("--rows", type=int, default=45)
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
    accum = max(1, args.grad_accum)

    def step():
        """一个完整优化步：accum 个微步 + 一次 optimizer.step()。"""
        for _ in range(accum):
            with ctx:
                out = model(input_ids=b["input_ids"], labels=b["labels"],
                            wcodes=b["wcodes"])
            (out.loss / accum).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    print(f"chunkwise kernel: {'batched' if FAST_CHUNKWISE else 'reference'} · "
          f"chunk={args.chunk_size} · compile={args.compile} · "
          f"ckpt={not args.no_grad_checkpoint} · batch={args.batch}×accum={accum} · "
          f"optim={args.optim or 'adamw_torch'}")
    for _ in range(6):
        step()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
    torch.cuda.synchronize()

    evs = prof.key_averages()
    buckets: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0])
    rows = []
    for e in evs:
        t = getattr(e, "self_device_time_total", 0.0) / 1000.0  # us -> ms
        if t <= 0:
            continue
        rows.append((t, e.key, e.count))
        c = classify(e.key)
        buckets[c][0] += t
        buckets[c][1] += e.count

    total = sum(v[0] for v in buckets.values())
    print(f"\n=== 按类别汇总（CUDA 自用时合计 {total:.0f} ms）===")
    print(f"{'类别':<26}{'ms':>9}{'占比':>8}{'调用数':>9}")
    for c, (t, n) in sorted(buckets.items(), key=lambda kv: -kv[1][0]):
        print(f"{c:<26}{t:>9.1f}{t/total*100:>7.1f}%{n:>9}")

    gemm = buckets["GEMM (mm)"][0] + buckets["GEMM (bmm)"][0]
    print(f"\nGEMM 合计 {gemm:.0f} ms（{gemm/total*100:.0f}%）  "
          f"非 GEMM {total-gemm:.0f} ms（{(total-gemm)/total*100:.0f}%）")
    print(f"若 GEMM 跑到实测峰值 40 TFLOPs，GEMM 部分本应约 257 ms → "
          f"当前 {gemm:.0f} ms，效率 {257/max(gemm,1)*100:.0f}%")

    print(f"\n=== 自用 CUDA 时间前 {args.rows} 名 ===")
    for t, k, n in sorted(rows, reverse=True)[:args.rows]:
        print(f"{t:>8.1f} ms  ×{n:<5} {k[:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
