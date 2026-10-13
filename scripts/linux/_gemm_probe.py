"""一分钟量出新卡的 bf16 稠密算力 —— 用来终结「这卡到底多少 TFLOPS」的口水。

## 为什么先测这个

换卡决策一直卡在一个纯实测的问题上：RTX PRO 6000 Blackwell 的 bf16 稠密算力，
公开资料只能给出营销口径（`4000 AI TOPS` = **FP4 稀疏**），除以 8 得 500、
除以 16 得 250 —— 差一倍，而它直接决定「换卡省不省钱」。

**与其继续除营销数字，不如直接量。** 用大矩阵 bf16 matmul（PyTorch 默认 FP32 累加，
这正是训练用的口径）测有效 TFLOP/s。这个值就是这套 workload 的硬上限 ——
任何训练步的吞吐都不可能超过它。

## 口径

- PyTorch 的 `a @ b` 在 bf16 下默认走 **FP32 累加**（`allow_bf16_reduced_precision_reduction`
  默认 False），与训练一致。
- 取 8192³：单次 2×8192³ = 1.1 TFLOP，足够把 launch 开销摊掉。
- 也打印 fp16（FP16 累加）作为对照 —— 若两者差一倍，说明这张卡和 Ampere 消费级一样
  有「累加精度翻倍」的特性，引用规格时必须带口径。
"""
from __future__ import annotations

import time

import torch

torch.backends.cuda.matmul.allow_tf32 = True

dev = torch.cuda.current_device()
prop = torch.cuda.get_device_properties(dev)
print(f"GPU        : {prop.name}")
print(f"显存       : {prop.total_memory / 2**30:.1f} GiB")
print(f"SM 数      : {prop.multi_processor_count}")
print(f"CC         : {prop.major}.{prop.minor}")
print(f"torch      : {torch.__version__}   cuda {torch.version.cuda}")
print(f"累加口径   : allow_bf16_reduced_precision_reduction="
      f"{torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")


def bench(n: int, dtype, iters: int = 30, warm: int = 5) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(warm):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = 2.0 * n**3
    print(f"{str(dtype).replace('torch.',''):9s} {n}x{n}x{n}: {dt * 1000:8.3f} ms → "
          f"{flops / dt / 1e12:7.1f} TFLOP/s")
    del a, b, c
    torch.cuda.empty_cache()
    return flops / dt / 1e12


print()
tf32 = bench(8192, torch.bfloat16)
print()
tf16 = bench(8192, torch.float16)
print()
# ⚠️ 这台实例的默认是 allow_bf16_reduced_precision_reduction=True，那条路径会略微抬高
# 数值。训练口径应以**严格 FP32 累加**为准，所以再测一次关掉它的版本。
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
tf32_strict = bench(8192, torch.bfloat16)
print()
print(f"结论：bf16 实测有效峰值 = {tf32:.0f} TFLOP/s（实例默认口径）"
      f" / {tf32_strict:.0f} TFLOP/s（严格 FP32 累加）")
print(f"      fp16(FP16 累加) = {tf16:.0f} TFLOP/s（比值 {tf16 / tf32:.2f}）")
print("这个数就是训练吞吐的硬上限：吞吐 = 该值 ÷ 每 token FLOP（本项目 3.136 GFLOP）。")
print(f"→ 本 workload 的理论上限 = {tf32_strict * 1e12 / 3.136e9:,.0f} token/s（按严格口径）")

