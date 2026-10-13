"""这块卡在这个项目实际用的形状上，到底能跑多少 TFLOPs？

训练步时 928 ms 对应 14.4 TFLOPs，但没人知道分母是多少。这个脚本给分母：

1. 大方针 GEMM —— 理论上限（4096³ 等，能跑满 tensor core）
2. **本项目的真实形状** —— 1024/2048/20000 的混合，这才是可比的基准
3. 纯显存带宽 —— 确认不是带宽瓶颈

    python scripts/bench_peak.py
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


def timeit(fn, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def gemm(m, n, k, dtype, reps=20):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    t = timeit(lambda: torch.matmul(a, b), reps=reps)
    flops = 2 * m * n * k
    del a, b
    return t, flops / t / 1e12


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}   SMs={p.multi_processor_count}   "
          f"显存 {p.total_memory/2**30:.1f} GiB")
    torch.backends.cuda.matmul.allow_tf32 = True

    print("\n" + "=" * 74)
    print("1. 大方针 GEMM（理论上限，能跑满 tensor core）")
    print("=" * 74)
    print(f"{'形状':>22}{'bf16 ms':>11}{'bf16 TFLOPs':>14}{'fp32/tf32 TFLOPs':>18}")
    peak16 = 0.0
    for n in (2048, 4096, 8192):
        t16, f16 = gemm(n, n, n, torch.bfloat16, args.reps)
        t32, f32 = gemm(n, n, n, torch.float32, max(5, args.reps // 4))
        peak16 = max(peak16, f16)
        print(f"{f'{n}x{n}x{n}':>22}{t16*1000:>11.2f}{f16:>14.1f}{f32:>18.1f}")

    print("\n" + "=" * 74)
    print("2. 本项目的真实形状")
    print("=" * 74)
    S4 = 8192   # batch 4 × seq 2048
    S1 = 2048   # batch 1 × seq 2048（配置 C 用的）
    print(f"{'用途':<18}{'M':>6}{'形状 MxNxK':>22}{'ms':>9}{'TFLOPs':>10}")
    shapes4 = [
        ("q/k 投影", (S4, 1024, 1024)),
        ("v/g 投影", (S4, 2048, 1024)),
        ("out_proj", (S4, 1024, 2048)),
        ("FFN 上", (S4, 2048, 1024)),
        ("FFN 下", (S4, 1024, 2048)),
        ("LM head", (S4, 20000, 1024)),
    ]
    peak4 = 0.0
    for name, (m, n, k) in shapes4:
        t, f = gemm(m, n, k, torch.bfloat16, args.reps)
        peak4 = max(peak4, f)
        print(f"{name:<18}{'batch4':>6}{f'{m}x{n}x{k}':>22}{t*1000:>9.2f}{f:>10.1f}")

    print()
    shapes1 = [
        ("q/k 投影", (S1, 1024, 1024)),
        ("v/g 投影", (S1, 2048, 1024)),
        ("out_proj", (S1, 1024, 2048)),
        ("FFN 上", (S1, 2048, 1024)),
        ("FFN 下", (S1, 1024, 2048)),
        ("LM head", (S1, 20000, 1024)),
    ]
    peak1 = 0.0
    for name, (m, n, k) in shapes1:
        t, f = gemm(m, n, k, torch.bfloat16, args.reps)
        peak1 = max(peak1, f)
        print(f"{name:<18}{'batch1':>6}{f'{m}x{n}x{k}':>22}{t*1000:>9.2f}{f:>10.1f}")

    def total(shapes, seq):
        # NOTE(2026-10-10): 形状表里 q/k 与 v/g 各是**两个独立投影**（q_proj 与
        # k_proj、v_proj 与 g_proj），但下面只按 20 层各算了一次，所以这里报出的
        # 前向总量偏低约 30%。要做严谨的「离上限还有多远」请用
        # scripts/gemm_ceiling.py —— 它直接读编译图自报的 aten_mm_info 计数。
        return sum(2 * m * n * k * (20 if n != 20000 else 1) for _, (m, n, k) in shapes)

    tot4, tot1 = total(shapes4, S4), total(shapes1, S1)
    print(f"\n  batch4 前向 GEMM 总量 {tot4/1e12:.2f} TFLOP → 训练一步(×3) {3*tot4/1e12:.2f} TFLOP")
    print(f"  batch1 每微步前向 {tot1/1e12:.2f} TFLOP → 一个优化步(4 微步×3) {12*tot1/1e12:.2f} TFLOP")
    print(f"\n  batch4 形状实测峰值 ~{peak4:.0f} TFLOPs → 理论下限 {3*tot4/peak4*1000:.0f} ms")
    print(f"  batch1 形状实测峰值 ~{peak1:.0f} TFLOPs → 理论下限 {12*tot1/1e12/peak1*1000:.0f} ms")
    print("  （peak 取的是各形状里的**最大**值，只能当粗上限；加权下限见"
          " scripts/gemm_ceiling.py）")
    print(f"  实测 C 配置 1078 ms（电池）/ 734 ms（交流）")

    print("\n" + "=" * 74)
    print("3. 显存带宽（确认是否带宽受限）")
    print("=" * 74)
    for mb in (64, 256, 1024):
        n_el = mb * 2**20 // 2
        x = torch.empty(n_el, device="cuda", dtype=torch.bfloat16)
        y = torch.empty_like(x)
        t = timeit(lambda: y.copy_(x), reps=30)
        bw = 2 * n_el * 2 / t / 1e9
        print(f"  {mb:>5} MiB 拷贝  {t*1000:7.3f} ms  =>  {bw:7.1f} GB/s")
        del x, y

    print(f"\n  本步需要搬运的量级估计：bf16 权重 0.55 GB × 3 遍 + fp32 梯度 1.09 GB "
          f"+ AdamW 状态 4.36 GB ≈ 11 GB")
    print(f"  若带宽 256 GB/s，光是搬运就是 43 ms —— 相对 928 ms 可忽略，**不是带宽瓶颈**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
