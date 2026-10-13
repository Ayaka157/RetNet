"""一步训练的 GEMM 到底离硬件上限有多远 —— 用编译图自报的真实形状算，不靠估计。

两个输入都是实测的，不是估的：

1. **形状清单**来自 `bench_trainer.py` 打印的 dynamo counter `aten_mm_info`，
   也就是 torch.compile 捕获到的那张图**实际发出**的 aten::mm / aten::bmm 调用。
   计数与 LM head 的 fwd+bwd 三条各为 1 完全自洽，所以那些计数就是**一个微步**。
2. **每个形状的可达峰值**用本文件的 `peak_mm` / `peak_bmm` 现场测，而不是拿
   大方针 GEMM 的 42 TFLOPs 冒充 —— batch 1 的形状实测只有 26~33 TFLOPs。

于是「GEMM 下限」= Σ FLOPs_i / peak_i，可以直接和 profiler 量到的
「GEMM 内核忙时」对比。比值接近 1 就说明 GEMM 已经到顶，剩下的只能从别处找。

    python scripts/gemm_ceiling.py --reps 30

注：`bench_peak.py` 的 `total()` 把 q/k 与 v/g 各漏乘了一次（那是两个独立投影），
所以它报的前向总量偏低约 30%；本脚本用实测计数，不存在这个问题。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch


def bench(fn, reps=20, target_s=0.006):
    """把 ``fn`` 连续跑 ``inner`` 次再除以次数。

    ⚠️ 关键：**不能在每次调用之间 synchronize**。小的 batched GEMM（本项目的
    chunkwise 批量乘只有 64x64/64x128，单次约 0.1 ms）会被 sync + Python 的
    固定开销淹掉，实测会低估三到十倍 —— 第一版就是这么把「GEMM 下限」算成
    142.9 ms 的，比 profiler 实测的忙时还高。这里先估一次耗时，再决定内层
    循环次数，让每次采样至少 ~6 ms。
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = max(time.perf_counter() - t0, 1e-7)
    inner = max(1, int(target_s / one))

    for _ in range(3):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()

    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / inner)
    return statistics.median(ts)


def peak_mm(m, n, k, dtype, reps):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    t = bench(lambda: a @ b, reps=reps)
    return 2 * m * n * k / t / 1e12, t


def peak_bmm(batch, m, n, k, dtype, reps):
    a = torch.randn(batch, m, k, device="cuda", dtype=dtype)
    b = torch.randn(batch, k, n, device="cuda", dtype=dtype)
    t = bench(lambda: torch.bmm(a, b), reps=reps)
    return 2 * batch * m * n * k / t / 1e12, t


# 来自 bench_trainer 的 dynamo counter `aten_mm_info`（一个微步的实测计数）
MM = [
    ("mm", (2048, 1024, 2048), 160),
    ("mm", (2048, 2048, 1024), 100),
    ("mm", (2048, 1024, 1024), 80),
    ("mm", (1024, 2048, 2048), 40),
    ("mm", (1024, 1024, 2048), 40),
    ("mm", (2048, 20000, 1024), 1),
    ("mm", (20000, 1024, 2048), 1),
    ("mm", (2048, 1024, 20000), 1),
    ("bmm", (512, 64, 128, 64), 120),
    ("bmm", (512, 64, 64, 64), 60),
    ("bmm", (512, 64, 64, 128), 60),
]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--measured-gemm-ms", type=float, default=104.0,
                    help="profiler 量到的 GEMM 内核忙时（ms/微步）。"
                         "当前值来自 profile_kernels.py：312 ms / 3 个微步。")
    ap.add_argument("--wall-ms", type=float, default=166.0,
                    help="优化步墙钟 / 4（ms/微步）")
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  SMs={p.multi_processor_count}")
    torch.backends.cuda.matmul.allow_tf32 = True
    dt = torch.bfloat16

    print(f"\n{'类型':<6}{'形状':>24}{'次数':>7}{'TFLOPs':>9}{'单次ms':>9}"
          f"{'小计GFLOP':>12}{'小计ms':>9}")
    tot_flop = 0.0
    floor_ms = 0.0
    for kind, shape, cnt in MM:
        if kind == "mm":
            m, n, k = shape
            pk, t = peak_mm(m, n, k, dt, args.reps)
            fl = 2 * m * n * k
        else:
            b, m, n, k = shape
            pk, t = peak_bmm(b, m, n, k, dt, args.reps)
            fl = 2 * b * m * n * k
        sub_fl = fl * cnt
        tot_flop += sub_fl
        floor_ms += sub_fl / (pk * 1e12) * 1000
        print(f"{kind:<6}{'x'.join(map(str, shape)):>24}{cnt:>7}{pk:>9.1f}"
              f"{t * 1000:>9.3f}{sub_fl / 1e9:>12.1f}{sub_fl / (pk * 1e12) * 1000:>9.3f}")

    print(f"\n  一个微步 GEMM 总量        {tot_flop / 1e12:8.3f} TFLOP")
    print(f"  按实测形状峰值算的下限    {floor_ms:8.1f} ms/微步")
    print(f"  profiler 实测 GEMM 忙时   {args.measured_gemm_ms:8.1f} ms/微步")
    ratio = args.measured_gemm_ms / floor_ms if floor_ms else 0
    print(f"  → 实测 / 下限 = {ratio:.2f}x   "
          f"（≈1 表示 GEMM 已到这批形状的硬件上限）")
    non = args.wall_ms - args.measured_gemm_ms
    print(f"\n  微步墙钟                  {args.wall_ms:8.1f} ms")
    print(f"  其中 GEMM                 {args.measured_gemm_ms:8.1f} ms "
          f"({args.measured_gemm_ms / args.wall_ms:.1%})")
    print(f"  其中非 GEMM（逐元素/归约）{non:8.1f} ms "
          f"({non / args.wall_ms:.1%})")
    print(f"\n  若 GEMM 已经 100% 达峰，则理论最快 ≈ "
          f"{floor_ms + non:8.1f} ms/微步（再快 {(1 - (floor_ms + non) / args.wall_ms):.1%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
