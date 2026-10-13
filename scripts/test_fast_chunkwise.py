"""Verify the batched-chunk chunkwise retention against the reference loop.

Two levels:

* unit  — same inputs, fp32, no autocast; checks the *math*.
* model — full RetNetForCausalLM forward under bf16 autocast, reference path vs
  batched path; checks the *integration* (this is what training actually runs).

    python scripts/test_fast_chunkwise.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.retention_fast import chunkwise_batched  # noqa: E402
from retnet_hf.modeling_retnet import (RetNetHFConfig, MultiScaleRetention,  # noqa: E402
                                       RetNetForCausalLM)


def stats(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    scale = b.abs().mean().clamp_min(1e-9)
    return d.max().item(), (d / b.abs().clamp_min(1e-6)).mean().item(), (d.mean() / scale).item()


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--key_dim", type=int, default=64)
    ap.add_argument("--value_head_dim", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    H, kd, vhd, L, B = args.heads, args.key_dim, args.value_head_dim, args.seq_len, args.batch
    d_model, value_dim = H * kd, H * vhd

    print("=" * 78)
    print("1. 单元级等价（fp32，无 autocast）—— 检验数学")
    print("=" * 78)
    cfg = RetNetHFConfig(vocab_size=1000, d_model=d_model, num_layers=1, num_heads=H,
                         value_dim=value_dim, ffn_dim=d_model, dropout=0.0)
    msr = MultiScaleRetention(cfg).to(dev)
    gamma = msr.gamma.detach().to(dev)
    q = torch.randn(B, H, L, kd, device=dev)
    k = torch.randn(B, H, L, kd, device=dev)
    v = torch.randn(B, H, L, vhd, device=dev)
    st0 = torch.randn(B, H, kd, vhd, device=dev) * 0.1

    print(f"{'chunk':>7}{'out max|d|':>13}{'out rel_mean':>15}{'state max|d|':>15}   判定")
    all_ok = True
    # 三种 gamma：默认初值（≈1）、真实 RP 门控范围（0.668~1.0117）、极端范围
    gammas = {
        "默认初值": msr.gamma.detach().clone(),
        "真实门控": torch.exp(torch.linspace(torch.log(torch.tensor(0.668)),
                                             torch.log(torch.tensor(1.0117193)), H, device=dev)),
        "极端下限": torch.exp(torch.linspace(torch.log(torch.tensor(0.30)),
                                             torch.log(torch.tensor(1.02)), H, device=dev)),
    }
    for gname, g in gammas.items():
        print(f"\n  --- gamma: {gname}  min={g.min():.4f} max={g.max():.4f} ---")
        for C in (64, 128, 256, 512, 1024):
            try:
                with torch.no_grad():
                    msr.gamma.copy_(g.to(msr.gamma.dtype))
                    o_ref, s_ref = msr._chunkwise_retention(q, k, v, st0, C)
                    o_new, s_new = chunkwise_batched(q, k, v, g, st0, C)
                bad = not (torch.isfinite(o_new).all() and torch.isfinite(s_new).all())
                mo, rmo, _ = stats(o_new, o_ref)
                ms, _, _ = stats(s_new, s_ref)
                ok = (not bad) and rmo < 1e-4 and ms < 1e-4
                all_ok &= ok
                note = "  ← 出现 Inf/NaN" if bad else ""
                print(f"{C:>7}{mo:>13.3e}{rmo:>15.3e}{ms:>15.3e}   "
                      f"{'等价 ✓' if ok else '不等价 ✗'}{note}")
            except Exception as exc:
                all_ok = False
                print(f"{C:>7}   异常: {type(exc).__name__}: {exc}")
    msr.gamma.copy_(gammas["默认初值"].to(msr.gamma.dtype))

    print()
    print("=" * 78)
    print("2. 模型级等价（bf16 autocast，整模型前向）—— 检验集成")
    print("=" * 78)
    mcfg = RetNetHFConfig(vocab_size=2000, d_model=256, num_layers=3, num_heads=4,
                          value_dim=512, ffn_dim=512, dropout=0.0)
    torch.manual_seed(1)
    model = RetNetForCausalLM(mcfg).to(dev).eval()
    ids = torch.randint(0, 2000, (2, 512), device=dev)

    def run(patch):
        orig = MultiScaleRetention._chunkwise_retention if patch else None

        def patched(self, qq, kk, vv, state, chunk_size):
            return chunkwise_batched(qq, kk, vv,
                                     self.gamma.detach().to(torch.float32), state, chunk_size)
        if patch:
            MultiScaleRetention._chunkwise_retention = patched
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return model._forward_hidden(ids, mode="chunkwise", return_state=True,
                                             chunk_size=128)[0].float()
        finally:
            if patch:
                MultiScaleRetention._chunkwise_retention = orig

    ref = run(False)
    new = run(True)
    mo, rmo, rmean = stats(new, ref)
    print(f"  logits max|d| = {mo:.3e}   rel_mean = {rmo:.3e}   mean|d|/mean|ref| = {rmean:.3e}")
    ok2 = rmean < 1e-5
    print(f"  => {'等价 ✓' if ok2 else '不等价 ✗'}")

    print()
    print("=" * 78)
    print("3. 速度（单层 retention，bf16 autocast）")
    print("=" * 78)
    qb = q.to(torch.bfloat16); kb = k.to(torch.bfloat16)
    vb = v.to(torch.bfloat16); sb = st0.to(torch.bfloat16)

    def timeit(fn, reps):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn()
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    print(f"{'chunk':>7}{'参考 (ms)':>12}{'批量 (ms)':>12}{'加速':>9}")
    for C in (64, 128, 256, 512, 1024):
        with ac():
            tr = timeit(lambda: msr._chunkwise_retention(qb, kb, vb, sb, C), args.reps)
            tn = timeit(lambda: chunkwise_batched(qb, kb, vb, gamma, sb, C), args.reps)
        print(f"{C:>7}{tr*1000:>12.2f}{tn*1000:>12.2f}{tr/tn:>8.2f}x")

    print(f"\n结论：单元级 {'通过 ✓' if all_ok else '失败 ✗'} · 模型级 {'通过 ✓' if ok2 else '失败 ✗'}")
    return 0 if (all_ok and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
