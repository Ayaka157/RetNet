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

import retnet_hf.modeling_retnet as MR                               # noqa: E402
from retnet_hf.retention_fast import chunkwise_batched  # noqa: E402
from retnet_hf.modeling_retnet import (RetNetHFConfig, MultiScaleRetention,  # noqa: E402
                                       RetNetForCausalLM)


class reference_path:
    """临时强制 ``_chunkwise_retention`` 走真正的参考实现。

    ⚠️ 这个测试原先有个致命缺陷：``modeling_retnet._chunkwise_retention`` 在
    ``RETNET_FAST_CHUNKWISE=1`` 时**直接转发给 ``chunkwise_batched``**，而训练
    环境恰好一直开着这个变量。于是「参考」就是被测函数本身，测试永远报逐位
    相同 —— 它从来没有真正验证过任何东西。

    所以取参考值时必须先把模块开关按下去。
    """

    def __enter__(self):
        self._saved = MR.FAST_CHUNKWISE
        MR.FAST_CHUNKWISE = False
        return self

    def __exit__(self, *exc):
        MR.FAST_CHUNKWISE = self._saved
        return False


def stats(a, b):
    """返回 (max|d|, rel_mean, mean|d|/mean|b|, rel_max)。

    第 4 项是关键：状态的**绝对**量级不设上界（γ>1 的头会把它放大到 1e13），
    拿绝对阈值 1e-4 去卡它毫无意义 —— 那正是原判据把一堆 1e-7 相对误差的
    正确结果判成「不等价」的原因。判等价一律看相对量。
    """
    a, b = a.float(), b.float()
    d = (a - b).abs()
    scale = b.abs().mean().clamp_min(1e-9)
    bmax = b.abs().max().clamp_min(1e-30)
    return (d.max().item(), (d / b.abs().clamp_min(1e-6)).mean().item(),
            (d.mean() / scale).item(), (d.max() / bmax).item())


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

    print(f"{'chunk':>7}{'out rel_mean':>15}{'out rel_max':>15}"
          f"{'state rel_max':>16}   判定")
    all_ok = True
    # 四种 gamma：默认初值（≈1）、真实 RP 门控范围（0.668~1.0117）、极端范围、
    # 以及**含 0** 门控（0 是合法值：门控可以学到「完全遗忘」，而 log(0) = -inf
    # 会让对数空间掩码在对角线算出 -inf*0 = NaN）。
    gammas = {
        "默认初值": msr.gamma.detach().clone(),
        "真实门控": torch.exp(torch.linspace(torch.log(torch.tensor(0.668)),
                                             torch.log(torch.tensor(1.0117193)), H, device=dev)),
        "极端下限": torch.exp(torch.linspace(torch.log(torch.tensor(0.30)),
                                             torch.log(torch.tensor(1.02)), H, device=dev)),
        "含零门控": torch.linspace(0.0, 1.0, H, device=dev),
    }
    for gname, g in gammas.items():
        print(f"\n  --- gamma: {gname}  min={g.min():.4f} max={g.max():.4f} ---")
        for C in (64, 128, 256, 512, 1024):
            try:
                with torch.no_grad():
                    msr.gamma.copy_(g.to(msr.gamma.dtype))
                    with reference_path():          # 必须关掉 FAST_CHUNKWISE！
                        o_ref, s_ref = msr._chunkwise_retention(q, k, v, st0, C)
                    o_new, s_new = chunkwise_batched(q, k, v, g, st0, C)
                bad = not (torch.isfinite(o_new).all() and torch.isfinite(s_new).all())
                _, rmo, _, rmax_o = stats(o_new, o_ref)
                _, rms, _, rmax_s = stats(s_new, s_ref)
                ok = (not bad) and rmo < 1e-4 and rmax_o < 1e-4 and rmax_s < 1e-4
                all_ok &= ok
                note = "  ← 出现 Inf/NaN" if bad else ""
                print(f"{C:>7}{rmo:>15.3e}{rmax_o:>15.3e}{rmax_s:>16.3e}   "
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

    def run(patch, amp):
        orig = MultiScaleRetention._chunkwise_retention if patch else None

        def patched(self, qq, kk, vv, state, chunk_size):
            return chunkwise_batched(qq, kk, vv,
                                     self.gamma.detach().to(torch.float32), state, chunk_size)
        if patch:
            MultiScaleRetention._chunkwise_retention = patched
        try:
            with torch.no_grad(), reference_path(), \
                    torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                return model._forward_hidden(ids, mode="chunkwise", return_state=True,
                                             chunk_size=128)[0].float()
        finally:
            if patch:
                MultiScaleRetention._chunkwise_retention = orig

    # --- 非空过闸门 -------------------------------------------------------
    # 这段模型级比较曾经**空过**：`MultiScaleRetention._init_weights` 把 1 维
    # Parameter 一律 zeros_，而 group_norm.weight 也是 1 维 —— 于是 retention
    # 支路输出恒为 0，两条路径当然逐位相同。判据是「差必须严格大于 0」：
    # 真的有差异才说明这个比较在跑。
    gn = [(n, p) for n, p in model.named_parameters() if n.endswith("group_norm.weight")]
    gn_ok = bool(gn) and all(float(p.detach().min()) != 0.0 for _, p in gn)
    print(f"  非空过闸门：group_norm.weight 不为 0 -> {gn_ok}  "
          f"（{len(gn)} 个张量）")
    if not gn_ok:
        print("  => 失败 ✗ retention 支路被初始化置零，这个比较没有意义")
        return 1

    # bf16 只做集成冒烟：bf16 有效位数 ~3 位十进制，阈值不可能压到 1e-5。
    ref_b, new_b = run(False, True), run(True, True)
    _, _, rmean_b, rmax_b = stats(new_b, ref_b)
    # 等价性判据放在 fp32 上（无 autocast），阈值才是有意义的 1e-5。
    ref_f, new_f = run(False, False), run(True, False)
    mo, rmo, rmean_f, rmax_f = stats(new_f, ref_f)
    print(f"  bf16  logits max|d| = {(new_b - ref_b).abs().max():.3e}   "
          f"rel_mean = {rmean_b:.3e}   rel_max = {rmax_b:.3e}   "
          f"（bf16 精度噪声量级，仅冒烟）")
    print(f"  fp32  logits max|d| = {mo:.3e}   rel_mean = {rmo:.3e}   "
          f"rel_max = {rmax_f:.3e}   mean|d|/mean|ref| = {rmean_f:.3e}")
    ok2 = rmean_f < 1e-5 and rmo < 1e-5 and rmax_f < 1e-5
    ok2_bf = rmean_b < 5e-2
    vacuous = mo == 0.0
    if vacuous:
        print("  => 失败 ✗ 两条路径逐位相同，比较退化为空过")
        ok2 = False
    print(f"  => fp32 {'等价 ✓' if ok2 else '不等价 ✗'} · "
          f"bf16 {'冒烟通过 ✓' if ok2_bf else '偏差过大 ✗'}")
    ok2 = ok2 and ok2_bf

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
            # 计时同理：不关掉开关的话「参考」列量的也是批量路径，加速比恒为 1。
            with reference_path():
                tr = timeit(lambda: msr._chunkwise_retention(qb, kb, vb, sb, C), args.reps)
            tn = timeit(lambda: chunkwise_batched(qb, kb, vb, gamma, sb, C), args.reps)
        print(f"{C:>7}{tr*1000:>12.2f}{tn*1000:>12.2f}{tr/tn:>8.2f}x")

    print(f"\n结论：单元级 {'通过 ✓' if all_ok else '失败 ✗'} · 模型级 {'通过 ✓' if ok2 else '失败 ✗'}")
    return 0 if (all_ok and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
