"""回归：GLA 逐头门控 gamma 被初始化置零 → 批量 chunkwise 前向 NaN。

发现经过（2026-10-11）
----------------------
`checkpoints/probe_500m` / `probe_600m` 的前向在**任意序列长度**下都是 NaN
（fp32、bf16、seq 128…2048 全是）。权重扫描显示 365 个张量**全部有限**，
所以不是「检查点损坏」那么简单。逐模块 hook 把断裂点定在
`blocks.0.msr`（输入 absmax 4.27、输出 NaN），而 `group_norm` 已经收到非有限输入。

根因有两层：

1. `MultiScaleRetention._init_weights` 里「1 维 Parameter 一律置零」那一支
   本意是兜底 bias，但本模块所有 `_proj` 都是 `bias=False`，于是 1 维
   Parameter **只剩 `gamma`** —— 门控被 `nn.init.zeros_` 置零。
2. `retention_fast._decay_tables` 用对数空间掩码 `exp(log γ · delta)`。
   γ=0 时 `log γ = -inf`，对角线上 `delta=0` 于是 `-inf * 0 = NaN`，
   而 `torch.where(causal, ...)` **保留**对角线 → 整层 NaN。
   参考实现用 `gamma ** delta`，`0**0 = 1` 是良定义的，所以两条路径在 γ=0 处不等价。

为什么既有测试没抓到
--------------------
`tests/test_fast_chunkwise.py` 的模型级用例用 `RetNetHFConfig(...)` 默认值构造，
而 `use_gla` 默认 **False**；此时 gamma 是 buffer 不是 Parameter，
`_init_weights` 的 `isinstance(p, nn.Parameter)` 判据把它排除了 → gamma 保持正确初值。
单元级用例用的是独立的 `MultiScaleRetention(cfg)`，根本不经过 `_init_weights`。
于是「use_gla=True 的整模型」这一条组合无人覆盖 —— 而生产配置正是 use_gla=True。

本测试覆盖的正是那条空缺组合。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retnet_hf.modeling_retnet as MR                                # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402
from retnet_hf.retention_fast import chunkwise_batched, _decay_tables  # noqa: E402

fails: list[str] = []


def check(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)
    return cond


def retnet_formula(H: int) -> torch.Tensor:
    """RetNet 论文的固定门控：gamma_h = 1 - 2^(-5-h)。"""
    return 1.0 - torch.pow(2.0, -5.0 - torch.arange(H, dtype=torch.float32))


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    dev = "cpu"
    H = 4

    # ---------------------------------------------------------------- 1
    print("1. use_gla=True 时，新建模型的 gamma 必须等于 RetNet 固定公式（不能是 0）")
    cfg = RetNetHFConfig(vocab_size=256, d_model=H * 16, num_layers=2, num_heads=H,
                         value_dim=H * 32, ffn_dim=64, dropout=0.0,
                         use_gla=True, chunk_size=64, mode="chunkwise")
    torch.manual_seed(0)
    model = RetNetForCausalLM(cfg).to(dev).eval()
    g = model.blocks[0].msr.gamma
    want = retnet_formula(H)
    check(g.numel() == H, f"gamma 形状 = {tuple(g.shape)}（期望 {H}）")
    check(float(g.detach().min()) > 0.0, f"gamma 最小值 {float(g.detach().min()):.6g} > 0（0 就是缺陷）")
    check(torch.allclose(g.detach().float(), want, atol=1e-7),
          f"gamma 等于公式 1-2^(-5-h)  实得 {[round(float(v), 6) for v in g.detach()]}  "
          f"期望 {[round(float(v), 6) for v in want]}")
    check(isinstance(dict(model.named_parameters()).get("blocks.0.msr.gamma"),
                     torch.nn.Parameter),
          "use_gla=True 时 gamma 是 nn.Parameter（可学习）")

    # 同一处缺陷的第二个受害者：group_norm.weight 也是 1 维 Parameter，
    # 被同一条「1 维一律置零」规则置了零 → retention 支路输出恒为 0。
    gn = [p for n, p in model.named_parameters() if n.endswith("group_norm.weight")]
    check(bool(gn) and all(float(p.detach().min()) > 0.0 for p in gn),
          f"group_norm.weight 不为 0（{len(gn)} 层，"
          f"min={min(float(p.detach().min()) for p in gn) if gn else float('nan'):.6g}）"
          f" —— 为 0 会让 retention 支路输出恒为 0")

    # ---------------------------------------------------------------- 2
    print("\n2. use_gla=True 整模型前向：批量路径必须有限，且与参考路径等价")
    ids = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        MR.FAST_CHUNKWISE = False
        out_ref = model(input_ids=ids, labels=ids)
        MR.FAST_CHUNKWISE = True
        out_fast = model(input_ids=ids, labels=ids)
    check(torch.isfinite(out_fast.logits).all(),
          f"FAST 路径 logits 全有限（absmax={float(out_fast.logits.abs().max()):.4g}）")
    check(torch.isfinite(out_fast.loss),
          f"FAST 路径 loss 有限（{float(out_fast.loss):.6f}）")
    check(torch.isfinite(out_ref.loss),
          f"参考路径 loss 有限（{float(out_ref.loss):.6f}）")
    d = (out_fast.logits.float() - out_ref.logits.float()).abs().max().item()
    check(d < 1e-3, f"两条路径 logits 最大差 {d:.3e} < 1e-3")

    # ---------------------------------------------------------------- 3
    print("\n3. gamma 恰好为 0 时，内核不能产生 NaN，且要与参考实现一致")
    B, kd, vhd, L, C = 1, 8, 8, 64, 64
    torch.manual_seed(1)
    q = torch.randn(B, H, L, kd)
    k = torch.randn(B, H, L, kd)
    v = torch.randn(B, H, L, vhd)
    gz = torch.zeros(H)
    im, cd, vd_, chd = _decay_tables(gz, C, torch.float32, torch.device("cpu"))
    check(torch.isfinite(im).all(), f"gamma=0 时 inner_mask 全有限（NaN {int(torch.isnan(im).sum())} 个）")
    check(torch.isfinite(cd).all(), f"gamma=0 时 cross_decay 全有限（NaN {int(torch.isnan(cd).sum())} 个）")
    check(torch.isfinite(vd_).all(), f"gamma=0 时 value_decay 全有限")
    check(torch.isfinite(chd).all(), f"gamma=0 时 chunk_decay 全有限")

    out_new, st_new = chunkwise_batched(q, k, v, gz, None, C)
    check(torch.isfinite(out_new).all(), f"gamma=0 时批量内核输出有限（NaN {int(torch.isnan(out_new).sum())} 个）")

    # 参考实现（gamma**delta，0**0=1）在 gamma=0 下的语义
    msr_cfg = RetNetHFConfig(vocab_size=64, d_model=H * kd, num_layers=1, num_heads=H,
                             value_dim=H * vhd, ffn_dim=64, dropout=0.0, use_gla=False)
    from retnet_hf.modeling_retnet import MultiScaleRetention     # noqa: E402
    msr = MultiScaleRetention(msr_cfg)
    with torch.no_grad():
        msr.gamma.copy_(gz)
        saved, MR.FAST_CHUNKWISE = MR.FAST_CHUNKWISE, False
        try:
            out_ref2, st_ref2 = msr._chunkwise_retention(q, k, v, None, C)
        finally:
            MR.FAST_CHUNKWISE = saved
    check(torch.isfinite(out_ref2).all(), "gamma=0 时参考内核输出有限")
    d2 = (out_new.float() - out_ref2.float()).abs().max().item()
    ds = (st_new.float() - st_ref2.float()).abs().max().item()
    check(d2 < 1e-5, f"gamma=0 时两路径输出最大差 {d2:.3e} < 1e-5")
    check(ds < 1e-5, f"gamma=0 时两路径状态最大差 {ds:.3e} < 1e-5")

    # ---------------------------------------------------------------- 4
    print("\n4. use_gla=False 的旧行为不能被改坏（gamma 仍是不可学习 buffer）")
    cfg2 = RetNetHFConfig(vocab_size=256, d_model=H * 16, num_layers=2, num_heads=H,
                          value_dim=H * 32, ffn_dim=64, dropout=0.0,
                          use_gla=False, chunk_size=64, mode="chunkwise")
    torch.manual_seed(0)
    m2 = RetNetForCausalLM(cfg2)
    check(not isinstance(dict(m2.named_parameters()).get("blocks.0.msr.gamma", None),
                         torch.nn.Parameter),
          "use_gla=False 时 gamma 仍是 buffer（不在 named_parameters 里）")
    check(torch.allclose(m2.blocks[0].msr.gamma.float(), retnet_formula(H), atol=1e-7),
          "use_gla=False 时 gamma 仍等于公式")

    print()
    if fails:
        print(f"[FAIL] {len(fails)} 项失败：")
        for f in fails:
            print("   - " + f)
        return 1
    print("[PASS] 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
