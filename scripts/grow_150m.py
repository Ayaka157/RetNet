"""Grow the 100M RetNet to a 152M model (d896/L14/H14) via net2net-style growth,
with the engineering upgrades from the research docs:
  - GLA: learnable per-head decay gamma (use_gla=True)
  - SwiGLU FFN (use_swiglu=True)

Growth rules (function-preserving where the activation is unchanged):
  * embedding: copy first 768 cols, small-noise the new 128 cols
  * width 768->896: block-copy old rows/cols, small-noise the new ones
  * depth 12->14: copy layers 0..11 to 0..11, insert 2 new layers with
    zero-initialized residual branches (out_proj=0, down_proj=0) so they start
    as identity and are learned gradually
  * heads 12->14: gamma sequence regenerated (1 - 2^{-5-h}), value_head_dim
    stays 128 (1792/14), so v/g/out projections migrate cleanly
  * GELU->SwiGLU: gate/up proj both seeded from the old FFN W1 (then perturbed
    to differ), down proj from old W2

    python scripts/grow_150m.py \
        --src checkpoints/retnet_100m_pretrain_v4_hf \
        --dst checkpoints/retnet_152m_hf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetHFConfig, RetNetForCausalLM  # noqa: E402

OLD_D, OLD_L, OLD_H = 768, 12, 12
NEW_D, NEW_L, NEW_H = 896, 14, 14
OLD_VD = 1536
NEW_VD = 1792


def grow_matrix(old: torch.Tensor, new_shape, noise_std=0.02, seed=0, zero_new=False):
    """Block-copy old into the top-left of new.

    zero_new=True: 新增行/列置 0（用于 MSR 投影，保证函数保持——新增 token 通道/新增
    头不向旧通道泄漏噪声）。False: 新增区域用小噪声（嵌入、SwiGLU 等）。
    """
    if zero_new:
        new = torch.zeros(new_shape, dtype=old.dtype)
    else:
        g = torch.Generator().manual_seed(seed)
        new = torch.randn(new_shape, generator=g, dtype=old.dtype) * noise_std
    new[:old.shape[0], :old.shape[1]] = old
    return new


def grow_vector(old: torch.Tensor, new_len, fill_new=1.0):
    new = torch.full((new_len,), fill_new, dtype=old.dtype)
    new[:old.shape[0]] = old
    return new


def grow_groupnorm_weight(old: torch.Tensor, old_groups, old_ch_per_group,
                          new_groups, new_ch_per_group):
    """GN weight: copy per-group blocks, new groups=1, new channels=1."""
    new = torch.ones((new_groups * new_ch_per_group,), dtype=old.dtype)
    for g in range(old_groups):
        new[g * new_ch_per_group:g * new_ch_per_group + old_ch_per_group] = \
            old[g * old_ch_per_group:(g + 1) * old_ch_per_group]
    return new


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    old = RetNetForCausalLM.from_pretrained(args.src)
    old_sd = old.state_dict()

    cfg = RetNetHFConfig(vocab_size=old.config.vocab_size, d_model=NEW_D, num_layers=NEW_L,
                         num_heads=NEW_H, value_dim=NEW_VD, ffn_dim=NEW_VD,
                         dropout=old.config.dropout, norm_eps=old.config.norm_eps,
                         tie_weights=True, use_gla=True, use_swiglu=True)
    new = RetNetForCausalLM(cfg)
    new_sd = new.state_dict()

    # embedding: (V, 896) <- (V, 768)
    new_sd["embedding.weight"] = grow_matrix(old_sd["embedding.weight"],
                                             (cfg.vocab_size, NEW_D), 0.02, args.seed)
    # lm_head tied, skip

    old_gn_ch = OLD_VD // OLD_H      # 128
    new_gn_ch = NEW_VD // NEW_H      # 128
    # layer mapping: old 12 -> new layers 0..11; insert new layers at index 7 and 12
    new_layers = set(range(NEW_L))
    for i in range(NEW_L):
        src_layer = None
        if i < 7:
            src_layer = i
        elif i == 7:
            src_layer = None            # inserted
        elif i < 12:
            src_layer = i - 1
        elif i == 12:
            src_layer = None            # inserted
        else:
            src_layer = i - 2
        p = f"blocks.{i}."
        if src_layer is not None:
            q = f"blocks.{src_layer}."
            # MSR projections (weight shape [out, in]) — 新增区域置 0，函数保持
            new_sd[p + "msr.q_proj.weight"] = grow_matrix(old_sd[q + "msr.q_proj.weight"], (NEW_D, NEW_D), 0.02, args.seed + i, zero_new=True)
            new_sd[p + "msr.k_proj.weight"] = grow_matrix(old_sd[q + "msr.k_proj.weight"], (NEW_D, NEW_D), 0.02, args.seed + i, zero_new=True)
            new_sd[p + "msr.v_proj.weight"] = grow_matrix(old_sd[q + "msr.v_proj.weight"], (NEW_VD, NEW_D), 0.02, args.seed + i, zero_new=True)
            new_sd[p + "msr.g_proj.weight"] = grow_matrix(old_sd[q + "msr.g_proj.weight"], (NEW_VD, NEW_D), 0.02, args.seed + i, zero_new=True)
            new_sd[p + "msr.out_proj.weight"] = grow_matrix(old_sd[q + "msr.out_proj.weight"], (NEW_D, NEW_VD), 0.02, args.seed + i, zero_new=True)
            new_sd[p + "msr.group_norm.weight"] = grow_groupnorm_weight(
                old_sd[q + "msr.group_norm.weight"], OLD_H, old_gn_ch, NEW_H, new_gn_ch)
            new_sd[p + "msr.group_norm.bias"] = grow_groupnorm_weight(
                old_sd[q + "msr.group_norm.bias"], OLD_H, old_gn_ch, NEW_H, new_gn_ch).zero_()
            new_sd[p + "norm1.weight"] = grow_vector(old_sd[q + "norm1.weight"], NEW_D, 1.0)
            new_sd[p + "norm1.bias"] = grow_vector(old_sd[q + "norm1.bias"], NEW_D, 0.0)
            new_sd[p + "norm2.weight"] = grow_vector(old_sd[q + "norm2.weight"], NEW_D, 1.0)
            new_sd[p + "norm2.bias"] = grow_vector(old_sd[q + "norm2.bias"], NEW_D, 0.0)
            # SwiGLU: gate/up 从旧 W1 拷贝、down 从旧 W2 拷贝；整体缩放 0.5 补偿
            # silu(x)·x 相对 gelu(x) 的幅值放大，避免 FFN 输出爆炸污染残差
            old_w1 = old_sd[q + "ffn.0.weight"]   # (1536, 768)
            old_w2 = old_sd[q + "ffn.2.weight"]   # (768, 1536)
            new_sd[p + "gate_proj.weight"] = grow_matrix(old_w1, (NEW_VD, NEW_D), 0.02, args.seed + i)
            new_sd[p + "up_proj.weight"] = grow_matrix(old_w1, (NEW_VD, NEW_D), 0.03, args.seed + i + 1)
            new_sd[p + "down_proj.weight"] = grow_matrix(old_w2, (NEW_D, NEW_VD), 0.02, args.seed + i) * 0.5
        else:
            # 插入层：残差零初始化（MSR out=0, FFN down=0），其余小噪声
            for name, shp in [(p + "msr.q_proj.weight", (NEW_D, NEW_D)),
                              (p + "msr.k_proj.weight", (NEW_D, NEW_D)),
                              (p + "msr.v_proj.weight", (NEW_VD, NEW_D)),
                              (p + "msr.g_proj.weight", (NEW_VD, NEW_D))]:
                new_sd[name] = torch.randn(shp, generator=torch.Generator().manual_seed(args.seed + i)) * 0.02
            new_sd[p + "msr.out_proj.weight"] = torch.zeros(NEW_D, NEW_VD)
            new_sd[p + "msr.group_norm.weight"] = torch.ones(NEW_VD)
            new_sd[p + "msr.group_norm.bias"] = torch.zeros(NEW_VD)
            new_sd[p + "norm1.weight"] = torch.ones(NEW_D)
            new_sd[p + "norm1.bias"] = torch.zeros(NEW_D)
            new_sd[p + "norm2.weight"] = torch.ones(NEW_D)
            new_sd[p + "norm2.bias"] = torch.zeros(NEW_D)
            new_sd[p + "gate_proj.weight"] = torch.randn(NEW_VD, NEW_D,
                generator=torch.Generator().manual_seed(args.seed + i)) * 0.02
            new_sd[p + "up_proj.weight"] = torch.randn(NEW_VD, NEW_D,
                generator=torch.Generator().manual_seed(args.seed + i + 1)) * 0.02
            new_sd[p + "down_proj.weight"] = torch.zeros(NEW_D, NEW_VD)
        # GLA gamma 初始 = 标准公式值（可学习）
        new_sd[p + "msr.gamma"] = (1.0 - torch.pow(2.0, -5.0 - torch.arange(NEW_H, dtype=torch.float32)))

    new_sd["final_norm.weight"] = grow_vector(old_sd["final_norm.weight"], NEW_D, 1.0)
    new_sd["final_norm.bias"] = grow_vector(old_sd["final_norm.bias"], NEW_D, 0.0)

    missing = set(new_sd.keys()) - set(old_sd.keys())
    # 检查：新模型所有 key 都已赋值（gamma 是 parameter 在 state_dict 里）
    new.load_state_dict(new_sd, strict=True)
    new.tie_weights()
    n_params = sum(p.numel() for p in new.parameters())
    print(f"grown model params = {n_params / 1e6:.2f}M")

    # 冒烟：前向正常、loss 不爆炸
    x = torch.randint(1, cfg.vocab_size, (2, 128))
    new.eval()
    with torch.inference_mode():
        out = new(x)
    print(f"smoke forward logits shape={out.logits.shape}, finite={torch.isfinite(out.logits).all().item()}")

    new.save_pretrained(args.dst, safe_serialization=True)
    print(f"saved -> {args.dst}")


if __name__ == "__main__":
    main()
