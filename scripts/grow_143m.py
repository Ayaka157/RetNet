"""Grow 100M RetNet to ~143M via DEPTH growth (strictly function-preserving).

Why not width growth: expanding d_model / heads under pre-LayerNorm adds new
channels that shift every LayerNorm/GroupNorm statistic, so the residual stream
diverges (measured val loss jumped 4.5 -> 17.8).  Depth growth with
zero-initialized residual branches IS strictly function-preserving: the new
blocks output exactly 0, so the whole model is unchanged at init.

Upgrades applied:
  * GLA: per-head decay gamma becomes learnable (use_gla=True) — old layers are
    initialized to their exact previous gamma values, so behaviour is unchanged
    and the gate adapts during training.
  * +6 layers (12 -> 18) with zeroed MSR out_proj and FFN down weight.

    python scripts/grow_143m.py --src checkpoints/retnet_100m_pretrain_v4_hf \
                                 --dst checkpoints/retnet_143m_hf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetHFConfig, RetNetForCausalLM  # noqa: E402

OLD_L, NEW_L = 12, 18
D, H = 768, 12
VD = 1536


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

    cfg = RetNetHFConfig(vocab_size=old.config.vocab_size, d_model=D, num_layers=NEW_L,
                         num_heads=H, value_dim=VD, ffn_dim=VD,
                         dropout=old.config.dropout, norm_eps=old.config.norm_eps,
                         tie_weights=True, use_gla=True, use_swiglu=False)
    new = RetNetForCausalLM(cfg)
    new_sd = new.state_dict()

    new_sd["embedding.weight"] = old_sd["embedding.weight"]

    for i in range(NEW_L):
        p = f"blocks.{i}."
        if i < OLD_L:
            q = f"blocks.{i}."
            for key in ["msr.q_proj.weight", "msr.k_proj.weight", "msr.v_proj.weight",
                        "msr.g_proj.weight", "msr.out_proj.weight", "msr.group_norm.weight",
                        "msr.group_norm.bias", "norm1.weight", "norm1.bias",
                        "ffn.0.weight", "ffn.2.weight", "norm2.weight", "norm2.bias"]:
                new_sd[p + key] = old_sd[q + key].clone()
            # GLA：旧 γ 值照搬（1 - 2^{-5-h}, h=0..11），新 parameter 初始=旧值
            new_sd[p + "msr.gamma"] = old_sd[q + "msr.gamma"].clone()
        else:
            # 新层：MSR/FFN 全小噪声，但 out_proj 与 FFN down 置 0 → 残差为 0，函数保持
            g = torch.Generator().manual_seed(args.seed + i)
            new_sd[p + "msr.q_proj.weight"] = torch.randn(D, D, generator=g) * 0.02
            new_sd[p + "msr.k_proj.weight"] = torch.randn(D, D, generator=g) * 0.02
            new_sd[p + "msr.v_proj.weight"] = torch.randn(VD, D, generator=g) * 0.02
            new_sd[p + "msr.g_proj.weight"] = torch.randn(VD, D, generator=g) * 0.02
            new_sd[p + "msr.out_proj.weight"] = torch.zeros(D, VD)
            new_sd[p + "msr.group_norm.weight"] = torch.ones(VD)
            new_sd[p + "msr.group_norm.bias"] = torch.zeros(VD)
            new_sd[p + "norm1.weight"] = torch.ones(D)
            new_sd[p + "norm1.bias"] = torch.zeros(D)
            new_sd[p + "ffn.0.weight"] = torch.randn(VD, D, generator=g) * 0.02
            new_sd[p + "ffn.2.weight"] = torch.zeros(D, VD)   # 残差零
            new_sd[p + "norm2.weight"] = torch.ones(D)
            new_sd[p + "norm2.bias"] = torch.zeros(D)
            new_sd[p + "msr.gamma"] = (1.0 - torch.pow(2.0, -5.0 - torch.arange(H, dtype=torch.float32)))

    new_sd["final_norm.weight"] = old_sd["final_norm.weight"].clone()
    new_sd["final_norm.bias"] = old_sd["final_norm.bias"].clone()

    new.load_state_dict(new_sd, strict=True)
    new.tie_weights()
    print(f"grown params = {sum(p.numel() for p in new.parameters()) / 1e6:.2f}M")
    x = torch.randint(1, cfg.vocab_size, (2, 128))
    new.eval()
    with torch.inference_mode():
        out = new(x)
    print(f"smoke forward finite={torch.isfinite(out.logits).all().item()}")
    new.save_pretrained(args.dst, safe_serialization=True)
    print(f"saved -> {args.dst}")


if __name__ == "__main__":
    main()
