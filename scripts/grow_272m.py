"""Grow 143M (d768/L18/H12, GLA) to 272M (d1024/L20/H16, GLA) — STRICTLY
function-preserving via: GrowthLayerNorm (two-segment LN), zero-injected new
token channels, zero-branch new heads, zero-residual new layers.

Verification: grown model logits must equal the 143M model's logits exactly
(max|Δ| = 0) because every new channel/layer is a zero side branch.

    python scripts/grow_272m.py --src checkpoints/retnet_143m_hf \
        --dst checkpoints/retnet_272m_hf --verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402

OLD_D, NEW_D = 768, 1024
OLD_L, NEW_L = 18, 20
OLD_H, NEW_H = 12, 16
OLD_VD, NEW_VD = 1536, 2048
INSERT_LAYERS = {18, 19}          # new zero-residual layers


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    old = RetNetForCausalLM.from_pretrained(args.src)
    old_sd = old.state_dict()

    cfg = RetNetHFConfig(vocab_size=old.config.vocab_size, d_model=NEW_D,
                         num_layers=NEW_L, num_heads=NEW_H, value_dim=NEW_VD,
                         ffn_dim=NEW_VD, dropout=old.config.dropout,
                         norm_eps=old.config.norm_eps, tie_weights=True,
                         use_gla=True, use_swiglu=False,
                         use_growth_ln=True, growth_old_dim=OLD_D,
                         embed_scale=OLD_D ** 0.5)   # 保持旧缩放，函数保真
    new = RetNetForCausalLM(cfg)
    new_sd = new.state_dict()

    def fill_proj(p: str, old_key: str, out_d: int, in_d: int):
        """Copy old into top-left of new (new rows/cols stay 0)."""
        w = torch.zeros(out_d, in_d)
        old_w = old_sd[old_key]
        w[: old_w.shape[0], : old_w.shape[1]] = old_w
        new_sd[p] = w

    # embedding: copy old cols, new cols = 0
    emb = torch.zeros(cfg.vocab_size, NEW_D)
    emb[:, :OLD_D] = old_sd["embedding.weight"]
    new_sd["embedding.weight"] = emb

    for i in range(NEW_L):
        p = f"blocks.{i}."
        if i not in INSERT_LAYERS and i < OLD_L:
            q = f"blocks.{i}."
            fill_proj(p + "msr.q_proj.weight", q + "msr.q_proj.weight", NEW_D, NEW_D)
            fill_proj(p + "msr.k_proj.weight", q + "msr.k_proj.weight", NEW_D, NEW_D)
            fill_proj(p + "msr.v_proj.weight", q + "msr.v_proj.weight", NEW_VD, NEW_D)
            fill_proj(p + "msr.g_proj.weight", q + "msr.g_proj.weight", NEW_VD, NEW_D)
            fill_proj(p + "msr.out_proj.weight", q + "msr.out_proj.weight", NEW_D, NEW_VD)
            # GroupNorm: old 12 groups -> new 16 groups (copy per-group blocks)
            gn_w = torch.ones(NEW_VD)
            gn_b = torch.zeros(NEW_VD)
            for g in range(OLD_H):
                gn_w[g * 128:(g + 1) * 128] = old_sd[q + "msr.group_norm.weight"][g * 128:(g + 1) * 128]
                gn_b[g * 128:(g + 1) * 128] = old_sd[q + "msr.group_norm.bias"][g * 128:(g + 1) * 128]
            new_sd[p + "msr.group_norm.weight"] = gn_w
            new_sd[p + "msr.group_norm.bias"] = gn_b
            # GLA gamma：旧 12 头拷贝（训练后已偏离公式值），新 4 头用公式值
            old_gamma = old_sd[q + "msr.gamma"]
            new_gamma = (1.0 - torch.pow(2.0, -5.0 - torch.arange(NEW_H, dtype=torch.float32)))
            new_gamma[:OLD_H] = old_gamma
            new_sd[p + "msr.gamma"] = new_gamma
            # norms (two-segment): copy ln_old, gamma_new=0(beta_new=0 => zero branch)
            for nk in ("norm1", "norm2"):
                new_sd[p + nk + ".ln_old.weight"] = old_sd[q + nk + ".weight"].clone()
                new_sd[p + nk + ".ln_old.bias"] = old_sd[q + nk + ".bias"].clone()
                new_sd[p + nk + ".gamma_new"] = torch.zeros(NEW_D - OLD_D)
                new_sd[p + nk + ".beta_new"] = torch.zeros(NEW_D - OLD_D)
            # FFN (GELU, unchanged activation): copy into top-left of larger proj
            w1 = old_sd[q + "ffn.0.weight"]     # (1536, 768)
            w2 = old_sd[q + "ffn.2.weight"]     # (768, 1536)
            n1 = torch.zeros(NEW_VD, NEW_D); n1[: w1.shape[0], : w1.shape[1]] = w1
            n2 = torch.zeros(NEW_D, NEW_VD); n2[: w2.shape[0], : w2.shape[1]] = w2
            new_sd[p + "ffn.0.weight"] = n1
            new_sd[p + "ffn.2.weight"] = n2
        else:
            # inserted zero-residual layers
            g = torch.Generator().manual_seed(i)
            new_sd[p + "msr.q_proj.weight"] = torch.randn(NEW_D, NEW_D, generator=g) * 0.02
            new_sd[p + "msr.k_proj.weight"] = torch.randn(NEW_D, NEW_D, generator=g) * 0.02
            new_sd[p + "msr.v_proj.weight"] = torch.randn(NEW_VD, NEW_D, generator=g) * 0.02
            new_sd[p + "msr.g_proj.weight"] = torch.randn(NEW_VD, NEW_D, generator=g) * 0.02
            new_sd[p + "msr.out_proj.weight"] = torch.zeros(NEW_D, NEW_VD)
            new_sd[p + "msr.group_norm.weight"] = torch.ones(NEW_VD)
            new_sd[p + "msr.group_norm.bias"] = torch.zeros(NEW_VD)
            new_sd[p + "msr.gamma"] = (1.0 - torch.pow(2.0, -5.0 - torch.arange(NEW_H, dtype=torch.float32)))
            for nk in ("norm1", "norm2"):
                new_sd[p + nk + ".ln_old.weight"] = torch.ones(OLD_D)
                new_sd[p + nk + ".ln_old.bias"] = torch.zeros(OLD_D)
                new_sd[p + nk + ".gamma_new"] = torch.zeros(NEW_D - OLD_D)
                new_sd[p + nk + ".beta_new"] = torch.zeros(NEW_D - OLD_D)
            new_sd[p + "ffn.0.weight"] = torch.randn(NEW_VD, NEW_D, generator=g) * 0.02
            new_sd[p + "ffn.2.weight"] = torch.zeros(NEW_D, NEW_VD)   # zero residual

    new_sd["final_norm.ln_old.weight"] = old_sd["final_norm.weight"].clone()
    new_sd["final_norm.ln_old.bias"] = old_sd["final_norm.bias"].clone()
    new_sd["final_norm.gamma_new"] = torch.zeros(NEW_D - OLD_D)
    new_sd["final_norm.beta_new"] = torch.zeros(NEW_D - OLD_D)

    new.load_state_dict(new_sd, strict=True)
    new.tie_weights()
    n_params = sum(pp.numel() for pp in new.parameters())
    print(f"grown params = {n_params / 1e6:.2f}M")

    if args.verify:
        x = torch.randint(1, cfg.vocab_size, (2, 128))
        new.eval(); old.eval()
        with torch.inference_mode():
            o_out = old(x).logits
            n_out = new(x).logits
            d = (o_out - n_out).abs().max().item()
        print(f"FUNCTION PRESERVING CHECK: max|Δ logits| = {d:.3e} "
              f"({'OK' if d < 1e-4 else 'FAIL'})")
    new.save_pretrained(args.dst, safe_serialization=True)
    print(f"saved -> {args.dst}")


if __name__ == "__main__":
    main()
