"""E16 的守卫：parallel 与 chunkwise 必须是同一个函数。

历史上 parallel 路径多做两次归一化（mask/√Σmask 与 scores/Σ|scores|），
而训练走 chunkwise、评测脚本又强制 parallel（eval_holdout.py:62），
于是"训练的函数"和"报告的数字"不是同一个东西。
删掉那两处归一化后，两条路径只该差浮点噪声。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402


def _model() -> RetNetForCausalLM:
    torch.manual_seed(0)
    cfg = RetNetHFConfig(vocab_size=128, d_model=32, num_layers=2, num_heads=2,
                         value_dim=64, ffn_dim=64, dropout=0.0, use_gla=True)
    m = RetNetForCausalLM(cfg)
    m.eval()
    return m


def test_parallel_matches_chunkwise():
    m = _model()
    ids = torch.randint(1, 128, (2, 16))
    with torch.no_grad():
        lp = m(input_ids=ids, mode="parallel").logits
        lc = m(input_ids=ids, mode="chunkwise", chunk_size=8).logits
    diff = (lp - lc).abs().max().item()
    scale = lp.abs().max().item()
    assert diff <= 0.02 * max(scale, 1.0), f"max|Δlogit|={diff:.4e}，scale={scale:.3e}"


def test_parallel_matches_recurrent():
    """prefill 走 parallel、decode 走 recurrent，这条也必须一致。"""
    m = _model()
    ids = torch.randint(1, 128, (1, 8))
    with torch.no_grad():
        lp = m(input_ids=ids, mode="parallel").logits
        lr, _ = m._forward_hidden(ids, mode="recurrent", return_state=True)
    diff = (lp - lr).abs().max().item()
    scale = lp.abs().max().item()
    assert diff <= 0.05 * max(scale, 1.0), f"max|Δlogit|={diff:.4e}，scale={scale:.3e}"
