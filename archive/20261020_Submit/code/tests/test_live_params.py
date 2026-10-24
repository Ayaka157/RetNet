"""E08 的守卫：参数量必须按"非零"统计，不能只数 config。

`numel()` 在 67% 参数精确为 0 时照样打印 504M。这里直接测
`live_param_ratio`：全零张量占比必须反映到比例上。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402
from retnet_hf.train import live_param_ratio  # noqa: E402


def _model() -> RetNetForCausalLM:
    cfg = RetNetHFConfig(vocab_size=128, d_model=32, num_layers=2, num_heads=2,
                         value_dim=64, ffn_dim=64, dropout=0.0, use_gla=True)
    return RetNetForCausalLM(cfg)


def test_fresh_model_all_live():
    live, total, ratio = live_param_ratio(_model())
    assert ratio > 0.99, f"新构造的模型不应有大片零参数，实际 live={live}/{total}"


def test_zeroed_part_is_detected():
    m = _model()
    with torch.no_grad():
        m.embedding.weight.zero_()          # 模拟"某个大张量整块是死的"
    live, total, ratio = live_param_ratio(m)
    assert live < total
    assert ratio < 0.99, "整块置零后 live 比例没有下降，统计逻辑有问题"


def test_ratio_is_exact_for_known_zero():
    """把一个已知张量的一半行置零，live 数必须精确减少对应的元素数。

    注意不能用"总数 - 被清零元素数"做基准：模型本身就有合法的零参数
    （GroupNorm 的 bias 初始为 0），所以要先测一次基线再比较差值。
    """
    m = _model()
    live0, total, _ = live_param_ratio(m)
    w = m.blocks[0].msr.q_proj.weight
    n = w.shape[0]
    with torch.no_grad():
        w[: n // 2].zero_()
    live1, total1, _ = live_param_ratio(m)
    assert total1 == total
    assert live1 == live0 - (n // 2) * w.shape[1]
