"""E06 的守卫：开启 no_decay_1d 后，1 维参数（尤其 msr.gamma）不得进入 decay 集合。

如果哪天有人把默认值改回 False、或把 `get_decay_parameter_names` 的过滤删掉，
这个测试会先红，而不是等 gamma 被 weight decay 压到半衰期 1.22 token 才发现。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from transformers import TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402
from retnet_hf.train import CompileSafeTrainer  # noqa: E402


def _tiny_model():
    cfg = RetNetHFConfig(vocab_size=128, d_model=32, num_layers=2, num_heads=2,
                         value_dim=64, ffn_dim=64, dropout=0.0, use_gla=True)
    return RetNetForCausalLM(cfg)


def _trainer(no_decay: bool, out: str):
    t = CompileSafeTrainer.__new__(CompileSafeTrainer)   # 不跑 __init__，只借方法
    t.args = TrainingArguments(output_dir=out, report_to=[])
    t._no_decay_1d = no_decay
    return t


def test_all_1d_params_excluded():
    m = _tiny_model()
    with tempfile.TemporaryDirectory() as td:
        names = set(_trainer(True, td).get_decay_parameter_names(m))
    one_d = {n for n, p in m.named_parameters() if p.ndim <= 1}
    assert one_d, "测试模型应当有 1 维参数"
    assert not (one_d & names), f"1 维参数仍在 decay 集合里: {sorted(one_d & names)}"
    assert not any(n.endswith("msr.gamma") for n in names)


def test_gamma_in_decay_set_when_disabled():
    """反向对照：关掉开关时必须能看到 gamma 在 decay 集合里，否则测试没在测东西。"""
    m = _tiny_model()
    with tempfile.TemporaryDirectory() as td:
        names = set(_trainer(False, td).get_decay_parameter_names(m))
    assert any(n.endswith("msr.gamma") for n in names), "关闭开关后 gamma 仍未进入 decay 集合"
