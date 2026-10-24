"""E07 的守卫：真的做一次梯度覆盖检查，而不是在注释里说做了。

旧实现只有一句注释（“另加一次梯度覆盖检查”），代码里没有任何统计；
`docs/01` 自己也写着“尚未落地”。这个测试证明三件事：

1. 反向时钩子真的收到每个参数的梯度（覆盖统计不是空的）；
2. 故意冻结一个参数 -> 它被识别为死参数；
3. 死参数占比超过阈值 -> ``on_step_end`` 抛 SystemExit（守卫会红）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402
from retnet_hf.train import (  # noqa: E402
    GradCoverageCallback, classify_dead_grad_params,
)


def _model() -> RetNetForCausalLM:
    torch.manual_seed(0)
    cfg = RetNetHFConfig(vocab_size=64, d_model=32, num_layers=2, num_heads=2,
                         value_dim=64, ffn_dim=64, dropout=0.0, use_gla=True)
    return RetNetForCausalLM(cfg)


def _step(model) -> None:
    x = torch.randint(1, 64, (2, 8))
    model(x, labels=x).loss.backward()


def _state(step: int = 1):
    return SimpleNamespace(global_step=step, is_world_process_zero=True)


def test_pure_classification():
    names = ["a", "b", "c"]
    assert classify_dead_grad_params(names, {"a": 1e-3, "b": 0.0}) == ["b", "c"]


def test_all_params_get_gradients_on_a_healthy_model():
    m = _model()
    cb = GradCoverageCallback(m, tol=1.0)
    _step(m)
    dead = cb.check_now()
    assert dead == [], f"健康模型不该有死参数：{dead}"


def test_frozen_parameter_is_detected_and_turns_red():
    m = _model()
    frozen = "blocks.1.msr.q_proj.weight"
    dict(m.named_parameters())[frozen].requires_grad_(False)
    cb = GradCoverageCallback(m, tol=0.0)
    _step(m)
    dead = cb.check_now()
    assert frozen in dead, f"冻结的参数没有被识别：dead={dead}"
    # 阈值 0 -> 任何一个死参数都必须让训练停下来
    cb2 = GradCoverageCallback(m, tol=0.0)
    _step(m)
    with pytest.raises(SystemExit) as exc:
        cb2.on_step_end(None, _state(), None)
    assert "死参数" in str(exc.value)


def test_hooks_are_removed_after_the_check():
    m = _model()
    cb = GradCoverageCallback(m, tol=1.0)
    _step(m)
    cb.check_now()
    assert cb._handles == [], "检查后必须摘掉钩子，否则每步都白付一次回调开销"
    # 再跑一步不报错（钩子已摘掉）
    _step(m)


def test_waits_for_the_configured_step():
    m = _model()
    cb = GradCoverageCallback(m, tol=0.0, check_step=3)
    with pytest.raises(SystemExit):
        # 先把一个参数弄死，再在 step 3 触发
        dict(m.named_parameters())["blocks.1.msr.q_proj.weight"].requires_grad_(False)
        _step(m)
        cb.on_step_end(None, _state(1), None)      # 未到 step 3：不检查
        cb.on_step_end(None, _state(3), None)      # 到了：检查并停机
