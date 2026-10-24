"""N01 的守卫：gamma 重初始化必须**按头**，不能按层。

旧实现把 20 个 `blocks.*.msr.gamma` 张量当成头，循环 h=0..19 生成一个标量再
`torch.full_like` 灌满每层 22 个头，结果同一层 22 个头完全相同。
这个测试直接加载脚本模块，断言 `build_gamma` 的输出逐头不同、半衰期落在给定区间，
并且旧式的“一个值刷满一层”会立刻被“取值个数必须等于头数”这条断言挡住。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "reinit_gamma.py"
    spec = importlib.util.spec_from_file_location("reinit_gamma_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_per_head_values_are_distinct():
    G = _load()
    g = G.build_gamma(22, "logspace", 128.0, 8192.0)
    assert g.shape == (22,)
    assert len(set(round(float(v), 10) for v in g)) == 22, "同一层内出现重复值：又退化成按层设置了"
    # 半衰期单调、落在区间内
    tau = G.half_lives(g)
    assert torch.isfinite(tau).all()
    assert float(tau.min()) > 127.0 and float(tau.max()) < 8193.0
    assert torch.all(tau[1:] > tau[:-1])


def test_orig_formula_is_per_head():
    G = _load()
    g = G.build_gamma(4, "orig")
    expected = 1.0 - 2.0 ** (-5.0 - torch.arange(4, dtype=torch.float64))
    assert torch.allclose(g, expected)
    assert len(set(round(float(v), 12) for v in g)) == 4


def test_bad_tau_range_rejected():
    G = _load()
    for lo, hi in ((0, 10), (100, 10), (-1, 10)):
        try:
            G.build_gamma(8, "logspace", lo, hi)
        except ValueError:
            continue
        raise AssertionError(f"非法 tau 范围 {lo}..{hi} 没有被拒绝")
