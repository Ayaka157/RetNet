"""E08 的守卫：活参数比例的判定必须能区分“坏初始化”和“已知剪枝”。

旧实现只有一条 `ratio < 0.95 -> SystemExit`，于是 docs/02 的 rescue 计划
（从 33% 活参数的 `retnet_500m_v21_export` 起步）第一条命令就中止。
这里把判定抽成纯函数 `check_live_ratio`，逐条测边界；再跑真实的 `_preflight`
证明两种情况的行为不同。
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
    _preflight, check_live_ratio, live_param_ratio, read_live_baseline,
    write_live_baseline,
)


def _model(use_gla: bool = True) -> RetNetForCausalLM:
    cfg = RetNetHFConfig(vocab_size=128, d_model=32, num_layers=2, num_heads=2,
                         value_dim=64, ffn_dim=64, dropout=0.0, use_gla=use_gla)
    return RetNetForCausalLM(cfg)


def _args(**kw):
    base = dict(min_live_ratio=0.95, allow_dead_params=False)
    base.update(kw)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------- 纯函数边界
def test_healthy_model_passes():
    ok, why = check_live_ratio(0.999)
    assert ok and ">=" in why


def test_pruned_checkpoint_is_refused_without_acknowledgement():
    ok, why = check_live_ratio(0.33)
    assert not ok
    assert "--allow_dead_params" in why, "拒绝时必须告诉操作者怎么继续"


def test_pruned_checkpoint_passes_with_explicit_flag():
    ok, why = check_live_ratio(0.33, allow_dead=True)
    assert ok and "allow_dead_params" in why


def test_recorded_baseline_exempts_only_without_regression():
    # 与历史基线一致：放行（续训自己导出的检查点）
    assert check_live_ratio(0.34, baseline=0.33)[0]
    # 比基线更死：拒绝（说明这一轮又把东西弄死了）
    ok, why = check_live_ratio(0.20, baseline=0.33)
    assert not ok
    assert "基线" in why


def test_custom_threshold_is_honoured():
    assert check_live_ratio(0.80, threshold=0.75)[0]
    assert not check_live_ratio(0.70, threshold=0.75)[0]


# ---------------------------------------------------------------- 基线读写
def test_live_baseline_roundtrip(tmp_path: Path):
    assert read_live_baseline(tmp_path) is None
    p = write_live_baseline(tmp_path, 330, 1000, 0.33)
    assert p.name == "live_params.json"
    assert read_live_baseline(tmp_path) == pytest.approx(0.33)
    # 坏文件不能把训练带崩
    p.write_text("{not json", encoding="utf-8")
    assert read_live_baseline(tmp_path) is None


def test_baseline_tracks_the_historical_minimum(tmp_path: Path):
    """缓慢侵蚀不能被“每轮重设基线”抹掉（P0-2 的同类漏洞）。"""
    write_live_baseline(tmp_path, 330, 1000, 0.33)
    # 下一轮掉到 0.325：单轮低于 2% 容差，但历史最低必须记住 0.325
    write_live_baseline(tmp_path, 325, 1000, 0.325, previous=read_live_baseline(tmp_path))
    assert read_live_baseline(tmp_path) == pytest.approx(0.325)
    # 再下一轮回到 0.33：ratio 记 0.33，但基线仍是 0.325
    write_live_baseline(tmp_path, 330, 1000, 0.33, previous=read_live_baseline(tmp_path))
    assert read_live_baseline(tmp_path) == pytest.approx(0.325)


# ---------------------------------------------------------------- _preflight
def test_preflight_refuses_broken_fresh_model():
    m = _model()
    with torch.no_grad():
        m.embedding.weight.zero_()
    with pytest.raises(SystemExit) as exc:
        _preflight(m, _args())
    assert "有效参数" in str(exc.value) or "活参数比例" in str(exc.value)


def test_preflight_allows_documented_pruned_start():
    """docs/02 §1 的起点：33% 活参数 + --allow_dead_params 必须能开训。"""
    m = _model()
    with torch.no_grad():
        m.embedding.weight.zero_()
    _preflight(m, _args(allow_dead_params=True))       # 不抛异常即通过


def test_preflight_uses_recorded_baseline():
    m = _model()
    with torch.no_grad():
        m.embedding.weight.zero_()
    _live, _total, ratio = live_param_ratio(m)
    _preflight(m, _args(), baseline=ratio)            # 与基线一致 -> 放行


def test_preflight_checks_gamma_even_when_it_is_a_buffer():
    """`use_gla=False` 时 gamma 是 buffer。旧实现只查 named_parameters()，
    这种模型上 gamma<=0 的检查完全不存在（P1-3）。"""
    m = _model(use_gla=False)
    with torch.no_grad():
        m.blocks[0].msr.gamma.zero_()
    with pytest.raises(SystemExit) as exc:
        _preflight(m, _args())
    assert "gamma" in str(exc.value)
