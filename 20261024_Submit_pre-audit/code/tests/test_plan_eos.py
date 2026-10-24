"""E12 的端到端守卫：计划 JSON 里的 `require_eos: false` 必须真的到达数据集。

旧测试只直接构造 ``_FlatStage(require_eos=...)``，绕过了
``parse_plan -> Stage -> stage dict -> StagePackedDataset`` 这条真实链路。
于是 ``Stage`` 没有 ``require_eos`` 字段、``parse_plan`` 把它丢掉、
``_FlatStage`` 的报错文案教人写的键根本不生效 —— 三处缺陷一个都测不出来。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.data import StagePackedDataset  # noqa: E402
from retnet_hf.plan import parse_plan  # noqa: E402
from retnet_hf.train import plan_stage_specs  # noqa: E402

SEQ = 8


def _eos_free_dir(tmp_path: Path) -> Path:
    """一个**没有 eos** 的 flat 分片（模拟--add-eos 未开时产出的流）。"""
    d = tmp_path / "noeos"
    d.mkdir()
    rng = np.random.default_rng(0)
    arr = rng.integers(3, 20000, size=(SEQ + 1) * 40, dtype=np.int16)   # 避开 2
    np.save(d / "shard_00000.npy", arr)
    return d


def _plan_raw(d: Path, **stage):
    s = {"name": "noeos", "data_dir": str(d), "steps": 10, "mode": "flat"}
    s.update(stage)
    return {"stages": [s], "seq_len": SEQ, "lr": {"warmup": 1},
            "save_steps": 10, "eval_steps": 10}


def test_require_eos_defaults_true_through_parse_plan(tmp_path: Path):
    plan = parse_plan(_plan_raw(_eos_free_dir(tmp_path)), Path("<test>"))
    assert plan.stages[0].require_eos is True


def test_require_eos_false_survives_json_to_dataset(tmp_path: Path):
    """报错文案承诺的补救办法必须真的能用。"""
    d = _eos_free_dir(tmp_path)
    plan = parse_plan(_plan_raw(d, require_eos=False), Path("<test>"))
    assert plan.stages[0].require_eos is False
    ds = StagePackedDataset(plan_stage_specs(plan), SEQ)
    ready, why = ds.stage_ready(0)
    assert ready is True, f"写了 require_eos: false 仍然打不开：{why}"
    assert len(ds) == 40


def test_without_the_key_the_guard_still_refuses(tmp_path: Path):
    d = _eos_free_dir(tmp_path)
    plan = parse_plan(_plan_raw(d), Path("<test>"))
    ds = StagePackedDataset(plan_stage_specs(plan), SEQ)
    ready, why = ds.stage_ready(0)
    assert ready is False
    assert "require_eos: false" in why, "拒绝时必须告诉操作者正确的键名"


def test_flat_stage_weight_reaches_wcodes(tmp_path: Path):
    """P2-2：计划里的 weight 以前在 flat 分支被静默丢掉（wcode 恒为 1）。"""
    d = tmp_path / "flat"
    d.mkdir()
    rng = np.random.default_rng(1)
    arr = rng.integers(3, 20000, size=(SEQ + 1) * 10, dtype=np.int16)
    arr[::16] = 2
    np.save(d / "shard_00000.npy", arr)
    plan = parse_plan(_plan_raw(d, weight=3), Path("<test>"))
    ds = StagePackedDataset(plan_stage_specs(plan), SEQ)
    ds.set_stage(0)
    assert set(ds[0]["wcodes"].tolist()) == {3}, "flat 段的 weight 没有生效"


def test_hot_reload_does_not_drop_require_eos(tmp_path: Path):
    """计划热更新也必须带上 require_eos/weight，否则改文件等于没改。"""
    d = _eos_free_dir(tmp_path)
    ds = StagePackedDataset(plan_stage_specs(
        parse_plan(_plan_raw(d, require_eos=True), Path("<test>"))), SEQ)
    assert ds.stage_ready(0)[0] is False
    ds.update_stages([{**plan_stage_specs(parse_plan(
        _plan_raw(d, require_eos=False, weight=2), Path("<test>")))[0]}])
    assert ds.stage_ready(0)[0] is True
