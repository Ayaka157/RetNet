"""E14 的守卫：验证集目录不得与任何训练阶段同目录。

上一轮 val_dir=chat_pack_v13，而 sft_mix 里 40% 就是它；训练期 eval_loss
因此在末段被"训练集成绩"污染，best-model 选择也跟着错。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.train import assert_val_disjoint  # noqa: E402


def test_same_dir_raises(tmp_path: Path):
    d = tmp_path / "chat_pack_v13"
    d.mkdir()
    with pytest.raises(SystemExit):
        assert_val_disjoint(d, [d])


def test_disjoint_passes(tmp_path: Path):
    val = tmp_path / "val"
    train = tmp_path / "train"
    val.mkdir()
    train.mkdir()
    assert_val_disjoint(val, [train])        # 不抛异常即通过


def test_relative_and_absolute_are_compared_after_resolve(tmp_path: Path):
    d = tmp_path / "stage"
    d.mkdir()
    rel = Path(str(d))                       # 同一个路径的另一种写法
    with pytest.raises(SystemExit):
        assert_val_disjoint(d, [rel])
