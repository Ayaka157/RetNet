"""E12 的守卫：flat 分片里没有 eos 就必须拒绝打开。

分词脚本曾经默认 `--add-eos` 关闭，产出的流是"上一篇末行 + 下一篇开头"。
这个测试造两个目录：一个有 eos、一个没有，断言只有前者能被 `_FlatStage` 打开。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.data import _FlatStage  # noqa: E402

SEQ = 8


def _write_shard(path: Path, n_tokens: int, with_eos: bool) -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(3, 20000, size=n_tokens, dtype=np.int16)   # 避开 2
    if with_eos:
        arr[::32] = 2                                             # eos id = 2
    np.save(path, arr)


def test_missing_eos_raises(tmp_path: Path):
    _write_shard(tmp_path / "shard_00000.npy", (SEQ + 1) * 200, with_eos=False)
    with pytest.raises(ValueError):
        _FlatStage("noeos", tmp_path, SEQ, require_eos=True)


def test_with_eos_opens(tmp_path: Path):
    _write_shard(tmp_path / "shard_00000.npy", (SEQ + 1) * 200, with_eos=True)
    st = _FlatStage("haseos", tmp_path, SEQ, require_eos=True)
    assert len(st) == 200


def test_can_opt_out_explicitly(tmp_path: Path):
    _write_shard(tmp_path / "shard_00000.npy", (SEQ + 1) * 50, with_eos=False)
    st = _FlatStage("noeos", tmp_path, SEQ, require_eos=False)
    assert len(st) == 50
