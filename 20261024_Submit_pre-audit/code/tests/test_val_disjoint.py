"""E14 的守卫：验证集不能和训练阶段**同源**（路径 + 声明来源 + 内容指纹）。

旧测试的第三个用例 `rel = Path(str(d))` 和 `d` 是同一个绝对路径，注释却写成
“相对路径的另一种写法”——它证明不了任何东西。真实事故也不是路径相等：
`val_dir=chat_pack_v13`，而 sft_mix 里 40% 就是 chat_pack_v13，
两个目录名不同、内容重叠。

这里逐层测：
1. 真的不同写法（os.path.relpath / 尾斜杠 / 大小写）都要被认出来；
2. meta.json 声明同一个来源 -> 拒绝；
3. 两个目录里存在完全相同的行 -> 拒绝（不看文件名）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.provenance import dir_row_hashes, norm_key, overlap_report  # noqa: E402
from retnet_hf.train import assert_val_disjoint  # noqa: E402

SEQ = 8


def _packed(d: Path, train_rows: int = 0, val_rows: int = 2, seed: int = 0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if train_rows:
        np.save(d / "train.ids.npy",
                rng.integers(1, 20000, (train_rows, SEQ + 1), dtype=np.int16))
        np.save(d / "train.wcode.npy",
                np.ones((train_rows, SEQ + 1), dtype=np.uint8))
    if val_rows:
        np.save(d / "val.ids.npy",
                rng.integers(1, 20000, (val_rows, SEQ + 1), dtype=np.int16))
        np.save(d / "val.wcode.npy", np.ones((val_rows, SEQ + 1), dtype=np.uint8))
    return d


def test_same_dir_raises(tmp_path: Path):
    d = _packed(tmp_path / "chat_pack_v13")
    with pytest.raises(SystemExit):
        assert_val_disjoint(d, [d], seq_len=SEQ)


def test_genuinely_disjoint_passes(tmp_path: Path):
    val = _packed(tmp_path / "val", seed=1)
    train = _packed(tmp_path / "train", train_rows=5, val_rows=0, seed=2)
    assert_val_disjoint(val, [train], seq_len=SEQ)      # 不抛异常即通过


def test_relative_spelling_is_caught(tmp_path: Path, monkeypatch):
    """真正的“另一种写法”：相对路径。"""
    d = _packed(tmp_path / "stage")
    monkeypatch.chdir(tmp_path)
    rel = os.path.relpath(d, tmp_path)
    assert rel != str(d)
    with pytest.raises(SystemExit):
        assert_val_disjoint(d, [rel], seq_len=SEQ)


def test_trailing_separator_is_caught(tmp_path: Path):
    d = _packed(tmp_path / "stage")
    with pytest.raises(SystemExit):
        assert_val_disjoint(d, [Path(str(d) + os.sep)], seq_len=SEQ)


@pytest.mark.skipif(os.name != "nt", reason="大小写折叠只在 Windows 文件系统上有意义")
def test_case_variation_is_caught(tmp_path: Path):
    d = _packed(tmp_path / "Stage")
    upper = Path(str(d).upper())
    assert str(upper) != str(d)
    assert norm_key(upper) == norm_key(d)
    with pytest.raises(SystemExit):
        assert_val_disjoint(d, [upper], seq_len=SEQ)


def test_declared_source_overlap_is_caught(tmp_path: Path):
    """名字不同、来源相同 —— 上一轮真实事故的形态。"""
    val = _packed(tmp_path / "val_pack", seed=1)
    train = _packed(tmp_path / "hf_pack_sft", train_rows=5, val_rows=0, seed=2)
    (val / "meta.json").write_text(json.dumps(
        {"source_ids": ["chat_pack_v13"]}), encoding="utf-8")
    (train / "meta.json").write_text(json.dumps(
        {"source_ids": ["chat_pack_v13"]}), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        assert_val_disjoint(val, [train], seq_len=SEQ)
    assert "共同来源" in str(exc.value)


def test_legacy_meta_sources_are_understood(tmp_path: Path):
    """旧产线只有 sources+text_fields，也要能判同源（否则等于没有声明）。"""
    from retnet_hf.provenance import declared_sources

    val = _packed(tmp_path / "val", seed=6)
    train = _packed(tmp_path / "train", train_rows=3, val_rows=0, seed=7)
    for d in (val, train):
        (d / "meta.json").write_text(json.dumps(
            {"sources": ["data_cache/chat_pack_v13"], "text_fields": ["prompt", "reply"]}),
            encoding="utf-8")
    assert declared_sources(val) == {"data_cache/chat_pack_v13#prompt|reply"}
    with pytest.raises(SystemExit):
        assert_val_disjoint(val, [train], seq_len=SEQ)


def test_content_overlap_is_caught_without_any_meta(tmp_path: Path):
    """没有任何 meta：只能靠内容指纹。训练侧 rows 里混进了验证集的两行。"""
    val = _packed(tmp_path / "val", seed=3)
    train = _packed(tmp_path / "train", train_rows=6, val_rows=0, seed=4)
    rows = np.load(train / "train.ids.npy")
    rows[:2] = np.load(val / "val.ids.npy")
    np.save(train / "train.ids.npy", rows)
    rep = overlap_report(val, [train], SEQ)
    assert rep["content_hits"], rep
    with pytest.raises(SystemExit) as exc:
        assert_val_disjoint(val, [train], seq_len=SEQ)
    assert "完全相同" in str(exc.value)


def test_val_side_hashes_read_val_ids(tmp_path: Path):
    """验证集目录里读的必须是 val.ids.npy。

    旧实现一律跳过文件名含 ``val`` 的分片，于是验证集侧哈希为空，
    内容层永远比较两个空集合 —— 看起来通过，其实没比。
    """
    val = _packed(tmp_path / "val", seed=5)
    h = dir_row_hashes(val, SEQ, prefer_val=True)
    assert h["rows"] == 2 and h["files"] == ["val.ids.npy"]
