"""P1-6：docs/02 的两个门依赖 `--by-position` / `--ctx-ablation`。

旧 `eval_holdout.py` 只有 `--chunk-size`：计划里写 `--by-position --ctx-ablation`
的命令在 argparse 阶段就失败，G1c 的 Δ_ctx 门没有任何脚本能算。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402

import eval_holdout  # noqa: E402

SEQ = 16
CHUNK = 8
ROWS = 3


def _model_dir(tmp_path: Path) -> Path:
    cfg = RetNetHFConfig(vocab_size=64, d_model=16, num_layers=1, num_heads=2,
                         value_dim=32, ffn_dim=32, dropout=0.0, use_gla=False)
    m = RetNetForCausalLM(cfg)
    d = tmp_path / "model"
    m.save_pretrained(d, safe_serialization=True)
    return d


def _val_dir(tmp_path: Path) -> Path:
    d = tmp_path / "val"
    d.mkdir()
    rng = np.random.default_rng(0)
    np.save(d / "val_a.npy", rng.integers(1, 64, (ROWS * (SEQ + 1),), dtype=np.int16))
    return d


def _run(tmp_path: Path, *flags: str) -> dict:
    out = tmp_path / "eval.jsonl"
    rc = eval_holdout.main([
        "--model", str(_model_dir(tmp_path)),
        "--val-glob", str(_val_dir(tmp_path) / "*.npy"),
        "--seq-len", str(SEQ), "--chunk-size", str(CHUNK), "--batch", "1",
        "--device", "cpu", "--tag", "t", "--out", str(out), *flags,
    ])
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])


def test_plain_eval_still_works(tmp_path: Path):
    rec = _run(tmp_path)
    assert rec["tokens"] == ROWS * SEQ and rec["loss"] > 0
    assert "by_position" not in rec and "ctx_ablation" not in rec


def test_by_position_and_ctx_ablation(tmp_path: Path):
    rec = _run(tmp_path, "--by-position", "--ctx-ablation", "--ctx", str(CHUNK))
    buckets = rec["by_position"]
    assert len(buckets) == 8, buckets
    assert [b["range"] for b in buckets] == \
        [[i * 2, i * 2 + 1] for i in range(8)], buckets
    assert all(b["tokens"] == ROWS * 2 for b in buckets)
    ctx = rec["ctx_ablation"]
    assert ctx["ctx"] == CHUNK and ctx["seq_len"] == SEQ
    assert ctx["tokens"] == ROWS * CHUNK, "Δ_ctx 只统计最后 ctx 个位置"
    assert abs(ctx["delta_ctx"] - (ctx["loss_short"] - ctx["loss_full_tail"])) < 1e-9


def test_misaligned_ctx_is_refused_up_front(tmp_path: Path):
    """短窗口也走 chunkwise 内核，ctx 不是 chunk 的倍数必须提前报错。"""
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, "--ctx-ablation", "--ctx", str(CHUNK + 1))
    assert "整数倍" in str(exc.value)
