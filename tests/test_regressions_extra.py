"""零散但关键的回归测试：每个都对应 docs/00 里一个编号。

这些都设计成"改回旧代码就会红"：
- E05：GrowthLayerNorm 的新维度门控默认必须是 1，不能是 0；
- E19：同一尺寸的分片被替换时 refresh 必须能发现；
- E20：plan 模式必须把 eval_batch_mult 传给 train.py；
- E21：recurrent 的单步快路径必须与并行路径一致；
- E22：分块交叉熵必须与整段 fp32 的朴素实现数值一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.data import _FlatStage  # noqa: E402
from retnet_hf.modeling_retnet import (  # noqa: E402
    GrowthLayerNorm, RetNetForCausalLM, RetNetHFConfig, _chunked_weighted_ce,
)


# ---------------------------------------------------------------- E05
def test_growth_ln_new_gate_defaults_to_one():
    ln = GrowthLayerNorm(8, 16)
    assert torch.all(ln.gamma_new == 1.0), "gamma_new 默认不是 1：新维度会被乘法门关死"
    assert torch.all(ln.beta_new == 0.0)


# ---------------------------------------------------------------- E19
def test_refresh_detects_same_size_replacement(tmp_path: Path):
    seq = 8
    p = tmp_path / "shard_00000.npy"
    arr = np.random.default_rng(0).integers(3, 20000, size=(seq + 1) * 20, dtype=np.int16)
    arr[::32] = 2
    np.save(p, arr)
    st = _FlatStage("r", tmp_path, seq)

    # Windows 下 mmap 会锁住文件，先手动释放映射（不清空 shards 列表，
    # 这样 refresh 仍有旧签名可比），再原地替换成同尺寸的新内容。
    st.shards[0][1]._mmap.close()
    arr2 = arr.copy()
    arr2[0] = 7
    np.save(p, arr2)
    import os
    os.utime(p, None)

    assert st.refresh() is True, "同尺寸分片被替换时 refresh 没有发现（E19 回归）"
    assert st.refresh() is False


# ---------------------------------------------------------------- E20
def test_build_cmd_plan_passes_eval_batch_mult(tmp_path: Path):
    import run_task

    plan = tmp_path / "plan.json"
    plan.write_text('{"seq_len": 8, "stages": [{"name": "a", "data_dir": ".", '
                    '"steps": 10}]}', encoding="utf-8")
    cfg = {
        "plan": str(plan), "model_dir": "m", "output_dir": "o",
        "batch_size": 1, "mode": "chunkwise", "chunk_size": 8,
        "precision": "fp32", "seed": 1, "eval_batch_mult": 2,
    }
    cmd = run_task.build_cmd(cfg)
    assert "--eval_batch_mult" in cmd, "plan 模式没有传 eval_batch_mult（E20 回归）"
    idx = cmd.index("--eval_batch_mult")
    assert cmd[idx + 1] == "2"


# ---------------------------------------------------------------- E21/E16
def _model() -> RetNetForCausalLM:
    torch.manual_seed(0)
    cfg = RetNetHFConfig(vocab_size=128, d_model=32, num_layers=2, num_heads=2,
                         value_dim=64, ffn_dim=64, dropout=0.0, use_gla=True)
    m = RetNetForCausalLM(cfg)
    m.eval()
    return m


def test_recurrent_single_step_matches_manual():
    """E21：tgt_len=1 的快路径必须与手写的一步递推逐位一致。"""
    m = _model()
    msr = m.blocks[0].msr
    x = torch.randn(1, 1, 32)
    q, k, v, _g = msr._project_and_rotate(x, 0)
    key_dim = 32 // 2
    value_dim = 64 // 2
    state = torch.randn(1, 2, key_dim, value_dim)

    out, ns = msr._recurrent_retention(q, k, v, state)

    gamma = msr.gamma.to(dtype=q.dtype).view(1, -1, 1, 1)
    manual_ns = gamma * state + k[:, :, 0, :].unsqueeze(-1) * v[:, :, 0, :].unsqueeze(-2)
    manual_out = (q[:, :, 0, :].unsqueeze(-1) * manual_ns).sum(dim=-2).unsqueeze(2)

    assert torch.allclose(ns, manual_ns, atol=1e-5)
    assert torch.allclose(out, manual_out, atol=1e-5), "recurrent 单步快路径算错了（E21）"


# ---------------------------------------------------------------- E22
def test_chunked_ce_matches_naive_weighted():
    torch.manual_seed(0)
    n, v = 37, 64
    logits = torch.randn(2, n, v)
    labels = torch.randint(0, v, (2, n))
    wcodes = torch.randint(0, 4, (2, n))
    plw = 0.1

    # 朴素实现：整段 fp32 + 权重码
    w = torch.zeros_like(wcodes, dtype=torch.float32)
    w = torch.where(wcodes == 1, torch.ones_like(w), w)
    w = torch.where(wcodes == 2, torch.full_like(w, plw), w)
    w = torch.where(wcodes == 3, torch.ones_like(w), w)
    ce = F.cross_entropy(logits.float().reshape(-1, v), labels.reshape(-1), reduction="none")
    naive = (ce * w.reshape(-1)).sum() / w.reshape(-1).sum().clamp_min(1.0)

    got = _chunked_weighted_ce(logits, labels, wcodes, plw, chunk_tokens=7)
    assert torch.allclose(got, naive, atol=1e-5), f"{got.item()} != {naive.item()}"


def test_chunked_ce_without_weights_is_plain_mean():
    torch.manual_seed(1)
    logits = torch.randn(1, 11, 32)
    labels = torch.randint(0, 32, (1, 11))
    naive = F.cross_entropy(logits.float().reshape(-1, 32), labels.reshape(-1))
    got = _chunked_weighted_ce(logits, labels, None, 0.1, chunk_tokens=4)
    assert torch.allclose(got, naive, atol=1e-5)
