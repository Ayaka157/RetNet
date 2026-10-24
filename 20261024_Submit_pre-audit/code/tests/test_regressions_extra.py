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
    import time
    # 显式把 mtime 推到 10 秒后：NTFS 的 mtime 粒度与写入速度可能让
    # "刚写完"和"刚创建"落在同一纳秒刻度上，那样签名会意外相同。
    future = int(time.time()) + 10
    os.utime(p, (future, future))

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


def test_recurrent_single_step_takes_fast_path(monkeypatch):
    """E21 是**性能**缺陷：只比数值的话，删掉快路径照样全绿。

    这里证明 tgt_len==1 时真的没走 `range(tgt_len)` 循环（循环版必须
    `torch.stack` 物化输出列表，快路径不需要）。
    """
    m = _model()
    msr = m.blocks[0].msr
    q, k, v, _g = msr._project_and_rotate(torch.randn(1, 1, 32), 0)

    def _boom(*a, **kw):
        raise AssertionError("tgt_len==1 仍然走了逐 token 循环（E21 回归）")

    monkeypatch.setattr(torch, "stack", _boom)
    out, ns = msr._recurrent_retention(q, k, v, None)
    assert out.shape == (1, 2, 1, 32) and ns.shape == (1, 2, 16, 32)


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


def test_ce_chunk_size_is_derived_from_a_byte_budget():
    """E22：chunk 必须由字节预算反推，不能是固定 token 数。

    旧默认 `chunk_tokens=262144` 与词表大小无关，且在 B=16/L=2048（n=32768）
    时只切一块 —— 2.6 GB 的整段 fp32 拷贝原封不动。
    """
    from retnet_hf.modeling_retnet import DEFAULT_CE_CHUNK_BYTES, ce_chunk_tokens
    # V=20000（生产词表）：64 MB / (20000*4) = 838 token/块
    assert ce_chunk_tokens(20000, DEFAULT_CE_CHUNK_BYTES) == 838
    assert ce_chunk_tokens(20000, DEFAULT_CE_CHUNK_BYTES) * 4 * 20000 <= DEFAULT_CE_CHUNK_BYTES
    # V 越大，块越小
    assert ce_chunk_tokens(40000) < ce_chunk_tokens(20000)
    # 极端词表也不能出现 0
    assert ce_chunk_tokens(10 ** 9) == 1


def test_chunked_ce_splits_at_production_token_count(monkeypatch):
    """生产形状 B=16 × L=2048（n=32768）必须切出多块，且每块不超过预算。

    用 V=1024 保持内存可控（n 与生产一致，预算 64 MB -> 16384 token/块），
    同时记录 `F.cross_entropy` 实际收到的切片形状。
    """
    import retnet_hf.modeling_retnet as MR

    torch.manual_seed(0)
    B, L, V = 16, 2048, 1024
    n = B * L
    logits = torch.randn(B, L, V, dtype=torch.float32)
    labels = torch.randint(0, V, (B, L))
    wcodes = torch.ones(B, L, dtype=torch.long)

    seen = []
    real_ce = MR.F.cross_entropy

    def _spy(inp, target, *a, **kw):
        seen.append((tuple(inp.shape), str(inp.dtype)))
        return real_ce(inp, target, *a, **kw)

    monkeypatch.setattr(MR.F, "cross_entropy", _spy)
    loss = MR._chunked_weighted_ce(logits, labels, wcodes, 0.1)
    assert torch.isfinite(loss)
    assert len(seen) >= 2, f"生产形状只切了 {len(seen)} 块 —— 分块参数被无视了"
    assert all(s[0][0] <= 16384 for s in seen), f"有块超过字节预算：{seen}"
    assert sum(s[0][0] for s in seen) == n, "分块必须覆盖全部 token"
    assert all(s[1] == "torch.float32" for s in seen), "块内必须先转 fp32 再算 CE"
