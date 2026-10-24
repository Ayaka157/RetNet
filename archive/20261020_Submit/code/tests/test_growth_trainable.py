"""生长后的模型必须"能学"：每一步都检查，而不是只看输出有没有变。

这个测试用**真实的 `scripts/grow_500m.py`**，把维度 monkeypatch 成很小的值，
然后断言两件事：

1. 正常初始化（新行随机、中间层 gate=1、final gate=0）→ 生长成功，
   并且 5 步反向之后每个参数都有非零梯度；
2. 把 `NEW_ROW_INIT_STD` 与 `BLOCK_GATE` 改回旧行为（0）→ 可训性校验必须失败，
   脚本必须拒绝落盘（返回 2）。

第 2 条是"守卫必须能被证明会失败"：如果哪天有人把 fill/grow_norm 改回全零，
这个测试会先变红，而不是等到训练烧完预算才发现 67% 参数是死的（docs/00 E04）。
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402


def _load_grow_module():
    path = ROOT / "scripts" / "grow_500m.py"
    spec = importlib.util.spec_from_file_location("grow_500m_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 只执行模块顶层，不会跑 main()
    return mod


def _tiny_source(tmp: Path) -> Path:
    cfg = RetNetHFConfig(
        vocab_size=256, d_model=64, num_layers=2, num_heads=2,
        value_dim=128, ffn_dim=128, dropout=0.0,
        tie_weights=True, use_gla=True, use_swiglu=False,
        use_growth_ln=False, growth_old_dim=None,
        embed_scale=64 ** 0.5,
    )
    cfg.mode = "chunkwise"
    cfg.chunk_size = 8
    model = RetNetForCausalLM(cfg)
    model.save_pretrained(tmp, safe_serialization=True)
    return tmp


def _patch_dims(G) -> None:
    """把 1024->1408 的配方缩成 64->96，其余比例保持一致。"""
    G.OLD_D, G.NEW_D = 64, 96
    G.OLD_L, G.NEW_L = 2, 2
    G.OLD_H, G.NEW_H = 2, 3
    G.OLD_VD, G.NEW_VD = 128, 192
    G.GROUP = 64
    G.TARGET_PARAMS = 0            # 只跳过参数量断言，不影响可训性校验


def _run(G, src: Path, dst: Path) -> int:
    argv = ["grow_500m.py", "--src", str(src), "--dst", str(dst),
            "--verify", "--seq", "16", "--tol", "1e-2"]
    old = sys.argv
    sys.argv = argv
    try:
        return G.main()
    finally:
        sys.argv = old


def test_growth_is_trainable():
    G = _load_grow_module()
    _patch_dims(G)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = _tiny_source(tmp / "src")
        dst = tmp / "dst"
        rc = _run(G, src, dst)
        assert rc in (0, 1), f"正常初始化下生长应当成功，实际 rc={rc}"
        assert (dst / "model.safetensors").exists(), "生长产物没有落盘"


def test_growth_guard_fails_when_reverted_to_zeros():
    """把新行随机撤掉、把中间层 gate 改成 0（旧行为），守卫必须拒绝落盘。"""
    G = _load_grow_module()
    _patch_dims(G)
    G.NEW_ROW_INIT_STD = 0.0
    G.BLOCK_GATE = 0.0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = _tiny_source(tmp / "src")
        dst = tmp / "dst"
        rc = _run(G, src, dst)
        assert rc == 2, f"旧行为下必须拒绝保存（rc=2），实际 rc={rc}"
        assert not (dst / "model.safetensors").exists(), "可训性校验失败却仍然落盘了"
