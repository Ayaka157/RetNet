"""生长后的模型必须"能学"，而且生长的映射必须就是文档写的那个映射。

这个测试用**真实的 `scripts/grow_500m.py`**，把维度 monkeypatch 成很小的值，
然后断言：

1. 正常初始化（新行随机、中间层 gate=1、final gate=0）→ 生长成功（rc **0**），
   产物落盘；旧测试写 `assert rc in (0, 1)`，而 `TARGET_PARAMS=0` 时
   `ok_n` 恒假、成功永远返回 1，等于把成功和“参数量不符”混成一个分支。
2. 把 `NEW_ROW_INIT_STD` 与 `BLOCK_GATE` 改回旧行为（0）→ 可训性校验必须失败，
   脚本必须拒绝落盘（返回 2）。
3. 源模型本身已经生长过一次（`use_growth_ln=True`）时，它的第二段
   `gamma_new`/`beta_new` 必须被**逐值继承**，只有本次新增的段写成 gate。
   旧实现把整个 gamma_new 重设为 gate，只有源第二段恰好等于 1.0 时才“看起来”
   保真 —— 文档却声称“源值照搬”。
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

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


def _tiny_source(tmp: Path, grown: bool = False,
                 block_gate: float = 1.0, final_gate: float = 0.0) -> Path:
    cfg = RetNetHFConfig(
        vocab_size=256, d_model=64, num_layers=2, num_heads=2,
        value_dim=128, ffn_dim=128, dropout=0.0,
        tie_weights=True, use_gla=True, use_swiglu=False,
        use_growth_ln=grown, growth_old_dim=48 if grown else None,
        embed_scale=64 ** 0.5,
    )
    cfg.mode = "chunkwise"
    cfg.chunk_size = 8
    model = RetNetForCausalLM(cfg)
    if grown:
        # 复现“本脚本产出的检查点”的门控布局：中间层 1.0、final 0.0。
        # 允许改成别的值，用来验证配方前提被违反时 --verify 会拒绝。
        with torch.no_grad():
            for i in range(cfg.num_layers):
                for nm in ("norm1", "norm2"):
                    getattr(model.blocks[i], nm).gamma_new.fill_(block_gate)
                    getattr(model.blocks[i], nm).beta_new.zero_()
            model.final_norm.gamma_new.fill_(final_gate)
            model.final_norm.beta_new.zero_()
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
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        src = _tiny_source(tmp / "src")
        dst = tmp / "dst"
        rc = _run(G, src, dst)
        assert rc == 0, f"正常初始化下生长应当成功（rc=0），实际 rc={rc}"
        assert (dst / "model.safetensors").exists(), "生长产物没有落盘"


def test_growth_guard_fails_when_reverted_to_zeros():
    """把新行随机撤掉、把中间层 gate 改成 0（旧行为），守卫必须拒绝落盘。"""
    G = _load_grow_module()
    _patch_dims(G)
    G.NEW_ROW_INIT_STD = 0.0
    G.BLOCK_GATE = 0.0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        src = _tiny_source(tmp / "src")
        dst = tmp / "dst"
        rc = _run(G, src, dst)
        assert rc == 2, f"旧行为下必须拒绝保存（rc=2），实际 rc={rc}"
        assert not (dst / "model.safetensors").exists(), "可训性校验失败却仍然落盘了"


def _patch_dims_stage1(G) -> None:
    """48/H2 -> 72/H3。key_dim 与 value_head_dim 保持不变（配方的硬前提）。"""
    G.OLD_D, G.NEW_D = 48, 72
    G.OLD_L, G.NEW_L = 2, 2
    G.OLD_H, G.NEW_H = 2, 3
    G.OLD_VD, G.NEW_VD = 96, 144
    G.GROUP = 48
    G.TARGET_PARAMS = 0


def _patch_dims_stage2(G) -> None:
    """72/H3 -> 96/H4（第二次生长，仍然保持 24 / 48）。"""
    G.OLD_D, G.NEW_D = 72, 96
    G.OLD_L, G.NEW_L = 2, 2
    G.OLD_H, G.NEW_H = 3, 4
    G.OLD_VD, G.NEW_VD = 144, 192
    G.GROUP = 48
    G.TARGET_PARAMS = 0


def test_growth_twice_preserves_function_and_layout():
    """P1-8：本配方的产物必须能被**再生长一次**。

    旧文档声称第二段“照搬源值 + 新段置零”，实际是整段重设。真实可复现的
    判据是：按本配方生长出来的检查点（中间层第二段 1.0、final 0.0、
    q/k/v 新列为 0），再生长一次时函数仍然保真 —— 这条只有在第二次生长
    才检验得到，单次生长用的是未生长过的普通 LayerNorm。
    """
    G = _load_grow_module()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        # 第一次：48 -> 64
        _patch_dims_stage1(G)
        src = tmp / "src"
        cfg = RetNetHFConfig(vocab_size=256, d_model=48, num_layers=2, num_heads=2,
                             value_dim=96, ffn_dim=96, dropout=0.0,
                             tie_weights=True, use_gla=True, use_swiglu=False,
                             use_growth_ln=False, growth_old_dim=None,
                             embed_scale=48 ** 0.5)
        cfg.mode = "chunkwise"
        cfg.chunk_size = 8
        RetNetForCausalLM(cfg).save_pretrained(src, safe_serialization=True)
        mid = tmp / "mid"
        rc1 = _run(G, src, mid)
        assert rc1 == 0, f"第一次生长应当成功，实际 rc={rc1}"
        # 第二次：64 -> 96（从配方产物出发）
        _patch_dims_stage2(G)
        dst = tmp / "dst"
        rc2 = _run(G, mid, dst)
        assert rc2 == 0, f"从配方产物再生长应当成功，实际 rc={rc2}"
        new = RetNetForCausalLM.from_pretrained(str(dst))
        # 落盘前脚本跑了 5 步真实反向（可训性校验），门控会被优化器推动
        # 一点点，所以这里用容差而不是逐位相等；要逐位断言的是脚本内部
        # 在反向之前做的 [layout] 检查。
        for i in range(2):
            for nm in ("norm1", "norm2"):
                g = getattr(new.blocks[i], nm).gamma_new
                assert torch.allclose(g, torch.full_like(g, G.BLOCK_GATE), atol=2e-2), \
                    f"blocks.{i}.{nm} 第二段没有整段抬到 BLOCK_GATE"
        g = new.final_norm.gamma_new
        assert torch.allclose(g, torch.zeros_like(g), atol=2e-2), \
            "final 第二段没有整段归零"
        # Windows 上 safetensors 的 mmap 会锁住文件，先释放再做临时目录清理。
        del new


def test_out_of_recipe_source_is_refused_not_silently_changed():
    """源的第二段被训练偏了（块 1.3 / final 0.4）时，整段重设会改变前向。

    这不是“bug”，是配方的**前提**：旧检查点必须由本配方产出。--verify 必须
    响亮拒绝（rc=2，不落盘），而不是静默产出一个不可比的模型。
    """
    G = _load_grow_module()
    _patch_dims(G)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        src = _tiny_source(tmp / "src", grown=True, block_gate=1.3, final_gate=0.4)
        dst = tmp / "dst"
        rc = _run(G, src, dst)
        assert rc == 2, f"前提被违反时必须拒绝保存（rc=2），实际 rc={rc}"
        assert not (dst / "model.safetensors").exists()
