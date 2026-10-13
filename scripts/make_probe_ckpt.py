"""生成 0.5B / 0.6B 的探针检查点（随机权重），只为测显存与步时。

    python scripts/make_probe_ckpt.py

随机权重对显存占用与步时无影响——计算图与张量形状完全一致。
权重用 fp32 保存，与现有 272M 检查点一致（否则 from_pretrained 的 dtype 会不同，
显存对照就不成立）。
"""
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402

# 沿用项目既有比例：key_dim = d/heads = 64，value_dim = ffn_dim = 2*d
CANDIDATES = [
    # name,        layers, d_model, heads
    ("probe_500m", 20, 1408, 22),
    ("probe_600m", 24, 1408, 22),
]


def build(name, L, d, heads, growth_old_dim=1024):
    out = ROOT / "checkpoints" / name
    if (out / "model.safetensors").exists():
        print(f"[{name}] 已存在，跳过")
        return
    cfg = RetNetHFConfig(
        vocab_size=20000, d_model=d, num_layers=L, num_heads=heads,
        value_dim=2 * d, ffn_dim=2 * d, tie_weights=True, use_gla=True,
        use_swiglu=False,
        # 镜像「从 272M 宽度增长而来」的形态：ln_old 覆盖 1024 维，
        # 新增 384 维用 gamma_new/beta_new。参数量与普通 LayerNorm 相同（都是 2d）。
        use_growth_ln=True, growth_old_dim=growth_old_dim,
        chunk_size=128, mode="chunkwise", dropout=0.0,
    )
    torch.manual_seed(1234)
    m = RetNetForCausalLM(cfg).to(torch.float32)
    n = sum(p.numel() for p in m.parameters())
    print(f"[{name}] d={d} L={L} heads={heads} value_dim={2*d} ffn={2*d}")
    print(f"[{name}] 参数量 = {n:,}")

    # --- 保存前的健全性闸门 ---------------------------------------------
    # 2026-10-11：这两个探针曾经是**坏**的 —— MultiScaleRetention._init_weights
    # 把 1 维 Parameter（=gamma）当成 bias 置了零，而 gamma=0 会让批量 chunkwise
    # 的 log 空间掩码在对角线算出 -inf*0 = NaN，于是整模型前向 NaN。
    # 更阴的是「权重全部有限」的扫描**抓不到它**（0 是有限值），
    # 所以必须在**前向**上验，而不是在权重上验。
    g = m.blocks[0].msr.gamma.detach()
    if not (float(g.min()) > 0.0):
        raise SystemExit(f"[{name}] gamma 最小值 {float(g.min()):.6g} <= 0，"
                         f"门控初值坏了，拒绝保存（见 tests/test_gla_gamma_init.py）")
    print(f"[{name}] gamma 初值 {float(g.min()):.6f} … {float(g.max()):.6f}")

    # 同一处缺陷的第二个受害者：`group_norm.weight` 也是 1 维 Parameter，
    # 被同一条 zeros_ 规则置了零 → retention 支路输出恒为 0（模型初始化即「没有注意力」），
    # 而且会让「快速路径 vs 参考路径」的等价测试**空过**（两边都是 0）。
    gn = [p.detach() for n, p in m.named_parameters() if n.endswith("group_norm.weight")]
    if not gn or any(float(p.min()) == 0.0 for p in gn):
        raise SystemExit(f"[{name}] group_norm.weight 含 0，retention 支路会被置零，拒绝保存")
    print(f"[{name}] group_norm.weight 初值 {float(gn[0].min()):.6f} … {float(gn[0].max()):.6f}"
          f"（{len(gn)} 层）")

    m.eval()
    with torch.no_grad():
        _ids = torch.randint(0, cfg.vocab_size, (1, 256))
        _out = m(input_ids=_ids, labels=_ids)
    _lg, _ls = _out.logits, _out.loss
    if not (torch.isfinite(_lg).all() and torch.isfinite(_ls)):
        raise SystemExit(f"[{name}] 短序列前向非有限（logits absmax={float(_lg.abs().max()):.4g}, "
                         f"loss={float(_ls):.4g}），拒绝保存")
    print(f"[{name}] 短序列前向健全：logits absmax={float(_lg.abs().max()):.4g}  "
          f"loss={float(_ls):.4f}（随机权重，量级合理即可）")
    m.train()

    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(out))
    # from_pretrained 需要 config.json 里的 architectures/model_type 正确；
    # save_pretrained 已写出，这里只补一条备注
    cj = json.loads((out / "config.json").read_text(encoding="utf-8"))
    cj["_note"] = "探针检查点：随机权重，仅用于显存/步时测量，不可用于评测"
    (out / "config.json").write_text(json.dumps(cj, indent=2, ensure_ascii=False), encoding="utf-8")
    sz = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"[{name}] 写出到 {out}  ({sz/1e9:.2f} GB)")
    del m
    gc.collect()


if __name__ == "__main__":
    for name, L, d, heads in CANDIDATES:
        build(name, L, d, heads)
    print("完成")
