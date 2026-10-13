"""生长产物的健全性检查（CPU，不占 GPU）。

比 grow_500m.py 自带的 --verify（seq=128、只比 logits）更强：
  1. gamma / group_norm.weight 是否被正确赋值（新头公式值、新头 weight=1）
  2. **seq=2048** 上批量路径与参考路径都有限
  3. **loss 与源模型逐位接近** —— 生长是函数保真的，loss 应当基本不变。
     这一条能同时抓住「NaN」和「生长把模型改坏了」两类问题。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import retnet_hf.modeling_retnet as MR                                # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM              # noqa: E402

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SEQ = 2048
torch.manual_seed(20261011)
ids = torch.randint(1, 20000, (1, SEQ))


def probe(path, chunk):
    m = RetNetForCausalLM.from_pretrained(path)
    m.config.mode = "chunkwise"
    m.config.chunk_size = chunk
    m.eval()
    g = m.blocks[0].msr.gamma.detach().float()
    gn = m.blocks[0].msr.group_norm.weight.detach().float()
    print(f"\n{path}   (chunk={chunk}, d={m.config.d_model}, L={m.config.num_layers}, "
          f"H={m.config.num_heads}, vdim={m.config.value_dim})")
    print(f"  gamma: H={g.numel()}  min={float(g.min()):.6f} max={float(g.max()):.6f} "
          f"为0个数={int((g == 0).sum())}")
    print(f"  group_norm.weight: min={float(gn.min()):.6f} max={float(gn.max()):.6f} "
          f"为0个数={int((gn == 0).sum())}  （新头的应为 1）")
    out = {}
    with torch.no_grad():
        for flag in (False, True):
            MR.FAST_CHUNKWISE = flag
            o = m(input_ids=ids, labels=ids)
            out[flag] = o
            print(f"  FAST_CHUNKWISE={flag!s:5s}  loss={float(o.loss):.6f}  "
                  f"logits absmax={float(o.logits.abs().max()):.4g}  "
                  f"finite={bool(torch.isfinite(o.logits).all())}")
    d = (out[True].logits - out[False].logits).abs().max().item()
    print(f"  两条路径 logits 最大差 {d:.4g}")
    return out[True].loss.item()


loss_src = probe("checkpoints/retnet_272m_v13", 64)
loss_new = probe("logs/_grow_check/retnet_500m", 128)
print(f"\n源 272M loss = {loss_src:.6f}   生长后 504M loss = {loss_new:.6f}   "
      f"|Δ| = {abs(loss_src - loss_new):.4g}")
print("  => 函数保真 ⇒ |Δ| 应当在数值噪声量级（<1e-2）")
