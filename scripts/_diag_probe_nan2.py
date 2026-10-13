"""probe 的 NaN 是「快速 chunkwise 内核的 bug」还是「模型/配置本身」？

判据：同一次前向分别走
  (a) retention_fast.chunkwise_batched（FAST_CHUNKWISE=1）
  (b) modeling_retnet 里的参考 per-chunk 循环（FAST_CHUNKWISE=0）
若 (b) 有限而 (a) 是 NaN ⇒ 内核 bug，与检查点无关，**真实模型同样会中招**。
若两条都 NaN ⇒ 配置/权重问题。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import retnet_hf.modeling_retnet as M                                # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM              # noqa: E402

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

print("=" * 70)
print("一、各检查点的 per-head gamma")
print("=" * 70)
for d in ("checkpoints/probe_500m", "checkpoints/probe_600m", "checkpoints/retnet_272m_v13"):
    m = RetNetForCausalLM.from_pretrained(d)
    g = m.blocks[0].msr.gamma.detach().float()
    lg = torch.log(g)
    print(f"{d}")
    print(f"  H={g.numel()}  gamma: min={g.min().item():.8g} max={g.max().item():.8g} "
          f"等于0的个数={int((g == 0).sum())} 等于1的个数={int((g == 1).sum())}")
    print(f"  log(gamma): min={lg.min().item():.6g} max={lg.max().item():.6g} "
          f"非有限={int((~torch.isfinite(lg)).sum())}")
    print(f"  use_gla={m.config.use_gla}")
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print()
print("=" * 70)
print("二、probe_500m 同一次前向，快速路径 vs 参考路径（seq=128，CPU fp32）")
print("=" * 70)
m = RetNetForCausalLM.from_pretrained("checkpoints/probe_500m")
m.config.mode = "chunkwise"
m.config.chunk_size = 128
m.eval()
ids = torch.randint(0, 20000, (1, 128))
with torch.no_grad():
    for flag in (True, False):
        M.FAST_CHUNKWISE = flag
        out = m(input_ids=ids, labels=ids)
        lg = out.logits
        print(f"  FAST_CHUNKWISE={flag!s:5s}  logits absmax={lg.abs().max().item():12.4g}  "
              f"finite={bool(torch.isfinite(lg).all())}  loss={out.loss.item():10.4f}")

print()
print("=" * 70)
print("三、把 gamma 强制置 1 后，快速路径还 NaN 吗（隔离 gamma 的作用）")
print("=" * 70)
import copy  # noqa: E402
m2 = RetNetForCausalLM.from_pretrained("checkpoints/probe_500m")
m2.config.mode = "chunkwise"
m2.config.chunk_size = 128
m2.eval()
with torch.no_grad():
    m2.blocks[0].msr.gamma.fill_(1.0)
M.FAST_CHUNKWISE = True
with torch.no_grad():
    out = m2(input_ids=ids, labels=ids)
    print(f"  blocks.0.gamma=1  FAST=1  logits absmax={out.logits.abs().max().item():12.4g}  "
          f"finite={bool(torch.isfinite(out.logits).all())}")

print()
print("=" * 70)
print("四、单块隔离：直接喂有限输入给 blocks.0.msr，看哪个中间量先坏")
print("=" * 70)
m3 = RetNetForCausalLM.from_pretrained("checkpoints/probe_500m")
m3.config.mode = "chunkwise"
m3.config.chunk_size = 128
m3.eval()
blk = m3.blocks[0].msr
M.FAST_CHUNKWISE = True
x = torch.randn(1, 128, m3.config.d_model) * 0.1
with torch.no_grad():
    q, k, v, g = blk._project_and_rotate(x, 0)
    print(f"  q {tuple(q.shape)} absmax={q.abs().max().item():.4g}")
    print(f"  k {tuple(k.shape)} absmax={k.abs().max().item():.4g}")
    print(f"  v {tuple(v.shape)} absmax={v.abs().max().item():.4g}")
    print(f"  g {tuple(g.shape)} absmax={g.abs().max().item():.4g}")
    from retnet_hf.retention_fast import _decay_tables
    im, cd, vd_, chd = _decay_tables(blk.gamma, 128, q.dtype, q.device)
    for nm, t in (("inner_mask", im), ("cross_decay", cd),
                  ("value_decay", vd_), ("chunk_decay", chd)):
        print(f"  {nm:12s} {tuple(t.shape)} absmax={t.abs().max().item():.4g} "
              f"finite={bool(torch.isfinite(t).all())} nan={int(torch.isnan(t).sum())}")
    out, st = blk._chunkwise_retention(q, k, v, None, 128)
    print(f"  内核输出 absmax={out.abs().max().item():.4g} "
          f"finite={bool(torch.isfinite(out).all())} nan={int(torch.isnan(out).sum())}")
