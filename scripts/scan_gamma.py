"""只读 safetensors，扫出每个检查点里所有层的 msr.gamma 范围。

目的：`MultiScaleRetention._init_weights` 会把 1 维 Parameter（=gamma）置零。
如果某个**真实训练出来的**检查点 gamma 全为 0，说明这个缺陷在它的训练过程中
就已经生效；如果都离 0 很远，说明缺陷只影响「全新构造」的模型。

只读 tensor，不加载整个模型（有些检查点 2.8 GB）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from safetensors import safe_open
except Exception as e:                                              # noqa: BLE001
    print(f"没有 safetensors：{e}")
    raise SystemExit(1)

rows = []
for d in sorted((ROOT / "checkpoints").iterdir()):
    if not d.is_dir():
        continue
    f = d / "model.safetensors"
    if not f.exists():
        continue
    try:
        with safe_open(str(f), framework="pt") as h:
            keys = [k for k in h.keys() if k.endswith("msr.gamma")]
            if not keys:
                rows.append((d.name, None, None, None, None, "无 gamma"))
                continue
            mn, mx, nz, tot = 1e9, -1e9, 0, 0
            for k in keys:
                t = h.get_tensor(k).float()
                mn = min(mn, t.min().item())
                mx = max(mx, t.max().item())
                nz += int((t == 0).sum())
                tot += t.numel()
            rows.append((d.name, len(keys), mn, mx, f"{nz}/{tot}", ""))
    except Exception as e:                                          # noqa: BLE001
        rows.append((d.name, None, None, None, None, f"{type(e).__name__}"))

print(f"{'检查点':<32}{'层数':>5}{'gamma min':>14}{'gamma max':>14}{'为0个数':>12}  备注")
print("-" * 92)
for name, n, mn, mx, z, note in rows:
    if n is None:
        print(f"{name:<32}{'-':>5}{'-':>14}{'-':>14}{'-':>12}  {note}")
    else:
        flag = "  ← 全零！" if mn == 0 and mx == 0 else ("  ← 含零" if mn == 0 else "")
        print(f"{name:<32}{n:>5}{mn:>14.6g}{mx:>14.6g}{z:>12}  {note}{flag}")

print()
print("参照：RetNet 论文的固定公式 gamma_h = 1 - 2^(-5-h)，H=16 时 min=0.96875, max=0.99997")
