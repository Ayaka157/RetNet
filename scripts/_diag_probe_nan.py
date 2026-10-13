"""定位 probe_500m 前向里 NaN 的第一个产生点。

权重全部有限，seq=128 就 NaN ⇒ 某个算子在**有限输入**上产生了 NaN。
用 forward hook 按**执行顺序**抓第一个输出非有限的模块，并区分：
  - 输入有限、输出 NaN   → 这个模块自己产生了 NaN（真凶）
  - 输入已含 NaN         → 更早的地方就坏了

排除 hook 顺序噪声：只报「输入有限而输出非有限」的第一个模块，
以及「输入已非有限」的最后一个有限模块（即断裂点）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retnet_hf.modeling_retnet import RetNetForCausalLM                 # noqa: E402

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MD = "checkpoints/probe_500m"
m = RetNetForCausalLM.from_pretrained(MD)
m.config.mode = "chunkwise"
m.config.chunk_size = 128
m.eval().to("cpu")

order: list[str] = []
first_bad: dict = {}
last_ok: dict = {}


def fin(x):
    if torch.is_tensor(x):
        return bool(torch.isfinite(x).all())
    if isinstance(x, (list, tuple)):
        return all(fin(v) for v in x)
    if isinstance(x, dict):
        return all(fin(v) for v in x.values())
    return True


def amax(x):
    if torch.is_tensor(x):
        t = x
        return float(t.abs().max()) if t.numel() and t.is_floating_point() else float("nan")
    if isinstance(x, (list, tuple)):
        return max((amax(v) for v in x), default=float("nan"))
    return float("nan")


def mk(name):
    def hook(mod, inp, out):
        order.append(name)
        oin, oout = fin(inp), fin(out)
        if oin and not oout and "gap" not in first_bad:
            first_bad["gap"] = (name, type(mod).__name__, amax(inp), amax(out))
        if oin and "last_ok" not in first_bad:
            first_bad["last_ok"] = (name, type(mod).__name__)
        if not oin and "input_bad" not in first_bad:
            first_bad["input_bad"] = (name, type(mod).__name__)
    return hook


hs = []
for name, mod in m.named_modules():
    if name == "":
        continue
    hs.append(mod.register_forward_hook(mk(name)))

ids = torch.randint(0, 20000, (1, 128))
# 必须带 labels：modeling_retnet.forward 在 labels=None 时 loss 未定义，
# 会抛 UnboundLocalError（models 里的既有缺陷，与本次 NaN 无关）。
with torch.no_grad():
    out = m(input_ids=ids, labels=ids)

for h in hs:
    h.remove()

print(f"共执行 {len(order)} 次模块前向")
print(f"第一个「输入有限、输出非有限」的模块：{first_bad.get('gap')}")
print(f"第一个「输入就已非有限」的模块：    {first_bad.get('input_bad')}")
print(f"最后一个「输入有限」的模块：        {first_bad.get('last_ok')}")
print()
print("执行顺序前 40 个模块名：")
for i, n in enumerate(order[:40]):
    print(f"  {i:3d}  {n}")
