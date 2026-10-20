"""把 checkpoint 里的 `msr.gamma`（每头可学习衰减）按**头**重初始化。

## 这个脚本以前是错的（docs/00 N01）

旧实现把 20 个 `blocks.*.msr.gamma` 张量当成 20 个"头"来遍历：

    keys = [k for k in sd if k.endswith(".msr.gamma")]   # 20 个，每层一个 [22]
    for h in range(len(keys)):                            # h = 0..19，是层号
        g = 1 - 2 ** (-5 - h)
        sd[key] = torch.full_like(sd[key], g)             # 22 个头被灌成同一个值

结果不是"按头设置"，而是"按层设置"，并且每层 22 个头完全相同——多尺度设计
被抹平成一个尺度。它打印的"旧值 0.399..0.586"也是每层 22 头的**均值**，
把真实的 0.106..0.999 抹掉了，所以连诊断都是错的。

## 现在的语义

对每一层的 [H] 向量，**按头索引** h 写入：

    tau_h   = tau_min * (tau_max / tau_min) ** (h / (H-1))
    gamma_h = 0.5 ** (1 / tau_h)

即半衰期从 tau_min 到 tau_max 对数均匀，每层相同、头与头不同。
默认 [128, 8192]：下限对齐 chunk_size=128（短于一个 chunk 的头等于没有跨块记忆），
上限对齐 8K 上下文。也可以用 `--formula orig` 取论文的 1-2^(-5-h)，
但注意它在 h=21 时半衰期约 4.6e7 token——那不是"长记忆"，是"永不遗忘"。

## 用法

    python scripts/reinit_gamma.py --model checkpoints/retnet_500m_v21_export \
        --out checkpoints/retnet_500m_v22_gamma --dry-run
    python scripts/reinit_gamma.py --model checkpoints/retnet_500m_v21_export \
        --out checkpoints/retnet_500m_v22_gamma            # 真写

只改 `msr.gamma`，其它权重原样搬运；同时写一份 `gamma_reinit.json` 记录参数。
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SUFFIX = ".msr.gamma"


def build_gamma(num_heads: int, formula: str = "logspace",
                tau_min: float = 128.0, tau_max: float = 8192.0) -> torch.Tensor:
    """返回 [num_heads] 的 gamma，按**头索引**排列。

    formula="logspace"：半衰期在 [tau_min, tau_max] 对数均匀；
    formula="orig"：论文的 1-2^(-5-h)，h 是头索引（半衰期 2^(5+h)）。
    """
    if num_heads <= 0:
        raise ValueError("num_heads 必须为正")
    if formula == "orig":
        return 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads, dtype=torch.float64))
    if formula != "logspace":
        raise ValueError(f"未知 formula: {formula}")
    if not (tau_min > 0 and tau_max > 0 and tau_max >= tau_min):
        raise ValueError(f"tau 范围非法: {tau_min}..{tau_max}")
    frac = torch.linspace(0.0, 1.0, steps=num_heads, dtype=torch.float64)
    tau = tau_min * (tau_max / tau_min) ** frac
    return torch.pow(0.5, 1.0 / tau)


def half_lives(gamma: torch.Tensor) -> torch.Tensor:
    """把 gamma 换算成半衰期（token）。gamma>=1 时返回 inf。"""
    g = gamma.double()
    out = torch.full_like(g, float("inf"))
    ok = g < 1.0
    out[ok] = math.log(0.5) / torch.log(g[ok])
    return out


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formula", default="logspace", choices=["logspace", "orig"])
    ap.add_argument("--tau-min", type=float, default=128.0)
    ap.add_argument("--tau-max", type=float, default=8192.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.model)
    if not src.is_absolute():
        src = ROOT / src
    dst = Path(args.out)
    if not dst.is_absolute():
        dst = ROOT / dst

    from safetensors.torch import load_file, save_file

    files = sorted(src.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"{src} 里没有 safetensors")
    sd = load_file(str(files[0]))

    keys = sorted(k for k in sd if k.endswith(SUFFIX))
    if not keys:
        raise SystemExit(f"没找到 *{SUFFIX}，请确认权重文件名")

    heads = int(sd[keys[0]].numel())
    for k in keys:
        if int(sd[k].numel()) != heads:
            raise SystemExit(f"{k} 的头数 {sd[k].numel()} 与 {keys[0]} 的 {heads} 不一致")

    old = torch.stack([sd[k].detach().double().flatten() for k in keys])  # [L, H]
    old_h = half_lives(old.flatten())
    print(f"  找到 {len(keys)} 层 x {heads} 头 = {len(keys) * heads} 个 gamma")
    print(f"  旧值: {float(old.min()):.6f} .. {float(old.max()):.6f}")
    print(f"  旧半衰期: p25={float(torch.quantile(old_h, 0.25)):.2f} "
          f"p50={float(torch.quantile(old_h, 0.50)):.2f} "
          f"p75={float(torch.quantile(old_h, 0.75)):.2f} token")

    gamma = build_gamma(heads, args.formula, args.tau_min, args.tau_max)
    tau = half_lives(gamma)
    print(f"  新 gamma: {float(gamma.min()):.8f} .. {float(gamma.max()):.8f}")
    print(f"  新半衰期: {float(tau.min()):.1f} .. {float(tau.max()):.1f} token，"
          f"不同取值 {len(set(round(float(v), 8) for v in gamma))} 个")
    # 同一层内不允许有重复值：重复就说明又退化成"按层设置"了
    if len(set(round(float(v), 8) for v in gamma)) != heads:
        raise SystemExit("新 gamma 在同一层内出现重复值——按头设置失败，拒绝写入")

    for k in keys:
        sd[k] = gamma.to(sd[k].dtype).reshape(sd[k].shape)

    if args.dry_run:
        print("  [dry-run] 不写文件")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix in (".json", ".txt"):
            shutil.copy2(p, dst / p.name)
    save_file(sd, str(dst / "model.safetensors"))

    rec = {
        "from": str(src), "formula": args.formula,
        "tau_min": args.tau_min, "tau_max": args.tau_max,
        "layers": len(keys), "heads": heads,
        "old_range": [float(old.min()), float(old.max())],
        "new_range": [float(gamma.min()), float(gamma.max())],
        "new_tau_range": [float(tau.min()), float(tau.max())],
    }
    (dst / "gamma_reinit.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已写入 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
