"""把 checkpoint 里的 `msr.gamma`（可学习衰减）重初始化。

## 为什么需要

v21 那一轮里 gamma 被 AdamW 的 weight decay 从初值一路拉到了 0.543：
- 初值 `1 - 2^(-5-h)`，半衰期 `τ_h = 2^(5+h)`，从 32 token 到 6.7e7 token
- ep1 0.667 → ep2 0.601 → ep3 0.543，**每 epoch 稳定 ×0.90**（正是 wd=0.1 的固定衰减率）
- 10 个头的 gamma 在 20 层里**逐位相同**（0.542977±1e-6）——不同层不同数据却一样，不可能是学出来的

后果：半衰期掉到约 1 token，**有效上下文只剩约 500 token**，2048 窗口后 3/4 用不上。

修法有两步，缺一不可：
1. **本脚本**：把 gamma 重置回设计初值
2. **训练侧**：`--no_decay_1d` 把 1 维参数移出 weight decay，否则再练一遍又会被拉下去

## 用法

    python scripts/reinit_gamma.py --model checkpoints/retnet_500m_v21_exported \\
        --out checkpoints/retnet_500m_v22_init

只改 `msr.gamma`，其它权重原样搬运。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "blocks."
SUFFIX = ".msr.gamma"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formula", default="orig", choices=["orig", "logspace"],
                    help="orig = 1-2^(-5-h)（架构设计初值）；"
                         "logspace = 半衰期在 200..16000 之间对数分布（为 8k 上下文设计）")
    ap.add_argument("--tau-min", type=float, default=200.0)
    ap.add_argument("--tau-max", type=float, default=16000.0)
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
    keys = [k for k in sd if k.startswith(PREFIX) and k.endswith(SUFFIX)]
    if not keys:
        raise SystemExit(f"没找到 {PREFIX}*{SUFFIX}，请确认权重文件名")

    old = {k: float(sd[k].mean()) for k in sorted(keys)}
    print(f"  找到 {len(keys)} 个 gamma（{len(set(round(v, 6) for v in old.values()))} 个不同取值）")
    print(f"  旧值范围 {min(old.values()):.6f} .. {max(old.values()):.6f}")

    n = len(keys)
    new_vals = []
    for h in range(n):
        if args.formula == "orig":
            g = 1.0 - 2.0 ** (-5.0 - h)
        else:
            # 半衰期在 [tau_min, tau_max] 之间对数分布，覆盖 8k 上下文
            import math
            frac = h / max(1, n - 1)
            tau = args.tau_min * (args.tau_max / args.tau_min) ** frac
            g = math.exp(-math.log(2.0) / tau)
        new_vals.append(g)

    for k, g in zip(sorted(keys), new_vals):
        sd[k] = torch.full_like(sd[k], g)

    print(f"  新值范围 {min(new_vals):.8f} .. {max(new_vals):.8f}")
    import math
    taus = [(-math.log(2) / math.log(g)) if g < 1 else float("inf") for g in new_vals]
    print(f"  对应半衰期 {min(taus):.0f} .. {max(taus):.0f} token")

    if args.dry_run:
        print("  [dry-run] 不写文件")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix in (".json", ".txt"):
            shutil.copy2(p, dst / p.name)
    save_file(sd, str(dst / "model.safetensors"))
    print(f"  已写入 {dst}")

    # 记录这次重初始化，便于以后追溯
    rec = {"from": str(src), "formula": args.formula, "heads": n,
           "old_mean_range": [min(old.values()), max(old.values())],
           "new_range": [min(new_vals), max(new_vals)]}
    (dst / "gamma_reinit.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
