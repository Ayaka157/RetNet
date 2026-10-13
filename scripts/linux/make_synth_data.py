"""就地生成一个**同形状**的极小数据集 —— 让测速不依赖那 1.55 GB 的真数据上传。

## 依据（都是实测）

真实数据的形状与类型（本地 numpy mmap 读出来的）：
    train.ids.npy    (269561, 2049) int16
    train.wcode.npy  (269561, 2049) uint8
wcode 的取值分布（前 2000 行实测）：**{1: 60%, 2: 40%}**，没有 0。
ids 范围 1~19998（vocab 20000）。

步时只取决于**形状**（`docs/23` §4.7(a)：显存由形状决定、与权重值无关，四次复测逐位一致），
所以取几千行就够 `bench_trainer.py` 采样。**但 wcode 不能全填 0** —— 0 很可能表示
"该位置不计损失"，全 0 会让 CE 退化成平凡计算，测出一个偏快的假步时。所以照抄真实分布。

用法：
    python scripts/linux/make_synth_data.py --out data_cache/_synth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--val-rows", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--width", type=int, default=2049, help="真实数据是 2049（= seq 2048 + 1）")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    def gen(n: int):
        ids = rng.integers(1, a.vocab - 1, size=(n, a.width)).astype(np.int16)
        wc = np.where(rng.random((n, a.width)) < 0.60, 1, 2).astype(np.uint8)
        return ids, wc

    ids, wc = gen(a.rows)
    np.save(out / "train.ids.npy", ids)
    np.save(out / "train.wcode.npy", wc)
    vids, vwc = gen(a.val_rows)
    np.save(out / "val.ids.npy", vids)
    np.save(out / "val.wcode.npy", vwc)

    print(f"已生成 {out}")
    print(f"  train {ids.shape} {ids.dtype} / wcode {wc.shape} {wc.dtype}")
    print(f"  val   {vids.shape} {vids.dtype} / wcode {vwc.shape} {vwc.dtype}")
    print(f"  真实数据 train (269561, 2049) int16 / wcode (269561, 2049) uint8 —— 形状一致，仅行数少")
    v, c = np.unique(wc, return_counts=True)
    dist = {int(k): round(float(n) / wc.size, 3) for k, n in zip(v, c)}
    print(f"  wcode 分布: {dist}（真实分布 60% 是 1、40% 是 2，没有 0）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
