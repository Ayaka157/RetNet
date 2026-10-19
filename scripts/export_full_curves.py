"""从训练日志导出完整曲线（loss / grad_norm / lr / eval），并校验覆盖范围。

为什么单独写一个：之前误判过"resume 覆盖了旧日志"，实际日志是**追加**的。
这个脚本会打印 step 覆盖范围与空洞，避免再猜。

用法：
    python scripts/export_full_curves.py logs/linux/v21_ddp.out logs/curves/
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/linux/v21_ddp.out")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "logs/curves")

STEP_RE = re.compile(r"\|\s*(\d+)/\d+")
LOSS_RE = re.compile(r"\{'loss': ([0-9.]+)")
GN_RE = re.compile(r"'grad_norm': ([0-9.eE+-]+)")
LR_RE = re.compile(r"'learning_rate': ([0-9.eE+-]+)")
EVAL_RE = re.compile(r"\{'eval_loss': ([0-9.]+)")


def main() -> int:
    raw = LOG.read_text(encoding="utf-8", errors="replace")
    # tqdm 用 \r 覆盖同一行，先统一拆成行
    lines = raw.replace("\r", "\n").split("\n")

    train = []          # (step, loss, grad_norm, lr)
    evals = []          # (step, eval_loss)
    cur = 0
    for ln in lines:
        m = STEP_RE.search(ln)
        if m:
            cur = max(cur, int(m.group(1)))
        ml = LOSS_RE.search(ln)
        if ml:
            g = GN_RE.search(ln)
            r = LR_RE.search(ln)
            train.append((cur, float(ml.group(1)),
                          float(g.group(1)) if g else None,
                          float(r.group(1)) if r else None))
            continue
        me = EVAL_RE.search(ln)
        if me:
            evals.append((cur, float(me.group(1))))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "train_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "grad_norm", "lr"])
        w.writerows(train)
    with open(OUT / "eval_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "eval_loss"])
        w.writerows(evals)

    # 覆盖度自检：按 1,000 步分桶，看哪些桶没有任何记录
    if train:
        lo, hi = train[0][0], train[-1][0]
        buckets = {}
        for s, *_ in train:
            buckets.setdefault(s // 1000, 0)
            buckets[s // 1000] += 1
        empty = [b for b in range(lo // 1000, hi // 1000 + 1) if b not in buckets]
        print(f"  训练记录 {len(train)} 条，step {lo:,} .. {hi:,}")
        print(f"  无记录的千步桶: {len(empty)} 个" + (f"（如 {[b*1000 for b in empty[:6]]}）" if empty else "（全覆盖）"))
        print(f"  eval 记录 {len(evals)} 条" + (f"，step {evals[0][0]:,} .. {evals[-1][0]:,}" if evals else ""))
    print(f"  已写入 {OUT}/train_curve.csv, {OUT}/eval_curve.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
