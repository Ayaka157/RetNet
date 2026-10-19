"""画 v21 训练状态图：训练 loss / 域内 held-out / chat 验证集 / 困惑度。

数据来源：
  logs/train_curve.csv   逐步 loss（来自当前日志，**只覆盖 step 32,500 之后**）
  logs/eval_curve.csv    chat 验证集 eval_loss
  logs/holdout_eval.jsonl 域内 held-out（判据用的那个）
  tasks/plan_ddp2.json    阶段边界
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures" / "v21_training_status.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 读数据
# 完整历史来自 checkpoint 的 trainer_state.json（log_history），
# 比 .out 日志更完整：日志在续训时只从 32,520 步开始，而 log_history 从第 20 步就有。
train_step, train_loss = [], []
with open(ROOT / "logs/curves/full_history.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["loss"]:
            train_step.append(int(r["step"]))
            train_loss.append(float(r["loss"]))
train_step = np.array(train_step)
train_loss = np.array(train_loss)

eval_step, eval_loss = [], []
with open(ROOT / "logs/curves/full_history.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["eval_loss"]:
            eval_step.append(int(r["step"]))
            eval_loss.append(float(r["eval_loss"]))
eval_step = np.array(eval_step)
eval_loss = np.array(eval_loss)

hold = []
with open(ROOT / "logs/holdout_eval.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            hold.append(json.loads(line))
hold = {h["tag"]: h for h in hold}
base = hold.get("ep3_full", {}).get("loss")
pts = []
if base:
    pts.append((0, base))
for tag, step in (("v21_step15000", 15000), ("v21_step17500", 17500)):
    if tag in hold:
        pts.append((step, hold[tag]["loss"]))
pts.sort()

plan = json.loads((ROOT / "tasks/plan_ddp2.json").read_text(encoding="utf-8"))
bounds, acc = [], 0
for s in plan["stages"]:
    acc += s["steps"]
    bounds.append((acc, s["name"]))
total = acc

# 早前那段（step 0..32,500）的日志被 resume 覆盖了，用当时健康检查记录的分段均值重建
# 不再需要重建：log_history 从第 20 步起就是真实记录

# ---------------------------------------------------------------- 画图
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle(f"RetNet-0.5B v21 训练状态（{total:,} 步计划，实时）", fontsize=15, fontweight="bold")

# (a) 训练 loss
ax = axes[0, 0]

# 按阶段着色：代码段 loss 只有 0.8 左右，混在一起会把预训练段压扁
edges = [0] + [b for b, _ in bounds]
colors = ["#1a73e8", "#188038", "#ea8600", "#a142f4", "#d93025", "#12b5cb", "#f439a0"]
cur_now = int(train_step[-1]) if len(train_step) else total
labelled = set()
for si in range(len(edges) - 1):
    lo, hi = edges[si], edges[si + 1]
    m = (train_step >= lo) & (train_step < hi)
    if not m.any():
        continue
    c = colors[si % len(colors)]
    name = bounds[si][1]
    ax.plot(train_step[m], train_loss[m], color=c, alpha=0.35, lw=0.6)
    ax.plot(train_step[m], train_loss[m], ".", color=c, ms=2.2)
    k = min(25, max(5, m.sum() // 4))
    if k >= 2:
        sm = np.convolve(train_loss[m], np.ones(k) / k, mode="valid")
        ax.plot(train_step[m][k - 1:], sm, color=c, lw=2.0,
                label=(name if name not in labelled else None))
        labelled.add(name)
ax.set_yscale("log")
ax.set_ylim(0.4, 4.2)
cur = int(train_step[-1]) if len(train_step) else total
for b, name in bounds:
    if b <= cur * 1.02:
        ax.axvline(b, color="#d93025", ls=":", lw=0.9, alpha=0.7)
        ax.text(b, ax.get_ylim()[1], name.replace("_", "\n"), fontsize=6.5,
                rotation=90, va="top", ha="right", color="#d93025")
ax.set_xlim(0, cur * 1.02)      # 不把横轴撑到未来的阶段
ax.set_title(f"(a) 训练 loss（按阶段着色，对数轴；已到 {cur:,} / {total:,} 步）", fontsize=10)
ax.set_xlabel("step"); ax.set_ylabel("loss（对数）")
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7, loc="lower left", ncol=2)

# (b) 域内 held-out
ax = axes[0, 1]
if pts:
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, "o-", color="#188038", lw=2, ms=7)
    for (x, y) in pts:
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    if base:
        ax.axhline(base * 0.9, color="#ea8600", ls="--", lw=1.2,
                   label=f"成功阈值 = -10% ({base * 0.9:.4f})")
        ax.axhline(base, color="#5f6368", ls=":", lw=1, label=f"ep3 基线 {base:.4f}")
ax.set_title("(b) 域内 held-out loss（真正的判据）", fontsize=11)
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_xlim(-500, 20000)
ax.grid(alpha=0.25); ax.legend(fontsize=8)

# (c) chat 验证集
ax = axes[1, 0]
if len(eval_step):
    ax.plot(eval_step, eval_loss, "o-", color="#a142f4", ms=5)
    ax.axhline(1.9459, color="#5f6368", ls=":", lw=1, label="ep3 的 chat-val = 1.9459")
ax.set_title("(c) chat 验证集（阶段 0–3 无对话数据，必然变差）", fontsize=11)
ax.set_xlabel("step"); ax.set_ylabel("loss")
ax.grid(alpha=0.25); ax.legend(fontsize=8)

# (d) 困惑度
ax = axes[1, 1]
# 两条 PPL 放**同一根对数轴**上：双轴会让人误读成同一条曲线
if pts:
    ax.plot([p[0] for p in pts], [np.exp(p[1]) for p in pts], "o-",
            color="#188038", lw=2, ms=7, label="域内 held-out PPL")
    for (x, y) in pts:
        ax.annotate(f"{np.exp(y):.1f}", (x, np.exp(y)), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color="#188038")
if len(eval_step):
    ax.plot(eval_step, np.exp(eval_loss), "s--", color="#a142f4", ms=5,
            label="chat 验证集 PPL")
    ax.annotate(f"{np.exp(eval_loss[-1]):.1f}", (eval_step[-1], np.exp(eval_loss[-1])),
                textcoords="offset points", xytext=(8, -12), fontsize=9, color="#a142f4")
ax.set_yscale("log")
ax.set_title("(d) 困惑度 PPL = exp(loss)（对数轴）", fontsize=11)
ax.set_xlabel("step"); ax.set_ylabel("PPL（对数）")
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"  已保存 {OUT}")
print(f"  当前进度 {train_step[-1] if len(train_step) else 0:,} / {total:,} 步")
if pts:
    print(f"  held-out 最新 {pts[-1][1]:.4f}（基线 {base:.4f}，"
          f"降 {(base - pts[-1][1]) / base * 100:.1f}%）")
