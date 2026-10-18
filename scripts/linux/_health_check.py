"""训练健康度体检：从日志里抽 loss / grad_norm / LR，看有没有发散迹象。"""
import json
import re
import statistics as st
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/linux/v21_ddp.out"
txt = open(LOG, encoding="utf-8", errors="replace").read().replace("\r", "\n")

recs = []
for line in txt.splitlines():
    line = line.strip()
    if not line.startswith("{") or "loss" not in line:
        continue
    try:
        d = json.loads(line.replace("'", '"'))
    except Exception:
        continue
    if "loss" in d:
        recs.append(d)

print(f"  解析到 {len(recs)} 条 loss 记录")
if not recs:
    raise SystemExit(0)

losses = [r["loss"] for r in recs if isinstance(r.get("loss"), (int, float))]
grads = [r["grad_norm"] for r in recs if isinstance(r.get("grad_norm"), (int, float))]
print("  loss: 首 %.3f  末 %.3f  最小 %.3f  最大 %.3f  中位 %.3f" % (
    losses[0], losses[-1], min(losses), max(losses), st.median(losses)))
print("  最近 8 条:", " ".join("%.3f" % x for x in losses[-8:]))
if grads:
    print("  grad_norm: 中位 %.2f  最大 %.2f  >5 的次数 %d / %d" % (
        st.median(grads), max(grads), sum(1 for g in grads if g > 5), len(grads)))
nan = sum(1 for x in losses if x != x)
print("  NaN loss 条数:", nan)
print("  异常大 loss (>10):", sum(1 for x in losses if x > 10))

# 每 10 条取一个点，看趋势（是否单调下降）
if len(losses) >= 20:
    seg = [st.mean(losses[i:i + 10]) for i in range(0, len(losses), max(1, len(losses) // 10))][:10]
    print("  分段均值:", " ".join("%.2f" % x for x in seg))
