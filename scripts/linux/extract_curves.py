"""从训练日志里抽出图表用的数据，输出 CSV（不画图，画图在本地做）。"""
import json
import re
import sys
from pathlib import Path

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/linux/v21_ddp.out"
OUT = sys.argv[2] if len(sys.argv) > 2 else "logs/train_curve.csv"
EVAL = sys.argv[3] if len(sys.argv) > 3 else "logs/eval_curve.csv"

txt = Path(LOG).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")

# 1) 逐步 loss / grad_norm / lr
step_rows = []
cur_step = 0
for line in txt.splitlines():
    line = line.strip()
    m = re.search(r"\|\s*(\d+)/\d+", line)
    if m:
        cur_step = max(cur_step, int(m.group(1)))
    if not line.startswith("{") or "loss" not in line:
        continue
    try:
        d = json.loads(line.replace("'", '"'))
    except Exception:
        continue
    if "loss" not in d:
        continue
    step_rows.append((cur_step, d.get("loss"), d.get("grad_norm"), d.get("learning_rate")))

# 2) eval（chat 验证集）
eval_rows = []
cur_step = 0
for line in txt.splitlines():
    m = re.search(r"\|\s*(\d+)/\d+", line)
    if m:
        cur_step = max(cur_step, int(m.group(1)))
    if "eval_loss" not in line:
        continue
    # tqdm 会往同一行前面塞 ANSI 控制符（如 "[A"），必须从第一个 { 开始截
    i = line.find("{")
    if i < 0:
        continue
    try:
        d = json.loads(line[i:].replace("'", '"'))
    except Exception:
        continue
    if "eval_loss" in d:
        eval_rows.append((cur_step, d["eval_loss"]))

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("step,loss,grad_norm,lr\n")
    for s, l, g, r in step_rows:
        f.write(f"{s},{l if l is not None else ''},{g if g is not None else ''},"
                f"{r if r is not None else ''}\n")
with open(EVAL, "w", encoding="utf-8") as f:
    f.write("step,eval_loss\n")
    for s, l in eval_rows:
        f.write(f"{s},{l}\n")

print(f"  逐步记录 {len(step_rows)} 条 -> {OUT}")
print(f"  eval 记录 {len(eval_rows)} 条 -> {EVAL}")
if step_rows:
    print(f"  step 范围 {step_rows[0][0]} .. {step_rows[-1][0]}")
    print(f"  loss 范围 {min(r[1] for r in step_rows if r[1] is not None):.3f} .. "
          f"{max(r[1] for r in step_rows if r[1] is not None):.3f}")
