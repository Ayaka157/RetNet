"""打印某次训练的 eval / train loss 曲线，用于外推还需要多少步。"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
run_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/retnet_272m_v13"
cs = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
            key=lambda s: int(s.split("-")[-1]))
if not cs:
    sys.exit(f"没找到 checkpoint：{run_dir}")
st = json.load(open(os.path.join(cs[-1], "trainer_state.json"), encoding="utf-8"))
print(f"日志来自 {cs[-1]}")
print(f"max_steps 配置 / 实际 global_step = {st.get('global_step')}")
print()
evals, trains = [], []
for e in st["log_history"]:
    if "eval_loss" in e:
        evals.append((e["step"], e["eval_loss"]))
    elif "loss" in e:
        trains.append((e["step"], e["loss"], e.get("learning_rate", 0.0)))

print(f"{'step':>8}{'train_loss':>12}{'eval_loss':>11}{'lr':>12}")
ev = dict(evals)
for s, l, lr in trains:
    if s % 500 == 0 or s in ev:
        print(f"{s:>8}{l:>12.4f}{ev.get(s, float('nan')):>11.4f}{lr:>12.3e}")

print("\n=== 每段 eval 的下降量（看是否还在陡降）===")
prev = None
for s, v in evals:
    if prev is not None:
        ds = s - prev[0]
        d = prev[1] - v
        print(f"  {prev[0]:>6} -> {s:<6}  Δloss {d:+.4f}   "
              f"{d/ds*1000:+.4f} / 千步   {'仍陡' if d/ds*1000 > 0.08 else '趋平'}")
    prev = (s, v)
