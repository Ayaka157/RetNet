"""Print per-segment step time from a run's trainer_state.json."""
import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
out = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/_smoke_opt")
cands = sorted(glob.glob(str(out / "checkpoint-*" / "trainer_state.json")))
p = cands[-1] if cands else (out / "trainer_state.json")
if not Path(p).exists():
    sys.exit(f"找不到 trainer_state.json（看了 {out}）")
state = json.load(open(p, encoding="utf-8"))
hist = [e for e in state["log_history"] if "loss" in e]
print(f"日志: {p}")
print(f"{'step':>6}{'loss':>10}{'本段秒':>10}{'步数':>6}{'ms/步':>9}")
prev_rt, prev_st = 0.0, 0
rows = []
for e in hist:
    st, rt = e["step"], e.get("train_runtime", 0.0)
    if st > prev_st and rt > prev_rt:
        dt, ds = rt - prev_rt, st - prev_st
        rows.append((st, e["loss"], dt, ds, dt / ds * 1000))
        print(f"{st:>6}{e['loss']:>10.4f}{dt:>10.1f}{ds:>6}{dt/ds*1000:>9.0f}")
    prev_rt, prev_st = rt, st
if len(rows) > 1:
    tail = rows[len(rows) // 2:]
    avg = sum(r[4] for r in tail) / len(tail)
    print(f"\n后半程平均 {avg:.0f} ms/步（{len(tail)} 段）")
    first = rows[0][4]
    print(f"第 1 段 {first:.0f} ms/步（含 inductor 编译）")
