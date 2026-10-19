#!/usr/bin/env bash
# 训练日志归档器：每 30 分钟把完整 loss 历史与原始日志落一份，防止丢。
#
#   setsid nohup bash scripts/linux/_archive_logs.sh > logs/linux/archiver.out 2>&1 &
#
# 为什么需要它：
#   1. `.out` 日志在**每次重启训练时会被截断**（本项目 19:36 那次重启就丢了
#      前 32,500 步的 loss 记录）。原始日志必须滚动归档。
#   2. 但完整历史其实在 checkpoint 的 `trainer_state.json: log_history` 里——
#      HF Trainer 续训时会累加，不会丢。所以真正的"完整日志"要从那里导。
#   3. checkpoint 会被轮转（save_total_limit=4）和磁盘守卫裁剪，所以不能指望
#      "结束再导"，必须持续导出。
set -uo pipefail
cd /root/autodl-tmp/RetNet
PY=/root/miniconda3/bin/python
ARCH=logs/archive
mkdir -p "$ARCH" logs/curves

while true; do
  ts=$(date '+%m%d_%H%M')

  # 1) 完整 loss 历史（来自最新 checkpoint 的 trainer_state.json）
  $PY - <<'PYEOF' >> logs/linux/archiver.out 2>&1
import csv, json
from pathlib import Path
cks = sorted(Path("checkpoints").glob("*/checkpoint-*"),
             key=lambda p: p.stat().st_mtime)
if not cks:
    raise SystemExit(0)
ck = cks[-1]
src = ck / "trainer_state.json"
if not src.exists():
    raise SystemExit(0)
d = json.loads(src.read_text(encoding="utf-8"))
h = d.get("log_history", [])
out = Path("logs/curves/full_history.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["step", "epoch", "loss", "grad_norm", "lr", "eval_loss"])
    for x in h:
        w.writerow([x.get("step", ""), x.get("epoch", ""), x.get("loss", ""),
                    x.get("grad_norm", ""), x.get("learning_rate", ""),
                    x.get("eval_loss", "")])
print("[%s] 导出 %s -> %d 条" % (__import__("time").strftime("%H:%M:%S"), src, len(h)))
PYEOF

  # 2) 原始日志滚动归档（每次训练重启会截断它，所以必须留副本）
  for f in logs/linux/v21_ddp.out logs/linux/v22_ctx16k.out; do
    [ -f "$f" ] && cp -f "$f" "$ARCH/$(basename "$f" .out)_$ts.out"
  done
  # 3) 评测记录
  [ -f logs/holdout_eval.jsonl ] && cp -f logs/holdout_eval.jsonl "$ARCH/holdout_eval_$ts.jsonl"

  # 4) 只留最近 24 份（约 12 小时），避免自己把磁盘吃满
  ls -1t "$ARCH"/*.out 2>/dev/null | tail -n +25 | xargs -r rm -f
  ls -1t "$ARCH"/holdout_eval_*.jsonl 2>/dev/null | tail -n +13 | xargs -r rm -f

  # 训练和 v22 都跑完了就再导一次然后退出
  if ! pgrep -f "retnet_hf/train[.]py" > /dev/null; then
    sleep 300
    if ! pgrep -f "retnet_hf/train[.]py" > /dev/null; then
      echo "[$(date '+%H:%M:%S')] 训练已结束，做最后一次导出"
      break
    fi
  fi
  sleep 1800
done

# 最后一次导出（训练结束后）
$PY - <<'PYEOF' >> logs/linux/archiver.out 2>&1
import csv, json, time
from pathlib import Path
for ck in sorted(Path("checkpoints").glob("*/checkpoint-*"), key=lambda p: p.stat().st_mtime)[-1:]:
    h = json.loads((ck / "trainer_state.json").read_text(encoding="utf-8")).get("log_history", [])
    with open("logs/curves/full_history_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "loss", "grad_norm", "lr", "eval_loss"])
        for x in h:
            w.writerow([x.get("step", ""), x.get("epoch", ""), x.get("loss", ""),
                        x.get("grad_norm", ""), x.get("learning_rate", ""),
                        x.get("eval_loss", "")])
    print("[%s] 最终导出 %d 条 -> logs/curves/full_history_final.csv" % (
        time.strftime("%H:%M:%S"), len(h)))
PYEOF
echo "[$(date '+%H:%M:%S')] 归档器退出"
