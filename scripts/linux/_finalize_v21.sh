#!/usr/bin/env bash
# v21 训后收尾：核对导出 → 域内 held-out 终评 → 聊天验证集 → 投入产出 →（可选）关机。
#
# 为什么不直接用 `_finalize.sh`：
#   那个脚本是单卡时代写的，第 7 步会**真起一次训练**做恢复验证，按现在的余额
#   （收尾时只剩个位数）会把钱烧完。这里只做只读检查 + 一次评测。
#
#   bash scripts/linux/_finalize_v21.sh              # 只检查
#   bash scripts/linux/_finalize_v21.sh --shutdown   # 检查完关机
set -uo pipefail
cd /root/autodl-tmp/RetNet
PY=/root/miniconda3/bin/python
OUT=checkpoints/retnet_500m_v21
LOG=logs/linux/v21_ddp.out

echo "=========== 1. 训练是否正常收尾 ==========="
if grep -aq "TRAINING DONE" "$LOG"; then
  echo "  OK  日志里有 TRAINING DONE"
else
  echo "  WARN  没有 TRAINING DONE（多为 tripwire 优雅中断）"
fi
tr '\r' '\n' < "$LOG" | grep -a '| [0-9]*/' | tail -1 | sed 's/^/  最后进度: /'

echo "=========== 2. 导出文件 ==========="
if [ ! -f "$OUT/model.safetensors" ]; then
  # tripwire 优雅中断时，HF 在 train() 里抛 KeyboardInterrupt，train.py 走不到
  # 最后的 trainer.save_model，输出目录就只有 checkpoint-XXXX，没有顶层导出。
  # 这里从最新 checkpoint 补一次导出（键名干净，已多次验证）。
  echo "  WARN  没有顶层 model.safetensors，从最新 checkpoint 补导出"
  CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  echo "  使用 $CKPT"
  CKPT="$CKPT" OUT="$OUT" "$PY" - <<'PYEXPORT'
import os
import sys
sys.path.insert(0, "/root/autodl-tmp/RetNet")
from retnet_hf import RetNetForCausalLM
m = RetNetForCausalLM.from_pretrained(os.environ["CKPT"])
m.save_pretrained(os.environ["OUT"])
print("  已导出到", os.environ["OUT"])
PYEXPORT
fi
ls -la "$OUT"/model.safetensors "$OUT"/config.json 2>/dev/null | awk '{printf "  %10d  %s\n", $5, $9}'

echo "=========== 3. 导出键是否干净（不能有 _orig_mod. 前缀）==========="
"$PY" - <<'PYEOF'
import glob
from safetensors.torch import load_file
f = sorted(glob.glob("checkpoints/retnet_500m_v21/model.safetensors"))
if not f:
    print("  FAIL  没有 safetensors")
else:
    d = load_file(f[0])
    bad = sum(1 for k in d if k.startswith("_orig_mod."))
    print(f"  {len(d)} 个键，带前缀 {bad} 个（应为 0）")
    print("  OK  可直接 from_pretrained" if bad == 0 else "  FAIL  键被污染，需 strip")
PYEOF

echo "=========== 4. 域内 held-out 终评（判据）==========="
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/eval_holdout.py --model "$OUT" --tag v21_final --batch 2 2>&1 | tail -9

echo "=========== 5. 与基线对比 ==========="
"$PY" - <<'PYEOF'
import json
recs = [json.loads(l) for l in open("logs/holdout_eval.jsonl", encoding="utf-8") if l.strip()]
by = {r["tag"]: r for r in recs}
base = by.get("ep3_full")
fin = by.get("v21_final")
if base and fin:
    drop = (base["loss"] - fin["loss"]) / base["loss"] * 100
    print(f"  ep3 基线 {base['loss']:.4f}  ->  v21 {fin['loss']:.4f}   降 {drop:.1f}%")
    print("  OK  达标（>=10%）" if drop >= 10 else "  FAIL  未达 10% 阈值")
else:
    print("  缺基线或终评记录")
PYEOF

echo "=========== 6. 聊天验证集（对话能力掉了多少）==========="
tr '\r' '\n' < "$LOG" | grep -a "eval_loss" | tail -3 | sed 's/^/  /'

echo "=========== 7. 投入产出 ==========="
"$PY" - <<'PYEOF'
import json
import subprocess
p = json.load(open("tasks/plan_ddp2.json", encoding="utf-8"))
tot = sum(s["steps"] for s in p["stages"])
print(f"  计划 {tot:,} 步 = {tot * 65536 / 1e9:.2f}B token")
for s in p["stages"]:
    print(f"    {s['name']:<24} {s['steps']:>6} 步  {s['steps'] * 65536 / 1e9:.2f}B")
try:
    out = subprocess.check_output(
        ["bash", "-c", "ps -eo etimes=,args= | grep '[r]etnet_hf/train.py' | head -1 | awk '{print $1}'"],
        text=True).strip()
    secs = int(out) if out else 0
    h = secs / 3600
    print(f"  训练进程已运行 {h:.2f} h  ->  ¥{h * 13.97:.1f}")
except Exception as exc:
    print("  运行时长读取失败:", exc)
PYEOF

echo "=========== 8. 磁盘 ==========="
df -h /root/autodl-tmp | tail -1 | sed 's/^/  /'
du -sh "$OUT" 2>/dev/null | sed 's/^/  /'

if [ "${1:-}" = "--shutdown" ]; then
  echo "=========== 9. 关机 ==========="
  sync
  /usr/bin/shutdown
fi
echo "=========== 收尾完成 ==========="
