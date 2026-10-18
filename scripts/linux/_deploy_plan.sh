#!/usr/bin/env bash
# 部署动态计划管线到实例，并做一次导入自检。
# 代码由本地 scp 到 _incoming/，本脚本负责备份旧文件 + 覆盖 + 自检。
set -uo pipefail
cd /root/autodl-tmp/RetNet
PY=/root/miniconda3/bin/python
STAMP=$(date +%Y%m%d-%H%M%S)

echo "=== 1) 备份将被覆盖的旧文件 ==="
mkdir -p logs/backup_$STAMP
for f in retnet_hf/plan.py retnet_hf/data.py retnet_hf/train.py retnet_hf/__init__.py tasks/plan.json; do
  if [[ -f "_incoming/$(basename "$f")" ]]; then
    [[ -f "$f" ]] && cp "$f" "logs/backup_$STAMP/$(basename "$f")" && echo "  备份 $f"
  fi
done

echo "=== 2) 覆盖 ==="
for f in plan.py data.py train.py __init__.py; do
  if [[ -f "_incoming/$f" ]]; then
    cp "_incoming/$f" "retnet_hf/$f"
    sed -i 's/\r$//' "retnet_hf/$f"
    echo "  retnet_hf/$f"
  else
    echo "  ⚠️ 缺少 _incoming/$f"
  fi
done
if [[ -f _incoming/plan.json ]]; then
  mkdir -p tasks
  cp _incoming/plan.json tasks/plan.json
  sed -i 's/\r$//' tasks/plan.json
  echo "  tasks/plan.json"
fi
if [[ -f _incoming/run_task.py ]]; then
  [[ -f run_task.py ]] && cp run_task.py "logs/backup_$STAMP/run_task.py" && echo "  备份 run_task.py"
  cp _incoming/run_task.py run_task.py
  sed -i 's/\r$//' run_task.py
  echo "  run_task.py（支持 --plan）"
fi

echo "=== 3) 导入自检 ==="
$PY - <<'PYEOF'
import sys
sys.path.insert(0, '/root/autodl-tmp/RetNet')
from retnet_hf.plan import PlanWatcher, parse_plan, make_plan_callback
from retnet_hf.data import StagePackedDataset
from retnet_hf import StagePackedDataset as S2
print('  import OK')
BATCH = 16  # 与任务配置的 batch_size 一致，仅用于打印估算
w = PlanWatcher('/root/autodl-tmp/RetNet/tasks/plan.json', verbose=False)
p = w.load(force=True)
print('  plan OK: %d 段, total_steps=%d, seq_len=%d' % (len(p.stages), p.total_steps, p.seq_len))
for i, s in enumerate(p.stages):
    print('    段 %d: %-18s %6d 步  %s' % (i, s.name, s.steps, s.data_dir))
PYEOF

echo "=== 4) 检查各阶段就绪性（训练中「边传边用」靠的就是这个）==="
$PY - <<'PYEOF'
import sys
sys.path.insert(0, '/root/autodl-tmp/RetNet')
from retnet_hf.plan import PlanWatcher
from retnet_hf.data import StagePackedDataset
BATCH = 16
w = PlanWatcher('/root/autodl-tmp/RetNet/tasks/plan.json', verbose=False)
p = w.load(force=True)
ds = StagePackedDataset(
    [{"name": s.name, "data_dir": str(s.resolved_dir()), "mode": s.mode,
      "gc": s.gc, "min_shards": s.min_shards, "exclude": s.exclude} for s in p.stages],
    p.seq_len)
for i, s in enumerate(p.stages):
    ready, why = ds.stage_ready(i)
    flag = "[OK]" if ready else "[..]"
    print('  %s 段 %d「%s」%6d 步 = %.3fB token  gc=%s  -> %s'
          % (flag, i, s.name, s.steps, s.steps * p.seq_len * BATCH / 1e9,
             s.gc, why[:60]))
print('  合计 %d 步 = %.2fB token 约 %.1f h / ¥%.1f'
      % (p.total_steps, p.total_steps * p.seq_len * BATCH / 1e9,
         p.total_steps / 1.75 / 3600, p.total_steps / 1.75 / 3600 * 6.98))
PYEOF
echo "=== 部署完成 ==="
