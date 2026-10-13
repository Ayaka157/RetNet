"""控制实验：run_task.py 必须把训练子进程的非零退出码传播出去。

## 为什么需要这个测试

2026-10-11 实测（AutoDL，日志 train_20261011_174631.log）：train.py 因
TrainingArguments 校验抛 ValueError，日志里明明有 traceback 和 `[exit] rc=1`，
但 run_train.sh 拿到的是「训练进程退出码: 0」—— 因为 run_task.py 的 while 循环
在 `rc in (0,1) and attempts >= auto_restart_max` 时 break，main() 正常返回，
进程以 0 退出。无人值守跑 10 小时时，任何崩溃都会被记成「训完了」并触发自动关机。

## 做法

拿 v15b 配置，只把 model_dir 指向一个不存在的目录。train.py 的**第一个动作**
就是 `RetNetForCausalLM.from_pretrained(model_dir)`（train.py:148），所以它会
立刻抛 OSError —— 纯 CPU、不碰 CUDA、不读数据集、秒级结束。

这个测试**能失败**：修复前 rc=0，修复后 rc!=0。不是那种永远通过的检查。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tasks" / "v15b_500m_24g.json"
TMP = ROOT / "tasks" / "_tmp_rc_prop.json"


def main() -> int:
    cfg = json.loads(BASE.read_text(encoding="utf-8"))
    cfg["model_dir"] = "checkpoints/__no_such_model__"
    cfg["max_steps"] = 1
    cfg["output_dir"] = "checkpoints/__no_such_out__"
    TMP.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        r = subprocess.run(
            [sys.executable, "run_task.py", "--config", str(TMP)],
            cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300)
    finally:
        TMP.unlink(missing_ok=True)

    out = (r.stdout or "") + (r.stderr or "")
    print("run_task.py 退出码 =", r.returncode)
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-8:]
    for ln in tail:
        print("   ", ln)

    ok = r.returncode != 0
    print()
    if ok:
        print("✅ 非零退出码已传播（崩溃不会再被记成成功）")
    else:
        print("❌ FAIL：子进程崩了，run_task.py 仍以 0 退出 —— 静默失败又回来了")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
