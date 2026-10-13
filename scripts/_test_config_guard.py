"""控制实验：check_task_config.py 的跨字段检查既不能不响，也不能乱响。

## 为什么需要这个测试

2026-10-11 我在 tasks/v15b_500m_24g.json 里写了 save_steps=2000 / eval_steps=5000。
train.py 开了 load_best_model_at_end=True（train.py:196），HF 要求 save_steps 必须是
eval_steps 的整数倍，于是训练在构造 TrainingArguments 时直接抛 ValueError —— 一步
都没跑，白等一次 GPU 排队。这纯 CPU 就能查出来，所以补进 check_task_config.py。

## 两向控制

- 正向：改好的 v15b（2500/2500）必须 **通过**（否则检查器会天天误报）。
- 负向：把它改回 2000/2500 必须 **报错**（否则这个检查等于没有）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tasks" / "v15b_500m_24g.json"
TMP = ROOT / "tasks" / "_tmp_neg_save_eval.json"


def run(cfg_path: Path):
    return subprocess.run(
        [sys.executable, "scripts/check_task_config.py", str(cfg_path)],
        cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300)


def main() -> int:
    good = run(BASE)

    cfg = json.loads(BASE.read_text(encoding="utf-8"))
    cfg["save_steps"] = 2000          # 2000 % eval_steps(2500) != 0
    cfg["eval_steps"] = 2500
    TMP.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        bad = run(TMP)
    finally:
        TMP.unlink(missing_ok=True)

    ok_good = good.returncode == 0
    bad_txt = (bad.stdout or "") + (bad.stderr or "")
    ok_bad = bad.returncode != 0 and "不是 eval_steps" in bad_txt

    print(f"正向 v15b(save=2500/eval=2500) 应通过 : "
          f"{'✅ OK' if ok_good else '❌ FAIL'}  rc={good.returncode}")
    print(f"负向 改回 save=2000/eval=2500 应报错 : "
          f"{'✅ OK' if ok_bad else '❌ FAIL'}  rc={bad.returncode}")
    if not ok_bad:
        print("  负向输出末尾：")
        for ln in bad_txt.strip().splitlines()[-8:]:
            print("   ", ln)

    return 0 if (ok_good and ok_bad) else 1


if __name__ == "__main__":
    sys.exit(main())
