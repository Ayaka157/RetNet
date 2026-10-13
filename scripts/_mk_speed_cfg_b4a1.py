"""派生一个临时测速配置：b4a1 版（对照 tasks/v15b_500m_24g.json 的 b2a2）。

两者每优化步的 token 数相同（4 × 2048 = 8192），所以步时间可直接比。

## 为什么要彩排而不只看步时间

⚠️ **b4a1 这一档从来没有被探测过**：`scripts/linux/speed_probe.sh` 第 21 行写着
「不含 batch>=4：batch4 的激活约 24 GiB，加上 state 4.81 GiB 必然 OOM（**算过，不试**）」。
（我一度在这里写过「探测里 b4a1 = 507 ms、b2a2 = 555 ms，快 8.6%」——那是推算，
没有日志来源，见 `docs/22` §0 第 21 条。而那条「必然 OOM」的推算本身也被证伪了：
b4a1 真实路径实测只用 23,230 MiB / 24,576 MiB。）

真实路径的同口径 300 步对比（有日志）是 b4a1 624 ms vs b2a2 672 ms，快 7.7%。
真正危险的不是训练步，而是**存档**（state_dict + safetensors 写盘）、**评测**
（b4a1 的 eval batch = 4×1）和收尾的 load_best_model_at_end。这三件事在只跑 100 步
的探测里一次都不会发生，却会在正式训练跑到第 10 小时时发生。


所以这个配置配 --save-steps 50 --max-steps 100 使用：跑完 100 步会经历
2 次评测 + 2 次存档 + 1 次收尾存档 + load_best，等于把正式训练的所有内存高峰
都提前演练一遍。跑得过就说明 b4a1 可用，跑不过就说明必须付 b2a2 那 9.7% 的溢价。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tasks" / "v15b_500m_24g.json"
OUT = ROOT / "tasks" / "_speed_b4a1.json"

cfg = json.loads(BASE.read_text(encoding="utf-8"))
cfg["name"] = "_speed_b4a1"
cfg["batch_size"] = 4
cfg["grad_accum"] = 1
cfg["output_dir"] = "checkpoints/_speed_b4a1"
cfg["_comment"] = [
    "临时测速配置（不参与正式训练）：b4a1，与 v15b 的 b2a2 每步 token 数相同。",
    "用法：bash scripts/linux/run_train.sh --config tasks/_speed_b4a1.json \\",
    "        --max-steps 100 --save-steps 50 --no-shutdown",
    "--save-steps 50 会同时把 eval_steps 设成 50，于是 100 步里包含 2 次评测 + 2 次存档。",
    "判据：① 能否全程无 OOM 跑到 TRAINING DONE；② ms/优化步是否真的低于 625 ms。",
    "生成方式：scripts/_mk_speed_cfg_b4a1.py（从 v15b 派生，只改 batch/accum/output）。",
]
OUT.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(OUT)
