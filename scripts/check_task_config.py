"""死配置字段检测器 —— `docs/22` §2.5 提议过、但一直没实现的那个守卫。

    python scripts/check_task_config.py tasks/v13_272m_2k.json
    python scripts/check_task_config.py --all          # 扫全部 tasks/*.json

## 它防的是什么

本项目已经**四例**同一形状的缺陷：任务 JSON 里写了一个键，`run_task.py` 从不把它
拼进命令行，于是 `train.py` 的硬编码默认值一直生效。**每一例都不报错**：

| 键 | 实际生效的值 | 后果 |
|---|---|---|
| `chunk_size` | 64（函数默认）而不是 128 | 一直跑 64 |
| `eval_batch_mult` | ×4（train.py 硬编码） | 评测慢、峰值冲到 7.65 GiB |
| `save_total_limit` | 3 而不是 4 | **静默删掉 checkpoint-10000** |
| `checkpoint_layers` | 无此参数 | 该开关在真实训练里**根本不存在** |

## 做法（不维护手工清单）

1. 跑 `run_task.py --dry`，解析它**真正拼出来的** `[cmd]` 命令行——用的是真实代码路径。
2. 把 JSON 的键映射到"应该出现的 flag"，检查它是否出现在 cmd 里。
   映射表只在**有语义差别**的地方需要（见 `KEY_TO_FLAG`），其余按 `--<key>` 同名推断。
3. 反向检查：cmd 里出现的**每一个** flag，都必须在 `retnet_hf/train.py` 的 argparse 里
   存在——否则 train.py 会直接崩（这一条能抓出上面第 4 例）。
4. 报告「JSON 没写、会走 DEFAULTS」的键——它们不报错，但值可能出乎意料。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# 不是命令行 flag 的「控制面」键：由 run_task.py 自己消费，或只是元数据
CONTROL_PLANE = {
    "name", "train_script", "auto_restart_max", "restart_wait_min",
    "load_best", "fast_chunkwise",        # fast_chunkwise 走环境变量
    "_comment",
}
# 有语义差别的键 -> 它应该产生的 flag（无条件出现的）
KEY_TO_FLAG = {
    "eval_batch_mult": "--eval_batch_mult",
}
# 条件 flag：(flag, 值为该值时出现)。
# ⚠️ 第一版把 `resume` 放进了 KEY_TO_FLAG，于是 `resume: false` 的配置被误报成死字段——
#    **检测器自己犯了它要检测的那类错（把条件当无条件）**。已挪到这里。
COND_FLAG = {
    "compile": ("--no-compile", False),              # False -> --no-compile；True -> 无
    "grad_checkpoint": ("--grad_checkpoint", True),  # True -> --grad_checkpoint；False -> 无
    "resume": ("--resume", True),                    # True -> --resume；False -> 无
    # 只在 compile=True 且值非空时才拼进命令行（见 run_task.py build_cmd_full）。
    # 值是真值判断：空字符串 -> 不出现，非空 -> 出现。
    "compile_mode": ("--compile_mode", True),
}


def train_py_flags() -> set:
    """从 retnet_hf/train.py 里抽出 argparse 声明的 flag 集合。

    ⚠️ 字符类必须**包含连字符**：第一版写成 `[A-Za-z0-9_]+`，于是
    `--no-compile`（`train.py:117`）永远匹配不到，5 个历史配置被误报成
    「train.py 不认识的 flag，会直接崩」——**检测器自己错了**。
    它们其实跑了几个月都没崩。教训：拿"历史上能跑通"当基准真值来校准检测器。
    """
    src = (ROOT / "retnet_hf" / "train.py").read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\(\s*["\'](--[A-Za-z0-9_-]+)["\']', src))


def load_best_on() -> bool:
    """train.py 是否开了 load_best_model_at_end —— 从源码读，不硬编码。"""
    src = (ROOT / "retnet_hf" / "train.py").read_text(encoding="utf-8")
    return bool(re.search(r"load_best_model_at_end\s*=\s*True", src))


def _flag_val(cmd, name):
    """从真实拼出的 cmd 里取一个整型 flag 的值。"""
    if name in cmd:
        i = cmd.index(name)
        if i + 1 < len(cmd):
            try:
                return int(cmd[i + 1])
            except ValueError:
                return None
    return None


def run_dry(cfg_path: Path):
    """跑 run_task.py --dry，返回 (cmd 列表, 原始输出)。"""
    p = subprocess.run([PY, "run_task.py", "--config", str(cfg_path), "--dry"],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"\[cmd\]\s*\n(.+)", out)
    if not m:
        return None, out
    return m.group(1).strip().split(), out


def check_one(cfg_path: Path, all_flags: set, verbose=True) -> int:
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ❌ 读不了 {cfg_path.name}: {e}")
        return 1

    cmd, raw = run_dry(cfg_path)
    if cmd is None:
        print(f"  ❌ {cfg_path.name}: --dry 没能吐出 [cmd]，原始输出末尾：")
        for ln in raw.strip().splitlines()[-6:]:
            print("      " + ln)
        return 1

    flags_in_cmd = {c for c in cmd if c.startswith("--")}
    problems, dead, effective = [], [], []

    # ---- 正向：JSON 的每个键是否真的产生了效果 ----
    for k, v in cfg.items():
        if k in CONTROL_PLANE:
            continue
        if k in COND_FLAG:
            flag, when = COND_FLAG[k]
            if bool(v) == when:
                (effective if flag in flags_in_cmd else dead).append(f"{k}={v} → 应出现 {flag}")
            else:
                effective.append(f"{k}={v}（不产生 flag，符合预期）")
            continue
        fl = KEY_TO_FLAG.get(k, f"--{k}")
        if fl in flags_in_cmd:
            effective.append(f"{k} → {fl}")
        else:
            dead.append(f"{k} → 期望 {fl}，**命令行里没有**")

    # ---- 反向：cmd 里的 flag 必须在 train.py 里存在 ----
    unknown = sorted(f for f in flags_in_cmd if f not in all_flags)

    # ---- 跨字段约束：load_best_model_at_end 要求 save_steps 整除 eval_steps ----
    # 2026-10-11 实际踩到：v15b 写 save_steps=2000 / eval_steps=5000，HF 在构造
    # TrainingArguments 时抛 ValueError，一步都没跑。纯 CPU 就能查，不该花 GPU 钱。
    cross = []
    if load_best_on():
        ss, es = _flag_val(cmd, "--save_steps"), _flag_val(cmd, "--eval_steps")
        if ss and es and ss % es != 0:
            cross.append(f"save_steps={ss} 不是 eval_steps={es} 的整数倍"
                         f"（train.py 开了 load_best_model_at_end，启动即 ValueError）")

    # ---- 提示：JSON 没写、会走 DEFAULTS 的键 ----
    from importlib import util as _u
    spec = _u.spec_from_file_location("rt", ROOT / "run_task.py")
    rt = _u.module_from_spec(spec)
    sys.modules["rt"] = rt
    spec.loader.exec_module(rt)
    absent = [k for k in rt.DEFAULTS if k not in cfg and k not in CONTROL_PLANE]

    if verbose:
        print(f"\n=== {cfg_path.name} ===")
        print(f"  生效 {len(effective)} 项")
        for e in effective:
            print(f"    ✅ {e}")
        if absent:
            print(f"  未写、走默认值的 {len(absent)} 项（不报错，但值可能出乎意料）：")
            print("    " + ", ".join(f"{k}={rt.DEFAULTS[k]!r}" for k in absent))
        if dead:
            print(f"  ❌ **死字段 {len(dead)} 项** —— 写了但没生效：")
            for d in dead:
                print(f"      {d}")
        if unknown:
            print(f"  ❌ **cmd 里有 {len(unknown)} 个 train.py 不认识的 flag**（会直接崩）：")
            print("      " + ", ".join(unknown))
        if cross:
            print(f"  ❌ **跨字段冲突 {len(cross)} 项**：")
            for c in cross:
                print(f"      {c}")
    return len(dead) + len(unknown) + len(cross)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--all", action="store_true", help="扫 tasks/*.json（排除 _smoke_*）")
    a = ap.parse_args()

    files = [Path(c) for c in a.configs]
    if a.all:
        files = sorted(p for p in (ROOT / "tasks").glob("*.json")
                       if not p.name.startswith("_"))
    if not files:
        print("没给配置。用 --all 或显式列出。")
        return 1

    all_flags = train_py_flags()
    print(f"retnet_hf/train.py 声明了 {len(all_flags)} 个 flag")
    total = 0
    for f in files:
        total += check_one(f, all_flags)

    print("\n" + "=" * 70)
    if total == 0:
        print("✅ 全部配置无死字段、无未知 flag。")
    else:
        print(f"❌ 共发现 {total} 处问题（死字段 + 未知 flag）。")
        print("   死字段：JSON 里写了但 run_task.py 没拼进命令行。")
        print("   未知 flag：run_task.py 拼了但 train.py 不认，会直接崩。")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
