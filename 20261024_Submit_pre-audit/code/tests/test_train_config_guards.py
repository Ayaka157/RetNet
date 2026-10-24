"""E06 / E15 / E17 / N04 的配置守卫。

旧版本这里只有**源码正则**：查 `train.py` 里有没有 `default=4`、有没有
`load_best_model_at_end=False`。它守的是字面量，不是行为 —— 所以
`run_task.py` 里仍然把 `save_total_limit` 兜底成 3（E17 的原始缺陷原样存在），
`no_decay_1d: false` 仍然被静默忽略（E06 的反向开关形同虚设），测试全绿。

现在改成：
* 默认值来自 `retnet_hf/defaults.py` **同一个常量**，两处 import；
* `build_parser()` 是纯函数，直接断言 `--decay_1d -> no_decay_1d=False`；
* `run_task` 的两个分支（plan / 非 plan）都断言 argv；
* 只保留一条源码断言：`main()` 真的把解析出来的值交给了 trainer
  （删掉那一行就变红 —— 这是旧测试完全覆盖不到的地方）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.defaults import DEFAULT_NO_DECAY_1D, DEFAULT_SAVE_TOTAL_LIMIT  # noqa: E402
from retnet_hf.train import build_parser  # noqa: E402

TRAIN = (ROOT / "retnet_hf" / "train.py").read_text(encoding="utf-8")

BASE_ARGV = ["--model_dir", "m", "--output_dir", "o", "--data_dir", "d"]


def _cfg(**kw):
    """build_cmd(_full) 需要的最小完整配置（不读盘）。"""
    cfg = {
        "name": "t", "model_dir": "m", "data_dir": "d", "output_dir": "o",
        "precision": "fp32", "mode": "chunkwise", "chunk_size": 64,
        "batch_size": 2, "max_steps": 10, "lr": 1e-4, "min_lr": 1e-5,
        "warmup": 1, "weight_decay": 0.1, "max_grad_norm": 1.0,
        "save_steps": 100, "eval_steps": 100, "seed": 1,
        "grad_checkpoint": False, "compile": False, "resume": False,
        "nproc": 1,
    }
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------- E15
def test_best_model_selection_is_disabled():
    """E15：load_best_model_at_end 必须是 False，且源码里不得再出现 True。"""
    assert "load_best_model_at_end=False" in TRAIN
    assert not re.search(r"load_best_model_at_end\s*=\s*True", TRAIN), \
        "load_best_model_at_end 又被打开了：eval 集被污染时会选错模型"


# ---------------------------------------------------------------- E17
def test_save_total_limit_is_one_shared_constant():
    assert DEFAULT_SAVE_TOTAL_LIMIT == 4
    args = build_parser().parse_args(BASE_ARGV)
    assert args.save_total_limit == DEFAULT_SAVE_TOTAL_LIMIT


def test_run_task_passes_save_total_limit_in_both_branches(tmp_path: Path):
    """E17：plan 与非 plan 两个分支都必须传同一个值 —— 旧代码只在非 plan 分支传 3。"""
    import run_task

    plan = tmp_path / "plan.json"
    plan.write_text('{"stages":[{"name":"a","data_dir":".","steps":10}]}',
                    encoding="utf-8")
    for extra in ({"plan": str(plan)}, {}):
        for want in (DEFAULT_SAVE_TOTAL_LIMIT, 6):
            cfg = _cfg(save_total_limit=want, **extra)
            cmd = run_task.build_cmd(cfg)
            assert "--save_total_limit" in cmd, f"分支 {extra} 没传 save_total_limit"
            got = cmd[cmd.index("--save_total_limit") + 1]
            assert got == str(want), f"分支 {extra} 传了 {got}，配置是 {want}"


# ---------------------------------------------------------------- E06
def test_decay_1d_flag_parses_to_false():
    p = build_parser()
    assert p.parse_args(BASE_ARGV).no_decay_1d is True
    assert p.parse_args(BASE_ARGV + ["--no_decay_1d"]).no_decay_1d is True
    assert p.parse_args(BASE_ARGV + ["--decay_1d"]).no_decay_1d is False


def test_run_task_emits_both_directions():
    """E06：`no_decay_1d: false` 以前被静默忽略，因为只在 truthy 时才拼 flag。"""
    import run_task

    assert DEFAULT_NO_DECAY_1D is True
    on = run_task.build_cmd_full(_cfg(no_decay_1d=True))
    off = run_task.build_cmd_full(_cfg(no_decay_1d=False))
    assert "--no_decay_1d" in on and "--decay_1d" not in on
    assert "--decay_1d" in off and "--no_decay_1d" not in off
    # 配置里没写时跟随默认值（与 train.py 的默认一致）
    default = run_task.build_cmd_full(_cfg())
    assert ("--no_decay_1d" in default) is DEFAULT_NO_DECAY_1D


def test_trainer_receives_parsed_flag():
    """接线：main() 必须把 args.no_decay_1d 交给 trainer。

    旧测试用 `CompileSafeTrainer.__new__` 手工赋值 `_no_decay_1d`，所以把
    `trainer._no_decay_1d = ...` 这一行删掉也全绿（E06 的接线零覆盖）。
    """
    assert re.search(r"trainer\._no_decay_1d\s*=\s*bool\(getattr\(args,\s*"
                     r"\"no_decay_1d\"", TRAIN), \
        "main() 没有把 --no_decay_1d 的解析结果交给 trainer"


def test_eval_batches_defaults_to_unlimited_and_is_passed_through():
    """P2-1：默认 0（不限）保持历史 eval_loss 口径；显式配置才截断。"""
    from retnet_hf.train import build_parser, cap_eval_rows

    assert build_parser().parse_args(BASE_ARGV).eval_batches == 0
    assert cap_eval_rows(100, 0, 32) == 100
    assert cap_eval_rows(100, 2, 32) == 64
    import run_task
    assert run_task.build_cmd_full(_cfg())[
        run_task.build_cmd_full(_cfg()).index("--eval_batches") + 1] == "0"
    cmd = run_task.build_cmd_full(_cfg(eval_batches=5))
    assert cmd[cmd.index("--eval_batches") + 1] == "5"


# ---------------------------------------------------------------- N04
def test_eval_uses_chunkwise():
    """N04：评测必须用 chunkwise（训练路径），不能再强制 parallel。"""
    src = (ROOT / "scripts" / "eval_holdout.py").read_text(encoding="utf-8")
    assert re.search(r'model\.config\.mode\s*=\s*"chunkwise"', src), \
        "eval_holdout 没有切到 chunkwise：评测测的会是与训练不同的函数"
    assert not re.search(r'model\.config\.mode\s*=\s*"parallel"', src)
    assert '"--chunk-size"' in src, "缺少 --chunk-size，无法与训练块大小对齐"


def test_eval_has_both_probes():
    """P1-6：docs/02 的门要求 --by-position 与 --ctx-ablation 真的存在。"""
    src = (ROOT / "scripts" / "eval_holdout.py").read_text(encoding="utf-8")
    assert '"--by-position"' in src
    assert '"--ctx-ablation"' in src
