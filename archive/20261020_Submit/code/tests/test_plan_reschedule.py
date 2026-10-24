"""E18 的守卫：延长计划把 LR 抬上去时，默认必须拒绝。

判定被抽成 `retnet_hf.plan.should_accept_reschedule`，所以这里可以直接构造
两个 Plan 测边界，不需要启动 Trainer。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.plan import parse_plan, should_accept_reschedule  # noqa: E402


def _plan(steps: int, allow: bool = False):
    raw = {
        "plan_version": 1, "seq_len": 8,
        "stages": [{"name": "a", "data_dir": ".", "steps": steps}],
        "lr": {"peak": 1e-4, "min": 1e-5, "warmup": 0},
        "save_steps": 10, "eval_steps": 10,
        "allow_reschedule": allow,
    }
    return parse_plan(raw, Path("<test>"))


def test_extending_plan_is_rejected():
    old, new = _plan(100), _plan(200)
    assert old.total_steps == 100 and new.total_steps == 200
    accept, lr_old, lr_new = should_accept_reschedule(old, new, 50)
    assert lr_new > lr_old * 1.02, "测试前提不成立：延长后 LR 并没有上跳"
    assert accept is False, "延长计划把 LR 抬高时，默认必须拒绝（E18 回归）"


def test_extending_plan_allowed_with_flag():
    old, new = _plan(100), _plan(200, allow=True)
    accept, _lr_old, _lr_new = should_accept_reschedule(old, new, 50)
    assert accept is True


def test_shrinking_plan_is_allowed():
    """缩短预算是止损操作，LR 只会变小，必须放行。"""
    old, new = _plan(100), _plan(50)
    accept, lr_old, lr_new = should_accept_reschedule(old, new, 25)
    assert lr_new <= lr_old * 1.02
    assert accept is True


def test_same_total_steps_is_allowed():
    old, new = _plan(100), _plan(100)
    accept, _a, _b = should_accept_reschedule(old, new, 30)
    assert accept is True
