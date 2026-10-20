"""E15/E17 的静态守卫：训练配置里两处默认值不能被改回去。

E15（不能拿被污染的 eval_loss 选最佳）与 E17（save_total_limit 默认 4）都落在
`TrainingArguments` 的构造参数上，而构造过程埋在 `main()` 里，直接跑起来很重。
这里改用源码级断言：一旦有人把 `load_best_model_at_end` 改回 True、
或把 `save_total_limit` 的默认改回 3，测试立刻变红。

静态断言比行为测试弱，但它守的是一个**字面量契约**，对这类回归足够；
真正需要行为验证的部分（val 隔离、EE6 的 decay 过滤）另有专门的测试。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = (ROOT / "retnet_hf" / "train.py").read_text(encoding="utf-8")


def test_best_model_selection_is_disabled():
    """E15：load_best_model_at_end 必须是 False，且源码里不得再出现 True。"""
    assert "load_best_model_at_end=False" in TRAIN
    assert not re.search(r"load_best_model_at_end\s*=\s*True", TRAIN), \
        "load_best_model_at_end 又被打开了：eval 集被污染时会选错模型"


def test_save_total_limit_default_is_four():
    """E17：默认值必须与任务模板一致（4），不能再是 3。"""
    m = re.search(r'"--save_total_limit",\s*type=int,\s*default=(\d+)', TRAIN)
    assert m, "找不到 --save_total_limit 的定义"
    assert m.group(1) == "4", f"默认值又变回 {m.group(1)}，与模板不一致"


def test_no_decay_1d_default_is_on():
    """E06：no_decay_1d 默认必须是开启，且保留显式的反向开关。"""
    assert re.search(r'"--no_decay_1d",\s*action="store_true",\s*default=True', TRAIN)
    assert '"--decay_1d"' in TRAIN, "缺少显式关闭的开关，无法复现旧行为"
