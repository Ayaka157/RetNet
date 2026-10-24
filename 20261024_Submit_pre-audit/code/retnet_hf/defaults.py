"""跨入口共享的默认值（**唯一来源**）。

E17 的根因不是“某个数字写错了”，而是同一个字段在 ``train.py`` 与
``run_task.py`` 里各有一套默认值：模板写 4、训练脚本写 3，谁生效取决于
调用路径，测试也只查了其中一处。把默认值放到一个模块里，两处 import
同一个常量，改一处就是真的改一处。
"""
from __future__ import annotations

#: 保留最近 N 个 checkpoint-*（docs/00 E17）。
DEFAULT_SAVE_TOTAL_LIMIT = 4

#: 1 维参数（norm scale / 门控 / msr.gamma）是否免 weight decay（docs/00 E06）。
DEFAULT_NO_DECAY_1D = True
