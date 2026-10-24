"""pytest 配置。

`tests/test_plan_gc.py` 与 `tests/test_plan_pipeline.py` 是**脚本式**测试：
模块顶层直接跑断言、结尾 `sys.exit(0/1)`。pytest 在收集阶段会 import 模块，
一 import 就会执行整个脚本并调用 `sys.exit`，于是报
`mainloop: caught unexpected SystemExit!`（2026-10-20 在本机与 4070 上都复现）。

它们本来就不是给 pytest 跑的，用 `collect_ignore_glob` 让 pytest 直接跳过导入，
改为手动执行：

    python tests/test_plan_gc.py
    python tests/test_plan_pipeline.py

不要把它们改写成 pytest 用例却不加 `__main__` 保护——那会让 `pytest tests`
每次都在收集阶段崩掉，而不是在断言阶段失败。
"""

collect_ignore_glob = [
    "test_plan_gc.py",
    "test_plan_pipeline.py",
]
