"""pytest 配置。

## 一、脚本式测试不交给 pytest

`tests/test_plan_gc.py` 与 `tests/test_plan_pipeline.py` 是**脚本式**测试：
模块顶层直接跑断言、结尾 `sys.exit(0/1)`。pytest 在收集阶段会 import 模块，
一 import 就会执行整个脚本并调用 `sys.exit`，于是报
`mainloop: caught unexpected SystemExit!`。

它们本来就不是给 pytest 跑的，用 `collect_ignore_glob` 让 pytest 直接跳过导入，
改为手动执行：

    python tests/test_plan_gc.py
    python tests/test_plan_pipeline.py

不要把它们改写成 pytest 用例却不加 `__main__` 保护——那会让 `pytest tests`
每次都在收集阶段崩掉，而不是在断言阶段失败。

## 二、五个“测试父仓库、不是测试本快照”的文件

下面五个文件导入的模块**不在本快照里**（它们属于 `E:\\Workspaces\\RetNet`
主仓库的 benchmark / 生成脚本 / 旧训练循环）：

| 文件 | 需要的模块 | 归属 |
|---|---|---|
| `test_checkers.py` | `retnet`（手写训练循环与 checkpoint 检查器） | 主仓库 `retnet/` |
| `test_model.py` | `retnet.model` | 主仓库 `retnet/` |
| `test_bench_equivalence.py` | `bench` | 主仓库 `bench/` |
| `test_play_v2_prompt.py` | `play_v2` | 主仓库 `play_v2.py` |
| `test_runner_lockstep.py` | `run_task` 的旧版内部函数 + `bench` | 主仓库 |

本快照只提交 `retnet_hf/` 这条训练路径（README 的目录表已写明）。让它们参与
收集的后果是 5 个 collection error —— `pytest tests` 退出码 2，看起来像“测试
挂了”，实际上一个用例都没跑。用 `collect_ignore` 显式排除，并在 README 里
写清真实数字与原因；不要用 `--ignore` 藏在命令行里，那对下一个人不可见。
"""

collect_ignore_glob = [
    # 脚本式测试：顶层就 sys.exit，pytest 收集即崩（手动执行，见上）
    "test_plan_gc.py",
    "test_plan_pipeline.py",
    # 依赖主仓库模块，不在本快照内（见上表）
    "test_checkers.py",
    "test_model.py",
    "test_bench_equivalence.py",
    "test_play_v2_prompt.py",
    "test_runner_lockstep.py",
]
