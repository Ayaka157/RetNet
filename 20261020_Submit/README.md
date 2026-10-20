# 20261020_Submit

一次可审核的快照：**重写后的训练代码 + 三份文档 + 独立复核脚本**。
权威来源是仓库 `E:\Workspaces\RetNet` 的 `main` 分支；本目录是它的时间点副本，
不参与训练、不被训练脚本引用。

## 目录

```
20261020_Submit/
  verify.py                    独立校验脚本（不联网、不改文件）
  MANIFEST.sha256              本目录每个文件的 sha256
  docs/
    00_错误总表.md              E01–E23 + N01–N04，逐行定位
    00F1_修复对照.md            每个编号：修复后的完整代码段 + >=500 字说明
    01_训练代码逐函数解释.md     83 个函数，代码段由 AST 抽取，逐条标注 00 编号
    02_续训恢复计划_220元.md     ¥220 预算下的恢复计划（附父 Agent 审核注记）
  code/
    retnet_hf/*.py             训练/模型/数据/内核完整包
    run_task.py                任务入口
    scripts/                   生长、建包、重初始化、评测、生成器
    tests/*.py                 19 个测试文件（41 个 pytest 用例 + 2 个脚本式测试）
  configs/tasks/               上一轮实际配置，仅作对照
```

## 修复状态（27 项）

| 类别 | 编号 | 状态 |
|---|---|---|
| 代码修复 | E01–E04、E06、E08–E14、E16–E22、N01、N04 | 已落地，各有测试或校验 |
| 默认值/注释 | E05（默认值 + 测试）、E07（只改注释）、E15（关闭自动选优） | 已落地 |
| 文档订正 | N02（gamma 口径）、N03（层宽 1024） | 无代码改动，理由写在 00F1 |

E23 是流程缺口：原来 11 个测试没有一个检查梯度，现在补了 6 个“改回旧行为就会红”的测试。

## 怎么复核

```powershell
cd 20261020_Submit
python verify.py
```

六项检查与实际输出：

| 检查 | 结果 |
|---|---|
| A 01 代码块 vs `code/` 源文件逐字节 | 83/83 一致 |
| B 01 函数覆盖（AST 枚举） | 83/83，0 遗漏 |
| C 01 解释 ≥100 字、无占位词 | 通过 |
| D 01 引用的 00 编号 | 19 个引用，0 悬空 |
| E 00F1 代码块逐字节 | 34 个代码块，0 不一致 |
| F 00F1 说明 ≥500 字、00 全覆盖 | 27/27 条目，通过 |

退出码 0。

## 测试

```powershell
python -m pytest tests -q          # 41 passed
python tests/test_plan_gc.py       # 全部通过
python tests/test_plan_pipeline.py # 全部通过
```

本机（RTX 3070 / Python 3.13 / torch 2.10）与远端笔记本
（192.168.31.23，C:\Ayaka\RetNet，4070 Laptop / Python 3.11 / torch 2.11）均已跑通。

## 注意

- `docs/01` 与 `docs/00F1` 的代码段都由脚本生成（`code/scripts/_gen_code_doc.py`、
  `_gen_fix_doc.py`），不是手抄；直接改文档里的代码块会被 `verify.py` 立刻发现。
- `configs/tasks/` 是**上一轮**的 v21 配置，保留是为了对照 E06/E20 的历史；
  新方案的训练配置不在本目录，等 02 计划里那批 `tasks/v22_*.json` 定稿后单独提交。
- 远端 clone 里没有 checkpoint（被 gitignore），所以 `reinit_gamma.py` 的产物体积验证
  只在本地做过；它的 pure 函数（`build_gamma`）由代码审查与 dry-run 覆盖。
