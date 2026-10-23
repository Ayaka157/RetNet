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
    03_模型结构变更.md          从原始 RetNet 到当前实现的逐项变更（GB/T 7714 引用）
    04_模型数学与损失.md        完整前向公式与损失，逐式标注代码函数
  code/
    retnet_hf/*.py             训练/模型/数据/计划/内核完整包
    run_task.py                任务入口
    scripts/                   生长、建包、重初始化、评测、生成器
    tests/*.py                 23 个测试文件（56 个 pytest 用例 + 2 个脚本式测试）
  configs/tasks/               上一轮实际配置，仅作对照
```

## 修复状态（27 项）

| 类别 | 编号 | 状态 |
|---|---|---|
| 代码修复 | E01–E04、E06、E08–E14、E16–E22、N01、N04 | 已落地，各有测试或校验 |
| 默认值/注释 | E05（默认值 + 测试）、E07（只改注释）、E15（关闭自动选优） | 已落地 |
| 文档订正 | N02（gamma 口径）、N03（层宽 1024） | 无代码改动，理由写在 00F1 |

对应测试：`test_growth_trainable`（E01–E04）、`test_no_decay_1d`（E06）、`test_live_params`（E08）、
`test_build_scripts`（E09/E10/E11）、`test_eos_present`（E12）、`test_plan_gc`（E13）、
`test_val_disjoint`（E14）、`test_train_config_guards`（E15/E17）、`test_parallel_chunkwise_equiv`（E16）、
`test_plan_reschedule`（E18）、`test_regressions_extra`（E19/E20/E21/E22）、`test_growth_trainable` 的反例（E23）。

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
python -m pytest tests -q          # 56 passed
python tests/test_plan_gc.py       # 全部通过
python tests/test_plan_pipeline.py # 全部通过
```

本机（RTX 3070 / Python 3.13 / torch 2.10）与远端笔记本
（192.168.31.23，C:\Ayaka\RetNet，4070 Laptop / Python 3.11 / torch 2.11）均已跑通。

注：`test_build_scripts.py` 需要 `data_cache/chat_pack_v13/bpe_tokenizer.json`，
远端 clone 没有该文件（被 gitignore），该文件里的 4 个用例会**显式 skip**，不是假通过。

## 注意

- `docs/01` 与 `docs/00F1` 的代码段都由脚本生成（`code/scripts/_gen_code_doc.py`、
  `_gen_fix_doc.py`），不是手抄；直接改文档里的代码块会被 `verify.py` 立刻发现。
- `configs/tasks/` 是**上一轮**的 v21 配置，保留是为了对照 E06/E20 的历史；
  新方案的训练配置不在本目录，等 02 计划里那批 `tasks/v22_*.json` 定稿后单独提交。
- 远端 clone 里没有 checkpoint（被 gitignore），所以 `reinit_gamma.py` 的产物体积验证
  只在本地做过；它的 pure 函数（`build_gamma`）由代码审查与 dry-run 覆盖。

## 文内引用

四份文档都按 `Papers/` 的文件前缀加了文内引用，编号 01–17 与论文一一对应：
01 RetNet、02 xPos、03 GLA、04 Transformers are RNNs、05 GroupNorm、06 SwiGLU、
07 Net2Net、08 Network Morphism、09 bert2BERT、10 StackBERT、11 MSLT、12 LiGO、
13 ReZero、14 Fixup、15 Chinchilla、16 FineWeb、17 Skywork。
03 单独采用 GB/T 7714—2015 顺序编码制，文中标注页码与章节；01 与 00F1 的引用由生成器写入（`code/scripts/_refs.py` 是唯一编号源），
00 与 02 由 `code/scripts/_add_citations.py` 按小节插入；每篇末尾附参考文献表。
