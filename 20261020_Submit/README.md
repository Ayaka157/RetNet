# 20261020_Submit

这是一次可审核的快照：**重写后的训练代码 + 两份文档 + 复核脚本**。
权威来源是仓库 `E:\Workspaces\RetNet` 的 `main` 分支；本目录是它的一个时间点副本，
用于你逐项核对，不参与训练、不被训练脚本引用。

## 目录

```
20261020_Submit/
  verify.py                 独立校验脚本（不联网、不改文件）
  MANIFEST.sha256           本目录每个文件的 sha256
  docs/
    00_错误总表.md           E01–E23，逐行定位到文件:行
    01_训练代码逐函数解释.md  78 个函数，代码段由 AST 抽取，逐条标注 00 编号
  code/
    retnet_hf/*.py          训练/模型/数据/内核完整包
    run_task.py             任务入口
    scripts/grow_500m.py    生长脚本（E01–E04 的修复点）
    scripts/build_flat_corpus.py  （E09 的修复点）
    scripts/_gen_code_doc.py     生成 01 的脚本
    tests/test_plan_gc.py   新增 4b：缺 meta.json 时 GC 必须拒绝
  configs/tasks/            上一轮实际使用的两份配置，仅作对照，不是新方案的配置
```

## 已修（10 项，对应 00）

| 编号 | 位置 | 改动 |
|---|---|---|
| E01 | `scripts/grow_500m.py` `fill()` | 新列置零、新行随机（原来两侧全零） |
| E02 | `scripts/grow_500m.py` `grow_norm()` | 中间层 `gate=1.0`、`final_norm gate=0.0`（原来全零） |
| E03 | `scripts/grow_500m.py` embedding | `[grow_old_dim:]` 重新随机（原来全零） |
| E04 | `scripts/grow_500m.py` | 新增 5 步反向的可训性校验，零梯度拒绝落盘 |
| E05 | `retnet_hf/modeling_retnet.py` | `GrowthLayerNorm` 默认 `gamma_new=1.0`，不再被覆写为 0 |
| E06 | `retnet_hf/train.py` | `--no_decay_1d` 默认开启，新增 `--decay_1d` 反向开关 |
| E07 | `retnet_hf/train.py` | 订正"梯度为 0 也算有参数"的错误注释 |
| E08 | `retnet_hf/train.py` `_preflight` | 活参数 <95% 或 gamma≤0 拒绝开训 |
| E09 | `scripts/build_flat_corpus.py` | `--add-eos` 默认 True |
| E13 | `retnet_hf/data.py` `gc_stage` | 缺 `meta.json` / `rebuild_cmd` 时拒绝删除 |
| E14 | `retnet_hf/train.py` | `val_dir` 与训练阶段同目录时拒绝开训 |
| E16 | `retnet_hf/modeling_retnet.py` | 删除 parallel 路径独有的两次归一化 |
| E17 | `retnet_hf/train.py` | `save_total_limit` 默认 3 → 4 |
| E20 | `run_task.py` | plan 分支补传 `eval_batch_mult` |

（E05 是结构性防御，不是独立修复；合计 10 个编号项 + 1 个防御项。）

## 未修（13 项，00 里逐条列着）

E10、E11、E12、E15、E18、E19、E21、E22、E23，以及 00 §4 里尚未落地的
"梯度覆盖断言""γ 衰减断言""最佳 checkpoint 按 held-out 保留"等。

**这些没修完就开机，等于把这轮预算再烧一遍。** `01` 文档开头的状态表逐条标了"未修"。

## 怎么复核

```powershell
cd 20261020_Submit
python verify.py
```

它做四项检查：

- **A**：文档里 78 个代码块，与 `code/` 下源文件同区间**逐字节**比较；
- **B**：用 AST 枚举代码里全部函数/方法，文档必须全覆盖；
- **C**：每条解释 ≥100 字，且不含"同上/简略"等占位词；
- **D**：文档引用的 00 编号必须真实存在。

当前结果：`[A] 78 个章节`、`[B] 78/78`、`[C] 全部 ≥100 字`、`[D] 悬空 0`，退出码 0。

## 注意

- `docs/01` 的正文由 `code/scripts/_gen_code_doc.py` 生成，代码段不是手抄；
  若直接改 `01` 的代码块而不改源码，`verify.py` 会立刻报不一致。
- `configs/tasks/` 是**上一轮**的配置（v21），保留是为了对照 E06/E20 的历史；
  新方案的训练配置不在本目录，等 ¥220 恢复计划定稿后单独提交。
