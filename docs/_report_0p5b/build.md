# RetNet-0.5B 报告 · 构建记录（task-6 / T2 改造流水线）

> 写入者：latex-builder ｜ 最后更新：2026-10-13
> 范围：`docs/latex/**` 与本文。**不包含** `docs/26_RetNet-0.5B_技术报告.md`（report-writer 的）。
> 状态：**task-6 完成**；task-7（正式构建）按 Lead 指示**等「中文定稿」后执行**，
> 期间 `docs/26_...md` 仍在修订（本轮期间它从 631 行涨到 645 行，§3.4 的分类也改过）。

---

## 一、本轮做了什么

把 `docs/latex/` 从「只支持 272M 双语单文件」改造成**多报告配置**：每个报告由一条
`mktex.Report` 记录描述，加报告只需加记录 + 手写片段，不动构建逻辑。

两种拆分模式：

| 模式 | 含义 | 用在 |
|---|---|---|
| `bilingual` | 取 `### N.N 中文` / `### N.N English` 成对标记的一半 | 272M |
| `mono` | Markdown 正文**就是**中文版正文，不拆；英文版全文来自手写补充稿 | 0.5B、selftest |

0.5B 的摘要处理：`## 摘要` 由构建器取出来排在目录之前（`\section*{摘要}` +
`\addcontentsline`），因此摘要与 Markdown 不会各自漂移；找不到 `## 摘要` 时构建器会在
stderr 打印 `WARNING`。

## 二、新增 / 改动文件

| 文件 | 变化 |
|---|---|
| `docs/latex/mktex.py` | **改**：新增 `Report` 记录 + `REPORTS`（272m / 0p5b / selftest）、`build_report` / `build_zh` / `build_en` 通用构建、`--report` / `--list` 三个 CLI 开关；`infer_caption` / `caption_supplement_tables` / `inject_figures` / `en_heading` / `merge_supplement` 参数化；`add_breaks` 增加「无分隔符长串」断行（按 `Report.compact_tables` 开关）；0.5B 的表题映射与插图锚点 |
| `docs/latex/makefigs.py` | **改**：新增 `load_05b_legs()` 与三张 0.5B 图（loss / lr / step），全部读自三个 `trainer_state.json` 与报告 §3.3 的实测分桶 |
| `docs/latex/verify.py` | **改**：272M 的 32 项检查**逐字保留**，另加对所有已构建报告的通用检查（TOC 无语言标记、编号合规、中文版有中文正文、英文版代码之外无中文）；未编译（无 `.toc`）的报告跳过 |
| `docs/latex/langcheck.py` | **新**：从 PDF 文本层实证语言互斥。中文版把参考文献与公式行**显式豁免并计数**，其余英文段落必须能追溯到代码；英文版含中文的行只能来自代码。页数 ≤ 8 会告警 |
| `docs/latex/reports/0p5b/zh_preamble.tex`、`zh_titles.tex` | **新**：0.5B 中文版文档类/字体/元信息 + 标题页（摘要与目录由构建器插入） |
| `docs/latex/reports/0p5b/en_preamble.tex`、`en_titles.tex` | **新**：0.5B 英文版文档类（Times + CJK 回退）、英文标题页与摘要 |
| `docs/latex/reports/0p5b/en_supplement.tex` | **新**：0.5B 英文正文，**完整翻译**（约 6,900 词；§1–§7 与附录 A–C 和中文终稿逐节一一对应，26 张表 + 3 张图）。数值逐节取自中文终稿；表题由 `SUPPLEMENT_TABLE_HEADERS_05B` 附着，手写表格里的长代码串由 `break_supplement_code` 加断行 |
| `docs/latex/reports/0p5b/{zh,en}_preamble.tex`、`{zh,en}_titles.tex` | **改**：封面标题同步为终稿标题（中：`RetNet-0.5B：逐段单卡训练到线上的 5.04 亿参数保留网络——宽度生长为何未能解锁`；英：`RetNet-0.5B: Training a 504M-Parameter Retentive Network to Production, and Why Width Growth Never Unlocked`），`pdftitle` 同步 |
| `docs/latex/selftest/**`（6 个文件） | **新**：流水线自测夹具——占位 md + 中英 preamble/标题 + 英文补充稿。`selftest` 不参与默认构建 |
| `docs/latex/figures/fig-0p5b-loss.pdf`、`fig-0p5b-lr.pdf`、`fig-0p5b-step.pdf` | **新**：由 `makefigs.py` 生成（矢量） |
| `docs/latex/README.md` | **改**：新增 0.5B、多报告配置、自测夹具、校验命令、新图数据源、已知取舍（`compact_tables`、列表渲染） |
| `docs/latex/.regress/` | **新**：272M `.tex` 哈希基线（改造前/后）、表题审计脚本与结果、overfull 审计脚本 |

产物命名：`docs/latex/zh/RetNet-0.5B-技术报告.pdf`、`docs/latex/en/RetNet-0.5B-Technical-Report.pdf`。

## 三、构建命令

```powershell
$env:PATH = "C:\Program Files\MiKTeX\miktex\bin\x64;$env:PATH"
cd E:\Workspaces\RetNet\docs\latex

python makefigs.py                  # 1) 插图（先清空 figures/）
python mktex.py                     # 2) 全部可用报告：272m（+ 0.5B，若源文件在）
python mktex.py --list              #    查看配置与源文件是否就位
python mktex.py --report 0p5b       #    只构建一个报告（源文件缺失即报错退出）

# 3) 每个 .tex 编译两遍（第二遍补齐目录/交叉引用）
cd zh; xelatex -interaction=nonstopmode "RetNet-272M-技术报告.tex"; xelatex -interaction=nonstopmode "RetNet-272M-技术报告.tex"; cd ..
cd en; xelatex -interaction=nonstopmode "RetNet-272M-Technical-Report.tex"; xelatex -interaction=nonstopmode "RetNet-272M-Technical-Report.tex"; cd ..
cd zh; xelatex -interaction=nonstopmode "RetNet-0.5B-技术报告.tex"; xelatex -interaction=nonstopmode "RetNet-0.5B-技术报告.tex"; cd ..
cd en; xelatex -interaction=nonstopmode "RetNet-0.5B-Technical-Report.tex"; xelatex -interaction=nonstopmode "RetNet-0.5B-Technical-Report.tex"; cd ..

# 4) 校验
python report.py        # 汇总 zh/、en/ 下 .log：页数 / error / undefined / Overfull
python verify.py        # 结构 + 语言互斥（272M 原 32 项 + 各报告通用项）
python langcheck.py     # PDF 文本抽取的语言实证（可审计的豁免计数）
python langcheck.py selftest
python pdfcheck.py      # 272M 两本 PDF 的关键内容点
```

自测夹具（不依赖真实报告）：

```powershell
python mktex.py --report selftest
cd selftest\zh; xelatex -interaction=nonstopmode "SelfTest-技术报告.tex"; xelatex -interaction=nonstopmode "SelfTest-技术报告.tex"; cd ..\..
cd selftest\en; xelatex -interaction=nonstopmode "SelfTest-Technical-Report.tex"; xelatex -interaction=nonstopmode "SelfTest-Technical-Report.tex"; cd ..\..
```

## 四、自测证据

### 4.1 流水线自测夹具（`selftest`，占位内容）

| 检查 | 中文版 | 英文版 |
|---|---|---|
| 页数（两遍编译） | 5 | 7 |
| `.log` error / undefined ref | 0 / 0 | 0 / 0 |
| Overfull hbox | 0 | 0 |
| `langcheck`（PDF 文本层） | PASS（417→461 CJK 字，英文段落候选 0） | PASS（**CJK 字符 0 个**） |
| `verify.py` 通用项 | 6/6 PASS | 6/6 PASS |

一条链路真正被验证：`selftest/placeholder-0p5b.md`（中文单语）→ 中文版 PDF；
`selftest/en/_supplement.tex`（手写补充稿）→ 英文版 PDF；两者都含表格题注、插图注入与
`\startappendix` 附录计数。

### 4.2 真实 0.5B 源文件试构建（**临时，产物已删除**）

Lead 指示在「中文定稿」前不做正式构建。为提前暴露真实正文（24 张表、公式、代码块）的问题，
我用**当时仍在修订的** `docs/26_...md` 跑过试构建，确认可编译后**删除了产物**
（`zh/en` 下不留任何 `RetNet-0.5B-*`），以免与 task-7 的正式产物混淆。
最新一轮（md 657 行、24 张表、封面已换新标题）：

| 检查 | 中文版 | 英文版 |
|---|---|---|
| 页数（两遍编译后稳定） | 27 | 18 |
| `.log` error / undefined ref | 0 / 0 | 0 / 0 |
| 字体形状警告 | 2 条（`FangSong` 无粗体/斜体，用正体替代；同 272M 的已知项） | 0 |
| Overfull hbox | 32 条，**全部 ≤ 4.17 pt**（修前最差 166 pt / 3 条 > 20 pt） | 1 条 2.74 pt |
| Markdown 表题 | 24/24 张有「表 N.」题注（`.regress/precheck.py`） | — |
| 插图注入 | `fig-0p5b-step`（§3.3）、`fig-0p5b-loss` + `fig-0p5b-lr`（§4.4） | 同左 |
| `verify.py`（含 0p5b 通用项） | **69/69 通过** | 同左 |
| `langcheck.py` | **PASS**：12 条英文段落候选中 10 条公式、1 条嵌在中文句子内（§5.6 的英文原样引文）、2 条待查、**0 泄漏**；CJK 11,957 字 | **PASS**：CJK **0** 字符 |

修掉的溢出问题（全部由 0.5B 侧新增的 `compact_tables` 开关解决，272M 逐字节不变）：

1. `sha256` 这类 64 字符无分隔符串在任何窄列里都不可断行（166 pt）→ 每 12 字符给一个断行机会；
2. 行内代码在窄列里同样不可断行（`train_batch_size/logging_steps/...` 95 pt）→ 打开开关时**所有**
   行内代码都加点断行；纯文本里的 `_ % /` 转义后也加点断行（`eval_loss` 10.88 pt → 消失）；
3. 数值列被压到比 `552,330,489` 这类不可断行数字还窄（7.63 pt）→ 把「最长的空白分隔连续段」
   也算作列宽下限；
4. 14 列（step + 13 次评测）的 loss 表在 `\small` 下每列都微溢出 → 列数 ≥ 10 的表改用 `\scriptsize`
   （宽度估算仍按 `\small`，因此留有富余）。

封面标题同步：中文 md 的 H1 不进正文，封面是我手写的 `reports/0p5b/zh_titles.tex`，
已按新标题改为「RetNet-0.5B：消费级单卡上的 5.04 亿参数保留网络 / ——宽度生长为何未能解锁」，
并同步 `zh_preamble.tex` 的 `pdftitle`。封面摘要表的数值也按新稿改为
504,179,896 / 1,656,209,408 token / 552,330,489 token 槽位 / 82 个仿射张量中 78 个为零。

### 4.3 272M 回归（task-6 硬约束，实测）

| 证据 | 结果 |
|---|---|
| `.tex` 逐字节 | `zh/RetNet-272M-技术报告.tex`、`en/RetNet-272M-Technical-Report.tex` 与改造前 SHA256 **完全一致**（`.regress/baseline-272m-tex.sha256` vs `.regress/after-272m-tex.sha256`） |
| 重建 | `python makefigs.py` + `python mktex.py` + 各编译两遍：沿用 README 步骤 |
| 页数 | 中文版 **25**、英文版 **29**（与 README 记录一致） |
| `.log` | 中文版 0 error / 0 undefined / 4 条 ≤4.07 pt 微溢出；英文版 0 error / 0 undefined / 0 Overfull |
| `verify.py` | **57/57 通过（未构建 0.5B 时）/ 69/69（构建后含 0p5b 通用项）**；其中原 272M 的 32 项逐字保留 |
| `langcheck.py` | 两本均 **PASS**：中文版 32 条英文段落候选全部是参考文献（18 行）或公式（14 行）已豁免，待查 0 条；英文版 192 个中文字符分布在 16 行，**全部可追溯到代码**（`lstlisting`/`\code{}`），无泄漏 |
| `pdfcheck.py` | **0 failures** |

> 三个 0.5B 新图不影响 272M：`makefigs.py` 重跑后 7 张旧图字节不变（尺寸逐个相同）。

## 五、已知取舍与遗留

- **`Report.compact_tables` 是逐报告开关，默认关**。原因：它会改变已交付的 272M 产物
  （272M 中文稿里确实有一个无分隔符长串）。0.5B 与 selftest 打开。开关包含四件事：
  无分隔符长串每 12 字符断行、所有行内代码加点断行、纯文本 `_ % /` 转义后加点断行、
  列数 ≥ 10 的表用 `\scriptsize`、以及「最长空白分隔连续段」作为列宽下限。
  这是本轮唯一的「为 0.5B 新增能力而不动 272M」的机制。
- **表题必须登记**：Markdown 表登记在 `TABLE_CAPTION_HINT_05B`（本轮已按真实表头
  逐张对齐，24/24 命中，含 §3.4「类别/缺陷/后果/证据」与 §4.1「组成/token/占比」两张新表），
  手写补充稿表登记在 `SUPPLEMENT_TABLE_HEADERS_05B`。
  中文稿若新增/改表头，需要重跑 `.regress/precheck.py` 复查。
- **插图锚点按真实中文标题选**：`fig-0p5b-step` → 「一步的时间构成」，
  `fig-0p5b-loss`/`fig-0p5b-lr` → 「热重启」（标题现为「4.4 热重启的代价」）。
  标题改动需同步锚点，否则插图静默不注入。
- **热重启差值的口径**：图里标注的是状态文件原始差值（`+0.0764` / `+0.1310`），
  报告表里是四位小数显示值相减（`+0.0765` / `+0.1309`）。因此**图题不重复这两个数**，
  只说「差值见图中标注」。这个 0.0001 级差异建议在 T4 核查时一并确认口径。
- **`logs/bench_v3/decode_speed.json` 属于 272M 的 checkpoint**，不能当作 0.5B 的解码吞吐；
  0.5B 英文稿里对此已写成「待该文件按 0.5B 重跑后引用」。
- **`render_list` 每个顶层列表项一个 list 环境**（272M 一直如此）——为保持 272M 逐字节不变
  本轮**刻意未改**，已写入 README「已知取舍」。
- **`fig-0p5b-step` 的数值来自报告 §3.3 表 6 的实测分桶**（与 272M 的 `fig-kernels` 同一做法：
  正文实测值写死在 `makefigs.py` 并注明来源），不是手绘示意。
- 试构建阶段的 `.regress/md-scan.txt`、`table-captions.txt` 是当时的审计快照，
  中文稿定稿后应视为过期并重跑。

## 六、交付顺序与最终状态

**第一批（已完成）**：`docs/26_...md` 终稿 + **中文 PDF**。

**第二批（已完成）**：英文 PDF（完整翻译）。

**最终构建记录（两本都用中文终稿 74,490 B / 694 行 / 表 1–28 连续 / `待补` = 0）**

| 项 | 中文版 | 英文版 |
|---|---|---|
| 绝对路径 | `E:\Workspaces\RetNet\docs\latex\zh\RetNet-0.5B-技术报告.pdf` | `E:\Workspaces\RetNet\docs\latex\en\RetNet-0.5B-Technical-Report.pdf` |
| 页数 | **28** | **31** |
| `.log` error | 0 | 0 |
| `.log` undefined ref | 0 | 0 |
| `.log` Overfull **vbox** | **0 条** | **0 条** |
| `.log` Overfull **hbox** | 35 条，最大值 **4.16501 pt**（无 >10 pt） | 2 条，最大值 **3.13135 pt**（无 >10 pt） |
| 最后一页内容底边（PyMuPDF） | y1 = **802.84 pt** < 页高 841.89 pt | y1 = **243.36 pt** < 841.89 pt |
| 字体形状警告 | 2 条（`FangSong` 无粗/斜体） | 0 |
| 表 | 正文 27 张，题注用 md 自带「表 2–28」；封面「表 1」手写 | 26 张，题注由 `SUPPLEMENT_TABLE_HEADERS_05B` 附着（附录 C 的跨页长表自带 `\caption`） |
| 评测结果表 | 表 21：7 列（n/配额、覆盖、得分、95% CI、think 泄漏、截断），泄漏/截断为 **1 位小数** | Table 19：同样 7 列、同样显示精度，15 类 + 加权总分逐行对齐（此前泄漏/截断两列误印成「—」，已按 `logs/bench_v3/report.500m_ep3.json` 的 `models.ep3.cats` 补齐） |
| 图 | 3 张（§3.3 步时构成、§4.4 loss、§4.4 lr） | 同左 |
| 章节 | 摘要 + §1–§7 + 附录 A–C | Abstract + §1–§7 + 附录 A–C（一一对应，含 §5 六个子节） |

> **跨页长表（vbox 修复）**：附录 C 的证据索引表原本是一整块不可断的 `tabular`，高度超过一页，
> 于是被整块顶出纸面（`Overfull \vbox 588.13 pt` / `549.29 pt`）。日志只统计 `\hbox` 时看不出它，
> 而 PDF 里文字对象存在、超界部分会被阅读器裁掉——实测两本最后一页内容底边 845.74 / 846.7 pt，
> 超出 841.89 pt 的页高。现在 Markdown 侧由 `render_table` 把 ≥18 行的表渲染成可断的 `longtable`
> （表头与 `\bottomrule` 每页重复），英文补充稿的附录 C 表同样改为 `longtable` 并自带 `\caption`。
> 修复后 vbox 归零，两本各多一页（表格正常跨页），`pdfbounds.py` 四本 PDF 全部 PASS。

| 校验 | 结果 |
|---|---|
| `python verify.py` | **69/69 通过**（272M 原 32 项 + 0p5b 两个语种 + selftest 的通用项） |
| `python langcheck.py` | 4/4 PDF 通过：中文 0.5B PASS（12 条英文候选：10 条公式 + 1 条 §5.6 原样引文 + 1 条实查，0 泄漏）；英文 0.5B PASS（CJK 23 字、5 行，全部可追溯到代码里的中文文件名） |
| `python pdfbounds.py` | **4/4 PASS**：任一页内容底边/右边都在纸张范围内（最坏 +0.00 pt）——这是 `\vbox` 溢出的独立复核 |
| 272M 回归 | 两个 `.tex` 与基线逐字节一致；重新编译后 25/29 页、0 error / 0 undefined、`verify.py` 通过、`pdfcheck.py` 0 failures（共享 preamble 新增 `longtable` 未影响其版面） |
| 关键值在 PDF 文本层 | 中文：`59.61` / `72.94` / `0.1037` / `0.15217` / `276` / `608` / `136` / `16.2` 全部命中；英文：另加 `0.102941` / `0.065147` / `weight mass` 命中 |
| 废弃值扫描 | 两本均 0 命中：`0.1074` / `0.1479` / `3.28494` / `82/82` / `0.0765` / `60,542.7` / `72.16` / `待补` |

英文稿的旧值同步清单（Lead 实测）已在上一版记录，全部落掉：`3.28496`、`78/82`、
`+0.0764`/`+0.1310`、`1,656,209,408`、`60,447.71`、`9.9782`、`338,196,736`、`165,983,160`、
`0.5430`、`67.08%`，并把英文标题换成终稿标题。

其余通用步骤（供后续改动复用）：`python makefigs.py`、`python mktex.py`、两个语种各
`xelatex` 两遍、`python report.py` / `verify.py` / `langcheck.py`。构建是幂等的；中文稿若改
表头或小节标题，先跑 `.regress/precheck.py`（表题与插图锚点哨兵）再重建。
