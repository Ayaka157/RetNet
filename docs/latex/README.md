# RetNet 技术报告 —— LaTeX 源码与构建说明

本目录把 Markdown 技术报告拆成**互相独立的 PDF**。目前配置了**两个交付报告**，外加一个
**自测夹具**（`selftest`），由 `mktex.py` 里的 `Report` 记录描述（见「多报告配置」）；
加新报告只需要加一条记录和它的手写片段，不必改构建逻辑。

## RetNet-272M（双语单文件源）

`docs/19_RetNet-272M_技术报告.md` 是**中英双语单文件**：每个顶层小节都有一对
`### N.N 中文` / `### N.N English` 标记。

| 版本 | 输出 | 页数 | 内容 |
|---|---|---|---|
| 中文版 | [`zh/RetNet-272M-技术报告.pdf`](zh/RetNet-272M-技术报告.pdf) | 25 | 全部中文正文；英文小节（`### N.N English`）已剔除 |
| 英文版 | [`en/RetNet-272M-Technical-Report.pdf`](en/RetNet-272M-Technical-Report.pdf) | 29 | 全部英文正文；§1 取源文件自带英文，§2 起由 `en/_supplement.tex` 提供 |

## RetNet-0.5B（中文单语源）

`docs/26_RetNet-0.5B_技术报告.md` 是**中文单语**，没有成对英文标记，因此拆分方式不同：

| 版本 | 输出 | 内容 |
|---|---|---|
| 中文版 | [`zh/RetNet-0.5B-技术报告.pdf`](zh/RetNet-0.5B-技术报告.pdf) | 直接取 Markdown 正文（`split_mode = "mono"`）；摘要取自 `## 摘要` 一节，排在目录之前 |
| 英文版 | [`en/RetNet-0.5B-Technical-Report.pdf`](en/RetNet-0.5B-Technical-Report.pdf) | 全文由手写英文补充稿 `reports/0p5b/en_supplement.tex` 提供，结构与中文版章节对齐 |

两版都是**完整报告**：各自有标题页、摘要、目录、正文、编号表格与插图、附录与参考文献。

> 0.5B 的英文补充稿目前是**结构完整的骨架**（章节标题 + 关键段落，数字取自
> `docs/_report_0p5b/facts.md` 已核实的事实）；待中文正文定稿后逐节补齐。
> 实际页数与日志摘要记录在 [`docs/_report_0p5b/build.md`](../_report_0p5b/build.md)。

**语言互斥**由 `verify.py`、`langcheck.py` 与 PDF 文本抽取双重实测。272M 英文版正文里
**没有任何中文字符**（残留的 192 个全部落在代码块的中文注释内），中文版不含任何英文正文段落。
0.5B 的实测数据见 [`docs/_report_0p5b/build.md`](../_report_0p5b/build.md) —— 该报告的
中文源文件尚未交付，本轮只用自测夹具验证了链路。

## 排版约定

对齐常规技术报告的写法，而不是幻灯片式的装饰排版：

- 单栏、衬线正文，章节标题为黑色粗体，不用彩色装饰线与色块；
- **表格与插图统一编号并配标题**（`表 N.` / `图 N.`），标题在表上方、图下方；
- 引用块不画灰底框，改为以 **Note.** 起头的缩进段落；
- 标题页只保留标题、项目名与日期，另附一张模型/硬件/数据的四行摘要表；
- 附录标题本身已含字母（`Appendix A: ...`），因此附录计数器改为字母（A–D），
  参考文献与引用可靠性说明作为「后置内容」另起编号（1.、2.）。

## 已知取舍

- **语言的分配依据源文件写法**：`### N.N 中文` 与 `### N.N English` 标记了同一节的两半，
  中文版取中文半、英文版取英文半。因此中文版无英文摘要，英文版无中文正文。
  （0.5B 走 `split_mode = "mono"`：源文件本身是中文单语，不拆，直接整篇作为中文版正文。）
- 英文版 §1 采用源文件自带的完整英文段落；§2 的英文半只有一段概括，若直接取用会让
  该节的中文半泄漏进来，故 §2 连同 §3–§10、附录一并在 `en/_supplement.tex` 中重新撰写。
- 附录 A–D 的代码块保留了源码中的**中文注释**（这是刻意的：注释是源码的一部分），
  因此英文版 PDF 里仍有少量中文字符，均位于代码块内。
- 中文版有 1 条字体形状警告（`FangSong` 无斜体，用正体替代），不影响输出。
- 表格列宽由 `mktex.py` 估算后写成显式 `p{}` 宽度；改动 Markdown 表格文字后重跑 `mktex.py` 即可重新估算。
  手写补充稿里的表格不受此保护：窄 `p{}` 列放不下不可断行的长代码串会溢出，需要手工断开。
- 表格标题由「表头行」匹配得到（Markdown 表格与补充稿的手写表格各有一套映射）；
  新增表格时需在对应的 `TABLE_CAPTION_HINT*` / `SUPPLEMENT_TABLE_HEADERS*` 中登记，否则该表不带编号。
- 标题页的「本 PDF 为中文版 / 英文版」仅用于区分两个文件，可按需删除。
- `render_list` 会把**每个顶层列表项渲染成独立的 list 环境**（272M 一直如此）。这是共用的
  已知行为，本轮为保持 272M 产物逐字节不变而**刻意未改**；如需修，应作为独立改动并重跑两本 PDF。
- `Report.compact_tables`（默认关，0.5B/selftest 打开）解决宽表与长串溢出：无分隔符长串
  （sha256 等）每 12 字符给一个断行机会、所有行内代码都给断行机会、列数 ≥ 10 的表改用
  `\scriptsize`、行数 ≥ 18 的表改用可跨页的 `longtable`。默认关是因为它会改变已交付的 272M
  产物；272M 若要受益，需作为独立改动重跑并重新验收。
- **超过一页高的表必须可断**：不可断的 `tabular` 会被整块顶出纸面（`Overfull \vbox`，
  超出部分被阅读器裁掉）。Markdown 表由 `render_table` 自动转 `longtable`；**手写补充稿里的
  长表要自己写成 `longtable`**（并用 `\caption{}` 自带题注，因为它不再是
  `\begin{center}\small` 块，`SUPPLEMENT_TABLE_HEADERS*` 不会给它加题注）。

## 插图

全部插图由 [`makefigs.py`](makefigs.py) 从仓库内的真实产物生成，**不依赖 matplotlib**
（本机未安装，也不引入新依赖）：图形直接以 PDF 绘图算符输出，因此是矢量图。数据来源：

| 图 | 数据源（272M） |
|---|---|
| `fig-loss` / `fig-lr` | `checkpoints/retnet_272m_v14/checkpoint-10000/trainer_state.json`（500 个点） |
| `fig-bench-cats` | `logs/bench_v3/report.T0.7_s1.json`、`report.c10k.json` |
| `fig-steps` / `fig-kernels` / `fig-gemm` / `fig-truncation` | 报告正文中的实测值 |

| 图 | 数据源（0.5B） |
|---|---|
| `fig-0p5b-loss` | 三个 `trainer_state.json`（`retnet_500m_v15b/checkpoint-67390`、`_ep2`、`_ep3`）的**每个日志步** loss（3369 / 842 / 842 点）+ **每次评测**的 `eval_loss`（13 / 8 / 8 点），横轴为累计 token |
| `fig-0p5b-lr` | 同一批日志步的 `learning_rate`（三段各自重启一条完整 cosine） |
| `fig-0p5b-step` | 报告 §3.3 表 6 的实测分桶（575 ms 步：GEMM 354.6 / 逐元素 160.5 / 归约 48.5 / 其它 11.6 ms） |

```powershell
python makefigs.py      # 写入 figures/*.pdf（先清空该目录）
```

## 多报告配置

报告由 [`mktex.py`](mktex.py) 顶部的 `Report` 记录描述，`REPORTS` 里登记了三条：

| key | 源文件 | 拆分方式 | 英文正文来源 | 默认构建 |
|---|---|---|---|---|
| `272m` | `docs/19_RetNet-272M_技术报告.md` | `bilingual`（取 `### N.N` 标记的一半） | `en/_supplement.tex` + 源文件 §1 英文 | 是 |
| `0p5b` | `docs/26_RetNet-0.5B_技术报告.md` | `mono`（正文即中文版） | `reports/0p5b/en_supplement.tex`（全文） | 是 |
| `selftest` | `selftest/placeholder-0p5b.md` | `mono` | `selftest/en/_supplement.tex`（全文） | 否（用 `--report selftest`） |

`selftest` 是**流水线自测夹具**：源文件是占位稿，产物写在 `selftest/zh`、`selftest/en`，
因此不需要真实报告就能验证「中文单语 md → 中文 PDF」和「英文补充稿 → 英文 PDF」两条链路。
它同时是一份可运行的最小样例（表格题注、按标题插图、附录计数、前置摘要都在里面）。

新增报告的步骤：

1. 加一条 `Report`（源文件、两个文件名、拆分方式、补充稿、preamble/标题片段、图表锚点）；
2. 写该报告的 `*_preamble.tex` / `*_titles.tex` / `en_supplement.tex` 片段；
3. 在对应的 `TABLE_CAPTION_HINT*`（Markdown 表格）与 `SUPPLEMENT_TABLE_HEADERS*`（手写表格）登记表题；
4. 新插图加进 `makefigs.py`，再登记到该报告的 `figures_zh` / `figures_en`。

## 构建

需要 XeLaTeX（MiKTeX）。在本目录下执行：

```powershell
$env:PATH = "C:\Program Files\MiKTeX\miktex\bin\x64;$env:PATH"

python makefigs.py       # 1) 生成插图
python mktex.py          # 2) 由 Markdown 生成各报告的 .tex（幂等）
#    python mktex.py --list            列出已配置的报告及其源文件是否存在
#    python mktex.py --report 0p5b     只构建一个报告（源文件缺失时报错）
#    python mktex.py --report selftest 只跑自测夹具

# 3) 各编译两遍（第二遍补齐目录与交叉引用）
cd zh; xelatex -interaction=nonstopmode "RetNet-272M-技术报告.tex"; xelatex -interaction=nonstopmode "RetNet-272M-技术报告.tex"; cd ..
cd en; xelatex -interaction=nonstopmode "RetNet-272M-Technical-Report.tex"; xelatex -interaction=nonstopmode "RetNet-272M-Technical-Report.tex"; cd ..
cd zh; xelatex -interaction=nonstopmode "RetNet-0.5B-技术报告.tex"; xelatex -interaction=nonstopmode "RetNet-0.5B-技术报告.tex"; cd ..
cd en; xelatex -interaction=nonstopmode "RetNet-0.5B-Technical-Report.tex"; xelatex -interaction=nonstopmode "RetNet-0.5B-Technical-Report.tex"; cd ..
```

自测夹具（不依赖真实报告）：

```powershell
python mktex.py --report selftest
cd selftest\zh; xelatex -interaction=nonstopmode "SelfTest-技术报告.tex"; xelatex -interaction=nonstopmode "SelfTest-技术报告.tex"; cd ..\..
cd selftest\en; xelatex -interaction=nonstopmode "SelfTest-Technical-Report.tex"; xelatex -interaction=nonstopmode "SelfTest-Technical-Report.tex"; cd ..\..
```

> 若 PDF 正被阅读器（Acrobat 等）打开，`dvipdfmx` 会报
> `Unable to open "...pdf"`。关掉阅读器再编译即可，这不是构建错误。

## 校验

```powershell
python verify.py     # 结构检查：对所有已构建的报告（272M 的原有检查保持原样）
python report.py     # 汇总 zh/、en/ 下的 .log：页数 / 报错 / 未定义引用 / Overfull hbox
python langcheck.py  # PDF 文本抽取：中文版无英文正文段落、英文版无中文正文（代码除外）
python pdfbounds.py  # PDF 页面越界检查：任一页内容底边/右边都必须落在纸张内
python pdfcheck.py   # 272M 两本 PDF 的关键内容点（pypdf 抽取）
```

- `verify.py` 的英文版检查会**先剔除代码**（`lstlisting`、`\code{}`、`\texttt{}`）与
  **LaTeX 注释**再找中文字符，因此「代码注释里的中文」和「源文件注释里的中文」都不算串味；
  对 0.5B 与 selftest 也做同样的检查。
- `langcheck.py` 对中文版的英文段落候选先把**参考文献**与**公式行**显式豁免并计数（可人工审计），
  其余候选必须能追溯到代码，否则判 FAIL；英文版里含中文的行只能来自代码。
- `langcheck.py` 会在任一 PDF 页数 ≤ 8 时打印 WARNING（0.5B 的验收要求是两本都 > 8 页）。
- `pdfbounds.py` 补的是 `report.py` 看不到的一类缺陷：`Overfull \vbox` 会让**整块**内容被
  顶出纸面（日志数字很大、但只报 hbox 时完全看不见），PDF 里文字对象存在、超界部分被阅读器
  裁掉。它用 PyMuPDF 逐页取「文字块 + 绘制路径」的最下/最右边缘与纸张尺寸比较，越界即 FAIL。
  超过一页高的表格由 `render_table` 渲染成可断的 `longtable`（≥18 行时），英文补充稿里同类
  长表也要手写为 `longtable`（见「已知取舍」）。

当前状态：四本 PDF 全部 `verify.py` 69/69、`langcheck.py` 4/4、`pdfbounds.py` 4/4；
0.5B 两本 vbox 0 条、hbox 最大值 ≤4.17 pt（中文）/ ≤3.14 pt（英文）；
272M 两本 0 error / 0 undefined、hbox ≤4.07 pt。

## 文件

| 文件 | 作用 |
|---|---|
| `makefigs.py` | 从仓库真实数据生成 `figures/*.pdf`（自带的极简 PDF 绘图层） |
| `mktex.py` | Markdown → LaTeX：`Report` 多报告配置、按语言拆分（`bilingual` / `mono`）、渲染表格/公式/代码、估算列宽、编号表格标题、按锚点插入插图 |
| `preamble-common.tex` | 两版共用的宏包、字体、代码样式、`note` 段落与附录计数器 |
| `zh/_preamble.tex`、`zh/_titles.tex` | 272M 中文版文档类、中文字体、标题页与摘要 |
| `en/_preamble.tex`、`en/_titles.tex` | 272M 英文版文档类、西文字体（含 CJK 回退）、标题页与摘要 |
| `en/_supplement.tex` | 272M 源文件只有中文的章节（§3–§10、附录 A–D）的英文正文 |
| `en/_supplement-notes.md` | 272M 英文补充稿的取舍与未核实项记录 |
| `reports/0p5b/zh_preamble.tex`、`zh_titles.tex` | 0.5B 中文版文档类与标题页（摘要与目录由构建器插入） |
| `reports/0p5b/en_preamble.tex`、`en_titles.tex` | 0.5B 英文版文档类、英文标题页与摘要 |
| `reports/0p5b/en_supplement.tex` | 0.5B 英文版全文（中文单语源，故英文版完全由手写补充稿提供） |
| `verify.py` | 结构检查（272M 原有检查 + 对所有报告的通用语言/编号检查） |
| `report.py` | 汇总 `zh/`、`en/` 下各 .log 的页数/报错/未定义引用/Overfull |
| `langcheck.py` | PDF 文本抽取的语言互斥检查（可审计的豁免计数 + 页数下限警告） |
| `pdfbounds.py` | PDF 页面越界检查：逐页取文字块与绘制路径的最下/最右边缘，越界即 FAIL（`Overfull \vbox` 的独立复核） |
| `pdfcheck.py` | 272M 两本 PDF 的关键内容点检查 |
| `selftest/` | 流水线自测夹具：占位 md + 中英 preamble/标题 + 英文补充稿 |
| `.regress/` | 272M `.tex` 的哈希基线（改造前后逐字节比对用） |

