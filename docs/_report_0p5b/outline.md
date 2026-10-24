# 0.5B 技术报告 · 大纲与文风规约（Lead 下达，写手必须遵守）

> 主体：`checkpoints/retnet_500m_v15b_ep3`。**只写 0.5B 这一代模型。**
> 前代（100M / 143M / 272M）的模型、训练历程、成绩一律不写。
> 参照文风：`2609.19969.pdf`（DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression，51 页）。
> 交付：`docs/26_RetNet-0.5B_技术报告.md`（中文单语）

---

## 一、参照文风的具体特征（从 2609.19969 抽出，逐条照做）

| # | 特征 | 说明 |
| --- | --- | --- |
| 1 | **标题 = `名称：一句话卖点`** | 例：`DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression`。副标题必须是一个**具体的可量化主张**，不是形容词。 |
| 2 | **摘要一段到底** | 不分点、不排比。顺序固定：问题 → 我们做了什么 → 关键数字 → 结论 → 产物地址（HF 链接那种）。 |
| 3 | **六段式主结构** | Introduction / Architecture / General Infrastructures / Pre-Training / Post-Training / Conclusion, Limitations, and Future Directions。 |
| 4 | **每节开头一句话交代"这节回答什么"** | 不铺垫、不抒情、不复述上一节。 |
| 5 | **图题在图下方，`Figure N \|` 粗体；表题在表上方，`Table N \|`** | 本项目中文版用 `图 N.` / `表 N.`。题注里把 (a)(b) 说明白。 |
| 6 | **细节下沉** | 正文只留解释性句子；超参、复现命令、逐项成绩全部进附录。"Evaluation Details" 是独立附录。 |
| 7 | **数字带单位与口径** | `890 bytes per token`、`45T tokens`、`552B backbone parameters`。绝不含糊成"很快""大幅"。 |
| 8 | **负结果与限制单列** | 标题里就有 Limitations；阴性结果不藏在正文里，专门列。 |
| 9 | **无"我们认为""值得注意的是""令人兴奋的是"** | 参照文里一次都没有。删掉所有情绪词。 |
| 10 | **不写双语对照** | 中文版就是纯中文正文。英文版另起文件，由英文写手写，不在同一份 md 里并置。 |

---

## 二、规定大纲

### 标题页块

```
# RetNet-0.5B：<副标题，一句话可量化主张>

> 日期：2026-10-13 ｜ 对象：checkpoints/retnet_500m_v15b_ep3
> 硬件：RTX PRO 6000（训练）/ RTX 3090 24 GB（首轮）/ RTX 3070 8 GB（评测与服务）
> 数据：data_cache/chat_pack_v13
> 说明：所有性能数字均为真实 Trainer 路径实测；未独立核实处显式标注。
```
另附一张四行摘要表（模型 / 硬件 / 数据 / 参数量）。

### 摘要

一段，不分点。必须出现：504,179,896 参数、20 层/1408 维/22 头、552.33 M token × 3 epoch、val loss 2.6358 → 1.9459、单次可切换的 parallel/chunkwise/recurrent 三范式、以及在线服务地址 `https://retnet.ayaka157.icu/play/`。

### 1. Introduction / 引言

- 约束是什么（单卡、显存、成本），为什么这个约束决定了一切设计选择。
- RetNet 的属性：线性递推 + 一个模型三种等价计算形式。
- **本报告不提出新架构**，记录一个具体模型从零训到线上的完整工程事实。
- 明确指出：**三处"不报错"的缺陷** 与 **阴性结果** 与正面结果同等重要。

### 2. Architecture / 架构

- 2.1 保留机制（Retention）：并行形式、循环形式、$\gamma_h$ 的语义。
- 2.2 **逐头可学习衰减门（GLA）**：本项目相对原论文的改动，$(0,1]$ 参数化的具体做法，梯度怎么走。
- 2.3 三范式等价性与 chunkwise：分块闭式解，`RETNET_FAST_CHUNKWISE=1`。
- 2.4 宽度生长的函数保持层归一化（`GrowthLayerNorm`，`growth_old_dim=768`）。
- 2.5 0.5B 配置表（照 `facts.md §A` 全量列）。
- 数学推导要有，但不许长于 2609.19969 相应节的比例：能一句说清的不要写一段。

### 3. General Infrastructures / 训练基础设施

这是**本报告最有价值的一节**，篇幅可以最大。

- 3.1 训练栈：`retnet_hf/train.py` + HF `Trainer`，任务配置驱动（`tasks/*.json`）。
- 3.2 显存与批量：8 GB / 24 GB / 48 GB 三张卡上的可行边界；`eval_batch_mult=1` 为什么是硬要求；
  8-bit AdamW 为什么在 0.5B 上是**必需**而非优化。
- 3.3 步时分解：一步的时间花在哪、带宽下限、结构性冗余（引 `docs/25` 的字节账）。
- 3.4 **三处静默正确性缺陷**：
  (a) `--resume` 被检查点里的 `train_batch_size` 静默覆盖（batch 16 → 4）；
  (b) 缺 `optimizer.pt` 时 scheduler 不恢复 → 学习率从峰值重来；
  (c) 等价性测试自己跟自己比。
  每处给出：现象 → 定位 → 修法 → 复现证据路径。**证据必须指到文件**。
- 3.5 阴性结果：`torch.compile` 各模式的实测、融合 CE（负收益）、GroupNorm 布局改写、物理 batch 2。
  每条给数字，不给形容词。

### 4. Pre-Training / 训练

- 4.1 数据构成（`facts.md §B`）。**v9 base 占比未知这一点必须显式写出来**，不许糊过去。
- 4.2 三段训练设置表（`facts.md §D`）。
- 4.3 loss 轨迹：`facts.md §C` 三张表全上（或折线图 + 表）。
- 4.4 **热重启代价**：`facts.md §C.4` 那张表必须出现在正文，不是附录。
- 4.5 算力与成本：按实测吞吐复算 1 epoch / 10 token-per-param / Chinchilla 20× 三档，
  给出「已花费」与「到 20× 还差多少」。单价必须写清是哪张卡、哪个时间点的价。
- 4.6 已训 token/参数比 = 3.285，与 Chinchilla 最优的差距，以及这个差距**如何解释第 5 节的成绩**。

### 5. Evaluation / 评测

- 5.1 基准设计：`bench_v3`，1,020 项 / 15 类 / 难度分层 / 判据 DSL / 逐位等价的解码器 / 可复现性保证
  （题库哈希、逐请求独立随机流、逐项温度、断点续跑）。
- 5.2 结果：全量表（15 类 + 加权总分 + 95% CI + think 泄漏率 + 退化重复率）。
  **数据源**：`logs/bench_v3/report.500m_ep3.json` 与 `summary.500m_ep3.md`（由 Lead 提供，跑完才有）。
- 5.3 难度分层 E/M/H 通过率。
- 5.4 停止行为：`<think>` 闭合率、截断率、`stop_reason` 分布。说明为什么"会停下来"比"会答题"更早修好。
- 5.5 **诚实结论**：把 val loss 的改善与任务正确率的绝对值放在一起看，说明
  「loss 下降 ≠ 任务可用」，并给出 token/参数预算不足这一量化解释。
- 5.6 定性观察：模型在开放闲聊上可用，在算术/知识题上不可用。给出真实回复摘录（可脱敏但不得伪造）。

### 6. Inference and Serving / 推理与服务

- 6.1 训练用 chunkwise、预填用 parallel、逐 token 解码用 recurrent —— 同一套权重。
- 6.2 解码吞吐实测（引 `logs/bench_v3/decode_speed.json` 与 `scripts/bench_decode_speed.py`）。
- 6.3 服务实现要点：SSE 流式、单工作线程 FIFO、`<think>` 三态渲染、并发与配额。
- 6.4 部署形态与访问地址。**不写域名注册商/隧道 token 等运维私密信息**，只写架构。

### 7. Conclusion, Limitations, and Future Directions

- 结论 3–5 句，不许新内容。
- Limitations 逐条列：token/参数不足、评测只 1 种子无统计功效、长上下文深度、服务为单实例、**未核实项**。
- Future Directions：下一档算力能把 val loss 推到哪（用实测吞吐外推，标明是外推）。

### 附录

- Appendix A：完整超参与任务配置（照 `facts.md §D`）。
- Appendix B：复现命令（训练 / 评测 / 服务，逐条可复制）。
- Appendix C：阴性结果与三处静默缺陷的完整证据索引（文件路径 + 关键行）。

---

## 三、篇幅与硬约束

| 项 | 要求 |
| --- | --- |
| 语言 | **纯中文单语**。不得出现成对的英文段落。术语首次出现给英文原词（如「保留网络（retentive network）」）。 |
| 篇幅 | 正文 6,000–11,000 字（不含附录）。DeepSeek 那份 51 页里有大量图，我们按字数对齐信息密度，不按页数。 |
| 表格 | 每个数字表都必须有 `表 N.` 题注，编号连续。 |
| 公式 | 用 `$$...$$`，与现有 `mktex.py` 的解析规则兼容（见 `docs/latex/README.md`）。 |
| 禁止 | 编造数字；照抄旧文档里已被推翻的推算值；把推算写成实测；抒情句；"我们认为"。 |
| 必须 | 每个数字可溯源到「文件路径 + 字段」；阴性结果与阳性同篇幅；未验证的写「假说，未验证」。 |

---

## 四、写作前必须读的东西（按顺序）

1. `docs/_report_0p5b/facts.md`（**先读这个，它是权威值**）
2. `docs/22_0.5B_提速研究.md`、`docs/25_一步计算构成与压缩空间.md`、`docs/23_0.5B扩容方案.md`
3. `docs/24_AutoDL上机手册.md`（云上流程与实测）
4. `logs/autodl_20261013/linux/v16_ep2.out` 与 `v17_ep3.out` 的**尾部**（真实 `train_runtime`、`it/s`）
5. `docs/14_基准测试v3.md` §1–§7
6. `retnet_hf/modeling_retnet.py`（架构数学，只读需要的部分）
7. `docs/19_RetNet-272M_技术报告.md` —— **只作为文风与章节粒度的反面/正面参照，内容一律不抄**
