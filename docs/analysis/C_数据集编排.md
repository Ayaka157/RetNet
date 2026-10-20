# C · 数据集编排审计与重设计方案

> 审计对象：`E:\Workspaces\RetNet`（RetNet 0.5B，504,179,896 参数，训练窗口 2048）
> 审计时间：2026-10-20 · 审计人：独立 Agent C
> 方法：只用 `C:\Tools\Miniconda3\envs\AI\python.exe` 以 `numpy.mmap_mode='r'` 只读 npy 头与分片内容统计；未修改任何已有文件、未 commit、未启动训练。
> 标注约定：**【实测】**＝我在本机跑脚本数出来的；**【文档转述】**＝项目文档/`meta.json`/计划注释里写的，我没有独立复核；**【我的推断】**＝基于实测的推理，未直接验证。
> 未读取 `docs/analysis/` 下任何既有文件（含 `D_数据集编排.md`）。

---

## 1. 结论摘要

1. **现役"数据集"其实只有一份 552 M token 的对话包（`chat_pack_v13`）被反复吃了 5 轮；预训练流 2.31 B 从未进入 v15b/v15c 那一轮的实际训练。** 【实测】v13 的 loss 权重里 88.3% 落在 reply 上，连续自然文本（`wcode=3` plain）只占 5.95%。
2. **最近一次 160 k 步训练的指标是反向的**：train loss 从 3.69 掉到 0.49（45 k 步后长期在 0.49–0.69），而 holdout 每 1 200 步评测从 **2.5995 单调涨到 4.546**（141.6 k 步），最后才因"训练集里就有验证集"掉回 2.445。【实测】`logs/curves/full_history_final.csv`。step 17 500 → 最终模型 holdout 由 **2.847 退化到 3.502**（+0.635 nats ≈ ppl ×1.89）。
3. **token 账是虚的**：`plan_ddp2.json` 声称唯一 token 10.09 B（20.0 tok/param），但 `hf_pack_code2` 与 `hf_pack_code` 是**同一份 Ling-Coder 26 个 parquet 从第一片重新分词**（`scripts/_prep_ext.py:95` 与 `_data_supervisor.py:41` 同一 glob），前 ~1.64 B token 必然重复 → 真实唯一 token ≈ **8.45 B = 16.8 tok/param**。
4. **新语料在分词时没有文档边界**：`build_flat_corpus.py` 的 `--add-eos` 是可选参数，而所有实际构建命令（`_prep_ext.py:80-84`、`_data_supervisor.py:182`）**都没有传**；代码/网页/教材/数学/中文编程共 ~8.4 B token 是"一整条无边界流"，`v22` 又把它按 16385 直接重切。这一点项目自己的 `build_ctx_mix.py:14` 已经写明。
5. **规划机制缺三样东西：唯一 token 去重校验、停机条件、不可篡改的 held-out。** 一天内 `total_steps` 被热改 6 次（60 955 → 78 999 → 76 000 → 148 000 → 152 557 → 156 057 → 159 857），cosine 被反复拉长，最终 lr = 1.2576e-05 未走到 min 1.1e-05；`val_dir` 指向 `chat_pack_v13`，而它同时是训练数据。

---

## 2. 现状盘点（逐包实测）

### 2.0 总账

| 项 | 数值 | 来源 |
|---|---:|---|
| `data_cache/` 子目录 / 文件 / 体积 | 26 / 2,532 / **43.28 GB** | 【实测】`Get-ChildItem -Recurse` |
| `.npy` 分片数 | **124**（其中 4 个是 wcode 掩码，不是 ids） | 【实测】 |
| 磁盘上 token（按 ids 计、train+val、2D 只算 `rows×seq_len`） | **8,365,214,145（8.365 B）** | 【实测】`totals.py` |
| 其中 12 个 chat_pack 合计 | **3.983 B**，唯一内容 **0.554 B**，重复 **7.19×** | 【实测】 |
| 词表兼容（剔 wikitext 16k/8k） | **8.238 B** | 【实测】 |
| 内容去重后可直接用 | **4.809 B** | 【实测】（= 8.238 − 3.429 冗余） |
| 未 token 化文本（jsonl/json/csv） | 34 个 jsonl/json 承载 **6.75 M 条**，约 **4.0 B token 估算** | 【实测】行数；token 数为【文档转述】doc27 系数 |
| 词表 | BPE **20 000**，`<pad>`=0 / `<bos>`=1 / `<eos>`=2 / `<unk>`=3 | 【实测】`bpe_tokenizer.json` |

> **与 `docs/27` 的口径差异（我以实测为准）**：doc27 写"`.npy` 合计 8,470,762,590 / 词表兼容 8.34 B"，我实测 **8.365 B / 8.238 B**，差 105 M（1.3%）。doc27 也没说明是否把 `*.wcode.npy` 算进去——若算进去会多出 6.4 B，所以它不是这个口径。差异来源我未追查；下文所有数字用我这一版。

> **token 计数约定**：packed 包的 `meta.json` 里 `train_tokens` 按 `rows×2049` 计（含 +1 的 shift 列），本文表格按 `rows×2048` 计（真正的训练 token）。两者差 `rows` 个，例如 v13：meta 552,330,489 vs 我 552,060,928。

### 2.1 每个目录一行（全部实测）

| 目录 | 内容 | 格式 / dtype | 宽度 | 文档边界 | token / 行数 | GB |
|---|---|---|---|---:|---|---:|---:|
| `pretrain_v2` | 预训练流 7 源（中文 FineWeb-Edu、SkyPile、FineWeb-Edu、Cosmopedia、wiki_zh、open-web-math、poetry_zh） | 一维 `int16` **62 片** + `manifest.json` | 无固定宽（flat） | **有**：每篇 `<bos>`…`<eos>` | **2,816,814,321** | 5.25 |
| `pretrain_merged` | 旧版合并流（v15 之前） | 一维 **`int64`** 3 文件 | flat | 有（eos 密度 1.19‰） | 1,311,846,870 + val 5.5 M | 9.82 |
| `chat_pack_v13` | **现役对话包**：v9 基座重切 + R1 全 CoT + i_wonder_why + ruozhiba | 二维 `int16 (269561, 2049)` + `uint8` wcode | **2049 = 2048+1** | 无（packed，48.4% 行不以 BOS 开头） | **269,561 行 / 552,330,489** | 1.55 |
| `chat_pack_v12` | v10 去 `<think>` | `int16 (919532, 513)` + wcode | 513 | packed | 471,719,916 | 1.32 |
| `chat_pack_v11` | v10 重打包 8K | `int16 (54015, 8193)` + wcode | 8193 | packed | 442,544,895 | 1.24 |
| `chat_pack_v11_4k` | v10 重打包 4K | `int16 (108030, 4097)` + wcode | 4097 | packed | 442,598,910 | 1.24 |
| `chat_pack_v10` | v9 + R1（1200 字符截断）+ wiki | `int16 (940232, 513)` + wcode | 513 | packed | 482,339,016 | 1.35 |
| `chat_pack_v9` | v13 的基座（a_grade 10 源 + RP + 闲聊） | `int16 (546950, 513)` + wcode | 513 | packed | 280,585,350 | 0.79 |
| `chat_pack_v8` / `v7` / `v6` / `v5` / `v4` | 同一数据体的早期重打包 | `int16 (438647/435279/409172/456023/423406, 513)` + wcode | 513 | packed | 225.0 / 223.3 / 209.9 / 233.9 / 217.2 M | 0.63/0.63/0.59/0.66/0.61 |
| `chat_pack_v3` | 第一代 | **`int32`**（浪费 2×）`(379546, 513)` + wcode | 513 | packed | 194,707,098 | 0.91 |
| `chat_recovery_mix` | 旧平铺对话混合，**无 wcode** | 一维 **`int64`** | flat | 仅靠 eos，密度 11.7‰（≈85 token/条，碎） | 82,901,765 | 0.62 |
| `chat_sft_mix` | 旧平铺 SFT 混合，**无 wcode** | 一维 **`int64`** | flat | eos 密度 2.36‰ | 36,529,709 | 0.28 |
| `a_grade` | SFT/CoT/RP/纯文本混装原始文本 13 份 | jsonl 一行一条 | — | 无 | 1,928,326 条 | 5.06 |
| `roleplay` | RP 原始文本 14 份 | jsonl | — | 无 | 2,329,926 条 | 4.68 |
| `literature` | 文学/对白挖掘 10 份 | jsonl | — | 无 | 652,987 条 | 1.40 |
| `extra_sft` / `extra_sft2` | 指令 SFT | jsonl 11 + 2 份 | — | 无 | 181,601 / 400,000 条 | 0.75 / 0.59 |
| `chat_corpus` | 分词器训练语料（LCCC 等）+ 百科 | jsonl/json/csv | — | 无 | 6,820,506 行（另 1 个 json 数组） | 1.90 |
| `nonhf` | chinese-poetry + Gutenberg 原始 | 2,318 个小文件（json/txt/zip/db） | — | 无 | 26 个 txt ≈ 23 MB | 0.46 |
| `wikitext103_v16k` / `wikitext2` | 英文基线 | 一维 **`int64`** | flat | 有 | 124,324,096 / 2,742,124 | 0.93 / 0.02 |
| `_resume_probe` | 续训夹具 | `int16 (64,129)` + `uint8` wcode | 129 | packed | 64 行 | ~0 |

补充实测细节：

- `pretrain_v2` 的文档边界是**真的**：`chinese_fineweb_edu_v2_00000.npy` 内 `<eos>`=35,840 个、`<bos>`=35,840 个，eos 间隔中位数 981 token；20 个分片 eos 求和 **840,704**，加 val 的 1,684，正好等于 `manifest.json` 的 `docs=842,388`。**【实测】**（唯一性交叉验证通过）
- 各源 eos 间隔中位数：chinese_fineweb 981 / fineweb_edu 719 / cosmopedia 847 / open_web_math 977 / wiki_zh 1,448 / skypile 509 / poetry_zh 71。**【实测】** 即"中文网页平均约 1 000 token 一篇"。
- `pretrain_v2` 各源实测 token：chinese_fineweb **1,013,826,215**、skypile **504,097,083**、fineweb_edu **504,765,133**、cosmopedia **403,355,751**、wiki_zh **202,958,320**、open_web_math **151,922,835**、poetry_zh **30,418,286**（sum 2.810 B；剔 skypile 后 2.307 B，与计划注释一致）。**【实测】**
- `chat_pack_v13` 的 wcode 直方图：**pad 16,126,035（2.92%）/ reply 358,896,326（64.98%）/ prompt 144,461,792（26.15%）/ plain 32,846,336（5.95%）**。**【实测】** `np.bincount`
- 掩码换算：`modeling_retnet.py:501-503` 把 wcode 2（prompt）映射成 `prompt_loss_weight`，2（默认 0.1，见 `train.py:5`）；wcode 3（plain）映射成 1.0。故 **loss 权重份额 = reply 88.3% / plain 8.1% / prompt 3.6%**。**【实测+代码核对】**
- `chat_pack_v13` 行首是 `<bos>` 的只占 **139,006 / 269,561 = 51.6%**；行尾是 pad 的有 **130,126** 行。**【实测】** → 一半的行是跨样本切出来的。
- `data_cache` 下 **不存在任何 `hf_pack_*` 目录**。**【实测】** 即 10-18 那天新建的 web/code/code2/textbook/math/prog/sft/ctx16k 共 ~8.4 B token 的包已全部消失（gc 或手工删除）。

### 2.2 12 个 chat_pack 的谱系（实测长度 + 文档交叉印证）

```
v3  194.7M (int32,512) → v4 217.2M → v5 233.9M → v6 209.9M → v7 223.3M → v8 225.0M
                                        └→ v9 280.6M  (a_grade 十源)
                                             ├→ v10 482.3M (+R1 1200字截断 +i_wonder_why)
                                             │    ├→ v11 442.5M (重打包 8192)
                                             │    ├→ v11_4k 442.6M (重打包 4096)
                                             │    └→ v12 471.7M (strip_think)
                                             └→ v13 552.3M (v9 重切 2048 + R1 4000字 + wiki) 【在用】
12 包合计 3,983.4 M token，唯一内容 552.1 M，冗余 3,431.3 M（7.19×）【实测】
```

### 2.3 训练计划（`tasks/`，全部读原文）

| 文件 | 阶段顺序（步数） | 总步数 | token/步 | 声称唯一 token | 关键守卫 |
|---|---|---:|---:|---:|---|
| `plan.json` (v2, 单卡) | pretrain_v2_noskypile 70,411 → chinese_web_hq 14,000 → code_ling 47,300 → textbook 5,500 → chat_anneal(v13) 6,000 | 143,211 | 32,768 | 5.039 B（9.995 tok/param） | `min_shards` 46/10/20/1；`exclude:["skypile_"]`；`gc:true` 前 4 段 |
| `plan_ddp2.json` (v13, 2 卡) | pretrain 35,205 → web 7,000 → code_ling 25,045 → textbook 2,750 → code_ling2 **68,557** → prog_zh 3,500 → math_zh 3,800 → sft_mix 14,000 | 159,857 | 65,536 | 10.09 B（20.0 tok/param） | `min_shards` 46/**1**/10/1/10/1/1；`gc` 前 5 段 true；`val_dir = chat_pack_v13` |
| `v21_plan_ddp2.json` (运行配置) | 复用 plan_ddp2 | — | 65,536（batch 16/卡 × 2 卡 × 2048） | — | bf16、chunkwise 128、`compile=max-autotune-no-cudagraphs`、`dataloader_workers=0` |
| `plan_ctx16k.json` (v22) | ctx16k_mix 7,629 步（16384×2×2） | 7,629 | 65,536 | 0.5 B | 数据 `hf_pack_ctx16k` = code2 60% + sft 25% + math 10% + prog 5% |

守卫的**真实语义**（`retnet_hf/plan.py`）：
- 切段由 `plan.stage_index_at(global_step)` 决定，即**纯步数驱动**；`stage_ready()` 不通过就"停在原段，每 200 步重试"（`plan.py:352-361`）——**没有超时、没有降级、没有告警升级**。
- 离开旧段时 `if prev.gc: ds.gc_stage(have)` 直接删该段数据文件（`plan.py:364-368`）。
- `min_free_gb` 只打印告警，不采取行动（`plan.py:392-398`）。
- 热加载：每 ~200 步比对 `plan.json` mtime，改 `steps` 会改变整个步→段映射与 cosine 分母（`_apply_intervals` 那段注释自己记录了 `state.save_steps` 不同步的 bug）。

### 2.4 数据构建脚本实际做了什么（读源码）

| 脚本 | 实际行为 | 关键缺陷（实测自源码） |
|---|---|---|
| `build_flat_corpus.py` | parquet/jsonl → 20k BPE → **一维 int16 分片**，默认 2.5e8 token/片；`--text-field auto` 按字段名猜正文，跳过含 id/url/hash/score/license/date 的列；`--max-tokens` 达额即停；**`--add-eos` 默认关闭** | ① 默认不写 eos → 全部 hf_pack 无文档边界；② `--max-tokens` 从**文件序列开头**截断，同一 glob 再跑一次 = 从头重复（code_ling/code_ling2 就是这么来的）；③ `auto` 字段选择会把 `TiWu-Lab/math-zh` 的 `tokens` 数值列当正文（列名不含 skip 关键词） |
| `build_sft_mix.py` | 取 v13 行 40% + code2/textbook/math flat 随机切 2049 行各 35/15/10%，混洗后写 packed；新语料 wcode=1 | ① chat 部分就是 v13 本体（模型已过拟合的同一份）；② 新语料行是**随机窗口**，可能从半个函数/半个句子开始；③ 验证集直接复制 v13 的 val（`build_sft_mix.py:137-140`）→ 验证集进入训练源 |
| `build_ctx_mix.py` | 按比例轮流取流拼 16K 包；SFT 来源剥 pad 补 eos；**flat 来源原样拼，无 eos** | 注释自认"flat 分片的原始边界在分词时就丢了"；16K 窗口内 60% 是无边界代码流 |
| `build_chat_mix_v13.py` | 读 `chat_pack_v9` 展平重切 2048，再 fresh-encode R1（截断 4000 字符）/i_wonder_why/ruozhiba | 重切丢原始样本边界；`r1_max_chars=4000` 硬截断长 CoT |
| `_data_supervisor.py` | 轮询下载清单 → 就绪后依次分词 code/web/textbook（`--tokens-per-shard 2.5e8`，无 `--add-eos`）→ 打包就绪后**删掉 `data/hf/<src>` 原始 parquet** | 原始数据被删 → 无法重分词、无法换 tokenizer、无法做去重；`_prep_ext.py:4` 明确写了这个副作用 |
| `_prep_ext.py` | code2（5 B 上限）、math（0.3 B）、prog（0.35 B，显式 `fields=text`）→ `build_sft_mix.py` | **code2 与 code_ling 同一 glob 同一顺序**：前 1.64 B token 是重复内容 |

---

## 3. 编排不足（逐条 + 证据）

### A. 配比问题

**A1. 一轮 5 epoch 的对话包压倒了全部预训练语料。**
【实测】`tasks/v15c_500m_pro6000.json` 的 `data_dir` 直接是 `data_cache/chat_pack_v13`（552 M token），从 `checkpoint-67390` 续训；`logs/curves/full_history_final.csv` 的 epoch 字段在 step 159,840 到 4.4087。5 轮 × 552 M ≈ 2.76 B 里，**连续自然文本只有 32.8 M × 5 = 164 M token（5.9%）**，而 reply 被重复 1.79 B 次。
【实测】loss 权重份额 reply **88.3%**。

**A2. 代码从 0 突然变成 56%。**
【实测】plan_ddp2 里 code_ling 25,045 + code_ling2 68,557 = 93,602 步 / 159,857 步 = **58.6% 的步数**、6.13 B token；而在 10-17 之前的全部训练里代码 token 为 0（`chat_pack_v13` 无代码，`pretrain_v2` 无代码）。

**A3. 数学/科学/教材被压到"凑数"级别。**
【实测】plan_ddp2 里 textbook 2,750 步（0.18 B）+ prog 3,500（0.23 B）+ math 3,800（0.25 B）= 合计 **4.1% 的步数**；而 `plan_ddp2._readme` 自己写着"补一个 math 段 3,800 步，把唯一 token 从 19.5 补到 20.0"——**配比由 token 总数倒推，不是由能力缺口决定**。

**A4. 中英比例从未被控制。**
【实测】`pretrain_v2` 里英文源 fineweb_edu 505 M + cosmopedia 403 M + open_web_math 152 M = **1.06 B（37.6%）**，中文源 1.25 B（44.4%），SkyPile 504 M（17.9%，后被 `exclude` 剔除）。剔除后英文占比升到 **45.8%**，而目标是"以中文为主"。没有任何脚本读 `language` 字段做配额。

### B. 顺序问题

**B1. 先对话 5 轮 → 再补预训练 → 再 6 B 代码 → 最后 1.4 B SFT，等于用最后 9% 的步数去挽回前 91% 造成的分布偏移。**
【实测】阶段边界与 loss 窗口：0–35 k 预训练（窗口均值 3.20→2.84）；35–42 k 中文网页（均值 3.12，明显高于预训练段——分布更难的证据）；45–130 k 代码（**均值 0.49–0.70**）；135–160 k sft_mix（均值 1.39，最大 4.20 出现在切换瞬间）。
【实测】holdout：step 42,000 = 3.951（预训练段末 3.118 → 网页段 +0.83）；67,200 = 4.405（代码段再 +0.45）；141,600 = **4.546（全程最差）**。

**B2. 遗忘是可量化的，而且发生在"看起来 loss 很好"的那一段。**
【实测】`logs/holdout_eval.jsonl`，同一套 5 个文件、同一 token 数（4,454,400）：

| 文件 | ep3_full | v21_step15000 | v21_step17500 | **v21_final** | 15000→final |
|---|---:|---:|---:|---:|---:|
| chinese_fineweb_edu_v2 | 3.8519 | 3.2353 | 3.2194 | **3.8605** | +0.641 |
| cosmopedia | 3.0564 | 1.9987 | 1.9740 | **2.8387** | +0.865 |
| fineweb_edu | 3.7223 | 2.9595 | 2.9391 | **3.4997** | +0.541 |
| open_web_math | 3.3976 | 2.6005 | 2.5796 | **2.7467** | +0.146 |
| wikipedia_zh | 3.6474 | 2.8145 | 2.7911 | **3.7319** | +0.941 |
| 合计 | 3.6287 | 2.8666 | 2.8471 | **3.5023** | **+0.635** |

即：**中间那个 checkpoint 比最终模型好 0.64 nats**，而且最终模型在中文百科上比"什么都没做的 ep3 基线"还差（3.7319 vs 3.6474）。`holdout_eval` 的 ep3_baseline 有 6 个文件（含 skypile_val），v21 只有 5 个 → 合计不可直接跨行比；但 v21_step15000 → v21_final 是同一口径，退化成立。

**B3. 代码段 6.1 B token 的"低 loss"不能当作学得好，而且它的低得不像学习。**
【实测】切到 `code_ling` 的那一刻（step 42,205）loss 从 3.00 直接掉到 2.52，**400 步内到 0.99**，之后 90 k 步稳定在 0.49–0.70。一个 0.5 B 模型在 2,600 万 token 内把交叉熵从 3.0 压到 1.0，不可能来自"学会写代码"——更可能是这股 `\n\n` 拼接、无 eos 的代码流本身**低熵且高度自相似**（1.4 M 条 messages 模板，`inclusionAI/Ling-Coder-SFT`），模型很快就能靠模板上下文预测。
【我的推断】同一时刻 holdout 从 4.40 涨到 4.55、最后到 4.546，训练 loss 与能力方向完全相反，说明**"这一段训练 loss 低"在旧编排里没有任何验收价值**。
【实测】另一条被忽略的重复消费：中文网页段只有 0.23 B，却给了 7,000 步 = 0.459 B → **同一份 0.23 B 被过了 2.0 遍**；而预训练段 35,205 步 = 2.307 B 正好 1.0 遍。

### C. 数据质量问题

**C1. 8.4 B 新语料没有文档边界，也没有 eos。**
【实测】`build_flat_corpus.py:188` `--add-eos` 是 `store_true`（默认 False）；`_prep_ext.py:80-84` 与 `_data_supervisor.py:182` 的构建命令里都没有它；`build_ctx_mix.py:14` 的注释直接承认"flat 分片的原始边界在分词时就丢了"。后果：代码文件之间用 `"\n\n"` 相接、网页文档首尾相连，模型无法学到"文档结束"这个事件，`<eos>` 在 20k 词表里只出现在 packed 对话数据中。v22 把这股流按 16,385 重切，等于把无边界问题从 2 K 放大到 16 K。

**C2. 同一份语料被分词两次、训练两遍（唯一 token 虚高 1.64 B）。**
【实测】`_data_supervisor.py:41` 与 `_prep_ext.py:95` 都是 `BASE/ling_coder` + `*.parquet`，`build_flat_corpus.py` 按 `sorted(glob)` 顺序读、达 `--max-tokens` 即 break。code_ling 上限 1.64 B、code2 上限 5 B → **code2 的前 1.64 B 与 code_ling 完全相同**。`plan_ddp2` 的注释却写"Ling-Coder 剩余分片，约 4.49 B 新 token"。

**C3. 训练损失里有 2.9% 是 pad。**
【实测】v13 `ids` 里 id=0 的 token 有 16,126,035 个（wcode=0），虽然权重为 0 不产生梯度，但占了 2.9% 的序列位置与显存/带宽；对照 v9 的 pad 是 11.7% 里的一部分、v10/v12 是 8.3–8.5%。

**C4. 长 CoT 被硬截断。**
【实测】`chat_pack_v13/meta.json`: `r1_max_chars=4000`；v10 更狠（1,200 字符）。R1 蒸馏样本里超过 4,000 字符的推理链被砍掉，而"链式推理"恰好是最需要完整长度的部分。doc27 记录：不截断原文 157.2 M → 截断后 147.9 M（砍掉 5.9%）。

**C5. 字段 auto 选择把数值元数据当正文。**
【我的推断】`TiWu-Lab/math-zh` 有 `tokens` 列（doc28 明写"是别的 tokenizer，别用"），`auto_text_fields` 的 skip 关键词是 `id/uuid/url/uri/hash/sha/score/count/index/license/source/date/time`，**不含 `tokens`** → `hf_pack_math` 里混进了整数 id 串。包已被删除，无法复核。

**C6. 重复包占盘 + dtype 浪费。**
【实测】12 个 chat_pack 合计 3.983 B token，唯一 552 M；`pretrain_merged` 用 `int64` 存 1.31 B token（10.49 GB，int16 只需 2.62 GB）；`wikitext*`、`chat_sft_mix`、`chat_recovery_mix` 同样 int64。43.28 GB 里约 **10 GB 是纯重复**，另有约 **7 GB 是 dtype 浪费**。

**C7. 原始文本层没有统一 schema，字段 auto 选择会把噪声和元数据一起喂进去。**
【实测】抽读 6 个原始 jsonl 目录的首行结构：

| 文件 | 顶层字段 | 陷阱 |
|---|---|---|
| `roleplay/belle_3_5m.jsonl` | `conversations`, `id` | `conversations` 是**字符串化的 Python list**：`"[{'from': 'human', 'value': …}]"` |
| `roleplay/claude_rp_30k.jsonl` | 同上 | 同上 |
| `literature/cornell_rp.jsonl` | `source`, `pairs` | `pairs` 是字符串化的嵌套 list |
| `a_grade/hundredcv_chat.jsonl` | `topic`,`user1`,`user2`,`dialog` | `dialog` 也是字符串化 list |
| `extra_sft/coig_*.jsonl` | 含 `metadata`/`answer_from`/`human_verified`/`copyright` | `build_flat_corpus.py` 的 auto 只跳过含 `id/url/hash/license/source/date` 的列，**这四列会被当正文拼进去** |
| `extra_sft2/zhihu_kol.jsonl` | **大写** `INSTRUCTION`/`RESPONSE`/`SOURCE`/`METADATA` | 按小写键名读会全部读空 |
| `chat_corpus/lccc_base_train.jsonl` | 首行是 **JSON 数组**而非单对象 | 按行 `json.loads` 能过，但结构不同 |

【实测】`build_flat_corpus.py:110-138` 的 `flatten_value` 只在值**已经是** list/dict 时展开；遇到"字符串化 list"直接原样返回 → `{'from': 'human', 'value':` 这类 Python repr 会原样进词表流。现有 `hf_pack_*` 是否踩到无法复核（包已删），但**任何重建都必须先做 schema 归一化**（见 §4.2 P10 的打包步骤）。

### D. 规划机制问题

**D1. 步数是"余额 ÷ 单价"算出来的，不是"token 需求"算出来的。**
【文档转述】`plan_ddp2._readme`："余额 ¥198.50 → 14.2 h → 目标 13.0 h → 60,955 步"；"用户追加预算到 ¥290.30，目标 20 tok/param … 插 code_ling2 段 72,000 步"。**先有钱后有数据配比**，这是这轮编排最根本的机制缺陷。

**D2. 停机条件不存在。**
【实测】`plan_ddp2` 的终点是 `steps` 用完 + deadline（`_readme`："79,000 步会跑到 03:38 撞上死手(03:26)，收敛到 76,000 步"）。**没有任何"holdout 变差就停/回滚"的条件**；实测 eval 从 2.60 涨到 4.55 期间仍继续跑了 117 k 步。

**D3. 验证集 = 训练集。**
【实测】`plan_ddp2.val_dir = "data_cache/chat_pack_v13"`；`build_sft_mix.py` 把 v13 的 40% 行放进 sft_mix，并把 v13 的 `val.ids.npy` 原样复制成 sft_mix 的 val。所以最后 14 k 步的 eval 从 4.55 掉到 2.45 **是记住训练数据，不是能力提升**。

**D4. 计划一天热改 6 次，cosine 分母跟着变。**
【文档转述】`plan_ddp2._readme` 记录 total_steps：60,955 → 78,999 → 76,000 → 148,000 → 152,557 → 156,057 → 159,857。【实测】最终 step 159,840 时 lr = **1.2576e-05**（min 应为 1.1e-05）→ 余弦没有走完，模型停在偏高的学习率上，而这正好是最后退化最严重的一段。

**D5. `min_shards` 守卫写错会让训练永久卡死。**
【文档转述+代码核对】`plan.json` 给 `chinese_web_hq` 写 `min_shards: 10`，实际只产出 1 个分片；`plan_ddp2` 的注释写明"写 10 会让段 1 永远不就绪，训练卡在 35,205 步"。代码语义就是"不就绪就停在原段无限重试"（`plan.py:352-361`），**没有超时告警**。

**D6. GC 把可复现性一起删掉了。**
【实测】`plan.py:364-368`：离开一段就 `gc_stage()` 删文件。今天 `data_cache` 里 **`hf_pack_*` 一个都不剩**；`_data_supervisor.py` 在包就绪后还会删 `data/hf/<src>` 原始 parquet。
【实测】矛盾点：`plan_ddp2` 第 0 段 `gc: true`，但 `pretrain_v2` 62 个分片（5.25 GB）仍在 —— 说明 GC 至少对第 0 段没有生效，**"删了什么、为什么删、删没删成"没有任何审计日志**。

**D7. 统计口径不可用（我已用 epoch 反解出真实阶段长度，但日志本身给不出"每个包吃了几轮"）。**
【实测】`full_history_final.csv` 的 `epoch` 不是"某语料被过几遍"：它只在 **step 70,000** 和 **step 146,000** 两次刷新分母（2.306 B → 4.998 B → 0.614 B），中间的 35,205（预训练→网页）与 42,205（网页→代码）**没有刷新**。所以 step 69,999 处 epoch≈1.99 并不代表"预训练跑了 2 遍"，而是"跑了 1 遍预训练 + 2 遍网页"。最终 epoch=4.4087 是 1.989+0.997+1.476 的累加。
**结论**：日志无法回答"这个包吃了几轮"，必须靠 `plan.steps × 65,536 ÷ 包大小` 手算；`plan.json` 的"9.995 token/参数"、`plan_ddp2` 的"20.0 tok/param"都建立在这个不可直接读出的量上。

### E. 从曲线与 holdout 能读出的退化/饱和

【实测】`logs/curves/full_history_final.csv`（8,111 行，step 20–159,840）：

| step 窗口 | train loss 均值 | 判读 |
|---|---:|---|
| 0–5 k | 3.199 | 预训练开始 |
| 25–30 k | 2.852 | 预训练收敛中 |
| 30–35 k | 2.839 | 预训练末尾，**已进入平台（每 5 k 只降 0.013）** |
| 35–40 k | 3.116 | 换到中文网页，loss 上跳 |
| 40–45 k | 1.821 | 换段瞬间 |
| 45–50 k | 0.696 | 代码段，**断崖式下跌** |
| 95–100 k | 0.522 | 代码段平台 |
| 130–135 k | 0.493 | 全程最低 |
| 135–140 k | 1.335（max 4.200） | sft_mix 切换，尖峰 |
| 155–160 k | 1.387 | 结束 |

【实测】eval（每 1,200 步）：**2.5995（2.5 k，全程最优）→ 2.806（5 k）→ 3.118（33.6 k）→ 3.951（42 k）→ 4.405（67.2 k）→ 4.546（141.6 k，全程最差）→ 3.018（146.4 k）→ 2.445（159.6 k）**。

**段边界的独立证据（实测，用来确认这段曲线确实按 `plan_ddp2` 跑完）**：

| 位置 | 证据 |
|---|---|
| 35,205 | 预训练→中文网页：loss 由 2.84 抬到 3.0–3.4（网页更难），eval 由 3.118 升到 3.951 |
| **42,205**（=35,205+7,000） | 中文网页→`code_ling`：**step 42,200 loss=3.0022，42,220 loss=2.5239（grad_norm 6.00），42,240=1.5806，42,600=0.9943** —— 400 步（约 2,600 万 token）内掉 2.0 nats |
| **70,000**（=35,205+7,000+25,045+2,750） | epoch 速率从 2.306 B/epoch 变为 **4.998 B/epoch** → 数据换成 `hf_pack_code2`（`--max-tokens 5e9` 实产 4.998 B） |
| **145,857**（+3,500 prog +3,800 math） | epoch 速率变为 **0.614 B/epoch**，即 300,000 行 × 2048 —— 与 `build_sft_mix.py --rows 300000` 默认值逐位吻合；该段 14,000 步 = **1.48 epoch** |

`sft_mix` 段被吃了 **1.48 个 epoch**，而它的 val 就是 v13 的 val → §3 D3 的泄漏在数字上闭环。

三点结论：
1. **饱和**：预训练段在 30–35 k 就平了（2.852→2.839），说明 2.31 B token 对这个模型的作用在 ~2 B 时已经吃满，继续加同质 token 收益趋零。
2. **退化**：从 35 k 到 141.6 k，holdout 单调恶化 2.599→4.546（+1.95 nats）。这段"训练 loss 从 3.12 掉到 0.49"的过程，在 held-out 上完全是负收益。
3. **最后的"回升"是泄漏**：146.4 k 之后 eval 掉到 2.445，正好是 sft_mix（含 40% v13）开始的位置。

---

## 4. 新数据方案

### 4.0 设计原则（先说为什么这么改）

1. **先定能力缺口与 token 预算，再倒推步数与钱**（与旧方案相反）。
2. **一包一职责、一包一来源族**；同一份原文只允许出现在一个"唯一 token 包"里，任何重复使用都必须走显式 replay 段并记账。
3. **所有分词都开 eos**；packed 只在对话/SFT 包使用，且必须保留 wcode；长上下文由 packed 包按文档边界拼，**不允许**对 flat 流裸切。
4. **验证集与训练集物理隔离**：来源级、doc 级、n-gram 级三重隔离，并在训练前跑重叠检查。
5. **停机条件写进计划**：holdout 连续 2 次恶化 > 0.05 nats 即触发回滚到最佳 ckpt 并停段。

### 4.1 包清单总表

| # | 包名 | 唯一 token | 占比 | 形态 | 位置 |
|---|---|---:|---:|---|---|
| P1 | `zh_web_core` | 3.80 B | 37.1% | flat+**eos**，int16 | S1 |
| P2 | `zh_edu_textbook` | 0.85 B | 8.3% | flat+eos | S1 |
| P3 | `en_web_edu` | 1.05 B | 10.2% | flat+eos | S1 |
| P4 | `zh_wiki_qa` | 0.32 B | 3.1% | flat+eos | S1 |
| P5 | `code_ling_full` | 2.15 B | 21.0% | flat+eos | S2 |
| P6 | `zh_code_prose` | 0.45 B | 4.4% | flat+eos | S2 |
| P7 | `math_zh_en` | 0.90 B | 8.8% | flat+eos | S3 |
| P8 | `zh_science` | 0.13 B | 1.3% | flat+eos | S3 |
| P9 | `long_ctx_8k` | 0.65 B（重排，非新增） | — | packed 8193 | S4 |
| P10 | `chat_sft_v2` | 0.58 B | 5.7% | packed 2049 + wcode | S5/S6 |
| P11 | `safety_refusal` | 0.02 B | 0.2% | packed | S5/S6 |
| — | 合计唯一 | **10.25 B** | 100% | | |
| E1 | `eval_holdout_v2`（不训练） | 0.06 B | — | flat+packed | 全程评测 |

### 4.2 逐包设计

#### P1 `zh_web_core` — 中文通用底座
- **内容与组成**：① 已有 `data_cache/pretrain_v2/chinese_fineweb_edu_v2_*` **1.01 B**（复用，零成本）；② `Morton-Li/ChineseWebText2.0-HighQuality` 取 **2.79 B** token（约 9.8 GB parquet，按 3.51 bytes/token 反算 ≈ 9.8 GB）。**不要**再混 SkyPile（协议受限，见 P1 备注）。
- **来源**：`Morton-Li/ChineseWebText2.0-HighQuality`，**apache-2.0**，279.15 GB / 155,493,566 行 / ≈83.9 B token，无需申请（public）；`chinese_fineweb_edu_v2` 为本地既有（上游 `opencsg` 系，需按原档案核许可）。
- **目的**：提供"事实密度高、句子完整、有段落结构"的连续中文文本，作为事实性知识与语法的主干——这正是旧方案里 5.95% 的 plain 无法支撑的部分。
- **预期表现 / 验收**：`E1.zh_wiki` 与 `E1.zh_news` 的 loss 从 ep3 基线的 3.65 / — 降到 **≤3.10**；中文 4-gram 复述探针（手工 200 条事实问答，答案必须逐字包含）命中率 ≥25%。训练中每 1,200 步记录，若 2 次连续恶化 >0.05 触发 D 节停机。
- **token 预算 / 占比**：3.80 B / 37.1%。
- **打包方式**：`seq_len=2048`，**flat**，`int16`，**每个文档末尾写 `<eos>`(2)、开头写 `<bos>`(1)**，分片 2.5e8 token；分词命令必须带 `--add-eos`。
- **计划位置**：S1，占 S1 的 67%。

#### P2 `zh_edu_textbook` — 中文教材/学科
- **内容与组成**：① `opencsg/Fineweb-Edu-Chinese-V3` 全量 **0.18 B**（0.284 GB，188,100 条，来自 100,442 本书/教材）；② `BAAI/IndustryCorpus2_subject_education_education` 取 **0.67 B**（208.5 GB 中抽 8 个 parquet，按质量分字段过滤 top 30%）。
- **来源**：前者 **other（OpenCSG 协议，需在页面确认商用条款）**、0.284 GB、无需申请；后者 **apache-2.0**、208.50 GB / ≈90 B token、无需申请。
- **目的**：结构化讲解型文本（定义—推导—例题），是"知识 0.000"最直接的补丁；教材的段落长、指代明确，也能改善长窗口利用率。
- **预期表现 / 验收**：新增 `E1.zh_textbook`（从教材里留出 300 篇，5 M token）loss ≤2.90；学科问答探针（物理/化学/生物各 50 题，选择题）准确率相对 ep3 基线 +10 pt。
- **token 预算 / 占比**：0.85 B / 8.3%。
- **打包方式**：同 P1（flat + eos + int16，2048）。
- **计划位置**：S1，占 15%。

#### P3 `en_web_edu` — 英文通用（知识迁移与代码前置）
- **内容与组成**：① 已有 `fineweb_edu` **0.505 B** + `cosmopedia` **0.403 B**（复用）；② 新增 `HuggingFaceFW/fineweb-edu` sample-100BT 取 **0.14 B**（补齐不同抽样）。
- **来源**：`HuggingFaceFW/fineweb-edu`，**apache-2.0**，sample-100BT 子集约 100 B token，无需申请；`cosmopedia` 为 `HuggingFaceTB/cosmopedia`（apache-2.0）。
- **目的**：英文是代码、数学、科学概念的载体；20k 中文词表对英文的压缩率差（约 4.5–5 bytes/token），所以**只用 10% 预算买"概念覆盖"**，不为英文流畅度付费。
- **预期表现 / 验收**：`E1.en_wiki`（wikitext103 重编码到 20k 词表，替代当前词表不兼容的 `wikitext103_v16k`）loss ≤3.00；英文 MMLU-lite 50 题 ≥ 随机 +8 pt。
- **token 预算 / 占比**：1.05 B / 10.2%。
- **打包方式**：flat + eos + int16，2048。
- **计划位置**：S1，占 12%；另在 S2 末尾 replay 0.30 B（防代码段把英文能力带走）。

#### P4 `zh_wiki_qa` — 百科与问答对
- **内容与组成**：① `wikipedia_zh` **0.203 B**（复用 `pretrain_v2`）；② `i_wonder_why` **0.119 B**（已在 v13 内，这里只保留其**自然段**部分，去掉模板化问答壳）。
- **来源**：zhwiki dump（CC BY-SA，需在发布物里署名）；`i_wonder_why` 为社区中文百科问答集（许可需按 `docs/05` 原档案核）。
- **目的**：高事实密度 + 定义式表达，是"知识问答"探针最敏感的语料；同时给 P1 的网页文本提供"实体—定义"锚点。
- **预期表现 / 验收**：`E1.zh_wiki` loss ≤2.95；中文事实补全探针（"X 的首都是"）top-1 ≥30%。
- **token 预算 / 占比**：0.32 B / 3.1%。
- **打包方式**：flat + eos + int16，2048；按"每篇一个 eos"保证条目边界。
- **计划位置**：S1，占 5%；S5 replay 0.05 B。

#### P5 `code_ling_full` — 代码（中英混合，**单次全量**）
- **内容与组成**：`inclusionAI/Ling-Coder-SFT` **26 个 parquet 全量一次过**，产出 **2.15 B** token；**不再拆成 code/code2 两段**，避免旧方案的 1.64 B 重复。
- **来源**：`inclusionAI/Ling-Coder-SFT`，**apache-2.0**，6.26 GB / 1,400,000 行 / ≈2.15 B token，无需申请。
- **目的**：代码语法 + 跨行结构 + 中英注释解释；这是唯一能同时改善"代码能力"和"结构化推理"的单一来源。
- **预期表现 / 验收**：`E1.code_humaneval-lite`（本地 40 题 pass@1）≥8%；`E1.code_zh_comment`（中文注释续写 100 条）loss ≤1.60。**不再用训练 loss 作为验收**（旧方案 0.49 的教训）。
- **token 预算 / 占比**：2.15 B / 21.0%（相对旧方案的 6.13 B 砍掉 65%，因为旧方案里 2/3 是重复或过量）。
- **打包方式**：flat + **eos（每条记录一条）** + int16，2048；`--text-field auto` 但**必须显式排除** `tags/languages/messages` 之外的元数据列；分词前先按 `languages` 字段统计中英比，目标中文样本 ≥25%。
- **计划位置**：S2，与 P6 混（P5 81% / P6 19%），另 replay P1 0.15 B。

#### P6 `zh_code_prose` — 中文编程与代码问答
- **内容与组成**：① `BAAI/IndustryCorpus2_computer_programming_code`（`chinese/high` 优先）取 **0.30 B**；② `Mxode/Chinese-StackOverflow-QA-C_Language` **0.15 B**（198,335 条，37.8% CJK）。
- **来源**：前者 **apache-2.0**，按 doc 记录 3 个 parquet ≈0.74 GB / ≈0.3 B token；后者 **cc-by-sa-4.0**，0.51 GB / ≈150 M token，无需申请（但 CC-BY-SA 有传染性，发布需标注）。
- **目的**：中文技术语域（报错、API、调试对话），补 P5 里"代码为主、中文为辅"的偏斜。
- **预期表现 / 验收**：`E1.zh_tech_qa`（200 条中文技术问答）loss ≤1.90；中文技术词 perplexity 相对 S2 前下降 ≥15%。
- **token 预算 / 占比**：0.45 B / 4.4%。
- **打包方式**：flat + eos + int16，2048；SO 数据按"问/答"两条各一 eos（保留轮次边界）。
- **计划位置**：S2，19%。

#### P7 `math_zh_en` — 数学
- **内容与组成**：① `TiWu-Lab/math-zh` 取 **0.60 B**（13 片里取 4 片，**显式 `--text-field text`，禁止 auto**，否则会把 `tokens` 数值列拼进正文）；② `Mxode/CMID-Chinese_Math_Instruct` **0.20 B**；③ `AI-MO/NuminaMath-1.5` **0.10 B**（英文，只买概念覆盖）。
- **来源**：`TiWu-Lab/math-zh` **odc-by**，16.36 GB / 7,117,922 行 / ≈10 B token；`Mxode/CMID-Chinese_Math_Instruct` **cc-by-sa-4.0**，0.59 GB / ≈218 M；`AI-MO/NuminaMath-1.5` **apache-2.0**，0.53 GB / ≈814 M。均无需申请。
- **目的**：数学是"推理链长度"和"符号操作"的唯一来源；旧方案只给 0.25 B 且混了噪声，等于没练。
- **预期表现 / 验收**：`E1.math_zh_k12`（K-12 计算 300 题，精确匹配）≥35%；`E1.math_cot`（中文 CoT 完整率：答案前必须有推理段）≥70%。**同时监控 `E1.zh_wiki` 不退化 >0.05**（数学段最容易带偏语言分布）。
- **token 预算 / 占比**：0.90 B / 8.8%。
- **打包方式**：flat + eos（每题一条）+ int16，2048；**保留原始公式字符**（不做 LaTeX 归一化，与词表一致）。
- **计划位置**：S3（与 P8 混），另在 S5 replay 0.10 B。

#### P8 `zh_science` — 中文科学/学科指令
- **内容与组成**：`BAAI/IndustryInstruction_Subject-Education` **0.13 B**（357,048 条）。
- **来源**：同上，**apache-2.0**，0.49 GB / ≈134 M token，无需申请。
- **目的**：把 P2 的教材知识转成"被提问"的形态，衔接 S5 的对话退火。
- **预期表现 / 验收**：`E1.sci_qa` 准确率相对 S1 末 +5 pt。
- **token 预算 / 占比**：0.13 B / 1.3%。
- **打包方式**：packed 2049 + wcode（question=2 权重 0.1，answer=1）；也可 flat+eos 作为 LM 数据，本方案选 packed 以保留指令形态。
- **计划位置**：S3，13%。

#### P9 `long_ctx_8k` — 长上下文（**重排包，不是新语料**）
- **内容与组成**：从 P1 抽 0.30 B、P5 抽 0.20 B、P2 抽 0.05 B、P10 抽 0.10 B，按**文档边界**拼成 8192 窗口（窗口内可以有多个文档，文档间必须有 eos；**禁止**跨文档切断单个样本，除非样本本身 >8192）。
- **来源**：同 P1/P2/P5/P10（复用，不新增语料，故不计入唯一 token）。
- **目的**：在不引入新分布的前提下把窗口从 2 K 拉到 8 K；旧方案 v22 用"无边界代码流按 16385 裸切"做长上下文，学到的是"在任意位置继续写代码"。
- **预期表现 / 验收**：`E1.long_needle`（8 K 内 5 个事实的检索，位置均匀分布）≥60%；`E1.long_doc_ppl`（整篇 6–8 K 文档的 loss）相对 2 K 模型的同文档 loss 下降 ≥8%。**并检查 `E1.zh_wiki` 不回退**。
- **token 预算 / 占比**：0.65 B（消费量），唯一 token 0。
- **打包方式**：`int16 (rows, 8193)` packed，`seq_len=8192`，每行以 `<bos>` 开头、文档间 `<eos>`；训练时 `batch 4 / 卡 × 2 卡 = 65,536 token/步`（与 2 K 同吞吐）。
- **计划位置**：S4，单独一段，lr 降到 2e-5。

#### P10 `chat_sft_v2` — 对话与指令（**替代 v13**）
- **内容与组成**：① 从 `chat_pack_v13` 里**只保留经过清洗的部分**：去掉 48.4% 的跨样本行、重打包成"一样本一行不跨行"、pad 降到 <0.5%，得 **0.36 B**；② `a_grade` + `extra_sft` + `extra_sft2` 里去重、去模板化后的指令数据 **0.16 B**；③ 新增 `BAAI/IndustryInstruction_Subject-Education` 的对话视图 0.06 B。
- **来源**：v13 上游（R1-110k、i_wonder_why、ruozhiba、LCCC、COIG、firefly、belle、sharegpt4、ultrachat 等，许可混杂，需逐项在 `docs/05` 核）；`a_grade`/`extra_sft*` 为本地采集。
- **目的**：对话能力是"最终交付形态"，但**不能靠 5 个 epoch 记住 552 M 模板**。做法是：**降量（0.58 B）+ 提质（去截断、去跨行）+ 与预训练 replay 混合**，让对话段只占最终分布的 15–20%。
- **预期表现 / 验收**：`E1.chat_quality`（人工 100 条盲评：相关性/格式/幻觉，1–5 分）≥3.8；`E1.chat_holdout`（与训练集**doc 级不相交**的 2,000 条）loss ≤1.60；**关键**：`E1.zh_wiki` 相对 S3 末退化 ≤0.10（旧方案退化 0.94）。
- **token 预算 / 占比**：0.58 B / 5.7%（旧方案：v13 552 M × 5 轮 = 2.76 B 消费，占实际训练 token 的 26%）。
- **打包方式**：packed `(rows, 2049)` + `uint8` wcode（0 pad / 1 reply / 2 prompt@0.1 / 3 plain@1.0），**每个样本独立成行、不足补 pad、绝不跨行**；`seq_len=2048`。
  **前置 schema 归一化（必须做，否则重蹈 §3 C7）**：① 对 `conversations`/`dialog`/`pairs` 这类"字符串化 list"先 `json.loads` 再 `ast.literal_eval` 兜底；② 键名归一（`INSTRUCTION/RESPONSE` → `instruction/output`，`from/value` → `role/content`）；③ 白名单取字段（只允许 `instruction/input/output/messages/role/content/text`），其余列（`metadata`/`copyright`/`answer_from`/`human_verified`/`id`/`url`）一律丢弃；④ 同一 instruction 的 MD5 去重 + 输出 <20 token 的样本丢弃。
- **计划位置**：S5 主体（0.42 B）+ S6 收尾（0.16 B）。

#### P11 `safety_refusal` — 安全与拒答
- **内容与组成**：自建 8,000 条中文安全/拒答/边界样本（**0.02 B**），来源为公开安全数据集改写 + 人工审核。
- **来源**：`PKU-Alignment/BeaverTails`（**apache-2.0**，需筛中文子集）+ 自建。
- **目的**：避免"什么都说"和"什么都不敢说"两个极端；这类样本绝不能多，0.2% 足够。
- **预期表现 / 验收**：`E1.safety`（100 条越狱/有害请求）拒答率 ≥85%，同时 `E1.chat_quality` 不因过度拒答掉分（正常请求误拒 ≤5%）。
- **token 预算 / 占比**：0.02 B / 0.2%。
- **打包方式**：packed 2049 + wcode（prompt 权重 0.1，reply 1.0）。
- **计划位置**：S5 末 + S6 各一半。

#### E1 `eval_holdout_v2` — 评测集（**永不进训练**）
- **内容与组成**：中文网页 20 M（从 ChineseWebText2.0 的**不同 tar 分片**取，做 13-gram 去重）、中文百科 8 M、教材 5 M、英文 5 M、代码 10 M（Ling-Coder 里**留出 3 个 parquet 不参与训练**）、数学 6 M、对话 2,000 条人工核验。
- **目的**：把"训练 loss 好看"和"模型真的好"分开。旧方案的 val_dir 就是训练集本身。
- **验收方式**：每个阶段结束、每 1,200 步都跑；**任何阶段让总分恶化 >0.05 nats 连续 2 次 → 回滚到该段起点 ckpt 并终止该段**。
- **隔离检查（上线前必跑）**：对 E1 的每条样本算 13-gram，与全部训练包做交集，命中率必须 <0.1%；否则重采样。

### 4.3 混合 / replay 策略

| 规则 | 做法 |
|---|---|
| 段内混合 | 同一段内的包按 token 比例**逐 batch 交错**（不是先 A 后 B），比例由 §4.1 给定 |
| 跨段 replay | 每段固定混入 **10–20% 的"上一阶段代表包"**：S2 混 P1 6%、S3 混 P1 20%、S4 混 P10 15%、S5 混 P1 37%+P3 22%+P5 7%、S6 混 P1/P3 40% |
| 唯一 token 记账 | 建 `data_cache/_ledger.json`：每个包记录 `source_url / sha256 / tokens / 是否已消费 / 消费于哪些段`。**同一 sha256 第二次出现必须报警**（旧方案缺这一步才让 1.64 B 重复混过去） |
| 采样权重 | packed 段用 wcode；flat 段 weight 恒 1。**不允许**用 `weight` 参数替代掩码 |
| 顺序 | 底座 → 代码 → 数理 → 长上下文 → 对话退火 → SFT；**代码/数理必须早于对话**，对话只做退火，不做主体 |
| 每段学习率 | S1 峰 1.1e-4；S2 8e-5；S3 6e-5；S4 2e-5；S5 3e-5；S6 1.5e-5，cosine 到 1.5e-6。**总步数在开跑后冻结**，热改只允许改 `data_dir`，不允许改 `steps` |

### 4.4 GC 与保留策略

| 对象 | 策略 |
|---|---|
| 原始 parquet（`data/hf/*`） | **永不删**；磁盘不够就先扩盘（43 GB→建议 250 GB），原始数据是复现的唯一来源 |
| 分词后的 flat 分片 | 该段消费完 **不删**，移到 `data_cache/_archive/`；只有确认"该段 holdout 与探针达标"后才允许删 |
| packed 包 | 永不 GC（体积小、复用率高：P10 在 S5/S6 都用） |
| 检查点 | `save_steps=1200`、`save_total_limit=6`；**另存 best-by-holdout 一份**，不受 limit 约束 |
| 触发 GC 的条件 | 只剩一个：`min_free_gb < 30` 且**已有 best ckpt 归档**。任何 GC 必须写 `logs/gc_audit.jsonl`（时间、段、删除文件、释放 GB、当时 holdout） |

### 4.5 成本估算

**单位成本（实测口径）**：2 卡 DDP 有效速率 1.607 it/s × 65,536 token/步 = **105,316 tok/s = 0.379 B token/h**；¥13.97/h → **¥36.85 / B token**（含 eval 开销；若加 10% 重跑/回滚余量 → **¥40.5 / B token**）。

| 段 | token | 步数（65,536/步） | 小时 | 费用 ¥ |
|---|---:|---:|---:|---:|
| S0 冒烟（100 步 + 健康检查） | 0.02 B | 305 | 0.05 | 0.7 |
| S1 中文底座（P1+P2+P3+P4） | 4.30 B | 65,613 | 11.34 | 158.4 |
| S2 代码（P5+P6+replay P1/P3） | 2.35 B | 35,858 | 6.20 | 86.6 |
| S3 数学/科学（P7+P8+replay P1） | 1.00 B | 15,259 | 2.64 | 36.8 |
| S4 长上下文 8K（P9+replay P10） | 0.75 B | 11,444 | 1.98 | 27.6 |
| S5 对话退火（P10+P11+replay P1/P3/P5/P7） | 1.35 B | 20,599 | 3.56 | 49.7 |
| S6 SFT 收尾（P10 子集+P11+replay） | 0.60 B | 9,155 | 1.58 | 22.1 |
| **合计** | **10.37 B** | **158,233** | **27.35 h** | **¥382.1** |

- 唯一 token **10.25 B**（计划实际消费 9.5 B 唯一 + 0.87 B replay）→ **18.8 token/参数**（504,179,896 参数）。坦率地低于 Chinchilla 20，但比旧方案"声称 20、真实 16.8"更可信；若余额允许，优先把 P1 补到 5.0 B 而非再加代码。
- 对照旧方案：159,857 步 × 65,536 = **10.476 B 消费**，27.63 h ≈ **¥386**，真实唯一 ≈8.45 B（16.8 tok/param）。

**非 GPU 成本（必须一起算）**：

| 项 | 量 | 单价假设 | 费用 |
|---|---|---|---|
| 新语料分词（9.3 B token，4 进程） | 0.2–1.2 M tok/s/进程（doc27 实测） | 4 进程 ≈3.5 h 本地/CPU 实例 | ¥0（本机）～¥20（按 ¥5/h 租 CPU） |
| 新增磁盘 | flat int16 ≈18.6 GB；packed ≈12 GB；E1 0.1 GB | 250 GB 数据盘 | 按月计，需提前扩盘 |
| 去重/13-gram 隔离检查 | 10.25 B token 的 13-gram 哈希，Bloom 过滤 | 约 2–3 h CPU + 12 GB 内存 | ¥0–15 |
| 失败/回滚重跑余量 | 10% | — | ¥38 |

**总预算：¥382（训练）+ ¥38（余量）+ ¥35（分词/检查/CPU）= ≈¥455**；按 ¥/B token 口径 = **¥36.9（纯训练）/ ¥44.3（全口径）**。

---

## 5. 阶段顺序与预算表（执行版）

| 段 | 段名 | 数据（比例） | token | 步数 | lr | 切换条件 | 验收门 |
|---|---|---|---:|---:|---|---|---|
| S0 | smoke | P1 前 200 万 token | 0.02 B | 305 | 1.1e-4 | 100 步 loss 单调降、无 NaN、ckpt 可载 | 能续训 |
| S1 | base_zh | P1 67% / P2 15% / P3 12% / P4 5% | 4.30 B | 65,613 | 1.1e-4→3e-5 | 步数用尽 **且** E1 总分相对 S1 起点降 ≥0.30 | `E1.zh_wiki` ≤3.10 |
| S2 | code | P5 81% / P6 19% + replay P1 6% + P3 末尾 13% | 2.35 B | 35,858 | 8e-5→2e-5 | 同上；**额外**：`E1.code_humaneval-lite` ≥8% | `E1.zh_wiki` 退化 ≤0.08 |
| S3 | math_sci | P7 70% / P8 13% + replay P1 20% | 1.00 B | 15,259 | 6e-5→1.5e-5 | 同 S1 | `E1.math_zh_k12` ≥35% |
| S4 | ctx8k | P9（P1 46%/P5 31%/P10 15%/P2 8%）+ replay P10 15% | 0.75 B | 11,444 | 2e-5→5e-6 | `E1.long_needle` ≥60% | `E1.zh_wiki` 退化 ≤0.05 |
| S5 | chat_anneal | P10 31% / replay P1 37% / P3 22% / P5 7% / P7 7% / P11 1% | 1.35 B | 20,599 | 3e-5→1e-5 | `E1.chat_quality` ≥3.8 | `E1.zh_wiki` 退化 ≤0.10（硬门） |
| S6 | sft_finish | P10 27% / P11 3% / replay P1+P3 40% / P5 20% / P7 10% | 0.60 B | 9,155 | 1.5e-5→1.5e-6 | 最终人工盲评 | 全面回归 + 打包发布 |

**停机规则（写进 `tasks/plan_v3.json` 的 `stop_rules`）**
1. 任一阶段内，`E1` 总分连续 2 次评测恶化且累计 >0.05 nats → 回滚到该段起点 ckpt，**跳过该段剩余步数**，进入下一段。
2. `E1.chat_quality` 人工分 <3.5 → 禁止进入 S6。
3. 13-gram 重叠 >0.1% → 立即停训，重采 E1。
4. 磁盘 <30 GB 且无 best ckpt 归档 → 硬停（不是告警）。

---

## 6. 风险与没验证的部分

**已验证（实测）**：目录/文件/体积、124 个 npy 的 dtype 与宽度、pretrain_v2 的 eos/bos 与 doc 数、v13 的 wcode 直方图与 pad 数、12 个 pack 的长度与冗余、`hf_pack_*` 已不存在、计划文件原文、构建脚本的 `--add-eos` 缺失、code_ling/code2 同 glob、曲线 8,111 行与 holdout 4 个点。

**没验证 / 只能标注为推断**：
1. **code_ling 与 code2 的实际重叠 token 数**：两个目录已被 GC 删除，我无法做 sha256 或内容比对。1.64 B 是"code_ling 上限 + 同一 glob 顺序"推出的下界，真实重叠可能更大（若 code_ling 实际只分到 1.0 B）。
2. **`hf_pack_math` 是否真混入 `tokens` 数值列**：包已删，只能从 `auto_text_fields` 的 skip 列表与 doc28 的列说明推断。
3. **`pretrain_v2` 是否做过跨源去重**：我没有读 `prepare_pretrain_v2.py`；仅知各源 doc 数与 manifest 一致。跨源重复（如 wiki 与千问语料）未排查。
4. **`holdout_eval.jsonl` 的 5 个 val 文件与训练集是否真的不相交**：`*_val.npy` 是独立切片，但没有去重脚本的证据；若它们是从同一 doc 池子里随机切的，那 holdout 的绝对值（3.5 vs 2.8）都要打折扣。
5. **`epoch` 字段的分母**：我从 `epoch/step` 速率反推出 2.306 B / 4.998 B / 0.614 B 三个分母，并在 2.306 B（=1,126,036 行 pretrain）与 0.614 B（=300,000 行 sft_mix）上与计划/build 脚本默认值逐位对上；但"为什么只在 70,000 和 146,000 刷新"我没有读 `train.py` 的 dataloader 重建逻辑，机制未证。
6. **HF 元数据**：P1–P8 的规模/许可/行数来自 `docs/28`（该文档自述用 HF API + datasets-server 实测），我本次**没有**重新调用 HF API 复核；gated 状态、许可变更需在执行前用 `HfApi().dataset_info` 再确认一次。特别是 `opencsg/Fineweb-Edu-Chinese-V3` 的 "other（OpenCSG 协议）" 与 `Mxode/*` 的 CC-BY-SA 传染性，商用前必须人工过一遍。
7. **速率假设**：¥36.85/B token 用的是 `plan_ddp2._readme` 的 1.607 it/s（含 eval）。S4 的 8 K 窗口在同样的 65,536 token/步下是否仍是 1.6 it/s，没有测过（v22 的设计注释声称"同吞吐"，未实测）。
8. **12 个 chat_pack 的"唯一 552 M"**：我只验证了长度与 v13 = v9 重切 + 新源的谱系；v4–v9 之间是否 100% 包含关系是【文档转述】doc27 的结论。
9. **`_resume_probe`、`nonhf` 里 2,318 个小文件**我只统计了数量与体积，没有逐个判断是否已被 `poetry_zh` 消化。

**已知的取舍（不是风险，是选择）**：
- 新方案把代码从 6.13 B 砍到 2.60 B：如果最终目标是"代码能力优先"，应把 P1 的 0.9 B 转给 P5，并接受中文网页 loss 上升。
- 新方案唯一 token 18.8/参数，低于 Chinchilla 20；在这张卡的钱/时间约束下，这是可辩护的（旧方案名义 20、真实 16.8，且 45% 消费在重复代码上）。
- 20k 词表对代码与英文都不划算（doc28 实测中文 3.51 bytes/token，英文约 5）——**换 32k/48k 词表是另一份报告的事**，本方案沿用 20k 以保证与现有 checkpoint 兼容；若重新开始训练（不续训），应优先做词表重训，这一步能省 15–20% 的 token 预算。
