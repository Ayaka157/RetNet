# D · 数据集编排审计与重设计

> 审计对象：`E:\Workspaces\RetNet`（RetNet 0.5B，504,179,896 参数，seq_len 2048，20,000 BPE）
> 审计时间：2026-10-20 · 方式：直接读 `.npy` 头 + 读训练日志 + 复算曲线，**不采信文档数字而重数**
> 标注约定：**【实测】**＝本次从文件/日志/命令行直接得到；**【文档转述】**＝仓库文档里写的、本次未独立复核；**【推断】**＝由实测数据推理得出，未直接观测
> 未读 `docs/analysis/` 下任何文件；未修改任何已有文件；未启动任何训练。

---

## 1. 结论摘要

1. **磁盘上的 8.37 B token 里，真正不重复的只有约 3.0 B**：`pretrain_v2` 2.817 B（其中 skypile 0.504 B 被 `exclude` 剔除、从未训练）＋ `chat_pack_v13` 0.552 B；14 个 chat pack 共 4.109 B token 是同一份内容的 7.4 次重切，`pretrain_merged` 的 1.317 B 是 v15 时代的旧合并流、没有任何计划引用。【实测】
2. **最严重的单个错误是代码段 4.49 B token 是重复数据**：`code_ling`(2.499 B) + `code_ling2`(4.997 B) 合计 7.50 B token，而来源 `inclusionAI/Ling-Coder-SFT` 全量只有 2.15 B token（文档实测值）→ 至少 3.5 倍重复，占总训练预算 8.0 B 的 56%。训练损失给出了独立佐证：code 段平均 loss **0.52**（p10 0.49），而同样规模的 `prog_zh` 段是 **3.29**——同一模型在两个"代码类"语料上差 2.8 nat，只可能是前者被记住了。【实测+推断】
3. **顺序编排让模型先坏后修**：在 chat_pack_v13 验证集上，eval_loss 从 step 2,500 的 **2.5995** 单调涨到 step 142,800 的 **4.5517**（+1.95 nat，+75%），直到最后 14,000 步 `sft_mix` 才降到 2.4450。也就是 90% 的训练时间里，对话能力处于被系统性破坏的状态。【实测】
4. **域内 held-out 直接证明后段训练是负收益**：同一 held-out 口径（5 文件 / 4,454,400 token）下，`v21_step17500` = **2.8471**，`v21_final` = **3.5023**，多训 14 万步反而退化 **+0.655 nat**；其中 wikipedia_zh 从 2.7911 → 3.7319（+0.94）。而基线 `ep3` 的 chinese_fineweb 是 3.8697、v21_final 是 3.8605——**5 B token 的增量训练在该指标上净收益为零**。【实测】
5. **规划机制本身是主要缺陷源**：步数不是由数据量或收敛决定，而是由账户余额倒推，`plan_ddp2.json` 的 `_readme` 里留下了 **8 次**计划热改记录（60,955→79,000→76,000→148,000→156,057→159,857，cosine 总步数被拉长两次）；`min_shards: 10` 曾让第 1 段永远不就绪（文件自己写了这个坑）；`gc: true` 把 `pretrain_v2` 的 46 个分片（4.4 GB）和全部 `hf_pack_*` 删干净，导致今天无法复核任何一个新语料包的真实内容。【实测】

---

## 2. 现状盘点

### 2.1 总账【实测】

- `data_cache/` 下 **24 个子目录**、**124 个 `.npy`**、**2,318 个非 npy 文件**（几乎全在 `nonhf/`），磁盘 **43.28 GiB**（逐目录求和，与 `docs/27` 的 43.28 GB 一致）。
- 可被训练读取的 token 流（只数 ids 流，不重复计 wcode 数组）：

| 组 | token | 文件 | dtype / 形状 |
|---|---:|---:|---|
| `pretrain_v2` | 2,816,814,321 | 62 | `<i2` 一维 flat（含 5 个 `*_val` 分片） |
| `pretrain_merged` | 1,317,371,268 | 2 | `<i8` 一维 flat（int64，8 B/token） |
| `chat_pack_v3…v13` + 2 个 mix | 4,108,801,299 | 28 | `<i2` `(N, seq+1)` + `wcode` `|u1` |
| `wikitext103_v16k` / `wikitext2` | 124,584,057 / 3,031,324 | 2 / 2 | `<i8` 一维 |
| **合计** | **8,370,602,269** | 124 | — |

### 2.2 `pretrain_v2` 分来源【实测】

行宽 = 无（一维流）。**有文档边界**：`scripts/prepare_pretrain_v2.py:118-131` 对每个文档 `append(bos) → encode(text) → append(eos)`，所以流里是 `<bos> … <eos><bos> … <eos>`。

| 来源 | train | val | 文件 | HF id |
|---|---:|---:|---:|---|
| `chinese_fineweb_edu_v2` | 1,013,826,215 | 0 | 20 | `opencsg/Chinese-FineWeb-Edu-V2` |
| `fineweb_edu` | 504,765,133 | 1,079,414 | 11 | `HuggingFaceFW/fineweb-edu` |
| `skypile` | 504,097,083 | 1,007,069 | 11 | `Skywork/SkyPile-150B`（**计划里 `exclude` 掉了，0 步训练过**） |
| `cosmopedia` | 403,355,751 | 811,385 | 9 | `HuggingFaceTB/smollm-corpus` (cosmopedia-v2) |
| `wikipedia_zh` | 202,958,320 | 370,375 | 5 | `wikimedia/wikipedia` 20231101.zh |
| `open_web_math` | 151,922,835 | 299,221 | 4 | `open-web-math/open-web-math` |
| `poetry_zh` | 30,418,286 | 0 | 1 | chinese-poetry（GitHub） |
| `chinese_fineweb_edu`（v1） | 0 | 1,903,234 | 1 | 只剩 val，train 已不存在 |

`_FlatStage._scan` 过滤掉名字含 `val` 的分片，所以训练实际可用 = **2,311,710,169 token**；日志里 `rows=1126011 × 2048 = 2,306,070,528`（余数被丢弃），与计划注释的 2,307,246,540 相差 1.2 M（0.05%）。

### 2.3 每个包一行【实测】

| 包 | 内容 | token | 文件 | 宽度 | dtype | eos/文档边界 | packed/flat | 在计划里的角色 |
|---|---|---:|---:|---|---|---|---|---|
| `pretrain_v2` | 7 源混合，见上表 | 2,816,814,321 | 62 | 一维 | i2 | **有**（bos+eos/文档） | flat | v21 Stage0，35,205 步 / 1.00 epoch |
| `pretrain_merged` | v15 时代旧合并流 | 1,317,371,268 | 2 | 一维 | **i8** | 有（同 v1 管线） | flat | **无任何计划引用（孤儿）** |
| `chat_pack_v3` | 15 源闲聊/指令 + pretrain_replay 38.9 M | 195,476,085 | 4 | 513 | i2 / u1 | 有（bos…eos/对话） | packed 512 | 历史 |
| `chat_pack_v4` | 26 源，加入 haruhi 系 | 217,814,670 | 4 | 513 | i2 / u1 | 有 | packed 512 | 历史（rp 占 37%） |
| `chat_pack_v5` | +style_rp | 234,604,134 | 4 | 513 | i2 / u1 | 有 | packed 512 | 历史（rp 41%） |
| `chat_pack_v6` | 修截断 | 210,495,699 | 4 | 513 | i2 / u1 | 有 | packed 512 | 历史 |
| `chat_pack_v7` | +cornell_rp / wikisource_rp | 223,934,247 | 4 | 513 | i2 / u1 | 有 | packed 512 | 历史 |
| `chat_pack_v8` | +gutenberg_rp | 225,667,674 | 4 | 513 | i2 / u1 | 有 | packed 512 | 历史 |
| `chat_pack_v9` | +a_grade 5 源（anime/rp 为主） | 281,386,143 | 4 | 513 | i2 / u1 | 有 | packed 512 | v13 的基座 |
| `chat_pack_v10` | v9 + R1(1200 字符截断) + i_wonder_why | 483,809,274 | 4 | 513 | i2 / u1 | 有 | packed 512 | 历史 |
| `chat_pack_v11` | v10 重切 8K | 443,904,933 | 4 | **8193** | i2 / u1 | **被重切破坏** | packed 8K | 历史 |
| `chat_pack_v11_4k` | v10 重切 4K | 443,959,114 | 4 | **4097** | i2 / u1 | 被重切破坏 | packed 4K | 历史 |
| `chat_pack_v12` | v10 + `strip_think` | 473,156,829 | 4 | 513 | i2 / u1 | 有 | packed 512 | 历史（把 CoT 删了） |
| **`chat_pack_v13`** | v9 重切 2048 + R1 完整 CoT + 百科 | **552,330,489** | 4 | **2049** | i2 / u1 | **跨行重叠/丢边界** | packed 2048 | v21 `val_dir`、退火段 |
| `chat_recovery_mix` | 恢复用混合 | 83,285,788 | 2 | 一维 | i8 | 有 | flat | 历史 |
| `chat_sft_mix` | 旧 SFT 混合 | 37,269,403 | 2 | 一维 | i8 | 有 | flat | 历史 |
| `a_grade`（13 jsonl） | anime/rp/CoT 原始文本 | 未分词（5.06 GiB） | 13 | — | — | — | — | 经 v9 进过训练 |
| `roleplay`（14 jsonl） | rp 原始文本 | 未分词（4.68 GiB） | 14 | — | — | — | — | 经 v4–v9 进过训练 |
| `literature`（10 jsonl） | wikisource/gutenberg | 未分词（1.40 GiB） | 10 | — | — | — | — | 经 v7–v9 进过训练 |
| `extra_sft` / `extra_sft2` | belle/coig/sharegpt/ultrachat/firefly/zhihu | 未分词（1.34 GiB） | 13 | — | — | — | — | 经 v3–v9 进过训练 |
| `chat_corpus` | lccc + wiki-cn 原始 | 未分词（1.90 GiB） | 3 | — | — | — | — | v3 之前 |
| `nonhf` | 2,318 个小文件 | 未分词（0.46 GiB） | 2,318 | — | — | — | — | 未进计划 |
| `wikitext103_v16k` | wikitext-103 | 124,584,057 | 2 | 一维 | i8 | 有 | flat | 早期实验 |
| `wikitext2` | wikitext-2 | 3,031,324 | 2 | 一维 | i8 | 有 | flat | 早期实验 |
| `_resume_probe` | 断点测试 | 17,028 | 4 | 129 | i2/u1 | — | packed | 测试 |

### 2.4 三个计划的真实内容【实测】

**`tasks/plan.json`**（plan_version 2，"唯一 token 账"）：5 段，总 143,211 步。
| # | 段 | 数据 | mode | 步数 | `min_shards` | `gc` | `exclude` |
|---|---|---|---|---:|---:|---|---|
| 0 | pretrain_v2_noskypile | `pretrain_v2` | flat | 70,411 | 46 | ✅ | `skypile_` |
| 1 | chinese_web_hq | `hf_pack_chinese_web` | flat | 14,000 | **10** | ✅ | — |
| 2 | code_ling | `hf_pack_code` | flat | 47,300 | 20 | ✅ | — |
| 3 | textbook | `hf_pack_textbook` | flat | 5,500 | 1 | ✅ | — |
| 4 | chat_anneal | `chat_pack_v13` | packed | 6,000 | — | ❌ | — |

**`tasks/plan_ddp2.json`**（plan_version 13，实际执行的那份）：8 段，159,857 步。step 边界与 `logs/linux/v21_ddp.out` 的 `[stage]` 行完全对上（35,205 / 42,205 / 67,250 / 70,000 / 138,557 / 142,057 / 145,857）。
| # | 段 | 数据 | mode | 步数 | `min_shards` | `gc` |
|---|---|---|---|---:|---:|---|
| 0 | pretrain_v2_noskypile | `pretrain_v2` | flat | 35,205 | 46 | ✅（实删 4.6 GB） |
| 1 | chinese_web_hq | `hf_pack_chinese_web` | flat | 7,000 | **1** | ✅（删 0.47 GB） |
| 2 | code_ling | `hf_pack_code` | flat | 25,045 | 10 | ✅（删 5.0 GB） |
| 3 | textbook | `hf_pack_textbook` | flat | 2,750 | 1 | ✅（删 0.42 GB） |
| 4 | code_ling2 | `hf_pack_code2` | flat | 68,557 | 10 | ❌ |
| 5 | prog_zh | `hf_pack_prog` | flat | 3,500 | 1 | ✅（删 0.65 GB） |
| 6 | math_zh | `hf_pack_math` | flat | 3,800 | 1 | ✅（删 0.35 GB） |
| 7 | sft_mix | `hf_pack_sft` | packed | 14,000 | — | ❌ |

`min_shards` 的真实作用（`retnet_hf/data.py:257-272`）：`shard_count < min_shards` 时返回"未就绪"，训练**停在当前段每 200 步重试**。所以 `plan.json` 里 `chinese_web` 的 `min_shards:10` 而实际只产出 1 个分片，会把训练永久卡在 35,205 步——`plan_ddp2.json` 的 `_note` 自己也记了这个坑（"写 10 会让段 1 永远不就绪"）。

**`tasks/v21_plan_ddp2.json`**（超参文件，不是阶段表）：`nproc=2`、`batch_size=16/卡`、`grad_accum=1`、`seq_len=2048` → **65,536 token/步**；`mode=chunkwise`、`chunk_size=128`、`optim=adamw_bnb_8bit`、`compile=max-autotune-no-cudagraphs`、`precision=bf16`。

### 2.5 五个构建脚本实际做了什么【实测】

| 脚本 | 实际行为 | 关键缺陷 |
|---|---|---|
| `scripts/build_flat_corpus.py` | parquet/jsonl → 一维 int16 flat 分片（默认 2.5e8 token/片）+ `meta.json`；`--text-field auto` 用字段名黑名单（id/url/score/license/date…）猜正文；`flatten_value` 把对话 list 压成 `role: content` 纯文本 | **`--add-eos` 默认关闭**，所以它产出的每个包**没有任何文档边界**；`auto` 会把多列无条件 `\n\n` 拼在一起 |
| `scripts/build_sft_mix.py` | 旧对话 40% + code2 35% + textbook 15% + math 10%（**按行数**，行宽相同故等价按 token）；从 flat 分片里随机切行、`wcode=1` | 40% 的"对话"桶里 `chat_pack_v13` 本身含 21.5% 的 `i_wonder_why` 百科和 26.8% 的 R1 CoT，**桶名与内容不符**；flat 来源整行标 1（全算 reply） |
| `scripts/build_ctx_mix.py` | code2 60% + sft 25% + math 10% + prog 5% 循环供料 → 16K 扩展包；从 packed 转来的部分**剥 pad 后补 `<eos>`(id=2)** | 代码注释直接承认"flat 分片的原始边界在分词时就丢了，无法恢复"；每个来源 `while True` 循环，**数据不够预算就静默重复** |
| `scripts/build_chat_mix_v13.py` | v9 的 `(N,513)` 行**先展平、丢掉 `wcode==0` 的 pad、再按 2049 重切**；R1 用 4,000 字符上限重编码（v10/v12 是 1,200） | 展平重切必然让对话跨行：一个对话被切成两半、下一行以半句话开头；**边界与 wcode 对齐被破坏** |
| `scripts/_data_supervisor.py` | 校验下载字节数→跑分词→**校验通过后 `rmtree` 原始 parquet**→`gc` 时删分片 | 原始语料与分片**都不留**，事后无法复核 `--text-field auto` 到底取了哪些列；`SPECS` 里只登记 code/web/textbook 三个包，`hf_pack_code2/prog/math/sft` 不在这条流水线上（来源不明） |

### 2.6 文档与实测的对照

| 文档说法 | 实测 | 判定 |
|---|---|---|
| `docs/27`：43.28 GB / 26 目录 / 2,532 文件 | 43.28 GiB / **24 目录** / 124 npy + 2,318 其他 | 体积对，目录/文件数对不上（`--` 可能是把根下 `dataset_survey.json` 也数进去了） |
| `docs/27`：已 token 化 8.47 B | **8.37 B**（只算 ids 流） | 差 0.1 B，可接受（wcode 计法不同） |
| `docs/27`：12 个 chat pack 重复系数 **7.2×** | 14 个 pack 共 4.109 B / 唯一 0.552 B = **7.4×** | ✅ 一致 |
| `docs/05`：Chinese-FineWeb-Edu-V2 = 1,014 M | 1,013,826,215 | ✅ |
| `docs/05`：SkyPile = 504 M | 504,097,083 + 1,007,069 val | ✅ |
| `docs/28`：Ling-Coder-SFT 全量 **2.15 B token** / 6.26 GB | 实际建了 2.499 B + 4.997 B = **7.50 B** | ❌ **超建 3.5 倍**（见 3.2） |
| `docs/28`：`TiWu-Lab/math-zh` 4 片 ≈ 3.7 B token | 计划只用 0.25 B（`math_zh` 段 3,800 步） | 只用掉 6.8% |

---

## 3. 编排不足（逐条 + 证据）

### 3.1 配比：一个动漫角色的同人扩写占了 SFT 的三分之一【实测】

`chat_pack_v4` 的 `meta.json`：`haruhi_expand` 54,501,224 token（340,695 docs）+ `haruhi_54k` 13,442,878 + `haruhi_zero_pippa` 614,616 = **68.6 M / 217.8 M = 31.5%**。到 `v9`（v13 的基座）仍是 43.1+11.6+0.3 = **55.0 M / 281.4 M = 19.6%**，加上 `coser_rp`、`style_rp`、`limarp_aug` 等，`by_kind.rp` = **130.97 M / 281.4 M = 46.5%**。
后果：模型的行为分布被单一角色/单一文风拉走。`logs/samples_v21_final.txt` 与 `logs/play_state.json` 里出现固定角色口癖，是这条配比的直接产物（此条为【推断】，未逐条统计采样文本）。

### 3.2 配比：代码段 56% 的预算是同一份数据的第 2–3 遍【实测+推断】

- 日志：`[stage] 打开阶段 2「code_ling」mode=flat rows=1220101` → 1,220,101 × 2048 = **2.499 B**；`阶段 4「code_ling2」rows=2440200` → **4.997 B**。【实测】
- 来源：`inclusionAI/Ling-Coder-SFT`，apache-2.0，26 parquet / 140 万行 / 6.26 GB，`docs/28` 用项目自己的 tokenizer 实测 **2.15 B token**。【文档转述】
- 7.50 B ÷ 2.15 B = **3.49 倍**。即 `code_ling` 与 `code_ling2` 是同一批文档的不同截取，第二段基本是在复读第一段。【推断，但算术上无可回避：除非 docs/28 的 2.15 B 错了 3.5 倍】
- 独立佐证（训练损失）：code_ling 段 loss 均值 **0.6537**、p10 0.578；code_ling2 段均值 **0.5202**、p10 0.4926。0.5 B 参数模型在真实源码上不可能做到 0.49 的逐 token 交叉熵（≈61% 单 token 命中）。而同为代码类中文语料的 `prog_zh` 段是 **3.2854**。两个"代码"语料差 2.8 nat，只能是前者被记住。【实测】
- 代价：8.0 B 训练预算里 4.49 B（56%）花在重复样本上；`docs/28` 自己算的"17.6 token/参数"因此是虚的。

### 3.3 顺序：对话能力先被摧毁 1.95 nat，最后 9% 的步数才修【实测】

`val_dir` 全程固定为 `chat_pack_v13`（日志：`[plan] 验证集固定用 … chat_pack_v13`），所以 `eval_loss` 序列同口径可比：

| step | 2,500 | 15,000 | 33,000 | 42,000 | 68,400 | 100,800 | 142,800 | 146,400 | 159,600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eval_loss | **2.5995** | 3.0496 | 3.1179 | 3.9508 | 4.4054 | **4.4911** | **4.5517** | 3.0178 | **2.4450** |

从 2.60 一路涨到 4.55（+1.95 nat / +75%），在 100,800 步之后基本饱和（4.49→4.55），最后 14,000 步 `sft_mix` 才把它压回 2.45。问题是：预训练段的配比里**没有任何对话锚点**（`pretrain_v2` 7 个源全是网页/百科/数学），而后段 4.49 B 的代码把分布推得更远。

### 3.4 顺序：后 6 万步是净负收益（域内 held-out）【实测】

`logs/holdout_eval.jsonl`，取 token 数完全相同的三次（4,454,400）：

| tag | 加权 | chinese_fineweb_edu_v2 | cosmopedia | fineweb_edu | open_web_math | wikipedia_zh |
|---|---:|---:|---:|---:|---:|---:|
| `v21_step15000` | 2.8666 | 3.2353 | 1.9987 | 2.9595 | 2.6005 | 2.8145 |
| `v21_step17500` | **2.8471** | 3.2194 | 1.9740 | 2.9391 | 2.5796 | 2.7911 |
| `v21_final` | **3.5023** | 3.8605 | 2.8387 | 3.4997 | 2.7467 | 3.7319 |
| Δ(17.5k→final) | **+0.655** | +0.641 | +0.865 | +0.561 | +0.167 | **+0.941** |

再看跨基线的对照（**注意口径不同、不可直接比**，见 3.9）：`ep3_baseline` chinese_fineweb = 3.8697，`v21_final` = 3.8605——5 B token 增量训练后，中文网页 held-out 只挪了 0.009 nat。**v21 的 pretrain/代码阶段对该指标是零收益**。

### 3.5 数据质量：后建的 flat 包全都没有文档边界【实测】

- `scripts/build_flat_corpus.py:188`：`--add-eos` 是 `store_true`，默认 **False**；`scripts/_data_supervisor.py:180-186` 的调用串里**没有** `--add-eos`。
- 因此 `hf_pack_code` / `hf_pack_code2` / `hf_pack_chinese_web` / `hf_pack_textbook` / `hf_pack_prog` / `hf_pack_math` / `hf_pack_ctx16k` 的 token 流是**记录首尾直接相接**：一篇代码的末行后面紧跟下一段 prompt。`_FlatStage.row()` 只是按 2049 滑动切片，训练目标里没有任何"文档结束"信号。
- `scripts/build_ctx_mix.py:14-16` 由项目自己写下这句结论：「flat 分片的原始边界在分词时就丢了（`--add-eos` 默认关闭），无法恢复」。
- 对照组：`pretrain_v2` 有 bos/eos（`prepare_pretrain_v2.py:118-131`），所以**v21 里只有第 0 段有文档边界，后面 124,652 步（78%）没有**。

### 3.6 数据质量：v13 的"重切"制造了跨界对话【实测】

`build_chat_mix_v13.py:34-54`：读 v9 的 `(N,513)`，`reshape(-1)` 展平、用 `c != 0` 丢掉 pad，再 `reshape(n, 2049)`。展平即丢失行边界 → 一个 512 宽的对话被切在第 2048 个 token 上、下一行从半句话开始。`meta.json` 显示 v13 无 pad（`552,330,489 = 269,561 × 2049` 整除），即**每一行都是硬切的产物**，且 `wcode` 也随之错位（reply/prompt 标记与语义不再对齐）。

### 3.7 数据质量：pad id = 0 被当作训练目标

`bpe_tokenizer.json` 的 `<pad>` = **id 0**、`<bos>`=1、`<eos>`=2、`<unk>`=3。【实测】
`retnet/flatdata.py:77-80` 对最后一个不满块补 0 并返回 `block[:-1], block[1:]`——**被补出来的 0 会作为 label 参与交叉熵**。`_FlatStage` 虽然丢余数（不补零），但 `wcode` 语义里 0 又表示 pad，两套 0 的含义混在一起。v13 的 `reslice` 正是靠 `c != 0` 区分 pad，一旦某行第 0 列是合法 token 就会误删。

### 3.8 规划机制：步数由余额倒推，8 次热改【实测】

`plan_ddp2.json` 的 `_readme` 逐条记录了：60,955（¥198.5 余额）→ 78,999（实测 1.69 it/s）→ 76,000（避免撞死手）→ 148,000（追加到 ¥290.30）→ 152,557 → 156,057 → 159,857。同一份 `_readme` 里两次承认 **cosine 被拉长**（"total_steps 从 60,955 涨到 78,999，余弦被拉长；当前还在 warmup 末端，LR 变化可忽略"）。`full_history_final.csv` 的 `lr` 列在 30k→40k 段是 9.4e-5→8.5e-5，说明调度确实随文件改动而跳变。【实测】

### 3.9 规划机制：token 账与实际产出脱节，靠小段凑数【实测】

- `plan.json` 给 `chinese_web` 记 0.459 B，实际 `hf_pack_chinese_web` 只产出 **114,523 行 = 0.2345 B**（`[stage]` 日志 + gc 只回收 0.47 GB），却仍跑 7,000 步 = 0.459 B → **1.96 epoch**。
- 为把"唯一 token"推过 20/参数，计划在尾部塞了 `prog_zh` 3,500 步（0.23 B）和 `math_zh` 3,800 步（0.25 B）两个"凑数段"（`_readme` 原话："使唯一 token 越过 10.08B = 20/参数"）。这两段实际训练 loss 是 3.29 / 1.58——模型在这两个域上明显欠拟合，2,750+3,800 步根本不够，塞进去只是为了让账面数字达标。
- 计划声称总 159,857 步 × 65,536 = 10.48 B token，但**唯一 token 只有约 3.0 B**（2.31 B pretrain + 0.234 B web + 7.50 B 代码里的 2.15 B + 0.21 B textbook + 0.15 B prog + 0.25 B math + 0.30 B sft_mix 取到的旧对话）。"20 token/参数"是账面数字，不是数据量。

### 3.10 GC：删掉了全部可复核的中间态【实测】

`logs/linux/v21_ddp.out` 的 `[gc]` 行：
```
[gc] 阶段 0「pretrain_v2_noskypile」已回收 2.28 GB … /data_cache/pretrain_v2
[gc] 阶段 1「chinese_web_hq」已回收 0.47 GB … /data_cache/hf_pack_chinese_web
[gc] 阶段 2「code_ling」已回收 4.00 GB + 1.00 GB … /data_cache/hf_pack_code
[gc] 阶段 3「textbook」已回收 0.42 GB … /data_cache/hf_pack_textbook
[gc] 阶段 5「prog_zh」已回收 0.65 GB … /data_cache/hf_pack_prog
[gc] 阶段 6「math_zh」已回收 0.35 GB … /data_cache/hf_pack_math
```
后果有三个：(a) 本地 `data_cache/` 现在**没有任何 `hf_pack_*`**，我无法复核 `--text-field auto` 到底把什么列拼进了训练集（这是本报告第 6 节列出的最大未验证项）；(b) 阶段 0 的 46 个分片被删，等于删掉了唯一有文档边界的那份预训练语料的训练态副本；(c) `code_ling2` 设了 `gc:false` 但产物也没回到本地——"留给 v22"的承诺没有兑现。

### 3.11 评测口径：held-out 三次不同大小，跨 run 不可比【实测】

`scripts/eval_holdout.py` 的 `--max-tokens` 默认 0（全量）。`logs/holdout_eval.jsonl` 里：`ep3_baseline` = 3,059,712 token / **6 个文件**（含 `skypile_val.npy`）；后续几次 = 4,454,400 token / **5 个文件**。单文件从 598,016 token 变成 1,900,544。也就是说基线用了约 `--max-tokens 600000`，后续没加这个参数。
后果：`ep3_baseline` 的 3.6027 与 `v21_final` 的 3.5023 **不能相减**（前者 6 文件含 skypile、后者 5 文件不含）。而 `v21_step17500` → `v21_final` 是同口径，那 +0.655 是真实退化。另外被剔除的恰好是 `skypile_val`——一个**从未参与训练**的域，把它从评测里删掉等于删掉了"跨域泛化"的唯一探针。

### 3.12 重复存储与孤儿数据【实测】

- 14 个 chat pack = 4.109 B token，唯一内容 0.552 B（7.4×），占 43 GiB 里的 12.8 GiB。
- `pretrain_merged` 1.317 B token（int64，2.5 GB）不被任何计划引用；我比对了它和 `pretrain_v2` 全部 46 个分片的前 64 个 token，**没有一个相同**（说明是另一次、不同顺序/不同分词批次的产物），但上游来源与 `pretrain_v2` 完全重叠（v15 时代同管线）。
- 未分词的 4.6 GiB jsonl（`a_grade`/`roleplay`/`literature`/`extra_sft*`/`chat_corpus`）留在盘上，但除了经 v9 进过一次训练外，没有可复现的构建脚本参数记录。

### 3.13 许可证：已训练数据里混进了 GPL-3.0【实测】

HF API 查证：`BelleGroup/train_3.5M_CN` = **gpl-3.0**，而 `belle_3_5m` 在 `chat_pack_v4`–`v9` 里都在（`meta.json`: v9 为 5,753,498 token）。`Skywork/SkyPile-150B` 在 HF 上**没有 license 标签**，计划只靠 `exclude: ["skypile_"]` 在读取时跳过——文件仍在盘上（504 M token），任何一次计划改动都可能把它重新纳入。`SkyPile` 被剔除的原因在 `plan.json` 的 `_readme` 里写的是"license 不合规"，但没有留证据链。

---

## 4. 新数据方案

### 4.0 设计约束（先把账算清楚）

- 目标模型 0.5B / 2048 窗口；Chinchilla 参考 10 B token。
- 吞吐：`v21_plan_ddp2.json` 实测 1.69 it/s × 65,536 token/步 = **110,724 token/s**。【实测：`full_history_final.csv` 的 step 斜率 + 日志进度条】
- 单价 ¥13.97/h → **398.6 M token/h** → **¥35.05 / B token**，即 **¥/B = 35.05 × 重复倍数**。这是本方案唯一的花钱口径。
- 硬约束：**每个包的总消费 ≤ 该包的唯一 token 池**（≤1.05 epoch），否则就是在重复 3.2 那个错误。做不到就加新来源，不加步数。

### P1 `zh_web_edu` — 中文教育价值网页（主力）

| 项 | 内容 |
|---|---|
| 组成 | `Chinese-FineWeb-Edu-V2` 1.014 B（已有，全用）+ `Fineweb-Edu-Chinese-V3` 0.18 B（已有）+ `chinese-fineweb-edu`(v1) 0.35 B（需重下，只保留 score≥4 的头部） |
| 来源 | `opencsg/Chinese-FineWeb-Edu-V2`（**apache-2.0**，HF 实测）、`opencsg/Fineweb-Edu-Chinese-V3`（**license:other**，需读协议，商用前要确认）、`opencsg/chinese-fineweb-edu`（apache-2.0） |
| 规模/申请 | 1.5 B token 量级；无需申请，直接 `snapshot_download` |
| 目的 | 中文世界知识与书面表达的主力分布；把"中文网页"从 0.234 B 提到 1.5 B 量级 |
| 验收 | held-out `chinese_fineweb_edu_v2_val`（现有，1.9 M token 全量）loss 从 ep3 的 3.87 降到 **≤3.35**；另建 2,000 条中文维基问答探针（见 P2）准确率 ≥ 基线 +8 pt |
| 预算 | 1.55 B（A 0.94 + B 0.25 + C 0.30 + D 0.06），占 15.6% |
| 打包 | flat，int16，`seq_len=2048`；**每文档尾部强制 `<eos>`(id=2)**，文档间不插 `<bos>`（省 1 token/文档）；`text` 字段显式指定，不用 `auto` |
| 位置 | A（与 P2/P3/P6 混）、B（replay 0.20 B）、C（退火头部 0.20 B） |

### P2 `zh_encyclopedia` — 百科/辞书事实源

| 项 | 内容 |
|---|---|
| 组成 | `wikimedia/wikipedia` 20231101.zh 全量（现有 0.203 B，补到 0.35 B）+ `wikimedia/wikisource` zh 0.06 B + `chinese-poetry` 0.030 B |
| 来源 | `wikimedia/wikipedia`（**cc-by-sa-3.0**，HF 实测）、`wikimedia/wikisource`（cc-by-sa）、chinese-poetry（GitHub，公共领域） |
| 目的 | 事实性知识、专名、定义式表达；抑制"百科问答像小说"的倾向 |
| 验收 | 自建 1,000 条中文事实 cloze（从 wiki 摘要挖空实体，答案唯一）top-1 从基线提升 ≥10 pt；`wikipedia_zh_val` loss ≤3.20 |
| 预算 | 0.40 B（A 0.25 + C 0.15），占 4.2% |
| 打包 | flat + `<eos>`/文档；标题与正文之间用 `\n\n`，不拼 URL/Infobox |
| 位置 | A、C（退火段加权到 8.8%） |

### P3 `en_general` — 英文通用（保双语与推理迁移）

| 项 | 内容 |
|---|---|
| 组成 | `HuggingFaceFW/fineweb-edu` sample-100BT 抽 0.505 B（已有）+ `smollm-corpus/cosmopedia-v2` 0.403 B（已有） |
| 来源 | fineweb-edu（**odc-by**）、smollm-corpus（**odc-by**）——两者都可商用 |
| 目的 | 英文能力 + 通过英文教科书式文本（cosmopedia）迁移推理；防止纯中文导致英文退化到不可用 |
| 验收 | `fineweb_edu_val` / `cosmopedia_val` loss 分别 ≤3.05 / ≤2.10；另跑 200 条英文 GSM8K-style 简单算术，≥基线 |
| 预算 | 0.75 B（A 0.45 + B 0.20 + C 0.10），占 7.5% |
| 打包 | flat + `<eos>`；不做中英混排拼接 |
| 位置 | A、B（B 段作为代码段的中和剂） |

### P4 `zh_books` — 长文/叙事/文学

| 项 | 内容 |
|---|---|
| 组成 | `wikisource` zh 长文 0.06 B + `chinese-poetry` 0.03 B（与 P2 不重复计，P2 只取辞书类）+ `zxbsmk/webnovel_cn` 0.02 B（现有）+ CC0 小说 0.02 B |
| 来源 | cc-by-sa / 公共领域 / CC0；**不再使用同人扩写**（`haruhi_expand` 这类 340 k 条合成对话永久弃用） |
| 目的 | 长距离连贯性、叙事结构、书面语韵律；这是 2048 窗口唯一能练到"跨段指代"的语料 |
| 验收 | 2,000 token 长文续写的重复率（4-gram 重复占比）在采样 200 条上 ≤15%；人工抽 30 条看是否成段 |
| 预算 | 0.13 B，占 1.4% |
| 打包 | flat + `<eos>`；**按书切，不跨书拼接**（一本书一个分片段），使 2048 窗口尽量落在同一本书内 |
| 位置 | A 0.08 + C 0.05（退火尾部，最后一千步前） |

### P5 `math_reason` — 数学（中英）

| 项 | 内容 |
|---|---|
| 组成 | `open-web-math` 0.152 B（已有）+ `TiWu-Lab/math-zh` 取 4 片 0.60 B（**新下载**，6.01 GB）+ `finemath`(可选) 0.10 B |
| 来源 | open-web-math（HF 无 license 标签 → **按 odc-by 处理，需在卡上确认**）、`TiWu-Lab/math-zh`（**odc-by**，HF 实测）、`HuggingFaceTB/finemath`（odc-by） |
| 规模/申请 | math-zh 4 片 6.01 GB ≈ 0.6 B token；无需申请 |
| 目的 | 数学符号、分步推导格式；这是 SFT 阶段 CoT 能"挂上去"的前提 |
| 验收 | 自建 300 题中文小学数学/初中代数（有确定答案），exact-match 从基线提升 ≥8 pt；`open_web_math_val` ≤2.45 |
| 预算 | 0.85 B（A 0.15 + B 0.40 + C 0.25 + D 0.05），占 8.5% |
| 打包 | flat + `<eos>`；**题面与解答之间插 `\n\n`，不拆题**；不做 LaTeX 归一化（保持原样，避免引入格式噪声） |
| 位置 | A、B（重点）、C、D |

### P6 `code_multi` — 多语言代码（严格 1 epoch）

| 项 | 内容 |
|---|---|
| 组成 | `inclusionAI/Ling-Coder-SFT` 全量 **2.15 B，只切一次，精确去重后按 1.00 epoch 消费**（A 0.85 + B 1.15 + C 0.15 = 2.15 B） |
| 来源 | `inclusionAI/Ling-Coder-SFT`（**apache-2.0**，HF 实测；26 parquet / 1.4 M 行 / 6.26 GB） |
| 目的 | 代码语法、跨行结构、注释-代码对应；也是长上下文扩展的骨架 |
| 验收 | 自建 500 条"函数补全"（取 held-out 文件的函数头，测 next-100-token exact match）；HumanEval 不做（0.5B 无意义）；train loss **必须落在 1.5–2.2**——若低于 1.2 立即停机查重复（这正是 3.2 的探针） |
| 预算 | 2.15 B，占 21.6%（**从 56% 降到 21.6%**） |
| 打包 | flat + `<eos>`；**每条样本 = 一个完整文件/函数，禁止把 prompt+response 的 `\n\n` 拼接当成一条**（这是 `--text-field auto` 的老毛病）；`text` 字段显式写死 |
| 位置 | A/B/C 三段，**不再单独占一整段 68,557 步** |

### P7 `code_zh` — 中文编程/技术文本

| 项 | 内容 |
|---|---|
| 组成 | `BAAI/IndustryCorpus2` 中文编程子集 0.25 B + 现有 `prog_zh` 0.05 B |
| 来源 | `BAAI/IndustryCorpus2`（**apache-2.0**，HF 实测） |
| 目的 | 中文技术表达（注释、文档、报错解释），补 P6 的英文偏向 |
| 验收 | 中文注释生成探针 100 条（给函数，写中文 docstring）人工 5 分制 ≥3.0；`prog_zh` 段 train loss 必须从 3.29 降到 **≤2.4**（当前明显欠拟合） |
| 预算 | 0.30 B（A 0.10 + B 0.12 + C 0.05 + D 0.03），占 3.0% |
| 打包 | flat + `<eos>`；中文注释与代码保持原顺序 |
| 位置 | A、B、D |

### P8 `zh_qa_knowledge` — 中文百科问答

| 项 | 内容 |
|---|---|
| 组成 | `Mxode/I_Wonder_Why-Chinese` 0.12 B（已有 118.9 M，全用） |
| 来源 | **cc-by-sa-4.0**（HF 实测） |
| 目的 | "提问→解释"的语体，是对话与知识之间的桥；v13 里它占 21.5% 却没被当成独立能力训练 |
| 验收 | 同 P2 的 1,000 条事实 cloze + 200 条"为什么"式问答的 ROUGE-L ≥ 基线 +5 |
| 预算 | 0.17 B（C 0.12 + D 0.05），占 1.7% |
| 打包 | packed（有 `wcode`），`<bos>`+`<eos>`/条；问句标 prompt(2)、答句标 reply(1) |
| 位置 | C（退火）、D（SFT 前段） |

### P9 `zh_cot` — 中文推理链

| 项 | 内容 |
|---|---|
| 组成 | `Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT` 0.15 B（**完整 CoT，不再截断到 1,200 字符**） |
| 来源 | **apache-2.0**（HF 实测） |
| 目的 | 显式推理格式；v13 用 4,000 字符上限重编码是对的，但只给了 147.9 M 且放在 SFT 桶里 |
| 验收 | 300 题中文数学 exact-match（与 P5 共用）；**另查"答案前是否出现 `</think>`"比例 ≥95%**（v12 的 `strip_think` 把 CoT 删过一次） |
| 预算 | 0.20 B（C 0.15 + D 0.05），占 2.1% |
| 打包 | packed；`<think>` 段标 reply(1)、题面标 prompt(2)；**禁止按字符截断**，超 2048 的样本整条丢弃并计数 |
| 位置 | C 尾、D |

### P10 `instruct_mix` — 通用指令（许可证过滤后）

| 项 | 内容 |
|---|---|
| 组成 | `m-a-p/COIG-CQIA` 0.10 B + `YeungNLP/firefly-train-1.1M` 0.08 B + `moss-003-sft` 0.06 B + `shibing624/sharegpt_gpt4` 0.06 B + `HuggingFaceH4/ultrachat_200k` 0.04 B |
| 来源/许可 | COIG-CQIA（HF 无 license 标签 → **须逐子集查**，其中部分子集为 nc）、firefly（无标签 → 只用自带声明的 cc-by-nc 部分则**丢弃**）、moss-003（apache-2.0）、sharegpt_gpt4（**cc-by-4.0**，HF 实测，注意它是 GPT-4 输出 → 多数平台禁止用其蒸馏）、ultrachat_200k（**mit**） |
| 目的 | 指令跟随、多轮、格式稳定 |
| 验收 | `logs/bench_v3` 的 `instr_robust` / `core_rp` 两项（已有 300+ 条 held-out）分数 ≥ 当前 v21_final；格式错误率（缺 `<eos>`、角色标签错）<1% |
| 预算 | 0.34 B（C 0.08 + D 0.26），占 3.4% |
| 打包 | packed，2048，`<bos>`…`<eos>`；**每条对话独立成行、不跨界**（用 packing 的"整条放入或丢弃"模式，不做 v13 那种展平重切） |
| 位置 | C 尾、D 主体 |

### P11 `roleplay_capped` — 角色扮演（硬上限 3%）

| 项 | 内容 |
|---|---|
| 组成 | `coser.jsonl` + `claude_rp_30k` + `roleplay_zh_gpt4` + `limarp_aug`，**按来源上限各 ≤0.03 B**，合计 0.08 B；**`haruhi_expand` / `haruhi_54k` / `haruhi_zero_pippa` 全部弃用** |
| 来源 | 现有 `data_cache/roleplay/`（许可未逐一核实，须按 3.13 的教训补） |
| 目的 | 保留角色扮演能力，但**不让单一人设主导** |
| 验收 | 采样 100 条，统计"同一角色名出现率" ≤10%（v21 的产物里这条明显超标）；同时 `core_rp` bench 不掉分 |
| 预算 | 0.08 B，占 0.85% |
| 打包 | packed，对话整条，`<eos>`/条 |
| 位置 | D 尾（最后 2,000 步） |

### P12 `replay_anchor` — 预训练回放（防遗忘，按比例注入，不单独成段）

| 项 | 内容 |
|---|---|
| 组成 | `pretrain_v2` 已训过的 2.31 B 中抽 0.60 B（含 bos/eos，边界完整） |
| 来源 | 本地已有，边际成本 0 |
| 目的 | 在代码段/退火段里按 12–15% 比例回放，压住 3.4 那种 held-out 退化 |
| 验收 | C 段结束时 `chinese_fineweb_edu_v2_val` 相对 B 段结束**不得上升 >0.05 nat**（这是硬停机条件） |
| 预算 | 0.60 B（A 混入 0.30 = 6.7%、B 0.20 = 7.1%、C 0.10 = 4.8%），占 6.0% |
| 打包 | flat，沿用原分片（有 eos），不重新分词 |
| 位置 | B、C 的混合流里，比例固定 12–15% |

### P13 `zh_general_web` — 中文通用网页（补量主力，新下载）

| 项 | 内容 |
|---|---|
| 组成 | `BAAI/IndustryCorpus2` 中文通用子集，按域名去重后取 2.44 B |
| 来源 | `BAAI/IndustryCorpus2`（**apache-2.0**，HF 实测；中文通用/教育/代码多子集，规模远超本项目需求） |
| 规模/申请 | 只取所需分片（约 5 GB parquet）；无需申请，`snapshot_download` 后**显式**指定 `text` 字段 |
| 目的 | 把中文网页从 v21 的 0.234 B 提到 2.44 B；这是全案唯一"从零补量"的包，也是把账从 7.5 B 推到 10 B 的唯一来源 |
| 验收 | 与 P1 共用 `chinese_fineweb_edu_v2_val`；另加"网页文本困惑度分档"抽查：随机 200 篇人工过一遍，明显噪声（导航条/版权声明/重复模板）>5% 就打回重洗 |
| 预算 | 2.44 B（A 1.43 + B 0.43 + C 0.58），占 24.5% |
| 打包 | flat，int16，2048；**必须 `--add-eos`**；分片前做 URL 级去重 + 段落级 128-token 前缀去重，去重率写进 `meta.json` |
| 位置 | A（主）、B（replay）、C（退火，只取 edu-score 头部 0.58 B） |

### P0 弃用与清洗（不出现在训练里）

- **弃用**：`pretrain_merged`（1.317 B，来源重叠且 int64）、`chat_pack_v3`–`v12`（4.109−0.552 = 3.56 B 冗余）、`chat_recovery_mix`、`chat_sft_mix`、`wikitext*`（不是中文目标）、`skypile`（无 license）、`belle_3_5M`（**GPL-3.0**）、全部 `haruhi_*`。
- **保留但隔离**：`roleplay/`、`a_grade/`、`literature/`、`extra_sft*/`、`chat_corpus/`、`nonhf/` 移到 `data_cache/_retired/`，只作为"历史上训过什么"的证据，不再进任何计划。
- **必须补做**：对 P6 做**精确去重**（按 128-token 的 sha1 前缀分桶），去重率与丢弃量写进 `meta.json`；这是防 3.2 复发的唯一机制。

---

## 5. 阶段顺序与预算表

### 5.1 顺序原则

1. **先广后窄，最后才对话**（与 v21 相反的地方在于：代码不再独占一大段，而是从头就以 ~23% 的固定比例混在 A/B 里）。
2. **退火段（C）只降 LR、只提高质量门槛，不换域**——v21 的 C 段是"换成纯代码 4.49 B"，这是 3.4 退化的直接原因。
3. **每一段都保留 12–15% 的 pretrain replay**（P12），且 replay 的样本必须含 bos/eos。
4. **对话锚点从 A 段就开始**：P8/P10 在 A 段占 3%，避免 3.3 那种"先崩 1.95 nat 再修"。

### 5.2 预算表（65,536 token/步，1.69 it/s，¥13.97/h）

| 阶段 | 步数 | token | 占比 | 混合构成（B） | 时长 | 费用 |
|---|---:|---:|---:|---|---:|---:|
| **A 基础**（LR 1.1e-4，warmup 500） | 68,665 | 4.500 | 45.2% | P13 1.43 / P1 0.94 / P6 0.85 / P3 0.45 / P12 0.30 / P2 0.20 / P5 0.15 / P7 0.10 / P4 0.08 | 11.3 h | ¥157.7 |
| **B 代码+数学**（LR 余弦 → 6e-5） | 42,725 | 2.800 | 28.1% | P6 1.15 / P13 0.43 / P5 0.40 / P1 0.25 / P3 0.20 / P12 0.20 / P7 0.12 / P2 0.05 | 7.0 h | ¥98.1 |
| **C 退火**（LR 6e-5 → 1.1e-5，质量门槛提高） | 31,738 | 2.080 | 20.9% | P13 0.58 / P1 0.30 / P5 0.25 / P2 0.15 / P6 0.15 / P9 0.15 / P8 0.12 / P3 0.10 / P12 0.10 / P10 0.08 / P4 0.05 / P7 0.05 | 5.2 h | ¥72.9 |
| **D SFT**（LR 3e-5 → 5e-6） | 8,850 | 0.580 | 5.8% | P10 0.26 / P11 0.08 / P1 0.06 / P8 0.05 / P9 0.05 / P5 0.05 / P7 0.03 | 1.5 h | ¥20.3 |
| **合计** | **151,978** | **9.960** | 100% | **9.96 B 唯一 token，每个包 ≤1.00 epoch，无重复包** | **24.99 h** | **¥349.1** |

- 加 5% eval 开销（`eval_steps 1200` 实测约 40 s/次）：**26.2 h / ¥366.6**。
- **¥/B token = ¥35.05**（100% 唯一）。对照 v21：账面 10.48 B token（159,857 步 × 65,536），其中 code 段消费 6.13 B 而唯一池只有 2.15 B（**3.98 B 是复读**），全案实际唯一约 5.65 B → 有效 **¥65 / unique-B**，是本案的 1.85 倍，约 ¥157 花在重复样本上。
- **19.75 token/参数**（10.0 B / 504.18 M），Chinchilla 的 98.8%。
- 可选 **E 段 16K 扩展**：P4 长文 0.35 B + P6 repo 级 0.15 B，7,629 步 = 0.50 B，1.3 h，¥17.5（LR 2e-5；`build_ctx_mix` 改成"长文优先"并**显式补 `<eos>`**）。
- 数据准备成本：新增下载 P13(≈5 GB) + P5 math-zh(6 GB) + P1 的 v1 头部(≈2 GB) ≈ 13 GB，分词 4–6 h 单进程（按 `docs/28` 实测 0.2–1.2 M tok/s），**不计入上面 GPU 时间**。

### 5.3 停机与验收条件（这是 v21 最缺的东西）

| 条件 | 阈值 | 动作 |
|---|---|---|
| code 段 train loss 低于 1.2 | 连续 500 步 | **立即停机查重复**（3.2 的探针） |
| `chinese_fineweb_edu_v2_val` 相对本段起点上升 >0.10 nat | 任一次 eval | 停机，回滚到上一个 checkpoint |
| `chat_pack_v13` val loss 相对本段起点上升 >0.30 nat | 任一次 eval | 提高 P12 replay 比例到 25%，否则停机 |
| 每段开头 | — | 必跑一次 `eval_holdout.py`（全量 5 文件 + skypile_val 恢复为 6 文件），写进 `logs/holdout_eval.jsonl`，**必须带 `tokens` 字段供口径校验** |
| 任何 eval 记录 | `tokens` 与前一次不同 | 标记为不可比，禁止写进同一张表 |

### 5.4 GC 与保留

- **分片保留策略**：每段消费完后，`gc` 只允许删"可从源 parquet 在 ≤2 h 内重建"的包；**`pretrain_v2` 切出的 replay 分片永不删**（它是有文档边界的唯一副本）。
- **每个包的 `meta.json` 必须记录**：`source_ids`、`text_fields`（显式，禁止 `auto`）、`add_eos: true`、`records_seen`、`records_errors`、`dup_dropped`、`tokenizer_sha1`。没有这份 meta 的包不许进计划（v21 的 `hf_pack_code` 就是反例）。
- **原始 parquet 至少保留 30 天**再删（`_data_supervisor.py` 现在分词完立刻 `rmtree`，这是 3.10 的根因）。
- 磁盘预算：9.96 B token × 2 B = 20 GB（int16）+ 源 parquet 约 12 GB + 中间态 → 预留 **40 GB**，与 v21 的 43 GiB 现状相当。

---

## 6. 风险与没验证的部分

1. **`hf_pack_code` / `code2` 的真实内容无法复核**（已被 gc 删除，本地不存在）。3.2 的"3.49 倍重复"是"7.50 B 建出量 ÷ 2.15 B 文档实测全量"这个算术结论 + 0.49 loss 的旁证，**不是直接看到重复样本**。若 `docs/28` 对 Ling-Coder 的 2.15 B 估算错得离谱（例如真实是 8 B），则该结论不成立——但那样 `code_ling2` 的 0.49 loss 就更无法解释。**建议用 3 分钟验证**：重下 Ling-Coder 的前 3 个 parquet，用项目 tokenizer 数一遍 token。
2. **`--text-field auto` 到底取了哪些列，无从得知**。`auto_text_fields` 只按字段名黑名单过滤，`flatten_value` 对 list/dict 的处理是启发式的。如果 Ling-Coder 的 parquet 里有 `instruction`/`input`/`output` 之外的列（如 `task_id` 之外的语言标签），会被拼进正文。这是我认为 code 段 loss 异常低的**第二个可能原因**（第一个是重复）。两者都指向同一个结论：重训必须显式指定 `--text-field`。
3. **训练损失 0.49 的替代解释未排除**：也可能是 `_FlatStage` 在 `refresh()` 后 `starts` 错位导致同一批行被反复取（`_scan` 每次重建 `starts`，而 sampler 持有旧长度）。我没有读到能证明或否证这一点的日志，故标为【未验证】。
4. **新方案里 4 个来源需要下载/确认**（`BAAI/IndustryCorpus2` 中文通用 2.44 B ≈ 5 GB、`TiWu-Lab/math-zh` 4 片 6 GB、`opencsg/Fineweb-Edu-Chinese-V3` 协议确认、`HuggingFaceTB/finemath`），下载+分词约 **6–9 h 单进程**（按 `docs/28` 的 0.2–1.2 M tok/s 实测），不计入上面的 24.9 h GPU 时间。
5. **许可证有 4 处不确定**：`Fineweb-Edu-Chinese-V3` 是 `license:other`（须读协议）；`open-web-math` / `Skywork/SkyPile-150B` / `m-a-p/COIG-CQIA` / `YeungNLP/firefly` 在 HF 上**没有 license 标签**；`shibing624/sharegpt_gpt4` 是 GPT-4 输出（cc-by-4.0 但上游 OpenAI 条款可能禁止蒸馏）。本报告只标注，不做法律判断。
6. **预算表里的"步数→token"用的是 1.69 it/s**。这是 v21 在 2×PRO 6000 上的实测稳态；若换卡、换 batch 或开 `find_unused_parameters`，费用按 35.05×(实际tok/s比) 等比放大。
7. **P4 `zh_books` 与 P3 `en_general` 的池子偏小**（0.13 B / 0.75 B）。这两个包的作用是"质量锚"而非"量"，但如果审稿方认为长文能力需要更多，就得补 `wikisource` 全量或 `Gutenberg`，池子上限我到 0.3 B 左右，我没有验证过更大规模的中文长文公开语料。
8. **`chat_pack_v13` 作为唯一 val_dir 是错的**（3.3 证明了它对 pretrain 分布不敏感），新方案要求 3 个探针并存：`chinese_fineweb_edu_v2_val`（域内）、`skypile_val`（未见域）、`chat_pack_v13`（对话）。这三个的 token 数必须写进每一条 eval 记录。
