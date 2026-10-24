# RetNet：从零手写 → 中英角色扮演对话模型

> 这是项目早期的中文 README，内容停在 **272M 阶段（2026-10 上旬）**，保留原样未改。
> 当前模型的英文总览见 [`README.md`](README.md)；逐段历史见 [`docs/archive/restored-2026-10-24/12_训练状态盘点.md`](docs/archive/restored-2026-10-24/12_训练状态盘点.md)。
> 文中引用的 `docs/paper/2307.08621v4.pdf` 因版权原因不随仓库分发，请从 [arXiv](https://arxiv.org/abs/2307.08621) 获取。

依据论文 [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)
（`docs/paper/2307.08621v4.pdf`）从零手写 RetNet——**没有用任何现成 Transformer/RetNet 模块**——
并在两块消费级显卡上把它从 23M 一路生长、训练到 **272M 的中英角色扮演对话模型**。

## 先看这几份

| 想知道什么 | 看哪份 |
| --- | --- |
| 项目现在什么状态、最后一个检查点在哪 | [`docs/archive/restored-2026-10-24/12_训练状态盘点.md`](docs/archive/restored-2026-10-24/12_训练状态盘点.md) |
| 模型什么水平、怎么评的、下一步做什么 | [`docs/archive/restored-2026-10-24/13_基准测试重设计与评测结果.md`](docs/archive/restored-2026-10-24/13_基准测试重设计与评测结果.md) |
| **当前基准（1,020 项）怎么用、判据怎么写出题** | [`docs/14_基准测试v3.md`](docs/14_基准测试v3.md) · [`bench/CONTRACT.md`](bench/CONTRACT.md) |
| 日常操作命令、故障处理 | [`RUNBOOK.md`](RUNBOOK.md) |
| 全部文档入口 | [`docs/archive/restored-2026-10-24/INDEX.md`](docs/archive/restored-2026-10-24/INDEX.md) |

## 硬件与环境

| | 本机 | 训练机 `192.168.1.100` |
| --- | --- | --- |
| GPU | RTX 3070 8GB（SM86 / 46 SMs） | RTX 4070 Laptop 8GB（SM89） |
| Python / torch | 3.13.7 / 2.10.0+cu128 | 3.11.15 / 2.11.0+cu128 |
| 角色 | 数据构建、评测、交互推理 | 272M 训练（可全天候） |

## 目录结构

```text
RetNet/
├── train.py                     手写训练入口（三种数据模式自动识别）
├── play.py                      交互推理：角色扮演 / 闲聊 / 英文续写
├── run_task.py                  配置驱动训练启动器（读 tasks/*.json）
├── retnet/                      ⭐ 手写核心
│   ├── model.py                 Retention / MSR / RetNetBlock / RetNetLM + generate
│   ├── tokenizer.py             byte-level BPE（vocab 20000）
│   ├── chat.py                  训练与推理共用的对话编码器（含「设定」persona 块）
│   ├── packing.py               文本清洗 / SFT 解析 / best-fit 行打包
│   ├── flatdata.py              int16 分片 mmap 数据集（3B+ token 不吃内存）
│   └── data.py                  早期 wikitext 管线
├── retnet_hf/                   ⭐ HF 封装（Trainer + safetensors），143M/272M 线在用
├── retnet_hf_fp16/              数值加固分支，v11 用过
├── scripts/                     数据构建 / 生长 / 评测 / 基准
├── bench/                       ⭐ 基准 v3 库：判据 DSL + 题库 + 运行器
│   ├── CONTRACT.md              出题契约（字段表 / DSL 全表 / 难度定义）
│   ├── checkers.py              判据 DSL 编译器（44 个谓词）
│   ├── registry.py              Item 校验 + 锚点审计 + 类别权重
│   ├── gen_basics.py            确定性生成器（填空/问答/语言/续写 390 项）
│   ├── fastgen.py               逐位等价的快速解码器 + 批处理解码器
│   ├── runner.py                跑分（断点续跑 / 逐项温度 / 贪婪基线）
│   ├── analysis.py              分层报分 + McNemar + 区分度诊断
│   ├── items/                   人工出题 JSONL（630 项）
│   ├── pools/                   原始内容包（知识 / 语言）
│   └── data/                    冻结题库 bench_v3.jsonl + SHA256 manifest
├── tasks/                       训练任务配置 + 启动器
├── data_cache/                  全部语料与 token 缓存（只放 E 盘）
├── checkpoints/                 各阶段权重（37+ GB）
├── docs/                        编号文档集 01–14 + scaling_study + paper/ + archive/
├── logs/                        训练日志 / 评测报告 / 归档
├── legacy/                      被取代的旧脚本 + 手写版代码快照
├── Kaggle/                      Kaggle TPU 线：notebook 世代 + 修复报告 + 审计留档
└── tests/                       test_model.py（三范式，⚠️玩具尺度）
                                 test_checkers.py（判据 DSL 语义）
                                 test_bench_equivalence.py（解码器逐位等价）
                                 test_runner_lockstep.py（批处理不改分）
```

## 模型核心

`retnet/model.py` 是手写实现，`retnet_hf/modeling_retnet.py` 是同一套数学的 HF 封装。
只用了 `nn.Linear` / `nn.Embedding` / `nn.LayerNorm` / `nn.GroupNorm` / `nn.Dropout`。

**Retention 机制**（论文式 5–8）

```text
S_n = γS_{n-1} + K_n^H V_n                          (6)  递推形
Retention(X_n) = Q_n S_n
D_nm = γ^(n-m)  if n >= m else 0                    (5)  并行形
Retention(X) = (QK^H ⊙ D)V
γ = 1 - 2^(-5 - arange(0, h))                       (8)  多尺度衰减
MSR(X) = (swish(XW_G) ⊙ GroupNorm_h(heads)) W_O           swish 门控 + 每头独立 Sub-LN
```

三种前向范式共享同一组权重：`parallel`（物化 L×L 分数矩阵）、`recurrent`（O(1) 逐 token）、
`chunkwise`（块内并行 + 块间递推，当前训练主线）。

**几个关键设计**

- **xPos 相对位置**：论文的复数 `e^{inθ}` 用等价的实数二维旋转实现；位置与频率全程 fp32，
  末端才 cast（bf16 位置在 8192 处相位误差可达 9.9 rad）
- **GLA 可学习衰减**（`use_gla=true`）：`γ` 是 `nn.Parameter` 而非固定 buffer。
  它必须留在 fp32——bf16 在 1.0 附近分辨率只有 2⁻⁸，会把 16 个头里的 **12 个衰减塌缩成恰好 1.0**，
  多尺度结构直接消失。三份实现的 decay 计算全程 fp32
- **宽度生长**（`GrowthLayerNorm` + `growth_old_dim`）：pre-LN 下直接扩宽会污染每层 LayerNorm 的
  均值/方差，18 层级联后 loss 从 4.5 涨到 17.8。解法是 LN 只对旧 D 维生效、新通道走 γ=0 的零侧分支，
  实测函数保真 Δ**7.25e-4**
- **推理状态 O(1)**：每层 `key_dim × value_head_dim`，272M 全天候约 2.4MB，与上下文长度无关
- **参数量闭式** `P ≈ 12·L·d² + V·d`；272M 精确值 **272,304,448**

> ⚠️ **三种范式在生产尺度上并不数值等价。** `tests/test_model.py` 断言的 `1e-4` 只在
> d_model=64 / L=16 的玩具尺度成立；272M fp32 实测 parallel↔chunkwise 的 `max|diff|` 到 **6.77**。
> 训练走 chunkwise，而 `generate()` 预填 prompt 走 parallel——这个不一致目前没有回归护栏。
> 详见 [`docs/11b_TileKernels可用性评估.md`](docs/11b_TileKernels可用性评估.md) §3.8。

## 数据管线

`retnet_hf/train.py` 只吃 packed 模式；shard 流式与旧扁平缓存是手写 `train.py` 的路径。

| 模式 | 参数 | 用途 |
| --- | --- | --- |
| packed + masked | `--prepared_data_dir data_cache/chat_pack_v13` | SFT：样本对齐 + 只在助手回复上算 loss |
| 分片流式 | `--shard_dirs data_cache/pretrain_v2` | 继续预训练：int16 分片 mmap，3B+ token |
| 旧扁平缓存 | `--prepared_data_dir data_cache/chat_mix_v20k` | 兼容早期数据 |

**每 token 带权重码**：`0` padding（不算 loss）、`1` 助手回复（算）、`2` 提问/设定
（`--prompt_loss_weight 0.1`）、`3` 纯文本回放（算）。打包用 best-fit 装箱，padding 占比约 1.6%。

**对话格式**（`retnet/chat.py` 是唯一实现，训练与推理逐 token 对齐）：

```text
<bos>设定：你是令狐冲（出自笑傲江湖）。……
用户：{提问}
助手：{回复}<eos>
```

> ⚠️ **两级截断都切尾部，而答案在尾部。** 读取端的 `clean(output, N)` 和打包端的
> `encode_dialogues_batch(max_len=seq_len)` 都从尾部截。R1 数据中位输出 1,912 字符，
> 被 `clean(output, 1200)` 砍掉一半——**正好砍掉 `</think>` 和答案**，于是模型只学会
> 「开一个 think 块，永远不收尾」。扩窗时必须同时抬高字符上限
> （v13 用 `--seq_len 2048 --r1_max_chars 4000`）。完整漏斗见 [`docs/13`](docs/archive/restored-2026-10-24/13_基准测试重设计与评测结果.md) §2。

## 快速开始

```powershell
cd E:\Workspaces\RetNet

# 交互聊天（默认预设指向 100M 的 rp_v2）
C:\Tools\Miniconda3\envs\AI\python.exe play.py --mode chat

# 玩 272M —— checkpoint 目录里没有 tokenizer，要去 data_cache 借
C:\Tools\Miniconda3\envs\AI\python.exe play.py --mode chat `
    --checkpoint checkpoints/retnet_272m_rp `
    --tokenizer data_cache/chat_pack_v9/bpe_tokenizer.json

# 能力基准 v3：1,020 项 × 15 类 × 难度分层（当前）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\compile_bench.py            # 构建 + 校验 + 冻结题库
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_v3.py `
    --models "rp=checkpoints/retnet_272m_rp,v10=checkpoints/retnet_272m_v10" `
    --tokenizer data_cache/chat_pack_v10/bpe_tokenizer.json `
    --seeds 42,43,44 --out logs/bench_v3

# 只重出报告（不用重跑 GPU）/ 冒烟（每 40 项取 1）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_report.py --dir logs/bench_v3 --families
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_v3.py --models "rp=checkpoints/retnet_272m_rp" `
    --tokenizer data_cache/chat_pack_v10/bpe_tokenizer.json --stride 40 --seeds 42 --out logs/bench_v3_smoke

# 自动判分能力基准 v2（143 项 × 9 类，历史对比用）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_v2.py `
    --models "rp=checkpoints/retnet_272m_rp" `
    --tokenizer data_cache/chat_pack_v10/bpe_tokenizer.json `
    --seeds 42,43,44 --out logs/bench_v2

# 工程基准（parallel↔chunkwise 等价性 / 吞吐 / 显存）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_engine.py `
    --model_dir checkpoints/retnet_272m_rp --seq_lens 256,512,1024

# 解码成本与批处理收益实测（决定基准吞吐的口径）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_decode_profile.py --model_dir checkpoints/retnet_272m_rp
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_batch_scaling.py --items 48

# 三范式等价性测试（⚠️ 只在玩具尺度成立，别当生产护栏用）
C:\Tools\Miniconda3\envs\AI\python.exe tests\test_model.py

# 基准自带的三套测试
C:\Tools\Miniconda3\envs\AI\python.exe tests\test_checkers.py            # 判据 DSL 语义
C:\Tools\Miniconda3\envs\AI\python.exe tests\test_bench_equivalence.py   # 基准解码器逐位等价
C:\Tools\Miniconda3\envs\AI\python.exe tests\test_runner_lockstep.py --stride 97
```

训练命令见 [`RUNBOOK.md`](RUNBOOK.md) §2；远程机操作见 [`docs/10_远程部署笔记.md`](docs/10_远程部署笔记.md)。

## 训练历程

| 阶段 | 数据 | 参数 | 结果 |
| --- | --- | --- | --- |
| `retnet_23m` | Wikitext-2 | 23.1M | val ppl 75.30 |
| `retnet_97m_wikitext103` | Wikitext-103 | 97.6M | val ppl 50.39 |
| `retnet_100m_chat` | 混合中英对话 | 100.4M | ppl 79.58，会闲聊但很笨 |
| `retnet_100m_pretrain` | seq 256 + 旧扁平缓存 | 100.4M | 全局最优 36.17 @98k；后 100k 步验证集净退化到 38.65 |
| `retnet_100m_chat_v2` | 对话恢复混合 | 100.4M | ppl 43.25，找回聊天格式 |
| `retnet_100m_chat_sft` | 指令 SFT 36.5M（扁平） | 100.4M | ppl 13.96 但答非所问 |
| `retnet_100m_chat_v3` | **对齐 + 掩码** 194.7M | 100.4M | 回复 ppl 91.72，闲聊自然 |
| `retnet_100m_rp` | 角色扮演混合 217M | 100.4M | ppl 10.31 |
| `retnet_100m_pretrain_v3` | 363M 新语料 | 100.4M | val ppl 56.84 |
| `retnet_100m_rp_v2` | v5 233.9M | 100.4M | val ppl 8.25 |
| `retnet_100m_pretrain_v4` | 2.66B token，520k 步 | 100.4M | val ppl 47.45 |
| `retnet_100m_rp_v3` | v6 | 100.4M | ⛔ 2,700 / 45,000 中断 |
| `retnet_143m_hf` | 深度生长 12→18 层 | 142.9M | 函数保持 |
| `retnet_143m_rp` | v8，45k 步 | 142.9M | eval_loss 2.6534 |
| `retnet_272m_hf` | 宽+深联合生长 | 272.3M | Δ7.25e-4 |
| `retnet_272m_rp` | v9 280.6M，20k 步 | 272.3M | eval_loss 2.9946，**基准最优** |
| `retnet_272m_v10` | v10 482.3M（R1+百科），65k 步 | 272.3M | eval_loss 2.3031；基准 0.3848 略优于 rp（p=0.084，不显著），但 **26.5% think 泄漏** |
| `retnet_272m_v13` | v13 552.3M @ seq 2048（完整 CoT） | 272.3M | 训练中（2026-10-09 夜） |

> 完整盘点（每段的日志出处、计划与实际偏差）见 [`docs/archive/restored-2026-10-24/12_训练状态盘点.md`](docs/archive/restored-2026-10-24/12_训练状态盘点.md)；
> 评测与路线重设计见 [`docs/archive/restored-2026-10-24/13_基准测试重设计与评测结果.md`](docs/archive/restored-2026-10-24/13_基准测试重设计与评测结果.md)。
>
> `chat_sft` 那版「答非所问」的三个根因（提示词被重复惩罚、数据未按样本对齐、loss 未做回复掩码）
> 与修法见 [`docs/archive/restored-2026-10-24/02_训练管线.md`](docs/archive/restored-2026-10-24/02_训练管线.md) §2 —— **别改回去**。

## 论文关键公式

```text
S_n = γS_{n-1} + K_n^H V_n                       (6)
Retention(X_n) = Q_n S_n

D_nm = γ^(n-m)  if n >= m else 0                 (5)
Retention(X) = (QK^H ⊙ D)V

γ = 1 - 2^(-5 - arange(0, h))                    (8)
MSR(X) = (swish(XW_G) ⊙ GroupNorm_h(heads)) W_O
```
