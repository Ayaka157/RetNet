# RetNet 从零实现 → 中英角色扮演对话模型

依据论文 *Retentive Network: A Successor to Transformer for Large Language Models*
（`docs/paper/2307.08621v4.pdf`）从零手写 RetNet，并在本机（RTX 3070 8GB）上
一路训练成一个 **100M 参数的中英角色扮演对话模型**。

- 日常操作命令、数据管线、故障处理 → [`RUNBOOK.md`](./RUNBOOK.md)
- 放大模型（222M / 生长方案）的理论研究 → [`docs/scaling_study.md`](./docs/scaling_study.md)

## 目录结构

```text
RetNet/
├── train.py                  唯一训练入口（三种数据模式，见下）
├── play.py                   交互推理：角色扮演 / 闲聊 / 英文续写
├── retnet/                   模型与数据核心（全部手写）
│   ├── model.py              Retention / MSR / RetNetBlock / RetNetLM + generate
│   ├── tokenizer.py          byte-level BPE（vocab 20000）
│   ├── chat.py               ⭐ 训练与推理共用的对话编码器（含 设定 persona 块）
│   ├── packing.py            文本清洗 / SFT 解析 / best-fit 行打包
│   ├── flatdata.py           int16 分片 mmap 数据集（3B+ token 不吃内存）
│   └── data.py               早期 wikitext 管线
├── scripts/
│   ├── build_chat_mix_v4.py  ⭐ 角色扮演混合数据（persona + 场景挖掘）
│   ├── build_sft_samples.py  通用 packed+masked SFT 数据（v3）
│   ├── prepare_pretrain_v2.py int16 预训练分片（可 --skip_docs 取新文本）
│   ├── download_roleplay_data.py 角色扮演/大规模中文对话数据下载
│   ├── probe_datasets.py     HF 数据集可达性探测（先探再下）
│   ├── eval_chat.py          ⭐ 固定题库评测（闲聊/指令/英文/多轮/角色扮演）
│   ├── train_loop.py         分段挂机训练（崩溃自动续训，LR 跨段连续）
│   └── merge_shards.py / validate_pretrain_data.py / monitor_training.py
├── data_cache/               所有语料与 token 缓存（只放 E 盘）
├── checkpoints/              各阶段权重
├── docs/                     论文、放大研究、Gemini 评论
├── logs/                     训练日志 / 评测报告 / 归档
├── legacy/                   已被 play.py / v4 管线取代的旧脚本
├── tests/                    test_model.py（三范式等价性等）
└── tools/, qq_decode_work/   QQ 聊天记录提取（另一条线，与本项目训练无关）
```

## 模型核心（手写，未使用现成 Transformer/RetNet 模块）

只用 `nn.Linear` / `nn.Embedding` / `nn.LayerNorm` / `nn.GroupNorm` / `nn.Dropout`，
以下逻辑全部在 `retnet/model.py` 里实现：

- Multi-Scale Retention（式 5–8）：`Q=(XW_Q)⊙Θ`、`K=(XW_K)⊙Θ̄`、`D_nm=γ^(n-m)`、
  `Retention(X)=(QKᵀ⊙D)V`、多尺度衰减 `γ=1-2^(-5-arange(h))`、
  swish 门控 `MSR(X)=(swish(XW_G)⊙Y)W_O`、每头独立 GroupNorm（Sub-LN）
- 三种范式共享同一组权重：`parallel`（训练）、`recurrent`（O(1) 推理）、`chunkwise`（块内并行）
- xPos 相对位置：论文的复数 `e^{inθ}` 用等价的实数二维旋转实现
- 输出层与词嵌入权重共享；参数量闭式 `P ≈ 12·L·d² + V·d`
- 生成：并行编码 prompt → recurrent 逐 token；支持 top-k / top-p /
  重复惩罚（**默认只惩罚新生成的 token**）/ `no_repeat_ngram_size`

## 训练数据的三种模式（train.py 自动识别）

| 模式 | 参数 | 用途 |
| --- | --- | --- |
| packed + masked | `--prepared_data_dir data_cache/chat_pack_v4` | SFT / 角色扮演：样本对齐 + 只在助手回复上算 loss |
| 分片流式 | `--shard_dirs data_cache/pretrain_v2` | 继续预训练：int16 分片 mmap，3B+ token |
| 旧扁平缓存 | `--prepared_data_dir data_cache/chat_mix_v20k` | 兼容早期数据 |

packed 数据每个 token 带权重码：`0` padding、`1` 助手回复、`2` 提问/设定、
`3` 纯文本回放；`--prompt_loss_weight 0.1` 控制提问的权重。

对话格式（`retnet/chat.py` 是唯一实现，训练与推理逐 token 对齐）：

```text
<bos>设定：你是令狐冲（出自笑傲江湖）。……
用户：{提问}
助手：{回复}<eos>
```

## 快速开始

```powershell
cd E:\Workspaces\RetNet

# 交互聊天（默认预设 chat）
C:\Tools\Miniconda3\envs\AI\python.exe play.py --mode chat

# 角色扮演固定题库评测
C:\Tools\Miniconda3\envs\AI\python.exe scripts\eval_chat.py `
    --checkpoint checkpoints/retnet_100m_chat_v3/model.pt `
    --tokenizer data_cache/chat_pack_v3/bpe_tokenizer.json --tag chat_v3

# 三范式等价性测试
C:\Tools\Miniconda3\envs\AI\python.exe tests\test_model.py
```

训练命令见 `RUNBOOK.md` 第 2 节。

## 训练历程

| 阶段 | 数据 | 参数 | 结果 |
| --- | --- | --- | --- |
| `retnet_23m` | Wikitext-2 | 23.1M | val ppl 75.30 |
| `retnet_97m_wikitext103` | Wikitext-103 | 97.6M | val ppl 50.39 |
| `retnet_100m_chat` | 混合中英对话 57.9M token | 100.4M | 会闲聊但很笨 |
| `retnet_100m_pretrain` | 1.31B token 预训练 | 100.4M | val ppl 36–39，但忘了对话格式 |
| `retnet_100m_chat_v2` | 对话恢复混合 82.9M | 100.4M | 找回聊天格式 |
| `retnet_100m_chat_sft` | 指令 SFT 36.5M（扁平） | 100.4M | ppl 13.96 但答非所问 |
| `retnet_100m_chat_v3` | **对齐 + 掩码** 194.7M | 100.4M | 回复 ppl 91.7，闲聊自然 |
| `retnet_100m_rp` | **角色扮演混合** 217M（RP 37%） | 100.4M | 训练中 |

> `chat_sft` 那版"答非所问"的三个根因（提示词被重复惩罚、数据未按样本对齐、
> loss 未做回复掩码）与修法记录在 `RUNBOOK.md` 第 1 节——**别改回去**。

## 论文关键公式

```text
S_n = γS_{n-1} + K_n^H V_n                       (6)
Retention(X_n) = Q_n S_n

D_nm = γ^(n-m)  if n >= m else 0                 (5)
Retention(X) = (QK^H ⊙ D)V

γ = 1 - 2^(-5 - arange(0, h))                    (8)
MSR(X) = (swish(XW_G) ⊙ GroupNorm_h(heads)) W_O
```
