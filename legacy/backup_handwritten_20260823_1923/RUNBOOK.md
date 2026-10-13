# RetNet 角色扮演对话模型 — 操作手册（2026-08-21 更新）

> 环境：`C:\Tools\Miniconda3\envs\AI\python.exe`　项目：`E:\Workspaces\RetNet`
> 所有大文件只放 E 盘；`HF_HOME` 已指向 `E:\Workspaces\RetNet\hf_cache`。
> 目标：**中文/英文角色扮演对话模型**，代码/数学/推理为加分项。

---

## 0. 当前状态

| 内容 | 路径 / 数值 |
| --- | --- |
| 模型架构 | RetNet 100.37M（d_model 768 / 12 层 / 12 头 / vocab 20000） |
| 预训练底座 | `checkpoints/retnet_100m_pretrain/model.pt`（1.31B token，val ppl 36–39） |
| 第一版 SFT | `checkpoints/retnet_100m_chat_sft/model.pt`（flat 数据，回复常跑偏） |
| **当前 SFT（masked 对齐版）** | `checkpoints/retnet_100m_chat_v3/`（chat_pack_v3，57k 步） |
| 对齐 + 掩码数据 | `data_cache/chat_pack_v3/`（194.7M token，reply 45% / prompt 33% / replay 20%） |
| 角色扮演数据 | `data_cache/chat_pack_v4/`（RP 为主，含 设定 persona 块） |
| 新预训练分片（int16） | `data_cache/pretrain_v2/*.npy`（目标 2.75B token，约 5.5GB） |
| 固定题库评测 | `logs/chat_eval/*.md`（`scripts/eval_chat.py` 生成） |
| 放大模型的理论研究 | `docs/scaling_study.md` |

## 1. 这一版做对了什么（关键，别改回去）

第一版 SFT 的问题不是「训练不够」，而是三个可修复的错误：

1. **数据没有按样本对齐**：旧管线把所有对话拼成一条 token 流再切 256 块，
   大部分样本从对话中间开始，模型很少见到 `<bos>用户：…\n助手：` 这个真正的推理输入。
2. **loss 算在所有 token 上**：用户提问也参与训练，模型把容量花在建模问题上。
   现在按 token 打权重码（0 padding / 1 助手回复 / 2 提问 / 3 纯文本回放），
   `--prompt_loss_weight 0.1` 表示提问只给 10% 权重。
3. **推理端两个 bug**：`repetition_penalty` 把**提问里的词**也惩罚了
   （直接把回答推离问题本身），且没有 n-gram 去重。
   现在默认 `penalize_prompt=False` + `--no_repeat_ngram_size 4`。

数据格式（`retnet/chat.py` 是唯一实现，训练与推理共用，保证 token 边界一致）：

```text
<bos>设定：你是令狐冲（出自笑傲江湖）。……      ← 权重码 2（角色扮演才有）
用户：{提问}\n助手：                            ← 权重码 2
{回复}                                          ← 权重码 1
<eos>                                           ← 权重码 1
```

## 2. 常用命令

### 2.1 聊天 / 试玩

```powershell
cd E:\Workspaces\RetNet

# 默认预设（chat = chat_v3）
C:\Tools\Miniconda3\envs\AI\python.exe play.py --mode chat

# 指定 checkpoint（角色扮演版训练完后换成 chat_v4）
C:\Tools\Miniconda3\envs\AI\python.exe play.py --mode chat `
    --checkpoint checkpoints/retnet_100m_chat_v3/model.pt `
    --tokenizer data_cache/chat_pack_v3/bpe_tokenizer.json `
    --prompt "你真好看"
```

采样参数建议（100M 模型别调太热）：`--temperature 0.7 --top_k 40 --top_p 0.9
--repetition_penalty 1.1 --no_repeat_ngram_size 4`。

### 2.2 固定题库评测（比看 ppl 靠谱）

```powershell
C:\Tools\Miniconda3\envs\AI\python.exe scripts\eval_chat.py `
    --checkpoint checkpoints/retnet_100m_chat_v3/model.pt `
    --tokenizer data_cache/chat_pack_v3/bpe_tokenizer.json `
    --tag chat_v3
```

输出 `logs/chat_eval/chat_v3.md`：闲聊 10 题 + 指令 10 题 + 英文 3 题 + 多轮 3 组。
对比不同阶段就看这几个文件。

### 2.3 继续 / 重跑 SFT（packed + masked）

```powershell
C:\Tools\Miniconda3\envs\AI\python.exe scripts\train_loop.py `
    --total_steps 57000 --segment_steps 19000 `
    --save_dir checkpoints/retnet_100m_chat_v3 --log_dir logs/chat_v3 `
    -- `
    --prepared_data_dir data_cache/chat_pack_v3 `
    --vocab_size 20000 --d_model 768 --num_layers 12 --num_heads 12 `
    --seq_len 512 --batch_size 10 `
    --warmup_steps 1000 --lr 1.2e-4 --min_lr 1.2e-5 `
    --prompt_loss_weight 0.1 `
    --mode parallel --amp bf16 --device cuda --compile `
    --init_from checkpoints/retnet_100m_chat_sft/model.pt `
    --eval_interval 2000 --eval_batches 40 `
    --chat_format --generate_prompt "你真好看"
```

中断后重跑同一条命令即可续训（`train_loop.py` 从 `last_model.pt` 接上，
并把全局 `--total_steps` 传给 cosine LR，不会再出现 PPL 锯齿）。

### 2.4 角色扮演 SFT（v4 数据）

```powershell
C:\Tools\Miniconda3\envs\AI\python.exe scripts\train_loop.py `
    --total_steps 40000 --segment_steps 20000 `
    --save_dir checkpoints/retnet_100m_rp --log_dir logs/rp `
    -- `
    --prepared_data_dir data_cache/chat_pack_v4 `
    --vocab_size 20000 --d_model 768 --num_layers 12 --num_heads 12 `
    --seq_len 512 --batch_size 10 `
    --warmup_steps 800 --lr 1e-4 --min_lr 1e-5 `
    --prompt_loss_weight 0.1 `
    --mode parallel --amp bf16 --device cuda --compile `
    --init_from checkpoints/retnet_100m_chat_v3/model.pt `
    --eval_interval 2000 --eval_batches 40 `
    --chat_format --generate_prompt "你真好看"
```

### 2.5 继续预训练（int16 分片，流式 mmap，不吃内存）

```powershell
C:\Tools\Miniconda3\envs\AI\python.exe scripts\train_loop.py `
    --total_steps 520000 --segment_steps 40000 `
    --save_dir checkpoints/retnet_100m_pretrain_v2 --log_dir logs/pretrain_v2 `
    -- `
    --shard_dirs data_cache/pretrain_v2 `
    --tokenizer_path data_cache/chat_pack_v3/bpe_tokenizer.json `
    --vocab_size 20000 --d_model 768 --num_layers 12 --num_heads 12 `
    --seq_len 512 --batch_size 10 `
    --warmup_steps 2000 --lr 2e-4 --min_lr 2e-5 `
    --mode parallel --amp bf16 --device cuda --compile `
    --init_from checkpoints/retnet_100m_chat_v3/model.pt `
    --eval_interval 4000 --eval_batches 40 `
    --generate_prompt "人工智能的发展"
```

`--total_steps 520000` ≈ 2.66B token（batch 10 × seq 512 = 5120 token/步），
按 28.8k tok/s 约 26 小时。想少跑就按比例调小。

## 3. 数据管线

### 3.1 预训练语料（int16 分片）

```powershell
# 中文教育网页（主力）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\prepare_pretrain_v2.py `
    --source chinese_fineweb_edu_v2 --max_tokens 1000000000

# 中文通用网页（含小说/论坛，角色扮演语感）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\prepare_pretrain_v2.py `
    --source skypile --max_tokens 500000000

# 英文
C:\Tools\Miniconda3\envs\AI\python.exe scripts\prepare_pretrain_v2.py `
    --source fineweb_edu --fineweb_config sample-100BT --max_tokens 500000000
C:\Tools\Miniconda3\envs\AI\python.exe scripts\prepare_pretrain_v2.py `
    --source cosmopedia --max_tokens 400000000

# 中文维基 + 数学（加分项）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\prepare_pretrain_v2.py --source wikipedia_zh --max_tokens 200000000
C:\Tools\Miniconda3\envs\AI\python.exe scripts\prepare_pretrain_v2.py --source open_web_math --max_tokens 150000000
```

要点：
- **int16 存储**，2 bytes/token（旧 int64 是 8 bytes）；vocab 20000 装得下。
- 想再取同一数据集的**新**文本，用 `--skip_docs`（脚本会在结束时打印下次该用的值，
  也记在 `data_cache/pretrain_v2/manifest.json`）。
- 训练同时准备数据时用 `RETNET_TOK_THREADS=6` 限制分词线程，别把 CPU 抢干。
- 中断了直接重跑，已完成分片不会重复（分片写入是先 `.tmp.npy` 再改名，不会留半截文件）。

### 3.2 角色扮演 / 对话数据

```powershell
# 角色扮演 + 大规模中文对话（原始 JSONL）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\download_roleplay_data.py --all

# 打包成训练用的 packed + masked 数据（RP 为主）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\build_chat_mix_v4.py `
    --out_dir data_cache/chat_pack_v4 --seq_len 512 `
    --rp_repeat 3 --instruct_scale 0.45
```

已验证可用的数据源（`scripts/probe_datasets.py` 探测，失败的会如实报错跳过）：

| 类别 | 数据集 | 规模 |
| --- | --- | --- |
| 中文 RP | silk-road/ChatHaruhi-Expand-118K | 118k 上下文 → 挖出 11.4 万条唯一场景样本 |
| 中文 RP | silk-road/ChatHaruhi-54K | 54.7k 角色问答 |
| 中文 RP | shibing624/roleplay-zh-sharegpt-gpt4-data | 2.8k（persona 质量最高） |
| 中文 RP | silk-road/Haruhi-Zero-RolePlaying-movie-PIPPA | 2.1k 影视角色卡 |
| 英文 RP | grimulkan/LimaRP-augmented、Neph0s/CoSER | 长篇 RP |
| 中文小说 | zxbsmk/webnovel_cn | 50k |
| 中文对话/指令 | Belle-3.5M、moss-003、ShareGPT-zh-90k、firefly-1.1M、smoltalk-chinese | 合计约 180 万条 |

### 3.3 数据自检

```powershell
# packed 数据抽查（解码 + 权重码分布）
C:\Tools\Miniconda3\envs\AI\python.exe -c "import numpy as np,sys; sys.path.insert(0,'.'); from retnet.tokenizer import BPETokenizer; tok=BPETokenizer.load('data_cache/chat_pack_v4/bpe_tokenizer.json'); ids=np.load('data_cache/chat_pack_v4/train.ids.npy',mmap_mode='r'); c=np.load('data_cache/chat_pack_v4/train.wcode.npy',mmap_mode='r'); print(ids.shape, ids.dtype); r=ids[0]; print(tok.decode([int(t) for t in r[c[0]>0]])[:400])"
```

`data_cache/chat_pack_v4/meta.json` 里有每个来源的 token 数与 `by_kind` 配比，
先看这个再决定要不要调 `--rp_repeat` / `--instruct_scale`。

## 4. 突发状况

### 4.1 CUDA OOM
`--batch_size 10 → 8` 或 `6`，必要时 `--grad_accum 2` 保持有效 batch；
`seq_len 512 → 256` 是最后手段（会削弱多轮/角色设定的上下文）。
显存占用参考：batch 10 ≈ 6.6GB，batch 12 ≈ 7.5GB（太满，别用）。

### 4.2 loss 变 NaN / Inf
停下来，从 `best_model.pt` 恢复，LR 砍半（`--lr 6e-5`），`--max_grad_norm 1.0` 不动。

### 4.3 进程被杀 / 断电
`train_loop.py` 每个 eval interval 存 `last_model.pt`，重跑同一条命令即可。
查当前步数：

```powershell
C:\Tools\Miniconda3\envs\AI\python.exe -c "import sys,torch; sys.path.insert(0,'.'); print(torch.load('checkpoints/retnet_100m_chat_v3/last_model.pt', map_location='cpu', weights_only=False)['step'])"
```

### 4.4 磁盘（预算 100GB）

```powershell
Get-ChildItem E:\Workspaces\RetNet -Directory | ForEach-Object { $s=(Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum; "{0,-30} {1,8:N1} GB" -f $_.Name, ($s/1GB) }
```

可回收的东西（按优先级）：
1. `data_cache/pretrain_shards`（旧 int64 分片，10GB，已合并进 `pretrain_merged`）；
2. `data_cache/pretrain_merged`（旧 int64，10GB；新 int16 分片已替代它的作用）；
3. 旧 checkpoint 的 `last_model.pt`（1.2GB/个，保留 `best_model.pt` + `model.pt` 即可）；
4. `data_cache/chat_corpus/*.jsonl`（原始语料，已编码成 npy）。

### 4.5 电脑睡眠 / 散热
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 10
```
温度：`nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv`
（3070 长跑 70℃ 属正常；想降温把 `--batch_size` 调小或限制功耗。）

## 5. 路线图（按收益排序）

1. ✅ 对齐 + 掩码 SFT 管线（chat_v3）——修掉「答非所问」的根因。
2. ⏳ 角色扮演 SFT（chat_pack_v4，设定 persona 块）。
3. ⏳ 继续预训练 +2.7B token（当前只有约 10 tokens/param，远低于 Chinchilla 的 20）。
4. 预训练后重做角色扮演 SFT（这一步的模型才是最终版）。
5. 可选：`seq_len` 提到 1024（RetNet 的二次项只占 1.4% 计算，成本低，利于长设定 + 多轮）。
6. 放大到 222M：见 `docs/scaling_study.md`，需要 ≥4 天连续 GPU 时间才划算。

---

## 6. 2026-08-21 夜间训练结果（最新状态）

| 阶段 | checkpoint | 数据 | 步数 | 结果 |
| --- | --- | --- | --- | --- |
| 继续预训练 | `checkpoints/retnet_100m_pretrain_v3/` | pretrain_v2 2.79B + 旧 1.31B（池 4.06B），实吃 363M token | 70,000 | 新验证集 ppl 57.60→56.84（已趋平） |
| **角色扮演 SFT v2（当前最佳）** | `checkpoints/retnet_100m_rp_v2/` | `data_cache/chat_pack_v5`（233.9M token，RP 41%，含 16.4M 口癖约束） | 35,000 | 回复 token ppl 18.62→**8.25**，仍在降 |

`play.py` 默认预设已切到 rp_v2：

```powershell
C:\Tools\Miniconda3\envs\AI\python.exe play.py --mode chat `
    --system "你是猫娘小白，说话可爱，**每句话结尾都要加“喵”**"
# 交互中 /set <设定> 换角色，clear 清空上下文
```

### 已验证学会的风格约束（rp_v2 vs rp v1）
句尾口癖（喵）✅、第三人称自称 ✅、高冷傲娇语气 ✅、【】动作描写 ✅、极简短句 ⚠️部分。
数据来源：`scripts/build_style_rp.py`（8 种规则改写，20.1 万条）。

### 已知待修（下一轮优先）
1. **半角逗号污染**：部分回复出现 `地方,比如说` 这类半角逗号与碎句，来自 LCCC/翻译类语料。
   修法：`retnet/packing.py` 的 `clean()` 增加半角→全角标点归一化 + 过滤碎句样本。
2. `t_cold` 模板小 bug：`rstrip('。！')` 没包含 `？`，会产出 `你怎么了？，我才不在乎。`。
3. 极简短句约束只部分生效：合成时应严格按字数截断并过滤超长样本。
4. 事实知识仍弱（100M 参数 + 约 1.4B token）。继续预训练剩余 3.7B token 语料未吃完，
   下一次长窗口直接跑满即可（约 26 小时）。
