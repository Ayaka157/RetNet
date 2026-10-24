# RUNBOOK — 操作手册

> 项目：`E:\Workspaces\RetNet`　环境：`C:\Tools\Miniconda3\envs\AI\python.exe`（本机）
> 远程机：`Ayaka@192.168.1.100`，环境 `C:\Users\Ayaka\miniconda3\envs\AI\python.exe`
> 所有大文件只放 E 盘；`HF_HOME` 已指向 `E:\Workspaces\RetNet\hf_cache`。
> 最后更新：2026-10-09

---

## 0. 当前状态

> 下表是 **2026-10-09** 的快照，不是实时状态。272M 的两个 checkpoint 目录
> （`retnet_272m_rp`、`retnet_272m_v10`）已从本地删除（见 `f66bd5c`），权重仅存于
> Hugging Face。项目当前具备的能力（与运行状态无关）见 [`FEATURES.md`](FEATURES.md)。

| 内容 | 路径 / 数值 |
| --- | --- |
| **模型（主线）** | RetNet **272.30M** — d_model 1024 / 20 层 / 16 头 / value_dim 2048 / ffn 2048 / `use_gla=true` / `use_growth_ln=true` / `growth_old_dim=768` |
| **基准最优** | `retnet_272m_rp`（20k 步，eval_loss 2.9946，基准加权总分 0.3587）— **本地目录已删除**，权重仅存于 Hugging Face |
| 最新产物 | `retnet_272m_v10`（65k 步，eval_loss 2.3031，加权总分 **0.3848**，**26.5% think 泄漏**）— **本地目录已删除**，权重仅存于 Hugging Face |
| 训练中 | `retnet_272m_v13` @ seq 2048，12,500 步，跑在远程机 |
| 次要模型线 | `retnet_143m_rp`（142.9M，eval_loss 2.6534） |
| 历史线 | `retnet_100m_*` 共 12 个目录（2026-08-19~08-23），已被取代 |
| 训练入口 | `retnet_hf/train.py`（HF Trainer）+ `run_task.py --config tasks/*.json` |
| 当前数据 | `data_cache/chat_pack_v13`（552.3M token @ seq 2048） |
| 基准工具 | `scripts/bench_v3.py` + `bench/`（**1,020 项 × 15 类，难度分层，当前**）、`scripts/bench_v2.py`（143 项，历史）、`scripts/bench_engine.py` |
| 状态盘点 / 评测 | `docs/archive/restored-2026-10-24/12_训练状态盘点.md`、`docs/archive/restored-2026-10-24/13_基准测试重设计与评测结果.md` |

---

## 1. 三条别改回去的设计

第一版 SFT（`retnet_100m_chat_sft`，ppl 13.96）评测时答非所问。贪心解码证明条件相关性其实是好的
（问番茄炒蛋答得对），真正的三个 bug：

1. **推理端**：`repetition_penalty` 惩罚了**提问里的词**，把回答推离问题。
   修法：默认只惩罚新生成的 token（`penalize_prompt=False`）+ `no_repeat_ngram_size=4`。
2. **数据未按样本对齐**：旧管线把对话拼成一条流再切 256 块，样本大多从对话中间开始，
   模型很少见到推理时真正的输入 `<bos>用户：…\n助手：`。
3. **loss 算在所有 token 上**：提问也参与训练，容量浪费在建模问题上。

现在的设计（见 `docs/archive/restored-2026-10-24/02_训练管线.md` §2）：

- 每样本 = `<bos>[设定：persona\n]用户：{q}\n助手：{a}<eos>`，BOS 对齐、EOS 收尾、**截断保头**
- 每 token 带权重码：`0` padding / `1` 助手回复 / `2` 提问与设定（权重 0.1）/ `3` 纯文本回放
- 编码器唯一实现 `retnet/chat.py`，**训练与推理共用**，保证 token 边界一致
- persona 走「设定：」块，prompt 权重——模型学「按设定说话」，不学「生成设定」

---

## 2. 常用命令

### 2.1 试玩

```powershell
cd E:\Workspaces\RetNet

# 默认预设（100M 的 rp_v2）
C:\Tools\Miniconda3\envs\AI\python.exe play.py --mode chat

# 272M —— checkpoint 目录里没有 tokenizer，必须去 data_cache 借
C:\Tools\Miniconda3\envs\AI\python.exe play.py --mode chat `
    --checkpoint checkpoints/retnet_272m_rp `
    --tokenizer data_cache/chat_pack_v9/bpe_tokenizer.json `
    --system "你是猫娘小白，说话可爱，**每句话结尾都要加“喵”**"
# 交互中 /set <设定> 换角色，clear 清空上下文
```

采样参数建议：`--temperature 0.7 --top_k 40 --top_p 0.9 --repetition_penalty 1.1 --no_repeat_ngram_size 4`。

### 2.2 评测与基准

```powershell
# === 基准 v3（当前）——1,020 项 × 15 类，难度分层，判据声明式 ===
# 第一步：构建题库（重新生成机械类 → 全量校验 → 冻结 + SHA256 manifest）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\compile_bench.py

# 第二步：跑分。1,020 项 × 1 种子 ≈ 42 分钟（272M fp32，RTX 3070）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_v3.py `
    --models "rp=checkpoints/retnet_272m_rp,v10=checkpoints/retnet_272m_v10" `
    --tokenizer data_cache/chat_pack_v10/bpe_tokenizer.json `
    --seeds 42,43,44 --out logs/bench_v3
# 输出 logs/bench_v3/<name>.<tag>.jsonl（逐条回复，流式追加、可断点续跑）
#      + summary.<tag>.md + report.<tag>.json（含 McNemar 配对检验与区分度诊断）

# 只重出报告（换聚合口径不用重跑 GPU）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_report.py `
    --dir logs/bench_v3 --out logs/bench_v3 --families

# 冒烟：每 40 项取 1，约 1.5 分钟
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_v3.py `
    --models "rp=checkpoints/retnet_272m_rp" `
    --tokenizer data_cache/chat_pack_v10/bpe_tokenizer.json `
    --stride 40 --seeds 42 --out logs/bench_v3_smoke

# ⚠ --batch 16 快约 1.5 倍，但批的 GEMM 与单行 GEMM 不是逐位相同，
#    ~1e-6 的 logit 差异会偶尔翻转采样（实测 107 项抽样里 8 项回复不同）。
#    默认 --batch 1 逐位一致。要可复现的分数就别开它。

# 基准自带的三套测试
C:\Tools\Miniconda3\envs\AI\python.exe tests\test_checkers.py          # 判据 DSL 语义
C:\Tools\Miniconda3\envs\AI\python.exe tests\test_bench_equivalence.py # 解码器逐位等价
C:\Tools\Miniconda3\envs\AI\python.exe tests\test_runner_lockstep.py --stride 97

# === 基准 v2（历史，143 项 × 9 类）——纵向对比时用 ===
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_v2.py `
    --models "rp=checkpoints/retnet_272m_rp,v10=checkpoints/retnet_272m_v10" `
    --tokenizer data_cache/chat_pack_v10/bpe_tokenizer.json `
    --seeds 42,43,44,45,46,47,48,49 --out logs/bench_v2
# 输出 logs/bench_v2/<name>.json（含逐条回复原文）+ summary.md + 逐项区分度报告

# 工程基准：parallel↔chunkwise 等价性 / 吞吐 / 显存
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_engine.py `
    --model_dir checkpoints/retnet_272m_rp --seq_lens 256,512,1024 `
    --out logs/bench/engine_272m_rp.json

# 解码成本分解 / batching 收益实测（决定基准吞吐口径的两个数）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_decode_profile.py `
    --model_dir checkpoints/retnet_272m_rp
C:\Tools\Miniconda3\envs\AI\python.exe scripts\bench_batch_scaling.py --items 48

# 旧的人工打分题库（保留用于与历史报告对比）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\eval_chat.py `
    --checkpoint <ckpt>/model.pt --tokenizer <data>/bpe_tokenizer.json --tag <名称>

# 三范式等价性（⚠️ 只在 d_model=64 / L=16 玩具尺度成立）
C:\Tools\Miniconda3\envs\AI\python.exe tests\test_model.py
```

### 2.3 训练（本机）

```powershell
# 272M 角色扮演 SFT —— 当前标准配置
C:\Tools\Miniconda3\envs\AI\python.exe retnet_hf\train.py `
    --model_dir checkpoints\retnet_272m_rp `
    --data_dir data_cache\chat_pack_v13 `
    --output_dir checkpoints\retnet_272m_v13 `
    --max_steps 12500 --batch_size 4 --lr 1.1e-4 --min_lr 1.1e-5 --warmup 300 `
    --mode chunkwise --chunk_size 64 --precision bf16 `
    --grad_checkpoint --no-compile

# 断点续训：加 --resume（从 output_dir 最后一个 checkpoint 接着跑）
```

实测吞吐（RTX 4070 Laptop 8GB，272M）：

| 配置 | 步时 | 吞吐 | 显存 |
| --- | --- | --- | --- |
| seq 512, batch 4, fp32 | 0.597 s/it | 3,430 tok/s | 5.5 GB |
| seq 512, batch 4, bf16 | 0.480 s/it | 4,267 tok/s | 5.5 GB |
| **seq 2048, batch 4, bf16** | **2.283 s/it** | 3,589 tok/s | 7.58 GB |
| seq 4096, batch 4, bf16（`retnet_hf_fp16/train.py`） | 12.34 s/it | 1,328 tok/s | — |

> 每 token 效率随窗口只轻微下降（512→2048 掉 16%），但 **fp16 加固分支在 4096 上慢 3 倍**，
> 别拿它的数字外推。每次 eval（`eval_batches=40`）约 97 秒 @512、约 390 秒 @2048。

### 2.4 训练（远程机，推荐）

```powershell
# 1) 传数据与配置
scp -i ~/.ssh/id_ed25519 -r data_cache\chat_pack_v13\* `
    Ayaka@192.168.1.100:C:/Users/Ayaka/RetNet/data_cache/chat_pack_v13/
scp -i ~/.ssh/id_ed25519 tasks\v13_272m_2k.json `
    Ayaka@192.168.1.100:C:/Users/Ayaka/RetNet/tasks/

# 2) 启动 —— 必须用计划任务，Start-Process 会被 SSH 断开时连带杀掉
ssh Ayaka@192.168.1.100 'schtasks /create /tn RetNet_v13 /tr "cmd /c C:\Users\Ayaka\RetNet\tasks\start_v13.bat" /sc once /st 23:58 /f'
ssh Ayaka@192.168.1.100 'schtasks /run /tn RetNet_v13'
ssh Ayaka@192.168.1.100 'schtasks /delete /tn RetNet_v13 /f'

# 3) 看进度
ssh Ayaka@192.168.1.100 'powershell -NoProfile -Command "Get-Content C:\Users\Ayaka\RetNet\logs\v13_272m_2k.log -Tail 5"'
ssh Ayaka@192.168.1.100 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```

> ⚠️ **`Start-Process` 起不来。** SSH 会话结束时 Windows 会连带杀掉整棵进程树，
> 日志里只留一行启动横幅、GPU 占用为 0。**必须走 `schtasks`**，它不隶属 SSH 会话。

---

## 3. 数据管线

### 3.1 对话 / SFT 数据

```powershell
# v13（当前）：v9 基座重切到 2048 + R1/wiki/锐智吧用完整 CoT 重编码
C:\Tools\Miniconda3\envs\AI\python.exe scripts\build_chat_mix_v13.py `
    --base data_cache/chat_pack_v9 --out_dir data_cache/chat_pack_v13 `
    --seq_len 2048 --r1_max_chars 4000

# v10 模板（512 窗口）：--strip_think 剥离 <think>，--r1_max_chars 抬高字符上限
C:\Tools\Miniconda3\envs\AI\python.exe scripts\build_chat_mix_v10.py `
    --base data_cache/chat_pack_v9 --out_dir data_cache/chat_pack_v12 `
    --seq_len 512 --strip_think

# v9 模板（RP/指令/闲聊主力，改预算用 --*_tokens）
C:\Tools\Miniconda3\envs\AI\python.exe scripts\build_chat_mix_v9.py `
    --out_dir data_cache/chat_pack_v9 --seq_len 512
```

> ⚠️ **不要用 `build_chat_mix_v9.py --seq_len 2048` 从原始语料重建大窗口**——
> 长文本读取器会涨到 13 GB RSS 并被 OOM 杀掉。宽窗口用 `build_chat_mix_v13.py`：
> 它只重切 v9 已有的 token 流，仅对 R1 重新编码。

### 3.2 预训练语料（int16 分片）

```powershell
C:\Tools\Miniconda3\envs\AI\python.exe scripts\prepare_pretrain_v2.py --source chinese_fineweb_edu_v2 --max_tokens 1000000000
C:\Tools\Miniconda3\envs\AI\python.exe scripts\prepare_pretrain_v2.py --source skypile --max_tokens 500000000
C:\Tools\Miniconda3\envs\AI\python.exe scripts\prepare_pretrain_v2.py --source fineweb_edu --fineweb_config sample-100BT --max_tokens 500000000
```

- **int16 存储**，2 bytes/token（旧 int64 是 8 bytes），vocab 20000 装得下
- 想再取同一数据集的**新**文本用 `--skip_docs`（脚本会打印下次该用的值，也记在 `manifest.json`）
- 训练同时准备数据时用 `RETNET_TOK_THREADS=6` 限制分词线程
- 中断直接重跑，已完成分片不会重复（先写 `.tmp.npy` 再改名）

### 3.3 数据自检

```powershell
C:\Tools\Miniconda3\envs\AI\python.exe -c "import numpy as np,sys; sys.path.insert(0,'.'); from retnet.tokenizer import BPETokenizer; tok=BPETokenizer.load('data_cache/chat_pack_v13/bpe_tokenizer.json'); ids=np.load('data_cache/chat_pack_v13/train.ids.npy',mmap_mode='r'); c=np.load('data_cache/chat_pack_v13/train.wcode.npy',mmap_mode='r'); print(ids.shape, ids.dtype); r=ids[0]; print(tok.decode([int(t) for t in r[c[0]>0]])[:400])"
```

`meta.json` 里有每个来源的 token 数与配比；宽窗口包还要看 `<think>` / `</think>` 的**闭合率**
（v13 是 86%，v10 只有 23.7%）。

---

## 4. 突发状况

### 4.1 CUDA OOM
`--batch_size 4 → 2`，必要时 `--grad_accum 2` 保持有效 batch。
seq 2048 batch 4 在 8GB 卡上占 7.58 GB，**只剩约 600 MB 余量**，别再加 batch。

### 4.2 loss 变 NaN / Inf
- 训练侧：停下来，从最后一个 checkpoint 恢复，LR 砍半，`--max_grad_norm 1.0` 不动
- **生成侧出现 `<think>` 收不了尾**：这不是 NaN，是数据缺陷——读取端字符上限 + 打包端窗口
  两级截断把 `</think>` 与答案切掉了。修数据，不是调采样参数

### 4.3 进程被杀 / 断电
`run_task.py` 配了 `auto_restart_max`，崩溃后自动 resume。手动续训：

```powershell
C:\Tools\Miniconda3\envs\AI\python.exe run_task.py --config tasks\v13_272m_2k.json --resume
C:\Tools\Miniconda3\envs\AI\python.exe run_task.py --config tasks\v13_272m_2k.json --status
```

### 4.4 远程训练没起来
按顺序查：`Get-Process python`（有没有进程）→ 日志字节数（只有横幅说明没进主循环）
→ `nvidia-smi`（占用是否为 0）。**用 `Start-Process` 起的必然是这个症状，改 `schtasks`。**

### 4.5 磁盘
```powershell
Get-ChildItem E:\Workspaces\RetNet -Directory | ForEach-Object { $s=(Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum; "{0,-30} {1,8:N1} GB" -f $_.Name, ($s/1GB) }
```
清理记录与可回收清单见 `docs/archive/restored-2026-10-24/06_清理与运维.md`。

### 4.6 电脑睡眠 / 散热
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 10
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv
```

---

## 5. 路线图

**以 `docs/archive/restored-2026-10-24/13_基准测试重设计与评测结果.md` §5 为准**。摘要：

| 优先级 | 内容 |
| --- | --- |
| **P0** | **基准 v2 扩项**——现有 143 项里仍有大量题贴地板（结构化/数学）或贴天花板（角色扮演），区分度不足 |
| **P1** | 重打包 v12：R1 数据剥离 `<think>` |
| **P2** | 用 `retnet_272m_v10` 当负对照跑 v12 冒烟，确认泄漏率从 26.5% 归零 |
| **P3-A** | 从 `retnet_272m_rp` 做 RP 特调（`docs/archive/restored-2026-10-24/09_272M_v10_方案备份.md` 阶段 B 原计划，**从未执行过**） |
| **P3-B** | 长上下文 4K——实测 12.34 s/step，跑满 30k 步要 103 小时 |
| **P4** | 生产尺度等价性回归 + `.git` 初始化 |

### 已验证走不通的（别再走）

- **在 272M 上追「综合能力」**：v10 的 65k 步在 143 项基准上只是**略优但不显著**
  （配对 McNemar p=0.084，总分 +0.026），代价 10 小时 46 分 + 一个 26.5% 的 think 泄漏。
  结构化/数学是尺度限制（`docs/archive/restored-2026-10-24/04_路线与扩展研究.md` 早判过 CoT 有效下限在 0.5–1B），不是数据缺口。
- **v11 的 8K/4K 路线**：`build_chat_mix_v11.py` 是在**已截断的 512 行上重切窗口**，
  截断在 v10 构建时就固化进 token 了，**重切救不回来**。两次运行都被人为中断、零产出。
- **fp16**：两个种子在第 20、40 步就 `loss 0.0 / grad_norm NaN`，bf16 同期正常。

---

## 6. 训练结果速查

完整 18 阶段表格（含每段日志出处）见 `docs/archive/restored-2026-10-24/12_训练状态盘点.md` §2。最近几段：

| 阶段 | checkpoint | 数据 / 步数 | 结果 |
| --- | --- | --- | --- |
| 272M 角色扮演 | `retnet_272m_rp` | chat_pack_v9 280.6M，20k 步 | eval_loss **2.9946**，1.498 steps/s，3h42m |
| 272M 综合 v10 | `retnet_272m_v10` | chat_pack_v10 482.3M，65k 步跑满 | eval_loss 2.3031，但 26.5% think 泄漏 |
| 272M 综合 v13 | `retnet_272m_v13` | chat_pack_v13 552.3M @ 2048，12.5k 步 | 训练中 |
| 143M 角色扮演 | `retnet_143m_rp` | chat_pack_v8，45k 步 | eval_loss 2.6534 |
| 100M 角色扮演 v2 | `retnet_100m_rp_v2` | chat_pack_v5 233.9M，35k 步 | 回复 ppl 8.25（100M 线最优） |

> ⚠️ **跨数据集的 eval_loss 不可比。** v10 的 2.3031 与 rp 的 2.9946 用的是不同验证集
> （v10 val 2,866 行 / rp val 1,561 行）。可比的是基准分：**v10 0.3848 vs rp 0.3587**
> （143 项 × 8 种子，配对 p=0.084，方向是 v10 略优但不显著）。
