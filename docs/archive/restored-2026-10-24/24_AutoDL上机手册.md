# 24. AutoDL 上机手册（0.5B 训练）

> 目的只有一个：**把 debug 从付费的 GPU 时段里挪出去**。
> 本机已经把这套东西准备好并做过正向对照；实例上只需要照着敲。

配套文件（都已在仓库里，且**已在本机验证过**）：

| 文件 | 作用 | 验证状态 |
|---|---|---|
| `scripts/linux/env_check.py` | 一条命令判断「这台机器现在能不能开训」 | 【实测】本机跑通：8 GB 卡上正确地只失败「显存≥20 GiB」一项，其余全过；真实前向+反向通过（loss 7.2722、287 个参数拿到非零梯度） |
| `scripts/linux/speed_probe.sh` | 量「免检查点」到底快多少，并算出完整训练要多少钱 | 【实测】`bash -n` 通过；解析逻辑经 4 个正向 + 1 个反向对照 |
| `scripts/linux/_test_parse.sh` | 上面前者的解析自测 | 【实测】全过（含「OOM 日志不得解析出指标」的反向对照） |
| `scripts/linux/run_train.sh` | 训练启动器（显存前置检查 / 超时 / 可续跑 / 结尾自动关机） | 【实测】`bash -n` 通过 |
| `scripts/linux/upload_manifest.txt` | 上传清单（3.43 GB，显式列举，避免误传几十 GB） | 【实测】逐项核对过大小 |

---

## 1. 预算与定位（先读这一段）

> ⚠️ **单价以实例页为准，不要照抄本文。** 创建时的报价页（另一台机器，12 核/43 GB）
> 显示 ¥1.32/时，但**实际开的这台是 14 vCPU / 60 GB，单价 ¥1.48/时（原价 ¥1.56/时）**。
> 差 12% 看着小，但在「¥20 买几小时」这种问题上会直接决定一次训练能不能跑完。
> **上机后第一件事：看实例详情页的「费用」那一行，用真实单价算预算。**

按 **¥1.48/时** 计，充值 **¥20** 可买 **约 13.5 个 GPU 小时**（按 ¥1.32 估算会高估到 15.2 小时）。

**下表已在 2026-10-11 夜换成实测**（`tasks/v15b_500m_24g.json`，b4a1 免检查点；
**最终**实测的有效步时间 = **601.6 ms/优化步**，含评测与存档摊销）：

| 目标 | 需要 | ¥20 够不够 |
|---|---|---|
| 验证日（上传 + env_check + 测速 + 彩排 + 冒烟） | **实测 2.4 GPU 小时 = ¥3.6** | ✅ 够（原推算 3.5 h / ¥5.2，实测更省） |
| 0.5B 跑满 1 个 epoch（552 M token，67,390 步） | **实测 11.21 h = ¥16.6**（纯计算 10.88 h + 评测/存档 19.6 min） | ⚠️ 一次充值不够：扣掉验证日的 ¥3.6 后只剩 11.2 h ≈ 刚好卡住（追加 ¥5 后才跑完，见 §9.6） |
| token/参数 到 10（5.04 B token） | 9.13 个 epoch ≈ **102.3 h = 4.3 天** | ❌ 约需 **¥151** |
| Chinchilla 20×（10.08 B token） | 18.26 个 epoch ≈ **204.7 h = 8.5 天** | ❌ 约需 **¥303** |

> ⚠️ **"步数"不是一把统一的尺子，比之前先问"每步多少 token"。** 上表的 67,390 步是
> **每优化步 8,192 token**（global batch 4）。若按常见文献口径 **global batch 512**
> （1.05 M token/步）记账，同一个 epoch 只有 **527 步**，同一个 10.08 B token 目标是
> **9,616 步** —— 差 128 倍，**而墙钟时间一模一样**（梯度累积不省时间，只改记账）。
> 详见 §9.11。

**推算 → 实测的修正**：原表写 1 epoch「约 12.8 h（¥18.9）」，实测 **11.2 h（¥16.6）**；
4.9 天 → **4.3 天**；9.8 天 → **8.5 天**。三个目标**全部比原估快约 9%**
（原推算隐含 684 ms/步，实测有效值 601.6 ms/步，含评测与存档摊销）。
⚠️ 原表那三个数是**推算**（口径：桌面 3070 实测 + 免检查点倍数外推），
本次是 3090 上的**真实训练路径**实测 —— **两者分母不同，不要混用**；
需要同卡对照时用 `docs/23` §4.7(e) 的 944 vs 510 ms（1.851×）。

> ⚠️ **余额归零会自动停实例。** 所以永远不要用余额去起一个比余额更长的训练——
> 你会得到一个腰斩的模型。所有启动命令都走 `run_train.sh`，它提供 `--max-hours` 硬上限
> （配合 `--save-steps` 可 `--resume` 续）。**更稳的是控制旗标**：`_graceful_stop.sh`
> 会到点先 `--ctrl save` 让检查点落盘，再 `--ctrl stop` 让训练正常退出（rc=0），
> 避免 `timeout` 的 SIGTERM 落在 60 秒写盘窗口里。§9 有整套实跑记录。

---

## 2. 上传（在**无卡模式**下做，¥0.1/时）

### 2.1 实例侧

1. 按量计费创建实例，镜像选 **PyTorch 2.8.0 / Python 3.12(ubuntu22.04) / CUDA 12.8**。
2. **先正常开机一次**，`nvidia-smi` 确认是 3090/24576MiB，然后**立刻关机**。
3. 关机后用 **无卡模式** 开机（0.5 核 / 2 GB 内存 / 无 GPU，¥0.1/时）。
4. 装依赖（走清华源，别重装 torch）：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers==4.57.6 accelerate huggingface_hub bitsandbytes safetensors
```

> `transformers==4.57.6` **不要改版本**：我们的检查点由它产出，而 HF Trainer 的
> loss 归一化跨版本会变（`modeling_retnet.py` 里有一条注释专讲 `num_items_in_batch`
> 的启发式）。版本不一致不会报错，只会让 loss 悄悄变了样。

### 2.2 本机侧上传

AutoDL 实例列表页有 SSH 连接信息。**不要把密码给 AI**——在本机配好别名与密钥即可：

```bash
# 本机 ~/.ssh/config 追加（HostName/Port 从实例页复制）
Host gpu-host
    HostName <gpu-host>
    Port 12345
    User root
    StrictHostKeyChecking no

# 一次配好免密（会提示输入一次密码，你自己输）
ssh-copy-id gpu-host
```

然后按清单上传（**只传 3.43 GB，不是整个仓库**）：

```bash
# 在 WSL 或 Git Bash 里跑（Windows 的 scp 也行，rsync 更省心）
cd /mnt/e/Workspaces/RetNet      # 或你的仓库路径
ssh gpu-host 'mkdir -p /root/gpu-scratch/RetNet'
rsync -avz --progress --files-from=scripts/linux/upload_manifest.txt \
      --exclude '__pycache__' ./ gpu-host:/root/gpu-scratch/RetNet/
```

**为什么用清单而不是 `rsync -av ./`**：仓库里的 `checkpoints/`、`hf_cache/`、`Kaggle/`
加起来是几十 GB，误传会白等几小时（无卡模式虽然便宜，但你的时间不便宜）。

---

## 3. 实例上的四步

```bash
cd /root/gpu-scratch/RetNet
source /etc/network_turbo          # 只有访问 GitHub/HuggingFace 才需要；用多了会 503
```

### 第 1 步：切回正常开机，跑环境自检

```bash
python scripts/linux/env_check.py
```

**预期**：所有项 `[PASS]`，包括 `显存 >= 20 GiB`、`transformers == 4.57.6`、
`参数量 == 504,179,896`、`gamma 最小值 > 0`，最后打印

```
✅ 全部通过。
   可以开训：bash scripts/linux/run_train.sh --config tasks/v15_500m.json
```

任何 `[FAIL]` 都带 `↳` 修复提示。**这一项不过就别往下走**——它就是为了把问题挡在付费时段之外。

### 第 2 步：量速度（这是验证日的核心）

```bash
bash scripts/linux/speed_probe.sh
```

三个探针串行跑，每个独立超时 600 秒，跑完自动跳过（可续跑）。约 10 分钟。
**预期**末尾会打印：

```
**免检查点的实测倍数 = 3300/1750 = 1.886×**（推算值 1.96×）
  nockpt       1 epoch = 11.5 小时 = ¥15.2；20×Chinchilla(10.08B token) = 8.8 天 = ¥278
  nockpt_b2a2  1 epoch = 9.8 小时 = ¥12.9；20×Chinchilla(10.08B token) = 7.5 天 = ¥237
```

（上面的数字是**示例格式**，真实值以实例输出为准。）

**把 `logs/opt_linux/results.txt` 贴回来即可** —— 不需要贴 GPU 日志全文。

### 第 3 步：冒烟（证明训练循环 + 存档 + 恢复都通）

```bash
bash scripts/linux/run_train.sh --max-steps 30 --save-steps 15 --no-shutdown
ls -1dt checkpoints/retnet_500m_v15/checkpoint-* | head -3
```

**预期**：30 步跑完，出现 `checkpoint-15`；loss 有限且在动。
`--no-shutdown` 是故意的——调试阶段别自动关机。

### 第 4 步：正式训练（想清楚再跑）

```bash
# 先算好：预算 ÷ 单价 = 可跑小时数。例：¥8 ÷ ¥1.32 = 6 小时
bash scripts/linux/run_train.sh --save-steps 1000 --max-hours 6
```

`--max-hours` 是硬上限，到点强杀（配合 `save_steps` 可 `--resume` 续）；结束后自动关机。

---

## 4. 省钱规则（逐条都有依据）

| 规则 | 依据 |
|---|---|
| 不需要 GPU 的事一律在**无卡模式**做 | 官方文档：无卡模式统一 ¥0.1/时 |
| 所有启动命令走 `run_train.sh`（默认结尾 `shutdown`） | 官方「省钱绝招」场景三 |
| **关机 ≠ 释放**：关机后数据保留 15 天且不计费 | 官方文档；释放才删数据 |
| 数据放 `/root/gpu-scratch/` | `/root/` 只有 30 GB，而 4 份检查点就 16 GB |
| 开训前 `env_check.py` + 跑完 `--max-hours` | 防止在付费时段 debug、防止忘记关机 |
| 同一时刻只跑一个 GPU 作业 | docs/23 §9.1/§9.2 的事故记录 |

---

## 5. 抢不到卡的备选（按顺序）

当前实测可用性（2026-10-11）：

| 型号 | 显存 | 约价 | 可用性 |
|---|---|---|---|
| **RTX 3090** | 24 GB | ¥1.32/时 | 很稀缺（六个区大多是 0/几百）——**首选，但要抢** |
| RTX 4090D | 24 GB | ¥1.88/时 | 西北B 曾有 1 台 |
| RTX 4090 | 24 GB | ¥1.98/时 | 北京B / 内蒙B 各 1 台 |
| A100-PCIE-40GB | 40 GB | 约 ¥3.5/时 | **佛山区 3 台**——最稳的保底，且 40 GB 能测 batch 4 |

**不要租**：`RTX 3080 Ti`（12 GB 不够免检查点）、`V100 / 2080 Ti`（Volta/Turing **无 bf16 tensor core**，
而训练是 `bf16=True`，会走退化路径）、`RTX PRO 6000 / H800 / A800 / H20`（显存够但单价太高，
且测出的速度外推不到你要训练的卡上）、`vGPU-*`（切分卡，性能与计价不透明）。

---

## 6. 故障排查

| 症状 | 原因 | 处置 |
|---|---|---|
| `env_check` 报「没有 nvidia-smi」/ `cuda.is_available()` False | 你在**无卡模式** | 关机后切回正常模式 |
| `transformers` 版本不是 4.57.6 | 装漏了或装了别的 | 按 §2.1 的 pip 命令重装 |
| `bitsandbytes` 导入失败 | 没装 | `pip install bitsandbytes` |
| `speed_probe` 三个探针全 TIMEOUT | 首次 `torch.compile` 在冷启动上很慢 | 调大 `--timeout 900` 重跑（会跳过已完成的） |
| `nockpt` 探针 OOM | 卡不是 24 GB，或其它进程占着卡 | 看 `nvidia-smi`；确认 `memory.total` |
| 训练 loss 是 `nan` 或恒为 0 | 模型 `gamma=0` 缺陷（docs/23 §8.9） | `env_check` 的 `gamma 最小值 > 0` 就是防这个；不满足则重新生长 |
| 余额归零实例被停 | — | 数据保留 15 天；充值后用 `--resume` 续训 |
| SSH 断开后训练进程没了 | 没用 `nohup`/`tmux` | `run_train.sh` 建议配 `tmux new -s t` 后再跑 |

---

## 7. 一句话总结流程

```
（本机）配好 SSH 别名 → rsync 清单 3.43 GB
（实例）创建 → 开机确认 3090 → 关机 → 无卡模式 → pip 装包
（实例）正常开机 → env_check.py → speed_probe.sh → 冒烟 30 步 → run_train.sh --max-hours N
（把 logs/opt_linux/results.txt 贴回来）
```

---

## 8. 首次上机实测踩坑记录（2026-10-11，内蒙B区 429机 / RTX 3090）

> 这一节全部是**实测**，不是推测。下次上机前先看这一节，能省掉当时花掉的那一两个小时。

### 8.1 ⚠️ 不要用无卡模式（与 §2.1 相反，因为卡源变了）

§2.1 让你用无卡模式（¥0.1/h）上传装包。**执行时改掉了**：AutoDL 官方文档写明
无卡模式**会释放 GPU**，而当时 3090 在西北B/重庆A/内蒙B/北京A 四个区的可用数**全是 0**。
为了省 ¥0.5 去赌一张抢不到的卡不划算。**正确做法：保持正常开机边付费边上传**
（实测上传 3.43 GB 期间总共多花约 ¥0.5–1）。

**判据**：先看目标卡型的区域可用数。充足 → 用无卡模式省钱；稀缺（个位数）→ 坚决别释放。

### 8.2 非交互 ssh 的 PATH 里没有 conda → `python3: command not found`

`ssh gpu-host "python3 -V"` 会报 `command not found`；因为镜像的 conda 只在交互式
登录 shell 里初始化，而 `ssh host "cmd"` 走的是**非交互 shell**。

解释器实际在 **`/root/miniconda3/bin/python` = Python 3.12.3**。
→ 所有脚本都加了 `find_python()` 兜底探测（依次试 `$PYTHON` / `python3` / `python` /
`/root/miniconda3/bin/python` / `/opt/conda/bin/python`），并把它的目录前置进 PATH。

### 8.3 ⚠️ PowerShell 往 ssh 送 stdin 会把 LF 变成 CRLF

这是当晚最耗时的坑。**实测**：把一个 12 字节的 `line1\nline2\n` 用 PowerShell 管道送进
`ssh gpu-host "cat > file"`，落地变成 `line1\nline2\n\r\n`；而本机的 `.sh` 文件本身
**CR 数量为 0**（纯 LF）。

后果：`Get-Content x.sh -Raw | ssh gpu-host "bash -s"` 送过去的脚本，在 Linux 上报
`bash: line N: $'\r': command not found`。

**规矩**：脚本一律 **scp 传（二进制安全）**，远程用 `bash <路径>` 执行；
**永远不要把脚本正文用 PowerShell 管道送进 ssh**。
另外：不要因为看到 `$'\r'` 就去改文件的换行——先分清是**文件**的问题还是**传输**的问题。

### 8.4 小文件多时，瓶颈是延迟不是带宽（别据此判断"网太慢"）

**实测**：280 个代码文件（共 6.3 MB）花了 **137 秒**，看着像 46 KB/s；
但单个 1.1 GB 大文件能跑到 **约 2.3 MB/s**。差异全在**每文件一次往返**。

当晚我一度用 `du -sb` 采样得到一个 **7.3 KB/s** 的结论，并据此判断"3.43 GB 要传 5.8 天"
——**那是测量假象**：采样点正好落在两个文件之间的间隙。
随后用单文件 20 MB 干净测速，得到 **1.57 MB/s**（36 分钟传完），才是真值。

**教训**：判断吞吐要用**单个大文件**测；用目录体积采样去估速率，在"多小文件"场景下会低估一两个数量级。

### 8.5 环境实测值（PyTorch 2.8.0 镜像）

| 项 | 实测 |
|---|---|
| Python | 3.12.3（`/root/miniconda3/bin/python`） |
| torch | **2.8.0+cu128**，CUDA 12.8，`cuda.is_available()=True` |
| GPU | RTX 3090，24576 MiB，驱动 595.71.05，算力 (8,6) |
| triton | 3.4.0（`torch.compile` 可用）；gcc/g++ 已装 |
| numpy | 2.3.2 |
| 预装缺失 | transformers / bitsandbytes / accelerate / safetensors → 需 pip 装 |
| 容器规格 | **14 核 / 60 GB 内存**（`nproc` 返回 112 是宿主机总核数，**不要拿它做规划**） |
| 磁盘 | 系统盘 `/` 30 GB；数据盘 `/root/gpu-scratch` 50 GB（独立盘） |

装依赖（实测成功，约 1 分钟）：

```bash
/root/miniconda3/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers==4.57.6 accelerate huggingface_hub bitsandbytes safetensors
# 实得：transformers-4.57.6 bitsandbytes-0.50.2 accelerate-1.15.0
#       safetensors-0.8.0 huggingface_hub-0.36.2 tokenizers-0.22.2
# 注意 bitsandbytes 版本与本机一致（0.50.2），这点很关键
```

### 8.6 SSH 别名写入：**永远不要「已存在就复用」**

见 §2.2 的脚本。第一版实现是「`Host gpu-host` 已存在就直接复用」，结果：一个写错的
`HostName root@<gpu-host>`（把 ssh 命令里的 `root@` 一起粘进去了）
被永久沿用，ssh 去解析 `root@connect...` 这个名字 → `Could not resolve hostname`。

现在的实现：**入口剥掉 `@`** + **总是校验并重建该段** + **占位符主机名直接拒绝**。
（与 docs/23 §8.9 那个「空过的等价性测试」是同一类错误：检查存在，但它没有能力失败。）

### 8.7 快照：`/etc/profile.d/gpu-host.env.sh` 里有面板 token

`head` 这个文件会输出 `AutodlAutoPanelToken=...`。**别 cat 它、别往对话里贴**。
上表信息（核数/内存/磁盘）在 AutoDL 自己的欢迎横幅里就有，不需要读这个文件。

---

## 9. 正式训练实跑记录（2026-10-11 夜，0.5B / RTX 3090）

### 9.1 实际起跑与配置

- 实例：内蒙B区 RTX 3090 24 GB，按量。
  **计费起点 = 容器里 `/etc/hostname` 的生成时间 16:07:39**——不是"我什么时候上传代码"，
  也不是"我什么时候创建"。`/proc/uptime` 是**宿主机**的 13 天，**不能用来算计费时长**。
- 配置：`tasks/v15b_500m_24g.json` —— 504.18M 稠密、bf16、chunk128、
  `grad_checkpoint: false`、`batch_size 4 / grad_accum 1`、`adamw_bnb_8bit`、
  `save_steps = eval_steps = 5000`、`max_steps 67390`（= 1 epoch）。
- 起跑 18:31:30，一条命令：

  ```bash
  bash scripts/linux/_launch_real.sh 10.5     # 峰值采样 + 训练 + 到点优雅收尾
  ```

- 起跑自检（全部通过）：`chunkwise kernel: batched (FAST_CHUNKWISE=1)` ·
  504.18M 参数 · train 269561 / val 833 · `torch.compile(inductor)` 生效 ·
  首 190 步 1.64 it/s = **610 ms/优化步** · GPU 23,230 MiB / 24,576、327 W、100%、79 °C。

### 9.2 预算算术（可复算；当时结论是"装不下一个 epoch"——⚠️ 20:13 追加 ¥5 后已推翻，见 §9.6）

| 量 | 值 | 来源 |
|---|---|---|
| 充值总额 | ¥20 | 用户 |
| 实际单价 | **¥1.48/时** | 控制台账单（创建页标的 ¥1.32 是另一台机器） |
| 计费起点 | 16:07:39 | 容器 `/etc/hostname` mtime |
| 验证日已花 | 2.35 h ≈ **¥3.48** | 16:07 → 18:28 |
| 剩余可跑 | ≈ 11.2 h | (20 − 3.48) ÷ 1.48 |
| 1 epoch 纯计算 | 11.7~13.1 h | 67390 步 × 625~700 ms（含热漂移） |
| 13 次评测 + 13 次存档 | ≈ 21 min | 38 s/次 + 60 s/次 |
| **1 epoch 合计** | **12.0~13.4 h ≈ ¥17.8~19.8** | 上两行 |

**差额恰好等于验证日花掉的那 ¥3.3。** 即使不花验证这 2.35 小时，¥20 也只够 13.5 h，
而一个 epoch 要 12.0~13.4 h——余量已被验证吃光。所以这次**不可能跑完完整 epoch**，
只能跑到约 10.5 小时（≈ 62000 步 ≈ 92%）。

### 9.3 用控制旗标优雅收尾（不赌 `timeout`）

```bash
setsid nohup bash scripts/linux/_graceful_stop.sh 10.5 240 20 > logs/linux/tripwire.out 2>&1 &
```

到点 → `--ctrl save`（训练在**下一个优化步边界**存一次检查点）→ 240 s 后 `--ctrl stop`
（步边界优雅退出）→ 等训练进程消失 → 汇总终态 → 20 分钟后 `/usr/bin/shutdown`。

为什么不用 `timeout` 硬杀：写一次检查点约 60 s，每 5000 步（约 51 min）才存一次，
SIGTERM 有约 **2%** 的概率落在写盘窗口里，留下半个 `checkpoint-*` 目录，`--resume`
可能读不了。控制旗标走 HF 正常路径（`on_step_end` → 本步 `_maybe_log_save_evaluate`
照常执行 → `should_training_stop` → `trainer.train()` 正常返回 → `save_model` +
`verify_export` → `TRAINING DONE` → **rc=0**），所以 `run_train.sh` 的两道校验都判成功。
`--max-hours 10.9` 仍作为硬背板保留。

**这条机制在正式训练跑到 step 5107 时做过一次实跑验证**（19:24:09，用一次性的 `save`
指令，它不停训练，所以可以安全地在运行中的训练上试）：

```bash
python run_task.py --ctrl save      # 写入后 1 秒内就被消费，旗标文件自动写回 run
```

日志与产物：

```
[ctrl] step 5107 读到指令 'save'
[ctrl] step 5107 立刻存一次检查点
→ checkpoints/retnet_500m_v15b/checkpoint-5107   生成
→ 训练继续（5314 步，未停）；峰值显存仍 23,230 MiB
```

**这一步把整晚计划里唯一的单点依赖（"训练进程会不会在步边界消费控制旗标"）从"假设"
变成了"实测"。** 剩下的未测分支只有 `stop`（它会真的终止训练，不能拿正在跑的作业试），
但它**有单元测试覆盖**：`tests/test_control.py` 用假的 `args/state/control` 把四个指令
的决策逻辑全测了（`stop` 置 `should_training_stop` 且粘性；`next-checkpoint` 在非保存步
不动、保存步才停；`save` 一次性；`run` 什么都不做），本轮重跑**全部通过**。
于是这条机制有两层证据：**单元层**（回调决策逻辑）+ **集成层**（实跑中 `save` 被消费）。
若它仍然失效，退路是 `--max-hours 10.9`（05:25）的硬背板，代价是少留 24 分钟、多花约 ¥0.6。

### 9.4 巡检用一条命令

```bash
bash scripts/linux/_status.sh    # 进程 / GPU / 进度 / loss / 评测 / 检查点 / 磁盘 / 旗标
```

### 9.5 峰值显存由谁记

```bash
bash scripts/linux/_peak_watch.sh logs/linux/_peak_v15b.txt 5
```

每 5 秒记一行 `memory.used,power.draw`，直到训练进程消失，末尾附最大值。

⚠️ 第一版直接进 while 判断，而训练进程总比采样器晚几秒起来，于是它**立刻判定"进程不在"
就退出**，峰值文件是空的。现在先等进程出现（最多 300 秒）。

### 9.6 追加余额后的计划变更：从「跑一部分」改成「跑完整个 epoch」

20:13:31 用户又充了 **¥5**（控制台账单），余额 **¥19.27**。这直接推翻了 §9.2 的结论，
所以计划改了，而且**用 `--resume` 改了**（不是从头跑）：

| 项 | 旧计划（20:13 之前） | 新计划 |
|---|---|---|
| 余额 | ¥14.27（20:59 前耗完） | **¥19.27**，够到约 **09:15**（÷¥1.48 = 13.0 h） |
| 目标 | 只跑到约 10.4 h，主动收尾（≈93% epoch） | **跑完 67,390 步整个 epoch**，预计 **05:56 完赛 + 评测/存档 ≈ 06:18** |
| 收尾 | `_graceful_stop.sh` 定时到点 `save`+`stop` | **`_watchdog.sh` 等训练自己结束**（跑完/崩溃/被停都行）→ 收尾 + 关机 |
| 兜底 | —— | `SKIP_FINALIZE=1 _graceful_stop.sh 11.9`：08:15 若还在跑就优雅停（防 GPU 挂死） |

**为什么不能只改计划不动作业**：正在跑的那次带着 `--max-hours 10.9`，会在 **05:25 硬杀**，
恰好把 epoch 截在 93%。启动参数改不了，所以只能重启 —— 而重启正好可以用 `--resume`，
于是顺手把目标第 (4) 项的「恢复验证」在**真实作业**上做掉了。

**优雅停止的全部四个指令现在都有实跑或单测证据**（`tests/test_control.py` 还全绿）：

```
[stop] 20:18:52 请求 save
[ctrl] step 10508 立刻存一次检查点          ← save 落地
[stop] 20:20:28 请求 stop
[ctrl] step 10661 收到 stop，本步结束后退出  ← stop 生效
[ctrl] 训练于 step 10661 按 'stop' 正常结束；最终模型仍会导出到 output_dir
[verify] ✅ 导出键名干净（365 个）
TRAINING DONE / 训练进程退出码: 0 / ✅ 已确认 TRAINING DONE
```

**`--resume` 的真实作业验证**：重启命令
`run_train.sh --config tasks/v15b_500m_24g.json --resume --max-hours 12.3 --no-shutdown`，
20:21:26 起跑，**步计数从 10508 续上**（首个刷新是 `10652/67390`，不是 `1/67390`），
优化器/LR 调度/RNG/数据采样位置都由 HF 一并恢复。这一条比 `_resume_test.sh` 更有说服力
——它证明**交付物本身**可以从一个中间检查点无缝续训。

**"恢复了"不能只看日志里有没有 `[resume]`，要验调度器真的续上**（否则会静默重新 warmup）。
用 `learning_rate` 对账：

| 时点 | step | `learning_rate` |
|---|---|---|
| 停止前最后一次 | ~10,660 | 0.00010447 |
| resume 后首条 | ~10,720 | **0.00010462** → 10460 / 10458 / 10456…（单调衰减） |
| 余弦调度理论值 | 10,720 | 1.0442e-4（`0.5(1+cos(π·(s−500)/(67390−500)))` 代入 1.1e-4→1.1e-5） |

五位有效数字对得上，且跨重启后**单调**下降 —— 没有在 restart 处跳回峰值，也没有重新
从 0 爬 warmup。**这一步才算把"可恢复"验完。**

> ⚠️ 一个必须记住的连带影响：定时哨兵与「跑完整个 epoch」是**互斥**的。
> 训练在 `max_steps` 到顶时会自己打 `TRAINING DONE` 正常退出，定时哨兵只会把它截断。
> 所以余额充足时用**看门狗**（等它结束），余额不足时用**定时哨兵**（主动收尾）。
> 顺带记一次实际踩到的坑：`SKIP_FINALIZE=1` 是为了防止兜底哨兵与看门狗**同时**跑
> `_finalize.sh` —— 两个进程同时抢 GPU 会让恢复验证必然失败。

**账单重建**（用户给的控制台流水，可与本文所有金额对账）：

| 时间 | 类型 | 金额 | 余额 | 备注 |
|---|---|---|---|---|
| 15:47:29 | 充值 | +¥10.00 | ¥10.00 | |
| 16:08:15 | 充值 | +¥10.00 | ¥20.00 | |
| 16:59:59 | 消费 | −¥1.29 | ¥18.71 | 首段不足整点，按 0.85 h 折算 |
| 17:59:59 | 消费 | −¥1.48 | ¥17.23 | |
| 18:59:59 | 消费 | −¥1.48 | ¥15.75 | |
| 19:59:59 | 消费 | −¥1.48 | ¥14.27 | |
| 20:13:31 | 充值 | +¥5.00 | **¥19.27** | ← 计划变更点（§9.6 开头） |

由 ¥1.48/时 反推：**余额归零点 ≈ 09:01**（20:13 起 ¥19.27 ÷ 1.48 = 13.02 h）。
一个 epoch 预计 05:50 跑完、06:40 关机 → **留约 2.3 h 余量、结余约 ¥4.5**。

**这就是"绝不让余额归零"这条硬约束的兑现方式**：先算零点，再定收尾时刻，
两者之间永远留出「收尾 + 恢复验证 + 关机」的时长（本例约 50 分钟）。
§9.2 那次之所以是"主动截断"，正是因为零点（05:38）落在完赛时刻（≈12:00）之前。

### 9.7 b4a1 的显存余量只剩 1.2 GiB，为什么还敢用

`nockpt_b4a1` **从来没有被探测过**（`speed_probe.sh` 第 21 行注释：「不含 batch>=4……
算过，不试」）。它真实的显存占用是**跑起来之后量的**：真实训练路径下 nvidia-smi
稳定显示 **23,230 MiB / 24,576 MiB = 22.68 GiB**，**余量 1,346 MiB**（这两个都是
nvidia-smi 直接读到的，相减即可）。真正危险的不是训练步，而是
**存档 / 评测 / 收尾的 `load_best_model_at_end`** —— 这三件事在"只跑 100 步"的普通探测里
一次都不会发生（`eval_steps` 远大于 100）。所以先做了一次**完整彩排**：

```bash
bash scripts/linux/run_train.sh --config tasks/_speed_b4a1.json \
  --max-steps 100 --save-steps 50 --output-dir checkpoints/_speed_b4a1 --no-shutdown
```

`--save-steps 50` 会同时把 `eval_steps` 设成 50，于是 100 步里包含 2 次评测、2 次检查点
存档、收尾的 load_best、最终 save_model + verify_export —— 正式训练的所有内存高峰都被
提前跑了一遍。结果：全程无 OOM、`TRAINING DONE`、rc=0、checkpoint-50/100 都在。
若正式跑在第一个存档点（step 5000）显存触顶，就退回 b2a2（672 ms，8 GiB 余量，只慢 7.7%）。

**正式跑的第一次实测验证（19:23:37，step 5000 首个存档点）**：

- `checkpoint-5000` 落盘，`eval_loss = 2.6358`，`eval_runtime = 47.0 s`
  （833 条 ÷ 4 = 209 batch，4.44 step/s）；
- **峰值显存仍为 23,230 MiB —— 与纯训练步完全一致，存档没有把峰值推高**。
  这直接否掉了「HF 存档会在 GPU 上克隆权重、2 GB 副本会撞上限」这个担心；
- 全程无 OOM、无 Traceback。

**启动开销很小**：t+2:00 时已到第 189 步（189 步纯计算 115 s，差值约 5 s 是模型/数据加载）
——因为 Triton 缓存已被探测与彩排预热。**冷缓存时 batch=4 的首次编译实测要 180 秒**，
所以「彩排一次」不只是验内存，也顺手把编译缓存热好了。

### 9.8 控制实验清单（这些都是"能失败"的检查，别删）

| 脚本 | 验什么 | 通过标准 |
|---|---|---|
| `scripts/_test_config_guard.py` | `check_task_config.py` 的跨字段检查 | 正向 v15b 通过 + 负向 2000/2500 报错，两向都要过 |
| `scripts/_test_rc_prop.py` | `run_task.py` 是否传播子进程的非零退出码 | 用不存在的 `model_dir` 造崩溃 → `run_task.py` 必须非零退出 |
| `scripts/linux/_test_run_train.sh` | `run_train.sh` 能否分辨四种结局 | 假解释器驱动：真跑完/崩溃/rc=0 但没跑完/校验不过 → rc 分别 0/1/1/2 |
| `scripts/linux/_test_parse.sh` | 日志解析与内核断言 | 4 个正向 fixture + OOM 反例 + 内核断言正反例 |
| `scripts/linux/_test_maxhours.sh` | `--max-hours` 小数换算与非法值拒绝 | 10.5→37800 s；`abc`/`-1`/`0`/空值被拒；旧的整数写法对 10.9 报错 |

### 9.9 如果没有人守着：结果去哪里读

正式训练的收尾由一个独立进程（哨兵 `_graceful_stop.sh`）负责，**不依赖任何人在线**：
到点发 `save` → 240 s 后发 `stop` → 等训练进程退出 → **自动跑 `_finalize.sh`
（含 `--resume` 恢复验证）** → 再等 20 分钟 → `/usr/bin/shutdown`。

| 文件 | 内容 |
|---|---|
| `logs/linux/watchdog.out` | 看门狗：等训练进程退出 → 收尾 → 40 分钟后关机（**余额充足时的主路径**） |
| `logs/linux/tripwire.out` | 定时哨兵时间线（余额不足时用）；`tripwire_fallback.out` 是兜底那次 |
| `logs/linux/finalize.out` | 训后收尾检查：终态日志 / 检查点 / 导出键 / 评测曲线 / 显存峰值 / **恢复验证** |
| `logs/linux/v15b_real.out` / `v15b_resume.out` | 训练主日志（第一段 / `--resume` 之后那段） |
| `logs/linux/_peak_v15b_seg1.txt` | 第一段的显存采样（每 5 s 一行，末行是峰值） |
| `logs/linux/_peak_v15b.txt` | `--resume` 之后那段的显存采样 |
| `checkpoints/retnet_500m_v15b/` | 交付物：`checkpoint-*`（含优化器状态）+ 最终模型导出 |

**判据（三条都满足才算这次训练"守住了"）**：

1. `finalize.out` 里 `TRAINING DONE` 在，且「存档恢复验证」4/4 通过；
2. `checkpoint-*` 至少一个，且其 `trainer_state.json` 的 `global_step` 是个大数（不是 2）；
3. `_peak_v15b.txt` 的峰值 < 卡的总量（实测 23,230 / 24,576 MiB，即没 OOM）。

一条命令看状态：`bash scripts/linux/_status.sh`。

---

### 9.10 最终结果：0.5B 跑满了整个 epoch（2026-10-12 05:51:46）

**结论先行：一个完整 epoch 落地了，而且全过程没有让余额归零、没有 OOM、没有崩溃。**

| 项 | 实测 |
|---|---|
| 训练步数 | **67,390 / 67,390**（= 1 epoch） |
| token | **552,330,489**（269,561 行 × 2048） |
| token/参数 | **1.095**（参数量 504,179,896） |
| 纯计算步时间 | **581 ms/优化步**（1.70~1.72 it/s，末 9.5 小时稳定） |
| 含评测/存档的有效步时间 | **601.6 ms**（56,882 步用 9.505 h） |
| 评测 + 存档开销 | 11 次评测（≈47 s/次）+ 11 次存档（≈60 s/次）= **19.6 min**（与实测差值吻合） |
| 显存峰值 | **23,118 / 24,576 MiB**，自 20:23 起 9 小时未增长 |
| 交付物 | `checkpoint-67390`（含优化器状态）+ `model.safetensors`（1.9 GB）+ `config.json` |
| 导出校验 | `[verify] ✅ 导出键名干净（365 个），可直接 from_pretrained` |
| 收尾 | `TRAINING DONE` · rc=0 · 已确认 |

**验证集 loss（每 5000 步，11 档，全程单调下降）**：

| step | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 50k | 55k | 60k | 65k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 2.6358 | 2.5655 | 2.5061 | 2.4515 | 2.3954 | 2.3445 | 2.2945 | 2.2468 | 2.2075 | 2.1701 | 2.1390 | 2.1187 | **2.1035** |

（5k→65k 降 **20.2%**，每 5000 步稳定降 2.0~2.3%，**没有出现平台期** —— 这说明 1.095
token/参数 远未吃满这段数据的收益，与 §9.2 的「10 token/参数要 4.5 天」是同一件事的两端。）
训练 loss：首条 2.6522 → 末条 **2.0009**。

**存档恢复验证（`_resume_test.sh`，4/4 通过）**：

```
源 checkpoint-67390 → --resume 跑到 67392
  ✅ train.py 报告从 checkpoint-67390 续训
  ✅ 新存档 checkpoint-67392 的 global_step = 67392（步计数确实续上，不是 2）
  ✅ 打印了 TRAINING DONE
  ✅ 退出码 0
```

> ⚠️ 这次补跑暴露了脚本自己的一个 bug：`_resume_test.sh` 里「删除前防呆」的判断写成
> `*/checkpoints/_resume_test`，而 `$DEST` 是**相对路径**（前面没有斜杠）→ case 不匹配 →
> **自己把自己拦下，恢复验证从来没跑过**（`finalize.out` 里那行
> `❌ 拒绝操作可疑路径`）。教训：**防呆写太窄，它自己就成了故障源**。
> 已改成同时接受相对/绝对两种写法，并把基准目录也钉死。

> ⚠️ 紧接着又暴露**第二个**同类缺陷，而且更危险：`_finalize.sh` 用
> `LOG="$(ls -1t logs/linux/train_*.log | head -1)"` 选主日志，而恢复验证自己也会
> 通过 `run_train.sh` 生成一份 `train_*.log` —— 于是那次重跑里第 **4/5/9 段全部读成 0**
> （读的是只跑 2 步的测试日志）。**"静默读了错文件"比"读不到"更危险**：它不会报错，
> 只会给你一份看起来"这部分没数据"的报告。
> 修法两层：① `_finalize.sh` 改成**按步计数最大**选日志（不按 mtime）；
> ② `_resume_test.sh` 跑完把新增的日志改名成 `_resume_test_run.log`，从源头消除污染。
> 重跑后报告全绿（见下）。**这两个 bug 都是"只在补跑时才出现"的**——第一次跑
> （05:51:49）恰好选中了正确的日志，所以只修第一个是不够的。

**预算兑现**：余额零点 09:01，训练 05:51 完赛、看门狗 06:31:49 执行 `/usr/bin/shutdown`
（次日 06:35 实测 ssh 已 `Connection refused`，确认关机成功）→ **留 2.5 小时余量，
结余约 ¥4.5**。这正是 §9.6「先算零点，再定收尾时刻，两者之间留出收尾+恢复验证+关机」
那条规则的兑现。数据在 AutoDL 保留 15 天，随时可再开机 `--resume` 续训。

**一个被源码确认的"警告"**：收尾阶段日志里有
`There were missing keys in the checkpoint model loaded: ['lm_head.weight']`。
**这是设计如此，不是缺陷**：`retnet_hf/modeling_retnet.py:398-400` 在 `config.tie_weights`
时把 `lm_head.weight` 绑到 `embedding.weight` 并登记进 `_tied_weights_keys` ——
HF 本就会把绑定键报成 missing，加载后由 `tie_weights()` 恢复。

**已知未做/待办**（诚实列出）：

1. 未做**定性生成抽样**（需要一个额外的 generation 脚本 + GPU 时间）。本轮的"模型学到了"
   证据是 val/train loss 曲线（双降且同步）与导出键校验，**不是**生成质量。
2. 笔记本上那个 1,020 题的 `v14_50k` 基准（`bench_v3`）因笔记本关机中断，未完成；
   本轮的评测口径里也没有它。
3. `_peak_watch.sh` 在 18:31 那次是旧版本，末行那句括号文字是硬编码的
   （峰值本身是实测）；源文件已改为运行时读 nvidia-smi 总量。

### 9.11 ⚠️ "步数"最容易被误解的地方：每优化步到底多少 token

外部资料（以及任何训练方案）说「0.5B 的 Chinchilla 最优约 100 亿 token，约 9,500 步」，
而本轮的正式训练是 **67,390 步**。两者都不是错的 —— **差的是每步 token 数，不是工作量**：

| 记账口径 | 每优化步 token | 552 M token（= 1 epoch） | 10.08 B token（Chinchilla 20×） |
|---|---|---|---|
| global batch **512** | 1,048,576 | **527 步** | **9,616 步** |
| 本轮 global batch **4** | 8,192 | **67,390 步** | **1,230,908 步** |
| 比值 | **128×** | 128× | 128× |

三件事必须分清：

1. **"步"必须配 batch 才有意义。** `global batch = per_device_batch × grad_accum`
   （本轮 4 = 4 × 1）。同一份 token 预算，batch 越小步数越多。
2. **梯度累积只改记账，不省时间。** 把本轮 `grad_accum` 从 1 改成 128，global batch
   就成了 512，同一个 epoch 变成 **527 步** —— 但每步耗时从 581 ms 变成
   128 × 581 ms = **74.4 s**，总时间一模一样（527 × 74.4 s ≈ 67,390 × 0.581 s ≈ 10.9 h）。
   **"步数少"不等于"更快"，真正的货币是 token。**
3. **本轮 67,390 步 ≠ 那个 10 B token 目标。** 67,390 × 8,192 = **552 M token**
   = 1.095 token/参数，只是把手上的语料喂了**一遍**，也就是 Chinchilla 20× 的 **5.5%**。
   真要跑到 10.08 B，在本轮这张卡上就是 **1,230,908 步 / 205 h / ¥303**。

**为什么本轮不把 batch 做大**：24 GB 卡、seq 2048、504 M 参数，`batch 8` 实测直接 OOM
（§4.7(e) 的探测试到上限），所以 per-device batch 只能 4；而梯度累积虽然能把
effective batch 做大，**却会改变优化动力学** —— 等效 batch 从 4 到 512，同样 lr 下
更新次数少 128 倍，必须重调 lr 与 warmup。本轮的目标是与 272M 那批**同口径**对照
（它们也是 global batch 4 = 8,192 token/步），所以选 `grad_accum=1`：
**为了可比性，不是为了快。**

> 换句话说：**"要跑多少步"这个问题没有唯一答案，"要喂多少 token"才有。**
> 本项目当前的约束是 token —— `docs/23` §8.6 的结论就是「先把 token/参数 从 2.5 提上去」，
> 而不是把步数做大或做小。


### 9.12 ⚠️ 2026-10-12 21:31–21:53：`--resume` 把 batch 16 **静默降成 4**（外加 LR 从谷底跳回峰值）

用户看显存曲线时问了一句「你此前也没吃满，但至少用了 70GB 左右」——
顺着这句查下去，发现不是"卡没吃满"，是**我的代码 + HF 默认行为叠加出的两个静默降级**。

#### 证据 1：显存对不上

| 时刻（PRO 6000，95.0 GiB 可用） | 显存 | 在跑什么 |
|---|---|---|
| 21:17:16 | 73,833 MiB | 0.5B **合成同架构**探针 batch 16（`checkpoints/_synth_500m/config.json` 建于 21:08；`logs/linux/` 21:27 才建，所以那时还没跑正式训练） |
| 21:31–21:53 | **30,177 MiB** | 正式训练，命令行写着 `--batch_size 16`，实际用 **4** |
| 22:00 起 | **75,903 MiB** | 修好之后，真 batch 16 |

#### 证据 2：日志里唯一的那行 WARNING

`logs/linux/train_20261012_212943.log:42`：

```
Warning: The following arguments do not match the ones in the `trainer_state.json` within the checkpoint directory:
	per_device_train_batch_size: 16 (from args) != 4 (from trainer_state.json)
```

#### 证据 3：源码 —— HF 用检查点里的 batch 覆盖命令行

`transformers/trainer.py:2298-2301`（4.57.6）：

```python
# In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
if state.train_batch_size is not None:
    self._train_batch_size = state.train_batch_size
```

`self._train_batch_size` 就是 `get_train_dataloader(batch_size=...)` 用的值（`trainer.py:1143`）。
第三方交叉验证：epoch 字段的推进速度也只对得上 4 ——
`steps_in_epoch = 269561 / 4 = 67390`，实测 step 70,180 时 `epoch = 1.04`（= 1 + 2,790/67,390 ✓）；
若真是 batch 16（`steps_in_epoch = 16848`）该显示 **1.17**。

#### 责任在我，而且它自锁

`checkpoint-67390/trainer_state.json` 是我为了"让 HF 找得到 config"从 3090 那段复制过去的
（里面 `train_batch_size = 4`）。更糟的是新存的检查点会把 4 记下去：
`checkpoint-70000/trainer_state.json` 实测 `train_batch_size: 4` —— **不修的话每一轮续训都回到 4**。

**修**：`retnet_hf/train.py` 新增 `_sync_resume_state()`。续训前把检查点里**会被 HF 采纳**的四个字段
（`train_batch_size` / `logging_steps` / `eval_steps` / `save_steps`）对齐到本次参数，逐条打印
`旧 -> 新`，并留一份 `trainer_state.json.orig`。以后再出现不一致，日志里会有一行明写的
`[resume] 修正 trainer_state.json: train_batch_size 4 -> 16`。

#### 第二个洞：没有 `optimizer.pt` 时，**LR 调度器也不恢复**

`transformers/trainer.py:3523`：

```python
if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
```

`checkpoint_file_exists` 指的是 `optimizer.pt`。本轮为了省时间没搬 1.11 GB 的 `optimizer.pt`
（内蒙B→本机链路实测 414 KB/s，并行分块 8 路全 0 字节），于是 `scheduler.pt` 明明在也不加载，
**LR 从 step 0 重新升温**：

| 口径 | learning_rate |
|---|---|
| 3090 那段最后几步（实测日志末尾） | **1.1000e-05**（余弦谷底） |
| 缺 `optimizer.pt` 续训后第一段 | **1.0987e-04** ≈ 峰值（10×） |
| 反推 scheduler 的 `last_epoch` | ≈ **2,790** = 70,180 − 67,390，正好是本段自起点的步数 |

所以续训后 loss 从 **2.0009** 反弹到 2.13~2.39 **不是训练坏了**，是 LR 被抬回峰值重新升温。
`train.py` 里那句"优化器 / LR 调度器状态由 HF 一并恢复"是**过度承诺**，已改。

#### 处置：不在这条脏路径上继续修补，改起一段干净的 epoch 2

`tasks/v16_ep2_500m_pro6000.json`（22:00 起跑）：

| 项 | 值 | 依据 |
|---|---|---|
| `model_dir` | `.../checkpoint-67390` | 只取权重（3090 那个 epoch 的末点） |
| `output_dir` | `.../retnet_500m_v15b_ep2` | **新目录**：不碰 v15b 的模型/检查点，也不触发 `--resume` 的三个陷阱 |
| `batch_size` / `grad_accum` | **16 / 1** | 合成探针 57.1k（+max-autotune-no-cudagraphs 60,951）tok/s；batch 32 需约 140 GiB 会 OOM |
| `max_steps` | **16,848** = ⌈269561/16⌉ | **正好一个 epoch** |
| `save_steps` / `eval_steps` | 2,106（= 16848/8） | HF 要求 `save_steps` 是 `eval_steps` 的整数倍 |
| LR | 1.1e-4 → 1.1e-5 余弦，覆盖这一个 epoch | fresh scheduler，不再被续训逻辑拉长 |
| `compile_mode` | `max-autotune-no-cudagraphs` | +6.7%（§9.9）；`max-autotune` 会在 `modeling_retnet.py:507` 崩 |
| 起点显存 | **75,903 MiB**（实测，`_peak_ep2.txt` 峰值 77,167） | 与 batch 4 的 30,177 MiB 对照 |

已知不完美：8-bit Adam 动量是新起的（`optimizer.pt` 没搬过来，见上）；
这一条也写进了配置注释 —— 之后每个检查点都自带完整优化器状态，后续续训才是真续训。

#### 修好之后的实测（22:00–22:02，开局 200 步，`_peak_ep2.txt` 峰值同值）

| 项 | batch 4（被覆盖的旧跑） | **batch 16（修好后）** | 倍数 |
|---|---|---|---|
| 优化步/s | 6.25 | **1.77** | 0.28×（每步 token 多 4 倍） |
| **token/s** | 51,200 | **58,000** | **1.13×** |
| 显存 | 30,177 MiB | **79,701 MiB** | 2.64× |
| 本项目口径 MFU | 41.1% | **46.5%** | 1.13× |
| 一个 epoch（552 M token） | 3.0 h / ¥20.9 | **2.64 h / ¥18.4** | 0.88× |

（MFU = token/s × 3.136 GFLOP/token ÷ 391 TFLOP/s；391 是本卡实测的 8192³ bf16 GEMM 峰值，
不是标称值。GPU 状态：471 W、SM 2407/2430 MHz、100% 利用率、69 ℃——**没有降频**，
所以 58,000 tok/s 就是这张卡在这个模型尺寸上的真实上限附近，不是被什么"黑洞"吃掉的。）

**与 3090 的同口径对照**（同一份数据、同一 seq、同一 chunk、同一模型）：

| | RTX 3090 24G | RTX PRO 6000 96G | 比值 |
|---|---|---|---|
| 单价 | ¥1.48/时 | ¥6.98/时 | **4.72×** |
| token/s（实测） | 14,100 | 58,000 | **4.11×** |
| MFU | 62% | 46.5% | 0.75× |
| **每 10 亿 token 成本** | ¥29.2 | **¥33.4** | **1.15×** |
| 一个 epoch 墙钟 | 11.21 h | 2.64 h | 0.24× |

**结论（这次是实测，不是推算）**：PRO 6000 快 4.1 倍，但**每 token 贵 15%** ——
因为「¥ / TFLOP/s·h」在这几张卡上几乎持平（3090 0.0208、A800 0.0179、PRO 6000 0.0179），
而 0.5B 模型在大卡上的 MFU 反而掉 25%（GEMM 太小、喂不满 188 个 SM）。
**大卡买的是墙钟时间，不是单位算力成本。**
要让大卡在单位成本上赢，只有两条路：把模型做大（1.45B 实测 51.7% MFU）或上 FP8。

**顺带修掉一处自己写死的假信息**：`run_train.sh:199` 打印"保持开机（记得手动关机，¥1.32/时）"，
¥1.32 是 3090 的价格常量，在 PRO 6000（¥6.98/时）上是错的，已改为读 `PRICE_PER_HOUR` 环境变量。



