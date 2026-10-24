# 0.5B 技术报告 · 已核实事实底稿

> 由 Lead 直接从**机器可读源头**读出，不是从散文文档转抄。每条都给出出处路径。
> 写作时任何与本文冲突的数字，一律以本文为准；本文没写的，必须自己回源头核，不许从旧文档照抄。
> 最后更新：2026-10-13

---

## A. 模型

| 项 | 值 | 出处 |
| --- | --- | --- |
| 参数量 | **504,179,896**（504.18 M） | `checkpoints/retnet_500m_v15b_ep3/model.safetensors` 头部张量形状求和 |
| 权重文件大小 | 2,016,755,680 B（三个存档逐字节相同） | 同上（`ep2` / `v15b/checkpoint-67390` 同尺寸） |
| 存储精度 | `dtype: float32`（config.json）；训练时 bf16 | `checkpoints/retnet_500m_v15b_ep3/config.json` |
| d_model | 1408 | 同上 |
| num_layers | 20 | 同上 |
| num_heads | 22 | 同上 |
| ffn_dim / value_dim | 2816 / 2816 | 同上 |
| vocab_size | 20000 | 同上 |
| chunk_size | 128 | 同上 |
| dropout | 0.1 | 同上 |
| norm_eps | 1e-05 | 同上 |
| mode | `chunkwise` | 同上 |
| use_gla | true（逐头可学习衰减门） | 同上 |
| use_growth_ln | true，`growth_old_dim: 768` | 同上 |
| use_swiglu | **false** | 同上 |
| tie_weights | true | 同上 |
| embed_scale | 27.712812921102035（= √768） | 同上 |
| prompt_loss_weight | 0.1 | 同上 |
| `checkpoint_layers` | −1 | 同上 |
| transformers_version | 4.57.6 | 同上 |
| model_type | `retnet_hf`（`RetNetForCausalLM`） | 同上 |

**sha256（传输校验时记录，未在本机重算）**

- ep3：`dd7cdc90ae76c81c4c7134bb5e614455f924a8ffa51075c6646728cd290a3c53`
- ep2：`dbaf8257119f22ad1b1a7f5590c0313f0867eee2922af430f21ce063bec3fba7`

> ⚠️ 报告若引用 sha256，需在交付前用 `Get-FileHash` 重算确认；这两个值来自云端传输时的校验输出，不是本次实测。

---

## B. 数据：`data_cache/chat_pack_v13`

出处：`data_cache/chat_pack_v13/meta.json`（唯一权威）

| 项 | 值 |
| --- | --- |
| seq_len | 2048 |
| train_rows | 269,561 |
| train_tokens | **552,330,489** |
| reply_tokens | 358,896,326（占 64.98%） |
| prompt_tokens | 144,461,792（占 26.15%） |
| plain_tokens | 32,846,336（占 5.95%） |
| strip_think | **false**（`<think>` 全量保留） |
| r1_max_chars | 4000 |

`meta.json` 的 `sources` 只登记了**新编码**的三个源，合计 266.9 M token，**不等于** 552.33 M：

| 源 | 类型 | tokens | docs | 编码耗时 |
| --- | --- | --- | --- | --- |
| r1_110k_sft | r1 | 147,945,725 | 109,999 | 155.0 s |
| i_wonder_why | wiki | 118,947,599 | 300,000 | 96.7 s |
| better_ruozhiba | multi | 119,657 | 1,475 | 0.1 s |

note 字段写明另有「v9 base re-sliced to 2048」的一部分。

> **已解决（核查员，2026-10-13）**：v9 base = **131,377 行 × 2049 = 269,191,473 token（48.73%）**，
> 三源新编码 267,012,981（48.34%）+ pad 16,126,035（2.92%）—— 三数逐位闭合到 552,330,489。
> ⇒ `reply_tokens` 占 64.98% 这个比例里**有一半来自 v9 base，不是 R1**。
> 报告写数据构成时用这组数，不要再用「未知」。

---

## C. 训练三段（全部逐位读出，`trainer_state.json`）

| 阶段 | 硬件 | batch×accum | max_steps | token/步 | 合计 token | total_flos |
| --- | --- | --- | --- | --- | --- | --- |
| ep1 | RTX 3090 24 GB | 4 × 1 | 67,390 | 8,192 | 552,058,880 | 1.576746e18 |
| ep2 | RTX PRO 6000 | 16 × 1 | 16,848 | 32,768 | 552,075,264 | 1.576752e18 |
| ep3 | RTX PRO 6000 | 16 × 1 | 16,848 | 32,768 | 552,075,264 | 1.576752e18 |

累计 **1,656,209,408 token ≈ 1.656 B**；token/参数 = **3.28496**。

> **勘误（2026-10-13）**：本文初版把 ep2/ep3 的每段 token 写成 552,140,800，是错的
> （`16848 × 32768 = 552,075,264`）。累计与 token/参数 已随之改正。
> 核查员当时"逐位复现 ✅"是在这个错数之上做加法，属内部自洽而非独立推导 ——
> **每个 token 数必须自己从 `max_steps × batch_size × seq_len` 乘出来，不许从别处的合计反推。**

### C.0 ⚠️ 三个必须区分的 token 口径（核查员提出，已复核）

| 口径 | 值 | 怎么来的 |
| --- | --- | --- |
| ① 语料总槽位 | **552,330,489** | `meta.json:train_tokens` = 269,561 行 × **2049**（`build_chat_mix_v13.py:80` 的 `row_len = seq_len + 1`），**含最后一列的 16,126,035 个 pad（2.92%）** |
| ② 语料有效损失位置 | **552,060,928** | 269,561 行 × 2048；`data.py:31-33` 取 `row[:-1]` / `[1:]` |
| ③ 实际训练消耗 | ep1 **552,058,880** / ep2、ep3 **552,075,264** | `max_steps × batch_size × seq_len`（ep1 = 269,560 行，比语料少 1 行；ep2/ep3 = 269,568 行，末批补 7 行） |

报告写「552.33 M token × 3 epoch」时**必须声明是哪个口径**。三者最大相差 0.049%。

> `total_flos` 与 `6 × N × T` 相差约 5.6%（实测 1.5767e18 vs 6×504,179,896×552,058,880 = 1.670e18），
> 原因未定位。**不要拿 `total_flos` 当算力证据**，它低于朴素估算。

### C.1 ep1 —— `checkpoints/retnet_500m_v15b/checkpoint-67390/trainer_state.json`

- best_metric **2.103548765182495** @ best_global_step **65000**（注：67,390 未评测）
- train loss：step 20 → **2.4091**；step 67380 → **2.0009**
- eval_loss 全部 13 条：

| step | 5000 | 10000 | 15000 | 20000 | 25000 | 30000 | 35000 | 40000 | 45000 | 50000 | 55000 | 60000 | 65000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval_loss | 2.6358 | 2.5655 | 2.5061 | 2.4515 | 2.3954 | 2.3445 | 2.2945 | 2.2468 | 2.2075 | 2.1701 | 2.1390 | 2.1187 | **2.1035** |

### C.2 ep2 —— `checkpoints/retnet_500m_v15b_ep2/trainer_state.json`

- best_metric **1.9864579439163208** @ **16848**
- train loss：2.0545 → 1.9779
- eval_loss 全部 8 条：

| step | 2106 | 4212 | 6318 | 8424 | 10530 | 12636 | 14742 | 16848 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval_loss | 2.1800 | 2.1603 | 2.1275 | 2.0906 | 2.0535 | 2.0206 | 1.9985 | **1.9865** |

### C.3 ep3 —— `checkpoints/retnet_500m_v15b_ep3/trainer_state.json`（当前线上版本）

- best_metric **1.9459127187728882** @ **16848**
- train loss：1.9624 → 1.9474
- eval_loss 全部 8 条：

| step | 2106 | 4212 | 6318 | 8424 | 10530 | 12636 | 14742 | 16848 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval_loss | 2.1174 | 2.1139 | 2.0848 | 2.0502 | 2.0134 | 1.9800 | 1.9578 | **1.9459** |

### C.4 热重启代价（可由上表直接复算，报告应给出）

链是 ep1 → ep2 → ep3，每一段都重启一条**完整 cosine**（峰值 1.1e-4，warmup 500），而不是接着上一段的 LR 尾巴继续。

| 衔接 | 上一段最优 | 下一段首次评测 | 差 |
| --- | --- | --- | --- |
| ep1 → ep2 | 2.1035 @65000 | 2.1800 @2106 | **+0.0765**（先变差） |
| ep2 → ep3 | 1.9865 @16848 | 2.1174 @2106 | **+0.1309**（先变差） |

两段最终都收敛到低于各自起点：ep2 终 1.9865（比 2.1035 低 0.1170），ep3 终 1.9459（比 1.9865 低 0.0406）。
**这是「热重启值不值」在本项目唯一的直接证据，报告必须写，不许只写最好的那个数。**

---

## D. 超参（三段实际生效值）

出处：`tasks/v15b_500m_24g.json`（ep1）、`tasks/v16_ep2_500m_pro6000.json`（ep2）、`tasks/v17_ep3_500m_pro6000.json`（ep3）

| 键 | ep1 | ep2 | ep3 |
| --- | --- | --- | --- |
| precision | bf16 | bf16 | bf16 |
| mode / chunk_size | chunkwise / 128 | 同 | 同 |
| batch_size × grad_accum | 4 × 1 | 16 × 1 | 16 × 1 |
| dataloader_workers | （未写，默认 0） | 0 | 0 |
| optim | adamw_bnb_8bit | adamw_bnb_8bit | adamw_bnb_8bit |
| grad_checkpoint | false | false | false |
| compile | true | true | true |
| compile_mode | （未写） | max-autotune-no-cudagraphs | max-autotune-no-cudagraphs |
| fast_chunkwise | true | true | true |
| max_steps | 67390 | 16848 | 16848 |
| lr / min_lr | 1.1e-4 / 1.1e-5 | 同 | 同 |
| warmup | 500 | 500 | 500 |
| weight_decay | 0.1 | 0.1 | 0.1 |
| max_grad_norm | 1.0 | 1.0 | 1.0 |
| save_steps = eval_steps | 5000 | 2106 | 2106 |
| save_total_limit | 4 | 4 | 4 |
| eval_batch_mult | 1 | 1 | 1 |
| seed | 1234 | 1234 | 1234 |
| resume | false | false | false |
| auto_restart_max | 0 | 0 | 0 |

起点：ep1 从 `checkpoints/retnet_500m`（生长产物）起；ep2 从 `checkpoints/retnet_500m_v15b/checkpoint-67390` 起；ep3 从 `checkpoints/retnet_500m_v15b_ep2` 起。

> ⚠️ ep1 的 `save_steps=5000` 与 `eval_steps=5000` 必须相等，否则 HF 在构造 `TrainingArguments` 时直接抛
> `ValueError`（`train.py:196` 的 `load_best_model_at_end=True` 要求 save 是 eval 的整数倍）。
> 这是踩过的坑，`scripts/check_task_config.py` 现在有跨字段检查。

> ⚠️ 另有一条 8 GB 线配置 `tasks/v15_500m.json`（batch 1 × accum 4、`grad_checkpoint: true`、
> `max_steps: 67390`、`save_steps: 2500`）。它与 ep1–ep3 是**同模型同数据同超参、不同硬件**，
> loss 可比但配置不可混用。报告若提及必须说清是哪条线。

---

## E. 待写手自行补齐的部分（本文不含，别猜）

以下内容必须在**源头**取证，本文不提供二手值：

| 报告章节 | 去哪取证 |
| --- | --- |
| 训练/推理步时、MFU、GEMM 归因、显存账 | `docs/22_0.5B_提速研究.md`、`docs/25_一步计算构成与压缩空间.md`、`docs/23_0.5B扩容方案.md`、`logs/autodl_2026101*/`（`_peak_*.txt`、`v15c_pro6000.out`、`v16_ep2.out`、`v17_ep3.out`、`finalize.out`、`watchdog*.out`） |
| 云上真实吞吐/成本 | `logs/autodl_20261013/linux/v16_ep2.out`（含 `it/s`、`train_runtime`）、`logs/autodl_20261012/logs/linux/probe_*.out` |
| 基准设计与历史成绩 | `docs/14_基准测试v3.md`、`bench/`、`logs/bench_v3/` |
| 续训两处静默陷阱（batch 覆盖、scheduler 不恢复） | `retnet_hf/train.py` 的 `_sync_resume_state` 与调用点；`logs/autodl_20261013/linux/_resume_test*.out`；`logs/local_resume_test_ep3.out`、`logs/resume_A.out`、`logs/resume_B.out` |
| 架构数学（保留机制、闭式 chunkwise、GLA 门控、GrowthLayerNorm） | `retnet_hf/modeling_retnet.py`、`scripts/verify_scale_probe.py`、`tests/test_model.py`、`docs/08`、`docs/16`、`docs/18` |
| 服务与推理 | `serve/retnet_play_server.py`、`serve/DEPLOY.md`、`logs/bench_v3/decode_speed.json`、`scripts/bench_decode_speed.py` |

### E.1 正在进行：0.5B 的基准评测

`logs/bench_v3/` 下正在跑 ep3 全量 1,020 项（1 种子 42）。产出：

- `logs/bench_v3/report.500m_ep3.json`
- `logs/bench_v3/summary.500m_ep3.md`
- 运行日志 `logs/bench_v3/run.500m_ep3.out`

冒烟（26 项，`--stride 40`，已跑完）参考值：加权总分 **0.0637**，think 泄漏 57.7%，退化重复 0.0%。
冒烟输出在 `logs/bench_v3_smoke/summary.smoke_500m_ep3.md`。

> **写手注意**：评测章节一律以全量结果为准，冒烟值只能用来判断"跑得通"。
> 同时**不允许**拿它跟 272M 的历史成绩直接比而不声明口径差异——那是前任模型，本次报告的主体范围里不写。

---

## F. 全局写作纪律

1. 任何数字必须能指到「文件路径 + 字段/行」。指不到的，标 `未核实` 或删掉。
2. 阴性结果（做了但没用）与阳性结果同等篇幅。
3. 三处静默缺陷（不报错的那种）必须写 —— 这是本项目最有价值的部分。
4. 不写"我们认为值得注意""这令人兴奋"这类句子。
5. 未验证的推断必须显式标注「假说，未验证」。本项目已经有过一次把推算当实测的教训（`docs/22 §0`）。

---

## G. 云上实测吞吐、显存、成本（Lead 从日志直接读出）

### G.1 三个 epoch 的整段实测

出处：`train_runtime` 与 `train_steps_per_second` 行，分别在
`logs/autodl_20261012/logs/linux/v15b_real.out`、`v15b_resume.out`（ep1 两段）、
`logs/autodl_20261013/linux/v16_ep2.out`、`v17_ep3.out`。

| 阶段 | 卡 | wall clock（`train_runtime`） | 步/s（**实算**） | token/s（**实算**） |
| --- | --- | --- | --- | --- |
| ep1 seg1 | 3090 24G | 6,540.3172 s（实际 10,661 步） | 1.6300 | 13,353 |
| ep1 seg2 | 3090 24G | 34,197.5076 s（实际 56,882 步） | 1.6633 | 13,626 |
| **ep1 合计** | 3090 24G | **40,737.82 s = 11.3161 h** | **1.6542** | **13,551.5** |
| **ep2** | PRO 6000 96G | **10,043.18 s = 2.7898 h** | **1.6776** | **54,970.2** |
| **ep3** | PRO 6000 96G | **9,666.71 s = 2.6852 h** | **1.7429** | **57,111.0** |
| ep2+ep3 | PRO 6000 96G | 19,709.88 s = 5.4750 h | — | 56,020.1 |

> **勘误（核查员，2026-10-13）**：本文初版用 `v15b_real.out` 的 `epoch: 0.16` 反推 seg1 停在 10,782 步，
> **被证伪**。真实停顿点是 **10,661 步**，三条独立证据：`v15b_real.out` 尾部 `[ctrl] step 10661 收到 stop`；
> tqdm 最大 10,661；末存档 `checkpoint-10508`，而 `v15b_resume.out` 的 tqdm 从 **10,509** 起
> ⇒ 153 步被丢弃重跑（10,661 + 56,882 = 67,543 = 67,390 + 153，逐位闭合）。
> `epoch` 字段在控制台只有 2 位小数（`log_history` 原始值 step10660 = 0.15818 → 印 0.16），
> 只能把步数限在 [10,445, 11,119]，不能当精确依据。
> **合计行（40,737.82 s / 1.6542 / 13,551.5）不受影响，仍然正确。**

ep3 口径 MFU = 57,111 × 3.136 GFLOP/token ÷ 391 TFLOP/s = **45.8%**（391 TFLOP/s 是实测 bf16 GEMM 峰值，非标称）。

**步/s 与 token/s 是实算的**：`token/s = 步/s × 每步 token`（ep1 每步 8,192；ep2/ep3 每步 32,768）。
**不要直接引用日志里的 `train_steps_per_second`** —— 理由见 G.2。

### G.2 ⚠️ 第三处静默陷阱：`train_steps_per_second` 在分段/续训运行上是错的

`transformers/trainer.py:2819-2825`（本机 4.57.6）训练结束时：

```python
metrics = speed_metrics(
    "train", start_time,
    num_samples=num_train_samples,
    num_steps=self.state.max_steps,   # ← 用的是 max_steps，不是真实跑过的步数
    num_tokens=num_train_tokens,
)
```

而 `transformers/trainer_utils.py:389` 定义的 `speed_metrics`，真正做 `num_steps / runtime` 的是 **`:411`**。
**结论：`train_steps_per_second = max_steps / train_runtime`。**

数值反证 5/5 命中：

| 日志 | `max_steps` | `train_runtime` | 相除 | 日志报的 |
| --- | --- | --- | --- | --- |
| `v15b_real.out` | 67,390 | 6,540.3172 | 10.3037 | 10.304 |
| `v15b_resume.out` | 67,390 | 34,197.5076 | 1.97067 | 1.971 |
| `v16_ep2.out` | 16,848 | 10,043.1782 | 1.67756 | 1.678 |
| `v17_ep3.out` | 16,848 | 9,666.7055 | 1.74289 | 1.743 |
| `v15c_pro6000.out` | 100,000 | 1,382.3627 | 72.34 | 72.34 |

因此对任何「提前停止」或「从检查点续训」的运行，这个字段都会严重失真：

| 证据 | 日志报的 | 真实情况 |
| --- | --- | --- |
| `v15b_real.out` | `train_steps_per_second: 10.304`，`epoch: 0.16` | max_steps=67390，实际只跑了 10,661 步 → 真实 1.6300 步/s。**报了 6.32 倍** |
| `v15b_resume.out` | `train_steps_per_second: 1.971`，`epoch: 1.0` | 从 10,661 步续到 67,390 步 → 真实 1.6633 步/s。**报了 1.185 倍** |
| `v15c_pro6000.out` | `train_steps_per_second: 72.34` | 该段只跑了 1,382 s，max_steps 是 100,000 → 最极端的一例 |
| `v16_ep2.out` / `v17_ep3.out` | 1.678 / 1.743 | 一步没少，**这两个是对的** |

这与 §3 另外两处缺陷同一家族：**不报错、数值看起来合理、只有在"步数对不上"时才是错的**。
报告里若引用吞吐，必须写清是自己按真实步数/真实 token 算的。

### G.3 峰值显存与功耗（`nvidia-smi` 每 5 秒采样，见 `scripts/linux/_peak_watch.sh`）

采样文件格式是 `memory.used, power.draw`。

| 运行 | 采样点数 | 峰值显存 | 峰值功耗 | 卡总量 |
| --- | --- | --- | --- | --- |
| ep1（3090） | 6,734 | 23,118 MiB | 348.5 W | 24,576 MiB |
| ep1 另一段 | 1,289 | 23,230 MiB | 347.5 W | 24,576 MiB |
| ep2（PRO 6000） | 1,640 | 80,433 MiB | 505.2 W | 97,887 MiB |
| ep3（PRO 6000） | 1,888 | 77,685 MiB | 500.3 W | 97,887 MiB |
| v15c 早期窗口 | 234 | 30,181 MiB | 461.3 W | 97,887 MiB |

- `_peak_ep3.txt` 带脚本自算的收尾行：`峰值显存 = 77685 MiB / 总量 97887 MiB（余量 20202 MiB）`。
  **更正（核查员）**：本文初版说"只有 ep3 有收尾行"，是错的 ——
  `_peak_v15b.txt`（`23118 MiB…余量 1458`）与 `_peak_v15b_seg1.txt`（`23230 MiB…余量 1346`）**也有**。
  表中 ep2/ep3 的峰值是 Lead 按同样算法（取第 1 列最大）复算的，且与脚本自算值逐一相符：23,230 / 23,118 / 80,433 / 77,685 / 30,181 与 347.53 / 348.51 / 505.25 / 500.34 / 461.26 W 全部命中。
- `_peak_ep2.txt` 的峰值（80,433 MiB）**高于** ep3（77,685 MiB），两段配置完全相同 ——
  说明峰值依赖采样窗口覆盖到哪一次存档/评测，不是配置差异。报告引用时要说明这一点。
- ⚠️ 3090 的余量只有 **1,346 MiB**（23,230 / 24,576）。这不是"够用"，是"擦着过"。
- ⚠️ `logs/autodl_20261012/logs/linux/_peak_b4a1.txt` 是**空文件**（0 字节），不能当证据用。

### G.4 单价与成本

| 卡 | 单价 | 出处 |
| --- | --- | --- |
| AutoDL RTX 3090（14 vCPU / 60 GB） | **¥1.48/时**（原价 ¥1.56） | `docs/24_AutoDL上机手册.md:21,336`；`tasks/v15b_500m_24g.json` 注释 |
| AutoDL RTX PRO 6000 96G | **¥6.98/时** | `docs/24_AutoDL上机手册.md:756` |

按 `train_runtime` 折算（不含实例开机但没训练的时间）：

| 阶段 | 小时 | 成本 |
| --- | --- | --- |
| ep1（3090） | 11.3161 | **¥16.75** |
| ep2（PRO 6000） | 2.7898 | **¥19.47** |
| ep3（PRO 6000） | 2.6852 | **¥18.74** |
| 三段合计 | 16.7911 | **¥54.96** |

> ⚠️ 这是**按训练时长折算的下界**。实例开机到关机之间的空转（上传、env_check、测速、彩排、收尾评测、
> 以及两次抢卡等待）不在内。报告若要写「总共花了多少钱」，必须说明是哪种口径，或去账单页取数。

### G.5 需要更正的历史数字

之前各处（含 Lead 自己的记录）流传过这些数，**与 G.1 的整段实测不一致**，报告不得直接引用：

| 旧数字 | 实际是什么 | 应该写成 |
| --- | --- | --- |
| 「1.78 it/s = 58,000 tok/s」 | 开局 200 步的**探测窗口**值（`docs/24:738-746`，batch 16 合成探针 57.1k，加 `max-autotune-no-cudagraphs` 后 60,951） | 整段实测 ep2 54,970 / ep3 57,111 tok/s；58,000 标注为探测值 |
| 「3090 14,100 tok/s」 | **全仓库无出处**（核查员确认：全 `docs/` 只有 `docs/24:757` 表格自身与本文转引，没有任何日志/脚本/推导给出它）；且与整段实测互斥——用它反推 MFU 是 62%，用实测 13,551.6 是 **59.6%**；同表 `:760` 的「11.21 h」既不符 14,100 口径（10.88 h）也不符实测 11.3161 h | 用整段实测 **13,551.5 tok/s**（分段 13,353 / 13,626）；若要保留 14,100，只能写「早期窗口近似值，出处仅 `docs/24:757` 表格自身」 |
| 「2.64 h / ¥18.4 一个 epoch」 | 探测值外推 | 整段实测 2.6852 h / ¥18.74（ep3） |
| `docs/24:742` 的「优化步/s 6.25 / 1.77」 | 又是 `max_steps/runtime` 口径，且是 200 步窗口 | 整段实测 1.6776 / 1.7429 步/s |

### G.6 ⚠️ 已核实的死字段（可写进报告，是"阴性结果"的好素材）

`retnet_hf/train.py:175` 声明了 `--eval_batches`（默认 40），**全文件只出现这一次**，
从未被读取 —— 命令行走廊里 `--eval_batches 40` 是装饰。每次评测都评满整个验证集
（`eval_samples_per_second ≈ 87.2`，`eval_runtime ≈ 9.53-9.56 s`）。
这是本项目第 4 个"死字段"类缺陷（前三个的清单见 `docs/23`/`tasks/v15_500m.json` 注释）。
另：`tasks/v15b_500m_24g.json` 注释里写的 `train.py:106` 是**空行**（陈旧引用），`--eval_batches` 的真身在 `:175`。

---

## H. 核查第一轮的更正 + Lead 的拍板（2026-10-13）

### H.1 已更正的错误（见对应小节）

| 位置 | 原写 | 更正 | 谁发现 |
| --- | --- | --- | --- |
| §C | ep2/ep3 每段 552,140,800 | **552,075,264**（`16848 × 32768`）；累计 1,656,340,480 → **1,656,209,408**；token/参数 3.285 → **3.28496** | Lead 自查 |
| §G.1 | seg1 停在 10,782 步（由 `epoch: 0.16` 反推） | **10,661 步**；seg1 1.6300 步/s、seg2 1.6633 步/s（合计行不变） | 核查员 |
| §G.2 | `trainer_utils.py:389` 做除法 | 定义在 `:389`，做 `num_steps/runtime` 的是 **`:411`** | 核查员 |
| §G.3 | "只有 ep3 带收尾行" | `_peak_v15b.txt` / `_peak_v15b_seg1.txt` **也有** | 核查员 |
| §G.5 | 14,100 出处"本轮未定位" | **全仓库无出处**，判 ❓，报告改用 13,551.5 | 核查员 |
| §B | v9 base 占比未知 | **48.73%**（131,377 行 × 2049 = 269,191,473） | 核查员 |

**教训（写给全组）**：核查员第一轮报「累计 1,656,340,480 ✅」是**在 Lead 的错误逐段数上做加法**，
属内部自洽而非独立推导。**任何 token 数必须自己从 `max_steps × batch × seq_len` 乘出来。**

### H.2 Lead 拍板：不报错的缺陷怎么组织

outline §3.4 原定「三处」，但实际候选已有 5 个。**不要硬凑数目**，改按后果分四类，
并显式写明「本文不主张这是穷尽清单」：

| 类别 | 缺陷 | 后果 | 证据 |
| --- | --- | --- | --- |
| ① 训练数值被静默改变 | `--resume` 被检查点的 `train_batch_size` 覆盖（16 → 4） | 实际 batch 与配置不符 | `trainer.py:2298`；`retnet_hf/train.py:91` 定义 / `:394` 调用 |
| ① 同上 | 缺 `optimizer.pt` 时 `scheduler.pt` 也不恢复 → LR 从峰值重来 | 学习率曲线错 | `trainer.py:3523` 是前置赋值，**真值条件行是 `:3524`** |
| ② 产物不可用 | `torch.compile` 的 `_orig_mod.` 前缀使检查点不可加载 | 存档等于废的 | 已核实 `train_20261012_212943.log:43` |
| ③ 度量失真 | `train_steps_per_second == max_steps / runtime` | 分段/续训运行的吞吐被高估最多 6.32× | `trainer.py:2823`；§G.2 |
| ④ 测试空转 | 等价性测试自己跟自己比 | 缺陷不会被发现 | — |

### H.3 Lead 拍板：阴性结果的代际红线

用户明确要求「只写 0.5B 这一代，完全不提前代」。核查员指出 §3.5 有 13 条阴性结果里，
**融合 CE（−13%）、GroupNorm 改写（−9.2%）、`max-autotune` 0% 这三条的数据全部来自 272M / 3070**，
照抄进本代报告就是越界。

**决定：§3.5 只保留本代实测的阴性结果。凡数据来自上一代的，一律删除，不写"未在本代复测"。**
例外（可保留，因为是本代数据）：
- `max-autotune` 在本代会**崩**，traceback 在 `modeling_retnet.py:295`（`docs/24:732` 与 `tasks/v16` 写的 **507 是错的**；证据 `logs/opt500m/O2_reduce-overhead.err:48`）
- `max-autotune-no-cudagraphs` 在 PRO 6000 + 0.5B 上 **+6.7%**（`docs/24:728`）
- 物理 batch 2 在本代是**吞吐**负结果（b4a1 624 ms vs b2a2 672 ms，快 7.7%），显存反而更省（15.49 vs 22.32 GiB）——**不要照抄"显存不可行"**

### H.4 Lead 拍板：§6.2 解码吞吐

`logs/bench_v3/decode_speed.json` 是**前代 272M 产物**（`model_dir: checkpoints/retnet_272m_rp`），
不得当成本代数据。**Lead 会在 bench 全量跑完后用 `scripts/bench_decode_speed.py` 对 ep3 重跑一次**，
把结果路径交给写手。写手在收到之前，§6.2 只写机制（三范式、recurrent 解码），不写数字。

### H.5 必须独立复核的头号发现（写手提出，尚未验证）

写手从 ep3 safetensors 逐张量读出：

- 全部 **41 个 `GrowthLayerNorm` 的 `gamma_new` / `beta_new`（82 张量）逐位为 0**
- embedding 的第 `768:1408` 列全零（12,800,000 个元素）
- L0–L17 每层 **70.2%** 权重恰好为 0；L18/L19 为 51.5%
- 全 checkpoint 恰好为 0 的元素 **338,196,736 / 504,179,896 = 67.08%**；非零 **165,983,160**

Lead 的独立核对：`1 − (768/1408)² = 70.25%` —— 与 L0–L17 实测的 70.2% 吻合，
说明新增的 640 维**读、写两侧全死**（不是只死一边）。
若成立，则这个"5.04 亿参数模型"**功能上等价于一个 768 维、约 1.66 亿非零参数的模型**，
`token/参数` 应按非零参数算 = **9.98**（正好落在「10 token/参数」这条经验线上）。
**这是本报告最重要的潜在发现，必须由核查员独立确认后才能写进结论。**

### H.6 其他待办

- **bench 方法论（核查员已给结论，采信）**：`bench/runner.py:47,99` 的 `encode_prompt` 与训练侧
  `encode_dialogue(s)_batch` 共用 `retnet/chat.py:29/34/40` 同一套原语（`chat.py:130` docstring 明写
  "tokenized exactly like the training data"）⇒ **不存在格式错配**。
  冒烟的 57.7% 泄漏**全部落在被截断样本上**（`(leak,truncated)` = (F,F)7 / (T,T)15 / (F,T)4），
  ⇒ `leak_untruncated = 0.0`、`truncated_rate = 73.1%`、`score_untruncated = 1/7 = 0.143`。
  低分主因是**题库 max_new 只有 16–160，而训练回复中位约 570 token** 的预算失配。
  §5.5 结论保留但**理由必须换**，且 `leak_untruncated = 0` 应当作**正面证据**写。
- **数据侧独立缺陷**：R1 的 4000 字符上限使 **6,364 / 110,000 = 5.79%** 的回复以未闭合 `<think>` 结尾
  （原始 output 未闭合为 0）。`docs/22` 第 10 条写的是 6,270，**以 `scripts/audit_r1_think.py` 为准**。
- **首次评测吃编译预热**：每次运行第一次 `eval_runtime` 是 ep2 90.86 s / ep3 23.08 s，之后 9.53–9.60 s。
  facts 里写的上界「9.56」应放宽到 **9.60**。
