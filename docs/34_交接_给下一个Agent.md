# 交接文档 —— 给下一个接手的人/Agent

> 写作时间：2026-10-19 23:35
> 前一个 agent（我）在这一轮里犯了较多错误，详细清单见 **`docs/33_失败合集.md`**，
> 建议先读那份，再读这份。**不要相信文档里的任何数字，都能复算。**

---

## 1. 三十秒状态摘要

| 项 | 状态 |
|---|---|
| RetNet-0.5B（504,179,896 参数）v21 | **训练完成** 159,857 步 / 20.8 小时 / 双卡 |
| 域内 held-out | **3.5023**（ep3 基线 3.6287）→ **判据未达成**（要求 ≤3.2659） |
| 中途最佳 | **2.8471** @ step 17,500（−21.5%）→ 之后被代码训练冲掉 |
| HuggingFace | **已上传** https://huggingface.co/Ayaka157/RetNet-0.5B-v21 |
| 本地服务 | **在跑** http://127.0.0.1:8077/play/（3070，26.6 tok/s） |
| 云实例 | **已关机**（数据盘保留模型、日志、语料） |
| 余额 | 约 ¥20（10-19 09:14 时有 ¥145.05） |
| 上下文扩展（16K） | **没做成**；16K 三次失败，**8K 从未尝试** |

---

## 2. 现在磁盘上有什么（本地）

```
checkpoints/retnet_500m_v21_export/
    model.safetensors      2,016,755,680 B
    config.json
    sha256 = a76da7163b2f9718fc1f112b273efb89347940b6385f9f5ba073a5c6a1235e80   ← 已校验

logs/curves/full_history_final.csv      8,111 行（step 20 → 159,840，119 条 eval）
logs/samples_v21_final.txt              6 条生成样本
logs/holdout_eval.jsonl                 4 次 held-out 评测原始记录
_archive_from_instance/                 从实例取回的完整包（20.6 MB，含 30 个归档快照 + 10 MB 原始日志）

docs/31_v21训练报告.md                  技术报告（§5 是真实结果）
docs/32_当前状态与接手须知.md            结果与失败原因
docs/33_失败合集.md                     我的错误清单（先读这个）
docs/figures/v21_training_status.png    四联图（held-out 呈 V 形）
```

**云实例数据盘里还有**（关机后保留，重新开机可访问）：
- `/root/autodl-tmp/RetNet/checkpoints/retnet_500m_v21/`（含 4 个训练 checkpoint + 导出）
- `/root/autodl-tmp/RetNet/data_cache/`（`hf_pack_ctx16k` 0.5B 混合包、`hf_pack_code2` 9.4 GB、
  `hf_pack_sft`、`hf_pack_math`、`hf_pack_prog`、`ctx16k_val`）
- `/root/autodl-tmp/RetNet/logs/`（全部日志与归档）

---

## 3. 已验证的事实（可用下面的命令复算）

### 3.1 gamma 已经塌了 —— **这是当前模型最大的缺陷**

实测 440 个 `blocks.*.msr.gamma`（20 层 × 22 头）：

| 分位 | 半衰期（token） |
|---|---|
| p25 | 0.45 |
| **p50** | **1.22** |
| p75 | 2.48 |
| p90 | 18.3 |
| max | 604 |

- 只有 **56/440** 个头半衰期 ≥10 token，**18/440** ≥100 token
- 设计初值是 `1 - 2^(-5-h)` = 0.9688 … 1.0000（半衰期 22 → 1400）
- 原因：gamma 是裸 `nn.Parameter`，HF 的 no-decay 规则只匹配 `bias` / `LayerNorm`，
  两头都不沾，于是被 AdamW 的 decoupled weight decay 一路压小（每个 epoch ×0.90）

**复算**：
```python
from safetensors.torch import load_file
import math, torch, numpy as np
sd = load_file('checkpoints/retnet_500m_v21_export/model.safetensors')
v = torch.cat([t.flatten() for k, t in sorted(sd.items()) if k.endswith('msr.gamma')]).numpy()
h = np.log(0.5) / np.log(v)
print(np.percentile(h, [25, 50, 75, 90]), (h >= 10).sum())
```

### 3.2 域内 held-out 是 V 形

| tag | 步数 | loss |
|---|---|---|
| `ep3_full`（基线） | 0 | 3.6287 |
| `v21_step15000` | 15,000 | 2.8666 |
| `v21_step17500` | 17,500 | **2.8471** |
| `v21_final` | 159,857 | **3.5023** |

分域（ep3 → 最终）：`chinese_fineweb_edu_v2` 3.852 → 3.861（**退回**）；
`wikipedia_zh` 3.647 → 3.732（**更差**）；`cosmopedia` 3.056 → 2.839；
`fineweb_edu` 3.722 → 3.500；`open_web_math` 3.398 → 2.747。

### 3.3 其它实测数字

- 训练吞吐：**1.69 it/s × 65,536 token/步 = 110,756 tok/s**（双卡，DDP `find_unused_parameters=False`）
- 双卡效率：**1.93×**（`SYS` 拓扑无 NVLink）
- 本地推理：**26.6 tok/s**（3070，bf16，`recurrent` 解码，context 2048）
- 数据：训练 token 10.23B；唯一 token **9.81B = 19.46 token/参数**

---

## 4. 服务怎么起（**别用 powershell 5.1**）

```powershell
# 正确：pwsh 7
pwsh -NoProfile -ExecutionPolicy Bypass -File serve\start_play.ps1 -NoTunnel

# 管理
pwsh -File serve\start_play.ps1 -Status
pwsh -File serve\start_play.ps1 -Stop
```

- `serve/start_play.ps1` 是 **UTF-8 无 BOM**，Windows PowerShell 5.1 会把中文当 GBK → 解析报错
- 它会把 pid 写进 `logs/play_state.json`，`-Status`/`-Stop` 靠这个精确定位
- **它的默认模型已经是 `checkpoints/retnet_500m_v21_export`**（我改过 4 处：`start_play.ps1`
  的 L14/L37 + `retnet_play_server.py` 的 L5/L364）
- `index.html` 是**每请求读文件**（`retnet_play_server.py` L356），改完刷新即可，不用重启
- 页面：`http://127.0.0.1:8077/play/`；健康检查 `/play/api/health`；对话 `POST /play/api/chat`
  （**流式 SSE**，每行 `data: {...}`，不是 JSON）

---

## 5. 下一步：我建议的优先级

### 优先级 1（零成本、十分钟、能立刻验证一个假设）

**本地重设 gamma，做前后对比。** 不需要 GPU，不需要云实例：

```bash
python scripts/reinit_gamma.py --model checkpoints/retnet_500m_v21_export \
    --out checkpoints/retnet_500m_v21_gammafix --formula orig --dry-run   # 先看
```

原理：RetNet 没有位置编码，位置信息通过 `γ^Δ` 进入。把 gamma 从 0.1–0.55 恢复成
`1-2^(-5-h)`（半衰期 22→1400），同一个权重的记忆通道立刻变长。

**但必须同时验证两件事**（不能只看 loss）：
1. `python scripts/gen_samples.py --model <新目录> --tag gammafix` 与
   `logs/samples_v21_final.txt` 逐条肉眼对比 —— **模型是在旧 gamma 下训练的，
   换 gamma 后分布可能不匹配，长文可能变差**；
2. 用一段 ≥800 token 的长输入测"能否引用前文"，这是旧 gamma 一定做不到的。

**如果两边都不变差，这就是一个免费的收益**；如果变差，说明 gamma 必须靠训练适配，
那就进入优先级 2。

### 优先级 2（需要重新开机，约 ¥5）

**测 8K。** 我完全没试过，这是明确的疏忽。正确流程：

```bash
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/inductor_cache   # 缓存复用，关键
# 先关 compile 测纯训练速率（这是唯一可靠的可行性判断）
python run_task.py --config tasks/v22_8k.json      # 需要新造：seq_len 8192, batch 4
# 看纯训练 it/s，>0.4 才算可行；可行再开 compile，且至少等 3 步
```

**注意**：`tasks/plan_ctx16k.json` 现在的 `val_dir` 是 `data_cache/ctx16k_val`（16385 宽）。
做 8K 需要另造一个 **8193 宽**的 val，否则会重现
`AssertionError: rows are 2049 wide, expected 8193`。

### 优先级 3（需要 ¥50–70 与 2–4 小时）

**补通用中文 + 混合 replay**。这是唯一能把 held-out 救回 2.9 以下的路：
重新下载 `Morton-Li/ChineseWebText2.0-HighQuality`（约 0.23B）或类似通用中文，
与现有语料配成 **通用 60% / 代码 25% / 数学 15%**，跑 0.5–1B token，
最后叠 3–5k 步 SFT。

**不要再犯的错**：这次必须**同时**在代码段掺通用文本，而不是先练完代码再补。

### 优先级 4（用户明确要求过，但从未执行）

**下游基准**：C-Eval / CMMLU / HumanEval / GSM8K 一个都没跑过。
用户要求"未经确认不要启动"，所以**先问，再跑**。

---

## 6. 必须知道的坑（都是我踩过的）

| 坑 | 说明 |
|---|---|
| `.out` 日志会被截断 | 每次重启训练用 `>` 重定向。**完整历史在 checkpoint 的 `trainer_state.json: log_history`** |
| 别杀裸 `sleep` | 死手是 `bash -c "sleep N; sync; shutdown"`，杀掉 sleep 会让 bash **继续执行 shutdown**（我把实例关过一次） |
| `--add-eos` 默认关闭 | 所有 flat 包**没有文档边界**。要边界必须显式传 |
| `--text-field auto` 会吃元数据列 | `prog_zh` 的 10 个数值列会被当正文。要显式 `--text-field text` |
| 部署前必须 diff 实例与本地代码 | 我因为实例上 `train.py` 是旧版，v22 白失败一次 |
| `min_shards` 要匹配现实 | 配 10 但只有 1 个分片 → 阶段永远不就绪 |
| 显存吃满就没法评测 | 我让训练吃到 99%，导致 14 小时无法测 held-out。**必须预留评测空间** |
| `plan.py` 会覆盖 `state.max_steps` | 热加载生效，所以 tqdm 进度条的总数可能是旧的，**看实际迭代数** |
| 磁盘守卫只删 `checkpoint-*` | 不会碰顶层 `model.safetensors`，可以放心 |
| 改 shell 脚本别用字符串替换 | PowerShell 会吃掉 `$VAR` 和反引号。用编辑工具 + `bash -n` |
| Python 写文件会转 CRLF | 仓库里多数文件是 LF，改完要还原，否则 diff 膨胀 |
| 关机前确认产物已取回 | 我在 shutdown 后才发现 1.9 GB 模型没拉，靠 17 块并行才救回来 |

---

## 7. 一句话给下一个 Agent

**这一轮真正买到的教训是：看不见的训练等于赌博，配比比总量重要。**
模型本身（3.5023）不值得当成果宣传；但它是一个可复现的、
带完整日志和明确缺陷定位的基线，比"重复同一份数据练 5 轮"要有价值。
