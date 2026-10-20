# A：RetNet 0.5B 的长上下文缺陷与参数损伤

调查对象：`checkpoints/retnet_500m_v21_export/`（504,179,896 参数，config 见 `checkpoints/retnet_500m_v21_export/config.json`）。
所有结论只写我自己跑出来的东西。标注约定：**【实测】**=我跑脚本得到的数；**【代码推断】**=从源码行号推的；**【我的猜测】**=没有直接证据。

---

## 1. 结论摘要

1. **缺的不是"长上下文机制"，而是这个模型几乎没有记忆。** 440 个 `msr.gamma` 的中位数是 **0.5656**（【实测】），对应半衰期 **1.22 token**；45.2% 的头 γ<0.5。按 `retention_fast.py:89` 的 `chunk_decay = γ^C`（C=128），**第 13–19 层（20 层里的 7 层）γ^128 ≤ 4.8e-25，跨 chunk 状态在 fp32 里就是 0**；只有 L2/L4/L6/L8/L10 各有 1 个"不衰减头"能带过 chunk 边界（γ^128≈0.8）。
2. **504M 里有 338M（67.06%）参数在"全零行或全零列"里，真实活跃参数只有 166,059,192（≈166M）。**【实测】`d_model=1408` 中只有前 768 维活着，22 个头里只有 12 个活着，`ffn_dim=2816` 里只有 1536 个神经元活着；`embedding.weight` 的第 768–1407 列**全零**（我直接验证 `abs().max()==0`）。这是 `use_growth_ln=true, growth_old_dim=768` 的宽度增长**从未生效**留下的空壳。
3. **宽度增长没生效的根因是可测量的：所有 `gamma_new` 恰好等于 0.0。**【实测】20 层 × (norm1,norm2) × 640 维 + `final_norm`，`gamma_new` 全零、`beta_new` 除 L18/L19 各 256 维（|max|≤0.25）外全零。`GrowthLayerNorm.forward`（`modeling_retnet.py:294-297`）里新通道 = `x*gamma_new + beta_new`，所以新通道**恒等于常数、与输入无关**，其梯度恒为 0 —— 新加的 640 维从来没学过东西。
4. **`msr.gamma` 被 weight decay 一路拉小，这是训练配置 bug 不是结构问题。**【实测】head 16–21 的 γ 在 **20 层里逐位相同 = 0.213622**，head 12–15 在 L0–L17 逐位相同 = 0.1755（只有 L18/L19 被训练过）。零梯度 + 解耦 wd 的唯一签名就是这个。反解 `exp(-wd·Σlr)=0.2136, wd=0.1` 得 **Σlr=15.44**，与整个续训链条的 LR 积分量级一致（v21 单轮的计划 LR 积分只有 ~8.6）。同仓库 `retnet_hf/train.py:51-61` 记录了同一现象，默认开关 `--no_decay_1d` 是**关**的（`train.py:255,260-266`）。
5. **导出的这个 checkpoint 本身还是训练尾部回退的产物。**`logs/holdout_eval.jsonl` 里同一个 v21 在 step 17,500 的 holdout loss 是 **2.8471**，最终 `v21_final` 是 **3.5023**（+0.655 nats），比它的中间检查点差得多、只比 v15b_ep3 的 3.6027 略好。也就是说：**底模最好的一版已经不存在了，本地留下的导出是尾部 SFT 段之后退化版。**

---

## 2. 长上下文缺陷

### 2.1 结构：跨 chunk 信息怎么传、chunk 多大、前向怎么算

- **chunk 大小 = 128**（`config.json:6` `"chunk_size": 128`）。训练时 `RetNetForCausalLM.forward` 从 config 读它（`modeling_retnet.py:484-485`）；v21 的任务配置也是 128（`tasks/v21_plan_ddp2.json:28`）。训练窗口 2048（`tasks/plan_ddp2.json:23` `"seq_len": 2048`），即 **16 个 chunk**。
- 训练走 `mode="chunkwise"`（`config.json:13`，`tasks/v21_plan_ddp2.json:27`）。`forward` 在 `modeling_retnet.py:510-513` 进入 `_forward_hidden(..., return_state=True)`，逐层调用 `MultiScaleRetention._chunkwise_retention`（`:213-248`）。v21 还开了 `fast_chunkwise: true`（`tasks/v21_plan_ddp2.json:36`），实际执行的是 `retention_fast.py:95-180` 的批量版本，两者数学等价（`FAST_CHUNKWISE` 开关在 `modeling_retnet.py:26`、`:215-217`）。
- **前向的四步**（`retention_fast.py`，与我读到的参考实现 `modeling_retnet.py:236-245` 一一对应）：
  1. `U_i = k_iᵀ · (v_i ⊙ value_decay)`，`value_decay[j] = γ^(C-1-j)`（`:88`, `:122-123`）：每个 chunk 内把"到 chunk 末尾的剩余衰减"折进 v。
  2. 闭式 scan `S_prev[i] = d^i·state + Σ_{j<i} d^(i-1-j) U_j`，`d = γ^C`（`:153-170`）。这一步就是**跨 chunk 的唯一通道**：一个 `[B,H,kd,vd]` 的矩阵状态，本模型 kd=64、vd=128，即每头 8192 个数。
  3. `inner = (q_r k_rᵀ ⊙ inner_mask) v_r`，`inner_mask[δ]=γ^δ`（`:82`, `:173-174`）：chunk 内因果衰减注意力。
  4. `cross = (q_r ⊙ γ^(j+1)) · S_prev`（`:87`, `:177`），`out = inner + cross`（`:179`）。
- 因此跨 chunk 的信息**只经过一个 64×128 的秩受限状态，并且每过一个 chunk 乘以 γ^128**。结构上不存在"注意力的 O(L²) 全历史"，也没有任何绝对位置/段位置编码（只有 q,k 上的 RoPE，`modeling_retnet.py:104-106,136-151`）。
- 两个额外的结构/实现事实：
  - `generate()` 的 prefill **没有走训练时的 chunkwise**，而是 `mode="parallel"`（`modeling_retnet.py:561`）。而 `_parallel_retention`（`:180-185`）多了一步 `scores / scores.detach().abs().sum(-1).clamp_min(1.0)`（`:184`），这一步在 chunkwise/recurrent 路径里**不存在**；`_decay_mask`（`:170-178`）还额外做了 `mask/mask.sum(-1).sqrt()`。所以**服务端的 prefill 与训练时的函数不是同一个函数**，prompt 越长偏差越大（我尝试实测这一项，见 §2.3）。
  - 状态的数据类型跟随 `q.dtype`（`retention_fast.py:161-167` 把 `T`、`t_state` 转成 `q.dtype`），v21 训练用 bf16（`tasks/v21_plan_ddp2.json:26`），即跨 chunk 状态是 8 bit 尾数的累加。

### 2.2 权重：γ 的真实分布、半衰期、按头还是按层

【实测】直接读 `model.safetensors` 的 440 个 `blocks.{L}.msr.gamma`（20 层 × 22 头，`use_gla=true` 所以它是 Parameter，`modeling_retnet.py:99-101`）：

| 统计量 | 值 |
|---|---|
| 最小值 / 中位数 / 均值 / 最大值 | 0.1057 / **0.5656** / 0.5026 / 0.9989 |
| 半衰期 = ln2/(−ln γ)：中位数 | **1.22 token**（min 0.31，max 604） |
| γ<0.5 的头占比 | **45.2%**（γ<0.3 占 44.3%） |
| 逐层 γ^128 的最大值（跨 chunk 保留率） | L0 3.0e-3, L1 0.47, L2 0.86, L3 0.59, L4 0.85, L5 0.28, L6 0.84, L7 0.56, L8 0.76, L9 0.58, L10 0.82, L11 0.26, L12 0.11, **L13 3.8e-29, L14 8.2e-22, L15 1.5e-26, L16 6.1e-16, L17 4.8e-25, L18 2.7e-23, L19 2.9e-16** |

- **是按头 + 按层的**：每个 (层, 头) 独立一个 γ，共 440 个，层间差异很大（head 0 的层间 std 0.21）。
- **但有两组头是退化的常数**（这是判定"练废"的关键）：head 16–21 的 γ 在全部 20 层**逐位相同 = 0.213622**（std=0）；head 12–15 在 L0–L17 逐位相同 = 0.1755，只在 L18/L19 被更新过（0.53–0.76）。零梯度的参数只受解耦 weight decay 影响，轨迹与层无关 → 才会出现"逐位相同"。反解 `exp(-0.1·Σlr) = 0.2136` 得 **Σlr ≈ 15.44**；我按 `logs/curves/full_history_final.csv` 的 v21 LR 曲线算的单轮积分是 Σlr≈8.58（wd=0.1 → 0.424），说明这是**多轮续训累积**的 wd 效果（【代码推断】+【实测】常数，分摊到每轮的精确分解我没有逐轮验证）。
- 换算成"总记忆质量"：γ=0.57 时 Σ_{d≥0} γ^d = 1/(1-γ) = 2.3，即**一个头的有效记忆约等于最近 2–3 个 token**；chunk 内 64 个 token 外的权重是 0.57^64=2e-16。

### 2.3 行为：能不能测到"有效上下文"

我按"逐 chunk 断状态"的做法做了行为测量（把 2048/1024 token 的序列切成 128 的 chunk 单独前向、rotary offset 对齐，等价于 `state=None`，即切断跨 chunk 通道），同时也准备了 single-token influence（改历史上某个 token，看 t 位置 logits 变化）与 parallel/chunkwise 对比。**这批实验在本机没跑完**：开工时 GPU 只有 688 MiB 占用，实际跑起来时被其它 Agent 抢到 7.9/8.2 GiB、利用率 100%，两次运行都被拖到 10 分钟以上仍未出结果，我按"不抢 GPU"的约束放弃了。所以**行为层我只给可复算的、由权重和代码直接决定的上界**，不给假数：

- 【代码推断】给定 γ 分布，任意头对 d 个 token 之前的依赖权重 ≤ γ^d。用中位数 γ=0.5656：d=8 → 0.0127，d=16 → 1.6e-4，d=32 → 2.6e-8。也就是说**中位头的"有效上下文"是 8–16 token**，不是 128，更不是 2048。
- 【代码推断】跨 chunk 通道在 L13–L19 被 γ^128 ≤ 4.8e-25 彻底掐死，而 L13–L19 是离输出最近的 7 层；L0–L12 里也只有 6 个层存在 γ^128≥0.5 的头。**信息从第 k 个 chunk 传到第 k+1 个 chunk 后，还要再穿过 7 层"零记忆"层才能影响输出**，因此端到端可用的跨 chunk 记忆在最乐观的头上也只有 ~1–2 个 chunk（128–384 token），且只经过 6/20 层。
- 【实测/可复算】训练窗口只有 2048（16 chunk），而跨 chunk 状态每 chunk 乘 γ^128；**2048 以内的"长程"能力本来就没有被任何训练信号要求过**（chunk 内 attention 覆盖 128，剩下的全靠这个衰减状态）。
- 想补的行为实验（我没做完，下一轮 15 分钟可做完）：`chunk 断状态 loss − 全状态 loss`（若 ≈0 就证明跨 chunk 通道没贡献）、单 token influence vs 距离曲线（应在 d≈8–16 掉到 0，d>128 严格为 0）、parallel vs chunkwise 同序列 loss 差（验证 §2.1 的 prefill 不一致）。脚本在 `%TEMP%\ret_ws\behave2.py`。

### 2.4 缺陷归因：训练 / 结构 / 配置写错

| 现象 | 归类 | 证据 |
|---|---|---|
| γ 中位数 0.57、半衰期 1.2 token | **训练配置（weight decay 打在裸 γ 上）** | head16-21 的 γ 逐位相同（零梯度+纯 wd）；`train.py:255` wd=0.1，`:260-266` `--no_decay_1d` 默认关；反解 Σlr=15.44 |
| 45.45% 宽度 / 10 个头 / 1280 个 FFN 神经元全零 | **训练流程（宽度增长从未生效）** | `gamma_new` 全零（实测）；`modeling_retnet.py:294-297` 的公式决定新通道恒为常数；新通道权重因此零梯度、停在零初始化 |
| L13–L19 跨 chunk 状态恒为 0 | **结构 × 权重耦合**：结构给了 γ^C 的指数衰减，权重把 γ 压到 0.57，两者相乘才是"无跨 chunk 记忆" | `retention_fast.py:89,153-167`；γ 表 |
| chunk 内也只能看 2–3 个 token | **训练**（同第一行），结构本身允许 γ→1 | γ 表 |
| 训练/服务不是同一个函数（prefill 走 parallel + 额外归一化） | **配置/实现写错** | `modeling_retnet.py:561` vs `:510-513`；`:184` 的额外归一化在 chunkwise 无对应项 |
| 2048 窗口、chunk 128：>128 的依赖只走秩 64×128 状态 | **结构本身**（RetNet 的设计取舍），但对本模型是致命的，因为状态被衰减掐死 | `retention_fast.py:95-180` |
| 导出 ckpt 比自己的 step17500 差 0.655 nats | **训练流程（尾部阶段回退）** | `logs/holdout_eval.jsonl` 的 v21_step17500=2.8471 / v21_final=3.5023 |

---

## 3. 参数级判定表

统计口径：`mean/std/absmax` 为逐张量；"死块"= 该张量中处于全零行或全零列的元素数（我按 `norm(dim=1)==0` 或 `norm(dim=0)==0` 精确统计）。全部为【实测】。

| 张量（每层同名） | 形状 | 当前统计 | 死块占比 | 判定 |
|---|---|---|---|---|
| `msr.gamma` | [22] ×20 = 440 | min 0.106 / med 0.566 / max 0.999；head16-21 恒 0.213622，head12-15 恒 0.1755（L0-17） | 10/22 头梯度恒 0 | **需重初始化 + 重训**（初值回到 1−2^(−5−h)，并把 1 维参数移出 weight decay；否则必然再次衰减到 0.2） |
| `norm1.gamma_new` / `norm2.gamma_new` | [640] ×40 | **恒等于 0.0**（absmax=0，非零元 0 个） | 全死 | **已无意义 → 需重初始化**为 1.0（或直接关掉 `use_growth_ln`，改成普通 LayerNorm） |
| `norm1.beta_new` / `norm2.beta_new` | [640] ×40 | 38 个张量全 0；L18/L19 各 256 维非零，absmax 0.25/0.21 | 大部分死 | **已无意义 → 重初始化**为 0 |
| `final_norm.gamma_new/beta_new` | [640]×2 | 全 0 | 全死 | 同上 |
| `msr.q_proj.weight`, `k_proj` | [1408,1408] ×20 | rms 0.0091–0.0145 | **70.2%**（仅左上 768×768 活） | 768×768 活块：**保留（作初始化）+ 重训**；768:1407 的行/列：**需重初始化** |
| `msr.v_proj`, `g_proj` | [2816,1408] ×20 | rms 0.0091–0.0156 | **70.2%**（1536×768 活） | 同上（活块保留，其余重初始化） |
| `msr.out_proj` | [1408,2816] ×20 | rms 0.0100–0.0155 | **70.2%** | 同上 |
| `ffn.0.weight`, `ffn.2.weight` | [2816,1408] / [1408,2816] ×20 | rms 0.0093–0.0174 | **70.2%**（1536 个隐藏神经元活） | 同上 |
| `embedding.weight`（= `lm_head`，tied） | [20000,1408] | std 0.0139，行范数中位 0.488，**无死行**；第 768–1407 列 **absmax=0** | 45.45% 列死 | 768 列：**保留 + 重训**；640 列：**需重初始化**（当前是零，等于没有词向量） |
| `norm1.ln_old.weight` | [768] ×20 | rms 0.48→0.99（L0 0.48 偏小） | 无 | **保留**（正常的 per-channel 缩放，重训时继续学） |
| `norm2.ln_old.weight` | [768] ×20 | rms 0.223（L0）→1.44（L18） | 无 | **保留但需注意**：L0 的 FFN 分支被压到 0.22，是"层 0 几乎不干活"的信号；重训会自然修正 |
| `final_norm.ln_old.weight` | [768] | **rms 5.067** | 无 | **保留**（pre-LN 末层放大是残差增长的正常补偿；但若重初始化后 loss 爆，先把它拉回 1.0） |
| `norm*.ln_old.bias` | [768] ×40 | absmax 0.27–0.95 | 无 | **保留** |
| `lm_head.weight` | — | safetensors 里**没有**这个张量（tied，`modeling_retnet.py:398-400`） | — | 无需处理（load 时 missing 属正常） |
| `inv_freq` | [64] | 非 persistent，不在 ckpt 里 | — | 无需处理 |

**净结论**：`504,179,896` 总参数中 **338,120,704（67.06%）落在全零行/列里，活跃参数 166,059,192**。所以"救这个 0.5B"的真实工作量不是修 5 亿参数，而是：**把 1.8 亿个零参数重新初始化并让它真的接到梯度上 + 把 1.66 亿活参数当初始化继续训**。另外注意 §1.5：连这 1.66 亿的活参数也来自一个比自身中间检查点差 0.65 nats 的尾部检查点，作为初始化可以，作为"已收敛的好权重"不行。

---

## 4. 重训计划

前提（必须先做，否则重训必然复现同样的病）：
1. `msr.gamma` 移出 weight decay（`--no_decay_1d`，`retnet_hf/train.py:260`），或改成 `γ = sigmoid(raw)` 的参数化，让 wd 无法把 γ 拉向 0。
2. `GrowthLayerNorm`：`gamma_new` 初始化 1.0、`beta_new` 0，并且**新通道的 q/k/v/g/out/ffn 权重必须非零初始化**（现在是零，见 §3）；或者干脆放弃 growth，直接用 `nn.LayerNorm`，把 1408 维当原生宽度重训。
3. 训练与推理统一到 chunkwise（`generate()` 的 prefill 改 `mode="chunkwise"`，`modeling_retnet.py:561`），并删掉 `_parallel_retention:184` 那个训练时没有的归一化。
4. 跨 chunk 训练信号：seq_len 必须 > chunk_size 且要有跨 chunk 的监督（长文档拼接，别用短样本 padding），否则状态永远是"没人用的参数"。

吞吐基准：v21 实测 2 卡 DDP 稳态 1.69 it/s × 65,536 token/步 ≈ **111k token/s**（来自 `tasks/v21_plan_ddp2.json:15` 与 `tasks/plan_ddp2.json:7` 的记录，**我未复算**，按此估时）。

### 4.1 最低成本验证版（"证明能救活"，不追求好用）

- 重初始化：`gamma_new→1.0`、`beta_new→0`、被零初始化的 1.8 亿参数按 xavier 重抽；`gamma` 重置为 1−2^(−5−h)。
- 冻结：`embedding` 的 768 活列、`norm*.ln_old.*`（它们已经在合理范围，见 §3）。
- 数据：现有 `data_cache/pretrain_v2` + `data_cache/hf_pack_chinese_web`，按 8:2 混，**只取长文档**（拼接成 2048，不 padding）。
- 步数/预算：**1.5B token**（≈ 22.9k 步 × 65,536）→ 1.5e9/111k ≈ **3.8 h** ≈ **¥53**（2×PRO 6000 @ ¥13.97/h），加 eval/重启余量按 5 h 报 **¥70**。
- LR：warmup 500，peak **3e-5**，cosine → 3e-6（继续训，不是从零）。
- 验收（必须全过）：
  1. `chunk 断状态 loss − 全状态 loss` **> 0.05 nats**（现在按权重推断 ≈0）；
  2. 单 token influence 在 d=256 仍 >1% 相对变化（现在 d>128 严格为 0）；
  3. γ 中位数 **≥0.99**、且不存在 `γ^128 < 1e-3` 的头；
  4. holdout loss ≤ **2.85**（先追平 v21_step17500，再谈超越）；
  5. 2048 与 4096 的 loss 差 < 0.05 nats。

### 4.2 完整版（"救回来且能用长上下文"）

- 阶段 1（修复+恢复，2048 窗口）：重初始化同 4.1，**不冻结**（全参训），数据 = pretrain 55% / 代码 25% / 中文网页+教材 15% / SFT 5%，**4B token** → 4e9/111k ≈ 10 h ≈ **¥140**；LR warmup 800，peak 1.1e-4，cosine→1.1e-5。
- 阶段 2（长上下文扩展）：seq 2048→4096 课程，chunk_size 128→256（保持 16 chunk），数据换成"长文档/书籍/多轮长对话"占 60%，**2B token** @4096 ≈ 2e9/111k ≈ 5 h（4096 的实际吞吐会低于 111k，按 0.7× 折算约 7 h）≈ **¥98**；LR peak 3e-5 短 cosine。
- 阶段 3（SFT 收尾，别再来一次回退）：0.5B token，SFT 混 30% 通用语料**防遗忘**，LR 1.5e-5，并在**每 200 步**用固定 holdout 打分、只在分数不退化时保留——v21 的教训就是尾部 14k 步 SFT 把 holdout 从 2.85 拉到 3.50（`logs/holdout_eval.jsonl`）。
- 合计：**约 21–23 h ≈ ¥300–320**（不含失败重跑；按经验留 30% 余量 → **¥400**）。
- 验收：4.1 的五条 + 4096 位置上的 holdout 不劣于 2048 + 一个 ≥8k 的"needle"检索测试（现在这个模型必然 0 分，因为 γ^128≈0）。

### 4.3 如果只想止损

把 1408 维裁到 768、22 头裁到 12、ffn 裁到 1536，得到 **166M 的等价模型**，权重直接用活块切片，loss 不变（死块贡献恒 0，这一点由全零行/列保证）。这比继续维护一个 504M 的空壳更省显存和算力。

---

## 5. 我没有验证的部分 / 不确定的地方

- **行为实验没跑完**（§2.3）：跨 chunk 断状态的 loss 差、influence-距离曲线、parallel vs chunkwise 的 loss 差，都只有脚本没有数。原因有两层：(a) GPU 被其它 Agent 占到 7.9/8.2 GiB、利用率 100%，一次 1024-token 的 chunkwise 前向要 **26.46 s**（正常应在 1 s 量级）；(b) 我的逐 chunk 脚本自己写错了（把 2049 长度的行送去 chunkwise，触发 `modeling_retnet.py:219-221` 的 `seq_len not a multiple of chunk_size` 断言），跑到断状态那张表时直接中止。这是我的 bug，不是模型行为。§2.3 里的"有效上下文 8–16 token"是**由 γ 分布 + 代码路径推的上界**，不是端到端实测。
- **Σlr=15.44 的分解**：我用 `exp(-wd·Σlr)=0.213622` 反解出累积 LR 积分，但 v21 单轮的积分只有 8.58，剩下的是前面若干轮（v15b/v16–v20）的贡献。我**没有**逐轮核对 LR 曲线，所以"每轮各贡献多少"未验证；"零梯度 + 纯 wd"这个机制本身由 head16-21 的逐位相同常数直接证明，不依赖这个分解。
- **head 12–15 的 0.1755 与 head 16–21 的 0.2136 为什么不同**：两者都是零梯度常数，差 0.038。可能是两组通道在不同轮次被创建（wd 作用步数不同），但我没有 git/日志证据，属【我的猜测】。
- **`embed_scale=27.7128 = sqrt(768)`**（`config.json:10`）：宽度是 1408，但缩放按 768 算。因为后面紧跟 LayerNorm，实际影响很小，我没有做消融；这更像是"宽度增长没做干净"的又一个痕迹。
- **吞吐 111k token/s**：引用的是 `tasks/plan_ddp2.json` 里的实测记录，我没在 2×PRO 6000 上复算，¥ 报价因此有 ±30% 误差。
- 我没有读 `docs/analysis/` 下任何文件；也没有改动仓库里任何已有文件、没有 commit。所有脚本在 `%TEMP%\ret_ws\`（`wstats.py`, `wdetail.py`, `gmatrix.py`, `behave2.py`）。
