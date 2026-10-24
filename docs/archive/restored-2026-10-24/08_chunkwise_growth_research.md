# 08 Chunkwise 与模型生长

> 最后更新：2026-10-09 · 初稿 2026-08-23
> 相关：[`01_项目总览.md`](./01_项目总览.md) §2、[`02_训练管线.md`](./02_训练管线.md) §4

## 1. 背景

100M 模型训完后要放大。从零重训 272M 需要重吃几十 B token，不可行。
本文记录**函数保持生长**的做法与实测，以及生长后训练用的 chunkwise 路径。

## 2. 方法论地图

| 方法 | 论文/来源 | 核心思想 | 对本项目的价值 |
| --- | --- | --- | --- |
| Net2Net | [1511.05641](https://ar5iv.labs.arxiv.org/html/1511.05641) | Net2WiderNet（宽度复制+缩放）、Net2DeeperNet（深度插入恒等层） | 理论基础；但原始版本假设**无归一化或后归一化** |
| LiGO | [ICLR 2023](https://vita-group.github.io/LiGO/) | 线性增长算子 `W' = L(W)`，学出来而非启发式拷贝 | 学习式生长更平滑；小规模性价比低 |
| MSG | [arXiv:2305.02869](https://arxiv.org/abs/2305.02869) | 新增结构掩码为 0，训练中渐进解锁 | 函数保真启动 + 边训练边激活 |
| SOLAR 10.7B | [NAACL 2024](https://aclanthology.org/2024.naacl-industry.3/) | 多层复制+重组为深堆叠 | 深度堆叠的工业级验证（本项目同思路） |
| BERT2BERT | Chen et al. 2021 | FPI（函数保真初始化）+ AKI | 从大模型迁移知识到特定任务 |
| Curse of Depth | [NeurIPS 2025](https://mlanthology.org/neurips/2025/sun2025neurips-curse/) | 深度增加引发训练不稳定，规范化放置/残差缩放是解药 | 本项目「零残差新层」正是残差缩放的函数保真版 |
| flash-linear-attention | [fla-org](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/multiscale_retention.py) | MultiScaleRetention 的 chunkwise Triton 内核 | **直接参考实现，不必从零写** |

## 3. 核心难题：pre-LN 下的宽度扩展

### 3.1 问题的本质（实测复现）

pre-LN 下把 `d_model` 从 D 扩到 D' 后，**新增通道会进入每一层 LayerNorm 的 mean/var 计算**：

```text
即使新通道输入恒 0：mean' = mean · (D / D')     ← 旧通道的归一化输出被系统性改变
```

18 层级联放大后，loss 从 4.5 涨到 **17.8**。
**Net2Net 的「复制 + 缩放」不处理归一化统计，所以它在 pre-LN 架构上不是函数保真的。**

### 3.2 解法对比

| 解法 | 原理 | 评价 |
| --- | --- | --- |
| **零通道 + 统计补偿** | 新通道零注入 + LN 只对旧 D 维计算统计 | ★★★★★ **本项目采用** |
| MSG 式掩码生长 | 新结构置 0 并 mask，训练中逐步 unmask | 需要正确处理 LN，否则统计仍被稀释 |
| LiGO 学习算子 | 用大模型权重训练线性算子 | 需额外训练，小规模性价比低 |
| 接受污染 + 愈合 | 生长后低 LR 愈合 | ❌ 实测 loss 17.8 太高，愈合困难 |

**关键认识**：mask 置零权重 ≠ 新通道激活为 0。真正让 LN 统计不受影响的唯一严格路径是
**把 LayerNorm 参数化改为「旧通道专用」**——即 `GrowthLayerNorm`。

## 4. 实施结果

### 4.1 `GrowthLayerNorm`

```text
old_out = ln_old(x[..., :D])           # LN 只作用在旧 D 维上，统计不受新通道影响
new_out = x[..., D:] * gamma_new + beta_new   # 新通道走独立缩放，初始 gamma_new=0（零侧分支）
```

- 新增通道（embedding 新列 + 全部投影新列/新行）在解锁前**恒为 0**
- 新通道的 `gamma_new = 0` 把输出压到 0 → 进残差后下一层输入的新通道仍是 0
- **`embed_scale = OLD_D**0.5` 是必要修正**：少了它 Δ 从 7.25e-4 爆到 23.5

### 4.2 两条生长线（实测）

| 生长 | 脚本 | 变换 | 函数保真 |
| --- | --- | --- | --- |
| 100M → 143M | `scripts/grow_143m.py` | 深度 12 → 18 层（零残差插入） | 保持 |
| 143M → 272M | `scripts/grow_272m.py` | 宽度 d768 → 1024 + 深度 18 → 20 层 | **Δ7.25e-4**（相对变化 1.6e-4） |

旧 12 个 head 的 `gamma`（GLA 可学习衰减）逐头复制，新 4 个 head 按
`gamma = 1 - 2^(-5-h)` 初始化（h 为 0-based 全局 head 编号）。

### 4.3 Chunkwise 训练级验收

272M RP-SFT 首次跑通（batch 4，v9 数据 280.6M token，seq 512）：

| 指标 | 值 |
| --- | --- |
| 显存 | 5,535 MiB / 8 GiB（开梯度检查点） |
| 速度 | 1.47–1.63 it/s ≈ 3.0–3.3k tok/s |
| 训练完成 | ✅ 20,000/20,000 步，3h42m24s |
| 最终 eval_loss | **2.9946** |
| train_loss 均值 | 3.4148 |
| grad_norm | 4.5–15.2，无 NaN |

**eval_loss 曲线（每 2,500 步）**：3.7946 → 3.5826 → 3.4385 → 3.3100 → 3.1865 → 3.0898 → 3.0279 → 2.9946
——8 次 eval 单调下降，到 20k 步仍在降，**是预算停止不是收敛**。

横向对比：143M RP（v8 数据，45k 步）最终 eval_loss 2.653；272M（v9 数据，20k 步）已达 2.9946。
272M 体积 +90%、数据量 +25%，同等步数下损失已接近 143M 的 45k 步终点。

> chunkwise 梯度正确性由此得到训练级验证——若梯度错了 loss 不可能下降。

### 4.4 模型线

| 模型 | 参数 | 说明 |
| --- | --- | --- |
| `retnet_100m_pretrain_v4_hf` | 100.37M | 预训练，Δ0 迁移 |
| `retnet_143m_hf` | 142.87M | 深度生长（12→18 层），函数保持 |
| `retnet_143m_rp` | 142.87M | RP-SFT，eval_loss 2.6534 |
| `retnet_272m_hf` | 272.30M | 宽度+深度联合生长，Δ7.25e-4 |
| `retnet_272m_rp` | 272.30M | RP-SFT（20k 步），eval_loss **2.9946** |

## 5. chunkwise 路径的当前状态

**三种范式在生产尺度上并不数值等价。** 272M fp32 B=1 实测：

| seq_len | max│diff│ parallel↔chunkwise | rel_p90 |
| --- | --- | --- |
| 256 | 2.461 | 3.3% |
| 512 | **6.770** | 7.1% |
| 1024 | 4.497 | 1.5% |

而 `tests/test_model.py` 断言的 `1e-4` 只在 d_model=64 / L=16 成立。
训练走 chunkwise、`generate()` 预填 prompt 走 parallel——**这条不一致没有回归护栏**，
是动任何 kernel 之前必须先补的（见 [`11b`](./11b_TileKernels可用性评估.md) §5.2）。

> 另注：HF 版的 `_parallel_retention`（**只被 `generate()` 用来预填 prompt**）多了一步
> 参考实现没有的分数归一化，使输出量级比 chunkwise 小约 44 倍。训练路径不经过它。

## 6. 参考文献

- RetNet 原始论文：https://arxiv.org/abs/2307.08621
- Net2Net：https://ar5iv.labs.arxiv.org/html/1511.05641
- LiGO (ICLR 2023)：https://vita-group.github.io/LiGO/
- MSG：https://arxiv.org/abs/2305.02869
- SOLAR 10.7B：https://aclanthology.org/2024.naacl-industry.3/
- Curse of Depth (NeurIPS 2025)：https://mlanthology.org/neurips/2025/sun2025neurips-curse/
- flash-linear-attention RetNet chunkwise：https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/multiscale_retention.py
