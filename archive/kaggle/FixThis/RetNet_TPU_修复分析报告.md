# RetNet 272M · Kaggle TPU v5e-8 训练修复分析报告

日期：2026-08-28 ｜ 最终修复版本：**`retnet-kaggle-train-8k-final3.ipynb`**（取代 final2 / final / fixed）

---

## -2. final2-run1 复盘：torch.utils.checkpoint 在 XLA 上根本不可用（2026-08-28 更新）

final2 上 TPU 实测，scan 崩溃**没有再出现**（前向建图顺利进入第一个 block），但新的报错：

```
AttributeError: module 'torch' has no attribute 'xla'
  File "torch/utils/checkpoint.py", line 1499, in _checkpoint_without_reentrant_generator
    device_module = _get_device_module(device_type)      # device_type = "xla"
  File "torch/utils/checkpoint.py", line 93, in _get_device_module
    device_module = getattr(torch, device)               # getattr(torch, "xla") → 爆炸
```

**根因（两层，都是框架侧）：**

1. `torch.xla` 模块是 **torch ≥ 2.9** 才有的；Kaggle 的 torch（2.7.x）没有。而 `torch.utils.checkpoint` 的非重入路径**无条件**调 `getattr(torch, device_type)`——XLA 上 `use_reentrant=False` 直接炸。换 `use_reentrant=True` 也没用：较新 torch 的重入路径同样调 `_get_device_module`。
2. 更本质的是（pytorch/xla issue **#7024**）：即使能跑，**上游 checkpoint 在 XLA 上根本不省显存**——XLA 的 CSE pass 会把 backward 里的重算图与原前向图合并，静默抵消检查点。torch_xla 官方因此在 `torch_xla/utils/checkpoint.py` 维护了自己的版本：对重算输入打 `xm.optimization_barrier_` 阻断 CSE，并管理 XLA RNG seed（`xm.get_rng_state`/`xm.fork_rng`）。它仅支持 reentrant 模式（issue **#8729**）。

**修复（final3，T1–T2）：** 改用官方 `from torch_xla.utils.checkpoint import checkpoint as _xla_checkpoint`，`RetNetBlock.forward` 中替换调用（默认 reentrant，支持非张量参数与 tuple 输出）。r2.7 分支源码已逐行核对：非张量参数经 `ctx.inputs` 原样保留、tuple 输出 `(y, new_state)` 兼容、backward 内有 optimization barrier + XLA RNG fork。

其余所有改动（无 scan、无 autocast、GLA fp32 门控、手动混合精度、optimizer 分组、逐 block 检查点开关）原样保留。RNG 语义（embed/block dropout）由 torch_xla 版 checkpoint 自动保存/恢复 XLA RNG seed 保证——这也是不能用自写 naive checkpoint 的第二个原因。

**正面信号**：本 run 已证明 §-1 的 scan 修复生效——报错从"scan HLO 签名冲突"推进到了"第一个 block 内的 checkpoint 调用"，两者是**互相独立**的框架缺陷。

---

## -1. run1 复盘：scan 本身才是病根（2026-08-27 晚更新）

`retnet-kaggle-train-8k-final.ipynb` 上 TPU 实测（run1）后仍崩，但报错**位置移动了**：

```
INVALID_ARGUMENT: Call parameter must match argument;
       got parameter 2 shape: f32[4,16,64,64], argument shape: bf16[4,16,64,64].
```

- 老错误：`parameter 0, f32[4,16,64,128]`（carry）；新错误：`parameter 2, f32[4,16,64,64]`（ki 位）。
- 8 个 worker 同一报错，发生在**前向建图**阶段（栈中无 autograd 帧）。
- run1 与 final 逐字节一致，所以问题在 final 版本身。

**证据链：**

1. scan 边界的所有张量（carry、qi、ki、vi、inner_mask、cross/value/chunk_decay）已逐一静态验证为 bf16；`xb.create_placeholder_tensor` 源码也确认保 dtype（`_PT_XLA_TYPE_MAP[torch.bfloat16] = Type.BF16`）。
2. 但 `fn_computation`（torch_xla scan 内部经 AOTAutograd 跟踪 + XLA `LoweringContext` 二次建图得到）把某个形参标成 **f32**，而 While 实参是 bf16。**报错位置随 body 内容变化而移动（param0→param2），证明 f32 是 scan trace 机制内部产生的，与模型代码无关。**
3. 该机制的参数映射脆弱性是 pytorch/xla 官方已知问题：issue **#7839**（"While loop can't yet handle additional tensors……参数顺序不一定与函数参数一致"）、issue **#8783**（"scan forces inputs to be differentiable"）。
4. 没有 TPU/torch_xla 环境无法继续向机制内部钉；每验证一个理论都要消耗一次 TPU 排队（已烧 2 次）。

**结论与决策**：`torch_xla.experimental.scan` 在此栈上不可用。final2 将其从关键路径移除，改用**普通 Python chunk 循环**——与原 scan 写法**逐运算完全等价**（scan 本来就是同一循环的 XLA While 封装），但走标准 XLA 惰性张量 trace 路径，从构造上消除 `fn_computation` 签名边界这整类问题。

**代价与对策**：
- 循环展开 128 chunk × 20 层 → 首步编译时间显著增加（一次性，图 hash 不变后续 step 不重编译）。
- 展开后逐 block 激活显存 ≈ 0.6GB/层，不检查点必然 OOM → final2 在 `train_worker` 里默认开启**逐 block 梯度检查点**（`block.gradient_checkpointing = True`，模型原有 `use_reentrant=False` 支持），显存峰值 ≈ 静态 3.5GB + 输入 1.3GB + logits/loss ~3.9GB + 单 block 重算 ~0.6GB，v5e 16GB HBM 可容。

final2 相对 final 的改动（S1–S5）：
| # | 位置 | 改动 |
|---|---|---|
| S1 | cell-4 import | 删除 `from torch_xla.experimental.scan import scan`，加 `import torch.utils.checkpoint` |
| S2 | `_retnet_scan_chunk` | 整个删除（不再需要） |
| S3 | `_chunkwise_retention` docstring | 更新说明 |
| S4 | `_chunkwise_retention` | scan 调用 → 普通 Python chunk 循环（数学逐运算等价） |
| S5 | `train_worker` | 默认开启逐 block 梯度检查点 |

### 上游定性：pytorch/xla 的已知缺陷，无法从用户侧修复（不提交 issue）

根因**不在本仓库代码**，而在 `torch_xla.experimental.scan` 及其底层的 XLA while-loop 实现。相关上游 issue 均为 **OPEN / 未修复**：

- **[#7839](https://github.com/pytorch/xla/issues/7839)** *"While loop can't yet handle additional tensors"*（2024-08-12 开）→ 至今 **逾 2 年**未修。报告人即 pytorch/xla 维护者，明确指出 body 里的额外张量会被 trace 成额外 parameter，且**参数顺序/类型映射不可靠**——这正是 `fn_computation` 把 ki 形参错标 f32 的机制（"even replacing `external` with `2` will still fail because `2` is turned into a tensor during lowering"）。
- **[#8918](https://github.com/pytorch/xla/issues/8918)** *"Standardize XLA loop APIs"*（2025-04-01）→ 维护者的**官方计划是重写/统一** scan/while-loop/fori-loop，**不做热修**当前 scan。
- **[#8783](https://github.com/pytorch/xla/issues/8783)** *"scan forces inputs to be differentiable"*（2025-03-04，PR #9083 仍未合）→ scan 对所有 carry 强制 `requires_grad=True`。
- **[#9405](https://github.com/pytorch/xla/issues/9405)**（已关）→ 维护者确认 scan 依赖 AOTAutograd、「非 aten op 不支持」；torchprime 靠精细 sharding 才跑通 Llama 3.1 8B。

**结论**：这是框架层陈年老 bug，用户代码无论怎么调 dtype / 加 `.to(bf16)` / 删 autocast / 改 fp32 decay 都**无法根治**——报错位置仅随 body 内容变化而漂移（param0→param2），证明 f32 是 scan trace 机制内部产生的。唯一可靠出路是**绕开 scan**：final2 的普通 chunk 循环即为此目的，与原 scan 写法逐运算等价，但走标准 XLA 惰性张量 trace 路径，从构造上消除 `fn_computation` 签名边界这整类问题。

**决策：不向 pytorch/xla 提交新 issue**。现有 #7839 已覆盖该缺陷，f32/bf16 变体不值得再开一条；本文档即作为记录归档。

---

## 0. TL;DR

本次共修复三个问题，全部通过静态分析 + 源码证据定位，未消耗 TPU 排队：

| # | 问题                                                | 修复                                                |
| - | ------------------------------------------------- | ------------------------------------------------- |
| 1 | cell-6 崩溃 `f32[4,16,64,128] vs bf16[4,16,64,128]` | 删除 `torch.autocast("xla")` 包裹（共 2 处），改手动混合精度      |
| 2 | **GLA 门控实际未启用**（`use_gla=true` 形同虚设）              | decay 门控 `gamma` 改 **fp32 主权重** + decay 计算全程 fp32 |
| 3 | 大量冗余 `.to(torch.bfloat16)`                        | 删除可证明为 no-op 的 cast，仅保留 fp32→bf16 边界 cast         |

checkpoint 兼容性：**state_dict 键完全不变**（366 个键，`blocks.N.msr.gamma` 名字/dtype 含义不变），初始权重照常 `strict=True` 加载。

---

## 1. 原崩溃根因（回顾，已修）

报错是 `scan` 编译出的 XLA `While` 循环里 `call fn_computation` 的实参/形参 dtype 不一致（HLO verifier 硬断言），形状 `[4,16,64,128]` = recurrence carry/state `[B, H, key_dim=64, value_head_dim=128]`。

源码证据（pytorch/xla master）：

- `torch_xla/csrc/autocast_mode.cpp`：TPU autocast 的 `lower_precision_fp` 列表（保持 bf16）含 `matmul/linear/mm/bmm/einsum`，**不含** `layer_norm/group_norm/gelu/silu`；fp32 提升列表只有 `batch_norm/_softmax/log_softmax/pow(Tensor_Scalar)/各 loss`；`promote` 仅 `stack/cat/index_copy`。
- `torch_xla/experimental/scan.py`：`value_and_grad_partitioned` 用 `torch.empty_like(v, dtype=v.dtype)` 跟踪（AOTAutograd 通道），`_scan_impl_flat` 又用 XLA `LoweringContext` 独立重跟踪建 `fn_computation`——两条 trace 通道，autocast 的 cast 注入在两条通道间不一致，在 carry 边界产生 f32/bf16 漂移。

修复：训练循环与 `evaluate` 中的两处 `with torch.autocast(device_type="xla", dtype=torch.bfloat16):` 全部删除。XLA 上它本就冗余（`XLA_USE_BF16=1` + `model.to(bf16)` 已覆盖整图），删除后全图由构造保证 bf16 一致。

> 注：此修复是**必要条件而非充分条件**。run1 实测证明，即使全图 bf16 一致、删掉 autocast，scan 内部仍报 f32/bf16 签名冲突——真正的根因是 scan 本身（见 §-1）。

---

## 2. GLA「没有启用」的真正原因

### 2.1 现象

`config.json` 里 `use_gla=true`，代码里 gamma 也确实注册成了 `nn.Parameter`（`blocks.N.msr.gamma`）。表面上看 GLA 已启用——**但数值上它是死的**。

### 2.2 证据一：checkpoint 的 gamma 是 fp32

```
blocks.0.msr.gamma   F32  [16]   ← 初始权重用 fp32 保存 decay 门控
```

### 2.3 证据二：bf16 把 decay 谱塌缩掉 12/16

RetNet 初始化 `gamma_i = 1 - 2^(-5-i)`，最慢的头到 `1 - 2^-20 ≈ 0.99999905`。bf16 在 1.0 附近的分辨率只有 2⁻⁸ ≈ 0.0039。逐头舍入结果（round-to-nearest-even，已用位运算精确复算）：

| head | gamma (fp32)           | gamma (bf16) | 塌缩? |
| ---- | ---------------------- | ------------ | --- |
| 0    | 0.96875                | 0.96875      |     |
| 1    | 0.984375               | 0.984375     |     |
| 2    | 0.9921875              | 0.9921875    |     |
| 3    | 0.99609375             | 0.99609375   |     |
| 4    | 0.998046875            | **1.0**      | ✗   |
| 5–15 | 0.9990234 … 0.99999905 | **1.0**      | ✗   |

**16 个头里 12 个在 bf16 下变成恰好 1.0——完全没有衰减（无限记忆）**。以 head 8 为例，chunk 级衰减 `gamma**64`：fp32 = 0.9922，bf16 = 1.0。多尺度衰减结构（Multi-Scale Retention 的核心）直接消失。

### 2.4 为什么等于「GLA 未启用」

原代码有两处把门控压成 bf16：

1. `__init__`：`gamma_bf16 = gamma.to(torch.bfloat16)` → `nn.Parameter(gamma_bf16.clone())`；
2. `train_worker`：`model.to(device=device, dtype=torch.bfloat16)` 把从 checkpoint 加载的 fp32 门控又转成 bf16。

后果有两层：

- **前向**：decay 谱塌缩（§2.3），12 个头退化为无衰减线性注意力，与预训练权重的 fp32 行为不一致；
- **训练**：可学习门控的主权重是 bf16，近 1.0 处最小步长 ~0.004，LR=1.1e-4 量级的梯度更新**全部被量化吞掉**——门控注册了 Parameter，却永远学不动。

> 附带说明：`previous_analysis` §6 称「`_chunkwise_retention` 没接收 `g`，GLA 门控是死的」——这个说法**不准确**：`g = g_proj(x)` 在 `MultiScaleRetention.forward` 的 `F.silu(g) * output` 中对所有 mode 都生效（那是输出门控，与 GLA 的 decay 门控是两回事）。真正的 bug 是本节的 bf16 塌缩，与 chunkwise/parallel/recurrent 哪个路径无关。

### 2.5 修复

- `gamma` 以 **fp32** 注册（Parameter 或 buffer 都是 fp32）；
- `train_worker` 在 `model.to(bf16)` 之后**显式恢复** 20 层门控为 fp32 主权重（在创建 optimizer 之前，因此 AdamW 的动量状态也是 fp32）；
- 所有 decay 计算（`inner_mask / cross_decay / value_decay / chunk_decay`，及 parallel/recurrent 路径的同类计算）**全程 fp32**，结果 `.to(bf16)` 后再进 scan / matmul；
- AdamW 参数分组：**1 维参数（门控、norm 权重、bias）不做 weight decay**——否则 0.1 的 weight decay 会把门控拉向 0，杀死长程记忆。

---

## 3. 手动混合精度设计（取代 autocast）

| 组件                                                | dtype                     | 理由                                  |
| ------------------------------------------------- | ------------------------- | ----------------------------------- |
| 投影矩阵 matmul（q/k/v/g/out_proj、FFN、lm_head）         | bf16                      | TPU 主力吞吐，`XLA_USE_BF16=1` + bf16 权重 |
| scan recurrence（carry、xs、body 全部）                 | bf16                      | 显存/吞吐关键路径；入口边界已统一 bf16，HLO 签名一致     |
| **decay 门控 gamma（主权重 + 全部指数运算）**                  | **fp32**                  | §2：bf16 塌缩 decay 谱、量化掉门控学习          |
| **所有 Norm（GrowthLayerNorm、GroupNorm）**            | **fp32 计算 → cast 回 bf16** | 手动 MP 标准做法，norm 的均值/方差对精度敏感         |
| **loss（`logits.float()` + cross_entropy + 加权求和）** | **fp32**                  | 原设计即如此，梯度自动 cast 回 bf16             |
| rotary（sin/cos、inv_freq）                          | bf16                      | 频率值域远离塌缩区，bf16 足够                   |

关键不变式：**进入 `scan` 的 `init`(state) 与 `xs` 全部 bf16、body 只含 matmul/mul/add、输出 bf16** —— HLO 签名由构造保证一致，不依赖任何 Python 层 cast 补救。

---

## 4. 逐处改动清单（cell-4，共 9 处）

| #  | 位置                               | 改动                                                                        |
| -- | -------------------------------- | ------------------------------------------------------------------------- |
| R1 | `_retnet_scan_chunk`             | 删除 9 处 `.to(bf16)`（7 输入 + 2 输出），body 纯 bf16 by construction               |
| R2 | `MultiScaleRetention.__init__`   | gamma 注册改为 fp32（删除 `gamma_bf16`），GLA 分支注释更新                               |
| R3 | `_decay_mask`                    | decay 指数运算 fp32，末尾 `.to(dtype)`                                           |
| R4 | `_parallel_final_state`          | `gamma32` fp32 计算，weights/gamma_pow 结果 cast bf16                          |
| R5 | `_chunkwise_retention`           | 删 q/k/v 冗余 cast；decay 四件套 fp32 计算→边界 cast bf16；xs 元组删 7 处冗余 cast          |
| R6 | `_head_group_norm`               | GroupNorm 改 fp32（`F.group_norm` + fp32 权重），cast 回 bf16                    |
| R7 | `GrowthLayerNorm.forward`        | LayerNorm 与新通道缩放改 fp32，cat 后 cast 回 bf16                                  |
| R8 | `train_worker`                   | `model.to(bf16)` 后恢复 gamma 为 fp32 主权重；AdamW 按 ndim 分组（≥2 维才 weight decay） |
| R9 | `train_worker` 训练循环 + `evaluate` | 删除两处 `torch.autocast("xla")` 包裹                                           |

验证：全 notebook `torch.autocast` 出现次数 = 0；所有 code cell 通过语法编译；state_dict 键集合不变。

---

## 5. 兼容性与行为变化（需要知道）

1. **初始权重加载**：键不变，`from_pretrained(..., strict=True)` 正常。checkpoint 里 fp32 的 gamma 现在**不再被压成 bf16**，慢头衰减真正生效——前几个 step 的 loss 会比 bf16 塌缩版略不同（这是修复生效的预期表现，不是异常）。
2. **断点 resume**：optimizer 改为参数分组（2 组），旧 `training.pt` 的 optimizer state 结构不兼容，**不能**从旧 checkpoint 续训；从头训无碍。model 部分仍可加载。
3. **数值语义**：norm 改 fp32 后输出与纯 bf16 有轻微差异（更准），loss 曲线与旧版不可逐位对比。
4. **recurrent/generate 路径**：保持 bf16 门控cast（仅推理用，影响可忽略），未改动。

---

## 6. 上 TPU 后的验证清单

0. **首步编译时间会明显变长**（循环展开 + 梯度检查点，可能数十分钟级，一次性）。若第一步 `[train] step=1` 打印出来，说明 scan 崩溃已根治。
1. cell-6 跑通首个 `mark_step`（不再报 `f32 vs bf16`）→ 修复 1 生效。
2. `[train] step=1` loss 与旧版量级一致（~ln(20000)≈9.9 附近起步）。
3. 探针（可选，加在 `model(...)` 后）：


   ```python
   if step == 1 and xm.is_master_ordinal():
       g = model.blocks[0].msr.gamma
       print("gate dtype:", g.dtype, "min:", g.min().item(), "max:", g.max().item())
   ```
   预期 `torch.float32`，且 max < 1.0（16 个头谱系完整）→ 修复 2 生效。
4. 训几百步后 `gamma` 与初始值的差应非零（门控真的在学）。
