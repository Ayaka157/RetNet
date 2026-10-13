# 11b TileKernels 可用性评估

> 编号 11b：初稿与 [`11_R1思维链与长上下文.md`](./11_R1思维链与长上下文.md) 同期（2026-08-26），
> 但正文引用了 TileKernels 仓库 2026-09-30 才发布的 News，实际定稿于 2026-10-01。
> 评估对象：[deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels)（main 分支）·
> 试用品：SCNet 国家超算互联网「AI计算-异构加速卡BW-1」图示试用品
> 结论口径：本项目代码的每条判断都给 `文件:行`；外部项目的每条判断都给 URL。带「背景」标记的段落是上下文补充，不是独立核验所得。

## 结论

对图示试用品：**用不上，而且不是配置问题**。试用品所在的卡是昆仑芯 P800（XPU），跑的是 XPyTorch（Python 3.10 + torch 2.5.1）；TileKernels 只有 NVIDIA CUDA 与华为昇腾 NPU 两个运行时后端，按 `/dev/davinci_manager` 存在与否二选一，第三方卡的路径不存在（`tile_kernels/config.py`）。版本侧也对不上：库要求 `torch>=2.13`、`python>=3.12`、CUDA toolkit ≥13.1、SM90 或 SM100，试用品一个都不满足。即使这些都满足，272M 模型含梯度与优化器状态约 4.4GB，对试用品单机 512GB HBM 是千分之二量级，而 TileKernels 的算子全部长在「大模型 + 低精度 + MoE + 显存带宽受限」这个前提上——租这台机器的理由是换卡型和换精度做实验，不是装这个库。

对本项目：**库本体用不上**，因为它的目录里没有 retention / 线性注意力 / chunkwise 衰减这一族算子，而我们唯一被测量证实值得写 kernel 的地方正好就在这里。**TileLang 这个 DSL 本身值得留意**（本机 SM86 在其 CUDA 后端覆盖范围内，PyPI 有 Windows wheel，依赖里不含 triton，与现有环境不冲突），但优先级排在两件事之后：先把 chunkwise 的 kernel 启动数打下来，先补上生产尺度的数值等价测试。在换 kernel 之前，还有一条免费回归线现在没用起来——同一个 state dict 同时装进 `retnet/model.py` 与 `retnet_hf/modeling_retnet.py`，parallel 与 chunkwise 两种模式的输出差异都是 0.0000e+00。

### 一页速览

| 问题 | 答案 | 决定性依据 |
| --- | --- | --- |
| 试用品能不能跑这个库 | 不能 | 只有 `cuda` / `npu` 两个后端，P800 是 XPU |
| 是不是装个旧版本就能跑 | 不能 | 旧版本同样只支持这两个后端；且 XPU 不执行 CUDA 二进制 |
| 是否值得为它调环境 | 不值得 | 量化 / MoE / Engram / MHC 四族我们全部为 0，仅 RoPE 形态相近 |
| 对我们项目是否可用 | 库不可用，DSL 可用 | 无 retention/chunkwise 算子族；TileLang 的 CUDA 后端覆盖 SM70–SM120 |
| 我们的瓶颈在哪 | 启动数与带宽，不在 FLOPs | 4K 前向 626.8 ms 中 elementwise+copy ≈ 55%，GEMM ≈ 41%，大 GEMM 已达 92% peak |
| 现在的第一优先级 | 生产尺度等价性 + 回退路径 | 生产尺度 parallel vs chunkwise `max\|diff\|=8.39`；项目没有 `.git` |

---

## 1. TileKernels 是什么（已核验）

仓库为 MIT 许可，自述为「A kernel library written in tilelang」；中文 README 称「数十个深度优化算子」，并说明已用于 DeepSeek 内部训练与推理。按 git tree 逐个目录核验，内容分布如下。

| 目录 | 包含的算子 | 说明 |
| --- | --- | --- |
| `quant/` | per-token / per-block / per-block-lossless / per-channel 的 FP8、FP4 cast 与 cast_back；融合 SwiGLU 前向与反向；norm 前向与反向；`batched_transpose_weight_sf` | 全部围绕低精度量化与缩放因子布局 |
| `moe/` | `topk_gate` 前向 / 反向、`normalize_weight`、`sqrt_softplus` | MoE 路由侧算子 |
| `engram/` | gate 前向 / 反向、hash、sinkhorn、fused_weight、`grad_w_reduce` | Engram 结构专用 |
| `mhc/` | expand、head_compute_mix、multilayer_recompute、norm_fn、post、pre_apply_mix、pre_big_fuse、pre_split_mixes、sinkhorn | 流形超连接（MHC） |
| `transform/` | rope | 只有 RoPE 一个文件族 |
| `rand/` | randn | 随机数 |
| `modeling/` | engram autograd 封装 | 只覆盖 engram |
| `torch/` | 参考实现（cast、swiglu、rope、norm、topk、moe、mhc、engram、reduce_fused） | 对照用 |
| `testing/`、`tests/` | 基准与单测 | — |

### 1.1 关键结构事实：每个算子只有两个后端

每个 kernel 都以「一算子三文件」的形式存在：`<name>_kernel.py` 是 TileLang 程序，`<name>_cuda.py` 是 NVIDIA 实现，`<name>_asc.py` 是华为昇腾实现。git tree 里**不存在第三个后端目录**。`tile_kernels/rand/` 下只有 `randn_asc.py` 与 `randn_kernel.py`，`tile_kernels/moe/sqrt_softplus.py` 是单文件实现，属于个别例外。

后端选择逻辑写死在 `tile_kernels/config.py`：

```python
@functools.lru_cache(maxsize=1)
def is_ascend() -> bool:
    return os.path.exists('/dev/davinci_manager')

@functools.lru_cache(maxsize=1)
def get_device() -> str:
    return 'npu' if is_ascend() else 'cuda'
```

也就是说，运行期只有 NVIDIA CUDA 与华为昇腾 NPU（`torch.npu`）两条路：设备节点不存在就一律落到 `cuda`。没有 XPU、ROCm 或其它第三方分支。

### 1.2 依赖与硬件门槛

`pyproject.toml` 声明 `dependencies = ["torch>=2.13", "tilelang>=0.1.15"]`、`requires-python = ">=3.12"`。README 的 Requirements 一节给出更细的门槛：

| 项 | 要求 |
| --- | --- |
| Python | ≥ 3.12 |
| PyTorch | ≥ 2.13 |
| TileLang | ≥ 0.1.15 |
| CUDA 后端 | NVIDIA SM90 或 SM100，且 CUDA Toolkit ≥ 13.1 |
| Ascend 后端 | Ascend 950 NPU，且 CANN ≥ 9.2.0 |

README 的 News 记录 `[2026-09-30]` 加入 Ascend 后端，运行时自动选择，两端 Python API 相同。

### 1.3 目录里没有的东西（本节是判断的直接依据）

按 git tree 核验，以下四类在仓库中不存在：

1. 没有 attention-flash 类算子；
2. 没有 linear attention / retention / chunkwise-decay 类算子；
3. 没有面向 bf16 的 LayerNorm / GroupNorm（带量化）算子——`quant/` 下的 norm 前向反向服务于量化缩放因子，不是通用归一化；
4. 没有以 API 形式暴露的通用 elementwise 融合设施。

第 2 条是本次评估的核心：**TileKernels 唯一缺的那一族，恰好是我们唯一值得写 kernel 的地方。**

---

## 2. 图示试用品是什么（已核验；产品参数部分标注为背景）

> 背景：以下两小节（2.1、2.2）来自商品页长图与厂商文档，属于采购侧参数，不是我们在机器上实测所得。2.3 起是平台识别与后果推演。

### 2.1 商品与配置

商品为 SCNet 国家超算互联网「AI计算-异构加速卡BW-1」，店铺为曙光智算旗舰店。商品页长图给出的配置：

| 项 | 内容 |
| --- | --- |
| 单节点加速卡 | 8 张异构加速卡 BW，每张 64GB 显存 |
| 面向场景 | 人工智能 / AI4S / 科学计算 |
| 交付形态 | K8S 容器 |
| 软件能力 | 支持自定义安装应用与容器、自定义镜像、公网网盘、本地直连、VSCode + PyCharm 连接、TensorBoard、性能监控、计费 |
| 共享存储 | 500G |

### 2.2 商品页的性能对照口径

性能附录把「异构加速卡BW（64GB 显存）」与「NVIDIA A800（80GB 显存）」并排比较，每个算例标题都写明这一对测试卡型。相对性能区间为：

| 算例类别 | 相对 A800 的性能区间 |
| --- | --- |
| DeepSeek-R1-Distill-Llama-70B / Llama3.1-70B-Dense / Meta-Llama-3.1-8B / Qwen2.5-72B / YOLOv11 / BevFormer / ControlNet / mineru / DBNet / tiGCN | 100.5% – 119% |
| VASP / GROMACS / LAMMPS | 104% – 135% |

商品页 URL 携带 `goodsResourceType=DCU` 与 `id=128C-512G-8*BW-K8S-1_TRIAL_SHARE`。`DCU` 这个字段与其余全部证据不一致，作为待澄清项记在第 6 节。

### 2.3 平台识别：这颗 BW 是昆仑芯 P800（XPU）

厂商侧的一手材料是 LoongForge 的《昆仑芯 P800 安装教程》。该教程构建 XPU 镜像时锁定的版本组合为：操作系统 Ubuntu 20.04、Python 3.10、PyTorch 2.5.1、CUDA 11.7；安装的包是从 `baidu-kunlun-public.su.bcebos.com` 取的 `xpytorch-cp310-torch251-ubuntu2004-x64.run`；工具链里用 `xpu-smi`，Dockerfile 名为 `Dockerfile.xpu`。

合起来可以确认：P800 跑的是**类 CUDA 的 XPyTorch 栈**，不是 CUDA 工具链。`xpu-smi`、`xpytorch` 这类命名与「8 张 64GB 异构加速卡 BW + A800 比照」的组合，指向百度昆仑芯 P800。

### 2.4 决定性后果：两端的门槛同时不成立

| 检查项 | TileKernels 要求 | 试用品实际 | 结果 |
| --- | --- | --- | --- |
| 运行时后端 | `cuda` 或 `npu` | XPU（XPyTorch） | 两端都不匹配。`cuda` 分支要求 `/dev/davinci_manager` 不存在且设备为 NVIDIA；Ascend 分支要求 Ascend 950 NPU + CANN 9.2，P800 两者都没有 |
| Python | ≥ 3.12 | 3.10 | 不满足 |
| PyTorch | ≥ 2.13 | 2.5.1 | 不满足 |
| CUDA Toolkit | ≥ 13.1 | 容器内是 CUDA 11.7 的类 CUDA 栈 | 不满足 |
| 计算能力 | SM90 或 SM100 | 非 NVIDIA | 不满足 |

即便版本全部对齐，前面还有一条更硬的约束：**面向 CUDA 目标编译出的 kernel 二进制不能在 XPU 上执行。** 这不是打补丁能绕过的，需要先把 kernel 移植到 XPU 后端（即 TileLang 的 XPU 后端尚不存在），移植本身才是工作量的主体。

### 2.5 机器规模与模型规模的错配

试用品单机为 8 卡 × 64GB = 512GB HBM，128 核 CPU，512GB 内存。我们把 272M 模型放进这个尺度里看：

| 量 | 数值 | 占单卡 64GB 的比例 |
| --- | --- | --- |
| 参数 | 272.3M（[checkpoints/retnet_272m_rp/config.json](../checkpoints/retnet_272m_rp/config.json)，`d_model 1024` / 20 层 / 16 头） | — |
| fp32 权重 | 约 1.1GB | 约 1.7% |
| bf16 权重 | 约 0.55GB | 约 0.9% |
| fp32 权重 + 梯度 + AdamW 状态 | 约 4.4GB | 约 7% |

按「整个模型」对「整个试用品单机」（8 × 64GB）算，占用在千分之二量级。这个错配是判断 A 的第三条支撑，也说明租它的合理理由是容量与卡型（例如 4K 序列长度在 8GB 卡上放不下这件事），不是装一个为大模型低精度训练的算子库。

---

## 3. 我们项目现在的实际情况（已实测）

### 3.1 两台机器

| 项 | 本机（A） | 远程机（B） |
| --- | --- | --- |
| 系统 | Windows | Windows（`192.168.1.100`） |
| GPU | RTX 3070 8GB | RTX 4070 Laptop 8GB |
| 计算能力 / SM 数 | SM86 / 46 SMs | SM89 / 未记录 |
| Python | 3.13.7 | 3.11.15 |
| PyTorch | 2.10.0+cu128 | 2.11.0（[docs/10_远程部署笔记.md:9-14](10_远程部署笔记.md#L9-L14)） |
| CUDA | 12.8 | 12.8 |
| 驱动 | 596.36 | — |
| triton | 3.7.1（已装） | — |
| tilelang | 未安装 | 未安装 |

4K / 8K 的实测都跑在 B 机上。

### 3.2 模型配置

| 模型线 | 配置 | 来源 |
| --- | --- | --- |
| 100M | d_model 768 / 12 层 / 12 头 / value_dim 1536 / ffn_dim 1536 | [RUNBOOK.md:13-14](../RUNBOOK.md#L13-L14)、[checkpoints/retnet_100m_pretrain_v4_hf/config.json](../checkpoints/retnet_100m_pretrain_v4_hf/config.json) |
| 272M（当前主线） | d_model 1024 / 20 层 / 16 头 / value_dim 2048 / ffn_dim 2048 / `use_gla true` / `use_growth_ln true` / mode `chunkwise` / chunk_size 64 | [checkpoints/retnet_272m_rp/config.json](../checkpoints/retnet_272m_rp/config.json) |

模型有三份实现，测试只覆盖其中一份：

| 实现 | 定位 | 测试覆盖 |
| --- | --- | --- |
| [retnet/model.py](../retnet/model.py) | 手写版 | 唯一有测试的一份（[tests/test_model.py](../tests/test_model.py) 只 `import retnet.model`） |
| [retnet_hf/modeling_retnet.py](../retnet_hf/modeling_retnet.py) | HF 封装，跑过 143M / 272M | 无 |
| [retnet_hf_fp16/modeling_retnet.py](../retnet_hf_fp16/modeling_retnet.py) | 数值加固分支，v11 使用 | 无 |

项目没有 `.git`，也就是说换 kernel 之后没有版本回退路径。

### 3.3 retention 的三条实现路径

三条路径都在 [retnet/model.py](../retnet/model.py)：

| 路径 | 位置 | 实现形态 | 用途 |
| --- | --- | --- | --- |
| parallel | [retnet/model.py:169-176](../retnet/model.py#L169-L176) | 物化 `[B,H,L,L]` 分数矩阵 + fp32 `[H,L,L]` 衰减掩码 + 3 遍 row-norm | 训练（历史） |
| recurrent | [retnet/model.py:190-207](../retnet/model.py#L190-L207) | Python 逐 time step 循环 | 只用于解码 |
| chunkwise | [retnet/model.py:209-252](../retnet/model.py#L209-L252) | Python 逐 chunk 循环（L/64 次），每 chunk 约 4 个小 GEMM + 约 4 个 elementwise，末尾 `torch.cat`（[:251](../retnet/model.py#L251)） | 当前训练主线 |

### 3.4 自定义 kernel 的现状：一个都没有

对 `retnet/`、`retnet_hf/`、`retnet_hf_fp16/`、`scripts/`、`tests/` 搜索 `triton|scaled_dot_product|load_inline|cpp_extension|tilelang`，**零命中**。`torch.compile` 存在（[train.py:331-340](../train.py#L331-L340)），但三份任务配置都写了 `"compile": false`（`tasks/v10_272m.json:18`、`tasks/v11_bf16_4k.json:18`、`tasks/v11_fp16.json:18`），并且 [run_task.py:114-115](../run_task.py#L114-L115) 会在 `compile` 为 false 时补 `--no-compile`。结论：v10 / v11 的训练是 eager 执行。

### 3.5 实测前向 profile：272M chunkwise bf16，B=4，RTX 3070，torch.profiler，仅前向

| 类别 | L=512 | L=4096 | 4K 占比 | 4K 启动数 |
| --- | --- | --- | --- | --- |
| 总计 | 81.3 ms | 626.8 ms | 100% | — |
| 常规 bf16 tensor-core GEMM | — | 206.4 ms | 32.9% | 120 |
| `cutlass_80_wmma_bf16_32x32_64x1`（每 chunk 的 q@k 与 score@v） | — | 29.5 ms | 4.7% | 5120 |
| bf16 copy | — | 63.8 ms | 10.2% | 5281 |
| `elementwise_kernel ILi128`（mask 乘、decay、加、除） | — | 144.0 ms | 23.0% | >13000 |
| `group_norm` RowwiseMoments | — | 16.4 ms | 2.6% | — |
| `CatArrayBatchedCopy` | — | 14.6 ms | 2.3% | — |
| 其余（未单列，含大 GEMM epilogue 等） | — | 152.1 ms | 24.3% | — |

两个可直接引用的比率：elementwise + copy ≈ 55% 的 GPU kernel 时间，GEMM ≈ 41%。而 retention 机制整体只占前向 FLOPs 的约 3%（实测 562.6 MFLOP/token，稠密基线 2N = 544.6）。**瓶颈是启动数与小 tile，不是 FLOPs。**

### 3.6 效率检查：大 GEMM 已经贴顶

`lm_head` 的 GEMM `[2048,1024]@[1024,20000]` 跑到 36.7 TFLOP/s，约为 3070 稠密 bf16 峰值的 92%。大 GEMM 已经没有余量，换 kernel 换不出收益。

### 3.7 实测吞吐

| 配置 | 步时 | 吞吐 | 显存 |
| --- | --- | --- | --- |
| 272M chunkwise bf16 B=4 L=512，开梯度检查点 | 0.638 s/it | 3212 tok/s | 5.46 GB |
| 272M chunkwise bf16 B=4 L=4096 | 21.9 s/it | 749 tok/s | 峰值 8.72 GB（B=4 在 4K 下已超出 8GB；`tasks/v11_bf16_4k.json` 正是 B=4） |
| 100M parallel bf16 `compile=inductor` B=10 L=512 | — | 约 29.0k tok/s | — |
| 272M chunkwise，chunk_size 64 → 128 → 256 | 0.664 → 0.645 → 0.643 s/it | — | — |

第一行与归档日志一致：[logs/rp_272m.log](../logs/rp_272m.log) 记录 `train_steps_per_second: 1.498`（约 3.07k tok/s）、`eval_loss 2.9946`。chunk_size 三档几乎不动步时，说明 chunk 大小不是杠杆。

### 3.8 数值发现（本次实测，仓库内无记录）

把 `checkpoints/retnet_100m_rp_v2/model.pt`（100.37M，d768/L12/H12）载入同一个模型，权重完全相同，B=1 L=512 fp32 下比较 parallel 与 chunkwise 的 logits：

| 指标 | 值 |
| --- | --- |
| `max\|diff\|` | 8.39 |
| `mean\|diff\|` | 0.314 |
| 相对误差 p50 | 3.45e-2 |
| 相对误差 p90 | 2.68e-1 |
| 交叉熵 | parallel 8.1806 vs chunkwise 8.1779 |

仓库自带的等价性测试断言 `< 1e-4`（[tests/test_model.py:24-45](../tests/test_model.py#L24-L45)），但只在 d_model=64、L=16、CPU、fp32 这个玩具尺度成立，**不能外推到生产尺度**。[docs/08_chunkwise_growth_research.md:82](08_chunkwise_growth_research.md#L82) 记录了一个「~8.75% 中位相对误差」，但生成该数字的脚本已经找不到。

### 3.9 一条被闲置的免费回归线

同一个 state dict 同时装进 [retnet/model.py](../retnet/model.py) 与 [retnet_hf/modeling_retnet.py](../retnet_hf/modeling_retnet.py)，在 parallel 与 chunkwise 两种模式下 B=1 L=512 fp32 的输出差异都是 `0.0000e+00`，加载报 0 missing / 0 unexpected keys。这条对照现在没有任何测试或脚本在用——换 kernel 之前，它是最便宜的正确性护栏。

### 3.10 fp16 是死路

`logs/precision_test/` 下的两组 fp16 种子在第 20、40 步就崩了，bf16 对照组正常：

| 运行 | step 20 loss / grad_norm | step 40 loss / grad_norm |
| --- | --- | --- |
| `fp16_seed1234` | 0.0 / NaN | 0.0 / NaN |
| `fp16_seed1235` | 0.0 / NaN | 0.0 / NaN |
| `bf16_seed1234` | 3.0547 / 6.82 | 3.0905 / 6.18 |
| `bf16_seed1235` | 2.7209 / 7.83 | 3.4172 / 8.56 |

[docs/10_远程部署笔记.md:39](10_远程部署笔记.md#L39) 的结论是 fp16「放弃」。而 v11 的 fp16 配置（`tasks/v11_fp16.json`）指向的正是这条已知会爆的路径。

### 3.11 项目自己已经点出了正确的工具

[docs/08_chunkwise_growth_research.md:17](08_chunkwise_growth_research.md#L17) 把 flash-linear-attention 的 Triton `MultiScaleRetention` 列为 chunkwise 路径的直接参考实现（[:49](08_chunkwise_growth_research.md#L49) 第 ① 条重复了这一建议）；[docs/scaling_study.md:93-95](scaling_study.md#L93-L95) 把「Python 逐 chunk 循环、compile 后仍慢」列为已知问题。TileLang 在仓库里没有任何提及。

---

## 4. 算子映射：TileKernels 目录 → 我们的代码

| TileKernels 算子族 | 我们代码里的对应物 | 判定 |
| --- | --- | --- |
| `quant/` per-token / per-block / per-channel FP8、FP4 cast + cast_back + 融合 SwiGLU + norm fwd/bwd | 全项目零量化。在 `retnet/`、`retnet_hf/`、`retnet_hf_fp16/`、`train.py`、`play.py` 上搜 `quantiz\|int8\|fp8\|bitsandbytes\|awq\|gptq`，只命中 [retnet/packing.py:223](../retnet/packing.py#L223) 与 [:261](../retnet/packing.py#L261) 的 `np.uint8` 权重码数组；磁盘上的权重全是 F32 | 无对应物，用不上。要谈量化，先回答的是「100M / 272M 这种小模型要不要量化」，而不是「选哪个算子库」 |
| `moe/` topk_gate fwd/bwd、normalize_weight | 无 MoE。FFN 是 Linear → GELU → Linear（[retnet/model.py:302-306](../retnet/model.py#L302-L306)） | 用不上 |
| `engram/` gate fwd/bwd、hash、sinkhorn、grad_w_reduce | 无 Engram、无 Sinkhorn | 用不上 |
| `mhc/`（流形超连接）全族 | 无 | 用不上 |
| `transform/` rope（RoPE） | 我们用的是 xPos 实值二维旋转：[retnet/model.py:54-59](../retnet/model.py#L54-L59) 的 `rotate_every_two` + [:62-69](../retnet/model.py#L62-L69) 的 `theta_shift`；sin/cos 表在每层内部重建（[`_rotations` retnet/model.py:131-134](../retnet/model.py#L131-L134)） | 算子形态同构（都是成对旋转 + cos/sin 调制），理论上可复用其 CUDA/ASC kernel 并换成我们的频率表。但它是为 LLM 的 RoPE 频率调度写的，且我们的表是 fp32 构建、每层重算，替换后要自己验证。收益有限 |
| `quant/swiglu_forward`（融合 SwiGLU） | FFN 是 GELU，不是 SwiGLU；`use_swiglu` 字段存在但两份 config 都是 `false`（[checkpoints/retnet_272m_rp/config.json](../checkpoints/retnet_272m_rp/config.json)），实现只在 HF 版里（[retnet_hf/modeling_retnet.py:28](../retnet_hf/modeling_retnet.py#L28)、[:268](../retnet_hf/modeling_retnet.py#L268)）。retention 输出上的门控 `F.silu(g) * y`（[retnet/model.py:283](../retnet/model.py#L283)）是 SwiGLU 的一半形态 | 只能借它的融合写法，不能直接调用 |
| `rand/` randn | torch 自带 | 无关 |

### 4.1 缺失的那一块

**TileKernels 的目录里没有 retention / 线性注意力 / chunkwise 衰减这一族算子——而它恰好是我们唯一真正值得写 kernel 的地方。**

从 git tree 看，`tile_kernels/` 下与序列建模相关的目录只有 `transform/`（仅 rope）。`quant/`、`moe/`、`engram/`、`mhc/` 四族覆盖的是 DeepSeek 自家模型结构，与 RetNet 的 retention 机制没有交集。

---

## 5. 判断

### 5.1 判断 A：对图示试用品，用不上，且不是「配置问题」

理由分三层，任一层单独成立就足够：

1. **后端不存在。** P800 是 XPU（XPyTorch，Python 3.10 + torch 2.5.1）。TileKernels 只有 CUDA 与 Ascend NPU 两个后端，判定逻辑是 `/dev/davinci_manager` 是否存在（`tile_kernels/config.py`）。CUDA kernel 二进制不能在 XPU 上执行；Ascend 后端要求 Ascend 950 NPU + CANN 9.2，P800 两者都不具备。这一层不是改配置能解决的，需要先做后端移植。
2. **版本对不上。** `torch>=2.13`、`python>=3.12`、CUDA toolkit ≥13.1、SM90 或 SM100——试用品一个都不满足。
3. **即使能用也没有意义。** 我们是 272M、fp32 磁盘权重、无量化、无 MoE 的模型，权重加梯度加优化器状态合计约 4.4GB，占单卡 64GB 的约 7%；与 TileKernels 假设的「大模型 + 低精度 + MoE + 显存带宽受限」不在同一量级。

适合这台机器的叙述是：8 × 64GB 的容量对 272M 模型是约三个数量级的错配。租它的合理用途是换卡型、换精度做实验（例如 4K 序列长度在 8GB 卡上放不下这件事），不是装这个库。

### 5.2 判断 B：对我们项目，库本体用不上，但 TileLang 这个 DSL 值得留意

优化目标需要从「找库」改成「写 kernel / 借 kernel」。优先级如下：

1. **先把 chunkwise retention 的 launch 数打下来。** 这是唯一被实测证实的大头：4K 前向 626.8 ms 里包含 5120 次 32×32 wmma、>13000 次 elementwise、5281 次 copy。现成路线是项目自己已经点名的 flash-linear-attention Triton `MultiScaleRetention`（[docs/08_chunkwise_growth_research.md:17](08_chunkwise_growth_research.md#L17)），本机 triton 3.7.1 已在。TileLang 是备选：PyPI 上 `tilelang 0.1.15` 提供 `cp39-abi3-win_amd64` wheel、`requires_python >=3.10`，CUDA 后端声明覆盖 SM70–SM120（本机 SM86 在内），`requires_dist` 里没有 triton，与现有环境不冲突。
2. **收掉零散 kernel。** xPos 旋转 + `k * scale` 折叠、per-head GroupNorm 前后的 transpose/reshape 布局拷贝、decay mask 的构建与三遍 row-norm、chunk 输出的 `torch.cat`（[retnet/model.py:251](../retnet/model.py#L251)）。这些是带宽与启动数的工作，不是数学工作。
3. **不要动大 GEMM。** `lm_head` 已经跑到 92% peak。
4. **前置条件比 kernel 本身更要紧。**
   - (a) 上生产规模的等价性测试。现在的 `1e-4` 只在小玩具尺度成立，生产尺度实测 `max\|diff\| = 8.39`。
   - (b) 项目没有 `.git`，换 kernel 没有回退路径。
   - (c) fp16 已知会爆，`tasks/v11_fp16.json` 需要先修或先放弃。

换 kernel 之前，先把「同一 state dict 装进两个实现得到 0 差异」这条免费回归线用起来（现状：没有在用）。

---

## 6. 需要澄清的一件事

商品页 URL 写着 `goodsResourceType=DCU`，但其余全部证据（BW 64GB、8 张/节点、与 A800 比照、昆仑芯 P800 的 XPyTorch 栈）都指向昆仑芯 P800 / XPU。这条不一致记为待确认项：

**下单前应向商家确认卡的准确型号，以及镜像里预装的 torch 版本（是 xpytorch 2.5.1 还是别的）。** 这个答案决定了「能不能用 Triton / TileLang 写自定义算子」会不会从「不能」变成「要先移植」——如果镜像里是标准 `torch` + XPU 设备后端且带 Triton 支持，第 5.2 节第 1 条的路径可以在试用品上试；如果只有 XPyTorch 2.5.1，那这条路在试用品上不存在。

---

## 来源

外部（均为 2026-08-26 核验）：

- TileKernels 仓库主页：https://github.com/deepseek-ai/TileKernels
- README（raw）：https://raw.githubusercontent.com/deepseek-ai/TileKernels/main/README.md
- pyproject.toml（raw）：https://raw.githubusercontent.com/deepseek-ai/TileKernels/main/pyproject.toml
- config.py（raw）：https://raw.githubusercontent.com/deepseek-ai/TileKernels/main/tile_kernels/config.py
- git tree（用于核对目录与 per-backend 文件分布）：https://api.github.com/repos/deepseek-ai/TileKernels/git/trees/main?recursive=1
- TileLang targets 文档（target 种类与后端支持级别）：https://tilelang.com/get_started/targets.html
- TileLang PyPI 元数据（0.1.15 wheel 列表、`requires_python`、`requires_dist`、后端表）：https://pypi.org/pypi/tilelang/json
- 昆仑芯 P800 安装教程（LoongForge）：https://loongforge.readthedocs.io/zh-cn/v0.1.0/_sources/kunlun_tutorial/install_p800.md

本仓库（判断题直接引用）：

- [retnet/model.py](../retnet/model.py)（:54-59、:62-69、:131-134、:169-176、:190-207、:209-252、:251、:283、:302-306）
- [tests/test_model.py:24-45](../tests/test_model.py#L24-L45)
- [retnet/packing.py:223](../retnet/packing.py#L223)、[:261](../retnet/packing.py#L261)
- [retnet_hf/modeling_retnet.py:28](../retnet_hf/modeling_retnet.py#L28)、[:268](../retnet_hf/modeling_retnet.py#L268)
- [train.py:331-340](../train.py#L331-L340)、[run_task.py:114-115](../run_task.py#L114-L115)
- [RUNBOOK.md:13-14](../RUNBOOK.md#L13-L14)
- [docs/08_chunkwise_growth_research.md:17](08_chunkwise_growth_research.md#L17)、[:49](08_chunkwise_growth_research.md#L49)、[:82](08_chunkwise_growth_research.md#L82)
- [docs/10_远程部署笔记.md:9-14](10_远程部署笔记.md#L9-L14)、[:39](10_远程部署笔记.md#L39)
- [docs/scaling_study.md:93-95](scaling_study.md#L93-L95)
- [checkpoints/retnet_100m_pretrain_v4_hf/config.json](../checkpoints/retnet_100m_pretrain_v4_hf/config.json)、[checkpoints/retnet_272m_rp/config.json](../checkpoints/retnet_272m_rp/config.json)
- `tasks/v10_272m.json`、`tasks/v11_bf16_4k.json`、`tasks/v11_fp16.json`（`compile: false`）
- [logs/rp_272m.log](../logs/rp_272m.log)、`logs/precision_test/*/checkpoint-40/trainer_state.json`
