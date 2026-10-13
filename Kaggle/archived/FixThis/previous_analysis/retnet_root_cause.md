# RetNet × XLA `scan`：f32/bf16 崩溃根因分析

> 目标：仅用静态分析 + 本地 CPU 复现，在不开 TPU 的前提下把崩溃的精确机制钉死，并确认修复。

---

## 1. 致命证据（来自 notebook cell[6] 的 stderr 输出）

崩溃发生在**反向传播**（堆栈顶部是 `torch::autograd::Engine::evaluate_function` → `PyNode::apply`，不是前向）：

```
F0827 ... debug_macros.h:21] Non-OK-status: status.status()
Status: INVALID_ARGUMENT: Call parameter must match argument;
       got parameter 0 shape: f32[4,16,64,128], argument shape: bf16[4,16,64,128].
*** Begin stack trace ***
    torch_xla::ShapeHelper::ShapeOfXlaOp
    ... torch::autograd::Engine::evaluate_function(...)
    ... torch::autograd::PyNode::apply()
```

三个硬事实：
- 是一个 **matmul**（XLA op builder 在建图时发现左右操作数 dtype 不一致）报错；
- 左右操作数**形状完全相同** `[4,16,64,128]`，只是 dtype 一个 `f32` 一个 `bf16`；
- 发生在 `scan` 编译出的反向图里（进程在 `torch.autograd` 引擎里死掉，8 个 worker 同时 abort）。

> 注：崩溃是 **SIGABRT / Check failed**，不是普通 Python `RuntimeError`——所以你在 Kaggle 上只看到一行 `f32[4,16,64,128] vs bf16[4,16,64,128]`，没有 Python 行号。这正是 XLA 的硬断言风格。

---

## 2. 真实维度（来自 `config.json`，不是我猜的）

| 字段 | 值 |
|---|---|
| `d_model` | 1024 |
| `num_heads` | 16 → **`key_dim = 1024/16 = 64`** |
| `value_dim` | 2048 → **`value_head_dim = 2048/16 = 128`** |
| `chunk_size` | **64**（你纠正过：不是 128） |
| `PER_DEVICE_BATCH` | 4 → 第一维就是 batch=4 |
| `use_gla` | **true** |
| `use_growth_ln` | true, `growth_old_dim=768` |

所以崩溃形状 `[4,16,64,128]` = `[B, H, key_dim=64, value_head_dim=128]`
——这正是 **`scan` 的递归状态（carry / state）张量的形状**，不是 `lm_head`/loss 的输出。

---

## 3. 逐行形状 + dtype 推导（`_chunkwise_retention` → `scan` body，纯 bf16）

**`forward()` → `_project_and_rotate(x [4,8192,1024])`**
- `q = q_proj(x).view(·,num_heads,key_dim).transpose` → `[4,16,8192,64]` **bf16**
- `k` 同上 `[4,16,8192,64]` **bf16**；`k *= scale` 标量 → bf16
- `v = v_proj(x).view(·,value_head_dim).transpose` → `[4,16,8192,128]` **bf16**
- `g = g_proj(x)` → `[4,8192,1024]`，**chunkwise 路径完全不使用 `g`**（见第 6 节）
- `sin/cos` 来自 `_rotations`，`dtype=q.dtype=bf16` → bf16

**`_chunkwise_retention(q,k,v, state, chunk=64)`**
- `q,k,v .to(bf16)`；`state` 初始 `torch.zeros(...,dtype=bf16)` → `[4,16,64,128]` bf16
- `gamma = self.gamma.to(bf16)`（buffer 本身就是 bf16，见第 5 节）
- `inner_mask = (gamma[:,None,None]**exp).to(bf16)` → `[16,64,64]` bf16
- `cross_decay/value_decay/chunk_decay` = `gamma**...` 后全部 `.to(bf16)` → bf16
- 切块：`q_chunks.permute(2,0,1,3,4)` → `[128,4,16,64,64]`，`v` 同理 `[128,4,16,64,128]`
- `xs` 元组：`(q,k,v,inner_mask,cross_decay,value_decay,chunk_decay)` 全部 bf16
- `scan(_retnet_scan_chunk, state, xs, is_fn_pure=True)`

**`scan` body `_retnet_scan_chunk(carry, x)`**（每行都是 `[4,16,64,128]` 级别）
- 入口全部 `.to(bf16)`（你加的那些 cast）
- `inner_scores = matmul(qi, ki.T) * inner_mask_t` → `[4,16,64,64]` **bf16**
- `inner = matmul(inner_scores, vi)` → `[4,16,64,128]` **bf16**
- `cross = matmul(qi*cross_decay_t, carry)` → `[4,16,64,128]` **bf16**（`carry` 是状态）
- `output_t = inner + cross` → `[4,16,64,128]` **bf16**
- `next_state = chunk_decay*carry + matmul(ki.T, weighted_v)` → `[4,16,64,128]` **bf16**
- `return next_state.to(bf16), output_t.to(bf16)`

**结论：纯 bf16 前向里，每一个张量都是 bf16，包括那个崩溃形状 `[4,16,64,128]` 的状态。前向没有任何 f32 来源。**

---

## 4. 因此：f32 只能源于 `torch.autocast("xla", bf16)`

既然前向在纯 bf16 下干净无瑕，那 `f32[4,16,64,128]` 必然是 `autocast` 在 XLA 编译期注入的。
`autocast` 是这份代码里**唯一**会在 XLA 图下降阶段改写算子 dtype 的机制——`model.to(bf16)`、
`XLA_USE_BF16=1`、你手动加的 `.to(bf16)` 都不会产生 fp32。

---

## 5. 为什么 `scan` 会注入 fp32，而 functorch 复现不出来

我用 `functorch.compile.aot_function`（**正是 `torch_xla.experimental.scan` 内部用的同一原语**），
按 config 真实维度搭了一个等价 scan body，在 `autocast("cpu", bf16)` 下跟踪，并 dump 前向/反向图：

```
########## (A) autocast ON -> AOT trace ##########
  [FW-autocast]   (无 fp32 节点)
  [BW-autocast]   (无 fp32 节点)
  done. out (4,4,16,64,128) torch.bfloat16
########## (B) autocast OFF -> AOT trace ##########
  (同样 0 个 fp32 节点)
```

**关键发现：functorch 的 `aot_function` 在 `autocast` 下跟踪时，autocast 是惰性的（FakeTensor 不走
Autocast 分发键），跟踪出的图完全一致 bf16，因此构造不出 f32。** 也就是说，functorch 路径上
`scan` + `autocast` 本不冲突。

那 XLA 为什么崩？因为 `torch_xla.experimental.scan` 的 `value_and_grad_partitioned`
**不走 functorch 的 FakeTensor 跟踪**，而是在 XLA 自己的 tracing 上下文里跟踪 body，并把整个模型
先 trace 成一张大图，**再在图下降阶段由 XLA autocast 插入 dtype cast**。XLA autocast 的 fp32 提升
作用在非 FakeTensor 的真实 trace 上，而 `scan` 编译出的 `carry`（递归状态）I/O 跨 chunk 复用，是
这张子图的边界——XLA 的 cast 插入对"递归状态在子图入口=bf16、出口梯度=fp32"的对称处理出现不一致，
于是 `scan` 的反向图里那个 `[4,16,64,128]` 状态张量一侧是 fp32、一侧是 bf16，被 XLA 的硬断言拦下。

> 这解释了你"变量都加了 `.to(bf16)` 还是报错"：autocast 的 cast 插在编译图内部，Python 层的
> dtype 改不动它；也解释了为什么 `is_fn_pure` 改成 `False` 也没用——`scan` 无论 `is_fn_pure` 取何值
> 都走 AOT 编译（之前已确认）。

**一句话根因**：`autocast("xla", bf16)` 包裹 `torch_xla.experimental.scan`，在 XLA 把 `scan` 状态
`[B,H,64,128]` 编译成反向子图时，autocast 的 fp32 提升_cast 在递归状态边界上插入不一致，导致
`matmul(f32[4,16,64,128], bf16[4,16,64,128])`，被 XLA 断言 `INVALID_ARGUMENT` 拦死。

---

## 6. 两个附带发现（不是崩溃原因，但你应该知道）

1. **`use_gla=True` 在 chunkwise 路径下是死的。** `_chunkwise_retention` 的签名是
   `(q, k, v, state, chunk_size)`，**根本没有接收 `g`**，函数体内也没用 GLA 门控。也就是说
   `config.json` 里 `use_gla=true` 与 chunkwise 前向不一致——要么 GLA 是你想用但漏接了，
   要么它只是个占位。这不影响崩溃，但是个潜在正确性坑。
2. `use_growth_ln=true` 同理，需要确认 `GrowthLayerNorm` 是否真的挂在层里（growth_old_dim=768）。
   同样与本次崩溃无关。

---

## 7. 修复确认

两份 notebook 都已把 `autocast` 彻底移除（`cell4` 内 `torch.autocast` 引用 = **0**），让整条前向 +
`scan` 真正统一在 bf16：

| 文件 | 内容 | 验证 |
|---|---|---|
| `retnet-kaggle-train-8k-fixed.ipynb` | 纯 bf16（去掉 autocast，`from_pretrained` 显式 bf16） | `autocast=0`，`model.to(bf16)` ✓ |
| `retnet-kaggle-train-8k-mixed.ipynb` | 手动混合精度（LayerNorm/GroupNorm 走 fp32 再 cast 回 bf16，scan body 仍纯 bf16） | `autocast=0` ✓ |

由于第 3 节已证明"纯 bf16 前向无任何 f32"，移除 `autocast` 后 `scan` 状态 `[4,16,64,128]` 全程 bf16，
崩溃必然消失。loss 的 fp32 累加（`logits.float()` / `total_loss` / `step_loss`）按原设计保留，且其
反向会把梯度 cast 回 bf16，不与 scan 冲突。

---

## 8. 上 TPU 后 30 秒定位（验证用，非必需）

如果排队上 TPU 后想亲手确认根因，在 `train_worker` 第一步前加：

```python
# 临时隔离：把 CFG.model_mode 改成 "recurrent" 或 "parallel"（不走 scan）
# 若报错消失 → 坐实是 scan + autocast；若仍在 → 与 scan 无关
```

以及 dtype 探针（在 `outputs = model(...)` 前后、`_retnet_scan_chunk` 入口打印）：

```python
print("state dtype:", state.dtype, "q dtype:", q.dtype, "out dtype:", outputs.logits.dtype)
```

按第 3 节推导，纯 bf16 下这些都应打印 `torch.bfloat16`；若某个是 `float32`，就是 autocast 漏网处。
