# `torch.autocast` 与 `torch_xla.experimental.scan` 的冲突机制

> 为什么在 `with torch.autocast("xla", dtype=torch.bfloat16):` 里调用 `torch_xla.experimental.scan(...)`
> 会报 “float32 和 bfloat16 精度不同”？
> 本文把两边的源码机制拆开，再定位冲突点。

---

## 0. 一句话结论

`autocast` 是**运行时按算子（op）动态插 cast** 的机制（靠 dispatcher 拦截每个 op）；
`scan` 是**先把 body 跟踪（trace）成一张静态图、再编译回放**的机制（靠 AOTAutograd）。
当你把 `scan(...)` 写进 `autocast` 上下文里，`scan` 在“跟踪”body 的那一刻把 autocast 的 fp32 提升
**烤进了静态图**；但 autocast 在运行时对 `Linear` 做的“把输入重新 cast 回 bf16”这一步，
并不会被 AOT 跟踪 faithfully 地保留。于是编译出的图里出现
`fp32 激活 → bf16 权重` 的 matmul 边界，XLA 把整张图当单一 HLO 编译时无法对齐 dtype，报错。

你手动 `.to(bfloat16)` 救不了，因为：
1. autocast 每次调用 `norm` 都会**重新把输出提回 fp32**，把你加的 cast 抵消；
2. 真正参与计算的是 AOT **跟踪出来的静态图**，Python 层的张量 dtype 动不到编译图内部。

---

## 1. `torch.autocast` 的机制

### 1.1 它只管一个“开关栈”，不碰张量

源码 `torch/amp/autocast_mode.py`：`autocast` 进入/退出时只保存和恢复**按设备线程局部**的
开关与 dtype，状态存在 C++ 内部（`torch.set_autocast_enabled` / `get_autocast_dtype` 等）：

```python
def __enter__(self):
    self.prev = torch.is_autocast_enabled(self.device)
    self.prev_fastdtype = torch.get_autocast_dtype(self.device)
    torch.set_autocast_enabled(self.device, self._enabled)
    torch.set_autocast_dtype(self.device, self.fast_dtype)
    torch.autocast_increment_nesting()
    ...

def __exit__(self, exc_type, exc_val, exc_tb):
    if torch.autocast_decrement_nesting() == 0:
        torch.clear_autocast_cache()
    torch.set_autocast_enabled(self.device, self.prev)
    torch.set_autocast_dtype(self.device, self.prev_fastdtype)
    ...
```

> 关键点：autocast **不会**改变任何 `nn.Parameter` 的 dtype，也不会预先改写你的张量。
> 它是在“每次 op 被分发（dispatch）执行时”临时插 cast。

### 1.2 真正的“提升规则”在 dispatcher（C++ Autocast 分发键）

op 级别的“谁升 fp32、谁留在 autocast dtype”由 PyTorch C++ 端的 **Autocast 分发键** 决定
（Python 文件里已经没有那些 `_cast` 列表了，它们搬到 C++ 里）。公认的默认策略：

| 类别 | 行为 | 代表 op |
|---|---|---|
| 归一化类 | **提升到 fp32** | `layer_norm`、`group_norm`、`batch_norm`、`softmax`、`log_softmax`、`pow` |
| 矩阵乘类 | 留在 autocast dtype（bf16） | `matmul`、`bmm`、`linear/addmm`、`conv` |
| 逐点类 | 取输入里最宽的 dtype | `silu`、`+`、`*`、`gelu` |

> ⚠️ **XLA:TPU 与 CUDA 的「fp32 清单」不同**：PyTorch/XLA 官方 AMP 文档给出的
> XLA:TPU `float32` 提升清单是 `batch_norm / log_softmax / binary_cross_entropy /
> prod / cdist / cholesky / inverse / mse_loss / nll_loss / …`，**不含 `layer_norm`、
> `group_norm`、`softmax`、`pow`**。也就是说在 TPU 上，`norm1/group_norm` 这类在
> CUDA 下会被提回 fp32 的算子，**默认仍留在 bf16**。本文档后面针对你这份 notebook 的
> 具体分析以 XLA:TPU 为准（fp32 来源不是 norm，而是显式的 `logits.float()` / loss 路径）。

这意味着：在 CUDA 的 `autocast` 内，`x = self.norm1(x)` 之后 `x` 会变成 **fp32**，
紧接着 `q = self.q_proj(x)`（权重是 bf16）会出现 “fp32 输入 × bf16 权重”；
而在**普通 eager 模式**下，autocast 会在分发 `linear` 时把输入重新 cast 回 bf16，所以通常能跑。
（在 XLA:TPU 上 norm 不提升，所以这一特定路径不会自发产生 fp32。）

### 1.3 它依赖“运行时真实张量 + 分发键”

autocast 这套“动态插 cast”只在**真正用真实张量执行、且 Autocast 分发键生效**时才起作用。
它不是一个会改变图结构的静态变换，而是“每次调用 op 时现场决定 dtype”。

---

## 2. `torch_xla.experimental.scan` 的机制

### 2.1 永远走 AOTAutograd 跟踪编译

`scan.py` 的公开函数 `scan()` **无条件**调用 `value_and_grad_partitioned`：

```python
def scan(fn, init, xs, partition_fn=default_partition, is_fn_pure=False, ...):
    if not callable(fn):
        raise ValueError(...)
    # 校验 xs 各叶长度一致 ...
    forward, alias_input, backward = value_and_grad_partitioned(
        fn, init, xs, partition_fn=partition_fn, is_fn_pure=is_fn_pure)
    carry, ys = Scan.apply(forward, alias_input, backward, init, xs)
    return carry, ys
```

而 `value_and_grad_partitioned` 内部是这样把 body `fn` 编译成静态图的
（注意：编译块里**只有 `torch.enable_grad()`，没有任何 autocast 上下文**）：

```python
with torch.enable_grad():
    fw_compiler, get_fwd = _make_get_graph_compiler()
    bw_compiler, get_bwd = _make_get_graph_compiler()
    fn_compiled = aot_function(                     # ← AOTAutograd 跟踪 fn
        fn_no_output_aliasing,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn)
    # 之后 fn_compiled 被 trace 出 fwd_graph / bwd_graph
```

`aot_function`（来自 `functorch.compile`）会**运行一次 `fn`** 来捕获算子序列、建成
前向图 + 反向图，并交给 XLA 编译器。这就是“跟踪（trace）”——把动态 Python 函数
固化成一张**静态、可缓存**的计算图。

### 2.2 `is_fn_pure` 只管缓存，不管是否走 AOT

源码里缓存分支明确受 `is_fn_pure` 控制，但编译路径不受它影响：

```python
# 仅当 is_fn_pure 才查/写缓存
if is_fn_pure and fn in _SCAN_COMPUTATION_CACHE:
    if second_level_cache_key in _SCAN_COMPUTATION_CACHE[fn]:
        return _SCAN_COMPUTATION_CACHE[fn][second_level_cache_key]
...
if is_fn_pure:
    if fn not in _SCAN_COMPUTATION_CACHE:
        _SCAN_COMPUTATION_CACHE[fn] = {}
    _SCAN_COMPUTATION_CACHE[fn][second_level_cache_key] = (forward, alias_input, backward)
```

所以无论 `is_fn_pure` 取何值，`aot_function` 都会被调用（除非命中纯函数缓存）。
改 `is_fn_pure` 救不了你的报错。

### 2.3 回放在 `Scan` 自定义 autograd.Function 里，且反向在 `no_grad` + 无 autocast 下跑

```python
@pytreeify
class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, forward, alias_input, backward, init, xs):
        with torch.no_grad():                     # ← 反向/前向回放都在 no_grad
            carry, ys = _scan_impl_pytree(forward, init, xs)
            ...
    @staticmethod
    def backward(ctx, grad_carry, grad_ys):
        saved = _load_from_context(ctx)
        with torch.no_grad():
            grad_init, grad_xs = _scan_impl_pytree(
                backward, grad_carry, (grad_ys, saved["activations"]), reverse=True)
        return None, None, None, grad_init, grad_xs
```

注意：`Scan.backward` 在 `torch.no_grad()` 下、且**不在任何 autocast 里**回放反向图。

---

## 3. 冲突具体发生在哪里

把上面两条线拼起来，你代码里的大致结构是这样的：

```python
with torch.autocast("xla", dtype=torch.bfloat16):     # (A) autocast 开启
    ...
    carry, ys = scan(body_fn, init, xs)               # (B) scan 在这里被调用
    ...
```

### 3.1 你这份 notebook 的真实 fp32 来源：不是 norm，是 `logits.float()`

先纠正一个常见误判：**在 XLA:TPU 上，autocast 并不会把 `norm1/group_norm/softmax`
提回 fp32**（见 1.2 的清单差异），而且你已经在 scan body 里把所有张量显式
`.to(torch.bfloat16)`（`q/k/v/gamma/inner_mask/cross_decay/...` 全是 bf16）。
所以 scan body **内部是纯 bf16 的**，不是冲突的发源地。

真正的 fp32 来自 **loss 路径**（第 478/487 行）：

```python
logits_f = logits.float()                       # ← 显式 fp32
loss = F.cross_entropy(logits_f.reshape(-1, ...), labels.reshape(-1))
```

`logits` 本身是 bf16（模型输出），但 `logits.float()` 把它变成 fp32，
cross_entropy 在 fp32 下算，于是 **loss 是 fp32，loss 的梯度也是 fp32**。

### 3.2 冲突：fp32 loss 梯度 ↔ AOT 编译出的 bf16 scan 反向图对不上

在 (B) 处 `scan` 调用 `aot_function` 跟踪 `body_fn` 时，(A) 的 autocast 仍生效，
且 XLA:TPU 下 body 算子都是 bf16，所以 AOT 抓到的**前向图是纯 bf16**，并据此
符号微分生成了一张**期待 bf16 梯度**的扫描反向图。

但 `loss.backward()` 时，跨 `cross_entropy` 流回来的梯度是 **fp32**（来自 3.1 的
fp32 loss）。XLA 要把 “fp32 的 cross_entropy 反向” 和 “bf16 的 scan 反向图”
编译进**同一张 HLO 图**。由于 `scan` 的 AOT 跟踪是在 autocast 上下文里做的、
它**不会在 fp32-loss ↔ bf16-scan 的边界插入对齐 cast**（autocast 的现场 cast
依赖运行时分发键，而 AOT 的 tracing 上下文不会 faithful 重放这套 cast），
于是编译出的 matmul 节点见到 `fp32 梯度 × bf16 权重` → dtype 不匹配。

> 一句话：body 内部很干净（全 bf16），但**最外层的 fp32 loss 梯度**在反向时
> 撞上 AOT 编译出的 bf16 scan 反向图，且这条边界缺一个 cast。

### 3.3 缓存让“坏图”反复复用

如果 `is_fn_pure=True`（你代码里正是如此），编译好的混合 dtype 图会被缓存，
即使你退出 autocast 上下文、或换数据，复用的是同一张带 fp32/bf16 边界的图。

### 3.4 XLA 把整图当单一 HLO 编译 → 边界炸开

XLA 会把 `cross_entropy 反向 + scan 反向` 编译成**一个 HLO 计算图**。HLO 对 matmul 节点
严格要求左右操作数 dtype 一致。`matmul(fp32 梯度, bf16 权重)`（或 fp32 激活 × bf16 权重）
是非法的，于是出现你看到的：

```
RuntimeError: ... expected both to be <dtype> (got float32 and bfloat16)
```

（这个错误正是 matmul/Linear 的 dtype 不匹配报错。）

> 补充（CUDA 用户注意）：在 **CUDA** 上，autocast 确实会把 `layer_norm/group_norm/softmax`
> 提回 fp32，那时冲突的发源地会转移到 “norm 输出 fp32 → Linear 权重 bf16” 的 body 内部边界，
> 机制同样是 “AOT 跟踪 + 缺对齐 cast”。本文档 3.1/3.2 针对你这份 **XLA:TPU** 代码给出的是
> loss 路径视角。

---

## 4. 为什么你手动 `.to(torch.bfloat16)` 没用

1. **body 内部你其实已经转对了**：`q/k/v/gamma/...` 都已是 bf16，问题不在 body 内部，
   而在**最外层 `logits.float()` 的 fp32 loss 梯度**——你没（也不该）把 loss 也转成 bf16。
2. **Graph ≠ Python 张量**：真正喂给 XLA 的是 AOT **跟踪出来的静态图**，图里
   `fp32-loss ↔ bf16-scan` 的边界在 trace 时就定死了，且缺对齐 cast。
   Python 层再怎么 cast 变量，动不到已经编译好的图内部。

所以“变量都加了 `.to(bfloat16)` 还是报错”完全符合这套机制——你补的是 Python 层，
冲突生在编译图层（loss 梯度与 scan 反向图的 dtype 边界）。

---

## 5. 正确解法（回顾）

既然模型已是 `model.to(device, dtype=torch.bfloat16)` 且 `XLA_USE_BF16=1`，
autocast 是**冗余且有害**的，应直接去掉：

| 方案 | 做法 | 说明 |
|---|---|---|
| 纯 bf16（推荐起点） | 删掉包住 `scan` 的两个 `autocast` | 权重/激活/scan 全 bf16，loss 保持 fp32 |
| 手动混合精度 | 把 `norm1/norm2/final_norm`、`_head_group_norm`、`ln_old` 改成
  `self.norm*(x.float()).to(x.dtype)`（fp32 算完 cast 回 bf16） | 数值敏感处保 fp32，residual stream 仍 bf16，scan body 不碰 autocast |
| ❌ 原做法 | `autocast` 包住 `scan` | 触发本文所述的 AOT 冲突 |

> 核心原则：**fp32 只能出现在“归一化层内部”和“loss 累加”这两类数值敏感点，且必须立刻 cast 回 bf16**
> 再进入下游 matmul / scan。这样 scan body 拿到的永远是纯 bf16，不会出现 fp32↔bf16 的编译图边界。

---

## 6. 这是已知 bug 吗？有 PR 修复吗？还是默认“特性”？

> 直接结论：**广义上是 PyTorch 社区已知的“autocast 不兼容 AOT/functorch/torch.compile”
> 类限制；但针对 `torch_xla.experimental.scan` + `autocast` 这个具体组合，没有专门的高赞
> issue/PR 来修它。** 它既不是被打上 “won't fix” 的 bug，也不是文档化的“特性”，
> 而是 **scan 从设计上完全无视 autocast 的未处理边界情况（unhandled edge case）**。

### 6.1 有人提过吗？

- **广义层面（有）**：autocast 与 `AOTAutograd` / `functorch` / `torch.compile` 不兼容是
  PyTorch 长期存在的已知限制。社区为此开过 issue，主线（Dynamo/torch.compile 路径）后来
  也补了“autocast 感知”的编译支持，使 `torch.compile` + `autocast` 在新版 PyTorch 下可用。
- **具体到 `torch_xla.experimental.scan` + `autocast`（检索未见专门高赞 issue/PR）**：
  `scan.py` 的模块 docstring 明确写它 **“symbolically traces the user-provided function
  `fn` to obtain the forward and backward computation graphs”**，且全文 **零提及**
  `autocast / mixed precision / AMP / dtype` 处理（见源码核对）。这属于“冷门但真实存在”的坑。

### 6.2 有 PR 修复这种 bug 吗？

- **`torch.compile` / Dynamo 路径（有）**：现代 PyTorch 已让 `torch.compile` 能感知
  autocast，相关 fix 已合入主线。
- **`torch_xla.experimental.scan` 路径（没有专门 PR）**：`scan` 直接调用底层
  `functorch.compile.aot_function`，**绕过了 Dynamo 的 autocast 处理**，
  scan 内部也没有任何 autocast 兼容代码。所以**对你这个具体场景，目前没有 PR 把它修好**。

### 6.3 是默认的“特性”吗？

- **不是“特性”，也不是“已登记为 bug”**，而是 **未处理的边界情况**：
  - 官方 XLA AMP 文档的 “Supported Operators” 只列出 fp32/bf16 提升算子，**完全没有提
    `scan` 与 AMP 的交互**；文档末尾还写 “Please file an issue or submit a pull request
    if there is an operator that should be autocasted that is not included.”
    这等于把“算子覆盖/行为缺失”当成**可上报、可扩展**的事，而非已修复的 bug。
  - 文档 Best Practices 只说 “`autocast` should wrap only the forward pass(es) and loss
    computation(s)”，未提及 scan/AOT 的例外。**等价于：scan + autocast = 不被支持、
    不报错、也不保证能跑。**

### 6.4 实践建议

- **不要在 `scan` 外面包 `autocast`**（纯 bf16 或手动混合精度，见第 5 节），这是当前唯一稳的路。
- 如果你希望它被正式跟踪/修复，最合适动作是去 `pytorch/xla` 提一个 issue，要求
  `scan` 对 autocast 友好（或在文档/源码里**显式声明不支持 autocast 包 scan**并给出报错）。
  这正合官方文档“file an issue or submit a PR”的口径。

---

## 7. 诚实修订：当前解释的局限与未决问题

> 本节是对第 3、4、6 节的修正与自我审查。**核心结论：结构性叙事（autocast 是运行时分发、
> scan 用 AOT 跟踪静态图、二者不兼容）站得住，但之前写的两个「具体触发机制」已被新证据
> 推翻；且按标准 autocast 语义静态分析这份代码，根本找不到那个 fp32×bf16 的 matmul。
> 因此「scan 是祸首」的归因很可能偏了，真正的锅更可能是 XLA autocast 在 XLA 设备下的
> 非标准行为。**

### 7.1 已被推翻的两个具体机制（附带验证）

- **机制 A「AOT 把 autocast 的 fp32 提升烤进图、又丢掉对齐 cast」→ 推翻。**
  CPU 实测（`functorch.compile.aot_function` 在 `torch.autocast("cpu", bf16)` 内跟踪一个
  含 `layer_norm` 的 body）：跟踪图中 `native_layer_norm` 的输出 meta dtype 是 **bf16**，
  `mm` 也是 bf16。原因——**AOT 跟踪用 FakeTensor，FakeTensor 不走 Autocast 分发键，
  autocast 在跟踪期间是惰性的**，所以跟踪出的图是**一致 bf16**，并未烤入 fp32，也不会
  “丢了 cast” 后 mismatch。AOT+autocast 对一致 bf16 的 body 在 CPU 上能正常前向+反向。

- **机制 B「fp32 loss 梯度撞上 bf16 scan 反向图」→ 推翻。**
  autograd 在 `logits.float()` 的反向会把 fp32 梯度 **cast 回 bf16**（梯度 dtype 跟随
  被求导张量 `logits` 的 dtype）。所以梯度进入 scan 反向图时已是 bf16，不是 fp32。

- **机制 C「XLA:TPU 上 norm 被 autocast 提回 fp32，导致 body 内 fp32→bf16」→ 不适用。**
  XLA 官方 AMP 文档的 fp32 提升清单**不含** `layer_norm/group_norm/softmax`；且实测
  `_retnet_scan_chunk` 内部无任何 norm/softmax，全 `.to(bfloat16)`。故 fp32 不来自 body 内 norm。

### 7.2 已经确认的事实（verified）

1. scan body `_retnet_scan_chunk` 是**纯 bf16**：所有 `xs` 输入、`carry` 状态、返回值都
   `.to(torch.bfloat16)`（见源码第 22–90 行）。
2. AOT 跟踪在 autocast 内产生**一致 bf16** 图，CPU 上 benign（7.1 实验）。
3. 整图静态分析：唯一 fp32 是 `logits.float()`（第 478/487 行）、loss 权重
   `w=...dtype=float32`（第 481 行）、`total_loss/step_loss` 累加器——**没有一处是 fp32 与
   bf16 做 matmul**。旋转编码 `_rotations`/`inv_freq`/`gamma` 全部 bf16。

### 7.3 未决的核心问题

按标准 autocast 语义**静态分析找不到**那个 fp32×bf16 matmul。所以报错来源只可能是：

- **(a) XLA autocast 的实际行为 ≠ eager/CUDA**：XLA 的 autocast 提升清单/lowering 可能与
  官方文档子集或 CUDA 不同——例如它若把 `layer_norm/group_norm` 提 fp32、却在
  `Linear` 处**不插回 bf16 的 cast**（eager CUDA 会插），就会在 RetNet 层的
  `norm1(x) → q_proj(...)` 处出现 `matmul(fp32, bf16)`，且发生在 **scan 之外**。
- **(b) XLA 编译器在 model→scan 边界 / 图融合时的 dtype 推断**：这是 XLA 特有，不是
  AOT 的通用事实。

由于本沙箱**无法安装/运行 `torch_xla`（且无 TPU）**，我无法在此复现并定位具体算子。

### 7.4 这对「scan 是祸首」意味着什么

`functorch.compile.aot_function`（XLA scan 用的同一原语）对**一致 bf16 的 body** 在 autocast
下是 benign 的（7.1）。而本代码的 scan body 正是一致 bf16。因此 **scan 不太可能是这份代码
报错的直接来源**；真正的锅更可能是「**bf16 模型 + XLA autocast**」这个组合在 XLA 下的非标准
行为（最可疑是 RetNet 层里的 norm→Linear，或 loss 路径在 XLA lowering 时的 dtype 处理）。

**但修复方向仍然正确**：既然模型已是 `model.to(bf16)` + `XLA_USE_BF16=1`，autocast 在任何
情况下都是冗余的；去掉它（或改手动混精，只让 norm fp32 算完立刻 cast 回 bf16）都会消除
XLA autocast 引入的 fp32 边界——无论那个边界究竟生成在哪个算子。

### 7.5 如何彻底闭环（建议用户执行）

1. **拿真实 traceback**：报错时打印完整堆栈，定位具体是哪个 op、哪一行
   （scan 内 matmul？RetNet 层 norm→Linear？还是 loss 路径）。
2. **最小隔离测试**：把 `CFG.model_mode` 临时改成 `"parallel"`（不走 scan）跑一步。
   - 若**仍**报 f32/bf16 → 与 scan 无关，是 autocast+bf16-model 的普遍问题；
   - 若**消失** → 确与 scan 相关（但按 7.4 推理，本代码概率低）。
3. **打印 dtype 探针**：在 `model(input_ids,...)` 前后、以及 `_retnet_scan_chunk` 入口
   打印各张量 dtype，看 XLA autocast 实际把哪些变成了 fp32。

---

## 附：关键源码位置速查

- `torch/amp/autocast_mode.py`：`autocast.__enter__` / `__exit__`（开关栈），
  `autocast_decorator`、`custom_fwd` / `custom_bwd`（装饰器）。
- `torch_xla/experimental/scan.py`：`scan()`（公开入口）、`value_and_grad_partitioned`
  （`aot_function` 调用点）、`Scan`（`torch.autograd.Function`，`no_grad` 回放）。
- op 级别 fp32 提升策略：PyTorch C++ 端的 **Autocast 分发键**（不在 Python 文件里）。
