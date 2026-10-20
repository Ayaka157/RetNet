# E. 文献核查：模型生长（宽度 768→1024→1408）时新增参数应如何初始化

> 核查对象：从零手写 RetNet 的宽度生长实现——每个线性层新矩阵 `w = zeros(out, in)` 后把旧权重贴左上角（新增行、新增列都是精确 0）；新增维度 LayerNorm 的 `gamma_new = 0`、`beta_new = 0`，且新维度不做归一化（`new = x_new * gamma_new + beta_new`）；embedding 新列也是 0。结果是两次生长后 67% 参数精确为 0，且这些参数从第 0 步起梯度恒为 0。
>
> 核查方法：用 web_search 定位论文，用 web_fetch 抓 arXiv abs/ar5iv 全文、ACL Anthology PDF、PMLR 官方 PDF、OpenAI 官方仓库源码。所有英文引文均逐字抄自抓取到的正文；中文为直译。未抓到的内容一律标注"未找到/无法核实"。
>
> 核查时间：2026-10-20。约 45 分钟，未运行任何训练。

---

## 1. 裁决

**文献里没有任何一篇论文把"新增权重行 + 新增权重列 + 新增 norm 门控 + 新 embedding 列同时置 0"当作正确做法；相反，所有函数保真生长都保证新参数至少有一侧非零（复制旧参数、随机初始化、或学习出的线性组合），置零只出现在"会被非零另一侧相乘"的单侧。** 两侧同时置零时，该维度的前向恒为 0、反向链路上每一环都含 0 因子，梯度从第 0 步起精确为 0，且没有任何机制能把它推出 0——这是死参数，不是"延迟学习"。**最接近的正确做法是 Network Morphism 的宽度生长：只把参数较少的一侧置 0，另一侧必须随机非零（Wei et al. 2016），或 Net2Net/bert2BERT 的"复制旧行列 + 按复制次数缩放 + 加噪破对称"。**

---

## 2. Q1：主流"函数保真生长"怎么初始化新参数

### 2.1 Net2Net: Accelerating Learning via Knowledge Transfer（Chen, Goodfellow, Shlens, 2015；ICLR 2016；arXiv:1511.05641）

链接：<https://arxiv.org/abs/1511.05641> ｜ 全文（ar5iv）：<https://ar5iv.labs.arxiv.org/html/1511.05641>

**做法（Net2WiderNet，加宽）**：不是补零，而是**随机复制旧神经元**。引入随机映射 `g: {1..q} → {1..n}`，新列直接取自旧列；同时把下一层对应行的权重除以该列的复制次数，使函数不变。

原文（§2.3 Net2WiderNet）：

> "Here, the first n columns of W(i) are copied directly into U(i). Columns n+1 through q of U(i) are created by choosing a random [column] as defined in g. The random selection is performed with replacement, so each column of W(i) is copied potentially many times. For weights in U(i+1), we must account for the replication by dividing the weight by replication factor given by 1/|{x|g(x)=g(j)}|, so all the units have the exactly the same value as the unit in the original net."

直译：W(i) 的前 n 列被直接复制进 U(i)；第 n+1 到 q 列按 g 的定义随机选取（有放回抽样，所以每一列都可能被复制多次）。对 U(i+1) 的权重，必须按复制因子 1/|{x|g(x)=g(j)}| 相除，以补偿复制，使所有单元与原网络中的那个单元取值完全相同。

原文（图 2 说明）：

> "To replicate the h[2] unit, we copy its weights c and d to the new h[3] unit. The weight f, going out of h[2], must be copied to also go out of h[3]. This outgoing weight must also be divided by 2 to compensate for the replication of h[2]."

直译：为复制 h[2] 单元，把它的权重 c、d 复制给新单元 h[3]；从 h[2] 出去的权重 f 也必须复制到 h[3] 出去，并且要除以 2 以补偿 h[2] 的复制。

**关键：还要加噪破对称**（§2.3 末）：

> "When using a training algorithm that does not use randomization to encourage identical units to learn different functions, one should add a small amount of noise to all but the first copy of each column of weights. This results in the student network representing only approximately the same function as the teacher, but this approximation is necessary to ensure that the student can learn to use its full capacity when training resumes."

直译：当训练算法不靠随机化来促使相同单元学出不同函数时，应当给每一列权重的所有副本（第一个副本除外）加少量噪声。这使学生网络只近似表示教师网络的函数，但这种近似是必要的，否则学生无法在恢复训练后用满其容量。

**做法（Net2DeeperNet，加深）**：新层初始化为**单位矩阵**，不是零。

原文（§2.4）：

> "The new matrix U is initialized to an identity matrix, but remains free to learn to take on any value later. This operation is only applicable when φ is chosen such that φ(Iφ(v)) = φ(v) for all vectors v. This property holds for the rectified linear activation."

直译：新矩阵 U 初始化为单位矩阵，但之后仍可自由学习成任意值。该操作只适用于 φ 满足 φ(Iφ(v)) = φ(v) 的情形；ReLU 满足这一性质。

原文（§2.4 末）：

> "When using Net2DeeperNet, there is no need to add noise. We can begin training with a student network that represents exactly the same function as the teacher network. Nevertheless, a small amount of noise can be added to break the symmetry more rapidly."

直译：用 Net2DeeperNet 时不必加噪，可以从与教师网络完全同函数的学生网络开始训练；不过也可以加少量噪声来更快打破对称。

**判读**：Net2Net 的两种生长都**没有"新参数置 0"**：加宽是复制列 + 缩放下一层 + 加噪，加深是单位阵。它甚至把"随机初始化"当作需要被函数保真替代的基线（"Random pad" / "Random initialization" 两个 baseline，§3.1）。这直接说明"置 0"不是 Net2Net 家族的做法。

---

### 2.2 bert2BERT: Towards Reusable Pretrained Language Models（Chen et al., 2021/2022；ACL 2022；arXiv:2110.07143）

链接：<https://arxiv.org/abs/2110.07143> ｜ ACL 官方 PDF：<https://aclanthology.org/2022.acl-long.151.pdf>

**做法（宽度）**：FPI（function-preserving initialization）= 复制 + 按复制次数缩放，与 Net2Net 同构。映射函数 `g_in`、`g_out` 把新维度随机映射回旧维度；被复制 k 次的行要除以 k。

原文（§4.3.1，式 7）：

> "C_{g_in(i)} = Σ_{i'=1}^{d_in^u} I(g_in(i') = g_in(i)); Ũ(i,*) = (1 / C_{g_in(i)}) W(g_in(i),*)."

直译：C 是 g_in(i) 在 g_in 取值中出现的次数；Ũ 的第 i 行等于 W 第 g_in(i) 行除以该次数。（即：新行是新维度对旧行的复制，再按复制次数缩放。）

**embedding 的处理**：只做 out-dimension 扩张，即**复制旧列**，不是补零。

原文（式 9）：

> "For the embedding matrix U^E, we only conduct the out-dimension expansion: U^E(*,j) = W^E(*, g_out^e(j))."

直译：对 embedding 矩阵 U^E，只做输出维扩张：其第 j 列等于 W^E 的第 g_out^e(j) 列。

**LayerNorm 的处理**：**复制旧 gamma/beta**，不是置 0。

原文（式 13）：

> "U^LN_j = W^LN_{g_out^2(j)}."

直译：LayerNorm 的第 j 个参数等于 W^LN 的第 g_out^2(j) 个参数。

同时作者承认 LN 的统计量会让严格函数保真出现缺口（这一点对后面判断"新维度不做归一化"很重要）：

> "Note that in layer normalization (Eq. 4), the mean μ and variance σ are calculated based on the hidden representations H. Thus, the expansion of this parameter inevitably induces a gap and prevents the target model from strictly following the function preserving principle. However, we empirically find that the gap is so small that it can hardly affect the initialization and convergence of the target model."

直译：注意在 LayerNorm 中，均值 μ 与方差 σ 是基于隐表示 H 计算的；因此该参数的扩张不可避免地引入缺口，使目标模型无法严格遵循函数保真原则。但我们经验上发现这个缺口很小，几乎不影响初始化与收敛。

**深度**：直接把加宽后的模型**堆叠**到目标深度。

原文（§4.4）：

> "To bridge the depth gap, we perform depth-wise expansion to increase model depth to the depth of the target model. We illustrate this process in Algorithm 1 and the main idea is to iteratively stack the widened model until its depth is equal to the target model (Gong et al., 2019)."

直译：为填补深度差距，我们做深度扩张，把模型深度增加到目标深度；Algorithm 1 的核心思想是反复堆叠加宽后的模型，直到深度等于目标模型（Gong et al., 2019）。

**为什么要 AKI（关键）**：作者明确说 FPI 会带来"对称"，需要**用相邻层的参数**去打破，而不是靠置零/加噪。

原文（§4.3.2）：

> "Moreover, the knowledge that comes from adjacent layers can break the symmetry (Chen et al., 2016) appeared in FPI, which has been demonstrated beneficial."

直译：此外，来自相邻层的知识可以打破 FPI 中出现的对称（Chen et al., 2016），这已被证明是有益的。

原文（§4.3.2 末）：

> "For example, FPI makes the attention patterns in the same layer repeated, which is redundant and called symmetry."

直译：例如，FPI 使同一层内的注意力模式重复，这种冗余被称为对称。

原文（附录 A）：

> "The FPI curve has an upward trend at the beginning. We conjecture that it is due to the symmetry brought by FPI and the model needs some optimization time to break this symmetry."

直译：FPI 的曲线在开始时有一个上升趋势。我们推测这是因为 FPI 带来的对称，模型需要一些优化时间才能打破这种对称。

**判读**：bert2BERT 的新参数全部是"复制旧参数（并对复制次数缩放）"，连 LayerNorm 的 gamma/beta 都是复制；没有任何一处"新增行列置 0"。它承认函数保真只保证输出一致，会留下对称性，需要额外手段（AKI）来破对称。

---

### 2.3 Progressive Stacking（Gong, He, Li, Qin, Wang, Liu, 2019；ICML 2019；PMLR 97:2337-2346）

链接（官方会议页）：<https://proceedings.mlr.press/v97/gong19a.html> ｜ PDF：<http://proceedings.mlr.press/v97/gong19a/gong19a.pdf> ｜ 官方代码：<https://github.com/gonglinyuan/StackingBERT>

**先纠正编号**：委托方给的 arXiv:1908.10951 不是这篇（该号是数学论文 *Higher order distance-like functions and Sobolev spaces*，math.DG）。我用 arXiv API 检索标题 "Progressively Stacking"，只返回 MSLT（2011.13635）与一篇 KAN 论文；StackBERT 的正式来源是 ICML/PMLR，未见 arXiv 版。**请以此 PMLR 链接为准。**

**做法（加深）**：把训练好的 L 层 BERT 复制成 2L 层——第 i 层与第 i+L 层共享第 i 层的参数。

原文（§3 Method）：

> "As is shown in Figure 3, if we have a L-layer trained BERT, we can construct a 2L-layer BERT by copying its parameters: for i ≤ L, the i-th layer and the (i + L)-th layer of the constructed BERT have the same parameter of the i-th layer of the trained BERT."

直译：如图 3 所示，若已有训练好的 L 层 BERT，可以通过复制其参数构造 2L 层 BERT：对 i ≤ L，构造出的 BERT 的第 i 层与第 i+L 层具有训练好的 BERT 第 i 层的相同参数。

**这篇论文最有价值的地方：它明确报告了"零初始化恒等层"这种函数保真生长是失败的。** 这正是 Q3 需要的直接文献记录。

原文（§2.2 Related Work，评 Net2Net 式函数保真）：

> "However, when dealing with sophisticated structures such as Transformer, function-preserving initialization is usually not effective. For example, the basic component in the Transformer is a composition of a self-attention layer and a feed-forward layer. According to our empirical study, simply setting the feed-forward layer to be near zero and randomly initializing the self-attention layer is a function-preserving initialization, but it is ineffective as most parameters in the self-attention layer stay untrained."

直译：然而，在处理 Transformer 这类复杂结构时，函数保真初始化通常并不有效。例如 Transformer 的基本组件是自注意力层与前馈层的复合。根据我们的实证研究，简单地把前馈层设成接近 0、并随机初始化自注意力层，是一种函数保真初始化，但它是无效的，因为自注意力层中的大多数参数始终得不到训练。

他们的 "Identity" 基线（§4.1 Implementation details）：

> "As is shown in Figure 1, each encoder layer of our BERT implementation consists of two residual blocks, and the last steps of both residual blocks are linear transformations (the second linear transformation of a feed forward layer and W^O of a multi-head self-attention layer). Therefore, if we initialize these linear transformations to be zero for the top L layers, these layers add nothing to the hidden states, so the top L layers are strictly equivalent to an identity mapping. ... This construction is strictly function-preserving, as is required by Net2Net (Chen et al., 2015)."

直译：如 Figure 1 所示，我们的 BERT 实现中每个编码层由两个残差块组成，两个残差块的最后一步都是线性变换（前馈层的第二个线性变换，以及多头自注意力的 W^O）。因此，若把顶部 L 层的这些线性变换初始化为 0，这些层对隐状态没有任何添加，顶部 L 层严格等价于恒等映射。……按 Net2Net 的要求，这一构造是严格函数保真的。

结果（§4.2）：

> "In contrast, as is shown in the last 2 rows of this figure, the heads in models trained by identity and identity+noise fail to learn such information."

直译：相反，如图最后两行所示，用 identity 和 identity+noise 训练的模型中的注意力头未能学到这类信息。

> "The algorithms based on function-preserving initialization do not provide a good initialization for attention layers on the top, so they converge slowly and fail to reduce training time."

直译：基于函数保真初始化的算法没有为顶部的注意力层提供好的初始化，因此收敛缓慢，未能缩短训练时间。

**判读**：注意这里的"零"只出现在**残差分支的输出投影**（W^O 与 FFN 第二个线性层），不是"新行新列同时置零"；即便如此，作者仍报告"大多数自注意力参数 stay untrained"、identity 初始化失败。**单侧零在 Transformer 上已经被记录为"无效"；双侧零只会更糟（数学上恒零）。**

---

### 2.4 MSLT: Multi-stage Layerwise Training（Yang, Wang, Yang, Li, He, Zhang, 2020；arXiv:2011.13635）

链接：<https://arxiv.org/abs/2011.13635> ｜ 全文（ar5iv）：<https://ar5iv.labs.arxiv.org/html/2011.13635>

**先纠正编号与名称**：委托方给的 arXiv:2002.02925 不是这篇（该号是 *BERT-of-Theseus: Compressing BERT by Progressive Module Replacing*，是压缩不是生长）。bert2BERT 引用的 "MSLT (Yang et al., 2020)" 就是本文；MSLT = **Multi-stage Layerwise Training**（多阶段逐层训练），不是 "Multi-Stage Generation"。正确编号 **arXiv:2011.13635**，标题 *Progressively Stacking 2.0: A Multi-stage Layerwise Training Method for BERT Training Speedup*。

**做法（加深）**：新加的顶层**复制上一阶段模型的顶部若干层**，随后只训练新层 + 输出层。

原文（§3.2）：

> "Similar to [Gong et al. 2019], at each stage, we initialize the parameters of the newly added N/k layers by copying the parameters of the top N/k layers of the model trained in the previous stage."

直译：与 Gong et al. (2019) 类似，在每个阶段，我们用上一阶段训练所得模型顶部 N/k 层的参数，来初始化新加入的 N/k 层。

原文（§3.2）：

> "Except the first stage in which we update all the parameters, in the other stages, only the parameters of the newly added top layers and the output layer are updated."

直译：除第一阶段更新全部参数外，其余阶段只更新新加入的顶部层与输出层的参数。

**判读**：MSLT 是纯深度生长，新层初始化 = 复制，不是零；而且它把"冻结旧层、只训新层"作为加速手段——这与"新参数梯度恒零"是两回事：冻结是显式选择，梯度可以非零只是不更新；本实现的死参数是梯度本身恒零。

---

### 2.5 LiGO: Learning to Grow Pretrained Models for Efficient Transformer Training（Wang et al., 2022/2023；ICLR 2023；arXiv:2303.00980）

链接：<https://arxiv.org/abs/2303.00980> ｜ 项目页：<https://vita-group.github.io/LiGO/>

**先纠正编号**：委托方给的 arXiv:2208.00317 不是这篇。LiGO 的正确编号是 **arXiv:2303.00980**（ICLR 2023）。

**它为什么认为逐层启发式不够**：

> "Noting the empirical effectiveness of such recipes, we observe that existing mechanisms generally do not have a learning component (e.g., randomly copying over neurons for width-expansion or stacking consecutive layers for depth-expansion)."

直译：注意到这些配方在经验上有效，我们观察到现有机制通常没有"学习"成分（例如宽度扩张时随机复制神经元，或深度扩张时堆叠相邻层）。

> "While existing operators have been empirically successful in accelerating transformer-based models such as BERT, we observe that generally do not have a learning component and perform the depth- and width-expansions separately."

直译：尽管现有算子在加速 BERT 这类 Transformer 上经验上成功，但我们观察到它们通常没有学习成分，而且把深度扩张与宽度扩张分开做。

**它提出的做法**：学习一个线性映射 Θ(large) = M Θ(small)，M 分解为 width 算子与 depth 算子，并用 Kronecker 分解参数化；新行/新列是**旧行/旧列的线性组合**。

原文（§3.2.2）：

> "Each new column/row is a linear combination of columns/rows of small model's weight matrix."

直译：每个新列/新行都是小模型权重矩阵的列/行的线性组合。

原文（§3.2.2 Training）：

> "LiGO expands a model in three steps: (1) for each layer, inserting new rows by linearly combining existing rows through B_l, (2) for each layer, inserting new columns by linearly combining existing columns through A_l, and then finally (3) reconstructing each layer by linearly combining the weight matrices with w along the depth. We then run a few steps (e.g., 100 iterations) of SGD to optimize M..."

直译：LiGO 分三步扩张模型：(1) 对每层，通过 B_l 线性组合现有行来插入新行；(2) 对每层，通过 A_l 线性组合现有列来插入新列；(3) 最后沿深度用 w 线性组合各层权重矩阵来重建每层。然后跑少量（如 100 步）SGD 优化 M……

**判读**：LiGO 的新参数既不是零也不是简单复制，而是**可学习线性组合的初始化**；它把"复制/堆叠"看作自己的特例（附录 A：StackBERT、Interpolation、Net2Net 都是 LiGO 的特例）。这进一步说明主流方向是"新参数必须携带非零信息"。

---

## 3. Q2：零初始化的正确用法，以及为什么它不会把梯度变成 0

### 3.1 先纠正一个前提：GPT-2 的 `wte` / `c_proj` 并不是零初始化

我核对了 OpenAI 官方仓库 `openai/gpt-2` 的 `src/model.py`（<https://github.com/openai/gpt-2/blob/master/src/model.py>，raw：<https://raw.githubusercontent.com/openai/gpt-2/master/src/model.py>）。原文代码：

```python
wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                     initializer=tf.random_normal_initializer(stddev=0.01))
wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                     initializer=tf.random_normal_initializer(stddev=0.02))
```

```python
def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
```

`c_proj` 走的就是 `conv1d` 的默认 `w_init_stdev=0.02`。**官方实现里 `wte` 与 `c_proj` 都是正态随机初始化，没有任何零初始化。**

GPT-2 论文里与"初始化"有关的原话也只有缩放，没有置零：

> "A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers."

直译：使用了一种计入残差路径随深度累积的修正初始化。我们把残差层权重在初始化时按 1/√N 缩放，N 是残差层数。

（来源：Radford et al., *Language Models are Unsupervised Multitask Learners*, 官方 PDF：<https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>，§2.3 Model。）

**结论**：委托描述中的"GPT-2 的 `wte`/`c_proj` 零初始化"**在官方代码与论文中都不成立，无法核实**，疑为误传（可能与"权重共享 `wte`↔`lm_head`"或 1/√N 缩放混淆）。真正使用"零初始化残差分支"的是 Fixup / ReZero / LayerScale，见下。

---

### 3.2 Fixup（Zhang, Dauphin, Ma, 2019；arXiv:1901.09321）

链接：<https://arxiv.org/abs/1901.09321>

Fixup 的规则 1：

> "Initialize the classification layer and the last layer of each residual branch to 0."

直译：把分类层以及每个残差分支的最后一层初始化为 0。

为什么这样不会死：**只零了分支的最后一层，它前面所有层仍是标准随机初始化，因此这一层的输入非零**。对最后一层权重 W_L，`∂L/∂W_L = (∂L/∂y) · h_{L-1}^T`，其中 h_{L-1} 非零，所以 W_L 在第 0 步就拿到非零梯度。原文也给了等价的标量解释：

> "Alternatively, we could start the residual branch as a zero function by setting a_m = 0 and ∀i ∈ [m−1], a_i = L^{−1/(2m−2)}. In the second option, the residual branch does not need to 'unlearn' its potentially bad random initial state, which can be beneficial for learning."

直译：另一种做法是把残差分支起始为零函数：令 a_m = 0，且对所有 i ∈ [m−1] 令 a_i = L^{−1/(2m−2)}。在后一方案中，残差分支不需要"遗忘"其可能糟糕的随机初始状态，这对学习有益。

注意 a_m = 0 是**乘在非零分支输出上的一个因子**（m 层复合中的最后一层），不是把整条分支的输入权重也置零。

---

### 3.3 ReZero（Bachlechner, Majumder, Mao, Cottrell, McAuley, 2020；arXiv:2003.04887）

链接：<https://arxiv.org/abs/2003.04887>

> "where α = 0 at the beginning of training. Initially the gradients for all parameters defining F vanish, but dynamically evolve to suitable values during initial stages of training."

直译：训练开始时 α = 0。最初，定义 F 的所有参数的梯度都消失，但在训练初期它们会动态演化到合适的值。

机制在论文的标量玩具模型里写得很清楚（式 8）：

> "α ← −λ L w x_0 ∂_x C(x)|_{x=x_L}"

直译：α 的更新量为 −λ·L·w·x_0·∂C/∂x（在 x = x_L 处取值）。

也就是说：**α 的梯度是"上游梯度 × 非零分支输出"，一般不为 0**；第 1 步 α 离开 0，第 2 步 F 的参数就拿到非零梯度。原文：

> "Initializing α = 0 solves both of these problems: assuming a sufficiently well-conditioned cost function, the first step of gradient descent will update the residual weights α to a value that avoids large outputs and keeps the parameter trajectory within a well-conditioned region while retaining the expressive power of the network."

直译：初始化 α = 0 同时解决这两个问题：在代价函数条件良好的前提下，梯度下降的第一步就会把残差权重 α 更新到一个避免输出过大、并使参数轨迹保持在良态区域的值，同时保留网络的表达能力。

**这与本实现的关键差别**：ReZero 只零一个标量 α，且 α 乘在**非零**的 F(x) 上；本实现是把 F 内部的所有新权重（输入侧和输出侧）都置零，F(x) 本身就是 0，`∂L/∂α` 也变成 0。

---

### 3.4 LayerScale（Touvron et al., 2021；arXiv:2103.17239）

链接：<https://arxiv.org/abs/2103.17239>

> "Formally, we add a learnable diagonal matrix on output of each residual block, initialized close to (but not at) 0."

直译：我们在每个残差块的输出上加一个可学习对角矩阵，初始化为接近 0（但不等于 0）。

而且他们明确说，即便只是把标量从 0 改成小 ε 也更好：

> "ReZero converges, we show (column α = ε) that it is better to initialize α to a small value instead of 0, as in LayerScale."

直译：ReZero 能收敛；我们展示（α = ε 那一列），把 α 初始化为一个小值而不是 0（如 LayerScale 所做）会更好。

> "ReZero simply initializes this parameter to α = 0."

直译：ReZero 只是把这个参数初始化为 α = 0。

**判读**：LayerScale 连"零标量"都不推荐，宁可用 1e-6 这种小非零值；这与"把整条新分支的权重置零"相距更远。

---

### 3.5 Network Morphism 的"单侧置零"（Wei, Wang, Rui, Chen, 2016；ICML 2016；arXiv:1603.01670）

链接：<https://arxiv.org/abs/1603.01670>

这是文献中**唯一**明确采用"置零"的函数保真构造，但它只允许**一侧**为零，另一侧必须随机非零。

原文（§3.4.1 Width Morphing，式 18 之后）：

> "It is obvious that we can either set F̃_l(c̄_l, c_{l−1}) or F̃_{l+1}(c_{l+1}, c̄_l) to 0, and the other can be set arbitrarily. Following the non-sparse practice, we set the one with less parameters to 0, and the other one to random noises."

直译：显然，我们可以把 F̃_l(c̄_l, c_{l−1}) 或 F̃_{l+1}(c_{l+1}, c̄_l) 之一置 0，另一个可以任意取值。遵循非稀疏原则，我们把参数较少的那一个置 0，另一个设为随机噪声。

原文（§3.2，明确反对零填充）：

> "Since the solutions to Equation (6) might not be unique, we shall make the morphism operation to follow the desired practices that: 1) the parameters will contain as many non-zero elements as possible, and 2) the parameters will need to be in a consistent scale. These two practices are widely adopted in existing work, since random initialization instead of zero filling for non-convex optimization problems is preferred (Bishop, 2006)..."

直译：由于式 (6) 的解可能不唯一，我们让 morphism 操作遵循以下期望做法：1) 参数包含尽可能多的非零元素；2) 参数保持一致的尺度。这两种做法在现有工作中被广泛采用，因为在非凸优化问题中，人们偏好随机初始化而不是零填充（Bishop, 2006）……

还提到零会成簇、需要打散：

> "The zeros and random noises in F̃_l and F̃_{l+1} may be clustered together. To break this unwanted behavior, we perform a random permutation on c̄_l, which will not change B_{l+1}."

直译：F̃_l 与 F̃_{l+1} 中的零和随机噪声可能聚在一起。为打破这种不良行为，我们对 c̄_l 做随机置换，这不会改变 B_{l+1}。

**这就是"正确置零"的模板**：零只出现在一侧，另一侧随机非零；而且作者仍然偏好"非零元素尽可能多"。

---

### 3.6 机制：为什么单侧零可活、双侧零必死

设某线性层 `y = W x`，损失为 L。

1. **对权重元素的梯度**：`∂L/∂W[i,j] = (∂L/∂y_i) · x_j`。
2. **单侧零（只零新输出行，新输入列随机非零）**：新行 `W[i_new, :] = 0`，但 `x` 的旧维度非零、且下一层从该新维度取数的列非零 → `∂L/∂y_{i_new} ≠ 0`，`x_j ≠ 0`（旧列）→ **新行梯度非零**，第一步就更新。新输入列 `W[:, j_new] = randn` 非零，但 `x_{j_new} = 0`（因为上一层的该维度输出为 0）→ 新列梯度为 0；不过新行一旦更新，下一层/本层该维度的输出就非零，新列随即获得梯度。**可恢复。**
3. **单侧零（只零新输入列，新输出行随机非零）**：新列 `W[:, j_new] = 0`，新行随机非零；`x_{j_new}` 来自上一层，若非零则新列梯度非零；若上一层该维也是 0，则同样存在一步延迟，但只要链路上有一侧非零即可解锁。
4. **双侧零（本实现）**：新行 `W[i_new, :] = 0` 且新输入列 `W[:, j_new] = 0`；同时上一层的对应输出行也是 0、下一层的对应输入列也是 0，LayerNorm 的 `gamma_new = 0` 又把该维的反向通路乘 0。于是对任意新维度 k：
   - 前向：`x_k = 0` 恒成立（embedding 新列为 0 → 第一层输入为 0；每层新输出行 × 任意输入 = 0）。
   - 反向：`∂L/∂W[i_new, j] = (∂L/∂y_{i_new}) · x_j`，而 `∂L/∂y_{i_new}` 要穿过下一层该维的列（=0）或 LN 的 gamma（=0），故为 0；`∂L/∂W[i, j_new] = (∂L/∂y_i) · x_{j_new}`，而 `x_{j_new} = 0`，故为 0。
   - 每个新参数都等于"非零量 × 0"或"0 × 非零量"——**精确 0，且没有任何一项能先动起来**。
5. **LayerNorm 的 `gamma_new = 0` 是额外的一刀**：即使新维度侥幸非零，`∂L/∂x_new = gamma_new · (...) = 0`，反向也回不去；而 `∂L/∂gamma_new = (∂L/∂y_new) · x_new_norm`，在 `x_new = 0` 时同样为 0。所以 gamma 自己也永远学不到 1。
6. **embedding 新列为 0**：本身不必然致命——若第一层新输入列非零，`∂L/∂E[:, j_new] = (∂L/∂x_{j_new}) · 1` 可非零。但在本实现里第一层新输入列也是 0，于是 embedding 新列同样恒零。

这就是"单侧零是延迟一步、双侧零是永久死亡"的数学区别。ReZero/Fixup/LayerScale 的零都只作用在**一个会被非零量相乘的因子**上；本实现的零作用在**乘积的两侧**，且被 LN gate 再乘一次 0。

---

## 4. Q3：有没有文献明确记录"两侧同时置零/零初始化导致单元永久死亡、梯度消失"

### 4.1 最直接的记录：StackBERT（Gong et al., 2019）报告"零初始化恒等层"失败

见 §2.3 引文。要点重述：

> "simply setting the feed-forward layer to be near zero and randomly initializing the self-attention layer is a function-preserving initialization, but it is ineffective as most parameters in the self-attention layer stay untrained."

直译：简单地把前馈层设成接近 0、并随机初始化自注意力层，是一种函数保真初始化，但它是无效的，因为自注意力层中的大多数参数始终得不到训练。

> "the heads in models trained by identity and identity+noise fail to learn such information."

直译：用 identity 与 identity+noise 训练的模型中的注意力头未能学到这类信息。

这是**生长语境下**、**函数保真**、**零初始化**、**参数学不动**四个要素同时出现的文献记录。注意：它零的是残差分支的输出投影（单侧），不是双侧；即便如此仍被报告为失败。**双侧零没有理由更好。**

### 4.2 bert2BERT 报告"函数保真带来对称，需要额外手段打破"

见 §2.2 引文。`"the knowledge that comes from adjacent layers can break the symmetry ... appeared in FPI"`、`"FPI makes the attention patterns in the same layer repeated, which is redundant and called symmetry"`、`"the model needs some optimization time to break this symmetry"`。

这是"函数保真只保输出、不保（新参数的）可学习性"的最接近表述：FPI 保证输出一致，但复制出来的单元彼此对称，必须靠 AKI（用相邻层参数）去打破；否则收敛变慢。

### 4.3 Net2Net 报告"复制出来的单元必须加噪破对称"

见 §2.1 引文：`"one should add a small amount of noise to all but the first copy of each column of weights"`。

### 4.4 Network Morphism 明确说"零会成簇，要随机置换打散"，并偏好非零填充

见 §3.5 引文：`"random initialization instead of zero filling for non-convex optimization problems is preferred (Bishop, 2006)"`、`"we perform a random permutation on c̄_l"`。

### 4.5 需要区分的相邻文献：dying ReLU

"Dying ReLU"（如 Xu et al., 2015, *Empirical Evaluation of Rectified Activations in Convolutional Network*, arXiv:1505.00853）记录的是：大学习率下 ReLU 的输入长期为负，输出恒 0、梯度恒 0，单元永久死亡。**机制不同**：那是"激活函数在负半轴的导数恒 0"，本实现是"乘法两侧同时为 0"。但两者共享同一个后果判据：**一旦某单元的梯度从第 0 步起恒为 0，它就不会再被任何后续更新激活**。（此文我只核对了标题与摘要层面的信息，未逐字引用原文，故不列入结论表。）

### 4.6 结论

- **没有找到**任何一篇论文逐字讨论"新增权重行与新增列同时置 0 → 该维度永久死亡"这一具体构造；也**没有找到**任何论文把"双侧置零"当作函数保真生长的正确做法。
- **找到了等价的负面记录**：StackBERT 明确报告零初始化恒等层"most parameters ... stay untrained"；bert2BERT 明确报告 FPI 的对称性会阻碍收敛、需要 AKI；Net2Net 明确要求加噪破对称；Network Morphism 明确说零填充不如随机初始化、零会成簇需要打散。
- 因此对"会不会导致死参数"的回答是：**会，而且比文献里被批评的单侧零更严重——单侧零只是延迟一步，双侧零是恒零。**

---

## 5. Q4：最终裁决

### 5.1 结论表

| 论文 | 生长类型 | 新参数怎么初始化 | 原文引用（英文） | 是否支持"两侧置 0" |
|---|---|---|---|---|
| Net2Net (Chen et al., 2015, arXiv:1511.05641) | 宽度 | 随机复制旧列；下一层对应行除以复制次数；其余副本加少量噪声 | "the first n columns of W(i) are copied directly into U(i) ... we must account for the replication by dividing the weight by replication factor"；"one should add a small amount of noise to all but the first copy of each column of weights" | **否**（复制 + 缩放 + 加噪） |
| Net2Net (同上) | 深度 | 新层权重 = 单位矩阵 | "The new matrix U is initialized to an identity matrix, but remains free to learn to take on any value later." | **否**（单位阵） |
| bert2BERT (Chen et al., 2022, arXiv:2110.07143) | 宽度 | 复制旧行/列，按复制次数缩放；embedding 只复制列；LN 的 gamma/beta 按索引复制 | "Ũ(i,*) = (1/C_{g_in(i)}) W(g_in(i),*)"; "For the embedding matrix U^E, we only conduct the out-dimension expansion"; "U^LN_j = W^LN_{g_out^2(j)}" | **否**（全复制） |
| bert2BERT (同上) | 深度 | 反复堆叠加宽后的模型 | "the main idea is to iteratively stack the widened model until its depth is equal to the target model" | **否**（堆叠） |
| StackBERT / Progressive Stacking (Gong et al., 2019, PMLR 97:2337) | 深度 | 第 i 层参数复制到第 i+L 层 | "for i ≤ L, the i-th layer and the (i + L)-th layer ... have the same parameter of the i-th layer of the trained BERT" | **否**；且明确报告零初始化恒等层失败："most parameters in the self-attention layer stay untrained" |
| MSLT (Yang et al., 2020, arXiv:2011.13635) | 深度 | 复制上一阶段顶部 N/k 层 | "we initialize the parameters of the newly added N/k layers by copying the parameters of the top N/k layers of the model trained in the previous stage" | **否**（复制） |
| LiGO (Wang et al., 2023, arXiv:2303.00980) | 宽度 + 深度 | 学习线性映射；新行/列是旧行/列的线性组合 | "Each new column/row is a linear combination of columns/rows of small model's weight matrix." | **否**（学习出的非零组合） |
| Network Morphism (Wei et al., 2016, arXiv:1603.01670) | 宽度 | **只零一侧**（参数少的那侧），另一侧随机非零；零要随机置换打散 | "we can either set F̃_l(...) or F̃_{l+1}(...) to 0, and the other can be set arbitrarily ... we set the one with less parameters to 0, and the other one to random noises"; "random initialization instead of zero filling ... is preferred" | **否**（只允许单侧零） |
| Fixup (Zhang et al., 2019, arXiv:1901.09321) | 残差分支 | 只零分支最后一层；其余层标准随机初始化 | "Initialize the classification layer and the last layer of each residual branch to 0." | **否**（单侧零，输入非零） |
| ReZero (Bachlechner et al., 2020, arXiv:2003.04887) | 残差门控 | α = 0 标量，乘在非零分支上 | "α = 0 at the beginning of training. Initially the gradients for all parameters defining F vanish, but dynamically evolve to suitable values" | **否**（零是标量，另一侧非零） |
| LayerScale (Touvron et al., 2021, arXiv:2103.17239) | 残差门控 | 对角阵初始化为 1e-6 / 1e-5 等小非零值 | "initialized close to (but not at) 0"；"it is better to initialize α to a small value instead of 0" | **否**（连零标量都不推荐） |
| GPT-2 (Radford et al., 2019) | 无生长 | `wte` N(0, 0.02)、`wpe` N(0, 0.01)、`c_proj` N(0, 0.02)；残差层按 1/√N 缩放 | 代码："initializer=tf.random_normal_initializer(stddev=0.02)"；论文："We scale the weights of residual layers at initialization by a factor of 1/√N" | **否**（无零初始化；委托描述的前提不成立） |
| **被审查实现** | 宽度 | 新行 + 新列 + LN gamma/beta + embedding 新列**全部 0**，且新维度不做归一化 | —（无对应论文） | **无先例** |

### 5.2 如果没有先例，最接近的正确做法是什么？差在哪一行代码？

**最接近的正确做法有两类：**

**A. Network Morphism 式"单侧零"（最接近本实现的意图，但只允许零一侧）**：宽度生长时，把新输入侧设为随机非零、只把新输出侧置 0（或反过来，把参数少的那侧置 0），这样函数保真，且至少有一半新参数立刻有梯度，另一半在一步后解锁。

**B. Net2Net / bert2BERT 式"复制 + 缩放 + 加噪"**：新行/新列复制旧行/旧列并按复制次数缩放，再对副本加小噪声破对称；LN 的 gamma/beta 也按索引复制；embedding 新列复制旧列。这是 Transformer 生长里被验证过能收敛的做法。

**差在哪几行代码（伪代码对比）：**

```python
# ── 被审查实现（死参数）─────────────────────────────
w = zeros(out_new, in_new)
w[:out_old, :in_old] = w_old
# 新行 = 0 且 新列 = 0  →  ∂L/∂W[新行] = 上游(0) × x，∂L/∂W[新列] = 上游 × x_new(0)
# 两边都是 0，永远解不开

gamma_new = 0      # ∂L/∂x_new = gamma × ... = 0；∂L/∂gamma = 上游 × x_new(0) = 0
beta_new  = 0
E_new[:, d_old:] = 0   # 第一层新输入列也是 0 → embedding 新列梯度也为 0

# ── 做法 A：Network Morphism 单侧零（函数保真，且可学）──
w = empty(out_new, in_new)
w[:out_old, :in_old] = w_old
w[:out_old, out_old:] = randn(...) * scale   # 新输入侧：随机非零
w[out_old:, :]         = 0                   # 只零新输出侧 → 第一步就有梯度
# 或者反过来：w[out_old:, :] = randn(...); w[:out_old, out_old:] = 0
# 关键：绝不能两侧同时为 0

# ── 做法 B：Net2Net / bert2BERT 复制（Transformer 上验证过）──
idx = randint(0, out_old, out_new - out_old)
w[:out_old, out_old:] = w_old[:, idx]          # 复制旧列
w[out_old:, :]         = w_old[idx, :] / count # 复制旧行 + 按复制次数缩放
w[out_old:, :]        += noise * mask          # 加小噪声破对称

gamma_new = gamma_old[idx]   # ≈1，绝不是 0
beta_new  = beta_old[idx]    # 0 可以，但必须和 gamma 一起复制
E_new[:, d_old:] = E_old[:, idx]   # 或 randn，绝不是 0
```

**一句话**：把 `w[out_old:, :] = 0` 和 `w[:, out_old:] = 0` 同时写，是致命的那一行；文献只允许其中一条成立，另一条必须是复制/随机/学习出的非零值；`gamma_new = 0` 应改为复制旧 gamma（≈1），`E_new[:, d_old:] = 0` 应改为复制或随机。

### 5.3 有没有论文明确说"函数保真必须只保输出、不保梯度，因此需要别的手段来打破对称"？

- **没有找到逐字这样的句子。** 不存在一篇论文用"function-preserving only preserves outputs, not gradients"这种措辞下结论。
- **最接近的是 bert2BERT**：它说 FPI"ensures that the initialized large model has almost the same behavior with the small model"（保证输出行为一致），但随后明确说 FPI 会带来 symmetry、`"the model needs some optimization time to break this symmetry"`，并用 AKI 去打破。这实际上就是"函数保真只保输出、不保新参数的可区分性/可学习性，需要额外手段破对称"的工程表述。
- **第二接近的是 StackBERT**：它把"零初始化 + 随机初始化"的组合称为 strictly function-preserving，但实测"most parameters ... stay untrained"、"fail to learn such information"——即函数保真成立、学习失败。
- **Net2Net** 用"加噪"、**Network Morphism** 用"随机置换 + 偏好非零"来破对称，属于同一逻辑的不同实现。

因此，若要在报告中下结论，建议表述为：**"函数保真只约束输出，不约束新参数能否收到梯度；文献的共识是必须靠复制+缩放+噪声/相邻层参数/学习算子来保证新参数可学，而不是靠置零。"** 这是对多篇原文的归纳（【我的转述】），不是任何单篇的原话。

---

## 6. 我没有核实的部分

1. **arXiv 编号纠正**：委托给的 1908.10951 是 math.DG 论文（*Higher order distance-like functions and Sobolev spaces*），2002.02925 是 *BERT-of-Theseus*（压缩），2208.00317 不是 LiGO。正确对应：StackBERT = PMLR 97:2337-2346（未在 arXiv 检索到）；MSLT = arXiv:2011.13635；LiGO = arXiv:2303.00980。以上均已用 arXiv abs 页/API 核对标题与作者。
2. **"MSG: Multi-Stage Generation" 这个名字未找到**。bert2BERT 引用的 "MSLT (Yang et al., 2020)" 实为 *Multi-stage Layerwise Training*（即 Progressively Stacking 2.0）。我没有找到任何叫 "Multi-Stage Generation" 的生长论文；若委托方另有出处，请提供。
3. **没有找到任何"两侧同时置零"的先例**，也没有找到逐字讨论该失效的论文；Q3 的结论建立在 StackBERT 的"零初始化恒等层失败"、bert2BERT 的"对称性"、Net2Net 的"加噪"、Network Morphism 的"反零填充"这四条**相邻证据**上，而非一篇直接命中的论文。
4. **GPT-2 的 `wte`/`c_proj` 零初始化无法核实**：官方 `src/model.py` 与论文都显示随机初始化 / 1/√N 缩放。若委托方指的是某个第三方复现或某篇生长论文里的 GPT-2 实验，请给出链接，我再核。
5. **SkipInit 原文未取到**：我尝试的 arXiv:2002.09758 是 *Unsupervised Question Decomposition for QA*，不是 SkipInit；本报告只在 LayerScale 的转述里提到 SkipInit（"ReZero/Skipinit ... add a learnable scalar initialized to α = 0"），未引用 SkipInit 原文。
6. **dying ReLU 原文未逐字核对**（Xu et al., 2015, arXiv:1505.00853），只据标题/摘要层面判断其机制与本例不同，故未写入结论表。
7. **未复现"67% 参数为 0、梯度恒 0"**：这是委托方给出的观测，我没有运行任何训练或梯度检查；本报告只从数学上论证该现象必然发生，并核对文献做法。
8. **版次差异**：Net2Net 用的是 arXiv 版（ar5iv 渲染），bert2BERT 同时核对了 arXiv 版与 ACL 2022 正式版（两者关键句一致）；LiGO 用的是 arXiv 版（ICLR 2023）；未逐字比对 camera-ready 与 arXiv 的每一处措辞差异。
9. **未核实的转述**：§5.3 最后那句归纳（"函数保真只约束输出，不约束新参数能否收到梯度"）是【我的转述】，不是原文；引用时请勿加引号。
