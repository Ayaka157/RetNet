# 04 Complete Mathematical Model and Loss of the Currently Submitted Model

> Citation convention as in `03_architecture_changes.md`: GB/T 7714—2015, sequential coding
> system; in-text citations are marked `[number]page (section)`, where page means the page index
> of the corresponding PDF in `Papers/`. Numbering in this document is reassigned by **order of
> first citation within this document**; a cross-reference table against
> `03_architecture_changes.md` / `Papers/` is appended at the end.
> Every equation is annotated with two counterparts: the function in the code
> (`retnet_hf/modeling_retnet.py` and others), and the same-named subsection in
> `01_function_reference.md` (search by the qualified function name, independent of line numbers).
> Notation follows `checkpoints/retnet_500m_v21_export/config.json`.

---

## 0.1 Master Table of Equations and Code

| Section | Mathematical content | Code (file::function/method) | Subsection in `01_function_reference.md` |
|---|---|---|---|
| 1 | Forward pass overview | `retnet_hf/modeling_retnet.py::RetNetBlock._block_forward`, `RetNetForCausalLM._forward_hidden` | same name |
| 2 | Embedding and embed_scale | `RetNetForCausalLM.__init__`, `RetNetForCausalLM._forward_hidden` | same name |
| 3.1 | LayerNorm | `RetNetBlock._norm` | same name |
| 3.2 | GrowthLayerNorm | `GrowthLayerNorm.__init__`, `GrowthLayerNorm.forward` | same name |
| 4.1 | Projections, scaling | `MultiScaleRetention.__init__`, `MultiScaleRetention._project_and_rotate` | same name |
| 4.2 | xPos rotation | `rotate_every_two`, `theta_shift`, `MultiScaleRetention._rotations` | same name |
| 4.3 | Multi-scale γ | `MultiScaleRetention.__init__` | same name |
| 4.4 | Three representations | `MultiScaleRetention._parallel_retention`, `._recurrent_retention`, `._chunkwise_retention`, `.forward` | same name |
| 4.5 | Closed-form scan | `retnet_hf/retention_fast.py::chunkwise_batched` | same name |
| 4.6 | GroupNorm and gate | `MultiScaleRetention._head_group_norm`, `MultiScaleRetention.forward` | same name |
| 5 | FFN / SwiGLU | `RetNetBlock.__init__`, `RetNetBlock._block_forward` | same name |
| 6 | Output head and tied embedding | `RetNetForCausalLM.__init__`, `.tie_weights`, `._forward_hidden` | same name |
| 7.1 | Weighted cross entropy | `RetNetForCausalLM.forward`, `_chunked_weighted_ce` | same name |
| 7.2 | Chunked CE | `_chunked_weighted_ce` | same name |
| 7.3 | Fused CE | `retnet_hf/fused_ce.py::_fused_forward`, `_fused_backward`, `FusedLinearCrossEntropy.forward`, `.backward`, `fused_linear_cross_entropy` | same name |
| 8.1 | packed data | `retnet_hf/data.py::PackedChatDataset.__getitem__`, `_PackedStage.row` | same name |
| 8.2 | flat data | `_FlatStage.row`, `_FlatStage._check_boundaries` | same name |
| 9 | Actual weight state | `retnet_hf/train.py::live_param_ratio`, `_preflight` | same name |
| 10 | Training objective | `retnet_hf/train.py::main`, `CompileSafeTrainer.get_decay_parameter_names`; `retnet_hf/plan.py::should_accept_reschedule` (plan.py is not among the 7 files covered by `01_function_reference.md`) | same name except plan.py |

> "same name" means the subsection in `01_function_reference.md` whose heading is that qualified
> function name (for example `MultiScaleRetention._chunkwise_retention`), where a line-by-line
> explanation and the complete code block can be found. Line numbers in
> `01_function_reference.md` move with the code, so searching by function name is enough.

## 0. Notation and Configuration

| Symbol | Meaning | Current value |
|---|---|---|
| $B$ | Batch size (per device) | configuration item |
| $L$ | Sequence length | 2048 |
| $D$ | `d_model` | 1408 |
| $n_l$ | Number of layers | 20 |
| $H$ | Number of heads | 22 |
| $K$ | `key_dim = D/H` | 64 |
| $V$ | `value_dim` | 2816 |
| $V_h$ | `value_head_dim = V/H` | 128 |
| $V_v$ | Vocabulary size | 20000 |
| $C$ | chunk size | 128 |
| $d_{\text{old}}$ | `growth_old_dim` | 768 |

The input is a sequence of token ids $x_{1:L}\in\{0,\dots,V_v-1\}^L$. The output is the logits at
each position $\ell_{1:L}\in\mathbb{R}^{V_v}$ together with the scalar loss $\mathcal{L}$.
Code entry point: `RetNetForCausalLM.forward`; see the same-named subsection in
`01_function_reference.md`.

## 1. Forward Pass Overview

$$
\begin{aligned}
e_t &= E[x_t]\cdot s_e, \qquad s_e=\texttt{embed\_scale}\ (\text{default }\sqrt{D}) \\
h^{(0)}_t &= \mathrm{Dropout}(e_t) \\
\tilde h^{(l-1)} &= \mathrm{Norm}_1^{(l)}\big(h^{(l-1)}\big) \\
u^{(l)} &= \mathrm{Dropout}\Big(W_O^{(l)}\big(\mathrm{SiLU}(g^{(l)})\odot
          \mathrm{GroupNorm}(\mathrm{Ret}(q,k,v))\big)\Big) \\
h'^{(l)} &= h^{(l-1)} + u^{(l)} \\
\tilde h'^{(l)} &= \mathrm{Norm}_2^{(l)}\big(h'^{(l)}\big) \\
f^{(l)} &= \mathrm{FFN}^{(l)}\big(\tilde h'^{(l)}\big) \\
h^{(l)} &= h'^{(l)} + \mathrm{Dropout}\big(f^{(l)}\big) \\
h^{\text{final}} &= \mathrm{FinalNorm}\big(h^{(n_l)}\big) \\
\ell &= h^{\text{final}} E^{\top}
\end{aligned}
$$

$E\in\mathbb{R}^{V_v\times D}$ is shared with the output head (`tie_weights=True`).

> **Placement of dropout** (the older version of the equations drew this wrongly): the order
> inside MSR is `Retention → GroupNorm → SiLU(g) gate → Dropout → out_proj`,
> that is, dropout acts **before out_proj**; the residual sum `h' = h + u` has **no** dropout.
> The dropout on the FFN branch comes before the residual sum and acts on `f^{(l)}`. The
> authoritative sources are `modeling_retnet.py::MultiScaleRetention.forward` and
> `RetNetBlock._block_forward`.
> **Corresponding code**: `RetNetBlock._block_forward`, `RetNetForCausalLM._forward_hidden`;
> in `01_function_reference.md`, `RetNetBlock._block_forward` and
> `RetNetForCausalLM._forward_hidden`.

## 2. Embedding

The word embedding $E\in\mathbb{R}^{V_v\times D}$ is initialized with `nn.init.normal_(std=0.02)`;
the forward pass first multiplies by `embed_scale` and then applies dropout:

$$
e_t = E[x_t]\cdot s_e .
$$

`embed_scale` is inherited explicitly during growth (currently $\sqrt{768}$, not $\sqrt{1408}$),
because growth does not change the embedding scale of the old dimensions. Code:
`RetNetForCausalLM.__init__` and `_forward_hidden`; see the same-named subsection in
`01_function_reference.md`.

> **Corresponding code**: `retnet_hf/modeling_retnet.py::RetNetForCausalLM.__init__` (the default
> value of `embed_scale` and its inheritance), `RetNetForCausalLM._forward_hidden`
> (`x = self.embedding(input_ids) * self.embed_scale` → dropout).
> In `01_function_reference.md` the two sections `RetNetForCausalLM.__init__` and
> `RetNetForCausalLM._forward_hidden` correspond.

## 3. Normalization

### 3.1 Plain Layers: LayerNorm

For a model that has never grown, $\mathrm{Norm}$ is a standard LayerNorm (pre-LN):

$$
\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad
\mu=\frac1D\sum_i x_i,\ \ \sigma^2=\frac1D\sum_i (x_i-\mu)^2 .
$$

> **Corresponding code**: `RetNetBlock._norm` returns `nn.LayerNorm` when `use_growth_ln=False`;
> in `01_function_reference.md`, `RetNetBlock._norm`.

### 3.2 Grown Layers: GrowthLayerNorm

The current 0.5B weights have `use_growth_ln=True`, and normalization is split into two segments:

$$
\mathrm{GLN}(x)=
\Big[\ \mathrm{LN}_{d_{\text{old}}}(x_{1:d_{\text{old}}})\ \Big\|\
x_{d_{\text{old}}+1:D}\odot\gamma_{\text{new}}+\beta_{\text{new}}\ \Big]
$$

where $\gamma_{\text{new}},\beta_{\text{new}}\in\mathbb{R}^{D-d_{\text{old}}}$; **the new segment
is not normalized**, it only receives a per-channel affine transform. In this way the mean/variance
of the old segment is determined by the old dimensions alone, and width growth does not dilute the
normalization statistics of the old dimensions. Code: `GrowthLayerNorm.forward`; in
`01_function_reference.md`, `GrowthLayerNorm.__init__` / `GrowthLayerNorm.forward`.

The motivation for function preservation comes from the width transformation of Net2Net (old
weights placed in the top-left corner and scaled by the number of copies) [1]3 (§2.3), the width
morphism of Network Morphism (zero on one side only, random on the other) [2]6 (§3.4.1), and the
FPI of bert2BERT (copy and scale; LayerNorm parameters are copied rather than zeroed) [3]4
(§3.3.1). The risk of zero-initializing one side is documented: StackBERT points out that
"setting the feed-forward layers close to zero and initializing the attention layers randomly"
does preserve the function, yet "most parameters in the self-attention layers are never trained"
[4]2 (§2); ReZero shows that a zero-initialized scalar gate is still learnable precisely because
it is a **learnable scalar** and not a two-sided zero [5]3 (§3); Fixup, arguing from variance
preservation, requires the residual branch to be initialized scaled by depth [6]4 (§3). In the
current weights of this project $\gamma_{\text{new}}$ is all zero (measured over 41 tensors), so
the new segment is identically zero after each layer's normalization, which is also the origin of
the "effective width" in Section 9.

> **Corresponding code**: `retnet_hf/modeling_retnet.py::GrowthLayerNorm.__init__` (the definitions
> of `ln_old`, `gamma_new`, `beta_new`), `GrowthLayerNorm.forward`
> (`new = x[..., old:] * self.gamma_new + self.beta_new`), `RetNetBlock._norm` (the
> `use_growth_ln` branch).
> In `01_function_reference.md`, `GrowthLayerNorm.__init__`, `GrowthLayerNorm.forward` and
> `RetNetBlock._norm` correspond.

## 4. Retention (MSR)

### 4.1 Projections and Scaling

$$
q = W_Q\,\tilde h,\quad k = W_K\,\tilde h,\quad v = W_V\,\tilde h,\quad g = W_G\,\tilde h,
$$

where $W_Q,W_K\in\mathbb{R}^{D\times D}$ and $W_V,W_G\in\mathbb{R}^{V\times D}$, all without
bias; $q,k$ are reshaped to $[B,H,L,K]$, $v$ to $[B,H,L,V_h]$, and $g$ to $[B,L,V]$.
$k$ is first multiplied by $K^{-1/2}$:

$$
k \leftarrow k\cdot K^{-1/2}.
$$

> **Corresponding code**: `MultiScaleRetention.__init__`, `_project_and_rotate`;
> see the same-named subsections in `01_function_reference.md`.

### 4.2 xPos: Rotary Positional Encoding with Decay

For every head, every position $t$ and every dimension pair $(2i,2i+1)$, define the frequency

$$
\theta_{t,i}=t\cdot 10000^{-i/(K/2)},\qquad i=0,\dots,K/2-1 .
$$

The rotation operator maps two adjacent dimensions to $R(\theta)(a,b)=(-b,a)$, hence

$$
\mathrm{xPos}(x)_t = x_t\odot\cos\theta_t + R(\theta_t)\,x_t\odot\sin\theta_t ,
$$

and the transform is applied to both $q$ and $k$ (to $k$ after the scaling). Code:
`rotate_every_two`, `theta_shift`, `MultiScaleRetention._rotations`; see the same-named
subsections in `01_function_reference.md`.
This form comes from xPos: the rotation makes the inner product depend only on the relative angle,
and the exponential decay keeps long-range inner products stable [7]3 (§3.2); the original RetNet
paper likewise adopts xPos [8]3 (§2.1).

> **Corresponding code**: `retnet_hf/modeling_retnet.py::rotate_every_two` (two adjacent dimensions
> → `(-x1, x0)`), `theta_shift` (`x·cos + rotate(x)·sin`), `MultiScaleRetention._rotations`
> (`inv_freq` and the sin/cos tables).
> In `01_function_reference.md`, `rotate_every_two`, `theta_shift` and
> `MultiScaleRetention._rotations` correspond.

### 4.3 Multi-Scale Decay

One scalar $\gamma_h$ per head, with initial value

$$
\gamma_h = 1-2^{-5-h},\qquad h=0,\dots,H-1,
$$

which yields a geometric series of half-lives from short to long. In the original RetNet $\gamma$
is fixed [8]4 (§2.2); this project makes it a learnable parameter (`use_gla=True`), still
initialized with the formula above:

$$
\gamma_h \in \mathbb{R}\ \text{(a `nn.Parameter`, initialized to }1-2^{-5-h}\text{)}.
$$

The justification is GLA: data-dependent gating outperforms a globally fixed decay [9]4 (§4.1).
> **Corresponding code**: `MultiScaleRetention.__init__`; see the same-named subsection in
> `01_function_reference.md`.

Define the decay matrix (for $n\ge m$)

$$
D_{nm}=\gamma_h^{\,n-m},\qquad D_{nm}=0\ (n<m).
$$

> **Corresponding code**: `retnet_hf/modeling_retnet.py::MultiScaleRetention._decay_mask` (the `D`
> mask of the parallel path), `MultiScaleRetention._chunkwise_retention` (the intra-chunk
> `inner_mask` and the cross-chunk `gamma^{t+1}`).
> In `01_function_reference.md`, `MultiScaleRetention._decay_mask` and
> `MultiScaleRetention._chunkwise_retention` correspond.

### 4.4 Three Equivalent Representations

**Parallel** (training history, `_parallel_retention`):

$$
o_t=\sum_{m\le t}\big(q_t^{\top}k_m\big)\,D_{tm}\,v_m .
$$

**Recurrent** (per-token decoding, `_recurrent_retention`):

$$
S_t=\gamma_h S_{t-1}+k_t^{\top}v_t,\qquad o_t=q_t S_t,\qquad S_0=0,
$$

where $S_t\in\mathbb{R}^{K\times V_h}$.

**Chunked** (main training path, `_chunkwise_retention`): the length $L$ is cut into $n_c=L/C$
chunks, and within chunk $i$ the positions are $t=0..C-1$:

$$
\begin{aligned}
S_i &= \gamma_h^{C} S_{i-1} + \sum_{j=0}^{C-1}\gamma_h^{C-1-j}\,k_{i,j}^{\top}v_{i,j},\\
o_{i,t} &= \underbrace{\sum_{s\le t}\gamma_h^{\,t-s}q_{i,t}^{\top}k_{i,s}\,v_{i,s}}_{\text{intra-chunk}}
+ \underbrace{\gamma_h^{\,t+1}q_{i,t}\,S_{i-1}}_{\text{inter-chunk}} .
\end{aligned}
$$

All three paths share the same set of weights and the same $\gamma$. Code:
`MultiScaleRetention.forward` dispatches on `mode`; in `01_function_reference.md`,
`_parallel_retention`, `_recurrent_retention`, `_chunkwise_retention` and `forward` correspond.
The chunked form agrees with the causal chunking of linear attention [10]3-4 (§3.2–§3.3); the
chunked form of GLA is given in [9]5 (§4.2); the original RetNet paper gives this representation
[8]4 (§2.2).

> **Corresponding code**: `retnet_hf/modeling_retnet.py::MultiScaleRetention._parallel_retention`,
> `._recurrent_retention`, `._chunkwise_retention`, `.forward` (dispatch on `mode`).
> In `01_function_reference.md` these four same-named subsections correspond; among them
> `_chunkwise_retention` is mathematically equivalent to `retention_fast.chunkwise_batched`.

### 4.5 Closed-Form Scan Used in Training

`retention_fast.chunkwise_batched` does not recurse chunk by chunk; it computes the states
preceding all chunks in one go. Let
$U_i=k_i^{\top}\big(v_i\odot\gamma_h^{C-1-j}\big)\in\mathbb{R}^{K\times V_h}$, then

$$
S_{\text{prev}}[i]=\gamma_h^{iC}S_0+\sum_{j<i}\gamma_h^{\,i-1-j}U_j ,
$$

written in matrix form: $T[h,i,j]=\gamma_h^{\,i-1-j}$ (for $i>j$, otherwise 0),

$$
S_{\text{all}}=T\,U_{\text{flat}}+\mathrm{diag}\big(\gamma_h^{iC}\big)S_0 ,
\qquad S_{\text{prev}}=S_{\text{all}}[:n_c],\quad S_{\text{new}}=S_{\text{all}}[n_c].
$$

Then the intra-chunk and cross-chunk terms are computed:

$$
o=\big(qk^{\top}\odot D^{\text{inner}}\big)v+\big(q\odot\gamma_h^{\,j+1}\big)S_{\text{prev}} .
$$

> **Corresponding code**: `chunkwise_batched`; see the same-named subsection in
> `01_function_reference.md`. It is mathematically equivalent to `_chunkwise_retention`; the only
> difference is "one batched matrix multiplication" versus "a loop over chunks".

### 4.6 Output Gate and GroupNorm

$$
y=\mathrm{GroupNorm}_{H}\Big(\big\|_{h=1}^{H}\mathrm{head}_h\Big),\qquad
o=\big(\mathrm{SiLU}(g)\odot y\big)W_O,
$$

$W_O\in\mathbb{R}^{D\times V}$. $\mathrm{GroupNorm}_H$ means grouping by head (each group has
$V_h$ channels), with every group normalized separately; $\mathrm{SiLU}(x)=x\,\sigma(x)$, i.e.
swish.
> **Corresponding code**: `_head_group_norm`, `MultiScaleRetention.forward`; see the same-named
> subsections in `01_function_reference.md`.
The definition of grouped normalization in GroupNorm is given in §3 of the original paper [11]3
(§3); the MSR output gate of RetNet is given in [8]4 (§2.2, Eq. (8)).

## 5. Feed-Forward Network

The default is a two-layer MLP (`use_swiglu=False`):

$$
\mathrm{FFN}(z)=W_2\,\mathrm{GELU}(W_1 z),\qquad
W_1\in\mathbb{R}^{F\times D},\ W_2\in\mathbb{R}^{D\times F},\ F=2816 .
$$

The optional SwiGLU branch (not enabled by the current weights):

$$
\mathrm{FFN}_{\text{swi}}(z)=W_{\text{down}}\big(\mathrm{SiLU}(W_{\text{gate}}z)\odot W_{\text{up}}z\big).
$$

> **Corresponding code**: `RetNetBlock.__init__` and `_block_forward`; see the same-named
> subsection in `01_function_reference.md`.
SwiGLU comes from the comparative experiments on GLU variants, outperforming the GELU baseline at
an equal parameter count [12]1-2 (§2).

## 6. Output Head

The final hidden state first passes through `FinalNorm`, then shares weights with the word
embedding:

$$
\ell_t = h^{\text{final}}_t E^{\top}\in\mathbb{R}^{V_v},\qquad E=\text{embedding.weight}.
$$

> **Corresponding code**: `tie_weights` in `RetNetForCausalLM.__init__` and `_forward_hidden`;
> in `01_function_reference.md`, `RetNetForCausalLM.__init__`, `tie_weights` and
> `_forward_hidden` correspond.

## 7. Loss Function

### 7.1 Definition

Each target position $t$ carries a weight code $w_t$ (supplied by the data packet, see Section 8):

$$
w_t=\begin{cases}
0, & \text{wcode}=0\ (\text{pad})\\
1, & \text{wcode}=1\ (\text{reply})\\
p, & \text{wcode}=2\ (\text{prompt}),\ p=\texttt{prompt\_loss\_weight}=0.1\\
1, & \text{wcode}=3\ (\text{plain})
\end{cases}
$$

Per-token cross entropy and total loss:

$$
\mathrm{CE}_t=-\log\frac{\exp(\ell_{t,y_t})}{\sum_{v}\exp(\ell_{t,v})},\qquad
\mathcal{L}=\frac{\sum_{t} w_t\,\mathrm{CE}_t}{\max\big(\sum_t w_t,\,1\big)} .
$$

> **Corresponding code**: the non-fused branch of `RetNetForCausalLM.forward` calls
> `_chunked_weighted_ce`; in `01_function_reference.md`, `RetNetForCausalLM.forward` and
> `_chunked_weighted_ce` correspond.
> Note that $\mathrm{CE}$ is computed in fp32 (`logits[s:e].float()`), while the logits themselves
> may be bf16.

### 7.2 Chunked Computation (Default Path)

`_chunked_weighted_ce` splits the $B\cdot L$ positions into chunks according to a **byte budget**:
the number of tokens per chunk is derived from $\lfloor \texttt{chunk\_bytes}/(4V_v)\rfloor$
(64 MB by default; 838 tokens/chunk when $V_v=20000$), and $\mathrm{CE}$ is computed chunk by
chunk while the numerator and denominator are accumulated:

$$
\mathcal{L}=\frac{\sum_{c}\sum_{t\in c} w_t\,\mathrm{CE}_t}{\sum_c\sum_{t\in c} w_t}.
$$

This is mathematically exactly equal to computing over the whole segment, **and it really does
bring peak memory down to roughly the order of `chunk_bytes`**: with $B=16,L=2048,V_v=20000$ it
splits into 40 chunks, and each chunk materializes only $838\times V_v\times 4\approx 67$ MB
instead of the whole segment's $32768\times V_v\times 4\approx 2.6$ GB.
(The old implementation hard-coded the chunk size to 262144 tokens; that shape yields a single
chunk, which amounts to no chunking at all.)
> **Corresponding code**: `_chunked_weighted_ce` and `ce_chunk_tokens`; see the same-named
> subsection in `01_function_reference.md`.

### 7.3 Fused Path (Optional)

When the environment variable `RETNET_FUSED_CE=1` is set, the code takes
`fused_linear_cross_entropy`: the forward pass computes logits chunk by chunk and saves only
`logsumexp` and the target logit; the backward pass recomputes logits chunk by chunk and derives
$\mathrm{softmax}-\mathrm{onehot}$, so the $[N,V_v]$ tensor is never stored at all.
The loss definition is exactly the same as in 7.1; code: `FusedLinearCrossEntropy` in `fused_ce.py`;
in `01_function_reference.md`, `_fused_forward`, `_fused_backward`,
`FusedLinearCrossEntropy.forward/backward` correspond.

> **Corresponding code**: `retnet_hf/fused_ce.py::_fused_forward` (chunked computation of
> `logsumexp` and the target logit), `_fused_backward` (chunked recomputation of
> `softmax-onehot`), `FusedLinearCrossEntropy.forward` / `.backward`,
> `fused_linear_cross_entropy` (the functional entry point). The same-named subsections in
> `01_function_reference.md` correspond.

## 8. Mathematical Form of the Data

### 8.1 packed Packet

Each sample is a row $r\in\mathbb{Z}^{L+1}$ together with a weight code $c\in\mathbb{Z}^{L+1}$:

$$
\text{input}_t=r_t,\qquad \text{label}_t=r_{t+1},\qquad w_t=c_{t+1},\quad t=0,\dots,L-1 .
$$

That is, the input and the label are offset by one position, and the weight code is shifted along
with them (the weight describes "how much weight to use when predicting this label").
> **Corresponding code**: `PackedChatDataset.__getitem__`, `_PackedStage.row`; see the same-named
> subsections in `01_function_reference.md`.

### 8.2 flat Packet

A flat shard is a one-dimensional token stream, split into rows by `seq_len+1`:

$$
r^{(k)}_t = \text{stream}[k\,(L+1)+t],\qquad w_t\equiv 1 .
$$

The current `_FlatStage` spot-checks on open whether eos (id=2) exists in the shard, and refuses
if it does not; `wcodes` is always 1 for flat. Code: `_FlatStage.row`, `_check_boundaries`; see
the same-named subsections in `01_function_reference.md`.

> **Corresponding code**: `retnet_hf/data.py::_FlatStage.row` (slices by `width = seq_len+1`,
> `_wc` is always 1), `_FlatStage._check_boundaries` (spot-checks the first 2M tokens for eos=2).
> In `01_function_reference.md`, `_FlatStage.row` and `_FlatStage._check_boundaries` correspond.

## 9. Actual Mathematical State of the Current Weights

The equations above describe **what the code is capable of**; the 0.5B weights currently on disk
are a degenerate instance:

- $D=1408$, but $\gamma_{\text{new}}\equiv 0$ in `GrowthLayerNorm`, so after each layer's
  normalization the new segment is identically zero, and the effective input width is only
  $d_{\text{old}}=768$;
- layers 0–17 have effective width 768, layers 18–19 have 1024 (their old segment is split at 768,
  but the live columns extend to 1023), so the model is a mixed-width "18×d768 + 2×d1024";
- 338,120,704 parameters (67.06%) are exactly zero, and the learnable parameters are about 166M.

This is not part of the equations but a consequence of the growth initialization; its mathematical
root is precisely the zero-initialization boundary cited in Section 3.2 [4]2 (§2). For
item-by-item localization see E01–E05 in `docs/00_error_table.md`, and for the change list see
`docs/03_architecture_changes.md`.

> **Corresponding code**: `retnet_hf/train.py::live_param_ratio` (per-tensor count of
> `(p != 0).sum()`), `_preflight` (refuses to start training when the live-parameter ratio is
> <95%, and prints the gamma distribution).
> In `01_function_reference.md`, `live_param_ratio` and `_preflight` correspond; the measured
> values are in E08 of `00_error_table.md` and in the `test_live_params.py` discussion of
> `01_function_reference.md`.

## 10. Training Objective (Minimal Set Relevant to Model Mathematics)

- Optimizer: AdamW (8-bit, `adamw_bnb_8bit`), weight decay $0.1$, but **all 1-D parameters are not
  decayed** (`--no_decay_1d` is on by default), including $\gamma$ and the scale of each norm;
- Learning rate: warmup + cosine down to `min_lr`; the plan file can be hot-edited (a change of
  the total step count that makes the LR jump upwards is rejected by default);
- Precision: bf16 autocast, with the loss computed in fp32;
- Gradient clipping: $\|\nabla\|_2\le 1.0$;
- Token budget: the compute-optimal ratio of Chinchilla is used as the reference (model and tokens
  scaled in proportion) [13]4 (§3).

These are training settings, not the model definition; they are listed because whether $\gamma$ is
decayed directly changes the $\gamma_h$ trajectory of Section 4.3.

> **Corresponding code**: `retnet_hf/train.py::main` (`optim`, `lr_scheduler_type`,
> `weight_decay`, `max_grad_norm`, `bf16` of `TrainingArguments`),
> `CompileSafeTrainer.get_decay_parameter_names` (when `_no_decay_1d=True`, moves all 1-D
> parameters out of decay); the LR-rejection logic for plan hot-edits is in
> `retnet_hf/plan.py::should_accept_reschedule`, and **plan.py is not among the 7 files covered by
> `01_function_reference.md`**, its code block is given in E18 of `00F1_fix_matrix.md`.
> In `01_function_reference.md`, `main` and
> `CompileSafeTrainer.get_decay_parameter_names` correspond.

---

## References

[1] CHEN T, GOODFELLOW I, SHLENS J. Net2Net: accelerating learning via knowledge transfer[EB/OL]. (2015)[2026-10-23]. https://arxiv.org/abs/1511.05641. (local: `Papers/07_Net2Net_1511.05641.pdf`)

[2] WEI T, WANG C, RUI Y, et al. Network morphism[EB/OL]. (2016)[2026-10-23]. https://arxiv.org/abs/1603.01670. (local: `Papers/08_Network-Morphism_1603.01670.pdf`)

[3] CHEN C, YIN Y, SHANG L, et al. bert2BERT: towards reusable pretrained language models[EB/OL]. (2021)[2026-10-23]. https://arxiv.org/abs/2110.07143. (local: `Papers/09_bert2BERT_2110.07143.pdf`)

[4] GONG L, HE D, LI Z, et al. Efficient training of BERT by progressively stacking[C]//Proceedings of the 36th International Conference on Machine Learning. PMLR, 2019: 2337-2346. (local: `Papers/10_StackBERT_PMLR-v97-gong19a.pdf`)

[5] BACHLECHNER T, MAJUMDER B P, MAO H H, et al. ReZero is all you need: fast convergence at large depth[EB/OL]. (2020)[2026-10-23]. https://arxiv.org/abs/2003.04887. (local: `Papers/13_ReZero_2003.04887.pdf`)

[6] ZHANG H, DAUPHIN Y N, MA T. Fixup initialization: residual learning without normalization[EB/OL]. (2019)[2026-10-23]. https://arxiv.org/abs/1901.09321. (local: `Papers/14_Fixup_1901.09321.pdf`)

[7] SUN Y, DONG L, PATRA B, et al. A length-extrapolatable transformer[EB/OL]. (2022)[2026-10-23]. https://arxiv.org/abs/2212.10554. (local: `Papers/02_Length-Extrapolatable-Transformer_2212.10554.pdf`)

[8] SUN Y, DONG L, HUANG S, et al. Retentive network: a successor to transformer for large language models[EB/OL]. (2023)[2026-10-23]. https://arxiv.org/abs/2307.08621. (local: `Papers/01_Retentive-Network_2307.08621.pdf`)

[9] YANG S, WANG B, SHEN Y, et al. Gated linear attention transformers with hardware-efficient training[EB/OL]. (2023)[2026-10-23]. https://arxiv.org/abs/2312.06635. (local: `Papers/03_Gated-Linear-Attention_2312.06635.pdf`)

[10] KATHAROPOULOS A, VYAS A, PAPPAS N, et al. Transformers are RNNs: fast autoregressive transformers with linear attention[EB/OL]. (2020)[2026-10-23]. https://arxiv.org/abs/2006.16236. (local: `Papers/04_Transformers-are-RNNs_2006.16236.pdf`)

[11] WU Y, HE K. Group normalization[EB/OL]. (2018)[2026-10-23]. https://arxiv.org/abs/1803.08494. (local: `Papers/05_Group-Normalization_1803.08494.pdf`)

[12] SHAZEER N. GLU variants improve transformer[EB/OL]. (2020)[2026-10-23]. https://arxiv.org/abs/2002.05202. (local: `Papers/06_GLU-Variants_2002.05202.pdf`)

[13] HOFFMANN J, BORGEAUD S, MENSCH A, et al. Training compute-optimal large language models[EB/OL]. (2022)[2026-10-23]. https://arxiv.org/abs/2203.15556. (local: `Papers/15_Chinchilla_2203.15556.pdf`)

---

## Appendix: Number Cross-Reference with `03_architecture_changes.md` / `Papers/`

| This document | `03_architecture_changes.md` | Papers | Paper |
|---|---|---|---|
| [1] | [3] | 07 | Net2Net |
| [2] | [4] | 08 | Network Morphism |
| [3] | [5] | 09 | bert2BERT |
| [4] | [6] | 10 | StackBERT |
| [5] | [7] | 13 | ReZero |
| [6] | [8] | 14 | Fixup |
| [7] | [12] | 02 | xPos |
| [8] | [1] | 01 | RetNet |
| [9] | [2] | 03 | GLA |
| [10] | [11] | 04 | Transformers are RNNs |
| [11] | [13] | 05 | GroupNorm |
| [12] | [14] | 06 | GLU Variants |
| [13] | [15] | 15 | Chinchilla |
