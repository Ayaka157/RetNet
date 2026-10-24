# Mathematical & Theoretical Consistency Audit — `04_模型数学与损失.md`

**Audit target:** `E:\Workspaces\RetNet_Submissions_Review\04_模型数学与损失.md` (475 lines, 22,889 B; identical to `E:\Workspaces\RetNet\docs\04_模型数学与损失.md` and to `20261020_Submit/docs/04_模型数学与损失.md`), cross-checked against the post-fix English rewrite `20261024_Submit_pre-audit/docs/04_math_and_loss.md` (548 lines).

**Code audited:** the post-fix snapshot `20261024_Submit_pre-audit/code` (authoritative for this review), plus the real checkpoint `E:\Workspaces\RetNet\checkpoints\retnet_500m_v21_export` for the “actual state” claims in §9.

**Papers:** all 13 references named in the document, extracted from the local `Papers/` PDFs; equations and page/section numbers below are from those PDFs (RetNet, GLA, xPos, Transformers-are-RNNs, GroupNorm, GLU Variants, Net2Net, Network Morphism, bert2BERT, StackBERT, ReZero, Fixup, Chinchilla).

**Environment:** Python 3.13.7, torch 2.10.0+cu128, transformers 4.57.6 (`C:\tools\miniconda3\envs\AI`). No files in the submission were modified; all probes were run on a copy.

**Severity:** Critical = silently wrong gradients or invalidates the model; High = a core mathematical statement is false or the actual-state description materially wrong; Medium = a real deviation/edge that can bite; Low = presentation/count/version.

---

## 0. What was checked

| Layer | What was compared |
|---|---|
| Document ↔ code | every numbered equation and every “corresponding code” claim in §§1–10 against the current functions |
| Document ↔ checkpoint | the “actual state” claims in §9 against `retnet_500m_v21_export/model.safetensors` |
| Code ↔ papers | the implemented retention, position encoding, decay, normalization, gating, FFN, loss, growth and data math against RetNet, GLA, xPos, linear-attention, GroupNorm, GLU, Net2Net, Network Morphism, bert2BERT, StackBERT, ReZero, Fixup, Chinchilla |
| Runtime probes | fused-CE vs reference gradient; parallel-with-nonzero-state vs chunkwise; exact zero/nonzero counts; per-tensor live rows/columns; eos sampling in the real `pretrain_v2` shards |

**Reproduced evidence**

| Command / probe | Result |
|---|---|
| `python verify.py` (post-fix snapshot) | PASS, 9/9 checks |
| `cd code && python -m pytest tests -q` | 82 passed, 4 skipped |
| fused CE vs `F.cross_entropy`, same logits/labels | loss identical; **gradient max diff 1.45e-3**, scale 0.1876 (≈0.8 %) |
| parallel + non-zero state vs chunkwise + same state | **max diff 0.2989**; with zero state 0.0 |
| checkpoint total / zero / nonzero | 504,179,896 / **338,196,736 (67.0786 %)** / 165,983,160 |
| `gamma_new` tensors | 41 tensors × 640, **all exactly 0** |
| q/k/v/g/ffn/out live rows and columns per layer | residual stream 768 everywhere; q/k 768 (layers 0–17) and 1024 (18–19); value 1536/2048; out rows 768 |
| `pretrain_v2` eos in the first per-shard sample | all 34/37/25 shards selected for `rescue_m1/2/3` contain eos |

---

## 1. Executive summary

The **core RetNet mathematics is largely correct and matches the code**: the three retention paradigms (parallel/recurrent/chunkwise), the causal decay matrix, GroupNorm per head, the SiLU gate, the GELU FFN, tied output head, weighted cross-entropy, data shifts and the optimizer are implemented consistently with the papers (with the exceptions below). The document is nevertheless **not yet a faithful mathematical specification of the current model**:

| # | Finding | Type | Severity |
|---|---|---|---|
| **F1** | Fused CE backward does not compute `softmax − onehot`; it overwrites the target entry with `-1`. The document’s §7.3 and `01`’s prose are false. | code + doc | **Critical (optional path)** |
| **F2** | Closed-form scan exponent is missing the chunk factor `C`: the document says `γ^{i-1-j}`, the code uses `(γ^C)^{i-1-j}`. | doc | **High** |
| **F3** | Dropout is placed after `W_O` in the equation, but the code applies it before `out_proj`; the residual add has no dropout. §4.6 omits it entirely. | doc | **High** |
| **F4** | §9 conflates weight-nonzero counts with functional width. The residual stream is 768 wide in every layer, not “18×768 + 2×1024”; the effective q/k/value head count is 12/16, not 22; ~4.19 M nonzero weights are unused; the “166 M learnable” is a nonzero count. | doc + data | **High** |
| **F5** | The learnable `gamma` is a global, input-independent per-head/per-layer scalar — **not** GLA’s data-dependent `α_t`. The document’s GLA justification and the `use_gla` name are wrong. | doc + naming | **High** |
| **F6** | `gamma` is an unconstrained `nn.Parameter`; `γ>1` makes the “decay” grow and can overflow; the preflight only rejects `γ<=0`. | code | Medium |
| **F7** | `mode="parallel"` with a non-zero `state` ignores the past state in the output; the API silently returns a different function. | code | Medium |
| **F8** | §8.2 says flat `wcodes` are always 1; `_FlatStage` now accepts a `weight` (P2-2), so the statement is false. | doc | Low |
| **F9** | §9’s exact zero count `338,120,704 (67.06 %)` is stale; the checkpoint has `338,196,736 (67.0786 %)`. | doc/data | Low |
| **F10** | The rotation is RetNet’s **scalar-decay** simplification of xPos, not xPos’s per-dimension `ζ_i^n`. | doc | Medium |
| **F11** | RetNet’s parallel score-normalization factors (`D/√rowsum`, `R/max(|rowsum|,1)`) are omitted; after per-head GroupNorm they are mathematically no-ops, but the omission is a numerical-stability tradeoff and is not disclosed. | doc | Medium |
| **F12** | The growth recipe is a hybrid of Network Morphism / FPI / ReZero, not FPI; `grow_norm` copies `ln_old` but resets the whole second segment, contradicting the FPI “LayerNorm parameters copied” motivation. | doc | Medium |
| **F13** | The eos spot-check samples `2M / n_shards` tokens per shard, not “the first 2M tokens” of the pack; a shard with late eos can be falsely refused. | doc | Low |
| **F14** | No `key_dim % 2 == 0` check; an odd `K` breaks `rotate_every_two`. | code | Low |
| **F15** | `embed_scale` is inherited as `sqrt(768)` for a width-1408 model; new dimensions use the old scale; the document explains the inheritance but not the new-dimension consequence. | doc | Low |
| **F16** | `gamma` is per layer × per head (440 parameters); RetNet shares γ across layers. The document’s `γ_h` notation hides the per-layer dimension. | doc/code | Low |
| **F17** | The audited original §7.2 says “`chunk_tokens`”; the current code uses a byte budget (`ce_chunk_tokens`). The English rewrite corrects this; the original does not. | version | Low |

**Bottom line: REVISE, not reject.** The model is a legitimate RetNet variant with an engineering growth layer; its core equations mostly hold. But the document must not be treated as the authoritative specification until F1–F5 are corrected, and `RETNET_FUSED_CE=1` must not be used until F1 is fixed. The cost of F1, F2, F3, F5, F8, F9, F10, F12, F13, F15, F16, F17 is **documentation only**; F6, F7, F14 are low-cost code guards; F4 is an analysis/documentation correction; true GLA or per-dim xPos would be a real model change (moderate/high).

---

## 2. Document ↔ code consistency

### 2.1 What is consistent

| Doc | Equation / claim | Code | Verdict |
|---|---|---|---|
| §1 | embedding `E[x]·s_e`, dropout | `_forward_hidden`: `embedding*scale` → `embed_dropout` | ✅ |
| §1 | `Norm1`, `Norm2`, FFN, final norm, tied logits | `_block_forward`, `_forward_hidden` | ✅ |
| §2 | `embed_scale` default `sqrt(D)`, inherited as `sqrt(768)` | `RetNetForCausalLM.__init__`, config `27.7128` | ✅ (consequence of growth; see F15) |
| §3.1 | LayerNorm formula with `1/D` variance | `nn.LayerNorm` | ✅ |
| §3.2 | two-segment GLN: `LN_old(x_old) ‖ x_new·γ_new + β_new` | `GrowthLayerNorm.forward` | ✅ |
| §4.1 | `q=W_Q x`, `k=W_K x`, `v=W_V x`, `g=W_G x`; `k←kK^{-1/2}` | `_project_and_rotate` | ✅ |
| §4.1 | `W_V,W_G∈R^{V×D}`, `W_O∈R^{D×V}`; `V=2D`, `F=2D` | config `value_dim=ffn_dim=2816`; RetNet paper’s 2d allocation | ✅ (matches RetNet’s parameter allocation, not a deviation) |
| §4.2 | RoPE-style pairwise rotation, frequencies `10000^{-i/(K/2)}` | `rotate_every_two`, `theta_shift`, `_rotations` | ✅ as a RoPE implementation; see F10 |
| §4.3 | `γ_h=1−2^{−5−h}`, `D_nm=γ^{n−m}` causal | `MultiScaleRetention.__init__`, `_decay_mask` | ✅ for the initial value and the mask |
| §4.4 | parallel `Σ(q·k)D v`; recurrent `S=γS+k^Tv`, `o=qS`; chunkwise `S_i`/`o_{i,t}` | `_parallel_retention`, `_recurrent_retention`, `_chunkwise_retention` | ✅ (T formula in §4.5 is not; see F2) |
| §4.6 | `GroupNorm_H(concat heads)`, `(SiLU(g)⊙y)W_O` | `_head_group_norm`, `forward` | ✅ for GroupNorm/gate; dropout missing (F3) |
| §5 | `W_2 GELU(W_1 z)`, optional SwiGLU | `RetNetBlock.__init__` | ✅ |
| §6 | tied `E` output | `tie_weights`, `lm_head` | ✅ |
| §7.1 | wcode mapping 0/1/2/3, `Σw·CE/max(Σw,1)` | `_chunked_weighted_ce` | ✅ |
| §7.2 (English rewrite) | byte-budget chunking | `ce_chunk_tokens`, `_chunked_weighted_ce` | ✅ |
| §8.1 | `input=r_t`, `label=r_{t+1}`, `w=c_{t+1}` | `PackedChatDataset.__getitem__`, `_PackedStage.row` | ✅ |
| §10 | AdamW 8-bit, wd 0.1, 1-D params no decay, cosine, bf16, clip 1.0 | `TrainingArguments`, `get_decay_parameter_names` | ✅ |

### 2.2 F1 — Fused CE backward is mathematically wrong (Critical; code + doc)

**Document claim.** §7.3 says the backward recomputes `softmax − onehot`; `01_function_reference.md` says the same.

**Code.** `fused_ce.py:64`:

```python
p = torch.softmax(lg, dim=-1)
p.scatter_(1, target[s:e].unsqueeze(1), -1.0)   # comment says "p - onehot"
```

`scatter_` **replaces** `p[target]` with `-1`; it does not subtract 1. The correct target entry is `p[target] − 1`, so the code is off by `+p[target]`. The loss forward is correct; the gradient is not.

**Derivation.** With `z` logits, `p=softmax(z)`, `L=Σ w·CE/Σw`, `∂CE/∂z = p − onehot`. The code implements `p′ = p` except `p′[y]=−1`, i.e. `p′ = p − onehot − p_y·e_y`. The gradient error is `+p_y` at the target entry, propagated into `grad_h` and `grad_w`.

**Probe.** On the same logits/labels, the fused loss equals `F.cross_entropy`, but `∂L/∂W` differs by `1.45e-3` at maximum (max |grad| = 0.188), i.e. ≈0.8 %; the error is systematic, not floating-point noise.

**Impact.** Every training run with `RETNET_FUSED_CE=1` has biased gradients. `docs/02_resume_plan_220cny.md` §2 E22 recommends this path as the memory fallback, so this is a live operational risk. The bug predates the recent audit-fix pass (the fix only hoisted the fp32 cast), but the document’s “softmax − onehot” claim is false and the P2-11 “no dedicated test” residual left it undetected.

**Fix (low-cost).** Replace `scatter_` with `scatter_add_(1, target, -1.0)` or `p[torch.arange(N), target] -= 1`; add a `fused_linear_cross_entropy` vs `F.cross_entropy` gradient-equivalence test; update §7.3 and `01`. One line + one test.

### 2.3 F2 — Closed-form scan exponent is missing the chunk factor (High; doc)

**Document claim.** §4.5:

```
S_prev[i] = γ_h^{iC} S0 + Σ_{j<i} γ_h^{i-1-j} U_j
T[h,i,j] = γ_h^{i-1-j}
```

**Code.** `retention_fast.py:165-179`:

```python
log_d = log(gamma) * chunk_size          # d = γ^C
T = exp(log_d * (i - 1 - j))             # d^{i-1-j} = γ^{C(i-1-j)}
S_all = T @ U_flat + diag(γ^{C i}) S0
```

**Derivation.** `U_j` is the state contribution of chunk `j` **at the end of chunk j**, already carrying `γ^{C−1−r}`. To carry it to the start of chunk `i`, one multiplies by `γ^{C(i−j−1)}` (the decay over the `i−j−1` whole chunks in between). The correct formulas are:

```
S_prev[i] = γ^{iC} S0 + Σ_{j<i} (γ^C)^{i-1-j} U_j
T[h,i,j]  = (γ_h^C)^{i-1-j}
```

The document’s `γ^{i-1-j}` is short by a factor `C` in the exponent. The code is correct; the document’s claimed equivalence is not supported by its own formula.

**Impact.** This is a core training-math statement (the closed-form scan is the fast chunkwise implementation). A reader re-deriving from §4.5 would get a different result; the document’s “mathematically equivalent” claim is false at the equation level. It does not affect the running code, only the specification.

**Fix (free).** Correct the equation and `T`; add a one-line note that `U_j` is the end-of-chunk state and the cross-chunk factor is `γ^C` per chunk. Optionally add a unit test that compares the closed form to the loop for a small `γ, C, nc` (the existing tests compare outputs, not the T formula).

### 2.4 F3 — Dropout is drawn on the wrong side of `W_O` (High; doc)

**Document claim.** §1:

```
u = Dropout(W_O(SiLU(g) ⊙ GroupNorm(Ret)))
h' = h + u
```

§4.6:

```
o = (SiLU(g) ⊙ y) W_O
```

**Code.** `modeling_retnet.py:298-301`:

```python
output = self._head_group_norm(...)
output = F.silu(g) * output
output = self.dropout(output)          # dropout BEFORE out_proj
output = self.out_proj(output)
...
y = x + y                              # no dropout on the residual
```

The correct equation is:

```
u = W_O( Dropout( SiLU(g) ⊙ GroupNorm(Ret) ) )
h' = h + u
```

The document applies dropout **after** `W_O` and omits it in §4.6. At eval (dropout off) the two are identical; at training they are different stochastic functions with different masks and gradients. The English rewrite adds a prose note that says the order is before `out_proj`, but the displayed equation is still the old one — the note and the equation contradict each other.

**Impact.** The document’s §1/§4.6 are not the training math of the code. This matters for anyone reproducing the model or deriving gradients; it is not a code bug. The prior report’s P2-5 failure is the same defect seen from the documentation side.

**Fix (free).** Rewrite §1 as `u = W_O(Dropout(SiLU(g)⊙GroupNorm(Ret)))` and §4.6 as `o = W_O(Dropout(SiLU(g)⊙y))`; remove the “older version drew this wrongly” implication or make it point at the corrected equation.

### 2.5 F4 — §9 is a weight-count, not the function’s actual width/heads (High; doc + data)

The document’s §9 says:

- `γ_new ≡ 0` in `GrowthLayerNorm`, so the new segment is 0;
- layers 0–17 effective width 768, layers 18–19 effective **input** width 1024; “18×d768 + 2×d1024”;
- 338,120,704 params (67.06 %) exactly zero, ~166 M learnable.

**What the checkpoint actually says** (`retnet_500m_v21_export`):

| Quantity | Value |
|---|---|
| total parameters | 504,179,896 |
| exactly zero | **338,196,736 (67.0786 %)** |
| nonzero | 165,983,160 |
| `gamma_new` tensors | 41 × 640, **all 0** (true) |
| `embedding` nonzero columns | 768 |
| `out_proj` nonzero rows (all 20 layers) | **768** |
| q/k nonzero rows/cols, layers 0–17 | 768 / 768 |
| q/k nonzero rows/cols, layers 18–19 | 1024 / 1024 |
| v/g/ffn0 live rows, layers 0–17 | 1536 |
| v/g/ffn0 live rows, layers 18–19 | 2048 |
| out_proj/ffn2 live cols, layers 0–17 | 1536 |
| out_proj/ffn2 live cols, layers 18–19 | 2048 |
| effective q/k heads | **12** (0–17), **16** (18–19), not 22 |
| effective value heads | **12** (0–17), **16** (18–19) |
| unused nonzero input columns, layers 18–19 | q/k/v/g/ffn0 columns 768–1023: **4,194,304** weights |

The residual stream is **768 wide in every layer**: `embedding` has 768 live columns and every `out_proj` has 768 live rows, so the residual additions never carry dimensions 768–1407. `γ_new=0` makes `Norm1`/`Norm2` output the new segment 0, so the MSR input is 768 wide in **all** layers. The q/k **output** width is 1024 in layers 18–19 because q/k have 1024 live rows; the q/k **input** is still 768. The “18×d768 + 2×d1024” is a q/k-width statement, not an input or residual-width statement.

The document’s own §3.2 (“the new segment is identically zero after each layer’s normalization”) contradicts §9’s “layers 18–19 effective input width 1024”. The 1024 columns are nonzero **weights**, but they are multiplied by zero activations. The effective head count is 12/16, not 22: 768/64 = 12, 1024/64 = 16; value 1536/128 = 12, 2048/128 = 16.

The “166 M learnable” is the `(p != 0)` count; it includes the 4.19 M nonzero-but-unused q/k/v/g/ffn0 input columns in layers 18–19. A functional-active estimate is ≈161.8 M (166.0 M minus the 4.19 M unused columns, ignoring the small zero-head `gamma` entries). This also explains why `live_param_ratio` (E08) and the per-parameter `GradCoverageCallback` (E07) can both report a “live” model while part of the nonzero mass is never reached: the guard is a **parameter-level nonzero** proxy, not activation-level liveness.

**Impact.** §9 is the document’s “actual mathematical state”; its conclusion about effective width/heads/learnable parameters is misleading. The nominal model is `D=1408, H=22, V=2816`; the current function is a **768-residual, 12/16-head, 1536/2048-value** model with unused nonzero weights. The exact zero count is also stale.

**Fix (free + low-cost analysis).** Either (a) restate §9 as “weight-count width” and distinguish residual width, q/k width, value width, head count, and nonzero vs functional counts; or (b) add a small `functional_width.py` probe that runs one forward with hooks / uses weight row-column reachability and prints the used widths and heads. The latter is a few dozen lines and no training change.

### 2.6 F5 — The “GLA” claim is not implemented (High; doc + naming)

**Document claim.** §4.3: “The justification is GLA: data-dependent gating outperforms a globally fixed decay [9]4 (§4.1).”

**Paper.** GLA (§4.1, Eq. 3) defines:

```
S_t = Diag(α_t) S_{t-1} + k_t^T v_t
α_t = σ(x_t Wα1 Wα2)   (data-dependent, per token)
```

**Code.** `modeling_retnet.py:119-124`:

```python
if config.use_gla:
    self.gamma = nn.Parameter(gamma.clone())   # per layer, per head scalar
else:
    self.register_buffer("gamma", gamma, persistent=True)
```

`_recurrent_retention` uses `state = gamma * state + k^T v`; `_chunkwise_retention` uses `γ^C`; there is no `x_t` dependence. The implementation is **RetNet with a learnable global decay** (per layer and head), not GLA. The `use_gla` flag and the document’s “data-dependent” sentence are misnomers.

**Impact.** The document justifies the design with a property the code does not have. The global learnable γ can adapt across training but cannot forget per token/content; the GLA paper’s reported gains from data-dependent gates are not expected here. This is a theoretical/attribution error, not a numerical one.

**Fix.** Either correct the prose (free): “RetNet with a learnable global decay; GLA would require input-dependent α_t and a different chunkwise form”; or implement GLA (high cost). A low-cost **approximation** is not equivalent and should not be presented as GLA.

### 2.7 F6 — `gamma` is unconstrained (Medium; code)

`gamma` is a raw `nn.Parameter`; `_decay_mask`, `_chunkwise_retention` and `retention_fast` use it without a domain check. RetNet/GLA assume `γ∈(0,1)`. For `γ>1`, `D_nm=γ^{n−m}` **grows** with distance; `γ^C` grows; the closed-form `T` grows; long sequences can overflow. The current checkpoint is safe (440 values, min 0.1057, max 0.99885, median 0.5656, none >1); the preflight only rejects `γ<=0`.

**Fix.** Cheap: add `if not 0 < γ < 1` to `_preflight` / a callback, or clamp in `forward`; for a future model, parameterize `γ=sigmoid(raw)` (state-dict/init migration, moderate). The document says `γ∈R`; it should say `(0,1)` for a decay.

### 2.8 F7 — `mode="parallel"` with a non-zero state is wrong (Medium; code)

`_parallel_retention` computes only the intra-sequence `Σ(qk)Dv`; `_parallel_final_state` updates the state, but the parallel output never adds the cross-state term `(q ⊙ γ^{t+1}) S_prev`. Probe: with a non-zero state from a 4-token prefix, `parallel` vs `chunkwise` max diff **0.2989**; with a zero state, 0.0. The current `generate` prefill uses `state=None`, so production generation is fine, but `_forward_hidden(..., mode="parallel", states=s)` is a public API that silently computes a different function. Fix: either raise when `state is not None` in parallel, or add the cross term (the same one chunkwise uses); low-cost.

### 2.9 F8 — Flat `wcodes` are not “always 1” (Low; doc)

`_FlatStage.__init__(..., weight=1, ...)` and `_wc = np.full(seq_len, self.weight)`; `StagePackedDataset._stage` passes the plan’s `weight`. The document §8.2 says “`wcodes` is always 1 for flat”, which was true before the P2-2 fix; it is false now. The current plan does not set a flat weight, but the code capability contradicts the doc. Fix the sentence (free).

### 2.10 F9 — The exact zero count is stale (Low; doc)

Document: `338,120,704 (67.06 %)`. Checkpoint: `338,196,736 (67.0786 %)`; difference 76,032. The qualitative 67 %/166 M is right; the exact number is not. Fix (free).

### 2.11 F10 — “xPos” is RetNet’s scalar simplification (Medium; doc)

xPos Eq (10), Alg. 1 uses a **per-dimension** length scaling `ζ_i^n` on q and `ζ_i^{-n}` on k. RetNet Eq (4) simplifies `γ` to a **scalar** per head; the code implements RoPE rotation on q/k and a scalar `γ^{n-m}` decay in `D`, i.e. RetNet’s scalar-decay xPos, not full xPos. This is mathematically consistent with RetNet (the product `qk · D` equals the scalar-decay xPos product), but the document’s §4.2 formula omits the `ζ`/decay and calls it xPos without qualification. Fix (free): say “RetNet’s scalar-decay xPos (RoPE + per-head γ)”. Per-dim xPos is a real change (moderate).

### 2.12 F11 — RetNet’s parallel score normalizations are omitted (Medium; doc)

RetNet §2.1, Eq (5) uses three numerical stabilizers: `QK^T/√d`, `D/√(ΣD)`, and `R/max(|ΣR|,1)`. The code keeps `k·K^{-1/2}` but omits the other two (the E16 fix removed them to make parallel = chunkwise). Because GroupNorm normalizes each head group separately, a per-row/per-head scalar scale is **mathematically a no-op after GroupNorm**; the omission is therefore exact up to fp, not a math error. It is, however, a numerical-stability tradeoff (the paper added the factors for fp precision) and is not disclosed. Fix (free): state the omission and the per-head scale invariance; if desired, add the factors to both parallel and chunkwise (moderate; no mathematical change).

### 2.13 F12 — Growth is a hybrid, not FPI/Net2Net (Medium; doc)

The document §3.2 cites Net2Net width (copy/scale), Network Morphism (zero one side, random other), bert2BERT FPI (copy/scale; LayerNorm parameters copied), StackBERT (zero init can leave units dead), ReZero (zero scalar gate), Fixup (variance-preserving init). The growth script:

- `fill`: copies old weights top-left, zeros new **columns**, randomizes new **rows** — Network Morphism-like, not Net2Net’s copy-with-scaling;
- `grow_norm`: copies `ln_old.weight/bias`, but sets the **whole** `gamma_new`/`beta_new` segment to `BLOCK_GATE=1.0` (blocks) or `FINAL_GATE=0.0` (final), discarding the source’s learned `gamma_new` — not FPI’s “copy the LayerNorm parameters”.

The fixed script’s header (P1-8) states the “source old gates are not inherited” precondition; the math document does not. For the current v21 checkpoint, `gamma_new` is all zero, so resetting the final to 0 and blocks to 1 is function-preserving **because the downstream new columns are zero**; for a source trained away from the recipe, `--verify` refuses. This is a defensible engineering choice, but the document should say it is a **modified FPI / Network Morphism hybrid**, not FPI. Fix (free); exact FPI would require copying the old segment and would not revive the zero-gated v21 new dims (low/moderate).

### 2.14 F13 — eos sampling is per-shard, not the first 2M tokens (Low; doc)

`_FlatStage._check_boundaries` sets `per = sample // len(shards)` and samples `per` tokens from each shard, not the first 2M tokens of the pack. A shard whose eos is after the per-shard slice but within 2M will be falsely refused. The real `pretrain_v2` shards pass for the current mixtures (all 34/37/25 selected shards have eos in the first per-shard sample), but the document’s wording should be “up to 2M tokens spread across shards” (free).

### 2.15 F14 — No odd `key_dim` guard (Low; code)

`rotate_every_two` and `repeat_interleave(2)` assume an even `key_dim = D/H`; there is no check. The current `K=64` is even; an odd `K` would fail at runtime. Add `assert key_dim % 2 == 0` in `MultiScaleRetention.__init__` (free/low).

### 2.16 F15 — `embed_scale` inheritance (Low; doc)

The current config uses `embed_scale=sqrt(768)` for a `D=1408` model, inherited to preserve the old dimensions’ scale. The new dimensions are also multiplied by `sqrt(768)`, not `sqrt(1408)`; the document explains the inheritance but not the consequence for new dims. With `gamma_new=0` they are dead today; for a revived model, the new dims’ scale differs from a fresh RetNet. Fix (free: state it; low: split the scale for old/new dims if growing).

### 2.17 F16 — `gamma` is per-layer, not shared (Low; doc)

RetNet §2.2 keeps `γ` identical across layers; the code creates one `gamma` per `MultiScaleRetention` block (20 × 22 = 440). The document’s `γ_h` notation does not expose the layer index. The per-layer learnable γ is more expressive and is still a global per-layer decay; it is a deviation from RetNet’s regularization. Fix (free: document); tying the layers is a low-cost code change but changes the model/training.

### 2.18 F17 — Original §7.2 predates the byte budget (Low; version)

The audited original §7.2 says “`chunk_tokens`”; the current code uses `ce_chunk_tokens(V, chunk_bytes)` and the English rewrite §7.2 correctly says “byte budget”. If the original is presented as the current math, it is stale; the English rewrite is the current one. No code issue.

---

## 3. Comparison with the cited papers

### 3.1 RetNet

| Component | RetNet paper | Code | Difference / judgement |
|---|---|---|---|
| Architecture | pre-LN residual blocks, MSR + FFN | same | ✅ |
| Parallel retention | `(QK^T ⊙ D)V`, Eq (5) | same after removing the two score normalizations | ✅ in exact arithmetic after per-head GroupNorm; numerical tradeoff (F11) |
| Recurrent | `S=γS+K^T V`, `Q S`, Eq (6) | same | ✅ |
| Chunkwise | Eq (7), `R_i=K^T(V⊙ζ)+γ^B R_{i-1}`, `ζ=γ^{B-i-1}`, cross `ξ=γ^{i+1}` | same | ✅ |
| Decay | `γ=1−2^{−5−h}`, fixed, shared across layers | per-layer/per-head learnable | **deviation** (F5/F6/F16) |
| xPos | complex rotation, `γ` scalar in Eq (4) | RoPE + scalar D | ✅ RetNet’s scalar form; not full xPos (F10) |
| Score normalization | `QK^T/√d`, `D/√rowsum`, `R/max(|rowsum|,1)` | only `1/√d` | omission mathematically no-op after GroupNorm; numerical (F11) |
| GroupNorm | per-head groups, SubLN | `GroupNorm(num_heads, value_dim)` | ✅ |
| Swish gate | `SiLU(XW_G)⊙Y`, `W_O` | same; dropout added | ✅ + regularization; doc placement wrong (F3) |
| FFN | `gelu(XW1)W2`, 2d intermediate | same | ✅ |
| Value dim | 2d (head dim V = 2×Q/K) | `value_dim=2*D` (2816) | ✅ matches RetNet’s parameter allocation |
| Tied output | not specified in the paper | tied embedding/head | engineering choice; consistent with the doc |
| Dropout | not in Eq (8)/(9) | embed/MSR/FFN dropout 0.1 | regularization deviation; doc includes; placement wrong |
| Growth LN | not in RetNet | `GrowthLayerNorm` | engineering extension; see growth papers |

**Advantages of the code’s RetNet variant.** (i) exact train/inference consistency after E16; (ii) per-layer learnable γ can adapt the multi-scale decay; (iii) 2d value/FFN matches RetNet’s parameter allocation; (iv) fast closed-form chunkwise scan.

**Disadvantages.** (i) learnable global γ is not data-dependent (F5); (ii) unbounded γ (F6); (iii) per-layer γ is a deviation from RetNet’s shared regularization (F16); (iv) omitted score normalizations may hurt fp16/bf16 stability at long L (F11); (v) growth layer is non-standard; current weights are degenerate (F4/F12).

**Migration cost.**
- Fix the doc (γ description, T, dropout): free.
- Add a γ domain guard / clamp: low.
- Share γ across layers: low code, but retraining.
- Add score normalizations to both modes: moderate (no mathematical change; numerical).
- True GLA-style gate: moderate/high (new parameters and chunkwise algorithm).
- Remove the growth layer / use standard LayerNorm: high (new weights/training).

### 3.2 GLA

| Component | GLA paper | Code | Difference |
|---|---|---|---|
| Recurrence | `S_t = Diag(α_t)S_{t-1} + k_t^T v_t` | `S_t = γ_h S_{t-1} + k_t^T v_t` | **no data dependence** (F5) |
| Gate | `α_t = σ(x_t Wα1 Wα2)` per token | scalar `γ_h` parameter | not GLA |
| Chunkwise | log-space cumulative products / hardware-efficient scan | constant-γ closed-form `T` | only valid for constant γ |
| Output | per-head LN + swish gate | GroupNorm + swish gate | different normalization; code follows RetNet |
| Length generalization | a claimed GLA advantage | not tested/reproduced | UNSURE |

**Advantages of GLA.** Data-dependent gates can forget per token/content; reported length generalization and recall improvements; hardware-efficient chunkwise.
**Disadvantages/cost.** More parameters (Wα), a more complex chunkwise scan, log-space stability work; no closed-form constant-γ shortcut.
**Code’s choice.** A global learnable γ keeps the RetNet-style closed form and costs only 440 parameters; it is a reasonable **low-cost** compromise, but it must not be called GLA.
**Migration.** (a) Free: call it “learnable global decay” and cite RetNet + the GLA paper only as motivation for *considering* data-dependent gates. (b) Moderate: add a per-token scalar gate and implement GLA’s chunkwise form; this changes the model and needs retraining. There is **no cost-free way** to obtain GLA’s data dependence; the current code is not a GLA implementation.

### 3.3 xPos

| Component | xPos paper | Code | Difference |
|---|---|---|---|
| Rotation | RoPE `q cos + rot(q) sin` | same | ✅ |
| Length scaling | per-dim `ζ_i^n` on q, `ζ_i^{-n}` on k | scalar `γ^{n-m}` in D | scalar simplification (RetNet Eq 4) |
| Frequency | `θ_i=10000^{-2i/d}` | `10000^{-i/(K/2)}` with `i=0..K/2-1` | ✅ |
| Decay | `ζ` chosen to stabilize long-range attention | `γ` per head, learnable | different parameterization |

**Advantages of xPos.** Better length extrapolation; stable long-range curves; per-dimension control.
**Disadvantages.** Extra per-dimension scaling; RetNet’s scalar γ is cheaper and supports the closed-form scan but loses per-dim extrapolation.
**Code.** Mathematically equivalent to RetNet’s scalar-decay xPos, **not** full xPos. The doc should say so (free). Per-dim xPos would be a moderate change; the scalar state recurrence would need per-key-dim decay or a q/k-only scaling, and retraining.

### 3.4 Transformers are RNNs / linear attention

| Component | Paper | Code | Difference |
|---|---|---|---|
| Recurrence | `S_t=S_{t-1}+k_t^T v_t`, `o=qS` | `S_t=γS_{t-1}+k^Tv` | decay added (RetNet) |
| Chunkwise | inter-chunk `S[i+1]=S[i]+K^T V`; intra `Q S + (QK^T⊙M)V` | same with γ | ✅ |
| Complexity | O(1) inference, O(L) memory | same | ✅ |

**Advantages.** Linear inference/memory; training parallelism via chunks.
**Disadvantages.** Lower expressivity than softmax attention; the decay/gate is the main remedy.
**Code.** Consistent; the document’s §4.4 citation is correct; no migration needed.

### 3.5 GroupNorm

| Component | Paper | Code | Difference |
|---|---|---|---|
| Groups | channel groups, per-sample normalization | `num_groups=num_heads`, group size `V_h` | ✅ |
| Affine | per-channel γ/β | `nn.GroupNorm` affine | ✅ |
| Role | normalizes each head’s output; scale-invariant | same | ✅ |

**Advantages.** Scale invariance (lets the score-normalization factors be no-ops), per-head variance handling; cheap.
**Disadvantages.** No sequence mixing; the group count must match heads; zero heads waste groups.
**Code.** Correct; no migration.

### 3.6 GLU variants

| Component | Paper | Code | Difference |
|---|---|---|---|
| SwiGLU | `W_down(SiLU(W_gate z)⊙W_up z)` | optional, current `use_swiglu=False` | ✅ |
| GELU FFN | baseline | default `W2 GELU(W1 z)` | ✅ |

**Advantages.** SwiGLU often better at equal parameters; gating adds non-linearity.
**Disadvantages.** Extra projection/parameters; current model uses GELU (RetNet default).
**Code.** Correct; the document’s §5 is accurate.

### 3.7 Growth papers

| Paper | Idea | Code | Match / difference |
|---|---|---|---|
| Net2Net (2015) | Net2WiderNet copies old units and divides outgoing weights by copy count | old weights top-left; new columns 0; new rows random | **not** Net2Net; no copy-with-scaling |
| Network Morphism (2016) | width morphism: zero one side, random other | same spirit | closest match |
| bert2BERT (2021) | FPI: copy/scale weight matrices; copy LayerNorm params | `ln_old` copied; `gamma_new`/`beta_new` reset to gate | **modified FPI**; not exact |
| StackBERT (2019) | zero-init residual branches can leave units dead | current v21 `gamma_new=0`; 338 M zeros | the observed failure mode |
| ReZero (2020) | zero-initialized scalar residual gate is learnable | `gamma_new` vector gate; final 0 → gradient via upstream×x | same principle; vector not scalar |
| Fixup (2019) | scale residual branches by depth for variance | pre-LN; no Fixup scaling | not implemented |

**Advantages of the code’s hybrid.** Function preservation through zero new columns; random new rows make the new columns trainable; block gate 1.0 + final gate 0 keeps the logits unchanged while reviving the block dims; the trainability probe refuses dead outputs.
**Disadvantages.** It discards the source’s learned second-segment gates (not FPI); it is only function-preserving for sources produced by the same recipe; the current v21 checkpoint is already destroyed (338 M zeros).
**Migration.** Free: correct the prose to “modified FPI / Network Morphism hybrid”. Low-cost: copy the old `gamma_new[:n_src]` and set only the truly new part, with a `--revive-dead` flag for zero-gated sources; but for v21 all block gates are 0, so copying would keep the dims dead — the current reset is intentional. Exact FPI/Net2Net would require a different growth recipe and retraining.

### 3.8 Chinchilla

The document cites Chinchilla only for the training token budget; there is no model math to compare. The reference is appropriate; no finding.

---

## 4. Advantages / disadvantages and migration costs

### 4.1 Design choices

| Choice | Current code | Paper / alternative | Advantage | Disadvantage | Migration |
|---|---|---|---|---|---|
| Decay parameterization | per-layer/per-head learnable global γ | RetNet fixed shared γ; GLA data-dependent α_t | adapts; cheap; keeps closed-form | not data-dependent; unbounded; not GLA | doc-free; γ guard low; GLA moderate/high |
| Position encoding | RoPE + scalar γ | xPos per-dim ζ | cheap; closed-form | less length extrapolation; not full xPos | doc-free; per-dim moderate |
| Parallel score normalization | omitted after E16 | RetNet three factors | exact parallel/chunkwise | numerical stability at low precision/long L | add to both (moderate) or document (free) |
| Normalization | GroupNorm per head | GroupNorm/SubLN | scale-invariant; per-head variance | zero heads waste groups; no sequence mixing | none |
| Value/FFN width | 2D / 2D | RetNet 2d | matches RetNet parameter allocation | more compute than 1D; but paper-consistent | none |
| Growth LN | two-segment, new dims affine | FPI/Net2Net | revives width; preserves old function | not exact FPI; requires recipe; current dead | doc-free; copy segment low; exact FPI high |
| Dropout | inside MSR before W_O; FFN before add | RetNet paper has no dropout | regularization | doc equation wrong; different training function | doc-free |
| Fused CE | chunked, custom autograd | standard CE | memory saving | gradient wrong; no test | one-line + test (low) |
| Embedding scale | sqrt(768) on 1408 dims | sqrt(D) fresh | preserves old dims | new dims under-scaled | doc-free; split low |
| Parallel state | ignored in parallel output | full sequence formula | fast prefill with zero state | silent wrong output with non-zero state | guard/cross low |

### 4.2 Cost-free fixes (documentation only)

1. F2: correct the `T`/`S_prev` exponent to `γ^{C(i-1-j)}`.
2. F3: rewrite the §1/§4.6 dropout equations to put dropout before `W_O`.
3. F5: call the decay “learnable global per-head/per-layer γ”, not GLA; cite GLA only as motivation.
4. F8: “flat wcodes are 1 unless the plan sets `weight`”.
5. F9: update the zero count to 338,196,736 / 67.0786 %.
6. F10: say “RetNet’s scalar-decay xPos”, not full xPos.
7. F11: state the three RetNet normalizations and why the two omitted ones are no-ops after per-head GroupNorm (numerical tradeoff).
8. F12: say “modified FPI / Network Morphism hybrid”; state the source-recipe precondition.
9. F13: say “up to 2M tokens spread across shards”.
10. F15: state the new-dimension `sqrt(768)` consequence.
11. F16: state 440 per-layer gammas; RetNet shares them.
12. F17: use the English rewrite’s byte-budget wording.

### 4.3 Low-cost code fixes (≤ a few lines + a test)

1. **F1**: `p.scatter_add_(1, target, -1.0)` or `p[arange, target] -= 1`; add fused-vs-reference gradient test. **Do this before any `RETNET_FUSED_CE=1` run.**
2. **F6**: add `0 < γ < 1` to `_preflight` / a callback; or `gamma.clamp(1e-6, 1-1e-6)`.
3. **F7**: raise if `mode=="parallel" and state is not None`, or add the cross term `(q*γ^{t+1}) @ state`.
4. **F14**: `assert key_dim % 2 == 0`.
5. **F4**: add a small `functional_width.py` probe that prints residual/qk/value widths, effective heads, used vs nonzero weights; update §9.

### 4.4 Moderate / high-cost changes

- **True GLA**: per-token `α_t` (low-rank projection), `S_t=Diag(α_t)S_{t-1}`, a log-space chunkwise scan; new parameters, retraining; moderate/high. No cost-free equivalent.
- **Per-dim xPos**: `ζ_i^n` scaling; moderate; retraining.
- **Add RetNet score normalizations to both paths**: moderate; mathematically no-op after GroupNorm, so primarily numerical.
- **Exact FPI growth**: copy/scale LayerNorm params; for v21 zero gates this would keep the new dims dead; needs a `--revive` flag; moderate.
- **Share γ across layers**: low code but changes the model; retraining.

---

## 5. Recommended order

**Before any further training / release**

1. **F1**: fix the fused-CE scatter and add a gradient test; until then, remove or qualify the `RETNET_FUSED_CE=1` recommendation in `docs/02` §2 E22.
2. **F2/F3**: correct the two equations; these are the document’s core math.
3. **F4/F5**: rewrite §9 (weight-count vs functional width/heads/nonzero vs active) and §4.3 (global learnable γ, not GLA).
4. **F6/F7**: add the γ domain guard and the parallel-state guard/cross term.
5. **F8–F17**: documentation and small guards.

**Acceptance checks**

- `pytest` still 82/4; `verify.py` still PASS.
- Fused CE: `fused_linear_cross_entropy` gradient `allclose` to `F.cross_entropy` (atol 1e-5, fp32) and a training-step loss/grad comparison.
- Closed form: a small `γ, C, nc` test computes `S_prev` both ways; `T` formula matches.
- Actual state: a probe prints residual 768, q/k 12/16 heads, value 1536/2048, 4,194,304 unused nonzero weights, 338,196,736 zeros.
- γ: a model with `γ=1.5` is rejected or clamped; current v21 values pass.
- Parallel: `parallel` with a non-zero state equals `chunkwise` or raises.

---

## 6. Verdict

**REVISE.** The model’s core RetNet mathematics is sound and largely faithful to the code and the cited RetNet/xPos/linear-attention/GroupNorm/GLU papers. The document is not yet a faithful specification: **F1** is a real (pre-existing) gradient bug in an optional path that the document claims is correct; **F2/F3/F5** are high-impact theoretical/documentation errors; **F4** materially misdescribes the actual weights (residual width 768, effective heads 12/16, 4.19 M unused nonzero weights); **F6/F7** are real edge risks. Most fixes are documentation-only; F1/F6/F7 are low-cost; true GLA / per-dim xPos / exact FPI would be real model changes and are not required to make the current model coherent.

---

## Appendix A — Derivations

### A.1 Fused CE gradient

For one token, `CE = −log p_y`, `p=softmax(z)`, `∂CE/∂z_i = p_i − 1[i=y]`. With weights `w` and `Z=Σw`, `L=Σ w CE/Z`; `∂L/∂z = (p − onehot) w/Z`. The code sets `p_y=-1`, so the target entry is `−1` instead of `p_y−1`; the error is `+p_y` at the target. The loss is unaffected because it uses `lse−z_y`, not the backward `p`.

### A.2 Closed-form chunk state

Let chunk `j` have positions `r=0..C−1`, and `U_j=Σ_r k_{j,r} v_{j,r}^T γ^{C−1−r}`. The state at the start of chunk `i` is

```
S_prev[i] = γ^{iC} S0 + Σ_{j<i} γ^{C(i−j−1)} U_j
```

because the state contribution of chunk `j` decays by `γ^C` for each whole chunk between `j` and `i`. The code implements this with `T=(γ^C)^{i−1−j}`; the document’s `γ^{i−1−j}` is missing the `C`.

### A.3 Parallel with a past state

For a sequence starting at offset 0 with a past state `S_{-1}` before position 0, the output at position `t` is

```
o_t = Σ_{m≤t} (q_t·k_m) γ^{t−m} v_m + γ^{t+1} q_t S_{-1}.
```

The code’s `_parallel_retention` has only the first term; the second is in `_chunkwise`’s cross term. Adding `(q*γ^{t+1}) @ S_{-1}` to `_parallel_retention` (or raising) fixes it.

### A.4 Per-head GroupNorm scale invariance

For a head group `g` at position `n`, `GroupNorm` normalizes the vector `y_g,n ∈ R^{V_h}`; multiplying it by a scalar `c_n` leaves the normalized vector unchanged (mean/variance scale by `c_n`, then cancel; the affine is applied after). RetNet’s `D/√rowsum` and `R/max(|rowsum|,1)` are per-row scalars, i.e. per-head scale factors; after per-head GroupNorm they are no-ops. This is why E16’s removal is mathematically exact; it is a numerical-stability tradeoff, not a mathematical error.

### A.5 Actual width/head count

- `embedding` live columns = 768 → residual input 768.
- every `out_proj` live rows = 768 → residual output 768.
- `gamma_new=0` → `Norm1`/`Norm2` new segment = 0 → MSR input 768 in every layer.
- q/k live rows 768 (layers 0–17), 1024 (18–19); `key_dim=64` → 12/16 heads.
- v/g live rows 1536 (0–17), 2048 (18–19); `V_h=128` → 12/16 value heads.
- out_proj live rows 768 = 12×64; the residual is 768.
- q/k/v/g/ffn0 input columns 768–1023 in layers 18–19 are nonzero (4,194,304 weights) but multiply zero activations; they are unused.
- nonzero count 165,983,160; functional-active ≈161.8M.

---

## Appendix B — Evidence and reproduction

- `python verify.py` in the snapshot: PASS, 9/9; MANIFEST 69 entries.
- `cd code && python -m pytest tests -q`: 82 passed, 4 skipped.
- Fused CE probe: same logits/labels; loss equal; `∂L/∂W` max diff 1.45e-3, scale 0.188.
- Parallel-state probe: non-zero state max diff 0.2989; zero state 0.0.
- Checkpoint probe: total 504,179,896; zero 338,196,736; `gamma_new` 41×640 all 0; q/k rows/cols, out rows, embedding cols as in §2.5.
- eos probe: each of the 34/37/25 shards selected for `rescue_m1/2/3` has eos in the first `2M/len(shards)` tokens.
- Paper text: `Papers/01_Retentive-Network_2307.08621.pdf` pp. 3–5; `03_Gated-Linear-Attention_2312.06635.pdf` pp. 4–5; `02_Length-Extrapolatable-Transformer_2212.10554.pdf` pp. 3–4; `04_Transformers-are-RNNs_2006.16236.pdf` pp. 3–4; `05_Group-Normalization_1803.08494.pdf` §3; `06_GLU-Variants_2002.05202.pdf` §2; growth papers as cited.

*End of report.*
