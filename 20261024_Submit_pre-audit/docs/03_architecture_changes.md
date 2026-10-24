# 03 Model Architecture Changes: From the Original RetNet to the Current Implementation

> Citation standard: GB/T 7714—2015 "Information and documentation — Rules for bibliographic references and citations to information resources", sequential numbering system.
> In-text markers take the form `[number]page (section)`; for example `[1]4 (§2.2)` means reference [1], page 4, §2.2;
> **Page numbering convention**: all page numbers in this document refer to the page index of the corresponding PDF under `Papers/` (counting from 1), so that the file can be opened and checked directly;
> Formal publication page numbers are given inside the reference entries (for example, 2337–2346 for StackBERT).
> Entries at the end are listed in order of first citation. Each entry's "local file" is the corresponding PDF under `Papers/` in the repository, for page-by-page verification.
> This document describes only changes to the **model itself**; training data, the loss function and the budget are out of scope.

---

## 1. Decay Rate γ: From a Fixed Multi-Scale Formula to a Learnable Gate (Primary)

Every head of the original RetNet uses a **fixed** decay rate:

$$\gamma = 1 - 2^{-5-\mathrm{arange}(0,h)} \in \mathbb{R}^{h}$$

The heads are distributed in a geometric progression from short memory to long memory, and are not updated during training [1]4 (§2.2, Eq. (5)).

This project turns γ into an `nn.Parameter`: the initial value still comes from the formula above, but it is updated by gradients during training (`config.use_gla=True`; see
`MultiScaleRetention.__init__` in `retnet_hf/modeling_retnet.py`). The rationale is GLA's conclusion that
data-dependent gating improves performance more than a global, data-independent fixed decay, and that gated linear attention still retains a
hardware-friendly chunkwise form [2]4 (§4, §4.1). The change concerns only whether the parameter is learnable; it does not alter the shape or the initial distribution of γ.

## 2. Normalization: A Newly Added Two-Segment GrowthLayerNorm (Primary)

The original RetNet uses a plain LayerNorm (pre-LN) in every layer [1]5 (§2.3). After width growth this project introduces
`GrowthLayerNorm`: the first `growth_old_dim` dimensions go through the original LayerNorm, while the remaining dimensions receive only the
per-channel affine transform `x * gamma_new + beta_new` and are **not normalized**, so that at the moment of growth the new dimensions do not
affect the normalization statistics of the old dimensions.

The design rests on function-preserving growth: Net2Net's width transform places the old weights in the top-left corner of the new matrix, scales the next layer by the number of copies, and adds noise to the copies to break symmetry [3]3 (§2.3); Network Morphism further points out that in a width morphism
only the side with fewer parameters can be set to 0, while the other side must be random [4]6 (§3.4.1); bert2BERT's FPI likewise adopts
"copy + scale by the number of copies", and copies the LayerNorm parameters rather than zeroing them [5]4 (§3.3.1).
The zero-initialized side is not free of cost: StackBERT explicitly records that setting the feed-forward layers close to zero and randomly initializing the attention layers
"is function-preserving, but ineffective, because most parameters in the self-attention layers are never trained" [6]2 (§2).
ReZero shows that a residual branch can be gated by a **zero-initialized but learnable** scalar α: the forward pass starts from the identity mapping,
and α still receives gradients, so the branch's gradients are not shut off along with it [7]3 (§3); Fixup argues from a variance-preservation perspective that
residual branches should be initialized scaled by depth and paired with a bias, rather than zeroed as a whole [8]4 (§3).
This project's later correction — "new rows random, new columns zeroed, final gate zeroed" — is precisely a trade-off between those two boundaries.

## 3. Scale and Shape: Three Rounds of Growth (Primary)

The current model was not produced by a single training run; it grew generation by generation along 100M → 143M → 272M → 0.5B:

| Generation | Layers | d_model | Heads | FFN width | Growth method |
|---|---|---|---|---|---|
| 100M | 12 | 768 | 12 | 1536 | Trained from scratch |
| 143M | 18 | 768 | 12 | 1536 | Depth 12→18 |
| 272M | 20 | 1024 | 16 | 2048 | Width 768→1024 + depth 18→20 |
| 0.5B | 20 | 1408 | 22 | 2816 | Width 1024→1408 |

Methodologically, depth growth corresponds to Net2Net's identity-layer insertion (the new matrix is initialized to the identity matrix) [3]5 (§2.4);
progressive deepening corresponds to StackBERT's segment-by-segment stacking [6]3 (§3); the multi-stage approach that trains only the newly added layers is described in MSLT [9]4 (§3);
the line of work that parameterizes growth itself as a learnable linear operator is LiGO [10]3 (§3.1).
The growth in this project is a hand-written `state_dict` mapping, not a complete implementation of any of the methods above, which has already been recorded as a defect in `docs/00`.

## 4. Training Forward Pass: Chunkwise and the Closed-Form Scan (Secondary)

The original RetNet gives three equivalent formulations, among which chunkwise recurrent runs in parallel within a chunk and recurs between chunks [1]4 (§2.2).
In this project, training along the **task configuration / launcher path** uses chunkwise (`run_task.py` passes
`--mode chunkwise`, as `tasks/v21_resume.json` does), and the chunk size is taken from `chunk_size` in the task JSON;
**when `train.py` is called directly the default is `--mode parallel`**, so to determine the actual mode you must look at the `chunkwise kernel` line
in the log and at the `--mode` that was passed, not just at the documentation.
Batched chunkwise is enabled by the environment variable `RETNET_FAST_CHUNKWISE=1` (the launcher's
`fast_chunkwise` field); it turns the per-chunk recurrence into a single closed-form scan: a lower-triangular matrix
`T[h,i,j]=γ_h^{i-1-j}` is used to obtain the states before all chunks in one batch, avoiding `nc` serial loop iterations
(`retnet_hf/retention_fast.py::chunkwise_batched`). The batched implementation and the reference loop are
**the same function, equivalent at the level of floating-point rounding, not bitwise identical** (measured max|Δ| 1.9e-6 @γ=0.5,
1.5e-5 @γ=0.99; `tests/test_fast_chunkwise.py` explicitly does not require bitwise equality).
For the chunkwise form of linear attention and causal masking, see the Linearized Attention and Causal Masking sections [11]3-4 (§3.2–§3.3),
and for GLA's chunkwise form see [2]5 (§4.2). This is a change at the implementation level and does not alter the mathematical definition of retention.

## 5. Positional Encoding: A Real-Valued Rotation Implementation of xPos (Secondary)

The original RetNet uses xPos: in the complex domain it multiplies q and k by a rotation factor with exponential decay [1]3 (§2.1).
This project keeps the same mathematical form but implements it with an equivalent real-valued two-dimensional rotation: each pair of adjacent components `(x0, x1)` is rotated to
`(-x1, x0)` and then combined with `cos` and `sin` (`rotate_every_two`, `theta_shift`),
with the frequency table generated as `10000^(-i/(key_dim/2))`. The xPos paper points out that rotation makes the inner product depend only on the relative angle, and that
once the exponential decay ζ is added, long-range inner products no longer oscillate strongly with the angle [12]3 (§3.2). This is therefore a **replacement of the implementation**,
not a change of positional-encoding scheme.

## 6. Normalization and Gating: An Engineering Implementation Equivalent to the Original Paper (Secondary)

The output gate of the original RetNet is `MSR(X) = (swish(XW_G) ⊙ GroupNorm_h(heads)) W_O`,
where GroupNorm normalizes the output of each head separately [1]4 (§2.2, Eq. (8)).
This project uses `nn.GroupNorm(num_heads, value_dim)` to implement the same "grouped by head" semantics,
and uses `F.silu(g) * output` for the gate (the SiLU spelling of swish); both are mathematically equivalent to the original paper and do not constitute a model change.
For the definition and grouping of GroupNorm, see §3 of the original paper [13]3 (§3).

## 7. Optional Branch: SwiGLU Feed-Forward (Secondary)

The original paper's FFN is the standard two-layer feed-forward (this project defaults to `ffn_dim = 2·d_model`, i.e. the 1536/2048/2816 in the table above).
The code retains a `use_swiglu` switch: when it is turned on, `silu(xW_gate) ⊙ (xW_up)` is computed and then passed through `W_down`.
This variant comes from the comparative experiments in GLU Variants, where SwiGLU outperforms the GELU baseline at an equal parameter count [14]1-2 (§2).
The current 0.5B weights **do not enable** this branch, so it is only an option, not an established change.

## 8. Parameter Count and Token Budget (Secondary)

The current 0.5B model has 504,179,896 parameters and was trained on 10.23B tokens, of which about 9.81B are unique,
i.e. roughly 19.5 tokens/parameter. This ratio refers to Chinchilla's compute-optimal conclusion that model size and training token
count should be scaled proportionally [15]4 (§3). Note that this is a **budget reference**, not an architecture change;
the actual learnable parameters number only about 166M (see the next section), and the true ratio computed on learnable parameters is about 59 tokens/parameter.

## 9. The Actual State After Growth (A Consequence, Not a Design Goal)

Because new rows, new columns and the gate were zeroed at the same time during width growth, **338,120,704 elements (67.06%) of the current weights are exactly 0**,
and the learnable parameters number about 166M; the effective width is a mixed width of "18 layers of d768 + 2 layers of d1024" rather than a uniform d1408.
This is exactly the failure mode recorded by StackBERT: a zero-initialized identity growth is function-preserving but "most parameters are never trained"
[6]2 (§2). It is not a new architectural design but follows from implementation defects in the growth scheme described in Sections 2 and 3;
for the detailed localization see E01–E05 in `docs/00_error_table.md`.

---

## References

[1] SUN Y, DONG L, HUANG S, et al. Retentive network: a successor to transformer for large language models[EB/OL]. (2023)[2026-10-23]. https://arxiv.org/abs/2307.08621. (Local: `Papers/01_Retentive-Network_2307.08621.pdf`)

[2] YANG S, WANG B, SHEN Y, et al. Gated linear attention transformers with hardware-efficient training[EB/OL]. (2023)[2026-10-23]. https://arxiv.org/abs/2312.06635. (Local: `Papers/03_Gated-Linear-Attention_2312.06635.pdf`)

[3] CHEN T, GOODFELLOW I, SHLENS J. Net2Net: accelerating learning via knowledge transfer[EB/OL]. (2015)[2026-10-23]. https://arxiv.org/abs/1511.05641. (Local: `Papers/07_Net2Net_1511.05641.pdf`)

[4] WEI T, WANG C, RUI Y, et al. Network morphism[EB/OL]. (2016)[2026-10-23]. https://arxiv.org/abs/1603.01670. (Local: `Papers/08_Network-Morphism_1603.01670.pdf`)

[5] CHEN C, YIN Y, SHANG L, et al. bert2BERT: towards reusable pretrained language models[EB/OL]. (2021)[2026-10-23]. https://arxiv.org/abs/2110.07143. (Local: `Papers/09_bert2BERT_2110.07143.pdf`)

[6] GONG L, HE D, LI Z, et al. Efficient training of BERT by progressively stacking[C]//Proceedings of the 36th International Conference on Machine Learning. PMLR, 2019: 2337-2346. (Local: `Papers/10_StackBERT_PMLR-v97-gong19a.pdf`)

[7] BACHLECHNER T, MAJUMDER B P, MAO H H, et al. ReZero is all you need: fast convergence at large depth[EB/OL]. (2020)[2026-10-23]. https://arxiv.org/abs/2003.04887. (Local: `Papers/13_ReZero_2003.04887.pdf`)

[8] ZHANG H, DAUPHIN Y N, MA T. Fixup initialization: residual learning without normalization[EB/OL]. (2019)[2026-10-23]. https://arxiv.org/abs/1901.09321. (Local: `Papers/14_Fixup_1901.09321.pdf`)

[9] YANG C, WANG S, YANG C, et al. Progressively stacking 2.0: a multi-stage layerwise training method for BERT training speedup[EB/OL]. (2020)[2026-10-23]. https://arxiv.org/abs/2011.13635. (Local: `Papers/11_Progressively-Stacking-2.0_2011.13635.pdf`)

[10] WANG P, PANDA R, TORROBA HENNIGEN L, et al. Learning to grow pretrained models for efficient transformer training[EB/OL]. (2023)[2026-10-23]. https://arxiv.org/abs/2303.00980. (Local: `Papers/12_LiGO_2303.00980.pdf`)

[11] KATHAROPOULOS A, VYAS A, PAPPAS N, et al. Transformers are RNNs: fast autoregressive transformers with linear attention[EB/OL]. (2020)[2026-10-23]. https://arxiv.org/abs/2006.16236. (Local: `Papers/04_Transformers-are-RNNs_2006.16236.pdf`)

[12] SUN Y, DONG L, PATRA B, et al. A length-extrapolatable transformer[EB/OL]. (2022)[2026-10-23]. https://arxiv.org/abs/2212.10554. (Local: `Papers/02_Length-Extrapolatable-Transformer_2212.10554.pdf`)

[13] WU Y, HE K. Group normalization[EB/OL]. (2018)[2026-10-23]. https://arxiv.org/abs/1803.08494. (Local: `Papers/05_Group-Normalization_1803.08494.pdf`)

[14] SHAZEER N. GLU variants improve transformer[EB/OL]. (2020)[2026-10-23]. https://arxiv.org/abs/2002.05202. (Local: `Papers/06_GLU-Variants_2002.05202.pdf`)

[15] HOFFMANN J, BORGEAUD S, MENSCH A, et al. Training compute-optimal large language models[EB/OL]. (2022)[2026-10-23]. https://arxiv.org/abs/2203.15556. (Local: `Papers/15_Chinchilla_2203.15556.pdf`)

---

## Appendix: Cross-Reference Between This Document's Numbers and `Papers/` File Numbers

| This document | Papers | Paper |
|---|---|---|
| [1] | 01 | RetNet |
| [2] | 03 | GLA |
| [3] | 07 | Net2Net |
| [4] | 08 | Network Morphism |
| [5] | 09 | bert2BERT |
| [6] | 10 | StackBERT |
| [7] | 13 | ReZero |
| [8] | 14 | Fixup |
| [9] | 11 | MSLT |
| [10] | 12 | LiGO |
| [11] | 04 | Transformers are RNNs |
| [12] | 02 | xPos |
| [13] | 05 | GroupNorm |
| [14] | 06 | GLU Variants |
| [15] | 15 | Chinchilla |

`Papers/16` (FineWeb) and `17` (Skywork) do not appear in this document: they are data sources rather than model architecture changes, and are therefore not listed.
