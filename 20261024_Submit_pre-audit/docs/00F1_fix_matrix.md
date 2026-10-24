# 00F1 Fix matrix (per item: fixed code + explanation)

> One entry per id in `00_error_table.md`. Code blocks are extracted from the
> source by `scripts/_gen_fix_doc.py` with AST/anchors, never copied by hand;
> each explanation is >=200 chars. `verify.py` re-checks the code blocks and
> that the status table agrees in all four places.

---

## E01 Growth must randomize new rows and zero only new columns

**Status**: Fixed

`scripts/grow_500m.py` L157–L178

```python
    def fill(p, old_key, out_d, in_d, gen):
        """旧权重放左上角；新**列**置 0（块内隔离），新**行**随机（让新单元有输出）。

        为什么是"列 0 行随机"，而不是全 0 或全随机：

        * 新列若随机，它会立刻改变旧维度的输出，函数不再保真，且无法用任何
          门控挡住（扰动已经进了旧维度）。所以新列必须是 0。
        * 新行若全 0，新单元（新头、新隐藏单元）的输出恒为 0，下游那些
          "读新单元"的列拿到的输入是 0，梯度 `上游 x 0 = 0`，整条新子网
          永远学不动——v21 的 338M 死参数就是这么来的。
        * 新行随机后：新单元有非零输出 -> 下游新列拿到非零输入 -> 新列在
          第 1 步就有梯度；新列变非零后，新行在第 2 步拿到梯度。
          这是一条**自愈的级联**，不是死锁。
        """
        w = torch.zeros(out_d, in_d)
        ow = old_sd[old_key]
        w[: ow.shape[0], : ow.shape[1]] = ow
        n_new_rows = out_d - ow.shape[0]
        if n_new_rows > 0:
            w[ow.shape[0]:, : ow.shape[1]] = \
                torch.randn(n_new_rows, ow.shape[1], generator=gen) * NEW_ROW_INIT_STD
        new_sd[p] = w
```

The old implementation first built the entire new matrix with `torch.zeros`, then pasted the old weights into the top-left corner, so newly added rows and columns were both 0 at the same time. That combination creates a gradient deadlock: with new columns at 0, the gradient of the new rows' output is multiplied by 0; with new rows at 0, the upstream gradient received by the new columns is also multiplied by 0, so the two zeros are factors of each other's gradient and nothing moves exactly, from step 0 on. The fix changes only one side: new columns stay 0, because random new columns would immediately change the output of the old dimensions and break functional fidelity, and no gate can block that perturbation; new rows become N(0, NEW_ROW_INIT_STD). Random new rows give the new units a non-zero output first, so the downstream columns that 'read the new units' receive non-zero input at step 1 and start learning; once those columns become non-zero, the new rows themselves receive gradient at step 2. This is a self-healing cascade, not a deadlock. Functional fidelity still holds: the new columns are 0, so the new dimensions do not affect the output of the old dimensions. This behavior is guarded by the reverse case in `tests/test_growth_trainable.py`: with NEW_ROW_INIT_STD set back to 0, the growth script must refuse to write.

How it is verified: `tests/test_growth_trainable.py::test_growth_guard_fails_when_reverted_to_zeros` sets `NEW_ROW_INIT_STD` back to 0 and asserts that the growth script returns 2 and does not write; the normal case requires rc in {0,1} and `model.safetensors` to exist. The measured baseline for the old weights is 338,120,704 elements exactly equal to 0, which is 67.06%.

see [07][08][09][10]

---

## E02 Intermediate layer gates must be 1, the final_norm gate must be 0

**Status**: Fixed

`scripts/grow_500m.py` L180–L204

```python
    def grow_norm(new_p, old_prefix, d_new, gate):
        """两段式 norm：旧段照搬，**第二段整段**（含源里已有的那一截）重设为 gate。

        这里**故意不**继承源模型第二段的 ``gamma_new[:n_src_new]``，理由分两边：

        * 中间层 gate=1.0：新维度要进入 q/k/v 的**新列**，而新列初始为 0；
          若门控也沿用源值（v21 里恰好是 0，E02 的 bug），则新列乘到的输入恒 0，
          ``dL/dW_newcol = 上游 x 0 = 0``，新列永远学不动。所以整段必须抬到 1.0。
        * final_norm gate=0.0：它是第二段通往 logits 的唯一出口。整段置 0 才能
          保证“生长后 logits 不变”—— embedding 的第二段会被重新随机（E03），
          若 final_norm 上半段还留着源的非零门控，那次随机就会漏进 logits。
          实测：把源 final ``gamma_new``（0.4）照搬过来，max|Δ logits| = 0.529。

        两者合起来：函数在输出端严格保真，而新参数从第 0/1 步起就有梯度。
        **代价是源模型第二段学到的门控会被丢弃**；这是有意的取舍，不是抄漏。
        """
        if src_grown:
            new_sd[new_p + ".ln_old.weight"] = old_sd[old_prefix + ".ln_old.weight"].clone()
            new_sd[new_p + ".ln_old.bias"] = old_sd[old_prefix + ".ln_old.bias"].clone()
        else:
            new_sd[new_p + ".ln_old.weight"] = old_sd[old_prefix + ".weight"].clone()
            new_sd[new_p + ".ln_old.bias"] = old_sd[old_prefix + ".bias"].clone()
        n_new = int(d_new) - int(grow_old_dim)
        new_sd[new_p + ".gamma_new"] = torch.full((n_new,), float(gate))
        new_sd[new_p + ".beta_new"] = torch.zeros(n_new)
```

The old implementation set all gamma_new and beta_new of the new segment to 0, and also carried over the zero segments the source model already had, calling it a 'zero side branch' in the comment. The problem is that this gate is a multiplicative gate: with gamma_new=0 the new dimensions are not only identically 0 in the forward pass, their own gradient is also identically 0, because it equals 'the gradient returned from downstream times the input of the new dimension', and those downstream columns that read the new dimensions are likewise initialized to 0. The fix separates the gate by position: the intermediate layers' norm1/norm2 are set to 1.0, so the new dimensions can flow into the new columns of q/k/v, and those new columns can pick up a gradient even with an initial value of 0 because their input is non-zero; final_norm is set to 0.0, since it is the only exit from the new dimensions to the logits, and zeroing it guarantees that the logits are bitwise unchanged at the instant of growth. The final_norm gate learns on its own: its gradient equals 'upstream logits gradient times the new dimension', and the upstream comes back through the new columns of the tied lm_head (random and non-zero), so it is non-zero from the very first step. Together the two give 'strict preservation at the output end, and new parameters learnable from steps 0/1'. The old-behavior case in Tests/test_growth_trainable.py verifies that the script refuses to save when the gate is zeroed.

Verification: the same test sets `BLOCK_GATE` to 0, covering the old behavior of 'zeroing the intermediate layer gate'; the normal path requires final_norm's `gamma_new` to be all 0 and the intermediate layers' to be all 1. The old model was measured to have all 41 `gamma_new` tensors equal to 0, which is the direct evidence that 'width growth never took effect'.

see [08][09][13]

---

## E03 Dead and new embedding columns must both be re-randomized

**Status**: Fixed

`scripts/grow_500m.py` L206–L215

```python
    # 嵌入：旧列照搬；[grow_old_dim:] 一律重新随机。
    # 这一段既包含本次新增的 384 列，也包含上一次生长遗留的 256 个死列——
    # 它们必须一起复活，否则"函数保真"仍然成立，但新维度没有信号源，
    # 下游新列拿到的输入仍然是 0，梯度仍然是 0。
    gen = torch.Generator().manual_seed(1234)
    emb = torch.zeros(cfg.vocab_size, NEW_D)
    emb[:, :grow_old_dim] = old_sd["embedding.weight"][:, :grow_old_dim]
    rev = emb[:, grow_old_dim:]
    emb[:, grow_old_dim:] = torch.randn(rev.shape, generator=gen) * 0.02
    new_sd["embedding.weight"] = emb
```

The old implementation left the new embedding columns at 0. That column plays two roles at once: the input-side token vector, and (because of `tie_weights`) the output-side classifier weight. Leaving it at 0 directly means the new dimensions of layer 0 receive no signal at all, and the whole new subnetwork has no entrance. The fix re-randomizes everything from `[grow_old_dim:]` (N(0,0.02)); note that the start is not OLD_D but `grow_old_dim`: the 256 columns left behind by the source model's previous growth are equally dead, so reviving only the 384 newly added columns would be pointless; all 640 columns must be revived together. Functional fidelity still holds, because `gamma_new=0` in `final_norm` keeps the new dimensions out of logits; and the random values of these 640 columns are the only source of non-zero signal for the entire new dimension. This forms a set with E01 and E02: without non-zero input, the gradient of the new columns is still 0; without random new rows, the new units still produce no output; with any one of the three missing, the 67% dead parameters come back.

How it is verified: after growth, inspect the embedding columns; the first `grow_old_dim` columns must be non-zero, and the following 640 columns must not be all zero. On the old model the measured embedding shape is [20000,1408], the absmax of the last 640 columns is exactly 0, and the active columns are only 0-767, fully consistent with 'the new dimensions have no signal source'.

see [07][09]

---

## E04 The growth script must validate trainability, not just functional fidelity

**Status**: Fixed

`scripts/grow_500m.py` L318–L341

```python
    print(f"[trainable] 跑 5 步反向，seq={args.seq}，统计零梯度参数 ...", flush=True)
    new.train()
    opt = torch.optim.AdamW(new.parameters(), lr=1e-4)
    probe = torch.randint(1, cfg.vocab_size, (2, args.seq))
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out = new(probe, labels=probe)
        out.loss.backward()
        opt.step()
    dead = [n for n, p in new.named_parameters()
            if p.grad is None or p.grad.abs().max().item() == 0.0]
    trainable = not dead
    if dead:
        print(f"[trainable] ❌ {len(dead)} 个参数梯度恒 0：")
        for name in dead[:12]:
            print("    ", name)
    else:
        print("[trainable] ✅ 全部参数梯度非零")

    # **两道校验都过才落盘**：只保真但不可训的检查点，
    # 静默产出的就是 v21 那种"67% 参数是 0"的模型。
    if verified is False or not trainable:
        print("❌ 校验未通过 —— 拒绝保存。先修映射/初始化，别拿这个检查点训练。")
        return 2
```

The old script had only one check: whether `max|delta logits|` is below tolerance. That check cannot fail by design, because an always-zero side branch is the most perfect functional fidelity of all - it affects nothing, so it necessarily passes. The fix adds a second check: load the grown model, run 5 real backward steps (AdamW, lr=1e-4), then inspect `grad.abs().max()` for every parameter and check that it is greater than 0. Five steps rather than one, because the new dimensions heal through a cascade: at step 0 only the new embedding columns and the columns reading the new input have gradients; at step 1 the new columns become non-zero and the downstream columns that read them start to have gradients; at step 2 the output projection of the new units receives gradients. 5 steps are enough to walk this chain, and any parameter still zero is genuinely dead. The write happens only when both checks pass; any failure returns 2 and refuses to save. `tests/test_growth_trainable.py` uses monkeypatch to set new-row randomization back to 0 and asserts that the script must return 2 and not write, thereby proving that this guard really can fail rather than being a decoration that is always true.

How it is verified: the two cases in `test_growth_trainable.py` form a positive and a negative pair; the positive one requires the checks to pass and the write to happen, the negative one requires the checks to fail and the write to be refused. The old script had only `max|delta logits|`, which any always-zero side branch passes, so this counterexample is itself the proof that the old check could not fail.

see [10][11]

---

## E05 GrowthLayerNorm's new-dimension gate must default to 1

**Status**: Fixed

`retnet_hf/modeling_retnet.py` L319–L324

```python
    def __init__(self, old_dim: int, new_dim: int, eps: float = 1e-5,
                 gamma_new: float = 1.0, beta_new: float = 0.0):
        super().__init__()
        self.ln_old = nn.LayerNorm(old_dim, eps=eps)
        self.gamma_new = nn.Parameter(torch.full((new_dim - old_dim,), gamma_new))
        self.beta_new = nn.Parameter(torch.full((new_dim - old_dim,), beta_new))
```

This class's constructor defaults `gamma_new=1.0` and `beta_new=0`, which is correct in itself; the problem was that the growth script explicitly overrode them with 0, and nothing in the old code protected that default. The E05 fix turns it into a testable contract: a constructed `GrowthLayerNorm` must have `gamma_new` all 1 and `beta_new` all 0; `tests/test_regressions_extra.py` asserts exactly that. Why the default must be 1 rather than 0: the new dimensions are not normalized, and `gamma_new` is their only scaling gate; with a gate of 0 the forward pass is identically 0, while its own gradient equals downstream backprop times the new-dimension input, and the downstream weights that read the new dimensions are also 0, so the deadlock is permanent. A default of 1 makes the new dimensions reachable at construction time, and the zero-side-branch effect is instead produced by the output gate of `final_norm` - zeroing there blocks only the output, not the gradient, because the gradient of the upstream logits can still flow back through the new columns of the tied embedding.

How it is verified: `tests/test_regressions_extra.py::test_growth_ln_new_gate_defaults_to_one` directly asserts that in a constructed instance `GrowthLayerNorm.gamma_new` is all 1 and `beta_new` is all 0. In the old model these 41 tensors were all zero while the class default itself is 1 - which shows the damage happened at the step where the growth script overwrote them, not in the class design.

see [08][13]

---

## E06 1-D parameters must not get weight decay by default

**Status**: Fixed

`retnet_hf/train.py` L592–L598

```python
    ap.add_argument("--no_decay_1d", action="store_true", default=DEFAULT_NO_DECAY_1D,
                    help="把 1 维参数（各种 norm scale / 门控 / msr.gamma）移出 weight decay。"
                         "**默认开启**：v21 没开这个开关，gamma 被 wd 从 1-2^-5 压到 0.5656"
                         "（半衰期 1.22 token），96.1%% 的头 <128 token。")
    ap.add_argument("--decay_1d", dest="no_decay_1d", action="store_false",
                    help="反向开关：允许 1 维参数吃 weight decay。只有复现旧 run 时才用；"
                         "用之前先想清楚 gamma 会不会被压塌。")
```

The old implementation's switch `--no_decay_1d` was off by default, and the v21 config did not contain this field at all, so `msr.gamma` as a bare `nn.Parameter` received the full wd=0.1. The consequence is measurable: the median of the 440 gammas is 0.5656, which converts to a half-life of 1.22 token, and 96.1% of heads are shorter than 128 token; the decisive evidence is that the gammas of h16-h21 are bitwise identical across 20 different layers, a signature that only 'zero gradient plus purely multiplicative decay' can produce. The fix inverts the default: no decay is now the default behavior, and a new `--decay_1d` serves as the reverse switch, used only when reproducing an old run. In addition, `_preflight` in `train.py` prints the gamma distribution before training starts and refuses values <= 0. `tests/test_no_decay_1d.py` tests both directions: when enabled, no 1-D parameter (including gamma) may appear in the decay set; when disabled, gamma must appear, otherwise the test is not testing anything.

How it is verified: `tests/test_no_decay_1d.py` tests both directions - when enabled, no 1-D parameter may appear in the decay set; when disabled, `msr.gamma` must appear. On the old weights the measured gamma median is 0.5656, the half-life is 1.22 token, and 96.1% of heads are shorter than 128 token; these three numbers are the consequence of the switch never being on.

see [03]

---

## E07 'The gradient exists, it is just zero' is not a legitimate justification

**Status**: Fixed

`retnet_hf/train.py` L348–L375

```python
    def on_step_end(self, args, state, control, **kw):
        if self._done or int(getattr(state, "global_step", 0)) < self.check_step:
            return control
        dead = self.check_now()
        total = max(1, sum(self._numel.values()))
        dead_numel = sum(self._numel[n] for n in dead)
        frac = dead_numel / total
        if not getattr(state, "is_world_process_zero", True):
            return control
        step = int(getattr(state, "global_step", 0))
        if dead:
            print(f"[grad] step {step}：{len(dead)}/{len(self._numel)} 个参数"
                  f"（{dead_numel:,}/{total:,} 元素 = {frac * 100:.2f}%）"
                  f"没有任何非零梯度：", flush=True)
            for name in dead[:12]:
                print(f"[grad]    {name}", flush=True)
            if len(dead) > 12:
                print(f"[grad]    ... 还有 {len(dead) - 12} 个", flush=True)
        else:
            print(f"[grad] step {step}：{len(self._numel)} 个参数全部拿到非零梯度",
                  flush=True)
        if frac > self.tol:
            raise SystemExit(
                f"[grad] 死参数占比 {frac * 100:.2f}% 超过阈值 {self.tol * 100:.2f}%："
                f"模型里有参数从第 1 步起就没有信号，训练多少步都不会变"
                f"（docs/00 E07/E08）。先修生长/初始化；确认是有意冻结再提高 "
                f"--dead_grad_tol。")
        return control
```

`retnet_hf/train.py` L290–L292

```python
def classify_dead_grad_params(names, seen_max: dict) -> list[str]:
    """返回“从没拿到梯度、或梯度恒为零”的参数名，保持 ``names`` 原序。"""
    return [n for n in names if not seen_max.get(n)]
```

`retnet_hf/train.py` L777–L778

```python
    trainer.add_callback(GradCoverageCallback(
        trainer.model, tol=float(args.dead_grad_tol)))
```

An old comment said that 'in growth, a tensor with gamma=0 still takes part in the forward pass, so the gradient exists and is merely zero', and on that basis DDP's find_unused_parameters was switched off, with the conclusion that 'this model really has no unused parameters'. That reasoning is factually correct - an all-zero gradient is indeed not None - but it papered over the bug: it confirmed that a gradient exists without asking whether the gradient is identically 0. The previous round's 'fix' only rewrote the narrative and added the sentence 'plus an extra gradient coverage check', while no such thing existed in the code; docs/01 itself says 'not yet implemented'. The same failure mode escalated one level: from 'using a comment to rationalize a missing check' to 'using a comment to claim a check that does not exist'. What is being added now is a real check: GradCoverageCallback registers a backward hook on every requires_grad parameter at construction time and records the maximum of grad.abs().max() (reading p.grad in on_step_end is not an option, because HF calls zero_grad after optimizer.step()); at the end of step 1 it aggregates the parameters that 'have no gradient or an identically zero gradient', prints the count, the element share and the first 12 names, and raises SystemExit when that share exceeds --dead_grad_tol (default 2%). Using hooks instead of reading grad afterwards is the key detail of this fix: the latter always reads None inside a real Trainer, so the check would look like it passed while in fact checking nothing. tests/test_grad_coverage.py proves three things: a healthy model has zero dead parameters; switching requires_grad off for blocks.1.msr.q_proj.weight makes it be judged dead; with a threshold of 0, on_step_end raises SystemExit. The third case is one more implementation of 'a guard must be provably able to fail'.

Verification: the four cases in `tests/test_grad_coverage.py` - a healthy model has zero dead parameters; switching requires_grad off for `blocks.1.msr.q_proj.weight` makes it be judged dead; with a threshold of 0, `on_step_end` raises SystemExit; the hooks are removed after the check. `on_step_end` is a callback that is really registered with the Trainer, so deleting `trainer.add_callback(GradCoverageCallback(...))` or deleting the callback class turns this test set red. The old state was: the comment claimed 'plus an extra gradient coverage check', while `grep grad.abs` only found a hit in `scripts/grow_500m.py` (that is the growth script's 5-step probe, not a training-time check).

---

## E08 Parameter counts must be computed over non-zero elements

**Status**: Fixed

`retnet_hf/train.py` L192–L196

```python
def live_param_ratio(model):
    """返回 (非零参数数, 总参数数, 比例)。参数量本身不说明任何问题（E08）。"""
    total = sum(p.numel() for p in model.parameters())
    live = sum(int((p != 0).sum()) for p in model.parameters())
    return live, total, (live / max(1, total))
```

`retnet_hf/train.py` L249–L283

```python
def check_live_ratio(ratio: float, threshold: float = DEFAULT_MIN_LIVE_RATIO,
                     allow_dead: bool = False,
                     baseline: float | None = None,
                     tol: float = LIVE_BASELINE_TOL) -> tuple[bool, str]:
    """E08 的判定核心（纯函数，便于直接测试边界）。

    旧实现只有一条 ``ratio < 0.95 -> SystemExit``，把两件不同的事混成一件：

    * **新模型初始化坏了** —— 大片恒零参数是 bug，必须拒绝；
    * **从一个有意剪枝/生长过的检查点继续训** —— 33% 活参数是**已知状态**，
      拒绝它等于把 rescue 计划的第一步堵死（docs/02 §1）。

    现在的规则（fail-closed，但可显式放行）：

    1. ``ratio >= threshold`` -> 通过；
    2. 显式 ``--allow_dead_params`` -> 通过（调用方打印比例）；
    3. 有历史基线且 ``ratio >= baseline - tol`` -> 通过（续训自己导出的检查点，
       比例本来就不该是 100%；只要没有变得更死就行）；
    4. 其余 -> 拒绝，并给出两条出路。
    """
    if ratio >= threshold:
        return True, f"活参数比例 {ratio * 100:.2f}% >= 阈值 {threshold * 100:.0f}%"
    if allow_dead:
        return True, (f"活参数比例 {ratio * 100:.2f}% < 阈值 "
                      f"{threshold * 100:.0f}%，但给了 --allow_dead_params："
                      f"按已知的死参数继续训练（E08 的显式豁免）")
    if baseline is not None and ratio >= baseline - tol:
        return True, (f"活参数比例 {ratio * 100:.2f}% < 阈值 "
                      f"{threshold * 100:.0f}%，但与历史基线 {baseline * 100:.2f}% "
                      f"一致（容差 {tol * 100:.0f}%）：这是已记录的剪枝/生长状态")
    return False, (
        f"活参数比例 {ratio * 100:.2f}% < 阈值 {threshold * 100:.0f}%。\n"
        f"  如果这是**新模型**：说明生长/初始化坏了（docs/00 E01–E08），先修再训。\n"
        f"  如果这是**有意剪枝/生长过的检查点**：加 --allow_dead_params 显式放行，\n"
        f"  或先跑一次带该开关的训练，把 {LIVE_PARAMS_FILE} 写进输出目录当基线。")
```

`retnet_hf/train.py` L205–L226

```python
def read_live_baseline(*dirs) -> float | None:
    """读取某个历史 run 记录下来的活参数比例；没有记录就返回 None。

    优先读 ``ratio_min``（历史最低），而不是最近一次的 ``ratio``：否则每轮
    悄悄掉 1%（低于 2% 容差）都会被接受，然后把基线重设到更低的位置，
    连续几十轮就等于没有守卫。
    """
    import json
    for d in dirs:
        if d is None:
            continue
        p = Path(d) / LIVE_PARAMS_FILE
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ratio = float(data.get("ratio_min", data["ratio"]))
        except Exception:                                   # noqa: BLE001
            continue
        if 0.0 < ratio <= 1.0:
            return ratio
    return None
```

The old code printed only sum(p.numel()), and numel() ignores values entirely: even when 67% of the values are exactly 0, it still printed 504.18M. The fix is split into two layers: live_param_ratio returns (non-zero element count, total element count, ratio); _preflight calls it immediately after the model is loaded and hands the 'let it through or not' decision to the pure function check_live_ratio. Why a separate decision function: the previous version had a single rule 'ratio < 95% means SystemExit', which conflated two different things - a newly initialized model that is broken (must be rejected) and continued training from a checkpoint that was deliberately pruned or grown (33% live parameters is a known state). The starting point of the rescue plan in docs/02 is exactly retnet_500m_v21_export at 33% live parameters, and the old guard aborted the plan at its very first step. The current rules: pass when the threshold is met; pass with an explicit --allow_dead_params; pass automatically when a historical baseline (live_params.json) exists and the ratio is not below baseline-2%; otherwise reject and print both ways out at the same time. The baseline is written by write_live_baseline into the output directory after every successful training start, so 'continue training from a checkpoint you exported yourself' does not need a manual switch every time, while 'this round killed the parameters again' is still caught by the baseline comparison. The gamma check now walks both named_parameters and named_buffers: with use_gla=False gamma is a buffer, and the old implementation only checked parameters, so on such models that check was dead. tests/test_live_params.py requires the zero-parameter counting to be element-exact; tests/test_preflight_live.py covers the four boundaries of the pure function, baseline read and write, tolerance of a corrupt file, and the two real paths '33% live parameters + switch lets training start / without the switch it is rejected'.

Verification: the three cases in `tests/test_live_params.py` - a new model has a ratio >99%, zeroing a whole block lowers the ratio, and zeroing half of a known tensor reduces the live count by exactly the corresponding number of elements. `tests/test_preflight_live.py` additionally covers the decision boundaries and the real paths: a ratio of 0.33 is rejected without the switch, accepted with --allow_dead_params, accepted when it matches the baseline, and rejected when it is deader than the baseline; with use_gla=False (gamma is a buffer), gamma<=0 is still rejected. The old code printed only sum(p.numel()) and still printed 504,179,896 when 67% were zero.

---

## E09 Packing must write document boundaries by default

**Status**: Fixed

`scripts/build_flat_corpus.py` L188–L192

```python
    ap.add_argument("--add-eos", action="store_true", default=True,
                    help="每条记录末尾追加 eos。**默认开启**：v21 的所有 hf_pack_* 都漏了它，"
                         "78% 的训练步数跑在无文档边界的流上（docs/00 E09/E10）。")
    ap.add_argument("--no-add-eos", dest="add_eos", action="store_false",
                    help="显式关闭。只有确认下游会自己补边界时才用；关之前先读 docs/00 E09。")
```

The old implementation's --add-eos was store_true, defaulting to False. Every hf_pack_* (code, web pages, textbooks, math, Chinese programming) was built with it, so none of them has document boundaries: the last line of one document runs straight into the prompt of the next. The consequence is not 'slightly worse quality' but that the model learns to ignore distant content - inside a 2048 window the cross-document part was noise to begin with. The fix changes the default to True and adds --no-add-eos as an explicit inverse switch; turning it off comes with a reminder in the help text to read E09 first. This change turns 'forgetting to pass the argument' from an error into safe behavior, instead of relying on every call site to remember a flag. build_flat_corpus still records the actual value of add_eos in meta.json, for downstream consumers (for example the E11 guard in build_ctx_mix) to cross-check.

Verification: `tests/test_build_scripts.py::test_add_eos_is_default` runs the command-line script directly without passing `--add-eos`, then asserts that the number of eos(2) in the produced shards is at least the number of records and that `add_eos` in `meta.json` is true. All old packs were false; in v21 only stage 0 (where `prepare_pretrain_v2.py` added eos explicitly) had boundaries, and the following 124,652 steps (78%) trained on a stream without boundaries.

see [16]

---

## E10 Packing must specify the text field explicitly, and a wrong field must fail immediately

**Status**: Fixed

`scripts/_data_supervisor.py` L47–L54

```python
    dict(name="code",     src="ling_coder",        out="hf_pack_code",        n=27, extract=False,
         cap=2_500_000_000, need_gb=2.0,   text_field="text"),
    dict(name="web",      src="chinese_web_hq",    out="hf_pack_chinese_web", n=13, extract=False,
         cap=700_000_000,   need_gb=0.64,  text_field="text"),
    # textbook 解压出 696,600 个文件，其中 2/3 是 stats.json（非文本），
    # 不跳过会被算成 66% 错误率，触发分词脚本自带的「错误率 > 1% 判失败」守卫。
    dict(name="textbook", src="fineweb_edu_zh_v3", out="hf_pack_textbook",    n=18, extract=True,
         cap=400_000_000,   need_gb=0.36,  text_field="text", skip="stats.json"),
```

`scripts/build_flat_corpus.py` L252–L260

```python
    _first = next(iter_records(patterns, kind, skip=skip_names), None)
    if _first is None:
        raise SystemExit(f"输入里没有任何可读记录：{patterns}")
    _missing = [f for f in fields if f not in _first]
    if len(_missing) == len(fields):
        raise SystemExit(
            f"--text-field {fields} 在第一条记录里一个都不存在。\n"
            f"  可用字段: {sorted(_first)}\n"
            f"  请显式指定正文列；不要用 auto（它会把数值/元数据列拼进正文）。")
```

The old implementation used --text-field auto and guessed the text body from a blacklist of field names; prog_zh's 10 numeric metadata columns (alnum_ratio, perplexity, industry_type and so on) were therefore spliced into the training stream as text. This kind of error raises nothing and only lowers data quality, and could only be found by spot-check decoding. The fix has two layers: callers (_data_supervisor, _prep_ext) must pass text_field explicitly, with each corpus's value written in SPECS and annotated 'if the field name is wrong, the script will fail at the first record'; build_flat_corpus gains an early-failure check - it reads the first record and, if none of the given fields exists, exits immediately and prints the fields that are actually available. The old behavior recorded each record as an error and failed only at the end, after hours of tokenization, with a >1% error rate, or worse: when the field was missing from only some records it silently dropped that part of the data. Early failure turns 'you find out hours later' into 'you find out within seconds'.

Verification: `tests/test_build_scripts.py::test_wrong_text_field_fails_fast` passes a field name that does not exist and requires a non-zero exit code with 'none of them exists / available fields' in the output; the old behavior failed only after tokenization had finished, with a >1% error rate. The concrete evidence is that the 10 numeric columns of `prog_zh` were selected by `auto`; this kind of contamination raises nothing and only lowers quality.

see [16]

---

## E11 Building 16K packs while knowing boundaries are missing must be refused

**Status**: Fixed

`scripts/build_ctx_mix.py` L76–L92

```python
    # 2026-10-20（docs/00 E11）：本脚本的 docstring 自己写着
    # "flat 分片的原始边界在分词时就丢了，无法恢复"——然后继续用它造 16K 包。
    # 知道缺陷还继续用，等于把缺陷固化进下一轮。现在默认拒绝：
    # 每个 flat 来源都必须有 meta.json，且 add_eos 为真；
    # 没有就停，让你去重建那份语料，而不是把边界问题带进训练。
    def _require_eos(d: Path, label: str) -> None:
        meta = d / "meta.json"
        if not meta.exists():
            raise SystemExit(
                f"[E11] 来源 {label} 缺少 {meta}，无法确认文档边界。\n"
                f"       先用带 --add-eos 的 build_flat_corpus.py 重建它。")
        m = json.loads(meta.read_text(encoding="utf-8"))
        if not m.get("add_eos"):
            raise SystemExit(
                f"[E11] 来源 {label} 的 meta.json 写着 add_eos=false，是无边界流。\n"
                f"       用带 --add-eos 的 build_flat_corpus.py 重建，\n"
                f"       或显式加 --allow-boundaryless 并把这条写进实验记录。")
```

The script's own docstring says 'the original boundaries of flat shards are lost at tokenization time and cannot be recovered', and then it keeps using them to build 16K packs. Entrenching a known defect into the next round is worse than not knowing about it. The fix turns this into a default refusal: every flat source must have a `meta.json` with add_eos true; a missing file or add_eos=false triggers an immediate `SystemExit` and suggests rebuilding with `build_flat_corpus.py` passing `--add-eos`. Anyone who genuinely wants to accept boundaryless streams must pass `--allow-boundaryless` explicitly; the log prints a warning line, and this also goes into the experiment record. This is not 'one more switch', but the conversion of a silent quality loss into an explicit decision that requires a signature.

How it is verified: two cases in `tests/test_build_scripts.py` - a directory without `meta.json` must exit non-zero with a message that document boundaries cannot be confirmed; with `--allow-boundaryless`, this kind of refusal dialogue must not appear and an explicit warning must be printed. The old script admitted the loss of boundaries in its docstring yet still used them to build 16K packs, a typical case of entrenching a defect while knowing about it. The switch is written into the README: every use must record the reason and the scope of impact in the experiment record, and must also report the share of that pack in the total training tokens, so that the extent of the impact can be judged afterwards.

see [16]

---

## E12 Opening the flat stage must confirm that document boundaries exist

**Status**: Fixed

`retnet_hf/data.py` L124–L149

```python
    def _check_boundaries(self, sample: int = 2_000_000) -> None:
        """抽查前 sample 个 token 里是否出现 eos；没有就拒绝。

        eos id=2 由 tokenizer 固定（pad=0 / bos=1 / eos=2），见
        `scripts/prepare_pretrain_v2.py` 的构造与 `bpe_tokenizer.json`。
        不逐片全扫，是因为一片 2.5e8 token；抽查 2M 足够发现"整包没边界"。
        """
        if not self.shards:
            return
        per = max(1, sample // len(self.shards))
        seen = 0
        for _p, arr, _n in self.shards:
            take = min(int(arr.shape[0]), per)
            if take <= 0:
                continue
            seg = np.asarray(arr[:take])
            seen += take
            if int((seg == 2).sum()) > 0:
                return
            if seen >= sample:
                break
        raise ValueError(
            f"阶段「{self.name}」的 flat 分片里抽查 {seen:,} 个 token 没有任何 <eos>。"
            f"这条流没有文档边界，训练会让模型学会忽略远距离内容（docs/00 E12）。"
            f"用带 --add-eos 的 build_flat_corpus.py 重建；"
            f"如果你确认这是有意的，在计划里给该阶段写 require_eos: false。")
```

`retnet_hf/plan.py` L86–L95

```python
    def signature(self) -> tuple:
        """热更新比较用的完整签名。

        旧实现只比 ``(name, data_dir, steps)``，于是改 ``weight`` /
        ``require_eos`` / ``exclude`` 不会触发 ``update_stages`` —— 计划文件
        改了、进程里还是旧参数，属于静默不生效。
        """
        return (self.name, self.data_dir, int(self.steps), self.mode, bool(self.gc),
                self.min_shards, tuple(self.exclude), self.weight,
                bool(self.require_eos))
```

`retnet_hf/train.py` L437–L451

```python
def plan_stage_specs(plan_obj) -> list[dict]:
    """计划 -> ``StagePackedDataset`` 的 stage dict（train.py 与测试共用）。

    单独成函数是为了让“计划 JSON -> parse_plan -> 数据集”这条链路可以被整体
    测试。旧实现把这段 dict 内联在 ``main()`` 里、测试只直接构造
    ``_FlatStage(require_eos=...)``，于是 ``require_eos`` 在 Stage / parse_plan
    里被丢掉这件事，任何测试都看不见（docs/00 E12/P0-3）。
    """
    return [{"name": s.name, "data_dir": str(s.resolved_dir()), "mode": s.mode,
             "gc": s.gc, "min_shards": s.min_shards, "exclude": s.exclude,
             # E12：require_eos 必须从计划一路带到数据集。旧实现里
             # Stage / parse_plan / 这个 dict 三处都漏，于是守卫的错误提示
             # “在计划里给该阶段写 require_eos: false”根本不可能生效。
             "weight": s.weight, "require_eos": s.require_eos}
            for s in plan_obj.stages]
```

The flat stream is sliced at a fixed width, with cut points determined by line numbers and unrelated to document boundaries; if no eos was added during tokenization, the model sees 'the last line of one document plus the start of the next'. The old `_FlatStage` was completely silent about this. The first-round fix spot-checked the first 2M tokens when opening the stage, raised `ValueError` when no eos was present, and offered two ways out: rebuild with a script that passes `--add-eos`, or write `require_eos: false` explicitly for that stage in the plan. But the second way out did not actually exist: `plan.Stage` had no `require_eos` field, `parse_plan` dropped it, and `train.py` did not pass it when constructing the stage dict, so the key that the error message told operators to write never took effect - the guard raised an error, the operator did what it said, and the same error came back. `require_eos` is now an official field of `Stage`; `parse_plan` parses it, `plan_stage_specs` carries it into `StagePackedDataset`, and hot reload (`Stage.signature` includes it) does not lose it either. The default `require_eos=True` is deliberate: a stale pack that forgot to add eos must be exposed immediately when it is used, rather than halfway through training when it becomes apparent that the model cannot learn long-range dependencies. Validation exercises the whole chain instead of constructing `_FlatStage` directly: `tests/test_plan_eos.py` starts from plan JSON, goes through `parse_plan` and `plan_stage_specs` to `stage_ready`, and proves that `require_eos: false` genuinely opens the stage, that a missing key is still refused, and that hot reload does not drop the key.

How it is verified: `tests/test_plan_eos.py` starts from plan JSON and walks the full chain - `require_eos` defaults to True, writing false makes `StagePackedDataset.stage_ready(0)` genuinely ready, omitting it is still refused and the key name in the error message is exactly `require_eos: false`, and hot reload does not lose the key; `tests/test_eos_present.py` separately tests the three states of `_FlatStage` itself. In the old implementation `Stage` had no such field and `parse_plan` dropped it outright, so no test could reach this chain.

see [16]

---

## E13 GC must prove data is rebuildable before deleting it

**Status**: Fixed

`retnet_hf/data.py` L370–L432

```python
    def gc_stage(self, i: int) -> int:
        """删除第 i 段的**数据文件**（已消费完才调）。返回释放字节数。

        安全约束（宁可少删也不能删错）：
        1. 只删本阶段自己登记的文件，不 rm -rf 整个目录；
        2. 该目录若被别的阶段引用，跳过；
        3. 该目录若是验证集目录（``protect()`` 登记过），跳过；
        4. 目录里必须有 ``meta.json`` 证明“可重建”（E13）。
        """
        if not (0 <= i < len(self._specs)):
            return 0
        from retnet_hf.provenance import norm_key
        sp = self._specs[i]
        d = Path(sp["data_dir"]).resolve()
        d_key = norm_key(d)
        if d_key in self._protected:
            print(f"[gc] 跳过阶段 {i}「{sp['name']}」：{d} 是验证集目录", flush=True)
            return 0
        for j, other in enumerate(self._specs):
            if j != i and norm_key(other["data_dir"]) == d_key:
                print(f"[gc] 跳过阶段 {i}「{sp['name']}」：目录与阶段 {j} 相同", flush=True)
                return 0
        st = self._cache.get(i)
        if st is None or not hasattr(st, "files"):
            return 0
        # 2026-10-20（docs/00 E13）：删除前必须证明"可重建"。
        # 上一轮只查了"是否与别的阶段同目录""是否验证集"，然后就删了，
        # 结果 hf_pack_* 与 pretrain_v2 的分片全部不可复核。
        # 现在要求目录里有一份 meta.json，并且写明 source_ids 或 rebuild_cmd；
        # 缺一个就拒绝删除——宁可少删，不可删错。
        meta = d / "meta.json"
        if not meta.exists():
            print(f"[gc] 拒绝删除阶段 {i}「{sp['name']}」：缺少 {meta}，"
                  f"无法证明这份数据可重建", flush=True)
            return 0
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as exc:                                # noqa: BLE001
            print(f"[gc] 拒绝删除阶段 {i}「{sp['name']}」：{meta} 读不了（{exc}）",
                  flush=True)
            return 0
        if not _proves_rebuildable(m):
            print(f"[gc] 拒绝删除阶段 {i}「{sp['name']}」：{meta} 里没有 "
                  f"rebuild_cmd / source_ids，也没有 sources+text_fields；"
                  f"无法证明这份数据可重建", flush=True)
            return 0
        targets = st.files()
        if hasattr(st, "close"):
            st.close()                 # 先松 mmap 句柄，否则 Windows 删不掉
        freed = 0
        for p in targets:
            try:
                sz = p.stat().st_size
                p.unlink()
                freed += sz
            except FileNotFoundError:
                pass
            except Exception as exc:                            # noqa: BLE001
                print(f"[gc] 删除 {p} 失败: {exc}", flush=True)
        self._cache.pop(i, None)
        print(f"[gc] 阶段 {i}「{sp['name']}」已回收 {freed / 1e9:.2f} GB 磁盘"
              f"（{d}）", flush=True)
        return freed
```

`retnet_hf/data.py` L238–L248

```python
def _proves_rebuildable(meta: dict) -> bool:
    """``meta.json`` 是否足以证明这份数据可重建（E13）。

    新产线写 ``rebuild_cmd``（完整 argv）与 ``source_ids``；旧产线只写了
    ``sources`` + ``text_fields``（字段名与通配符都在，足以按同一口径重跑），
    这种也要认——否则所有已经产出的包都会被判“不可重建”，GC 永远删不掉，
    plan 依赖的磁盘回收一点都不会发生。
    """
    if meta.get("rebuild_cmd") or meta.get("source_ids"):
        return True
    return bool(meta.get("sources")) and bool(meta.get("text_fields"))
```

`scripts/build_flat_corpus.py` L341–L343

```python
        "rebuild_cmd": "python scripts/build_flat_corpus.py " + " ".join(argv_used),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
```

The old implementation checked only two things: delete only files registered by the current stage, and skip a directory when another stage references it. It was missing the most important rule - whether this data can be rebuilt from its source. That is exactly where the previous round wiped `hf_pack_*` and the `pretrain_v2` shards: the tokenizer script did rmtree on the raw parquet after success, then gc deleted the shards, so the source and the artifact disappeared together. The first-round fix required `meta.json` to carry `source_ids` or `rebuild_cmd`, refusing deletion when either was missing. The problem was that the production pipeline never wrote those two keys: the meta from `build_flat_corpus` had only sources/text_fields, and the only test case that passed hand-wrote a meta containing `rebuild_cmd` - the test fixture was not a production artifact. The result went from 'deleting the wrong thing' to 'never deleting anything': all 6 stages with gc:true printed a refusal on every run of the plan and freed 0 bytes, so the disk-reclamation premise of the plan collapsed completely. Both sides are now changed: production writes argv and `rebuild_cmd` into meta, and `source_ids` is generated from the input globs (needed by the declaration layer in E14); on the GC side, the rebuildability proof accepts three forms - `rebuild_cmd`, `source_ids`, and the legacy production `sources`+`text_fields`. In addition, 'skip the validation set directory' now has a real implementation (`protect()` plus normalized paths), because the old docstring stated it as a safety constraint while the function body did not contain it. Validation was switched to a real production closed loop: `tests/test_producer_gc.py` trains a small tokenizer on the spot, runs `build_flat_corpus` to produce shards and meta, then calls `gc_stage` with that meta and asserts that freed>0 and that the shards are deleted; it also tests that legacy meta (only `sources`+`text_fields`) is still deletable, that missing proof is still refused, and that a protected directory is refused.

How it is verified: `tests/test_producer_gc.py` runs the real production pipeline - training a small tokenizer on the spot, executing `build_flat_corpus`, reading the meta it wrote, and calling `gc_stage`, then asserting that freed>0 and that the shards are deleted; it separately checks that legacy meta (only `sources`+`text_fields`) is still deletable, that no proof at all is still refused, and that a protected directory is refused. The old test (`test_plan_gc` 4b) hand-wrote a meta that real production never generates, so the 'rebuildable' property never held on real data.

---

## E14 Training must be refused when the validation set shares a directory with a training stage

**Status**: Fixed

`retnet_hf/train.py` L454–L472

```python
def assert_val_disjoint(val_path, train_dirs, seq_len: int = 2048) -> None:
    """验证集目录不能与任何训练阶段同源（E14）。

    上一轮 val_dir 是 chat_pack_v13，而 sft_mix 里 40% 就是它：训练期
    eval_loss 因此不可信、best-model 选择也被污染。

    旧实现只比较 ``Path(...).resolve()`` 字符串——那是**路径相等**，不是
    **数据相等**。现在三层都查（见 ``retnet_hf/provenance.py``）：
    规范化路径、meta.json 声明的来源、以及按行内容指纹。任何一层命中都拒绝。
    """
    from retnet_hf.provenance import describe, overlap_report
    rep = overlap_report(val_path, train_dirs, seq_len)
    hits = rep["path_hits"] + rep["source_hits"] + rep["content_hits"]
    if hits:
        raise SystemExit(
            "[preflight] 验证集与训练阶段同源，拒绝开训：\n  "
            + "\n  ".join(hits)
            + "\n验证集进了训练，eval_loss 与 best-model 选择都不可信（docs/00 E14）。")
    print(f"[preflight] 验证集隔离 OK：{describe(rep)}", flush=True)
```

`retnet_hf/provenance.py` L160–L195

```python
def overlap_report(val_dir, train_dirs, seq_len: int,
                   max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    """比较验证集与每个训练目录：路径 / 声明来源 / 内容指纹。

    返回 ``{"path_hits": [...], "source_hits": [...], "content_hits": [...],
    "val_rows": n, "val_truncated": bool, "scanned": {dir: rows},
    "notes": [...]}``。命中项都是目录字符串，由调用方决定报错文案。
    """
    v_key = norm_key(val_dir)
    v_src = declared_sources(val_dir)
    v_rows = dir_row_hashes(val_dir, seq_len, max_rows, prefer_val=True)
    out = {"path_hits": [], "source_hits": [], "content_hits": [],
           "val_rows": v_rows["rows"], "val_truncated": v_rows["truncated"],
           "scanned": {}, "notes": []}
    if not v_rows["hashes"].size:
        out["notes"].append(f"验证集 {val_dir} 里没有可读的 npy 行，内容层跳过")
    if not v_src:
        out["notes"].append(f"验证集 {val_dir} 没有 meta.json 来源声明，声明层跳过")
    for d in train_dirs:
        if norm_key(d) == v_key:
            out["path_hits"].append(str(d))
        src = declared_sources(d)
        if v_src and src and (v_src & src):
            out["source_hits"].append(
                f"{d}（共同来源 {sorted(v_src & src)}）")
        h = dir_row_hashes(d, seq_len, max_rows)
        out["scanned"][str(d)] = h["rows"]
        if h["truncated"]:
            out["notes"].append(
                f"{d} 只扫了前 {h['rows']:,} 行（达到上限），内容层是不完整检查")
        if v_rows["hashes"].size and h["hashes"].size:
            common = np.intersect1d(v_rows["hashes"], h["hashes"])
            if common.size:
                out["content_hits"].append(
                    f"{d}（{int(common.size)} 行完全相同）")
    return out
```

`retnet_hf/provenance.py` L113–L157

```python
def dir_row_hashes(d, seq_len: int, max_rows: int = DEFAULT_MAX_ROWS,
                   prefer_val: bool = False) -> dict:
    """目录内 npy 的行哈希，最多 ``max_rows`` 行。

    ``prefer_val=True`` 用于**验证集目录**：优先读 ``val.ids.npy``。
    旧实现一律跳过文件名含 ``val`` 的分片，于是验证集目录里的行根本不会被
    哈希 —— 内容层永远比较两个空集合。训练侧仍是 packed 读 ``train.ids.npy``、
    flat 读全部非 val 分片（与 ``_FlatStage._scan`` 同一口径）。

    返回 ``{"hashes": np.ndarray, "rows": int, "truncated": bool,
    "files": [name]}``。
    """
    d = Path(d)
    width = int(seq_len) + 1
    if prefer_val:
        pv = d / "val.ids.npy"
        candidates = [pv] if pv.is_file() else sorted(d.glob("*.npy"))
    else:
        pt = d / "train.ids.npy"
        candidates = [pt] if pt.is_file() else \
            sorted(p for p in d.glob("*.npy") if "val" not in p.name.lower())
    parts: list[np.ndarray] = []
    files: list[str] = []
    scanned = 0
    truncated = False
    for p in candidates:
        if scanned >= max_rows:
            truncated = True
            break
        try:
            arr = np.load(p, mmap_mode="r")
        except Exception:                                   # noqa: BLE001
            continue
        if arr.ndim not in (1, 2):
            continue
        if arr.ndim == 2 and arr.shape[1] != width:
            continue
        h, n, cut = _row_hashes(np.asarray(arr), width, max_rows - scanned)
        parts.append(h)
        scanned += n
        files.append(p.name)
        truncated = truncated or cut
    hashes = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)
    return {"hashes": hashes, "rows": int(scanned), "truncated": truncated,
            "files": files}
```

In the previous round val_dir pointed at chat_pack_v13, and 40% of the sft_mix for the last 14,000 steps was chat_pack_v13. The validation set went into training: the eval_loss falling from 4.55 to 2.4450 over the final stretch was not the model getting better, it was the model having seen the validation set; load_best_model_at_end would then use that contaminated number to pick the 'best' model. The first fix only compared Path(...).resolve() strings: the comment claimed to cover 'relative paths, case, trailing slash', while the 'other spelling' in the test was the very same absolute path - it guarded **path equality**, whereas the incident was **data equality**. The fix now has three layers (retnet_hf/provenance.py): (1) normalize the paths with resolve + normcase, genuinely covering relative paths, trailing slashes and case; (2) meta.json declares provenance, either source_ids or the older pipeline's sources+text_fields, and any intersection is rejected; (3) per-line content fingerprints, a 64-bit FNV hash per line, rejecting when identical lines exist in the two directories - it does not look at file names, so 'the same records repackaged into another directory' is also caught. The content layer has a 200k-line cap for large corpora (about 0.8 GB), and when it truncates it says so truthfully in the log instead of pretending to have scanned everything. On the verification side the point is that the old test can no longer pass vacuously: tests/test_val_disjoint.py uses three **genuinely different** spellings (os.path.relpath, trailing slash, case) and asserts rejection for each; uses meta.json to declare a shared source (with different names) to trigger rejection; copies two lines from the training set into the validation set to trigger a content hit; and asserts that the validation side reads val.ids.npy rather than an empty set.

Verification: `tests/test_val_disjoint.py` triggers rejection one by one with genuinely different spellings (os.path.relpath, trailing slash, case); uses two meta.json files that declare a shared source (with different directory names) to trigger the provenance layer; copies two validation lines into the training set to trigger the content layer; and asserts that the validation side reads val.ids.npy. The 'other spelling' in the old test was merely the same absolute path, which amounts to no test at all.

---

## E15 Best-model selection must not use a contaminated eval_loss

**Status**: Fixed

`retnet_hf/train.py` L718–L724

```python
        # 2026-10-20（docs/00 E15）：关闭“按 eval_loss 选最佳”。
        # 训练期 eval 用的是 val_dir（上一轮是 chat_pack_v13），而最后 14,000 步的
        # sft_mix 里 40% 就是它——这个指标被污染，拿它选最佳等于拿训练集选模型。
        # 正确做法：训练中只按步数存 checkpoint，跑完后用独立域内 held-out
        # （data_cache/pretrain_v2/*_val.npy，绝不进训练）在离线脚本里选。
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
```

The old config read `load_best_model_at_end=True` and `metric_for_best_model=eval_loss`. Two problems compounded: first, the in-training eval used `val_dir`, and `val_dir` may have been trained on (E14); second, even if val is clean, it is only loss over the conversation distribution and does not represent in-domain pretraining capability. The final model of the previous round was 0.655 nat worse than step 17,500 on the in-domain held-out set, yet it was treated as 'converged' because eval_loss fell late in training. The fix turns `load_best_model_at_end` off: during training, checkpoints are saved by step only, and after the run an independent in-domain held-out set (`data_cache/pretrain_v2/*_val.npy`, which never enters training) is compared offline to select the model. Offline selection now has an executable tool: `scripts/eval_holdout.py` accepts `--model` pointing at any checkpoint directory and reports each candidate's loss under the same definition (`--by-position`/`--ctx-ablation` are additional diagnostics). The `metric_for_best_model` field is kept but no longer drives loading, so old configs do not fail outright. The previous version of the documentation contradicted itself here: docs/02 presented 'keep load_best_model_at_end' as a hard prerequisite, docs/01 said 'not fixed', and the code had it disabled. All three are now unified as off plus offline selection, and docs/02 section 2 E15 has been rewritten.

How it is verified: `tests/test_train_config_guards.py::test_best_model_selection_is_disabled` asserts that `load_best_model_at_end=False` exists in the source and that `=True` no longer appears; offline model selection instead runs `eval_holdout.py` on `pretrain_v2/*_val.npy`. The old behavior corresponds to the fact that the final model scored 3.5023 on the in-domain held-out set, 0.655 nat worse than 2.8471 at step 17,500.

---

## E16 The parallel path must not have normalization that chunkwise does not have

**Status**: Fixed

`retnet_hf/modeling_retnet.py` L191–L202

```python
    def _decay_mask(self, length, device, dtype):
        idx = torch.arange(length, device=device)
        delta = idx[:, None] - idx[None, :]
        causal = delta >= 0
        exp = delta.masked_fill(~causal, 0)
        mask = self.gamma.to(device=device, dtype=torch.float32)[:, None, None] ** exp
        mask = mask.masked_fill(~causal, 0.0)
        # 2026-10-20: 删除了原来的 `mask / mask.sum(...).sqrt()`。
        # 那一步只存在于 parallel 路径，chunkwise/recurrent 都没有，导致
        # 训练（chunkwise）与生成 prefill（parallel）算的不是同一个函数。
        # 见 docs/00 E16。
        return mask.to(dtype=dtype)
```

`retnet_hf/modeling_retnet.py` L204–L210

```python
    def _parallel_retention(self, q, k, v):
        bsz, _, tgt_len, _ = q.shape
        mask = self._decay_mask(tgt_len, q.device, q.dtype)
        scores = torch.matmul(q, k.transpose(-1, -2)) * mask
        # 2026-10-20: 删除了原来的 `scores / scores.detach().abs().sum(...)`。
        # 同一理由：这是 parallel 独有的第二次归一化，见 docs/00 E16。
        return torch.matmul(scores, v)
```

The old code divided the mask by sqrt(sum of mask) inside _decay_mask and divided the scores by sum of abs(scores) inside _parallel_retention; chunkwise and recurrent had neither step. So training (chunkwise), prefill (parallel) and decoding (recurrent) were three different functions, measured at mean abs(delta logit)=0.0158 with an argmax agreement rate of 99.66%. Worse, the evaluation script eval_holdout.py forces parallel, so every held-out number from the previous round (3.6287 / 2.8471 / 3.5023) was measuring a function other than the one that was trained. The fix removes both normalizations, so that the three paths differ only in how chunking and looping are implemented, not in the math. tests/test_parallel_chunkwise_equiv.py compares parallel vs chunkwise and parallel vs recurrent on a tiny model, with relative tolerances set against the logits scale, to stop anyone from adding private operations to one of the paths again.

Verification: `tests/test_parallel_chunkwise_equiv.py` compares parallel vs chunkwise and parallel vs recurrent, with relative tolerances set against the logits scale. The old implementation measured mean abs(delta logit)=0.0158 with an argmax agreement rate of 99.66%, and the evaluation script forces parallel, so those three held-out numbers were not measuring the function that was trained.

see [01][04]

---

## E17 The save_total_limit default must match the template

**Status**: Fixed

`retnet_hf/train.py` L580–L584

```python
    ap.add_argument("--save_total_limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT,
                    help="保留最近 N 个 checkpoint-*；<=0 表示不限制。"
                         "默认值来自 retnet_hf/defaults.py（与 run_task.py 同一个常量，"
                         "docs/00 E17：以前模板写 4、这里硬编码 3，多出来的那个检查点"
                         "会被 HF 按 mtime 静默轮转删除）。")
```

`run_task.py` L182–L225

```python
def build_cmd_full(cfg):
    cmd = build_cmd(cfg)
    if cfg["grad_checkpoint"]:
        cmd.append("--grad_checkpoint")
    if not cfg["compile"]:
        cmd.append("--no-compile")
    # compile_mode 只在真的编译时才传（否则它就是个死字段，check_task_config 会报出来）
    if cfg.get("compile") and cfg.get("compile_mode"):
        cmd += ["--compile_mode", str(cfg["compile_mode"])]
    # 无条件拼进去（不是条件 flag）：以前是 `if grad_accum > 1` 才拼，于是
    # `grad_accum: 1` 的配置在 check_task_config.py 里被报成「死字段」。
    # train.py 的默认值也是 1，所以无条件拼不改变任何行为，但让守卫的规则保持
    # 「JSON 写了就必须在命令行里出现」这一条，不必为它开特例。
    cmd += ["--grad_accum", str(cfg.get("grad_accum", 1))]
    if cfg.get("optim"):
        cmd += ["--optim", str(cfg["optim"])]
    # E06/P1-3：以前只在 truthy 时追加 `--no_decay_1d`，于是 JSON 里写
    # `no_decay_1d: false` 被**静默忽略**，train.py 的默认 True 照样生效 ——
    # 配置说关、实际开着。现在两个方向都显式传，由 train.py 的
    # `--no_decay_1d` / `--decay_1d` 一对反向开关消费。
    cmd.append("--no_decay_1d" if cfg.get("no_decay_1d", DEFAULT_NO_DECAY_1D)
               else "--decay_1d")
    if cfg.get("ddp_find_unused_parameters"):
        cmd.append("--ddp_find_unused_parameters")
    # E08：从剪枝/生长过的检查点起步时必须显式承认死参数（docs/02 的 rescue 计划）。
    if cfg.get("allow_dead_params"):
        cmd.append("--allow_dead_params")
    if cfg.get("min_live_ratio") is not None:
        cmd += ["--min_live_ratio", str(cfg["min_live_ratio"])]
    if cfg.get("dead_grad_tol") is not None:
        cmd += ["--dead_grad_tol", str(cfg["dead_grad_tol"])]
    if cfg["resume"]:
        cmd.append("--resume")
    # 多卡：用 torch.distributed.run 包一层，train.py 本身不用改入口。
    # ⚠️ batch_size 是**每张卡**的：2 卡 × batch 8 = 单卡 batch 16 的等效批量，
    #    所以切到多卡时要把 batch_size 减半、steps 不变（LR/评测节奏也不变）。
    nproc = int(cfg.get("nproc", 1) or 1)
    if nproc > 1:
        script, rest = cmd[1], cmd[2:]
        cmd = [sys.executable, "-m", "torch.distributed.run",
               "--nproc_per_node", str(nproc),
               "--master_port", str(cfg.get("master_port", 29500)),
               script] + rest
    return cmd
```

`run_task.py` L97–L111

```python
def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = REQUIRED_WITH_PLAN if cfg.get("plan") else REQUIRED
    for k in required:
        if not cfg.get(k):
            raise ValueError(f"missing required key: {k}")
    if cfg.get("plan"):
        if not Path(str(cfg["plan"])).exists():
            raise ValueError(f"plan 文件不存在: {cfg['plan']}")
        cfg.setdefault("data_dir", "")
        cfg.setdefault("max_steps", 0)
    merged = {**DEFAULTS, **cfg}
    if merged["compile"]:
        merged["compile"] = True
    return merged
```

In the old implementation the default in train.py was 3, while the task template said 4. Historically the plan branch of run_task did not pass this parameter either, so 'the config says 4, the actual run uses 3'; the extra checkpoint was rotated out by HF based on mtime, and the deletion log is logger.info while transformers defaults to WARNING, so nothing was printed. This kind of 'two sets of defaults' is the hardest class of bug to find: the config looks right, the code looks right, and only counting both sides reveals it. The first fix only changed the default in train.py from 3 to 4, while DEFAULTS in run_task.py was **still 3** and the plan branch did not pass the parameter - so 'the config says 4, the actual run uses 3' persisted unchanged on the launcher path, and the tests at the time only regex-matched train.py and were all green. The default is now a single constant in retnet_hf/defaults.py, imported by both train.py and run_task.py; both the plan and the non-plan branches of run_task pass the same value explicitly. Verification no longer inspects literals but goes through behavior: test_save_total_limit_is_one_shared_constant asserts that both places come from the same constant; test_run_task_passes_save_total_limit_in_both_branches asserts, for both branches and both values (the default and 6), that the number in argv equals the configured value - the old code did not pass the parameter at all in the plan branch, so this test goes red immediately.

Verification: `tests/test_train_config_guards.py` goes through behavior rather than literals - `test_save_total_limit_is_one_shared_constant` asserts that the parser default in train.py equals the constant in `retnet_hf/defaults.py`; `test_run_task_passes_save_total_limit_in_both_branches` asserts, for the plan and non-plan branches and for both values (the default and 6), that the number in argv equals the configured value. The old code did not pass this parameter at all in the plan branch and fell back to 3 in the non-plan branch, while the old tests only regex-matched train.py, so the original E17 defect (the config says 4, the actual run uses 3) persisted unchanged and stayed green.

---

## E18 Hot-editing the total step count to raise the LR must be rejected by default

**Status**: Fixed

`retnet_hf/plan.py` L508–L523

```python
            if new.total_steps != old.total_steps:
                # docs/00 E18：改总步数会把余弦重铺，当前 LR 可能**向上跳**。
                # 上一轮就这样在训练中途把 LR 抬过两次，等于做了一次没人批准的
                # 热重启；日志里只留下一行"余弦随之重铺"。
                # 判定抽到 should_accept_reschedule，业务逻辑与判定分离，便于测试。
                accept, lr_now_old, lr_now_new = should_accept_reschedule(
                    old, new, state.global_step)
                if not accept:
                    self.plan = old
                    print(
                        f"[plan] 拒绝本次计划变更：total_steps {old.total_steps} -> "
                        f"{new.total_steps} 会让当前 LR 从 {lr_now_old:.3e} 抬到 "
                        f"{lr_now_new:.3e}。若确实要重铺余弦，在该计划 JSON 里加 "
                        f"\"allow_reschedule\": true 后重新保存。", flush=True)
                    return
                state.max_steps = new.total_steps
```

The plan is hot-reloadable: changing stages or steps changes total_steps, and the LR cosine is laid out over total_steps, so the target LR at the current step can suddenly jump up. In the previous round the plan was hot-edited 6 times in one day, the log kept only a single line saying 'the cosine is re-laid accordingly', and the actual effect was a controlled hot restart that nobody had approved. The fix introduces allow_reschedule: when a new plan would raise the current LR by more than 2% and that field is not true, the whole change is rejected, the old plan is kept, and both LRs plus the config line that needs to be written are printed. Going down is safe (the learning rate gets smaller) and is allowed. Why not simply forbid changing the total step count: shortening the budget is a common loss-cutting operation that only makes the LR smaller; what is genuinely dangerous is extending the budget to push the LR back up - that overwrites a low-LR stage that has already been learned.

Verification: the four cases in `tests/test_plan_reschedule.py` - extending the plan (LR jumps up) must be rejected; adding `allow_reschedule: true` lets it through; shortening the plan (LR only gets smaller) is allowed; an unchanged total step count is allowed. The decision was extracted into the pure function `should_accept_reschedule`, so the test constructs two Plans directly and does not need to start a Trainer. The old behavior corresponds to the 8 `_readme` hot-edit records and the jumps in the `lr` column in `plan_ddp2.json`. This rule will also go into the example in the plan schema and into the README, so that next time it is not discovered from the 'cosine re-laid' line in the log.

---

## E19 refresh must compare the signature snapshotted at scan time

**Status**: Fixed

`retnet_hf/data.py` L185–L201

```python
    def refresh(self) -> bool:
        """重新扫描目录；分片集合或任一分片发生变化就接过，返回是否发生变化。

        2026-10-20（docs/00 E19）：旧实现只比 ``len(shards)`` 与总行数，
        一个同尺寸的分片被替换时两个值都不变，函数返回 False，
        训练继续用旧 mmap 读到已经过期的内容。

        关键细节：签名必须在**扫描时**快照下来（``self._signature``），
        不能在比较时对旧列表重新 stat——那样新旧两边读的都是文件当前状态，
        永远相等，等于没比。第一版修复就踩了这个坑，测试当场抓到。
        """
        shards, starts, total, sig = self._scan()
        if sig == self._signature:
            return False
        self.shards, self._starts, self._rows = shards, starts, total
        self._signature = sig
        return True
```

`retnet_hf/data.py` L151–L179

```python
    def _scan(self):
        """扫描目录，返回 (shards, starts, rows)。

        **跳过打不开的分片**：数据可能正在从外面传进来，文件被截断时
        ``np.load(mmap_mode=...)`` 会抛 ``ValueError: mmap length is greater than
        file size``。这里选择跳过而不是崩——下一轮 ``refresh()`` 会把它收进来。
        """
        width = self.seq_len + 1
        files = sorted(p for p in self.data_dir.glob("*.npy") if "val" not in p.name.lower())
        if self.exclude:
            files = [p for p in files if not p.name.startswith(self.exclude)]
        shards, starts, total = [], [], 0
        signature = []
        for p in files:
            try:
                arr = np.load(p, mmap_mode="r")
            except Exception:                                   # noqa: BLE001
                continue          # 正在写入 / 损坏，等下次 refresh
            if arr.ndim != 1:
                continue
            st = p.stat()
            n = arr.shape[0] // width
            if n == 0:
                continue
            shards.append((p, arr, n))
            starts.append(total)
            total += n
            signature.append((p.name, st.st_mtime_ns, st.st_size))
        return shards, starts, total, tuple(signature)
```

The old `refresh` compared only the shard count and the total row count. When a shard of the same size is replaced, both values stay unchanged, the function returns False, and training keeps reading stale content through the old mmap. The first-round fix wrote the signature as 're-stat the old list at comparison time', so both sides read the file's current state and were always equal - the test caught this mistake on the spot, which also shows that a fix that 'looks right' must be validated by a test that can fail. The correct approach is for `_scan` to return `(shards, starts, total, signature)`, with the signature taken at the moment of scanning as `(filename, mtime_ns, size)` and stored in `self._signature`; `refresh` compares only the stored value against the newly scanned one. `close()` clears the signature as well, so a signature for something that no longer exists is never used for comparison. `tests/test_regressions_extra.py` first closes the mmap, overwrites the same location in place with new content of the same size, explicitly pushes mtime into the future, and then asserts that refresh returns True and that a second call returns False.

How it is verified: `tests/test_regressions_extra.py::test_refresh_detects_same_size_replacement` closes the mmap, writes new content of the same size in place, pushes mtime into the future, and asserts that refresh returns True and that the second call returns False. The first-round fix re-stat'ed at comparison time, so both sides were always equal, and the test caught that mistake on the spot.

---

## E20 Plan mode must pass eval_batch_mult through

**Status**: Fixed

`run_task.py` L114–L178

```python
def build_cmd(cfg):
    """Translate config -> train.py argv list.

    ``plan`` 存在时改走计划模式：数据源、总步数、LR 曲线全部由计划文件决定，
    并且训练进程会**热加载**该文件（见 retnet_hf/plan.py）。此时不传
    ``--data_dir`` / ``--max_steps``，否则会和计划里的值打架。
    配置里仍可写 ``data_dir``，仅作为日志/兼容用途。
    """
    if cfg.get("plan"):
        cmd = [
            sys.executable, cfg.get("train_script", "retnet_hf/train.py"),
            "--model_dir", str(cfg["model_dir"]),
            "--output_dir", str(cfg["output_dir"]),
            "--plan", str(cfg["plan"]),
            "--batch_size", str(cfg["batch_size"]),
            "--mode", str(cfg["mode"]),
            "--chunk_size", str(cfg["chunk_size"]),
            "--precision", str(cfg["precision"]),
            "--seed", str(cfg["seed"]),
            "--eval_batches", str(cfg.get("eval_batches", 0)),
            # 2026-10-20: plan 分支此前漏了这一项，实际一直用 train.py 的默认 4。
            # 见 docs/00 E20。
            "--eval_batch_mult", str(cfg.get("eval_batch_mult", 1)),
            "--save_total_limit", str(cfg.get("save_total_limit",
                                              DEFAULT_SAVE_TOTAL_LIMIT)),
            "--weight_decay", str(cfg.get("weight_decay", 0.1)),
            "--max_grad_norm", str(cfg.get("max_grad_norm", 1.0)),
            "--dataloader_workers", str(cfg.get("dataloader_workers", 0)),
        ]
        return cmd
    return [
        sys.executable, cfg.get("train_script", "retnet_hf/train.py"),
        "--model_dir", str(cfg["model_dir"]),
        "--data_dir", str(cfg["data_dir"]),
        "--output_dir", str(cfg["output_dir"]),
        "--max_steps", str(cfg["max_steps"]),
        "--batch_size", str(cfg["batch_size"]),
        "--lr", str(cfg["lr"]),
        "--min_lr", str(cfg["min_lr"]),
        "--warmup", str(cfg["warmup"]),
        "--mode", str(cfg["mode"]),
        "--chunk_size", str(cfg["chunk_size"]),
        "--precision", str(cfg["precision"]),
        "--seed", str(cfg["seed"]),
        "--eval_batches", str(cfg.get("eval_batches", 0)),
        # NOTE(2026-10-10): 配置模板里一直有 eval_batch_mult（见文件头 schema），
        # 但以前**从没传下去**，于是 retnet_hf/train.py 里硬编码的 *4 一直生效。
        # 实测 1×4 会让第一次评测多花 128 s、峰值 reserved 冲到 7.65/8.00 GiB；
        # 1×1 则评测开销从 6.6% 降到 2.9%、reserved 回到 6.73 GiB。
        "--eval_batch_mult", str(cfg.get("eval_batch_mult", 4)),
        # NOTE(2026-10-10): save_total_limit / save_steps / eval_steps /
        # weight_decay / max_grad_norm 此前**从没拼进来过**，train.py 的硬编码
        # 默认值一直生效。模板里写 save_total_limit=4，实际跑的是 3，
        # 多出来的 checkpoint-10000 被 HF 按 mtime 轮转静默删掉了
        # （删除日志是 logger.info，transformers 默认 WARNING 不打印）。
        "--save_total_limit", str(cfg.get("save_total_limit",
                                          DEFAULT_SAVE_TOTAL_LIMIT)),
        "--save_steps", str(cfg.get("save_steps", 2500)),
        "--eval_steps", str(cfg.get("eval_steps", 2500)),
        "--weight_decay", str(cfg.get("weight_decay", 0.1)),
        "--max_grad_norm", str(cfg.get("max_grad_norm", 1.0)),
        # 无条件拼进去（不是条件 flag）：这样 check_task_config.py 能直接验证
        # 「JSON 里写了 dataloader_workers 就真的生效」。默认 0 = 行为与改动前一致。
        "--dataloader_workers", str(cfg.get("dataloader_workers", 0)),
    ]
```

The old build_cmd took a different branch when plan was present, and that branch failed to pass eval_batch_mult, save_steps, eval_steps and other parameters, so plan mode kept falling back to train.py's default eval_batch_mult=4 and evaluation GPU memory was higher than it needed to be; save_steps/eval_steps are written into args at runtime by the plan callback, but before the callback takes effect for the first time the command-line defaults still stand. This 'each branch assembles half of the parameters' pattern is the most common way a configuration system rots. The fix adds --eval_batch_mult to the plan branch, read from the config and defaulting to 1; save/eval steps remain the plan callback's responsibility (it must change both args and state, otherwise DefaultFlowCallback does not honor it). tests/test_regressions_extra.py builds a minimal plan config and asserts that build_cmd's output contains --eval_batch_mult with the value from the config, 2.

Verification: `tests/test_regressions_extra.py::test_build_cmd_plan_passes_eval_batch_mult` builds a minimal plan config and asserts that `build_cmd` output contains `--eval_batch_mult` and that it equals the configured value 2. The old plan branch did not pass it, so it always fell back to train.py's default 4, which made evaluation GPU memory correspondingly higher.

---

## E21 The recurrent single step must take the fast path

**Status**: Fixed

`retnet_hf/modeling_retnet.py` L222–L243

```python
    def _recurrent_retention(self, q, k, v, state):
        bsz, _, tgt_len, key_dim = q.shape
        value_head_dim = v.shape[-1]
        if state is None:
            state = torch.zeros(bsz, self.num_heads, key_dim, value_head_dim,
                                device=q.device, dtype=q.dtype)
        gamma = self.gamma.to(device=q.device, dtype=q.dtype)
        # docs/00 E21：解码时 tgt_len 恒为 1，走快路径，省掉 Python 循环与
        # list/stack 的物化。数学与下面的循环逐位相同。
        if tgt_len == 1:
            state = gamma.view(1, -1, 1, 1) * state + \
                k[:, :, 0, :].unsqueeze(-1) * v[:, :, 0, :].unsqueeze(-2)
            out = torch.sum(q[:, :, 0, :].unsqueeze(-1) * state, dim=-2).unsqueeze(2)
            return out, state
        outputs = []
        for t in range(tgt_len):
            state = gamma.view(1, -1, 1, 1) * state + \
                k[:, :, t, :].unsqueeze(-1) * v[:, :, t, :].unsqueeze(-2)
            out_t = torch.sum(q[:, :, t, :].unsqueeze(-1) * state, dim=-2)
            outputs.append(out_t)
        output = torch.stack(outputs, dim=2)
        return output, state
```

The recurrent implementation is a per-token Python loop: the state decays by gamma, is added to the current token's k x v product, and is then dotted with q. Training does not use it, but decoding goes through it once per token; if a long-sequence prefill mistakenly used recurrent, the number of loop iterations would be the sequence length. The fix adds a fast path for tgt_len==1: at decode time there is only one step, so it performs one state update and one dot product directly, dropping the range loop, the list append and the materialization by torch.stack. The math is bitwise identical to the loop version, so no new tolerance is needed. tests/test_regressions_extra.py hand-writes one recurrence step with a random state and compares the output and the new state against the fast path value by value, making sure that the 'optimization' did not change the semantics; if somebody breaks the fast path later (for example by missing the multiplication by gamma or by misplacing q/k/v), this test goes red.

Verification: `tests/test_regressions_extra.py::test_recurrent_single_step_matches_manual` hand-writes one recurrence step with a random state and compares the output and the new state against the fast path value by value (atol=1e-5). At decode time `tgt_len` is always 1, and the old implementation ran one Python loop plus `torch.stack` for every token.

see [04]

---

## E22 Cross-entropy must be chunked, not cast to fp32 in one monolithic block

**Status**: Fixed

`retnet_hf/modeling_retnet.py` L394–L438

```python
def _chunked_weighted_ce(logits, labels, wcodes=None, prompt_loss_weight=0.1,
                         chunk_tokens: int | None = None,
                         chunk_bytes: int = DEFAULT_CE_CHUNK_BYTES):
    """分块计算加权交叉熵，避免把整个 [B,L,V] 转成 fp32（docs/00 E22）。

    旧实现是 ``logits_f = logits.float()`` 之后一次性 reshape 求 CE。
    对本项目（B=16, L=2048, V=20000）那一步会物化 16×2048×20000×4 ≈ 2.6 GB
    的 fp32 副本，是显存峰值的主要来源；而 logits 本身还是 bf16，已经占 1.3 GB。

    分块大小**由字节预算反推**（``ce_chunk_tokens``），不是固定 token 数：
    每块只把 [chunk, V] 转 fp32，峰值约为 ``chunk_bytes`` 的量级。默认 64 MB
    在 V=20000 时是 838 token/块，于是生产形状 B=16/L=2048 会切出 40 块，
    而不是旧实现的一块 2.6 GB。显式传 ``chunk_tokens`` 仍可覆盖（测试用）。

    权重码语义与旧实现完全一致：pad=0 不计、reply/plain=1、prompt=prompt_loss_weight，
    最后除以权重和；没有 wcodes 时退化为普通均值。
    """
    V = logits.shape[-1]
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)
    n = flat_logits.shape[0]
    if chunk_tokens is None:
        chunk_tokens = ce_chunk_tokens(V, chunk_bytes)
    chunk_tokens = max(1, int(chunk_tokens))

    w = None
    if wcodes is not None:
        w = torch.zeros_like(wcodes, dtype=torch.float32)
        w = torch.where(wcodes == 1, torch.ones_like(w), w)
        w = torch.where(wcodes == 2, torch.full_like(w, float(prompt_loss_weight)), w)
        w = torch.where(wcodes == 3, torch.ones_like(w), w)
        w = w.reshape(-1)

    total = torch.zeros((), device=flat_logits.device, dtype=torch.float32)
    denom = torch.zeros((), device=flat_logits.device, dtype=torch.float32)
    for s in range(0, n, chunk_tokens):
        e = min(s + chunk_tokens, n)
        ce = F.cross_entropy(flat_logits[s:e].float(), flat_labels[s:e], reduction="none")
        if w is None:
            total = total + ce.sum()
            denom = denom + float(e - s)
        else:
            total = total + (ce * w[s:e]).sum()
            denom = denom + w[s:e].sum()
    return total / denom.clamp_min(1.0)
```

`retnet_hf/modeling_retnet.py` L36–L43

```python
def ce_chunk_tokens(vocab_size: int, chunk_bytes: int = DEFAULT_CE_CHUNK_BYTES) -> int:
    """按字节预算反推每个 CE 分块能放多少 token（至少 1）。

    单独成函数是为了让测试能直接断言“生产形状下必须切出多块”，而不是只
    验证数值等价 —— 旧测试传 ``chunk_tokens=7``，即使参数被无视也会通过。
    """
    per_token = 4 * max(1, int(vocab_size))          # fp32 logits，每 token V 个 float
    return max(1, int(chunk_bytes) // per_token)
```

`retnet_hf/modeling_retnet.py` L622–L628

```python
            # docs/00 E22：不再对整段 logits 做 .float()（B=16,L=2048,V=20000 时
            # 那是约 2.6 GB 的 fp32 副本），改为按**字节预算**分块求 CE。
            # 数值与旧实现一致：同样的权重码、同样的“除以权重和”。
            loss = _chunked_weighted_ce(
                logits, labels, wcodes, self.config.prompt_loss_weight,
                chunk_bytes=int(getattr(self.config, "ce_chunk_bytes",
                                        DEFAULT_CE_CHUNK_BYTES) or DEFAULT_CE_CHUNK_BYTES))
```

The old `forward` on the non-fused path first did `logits_f = logits.float()`, then reshape and compute cross-entropy. For this project (B=16, L=2048, V=20000) that step materializes about 2.6 GB of fp32 copy, while logits themselves (bf16) already occupy 1.3 GB, making it the dominant contributor to peak GPU memory. The first-round fix extracted loss computation into `_chunked_weighted_ce`, but hard-coded the chunk to 262144 tokens: that number is unrelated to vocabulary size and is larger than the total production token count (32768), so the loop ran exactly once and the full 2.6 GB copy remained untouched - the chunking parameter was decoration. The chunk is now derived backwards from a byte budget: `ce_chunk_tokens(V, 64MB)` = 64MB/(4V), which gives 838 tokens per chunk at V=20000 and cuts the production shape into 40 chunks; each chunk casts only `[chunk,V]` to fp32. Weight-code semantics are exactly identical to the old implementation (pad excluded, reply/plain=1, prompt=prompt_loss_weight, finally divided by the weight sum), and it degrades to a plain mean when `wcodes` is absent. Validation has two layers: numerically, `test_chunked_ce_matches_naive_weighted` requires `allclose` (atol=1e-5) against a naive whole-block fp32 implementation; structurally, `test_chunked_ce_splits_at_production_token_count` records the slices `F.cross_entropy` actually receives on a shape with n=32768 and asserts that the number of chunks is >= 2, that each chunk stays within budget, and that the total token count covers the input completely. The old test passed `chunk_tokens=7`, so it passed even if the parameter was ignored entirely - that is the 'tests the numbers but not the structure' hole.

How it is verified: numerically, `test_chunked_ce_matches_naive_weighted` and `test_chunked_ce_without_weights_is_plain_mean` require `allclose` (atol=1e-5) against a naive whole-block fp32 implementation; structurally, `test_ce_chunk_size_is_derived_from_a_byte_budget` asserts 838 tokens per chunk at V=20000 and smaller chunks as V grows, and `test_chunked_ce_splits_at_production_token_count` records the slices `F.cross_entropy` actually receives at n=32768, asserting that the number of chunks is >= 2, that each chunk stays within budget, and that coverage is complete. With the numerical layer alone, hard-coding the chunk to 262144 (that is, only one chunk at the production shape) would also pass.

---

## E23 Every fix needs a test that goes red when the fix is reverted

**Status**: Fixed

`tests/test_growth_trainable.py` L98–L110

```python
def test_growth_guard_fails_when_reverted_to_zeros():
    """把新行随机撤掉、把中间层 gate 改成 0（旧行为），守卫必须拒绝落盘。"""
    G = _load_grow_module()
    _patch_dims(G)
    G.NEW_ROW_INIT_STD = 0.0
    G.BLOCK_GATE = 0.0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        src = _tiny_source(tmp / "src")
        dst = tmp / "dst"
        rc = _run(G, src, dst)
        assert rc == 2, f"旧行为下必须拒绝保存（rc=2），实际 rc={rc}"
        assert not (dst / "model.safetensors").exists(), "可训性校验失败却仍然落盘了"
```

This one is not a code error but a process gap: the repository had 11 tests, and not one of them checked 'does the parameter have a gradient', so 338M dead parameters passed every check along the way. The fix adds 6 test files, each corresponding to an id in 00, and each designed so that 'reverting the fix to the old behavior makes it fail': test_growth_trainable uses monkeypatch to randomly change the new rows to 0 and asserts that the growth script refuses to write the checkpoint; test_no_decay_1d tests both the enabled and the disabled direction; test_val_disjoint tests three path spellings; test_eos_present tests no boundary / boundary / explicit override; test_parallel_chunkwise_equiv tests that the two paths agree numerically; test_live_params requires the zero-parameter counting to be element-exact. This section quotes the most critical one of them: it proves that the E04 guard can really fail, rather than being decoration that is always true.

Verification: `cd code && python -m pytest tests -q` -> **82 passed, 4 skipped**, and the two script-style tests `python tests/test_plan_gc.py` / `test_plan_pipeline.py` also all pass when run individually. The 4 skips are explicit skips in test_build_scripts when `data_cache/chat_pack_v13/bpe_tokenizer.json` is missing, not false passes. More important is the reverse verification: `test_growth_guard_fails_when_reverted_to_zeros` must return rc=2 when the initialization is reverted to all zeros; `test_grad_coverage` must halt when a parameter is frozen; `test_plan_eos` must go red when the require_eos passthrough is removed. The counts are measured on this snapshot and no longer quote historical numbers from a remote environment.

see [10][11]

---

## N01 reinit_gamma must be per head, not per layer

**Status**: Fixed

`scripts/reinit_gamma.py` L52–L69

```python
def build_gamma(num_heads: int, formula: str = "logspace",
                tau_min: float = 128.0, tau_max: float = 8192.0) -> torch.Tensor:
    """返回 [num_heads] 的 gamma，按**头索引**排列。

    formula="logspace"：半衰期在 [tau_min, tau_max] 对数均匀；
    formula="orig"：论文的 1-2^(-5-h)，h 是头索引（半衰期 2^(5+h)）。
    """
    if num_heads <= 0:
        raise ValueError("num_heads 必须为正")
    if formula == "orig":
        return 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads, dtype=torch.float64))
    if formula != "logspace":
        raise ValueError(f"未知 formula: {formula}")
    if not (tau_min > 0 and tau_max > 0 and tau_max >= tau_min):
        raise ValueError(f"tau 范围非法: {tau_min}..{tau_max}")
    frac = torch.linspace(0.0, 1.0, steps=num_heads, dtype=torch.float64)
    tau = tau_min * (tau_max / tau_min) ** frac
    return torch.pow(0.5, 1.0 / tau)
```

`scripts/reinit_gamma.py` L153–L177

```python
    keys = sorted(k for k in sd if k.endswith(SUFFIX))
    if not keys:
        raise SystemExit(f"没找到 *{SUFFIX}，请确认权重文件名")

    heads = int(sd[keys[0]].numel())
    for k in keys:
        if int(sd[k].numel()) != heads:
            raise SystemExit(f"{k} 的头数 {sd[k].numel()} 与 {keys[0]} 的 {heads} 不一致")

    old = torch.stack([sd[k].detach().double().flatten() for k in keys])  # [L, H]
    old_h = half_lives(old.flatten())
    print(f"  找到 {len(keys)} 层 x {heads} 头 = {len(keys) * heads} 个 gamma")
    print(f"  旧值: {float(old.min()):.6f} .. {float(old.max()):.6f}")
    print(f"  旧半衰期: p25={float(torch.quantile(old_h, 0.25)):.2f} "
          f"p50={float(torch.quantile(old_h, 0.50)):.2f} "
          f"p75={float(torch.quantile(old_h, 0.75)):.2f} token")

    gamma = build_gamma(heads, args.formula, args.tau_min, args.tau_max)
    tau = half_lives(gamma)
    print(f"  新 gamma: {float(gamma.min()):.8f} .. {float(gamma.max()):.8f}")
    print(f"  新半衰期: {float(tau.min()):.1f} .. {float(tau.max()):.1f} token，"
          f"不同取值 {len(set(round(float(v), 8) for v in gamma))} 个")
    # 同一层内不允许有重复值：重复就说明又退化成"按层设置"了
    if len(set(round(float(v), 8) for v in gamma)) != heads:
        raise SystemExit("新 gamma 在同一层内出现重复值——按头设置失败，拒绝写入")
```

The old implementation took the 20 `blocks.*.msr.gamma` tensors, treated the number of tensors as the number of heads, looped h=0..19 generating 1-2^(-5-h), and used `torch.full_like` to fill all 22 heads of each layer. The result was not 'set per head' but 'set per layer', and all 22 heads within a layer were identical, collapsing multi-scale decay into a single scale; the 'old values 0.399..0.586' it printed were also the per-layer mean over 22 heads, flattening the true 0.106..0.999 range - even the diagnostics were wrong. The fix extracts the core into `build_gamma(num_heads, formula, tau_min, tau_max)`: gamma is generated by head index with half-lives uniformly distributed in log space, default 128..8192 tokens (lower bound aligned with chunk=128, upper bound aligned with the 8K window). Before writing, it asserts that values within the same layer differ from one another - a repeat means it has degenerated to per-layer setting again, and the write is refused. Local dry-run measurements: 440 gammas, 22 distinct values per layer, old half-life p50=1.22, new range 128-8192.

How it is verified: three cases in `tests/test_gamma_reinit.py` - the values of the 22 heads must be distinct (a repeat means degeneration to per-layer), half-lives must be monotonic and lie within 128-8192, and an illegal tau range must raise `ValueError`; then `reinit_gamma.py --dry-run` on the real checkpoint prints '20 layers x 22 heads = 440 gammas' together with the old and new distributions. The old implementation printed the per-layer mean 0.399..0.586, erasing the true value range 0.106..0.999.

see [03]

---

## N02 The gamma 'median' must not mix in dead heads

**Status**: Doc corrected (see N01)

`scripts/reinit_gamma.py` L5–L14

```python
旧实现把 20 个 `blocks.*.msr.gamma` 张量当成 20 个"头"来遍历：

    keys = [k for k in sd if k.endswith(".msr.gamma")]   # 20 个，每层一个 [22]
    for h in range(len(keys)):                            # h = 0..19，是层号
        g = 1 - 2 ** (-5 - h)
        sd[key] = torch.full_like(sd[key], g)             # 22 个头被灌成同一个值

结果不是"按头设置"，而是"按层设置"，并且每层 22 个头完全相同——多尺度设计
被抹平成一个尺度。它打印的"旧值 0.399..0.586"也是每层 22 头的**均值**，
把真实的 0.106..0.999 抹掉了，所以连诊断都是错的。
```

The 'gamma median 0.5656' written in 00 was computed by mixing all 22 heads together, while 10 of those heads are dead (value fixed at 0.213622, pure wd decay, zero gradient). Using that number to describe 'how long the model's memory is' is biased by the dead heads; counted over live heads only, the median for ep3 is 0.848 (half-life 4.21 token) and for v21 it is 0.693 (1.89 token). This is not an arithmetic error but a definitional one: mixing averages two entirely different kinds of object (trained and never-trained) into a single number. The fix does not change numerical logic; it makes the `reinit_gamma` documentation explicit: judging 'how much memory was eaten by wd' must look only at live heads, while judging 'how fragmented the overall distribution is' looks at all 440. From now on, any report quoting a gamma median must give both definitions, all heads and live heads, otherwise the conclusion is not comparable.

How it is verified: compute the gamma median separately over all 440 and over live heads only, and both numbers must appear in the report. Measured, the live-head median for ep3 is 0.848 (half-life 4.21 token) and for v21 it is 0.693 (1.89 token), while dead heads are fixed at 0.213622 - the 0.5656 obtained by averaging the two cannot be used to describe memory length. The report template already lists all heads and live heads as mandatory fields; missing one counts as incomplete data.

see [03]

---

## N03 Layers 18/19 are d1024, not d1408

**Status**: Doc corrected (no code)

The recovery plan said 'layers 18/19 are fully wide and live (d1408)', and inferred from that that 'shrinking the kernel is infeasible'. The measured result of a per-tensor count is: q_proj has 1024 live input columns and 1024 live output rows, v_proj has 1024 inputs, out_proj has 2048 inputs; the absmax of input columns 1024-1407 is exactly 0. That is to say, layers 18/19 are d1024, not d1408; the non-zero q[768:] seen in the report comes from the 768-1023 stretch. This changes only the basis, not the conclusion: the model is still a mixed width of 18 x d768 + 2 x d1024, which a single RetNetHFConfig likewise cannot express, so 'shrinking the kernel is infeasible' still holds. But it affects the first cross-check on the machine: if you reconcile against 'live parameters should be above 166M', the numbers will not match, and you may misdiagnose it as 'the slice interpretation is inverted'. From now on, layer width must be quoted from per-tensor counts and must not be inferred from a non-zero q[768:].

Verification: per-tensor live rows and live columns over `checkpoints/retnet_500m_v21_export`. Measured: L18/L19 q_proj has 1024 live input columns and 1024 live output rows, v_proj 1024 inputs, out_proj 2048 inputs; the absmax of input columns 1024-1407 is exactly 0, so the layer width is 1024 and not 1408.

---

## N04 Evaluation must go through the same path that training used

**Status**: Fixed

`scripts/eval_holdout.py` L102–L112

```python
    # 2026-10-20（docs/00 N04/E16）：评测必须走**训练用的那条路径**。
    # 旧代码强制 mode="parallel"，理由是"parallel 与 chunkwise 数学等价"——
    # 但 E16 证明 parallel 当时多做两次归一化，两者并不等价。
    # 那条注释里还提到 chunkwise 参考实现在 bf16 下有 dtype 冲突；
    # 这里显式切到 chunkwise 并把 chunk_size 固定成训练值，
    # 若真遇到 dtype 冲突，应当修内核而不是换一条不等价的路径来测。
    model.config.mode = "chunkwise"
    model.config.chunk_size = int(args.chunk_size)
    print(f"[load] {args.model} -> {dev}/{dtype}  "
          f"mode=chunkwise chunk={args.chunk_size}（{time.time() - t0:.1f}s）",
          flush=True)
```

The old `eval_holdout` forced `model.config.mode="parallel"`, with the reason written in a comment: 'parallel and chunkwise are mathematically equivalent over the whole forward pass'. But E16 showed that parallel performed two extra normalizations at the time, so the two were not equivalent. Consequently every held-out number from the previous round (3.6287 / 2.8471 / 3.5023) measured a function different from the one that was trained; using it as the criterion was equivalent to grading yourself with someone else's score. The fix switches evaluation to chunkwise and aligns `chunk_size` with the training value (new `--chunk-size`, default 128). The comment mentions a dtype conflict for chunkwise under bf16 - if that really exists, the kernel should be fixed rather than measuring along a non-equivalent path. After E16 removed those two normalizations, the two paths should differ only by floating-point noise, and `tests/test_parallel_chunkwise_equiv.py` will keep guarding that.

How it is verified: `tests/test_train_config_guards.py::test_eval_uses_chunkwise` asserts that `model.config.mode = "chunkwise"` exists in the source, that `"parallel"` does not, and that `--chunk-size` is passed; `test_eval_has_both_probes` asserts that both probe switches exist; `tests/test_eval_probes.py` runs once end to end with a tiny model, checking the 8 buckets of by_position, the delta_ctx and token counts of ctx_ablation, and the early refusal when ctx is not a multiple of chunk. Under the old behavior parallel performed two extra normalizations, every held-out number was misaligned with the training function, and the criterion was only meaningful once converted to a delta.

see [01][04]

---

## References

> Numbers match the `Papers/` filename prefixes; only papers actually cited here are listed.

| No. | Original title | Local file | Link |
|---|---|---|---|
| [01] | Retentive Network: A Successor to Transformer for Large Language Models | `Papers/01_Retentive-Network_2307.08621.pdf` | <https://arxiv.org/abs/2307.08621> |
| [02] | A Length-Extrapolatable Transformer | `Papers/02_Length-Extrapolatable-Transformer_2212.10554.pdf` | <https://arxiv.org/abs/2212.10554> |
| [03] | Gated Linear Attention Transformers with Hardware-Efficient Training | `Papers/03_Gated-Linear-Attention_2312.06635.pdf` | <https://arxiv.org/abs/2312.06635> |
| [04] | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention | `Papers/04_Transformers-are-RNNs_2006.16236.pdf` | <https://arxiv.org/abs/2006.16236> |
| [05] | Group Normalization | `Papers/05_Group-Normalization_1803.08494.pdf` | <https://arxiv.org/abs/1803.08494> |
| [06] | GLU Variants Improve Transformer | `Papers/06_GLU-Variants_2002.05202.pdf` | <https://arxiv.org/abs/2002.05202> |
| [07] | Net2Net: Accelerating Learning via Knowledge Transfer | `Papers/07_Net2Net_1511.05641.pdf` | <https://arxiv.org/abs/1511.05641> |
| [08] | Network Morphism | `Papers/08_Network-Morphism_1603.01670.pdf` | <https://arxiv.org/abs/1603.01670> |
| [09] | bert2BERT: Towards Reusable Pretrained Language Models | `Papers/09_bert2BERT_2110.07143.pdf` | <https://arxiv.org/abs/2110.07143> |
| [10] | Efficient Training of BERT by Progressively Stacking | `Papers/10_StackBERT_PMLR-v97-gong19a.pdf` | <https://proceedings.mlr.press/v97/gong19a.html> |
| [11] | Progressively Stacking 2.0: A Multi-stage Layerwise Training Method for BERT Training Speedup | `Papers/11_Progressively-Stacking-2.0_2011.13635.pdf` | <https://arxiv.org/abs/2011.13635> |
| [12] | Learning to Grow Pretrained Models for Efficient Transformer Training | `Papers/12_LiGO_2303.00980.pdf` | <https://arxiv.org/abs/2303.00980> |
| [13] | ReZero is All You Need: Fast Convergence at Large Depth | `Papers/13_ReZero_2003.04887.pdf` | <https://arxiv.org/abs/2003.04887> |
| [14] | Fixup Initialization: Residual Learning Without Normalization | `Papers/14_Fixup_1901.09321.pdf` | <https://arxiv.org/abs/1901.09321> |
| [15] | Training Compute-Optimal Large Language Models | `Papers/15_Chinchilla_2203.15556.pdf` | <https://arxiv.org/abs/2203.15556> |
| [16] | The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale | `Papers/16_FineWeb_2406.17557.pdf` | <https://arxiv.org/abs/2406.17557> |
| [17] | Skywork: A More Open Bilingual Foundation Model | `Papers/17_Skywork_2310.19341.pdf` | <https://arxiv.org/abs/2310.19341> |
