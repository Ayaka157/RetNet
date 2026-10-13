"""Build 202610011026_fix.ipynb from 202610011026.ipynb.

Only cell 4 (the model + training-loop cell) and cell 0's markdown are edited;
every other cell, cell id, output and notebook byte is preserved so the file
stays a drop-in replacement for the original Kaggle notebook.

Each edit asserts that it matched exactly once -- a silent no-op here would
ship an unfixed notebook.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"E:\Workspaces\RetNet\Kaggle")
SRC = ROOT / "202610011026.ipynb"
DST = ROOT / "202610011026_fix.ipynb"


def edit(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"[FAIL] {label}: expected 1 match, found {n}")
    print(f"[ok] {label}")
    return text.replace(old, new)


def main() -> int:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # ---- cell 0: markdown, document the actual (chunk-loop) design ----
    md_old = (
        "- `chunkwise` 使用 `torch_xla.experimental.scan`，把原来的 Python chunk loop 表达为 XLA scan；"
        "训练后的 `state_dict` 与原模型完全同构。"
    )
    md_new = (
        "- `chunkwise` 使用普通 Python chunk loop（**不用** `torch_xla.experimental.scan`：scan 的 "
        "AOT/HLO-While 路径在本栈上会把某个 fn_computation 形参标成 f32，与 bf16 实参冲突而崩）；"
        "训练后的 `state_dict` 与原模型完全同构。"
    )

    # ---- cell 4: the training script ----
    cell4 = cells[4]
    assert cell4["cell_type"] == "code", cell4["cell_type"]
    src = "".join(cell4["source"])

    # F1: `_latest_checkpoint` calls re.fullmatch but `re` was never imported,
    # so resume (any existing checkpoint-N dir) died with NameError.
    src = edit(
        src,
        "import json\nimport os\nfrom typing import List, Optional, Tuple\n",
        "import json\nimport os\nimport re\nfrom typing import List, Optional, Tuple\n",
        "F1 missing `import re`",
    )

    # F2: the fp32 decay gate must survive model.to(bfloat16).  The old code
    # only patched nn.Parameter, leaving the use_gla=False persistent buffer
    # path (also fp32 gamma) collapsed to bf16 -> 12/16 heads decay == 1.0.
    #
    # F2b/F2c (found by the CPU verifier, proven against the real checkpoint):
    # restoring the DTYPE after the cast is not enough -- `model.to(bfloat16)`
    # has already ROUNDED the values, and `.to(torch.float32)` cannot undo that.
    # Measured on retnet_272m_rp/model.safetensors: 0/20 gates bitwise recovered,
    # 4 heads merge onto one value (0.8862733/…768/…786/…795 -> 0.88671875) and
    # the cross-chunk decay gamma**64 carries up to 1.37e-01 relative error.
    # So snapshot the fp32 master weights BEFORE the cast and restore the values.
    # F11: the shipped checkpoint's learned decay gates are NOT all < 1 --
    # blocks.12.msr.gamma[10] = 1.0117193.  chunkwise multiplies the state by
    # chunk_decay = gamma**64 per chunk, so 128 chunks give gamma**8192 =
    # 2.83e41, far beyond bf16 max 3.39e38.  Measured with the real gamma
    # vector: seq=1024 clean, seq=4096 max 1.2e20, seq=8192 -> 48896 NaN and
    # 1527 Inf in the retention output.  One explosive head poisons the whole
    # model through GroupNorm, so loss=nan from step 1 and every checkpoint
    # would be garbage.  Clamp the gates to a representable ceiling at load and
    # after every optimizer step (training can push them over again).
    src = edit(
        src,
        "        # optimizer_step = gradient all-reduce + optimizer.step() + mark_step()\n"
        "        xm.optimizer_step(optimizer, barrier=False)\n"
        "        optimizer.zero_grad(set_to_none=True)\n"
        "        scheduler.step()\n",
        "        # optimizer_step = gradient all-reduce + optimizer.step() + mark_step()\n"
        "        xm.optimizer_step(optimizer, barrier=False)\n"
        "        optimizer.zero_grad(set_to_none=True)\n"
        "        scheduler.step()\n"
        "\n"
        "        # Keep the decay gates in a representable range. At seq_len=8192 the\n"
        "        # state is multiplied by gamma**64 per chunk over 128 chunks, i.e. by\n"
        "        # gamma**8192; a single gate above 1 (the shipped checkpoint has\n"
        "        # blocks.12.msr.gamma[10]=1.0117) overflows bf16 and turns the whole\n"
        "        # batch into inf/nan, which GroupNorm then spreads across the model.\n"
        "        if GAMMA_MAX is not None and _gate_list:\n"
        "            with torch.no_grad():\n"
        "                for _g in _gate_list:\n"
        "                    _over = _g > GAMMA_MAX\n"
        "                    if bool(_over.any().item()):\n"
        "                        clamped_gates += int(_over.sum().item())\n"
        "                        _g.copy_(torch.clamp(_g, max=GAMMA_MAX))\n",
        "F11b clamp gates after every optimizer step",
    )

    # F11a: the gate ceiling is defined in cell 1 and must exist here.
    src = edit(
        src,
        "from torch.utils.data import DataLoader, DistributedSampler\n",
        "from torch.utils.data import DataLoader, DistributedSampler\n"
        "\n"
        "# Decay-gate ceiling, defined in cell 1 (GAMMA_MAX). The learned gates are\n"
        "# per-head decays: the chunkwise state grows like gamma**seq_len, so any\n"
        "# gate above 1.0 diverges at long context (a gate of 1.0117 reaches\n"
        "# 2.8e41 over 8192 tokens, past bf16's 3.4e38 ceiling). Clamping to just\n"
        "# under 1 keeps the recurrence representable; the gates stay learnable.\n"
        "assert \"GAMMA_MAX\" in dir(), \"cell 1 must define GAMMA_MAX\"\n",
        "F11a GAMMA_MAX is supplied by cell 1",
    )

    # F11c: collect the gates once, before the loop
    src = edit(
        src,
        "    optimizer.zero_grad(set_to_none=True)\n"
        "    t0 = time.perf_counter()\n",
        "    # Decay gates the loop must keep finite (see GAMMA_MAX).\n"
        "    _gate_list = [\n"
        "        _v for _k, _v in model.named_parameters()\n"
        "        if _k.endswith(\"msr.gamma\") and _v.dtype == torch.float32\n"
        "    ]\n"
        "    clamped_gates = 0\n"
        "\n"
        "    # The gates are clamped once here, before the first step, so a run that\n"
        "    # resumes an already-diverged checkpoint is repaired as well.\n"
        "    if GAMMA_MAX is not None:\n"
        "        with torch.no_grad():\n"
        "            for _g in _gate_list:\n"
        "                clamped_gates += int((_g > GAMMA_MAX).sum().item())\n"
        "                _g.copy_(torch.clamp(_g, max=GAMMA_MAX))\n"
        "        if xm.is_master_ordinal():\n"
        "            _g0 = model.blocks[0].msr.gamma\n"
        "            print(\n"
        "                f\"[gates] GAMMA_MAX={GAMMA_MAX} clamped_at_load={clamped_gates} \"\n"
        "                f\"dtype={_g0.dtype} min={_g0.min().item():.7f} \"\n"
        "                f\"max={_g0.max().item():.7f}\",\n"
        "                flush=True,\n"
        "            )\n"
        "\n"
        "    optimizer.zero_grad(set_to_none=True)\n"
        "    t0 = time.perf_counter()\n",
        "F11c clamp at load + collect gate list once",
    )

    # report clamped gates in the periodic log
    src = edit(
        src,
        "                    f\"elapsed={elapsed/60:.1f}m\",\n"
        "                    flush=True,\n"
        "                )\n",
        "                    f\"elapsed={elapsed/60:.1f}m \"\n"
        "                    f\"gate_clamps={clamped_gates}\",\n"
        "                    flush=True,\n"
        "                )\n",
        "F11d log clamped-gate count",
    )

    # F2b/F2c/F2d: the fp32 master tensors must survive `model.to(bfloat16)`.
    # The gate restore after the cast cannot recover values that were already
    # rounded (0/20 gates bitwise recovered, 4 heads merged, gamma**64 off by up
    # to 1.37e-01).  The same is true for `inv_freq`, which is a NON-persistent
    # buffer: `.to(dtype)` casts non-persistent buffers too (verified in
    # isolation), and since it never enters the state_dict nothing restores it,
    # so the rotary frequencies drift by up to 2.2e-3 relative / 9.9 rad at 8192.
    # Snapshot every fp32 buffer that must stay fp32, restore after the cast.
    src2 = edit(
        src,
        "    model.config.chunk_size = CFG.chunk_size\n"
        "    model.to(device=device, dtype=torch.bfloat16)\n"
        "    # Manual mixed precision: restore the decay gates to fp32 master weights\n"
        "    # (model.to(bfloat16) casts them too). bf16 resolves only ~2**-8 near 1.0,\n"
        "    # which collapses 12/16 heads' decay to exactly 1.0 and quantizes away\n"
        "    # gate learning -- this restore is what actually enables the GLA gate.\n"
        "    for _block in model.blocks:\n"
        "        _gate = _block.msr.gamma\n"
        "        if isinstance(_gate, nn.Parameter):\n"
        "            _gate.data = _gate.data.to(torch.float32)\n",
        "    model.config.chunk_size = CFG.chunk_size\n"
        "    # Manual mixed precision. Two things must not be cast to bf16:\n"
        "    #  * the decay gates gamma -- they sit within ~2**-8 of 1.0 and bf16\n"
        "    #    cannot resolve that region;\n"
        "    #  * the rotary `inv_freq` buffer -- a bf16 frequency loses ~0.2%,\n"
        "    #    which accumulates into radians of phase error over 8192 steps.\n"
        "    # `model.to(dtype=...)` casts every float buffer, including the\n"
        "    # non-persistent `inv_freq` that never enters the state_dict, so there\n"
        "    # is nothing to restore it from later.  Snapshot the exact fp32 values\n"
        "    # BEFORE the cast, then put them back afterwards -- casting back up\n"
        "    # cannot undo rounding (measured: 0/20 gates bitwise recovered).\n"
        "    preserved_fp32 = {\n"
        "        _name: _p.detach().clone()\n"
        "        for _name, _p in model.named_parameters()\n"
        "        if _name.endswith(\"msr.gamma\") and _p.dtype == torch.float32\n"
        "    }\n"
        "    for _name, _buf in model.named_buffers():\n"
        "        if _name.endswith(\"msr.inv_freq\") and _buf.dtype == torch.float32:\n"
        "            preserved_fp32[_name] = _buf.detach().clone()\n"
        "    model.to(device=device, dtype=torch.bfloat16)\n"
        "    if preserved_fp32:\n"
        "        _tensors = dict(model.named_parameters())\n"
        "        _tensors.update(dict(model.named_buffers()))\n"
        "        for _key, _master in preserved_fp32.items():\n"
        "            _dst = _tensors[_key]\n"
        "            # Rebind .data to a fresh fp32 tensor on the right device.\n"
        "            # copy_ into a bf16 destination only round-trips through bf16,\n"
        "            # so it silently keeps the damaged values AND the bf16 dtype.\n"
        "            _dst.data = _master.to(device=_dst.device, dtype=torch.float32)\n"
        "    if xm.is_master_ordinal():\n"
        "        _g0 = model.blocks[0].msr.gamma\n"
        "        print(f\"[cast] preserved {len(preserved_fp32)} fp32 tensors; \"\n"
        "              f\"gamma dtype={_g0.dtype} inv_freq dtype=\"\n"
        "              f\"{model.blocks[0].msr.inv_freq.dtype}\", flush=True)\n",
        "F2b/F2c/F2d preserve fp32 gates and inv_freq across the cast",
    )
    src = src2

    # F3: rotary positions/frequencies in fp32.  torch.set_default_dtype(bfloat16)
    # makes dtype-less arange produce bf16, and bf16 spacing is 1.0 from 512 up,
    # so positions past ~256 are quantized -> sin/cos phase garbage over an 8192
    # context.  Same bug class as F4/F5.
    src = edit(
        src,
        "        self.register_buffer(\"inv_freq\", inv_freq.repeat_interleave(2).to(torch.bfloat16), persistent=False)\n",
        "        # Keep inv_freq in fp32: bf16 cannot resolve these frequencies near\n"
        "        # the low end (1e-4), and the rotary path below computes in fp32\n"
        "        # anyway. Non-persistent, so it stays out of the state_dict.\n"
        "        self.register_buffer(\"inv_freq\", inv_freq.repeat_interleave(2), persistent=False)\n",
        "F3a inv_freq stays fp32",
    )

    src = edit(
        src,
        "    def _rotations(self, length, offset, device, dtype):\n"
        "        # Keep the whole rotary-frequency path in the surrounding compute dtype.\n"
        "        position = torch.arange(offset, offset + length, device=device, dtype=dtype)\n"
        "        inv_freq = self.inv_freq.to(device=device, dtype=dtype)\n"
        "        freqs = torch.outer(position, inv_freq)\n"
        "        return torch.sin(freqs), torch.cos(freqs)\n",
        "    def _rotations(self, length, offset, device, dtype):\n"
        "        # Compute the rotary phase in fp32 and cast once at the end.\n"
        "        # This function is called with dtype=q.dtype == bfloat16, and bf16\n"
        "        # cannot represent integers above 256 exactly (spacing is 1.0 from\n"
        "        # 512 up, 31.0 by the end of an 8192 context): passing that dtype\n"
        "        # straight into arange quantizes the positions and destroys the\n"
        "        # phase -- measured max |sin - sin(fp32)| was 2.0 before this fix.\n"
        "        position = torch.arange(offset, offset + length, device=device, dtype=torch.float32)\n"
        "        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)\n"
        "        freqs = torch.outer(position, inv_freq)\n"
        "        return torch.sin(freqs).to(dtype), torch.cos(freqs).to(dtype)\n",
        "F3b _rotations fp32 positions/freqs",
    )

    # F4/F5 (REFUTED -- do not apply): I first claimed the dtype-less
    # `torch.arange(length, device=device)` in _decay_mask and
    # _parallel_final_state inherited the bf16 default dtype.  It does not:
    # torch.set_default_dtype only affects FLOAT factories, so bare arange is
    # int64 (verified: arange -> int64, zeros -> bfloat16, tensor(1.0) ->
    # bfloat16).  Those indexes were always exact; the conversion to fp32 is a
    # harmless no-op.  Recorded so it is not "re-fixed" again.

    # F6: `evaluate` builds its NaN sentinel with no dtype, so under cell 4's
    # `torch.set_default_dtype(bfloat16)` it becomes bf16 while total_loss is
    # fp32.  nan is nan, so this is a type-consistency fix, not a bug fix -- but
    # the surrounding path is now uniformly fp32, and the empty-loader case is
    # made loud instead of silently returning nan into best_eval_loss.
    src = edit(
        src,
        "    # DistributedSampler(drop_last=True) gives every worker the same number\n"
        "    # of batches, so a sum/mean across replicas is deterministic.\n"
        "    if batches == 0:\n"
        "        mean_loss = torch.tensor(float(\"nan\"), device=device)\n"
        "    else:\n"
        "        mean_loss = total_loss / batches\n",
        "    # DistributedSampler(drop_last=True) gives every worker the same number\n"
        "    # of batches, so a sum/mean across replicas is deterministic.\n"
        "    if batches == 0:\n"
        "        # Never let this reach best_eval_loss comparison as a silent nan.\n"
        "        raise RuntimeError(\n"
        "            \"validation loader produced 0 batches: DistributedSampler with \"\n"
        "            \"drop_last=True dropped everything (dataset smaller than \"\n"
        "            f\"world_size * per_device_batch = {CFG.per_device_batch})\"\n"
        "        )\n"
        "    mean_loss = total_loss / batches\n",
        "F6 evaluate empty-loader guard (replaces bf16 nan sentinel)",
    )

    # F7 (REFUTED -- do not apply): `world_size` looked like it was bound only
    # inside `if xm.is_master_ordinal():`, but line 764 of cell 4 binds it at
    # FUNCTION scope (indent 4) before that branch, so every rank sees it.  The
    # empirical reproduction at a simulated rank 1 of 8 confirms it.  Kept here
    # as a recorded refutation so it is not "re-fixed" again later.
    #
    # F8: xm.mark_step() is deprecated on this stack; the on-TPU run emitted a
    # DeprecationWarning per call.  Keep behaviour, drop the noise.
    src = edit(
        src,
        "    mean_loss = xm.all_reduce(xm.REDUCE_SUM, mean_loss) / xr.world_size()\n    xm.mark_step()\n",
        "    mean_loss = xm.all_reduce(xm.REDUCE_SUM, mean_loss) / xr.world_size()\n"
        "    torch_xla.sync()\n",
        "F8a evaluate: xm.mark_step -> torch_xla.sync",
    )
    src = edit(
        src,
        "            ) / world_size\n            xm.mark_step()\n",
        "            ) / world_size\n            torch_xla.sync()\n",
        "F8b train loop: xm.mark_step -> torch_xla.sync",
    )
    src = edit(
        src,
        "    # Final rank-0 HF-format export. Convert tensors to CPU before asking\n"
        "    # Transformers to write safetensors, avoiding XLA-device serialization.\n"
        "    xm.mark_step()\n",
        "    # Final rank-0 HF-format export. Convert tensors to CPU before asking\n"
        "    # Transformers to write safetensors, avoiding XLA-device serialization.\n"
        "    torch_xla.sync()\n",
        "F8c finalize: xm.mark_step -> torch_xla.sync",
    )

    # F9: make the XLA lazy-device handle part of the worker's own imports.
    # Cell 4 does `import torch_xla` at module level, but train_worker is the
    # one place that needs the lazy device, and it is the function whose
    # execution environment (spawned worker) is the fragile part.
    src = edit(
        src,
        "def train_worker():\n"
        "    # Import/access XLA only inside the spawned worker. This is a key PJRT/XLA\n"
        "    # multiprocessing invariant.\n"
        "    import torch_xla.core.xla_model as xm\n"
        "    import torch_xla.runtime as xr\n"
        "    import torch_xla.distributed.parallel_loader as pl\n",
        "def train_worker():\n"
        "    # Import/access XLA only inside the spawned worker. This is a key PJRT/XLA\n"
        "    # multiprocessing invariant.\n"
        "    global torch_xla\n"
        "    import torch_xla\n"
        "    import torch_xla.core.xla_model as xm\n"
        "    import torch_xla.runtime as xr\n"
        "    import torch_xla.distributed.parallel_loader as pl\n",
        "F9 train_worker imports torch_xla itself",
    )

    # F10: optimizer parameter grouping lived only inside train_worker, so any
    # other caller (resume/eval tooling) built a 1-group AdamW and then
    # _load_checkpoint died with "loaded state dict has a different number of
    # parameter groups".  One shared builder = one canonical grouping.
    src = edit(
        src,
        "    # No weight decay on 1-D tensors (norm scales, biases, decay gates):\n"
        "    # decaying the fp32 gate gamma toward 0 would kill long-range memory.\n"
        "    _wd = [p for p in model.parameters() if p.ndim >= 2]\n"
        "    _nowd = [p for p in model.parameters() if p.ndim < 2]\n"
        "    optimizer = AdamW(\n"
        "        [\n"
        "            {\"params\": _wd, \"weight_decay\": CFG.weight_decay},\n"
        "            {\"params\": _nowd, \"weight_decay\": 0.0},\n"
        "        ],\n"
        "        lr=CFG.lr,\n"
        "        betas=(0.9, 0.95),\n"
        "        eps=1e-8,\n"
        "    )\n"
        "    scheduler = CosineWithMinLRScheduler(\n"
        "        optimizer,\n"
        "        warmup_steps=CFG.warmup_steps,\n"
        "        total_steps=CFG.max_steps,\n"
        "        min_lr=CFG.min_lr,\n"
        "    )\n",
        "    optimizer = build_optimizer(model)\n"
        "    scheduler = CosineWithMinLRScheduler(\n"
        "        optimizer,\n"
        "        warmup_steps=CFG.warmup_steps,\n"
        "        total_steps=CFG.max_steps,\n"
        "        min_lr=CFG.min_lr,\n"
        "    )\n",
        "F10a train_worker uses build_optimizer",
    )

    src = edit(
        src,
        "class CosineWithMinLRScheduler(torch.optim.lr_scheduler.LambdaLR):\n",
        "def build_optimizer(model) -> AdamW:\n"
        "    \"\"\"Canonical AdamW grouping. Must be identical on every code path\n"
        "    that later calls _load_checkpoint: optimizer.load_state_dict compares\n"
        "    the number of parameter groups, and a 1-group vs 2-group mismatch\n"
        "    makes resume fail outright.\n"
        "\n"
        "    No weight decay on 1-D tensors (norm scales, biases, decay gates):\n"
        "    decaying the fp32 gate gamma toward 0 would kill long-range memory.\n"
        "    \"\"\"\n"
        "    _wd = [p for p in model.parameters() if p.ndim >= 2]\n"
        "    _nowd = [p for p in model.parameters() if p.ndim < 2]\n"
        "    return AdamW(\n"
        "        [\n"
        "            {\"params\": _wd, \"weight_decay\": CFG.weight_decay},\n"
        "            {\"params\": _nowd, \"weight_decay\": 0.0},\n"
        "        ],\n"
        "        lr=CFG.lr,\n"
        "        betas=(0.9, 0.95),\n"
        "        eps=1e-8,\n"
        "    )\n"
        "\n"
        "\n"
        "class CosineWithMinLRScheduler(torch.optim.lr_scheduler.LambdaLR):\n",
        "F10b add build_optimizer",
    )

    # F12 (A3, found by the notebook auditor): checkpoint pruning parses every
    # `checkpoint-*` DIRECTORY name as an int, so one stray non-numeric dir (a
    # partially copied checkpoint, a manual rename) raises ValueError at the
    # first save, i.e. step 500.  `_latest_checkpoint` already tolerates it.
    src = edit(
        src,
        "        checkpoints = sorted(\n"
        "            (p for p in Path(output_dir).glob(\"checkpoint-*\") if p.is_dir()),\n"
        "            key=lambda p: int(p.name.split(\"-\")[-1]),\n"
        "        )\n",
        "        def _step_of(path):\n"
        "            m = re.fullmatch(r\"checkpoint-(\\d+)\", path.name)\n"
        "            return int(m.group(1)) if m else None\n"
        "\n"
        "        checkpoints = sorted(\n"
        "            (p for p in Path(output_dir).glob(\"checkpoint-*\")\n"
        "             if p.is_dir() and _step_of(p) is not None),\n"
        "            key=_step_of,\n"
        "        )\n",
        "F12 pruning ignores non-numeric checkpoint dirs",
    )

    # F13 (auditor note): cell 0 still advertised torch.autocast("xla"), which
    # cell 4 has not used for several revisions.
    md_old2 = (
        "**精度策略：** TPU 使用 `XLA_USE_BF16=1` + `torch.autocast(\"xla\", dtype=torch.bfloat16)`；"
        "模型参数保持可加载的普通 PyTorch/HF `state_dict`，loss/reduction 明确保留 FP32。"
    )
    md_new2 = (
        "**精度策略：** TPU 使用 `XLA_USE_BF16=1` + 手动混合精度（**不用 `torch.autocast`**："
        "它在 XLA 上会与 scan/lazy 图产生 f32/bf16 签名不一致而崩，且与 `XLA_USE_BF16=1` 冗余）。"
        "模型参数保持可加载的普通 PyTorch/HF `state_dict`，decay 门控、norm、loss/reduction 明确保留 FP32。"
        "\n\n"
        "**decay 门控上限：** cell 1 的 `GAMMA_MAX` 把学习到的逐头衰减门控钳在 1.0 以下。"
        "chunkwise 状态按 `gamma**seq_len` 增长，初始权重里 `blocks.12.msr.gamma[10]=1.0117` "
        "在 8K 下会达到 2.8e41、超出 bf16 上限而 NaN。"
    )

    cells[4]["source"] = src.splitlines(keepends=True)
    cells[0]["source"] = "".join(cells[0]["source"]).replace(md_old, md_new)
    assert md_new in "".join(cells[0]["source"]), "cell 0 markdown edit failed"
    cells[0]["source"] = "".join(cells[0]["source"]).replace(md_old2, md_new2)
    assert md_new2 in "".join(cells[0]["source"]), "cell 0 precision-policy edit failed"
    print("[ok] F13 cell 0 markdown: autocast + gate-ceiling documented")

    # ---- cell 1: add the decay-gate ceiling knob ----
    c1 = "".join(cells[1]["source"])
    g_old = "SAVE_LIMIT       = 3\n"
    g_new = (
        "SAVE_LIMIT       = 3\n"
        "\n"
        "# Hard ceiling for the learned per-head decay gates (see cell 4).  The\n"
        "# shipped checkpoint has blocks.12.msr.gamma[10] = 1.0117; a gate above 1\n"
        "# makes the chunkwise state grow like gamma**seq_len (2.8e41 at 8192),\n"
        "# which overflows bf16 and yields NaN loss from step 1.  1 - 1e-6 keeps\n"
        "# the recurrence stable while staying numerically ~1.0.  Set to None to\n"
        "# disable the clamp (not recommended at seq_len 8192).\n"
        "GAMMA_MAX        = 1.0 - 1e-6\n"
    )
    assert c1.count(g_old) == 1, "cell 1 SAVE_LIMIT anchor"
    c1 = c1.replace(g_old, g_new)
    cells[1]["source"] = c1.splitlines(keepends=True)
    print("[ok] F11e cell 1: GAMMA_MAX knob")

    DST.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
