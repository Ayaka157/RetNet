"""Lead's acceptance harness: execute notebook cell 4 unmodified on CPU.

A fake torch_xla is injected so the cell's real code path runs unchanged:
  - `torch_xla.device()` returns a device-like object whose device ops are
    stripped back to CPU by a tensor-method patch (scoped, restored at exit),
  - `torch_xla.utils.checkpoint.checkpoint` is the plain autograd checkpoint
    (reentrant), matching the semantics trainer relies on,
  - `xm` / `xr` / `pl` are stubbed to single-device behaviour.

A tiny model config keeps it fast while keeping cell 4's
`torch.set_default_dtype(torch.bfloat16)` so the dtype defects reproduce.

usage: python run_fix_acceptance.py <notebook-or-cell4-path> [--out OUT_DIR]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

NB_DEFAULT = r"E:\Workspaces\RetNet\Kaggle\202610011026_fix.ipynb"


def load_cell4(path: Path) -> str:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        return "".join(nb["cells"][4]["source"])
    return path.read_text(encoding="utf-8")


class FakeDevice(str):
    """Acts like torch.device('cpu')/str for printing, tagged as the XLA device."""

    @property
    def type(self):
        return "xla"

    def __repr__(self):
        return "xla:0"


def make_torch_xla(torch, *, real_checkpoint, rank=0):
    import importlib.machinery
    import types

    def new_mod(name, is_package=False):
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
        if is_package:
            spec.submodule_search_locations = []
        m = types.ModuleType(name)
        m.__spec__ = spec
        if is_package:
            m.__path__ = []
        return m

    mod = new_mod("torch_xla", is_package=True)
    dev = FakeDevice("cpu")
    mod.device = lambda *a, **k: dev
    mod.sync = lambda *a, **k: None
    mod.__version__ = "fake-2.7"

    # --- torch_xla.utils(.checkpoint) ---
    utils = new_mod("torch_xla.utils", is_package=True)
    ckpt_mod = new_mod("torch_xla.utils.checkpoint")
    ckpt_mod.checkpoint = real_checkpoint
    utils.checkpoint = ckpt_mod
    mod.utils = utils

    # --- torch_xla.core.xla_model ---
    core = new_mod("torch_xla.core", is_package=True)
    xm = new_mod("torch_xla.core.xla_model")
    xm.REDUCE_SUM = "sum"
    xm.is_master_ordinal = lambda *a, **k: rank == 0
    xm.xla_device = lambda *a, **k: dev
    xm.rendezvous = lambda tag, *a, **k: None
    xm.mark_step = lambda *a, **k: None

    def all_reduce(reduce_type, value, *a, **k):
        return value

    def optimizer_step(optimizer, *a, **k):
        optimizer.step()

    def save(data, path, master_only=False, *a, **k):
        torch.save(data, path)

    xm.all_reduce = all_reduce
    xm.optimizer_step = optimizer_step
    xm.save = save
    xm.get_rng_state = lambda *a, **k: torch.get_rng_state()
    xm.set_rng_state = lambda s, *a, **k: torch.set_rng_state(s)
    core.xla_model = xm
    mod.core = core

    # --- torch_xla.runtime ---
    rt = new_mod("torch_xla.runtime")
    rt.global_ordinal = lambda *a, **k: rank
    rt.world_size = lambda *a, **k: (1 if rank == 0 else 8)
    rt.device_type = lambda *a, **k: "CPU"
    mod.runtime = rt

    # --- torch_xla.distributed(.parallel_loader, .xla_multiprocessing) ---
    dist = new_mod("torch_xla.distributed", is_package=True)
    pl = new_mod("torch_xla.distributed.parallel_loader")

    class MpDeviceLoader:
        def __init__(self, loader, device, **kw):
            self._loader = loader

        def __iter__(self):
            return iter(self._loader)

        def __len__(self):
            return len(self._loader)

    pl.MpDeviceLoader = MpDeviceLoader
    xmp = new_mod("torch_xla.distributed.xla_multiprocessing")
    xmp.spawn = lambda *a, **k: None
    dist.parallel_loader = pl
    dist.xla_multiprocessing = xmp
    mod.distributed = dist
    return mod


def main() -> int:
    nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(NB_DEFAULT)
    rank = 0
    if "--rank" in sys.argv:
        rank = int(sys.argv[sys.argv.index("--rank") + 1])
    cell4 = load_cell4(nb_path)
    print(f"[harness] cell4 from {nb_path}  ({len(cell4.splitlines())} lines)  rank={rank}",
          flush=True)

    import torch
    from torch.utils.checkpoint import checkpoint as real_checkpoint

    txl = make_torch_xla(torch, real_checkpoint=real_checkpoint, rank=rank)
    sys.modules["torch_xla"] = txl
    sys.modules["torch_xla.utils"] = txl.utils
    sys.modules["torch_xla.utils.checkpoint"] = txl.utils.checkpoint
    sys.modules["torch_xla.core"] = txl.core
    sys.modules["torch_xla.core.xla_model"] = txl.core.xla_model
    sys.modules["torch_xla.runtime"] = txl.runtime
    sys.modules["torch_xla.distributed"] = txl.distributed
    sys.modules["torch_xla.distributed.parallel_loader"] = txl.distributed.parallel_loader
    sys.modules["torch_xla.distributed.xla_multiprocessing"] = txl.distributed.xla_multiprocessing

    # Strip the fake device from any op that receives it.
    def patch_method(cls, name, device):
        orig = getattr(cls, name)
        if getattr(orig, "_dsh_patched", False):
            return
        import functools

        @functools.wraps(orig)
        def wrapper(self, *args, **kwargs):
            if self is device or self == device:
                self = "cpu"
            if "device" in kwargs and kwargs["device"] is device:
                kwargs["device"] = "cpu"
            args = tuple("cpu" if (a is device) else a for a in args)
            return orig(self, *args, **kwargs)

        wrapper._dsh_patched = True
        setattr(cls, name, wrapper)

    for meth in ("to", "cpu", "cuda"):
        for cls in (torch.Tensor,):
            if hasattr(cls, meth):
                patch_method(cls, meth, txl.device())
    patch_method(torch.nn.Module, "to", txl.device())

    tmp = Path(tempfile.mkdtemp(prefix="retnet_accept_"))
    out_dir = tmp
    dataset_dir = tmp / "ds"
    model_dir = tmp / "model"
    dataset_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    cfg = dict(
        vocab_size=256, d_model=64, num_layers=2, num_heads=4, ffn_dim=128,
        value_dim=128, dropout=0.0, norm_eps=1e-5, tie_weights=True,
        embed_scale=None, prompt_loss_weight=0.1, use_gla=True, use_swiglu=False,
        use_growth_ln=True, growth_old_dim=48, chunk_size=64, mode="chunkwise",
    )
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    print(f"[harness] tmp={tmp}", flush=True)
    g = {
        "__name__": "__main__",
        "DATASET_DIR": str(dataset_dir),
        "MODEL_DIR": str(model_dir),
        "OUTPUT_DIR": str(out_dir),
        "MODEL_MODE": "chunkwise",
        "SEQ_LEN": 128,
        "PER_DEVICE_BATCH": 2,
        "GRAD_ACCUM": 1,
        "MAX_STEPS": 4,
        "LR": 1e-3,
        "MIN_LR": 1e-4,
        "WARMUP": 2,
        "WEIGHT_DECAY": 0.1,
        "MAX_GRAD_NORM": 1.0,
        "CHUNK_SIZE": 64,
        "SAVE_STEPS": 2,
        "EVAL_STEPS": 2,
        "SAVE_LIMIT": 2,
        # cell 4 asserts this exists (defined in cell 1)
        "GAMMA_MAX": 1.0 - 1e-6,
    }

    print("=" * 70, flush=True)
    print("STAGE 1: exec cell 4 (module level)", flush=True)
    print("=" * 70, flush=True)
    exec(compile(cell4, "cell4.py", "exec"), g)
    print("[harness] cell 4 module level OK", flush=True)

    PackedChatDataset = g["PackedChatDataset"]
    RetNetForCausalLM = g["RetNetForCausalLM"]
    RetNetHFConfig = g["RetNetHFConfig"]
    CFG = g["CFG"]

    # Materialise real weights so train_worker's from_pretrained has a
    # config.json + model.safetensors to read, like the Kaggle input dir.
    _seed_model = RetNetForCausalLM(RetNetHFConfig(**cfg))
    _seed_model.save_pretrained(str(model_dir), safe_serialization=True)
    print(f"[harness] seeded model_dir: "
          f"{sorted(p.name for p in model_dir.iterdir())}", flush=True)
    del _seed_model

    # ---- dataset ----
    import numpy as np

    rng = np.random.default_rng(0)
    n = 8
    ids = rng.integers(0, 256, size=(n, 129), dtype=np.int32)
    codes = rng.integers(0, 4, size=(n, 129), dtype=np.int32)
    np.save(dataset_dir / "train.ids.npy", ids)
    np.save(dataset_dir / "train.wcode.npy", codes)
    np.save(dataset_dir / "val.ids.npy", ids[:4])
    np.save(dataset_dir / "val.wcode.npy", codes[:4])

    print("=" * 70, flush=True)
    print("STAGE 2: numerically compare chunkwise vs parallel (RetNet eq. 1 vs 7)", flush=True)
    print("=" * 70, flush=True)
    import torch as T

    model = RetNetForCausalLM(RetNetHFConfig(**cfg)).eval()
    toks = T.randint(0, 256, (2, 128))
    captured = {}
    _orig_fh = model._forward_hidden

    def _cap_fh(*a, **k):
        if k.get("return_state"):
            captured["states"] = _orig_fh(*a, **k)[1]
        return _orig_fh(*a, **k)

    model._forward_hidden = _cap_fh
    with T.no_grad():
        r_chunk = model(toks, mode="chunkwise", return_state=True, chunk_size=64)
        st_chunk = captured["states"]
        r_par = model(toks, mode="parallel", return_state=True)
        st_par = captured["states"]
        r_par_nostate = model(toks, mode="parallel", return_state=False)
    model._forward_hidden = _orig_fh
    d_out = (r_chunk.logits - r_par.logits).abs().max().item()
    scale = r_par.logits.abs().max().item()
    d_state = max((a.float() - b.float()).abs().max().item() for a, b in zip(st_chunk, st_par))
    print(f"[cmp] chunkwise vs parallel logits: max|diff|={d_out:.3e} (scale {scale:.3e})", flush=True)
    print(f"[cmp] chunkwise vs parallel final states: max|diff|={d_state:.3e}", flush=True)
    print(f"[cmp] return_state=True vs False logits identical: "
          f"{(r_par.logits == r_par_nostate.logits).all().item()}", flush=True)

    print("=" * 70, flush=True)
    print("STAGE 3: rotary sanity at a long context (the F3 fix)", flush=True)
    print("=" * 70, flush=True)
    T.set_default_dtype(T.bfloat16)
    block = model.blocks[0].msr
    pos = T.arange(0, 8192, dtype=T.float32)
    inv = block.inv_freq.float()
    freqs = T.outer(pos, inv)
    sin_ref, cos_ref = T.sin(freqs), T.cos(freqs)
    s, c = block._rotations(8192, 0, T.device("cpu"), T.bfloat16)
    print(f"[rot] returned dtype sin={s.dtype} cos={c.dtype}", flush=True)
    print(f"[rot] max|sin - sin(fp32)| = {(s.float()-sin_ref).abs().max().item():.3e}", flush=True)
    print(f"[rot] max|cos - cos(fp32)| = {(c.float()-cos_ref).abs().max().item():.3e}", flush=True)
    bf16_pos = T.arange(0, 8192, dtype=T.bfloat16)
    print(f"[rot] bf16 arange position error at idx 512 = "
          f"{(bf16_pos[512].float()-512).item():.1f}, at 8191 = "
          f"{(bf16_pos[8191].float()-8191).item():.1f}", flush=True)

    print("=" * 70, flush=True)
    print("STAGE 4: train_worker() end-to-end (the world_size / re / checkpoint path)", flush=True)
    print("=" * 70, flush=True)
    try:
        g["train_worker"]()
        print("[harness] train_worker COMPLETED", flush=True)
    except Exception:
        traceback.print_exc()
        print("[harness] train_worker FAILED", flush=True)
        return 1

    ckpts = sorted(p.name for p in out_dir.glob("checkpoint-*"))
    print(f"[harness] checkpoints written: {ckpts}", flush=True)
    print(f"[harness] final export present: "
          f"{(out_dir / 'model.safetensors').is_file() or (out_dir / 'config.json').is_file()}",
          flush=True)

    print("=" * 70, flush=True)
    print("STAGE 5: resume path (_latest_checkpoint + reload), where `re` is needed", flush=True)
    print("=" * 70, flush=True)
    try:
        latest_step, latest_dir = g["_latest_checkpoint"](str(out_dir))
        print(f"[resume] _latest_checkpoint -> step={latest_step} dir={latest_dir.name if latest_dir else None}",
              flush=True)
        model2 = RetNetForCausalLM(RetNetHFConfig(**cfg))
        opt2 = g["build_optimizer"](model2)
        sch2 = g["CosineWithMinLRScheduler"](opt2, 2, 10, 1e-4)
        st, best = g["_load_checkpoint"](model2, opt2, sch2, latest_dir)
        print(f"[resume] reloaded step={st} best={best:.4f}", flush=True)
    except Exception:
        traceback.print_exc()
        print("[harness] resume FAILED", flush=True)
        return 1

    print("=" * 70, flush=True)
    print("STAGE 6: which factories cell 4's default_dtype actually poisons", flush=True)
    print("=" * 70, flush=True)
    import torch as T2

    T2.set_default_dtype(T2.bfloat16)             # exactly what cell 4 does
    probes = [
        ("torch.arange(8192)", lambda: T2.arange(8192)),
        ("torch.arange(8192, device=...)", lambda: T2.arange(8192, device=T2.device("cpu"))),
        ("torch.zeros(2)", lambda: T2.zeros(2)),
        ("torch.ones(2)", lambda: T2.ones(2)),
        ("torch.tensor(1.0)", lambda: T2.tensor(1.0)),
        ("torch.tensor(float('nan'))", lambda: T2.tensor(float("nan"))),
    ]
    for label, fn in probes:
        t = fn()
        note = ""
        if t.dtype == T2.int64:
            note = "  <- integer factory: NOT poisoned, always exact"
        print(f"[dtype] {label:32s} -> {str(t.dtype).replace('torch.','')}{note}", flush=True)
    p = T2.arange(8192, dtype=T2.bfloat16).float()
    e = T2.arange(8192, dtype=T2.float32)
    err = (p - e).abs()
    print(f"[dtype] bf16 positions (the F3 defect)      : "
          f"{int((err > 0).sum())}/8192 inexact, max err {err.max().item():.1f}", flush=True)
    T2.set_default_dtype(T2.bfloat16)

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n[harness] ALL STAGES PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
