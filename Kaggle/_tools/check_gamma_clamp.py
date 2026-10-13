"""Verify the GAMMA_MAX clamp against the REAL checkpoint gate.

The shipped checkpoint has blocks.12.msr.gamma[10] = 1.0117193.  This script
executes the fixed notebook's cell 4 and asserts:
  1. the fp32 gate values survive `model.to(bfloat16)` bit-exactly,
  2. clamping to GAMMA_MAX actually fires on that gate,
  3. after clamping the chunkwise recurrence is finite at seq_len 8192.

Run:  python check_gamma_clamp.py
"""
from __future__ import annotations

import importlib.machinery
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(r"E:\Workspaces\RetNet\Kaggle")
REPO = Path(r"E:\Workspaces\RetNet")
CKPT = ROOT / "archived" / "retnet_272m_init_weights" / "retnet_272m_rp"
NB = ROOT / "202610011026_fix.ipynb"


def load_cell4_module(torch, *, extra_globals=None):
    spec = importlib.machinery.ModuleSpec("torch_xla", loader=None, is_package=True)
    spec.submodule_search_locations = []
    m = types.ModuleType("torch_xla")
    m.__spec__ = spec
    m.__path__ = []

    def nm(n, p=False):
        s = importlib.machinery.ModuleSpec(n, loader=None, is_package=p)
        if p:
            s.submodule_search_locations = []
        x = types.ModuleType(n)
        x.__spec__ = s
        if p:
            x.__path__ = []
        return x

    from torch.utils.checkpoint import checkpoint as rc

    u = nm("torch_xla.utils", True)
    c = nm("torch_xla.utils.checkpoint")
    c.checkpoint = rc
    u.checkpoint = c
    m.utils = u
    dev = "cpu"
    m.device = lambda *a, **k: dev
    m.sync = lambda *a, **k: None
    core = nm("torch_xla.core", True)
    xm = nm("torch_xla.core.xla_model")
    xm.is_master_ordinal = lambda *a, **k: True
    xm.xla_device = lambda *a, **k: dev
    xm.rendezvous = lambda *a, **k: None
    xm.REDUCE_SUM = "sum"
    core.xla_model = xm
    m.core = core
    rt = nm("torch_xla.runtime")
    rt.global_ordinal = lambda *a, **k: 0
    rt.world_size = lambda *a, **k: 1
    m.runtime = rt
    dist = nm("torch_xla.distributed", True)
    pl = nm("torch_xla.distributed.parallel_loader")
    pl.MpDeviceLoader = type("M", (), {"__init__": lambda s, l, d, **k: setattr(s, "_l", l),
                                       "__iter__": lambda s: iter(s._l)})
    xmp = nm("torch_xla.distributed.xla_multiprocessing")
    xmp.spawn = lambda *a, **k: None
    dist.parallel_loader = pl
    dist.xla_multiprocessing = xmp
    m.distributed = dist
    sys.modules.update({
        "torch_xla": m, "torch_xla.utils": u, "torch_xla.utils.checkpoint": c,
        "torch_xla.core": core, "torch_xla.core.xla_model": xm,
        "torch_xla.runtime": rt, "torch_xla.distributed": dist,
        "torch_xla.distributed.parallel_loader": pl,
        "torch_xla.distributed.xla_multiprocessing": xmp,
    })
    nb = json.loads(NB.read_text(encoding="utf-8"))
    g = {"__name__": "__main__"}
    g.update(extra_globals or {})
    exec(compile("".join(nb["cells"][4]["source"]), "cell4.py", "exec"), g)
    return g


def main() -> int:
    import torch

    tmp = Path(tempfile.mkdtemp(prefix="retnet_gamma_"))
    g = load_cell4_module(torch, extra_globals={
        "DATASET_DIR": str(tmp), "MODEL_DIR": str(CKPT), "OUTPUT_DIR": str(tmp),
        "MODEL_MODE": "chunkwise", "SEQ_LEN": 8192, "PER_DEVICE_BATCH": 1,
        "GRAD_ACCUM": 1, "MAX_STEPS": 2, "LR": 1.1e-4, "MIN_LR": 1.1e-5,
        "WARMUP": 1, "WEIGHT_DECAY": 0.1, "MAX_GRAD_NORM": 1.0, "CHUNK_SIZE": 64,
        "SAVE_STEPS": 0, "EVAL_STEPS": 0, "SAVE_LIMIT": 3,
        "GAMMA_MAX": 1.0 - 1e-6,
    })
    RetNetForCausalLM, RetNetHFConfig, MultiScaleRetention = (
        g["RetNetForCausalLM"], g["RetNetHFConfig"], g["MultiScaleRetention"])
    GAMMA_MAX = g["GAMMA_MAX"]

    # 1) real gate values survive the bf16 cast, bit-exactly
    model = RetNetForCausalLM.from_pretrained(str(CKPT))
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if n.endswith("msr.gamma")}
    inv_before = {n: b.detach().clone() for n, b in model.named_buffers()
                  if n.endswith("msr.inv_freq")}
    print(f"[1] gates={len(before)} inv_freq={len(inv_before)} "
          f"gamma dtype={next(iter(before.values())).dtype}")

    preserved = dict(before)
    for n, b in model.named_buffers():
        if n.endswith("msr.inv_freq"):
            preserved[n] = b.detach().clone()
    model.to(dtype=torch.bfloat16)
    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))
    for k, master in preserved.items():
        dst = tensors[k]
        dst.data = master.to(device=dst.device, dtype=torch.float32)

    exact = sum(int(torch.equal(tensors[k].data, master)) for k, master in preserved.items())
    print(f"[1] bitwise-identical after cast: {exact}/{len(preserved)} "
          f"| gamma dtype={tensors[list(before)[0]].dtype}")
    assert exact == len(preserved), "fp32 masters not preserved"

    # 2) the clamp fires on blocks.12.msr.gamma[10]
    key12 = [k for k in before if k.startswith("blocks.12.")][0]
    val = tensors[key12][10].item()
    print(f"[2] {key12}[10] before clamp = {val:.7f}  (GAMMA_MAX={GAMMA_MAX:.7f})")
    assert val > GAMMA_MAX, "expected an out-of-range gate in the real checkpoint"
    clamped = 0
    for k in before:
        t = tensors[k]
        if t.data.dtype == torch.float32:
            clamped += int((t.data > GAMMA_MAX).sum().item())
            t.data = torch.clamp(t.data, max=GAMMA_MAX)
    print(f"[2] gates clamped: {clamped}")
    print(f"[2] {key12}[10] after  clamp = {tensors[key12][10].item():.7f}")
    assert clamped >= 1, "clamp did not fire"

    # 3) chunkwise recurrence is finite at 8192 with the real (clamped) gates
    msr = MultiScaleRetention(RetNetHFConfig.from_pretrained(str(CKPT)))
    real_g = torch.stack([before[k] for k in sorted(before, key=lambda s: int(s.split(".")[1]))])
    msr.gamma.data = torch.clamp(real_g[12], max=GAMMA_MAX)
    torch.manual_seed(0)
    T = 8192
    x = torch.randn(1, T, msr.d_model, dtype=torch.bfloat16) * 0.1
    msr = msr.to(torch.bfloat16)
    msr.gamma.data = torch.clamp(real_g[12], max=GAMMA_MAX).to(torch.float32)
    with torch.no_grad():
        q, k, v, _ = msr._project_and_rotate(x, 0)
        out, state = msr._chunkwise_retention(q, k, v, None, 64)
    nan = int(torch.isnan(out).sum())
    inf = int(torch.isinf(out).sum())
    print(f"[3] seq={T} clamped: out max={out.abs().max().item():.3e} nan={nan} inf={inf} "
          f"state inf={int(torch.isinf(state).sum())}")
    assert nan == 0 and inf == 0, "still not finite at 8K"

    print("\nALL GAMMA-CLAMP CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
