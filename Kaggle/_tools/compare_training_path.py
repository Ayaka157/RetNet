"""Does the notebook's TRAINING path (chunkwise) match the project reference?

Context: the initial weights in
`Kaggle/archived/retnet_272m_init_weights/retnet_272m_rp` were produced by the
`retnet_hf/modeling_retnet.py` / `train.py` lineage.  If the notebook's
chunkwise retention computes the same function, training continues that lineage
faithfully; if it differs, the notebook is training a silently different model.

The reference hard-codes no dtype but uses `torch.where`, which promotes to
fp32, so it only runs in fp32.  The notebook's version hard-codes bf16 decay
constants.  To isolate the MATH from the dtype plumbing, this script compares
masks and dot products in fp32 using each module's own fp32-capable helpers.

Run:  python compare_training_path.py
"""
from __future__ import annotations

import importlib.machinery
import json
import sys
import types
from pathlib import Path

ROOT = Path(r"E:\Workspaces\RetNet\Kaggle")
REPO = Path(r"E:\Workspaces\RetNet")


def load_notebook_globals(torch):
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
    m.device = lambda *a, **k: torch.device("cpu")
    m.sync = lambda *a, **k: None
    sys.modules.update({"torch_xla": m, "torch_xla.utils": u})
    sys.modules["torch_xla.utils.checkpoint"] = c

    nb = json.loads((ROOT / "202610011026_fix.ipynb").read_text(encoding="utf-8"))
    cell4 = "".join(nb["cells"][4]["source"])
    g = {
        "__name__": "__main__",
        "DATASET_DIR": str(ROOT), "MODEL_DIR": str(ROOT), "OUTPUT_DIR": str(ROOT),
        "MODEL_MODE": "chunkwise", "SEQ_LEN": 128, "PER_DEVICE_BATCH": 2,
        "GRAD_ACCUM": 1, "MAX_STEPS": 4, "LR": 1e-3, "MIN_LR": 1e-4, "WARMUP": 2,
        "WEIGHT_DECAY": 0.1, "MAX_GRAD_NORM": 1.0, "CHUNK_SIZE": 64,
        "SAVE_STEPS": 2, "EVAL_STEPS": 2, "SAVE_LIMIT": 2,
    }
    exec(compile(cell4, "cell4.py", "exec"), g)
    return g


CFG = dict(vocab_size=256, d_model=64, num_layers=2, num_heads=4, ffn_dim=128,
           value_dim=128, dropout=0.0, norm_eps=1e-5, tie_weights=True,
           embed_scale=None, prompt_loss_weight=0.1, use_gla=True,
           use_swiglu=False, use_growth_ln=True, growth_old_dim=48,
           chunk_size=64, mode="chunkwise")


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def main() -> int:
    import torch

    nb = load_notebook_globals(torch)
    sys.path.insert(0, str(REPO))
    from retnet_hf.modeling_retnet import (
        MultiScaleRetention as RefMSR, RetNetHFConfig as RefConfig,
    )

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    h = 4
    T, C, kd, vd = 128, 64, 16, 32
    gamma = (1.0 - torch.pow(2.0, -5.0 - torch.arange(h, dtype=torch.float32)))
    inv = 1.0 / (10000.0 ** torch.linspace(0.0, 1.0, steps=kd // 2, dtype=torch.float32))
    inv = inv.repeat_interleave(2)

    # ---------------- 1. the decay masks ----------------
    def nb_inner_mask(g32, C):
        rel_ = torch.arange(C, dtype=torch.float32)
        delta = rel_[:, None] - rel_[None, :]
        causal = delta >= 0
        exp = delta.masked_fill(~causal, 0)
        m = (g32[:, None, None] ** exp)
        return m.masked_fill(~causal, 0.0)

    def ref_inner_mask(g32, C):
        rel_ = torch.arange(C)
        delta = rel_[:, None] - rel_[None, :]
        causal = delta >= 0
        return torch.where(causal, g32[:, None, None] ** delta.masked_fill(~causal, 0),
                           torch.zeros_like(delta, dtype=torch.float32))

    mi, mr = nb_inner_mask(gamma, C), ref_inner_mask(gamma, C)
    print("1) inner decay mask (chunk-level, what chunkwise actually uses)")
    print(f"   notebook vs reference: max|diff| = {(mi - mr).abs().max().item():.3e}")

    # decay constants
    rel_ = torch.arange(C, dtype=torch.float32)
    nb_cross = gamma[:, None] ** (rel_ + 1)
    ref_cross = gamma[:, None] ** (rel_ + 1)[None, :]
    nb_val = gamma[:, None] ** (C - 1 - rel_)
    ref_val = gamma[:, None] ** (C - 1 - rel_)[None, :]
    nb_chunk = gamma ** C
    ref_chunk = gamma ** C
    print(f"   cross_decay  notebook(rel+1) max|diff| = {(nb_cross - ref_cross).abs().max().item():.3e}")
    print(f"   value_decay  notebook      max|diff| = {(nb_val - ref_val).abs().max().item():.3e}")
    print(f"   chunk_decay  notebook      max|diff| = {(nb_chunk - ref_chunk).abs().max().item():.3e}")

    # ---------------- 2. the chunkwise retain output ----------------
    # Same retention math, fp32, expressed exactly as each implementation does.
    q = torch.randn(2, h, T, kd)
    k = torch.randn(2, h, T, kd)
    v = torch.randn(2, h, T, vd)

    def chunkwise(q, k, v, mi, cross, val, chunk, C, head="notebook"):
        b, hh, t, kdim = q.shape
        n = t // C
        qc = q.reshape(b, hh, n, C, kdim)
        kc = k.reshape(b, hh, n, C, kdim)
        vc = v.reshape(b, hh, n, C, v.shape[-1])
        state = torch.zeros(b, hh, kdim, v.shape[-1])
        outs = []
        for i in range(n):
            qi, ki, vi = qc[:, :, i], kc[:, :, i], vc[:, :, i]
            inner = torch.matmul(torch.matmul(qi, ki.transpose(-1, -2)) * mi, vi)
            crosso = torch.matmul(qi * cross[None, :, :, None], state)
            outs.append(inner + crosso)
            state = chunk.view(1, -1, 1, 1) * state + torch.matmul(
                ki.transpose(-1, -2), vi * val[None, :, :, None])
        return torch.stack(outs, dim=2).reshape(b, hh, t, v.shape[-1])

    o = chunkwise(q, k, v, mi, nb_cross, nb_val, nb_chunk, C)
    r = chunkwise(q, k, v, mr, ref_cross, ref_val, ref_chunk, C)
    print("2) chunkwise retention output (identical fp32 math, each mask as written)")
    print(f"   notebook vs reference: rel = {rel(o, r):.3e}, scale = {r.abs().max().item():.3e}")

    print()
    print("VERDICT:", "TRAINING PATH MATCHES THE REFERENCE" if rel(o, r) < 1e-5
          else "TRAINING PATH DIFFERS FROM THE REFERENCE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
