"""Compare the notebook's chunkwise retention against the project reference.

`E:\\Workspaces\\RetNet\\retnet_hf\\modeling_retnet.py` (and `train.py`) are the
canonical implementations this notebook must match: the initial weights under
`Kaggle/archived/retnet_272m_init_weights/retnet_272m_rp` were trained with
them, and `state_dict` keys/shapes must stay identical.

This loads both `MultiScaleRetention` implementations, feeds them identical
q/k/v, and reports per-path relative differences.  If the notebook diverges
from the reference, the fix must restore agreement.

Run:  python compare_with_reference.py
"""
from __future__ import annotations

import importlib.machinery
import json
import sys
import types
from pathlib import Path

ROOT = Path(r"E:\Workspaces\RetNet\Kaggle")
REPO = Path(r"E:\Workspaces\RetNet")


def load_notebook_module(torch):
    """Exec notebook cell 4 and hand back its globals."""
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


def main() -> int:
    import torch

    nb = load_notebook_module(torch)
    sys.path.insert(0, str(REPO))
    from retnet_hf.modeling_retnet import (
        MultiScaleRetention as RefMSR,
        RetNetHFConfig as RefConfig,
    )

    cfg = dict(vocab_size=256, d_model=64, num_layers=2, num_heads=4, ffn_dim=128,
               value_dim=128, dropout=0.0, norm_eps=1e-5, tie_weights=True,
               embed_scale=None, prompt_loss_weight=0.1, use_gla=True,
               use_swiglu=False, use_growth_ln=True, growth_old_dim=48,
               chunk_size=64, mode="chunkwise")

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    a = nb["MultiScaleRetention"](nb["RetNetHFConfig"](**cfg)).eval()
    b = RefMSR(RefConfig(**cfg)).eval()
    # The notebook's _chunkwise_retention hard-codes bf16 decay constants, so
    # compare in the dtype the notebook actually trains in.
    a = a.to(torch.bfloat16)
    b = b.to(torch.bfloat16)

    # identical weights so only the math differs
    b.load_state_dict(a.state_dict(), strict=True)
    print("[setup] state_dict keys identical and loadable:", True)

    x = torch.randn(2, 128, 64).to(torch.bfloat16)
    qa, ka, va, ga = a._project_and_rotate(x, 0)
    qb, kb, vb, gb = b._project_and_rotate(x, 0)
    print(f"[setup] _project_and_rotate  q rel={((qa-qb).abs().max()/qb.abs().max()).item():.3e} "
          f"k rel={((ka-kb).abs().max()/kb.abs().max()).item():.3e} "
          f"v rel={((va-vb).abs().max()/vb.abs().max()).item():.3e}")

    def rel(p, r):
        return ((p.float() - r.float()).abs().max() / r.float().abs().max().clamp_min(1e-30)).item()

    with torch.no_grad():
        for mode, fa, fb in (
            ("chunkwise(64)", lambda: a._chunkwise_retention(qa, ka, va, None, 64),
                              lambda: b._chunkwise_retention(qb, kb, vb, None, 64)),
            ("parallel", lambda: a._parallel_retention(qa, ka, va),
                         lambda: b._parallel_retention(qb, kb, vb)),
        ):
            oa = fa()[0]
            ob = fb()[0]
            print(f"[{mode:14s}] notebook-vs-reference out rel={rel(oa, ob):.3e}  "
                  f"(nb scale {oa.abs().max().item():.3e}, ref scale {ob.abs().max().item():.3e})")
        # full msr forward
        oa = a(x)[0] if isinstance(a(x), tuple) else a(x)
        ob = b(x)[0] if isinstance(b(x), tuple) else b(x)
        print(f"[msr forward   ] rel={rel(oa, ob):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
