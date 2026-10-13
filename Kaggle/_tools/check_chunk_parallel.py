"""Is `chunkwise` (RetNet eq. 7) numerically equivalent to `parallel` (eq. 1)?

The bf16 model shows max|diff| ~ 1.5e-01 on logits of magnitude ~0.7, which is
far too large to dismiss without a look.  Re-run the same comparison in pure
fp32: if the diff collapses to rounding noise, the bf16 gap is accumulation
error, not an algorithmic mismatch between the two paths.

Run:  python check_chunk_parallel.py
"""
from __future__ import annotations

import importlib.machinery
import json
import sys
import types
from pathlib import Path

ROOT = Path(r"E:\Workspaces\RetNet\Kaggle")
NB = ROOT / "202610011026_fix.ipynb"


def install_torch_xla_stub(torch):
    """Minimal stub; cell 4 only needs the checkpoint symbol at module level."""
    spec = importlib.machinery.ModuleSpec("torch_xla", loader=None, is_package=True)
    spec.submodule_search_locations = []
    mod = types.ModuleType("torch_xla")
    mod.__spec__ = spec
    mod.__path__ = []

    def new_mod(name, pkg=False):
        s = importlib.machinery.ModuleSpec(name, loader=None, is_package=pkg)
        if pkg:
            s.submodule_search_locations = []
        m = types.ModuleType(name)
        m.__spec__ = s
        if pkg:
            m.__path__ = []
        return m

    from torch.utils.checkpoint import checkpoint as real_ckpt

    utils = new_mod("torch_xla.utils", True)
    cm = new_mod("torch_xla.utils.checkpoint")
    cm.checkpoint = real_ckpt
    utils.checkpoint = cm
    mod.utils = utils
    mod.device = lambda *a, **k: torch.device("cpu")
    mod.sync = lambda *a, **k: None
    return mod


def main() -> int:
    import torch

    sys.modules["torch_xla"] = install_torch_xla_stub(torch)
    sys.modules["torch_xla.utils"] = sys.modules["torch_xla"].utils
    sys.modules["torch_xla.utils.checkpoint"] = sys.modules["torch_xla"].utils.checkpoint

    nb = json.loads(NB.read_text(encoding="utf-8"))
    cell4 = "".join(nb["cells"][4]["source"])
    g = {
        "__name__": "__main__",
        # cell 4 reads the cell-1 globals when building CFG; any path works here
        # because this script never touches the dataset or the output dir.
        "DATASET_DIR": str(ROOT),
        "MODEL_DIR": str(ROOT),
        "OUTPUT_DIR": str(ROOT),
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
    }
    exec(compile(cell4, "cell4.py", "exec"), g)
    RetNetForCausalLM, RetNetHFConfig = g["RetNetForCausalLM"], g["RetNetHFConfig"]

    # `_chunkwise_retention` hard-codes bfloat16 for its decay constants and
    # initial state (correct for the real bf16 training run).  To compare the
    # two RetNet recurrences in fp32 we re-bind the SAME body with the dtype
    # parametrised -- no algorithmic change, only the cast target.
    import torch.nn.functional as F

    def chunkwise_as_dtype(self, q, k, v, state, chunk_size):
        dt = q.dtype
        bsz, _, tgt_len, key_dim = q.shape
        assert tgt_len % chunk_size == 0
        if state is None:
            state = torch.zeros(bsz, self.num_heads, key_dim, v.shape[-1],
                                device=q.device, dtype=dt)
        else:
            state = state.to(device=q.device, dtype=dt)
        n_chunks = tgt_len // chunk_size
        gamma32 = self.gamma.to(device=q.device, dtype=torch.float32)
        rel = torch.arange(chunk_size, device=q.device, dtype=torch.float32)
        delta = rel[:, None] - rel[None, :]
        causal = delta >= 0
        exp = delta.masked_fill(~causal, 0)
        inner_mask = (gamma32[:, None, None] ** exp).to(dt).masked_fill(~causal, 0.0)
        cross_decay = (gamma32[:, None] ** (rel + 1)).to(dt)
        value_decay = (gamma32[:, None] ** (chunk_size - 1 - rel)).to(dt)
        chunk_decay = (gamma32 ** chunk_size).to(dt)
        q_c = q.reshape(bsz, self.num_heads, n_chunks, chunk_size, key_dim)
        k_c = k.reshape(bsz, self.num_heads, n_chunks, chunk_size, key_dim)
        v_c = v.reshape(bsz, self.num_heads, n_chunks, chunk_size, v.shape[-1])
        cur, outs = state, []
        for i in range(n_chunks):
            qi, ki, vi = q_c[:, :, i], k_c[:, :, i], v_c[:, :, i]
            inner = torch.matmul(torch.matmul(qi, ki.transpose(-1, -2)) * inner_mask, vi)
            cross = torch.matmul(qi * cross_decay[None, :, :, None], cur)
            outs.append(inner + cross)
            cur = (chunk_decay.view(1, -1, 1, 1) * cur
                   + torch.matmul(ki.transpose(-1, -2), vi * value_decay[None, :, :, None]))
        return torch.stack(outs, dim=2).reshape(bsz, self.num_heads, tgt_len, v.shape[-1]), cur

    g["MultiScaleRetention"]._chunkwise_retention = chunkwise_as_dtype

    cfg = dict(vocab_size=256, d_model=64, num_layers=2, num_heads=4, ffn_dim=128,
               value_dim=128, dropout=0.0, norm_eps=1e-5, tie_weights=True,
               embed_scale=None, prompt_loss_weight=0.1, use_gla=True,
               use_swiglu=False, use_growth_ln=True, growth_old_dim=48,
               chunk_size=64, mode="chunkwise")

    torch.manual_seed(0)
    toks = torch.randint(0, 256, (2, 128))

    # ---- A: isolate the retention math from norm/gate/FFN/layernorm ----
    # A plain nn.LayerNorm has parameters at their defaults (ones/zeros), so a
    # GrowthLayerNorm replaced by identity keeps `msr` the only thing compared.
    msr = g["MultiScaleRetention"](RetNetHFConfig(**cfg))
    msr.eval()
    for dtype in (torch.float32, torch.bfloat16):
        msr = msr.to(dtype)
        msr.gamma.data = msr.gamma.data.to(torch.float32)   # manual mixed precision
        x = torch.randn(2, 128, 64).to(dtype)
        q, k, v, _ = msr._project_and_rotate(x, 0)
        with torch.no_grad():
            out_c, st_c = msr._chunkwise_retention(q, k, v, None, 64)
            out_c1, _ = msr._chunkwise_retention(q, k, v, None, 128)
            out_p = msr._parallel_retention(q, k, v)
            st_p = msr._parallel_final_state(k, v, None)

        def rel(a, b):
            return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()

        print(f"[retention {str(dtype).replace('torch.',''):8s}] "
              f"state chunk-vs-parallel rel={rel(st_c, st_p):.3e}   (recurrence check)")
        print(f"[retention {str(dtype).replace('torch.',''):8s}] "
              f"out  chunk(64)-vs-parallel rel={rel(out_c, out_p):.3e}   <-- current design")
        print(f"[retention {str(dtype).replace('torch.',''):8s}] "
              f"out  chunk(128)-vs-parallel rel={rel(out_c1, out_p):.3e}  <-- 1 chunk == eq.(1)")

        # Is the gap exactly _parallel_retention's score normalisation?
        mask = msr._decay_mask(128, q.device, q.dtype)
        sc = torch.matmul(q, k.transpose(-1, -2)) * mask
        norm = sc / sc.detach().abs().sum(-1, keepdim=True).clamp_min(1.0)
        print(f"[retention {str(dtype).replace('torch.',''):8s}] "
              f"matmul(q,k)*mask            vs chunk(128) rel={rel(torch.matmul(sc, v), out_c1):.3e}")
        print(f"[retention {str(dtype).replace('torch.',''):8s}] "
              f"normalised scores * v       vs chunk(128) rel={rel(torch.matmul(norm, v), out_c1):.3e}")
        print(f"[retention {str(dtype).replace('torch.',''):8s}] "
              f"mean |score| row-normaliser = "
              f"{(sc.detach().abs().sum(-1, keepdim=True)[0,0,0].item() / 128):.3e}")
        if dtype is torch.float32:
            print("[retention fp32 ] VERDICT:",
                  "EQUIVALENT" if rel(out_c1, out_p) < 1e-3
                  else "MISMATCH between eq.(1) and eq.(7) paths")

    # ---- B: end-to-end logits ----
    for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        m = RetNetForCausalLM(RetNetHFConfig(**cfg)).eval().to(dtype)
        # decay gates must stay fp32 (manual mixed precision), as in cell 4
        for b in m.blocks:
            b.msr.gamma.data = b.msr.gamma.data.to(torch.float32)
        with torch.no_grad():
            ch = m(toks, mode="chunkwise", return_state=True, chunk_size=64).logits.float()
            pa = m(toks, mode="parallel", return_state=True).logits.float()
        d = (ch - pa).abs()
        print(f"[{label:4s}] logits scale={pa.abs().max().item():.4e}  "
              f"max|chunk-parallel|={d.max().item():.4e}  "
              f"rel={(d.max()/pa.abs().max()).item():.3e}")
        if dtype is torch.float32:
            verdict = ("EQUIVALENT within fp32 rounding"
                       if (d.max() / pa.abs().max()).item() < 1e-3 else "MISMATCH -- algorithmic")
            print(f"[fp32] VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
