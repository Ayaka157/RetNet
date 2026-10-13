"""Locate 8K fp16 NaN: check q/k/v projection and first-layer internals."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf_fp16.modeling_retnet import RetNetForCausalLM  # noqa: E402


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    torch.manual_seed(0)
    model = RetNetForCausalLM.from_pretrained("checkpoints/retnet_272m_rp").cuda().train()
    model.config.mode = "chunkwise"
    model.config.chunk_size = 64
    for b in model.blocks:
        b.gradient_checkpointing = True

    ids_np = np.load("data_cache/chat_pack_v11/val.ids.npy", mmap_mode="r")[0][:8193]
    x = torch.tensor(ids_np[:-1].astype(np.int64), device="cuda").unsqueeze(0)

    with torch.autocast("cuda", dtype=torch.float16):
        h = model.embedding(x) * model.embed_scale
        print(f"embedding out: max={h.float().abs().max().item():.4g} nan={torch.isnan(h.float()).any().item()}")
        for bi, blk in enumerate(model.blocks):
            n1 = blk.norm1(h)
            q, k, v, g = blk.msr._project_and_rotate(n1, 0)
            qmax = q.float().abs().max().item()
            knan = torch.isnan(k.float()).any().item()
            vinf = torch.isinf(v.float()).any().item()
            if bi < 2 or torch.isnan(q.float()).any() or torch.isnan(k.float()).any() or torch.isnan(v.float()).any():
                print(f"L{bi}: q_max={qmax:.4g} k_nan={knan} v_inf={vinf}  [q nan={torch.isnan(q.float()).any().item()} k inf={torch.isinf(k.float()).any().item()} v nan={torch.isnan(v.float()).any().item()}]")
            if torch.isnan(q.float()).any() or torch.isnan(k.float()).any() or torch.isnan(v.float()).any():
                print(f"  >>> NaN in projection at L{bi}")
                break


if __name__ == "__main__":
    main()
