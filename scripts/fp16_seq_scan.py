"""Test fp16 2K with gradient checkpointing (remote)."""
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
    for seq in (2048, 4096, 8192):
        torch.manual_seed(0)
        model = RetNetForCausalLM.from_pretrained("checkpoints/retnet_272m_rp").cuda().train()
        model.config.mode = "chunkwise"
        model.config.chunk_size = 64
        for b in model.blocks:
            b.gradient_checkpointing = True
        ids = np.load("data_cache/chat_pack_v11/val.ids.npy", mmap_mode="r")[0][: seq + 1]
        x = torch.tensor(ids[:-1].astype(np.int64), device="cuda").unsqueeze(0)
        wc = torch.ones_like(x)
        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, labels=x, wcodes=wc)
            l = out.loss
        l.backward()
        bad = sum(1 for p in model.parameters() if p.grad is not None and
                  (torch.isnan(p.grad.float()).any() or torch.isinf(p.grad.float()).any()))
        print(f"seq={seq}: loss={l.item():.4f} nan={torch.isnan(l).item()} NaN_grads={bad}", flush=True)


if __name__ == "__main__":
    main()
