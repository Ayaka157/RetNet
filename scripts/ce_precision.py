"""Controlled CE precision check: logits.float() vs bf16 logits directly.
Same frozen model, same input, same labels -> is the bf16 path numerically
equivalent (PyTorch log_softmax accumulates in fp32 for bf16)? """
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

torch.manual_seed(0)
model = RetNetForCausalLM.from_pretrained("checkpoints/retnet_272m_rp").cuda().eval()
model.config.mode = "chunkwise"
model.config.chunk_size = 64
V = model.config.vocab_size

B, L = 2, 4096
x = torch.randint(0, V, (B, L), device="cuda")
labels = torch.randint(0, V, (B, L), device="cuda")

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(input_ids=x).logits   # bf16

    # reference: fp32
    lf = logits.float()
    ref = F.cross_entropy(lf.reshape(-1, V), labels.reshape(-1), reduction="none")

    # candidate: bf16 logits directly
    cand = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="none")
    cand_f = cand.float()

diff = (cand_f - ref).abs()
print(f"ref  (float)  loss mean = {ref.mean().item():.6f}")
print(f"cand (bf16)   loss mean = {cand_f.mean().item():.6f}")
print(f"max |diff|              = {diff.max().item():.6f}")
print(f"mean |diff|             = {diff.mean().item():.6f}")
print(f"rel mean diff           = {(diff / ref.abs().clamp_min(1e-6)).mean().item():.3e}")

# also check the weighted-loss end-to-end equivalence
w = torch.ones((B, L), device="cuda", dtype=torch.float32)
loss_ref = (ref * w.reshape(-1)).sum() / w.sum()
loss_cand = (cand_f * w.reshape(-1)).sum() / w.sum()
print(f"\nweighted loss  ref = {loss_ref.item():.6f}   cand = {loss_cand.item():.6f}")
