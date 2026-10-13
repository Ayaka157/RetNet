"""Gradient-level fp16 test: is NaN in backward or in loss value?

Runs ONE training step under fp16 autocast with a real forward+backward,
probing: loss value, grad norm per param group, and the first param with NaN
grad.  Compare against bf16 reference.  Also tests whether disabling the
GroupNorm fp16 path (post-backward) changes anything.

    python scripts/fp16_grad_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402
from retnet.chat import encode_prompt  # noqa: E402


def run(dtype: torch.dtype, mode: str, chunk_size: int = 64):
    torch.manual_seed(0)
    model = RetNetForCausalLM.from_pretrained("checkpoints/retnet_272m_rp").cuda().eval()
    model.config.mode = mode
    model.config.chunk_size = chunk_size
    tok = BPETokenizer.load("data_cache/chat_pack_v9/bpe_tokenizer.json")
    seq = 512
    ids = encode_prompt(tok, [], "测试模型稳定性")
    ids = ([tok.bos_id] * (seq - len(ids)) + ids)[:seq]
    x = torch.tensor([ids], device="cuda")
    wc = torch.ones_like(x)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    opt.zero_grad()

    with torch.autocast("cuda", dtype=dtype):
        out = model(x, labels=x, wcodes=wc)
        loss = out.loss
    print(f"  [{dtype} {mode}] loss={loss.item():.4f} (nan={torch.isnan(loss).item()})")
    scaler.scale(loss).backward()
    scaler.unscale_(opt)

    bad_n = 0
    first_bad = None
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.float()
        gn = g.norm().item()
        total_norm += gn * gn
        if torch.isnan(g).any() or torch.isinf(g).any():
            bad_n += 1
            if first_bad is None:
                first_bad = (name, g.abs().max().item())
    print(f"  [grad] NaN/inf params={bad_n} first_bad={first_bad} total_norm={total_norm**0.5:.3f}")
    scaler.step(opt)
    scaler.update()
    return loss.item()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("=== bf16 reference (parallel) ===")
    run(torch.bfloat16, "parallel")
    print("=== fp16 (parallel) ===")
    run(torch.float16, "parallel")
    print("=== fp16 (chunkwise) ===")
    run(torch.float16, "chunkwise")
