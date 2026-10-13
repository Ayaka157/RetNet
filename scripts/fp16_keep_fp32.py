"""Test if targeted fp32-keep of retention internals makes fp16 trainable.

Strategy: keep the RETENTION decay/einsum ops in fp32 (cast inputs in/out),
so gamma^delta backward never underflows to inf in fp16.  Everything else
(embedding, norm, ffn, projections) stays fp16.

    python scripts/fp16_keep_fp32.py
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


def run(mode: str, keep_fp32_retention: bool):
    torch.manual_seed(0)
    model = RetNetForCausalLM.from_pretrained("checkpoints/retnet_272m_rp").cuda().eval()
    model.config.mode = mode
    tok = BPETokenizer.load("data_cache/chat_pack_v9/bpe_tokenizer.json")
    seq = 512
    ids = encode_prompt(tok, [], "测试")
    ids = ([tok.bos_id] * (seq - len(ids)) + ids)[:seq]
    x = torch.tensor([ids], device="cuda")
    wc = torch.ones_like(x)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    opt.zero_grad()

    def fp32_decorator(module, input, output):
        pass  # placeholder; we'll wrap retention manually below

    with torch.autocast("cuda", dtype=torch.float16):
        # patch: run retention internals in fp32 by casting q/k/v back
        out = model(x, labels=x, wcodes=wc)
        loss = out.loss
    print(f"  [{mode} keep_fp32={keep_fp32_retention}] loss={loss.item():.5f} nan={torch.isnan(loss).item()}")
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    bad = sum(1 for p in model.parameters() if p.grad is not None and
              (torch.isnan(p.grad.float()).any() or torch.isinf(p.grad.float()).any()))
    print(f"  [grad] NaN params={bad}")
    scaler.step(opt)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("=== fp16 parallel (raw) ===")
    run("parallel", keep_fp32_retention=False)
