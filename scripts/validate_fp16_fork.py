"""Validate the fp16-fork fixes: one fp16 forward+backward step must be NaN-free.

Compares fork (fp16) against original (bf16) on the same model/seed/step.
Run locally:  python scripts/validate_fp16_fork.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM as Orig  # noqa: E402
from retnet_hf_fp16.modeling_retnet import RetNetForCausalLM as Fork  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402
from retnet.chat import encode_prompt  # noqa: E402


def run_step(model_cls, dtype, mode, seq=512, chunk=64, keep_fp32=False):
    torch.manual_seed(0)
    model = model_cls.from_pretrained("checkpoints/retnet_272m_rp").cuda().eval()
    model.config.mode = mode
    model.config.chunk_size = chunk
    tok = BPETokenizer.load("data_cache/chat_pack_v9/bpe_tokenizer.json")
    ids = encode_prompt(tok, [], "测试")
    ids = ([tok.bos_id] * (seq - len(ids)) + ids)[:seq]
    x = torch.tensor([ids], device="cuda")
    wc = torch.ones_like(x)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    opt.zero_grad()
    if keep_fp32:
        # 禁用 autocast 对 retention 的 fp16 转换：手动跑 fp16 autocast 但
        # 在 retention forward 外层用 fp32 上下文？不——改用 autocast 黑名单。
        pass
    with torch.autocast("cuda", dtype=dtype):
        out = model(x, labels=x, wcodes=wc)
        loss = out.loss
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    bad = sum(1 for p in model.parameters() if p.grad is not None and
              (torch.isnan(p.grad.float()).any() or torch.isinf(p.grad.float()).any()))
    scaler.step(opt)
    return loss.item(), bad


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("=== fp16 修复验证（原版 vs fork） ===")
    print("--- 原版 fp16 parallel ---")
    try:
        l, b = run_step(Orig, torch.float16, "parallel")
        print(f"  loss={l:.4f} NaN_params={b}")
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:120]}")
    print("--- fork fp16 parallel ---")
    l, b = run_step(Fork, torch.float16, "parallel")
    print(f"  loss={l:.4f} NaN_params={b}")
    print("--- fork fp16 chunkwise ---")
    l, b = run_step(Fork, torch.float16, "chunkwise")
    print(f"  loss={l:.4f} NaN_params={b}")
    print("--- fork bf16 chunkwise (回归) ---")
    l, b = run_step(Fork, torch.bfloat16, "chunkwise")
    print(f"  loss={l:.4f} NaN_params={b}")
