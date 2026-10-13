"""Find the FIRST backward gradient NaN in fp16 chunkwise (fork model).

Backward pass, then scan model params in order; report first NaN-grad param
with its name/shape.  Also try a variant with grad_scaler disabled scaling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf_fp16.modeling_retnet import RetNetForCausalLM  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402
from retnet.chat import encode_prompt  # noqa: E402


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    torch.manual_seed(0)
    model = RetNetForCausalLM.from_pretrained("checkpoints/retnet_272m_rp").cuda().eval()
    model.config.mode = "chunkwise"
    model.config.chunk_size = 64
    tok = BPETokenizer.load("data_cache/chat_pack_v9/bpe_tokenizer.json")
    seq = 512
    ids = encode_prompt(tok, [], "测试")
    ids = ([tok.bos_id] * (seq - len(ids)) + ids)[:seq]
    x = torch.tensor([ids], device="cuda")
    wc = torch.ones_like(x)

    model.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16):
        h = model.embedding(x) * model.embed_scale
        for bi, blk in enumerate(model.blocks):
            n1 = blk.norm1(h)
            if bi in (0, 1, 5):
                print(f"  [grad-on] L{bi} norm1 min={n1.float().min().item():.4g} max={n1.float().max().item():.4g} "
                      f"nan={torch.isnan(n1.float()).any().item()}")
            r = blk.msr(n1)
            if bi in (0, 1, 5):
                print(f"  [grad-on] L{bi} msr-out min={r.float().min().item():.4g} max={r.float().max().item():.4g} "
                      f"nan={torch.isnan(r.float()).any().item()}")
            n2 = blk.norm2(r)
            f = blk.ffn(n2)
            h = h + r + f
        logits = model.lm_head(model.final_norm(h))
        print(f"  [grad-on] logits nan={torch.isnan(logits.float()).any().item()}")
    print("logits 正常 -> 问题在 loss 计算")

    order = []
    for name, p in model.named_parameters():
        order.append((name, p))
    print("--- NaN grad params (first 20, name: nan/inf) ---")
    cnt = 0
    for name, p in order:
        if p.grad is None:
            continue
        g = p.grad.float()
        n = torch.isnan(g).any().item()
        i = torch.isinf(g).any().item()
        if n or i:
            print(f"  {name}  nan={n} inf={i} grad_absmax={g.abs().max().item():.4g}")
            cnt += 1
            if cnt >= 20:
                break
    print(f"total NaN params: {cnt}+")


if __name__ == "__main__":
    main()
