"""Locate the first fp16 overflow point in RetNet-272M forward pass.

Runs a single forward (chunkwise mode) with fp16 autocast, probing tensor
stats at each stage: input -> embedding -> each block's q/k/v, retention
scores/output, ff hidden -> logits.  Reports where inf/nan first appears.

    python scripts/fp16_diag.py --model_dir checkpoints/retnet_272m_rp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402
from retnet.chat import encode_prompt  # noqa: E402


def probe(name, t):
    if not torch.is_tensor(t):
        print(f"  {name}: (not tensor)")
        return
    fn = t.float()
    bad = torch.isnan(fn).any().item() or torch.isinf(fn).any().item()
    flag = " <-- BAD" if bad else ""
    print(f"  {name:<28} min={fn.min():>10.3f} max={fn.max():>10.3f} "
          f"nan={torch.isnan(fn).any().item()} inf={torch.isinf(fn).any().item()}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="checkpoints/retnet_272m_rp")
    ap.add_argument("--tokenizer", default="data_cache/chat_pack_v9/bpe_tokenizer.json")
    ap.add_argument("--seq_len", type=int, default=512)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = RetNetForCausalLM.from_pretrained(args.model_dir).cuda().eval()
    model.config.mode = "chunkwise"
    tok = BPETokenizer.load(args.tokenizer)
    ids = encode_prompt(tok, [], "测试一下模型在fp16下的稳定性")
    ids = ([tok.bos_id] * (args.seq_len - len(ids)) + ids)[:args.seq_len]
    x = torch.tensor([ids], device="cuda")
    wc = torch.ones_like(x)

    print("=== embedding ===")
    with torch.autocast("cuda", dtype=torch.float16):
        h = model.embedding(x)
        probe("embedding output", h)
        for bi, blk in enumerate(model.blocks):
            with torch.no_grad():
                ln = blk.norm1(h)
                probe(f"L{bi} norm1", ln)
                r = blk.msr(ln)
                probe(f"L{bi} retention out", r)
                a = blk.norm2(r)
                probe(f"L{bi} norm2", a)
                f = blk.ffn(a)
                probe(f"L{bi} ffn out", f)
                h = h + r + f
            probe(f"L{bi} residual (h)", h)
        final_ln = model.final_norm(h)
        probe("final_norm", final_ln)
        logits = model.lm_head(final_ln)
        probe("lm_head logits", logits)

    print("\n=== 结论: 若所有 BAD 标记出现前的第一个 BAD 点 === 即为溢出源头")


if __name__ == "__main__":
    main()
