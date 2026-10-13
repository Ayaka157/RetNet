"""Pinpoint fp16 NaN: inspect chunkwise retention internals directly."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf_fp16.modeling_retnet import RetNetForCausalLM  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402
from retnet.chat import encode_prompt  # noqa: E402


def chk(name, t):
    fn = t.float()
    bad = torch.isnan(fn).any().item() or torch.isinf(fn).any().item()
    print(f"  {name:<30} min={fn.min():>12.4g} max={fn.max():>12.4g} nan={torch.isnan(fn).any().item()} inf={torch.isinf(fn).any().item()}{'<<<BAD' if bad else ''}")


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

    with torch.autocast("cuda", dtype=torch.float16):
        h = model.embedding(x) * model.embed_scale
        chk("hidden0", h)
        for bi, blk in enumerate(model.blocks):
            with torch.no_grad():
                n1 = blk.norm1(h)
                r = blk.msr(n1)
                chk(f"L{bi} msr-out", r)
                n2 = blk.norm2(r)
                f = blk.ffn(n2)
                h = h + r + f
                chk(f"L{bi} residual-h", h)
        fn = model.final_norm(h)
        lg = model.lm_head(fn)
        chk("final_norm", fn)
        chk("lm_head", lg)


if __name__ == "__main__":
    main()
