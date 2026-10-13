"""Track chunkwise state across chunks in fp16 (fork) — find state explosion."""
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
    print(f"  {name:<28} min={fn.min():>12.4g} max={fn.max():>12.4g} "
          f"nan={torch.isnan(fn).any().item()} inf={torch.isinf(fn).any().item()}")


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
        msr = model.blocks[0].msr
        n1 = model.blocks[0].norm1(h)
        q, k, v, g = msr._project_and_rotate(n1, 0)
        # manually walk chunks of layer 0
        C = 64
        bsz, _, tgt_len, key_dim = q.shape
        state = torch.zeros(bsz, 16, key_dim, 128, device=q.device, dtype=torch.float32)
        gamma = msr.gamma.to(device=q.device, dtype=q.dtype)
        rel = torch.arange(C, device=q.device)
        delta = rel[:, None] - rel[None, :]
        causal = delta >= 0
        im = torch.where(causal, gamma[:, None, None] ** delta.masked_fill(~causal, 0),
                         torch.zeros_like(delta, dtype=torch.float32)).to(q.dtype)
        cd = (gamma.float()[:, None] ** (rel + 1)[None, :]).to(q.dtype)
        vd = (gamma.float()[:, None] ** (C - 1 - rel)[None, :]).to(q.dtype)
        chd = (gamma.float() ** C).to(q.dtype)
        for i in range(tgt_len // C):
            sl = slice(i * C, (i + 1) * C)
            qi, ki, vi = q[:, :, sl], k[:, :, sl], v[:, :, sl]
            ins = torch.matmul(qi, ki.transpose(-1, -2)) * im
            inner = torch.matmul(ins, vi)
            cross = torch.matmul(qi * cd[None, :, :, None], state.to(q.dtype))
            chk(f"chunk{i} inner", inner)
            chk(f"chunk{i} cross", cross)
            wv = vi * vd[None, :, :, None]
            state = chd.float().view(1, -1, 1, 1) * state + torch.matmul(ki.transpose(-1, -2), wv).float()
            chk(f"chunk{i} state(upd)", state)
            if torch.isnan(state.float()).any():
                print(f"  >>> NaN at chunk {i}")
                break


if __name__ == "__main__":
    main()
