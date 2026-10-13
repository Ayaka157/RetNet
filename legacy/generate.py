"""Generate text with a trained RetNet checkpoint using O(1) recurrent inference."""
from __future__ import annotations

import argparse

import torch

from retnet.model import RetNetLM, strip_optimized_prefix
from retnet.tokenizer import BPETokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/retnet_97m_wikitext103/model.pt")
    p.add_argument("--tokenizer", type=str, default="data_cache/wikitext103_v16k/bpe_tokenizer.json")
    p.add_argument("--prompt", type=str, default="The history of the English language begins")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--show_special", action="store_true",
                   help="show <bos>/<eos> in the decoded text")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else \
             ("cpu" if args.device == "auto" else args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RetNetLM(ckpt["config"]).to(device)
    model.load_state_dict(strip_optimized_prefix(ckpt["model_state"]))
    model.eval()
    tokenizer = BPETokenizer.load(args.tokenizer)

    ids = [tokenizer.bos_id] + tokenizer.encode(args.prompt)
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        out = model.generate(input_ids, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             eos_id=tokenizer.eos_id)
    print(tokenizer.decode(out[0].tolist(), skip_special_tokens=not args.show_special))


if __name__ == "__main__":
    main()
