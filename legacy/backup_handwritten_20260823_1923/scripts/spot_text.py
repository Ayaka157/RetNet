"""Plain-text continuation spot check (CPU) — measures what continued pretraining
actually bought: fluency, knowledge density, cross-domain coverage.

    python scripts/spot_text.py --checkpoint ... --tokenizer ... --tag NAME
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.model import RetNetLM, strip_optimized_prefix  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402

PROMPTS = [
    ("中文科普", "人工智能是一种让机器"),
    ("中文叙事", "那是一个寒冷的冬夜，他独自走在"),
    ("英文", "The history of the English language begins"),
    ("物理", "牛顿第二定律指出，物体的加速度"),
    ("历史", "唐朝是中国历史上"),
    ("健康", "长期熬夜对身体的影响，主要体现在"),
]


def main():
    fix_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tag", default="text")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RetNetLM(ck["config"]).to(torch.device("cpu"))
    model.load_state_dict(strip_optimized_prefix(ck.get("model_state", ck)))
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)
    step = ck.get("step", "?")
    t0 = time.time()

    print(f"===== {args.tag} (step={step}) =====")
    for label, prompt in PROMPTS:
        ids = [tok.bos_id] + tok.encode(prompt)
        x = torch.tensor([ids], device="cpu")
        with torch.inference_mode():
            out = model.generate(x, max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 top_p=args.top_p,
                                 repetition_penalty=args.repetition_penalty,
                                 no_repeat_ngram_size=4, eos_id=tok.eos_id)
        text = tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True)
        text = text.split("\n")[0].strip()
        print(f"[{label}] {prompt}…")
        print(f"  → {text[:200]}")
        print("", flush=True)
    print(f"(elapsed {time.time() - t0:.0f}s)", flush=True)


def fix_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


if __name__ == "__main__":
    main()
