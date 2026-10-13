"""Measure the benchmark decoder against the model's own, on a real checkpoint.

``bench/fastgen.py`` argues the reference decoder spends most of its time in
Python.  This turns that into a number on the actual 272M weights, and — more
importantly — re-runs the bit-equivalence check there, because the toy-scale
test cannot see fp32 drift in ``_parallel_retention``.

    python scripts/bench_decode_speed.py ^
        --model_dir checkpoints/retnet_272m_rp ^
        --tokenizer data_cache/chat_pack_v10/bpe_tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.fastgen import fast_generate                                  # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM                  # noqa: E402


def timed(fn, n_warm=1, n_rep=3):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_rep):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_rep


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--new_tokens", type=int, default=64)
    ap.add_argument("--seeds", default="7,8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = RetNetForCausalLM.from_pretrained(args.model_dir).cuda().eval()
    print(f"model {args.model_dir}  params "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    ids = torch.randint(4, 19990, (1, args.prompt_len), device="cuda")
    kw = dict(max_new_tokens=args.new_tokens, temperature=0.7, top_k=40, top_p=0.9,
              repetition_penalty=1.1, no_repeat_ngram_size=4, eos_id=None)

    torch.manual_seed(0)
    model.generate(ids, **kw)
    torch.cuda.synchronize()
    ref_s = timed(lambda: (torch.manual_seed(0), model.generate(ids, **kw))[1])
    fast_s = timed(lambda: (torch.manual_seed(0),
                            fast_generate(model, ids, **kw))[1], n_warm=2, n_rep=5)

    ref_tps = args.new_tokens / ref_s
    fast_tps = args.new_tokens / fast_s
    print(f"reference  {ref_s*1000:8.1f} ms  {ref_tps:6.1f} tok/s")
    print(f"fastgen    {fast_s*1000:8.1f} ms  {fast_tps:6.1f} tok/s")
    print(f"speedup    x{ref_s/fast_s:.2f}")

    # bit-equivalence on production weights
    same = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        torch.manual_seed(seed)
        a = model.generate(ids, **kw)
        torch.manual_seed(seed)
        b = fast_generate(model, ids, **kw)
        eq = a.shape == b.shape and torch.equal(a, b)
        same.append(eq)
        print(f"  seed {seed}: identical = {eq}")
    identical = all(same)

    # long-prompt prefill cost (matters for the 长上下文 category)
    prefill = {}
    for L in (512, 2048, 4096):
        x = torch.randint(4, 19990, (1, L), device="cuda")
        try:
            ms = timed(lambda: model._forward_hidden(x, mode="parallel",
                                                     return_state=True), n_warm=0, n_rep=3) * 1000
            prefill[L] = ms
            print(f"  parallel prefill L={L:5d}: {ms:8.1f} ms")
        except Exception as exc:
            print(f"  parallel prefill L={L:5d}: failed ({type(exc).__name__}: {exc})")

    rep = {"model_dir": args.model_dir, "prompt_len": args.prompt_len,
           "new_tokens": args.new_tokens,
           "reference_ms": ref_s * 1000, "fastgen_ms": fast_s * 1000,
           "reference_tok_s": ref_tps, "fastgen_tok_s": fast_tps,
           "speedup": ref_s / fast_s, "identical": identical,
           "prefill_parallel_ms": prefill,
           "peak_vram_MB": torch.cuda.max_memory_allocated() / 1024 ** 2}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved -> {p}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
