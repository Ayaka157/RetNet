"""Engineering benchmark: numerical equivalence and throughput.

The capability battery says whether a model is any good.  This says whether the
*implementation* can be trusted and how fast it is.  docs/11_TileKernels calls
the first section the gate that must be cleared before touching any kernel:
the repo's own equivalence test asserts < 1e-4 but only at d_model=64 / L=16.

    python scripts/bench_engine.py --model_dir checkpoints/retnet_272m_v10 ^
        --tokenizer data_cache/chat_pack_v10/bpe_tokenizer.json ^
        --out logs/bench/engine_272m_v10.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402


def xentropy(logits: torch.Tensor, ids: torch.Tensor) -> float:
    """Cross-entropy of the same tokens under a given logits tensor."""
    lo = logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = torch.log_softmax(lo, dim=-1)
    return float(-lp.gather(-1, tgt.unsqueeze(-1)).mean())


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--seq_lens", default="256,512,1024")
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--decode_tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(42)
    model = RetNetForCausalLM.from_pretrained(args.model_dir)
    model = model.cuda().eval()
    n_param = sum(p.numel() for p in model.parameters())
    print(f"params = {n_param/1e6:.2f}M  dtype = {next(model.parameters()).dtype}")
    torch.cuda.reset_peak_memory_stats()

    report = {"model_dir": args.model_dir, "params_M": n_param / 1e6, "equiv": [],
              "decode": {}, "peak_vram_MB": 0.0}

    for L in [int(x) for x in args.seq_lens.split(",")]:
        ids = torch.randint(0, 20000, (1, L), device="cuda")
        row = {"seq_len": L}

        with torch.inference_mode():
            t0 = time.time()
            lp = model(ids, mode="parallel").logits.float()
            row["parallel_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            lc = model(ids, mode="chunkwise", chunk_size=args.chunk_size).logits.float()
            row["chunkwise_ms"] = (time.time() - t0) * 1000

        d = (lc - lp).abs()
        denom = lp.abs().clamp_min(1e-6)
        rel = (d / denom).flatten()
        row.update({
            "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "rel_p50": float(torch.quantile(rel[: min(rel.numel(), 2_000_000)], 0.5)),
            "rel_p90": float(torch.quantile(rel[: min(rel.numel(), 2_000_000)], 0.9)),
            "xent_parallel": xentropy(lp, ids),
            "xent_chunkwise": xentropy(lc, ids),
        })
        del lp, lc, d, rel
        torch.cuda.empty_cache()
        report["equiv"].append(row)
        print(f"L={L:5d}  max|diff|={row['max_abs_diff']:.6g}  "
              f"mean|diff|={row['mean_abs_diff']:.6g}  "
              f"rel_p90={row['rel_p90']:.4g}  "
              f"xent {row['xent_parallel']:.4f} vs {row['xent_chunkwise']:.4f}  "
              f"| {row['parallel_ms']:.0f}ms / {row['chunkwise_ms']:.0f}ms", flush=True)

    # ---- decode throughput (recurrent, the O(1) path)
    ids = torch.randint(0, 20000, (1, 32), device="cuda")
    with torch.inference_mode():
        model.generate(ids, max_new_tokens=4, temperature=0.0, eos_id=None)
        torch.cuda.synchronize()
        t0 = time.time()
        model.generate(ids, max_new_tokens=args.decode_tokens, temperature=0.0, eos_id=None)
        torch.cuda.synchronize()
        dt = time.time() - t0
    report["decode"] = {"tokens": args.decode_tokens, "seconds": dt,
                        "tok_per_s": args.decode_tokens / dt}
    print(f"decode: {args.decode_tokens} tok in {dt:.2f}s = "
          f"{args.decode_tokens/dt:.1f} tok/s", flush=True)

    report["peak_vram_MB"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"peak VRAM allocated = {report['peak_vram_MB']:.0f} MB")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved -> {p}")


if __name__ == "__main__":
    main()
