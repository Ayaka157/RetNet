"""Where does a decode step actually spend its time?

``bench/fastgen.py`` assumed the reference decoder's Python overhead dominated.
On the 272M checkpoint the measured speed-up was only 1.08x, so that assumption
is wrong and this script finds out why before the design is committed to.

    python scripts/bench_decode_profile.py --model_dir checkpoints/retnet_272m_rp
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM            # noqa: E402


def sync_time(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--seq_lens", default="1,2,8,64,256")
    ap.add_argument("--batches", default="1,4,16")
    args = ap.parse_args()

    model = RetNetForCausalLM.from_pretrained(args.model_dir).cuda().eval()
    n_blocks = len(model.blocks)
    print(f"model {args.model_dir}  blocks={n_blocks}  params "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")

    print(f"{'B':>3} {'L':>5} | {'parallel ms':>11} {'chunkwise ms':>12} "
          f"{'recurrent ms':>12} | {'GB':>5}")
    print("-" * 62)
    for B in [int(x) for x in args.batches.split(",")]:
        for L in [int(x) for x in args.seq_lens.split(",")]:
            ids = torch.randint(4, 19990, (B, L), device="cuda")
            row = {}
            torch.cuda.reset_peak_memory_stats()
            for mode, kw in (("parallel", {}), ("chunkwise", {"chunk_size": 64}),
                             ("recurrent", {})):
                with torch.inference_mode():
                    if mode == "chunkwise":
                        pad = (-L) % 64
                        x = torch.nn.functional.pad(ids, (0, pad)) if pad else ids
                    else:
                        x = ids
                    try:
                        row[mode] = sync_time(
                            lambda m=mode, xx=x: model._forward_hidden(
                                xx, mode=m, return_state=True, **kw), n=10)
                    except Exception as exc:
                        row[mode] = float("nan")
                        print(f"   {mode} L={L} B={B} failed: {exc}")
            gb = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"{B:>3} {L:>5} | {row['parallel']:>11.2f} "
                  f"{row['chunkwise']:>12.2f} {row['recurrent']:>12.2f} | {gb:>5.2f}")
            del ids
            torch.cuda.empty_cache()

    # decomposition of one recurrent step at B=1, L=1: kernel vs python
    print("\nper-step decomposition (B=1, L=1, recurrent):")
    tok = torch.randint(4, 19990, (1, 1), device="cuda")
    states = [None] * n_blocks
    lp, states = model._forward_hidden(torch.randint(4, 19990, (1, 16),
                                                     device="cuda"),
                                       mode="parallel", return_state=True)
    ms_kernel = sync_time(lambda: model._forward_hidden(tok, mode="recurrent",
                                                        states=states, offset=16,
                                                        return_state=True), n=50)
    print(f"  forward only            {ms_kernel:7.2f} ms")
    ms_item = sync_time(lambda: float(model._forward_hidden(
        tok, mode="recurrent", states=states, offset=16,
        return_state=True)[0][0, -1].max().item()), n=50)
    print(f"  forward + .item() sync  {ms_item:7.2f} ms")
    print(f"  -> python/sync overhead {ms_item - ms_kernel:7.2f} ms "
          f"({(ms_item - ms_kernel) / ms_item * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
