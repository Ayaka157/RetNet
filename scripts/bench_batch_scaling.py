"""Does lockstep batching actually buy throughput on this model?

``scripts/bench_decode_profile.py`` measured a decode step at ~22–25 ms for
batch 1, 4 *and* 16, which is suspicious: if a step costs the same regardless of
batch size, lockstep decoding should be ~N times faster, and if it scales with N
there is nothing to gain.

The profile only timed the raw recurrent forward.  This times the whole runner
on a fixed slice of the battery at several batch sizes, which is the number that
decides whether the extra complexity (and the numerical caveat) is worth it.

    python scripts/bench_batch_scaling.py --items 48
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.registry import load_jsonl                            # noqa: E402
from bench.runner import BatteryRunner, GenCfg, load_model, load_tokenizer  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / "checkpoints" / "retnet_272m_rp"))
    ap.add_argument("--tokenizer",
                    default=str(ROOT / "data_cache" / "chat_pack_v10" / "bpe_tokenizer.json"))
    ap.add_argument("--dataset", default=str(ROOT / "bench" / "data" / "bench_v3.jsonl"))
    ap.add_argument("--items", type=int, default=48)
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = load_jsonl(pathlib.Path(args.dataset))[:args.items]
    tok = load_tokenizer(args.tokenizer)
    model, _ = load_model(args.model_dir)
    print(f"{len(items)} items, seed {args.seed}\n")
    print(f"{'batch':>6} {'seconds':>9} {'s/item':>8} {'speedup':>8} {'tokens':>8}")
    print("-" * 44)
    base = None
    for B in [int(x) for x in args.batches.split(",")]:
        cfg = GenCfg(batch=B)
        runner = BatteryRunner(model, tok, items, cfg)
        t0 = time.time()
        recs = runner.run_group(items, args.seed) if B > 1 else \
            [runner.run_item(it, args.seed) for it in items]
        dt = time.time() - t0
        ntok = sum(r["n_new"] for r in recs)
        if base is None:
            base = dt
        print(f"{B:>6} {dt:>9.1f} {dt/len(items):>8.2f} {base/dt:>7.2f}x {ntok:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
