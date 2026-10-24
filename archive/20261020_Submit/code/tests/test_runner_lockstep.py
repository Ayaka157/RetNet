"""The property the whole throughput story rests on, on real weights.

``tests/test_bench_equivalence.py`` proves the batched decoder equals the
single-item path on a toy model.  This checks the *runner*: an item decoded
inside a lockstep group of 16 must get exactly the reply it gets when decoded
alone, on the production 272M checkpoint, for a real slice of the battery —
including multi-turn items, greedy items and long-prompt items.

If this fails, batching is buying speed by measuring a different model.

    python tests/test_runner_lockstep.py --stride 97
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.registry import load_jsonl                          # noqa: E402
from bench.runner import (BatteryRunner, GenCfg, load_model,   # noqa: E402
                          load_tokenizer)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / "checkpoints" / "retnet_272m_rp"))
    ap.add_argument("--tokenizer",
                    default=str(ROOT / "data_cache" / "chat_pack_v10" / "bpe_tokenizer.json"))
    ap.add_argument("--dataset", default=str(ROOT / "bench" / "data" / "bench_v3.jsonl"))
    ap.add_argument("--stride", type=int, default=97)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    items = load_jsonl(pathlib.Path(args.dataset))[::args.stride]
    print(f"sanity slice: {len(items)} items "
          f"({len({i.cat for i in items})} categories, "
          f"{sum(1 for i in items if i.is_multi_turn)} multi-turn)")

    tok = load_tokenizer(args.tokenizer)
    model, _ = load_model(args.model_dir)

    solo = BatteryRunner(model, tok, items, GenCfg(batch=1))
    group = BatteryRunner(model, tok, items, GenCfg(batch=args.batch))

    solo_recs = [solo.run_item(it, args.seed) for it in items]
    lock_recs = group.run_group(items, args.seed)

    by_id = {r["id"]: r for r in lock_recs}
    bad = []
    for s in solo_recs:
        l = by_id.get(s["id"])
        if l is None:
            bad.append(f"{s['id']}: missing from the lockstep run")
            continue
        if s["reply"] != l["reply"]:
            bad.append(f"{s['id']}: reply differs\n"
                       f"      solo   {s['reply'][:90]!r}\n"
                       f"      lock   {l['reply'][:90]!r}")
        elif bool(s["pass"]) != bool(l["pass"]):
            bad.append(f"{s['id']}: same reply but pass flips "
                       f"{s['pass']} vs {l['pass']}")

    print(f"\ncompared {len(solo_recs)} items, solo vs lockstep batch={args.batch}")
    if bad:
        print(f"{len(bad)} MISMATCHES:")
        for b in bad[:12]:
            print(f"   - {b}")
        return 1
    print("PASS: lockstep replies are identical to single-item replies "
          "(reply text and pass/fail)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
