"""Compare bf16 vs fp16 training stability on the SAME model/data/seed/steps.

Decides whether the T4 (no bf16) cloud route is viable for RetNet-272M:
  - fp16: risk = retention KV accumulation overflow (>65504) & grad spikes
  - bf16: our current baseline (local RTX 3070)

Runs 3 separate trainer runs: bf16 (ref), fp16 (dup of ref).  Compare loss
curves: if fp16 loss tracks bf16 within noise (no NaN, no divergence),
T4 route is viable.  If fp16 diverges/NaNs, we need GLA/retention fp32
keep or grad_scaler tuning before cloud training.

    python scripts/precision_test.py --model_dir checkpoints/retnet_272m_rp \
        --data_dir data_cache/chat_pack_v9 --out_root logs/precision_test
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.train import main as train_main  # noqa: E402

# train.py's main() reads sys.argv; emulate by passing explicit argv list.
# We re-invoke train_main with patched sys.argv per precision.


def run(model_dir, data_dir, out_root, precision, max_steps, seed):
    out = Path(out_root) / f"{precision}_seed{seed}"
    if out.exists():
        shutil.rmtree(out)
    old = sys.argv
    sys.argv = [
        "train.py",
        "--model_dir", str(model_dir),
        "--data_dir", str(data_dir),
        "--output_dir", str(out),
        "--max_steps", str(max_steps),
        "--batch_size", "2",
        "--lr", "1.1e-4",
        "--warmup", "10",
        "--no-compile",
        "--mode", "chunkwise",
        "--grad_checkpoint",
        "--precision", precision,
        "--seed", str(seed),
    ]
    try:
        train_main()
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="checkpoints/retnet_272m_rp")
    ap.add_argument("--data_dir", default="data_cache/chat_pack_v9")
    ap.add_argument("--out_root", default="logs/precision_test")
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--seeds", default="1234,1235")
    ap.add_argument("--precision_list", default="bf16,fp16",
                    help="comma-separated precisions to compare")
    args = ap.parse_args()

    precisions = args.precision_list.split(",")
    for prec in precisions:
        for seed in args.seeds.split(","):
            run(args.model_dir, args.data_dir, args.out_root, prec,
                args.max_steps, int(seed))
            print(f"DONE {prec} seed={seed}", flush=True)

    # dump comparison
    print()
    print("=== loss 对比（text, step=loss） ===")
    import json
    for prec in precisions:
        for seed in args.seeds.split(","):
            p = (Path(args.out_root) / f"{prec}_seed{seed}" / "trainer_state.json")
            if not p.exists():
                print(f"{prec} seed={seed}: NO STATE", flush=True)
                continue
            s = json.loads(p.read_text(encoding="utf-8"))
            hist = [e for e in s["log_history"] if "loss" in e]
            seq = " ".join(f"{e['step']}:{e['loss']:.3f}" for e in hist)
            print(f"{prec} seed={seed}: {seq}", flush=True)


if __name__ == "__main__":
    main()
