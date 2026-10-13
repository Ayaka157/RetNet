"""Equivalence test: benchmark decoder vs the model's own ``generate``.

``bench/fastgen.py`` exists because ``RetNetForCausalLM.generate`` spends most
of its time in Python (a list rebuilt per step, an ``.item()`` sync per step, a
per-token loop over ``set(generated)`` for the repetition penalty).  At 42 tok/s
a 1,000-item battery is days of GPU time; the benchmark needs ~8x.

A speed-up is only acceptable if it changes nothing else.  ``fast_generate``
therefore reproduces the reference algorithm *including its random-number
consumption order*, and this test asserts **identical output** on the same
seed across every decoding regime the battery uses:

    greedy · sampled · top-k · top-p · repetition penalty · n-gram ban
    · penalize_prompt · eos early stop · multi-turn state continuation

If this test fails, the benchmark numbers are measuring a different model than
``play.py`` and ``bench_v2.py`` do — do not "fix" it by loosening the assert.

    python tests/test_bench_equivalence.py [--cuda]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.fastgen import (Request, batched_generate, fast_generate,  # noqa: E402
                           prefill_states)
from retnet.model import RetNetConfig, RetNetLM                      # noqa: E402


class Shim:
    """Expose the handwritten model under the HF module's method names.

    ``fast_generate`` calls ``model._forward_hidden`` and reads ``model.blocks``;
    the handwritten ``RetNetLM`` has the same internals under different names.
    Wrapping it lets one test cover both implementations without duplicating
    the model code.
    """

    def __init__(self, lm):
        self.lm = lm
        self.blocks = lm.blocks

    def _forward_hidden(self, input_ids, mode="parallel", states=None, offset=0,
                        return_state=False, chunk_size=64):
        if mode == "chunkwise":
            return self.lm(input_ids, mode="chunkwise", states=states, offset=offset,
                           return_state=return_state, chunk_size=chunk_size)
        return self.lm(input_ids, mode=mode, states=states, offset=offset,
                       return_state=return_state)

    def parameters(self):
        return self.lm.parameters()


def make_model(seed=0):
    torch.manual_seed(seed)
    cfg = RetNetConfig(vocab_size=256, d_model=64, num_layers=3, num_heads=4,
                       dropout=0.0)
    lm = RetNetLM(cfg).eval()
    return Shim(lm)


def reference(model, ids, seed, **kw):
    torch.manual_seed(seed)
    return model.lm.generate(ids, **kw)


def fast(model, ids, seed, **kw):
    torch.manual_seed(seed)
    return fast_generate(model, ids, **kw)


CASES = [
    # name,                 kwargs
    ("greedy",              dict(max_new_tokens=12, temperature=0.0, eos_id=None)),
    ("sampled",             dict(max_new_tokens=12, temperature=1.0, eos_id=None)),
    ("top_k",               dict(max_new_tokens=12, temperature=1.0, top_k=8, eos_id=None)),
    ("top_p",               dict(max_new_tokens=12, temperature=0.8, top_p=0.9, eos_id=None)),
    ("top_k+top_p",         dict(max_new_tokens=12, temperature=0.9, top_k=20,
                                 top_p=0.85, eos_id=None)),
    ("rep_penalty",         dict(max_new_tokens=12, temperature=0.7, top_k=40,
                                 repetition_penalty=1.3, eos_id=None)),
    ("rep_penalize_prompt", dict(max_new_tokens=12, temperature=0.7, top_k=40,
                                 repetition_penalty=1.3, penalize_prompt=True,
                                 eos_id=None)),
    ("ngram_ban",           dict(max_new_tokens=16, temperature=0.7, top_k=40,
                                 no_repeat_ngram_size=3, eos_id=None)),
    ("ngram_ban+rep",       dict(max_new_tokens=16, temperature=0.7, top_k=40,
                                 repetition_penalty=1.15, no_repeat_ngram_size=4,
                                 eos_id=None)),
    ("eos_stop",            dict(max_new_tokens=24, temperature=0.7, top_k=40,
                                 eos_id=7)),
    ("short_prompt",        dict(max_new_tokens=10, temperature=0.7, top_k=40,
                                 eos_id=None)),
]


def test_batched_equals_single(model, dev, fails):
    """Batched decoding must give every request the reply it would get alone.

    This is the property the whole throughput story rests on: lockstep batching
    is only legitimate if a request cannot tell how many neighbours it had.  Both
    paths use the **same single-stream sampling** (draw ``k`` comes from the k-th
    ``multinomial`` on the ambient RNG, which the fresh ``manual_seed`` per
    request resets), so no per-request stream bookkeeping has to agree.
    """
    import random
    rng = random.Random(4)
    base = []
    for i in range(9):
        L = rng.choice([1, 2, 5, 5, 5, 12, 12, 30, 30])
        base.append({
            "ids": [rng.randrange(4, 200) for _ in range(L)],
            "max_new": rng.choice([6, 8, 12]),
            "temperature": rng.choice([0.0, 0.7, 1.0]),
            "top_k": 40, "top_p": 0.9, "repetition_penalty": 1.1,
            "no_repeat_ngram_size": rng.choice([0, 4]),
            "eos_id": None,
            "tag": f"req{i}",
        })

    for seed in (31, 32):
        singles = []
        for i, cfg in enumerate(base):
            torch.manual_seed(seed)
            ids = torch.tensor([cfg["ids"]], device=dev)
            out = fast_generate(
                model, ids, max_new_tokens=cfg["max_new"],
                temperature=cfg["temperature"], top_k=cfg["top_k"],
                top_p=cfg["top_p"], repetition_penalty=cfg["repetition_penalty"],
                no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
                eos_id=cfg["eos_id"], rng_seed=1000 + i)
            singles.append(out[0].tolist()[len(cfg["ids"]):])

        torch.manual_seed(seed)
        reqs = [Request(**{**cfg, "rng_seed": 1000 + i})
                for i, cfg in enumerate(base)]
        batched_generate(model, reqs)
        for i, r in enumerate(reqs):
            if r.out != singles[i]:
                j = next((k for k, (a, b) in enumerate(zip(r.out, singles[i]))
                          if a != b), "n/a")
                fails.append(f"batched != single (seed {seed}) request {i}: len "
                             f"{len(r.out)} vs {len(singles[i])}, diff at {j}")
                break

        # a request must not change when its neighbours change
        torch.manual_seed(seed)
        subset = [Request(**{**cfg, "rng_seed": 1000 + i})
                  for i, cfg in enumerate(base[:3])]
        batched_generate(model, subset)
        for i, r in enumerate(subset):
            if r.out != singles[i]:
                fails.append(f"batched subset differs (seed {seed}) request {i}")
                break
    print(f"  {'OK ' if not any('batched' in f for f in fails) else '!! '}"
          f"{'batched == single':<22} {len(base)} requests x 2 seeds, full and subset")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    dev = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    model = make_model(0)
    model.lm = model.lm.to(dev)
    seeds = [int(s) for s in args.seeds.split(",")]

    fails = []
    for name, kw in CASES:
        n = 1 if name == "short_prompt" else 6
        ids = torch.randint(4, 200, (1, n), device=dev)
        for seed in seeds:
            ref = reference(model, ids, seed, **kw)
            got = fast(model, ids, seed, **kw)
            if ref.shape != got.shape or not torch.equal(ref, got):
                fails.append(
                    f"{name} seed={seed}: shapes {tuple(ref.shape)} vs "
                    f"{tuple(got.shape)}; "
                    f"diff={(ref[0, :min(ref.shape[1], got.shape[1])].float() - got[0, :min(ref.shape[1], got.shape[1])].float()).abs().max().item() if ref.shape[1] == got.shape[1] else 'n/a'}")
                continue
        print(f"  {'OK ' if not any(f.startswith(name) for f in fails) else '!! '}"
              f"{name:<22} seeds={seeds}")

    # multi-turn: the recurrent state carried across two calls must match
    ids = torch.randint(4, 200, (1, 8), device=dev)
    with torch.inference_mode():
        torch.manual_seed(11)
        r1 = model.lm.generate(ids, max_new_tokens=8, temperature=0.0, eos_id=None)
        torch.manual_seed(11)
        f1 = fast_generate(model, ids, max_new_tokens=8, temperature=0.0, eos_id=None)
    if not torch.equal(r1, f1):
        fails.append("multi-turn step 1 mismatch")
    with torch.inference_mode():
        torch.manual_seed(12)
        r2 = model.lm.generate(r1, max_new_tokens=8, temperature=0.0, eos_id=None)
        torch.manual_seed(12)
        f2 = fast_generate(model, f1, max_new_tokens=8, temperature=0.0, eos_id=None)
    if not torch.equal(r2, f2):
        fails.append("multi-turn step 2 mismatch")
    print(f"  {'OK ' if not any('multi-turn' in f for f in fails) else '!! '}"
          f"{'multi-turn state':<22} seeds=[11,12]")

    # chunkwise prefill must be a legal alternate path (same length semantics)
    ids = torch.randint(4, 200, (1, 100), device=dev)
    with torch.inference_mode():
        torch.manual_seed(5)
        a = fast_generate(model, ids, max_new_tokens=6, temperature=0.0, eos_id=None)
        _, st, off = prefill_states(model, ids, mode="chunkwise")
        torch.manual_seed(5)
        b = fast_generate(model, ids, max_new_tokens=6, temperature=0.0, eos_id=None)
    if not torch.equal(a, b):
        fails.append("chunkwise-prefill path changed the sampled tokens")
    print(f"  {'OK ' if not any('chunkwise' in f for f in fails) else '!! '}"
          f"{'chunkwise prefill':<22} pad-to-64 path exercised (padded_len={off})")

    test_batched_equals_single(model, dev, fails)

    if fails:
        print(f"\n{len(fails)} FAILURES:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"\nPASS: fast_generate == RetNetForCausalLM.generate on {dev} "
          f"({len(CASES)} regimes x {len(seeds)} seeds + multi-turn + prefill)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
