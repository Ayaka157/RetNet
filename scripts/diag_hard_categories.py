"""Is the 长上下文 zero a format failure or a genuine recall failure?

50/50 items scored 0.  Two very different explanations produce that number:

* **format** — the model has never seen a 1,200-token user turn followed by a
  question, so it emits a continuation of the document and never answers;
* **recall** — it reads the question fine and cannot retrieve the fact.

The control that separates them: put the **same question and the same material
in one turn** (which is the shape the model was trained on) and compare.  If the
single-turn control also scores 0, the two-turn structure is not the problem and
the number is honest.  This also prints the cloze and instruction-following
controls, for the same reason.

    python scripts/diag_hard_categories.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch                                                       # noqa: E402

from bench.checkers import Ctx                                     # noqa: E402
from bench.fastgen import fast_generate, make_rng_seed             # noqa: E402
from bench.registry import load_jsonl                              # noqa: E402
from bench.runner import cut, load_model, load_tokenizer           # noqa: E402
from retnet.chat import encode_prompt                              # noqa: E402


def decode(model, tok, turns, system=None, max_new=48, temp=0.0, seed=42,
           item_id="x", device="cuda"):
    history = []
    reply = ""
    for t in turns:
        ids = encode_prompt(tok, history, t, system=system)
        x = torch.tensor([ids], device=device)
        out = fast_generate(model, x, max_new_tokens=max_new, temperature=temp,
                            top_k=None if temp <= 0 else 40,
                            top_p=None if temp <= 0 else 0.9,
                            repetition_penalty=1.0 if temp <= 0 else 1.1,
                            no_repeat_ngram_size=4, eos_id=tok.eos_id,
                            rng_seed=make_rng_seed(seed, item_id),
                            device=device)
        reply = cut(tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True))
        history.append((t, reply))
    return reply


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
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    items = {i.id: i for i in load_jsonl(pathlib.Path(args.dataset))}
    tok = load_tokenizer(args.tokenizer)
    model, _ = load_model(args.model_dir)

    lc = [i for i in items.values() if i.cat == "长上下文"][:args.n]
    print("=== 长上下文：两轮（原题） vs 单轮（对照） ===")
    for it in lc:
        material, question = it.turns[0], it.turns[1]
        orig = decode(model, tok, [material, question], max_new=it.max_new,
                      seed=42, item_id=it.id)
        merged = decode(model, tok, [material + "\n\n问题：" + question],
                        max_new=it.max_new, seed=42, item_id=it.id)
        ctx_o = Ctx(reply=orig, prompt=question, users=it.turns)
        ctx_m = Ctx(reply=merged, prompt=question, users=it.turns)
        print(f"  {it.id} [{it.tier}] 材料 {len(material)} 字")
        print(f"     两轮 {int(bool(it.checker()(ctx_o)))} {orig[:76]!r}")
        print(f"     单轮 {int(bool(it.checker()(ctx_m)))} {merged[:76]!r}")

    print("\n=== 填空：贪婪 vs 采样 ===")
    for it in [i for i in items.values() if i.cat == "填空"][:args.n]:
        g = decode(model, tok, it.turns, max_new=it.max_new, temp=0.0,
                   seed=42, item_id=it.id)
        s = decode(model, tok, it.turns, max_new=it.max_new, temp=0.7,
                   seed=42, item_id=it.id)
        cg = Ctx(reply=g, prompt=it.turns[-1], users=it.turns)
        cs = Ctx(reply=s, prompt=it.turns[-1], users=it.turns)
        print(f"  {it.id:<26} [{it.tier}] {it.turns[0][:26]!r}")
        print(f"     贪婪 {int(bool(it.checker()(cg)))} {g[:60]!r}")
        print(f"     采样 {int(bool(it.checker()(cs)))} {s[:60]!r}")

    print("\n=== 指令遵循：贪婪 ===")
    for it in [i for i in items.values() if i.cat == "指令遵循"][:args.n]:
        g = decode(model, tok, it.turns, max_new=it.max_new, temp=0.0,
                   seed=42, item_id=it.id)
        cg = Ctx(reply=g, prompt=it.turns[-1], users=it.turns)
        print(f"  {it.id:<24} [{it.tier}] {int(bool(it.checker()(cg)))} "
              f"{it.turns[0][:44]!r} -> {g[:56]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
