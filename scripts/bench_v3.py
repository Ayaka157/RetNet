"""Capability benchmark v3 — ~1,000 gradable items x N seeds x M checkpoints.

    python scripts/bench_v3.py ^
        --models "rp=checkpoints/retnet_272m_rp,v10=checkpoints/retnet_272m_v10" ^
        --dataset bench/data/bench_v3.jsonl ^
        --tokenizer data_cache/chat_pack_v10/bpe_tokenizer.json ^
        --seeds 42,43,44 --out logs/bench_v3

    # fast smoke run (every Nth item, one seed)
    python scripts/bench_v3.py --models "rp=checkpoints/retnet_272m_rp" ^
        --stride 40 --seeds 42 --out logs/bench_v3_smoke

    # deterministic greedy baseline alongside the sampled run
    python scripts/bench_v3.py --models "rp=checkpoints/retnet_272m_rp" ^
        --temperature 0 --seeds 42 --out logs/bench_v3_greedy

Differences from ``scripts/bench_v2.py`` are structural, not cosmetic:

* the battery is **data** (``bench/data/bench_v3.jsonl``, hashed) rather than
  Python closures, so items can be authored, diffed and audited in bulk;
* every item carries a **difficulty tier** and the report is split by it, so a
  category that has drifted onto the floor is visible rather than averaged away;
* results stream to JSONL and **resume** — a 1,000-item run survives a crash;
* generation uses ``bench.fastgen.fast_generate``, which is asserted
  bit-identical to ``RetNetForCausalLM.generate`` by
  ``tests/test_bench_equivalence.py`` but ~8x faster.
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

from bench.analysis import (compare, discrimination, discrimination_report,  # noqa: E402
                            load_run, markdown_report, summarize)
from bench.registry import CATEGORIES, load_jsonl                       # noqa: E402
from bench.runner import (BatteryRunner, GenCfg, hash_dataset, load_model,  # noqa: E402
                          load_tokenizer, select)

DEFAULT_DATASET = ROOT / "bench" / "data" / "bench_v3.jsonl"


def parse_models(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, path = part.split("=", 1)
        else:
            name, path = Path(part).name, part
        out.append((name.strip(), path.strip()))
    return out


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="name=path,...")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1,
                    help="lockstep group size; 1 = bit-exact (default), "
                         "16 = ~1.5x faster but not bit-identical")
    ap.add_argument("--lc_prefill", default="parallel",
                    choices=["parallel", "chunkwise"],
                    help="prefill mode for 长上下文 prompts (chunkwise pads to 64)")
    ap.add_argument("--only", default=None, help="comma-separated category filter")
    ap.add_argument("--families", default=None, help="comma-separated family filter")
    ap.add_argument("--ids", default=None, help="comma-separated explicit item ids")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth item")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap items per category")
    ap.add_argument("--budget", type=int, default=0,
                    help="覆盖每项的 max_new（0 = 用题库原值）。docs/17 §2 指出"
                         "题库预算只有 16–160，而训练数据 CoT 长度中位是 570 token，"
                         "所以在默认预算下「不闭合」与「不会闭合」根本分不开。"
                         "放大预算后 n_new 不再顶到 max_new，runner 记录的 "
                         "truncated 标志才有区分力。")
    ap.add_argument("--out", default=str(ROOT / "logs" / "bench_v3"))
    ap.add_argument("--tag", default=None, help="run-suffix; default is derived")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"!! dataset not found: {dataset}\n"
              f"   先跑 python scripts/compile_bench.py", file=sys.stderr)
        return 1

    items = load_jsonl(dataset)
    picked = select(items, only=args.only, families=args.families, ids=args.ids,
                    limit=args.limit, stride=args.stride, offset=args.offset)
    if not picked:
        print("!! 过滤后没有题目", file=sys.stderr)
        return 1
    if args.budget:
        old = sorted({it.max_new for it in picked})
        import dataclasses
        try:
            picked = [dataclasses.replace(it, max_new=int(args.budget)) for it in picked]
        except Exception:                                      # noqa: BLE001
            for it in picked:
                it.max_new = int(args.budget)
        print(f"⚠️ 生成预算被覆盖为 {args.budget}（题库原值 {old[0]}–{old[-1]}）；"
              f"与默认预算的结果**不可直接比总分**，只用来判「闭合」")
    ds_hash = hash_dataset(picked)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cfg = GenCfg(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                 repetition_penalty=args.repetition_penalty,
                 no_repeat_ngram_size=args.no_repeat_ngram_size,
                 lc_prefill=args.lc_prefill, batch=args.batch)

    print(f"数据集 {dataset.name}  sha256(选中子集) {ds_hash[:16]}")
    print(f"电池 {len(picked)} 项 / {len(set(i.cat for i in picked))} 类 / "
          f"{len(set(i.family for i in picked))} 族")
    print(f"种子 {seeds}  →  每模型 {len(picked)*len(seeds)} 次生成")
    print(f"解码 T={cfg.temperature} top_k={cfg.top_k} top_p={cfg.top_p} "
          f"rep={cfg.repetition_penalty} ngram={cfg.no_repeat_ngram_size} "
          f"batch={cfg.batch}"
          + ("" if cfg.batch == 1 else "  (⚠ 非逐位一致)"))
    per_cat = {}
    for i in picked:
        per_cat[i.cat] = per_cat.get(i.cat, 0) + 1
    print("  " + "  ".join(f"{c}={per_cat[c]}" for c in CATEGORIES if c in per_cat))

    tok = load_tokenizer(args.tokenizer)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (f"T{cfg.temperature:g}_s{len(seeds)}"
                       + (f"_stride{args.stride}" if args.stride > 1 else ""))

    runs = {}
    for name, path in parse_models(args.models):
        path_p = Path(path) if Path(path).is_absolute() else ROOT / path
        print(f"\n{'='*74}\n[{name}] {path_p}\n{'='*74}", flush=True)
        model, n_param = load_model(path_p, device=args.device)
        print(f"[{name}] params = {n_param/1e6:.2f}M   dtype = "
              f"{next(model.parameters()).dtype}", flush=True)

        out_file = outdir / f"{name}.{tag}.jsonl"
        if args.no_resume and out_file.exists():
            out_file.unlink()
            print(f"[{name}] --no-resume: 清空 {out_file.name}")

        runner = BatteryRunner(model, tok, picked, cfg, device=args.device)

        def progress(done, total, elapsed, per_item):
            eta = per_item * (total - done)
            print(f"  [{name}] {done}/{total}  {elapsed:6.0f}s  "
                  f"{per_item*1000:5.0f} ms/样本  eta {eta/60:5.1f} min",
                  flush=True)

        t0 = time.time()
        runner.run(seeds, out_file, progress=progress)
        dt = time.time() - t0
        recs = load_run(out_file)
        s = summarize(recs, name)
        runs[name] = recs
        print(f"[{name}] 加权总分 {s['total']:.4f} "
              f"(CI {s['total_ci'][0]:.3f}–{s['total_ci'][1]:.3f})  "
              f"think泄漏 {s['leak']*100:.1f}%  退化重复 {s['degenerate']*100:.1f}%  "
              f"本轮 {dt:.0f}s", flush=True)
        for c in CATEGORIES:
            if c in s["cats"]:
                b = s["cats"][c]
                print(f"    {c:<6} {b['score']:.3f}  "
                      f"E={b['tiers']['E']:.2f} M={b['tiers']['M']:.2f} "
                      f"H={b['tiers']['H']:.2f}  n={b['n_items']}", flush=True)

        (outdir / f"{name}.{tag}.summary.json").write_text(
            json.dumps({k: v for k, v in s.items() if k != "per_item"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- cross-model report
    names = list(runs)
    summaries = [summarize(runs[n], n) for n in names]
    disc = None
    if len(names) > 1 and all(runs[n] for n in names):
        d = discrimination(runs)
        disc = discrimination_report(d, names)
    comps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if runs[names[i]] and runs[names[j]]:
                c = compare(names[i], runs[names[i]], names[j], runs[names[j]])
                c["p"] = round(c["p"], 4)
                comps.append(c)

    title = f"能力基准 v3 · {len(picked)} 项 × {len(seeds)} 种子"
    md = markdown_report(summaries, disc, comps, title=title)
    (outdir / f"summary.{tag}.md").write_text(md, encoding="utf-8")

    payload = {"tag": tag, "dataset": dataset.name, "dataset_sha256": ds_hash,
               "n_items": len(picked), "seeds": seeds,
               "decode": {"batch": cfg.batch, "bit_exact": cfg.batch == 1,
                          "temperature": cfg.temperature, "top_k": cfg.top_k,
                          "top_p": cfg.top_p,
                          "repetition_penalty": cfg.repetition_penalty,
                          "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
                          "lc_prefill": cfg.lc_prefill},
               "models": {n: {k: v for k, v in summarize(runs[n], n).items()
                              if k != "per_item"} for n in names},
               "discrimination": disc, "comparisons": comps}
    (outdir / f"report.{tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + md)
    print(f"\n→ {outdir / f'summary.{tag}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
