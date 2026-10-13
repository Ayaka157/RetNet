"""Re-analyse finished runs without touching the GPU.

``scripts/bench_v3.py`` writes one JSONL per (model, tag).  This script reads
any set of those files back and rebuilds the report — useful when a run was
produced on the training machine and copied over, when a checkpoint is added
after the fact, or when the aggregation logic changes and old results need
re-scoring.

    python scripts/bench_report.py --runs "rp=logs/bench_v3/rp.*.jsonl,v10=..." ^
        --out logs/bench_v3

    # every *.jsonl in a directory, grouped by the name before the first dot
    python scripts/bench_report.py --dir logs/bench_v3 --out logs/bench_v3

    # show what the battery measures per family, for auditing item quality
    python scripts/bench_report.py --dir logs/bench_v3 --families
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.analysis import (compare, discrimination, discrimination_report,  # noqa: E402
                            load_run, markdown_report, summarize)


def parse_runs(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--runs 需要 name=glob 形式，收到 {part!r}")
        name, glob = part.split("=", 1)
        files = sorted(Path().glob(glob.strip())) if not Path(glob.strip()).is_absolute() \
            else sorted(Path(glob.strip()).parent.glob(Path(glob.strip()).name))
        files = [f for f in files if f.suffix == ".jsonl"]
        if not files:
            print(f"~~ {name}: glob 没匹配到文件 ({glob})", file=sys.stderr)
            continue
        recs = []
        for f in files:
            recs.extend(load_run(f))
        out.append((name, recs, [f.name for f in files]))
    return out


def parse_dir(d: Path, exclude: set):
    groups = {}
    for f in sorted(d.glob("*.jsonl")):
        if f.name in exclude:
            continue
        name = f.name.split(".")[0]
        groups.setdefault(name, []).append(f)
    return [(n, [r for f in fs for r in load_run(f)], [f.name for f in fs])
            for n, fs in sorted(groups.items())]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="name=glob,...")
    ap.add_argument("--dir", default=None, help="directory of *.jsonl")
    ap.add_argument("--out", default=str(ROOT / "logs" / "bench_v3"))
    ap.add_argument("--title", default=None)
    ap.add_argument("--families", action="store_true",
                    help="print the per-family breakdown table")
    ap.add_argument("--items", default=None,
                    help="print per-item pass rates for a category")
    ap.add_argument("--tag", default="report")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.runs and not args.dir:
        print("需要 --runs 或 --dir", file=sys.stderr)
        return 1

    groups = []
    if args.runs:
        groups += parse_runs(args.runs)
    if args.dir:
        d = Path(args.dir)
        if not d.is_absolute():
            d = ROOT / args.dir
        groups += parse_dir(d, exclude=set())

    # collapse duplicate names
    merged = {}
    for name, recs, files in groups:
        if name in merged:
            merged[name][0].extend(recs)
            merged[name][1].extend(files)
        else:
            merged[name] = [list(recs), list(files)]
    runs = {n: v[0] for n, v in merged.items()}
    if not runs:
        print("没有可分析的结果", file=sys.stderr)
        return 1

    for n, v in merged.items():
        seen = {}
        for r in v[0]:
            seen[(r["id"], r["seed"])] = r
        print(f"[{n}] {len(seen)} 个 (item,seed) 单元 来自 {len(v[1])} 个文件")

    summaries = [summarize(runs[n], n) for n in sorted(runs)]
    disc = None
    if len(runs) > 1:
        disc = discrimination_report(discrimination(runs), sorted(runs))
    comps = []
    names = sorted(runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            comps.append(compare(names[i], runs[names[i]], names[j], runs[names[j]]))

    md = markdown_report(summaries, disc, comps,
                         title=args.title or "能力基准 v3")
    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"summary.{args.tag}.md"
    p.write_text(md, encoding="utf-8")
    (outdir / f"report.{args.tag}.json").write_text(
        json.dumps({"models": {s["name"]: {k: v for k, v in s.items()
                                           if k != "per_item"} for s in summaries},
                    "discrimination": disc, "comparisons": comps},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet:
        print("\n" + md)

    if args.families:
        print("\n## 逐族通过率\n")
        fams = sorted({f for s in summaries for f in s["families"]})
        hdr = "| 族 | 项数 | " + " | ".join(s["name"] for s in summaries) + " |"
        print(hdr)
        print("| --- | --- | " + " | ".join("---" for _ in summaries) + " |")
        for f in fams:
            n = max((s["families"][f]["n_items"] for s in summaries
                     if f in s["families"]), default=0)
            cells = [f"{s['families'][f]['score']:.2f}" if f in s["families"] else "—"
                     for s in summaries]
            print(f"| `{f}` | {n} | " + " | ".join(cells) + " |")

    if args.items:
        print(f"\n## 逐项通过率 · {args.items}\n")
        names = sorted(runs)
        per = {n: {} for n in names}
        cats = {}
        for n in names:
            for r in runs[n]:
                cats[r["id"]] = r["cat"]
                per[n].setdefault(r["id"], []).append(int(bool(r["pass"])))
        rows = [(i, cats[i], {n: sum(v) / len(v) for n, v in
                              ((nn, per[nn].get(i, [])) for nn in names) if v})
                for i in sorted(cats) if cats[i] == args.items]
        print("| 项 | tier | " + " | ".join(names) + " |")
        print("| --- | --- | " + " | ".join("---" for _ in names) + " |")
        for i, cat, rates in rows:
            print(f"| `{i}` | | " +
                  " | ".join(f"{rates[n]:.2f}" if n in rates else "—" for n in names) + " |")

    print(f"\n→ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
