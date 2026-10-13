"""Build the frozen battery — regenerate the mechanical families, validate
everything, and emit a hashed dataset + audit report.

    python scripts/compile_bench.py                # full build
    python scripts/compile_bench.py --no-gen       # skip regeneration (audit only)
    python scripts/compile_bench.py --check        # validate and exit non-zero on error

Outputs:

``bench/data/bench_v3.jsonl``
    the battery, sorted by (category order, id) — this is what the runner reads
``bench/data/bench_v3.manifest.json``
    SHA256, counts, and the coverage / lint / discrimination-design report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import gen_basics                                  # noqa: E402
from bench.registry import (CATEGORIES, CATEGORY_TARGET,        # noqa: E402
                            ItemError, battery_summary, check_anchors_in_prompt,
                            check_duplicates, check_persona_anchors,
                            check_tier_quota, coverage_vs_target, lint_items,
                            load_items_dir, reviewed_anchor_exceptions)

ITEMS = ROOT / "bench" / "items"
POOLS = ROOT / "bench" / "pools"
DATA = ROOT / "bench" / "data"


def regenerate(quiet: bool = False) -> None:
    """Rebuild the deterministic families from their content packs.

    Only 填空 / 问答 / 语言 / 续写 are generated; the other eleven categories are
    authored and must never be rewritten here, or a build would silently discard
    hand-written items.
    """
    if not (POOLS / "knowledge_pack.json").exists() or \
       not (POOLS / "language_pack.json").exists():
        print(f"!! pools missing under {POOLS}; run with --no-gen to keep the "
              f"existing generated file", file=sys.stderr)
        return
    reps = gen_basics.write_generated(ITEMS, POOLS)
    for r in reps:
        flag = "OK " if r["n"] == r["target"] else "!! "
        if not quiet or r["n"] != r["target"]:
            print(f"{flag}{r['cat']:<6} {r['n']:>4}/{r['target']:<4} "
                  f"tiers={r['tiers']} dropped={len(r['dropped'])}")
        for d in r["dropped"][:5]:
            print(f"      - {d}")


def order_key(it) -> tuple:
    return (CATEGORIES.index(it.cat), it.id)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gen", action="store_true",
                    help="do not regenerate the mechanical families")
    ap.add_argument("--check", action="store_true",
                    help="validate only; exit 1 on any hard error")
    ap.add_argument("--strict-lint", action="store_true",
                    help="treat lint warnings as errors")
    args = ap.parse_args()

    if not args.no_gen:
        regenerate()

    print(f"\n== 载入 {ITEMS}")
    try:
        items = load_items_dir(ITEMS)
    except ItemError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 1
    print(f"   {len(items)} 项，来自 {len(set(i.src for i in items))} 个文件")

    hard: list[str] = []
    dups = check_duplicates(items)
    hard += dups
    anchors = check_anchors_in_prompt(items)
    hard += anchors
    persona = check_persona_anchors(items)

    warns = lint_items(items)
    quota = check_tier_quota(items)

    s = battery_summary(items)
    print(f"\n== 结构")
    print(f"   类别 {len(s['by_cat'])}/{len(CATEGORIES)}  题目族 {len(s['families'])}")
    print(f"   tier  E={s['by_tier'].get('E',0)}  M={s['by_tier'].get('M',0)}"
          f"  H={s['by_tier'].get('H',0)}")
    n = max(1, s["n"])
    for t, target in (("E", 0.20), ("M", 0.50), ("H", 0.30)):
        got = s["by_tier"].get(t, 0) / n
        flag = "!!" if abs(got - target) > 0.06 else "  "
        print(f"   {flag} tier {t}: {got:.1%} (目标 {target:.0%})")

    print(f"\n== 覆盖率（实有/目标）")
    gaps = []
    for cat, have, want in coverage_vs_target(items):
        flag = "OK" if have >= want else ("~~" if have >= want * 0.8 else "!!")
        if have < want:
            gaps.append((cat, have, want))
        print(f"   {flag} {cat:<6} {have:>4}/{want:<4}")
    total_target = sum(CATEGORY_TARGET.values())
    print(f"   合计 {s['n']}/{total_target}")

    print(f"\n== 能力轴")
    for a, c in sorted(s["by_axis"].items(), key=lambda kv: -kv[1]):
        print(f"   {a:<14} {c}")

    print(f"\n== 判据谓词使用")
    print("   " + ", ".join(f"{k}×{v}" for k, v in
                            sorted(s["predicates"].items(), key=lambda kv: -kv[1])))

    if dups:
        print(f"\n!! 重复 id {len(dups)} 处", file=sys.stderr)
        for d in dups[:10]:
            print(f"   {d}", file=sys.stderr)
    if anchors:
        print(f"\n!! 判据锚点已被「正在被测的那一轮」直接给出，会白送分："
              f"{len(anchors)} 处", file=sys.stderr)
        for d in anchors:
            print(f"   {d}", file=sys.stderr)
    exc = reviewed_anchor_exceptions(items)
    if exc:
        print(f"\n~~ 已人工复核的锚点豁免 {len(exc)} 项（对抗式探针，不是白送分）")
        for e in exc:
            print(f"   {e}")
    if persona:
        print(f"\n~~ 判据锚点由 system 设定块提供：{len(persona)} 处"
              f"（角色内探针的常见正确写法，仅提示）")
        for d in persona[:8]:
            print(f"   {d}")
    if warns:
        print(f"\n~~ lint 警告 {len(warns)} 条")
        for w in warns[:20]:
            print(f"   {w}")
        if len(warns) > 20:
            print(f"   ... 另 {len(warns)-20} 条")
    if quota:
        print(f"\n~~ 难度配额偏差 {len(quota)} 条")
        for q in quota:
            print(f"   {q}")

    if args.check:
        ok = not hard and (not args.strict_lint or not warns)
        print("\n== CHECK " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1

    # ---- freeze
    items.sort(key=order_key)
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "bench_v3.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            fh.write(json.dumps(it.to_json(), ensure_ascii=False) + "\n")

    h = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "dataset": "bench_v3",
        "path": "bench/data/bench_v3.jsonl",
        "sha256": h,
        "n_items": len(items),
        "n_families": len(s["families"]),
        "categories": {c: s["by_cat"].get(c, 0) for c in CATEGORIES},
        "category_target": CATEGORY_TARGET,
        "tiers": {t: s["by_tier"].get(t, 0) for t in ("E", "M", "H")},
        "axes": s["by_axis"],
        "predicates": s["predicates"],
        "hard_errors": hard,
        "lint_warnings": warns,
        "persona_anchor_notes": persona,
        "reviewed_anchor_exceptions": exc,
        "tier_quota_warnings": quota,
        "coverage_gaps": [{"cat": c, "have": h_, "want": w} for c, h_, w in gaps],
        "sources": sorted({i.src for i in items}),
    }
    (DATA / "bench_v3.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 冻结 → {out}")
    print(f"   sha256 {h}")
    print(f"   manifest bench/data/bench_v3.manifest.json")
    if hard:
        print(f"   !! 有 {len(hard)} 处硬错误，请先修（见上）")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
