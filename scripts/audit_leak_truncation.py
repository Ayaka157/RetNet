"""独立复核 docs/17 §2 的核心断言：52% 的 think 泄漏率是不是评测预算造成的。

docs/17 说泄漏判据（`bench/runner.py:155`）不看生成是否被预算截断，而题库每项的
``max_new`` 只有 16–160，训练数据的 CoT 长度中位却是 570 token。它的切分结论是：

    单轮 · 被预算截断   644 项 → 泄漏 82.1%
    单轮 · 未被截断     235 项 → 泄漏  0.0%

旧结果文件里**没有** ``max_new``（那是本轮才加进 runner 的），所以必须拿题库
``bench/data/bench_v3.jsonl`` 按 id join 回去。这个脚本就是做这件事的——
不依赖 docs/17 的任何中间产物，只读原始 jsonl。

    python scripts/audit_leak_truncation.py --run logs/bench_v3/v13.T0.7_s1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.registry import load_jsonl  # noqa: E402

DEFAULT_DATASET = ROOT / "bench" / "data" / "bench_v3.jsonl"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="结果 jsonl")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--by-cat", action="store_true", help="额外按类别拆截断率")
    args = ap.parse_args()

    items = {it.id: it for it in load_jsonl(Path(args.dataset))}
    recs = []
    for line in Path(args.run).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))

    print(f"题库 {len(items)} 项 · 结果 {len(recs)} 条")
    miss = [r["id"] for r in recs if r["id"] not in items]
    if miss:
        print(f"⚠️ {len(miss)} 条结果在题库里找不到对应项（头 3 个：{miss[:3]}）")

    # 预算分布：验证 docs/17「max_new 全部落在 16–160」这句
    budgets = sorted(it.max_new for it in items.values())
    print(f"题库 max_new 范围 {budgets[0]}–{budgets[-1]}  "
          f"中位 {budgets[len(budgets)//2]}")

    # 单轮项的判据是精确的：n_new == max_new 即被截断
    st = [r for r in recs if len(items[r["id"]].turns) == 1] if not miss else \
         [r for r in recs if r["id"] in items and len(items[r["id"]].turns) == 1]
    mt = [r for r in recs if r["id"] in items and len(items[r["id"]].turns) > 1]

    def split(rs, label):
        trunc = [r for r in rs if r["n_new"] >= items[r["id"]].max_new]
        clean = [r for r in rs if r["n_new"] < items[r["id"]].max_new]
        lt = sum(1 for r in trunc if r["think_leak"])
        lc = sum(1 for r in clean if r["think_leak"])
        print(f"\n  {label}")
        print(f"    被预算截断 {len(trunc):>4} 项  泄漏 {lt:>4} "
              f"({lt/len(trunc)*100 if trunc else 0:5.1f}%)  "
              f"判据通过 {sum(1 for r in trunc if r['pass'])}")
        print(f"    未被截断   {len(clean):>4} 项  泄漏 {lc:>4} "
              f"({lc/len(clean)*100 if clean else 0:5.1f}%)  "
              f"判据通过 {sum(1 for r in clean if r['pass'])}")
        return trunc, clean

    print("\n== 按是否被预算截断切开 ==")
    t_all, c_all = split(recs, "全部项")
    t_st, c_st = split(st, "单轮项")
    split(mt, "多轮项（判据只是保守下界）")

    tot_leak = sum(1 for r in recs if r["think_leak"])
    print(f"\n  裸泄漏率 {tot_leak}/{len(recs)} = {tot_leak/len(recs)*100:.1f}%")
    print(f"  其中落在截断子集里：{sum(1 for r in t_all if r['think_leak'])}"
          f" / {tot_leak}")

    if args.by_cat:
        print("\n== 按类别的截断率与泄漏率 ==")
        by = defaultdict(list)
        for r in recs:
            by[items[r["id"]].cat].append(r)
        print(f"  {'类别':<10}{'项数':>6}{'中位预算':>9}{'截断':>9}{'泄漏率':>9}{'得分':>8}")
        for c in sorted(by, key=lambda k: -len(by[k])):
            rs = by[c]
            tr = sum(1 for r in rs if r["n_new"] >= items[r["id"]].max_new)
            bk = sorted(items[r["id"]].max_new for r in rs)
            leak = sum(1 for r in rs if r["think_leak"]) / len(rs)
            score = sum(1 for r in rs if r["pass"]) / len(rs)
            print(f"  {c:<10}{len(rs):>6}{bk[len(bk)//2]:>9}"
                  f"{tr:>4}/{len(rs):<4}{leak:>9.2f}{score:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
