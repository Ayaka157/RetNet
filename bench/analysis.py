"""Aggregation, discrimination diagnostics and paired significance tests.

Battery v3 exists because v2 could not answer its own questions: half the items
had under 0.05 discrimination, so a total-score difference was mostly noise
(`docs/13` §3).  The report therefore refuses to print a bare number.  Alongside
every score it prints

* the **bootstrap CI** over items (the item is the sampling unit, not the
  generation — seeds within an item are correlated and treating them as
  independent is how v2 got a 1.8-sigma result it could not interpret),
* the **tier split**, so "this category is on the floor" is visible as
  ``H=E=0`` rather than hidden inside a mean,
* **per-item discrimination** across checkpoints, and
* **McNemar's exact test** on same-item/same-seed pairs for two checkpoints.

think-leak and degenerate repetition are reported separately and never enter a
score (`docs/13` §1).
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .registry import CATEGORY_WEIGHTS, TIERS

N_BOOT = 2000


# --------------------------------------------------------------------- loading


def load_run(path: Path) -> List[Dict]:
    recs: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return recs


def item_pass_rates(recs: Sequence[Dict]) -> Dict[str, float]:
    """``{item_id: pass_rate}`` over the seeds present."""
    acc: Dict[str, List[int]] = defaultdict(list)
    for r in recs:
        acc[r["id"]].append(int(bool(r["pass"])))
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def _cat_of(recs: Sequence[Dict]) -> Dict[str, str]:
    return {r["id"]: r["cat"] for r in recs}


# ----------------------------------------------------------------------- stats


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(per_item: Sequence[float], n_boot: int = N_BOOT,
                 alpha: float = 0.05, seed: int = 20261009) -> Tuple[float, float]:
    """Percentile CI over a resample of items."""
    if not per_item:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(per_item)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(m):
            s += per_item[rng.randrange(m)]
        means.append(s / m)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation without scipy (ties get average ranks)."""
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else 0.0


# ------------------------------------------------------------------- summary


def degenerate_rate(recs: Sequence[Dict]) -> float:
    """Share of replies that burn the budget on a repeated n-gram."""
    bad = 0
    for r in recs:
        s = (r.get("reply") or "").strip()
        if len(s) < 40:
            continue
        g = [s[i:i + 4] for i in range(len(s) - 3)]
        if g and len(set(g)) / len(g) < 0.65:
            bad += 1
    return bad / len(recs) if recs else 0.0


def summarize(recs: Sequence[Dict], name: str = "model") -> Dict:
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    by_axis: Dict[str, List[Dict]] = defaultdict(list)
    by_tier: Dict[str, List[Dict]] = defaultdict(list)
    by_fam: Dict[str, List[Dict]] = defaultdict(list)
    for r in recs:
        by_cat[r["cat"]].append(r)
        by_tier[r.get("tier", "?")].append(r)
        by_fam[r.get("family", "?")].append(r)
        for a in r.get("axes") or []:
            by_axis[a].append(r)

    def block(group: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        out = {}
        for k, rs in sorted(group.items()):
            per_item = item_pass_rates(rs)
            vals = list(per_item.values())
            lo, hi = bootstrap_ci(vals) if len(vals) > 3 else (0.0, 1.0)
            out[k] = {"n_records": len(rs), "n_items": len(per_item),
                      "score": sum(vals) / len(vals) if vals else 0.0,
                      "ci": [lo, hi],
                      "leak": sum(r["think_leak"] for r in rs) / len(rs),
                      **_trunc_split(rs),
                      "tiers": {t: _rate([r for r in rs if r.get("tier") == t])
                                for t in TIERS}}
        return out

    cats = block(by_cat)
    # Weighted total over the categories actually present, using the frozen
    # weights.  Reported with the weight mass so a partial run is not mistaken
    # for a full-battery score.
    wsum = sum(CATEGORY_WEIGHTS.get(c, 1.0) for c in cats)
    total = (sum(cats[c]["score"] * CATEGORY_WEIGHTS.get(c, 1.0) for c in cats) / wsum
             if wsum else 0.0)
    per_cat_vals = [cats[c]["score"] for c in cats]
    tot_lo, tot_hi = (bootstrap_ci(per_cat_vals) if len(per_cat_vals) > 3
                      else (0.0, 1.0))

    return {
        "name": name,
        "n_records": len(recs),
        "n_items": len({r["id"] for r in recs}),
        "seeds": sorted({r["seed"] for r in recs}),
        "total": total,
        "total_ci": [tot_lo, tot_hi],
        "weight_mass": wsum,
        "cats": cats,
        "axes": block(by_axis),
        "tiers": {t: _rate([r for r in recs if r.get("tier") == t]) for t in TIERS},
        "tier_counts": {t: sum(1 for r in recs if r.get("tier") == t) for t in TIERS},
        "families": block(by_fam),
        "leak": sum(r["think_leak"] for r in recs) / len(recs) if recs else 0.0,
        **_trunc_split(recs),
        "degenerate": degenerate_rate(recs),
        "mean_len": (sum(len(r.get("reply") or "") for r in recs) / len(recs)
                     if recs else 0.0),
        "seconds": sum(r.get("seconds", 0.0) for r in recs),
        "per_item": item_pass_rates(recs),
    }


def _trunc_split(rs: Sequence[Dict]) -> Dict:
    """把「是否被生成预算截断」这个维度单独拆出来。

    NOTE(2026-10-10, docs/17 §2)：这是本轮最重要的度量修正。裸的 ``leak`` 会把
    「模型不会收尾」和「预算根本不够写 CoT」混成一个数——v13 那个 52% 泄漏率
    就是这么来的：题库每项 ``max_new`` 只有 16–160，而训练数据的 CoT 长度中位是
    570 token。按截断切开之后，未被截断的 235 个单轮项泄漏是 **0**。

    ⚠️ **在 ``truncated`` 字段存在之前产生的旧结果文件里，这几个键返回 ``None``，
    不是 0** —— 那种文件无法支持任何关于泄漏率的结论。
    """
    if not any("truncated" in r for r in rs):
        return {"truncated_rate": None, "n_untruncated": 0,
                "leak_untruncated": None, "score_untruncated": None}
    clean = [r for r in rs if r.get("truncated") is False]
    cv = list(item_pass_rates(clean).values())
    return {
        "truncated_rate": sum(1 for r in rs if r.get("truncated")) / len(rs),
        "n_untruncated": len(clean),
        "leak_untruncated": (sum(r["think_leak"] for r in clean) / len(clean)
                             if clean else None),
        "score_untruncated": (sum(cv) / len(cv)) if cv else None,
    }


def _rate(rs: Sequence[Dict]) -> float:
    if not rs:
        return 0.0
    pi = item_pass_rates(rs)
    return sum(pi.values()) / len(pi) if pi else 0.0


# ------------------------------------------------------------- discrimination


def discrimination(runs: Dict[str, Sequence[Dict]]) -> List[Dict]:
    """Per item: pass rate per model, spread, and a verdict.

    * ``dead``  — every model passed (ceiling) or every model failed (floor)
    * ``weak``  — spread below 0.05 across models, i.e. carries no signal
    * ``live``  — spread at least 0.05; ``spread`` is the effect size
    """
    names = list(runs)
    per: Dict[str, Dict[str, float]] = {n: item_pass_rates(runs[n]) for n in names}
    cats = _cat_of(runs[names[0]]) if names else {}
    out: List[Dict] = []
    for iid in sorted(set().union(*[set(p) for p in per.values()]) if per else set()):
        vals = {n: per[n].get(iid) for n in names if iid in per[n]}
        present = [v for v in vals.values() if v is not None]
        if not present:
            continue
        spread = max(present) - min(present)
        if all(v >= 0.95 for v in present):
            verdict = "ceiling"
        elif all(v <= 0.05 for v in present):
            verdict = "floor"
        elif spread < 0.05:
            verdict = "weak"
        else:
            verdict = "live"
        out.append({"id": iid, "cat": cats.get(iid, "?"), "rates": vals,
                    "spread": spread, "verdict": verdict,
                    "mean": sum(present) / len(present)})
    out.sort(key=lambda d: -d["spread"])
    return out


def discrimination_report(disc: Sequence[Dict], names: Sequence[str]) -> Dict:
    n = len(disc) or 1
    live = [d for d in disc if d["verdict"] == "live"]
    return {
        "n_items": len(disc),
        "live": len(live),
        "weak": sum(1 for d in disc if d["verdict"] == "weak"),
        "floor": sum(1 for d in disc if d["verdict"] == "floor"),
        "ceiling": sum(1 for d in disc if d["verdict"] == "ceiling"),
        "live_share": len(live) / n,
        "top": disc[:20],
        "worst_cats": _cat_disc(disc),
    }


def _cat_disc(disc: Sequence[Dict]) -> List[Dict]:
    g: Dict[str, List[Dict]] = defaultdict(list)
    for d in disc:
        g[d["cat"]].append(d)
    rows = []
    for cat, ds in sorted(g.items()):
        live = sum(1 for d in ds if d["verdict"] == "live")
        rows.append({"cat": cat, "n": len(ds), "live": live,
                     "live_share": live / len(ds),
                     "mean_spread": sum(d["spread"] for d in ds) / len(ds),
                     "floor": sum(1 for d in ds if d["verdict"] == "floor"),
                     "ceiling": sum(1 for d in ds if d["verdict"] == "ceiling")})
    rows.sort(key=lambda r: r["live_share"])
    return rows


# -------------------------------------------------------------------- paired


def compare(a_name: str, a: Sequence[Dict], b_name: str, b: Sequence[Dict]) -> Dict:
    """Paired comparison on the same (item, seed) cells."""
    ka = {(r["id"], r["seed"]): bool(r["pass"]) for r in a}
    kb = {(r["id"], r["seed"]): bool(r["pass"]) for r in b}
    shared = sorted(set(ka) & set(kb))
    a_win = sum(1 for k in shared if ka[k] and not kb[k])
    b_win = sum(1 for k in shared if kb[k] and not ka[k])
    both = sum(1 for k in shared if ka[k] and kb[k])
    neither = sum(1 for k in shared if not ka[k] and not kb[k])
    p = mcnemar_exact(a_win, b_win)

    cats = _cat_of(a)
    per_cat: Dict[str, Dict] = {}
    for cat in sorted(set(cats[k[0]] for k in shared)):
        ks = [k for k in shared if cats[k[0]] == cat]
        aw = sum(1 for k in ks if ka[k] and not kb[k])
        bw = sum(1 for k in ks if kb[k] and not ka[k])
        sa = sum(int(ka[k]) for k in ks) / len(ks)
        sb = sum(int(kb[k]) for k in ks) / len(ks)
        per_cat[cat] = {"n": len(ks), "a": sa, "b": sb, "diff": sa - sb,
                        "a_win": aw, "b_win": bw, "p": mcnemar_exact(aw, bw)}

    ids = sorted({k[0] for k in shared})
    ipa, ipb = {}, {}
    for k in shared:
        ipa.setdefault(k[0], []).append(int(ka[k]))
        ipb.setdefault(k[0], []).append(int(kb[k]))
    item_a = [sum(ipa[i]) / len(ipa[i]) for i in ids]
    item_b = [sum(ipb[i]) / len(ipb[i]) for i in ids]

    return {
        "a": a_name, "b": b_name, "n_pairs": len(shared),
        "a_win": a_win, "b_win": b_win, "both": both, "neither": neither,
        "p": p,
        "a_score": sum(item_a) / len(item_a) if item_a else 0.0,
        "b_score": sum(item_b) / len(item_b) if item_b else 0.0,
        "item_corr": spearman(item_a, item_b),
        "per_cat": per_cat,
        "a_items_won": sum(1 for i in ids
                           if sum(ipa[i]) / len(ipa[i]) > sum(ipb[i]) / len(ipb[i])),
        "b_items_won": sum(1 for i in ids
                           if sum(ipb[i]) / len(ipb[i]) > sum(ipa[i]) / len(ipa[i])),
        "items_tied": sum(1 for i in ids
                          if sum(ipa[i]) / len(ipa[i]) == sum(ipb[i]) / len(ipb[i])),
    }


# ------------------------------------------------------------------- markdown


def markdown_report(summaries: Sequence[Dict], disc: Optional[Dict] = None,
                    comps: Optional[Sequence[Dict]] = None,
                    title: str = "能力基准 v3") -> str:
    cats = sorted({c for s in summaries for c in s["cats"]})
    lines = [f"# {title}", ""]
    for s in summaries:
        lines.append(f"- **{s['name']}**：{s['n_items']} 项 × {len(s['seeds'])} 种子 "
                     f"= {s['n_records']} 次生成；权重覆盖 {s['weight_mass']:.1f}")
    lines += ["", "| 模型 | " + " | ".join(cats) + " | **加权总分** | 95% CI | think泄漏 | 退化重复 |",
              "| --- | " + " | ".join(["---"] * len(cats)) + " | --- | --- | --- | --- |"]
    for s in summaries:
        cells = [f"{s['cats'][c]['score']:.2f}" if c in s["cats"] else "—" for c in cats]
        lines.append(f"| {s['name']} | " + " | ".join(cells) +
                     f" | **{s['total']:.4f}** | "
                     f"{s['total_ci'][0]:.3f}–{s['total_ci'][1]:.3f} | "
                     f"{s['leak']*100:.1f}% | {s['degenerate']*100:.1f}% |")

    lines += ["", "## 难度分层（E/M/H 通过率）", "",
              "| 模型 | E | M | H | E 占比 | M 占比 | H 占比 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in summaries:
        tc = s["tier_counts"]
        tot = max(1, sum(tc.values()))
        lines.append(f"| {s['name']} | {s['tiers']['E']:.2f} | {s['tiers']['M']:.2f} | "
                     f"{s['tiers']['H']:.2f} | {tc['E']/tot:.0%} | {tc['M']/tot:.0%} | "
                     f"{tc['H']/tot:.0%} |")

    axes = sorted({a for s in summaries for a in s["axes"]})
    if axes:
        lines += ["", "## 能力轴画像", "",
                  "| 模型 | " + " | ".join(axes) + " |",
                  "| --- | " + " | ".join(["---"] * len(axes)) + " |"]
        for s in summaries:
            lines.append(f"| {s['name']} | " +
                         " | ".join(f"{s['axes'][a]['score']:.2f}" if a in s["axes"] else "—"
                                    for a in axes) + " |")

    if comps:
        lines += ["", "## 配对检验（同题同种子）", "",
                  "| A | B | 对数 | A胜 | B胜 | 平 | A分 | B分 | McNemar p | 判定 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c in comps:
            verdict = ("**显著**" if c["p"] < 0.05 else
                       "边缘" if c["p"] < 0.10 else "无差异")
            sign = "A 优" if c["a_score"] > c["b_score"] else "B 优"
            lines.append(f"| {c['a']} | {c['b']} | {c['n_pairs']} | {c['a_win']} | "
                         f"{c['b_win']} | {c['both']+c['neither']} | "
                         f"{c['a_score']:.3f} | {c['b_score']:.3f} | "
                         f"{c['p']:.3f} | {verdict}（{sign}） |")

    if disc:
        lines += ["", "## 区分度诊断", "",
                  f"- 共 {disc['n_items']} 项：**live {disc['live']}**"
                  f"（{disc['live_share']:.0%}）、weak {disc['weak']}、"
                  f"floor {disc['floor']}、ceiling {disc['ceiling']}",
                  "", "| 类别 | 项数 | live | live占比 | 平均极差 | floor | ceiling |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in disc["worst_cats"]:
            lines.append(f"| {r['cat']} | {r['n']} | {r['live']} | "
                         f"{r['live_share']:.0%} | {r['mean_spread']:.3f} | "
                         f"{r['floor']} | {r['ceiling']} |")
        lines += ["", "### 区分度最高的 20 项", "",
                  "| 项 | 类别 | 极差 | 各模型通过率 |", "| --- | --- | --- | --- |"]
        for d in disc["top"]:
            rates = "  ".join(f"{k}={v:.2f}" for k, v in d["rates"].items())
            lines.append(f"| `{d['id']}` | {d['cat']} | {d['spread']:.2f} | {rates} |")
    return "\n".join(lines) + "\n"
