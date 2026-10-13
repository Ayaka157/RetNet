"""把满预算基准的 .jsonl 变成「七个零分类到底是不是评测假象」的裁决表。

    python scripts/analyze_zero_categories.py logs/bench_v3_full/v13.b1024.jsonl
    python scripts/analyze_zero_categories.py <jsonl> --dataset bench/data/bench_v3.jsonl

## 它回答什么

`docs/17` §2.4 用实测指出：v13 那七个挂零的类别里，**5 类是被生成预算截断**
（结构化 60/60、编码 25/25、数学 25/25、指令遵循 77/80、填空 110/120），
模型根本没走到答案；并且明确说**「在这一步做完之前，七个零分类的结论不能作为决策依据」**。

满预算基准（`--budget 1024`）就是那一步。本脚本按类别同时报
**通过率**与**截断率**，并对每个「零分类」给出一句判定：

  * 通过率 > 0                    → 非零，不再是零分类
  * 通过率 = 0 且截断率 ≥ 0.5     → **不可归因于尺度**（预算烧光，测的不是能力）
  * 通过率 = 0 且截断率 < 0.5     → 零分**可以**归因于能力（这才是有信息量的零）

## 口径

* 「截断」= `truncated` 字段（`bench/runner.py` 已改成**逐轮精确**：
  `any(x >= item.max_new for x in n_new_per_turn)`，不是"整体超预算"）
* **部分数据**：jsonl 可能只写了一半的题。本脚本会**对照数据集报出 n/总数**，
  并且当 `n < 总数` 时在结论里打 `[部分]`——**部分数据上的类别通过率不可当作最终值**。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `docs/17` §0.2「容量无关 vs 容量相关」的判定，按**类别名**落成表。
# 目的：让本脚本的输出**自己带解释**——因为这张表可能在我（写脚本的人）不在时被读。
# 只收录 docs/17 明确判过的类别；其余填 None（未判定），不猜。
DOC17_CLASS = {
    "填空":   ("容量无关", "需目标格式的监督数据；docs/14 §9.3 有独立格式探针"),
    "语言":   ("容量无关", "同上"),
    "结构化": ("弱容量相关·当前不可判", "docs/17 明确说先修口径才能判"),
    "指令遵循": ("弱容量相关·当前不可判", "同上"),
    "长上下文": ("容量相关", "跨 1.2K token 取事实，docs/14 §9.3 单轮对照已排除格式解释"),
    "数学":   ("容量相关", "v10（0.28 epoch）也是 0"),
    "编码":   ("容量相关", "地板锚点"),
}


def load_jsonl(p: Path):
    out = []
    with p.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue          # 正在写的那一行可能是半行
    return out


def dataset_counts(p: Path):
    tot = defaultdict(int)
    if not p.exists():
        return tot
    with p.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                tot[json.loads(ln).get("cat", "?")] += 1
            except json.JSONDecodeError:
                continue
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="基准输出的 .jsonl")
    ap.add_argument("--dataset", default=str(ROOT / "bench" / "data" / "bench_v3.jsonl"))
    a = ap.parse_args()

    p = Path(a.jsonl)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        print(f"❌ 找不到 {p}")
        return 1

    rows = load_jsonl(p)
    total = dataset_counts(Path(a.dataset))
    if not rows:
        print(f"❌ {p} 里没有可解析的记录")
        return 1

    print("=" * 86)
    print(f"零分类裁决表 —— {p.name}")
    print("=" * 86)
    print(f"已解析 {len(rows)} 条记录；数据集共 {sum(total.values())} 项"
          f"（{'完整' if len(rows) >= sum(total.values()) else '**部分数据**'}）")

    g = defaultdict(lambda: {"n": 0, "pass": 0, "trunc": 0, "nnew": 0,
                             "nu": 0, "passu": 0})
    for r in rows:
        c = r.get("cat", "?")
        g[c]["n"] += 1
        g[c]["pass"] += 1 if r.get("pass") else 0
        g[c]["trunc"] += 1 if r.get("truncated") else 0
        g[c]["nnew"] += int(r.get("n_new") or 0)
        if not r.get("truncated"):
            g[c]["nu"] += 1
            g[c]["passu"] += 1 if r.get("pass") else 0

    print()
    print(f"  {'类别':<12}{'n/总':>9}{'通过率':>8}{'截断率':>8}"
          f"{'未截断 n':>9}{'未截断通过率':>13}   判定")
    zero_real, zero_artifact, nonzero, partial = [], [], [], []
    for c, s in sorted(g.items(), key=lambda kv: (kv[1]["pass"], -kv[1]["n"])):
        n = s["n"]
        pr = s["pass"] / n
        tr = s["trunc"] / n
        nu = s["nu"]
        pru = (s["passu"] / nu) if nu else float("nan")
        nt = total.get(c, 0)
        is_partial = bool(nt and n < nt)
        if is_partial:
            partial.append(c)
        # 判定只用**未截断子集**的通过率：截断项本来就"没走到答案"，不该混进来
        if pru > 0:
            tag = "✅ 非零"
            nonzero.append(c)
        elif nu == 0:
            tag = "⚠️ 全部被截断 → 读不出任何东西"
            zero_artifact.append(c)
        elif tr >= 0.5:
            tag = "❌ 零分不可归因于尺度（半数以上预算烧光）"
            zero_artifact.append(c)
        else:
            tag = f"⚠️ 零分可归因于能力（未截断的 {nu} 项也全 0）"
            zero_real.append(c)
        if is_partial:
            tag += "  [部分]"
        pru_s = "  n/a" if nu == 0 else f"{pru:.3f}"
        print(f"  {c:<12}{f'{n}/{nt}' if nt else str(n):>9}{pr:>8.3f}{tr:>8.3f}"
              f"{nu:>9}{pru_s:>13}   {tag}")

    print("\n" + "=" * 86)
    print("结论")
    print("=" * 86)
    print(f"  非零类别      : {len(nonzero)}  {nonzero}")
    print(f"  零·评测假象   : {len(zero_artifact)}  {zero_artifact}")
    print(f"  零·可归因能力 : {len(zero_real)}  {zero_real}")

    # 把零分类与 docs/17 的容量判定交叉，直接给「该不该扩」的读数
    cap_real = [c for c in zero_real if DOC17_CLASS.get(c, ("", ""))[0] == "容量相关"]
    cap_soft = [c for c in zero_real if DOC17_CLASS.get(c, ("", ""))[0].startswith("弱容量")]
    cap_no = [c for c in zero_real if DOC17_CLASS.get(c, ("", ""))[0] == "容量无关"]
    cap_unk = [c for c in zero_real if c not in DOC17_CLASS]

    print("\n  --- 与 `docs/17` §0.2 的容量判定交叉（决定「该不该扩到 0.5B」）---")
    for label, lst in (("容量相关（扩容才是正解）", cap_real),
                       ("弱容量相关·当前不可判", cap_soft),
                       ("容量无关（该修数据/格式，加参数买不到）", cap_no),
                       ("docs/17 未判定", cap_unk)):
        print(f"    {label:<34}: {lst if lst else '—'}")

    if cap_real:
        print(f"\n  ▶ **真正支持扩容的证据**：{cap_real}")
        print("    这几类是 docs/17 判定容量相关、且本次实测在**未截断子集上仍为 0**。")
    if cap_no:
        print(f"\n  ▶ **不支持扩容的证据**：{cap_no} 是容量无关项，加参数买不到，")
        print("    该做的是目标格式的监督数据。")
    if not cap_real:
        print("\n  ▶ **本次数据里没有任何「容量相关且实测为零」的类别**：")
        print("    要么那几类还没跑到（数学/编码/长上下文在数据集里排在最后），")
        print("    要么它们已经非零。**在它们测出来之前不要用这张表决定扩容。**")

    if partial:
        print(f"\n  ⚠️ 这些类别是**部分数据**：{partial}。等基准跑完再重跑本脚本。")
    print("\n  ⚠️ **口径警告**：截断率 > 0 的类别里，整体通过率不可直接解读。")
    print("     判定一律用「未截断子集通过率」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
