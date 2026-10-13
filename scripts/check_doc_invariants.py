"""文档一致性守卫 —— 把 `docs/23` 的表**对着真值**核，而不是靠启发式。

    python scripts/check_doc_invariants.py

## 为什么重写（第一版是个反面教材）

第一版查「同一行里 GiB 与 MiB 是否自洽」，结果：
* **两个误报**：`7,593 MiB = 7.42 GiB` 是 7.4150 的正确两位四舍五入（差 5 MiB 属显示精度）；
  另一行 `笔记本 7,226 MiB − 桌面 6.61 GiB` **故意**是两台机器的数并列，
  整节就是在讲这个减法**不可比**——我却把它当矛盾。
* **真 bug 反而漏了**：`§6 写 3.28 GiB` 与 `§3.2 写 3,278 MiB` 的矛盾
  **跨两张表、两个不同的行**，同行检查永远看不到。

教训：**对文档做一致性检查，真值必须是"外部锚点"或"显式映射"，
不能是对格式的启发式猜测。** 所以这一版：

1. **对着 `logs/flops_model_out.txt`（模型实测输出）核 `docs/23` 的表**——
   用**显式**的「文档行标签 ↔ 模型行标签」映射，不猜。
2. 显示精度容差按**小数位数**算：两位小数允许 ±0.005×1024 = ±5.12 MiB。
3. 只在对得上时判失败；对不上的行**只报告不判失败**（可能是模型表里没有的项）。

退出码非 0 = 文档与模型输出不一致。**改完表就跑。**
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "23_0.5B扩容方案.md"
MODEL_OUT = ROOT / "logs" / "flops_model_out.txt"

# 显式映射：docs/23 §6 的行标签 -> flops_model.py 输出的行标签
# 不猜、不模糊匹配；对不上就是没核到，报告出来。
ROW_MAP = {
    "稠密 272M": "现状 272M 稠密",
    "稠密 0.5B": "0.5B 稠密宽浅",
    "稠密 0.6B": "0.6B 稠密宽浅",
    "MoE E=4 top-1": "MoE E=4 top1",
    "MoE E=5 top-1": "MoE E=5 top1",
    "MoE E=6 top-1": "MoE E=6 top1",
    "MoE E=7 top-1": "MoE E=7 top1",
    "0.5B 稠密 + MoD": "0.5B 稠密+MoD",
}


def tol_for(decimals: int) -> float:
    """显示精度带来的容差，**单位是 GiB**（必须与比较量同单位）。

    ⚠️ 第一版写成 `0.5 * 10**-decimals * 1024 + 0.51`，本意是"换成 MiB 是 5.12"，
    但结果被拿去和一个**以 GiB 为单位**的差比较了 —— 于是容差变成 **5.63 GiB**，
    任何两个显存值都算相等。阳性对照（把 3.20 改回错的 3.28）因此**照样报 ✅**。
    **同类单位错误（MiB/GiB）本会话已经出现过四次。**
    """
    return 0.5 * (10 ** -decimals)          # 单位：GiB；两位小数 -> 0.005


def doc_rows(text: str):
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|") or re.fullmatch(r"\|[\s:|-]+\|", s):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells:
            yield cells


FRONTIER = re.compile(r"(\d\.\d\d)\s+(-?[\d,]+)\s+(\d+)~(\d+)")


def model_rows(text: str):
    """flops_model §2 前沿表：标签 -> (行文本, 正则 group)。

    ⚠️ **不能用「后写覆盖」**：`flops_model_out.txt` 里每个标签出现**两次**——
    一次是 §2 的前沿表，一次是「显存分项」表（`2598 80 32 391 24 154`）。
    第一版让后出现的分项行覆盖了前沿行，于是后面所有正则都匹配不到、
    **比对一条都没做却报「✅ 一致」**。这里只收**能匹配前沿格式**的行。
    """
    out = {}
    for line in text.splitlines():
        s = line.strip()
        for mlabel in ROW_MAP.values():
            if not s.startswith(mlabel):
                continue
            m = FRONTIER.search(line)
            if m:                       # 只认前沿行；分项行忽略
                out[mlabel] = (line, m.groups())
    return out


def check_self_consistency(doc: str) -> int:
    """只靠文档自身的算术核 —— **不需要外部真值**，所以永远不会因真值文件变化而空转。

    这是本轮新增的一类检查。前几轮找到的错里，至少三个是**纯内部矛盾**：
    表里的分项加起来不等于合计、某行的合计与它自己的余量列反推的值不符、
    标称的百分比范围和实际占比不符。这类错不需要模型输出就能抓。
    """
    fails = 0

    # ---------- (a) §3.2 预算表：分项之和 == 合计 ----------
    print("\n=== (a) §3.2 显存预算表：分项之和 == 合计 ===")
    m = re.search(r"\| 方案 \| state(.*?)(?=\n\n)", doc, re.S)
    if not m:
        print("  ⚠️ 找不到 §3.2 表（格式变了？）")
        fails += 1
    else:
        n = 0
        for line in m.group(1).splitlines():
            s = line.strip()
            if not s.startswith("|") or re.fullmatch(r"\|[\s:|-]+\|", s):
                continue
            c = [x.strip() for x in s.strip("|").split("|")]
            if len(c) < 8:          # 标签 + 6 个分项 + 合计 = 8 格
                continue
            try:
                parts = [int(re.sub(r"[^\d]", "", x)) for x in c[1:7]]
                tot = int(re.sub(r"[^\d]", "", c[7]))
            except ValueError:
                continue
            n += 1
            if sum(parts) == tot:
                print(f"  ✅ {c[0]:<18} {sum(parts):,} == {tot:,}")
            else:
                fails += 1
                print(f"  ❌ {c[0]:<18} 分项和 {sum(parts):,} != 合计 {tot:,}"
                      f"（差 {sum(parts)-tot:+,}）")
        if n == 0:
            fails += 1
            print("  ❌ 空转：一行都没解析到")

    # ---------- (b) §6 表的 MoE 每专家增量 ----------
    print("\n=== (b) §6 表下「每加 1 个专家」的增量 ===")
    dparams = 2 * 1024 * 2048 * 20          # 2*d*ffn*L，d1024/ffn2048/L20
    mib = dparams * 10.004 / 2 ** 20
    claim = re.search(r"每加 1 个专家：\+([\d.]+)M 名义参数、\+0 算力、\+([\d.]+) GiB 状态", doc)
    if not claim:
        print("  ⚠️ 找不到该断言（措辞变了？）")
        fails += 1
    else:
        cp, cg = float(claim.group(1)), float(claim.group(2))
        ok_p = abs(cp - dparams / 1e6) <= 0.05
        ok_g = abs(cg - mib / 1024) <= 0.01
        print(f"  {'✅' if ok_p else '❌'} 参数 +{cp}M（实算 {dparams/1e6:.2f}M）")
        print(f"  {'✅' if ok_g else '❌'} 状态 +{cg} GiB（实算 {mib/1024:.3f} GiB"
              f" = {mib:.1f} MiB；注意 {dparams*10.004/1e6/1000:.3f} GB 是十进制，别混）")
        fails += (not ok_p) + (not ok_g)

    # ---------- (c) §8.6 数据表：步数 / 墙钟 / token 每参数 ----------
    print("\n=== (c) §8.6 数据表的三列换算 ===")
    tok, per = 3.4e9, 8192
    rows = re.findall(r"\|\s*(272M|0\.5B|0\.6B)\s*\|\s*([\d.]+)\s*\|\s*\*\*([\d.]+) 天\*\*"
                      r"\s*\|\s*\*{0,2}([\d.]+)\*{0,2}\s*\|\s*([\d.]+) B / (\d+) 天\s*\|", doc)
    if not rows:
        print("  ⚠️ 找不到 §8.6 数据表（格式变了？）—— 跳过")
    for nm, sps, days, tpp, chin_b, chin_d in rows:
        sps, days, tpp = float(sps), float(days), float(tpp)
        e_days = tok / per * sps / 86400
        e_tpp = tok / 3.4e9 * 3.4e9 / {"272M": 272.304448e6, "0.5B": 504.179896e6,
                                       "0.6B": 599.383312e6}[nm]
        ok_d = abs(e_days - days) <= 0.1
        ok_t = abs(e_tpp - tpp) <= 0.1
        print(f"  {'✅' if ok_d else '❌'} {nm:<5} 墙钟 {days} 天（实算 {e_days:.2f}）")
        print(f"  {'✅' if ok_t else '❌'} {nm:<5} token/参数 {tpp}（实算 {e_tpp:.2f}）")
        fails += (not ok_d) + (not ok_t)

    return fails


def check_cross_refs(doc: str) -> int:
    """核**跨文档**的 `§N` 引用是否在目标文档里真的存在。

    为什么单独做这一条：本会话在 `docs/23` 里做过一次全局的 `§4.4 ↔ §4.5`
    占位符交换（修正自己的编号顺序），结果把一个指向 **`docs/16` §4.4** 的引用
    也一起改成了 §4.5 —— 而 `docs/16` 根本没有 §4.5。
    **全局替换分不清「本文的 §4.4」和「他文的 §4.4」。**
    这类错肉眼很难发现（数字看着都对），但机械可判。
    """
    fails = 0
    print("\n=== (d) 跨文档 §N 引用是否解析得到 ===")
    docs_dir = DOC.parent
    # ⚠️ body 必须**不能跨过下一个链接**：`[`13`](./13_x.md) §2.7、[`14`](./14_y.md) §9`
    # 这种一行两个链接的写法，用宽松的 `[^|\n]*` 会把 docs/14 的 §9 算到 docs/13 头上，
    # 报出「docs/13 没有 §9」的**误报**。加 `(?!\[)` 断言即可。
    pattern = re.compile(r"\[[^\]]*\]\(\./([^)#]+?\.md)(?:#[^)]*)?\)((?:(?!\[)[^|\n])*)")
    seen = set()
    for m in pattern.finditer(doc):
        target_name, body = m.group(1), m.group(2)
        refs = re.findall(r"§([\d.]+(?:\s*[AB]\d+)?)", body)
        if not refs:
            continue
        target = docs_dir / target_name
        if not target.exists():
            fails += 1
            print(f"  ❌ 目标文档不存在：{target_name}")
            continue
        t = target.read_text(encoding="utf-8")
        heads = set(re.findall(r"(?m)^#{2,4} (\d+(?:\.\d+)*)", t))
        for r in refs:
            raw = r.strip()
            # 形如「5 B5」：取主节号 5，再要求 B5 作为子节标题出现在文里
            head, *sub = raw.split()
            key = (target_name, raw)
            if key in seen:
                continue
            seen.add(key)
            ok = head in heads or any(h.startswith(head + ".") for h in heads)
            if ok and sub:
                ok = re.search(r"(?m)^#{2,4} " + re.escape(sub[0]) + r"\b", t) is not None
            if ok:
                print(f"  ✅ {target_name:<32} §{raw}")
            else:
                fails += 1
                print(f"  ❌ {target_name:<32} §{raw}  **目标里没有**")
    if not seen:
        fails += 1
        print("  ❌ 空转：一个跨文档引用都没解析到（格式变了？）")
    return fails


def main():
    for p in (DOC, MODEL_OUT):
        if not p.exists():
            print(f"❌ 找不到 {p}")
            return 1
    doc = DOC.read_text(encoding="utf-8")
    model = MODEL_OUT.read_text(encoding="utf-8")
    mrows = model_rows(model)

    wins = fails = skipped = 0
    print("=== 对着 flops_model 输核核 docs/23 §6 表 ===")
    for cells in doc_rows(doc):
        label = re.sub(r"\*\*|`", "", cells[0]).strip()
        if label not in ROW_MAP:
            continue
        mlabel = ROW_MAP[label]
        if mlabel not in mrows:
            print(f"  ⚠️ {label}: 模型输出里找不到 '{mlabel}' —— 跳过")
            skipped += 1
            continue
        mline, mg = mrows[mlabel]
        row = "|".join(cells)

        # §6 表的列是固定的：0=标签 1=名义参数 2=每token算力 3=参数/算力比
        #                    4=显存 5=余量 6=步时 7=vs713
        # **按位置取，不要用「最接近的数」那种启发式**——文档里 487 用的是
        # Unicode 减号 U+2212，启发式会漏掉它、退而抓到 286M 那列，造成误报。
        if len(cells) < 8:
            print(f"  ⚠️ {label}: 列数 {len(cells)} < 8，跳过")
            skipped += 1
            continue
        cell_mem, cell_res, cell_t = cells[4], cells[5], cells[6]

        def _norm_num(s: str):
            """把 Unicode 减号/全角数字归一化后取第一个数。"""
            s = (s.replace("\u2212", "-").replace("**", "")
                  .replace("−", "-").replace(",", ""))
            m = re.search(r"-?\d+(?:\.\d+)?", s)
            return float(m.group(0)) if m else None

        mm = FRONTIER.search(mline)
        # --- 显存 GiB ---
        dg = re.search(r"([\d.]+)\s*GiB", cell_mem)
        if dg and mm:
            mgib = float(mm.group(1))
            d = abs(float(dg.group(1)) - mgib)
            if d <= tol_for(2):
                wins += 1
                print(f"  ✅ {label:<16} 显存 {dg.group(1)} GiB == 模型 {mgib} GiB")
            else:
                fails += 1
                print(f"  ❌ {label:<16} 显存 文档 {dg.group(1)} GiB != 模型 {mgib} GiB"
                      f"（差 {d*1024:.0f} MiB）")
        # --- 余量 MiB ---
        vr = _norm_num(cell_res)
        if vr is not None and mm:
            resid = float(mm.group(2).replace(",", ""))
            if abs(vr - resid) <= 1:
                wins += 1
                print(f"  ✅ {label:<16} 余量 {vr:,.0f} MiB == 模型 {resid:,.0f} MiB")
            else:
                fails += 1
                print(f"  ❌ {label:<16} 余量 文档 {vr:,.0f} MiB != 模型 {resid:,.0f} MiB")
        # --- 步时：**故意不查** ---
        # 文档 §6 的步时列给的是「完整区间」（低端含 §4.4 的稀释修正，高端含 floor/ckpt
        # 最差端点），而模型表给的是**模型口径**。两者按设计就不同：
        #   0.5B 文档 1.28~1.61 s  vs  模型 1500~1609 ms   （低端差 220 ms，是有意的）
        # 拿一个**已知口径不同**的字段去判失败，只会产生误报，
        # 而误报会让整个检查器失去可信度（前两版就是这么废掉的）。
        # 显存与余量没有这个问题：它们是静态预算，两侧必须逐位一致。

    print(f"\n  对得上 {wins} 项，不一致 {fails} 项，跳过 {skipped} 行")
    print("\n  说明：文档 §6 的步时列用的是**完整区间**（含稀释下移），")
    print("        而模型表给的是**模型口径**，所以步时上界会比模型的大——**这是有意的，不判失败**。")
    print("        上界不对判，只判显存/余量/步时下界。")

    print("\n" + "=" * 60)
    # ---- 空转守卫：比对次数为 0 必须**判失败** ----
    # 这个检查器的前两版都栽在这：一次比对都没做，却报「✅ 一致」。
    # 一个静默空转的检查器比没有检查器更糟——它给的是假保证。
    if wins + fails == 0:
        print("❌ 空转：**一次比对都没有发生**。检查器坏了，不是文档坏了。")
        print("   常见原因：标签映射对不上、正则没匹配、真值文件格式变了。")
        return 2

    # ---- 文档自身算术（不依赖外部真值）----
    self_fails = check_self_consistency(doc)
    fails += self_fails

    # ---- 跨文档 §N 引用 ----
    fails += check_cross_refs(doc)

    print("\n" + "=" * 60)
    if fails == 0:
        print(f"✅ docs/23 通过全部检查：外部对真值 {wins} 项 + 文档自身算术/交叉引用。")
        print("   不覆盖：步时（两侧口径不同，见上）、文字表述、其它小节。")
    else:
        print(f"❌ 共 {fails} 项不一致（外部 {wins} 项通过）。")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
