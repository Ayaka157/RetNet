"""只读审计：R1 语料里到底有多少「有 <think> 无 </think>」的未闭合样本。

    python scripts/audit_r1_think.py
    python scripts/audit_r1_think.py --max_chars 4000 --ag_dir data_cache/a_grade

## 为什么要单独测一遍

`docs/17` §0.3 说「清掉/救回那 **9,769** 条未闭合样本（占 R1 的 **8.88%**、全包 **3.6%**）」，
`docs/23` §8.2 把「修数据」排成扩容的第 0 步。但我**一直在引用这个数，没有自己验过**
（本会话已经栽过三次"引用别人的结论"）。本脚本把它测出来，
并给出 `docs/17` 没分开的两个口径：

| 口径 | 什么时候重要 |
|---|---|
| **原始 output** 里未闭合 | 反映源数据质量 |
| **过完 `max_chars` 截断后**未闭合 | **这才是模型真正看到的**——`docs/13` 那 27.7% think 泄漏就是被这个截断造出来的 |

## 做法

**直接 import 真实的 `r1_reader`**（`scripts/build_chat_mix_v10.py`），不重写一遍。
重写就有转写错误的风险，而这类错误正是本脚本要防的东西。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from retnet.packing import clean            # noqa: E402
from build_chat_mix_v10 import r1_reader    # noqa: E402  ← 用真实现，不重写

OPEN_RE = re.compile(r"<think>")
CLOSE_RE = re.compile(r"</think>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag_dir", default="data_cache/a_grade")
    ap.add_argument("--max_chars", type=int, default=4000)
    ap.add_argument("--file", default="r1_110k_sft.jsonl")
    a = ap.parse_args()

    path = ROOT / a.ag_dir / a.file
    if not path.exists():
        print(f"❌ 找不到 {path}")
        return 1

    print(f"审计文件 : {path}  ({path.stat().st_size/1e6:.1f} MB)")
    print(f"max_chars: {a.max_chars}（当前 chat_pack_v13 用的值）")

    n = 0
    raw_open = raw_unclosed = 0          # 原始 output 里
    fin_open = fin_unclosed = 0          # 过完截断后（模型真正看到的）
    capped = 0                           # 被 max_chars 砍过的样本
    examples = []

    for pairs, _ in r1_reader(path, strip_think=False, max_chars=a.max_chars):
        user, ans = pairs[0]
        n += 1

        # --- 过完截断后的（= 真实进训练的那份）---
        has_o = bool(OPEN_RE.search(ans))
        has_c = bool(CLOSE_RE.search(ans))
        if has_o:
            fin_open += 1
            if not has_c:
                fin_unclosed += 1
                if len(examples) < 3:
                    examples.append((user[:50], ans[-90:]))

        # --- 原始 output 里的（重跑 reader 的清洗但不截断）---
        # 这里只看「截断是否把闭合标签切掉了」：ans 已被截断，
        # 所以用 clean(raw, None) 拿到完整版比对
        # （reader 是生成器，拿不到 raw；改用下面的独立采样）
        if len(ans) >= a.max_chars - 1:
            capped += 1

    # 原始 output 的口径单独走一遍 iter_jsonl，避免猜
    from retnet.packing import iter_jsonl
    total_rows = 0
    for obj in iter_jsonl(path):
        total_rows += 1
        raw = obj.get("output") or obj.get("response") or ""
        if OPEN_RE.search(raw):
            raw_open += 1
            if not CLOSE_RE.search(raw):
                raw_unclosed += 1

    print(f"\nJSONL 总行数            : {total_rows:,}")
    print(f"reader 产出的可用样本数  : {n:,}")
    print()
    print("【口径 1】原始 output 里")
    print(f"  含 <think>            : {raw_open:,}  ({raw_open/max(total_rows,1)*100:.2f}% of 总行数)")
    print(f"  **未闭合**            : {raw_unclosed:,}  ({raw_unclosed/max(raw_open,1)*100:.2f}% of 含 think 的)")
    print()
    print("【口径 2】过完 max_chars 截断后（= 模型真正看到的）")
    print(f"  含 <think>            : {fin_open:,}  ({fin_open/max(n,1)*100:.2f}% of 可用样本)")
    print(f"  **未闭合**            : {fin_unclosed:,}  ({fin_unclosed/max(fin_open,1)*100:.2f}% of 含 think 的)")
    print(f"  被 max_chars 砍过      : {capped:,}  ({capped/max(n,1)*100:.2f}%)")
    print()
    print("对照 docs/17 §0.3 的说法：未闭合 9,769（R1 的 8.88%、全包 3.6%）")
    print(f"  本脚本口径 2 得到 {fin_unclosed:,}；占 R1 可用样本 {fin_unclosed/max(n,1)*100:.2f}%")
    if examples:
        print("\n未闭合样本示例（尾部 90 字符，可见 CoT 被砍在半路）：")
        for u, tail in examples:
            print(f"  prompt: {u!r}")
            print(f"    tail: ...{tail!r}")

    # 两种口径差很大 → 说明「修原始数据」和「抬 max_chars」是**两件不同的事**
    if fin_unclosed > raw_unclosed * 1.5:
        print(f"\n⚠️ 口径 2（{fin_unclosed:,}）远大于口径 1（{raw_unclosed:,}）：")
        print("   大部分未闭合是 **max_chars 截断造成的**，不是源数据本身缺闭合标签。")
        print("   → 只过滤源数据**治不了**这个问题；必须同时抬 max_chars 或硬过滤截断后的样本。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
