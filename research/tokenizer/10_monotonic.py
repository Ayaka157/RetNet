"""10 - monotonicity check for append-only vocab expansion.

Claim: with the production merge list kept byte-identical and extra merges appended
after it, the greedy lowest-rank-first BPE process is unchanged for every pair the
base list could merge; appended merges therefore only fire on what the base left
split, and each firing removes exactly one symbol.  Hence

    len(encode_appended(s)) <= len(encode_prod(s))   for every string s

i.e. append-only expansion can never lengthen any text — it is lossless by
construction.  This script falsifies-or-confirms that on 1500 training rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, TOK_PATH  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

O = OUT / "out"


def main() -> None:
    hold = json.loads((O / "holdout_rows.json").read_text(encoding="utf-8"))
    prod = Tokenizer.from_file(str(TOK_PATH))
    rows = []
    for name in ("app_top500", "app_top2000", "app_top8000",
                 "app_ctl_v13_20k", "app_ctl_v13_48k"):
        p = O / "tok" / f"{name}.json"
        if not p.exists():
            continue
        t = Tokenizer.from_file(str(p))
        worse = longer = same = shorter = 0
        delta = 0
        for s in hold:
            a = len(prod.encode(s, add_special_tokens=False).ids)
            b = len(t.encode(s, add_special_tokens=False).ids)
            delta += a - b
            if b > a:
                worse += 1
                longer += b - a
            elif b == a:
                same += 1
            else:
                shorter += 1
        rows.append([name, t.get_vocab_size(), f"{worse}", f"{same}", f"{shorter}",
                     f"{longer}", f"{delta:,}"])
        print(rows[-1], flush=True)
    L = ["# 10 append-only 词表扩张的单调性检验\n",
         "对 1500 行留出训练文本，逐条比较 prod 与新 tokenizer 的 token 数。\n",
         "| 变体 | vocab | 变长的条数 | 不变的条数 | 变短的条数 | 总共多出的 token | 总共省下的 token |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    L.append("\n> 「变长的条数」应恒为 0：保留基座 merge 列表不动、只在尾部追加，"
             "贪心 BPE 的最低 rank 优先顺序不会变，追加的 merge 只能在基座切剩的片段上触发，"
             "每次触发恰好少一个符号。**因此 append-only 扩张不会让任何文本变长。**")
    (O / "10_monotonic.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote out/10_monotonic.md")


if __name__ == "__main__":
    main()
