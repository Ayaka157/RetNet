"""03b - fine-grained token budget of the real training stream.

Reads the id histogram cached by 03_stream_census.py (out/03_counts.npz) so it is
instant.  Splits the coarse "cjk" class of 03 into CJK ideographs / CJK punctuation /
kana-hangul, which matters because CJK punctuation costs a whole token per character.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, TOK_PATH, token_to_bytes  # noqa: E402

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def fine_class(ch: str) -> str:
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF \
            or 0x20000 <= o <= 0x2FA1F:
        return "cjk_ideo"
    if 0x3000 <= o <= 0x303F or 0xFF01 <= o <= 0xFF20 or 0xFF3B <= o <= 0xFF40 \
            or 0xFF5B <= o <= 0xFF65 or o in (0x2018, 0x2019, 0x201C, 0x201D, 0x2014, 0x2026):
        return "cjk_punct"
    if 0x3040 <= o <= 0x30FF:
        return "kana"
    if 0xAC00 <= o <= 0xD7AF:
        return "hangul"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    import unicodedata
    cat = unicodedata.category(ch)
    if cat.startswith("P") or cat.startswith("S"):
        return "punct"
    return "other"


def main() -> None:
    tok = Tokenizer.from_file(str(TOK_PATH))
    z = np.load(OUT / "out" / "03_counts.npz")
    counts = z["counts"]
    pad, bos, eos, unk = (int(x) for x in z["specials"])
    V = len(counts)
    SPECIAL = {0: "pad", 1: "bos", 2: "eos", 3: "unk"}
    total = int(counts.sum())

    ct = collections.Counter()
    cb = collections.Counter()
    cc = collections.Counter()
    multi = collections.Counter()
    for i in range(V):
        c = int(counts[i])
        if c == 0:
            continue
        if i in SPECIAL:
            ct[SPECIAL[i]] += c
            continue
        b = token_to_bytes(tok.id_to_token(i))
        if b is None:
            ct["non_bytelevel"] += c
            continue
        s = b.decode("utf-8", errors="replace")
        body = [fine_class(ch) for ch in s if not ch.isspace()]
        if not body:
            k = "space"
        else:
            u = set(body)
            k = body[0] if len(u) == 1 else "mixed"
        ct[k] += c
        cb[k] += len(b) * c
        cc[k] += len(s) * c
        if len(s) > 1:
            multi[k] += c

    content = total - pad - bos - eos - unk
    L = ["# 03b 真实训练流的细粒度 token 预算\n",
         f"数据源：`out/03_counts.npz`（对 `train.ids.npy` 全量 552,330,489 槽位的精确 id 直方图）。",
         f"总槽位 {total:,}；非 special 内容 token = **{content:,}**。\n",
         "| 类别 | token 数 | 占非 special | 字符数 | bytes/token | chars/token |",
         "|---|---|---|---|---|---|"]
    rows = []
    for k, c in ct.most_common():
        if k in ("pad", "bos", "eos", "unk"):
            share = f"({c / total * 100:.2f}% 全槽位)"
        else:
            share = f"{c / content * 100:.2f}%"
        rows.append([k, f"{c:,}", share, f"{cc[k]:,}",
                     f"{cb[k] / c:.3f}" if c else "-", f"{cc[k] / c:.3f}" if c else "-"])
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    tot_cb = sum(cb.values())
    tot_cc = sum(cc.values())
    L.append(f"\n- 非 special token 承载的字符总量 = {tot_cc:,}，字节总量 = {tot_cb:,}")
    L.append(f"- 由此得内容侧整体效率：**{tot_cc / content:.3f} chars/token**，"
             f"**{tot_cb / content:.3f} bytes/token**")
    L.append(f"- 含 pad/bos/eos 的原始效率：{tot_cc / total:.3f} chars/token（pad 不承载字符）")
    L.append("\n## 类别内部的多字符 token 占比（衡量该类别被压得多紧）\n")
    L.append("| 类别 | token 数 | 其中 len>1 的 token 数 | 占比 |")
    L.append("|---|---|---|---|")
    for k, c in ct.most_common():
        if multi[k] or c:
            L.append(f"| {k} | {c:,} | {multi[k]:,} | {multi[k] / c * 100:.2f}% |")
    (OUT / "out" / "03b_fine_census.md").write_text("\n".join(L), encoding="utf-8")
    json.dump({"tokens": dict(ct), "bytes": dict(cb), "chars": dict(cc),
               "multi": dict(multi), "total_slots": total, "content_tokens": content,
               "content_chars": tot_cc, "content_bytes": tot_cb},
              open(OUT / "out" / "03b_fine_census.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("wrote out/03b_fine_census.{md,json}")
    for k, c in ct.most_common():
        if k in ("pad", "bos", "eos", "unk"):
            continue
        print(f"{k:12s} tokens={c:>12,} ({c / content * 100:5.2f}%) chars={cc[k]:>12,} "
              f"chars/tok={cc[k] / c:.3f} bytes/tok={cb[k] / c:.3f}")
    print(f"content chars/token={tot_cc / content:.4f} bytes/token={tot_cb / content:.4f}")


if __name__ == "__main__":
    main()
