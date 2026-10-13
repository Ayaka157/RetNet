"""02 - token efficiency profile of the v13 tokenizer on real corpora + the bench.

Measured, per test set:
  * chars/token and bytes/token overall
  * the same split by content class (cjk / latin / digit / code), where each class
    is measured on maximal class-pure runs extracted from the very same documents
  * the token-budget share each class consumes (exact, by classifying every token)

Test sets:
  A bench_v3       : all 1020 items, `turns` joined exactly as bench_v3.py feeds them
  B corpus_r1      : sampled docs from data_cache/a_grade/r1_110k_sft.jsonl
  C corpus_wiki    : sampled docs from data_cache/a_grade/i_wonder_why.jsonl
  D corpus_ruozhi  : data_cache/a_grade/better_ruozhiba.jsonl (all)
  E stream_decoded : rows decoded back out of data_cache/chat_pack_v13/train.ids.npy
The segmentation for B/C/D is built with the project's own readers + retnet.chat so
the strings are byte-identical to what build_chat_mix_v13.py feeds the tokenizer.

CPU only.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common import (OUT, ROOT, TOK_PATH, char_class, cjk_ratio, fmt_table,  # noqa: E402
                    iter_jsonl, token_to_bytes)

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from retnet.chat import system_prefix, turn_prefix  # noqa: E402

BENCH = ROOT / "bench" / "data" / "bench_v3.jsonl"
AG = ROOT / "data_cache" / "a_grade"
N_DOCS = 1500
CODE_FENCE = re.compile(r"```.*?```", re.S)


# ------------------------------------------------------------------ helpers
def clean(text: str, limit: int) -> str:
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text[:limit]


def r1_segments(obj: dict, max_chars: int = 4000) -> list[str]:
    """Exact re-implementation of build_chat_mix_v10.r1_reader's text content."""
    ins = clean(obj.get("instruction") or obj.get("prompt") or "", 600)
    u = clean(obj.get("input") or "", 400) if obj.get("input") else ""
    raw = obj.get("output") or obj.get("response") or ""
    a = clean(raw, max_chars)
    user = (ins + " " + u).strip() if u else ins
    if not (user and a):
        return []
    return [turn_prefix(user[:600], first=True), a[:max_chars]]


def wiki_segments(obj: dict) -> list[str]:
    q = clean(obj.get("prompt") or obj.get("instruction") or obj.get("question") or "", 400)
    a = clean(obj.get("response") or obj.get("output") or obj.get("answer") or "", 800)
    if not (q and a):
        return []
    return [turn_prefix(q, first=True), a]


def ruozhi_segments(obj: dict) -> list[str]:
    q = clean(obj.get("instruction") or obj.get("question") or "", 300)
    a = clean(obj.get("output") or obj.get("answer") or "", 600)
    if not (q and a):
        return []
    return [turn_prefix(q, first=True), a]


def class_of_token(tb: bytes) -> str:
    s = tb.decode("utf-8", errors="replace")
    cc = collections.Counter(char_class(c) for c in s)
    body = {k: v for k, v in cc.items() if k != "space"}
    if not body:
        return "space"
    if len(body) == 1:
        return next(iter(body))
    return "mixed"


def runs_of_class(text: str, cls: str, min_len: int = 4) -> list[str]:
    """Maximal runs where every char is of `cls` (spaces allowed inside)."""
    out, cur = [], []
    for ch in text:
        c = char_class(ch)
        if c == cls or (c == "space" and cur and char_class(cur[-1]) == cls):
            cur.append(ch)
        else:
            if len(cur) >= min_len:
                out.append("".join(cur))
            cur = []
    if len(cur) >= min_len:
        out.append("".join(cur))
    return out


# ------------------------------------------------------------------ measurement
class Acc:
    def __init__(self, name: str):
        self.name = name
        self.docs = 0
        self.chars = 0
        self.bytes = 0
        self.tokens = 0
        self.by_class_chars = collections.Counter()
        self.by_class_bytes = collections.Counter()
        self.by_class_tokens = collections.Counter()
        self.run_chars = collections.Counter()
        self.run_bytes = collections.Counter()
        self.run_tokens = collections.Counter()
        self.run_count = collections.Counter()

    def add_doc(self, tok: Tokenizer, text: str) -> None:
        if not text:
            return
        ids = tok.encode(text, add_special_tokens=False).ids
        self.docs += 1
        self.chars += len(text)
        self.bytes += len(text.encode("utf-8"))
        self.tokens += len(ids)
        for i in ids:
            t = tok.id_to_token(i)
            b = token_to_bytes(t)
            if b is None:
                continue
            c = class_of_token(b)
            self.by_class_tokens[c] += 1
            s = b.decode("utf-8", errors="replace")
            self.by_class_chars[c] += len(s)
            self.by_class_bytes[c] += len(b)

    def add_runs(self, tok: Tokenizer, text: str, cls: str, min_len: int = 4) -> None:
        for r in runs_of_class(text, cls, min_len):
            n = len(tok.encode(r, add_special_tokens=False).ids)
            self.run_count[cls] += 1
            self.run_chars[cls] += len(r)
            self.run_bytes[cls] += len(r.encode("utf-8"))
            self.run_tokens[cls] += n

    def row(self) -> list:
        c = self.chars
        return [self.name, f"{self.docs:,}", f"{c:,}", f"{self.bytes:,}", f"{self.tokens:,}",
                f"{c / max(1, self.tokens):.3f}", f"{self.bytes / max(1, self.tokens):.3f}",
                f"{self.tokens / max(1, c) * 1000:.1f}"]

    def class_rows(self, cls: str) -> list:
        ct = self.by_class_tokens[cls]
        return [self.name, cls, f"{self.by_class_chars[cls]:,}", f"{ct:,}",
                f"{self.by_class_chars[cls] / max(1, ct):.3f}",
                f"{self.by_class_bytes[cls] / max(1, ct):.3f}",
                f"{ct / max(1, self.tokens) * 100:.2f}%"]

    def run_row(self, cls: str) -> list:
        if not self.run_count[cls]:
            return [self.name, cls, 0, 0, 0, "-", "-", "-"]
        return [self.name, cls, f"{self.run_count[cls]:,}", f"{self.run_chars[cls]:,}",
                f"{self.run_tokens[cls]:,}",
                f"{self.run_chars[cls] / self.run_tokens[cls]:.3f}",
                f"{self.run_bytes[cls] / self.run_tokens[cls]:.3f}",
                f"{self.run_tokens[cls] / self.run_chars[cls] * 1000:.1f}"]


def main() -> None:
    tok = Tokenizer.from_file(str(TOK_PATH))
    accs: dict[str, Acc] = {}
    docs_for_filter: list[tuple[str, str]] = []   # (set_name, text) for code fences
    lc_detail: list[dict] = []
    bench_per_item: list[dict] = []

    # ---------------------------------------------------------- A bench_v3
    A = accs["bench_v3"] = Acc("bench_v3")
    bench = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in bench:
        text = "\n".join(r["turns"])
        A.add_doc(tok, text)
        for cls in ("cjk", "latin", "digit"):
            A.add_runs(tok, text, cls, 4)
        for blk in CODE_FENCE.findall(text):
            A.add_runs(tok, blk, "code", 1)
        bench_per_item.append({
            "id": r["id"], "cat": r["cat"], "family": r.get("family"), "tier": r["tier"],
            "chars": len(text), "bytes": len(text.encode("utf-8")),
            "tokens": len(tok.encode(text, add_special_tokens=False).ids),
            "cjk_ratio": round(cjk_ratio(text), 4),
        })
        if r["id"].startswith("lc_"):
            lc_detail.append(bench_per_item[-1] | {"note": r.get("note", "")})

    # ---------------------------------------------------------- B/C/D corpus
    specs = [("corpus_r1", AG / "r1_110k_sft.jsonl", r1_segments),
             ("corpus_wiki", AG / "i_wonder_why.jsonl", wiki_segments),
             ("corpus_ruozhi", AG / "better_ruozhiba.jsonl", ruozhi_segments)]
    for name, path, segf in specs:
        acc = accs[name] = Acc(name)
        n = 0
        for obj in iter_jsonl(path):
            segs = segf(obj)
            if not segs:
                continue
            text = "\n".join(segs)
            acc.add_doc(tok, text)
            for cls in ("cjk", "latin", "digit"):
                acc.add_runs(tok, text, cls, 4)
            for blk in CODE_FENCE.findall(text):
                acc.add_runs(tok, blk, "code", 1)
            if n < 400:
                docs_for_filter.append((name, text))
            n += 1
            if n >= N_DOCS:
                break

    # ---------------------------------------------------------- E stream decode
    ids_path = ROOT / "data_cache" / "chat_pack_v13" / "train.ids.npy"
    arr = np.load(ids_path, mmap_mode="r")
    E = accs["stream_decoded"] = Acc("stream_decoded")
    rng = np.random.default_rng(20260418)
    rows = np.sort(rng.choice(arr.shape[0], size=2000, replace=False))
    rt_exact = rt_diff = 0
    for ri in rows:
        row = np.asarray(arr[int(ri)])
        ids = [int(x) for x in row]
        while ids and ids[-1] == tok.token_to_id("<pad>"):
            ids.pop()
        if not ids:
            continue
        text = tok.decode(ids, skip_special_tokens=True)
        E.add_doc(tok, text)
        re_ids = tok.encode(text, add_special_tokens=False).ids
        if len(re_ids) == len(ids):
            rt_exact += 1
        else:
            rt_diff += 1

    # ---------------------------------------------------------- report
    L: list[str] = []
    L.append("# 02 分词效率画像（全部实测）\n")
    L.append("所有数字由 `02_efficiency.py` 在同一 tokenizer "
             "`data_cache/chat_pack_v13/bpe_tokenizer.json` 上跑出。\n")
    L.append("## 总体 chars/token、bytes/token\n")
    L.append(fmt_table([a.row() for a in accs.values()],
                       ["测试集", "文档数", "字符数", "字节数", "token 数",
                        "chars/token", "bytes/token", "token/千字符"]))
    L.append("\n- `bench_v3` = 全 1020 题，`\\n`.join(turns)，即 bench 实际喂给模型的文本。")
    L.append(f"- `corpus_r1` / `corpus_wiki` = `data_cache/a_grade/` 原始 jsonl 前 "
             f"{N_DOCS} 条有效文档，经 build_chat_mix_v10 的 reader + `retnet.chat.turn_prefix` "
             f"拼成与训练完全一致的字符串。")
    L.append("- `corpus_ruozhi` = better_ruozhiba.jsonl 全量（1475 条）。")
    L.append("- `stream_decoded` = 从 train.ids.npy 随机抽 2000 行，去 pad 后用 ByteLevel "
             "decoder 解回文本再测。\n")

    L.append("## 按字符类别拆分的 token 预算（对每个 token 解码后归类，精确可加）\n")
    rows = []
    for a in accs.values():
        for cls in ("cjk", "latin", "digit", "punct", "mixed", "space", "other"):
            if a.by_class_tokens[cls]:
                rows.append(a.class_rows(cls))
    L.append(fmt_table(rows, ["测试集", "类别", "该类别字符数", "该类别 token 数",
                              "chars/token", "bytes/token", "占该集 token 预算"]))
    L.append("\n说明：token 数是该类别 token 的**全部**计数（一个 token 只归一类）。"
             "纯 CJK token 里既有 1 个汉字也有 4 个汉字，所以这里的 chars/token 直接度量"
             "该类别内容被压得多紧。\n")

    L.append("## 类别纯净 run 上的 chars/token（同一批文档里抽最大同类 run，单独编码）\n")
    rows = []
    for a in accs.values():
        for cls in ("cjk", "latin", "digit", "code"):
            rows.append(a.run_row(cls))
    L.append(fmt_table(rows, ["测试集", "类别", "run 数", "run 字符数", "run token 数",
                              "chars/token", "bytes/token", "token/千字符"]))
    L.append("\n- cjk / latin / digit run：长度 ≥4 的同类字符极大 run（空格可包含在 run 内）。")
    L.append("- code run：文本中 ``` 围栏代码块的整块（`min_len=1`）。\n")

    # language split of bench
    L.append("## bench_v3 按题目类别\n")
    bycat = collections.defaultdict(lambda: [0, 0, 0])
    for it in bench_per_item:
        b = bycat[it["cat"]]
        b[0] += 1
        b[1] += it["chars"]
        b[2] += it["tokens"]
    rows = [[k, v[0], f"{v[1]:,}", f"{v[2]:,}", f"{v[1] / max(1, v[2]):.3f}"]
            for k, v in sorted(bycat.items(), key=lambda x: -x[1][2])]
    L.append(fmt_table(rows, ["cat", "题数", "字符", "token", "chars/token"]))

    L.append("\n## bench_v3 按文本语言（CJK 字符占比）\n")
    bins = [(0.0, 0.05, "纯英文 ≤5% CJK"), (0.05, 0.5, "英文为主 5-50%"),
            (0.5, 0.95, "中文为主 50-95%"), (0.95, 1.01, "纯中文 ≥95%")]
    rows = []
    for lo, hi, name in bins:
        sel = [it for it in bench_per_item if lo <= it["cjk_ratio"] < hi]
        if not sel:
            continue
        ch = sum(i["chars"] for i in sel)
        tk = sum(i["tokens"] for i in sel)
        by = sum(i["bytes"] for i in sel)
        rows.append([name, len(sel), f"{ch:,}", f"{by:,}", f"{tk:,}",
                     f"{ch / tk:.3f}", f"{by / tk:.3f}"])
    L.append(fmt_table(rows, ["语言档", "题数", "字符", "字节", "token",
                              "chars/token", "bytes/token"]))

    # slowest English items
    en = sorted([i for i in bench_per_item if i["cjk_ratio"] <= 0.05],
                key=lambda x: x["chars"] / max(1, x["tokens"]))[:10]
    L.append("\n### 英文题里 chars/token 最低的 10 条\n")
    L.append(fmt_table([[i["id"], i["cat"], i["chars"], i["tokens"],
                         f"{i['chars'] / i['tokens']:.3f}"] for i in en],
                       ["id", "cat", "字符", "token", "chars/token"]))
    L.append("")
    en_best = sorted([i for i in bench_per_item if i["cjk_ratio"] <= 0.05],
                     key=lambda x: -(x["chars"] / max(1, x["tokens"])))[:10]
    L.append("### 英文题里 chars/token 最高的 10 条\n")
    L.append(fmt_table([[i["id"], i["cat"], i["chars"], i["tokens"],
                         f"{i['chars'] / i['tokens']:.3f}"] for i in en_best],
                       ["id", "cat", "字符", "token", "chars/token"]))

    # long context detail
    L.append("\n## 长上下文题（lc_*）逐条明细\n")
    L.append(fmt_table([[i["id"], i["chars"], i["tokens"], f"{i['chars'] / i['tokens']:.3f}",
                         f"{i['cjk_ratio'] * 100:.0f}%", i["note"][:42]] for i in lc_detail],
                       ["id", "字符", "token", "chars/token", "CJK 占比", "note"]))
    L.append(f"\nlc_* 汇总：{sum(i['chars'] for i in lc_detail):,} 字符 → "
             f"{sum(i['tokens'] for i in lc_detail):,} token，"
             f"合计 {sum(i['chars'] for i in lc_detail) / sum(i['tokens'] for i in lc_detail):.3f} chars/token")

    (OUT / "out" / "02_efficiency.md").write_text("\n".join(L), encoding="utf-8")

    detail = {
        "overall": {a.name: {"docs": a.docs, "chars": a.chars, "bytes": a.bytes,
                             "tokens": a.tokens,
                             "chars_per_token": a.chars / max(1, a.tokens),
                             "bytes_per_token": a.bytes / max(1, a.tokens),
                             "class_tokens": dict(a.by_class_tokens),
                             "class_chars": dict(a.by_class_chars),
                             "run_tokens": dict(a.run_tokens),
                             "run_chars": dict(a.run_chars),
                             "run_bytes": dict(a.run_bytes),
                             "run_count": dict(a.run_count)}
                   for a in accs.values()},
        "bench_per_item": bench_per_item,
        "round_trip": {"rows": len(rows), "exact": rt_exact, "diff": rt_diff},
        "n_docs_corpus": N_DOCS,
    }
    (OUT / "out" / "02_efficiency.json").write_text(
        json.dumps(detail, ensure_ascii=False), encoding="utf-8")
    print("wrote out/02_efficiency.{md,json}")
    for a in accs.values():
        print(a.row())
    print("round-trip re-encode:", rt_exact, "exact /", rt_diff, "differ /", len(rows))


if __name__ == "__main__":
    main()
