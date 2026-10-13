"""Shared measurement harness: fixed test sets + a tokenizer scorer.

Test sets are cached to research/tokenizer/out/testsets.json so every tokenizer in
the comparison is scored on byte-identical strings.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, ROOT, char_class, cjk_ratio, iter_jsonl, token_to_bytes  # noqa: E402

BENCH = ROOT / "bench" / "data" / "bench_v3.jsonl"
AG = ROOT / "data_cache" / "a_grade"
TESTSETS = OUT / "out" / "testsets.json"
N_AG_DOCS = 1500


def clean(text: str, limit: int) -> str:
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text[:limit]


def r1_text(obj: dict, max_chars: int = 4000) -> str:
    ins = clean(obj.get("instruction") or obj.get("prompt") or "", 600)
    u = clean(obj.get("input") or "", 400) if obj.get("input") else ""
    raw = obj.get("output") or obj.get("response") or ""
    a = clean(raw, max_chars)
    user = (ins + " " + u).strip() if u else ins
    if not (user and a):
        return ""
    return f"用户：{user[:600]}\n助手：{a[:max_chars]}"


def wiki_text(obj: dict) -> str:
    q = clean(obj.get("prompt") or obj.get("instruction") or obj.get("question") or "", 400)
    a = clean(obj.get("response") or obj.get("output") or obj.get("answer") or "", 800)
    return f"用户：{q}\n助手：{a}" if q and a else ""


def ruozhi_text(obj: dict) -> str:
    q = clean(obj.get("instruction") or obj.get("question") or "", 300)
    a = clean(obj.get("output") or obj.get("answer") or "", 600)
    return f"用户：{q}\n助手：{a}" if q and a else ""


def build_testsets(force: bool = False) -> dict[str, list[str]]:
    if TESTSETS.exists() and not force:
        return json.loads(TESTSETS.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    bench = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]
    out["bench_all"] = ["\n".join(r["turns"]) for r in bench]
    out["bench_zh"] = [t for t in out["bench_all"] if cjk_ratio(t) >= 0.5]
    out["bench_en"] = [t for t in out["bench_all"] if cjk_ratio(t) <= 0.05]
    out["bench_lc"] = ["\n".join(r["turns"]) for r in bench if r["id"].startswith("lc_")]
    for name, path, fn in (("aga_r1", AG / "r1_110k_sft.jsonl", r1_text),
                           ("aga_wiki", AG / "i_wonder_why.jsonl", wiki_text),
                           ("aga_ruozhi", AG / "better_ruozhiba.jsonl", ruozhi_text)):
        texts = []
        for obj in iter_jsonl(path):
            t = fn(obj)
            if t:
                texts.append(t)
            if len(texts) >= N_AG_DOCS:
                break
        out[name] = texts
    TESTSETS.parent.mkdir(parents=True, exist_ok=True)
    TESTSETS.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# ------------------------------------------------------------------ scoring
CODE_FENCE = re.compile(r"```.*?```", re.S)


def _runs(text: str, cls: str, min_len: int = 4) -> list[str]:
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


def score(tok, texts: list[str]) -> dict:
    chars = bytes_ = tokens = 0
    cls_tok = collections.Counter()
    cls_ch = collections.Counter()
    run_ch = collections.Counter()
    run_by = collections.Counter()
    run_tk = collections.Counter()
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False).ids
        if not ids:
            continue
        chars += len(t)
        bytes_ += len(t.encode("utf-8"))
        tokens += len(ids)
        for i in ids:
            b = token_to_bytes(tok.id_to_token(i))
            if b is None:
                continue
            s = b.decode("utf-8", errors="replace")
            cc = collections.Counter(char_class(c) for c in s)
            body = {k: v for k, v in cc.items() if k != "space"}
            k = "space" if not body else (next(iter(body)) if len(body) == 1 else "mixed")
            cls_tok[k] += 1
            cls_ch[k] += len(s)
        for cls in ("cjk", "latin", "digit"):
            for r in _runs(t, cls, 4):
                run_ch[cls] += len(r)
                run_by[cls] += len(r.encode("utf-8"))
                run_tk[cls] += len(tok.encode(r, add_special_tokens=False).ids)
        for blk in CODE_FENCE.findall(t):
            run_ch["code"] += len(blk)
            run_by["code"] += len(blk.encode("utf-8"))
            run_tk["code"] += len(tok.encode(blk, add_special_tokens=False).ids)
    return {
        "docs": len(texts), "chars": chars, "bytes": bytes_, "tokens": tokens,
        "chars_per_token": chars / max(1, tokens),
        "bytes_per_token": bytes_ / max(1, tokens),
        "tokens_per_kchar": tokens / max(1, chars) * 1000,
        "class_tokens": dict(cls_tok), "class_chars": dict(cls_ch),
        "run_chars": dict(run_ch), "run_bytes": dict(run_by), "run_tokens": dict(run_tk),
    }


def main() -> None:
    ts = build_testsets(force="--force" in sys.argv)
    print("testsets:", {k: (len(v), sum(len(x) for x in v)) for k, v in ts.items()})


if __name__ == "__main__":
    main()
