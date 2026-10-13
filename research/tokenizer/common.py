"""Shared helpers for the tokenizer audit (CPU only, no GPU anywhere)."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent.parent
TOK_PATH = ROOT / "data_cache" / "chat_pack_v13" / "bpe_tokenizer.json"
OUT = Path(__file__).resolve().parent


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / tokenizers ByteLevel byte -> printable-unicode map."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_B2U = _bytes_to_unicode()
U2B = {v: k for k, v in _B2U.items()}


def token_to_bytes(tok: str) -> bytes | None:
    """ByteLevel token string -> raw bytes. None for non-ByteLevel (special) tokens."""
    try:
        return bytes(U2B[c] for c in tok)
    except KeyError:
        return None


# ---------------------------------------------------------------- script classes
def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF        # CJK Unified Ideographs
        or 0x3400 <= o <= 0x4DBF     # Ext A
        or 0xF900 <= o <= 0xFAFF     # Compat Ideographs
        or 0x3000 <= o <= 0x303F     # CJK punctuation
        or 0xFF00 <= o <= 0xFFEF     # Fullwidth forms (halfwidth katakana etc.)
        or 0x20000 <= o <= 0x2FA1F   # Ext B+
        or 0x3040 <= o <= 0x30FF     # kana
        or 0xAC00 <= o <= 0xD7AF     # hangul
    )


def char_class(ch: str) -> str:
    if _is_cjk(ch):
        return "cjk"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    cat = unicodedata.category(ch)
    if cat.startswith("P") or cat.startswith("S"):
        return "punct"
    return "other"


def text_class(s: str) -> str:
    """Dominant character class of a string."""
    counts = {"cjk": 0, "latin": 0, "digit": 0, "space": 0, "punct": 0, "other": 0}
    for ch in s:
        counts[char_class(ch)] += 1
    if not s:
        return "empty"
    # decide by non-space content
    body = {k: v for k, v in counts.items() if k != "space"}
    dom = max(body, key=body.get)
    return dom if body[dom] > 0 else "space"


def cjk_ratio(s: str) -> float:
    body = [c for c in s if not c.isspace()]
    if not body:
        return 0.0
    return sum(1 for c in body if _is_cjk(c)) / len(body)


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def die(msg: str) -> None:
    raise SystemExit(msg)


def fmt_table(rows: list[list], header: list[str]) -> str:
    """Minimal markdown table so every script can emit a report fragment."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
