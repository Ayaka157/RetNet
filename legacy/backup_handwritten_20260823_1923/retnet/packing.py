"""Shared text cleaning, SFT parsing and row-packing helpers for data builders.

Extracted so ``scripts/build_chat_mix_v4.py`` (role-play mixture) and the older
``scripts/build_sft_samples.py`` can use the same battle-tested pieces.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np

csv.field_size_limit(10 ** 9)

Pairs = List[Tuple[str, str]]


# --------------------------------------------------------------------- text
def clean(s, max_chars: int | None = 1200) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ")
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
    s = re.sub(r"\s+(?=[，。！？；：、）》”’…])", "", s)
    s = re.sub(r"(?<=[，。！？；：、（《“‘…])\s+(?=[\u4e00-\u9fff])", "", s)
    # 夹在两个汉字之间的半角标点转全角：修掉「地方,比如说」这类污染，
    # 英文里的逗号句号不受影响。
    HALF2FULL = str.maketrans({",": "，", ".": "。", "!": "！", "?": "？",
                               ";": "；", ":": "："})
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s*([,\.!?;:])\s*(?=[\u4e00-\u9fff])",
               lambda m: m.group(1).translate(HALF2FULL), s)
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars and len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s


def is_fragment(text: str) -> bool:
    """Drop replies that are visibly broken (sentence fragments).

    ``True`` = throw this sample away.  Translation/social-media corpora are full
    of comma-splattered fragments; they teach the model to speak in broken
    sentence fragments.
    """
    t = text.strip()
    if len(t) < 8:
        return True
    if t[-1] in "，,、(" or t[-1] in "（(《“「":
        return True
    # 逗号密度过高且几乎没有句尾标点 = 碎句串
    enders = sum(t.count(ch) for ch in "。！？…!?")
    commas = sum(t.count(ch) for ch in "，,、")
    if commas >= 4 and enders <= 1:
        return True
    return False


def role_of(raw) -> str | None:
    r = str(raw).lower()
    if r in {"user", "human", "用户", "客户", "prompter"}:
        return "user"
    if r in {"assistant", "gpt", "bot", "ai", "助手", "chatgpt", "character"}:
        return "assistant"
    return None


def cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return cjk / len(s)


# ------------------------------------------------------------------ readers
def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def messages_to_pairs(seq, max_chars: int) -> Pairs:
    from retnet.chat import pairs_from_roles
    msgs: List[Tuple[str, str]] = []
    for m in seq:
        if not isinstance(m, dict):
            continue
        if "human" in m or "assistant" in m:          # sharegpt-cn style
            if m.get("human"):
                msgs.append(("user", clean(m["human"], max_chars)))
            if m.get("assistant"):
                msgs.append(("assistant", clean(m["assistant"], max_chars)))
            continue
        role = role_of(m.get("role", m.get("from", "")))
        text = clean(m.get("content", m.get("value", "")), max_chars)
        if role and text:
            msgs.append((role, text))
    return pairs_from_roles(msgs)


def obj_to_pairs(obj: dict, max_chars: int) -> Pairs:
    """Convert one HF SFT record (many possible schemas) into (user, assistant) pairs."""
    from retnet.chat import pairs_from_roles
    if obj and all(k.isupper() for k in obj.keys()):
        obj = {k.lower(): v for k, v in obj.items()}
    for key in ("messages", "conversations", "conversation"):
        seq = obj.get(key)
        if isinstance(seq, list) and seq:
            pairs = messages_to_pairs(seq, max_chars)
            if pairs:
                return pairs

    if "instruction" in obj:
        ins = clean(obj.get("instruction"), max_chars)
        out = clean(obj.get("output", obj.get("response", "")), max_chars)
        if re.search(r"Human\s*[:：]", ins):                   # Belle multi-turn
            msgs = []
            for m in re.finditer(r"(Human|Assistant)\s*[:：]\s*(.*?)(?=(?:Human|Assistant)\s*[:：]|$)",
                                 ins, flags=re.S):
                role = "user" if m.group(1) == "Human" else "assistant"
                text = clean(m.group(2), max_chars)
                if text:
                    msgs.append((role, text))
            if out:
                msgs.append(("assistant", out))
            pairs = pairs_from_roles(msgs)
            if pairs:
                return pairs
        inp = clean(obj.get("input"), max_chars)
        user = f"{ins} {inp}".strip() if inp else ins
        if user and out:
            return [(user, out)]

    for uk, ak in (("question", "answer"), ("prompt", "completion"),
                   ("query", "response"), ("instruction", "response"),
                   ("input", "target"), ("input", "output")):
        u, a = clean(obj.get(uk), max_chars), clean(obj.get(ak), max_chars)
        if u and a:
            return [(u, a)]
    return []


def iter_sft_file(path: Path, max_chars: int) -> Iterator[Pairs]:
    for obj in iter_jsonl(path):
        pairs = obj_to_pairs(obj, max_chars)
        if pairs:
            yield pairs


def iter_lccc(path: Path, stride: int, max_chars: int,
              min_turn_chars: int = 0) -> Iterator[Pairs]:
    """LCCC-base: one JSON list of alternating utterances per line."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if stride > 1 and i % stride:
                continue
            try:
                utts = json.loads(line)
            except Exception:
                continue
            if not isinstance(utts, list) or len(utts) < 2:
                continue
            texts = [clean(u, max_chars) for u in utts]
            texts = [t for t in texts if t]
            if len(texts) < 2:
                continue
            if min_turn_chars and min(len(t) for t in texts) < min_turn_chars:
                continue
            pairs = [(texts[j], texts[j + 1]) for j in range(0, len(texts) - 1, 2)]
            if pairs:
                yield pairs


def iter_glm(path: Path, stride: int, max_chars: int) -> Iterator[Pairs]:
    """GLM open-dialogue CSV (上下文 ... 答案 ... / target_text)."""
    with open(path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if stride > 1 and i % stride:
                continue
            src = row.get("source_text", "") or ""
            tgt = clean(row.get("target_text", ""), max_chars)
            if not tgt:
                continue
            m = re.search(r"上下文[:：]\s*(.*?)\s*答案[:：]", src, flags=re.S)
            utts = []
            if m:
                utts = [clean(x, max_chars) for x in re.split(r"\[SEP\]|\n", m.group(1))]
                utts = [u for u in utts if u]
            if not utts:
                utts = [clean(src, max_chars)]
            if not utts[0]:
                continue
            texts = utts + [tgt]
            pairs = [(texts[j], texts[j + 1]) for j in range(0, len(texts) - 1, 2)]
            if pairs:
                yield pairs


# ------------------------------------------------------------------ packing
class Packer:
    """Best-fit pack variable-length samples into fixed-length rows (int16 ids)."""

    def __init__(self, row_len: int, pad_id: int, open_rows: int = 64,
                 dtype=np.int16):
        self.row_len = row_len
        self.pad_id = pad_id
        self.open_rows = max(1, open_rows)
        self.dtype = dtype
        self.done_ids: List[np.ndarray] = []
        self.done_codes: List[np.ndarray] = []
        self.bins: List[list] = []
        self.tokens = 0
        self.padding = 0
        self.rows_done = 0

    def _new_bin(self):
        return [np.full(self.row_len, self.pad_id, dtype=self.dtype),
                np.zeros(self.row_len, dtype=np.uint8), 0]

    def _retire(self, b):
        self.padding += self.row_len - b[2]
        self.done_ids.append(b[0])
        self.done_codes.append(b[1])
        self.rows_done += 1

    def add(self, ids: Sequence[int], codes: Sequence[int]):
        n = len(ids)
        if n > self.row_len:
            ids, codes, n = ids[:self.row_len], codes[:self.row_len], self.row_len
        best, best_left = -1, self.row_len + 1
        for i, b in enumerate(self.bins):
            left = self.row_len - b[2]
            if n <= left < best_left:
                best, best_left = i, left
        if best < 0:
            if len(self.bins) >= self.open_rows:
                fullest = max(range(len(self.bins)), key=lambda i: self.bins[i][2])
                self._retire(self.bins.pop(fullest))
            self.bins.append(self._new_bin())
            best = len(self.bins) - 1
        b = self.bins[best]
        b[0][b[2]:b[2] + n] = ids
        b[1][b[2]:b[2] + n] = codes
        b[2] += n
        self.tokens += n
        if b[2] >= self.row_len:
            self._retire(self.bins.pop(best))

    def finish(self):
        for b in self.bins:
            if b[2] > 0:
                self._retire(b)
        self.bins = []
        if not self.done_ids:
            return (np.zeros((0, self.row_len), dtype=self.dtype),
                    np.zeros((0, self.row_len), dtype=np.uint8))
        return np.stack(self.done_ids), np.stack(self.done_codes)
