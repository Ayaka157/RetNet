"""Deterministic generator for the mechanical categories of battery v3.

Four of the fifteen categories are *families over pools* rather than one-off
authored prose: 填空, 问答, 语言, 续写.  Writing them by hand would be both slow
and redundant, so the raw material lives in ``bench/pools/*.json`` (facts,
idioms, poems, word pairs, openings) and this module expands it into ~390
gradable items with a fixed, reproducible allocation.

Determinism rules:

* pool entries are consumed in order from a ``start`` offset, wrapping around,
  so the same pack always yields the same items;
* tier allocation uses the largest-remainder method against the target
  E/M/H split, so per-category totals are exact rather than approximately right;
* every anchor that already occurs in its own prompt is dropped, because such
  an item is passed for free.

Tiers here mean: ``E`` single-hop recall, ``M`` two-hop or a second constraint,
``H`` three hops, arithmetic with structure, or a long free-form answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .registry import CATEGORY_TARGET, Item, anchor_problems

POOLS = Path(__file__).resolve().parent / "pools"
TIER_ORDER = ("E", "M", "H")


# ------------------------------------------------------------------- utilities


def largest_remainder(total: int, shares: Dict[str, float]) -> Dict[str, int]:
    """Split ``total`` over keys of ``shares`` by largest remainder."""
    keys = list(shares)
    raw = {k: total * shares[k] for k in keys}
    out = {k: int(v) for k, v in raw.items()}
    rest = total - sum(out.values())
    for k in sorted(keys, key=lambda k: (raw[k] - out[k], -keys.index(k)), reverse=True):
        if rest <= 0:
            break
        out[k] += 1
        rest -= 1
    return out


def tier_split(total: int) -> Dict[str, int]:
    return largest_remainder(total, {"E": 0.20, "M": 0.50, "H": 0.30})


def pick(pool: Sequence, n: int, start: int = 0) -> List:
    """``n`` entries from ``pool`` at a fixed stride, wrapping around."""
    if not pool:
        return []
    m = len(pool)
    if n >= m:
        return [pool[(start + i) % m] for i in range(n)]
    stride = max(1, m // n)
    return [pool[(start + i * stride) % m] for i in range(n)]


def alts(answer: str) -> List[str]:
    return [a for a in answer.split("|") if a]


def anchor_in(text: str, *cands: str) -> bool:
    return any(c and c in text for c in cands)


def _mk(iid: str, cat: str, tier: str, family: str, turns: Sequence[str],
        max_new: int, chk: str, axes: Sequence[str], note: str,
        system: Optional[str] = None, temp: Optional[float] = None) -> Dict:
    d = {"id": iid, "cat": cat, "tier": tier, "family": family,
         "turns": list(turns), "max_new": max_new, "chk": chk,
         "axes": list(axes), "note": note}
    if system:
        d["system"] = system
    if temp is not None:
        d["temp"] = temp
    return d


class Builder:
    """Accumulates items while enforcing id uniqueness and anchor sanity."""

    def __init__(self, cat: str, target: int):
        self.cat = cat
        self.target = target
        self.rows: List[Dict] = []
        self.ids: set = set()
        self.tiers: Dict[str, int] = {t: 0 for t in TIER_ORDER}
        self.dropped: List[str] = []
        self.quota: Dict[str, int] = tier_split(target)

    def add(self, row: Dict) -> bool:
        iid = row["id"]
        if iid in self.ids:
            self.dropped.append(f"dup id {iid}")
            return False
        prompt = "\n".join(row["turns"])
        free = anchor_problems(iid, row["chk"], "\n".join(row["turns"][:-1]),
                               row["turns"][-1], row.get("family", ""))
        if free:
            self.dropped.append(free[0].split(": ", 1)[-1])
            return False
        self.ids.add(iid)
        self.rows.append(row)
        self.tiers[row["tier"]] += 1
        return True

    def full(self) -> bool:
        return len(self.rows) >= self.target

    def take(self, tier: str) -> bool:
        """True while this tier still has room (keeps the split on target)."""
        return self.tiers[tier] < self.quota[tier]

    def report(self) -> Dict:
        return {"cat": self.cat, "n": len(self.rows), "target": self.target,
                "tiers": dict(self.tiers), "quota": dict(self.quota),
                "dropped": list(self.dropped)}


# --------------------------------------------------------------- pack loading


def load_packs(pools: Path = POOLS) -> Tuple[Dict, Dict]:
    kp = json.loads((pools / "knowledge_pack.json").read_text(encoding="utf-8"))
    lp = json.loads((pools / "language_pack.json").read_text(encoding="utf-8"))
    return kp, lp


def _sections(pack: Dict, key: str, kind: Optional[str] = None) -> List[Dict]:
    out = []
    for s in pack.get(key, []):
        if kind is not None and s.get("kind") != kind:
            continue
        out.append(s)
    return out


# ------------------------------------------------------------------ 填空 cloze
# The pool is the source of truth for section names, sizes and difficulty: the
# pack declares a tier per section and roughly 12 zh / 7 en entries each, and
# the generator must not hard-code a parallel list of ids that can drift out of
# sync with it (an early version did, and silently produced zero items).

CLOZE_ZH_SHARE = 0.62          # the item set is bilingual, zh-weighted


def gen_cloze(pack: Dict, b: Builder) -> None:
    secs = _sections(pack, "cloze")
    if not secs:
        b.dropped.append("cloze: pool has no sections")
        return
    # Per-section capacity, then scale the requested total down to what the
    # pools can actually supply.  Demanding 120 from a 254-entry pack and then
    # reporting "pool exhausted" 14 times is a generator bug, not a data gap.
    cap = {s["id"]: len(s.get("zh", [])) + len(s.get("en", [])) for s in secs}
    total_cap = sum(cap.values())
    want_total = min(b.target, total_cap)
    if want_total < b.target:
        b.dropped.append(f"cloze: pools supply {total_cap} items, "
                         f"target was {b.target}")
    counts = largest_remainder(want_total, cap)

    plan: List[Tuple[Dict, str, int]] = []
    for sec in secs:
        want = counts.get(sec["id"], 0)
        if want <= 0:
            continue
        zh, en = sec.get("zh", []), sec.get("en", [])
        n_zh = min(len(zh), int(round(want * CLOZE_ZH_SHARE)))
        n_en = min(len(en), want - n_zh)
        if n_en < want - n_zh:                  # English pool short: top up zh
            n_zh = min(len(zh), want - n_en)
        plan.append((sec, "zh", n_zh))
        plan.append((sec, "en", n_en))

    tiers = _tier_schedule(b.target)
    ti = 0
    for sec, lang, n in plan:
        if n <= 0:
            continue
        pairs = sec.get(lang, [])
        hint = sec.get("tier")
        for i, (prefix, answer) in enumerate(pairs[:n]):
            if b.full():
                return
            tier = _next_tier(b, tiers, ti, hint)
            ti += 1
            opts = alts(answer)
            guard = "dedup:0.65,4,40" if len(prefix) > 12 else None
            chk = ";".join(filter(None, [
                f"any:{','.join(opts)}" if len(opts) > 1 else f"any:{opts[0]}",
                "len:1..60",
                "not_think",
                guard,
            ]))
            iid = f"clz_{sec['id']}_{lang}{i + 1:02d}"
            # Kept as a bare incomplete sentence, deliberately, even though the
            # first full run showed this model cannot do it (0/120, and it echoes
            # an explicit "（只输出横线上的内容）" instruction back verbatim —
            # see docs/14 §9.2).  Rewriting these into questions would inflate the
            # score and destroy the evidence that "填空" is a format this model
            # has no representation for.  No "___" marker: it was never in the
            # training data either.
            b.add(_mk(iid, "填空", tier, f"clz_{sec['id']}",
                      [prefix], 32, chk, sec.get("axes") or ["factual"],
                      f"{lang} cloze · {sec['id']} · ans={opts[0]}"))


def _tier_schedule(total: int) -> List[str]:
    """A flat E/M/H schedule in the 20/50/30 design ratio."""
    split = tier_split(total)
    out = ["E"] * split["E"] + ["M"] * split["M"] + ["H"] * split["H"]
    while len(out) < total:
        out.append("M")
    # interleave so the schedule is not consumed in blocks
    order = {"E": 0, "M": 1, "H": 2}
    out.sort(key=lambda t: order[t])
    mixed: List[str] = []
    buckets = {t: [x for x in out if x == t] for t in TIER_ORDER}
    while any(buckets.values()):
        for t in TIER_ORDER:
            if buckets[t]:
                mixed.append(buckets[t].pop(0))
    return mixed


def _next_tier(b: Builder, schedule: Sequence[str], i: int,
               hint: Optional[str]) -> str:
    """Take the next scheduled tier that still has room; fall back to the hint.

    The schedule keeps the category-level E/M/H mix on target; the hint keeps a
    section's own intent (an ``E`` pool section should not emit ``H`` items)
    when the schedule has already filled that tier.
    """
    for j in range(len(schedule)):
        t = schedule[(i + j) % len(schedule)]
        if b.take(t):
            return t
    if hint in TIER_ORDER:
        return hint
    for t in TIER_ORDER:                       # last resort: least-full tier
        if b.tiers[t] < b.quota[t] + 2:
            return t
    return "M"


def _tier_for(hint: Optional[str], i: int, n: int) -> str:
    """Spread items inside a pool section over E/M/H by index."""
    if hint in TIER_ORDER and n <= 3:
        return hint
    frac = (i + 0.5) / max(1, n)
    return "E" if frac < 0.30 else ("M" if frac < 0.80 else "H")


# -------------------------------------------------------------------- 问答 qa
# Allocation is by *kind* (math / fact / open / regex / list), because that is
# what determines the checker and the failure mode.  Which pool sections feed a
# kind is read from the pack, so adding a section never needs a code change.

QA_KIND_WEIGHTS: Dict[str, int] = {
    "fact": 38, "math": 34, "open": 24, "regex": 12, "list": 12,
}
QA_ZH_SHARE = 0.72


def gen_qa(pack: Dict, b: Builder) -> None:
    secs = _sections(pack, "qa")
    if not secs:
        b.dropped.append("qa: pool has no sections")
        return
    kind_counts = largest_remainder(
        b.target, {k: v / sum(QA_KIND_WEIGHTS.values())
                   for k, v in QA_KIND_WEIGHTS.items()})
    per_kind: Dict[str, List[Dict]] = {}
    for s in secs:
        per_kind.setdefault(s.get("kind", "fact"), []).append(s)

    tiers = _tier_schedule(b.target)
    ti = 0
    for kind, count in kind_counts.items():
        pool_secs = per_kind.get(kind) or []
        if not pool_secs:
            b.dropped.append(f"qa kind {kind}: no pool section")
            continue
        ws = {s["id"]: max(1, len(s.get("zh", [])) + len(s.get("en", [])))
              for s in pool_secs}
        per_sec = largest_remainder(count, ws)
        for sec in pool_secs:
            want = per_sec.get(sec["id"], 0)
            if want <= 0:
                continue
            zh, en = list(sec.get("zh", [])), list(sec.get("en", []))
            n_zh = min(len(zh), int(round(want * QA_ZH_SHARE)))
            n_en = want - n_zh
            if n_en > len(en):
                n_en, n_zh = len(en), min(len(zh), want - len(en))
            axes = sec.get("axes") or (["reasoning"] if kind == "math" else ["factual"])
            for lang, entries, n in (("zh", zh, n_zh), ("en", en, n_en)):
                for i, (q, a) in enumerate(pick(entries, n, start=len(sec["id"]) % 5)):
                    if b.full():
                        return
                    tier = _next_tier(b, tiers, ti, sec.get("tier"))
                    ti += 1
                    iid = f"qa_{sec['id']}_{lang}{i + 1:02d}"
                    row = _qa_row(iid, kind, q, a, lang, tier, axes, sec["id"])
                    if row is not None:
                        b.add(row)
            if b.full():
                return


def _qa_row(iid: str, kind: str, q: str, a, lang: str, tier: str,
            axes: Sequence[str], sec: str) -> Optional[Dict]:
    base_axes = list(axes)
    if lang == "en" and "lang_en" not in base_axes:
        base_axes = (base_axes + ["lang_en"])[:3]
    if lang == "zh" and "lang_zh" not in base_axes:
        base_axes = (base_axes + ["lang_zh"])[:3]

    if kind == "math":
        val = a
        if not isinstance(val, (int, float)):
            try:
                val = float(a)
            except (TypeError, ValueError):
                return None
        sv = f"{val:g}"
        chk = ";".join([f"num:{sv}", "len:1..160", "not_think"])
        wrong = _distractor(val)
        if wrong is not None:
            chk += f";!num:{wrong:g}"
        return _mk(iid, "问答", tier, f"qa_{sec}", [q], 72, chk, base_axes,
                   f"{lang} arithmetic; answer={sv}", temp=0.0)

    if kind == "fact":
        opts = alts(str(a))
        if not opts:
            return None
        inside = f"any:{','.join(opts)}"
        chk = ";".join([inside, "len:1..120", "not_think"])
        return _mk(iid, "问答", tier, f"qa_{sec}", [q], 56, chk, base_axes,
                   f"{lang} fact recall; accept={opts[:2]}", temp=0.0)

    if kind == "regex":
        pat = str(a)
        try:
            re.compile(pat)
        except re.error:
            return None
        chk = ";".join([f"re:{pat}", "len:1..160", "not_think"])
        return _mk(iid, "问答", tier, f"qa_{sec}", [q], 72, chk, base_axes,
                   f"{lang} regex-graded answer", temp=0.0)

    if kind == "open":
        cands = [c for c in a if c]
        if not cands:
            return None
        chk = ";".join([f"any:{','.join(cands)}", "len:12..400",
                        "dedup:0.60,4,60", "not_think"])
        return _mk(iid, "问答", tier, f"qa_{sec}", [q], 128, chk, base_axes,
                   f"{lang} open answer; anchors={len(cands)}")

    if kind == "list":
        req = [c for c in a if c]
        if not req:
            return None
        chk = ";".join([f"all:{','.join(req)}", "len:6..400",
                        "dedup:0.60,4,60", "not_think"])
        return _mk(iid, "问答", tier, f"qa_{sec}", [q], 128, chk, base_axes,
                   f"{lang} must name all of {req}")
    return None


def _distractor(val) -> Optional[float]:
    """A plausible wrong answer to forbid, when one exists and is distinct."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v != int(v) or abs(v) > 5000:
        return None
    cand = v * 10 if abs(v) < 100 else v + 10
    return cand if cand != v else None


# ------------------------------------------------------------------ 语言 language

LANG_KINDS: List[Tuple[str, int, str]] = [
    ("idiom", 16, "lang_ix"),
    ("poem", 12, "lang_ix"),
    ("measure", 10, "lang_ix"),
    ("antonym", 10, "lang_ix"),
    ("synonym", 8, "lang_ix"),
    ("pinyin", 8, "lang_ix"),
    ("zh2en", 10, "translate"),
    ("en2zh", 10, "translate"),
    ("pattern", 10, "lang_ix"),
    ("grammar", 6, "lang_ix"),
]


def gen_language(lp: Dict, b: Builder) -> None:
    tiers = _tier_schedule(b.target)
    ti = 0
    idx = 0
    for kind, count, axis in LANG_KINDS:
        pool = list(lp.get(_POOL_KEY[kind], []))
        if not pool:
            b.dropped.append(f"language {kind}: empty pool")
            continue
        for i, entry in enumerate(pick(pool, count, start=idx)):
            if b.full():
                return
            tier = _next_tier(b, tiers, ti, None)
            ti += 1
            iid = f"lang_{kind}_{i + 1:02d}"
            row = _lang_row(iid, kind, entry, tier, axis)
            if row is not None:
                b.add(row)
        idx += 3


_POOL_KEY = {
    "idiom": "idioms", "poem": "poems", "measure": "measure_words",
    "antonym": "antonyms", "synonym": "synonyms", "pinyin": "pinyin",
    "zh2en": "translate_zh2en", "en2zh": "translate_en2zh",
    "pattern": "sentence_patterns", "grammar": "error_correction",
}


def _lang_row(iid: str, kind: str, entry, tier: str, axis: str) -> Optional[Dict]:
    ax = [axis] if axis != "lang_ix" else ["lang_zh"]
    if kind == "idiom":
        word, anchors = entry[0], list(entry[1])
        chk = ";".join([f"any:{','.join(anchors)}", "len:8..200",
                        "dedup:0.60,4,40", "not_think"])
        return _mk(iid, "语言", tier, "lang_idiom", [f"成语「{word}」是什么意思？"],
                   96, chk, ["idiom", "lang_zh"], f"explain 成语 {word}")
    if kind == "poem":
        head, nxt = entry[0], entry[1]
        parts = alts(nxt)
        chk = ";".join([f"any:{','.join(parts)}", "len:1..60", "not_think"])
        return _mk(iid, "语言", tier, "lang_poem", [f"「{head}」的下一句是什么？"],
                   40, chk, ["lang_zh", "idiom"], f"next line after {head[:12]}",
                   temp=0.0)
    if kind == "measure":
        stem, word = entry[0], entry[1]
        chk = ";".join([f"any:{word}", "len:1..40", "not_think"])
        return _mk(iid, "语言", tier, "lang_measure", [f"填量词：{stem}"],
                   24, chk, ["lang_zh"], f"measure word for {stem}")
    if kind == "antonym":
        w, anti = entry[0], entry[1]
        chk = ";".join([f"any:{anti}", "len:1..60", "not_think"])
        return _mk(iid, "语言", tier, "lang_antonym",
                   [f"「{w}」的反义词是什么？"], 32, chk, ["lang_zh"],
                   f"antonym of {w}", temp=0.0)
    if kind == "synonym":
        w, syns = entry[0], list(entry[1])
        chk = ";".join([f"any:{','.join(syns)}", "len:1..80", "not_think"])
        return _mk(iid, "语言", tier, "lang_synonym",
                   [f"「{w}」的近义词是什么？"], 32, chk, ["lang_zh"],
                   f"synonym of {w}")
    if kind == "pinyin":
        w, syls = entry[0], list(entry[1])
        chk = ";".join([f"any:{','.join(syls)}", "len:1..60", "not_think"])
        return _mk(iid, "语言", tier, "lang_pinyin",
                   [f"「{w}」的拼音怎么写？"], 40, chk, ["lang_zh"],
                   f"pinyin of {w}")
    if kind == "zh2en":
        zh, musts = entry[0], list(entry[1])
        chk = ";".join([f"any:{','.join(musts)}", "ascii:0.6,4",
                        "len:2..200", "not_think"])
        return _mk(iid, "语言", tier, "lang_zh2en",
                   [f"把「{zh}」翻译成英文。"], 72, chk, ["translate", "lang_en"],
                   f"zh->en {zh}")
    if kind == "en2zh":
        en, musts = entry[0], list(entry[1])
        chk = ";".join([f"any:{','.join(musts)}", "zh_ratio:0.4,2",
                        "len:2..200", "not_think"])
        return _mk(iid, "语言", tier, "lang_en2zh",
                   [f"把 {en} 翻译成中文。"], 72, chk, ["translate", "lang_zh"],
                   f"en->zh {en}")
    if kind == "pattern":
        a, bb = entry[0], entry[1]
        # `all:` alone is passed for free by an echoed instruction — the prompt
        # itself quotes the two connectives.  `not_echo_prompt` is the guard
        # that makes the item mean "produce a sentence" rather than "repeat the
        # task", so it is required here, not optional.
        chk = ";".join([f"all:{a},{bb}", "len:12..200", "dedup:0.60,4,40",
                        "not_echo_prompt", "not_think"])
        return _mk(iid, "语言", tier, "lang_pattern",
                   [f"用「{a}……{bb}……」造一个句子。"], 96, chk,
                   ["lang_zh", "creative"], f"sentence pattern {a}/{bb}")
    if kind == "grammar":
        bad, fixes = entry[0], list(entry[1])
        chk = ";".join([f"any:{','.join(fixes)}", "len:2..200", "not_think"])
        return _mk(iid, "语言", tier, "lang_grammar",
                   [f"修改这个病句：「{bad}」"], 80, chk, ["lang_zh"],
                   "grammar repair")
    return None


# ---------------------------------------------------------------- 续写 continuation

CONT_TIERS = {"E": (40, 0.55), "M": (70, 0.62), "H": (110, 0.68)}


def gen_continuation(lp: Dict, b: Builder) -> None:
    entries = list(lp.get("continuation", []))
    if not entries:
        b.dropped.append("continuation: empty pool")
        return
    tiers = _tier_schedule(b.target)
    for i, entry in enumerate(pick(entries, b.target, start=1)):
        if b.full():
            return
        prefix, lang = entry[0], entry[1]
        topic = entry[2] if len(entry) > 2 else ""
        tier = _next_tier(b, tiers, i, None)
        lo, ratio = CONT_TIERS[tier]
        parts = [f"len:{lo}..600", f"dedup:{ratio},4,{lo}", "not_think",
                 "not_echo_user"]
        if lang == "en":
            parts.insert(0, "ascii:0.75,20")
            axes = ["creative", "lang_en"]
        else:
            axes = ["creative", "lang_zh"]
        if topic and not anchor_in(prefix, topic):
            parts.append(f"any:{topic}")
        iid = f"cont_{lang}_{i + 1:02d}"
        # Bare prefix for the same reason as 填空: "___" is out of distribution.
        b.add(_mk(iid, "续写", tier, "cont_open", [prefix], 160,
                  ";".join(parts), axes, f"{lang} continuation; topic={topic}"))


# ------------------------------------------------------------------------ main


def build(pools: Path = POOLS) -> Tuple[List[Item], List[Dict]]:
    kp, lp = load_packs(pools)
    builders = {
        "填空": Builder("填空", CATEGORY_TARGET["填空"]),
        "问答": Builder("问答", CATEGORY_TARGET["问答"]),
        "语言": Builder("语言", CATEGORY_TARGET["语言"]),
        "续写": Builder("续写", CATEGORY_TARGET["续写"]),
    }
    gen_cloze(kp, builders["填空"])
    gen_qa(kp, builders["问答"])
    gen_language(lp, builders["语言"])
    gen_continuation(lp, builders["续写"])

    rows: List[Dict] = []
    reports: List[Dict] = []
    for b in builders.values():
        rows.extend(b.rows)
        reports.append(b.report())
    files: List[Item] = []
    for r in rows:
        from .registry import parse_item
        files.append(parse_item(r, f"gen:{r['id']}"))
    return files, reports


def write_generated(out_dir: Path, pools: Path = POOLS) -> List[Dict]:
    """Emit ``bench/items/gen_basics.jsonl`` plus a build report."""
    items, reports = build(pools)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "gen_basics.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            fh.write(json.dumps(it.to_json(), ensure_ascii=False) + "\n")
    return reports
