"""Checker DSL — compiles a declarative ``chk`` string into a predicate.

Why a DSL instead of functions
-----------------------------
``scripts/bench_v2.py`` welded prompts and Python checker closures together in
one file, which made the battery impossible to audit or author in bulk.  Here
the item is *data* and the judgement is a *declared spec*:

    "all:len:2..220;not_think;any:令狐冲,华山,师兄"

compiles to a closure that returns ``True`` only when every ``;``-separated
predicate holds.  Compilation happens once, at dataset build time, so a typo in
a checker surfaces as a compile error with the offending item id — never as a
silently-always-false item in the middle of a 1,000-item run.

Grammar
-------
::

    spec    := pred (';' pred)*                 # AND over predicates
    pred    := ['!'] name [':' arglist]
    arglist := arg (',' arg)*                   # OR over args (per-arg predicates
                                                # consume the whole arglist)
    escape  := '\\,'  '\\;'  '\\\\'             # literal comma / semicolon / backslash

The ``!`` prefix negates the predicate.  Combined with the OR-over-args rule,
``!any:喵,汪`` means "contains neither 喵 nor 汪" — the single most common
shape in the battery.

Every predicate receives a :class:`Ctx` carrying the model reply, the encoded
prompt text and the raw user turns, so predicates such as ``not_echo_user`` can
compare against what was asked.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- ctx


@dataclass
class Ctx:
    """Everything a predicate is allowed to look at."""

    reply: str                              # cut + stripped assistant reply
    prompt: str = ""                        # the last user turn, verbatim
    users: Sequence[str] = field(default_factory=tuple)   # every user turn
    system: Optional[str] = None
    n_new: int = 0                          # generated token count

    # v1 of bench_v2 exposed ``raw`` as "untruncated" but truncated anyway.
    # v3 removes the ambiguity: raw is the reply.
    @property
    def raw(self) -> str:
        return self.reply


Predicate = Callable[[Ctx], bool]

_SPEC_CACHE: Dict[str, Predicate] = {}


class CheckerError(ValueError):
    """Raised at compile time for a malformed ``chk`` spec."""


# ------------------------------------------------------------------- splitting


def _split_top(text: str, sep: str) -> List[str]:
    """Split on ``sep``, decoding ``\\x2c`` / ``\\x3b`` to a literal separator.

    **Every other backslash is passed through verbatim**, which is what makes a
    regex written in a ``chk`` string behave the way its author expects.  The
    alternative — "a backslash escapes the next character" — silently eats the
    backslash in ``\\d``, ``\\s``, ``\\(`` and ``\\|``, turning a correct pattern
    into a different pattern that still compiles.  That failure is invisible in
    every direction, so the rule here is deliberately the one with no surprises.

    The DSL therefore claims exactly two sequences:

    ==================  ===========  ==================================
    sequence            becomes      why this spelling
    ==================  ===========  ==================================
    ``\\x2c``           ``,``        one character, so a regex keeps its
                                     length budget and ``{n,m}`` stays put
    ``\\x3b``           ``;``        same
    everything else     unchanged    ``\\d`` stays ``\\d``
    ==================  ===========  ==================================

    The cost is that matching a *literal* comma needs ``\\x2c`` rather than
    ``\\,`` — a fair trade for a DSL whose main payload is regular expressions.
    """
    out, buf, i, n = [], [], 0, len(text)
    while i < n:
        if text.startswith("\\x2c", i):
            buf.append(",")
            i += 4
            continue
        if text.startswith("\\x3b", i):
            buf.append(";")
            i += 4
            continue
        ch = text[i]
        if ch == sep:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    out.append("".join(buf))
    return out


def _split_specs(spec: str) -> List[str]:
    """Split a ``chk`` string into predicates on unescaped ``;``.

    ``\\x3b`` becomes a literal ``;`` **inside** the predicate (so a regex may
    contain one) while ``\\x2c`` is deliberately left encoded: it is the
    *argument* separator for single-value predicates such as ``dedup:`` and
    ``csv:``, and decoding it here would let a literal comma split a value list.
    """
    out, buf, i, n = [], [], 0, len(spec)
    while i < n:
        if spec.startswith("\\x3b", i):
            buf.append(";")
            i += 4
            continue
        ch = spec[i]
        if ch == ";":
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    out.append("".join(buf))
    return out


def _decode(text: str) -> str:
    """Decode the escapes that are meaningful *inside* one predicate."""
    return text.replace("\\x2c", ",")


def _rx(pattern: str) -> "re.Pattern[str]":
    try:
        return re.compile(pattern)
    except re.error as exc:                                  # pragma: no cover
        raise CheckerError(f"bad regex {pattern!r}: {exc}") from exc


def _ints(spec: str, lo: int, hi: int) -> List[int]:
    vals: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError as exc:
            raise CheckerError(f"expected int, got {part!r} in {spec!r}") from exc
    if not lo <= len(vals) <= hi:
        raise CheckerError(f"expected {lo}..{hi} ints in {spec!r}, got {len(vals)}")
    return vals


# ---------------------------------------------------------------- predicate ops


def _json_blocks(text: str) -> List[object]:
    """All parseable ``{...}`` / ``[...]`` substrings, outermost first."""
    out: List[object] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            end = text.rfind(closer)
            while end > start:
                try:
                    out.append(json.loads(text[start:end + 1]))
                    break
                except Exception:
                    end = text.rfind(closer, start, end)
            start = text.find(opener, start + 1)
    return out


def _parse_json_obj(text: str) -> Optional[object]:
    """Best-effort: the last parseable JSON value in the reply.

    Objects are preferred over arrays, and the *outermost* object wins: for
    ``{"user": {"name": "小明"}}`` a naive "last parseable value" search also
    matches the inner ``{"name": "小明"}`` and would make ``json_nested:``
    unusable.  Arrays are only considered when no object parses at all.
    """
    objs = [o for o in _json_blocks(text) if isinstance(o, dict)]
    if objs:
        return objs[0]
    arrs = [o for o in _json_blocks(text) if isinstance(o, list)]
    return arrs[0] if arrs else None


def _n_grams(text: str, n: int) -> List[str]:
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def _roman_count(text: str) -> int:
    return len({m.group(1).lower()
                for m in re.finditer(r"(?mi)^\s*(i{1,3}|iv|v|vi{1,3})[.、)]", text)})


def _arabic_count(text: str) -> int:
    hits = set()
    for line in text.splitlines():
        m = re.match(r"\s*([0-9０-９]{1,2})\s*[.、)）]", line)
        if m:
            hits.add(m.group(1))
    cn = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
          "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}
    for line in text.splitlines():
        m = re.match(r"\s*([一二三四五六七八九十]{1,3})\s*[、.．)）]\s*\S", line)
        if m and m.group(1) in cn:
            hits.add(cn[m.group(1)])
    for m in re.finditer(r"[（(]\s*([0-9０-９]{1,2})\s*[）)]", text):
        hits.add(m.group(1))
    return len(hits)


def _is_ascii_ratio(text: str, ratio: float, minlen: int) -> bool:
    if len(text) < minlen:
        return False
    return sum(1 for ch in text if ord(ch) < 128) / len(text) >= ratio


def _zh_ratio(text: str, ratio: float, minlen: int) -> bool:
    if len(text) < minlen:
        return False
    n = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return n / len(text) >= ratio


def _dedup(text: str, ratio: float, n: int, minlen: int) -> bool:
    """``False`` only when the text is long enough to judge *and* repetitive.

    ``minlen`` is a **gate on judging, not on passing**: a reply shorter than
    ``minlen`` cannot exhibit a runaway 4-gram loop, so the predicate abstains
    (returns ``True``) and the length requirement is left to ``len:``.  Failing
    short replies here would double-penalise them and silently turn every
    ``dedup:`` item into a length item — the two guards must stay independent.
    """
    if len(text) < minlen:
        return True
    g = _n_grams(text, n)
    if not g:
        return True
    return len(set(g)) / len(g) >= ratio


# name -> (builder, min_args, max_args)
#
# Arity matters at compile time, not just at call time: predicates registered
# ``(1, 1)`` take **one spec argument that may itself contain commas**
# (``dedup:RATIO,N,MINLEN``, ``csv:ROWS,COLS``), while ``(1, N)`` predicates
# take a comma-separated *list* of alternatives (``any:A,B``).  Getting this
# wrong splits ``dedup:0.65,4,40`` into three arguments and raises TypeError.
_BUILDERS: Dict[str, Tuple[Callable[..., Predicate], int, int]] = {}


def _reg(name: str, lo: int = 1, hi: int = 1):
    def deco(fn):
        _BUILDERS[name] = (fn, lo, hi)
        return fn
    return deco


@_reg("any", 1, 64)
def _b_any(*vals: str) -> Predicate:
    if not any(vals):
        raise CheckerError("any: needs at least one non-empty value")
    return lambda c: any(v in c.reply for v in vals)


@_reg("all", 1, 64)
def _b_all(*vals: str) -> Predicate:
    return lambda c: all(v in c.reply for v in vals)


@_reg("none", 1, 64)
def _b_none(*vals: str) -> Predicate:
    return lambda c: not any(v in c.reply for v in vals)


@_reg("contains_all", 1, 64)
def _b_contains_all(*vals: str) -> Predicate:
    return lambda c: all(v in c.reply for v in vals)


@_reg("re", 1, 64)
def _b_re(*pats: str) -> Predicate:
    rxs = [_rx(p) for p in pats]
    return lambda c: any(r.search(c.reply) for r in rxs)


@_reg("not_re", 1, 64)
def _b_not_re(*pats: str) -> Predicate:
    rxs = [_rx(p) for p in pats]
    return lambda c: not any(r.search(c.reply) for r in rxs)


@_reg("regex_count", 1, 1)
def _b_regex_count(spec: str) -> Predicate:
    parts = spec.rsplit(",", 1)
    if len(parts) != 2:
        raise CheckerError(f"regex_count: needs PAT,N, got {spec!r}")
    rx, n = _rx(parts[0]), int(parts[1])
    return lambda c: len(rx.findall(c.reply)) >= n


@_reg("len", 1, 1)
def _b_len(spec: str) -> Predicate:
    m = re.fullmatch(r"\s*(\d+)\s*\.\.\s*(\d*)\s*", spec)
    if not m:
        raise CheckerError(f"len: needs LO..HI (HI may be empty for unbounded), "
                           f"got {spec!r}")
    lo = int(m.group(1))
    explicit_hi = m.group(2)
    hi = int(explicit_hi) if explicit_hi else 100_000
    if lo > hi:
        raise CheckerError(f"len: LO > HI in {spec!r}")
    if explicit_hi and int(explicit_hi) > 99_999:
        raise CheckerError(f"len:{spec}: HI of 99999 or more can never fail — "
                           f"max_new is capped at 512 tokens, so the reply "
                           f"cannot reach it")
    return lambda c: lo <= len(c.reply.strip()) <= hi
    return lambda c: lo <= len(c.reply.strip()) <= hi


@_reg("startswith", 1, 64)
def _b_startswith(*vals: str) -> Predicate:
    return lambda c: any(c.reply.strip().startswith(v) for v in vals)


@_reg("endswith", 1, 64)
def _b_endswith(*vals: str) -> Predicate:
    return lambda c: any(c.reply.strip().endswith(v) for v in vals)


@_reg("numbered", 1, 1)
def _b_numbered(*specs: str) -> Predicate:
    ns = [int(s) for s in specs]
    return lambda c: max(_arabic_count(c.reply), _roman_count(c.reply)) >= min(ns)


@_reg("dash_list", 1, 1)
def _b_dash_list(*specs: str) -> Predicate:
    ns = [int(s) for s in specs]
    return lambda c: len(re.findall(r"(?m)^\s*[-*•]\s*\S", c.reply)) >= min(ns)


@_reg("bullets", 1, 1)
def _b_bullets(*specs: str) -> Predicate:
    ns = [int(s) for s in specs]
    pat = r"(?m)^\s*(?:[-*•]|\d{1,2}[.、)）]|[一二三四五六七八九十]{1,3}[、.．])\s*\S"
    return lambda c: len(re.findall(pat, c.reply)) >= min(ns)


@_reg("json", 1, 64)
def _b_json(*keys: str) -> Predicate:
    want = tuple(keys)

    def f(c: Ctx) -> bool:
        o = _parse_json_obj(c.reply)
        return isinstance(o, dict) and all(k in o for k in want)
    return f


@_reg("json_array", 1, 1)
def _b_json_array(*specs: str) -> Predicate:
    n = int(specs[0])

    def f(c: Ctx) -> bool:
        o = _parse_json_obj(c.reply)
        return isinstance(o, list) and len(o) == n
    return f


@_reg("json_array_min", 1, 1)
def _b_json_array_min(*specs: str) -> Predicate:
    n = int(specs[0])

    def f(c: Ctx) -> bool:
        o = _parse_json_obj(c.reply)
        return isinstance(o, list) and len(o) >= n
    return f


@_reg("json_nested", 1, 1)
def _b_json_nested(*specs: str) -> Predicate:
    outer, inner = specs[0].split(".", 1)

    def f(c: Ctx) -> bool:
        o = _parse_json_obj(c.reply)
        return (isinstance(o, dict) and isinstance(o.get(outer), dict)
                and inner in o[outer])
    return f


@_reg("json_has", 1, 64)
def _b_json_has(*specs: str) -> Predicate:
    """``json_has:key=value`` — flat dict whose key holds the given value."""
    pairs = [s.split("=", 1) for s in specs]
    for p in pairs:
        if len(p) != 2:
            raise CheckerError(f"json_has: needs key=value, got {p!r}")

    def f(c: Ctx) -> bool:
        o = _parse_json_obj(c.reply)
        if not isinstance(o, dict):
            return False
        for k, v in pairs:
            if k not in o or str(o[k]).strip().lower() != v.strip().lower():
                return False
        return True
    return f


@_reg("csv", 1, 1)
def _b_csv(spec: str) -> Predicate:
    rows, cols = _ints(spec, 2, 2)
    return lambda c: sum(
        1 for ln in c.reply.splitlines() if ln.strip() and ln.count(",") >= cols - 1
    ) >= rows


@_reg("md_table", 0, 0)
def _b_md_table() -> Predicate:
    rx = _rx(r"(?m)^\s*\|?[\s:|-]*\|[\s:|-]*$")
    return lambda c: "|" in c.reply and bool(rx.search(c.reply))


@_reg("code_block", 0, 0)
def _b_code_block() -> Predicate:
    return lambda c: c.reply.count("```") >= 2


@_reg("yaml", 1, 64)
def _b_yaml(*keys: str) -> Predicate:
    rxs = [_rx(rf"(?m)^\s*{re.escape(k)}\s*:") for k in keys]
    return lambda c: all(r.search(c.reply) for r in rxs)


@_reg("xml", 1, 64)
def _b_xml(*tags: str) -> Predicate:
    return lambda c: all(f"<{t}>" in c.reply and f"</{t}>" in c.reply for t in tags)


@_reg("kv", 1, 64)
def _b_kv(*keys: str) -> Predicate:
    rxs = [_rx(rf"{re.escape(k)}\s*[:=]") for k in keys]
    return lambda c: all(r.search(c.reply) for r in rxs)


@_reg("steps", 1, 64)
def _b_steps(*vals: str) -> Predicate:
    def f(c: Ctx) -> bool:
        pos = 0
        for v in vals:
            i = c.reply.find(v, pos)
            if i < 0:
                return False
            pos = i + len(v)
        return True
    return f


@_reg("repeat", 1, 1)
def _b_repeat(spec: str) -> Predicate:
    parts = spec.rsplit(",", 1)
    if len(parts) != 2:
        raise CheckerError(f"repeat: needs SUB,N, got {spec!r}")
    sub, n = parts[0], int(parts[1])
    if not sub:
        raise CheckerError("repeat: empty substring")
    return lambda c: c.reply.count(sub) >= n


@_reg("num", 1, 64)
def _b_num(*specs: str) -> Predicate:
    """``num:V`` / ``num:V1,V2`` — standalone number, tolerant of formatting."""
    vals = [s.strip() for s in specs if s.strip()]
    if not vals:
        raise CheckerError("num: needs at least one value")
    nums = []
    for v in vals:
        try:
            nums.append(float(v))
        except ValueError as exc:
            raise CheckerError(f"num: {v!r} is not a number") from exc
    return _b_num_f(*[f"{n:g}" for n in nums])


@_reg("num_any", 1, 64)
def _b_num_any(*specs: str) -> Predicate:
    """Alias of ``num:`` kept because the contract documents both spellings."""
    return _b_num(*specs)


@_reg("num_f", 1, 64)
def _b_num_f(*specs: str) -> Predicate:
    """Numeric match by value, so ``3.50`` counts for ``3.5`` and ``042`` for 42."""
    try:
        vals = [float(s) for s in specs if s.strip()]
    except ValueError as exc:
        raise CheckerError(f"num_f: non-numeric argument in {specs!r}") from exc
    if not vals:
        raise CheckerError("num_f: needs at least one value")

    def f(c: Ctx) -> bool:
        for m in re.finditer(r"-?\d+(?:\.\d+)?", c.reply):
            try:
                got = float(m.group(0))
            except ValueError:                               # pragma: no cover
                continue
            if any(abs(got - v) < 1e-9 for v in vals):
                return True
        return False
    return f


@_reg("number_only", 1, 1)
def _b_number_only(spec: str) -> Predicate:
    """The reply must *be* the number — ``MAXLEN`` is measured on the trimmed reply.

    Optional third field: ``delims``.  Without it the reply must equal the value
    (after stripping a trailing ``。``/``.``); with ``delims=1`` a short
    ``答案是 123`` wrapper is also accepted, which is what a chat model usually
    produces and what the item often actually wants to test.
    """
    parts = _split_top(spec, ",")
    if not parts or not parts[0].strip():
        raise CheckerError(f"number_only: needs VALUE[,MAXLEN[,delims]], got {spec!r}")
    val = parts[0].strip()
    try:
        want = float(val)
    except ValueError as exc:
        raise CheckerError(f"number_only: {val!r} is not a number") from exc
    maxlen = int(parts[1]) if len(parts) > 1 and parts[1].strip() else 12
    loose = len(parts) > 2 and parts[2].strip() == "delims"

    def f(c: Ctx) -> bool:
        s = c.reply.strip()
        if len(s) > maxlen:
            return False
        s = re.sub(r"[。.！!？?\s]+$", "", s)
        if loose:
            for m in re.finditer(r"-?\d+(?:\.\d+)?", s):
                if abs(float(m.group(0)) - want) < 1e-9:
                    return True
            return False
        try:
            return abs(float(s) - want) < 1e-9
        except ValueError:
            return False
    return f


@_reg("ascii", 1, 1)
def _b_ascii(spec: str) -> Predicate:
    parts = _split_top(spec, ",")
    ratio = float(parts[0]) if parts[0] else 0.8
    minlen = int(parts[1]) if len(parts) > 1 and parts[1] else 1
    return lambda c: _is_ascii_ratio(c.reply.strip(), ratio, minlen)


@_reg("zh_ratio", 1, 1)
def _b_zh_ratio(spec: str) -> Predicate:
    parts = _split_top(spec, ",")
    ratio = float(parts[0]) if parts[0] else 0.5
    minlen = int(parts[1]) if len(parts) > 1 and parts[1] else 1
    return lambda c: _zh_ratio(c.reply.strip(), ratio, minlen)


@_reg("dedup", 1, 1)
def _b_dedup(spec: str) -> Predicate:
    parts = _split_top(spec, ",")
    ratio = float(parts[0]) if parts and parts[0] else 0.65
    n = int(parts[1]) if len(parts) > 1 and parts[1] else 4
    minlen = int(parts[2]) if len(parts) > 2 and parts[2] else 40
    return lambda c: _dedup(c.reply.strip(), ratio, n, minlen)


@_reg("not_think", 0, 0)
def _b_not_think() -> Predicate:
    def f(c: Ctx) -> bool:
        r = c.reply
        i = r.find("<think>")
        if i < 0:
            return True
        return "</think>" in r[i:]
    return f


@_reg("think_leak", 0, 0)
def _b_think_leak() -> Predicate:
    def f(c: Ctx) -> bool:
        r = c.reply
        i = r.find("<think>")
        return i >= 0 and "</think>" not in r[i:]
    return f


@_reg("not_echo_prompt", 0, 1)
def _b_not_echo_prompt(*specs: str) -> Predicate:
    """The reply must not contain a long verbatim chunk of the instruction.

    The failure mode this catches is the model repeating the task instead of
    doing it — ``用「要是……就……」造一个句子`` answered with the instruction
    itself.  Such an echo contains one of the connectives the checker looks for,
    so a plain ``all:`` anchor is satisfied for free; this is the missing guard
    for template-shaped items.

    ``specs[0]`` (optional) is the minimum chunk length that counts as an echo.
    It defaults to 8 and is capped at ``max(4, len(prompt) // 2)`` so a very
    short prompt cannot make the predicate unsatisfiable.
    """
    try:
        minchunk = int(specs[0]) if specs and specs[0].strip() else 8
    except ValueError as exc:
        raise CheckerError(f"not_echo_prompt: bad chunk size {specs!r}") from exc

    def f(c: Ctx) -> bool:
        r = c.reply.strip()
        p = (c.prompt or "").strip()
        if not p:
            return True
        n = min(minchunk, max(4, len(p) // 2))
        if len(r) < n:
            return True
        chunks = {p[i:i + n] for i in range(len(p) - n + 1)}
        return not any(ch in r for ch in chunks)
    return f


@_reg("not_echo_user", 0, 0)
def _b_not_echo_user() -> Predicate:
    def f(c: Ctx) -> bool:
        r = c.reply.strip()
        for u in c.users:
            u = u.strip()
            if len(u) >= 4 and len(r) >= 4 and r[:12] == u[:12]:
                return False
        return True
    return f


@_reg("sentences", 1, 1)
def _b_sentences(*specs: str) -> Predicate:
    """Count sentence terminators, counting ASCII ``.`` and ``;`` as well.

    English text ends in ``.``, so a Chinese-only terminator set silently scored
    every English probe as 0 — the failure mode was invisible because the item
    just looked hard.  ``.`` inside a decimal (``3.5``) also counts, which only
    ever makes the predicate more permissive.
    """
    ns = [int(s) for s in specs if s.strip()]
    if not ns:
        raise CheckerError("sentences: needs a count")
    n = min(ns)
    return lambda c: len(re.findall(r"[。！？!?.;；\n]", c.reply)) >= n


@_reg("punct_free", 0, 0)
def _b_punct_free() -> Predicate:
    return lambda c: not re.search(r"[。，！？、；：,.!?;:]", c.reply)


@_reg("caps", 1, 64)
def _b_caps(*words: str) -> Predicate:
    return lambda c: any(w.upper() in c.reply for w in words)


@_reg("eq", 1, 1)
def _b_eq(spec: str) -> Predicate:
    """Verbatim equality (after stripping).  ``\\x2c`` yields a literal comma."""
    val = _decode(spec)
    if not val:
        raise CheckerError("eq: needs a value")
    return lambda c: c.reply.strip() == val


@_reg("is_int_in", 1, 1)
def _b_is_int_in(spec: str) -> Predicate:
    lo, hi = _ints(spec, 2, 2)

    def f(c: Ctx) -> bool:
        s = c.reply.strip()
        if not re.fullmatch(r"-?\d+", s):
            return False
        return lo <= int(s) <= hi
    return f


@_reg("not_empty", 0, 0)
def _b_not_empty() -> Predicate:
    return lambda c: bool(c.reply.strip())


@_reg("max_tokens", 1, 1)
def _b_max_tokens(*specs: str) -> Predicate:
    n = int(specs[0])
    return lambda c: c.n_new <= n


# --------------------------------------------------------------------- compile


def _compile_pred(spec: str) -> Predicate:
    spec = spec.strip()
    if not spec:
        raise CheckerError("empty predicate")
    negate = spec.startswith("!")
    if negate:
        spec = spec[1:].strip()
    name, _, argstr = spec.partition(":")
    name = name.strip()
    if name not in _BUILDERS:
        raise CheckerError(f"unknown predicate {name!r} (see bench/CONTRACT.md §4)")
    fn, lo, hi = _BUILDERS[name]

    # Arity decides how the argument text is cut.  ``(1, 1)`` predicates take a
    # single spec that legitimately contains commas — ``dedup:0.65,4,40``,
    # ``csv:2,2``, ``number_only:123,8``, ``regex_count:PAT,N`` — so the whole
    # argstr is one argument.  ``(1, N)`` predicates take a list of alternatives.
    #
    # ``\x2c`` is decoded *per argument, after the split*: decoding first would
    # turn an escaped literal comma back into a separator and split the value
    # list (``any:a\x2cb`` would silently mean "a OR b").
    if hi <= 1:
        args = [_decode(argstr)] if argstr else []
    else:
        args = [_decode(a) for a in _split_top(argstr, ",")] if argstr else []
    if len(args) < lo:
        raise CheckerError(
            f"{name}: needs at least {lo} argument(s), got {len(args)} "
            f"(in {spec!r})")
    try:
        base = fn(*args)
    except CheckerError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise CheckerError(f"{name}: bad arguments {args!r}: {exc}") from exc
    return (lambda c: not base(c)) if negate else base


def compile_checker(spec: str) -> Predicate:
    """``"a;b;c"`` -> predicate that is the AND of ``a``, ``b`` and ``c``.

    ``\\x3b`` is decoded to a literal ``;`` as part of the split, so a regex may
    contain a semicolon; ``\\x2c`` stays escaped here and is decoded per
    predicate, so a literal comma cannot create a spurious extra predicate.
    """
    key = spec.strip()
    hit = _SPEC_CACHE.get(key)
    if hit is not None:
        return hit
    if not key:
        raise CheckerError("empty checker spec")
    preds: List[Predicate] = []
    for part in _split_specs(key):
        part = part.strip()
        if not part:
            continue
        if part == "true":                            # explicit no-op, for H items
            continue
        if part.startswith("re:^") and part in ("re:^", "re:^$"):
            raise CheckerError(f"always-true regex predicate {part!r}")
        preds.append(_compile_pred(part))
    if not preds:
        raise CheckerError(f"checker {spec!r} has no effective predicate")
    eff: Predicate = (lambda c: all(p(c) for p in preds)) if len(preds) > 1 else preds[0]
    _SPEC_CACHE[key] = eff
    return eff


def predicate_names(spec: str) -> List[str]:
    """Predicate names used by a spec — for coverage audits over the battery."""
    out = []
    for part in _split_specs(spec):
        part = part.strip().lstrip("!")
        if not part:
            continue
        out.append(part.partition(":")[0].strip())
    return out


def known_predicates() -> List[str]:
    return sorted(_BUILDERS)


# ---------------------------------------------------------------- sanity checks

_WEAK_SIGNS = ("len:0..", "re:^", "re:.$", "not_empty", "true")


def lint_checker(spec: str, prompt: str = "") -> List[str]:
    """Warnings that are about *the checker*, empty list means sound.

    Anchor-does-not-discriminate analysis deliberately does **not** live here:
    it needs the item's turn structure (which turn is under test) and a reviewed
    exception list, both of which are in :mod:`bench.registry`.  Duplicating a
    crude version of it here produced hundreds of false warnings on the battery's
    best adversarial probes.
    """
    warns: List[str] = []
    for part in _split_specs(spec):
        p = part.strip()
        if p in _WEAK_SIGNS:
            warns.append(f"weak predicate {p!r}")
    names = set(predicate_names(spec))
    if not names:
        warns.append("no predicates")
    if "dedup" not in names and "len" not in names and any(
            n in names for n in ("any", "all", "contains_all")):
        warns.append("open-ended item without a dedup:/len: guard")
    return warns
