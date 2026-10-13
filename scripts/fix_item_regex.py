"""Repair broken regex predicates in an authored item file, at JSON level.

The checker DSL passes backslashes through **verbatim** (see
``bench/checkers.py:_split_top``), so a correctly written ``re:2\\D*5\\D*9``
needs no help.  What this tool fixes is the damage left by an editing pass that
re-escaped a string as if the DSL were a shell or a regex literal — which
produces patterns that either fail to compile or, worse, compile into something
that matches the wrong text:

* a stray unbalanced ``(`` or ``)`` from a truncated edit;
* a bare quantifier with nothing to repeat — ``**2``, ``*x`` — usually literal
  text that was meant to be matched;
* ``\\\\<metachar>`` doubling, where a pattern was escaped twice and now looks
  for a literal backslash.

Every repaired pattern is re-checked with ``re.compile`` before being written,
and anything still invalid is reported instead of guessed at.

    python scripts/fix_item_regex.py --file bench/items/longctx_code.jsonl
    python scripts/fix_item_regex.py --file ... --write      # apply
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.checkers import _split_specs  # noqa: E402

REGEX_PREDS = ("re", "not_re", "regex_count")
_DEBUG = False

# Characters that carry regex meaning.  ``\\X`` for one of these is almost always
# a doubled escape rather than a deliberate literal backslash.
METACHARS = set("dDwWsSbBAZz0123456789ntrfvuxNpP.^$|?*+()[]{}-/")
# Shortcut escapes whose *uppercase* form is a different character class and is
# therefore just as likely to be over-escaped: \D \W \S \B \A \Z.
METACHARS |= {c.upper() for c in "dwsbaz"}


def _split_regex(arg: str, name: str):
    """Separate ``PATTERN`` from the trailing ``,N`` of ``regex_count:``."""
    if name != "regex_count":
        return arg, None
    m = re.fullmatch(r"(.*),(\d+)\s*", arg)
    if not m:
        return arg, None
    return m.group(1), m.group(2)


def _escape_unbalanced(pat: str) -> str:
    """Escape only the parens that have no partner, leaving groups intact.

    Blanket-escaping every ``(`` would destroy valid groups such as
    ``([a-z]+)``; the corruption here is always a *single* stray paren.
    """
    opens: list[int] = []
    unmatched: set[int] = set()
    for i, ch in enumerate(pat):
        if ch == "\\" and i + 1 < len(pat):
            continue
        if ch == "(":
            opens.append(i)
        elif ch == ")":
            if opens:
                opens.pop()
            else:
                unmatched.add(i)
    unmatched.update(opens)
    if not unmatched:
        return pat
    return "".join(("\\" + ch) if i in unmatched else ch
                   for i, ch in enumerate(pat))


def repair_pattern(pat: str, name: str = "re") -> tuple[str, list[str]]:
    """Return ``(fixed, notes)`` for one regex argument."""
    notes: list[str] = []
    fixed = pat

    # 1) over-escaping: `\\X` -> `\X` for regex-significant X.  A deliberate
    #    literal backslash is written `\\\\`, which this leaves alone.
    for m in sorted(set(re.findall(r"\\\\(.)", fixed))):
        if m in METACHARS:
            fixed = fixed.replace("\\\\" + m, "\\" + m)
            notes.append(f"\\\\{m} -> \\{m}")

    # 2) `\x2c`/`\x3b` are the DSL's own escapes and must stay encoded until
    #    after the split, so they are never touched above (the x is in
    #    METACHARS, so `\\x2c` would be collapsed) — put them back.
    for esc, lit in (("\\x2c", ","), ("\\x3b", ";")):
        if esc.replace("\\", "\\\\") in pat and esc not in fixed:
            fixed = fixed.replace(lit, esc)

    # 3) parens with no partner do not compile; escape just those
    try:
        re.compile(fixed)
    except re.error:
        cand = _escape_unbalanced(fixed)
        if cand != fixed:
            try:
                re.compile(cand)
                fixed = cand
                notes.append("escaped the unpaired ( or )")
            except re.error:
                pass

    # 4) a quantifier with nothing to repeat — `**2`, `*x`, `+(` — is almost
    #    always literal text that was meant to be matched, so escape it
    try:
        re.compile(fixed)
    except re.error as exc:
        if "nothing to repeat" in str(exc):
            cand = re.sub(r"(?<!\\)([*+?])", r"\\\1", fixed)
            try:
                re.compile(cand)
                fixed = cand
                notes.append("escaped a bare quantifier")
            except re.error:
                pass

    try:
        re.compile(fixed)
    except re.error as exc:
        notes.append(f"STILL INVALID: {exc}")
    return fixed, notes


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.is_absolute():
        path = ROOT / args.file
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    out_lines: list[str] = []
    changed = 0
    still_bad = 0
    for n, line in enumerate(lines, 1):
        if not line.strip():
            out_lines.append(line)
            continue
        obj = json.loads(line)
        chk = obj.get("chk", "")
        new_parts: list[str] = []
        touched = False
        if _DEBUG and "re:" in chk:
            print(f"[debug] line {n} id={obj.get('id')} chk={chk!r} "
                  f"parts={_split_specs(chk)!r}")
        for part in _split_specs(chk):
            stripped = part.strip()
            neg = stripped.startswith("!")
            body = stripped[1:] if neg else stripped
            name, _, arg = body.partition(":")
            name = name.strip()
            if name not in REGEX_PREDS or not arg:
                new_parts.append(part)
                continue
            pat, n_suffix = _split_regex(arg, name)
            fixed, notes = repair_pattern(pat, name)
            if _DEBUG:
                print(f"[debug]   {obj.get('id')} {name} pat={pat!r} "
                      f"fixed={fixed!r} notes={notes!r}")
            if not notes:
                new_parts.append(part)
                continue
            if any(x.startswith("STILL INVALID") for x in notes):
                still_bad += 1
                print(f"!! {path.name}:{n} {obj.get('id')} {name}: {notes[-1]}")
                print(f"     pat = {pat!r}")
                new_parts.append(part)
                continue
            rebuilt = f"{name}:{fixed}" + (f",{n_suffix}" if n_suffix else "")
            new_parts.append(("!" if neg else "") + rebuilt)
            touched = True
            print(f"   {path.name}:{n} {obj.get('id')} {name}: {'; '.join(notes)}")
        if touched:
            changed += 1
            obj["chk"] = ";".join(p.strip() for p in new_parts)
            out_lines.append(json.dumps(obj, ensure_ascii=False))
        else:
            out_lines.append(line)

    print(f"\n{changed} line(s) repaired, {still_bad} needing human attention")
    if args.write and changed:
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    elif changed:
        print("(dry run — pass --write to apply)")
    return 1 if still_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
