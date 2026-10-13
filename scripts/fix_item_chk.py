"""Fix two authoring classes the compile step surfaced, in the authored files.

Both are cases where a legitimate-looking ``chk`` means something other than
what its author intended — the compile step catches the mechanical ones, and
these two need a decision:

1. ``re:max\\\\(;re:def larger,lambda`` — the second ``re:`` splits on its comma,
   so it means "def larger" OR "lambda", which happens to be the intent.  But
   ``re:range(1, 6),range(1,6)`` also splits, turning one pattern into two
   fragments ``range(1`` and `` 6)`` — the first does not compile at all.
2. Multiple ``re:`` predicates in one ``chk`` are ANDed, which for alternative
   spellings is wrong: ``re:return;re:a + b,a+b`` requires *both*.  They should
   be a single multi-value predicate.

    python scripts/fix_item_chk.py --file bench/items/longctx_code.jsonl --write
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


def normalise(chk: str) -> tuple[str, list[str]]:
    """Merge consecutive ``re:`` predicates; repair comma-bearing patterns."""
    notes: list[str] = []
    parts = _split_specs(chk)
    merged: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i].strip()
        neg = p.startswith("!")
        body = p[1:] if neg else p
        name, _, arg = body.partition(":")
        if neg or name.strip() != "re":
            merged.append(p)
            i += 1
            continue
        # collect the run of consecutive positive `re:` predicates
        alts = [a.strip() for a in arg.split(",") if a.strip()]
        j = i + 1
        while j < len(parts):
            q = parts[j].strip()
            if q.startswith("!") or q.partition(":")[0].strip() != "re":
                break
            alts.extend(a.strip() for a in q.partition(":")[2].split(",") if a.strip())
            j += 1
        if j - i > 1:
            notes.append(f"merged {j - i} re: predicates")
        # a pattern containing an unescaped comma and no other alternative
        fixed_alts = []
        for a in alts:
            try:
                re.compile(a)
                fixed_alts.append(a)
                continue
            except re.error:
                pass
            # escape the comma(s) so the pattern survives as one alternative
            cand = a.replace(",", "\\x2c")
            try:
                re.compile(cand)
                fixed_alts.append(cand)
                notes.append(f"escaped comma inside {a!r}")
            except re.error:
                fixed_alts.append(a)
        merged.append("re:" + ",".join(fixed_alts))
        i = j
    return ";".join(merged), notes


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
    lines = path.read_text(encoding="utf-8").splitlines()
    out, changed = [], 0
    for n, line in enumerate(lines, 1):
        if not line.strip():
            out.append(line)
            continue
        obj = json.loads(line)
        new, notes = normalise(obj.get("chk", ""))
        if notes and new != obj.get("chk"):
            changed += 1
            print(f"  {path.name}:{n} {obj.get('id')}: {'; '.join(notes)}")
            obj["chk"] = new
            out.append(json.dumps(obj, ensure_ascii=False))
        else:
            out.append(line)
    print(f"\n{changed} line(s) changed")
    if args.write and changed:
        path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
