"""Harden multi-turn recall checks against the question-echo failure mode.

The multi-turn recall families ask things like "我叫什么名字？" or "我现在住哪？"
whose answer is a fact introduced in an earlier turn.  A model that has **not**
recalled the fact still tends to answer plausibly by restating the question —
and for prompts where the fact is named anyway ("你的狗叫什么名字？" against
``any:两只猫``) the item scores on the question alone.

The distinguishing test is not "does the anchor appear" but "is this an answer or
an echo".  An echo is a question, so the guard is `!endswith:？,?`: the reply must
carry the anchor **and** not be shaped like the question it was asked.

That is deliberately preferred over the obvious alternative of requiring the
anchor twice (``regex_count:anchor,2``), which misreads a perfectly good short
answer — the name usually appears once in "你叫小明呀".

    python scripts/fix_recall_checks.py --write
    python scripts/fix_recall_checks.py            # dry run, shows every change
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.checkers import (CheckerError, _split_specs, _split_top,  # noqa: E402
                            compile_checker)
from bench.registry import _NEGATION_CUES, load_items_dir            # noqa: E402

GENERATED = ("gen_basics.jsonl",)
ECHO_GUARD = "!endswith:？,?,。?,?。"
MIN_ANCHOR = 2
FAIL_ONLY = ("any", "all")


def _in_question(anchor: str, users: str) -> bool:
    """Anchor present in a user turn, not inside a negating cue."""
    for m in re.finditer(re.escape(anchor), users):
        window = users[max(0, m.start() - 14):m.end() + 8]
        if not any(c in window for c in _NEGATION_CUES):
            return True
    return False


def _asks_question(users: str) -> bool:
    last = users.strip().splitlines()[-1] if users.strip() else ""
    return last.rstrip().endswith(("？", "?"))


def strengthen(family: str, chk: str, users: str) -> tuple[str, list[str]]:
    """Add the echo guard when a recall item's anchor is named by the question."""
    notes: list[str] = []
    if not family.startswith("mt_"):
        return chk, notes
    if not _asks_question(users):
        return chk, notes
    parts = _split_specs(chk)
    named = set()
    for part in parts:
        p = part.strip()
        if p.startswith("!"):
            continue
        name, _, arg = p.partition(":")
        if name.strip() not in FAIL_ONLY or not arg:
            continue
        for v in _split_top(arg, ","):
            v = v.strip()
            if len(v) >= MIN_ANCHOR and _in_question(v, users):
                named.add(v)
    if not named:
        return chk, notes
    if any(p.strip().startswith("!endswith") for p in parts):
        return chk, ["already guarded"]
    notes.append(f"echo guard for {sorted(named)}")
    return chk + ";" + ECHO_GUARD, notes


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dir", default=str(ROOT / "bench" / "items"))
    args = ap.parse_args()

    d = Path(args.dir)
    total = 0
    by_family: Counter = Counter()
    for path in sorted(d.glob("*.jsonl")):
        if path.name in GENERATED:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        out, changed = [], 0
        for n, line in enumerate(lines, 1):
            if not line.strip():
                out.append(line)
                continue
            o = json.loads(line)
            family = o.get("family", "")
            users = "\n".join(t.get("user", "") for t in o.get("turns", []))
            new_chk, notes = strengthen(family, o.get("chk", ""), users)
            if notes and new_chk != o.get("chk"):
                try:
                    compile_checker(new_chk)
                except CheckerError as exc:
                    print(f"!! {o['id']}: {exc}")
                    out.append(line)
                    continue
                changed += 1
                by_family[family] += 1
                if changed <= 30:
                    print(f"  {path.name}:{n} {o['id']}  [{notes[0]}]")
                    print(f"      - {o['chk']}")
                    print(f"      + {new_chk}")
                o["chk"] = new_chk
                out.append(json.dumps(o, ensure_ascii=False))
            else:
                out.append(line)
        if changed and args.write:
            path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
        total += changed
    print(f"\n{total} item(s) strengthened")
    for k, v in by_family.most_common():
        print(f"   {k:<20} {v}")
    if not args.write:
        print("(dry run — pass --write to apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
