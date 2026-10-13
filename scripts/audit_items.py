"""Damage assessment for an authored item file, without the checker compiler.

`load_jsonl` refuses to load a file whose `chk` does not compile, which hides
how much of the file is otherwise intact.  This reads the JSON directly and
reports the split between "content is fine, judgement is broken" and "content is
also damaged", so a repair can be scoped instead of guessed at.

    python scripts/audit_items.py --file bench/items/longctx_code.jsonl
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

from bench.checkers import CheckerError, compile_checker, predicate_names  # noqa: E402


def audit(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    total = ok = bad_chk = unparseable = 0
    fams: Counter = Counter()
    cats: Counter = Counter()
    tiers: Counter = Counter()
    problems: list[str] = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        total += 1
        try:
            o = json.loads(line)
        except json.JSONDecodeError as exc:
            unparseable += 1
            problems.append(f"{n}: invalid JSON ({exc.msg} at col {exc.colno})")
            continue
        cats[o.get("cat", "?")] += 1
        fams[o.get("family", "?")] += 1
        tiers[o.get("tier", "?")] += 1
        try:
            compile_checker(o.get("chk", ""))
            ok += 1
        except CheckerError as exc:
            bad_chk += 1
            problems.append(f"{n}: {o.get('id','?')}: {exc}")
    return {"total": total, "ok": ok, "bad_chk": bad_chk,
            "unparseable": unparseable, "problems": problems,
            "cats": cats, "fams": fams, "tiers": tiers,
            "preds": Counter(p for line in lines if line.strip()
                             for p in predicate_names(json.loads(line).get("chk", "")))}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--problems", type=int, default=25)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.is_absolute():
        path = ROOT / args.file
    r = audit(path)
    print(f"== {path.name}")
    print(f"   lines        {r['total']}")
    print(f"   chk OK       {r['ok']}")
    print(f"   chk broken   {r['bad_chk']}")
    print(f"   bad JSON     {r['unparseable']}")
    print(f"   cats  {dict(r['cats'])}")
    print(f"   tiers {dict(r['tiers'])}")
    print(f"   families ({len(r['fams'])})")
    if r["problems"]:
        print(f"\n   first {args.problems} problems:")
        for p in r["problems"][:args.problems]:
            print(f"     {p}")
    return 1 if (r["bad_chk"] or r["unparseable"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
