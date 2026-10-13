"""Rewrite leaked multi-turn recall checks so the item actually tests recall.

A "remember my name" item looks like this::

    turn 1  我叫小明，很高兴认识你。
    turn 2  我叫什么名字？            chk = any:小明

The answer is written verbatim in an earlier user turn, so the model can score
by **copying from the visible context** — recall is never exercised, and the item
cannot separate a model that remembers from one that reads the transcript.

The repair keeps the fact where it is (that is the point of the item) and
rephrases the question so it no longer names the fact::

    turn 2  我刚才让你记住的名字是什么？

An item is considered leaked when one of its ``any:``/``all:`` anchors occurs in
an **earlier** user turn.  Only the question is rewritten — never the fact, never
the checker — so the item keeps its tier, its category and its difficulty.

    python scripts/fix_leaked_recall.py            # dry run
    python scripts/fix_leaked_recall.py --write
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

from bench.checkers import CheckerError, _split_specs, _split_top, compile_checker  # noqa: E402
from bench.registry import anchor_problems, load_items_dir                          # noqa: E402

GENERATED = ("gen_basics.jsonl",)

# Question shapes, ordered by specificity.  The rewrite removes references to the
# fact itself and refers to it as something the user asked to be remembered.
SHAPES: list[tuple[str, str]] = [
    (r"我叫什么名字[？?]?$", "我刚才让你记住的名字是什么？"),
    (r"我叫什么[？?]?$", "我刚才让你记住的名字是什么？"),
    (r"我(现在)?住(在)?哪(里|儿|个城市)?[？?]?$", "我刚才让你记住的住址是哪里？"),
    (r"我(现在)?住在什么城市[？?]?$", "我刚才让你记住的城市是哪个？"),
    (r"我(妻子|老婆)叫什么[？?]?$", "我刚才让你记住的那个名字是什么？"),
    (r"^(我 )?(刚才|之前|上次)?(让你)?(记的|记住的)?(那个)?数字是(几|多少)[？?]?$",
     "我刚才让你记住的那个数字是多少？"),
    (r"我(最)?(喜欢|爱)什么(颜色|水果|动物|书|科目)[？?]?$",
     "我刚才告诉你我喜欢什么？"),
    (r"我(的)?(狗|猫|宠物)叫什么名字[？?]?$", "我刚才提到的宠物叫什么名字？"),
    (r"我(下周三)?去哪儿[？?]?$", "我刚才说我要去哪里？"),
    (r"我(的)?(车)?牌(照)?号(码)?是(多少|什么)[？?]?$",
     "我刚才让你记住的号码是多少？"),
    (r"我多大了[？?]?$", "我刚才说的年龄是多少？"),
    (r"我是做什么的[？?]?$", "我刚才说的职业是什么？"),
    (r"我(最)?讨厌吃什么[？?]?$", "我刚才说我不吃什么？"),
    (r"我(的)?航班几点起飞[？?]?$", "我刚才说的起飞时间是几点？"),
    (r"我(的)?猫叫什么[？?]?$", "我刚才提到的宠物的名字是什么？"),
    (r"我女儿的生日是哪天[？?]?$", "我刚才说的那个日期是哪天？"),
    (r"我(的)?工位在哪个房间[？?]?$", "我刚才说的房间号是多少？"),
    (r"我(的)?(快递)?单号是多少[？?]?$", "我刚才让你记住的那串号码是什么？"),
    (r"我(的)?妻子叫什么[？?]?我们结婚几年了[？?]?$",
     "我刚才说的那两个信息分别是什么？"),
    (r"我(下周三)?去哪儿[？?]?住哪个饭店[？?]?$", "我刚才说的行程和住处的名字是什么？"),
    (r"我叫什么[？?]?住(在)?哪[？?]?$", "我刚才让你记住的名字和住址分别是什么？"),
    (r"我(的)?猫和狗叫什么[？?]?$", "我刚才提到的宠物的名字分别是什么？"),
    (r"我是哪的人[？?]?做什么工作[？?]?$", "我刚才说的籍贯和职业分别是什么？"),
    (r"我过敏什么[？?]?$", "我刚才说我对什么过敏？"),
]

# Applied to questions with no specific shape: prepend a reference to the earlier
# turn so the fact is no longer written in the question.  Generic on purpose —
# the anchor check is what decides whether a given rewrite is acceptable, and a
# wrong rewrite is skipped rather than guessed at.
GENERIC = "我刚才告诉过你的，"


def _no_leak(candidate: str, anchors: list[str]) -> bool:
    return not any(a in candidate for a in anchors)


def rewrite_question(q: str, anchors: list[str]) -> str | None:
    s = q.strip()
    for pat, repl in SHAPES:
        if re.fullmatch(pat, s) and _no_leak(repl, anchors):
            return repl
    generic = GENERIC + s
    if _no_leak(generic, anchors) and len(s) <= 40:
        return generic
    return None


def leaked_anchors(item) -> list[str]:
    """Anchors checked on the last reply that also appear in an earlier turn."""
    prior = "\n".join(item.turns[:-1])
    if not prior.strip():
        return []
    from bench.checkers import _split_top as st
    out = []
    for part in _split_specs(item.chk):
        p = part.strip()
        if p.startswith("!"):
            continue
        name, _, arg = p.partition(":")
        if name.strip() not in ("any", "all", "contains_all") or not arg:
            continue
        for v in st(arg, ","):
            v = v.strip()
            if len(v) >= 2 and v in prior and v not in out:
                out.append(v)
    return out


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
    total = rewritable = 0
    by_family: Counter = Counter()
    unresolved: list[str] = []
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
            if not o["id"].startswith("mt_"):
                out.append(line)
                continue
            anchors = leaked_anchors(_Item(o))
            if not anchors:
                out.append(line)
                continue
            total += 1
            new_q = rewrite_question(o["turns"][-1].get("user", ""), anchors)
            if new_q is None:
                unresolved.append(f"{o['id']}: {o['turns'][-1]['user'][:60]!r} "
                                  f"anchors={anchors}")
                out.append(line)
                continue
            rewritable += 1
            changed += 1
            by_family[o.get("family", "")] += 1
            print(f"  {o['id']}  anchors={anchors}")
            print(f"      - {o['turns'][-1]['user']}")
            print(f"      + {new_q}")
            o["turns"][-1] = {**o["turns"][-1], "user": new_q}
            try:
                compile_checker(o["chk"])
            except CheckerError as exc:
                print(f"      !! chk broken: {exc}")
            out.append(json.dumps(o, ensure_ascii=False))
        if changed and args.write:
            path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")

    print(f"\nleaked recall items: {total}; rewritten: {rewritable}; "
          f"left alone: {len(unresolved)}")
    for k, v in by_family.most_common():
        print(f"   {k:<20} {v}")
    if unresolved:
        print("\nquestions with no matching shape (still leaked):")
        for u in unresolved:
            print(f"   {u}")
    if not args.write:
        print("(dry run — pass --write to apply)")
    return 0


class _Item:
    """Minimal shim so :func:`leaked_anchors` can take a raw dict."""

    def __init__(self, o: dict):
        self.chk = o.get("chk", "")
        self.turns = [t.get("user", "") for t in o.get("turns", [])]


if __name__ == "__main__":
    raise SystemExit(main())
