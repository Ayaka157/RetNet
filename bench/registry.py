"""Item schema, validation and the global id registry.

An authored line is *data*; :class:`Item` is its validated form.  Validation is
deliberately strict because the battery is 1,000+ items produced by several
authors (human and agent): a broken line must fail loudly at build time, with
the source file and line number, rather than quietly scoring 0 in a long run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .checkers import CheckerError, compile_checker, lint_checker, predicate_names

# ------------------------------------------------------------------- taxonomy

CATEGORIES: Tuple[str, ...] = (
    "角色扮演", "多轮", "一致性",            # core of the project's main line
    "指令遵循", "结构化", "鲁棒性", "双语", "安全",
    "填空", "问答", "语言", "续写",
    "长上下文", "编码", "数学",
)

# v2 weights, rebalanced.  docs/13 §2.1 showed category weight is an amplifier:
# a 1-item category carrying 2.0 weight flipped the whole conclusion.  v3 keeps
# the relative order of the project's priorities but every category now has
# 25-120 items, so no single item can move the total by more than ~0.002.
CATEGORY_WEIGHTS: Dict[str, float] = {
    "角色扮演": 2.0, "多轮": 2.0, "一致性": 1.5,
    "指令遵循": 1.2, "结构化": 1.0, "鲁棒性": 1.0, "双语": 1.0, "安全": 1.0,
    "填空": 0.8, "问答": 0.8, "语言": 0.8, "续写": 0.6,
    "长上下文": 1.5, "编码": 0.5, "数学": 0.5,
}

TIERS: Tuple[str, ...] = ("E", "M", "H")
TIER_TARGET_RATE = {"E": (0.70, 0.95), "M": (0.25, 0.75), "H": (0.02, 0.35)}
TIER_QUOTA = {"E": 0.20, "M": 0.50, "H": 0.30}
TIER_LABEL = {"E": "easy 热身", "M": "medium 主力", "H": "hard 压力"}

AXES: Tuple[str, ...] = (
    "rp_core", "setting_follow", "style_hold", "emotion", "memory", "identity",
    "format", "constraint", "factual", "lang_zh", "lang_en", "translate",
    "idiom", "creative", "reasoning", "robust", "safety", "bilingual", "longctx",
)

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")
_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")
_SEPARATORS = ("用户：", "助手：", "设定：")

# Expected size of each category, used by the coverage report.  Keeping this in
# one place makes "we shipped 1,000 items" a checkable claim rather than a hope.
CATEGORY_TARGET: Dict[str, int] = {
    "角色扮演": 100, "多轮": 90, "一致性": 70,
    "指令遵循": 80, "结构化": 60, "鲁棒性": 50, "双语": 45, "安全": 35,
    "填空": 120, "问答": 120, "语言": 90, "续写": 60,
    "长上下文": 50, "编码": 25, "数学": 25,
}


@dataclass
class Item:
    id: str
    cat: str
    tier: str
    family: str
    turns: List[str]
    max_new: int
    chk: str
    axes: List[str]
    system: Optional[str] = None
    temp: Optional[float] = None
    note: str = ""
    # build-time provenance, filled in by the loader
    src: str = ""
    line: int = 0
    # compiled lazily so the frozen JSONL stays pure data
    _fn: Optional[Callable] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ basics
    @property
    def prompt(self) -> str:
        """First user turn — what the schema calls the item's prompt."""
        return self.turns[0]

    @property
    def is_multi_turn(self) -> bool:
        return len(self.turns) > 1

    @property
    def axes_set(self) -> List[str]:
        return list(self.axes)

    def checker(self) -> Callable:
        if self._fn is None:
            self._fn = compile_checker(self.chk)
        return self._fn

    def to_json(self) -> Dict:
        d = {"id": self.id, "cat": self.cat, "tier": self.tier,
             "family": self.family, "turns": list(self.turns),
             "max_new": self.max_new, "chk": self.chk, "axes": list(self.axes)}
        if self.system:
            d["system"] = self.system
        if self.temp is not None:
            d["temp"] = self.temp
        if self.note:
            d["note"] = self.note
        return d


class ItemError(ValueError):
    pass


def _fail(where: str, msg: str):
    raise ItemError(f"{where}: {msg}")


# -------------------------------------------------------------------- loading


def parse_item(obj: Dict, where: str) -> Item:
    """Validate one authored JSON object, raising :class:`ItemError`."""
    if not isinstance(obj, dict):
        _fail(where, "line is not a JSON object")

    missing = [k for k in ("id", "cat", "tier", "turns", "max_new", "chk", "axes")
               if k not in obj]
    if missing:
        _fail(where, f"missing required field(s) {missing}")

    iid = obj["id"]
    if not isinstance(iid, str) or not _ID_RE.match(iid):
        _fail(where, f"bad id {iid!r} (want ^[a-z][a-z0-9_]{{2,47}}$)")

    cat = obj["cat"]
    if cat not in CATEGORIES:
        _fail(where, f"unknown cat {cat!r}; must be one of {list(CATEGORIES)}")

    tier = obj["tier"]
    if tier not in TIERS:
        _fail(where, f"bad tier {tier!r}; must be E/M/H")

    family = obj.get("family") or iid.rsplit("_", 1)[0]
    if not isinstance(family, str) or not _FAMILY_RE.match(family):
        _fail(where, f"bad family {family!r}")

    turns = obj["turns"]
    if isinstance(turns, str):
        turns = [turns]
    if not isinstance(turns, list) or not turns:
        _fail(where, "turns must be a non-empty list")
    norm_turns: List[str] = []
    for i, t in enumerate(turns):
        if isinstance(t, dict):
            t = t.get("user", t.get("text", ""))
        if not isinstance(t, str) or not t.strip():
            _fail(where, f"turns[{i}] is empty or not a string")
        for sep in _SEPARATORS:
            if sep in t:
                _fail(where, f"turns[{i}] contains the encoding separator {sep!r}")
        norm_turns.append(t)

    mn = obj["max_new"]
    if not isinstance(mn, int) or not 8 <= mn <= 512:
        _fail(where, f"max_new must be an int in [8, 512], got {mn!r}")

    chk = obj["chk"]
    if not isinstance(chk, str) or not chk.strip():
        _fail(where, "chk must be a non-empty string")
    try:
        compile_checker(chk)
    except CheckerError as exc:
        _fail(where, f"checker does not compile: {exc}")

    axes = obj["axes"]
    if isinstance(axes, str):
        axes = [axes]
    if not isinstance(axes, list) or not 1 <= len(axes) <= 3:
        _fail(where, f"axes must be a list of 1..3 labels, got {axes!r}")
    bad = [a for a in axes if a not in AXES]
    if bad:
        _fail(where, f"unknown axis label(s) {bad}; see CONTRACT §5")

    system = obj.get("system")
    if system is not None:
        if not isinstance(system, str) or not system.strip():
            _fail(where, "system must be a non-empty string when present")
        for sep in _SEPARATORS:
            if sep in system:
                _fail(where, f"system contains the encoding separator {sep!r}")

    temp = obj.get("temp")
    if temp is not None:
        if not isinstance(temp, (int, float)) or float(temp) not in (0.0,):
            _fail(where, f"temp may only be 0.0 or omitted (got {temp!r}); "
                         f"non-zero temperatures use the global default")

    note = obj.get("note", "")
    if not isinstance(note, str):
        _fail(where, "note must be a string")
    if len(note) > 200:
        _fail(where, f"note is {len(note)} chars (max 200)")

    return Item(id=iid, cat=cat, tier=tier, family=family, turns=norm_turns,
                max_new=mn, chk=chk, axes=list(axes), system=system,
                temp=float(temp) if temp is not None else None, note=note)


def load_jsonl(path: Path) -> List[Item]:
    items: List[Item] = []
    text = path.read_text(encoding="utf-8")
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name}:{n}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ItemError(f"{where}: invalid JSON ({exc.msg} at col {exc.colno})") from exc
        it = parse_item(obj, where)
        it.src, it.line = path.name, n
        items.append(it)
    return items


def load_items_dir(d: Path) -> List[Item]:
    items: List[Item] = []
    for p in sorted(d.glob("*.jsonl")):
        items.extend(load_jsonl(p))
    return items


# ------------------------------------------------------------------- validation


def check_duplicates(items: Sequence[Item]) -> List[str]:
    seen: Dict[str, str] = {}
    problems: List[str] = []
    for it in items:
        if it.id in seen:
            problems.append(f"duplicate id {it.id!r}: {seen[it.id]} and {it.src}:{it.line}")
        else:
            seen[it.id] = f"{it.src}:{it.line}"
    return problems


# Predicates whose arguments are legitimately echoed by the prompt itself (a
# sentence-pattern task quotes the pattern; an "answer in exactly this form"
# task quotes the form).  Free-pass detection must skip those.
_ANCHOR_EXEMPT = frozenset({
    "eq", "number_only", "num", "num_f", "num_any", "is_int_in",
    "steps", "repeat", "numbered", "startswith", "endswith", "regex_count",
    "json", "json_has", "json_nested",           # the task names the keys
    "kv", "yaml", "xml", "caps",
    "not_echo_prompt", "not_echo_user", "max_tokens",
})

# Items whose anchors are *quoted by the prompt by design*: a sentence-pattern
# task names the pattern, so `all:要是,就` is not a free pass — it is the task.
# Those items carry `not_echo_prompt`, which is what stops a pure echo scoring.
_TEMPLATE_FAMILIES = frozenset({"lang_pattern"})

# Cues that the user turn is *attacking* the anchor rather than supplying it.
# "别说老孙了" / "哪有什么白狐" / "你其实是 AI 吧" quote the anchor precisely so
# that quoting it is the failure mode; an item whose check requires the model to
# keep using it is testing hold-under-pressure, not recall.  Detecting these is
# what keeps `check_anchors_in_prompt` from flagging the battery's best probes.
_NEGATION_CUES = (
    "不要", "别说", "不准", "别再", "不许", "没有", "哪有什么", "早就",
    "别装了", "假的", "不是", "别演", "改成", "改叫", "从现在起你叫",
    "请用正常", "忽略", "其实", "结束", "不说", "哪来", "骗", "真身份",
    "别叫", "不叫", "不对",
)


def _attacked_by_user(anchor: str, users: str) -> bool:
    """True when every mention of ``anchor`` sits inside a negating cue."""
    for m in re.finditer(re.escape(anchor), users):
        lo = max(0, m.start() - 14)
        window = users[lo:m.end() + 8]
        if not any(cue in window for cue in _NEGATION_CUES):
            return False
    return True


def _has_echo_guard(chk: str) -> bool:
    """``!endswith:？`` — the item already refuses a question-shaped echo.

    For a multi-turn recall item the question often has to name the fact
    ("我现在住哪？" after the user said "我搬到苏州了"), so the anchor sits in the
    final user turn by design and *the guard*, not the anchor, is what stops a
    parrot from scoring.  See ``scripts/fix_recall_checks.py``.
    """
    from .checkers import _split_specs  # same package, private helper

    return any(p.strip().startswith("!endswith") for p in _split_specs(chk))


def anchor_problems(iid: str, chk: str, prior: str = "", final: str = "",
                    family: str = "", include_system: bool = False) -> List[str]:
    """An ``any:`` anchor the prompt hands to the model for free.

    The rule follows what the model can see when it answers:

    * **single-turn item** — the whole prompt is visible, so any anchor in it is
      a free pass.  "北京有多少人口" with ``any:北京`` scores on the question.
    * **multi-turn item** — only the *final* user turn is under test.  A fact
      introduced in turn 1 and asked about in turn 4 is not a free pass, it is
      the recall task; the model still has to carry it across its own replies.
      An anchor quoted by the **final** turn is a free pass unless the item
      guards against a question-shaped echo (``!endswith:？``, added by
      ``scripts/fix_recall_checks.py``).

    Two situations are reported but never rejected, because rejecting them would
    delete the battery's best probes:

    * **hold-under-pressure** — every mention sits in a negating cue
      ("别说老孙了", "哪有什么白狐").  The user quoting the anchor is the *setup*;
      the check is that the model keeps producing it anyway.
    * **persona** (``include_system=True``) — the ``system`` block establishes
      the anchor.  Correct for an in-persona probe: without it silence passes.

    ``family`` exempts template items (see ``_TEMPLATE_FAMILIES``).
    """
    from .checkers import _split_specs, _split_top  # same package, private helper

    if family in _TEMPLATE_FAMILIES:
        return []
    guarded = _has_echo_guard(chk)
    visible = final if prior.strip() else (final or prior)
    scope = (prior + "\n" + final).strip() if include_system else visible
    if not scope.strip():
        return []

    out: List[str] = []
    for part in _split_specs(chk):
        p = part.strip()
        if p.startswith("!"):          # `!any:` names what must NOT appear, so
            continue                   # its presence in the prompt is the point
        name, _, arg = p.partition(":")
        name = name.strip()
        if name not in ("any", "all", "contains_all") or not arg:
            continue
        for v in _split_top(arg, ","):
            v = v.strip()
            if len(v) < 2 or v not in scope:
                continue
            # A short alternative that is a substring of a longer one in the same
            # predicate is not a second finding: `any:[1, 2, 3],[1,2,3]` splits on
            # both commas and reports the fragment "[1" for a prompt that happens
            # to contain it.  Report the longest form only.
            if any(v != w and v in w for w in _split_top(arg, ",")):
                continue
            if not include_system and _attacked_by_user(v, scope):
                continue
            if guarded and v in final:
                out.append(f"{iid}: anchor {v!r} is a guarded recall target "
                           f"(!endswith blocks the echo)")
                continue
            if prior.strip() and v not in final:
                out.append(f"{iid}: anchor {v!r} comes from an earlier turn "
                           f"(recall setup, not a free pass)")
                continue
            out.append(f"{iid}: anchor {v!r} already appears in the prompt")
    return out


# Anchors the question hands over *on purpose*: the user names the thing in order
# to attack it, deny it, or ask about it, and the item's whole point is that the
# model keeps producing it anyway.  These are the battery's best adversarial
# probes; rejecting them would delete the ability to measure hold-under-pressure,
# which is the project's main line.
#
# Listed by id so a *changed* item re-triggers the finding instead of inheriting
# the exemption.  Each entry was read by hand — see docs/14 §"锚点白名单".
REVIEWED_ANCHOR_EXCEPTIONS = {
    "rp_style_08": "用户把「长安」写进对比问句；判据是文言虚词数，锚点只是话题",
    "rp_style_09": "「钥匙」是用户自己的问题；判据「从逻辑上讲」来自 system",
    "rp_conflict_07": "用户索要银子；判据要求财主拒绝，锚点在拒绝语里是自然的",
    "rp_hold_09": "越狱探针：用户自称系统消息并直接问设定内容，守住才算过",
    "rp_longpersona_04": "追问长设定细节；锚点来自 system 的誓言与地名",
    "rp_longpersona_05": "用户点出「食指」追问伤疤来历，考的是长设定记忆",
    "rp_longpersona_08": "用户点名问「团团是谁」，锚点由 system 定义",
    "rp_longpersona_12": "诱导下塔：「灯塔」在用户句里，守住誓言才算过",
    "rp_longpersona_17": "用户出钱买隐私；判据要求拒绝并保留「老太太/粥」",
    "rp_longpersona_18": "用户质疑经历是编的；判据要求守住细节且不认编",
    "rp_bilingual_02": "中英夹杂人设：用户问「酱油」，判据要求英文词出现",
    "cons_conflict_08": "用户拿学费换「加料」；判据要求拒绝且保留女儿/老人",
    "code_23": "输出预测题，判据锚点是被问代码里的字面量",
    "code_24": "改错题，锚点是应改成的写法，与被问代码同族",
}


def check_anchors_in_prompt(items: Sequence[Item]) -> List[str]:
    """Hard errors only: an anchor the *question under test* hands over for free.

    Multi-turn recall setups and guarded recall targets are returned by
    :func:`anchor_notes`, not here — see :func:`anchor_problems` for why they are
    legitimate.
    """
    problems: List[str] = []
    for it in items:
        prior = "\n".join(it.turns[:-1])
        final = it.turns[-1] if it.turns else ""
        problems.extend(
            p for p in anchor_problems(it.id, it.chk, prior, final, it.family,
                                       include_system=False)
            if "comes from an earlier turn" not in p
            and "guarded recall target" not in p
            and it.id not in REVIEWED_ANCHOR_EXCEPTIONS)
    return problems


def reviewed_anchor_exceptions(items: Sequence[Item]) -> List[str]:
    """Which hand-reviewed exceptions were actually triggered by this battery."""
    ids = {i.id for i in items}
    return [f"{k}: {v}" for k, v in sorted(REVIEWED_ANCHOR_EXCEPTIONS.items())
            if k in ids]


def anchor_notes(items: Sequence[Item]) -> List[str]:
    """Reported-not-rejected: persona anchors, hold-under-pressure anchors,
    multi-turn recall setups and guarded recall targets."""
    out: List[str] = []
    for it in items:
        prior = "\n".join(it.turns[:-1])
        final = it.turns[-1] if it.turns else ""
        hard = set(check_anchors_in_prompt([it]))
        soft = set(anchor_problems(it.id, it.chk, prior, final, it.family, True))
        out.extend(sorted(soft - hard))
    return out


def check_persona_anchors(items: Sequence[Item]) -> List[str]:
    """Alias kept for the compile report: persona + recall-anchor notes."""
    return anchor_notes(items)


def check_tier_quota(items: Sequence[Item], tol: float = 0.09) -> List[str]:
    """Warn when a category's tier mix drifts far from the 20/50/30 design."""
    problems: List[str] = []
    by_cat: Dict[str, Dict[str, int]] = {}
    for it in items:
        by_cat.setdefault(it.cat, {"E": 0, "M": 0, "H": 0})[it.tier] += 1
    for cat, counts in sorted(by_cat.items()):
        n = sum(counts.values())
        if n < 20:
            continue
        for tier, target in TIER_QUOTA.items():
            share = counts[tier] / n
            if abs(share - target) > tol:
                problems.append(
                    f"{cat}: tier {tier} share {share:.0%} deviates from "
                    f"{target:.0%} by more than {tol:.0%}")
    return problems


def lint_items(items: Sequence[Item]) -> List[str]:
    out: List[str] = []
    for it in items:
        for w in lint_checker(it.chk, "\n".join(it.turns)):
            out.append(f"{it.id}: {w}")
    return out


def battery_summary(items: Sequence[Item]) -> Dict:
    by_cat: Dict[str, int] = {}
    by_tier: Dict[str, int] = {}
    by_axis: Dict[str, int] = {}
    fams: Dict[str, int] = {}
    preds: Dict[str, int] = {}
    for it in items:
        by_cat[it.cat] = by_cat.get(it.cat, 0) + 1
        by_tier[it.tier] = by_tier.get(it.tier, 0) + 1
        for a in it.axes:
            by_axis[a] = by_axis.get(a, 0) + 1
        fams[it.family] = fams.get(it.family, 0) + 1
        for p in predicate_names(it.chk):
            preds[p] = preds.get(p, 0) + 1
    return {"n": len(items), "by_cat": by_cat, "by_tier": by_tier,
            "by_axis": by_axis, "families": fams, "predicates": preds}


def coverage_vs_target(items: Sequence[Item]) -> List[Tuple[str, int, int]]:
    have = battery_summary(items)["by_cat"]
    return [(c, have.get(c, 0), t) for c, t in CATEGORY_TARGET.items()]
