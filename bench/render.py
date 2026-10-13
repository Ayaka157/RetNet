"""Prompt composition — mirrors ``retnet.chat`` so item lengths are measurable.

The corpus packs dialogue with::

    <bos>设定：{system}
    用户：{u1}
    助手：{a1}
    用户：{u2}
    助手：

``retnet/chat.py`` is the single implementation used by training *and* by
``scripts/bench_v2.py`` at generation time; the benchmark must keep using it or
the numbers stop being comparable.  What it does not offer is a way to know how
long a prompt will be *before* a tokenizer is available — and the 长上下文
category is defined by character depth.  These helpers reproduce the exact same
layout as plain text so the item generator can size a passage, and
``assert_layout_matches`` guards the duplication against drift.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from retnet.chat import ASSISTANT, SYSTEM, USER

BOS = "<bos>"


def system_block(system: Optional[str]) -> str:
    return f"{SYSTEM}：{system}\n" if system else ""


def turn_block(user: str, first: bool) -> str:
    head = "" if first else "\n"
    return f"{head}{USER}：{user}\n{ASSISTANT}："


def render_prompt(turns: Sequence[str], system: Optional[str] = None) -> str:
    """Exact text form of the ids ``encode_prompt`` would produce."""
    out = [BOS, system_block(system)]
    first = not system
    for u in turns:
        out.append(turn_block(u, first=first))
        first = False
    return "".join(out)


def render_dialogue(turns: Sequence[str], replies: Sequence[str],
                    system: Optional[str] = None) -> str:
    """Full dialogue text — used for length budgeting, not for generation."""
    out = [BOS, system_block(system)]
    first = not system
    for u, r in zip(turns, replies):
        out.append(turn_block(u, first=first))
        out.append(r)
        first = False
    return "".join(out)


def prompt_chars(item) -> int:
    """Characters in the encoded prompt for a single-turn item."""
    return len(render_prompt(item.turns, item.system))


def passage_chars(turns: Sequence[str], system: Optional[str] = None) -> int:
    """Characters an assistant of unknown length contributes nothing to.

    Long-context depth is defined as the size of the *user* material that must
    survive to the final question, so this counts the user turns plus the
    scaffolding — not any model-generated reply.
    """
    return len(render_prompt(turns, system))


def assert_layout_matches(tok, cases: Iterable[Tuple[Sequence[str], Optional[str]]]) -> None:
    """Assert the text layout and the tokenizer layout agree on the boundary.

    ``encode_prompt`` is authoritative; this checks that ``render_prompt`` is a
    faithful text shadow of it, so a future edit to ``retnet/chat.py`` cannot
    silently invalidate the character-depth budgets of the 长上下文 category.
    """
    from retnet.chat import encode_prompt

    for turns, system in cases:
        ids = encode_prompt(tok, [], turns[0], system=system) if len(turns) == 1 else None
        text = render_prompt(turns, system)
        if ids is not None:
            assert tok.decode(ids) .strip().endswith(ASSISTANT + "："), text
        for u in turns:
            assert f"{USER}：{u}" in text, u[:40]
        if system:
            assert text.startswith(f"{BOS}{SYSTEM}：{system}"), "system block drift"
