"""Shared 用户/助手 dialogue encoding used by both data building and inference.

Keeping one implementation guarantees that the token boundaries the model is
trained on are exactly the boundaries it sees at generation time.

Loss-weight codes stored alongside the ids:

===  ============================================================
0    padding, never contributes to the loss
1    assistant reply (the tokens SFT should actually optimize)
2    prompt tokens: user turn text plus the ``助手：`` marker
3    plain language-model text (pretrain replay), always optimized
===  ============================================================
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

USER = "用户"
ASSISTANT = "助手"
SYSTEM = "设定"

W_PAD = 0
W_REPLY = 1
W_PROMPT = 2
W_PLAIN = 3


def system_prefix(system_text: str) -> str:
    """Persona / scene block placed before the first user turn (role-play)."""
    return f"{SYSTEM}：{system_text}\n"


def turn_prefix(user_text: str, first: bool) -> str:
    """Text that precedes an assistant reply (never optimized as a reply)."""
    head = "" if first else "\n"
    return f"{head}{USER}：{user_text}\n{ASSISTANT}："


def reply_text(assistant_text: str) -> str:
    return assistant_text


def encode_dialogue(tok, pairs: Sequence[Tuple[str, str]],
                    max_len: int | None = None,
                    system: str | None = None) -> Tuple[List[int], List[int]]:
    """Encode ``[(user, assistant), ...]`` into ids + loss-weight codes.

    ``system`` carries a role-play persona / scene block; it is prompt-weighted
    so the model conditions on it without being trained to generate it.
    The returned ids already include BOS and EOS.  Truncation keeps the head of
    the conversation so every training sample starts at a real dialogue start.
    """
    ids: List[int] = [tok.bos_id]
    codes: List[int] = [W_PROMPT]
    if system:
        head = tok.encode(system_prefix(system))
        ids.extend(head)
        codes.extend([W_PROMPT] * len(head))
    for i, (user, assistant) in enumerate(pairs):
        prefix = tok.encode(turn_prefix(user, first=(i == 0 and not system)))
        ids.extend(prefix)
        codes.extend([W_PROMPT] * len(prefix))
        reply = tok.encode(reply_text(assistant))
        ids.extend(reply)
        codes.extend([W_REPLY] * len(reply))
    ids.append(tok.eos_id)
    codes.append(W_REPLY)
    if max_len is not None and len(ids) > max_len:
        ids = ids[:max_len]
        codes = codes[:max_len]
        ids[-1] = tok.eos_id
        codes[-1] = W_REPLY
    return ids, codes


def encode_dialogues_batch(tok, dialogues: Sequence[Sequence[Tuple[str, str]]],
                           max_len: int | None = None,
                           systems: Sequence[str | None] | None = None):
    """Batched version of :func:`encode_dialogue` (uses the fast tokenizer).

    ``systems[i]`` optionally supplies a role-play persona for dialogue ``i``.
    Yields ``(ids, codes)`` in the same order as ``dialogues``.
    """
    segments: List[str] = []
    layout: List[Tuple[int, bool]] = []      # (segment count, has_system)
    for idx, pairs in enumerate(dialogues):
        system = systems[idx] if systems is not None else None
        n = 0
        if system:
            segments.append(system_prefix(system))
            n += 1
        for i, (user, assistant) in enumerate(pairs):
            segments.append(turn_prefix(user, first=(i == 0 and not system)))
            segments.append(reply_text(assistant))
            n += 2
        layout.append((n, bool(system)))
    encoded = tok.encode_batch(segments)

    out = []
    pos = 0
    for n, has_system in layout:
        ids: List[int] = [tok.bos_id]
        codes: List[int] = [W_PROMPT]
        start = 0
        if has_system:
            seg = encoded[pos]
            ids.extend(seg)
            codes.extend([W_PROMPT] * len(seg))
            start = 1
        for j in range(start, n):
            seg = encoded[pos + j]
            is_prompt = ((j - start) % 2 == 0)
            ids.extend(seg)
            codes.extend([W_PROMPT if is_prompt else W_REPLY] * len(seg))
        pos += n
        ids.append(tok.eos_id)
        codes.append(W_REPLY)
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
            codes = codes[:max_len]
            ids[-1] = tok.eos_id
            codes[-1] = W_REPLY
        out.append((ids, codes))
    return out


def encode_prompt(tok, history: Iterable[Tuple[str, str]], user_text: str,
                  system: str | None = None) -> List[int]:
    """Ids for a generation prompt, tokenized exactly like the training data."""
    ids: List[int] = [tok.bos_id]
    if system:
        ids.extend(tok.encode(system_prefix(system)))
    first = not system
    for prev_user, prev_reply in history:
        ids.extend(tok.encode(turn_prefix(prev_user, first=first)))
        ids.extend(tok.encode(reply_text(prev_reply)))
        first = False
    ids.extend(tok.encode(turn_prefix(user_text, first=first)))
    return ids


def pairs_from_roles(messages: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Turn a flat ``[(role, text), ...]`` list into (user, assistant) pairs.

    Consecutive same-role messages are merged; a leading assistant message and a
    trailing user message are dropped so every pair is complete.
    """
    merged: List[Tuple[str, str]] = []
    for role, text in messages:
        if not text:
            continue
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + " " + text)
        else:
            merged.append((role, text))
    while merged and merged[0][0] != "user":
        merged.pop(0)
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(merged) - 1, 2):
        if merged[i][0] == "user" and merged[i + 1][0] == "assistant":
            pairs.append((merged[i][1], merged[i + 1][1]))
        else:
            break
    return pairs
