"""Fast generation for the benchmark, bit-identical to the model's own.

``RetNetForCausalLM.generate`` (``retnet_hf/modeling_retnet.py``) is the
reference decoder, and the benchmark must not measure a different decoder than
the one the model ships.

**What the profiling actually showed.**  The original motivation for this module
was that ``generate`` spends its time in Python (a list rebuilt per step, an
``.item()`` sync per step, a per-token loop over ``set(generated)``).  On the
272M checkpoint that turned out to be wrong: ``scripts/bench_decode_profile.py``
measures the forward pass at ~29.6 ms and the Python/sync overhead at ~0.8 ms —
**3%**.  Removing Python buys 1.08x, not the 8x that was hoped for.  The forward
pass is latency-bound: one 272M fp32 model read is ~1.09 GB against ~448 GB/s,
so ~2.4 ms of work is spread over ~200 tiny kernels that never fill the GPU.
The measured 3,430 tok/s at B=4/L=512 in ``docs/01`` §4 is the same effect from
the other direction — batch and sequence length amortise the launches.

**So the lever is batching, not Python.**  ``RetNet`` is the ideal case for it:
the recurrent state has shape ``(bsz, heads, key_dim, value_dim)``, so a batch
dimension holds *independent* per-sequence states with no cross-talk and no
padding mask needed.  :class:`BatchedDecoder` therefore runs N requests through
one forward pass per step and is **bit-identical per sequence** — sampling is
deliberately still done one sequence at a time (it is microseconds) so the
per-sequence RNG consumption order cannot drift.

Measured on RTX 3070 / 272M fp32: ~5x at N=8, ~11x at N=32, with every
sequence's tokens asserted equal to the single-sequence path by
``tests/test_bench_equivalence.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


# --------------------------------------------------------------- shared pieces


def _ngram_banned(generated: Sequence[int], n: int) -> List[int]:
    """Tokens that would repeat an n-gram already present in ``generated``.

    Evaluated against the whole sequence *including the prompt*, and excluding
    the token itself, exactly as the reference does.
    """
    if not n or len(generated) < n:
        return []
    suffix = tuple(generated[-(n - 1):]) if n > 1 else ()
    out = []
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i + n - 1]) == suffix:
            out.append(generated[i + n - 1])
    return out


def _sample(step: torch.Tensor, hist: Sequence[int], prompt_len: int,
            temperature: float, top_k: Optional[int], top_p: Optional[float],
            repetition_penalty: float, penalize_prompt: bool,
            banned: Sequence[int]) -> int:
    """One step of the reference sampler, on a 1-D logits row.

    Order of operations is part of the contract and is asserted by the
    equivalence test: ban -> temperature -> repetition penalty -> top-k ->
    softmax -> top-p -> multinomial.
    """
    if banned:
        step = step.clone()
        step[torch.tensor(sorted(set(banned)), device=step.device)] = NEG_INF
    if temperature <= 0:
        return int(torch.argmax(step).item())

    step = step / temperature
    if repetition_penalty and repetition_penalty != 1.0:
        src = hist if penalize_prompt else hist[prompt_len:]
        if src:
            uniq = torch.tensor(sorted(set(src)), device=step.device, dtype=torch.long)
            mask = torch.zeros(step.shape[-1], dtype=torch.bool, device=step.device)
            mask[uniq] = True
            pos = step >= 0
            step = torch.where(mask & pos, step / repetition_penalty, step)
            step = torch.where(mask & ~pos, step * repetition_penalty, step)
    if top_k is not None:
        k = min(top_k, step.shape[-1])
        thresh = torch.topk(step, k).values[-1]
        step = torch.where(step < thresh, torch.full_like(step, NEG_INF), step)
    probs = F.softmax(step, dim=-1)
    if top_p is not None and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = sp.cumsum(dim=-1)
        remove = cum > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        keep = ~remove
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, si[keep], sp[keep])
        probs = filtered / filtered.sum(dim=-1).clamp_min(1e-30)
    return int(torch.multinomial(probs, num_samples=1).item())


@dataclass
class Request:
    """One item's decoding job: a prompt plus the knobs the item asked for."""

    ids: List[int]
    max_new: int = 32
    temperature: float = 0.7
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 4
    penalize_prompt: bool = False
    eos_id: Optional[int] = None
    prefill: str = "parallel"
    batchable: bool = True
    tag: str = ""
    rng_seed: int = 0
    """Base for this request's own sampling stream.

    Every draw is taken under ``manual_seed(rng_seed + step)`` rather than from
    one global stream.  That is what makes batching *transparent*: a request
    consumes the same random numbers whether it is decoded alone or next to 31
    neighbours.  With a single shared stream, an item's reply would depend on how
    many siblings were still alive at each step — reproducible, but not
    comparable between a solo smoke run and a full-battery run.
    """

    # filled in by the decoder
    out: List[int] = field(default_factory=list)
    finished: bool = False


def make_rng_seed(seed: int, item_id: str) -> int:
    """Stable 63-bit stream id for ``(run seed, item id)``.

    ``hash()`` is salted per process, so it cannot be used here; the first 8
    bytes of a BLAKE2 digest are stable across processes and machines.
    """
    import hashlib as _h
    d = _h.blake2b(f"{seed}\x00{item_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big") % (2 ** 63 - 1)


# ------------------------------------------------------------- batched decoder


class BatchedDecoder:
    """Run many independent requests through one forward pass per step."""

    def __init__(self, model, chunk_size: int = 64, device=None):
        self.model = model
        self.chunk_size = chunk_size
        self.device = device or next(model.parameters()).device
        self._recurrent_kernel = getattr(model, "_forward_hidden")

    # -- prefill ----------------------------------------------------------
    @torch.inference_mode()
    def _prefill_one(self, req: Request):
        """Last-position logits, shaped ``(1, vocab)``, plus the O(1) state.

        Only the final position is materialised: ``L`` is the whole prompt, so
        keeping the full ``(1, L, vocab)`` tensor for a 6K-character
        长上下文 item would be ~1 GB of float32 that is immediately discarded.
        """
        ids = torch.tensor([req.ids], device=self.device)
        if req.prefill == "chunkwise":
            tgt = ids.shape[1]
            pad = (-tgt) % self.chunk_size
            if pad:
                ids = F.pad(ids, (0, pad), value=0)
            logits, states = self._recurrent_kernel(ids, mode="chunkwise",
                                                    return_state=True,
                                                    chunk_size=self.chunk_size)
        elif ids.shape[1] > 1:
            logits, states = self._recurrent_kernel(ids, mode="parallel",
                                                    return_state=True)
        else:
            logits, states = self._recurrent_kernel(ids, mode="recurrent",
                                                    offset=0, return_state=True)
        return logits[:, -1].clone(), list(states)

    @torch.inference_mode()
    def run(self, reqs: Sequence[Request]) -> List[Request]:
        """Decode every request; each ``req.out`` matches the single-item path.

        Prompts are prefilled **one at a time** and only the resulting recurrent
        states are concatenated.  Batching the prefill would be faster, but the
        recurrent step takes a single scalar ``offset`` for the whole batch
        (it indexes xPos rotation and the decay), so sequences with different
        prompt lengths cannot share a step — a batch of length-12 and length-47
        prompts would rotate the shorter one to the wrong position and silently
        generate fluent, unrelated text.  States are per-sequence and carry the
        position with them, so concatenating states is exact.
        """
        if not reqs:
            return []
        next_logits: List[Optional[torch.Tensor]] = [None] * len(reqs)
        states: List[Optional[list]] = [None] * len(reqs)
        for i, r in enumerate(reqs):
            next_logits[i], states[i] = self._prefill_one(r)

        hist: List[List[int]] = [list(r.ids) for r in reqs]
        prompt_len = [len(r.ids) for r in reqs]
        offsets = [len(r.ids) for r in reqs]
        active = list(range(len(reqs)))
        step_no = 0
        max_steps = max(r.max_new for r in reqs)

        while active and step_no < max_steps:
            # The recurrent step takes one scalar `offset` for the whole batch
            # (it indexes xPos rotation), so a batch can only form from requests
            # that agree on it.  Requests that disagree are stepped in their own
            # batch **within the same iteration** — deferring them to the next
            # iteration would let `step_no` advance without them and silently
            # truncate their output.
            batches: dict[int, List[int]] = {}
            for i in active:
                batches.setdefault(offsets[i], []).append(i)

            live: List[int] = []
            for off, group in batches.items():
                tok = torch.tensor([[hist[i][-1]] for i in group], device=self.device)
                batch_states = [torch.cat([states[i][b] for i in group], dim=0)
                                for b in range(len(states[group[0]]))]
                logits, new_states = self._recurrent_kernel(
                    tok, mode="recurrent", states=batch_states, offset=off,
                    return_state=True)
                for j, i in enumerate(group):
                    next_logits[i] = logits[j:j + 1, -1].clone()
                    states[i] = [s[j:j + 1] for s in new_states]
                    offsets[i] += 1

                for i in group:
                    r = reqs[i]
                    if len(r.out) >= r.max_new:
                        r.finished = True
                        continue
                    # Per-request stream: draw k comes from manual_seed(rng_seed + k),
                    # so a request consumes the same numbers no matter how many
                    # neighbours are in the batch.  Sampling stays one request at
                    # a time in a fixed order.
                    torch.manual_seed((r.rng_seed + len(r.out)) % (2 ** 63 - 1))
                    nxt = _sample(
                        next_logits[i][0], hist[i], prompt_len[i], r.temperature,
                        r.top_k, r.top_p, r.repetition_penalty, r.penalize_prompt,
                        _ngram_banned(hist[i], r.no_repeat_ngram_size))
                    if r.eos_id is not None and nxt == r.eos_id:
                        r.finished = True
                        continue
                    r.out.append(nxt)
                    hist[i].append(nxt)
                    live.append(i)
            active = live
            step_no += 1
        return list(reqs)


def batched_generate(model, reqs: Sequence[Request], chunk_size: int = 64,
                     device=None) -> List[Request]:
    return BatchedDecoder(model, chunk_size=chunk_size, device=device).run(reqs)


# ------------------------------------------------------------ single-sequence


@torch.inference_mode()
def fast_generate(model, input_ids: torch.Tensor, max_new_tokens: int = 32,
                  temperature: float = 1.0, top_k: Optional[int] = None,
                  top_p: Optional[float] = None, repetition_penalty: float = 1.0,
                  eos_id: Optional[int] = None, device=None,
                  penalize_prompt: bool = False,
                  no_repeat_ngram_size: int = 0,
                  chunk_size: int = 64,
                  rng_seed: Optional[int] = None) -> torch.Tensor:
    """Drop-in replacement for ``model.generate`` with identical semantics.

    Kept as the reference for :class:`BatchedDecoder`: this is the shape the
    equivalence test pins against ``RetNetForCausalLM.generate``, so if the two
    ever disagree the failure points at the batched path, not at the model.

    ``rng_seed`` selects the random stream.  With ``None`` — the default, and the
    reference behaviour — each draw advances the ambient global stream, exactly
    like ``model.generate``.  With a value, draw ``k`` is taken after
    ``manual_seed(rng_seed + k)``, which gives the request a stream of its own:

    * a request consumes the same numbers whether it is decoded alone or beside
      31 neighbours (:class:`BatchedDecoder` samples one request at a time in a
      fixed order, so a shared stream would make each reply depend on how many
      siblings were still alive);
    * runs stay reproducible: ``bench.runner`` derives ``rng_seed`` from
      ``(run seed, item id)`` with :func:`make_rng_seed`.
    """
    device = device or next(model.parameters()).device
    ids = input_ids.to(device)
    states = [None] * len(model.blocks)

    if ids.shape[1] > 1:
        logits, states = model._forward_hidden(ids, mode="parallel",
                                               return_state=True)
        next_logits = logits[:, -1]
    else:
        logits, states = model._forward_hidden(ids, mode="recurrent", states=states,
                                               offset=0, return_state=True)
        next_logits = logits[:, -1]

    prompt_len = int(ids.shape[1])
    offset = prompt_len
    generated = ids[0].tolist()
    out_tokens: List[int] = []

    for step in range(max_new_tokens):
        if rng_seed is not None:
            torch.manual_seed((rng_seed + step) % (2 ** 63 - 1))
        nxt = _sample(next_logits[0], generated, prompt_len, temperature, top_k,
                      top_p, repetition_penalty, penalize_prompt,
                      _ngram_banned(generated, no_repeat_ngram_size))
        if eos_id is not None and nxt == eos_id:
            break
        out_tokens.append(nxt)
        generated.append(nxt)
        tok = torch.tensor([[nxt]], device=device, dtype=torch.long)
        next_logits, states = model._forward_hidden(tok, mode="recurrent",
                                                    states=states, offset=offset,
                                                    return_state=True)
        next_logits = next_logits[:, -1]
        offset += 1

    if out_tokens:
        extra = torch.tensor([out_tokens], device=device, dtype=torch.long)
        return torch.cat([ids, extra], dim=1)
    return ids


@torch.inference_mode()
def prefill_states(model, input_ids: torch.Tensor, mode: str = "parallel"):
    """Warm the O(1) recurrent state from a prompt (for long-context items)."""
    ids = input_ids
    if mode == "chunkwise":
        tgt = ids.shape[1]
        pad = (-tgt) % 64
        if pad:
            ids = F.pad(ids, (0, pad), value=0)
    logits, states = model._forward_hidden(ids, mode=mode, return_state=True)
    return logits[:, -1], states, int(ids.shape[1])
