"""Battery runner — executes the frozen dataset against checkpoints.

Design constraints that come straight from ``docs/13``:

1. **Determinism.**  The item order is fixed, and each lockstep group resets the
   global RNG once before generating, so a rerun on the same checkpoint and seed
   reproduces the same replies.  No shuffling, no worker-order dependence.
2. **Resumability.**  A 1,000-item run is hours of GPU time; a crash at item 900
   must not cost the first 899.  Results are appended to a JSONL as they
   complete, and a rerun skips what is already there.
3. **Per-item temperature.**  ``temp: 0.0`` in the item means greedy — the
   deterministic baseline ``docs/13`` §5.1 item 5 asks for.
4. **The decoder is the model's own.**  ``bench.fastgen`` is asserted
   bit-identical to ``RetNetForCausalLM.generate``; the runner only calls that
   path.
5. **think-leak is a first-class metric** and never enters the score.

Throughput
----------
``scripts/bench_decode_profile.py`` shows a decode step is ~29.6 ms of forward
pass against ~0.8 ms of Python, so the model is latency-bound at batch 1 and no
amount of interpreter tuning helps.  ``scripts/bench_batch_scaling.py`` then
measures what lockstep batching is worth end to end: **1.46x**, flat from batch
8 to batch 32, because a step is dominated by streaming 272M fp32 weights and
that cost does not grow with the batch.

That is a poor trade, because a batched GEMM is not bit-identical to a
single-row one: a ~1e-6 logit difference flips an occasional multinomial draw
and the reply then diverges (measured: 8 of 107 items on a stride-97 slice).

``--batch`` therefore defaults to **1** — bit-exact, reproducible, and directly
comparable with ``bench_v2`` and ``play.py``.  ``--batch 16`` is available when
a run needs to be ~1.5x faster and a small numerical deviation is acceptable;
the report records which mode produced it.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from retnet.chat import encode_prompt
from retnet_hf.modeling_retnet import RetNetForCausalLM

from .checkers import Ctx
from .fastgen import BatchedDecoder, Request, fast_generate, make_rng_seed
from .registry import Item

# ------------------------------------------------------------------- decoding


@dataclass
class GenCfg:
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 4
    lc_prefill: str = "parallel"          # parallel | chunkwise (长上下文 prompt)
    batch: int = 1                        # 1 = bit-exact; >1 = lockstep, ~1.5x
    long_prompt_tokens: int = 512         # longer prompts always decode alone


def cut(text: str) -> str:
    """Keep only the assistant's own reply, as bench_v2 did."""
    return text.split("用户：")[0].split("助手：")[0].strip()


def hash_dataset(items: Sequence[Item]) -> str:
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x.id):
        h.update(json.dumps(it.to_json(), ensure_ascii=False,
                            sort_keys=True).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------- runner


class BatteryRunner:
    def __init__(self, model: RetNetForCausalLM, tok, items: Sequence[Item],
                 cfg: GenCfg, device: str = "cuda"):
        self.model = model
        self.tok = tok
        self.items = list(items)
        self.cfg = cfg
        self.device = device
        self.eos_id = tok.eos_id
        self.decoder = BatchedDecoder(model, device=device)

    # -- helpers ----------------------------------------------------------
    def _encode(self, item: Item, history: Sequence[Tuple[str, str]],
                turn: str) -> List[int]:
        return encode_prompt(self.tok, history, turn, system=item.system)

    def _item_temp(self, item: Item) -> float:
        return item.temp if item.temp is not None else self.cfg.temperature

    def _req(self, item: Item, ids: List[int], batchable: bool,
             seed: int = 0) -> Request:
        temp = self._item_temp(item)
        greedy = temp <= 0
        return Request(
            ids=ids, max_new=item.max_new,
            temperature=0.0 if greedy else temp,
            top_k=None if greedy else self.cfg.top_k,
            top_p=None if greedy else self.cfg.top_p,
            repetition_penalty=1.0 if greedy else self.cfg.repetition_penalty,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            eos_id=self.eos_id,
            prefill=self.cfg.lc_prefill,
            batchable=batchable,
            tag=item.id,
            rng_seed=make_rng_seed(seed, item.id))

    def _continue(self, item: Item, history: List[Tuple[str, str]],
                  turn: str, seed: int = 0) -> Tuple[str, int]:
        """One follow-up assistant turn (always single-item)."""
        ids = self._encode(item, history, turn)
        x = torch.tensor([ids], device=self.device)
        temp = self._item_temp(item)
        greedy = temp <= 0
        out = fast_generate(
            self.model, x, max_new_tokens=item.max_new,
            temperature=0.0 if greedy else temp,
            top_k=None if greedy else self.cfg.top_k,
            top_p=None if greedy else self.cfg.top_p,
            repetition_penalty=1.0 if greedy else self.cfg.repetition_penalty,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            eos_id=self.eos_id,
            rng_seed=make_rng_seed(seed, item.id))
        new_ids = out[0].tolist()[len(ids):]
        return cut(self.tok.decode(new_ids, skip_special_tokens=True)), len(new_ids)

    def _judge(self, item: Item, replies: Sequence[str], n_new: int,
               dt: float, temperature: float, seed: Optional[int] = None,
               per_turn: Optional[Sequence[int]] = None) -> Dict:
        reply = replies[-1] if replies else ""
        ctx = Ctx(reply=reply, prompt=item.turns[-1], users=item.turns,
                  system=item.system, n_new=n_new)
        try:
            ok = bool(item.checker()(ctx))
        except Exception as exc:                               # pragma: no cover
            ok = False
            replies = list(replies) + [f"<<checker error: {exc}>>"]
        # NOTE(2026-10-10 晚)：修两个会让整轮跑作废的缺陷。
        # (1) 旧实现在 replies[-1] 上找 <think>，多轮题第 1..n-1 轮的泄漏**完全不可见**
        #     （141/1020 题受影响，占 13.8%）。现在逐轮检查，任一轮未闭合即算泄漏，
        #     并把出问题的轮号记进 leak_turns 供复核。
        # (2) 旧实现把回复截断存储（reply[:600] / all_replies 每条 [:400]）。满预算下
        #     回复能到 1000+ token、上千字符，截断后**事后无法重新判定闭合**——
        #     而「给满预算会不会闭合」正是这轮跑的唯一目的。现在存全文。
        leak_turns = []
        for _ti, _t in enumerate(replies):
            _at = (_t or "").find("<think>")
            if _at >= 0 and "</think>" not in _t[_at:]:
                leak_turns.append(_ti)
        leak = bool(leak_turns)
        n_turns = max(len(item.turns), 1)
        if per_turn:
            # 逐轮长度是精确判据。旧的 n_new >= max_new × 轮数 在多轮上只是保守下界：
            # 若前几轮写满、最后一轮提前收尾，总数就够不到上限，会被漏判成未截断。
            truncated = bool(any(x >= item.max_new for x in per_turn))
        else:
            truncated = bool(n_new >= item.max_new * n_turns)
        rec = {
            "id": item.id, "cat": item.cat, "tier": item.tier,
            "family": item.family, "axes": list(item.axes),
            "temp": temperature, "pass": ok,
            "think_leak": leak,
            # 判据：单轮精确；多轮在 per_turn 可用时是逐轮精确判据，否则退回保守下界。
            "truncated": truncated,
            "max_new": item.max_new,
            # 只在没被截断的子集上才有意义的泄漏标志（见 analysis.py 的聚合）
            "think_leak_untruncated": bool(leak and not truncated),
            "n_new": n_new, "seconds": round(dt, 3),
            "n_new_per_turn": list(per_turn) if per_turn else None,
            "leak_turns": leak_turns,
            "reply": reply,
            "all_replies": list(replies) if len(replies) > 1 else None,
        }
        if seed is not None:
            rec["seed"] = seed
        return rec

    # -- single item (reference path) -------------------------------------
    def run_item(self, item: Item, seed: int) -> Dict:
        """Run one item alone — the semantics the lockstep path must match."""
        torch.manual_seed(seed)
        t0 = time.time()
        history: List[Tuple[str, str]] = []
        replies: List[str] = []
        n_new = 0
        per_turn: List[int] = []
        for ti, turn in enumerate(item.turns):
            reply, k = self._continue(item, history, turn, seed)
            replies.append(reply)
            n_new += k
            per_turn.append(k)
            history.append((turn, reply))
            _ = ti
        return self._judge(item, replies, n_new, time.time() - t0,
                           self._item_temp(item), seed, per_turn)

    # -- lockstep group ---------------------------------------------------
    def plan_group(self, group: Sequence[Item],
                   seed: int) -> Tuple[List[Request], List[int]]:
        """First-turn requests for the group, and the indexes decoded solo."""
        reqs, solo = [], []
        for i, it in enumerate(group):
            ids = self._encode(it, [], it.turns[0])
            batchable = (not it.is_multi_turn
                         and len(ids) <= self.cfg.long_prompt_tokens)
            if not batchable:
                solo.append(i)
            reqs.append(self._req(it, ids, batchable, seed))
        return reqs, solo

    def run_group(self, group: Sequence[Item], seed: int) -> List[Dict]:
        """Decode a lockstep group under one seeding, then finish each item."""
        torch.manual_seed(seed)
        t0 = time.time()
        reqs, solo = self.plan_group(group, seed)
        joint = [i for i in range(len(group)) if i not in set(solo)]

        if joint:
            self.decoder.run([reqs[i] for i in joint])
        for i in solo:
            self.decoder.run([reqs[i]])

        out: List[Dict] = []
        for i, it in enumerate(group):
            replies = [cut(self.tok.decode(reqs[i].out, skip_special_tokens=True))]
            n_new = len(reqs[i].out)
            per_turn = [n_new]
            history: List[Tuple[str, str]] = [(it.turns[0], replies[0])]
            for turn in it.turns[1:]:
                reply, k = self._continue(it, history, turn, seed)
                replies.append(reply)
                n_new += k
                per_turn.append(k)
                history.append((turn, reply))
            out.append(self._judge(it, replies, n_new,
                                   (time.time() - t0) / max(1, len(group)),
                                   self._item_temp(it), seed, per_turn))
        return out

    # -- whole battery ----------------------------------------------------
    def groups(self) -> Iterator[List[Item]]:
        """Contiguous item spans of at most ``cfg.batch`` items.

        With the default ``batch=1`` this is one item per group, i.e. the
        bit-exact path; ``run_group`` then degenerates to ``run_item`` with the
        same per-item RNG stream.
        """
        if self.cfg.batch <= 1:
            for it in self.items:
                yield [it]
            return
        cur: List[Item] = []
        for it in self.items:
            cur.append(it)
            if len(cur) >= self.cfg.batch:
                yield cur
                cur = []
        if cur:
            yield cur

    def run(self, seeds: Sequence[int], out_path: Path,
            progress=None) -> List[Dict]:
        done = load_done(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records: List[Dict] = []
        t_start = time.time()
        total = len(self.items) * len(seeds)
        n_done = 0
        fresh = 0
        with out_path.open("a", encoding="utf-8", newline="\n") as fh:
            for seed in seeds:
                for group in self.groups():
                    pending = [it for it in group if (it.id, seed) not in done]
                    n_done += len(group) - len(pending)
                    if not pending:
                        continue
                    recs = self.run_group(pending, seed)
                    for rec in recs:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    records.extend(recs)
                    fresh += len(pending)
                    n_done += len(pending)
                    if progress and fresh >= 10:
                        # NOTE(2026-10-10 晚)：这里原本是 `el / max(1, fresh)`，把**累计**
                        # elapsed 除以**本窗口**的样本数（fresh 每轮清零），于是打印出的
                        # ms/样本 与 eta 会随进度线性膨胀（实测 80 条时虚报 9282 ms，
                        # 真值 1160 ms，差 8 倍）。改成除以本次运行真正生成的条数。
                        el = time.time() - t_start
                        progress(n_done, total, el,
                                 el / max(1, n_done - len(done)))
                        fresh = 0
        if progress:
            progress(n_done, total, time.time() - t_start,
                     (time.time() - t_start) / max(1, n_done - len(done) or 1))
        return records


def load_done(path: Path) -> set:
    """``{(item_id, seed)}`` already present in a results file."""
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["id"], r["seed"]))
            except Exception:
                continue
    return done


# ------------------------------------------------------------------ model load


def load_model(path: str | Path, device: str = "cuda"):
    model = RetNetForCausalLM.from_pretrained(str(path))
    n_param = sum(p.numel() for p in model.parameters())
    model = model.to(device).eval()
    return model, n_param


def load_tokenizer(path: str | Path):
    from retnet.tokenizer import BPETokenizer
    return BPETokenizer.load(str(path))


# ------------------------------------------------------------------ selection


def select(items: Sequence[Item], only: Optional[str] = None,
           families: Optional[str] = None, ids: Optional[str] = None,
           limit: int = 0, stride: int = 1, offset: int = 0) -> List[Item]:
    """Filter the battery.  ``stride``/``offset`` give a cheap representative
    subset without disturbing the deterministic order (smoke runs)."""
    out = list(items)
    if only:
        keep = {c.strip() for c in only.split(",") if c.strip()}
        out = [i for i in out if i.cat in keep]
    if families:
        keep = {c.strip() for c in families.split(",") if c.strip()}
        out = [i for i in out if i.family in keep]
    if ids:
        keep = [c.strip() for c in ids.split(",") if c.strip()]
        order = {k: n for n, k in enumerate(keep)}
        out = sorted((i for i in out if i.id in order), key=lambda i: order[i.id])
    if stride > 1:
        out = out[offset::stride]
    if limit:
        per: Dict[str, int] = {}
        trimmed = []
        for i in out:
            per[i.cat] = per.get(i.cat, 0) + 1
            if per[i.cat] <= limit:
                trimmed.append(i)
        out = trimmed
    return out
