"""Build a sample-aligned, loss-masked chat dataset (packed rows).

Why this exists
---------------
The earlier pipeline concatenated every conversation into one flat token stream
and then cut it into fixed 256-token blocks.  Two problems followed:

1. Most training samples started in the middle of a conversation, so the model
   rarely saw the aligned ``<bos>用户：…\\n助手：`` pattern it must complete at
   inference time.
2. The loss was computed on user tokens too, so the model spent capacity
   modelling questions instead of sharpening answers.

This script instead builds *aligned samples* and stores a per-token loss-weight
code (see :mod:`retnet.chat`), then greedily packs samples into fixed-length
rows so no GPU time is wasted on padding.

It also rebalances the mixture: the previous SFT set was almost pure
"assistant task" data (UltraChat / ShareGPT), which is why casual prompts like
"你真好看" were answered with an essay.  Casual Chinese dialogue (LCCC, GLM,
豆瓣/小红书, everyday-conversations) now gets an explicit token budget.

Example:
    python scripts/build_sft_samples.py --out_dir data_cache/chat_pack_v3 --seq_len 512
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.chat import (W_PAD, W_PLAIN, encode_dialogues_batch,  # noqa: E402
                         pairs_from_roles)
from retnet.tokenizer import BPETokenizer  # noqa: E402

csv.field_size_limit(10 ** 9)

Pairs = List[Tuple[str, str]]


# ----------------------------------------------------------------- text utils
def clean(s, max_chars: int | None = 1200) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ")
    # LCCC stores Chinese with a space between every character.
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
    # ... and around full-width punctuation.  Restricted to CJK punctuation so
    # English sentences keep the space after their periods.
    s = re.sub(r"\s+(?=[，。！？；：、）》”’…])", "", s)
    s = re.sub(r"(?<=[，。！？；：、（《“‘…])\s+(?=[\u4e00-\u9fff])", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars and len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s


def role_of(raw) -> str | None:
    r = str(raw).lower()
    if r in {"user", "human", "用户", "客户", "prompter"}:
        return "user"
    if r in {"assistant", "gpt", "bot", "ai", "助手", "chatgpt"}:
        return "assistant"
    return None


# ------------------------------------------------------------- source readers
def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def obj_to_pairs(obj: dict, max_chars: int) -> Pairs:
    """Convert one HF SFT record into (user, assistant) pairs."""
    # Some repos use upper-case field names (Zhihu-KOL: INSTRUCTION/RESPONSE).
    if obj and all(k.isupper() for k in obj.keys()):
        obj = {k.lower(): v for k, v in obj.items()}
    for key in ("messages", "conversations", "conversation"):
        seq = obj.get(key)
        if isinstance(seq, list) and seq:
            msgs: List[Tuple[str, str]] = []
            for m in seq:
                if not isinstance(m, dict):
                    continue
                if "human" in m or "assistant" in m:      # sharegpt-cn style
                    if m.get("human"):
                        msgs.append(("user", clean(m["human"], max_chars)))
                    if m.get("assistant"):
                        msgs.append(("assistant", clean(m["assistant"], max_chars)))
                    continue
                role = role_of(m.get("role", m.get("from", "")))
                text = clean(m.get("content", m.get("value", "")), max_chars)
                if role and text:
                    msgs.append((role, text))
            pairs = pairs_from_roles(msgs)
            if pairs:
                return pairs

    if "instruction" in obj:
        ins = clean(obj.get("instruction"), max_chars)
        out = clean(obj.get("output", obj.get("response", "")), max_chars)
        if re.search(r"Human\s*[:：]", ins):               # Belle multi-turn
            msgs = []
            for m in re.finditer(r"(Human|Assistant)\s*[:：]\s*(.*?)(?=(?:Human|Assistant)\s*[:：]|$)",
                                 ins, flags=re.S):
                role = "user" if m.group(1) == "Human" else "assistant"
                text = clean(m.group(2), max_chars)
                if text:
                    msgs.append((role, text))
            if out:
                msgs.append(("assistant", out))
            pairs = pairs_from_roles(msgs)
            if pairs:
                return pairs
        inp = clean(obj.get("input"), max_chars)
        user = f"{ins} {inp}".strip() if inp else ins
        if user and out:
            return [(user, out)]

    for uk, ak in (("question", "answer"), ("prompt", "completion"),
                   ("query", "response"), ("instruction", "response"),
                   ("input", "target"), ("input", "output")):
        u, a = clean(obj.get(uk), max_chars), clean(obj.get(ak), max_chars)
        if u and a:
            return [(u, a)]
    return []


def iter_sft_file(path: Path, max_chars: int) -> Iterator[Pairs]:
    for obj in iter_jsonl(path):
        pairs = obj_to_pairs(obj, max_chars)
        if pairs:
            yield pairs


def iter_lccc(path: Path, stride: int, max_chars: int) -> Iterator[Pairs]:
    """LCCC-base: each line is a list of alternating utterances."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if stride > 1 and i % stride:
                continue
            try:
                utts = json.loads(line)
            except Exception:
                continue
            if not isinstance(utts, list) or len(utts) < 2:
                continue
            texts = [clean(u, max_chars) for u in utts]
            texts = [t for t in texts if t]
            if len(texts) < 2:
                continue
            pairs = [(texts[j], texts[j + 1]) for j in range(0, len(texts) - 1, 2)]
            if pairs:
                yield pairs


def iter_glm(path: Path, stride: int, max_chars: int) -> Iterator[Pairs]:
    """GLM open-dialogue CSV: 上下文 ... 答案 ... / target_text."""
    with open(path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if stride > 1 and i % stride:
                continue
            src = row.get("source_text", "") or ""
            tgt = clean(row.get("target_text", ""), max_chars)
            if not tgt:
                continue
            m = re.search(r"上下文[:：]\s*(.*?)\s*答案[:：]", src, flags=re.S)
            utts = []
            if m:
                utts = [clean(x, max_chars) for x in re.split(r"\[SEP\]|\n", m.group(1))]
                utts = [u for u in utts if u]
            if not utts:
                utts = [clean(src, max_chars)]
            if not utts[0]:
                continue
            texts = utts + [tgt]
            pairs = [(texts[j], texts[j + 1]) for j in range(0, len(texts) - 1, 2)]
            if pairs:
                yield pairs


# ------------------------------------------------------------------- packing
class Packer:
    """Pack variable-length samples into fixed-length rows.

    Keeps several rows open at once and puts each sample into the open row where
    it fits best (best-fit bin packing), which cuts padding from ~16% to ~1%.
    """

    def __init__(self, row_len: int, pad_id: int, chunk_rows: int = 8192, open_rows: int = 64):
        self.row_len = row_len
        self.pad_id = pad_id
        self.chunk_rows = chunk_rows
        self.open_rows = max(1, open_rows)
        self.ids_chunks: List[np.ndarray] = []
        self.code_chunks: List[np.ndarray] = []
        self.done_ids: List[np.ndarray] = []
        self.done_codes: List[np.ndarray] = []
        self.bins: List[Tuple[np.ndarray, np.ndarray, int]] = []
        self.tokens = 0
        self.padding = 0
        self.rows_done = 0

    def _new_bin(self):
        ids = np.full(self.row_len, self.pad_id, dtype=np.int32)
        codes = np.zeros(self.row_len, dtype=np.uint8)
        return [ids, codes, 0]

    def _retire(self, b):
        self.padding += self.row_len - b[2]
        self.done_ids.append(b[0])
        self.done_codes.append(b[1])
        self.rows_done += 1

    def add(self, ids: Sequence[int], codes: Sequence[int]):
        n = len(ids)
        if n > self.row_len:
            ids, codes, n = ids[:self.row_len], codes[:self.row_len], self.row_len
        # best fit: the open row with the least remaining space that still fits
        best = -1
        best_left = self.row_len + 1
        for i, b in enumerate(self.bins):
            left = self.row_len - b[2]
            if n <= left < best_left:
                best, best_left = i, left
        if best < 0:
            if len(self.bins) >= self.open_rows:
                # retire the fullest row to make space
                fullest = max(range(len(self.bins)), key=lambda i: self.bins[i][2])
                self._retire(self.bins.pop(fullest))
            self.bins.append(self._new_bin())
            best = len(self.bins) - 1
        b = self.bins[best]
        b[0][b[2]:b[2] + n] = ids
        b[1][b[2]:b[2] + n] = codes
        b[2] += n
        self.tokens += n
        if b[2] >= self.row_len:
            self._retire(self.bins.pop(best))

    def finish(self):
        for b in self.bins:
            if b[2] > 0:
                self._retire(b)
        self.bins = []
        if not self.done_ids:
            return (np.zeros((0, self.row_len), dtype=np.int32),
                    np.zeros((0, self.row_len), dtype=np.uint8))
        return np.stack(self.done_ids), np.stack(self.done_codes)


# ---------------------------------------------------------------------- main
def source_specs(args):
    sft = Path(args.sft_dir)
    corpus = Path(args.corpus_dir)
    extra2 = Path(args.extra_sft_dir) if args.extra_sft_dir else Path("data_cache/_none")
    # (name, kind, iterator factory, token budget)
    specs = [
        # --- casual Chinese chit-chat: fixes "闲聊被当成写作任务" ---
        ("lccc", "casual", lambda: iter_lccc(corpus / "lccc_base_train.jsonl",
                                             args.lccc_stride, 300), args.lccc_tokens),
        ("glm_dialogue", "casual", lambda: iter_glm(corpus / "train_all_df.csv",
                                                    1, 300), args.glm_tokens),
        ("coig_douban", "casual", lambda: iter_sft_file(sft / "coig_douban.jsonl", 600), 3_000_000),
        ("coig_xhs", "casual", lambda: iter_sft_file(sft / "coig_xhs.jsonl", 600), 2_000_000),
        ("smoltalk_everyday", "casual",
         lambda: iter_sft_file(sft / "smoltalk_everyday-conversations.jsonl", 600), 2_000_000),
        # --- instruction / assistant data ---
        ("ultrachat", "instruct", lambda: iter_sft_file(sft / "ultrachat.jsonl", 1200), args.ultrachat_tokens),
        ("sharegpt4", "instruct", lambda: iter_sft_file(sft / "sharegpt4.jsonl", 1200), args.sharegpt_tokens),
        ("belle", "instruct", lambda: iter_sft_file(sft / "belle.jsonl", 800), args.belle_tokens),
        ("smoltalk_chinese", "instruct",
         lambda: iter_sft_file(sft / "smoltalk_chinese.jsonl", 1000), args.smoltalk_zh_tokens),
        ("coig_zhihu", "instruct", lambda: iter_sft_file(sft / "coig_zhihu.jsonl", 1000), 5_000_000),
        ("coig_pc", "instruct", lambda: iter_sft_file(sft / "coig_coig_pc.jsonl", 1000), 3_000_000),
        ("ruozhiba", "instruct", lambda: iter_sft_file(sft / "ruozhiba.jsonl", 800), 2_000_000),
        # --- short-answer Chinese instructions (firefly) + colloquial long-form (zhihu) ---
        ("firefly", "instruct", lambda: iter_sft_file(extra2 / "firefly.jsonl", 800),
         args.firefly_tokens),
        ("zhihu_kol", "instruct", lambda: iter_sft_file(extra2 / "zhihu_kol.jsonl", 900),
         args.zhihu_tokens),
    ]
    named = {"firefly.jsonl", "zhihu_kol.jsonl"}
    if extra2.exists():
        for path in sorted(extra2.glob("*.jsonl")):
            if path.name in named:
                continue
            specs.append((f"extra_{path.stem}", "instruct",
                          lambda p=path: iter_sft_file(p, 1000), args.extra_tokens))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_cache/chat_pack_v3")
    ap.add_argument("--tokenizer", default="data_cache/chat_sft_mix/bpe_tokenizer.json")
    ap.add_argument("--sft_dir", default="data_cache/extra_sft")
    ap.add_argument("--extra_sft_dir", default="data_cache/extra_sft2")
    ap.add_argument("--corpus_dir", default="data_cache/chat_corpus")
    ap.add_argument("--pretrain_dir", default="data_cache/pretrain_merged")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_docs", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=200,
                    help="hold out one sample out of N for validation")
    ap.add_argument("--val_max_tokens", type=int, default=1_500_000)
    # token budgets
    ap.add_argument("--lccc_tokens", type=int, default=45_000_000)
    ap.add_argument("--lccc_stride", type=int, default=3)
    ap.add_argument("--glm_tokens", type=int, default=15_000_000)
    ap.add_argument("--ultrachat_tokens", type=int, default=25_000_000)
    ap.add_argument("--sharegpt_tokens", type=int, default=25_000_000)
    ap.add_argument("--belle_tokens", type=int, default=20_000_000)
    ap.add_argument("--smoltalk_zh_tokens", type=int, default=12_000_000)
    ap.add_argument("--firefly_tokens", type=int, default=15_000_000)
    ap.add_argument("--zhihu_tokens", type=int, default=6_000_000)
    ap.add_argument("--extra_tokens", type=int, default=15_000_000)
    ap.add_argument("--replay_tokens", type=int, default=40_000_000)
    ap.add_argument("--replay_spans", type=int, default=4000,
                    help="number of contiguous spans sampled from the pretrain corpus")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    tok = BPETokenizer.load(args.tokenizer)
    row_len = args.seq_len + 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = Packer(row_len, tok.pad_id)
    val = Packer(row_len, tok.pad_id)
    stats = {}
    val_tokens = 0

    for name, kind, factory, budget in source_specs(args):
        t0 = time.time()
        used = 0
        docs = 0
        try:
            it = factory()
        except FileNotFoundError:
            print(f"[skip] {name}: file not found", flush=True)
            continue
        batch: List[Pairs] = []
        stop = False
        while not stop:
            batch.clear()
            try:
                for _ in range(args.batch_docs):
                    batch.append(next(it))
            except StopIteration:
                stop = True
            except FileNotFoundError:
                print(f"[skip] {name}: file not found", flush=True)
                break
            if not batch:
                break
            for ids, codes in encode_dialogues_batch(tok, batch, max_len=args.seq_len):
                if len(ids) < 12:
                    continue
                docs += 1
                if docs % args.val_every == 0 and val_tokens < args.val_max_tokens:
                    val.add(ids, codes)
                    val_tokens += len(ids)
                else:
                    train.add(ids, codes)
                    used += len(ids)
            if used >= budget:
                break
        stats[name] = {"kind": kind, "tokens": used, "docs": docs,
                       "sec": round(time.time() - t0, 1)}
        print(f"[{kind}] {name}: tokens={used:,} docs={docs:,} "
              f"({stats[name]['sec']}s)", flush=True)

    # --- plain-text replay so the general knowledge does not wash out ---
    if args.replay_tokens > 0:
        t0 = time.time()
        src = np.load(str(Path(args.pretrain_dir) / "train.ids.npy"), mmap_mode="r")
        n = min(args.replay_tokens, src.shape[0])
        chunk = args.seq_len
        # Read a few thousand CONTIGUOUS spans instead of striding across the
        # whole 1.3B-token file: strided reads become random disk I/O and are
        # hundreds of times slower for the same token count.
        spans = max(1, min(args.replay_spans, n // chunk))
        span_tokens = max(chunk, (n // spans) // chunk * chunk)
        starts = np.linspace(0, max(0, src.shape[0] - span_tokens - 1), spans, dtype=np.int64)
        taken = 0
        for start in starts:
            if taken >= n:
                break
            span = np.asarray(src[start:start + span_tokens], dtype=np.int32)
            for off in range(0, span.shape[0] - chunk + 1, chunk):
                train.add(span[off:off + chunk].tolist(), [W_PLAIN] * chunk)
                taken += chunk
                if taken >= n:
                    break
        stats["pretrain_replay"] = {"kind": "plain", "tokens": int(taken), "docs": int(spans),
                                    "sec": round(time.time() - t0, 1)}
        print(f"[plain] pretrain_replay: tokens={taken:,} spans={spans} "
              f"({stats['pretrain_replay']['sec']}s)", flush=True)

    train_ids, train_codes = train.finish()
    val_ids, val_codes = val.finish()

    np.save(str(out / "train.ids.npy"), train_ids)
    np.save(str(out / "train.wcode.npy"), train_codes)
    np.save(str(out / "val.ids.npy"), val_ids)
    np.save(str(out / "val.wcode.npy"), val_codes)
    shutil.copyfile(args.tokenizer, out / "bpe_tokenizer.json")

    reply_tokens = int((train_codes == 1).sum())
    prompt_tokens = int((train_codes == 2).sum())
    plain_tokens = int((train_codes == 3).sum())
    pad_tokens = int((train_codes == W_PAD).sum())
    meta = {
        "seq_len": args.seq_len,
        "row_len": row_len,
        "train_rows": int(train_ids.shape[0]),
        "val_rows": int(val_ids.shape[0]),
        "train_tokens": int(train_ids.size),
        "reply_tokens": reply_tokens,
        "prompt_tokens": prompt_tokens,
        "plain_tokens": plain_tokens,
        "pad_tokens": pad_tokens,
        "sources": stats,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
