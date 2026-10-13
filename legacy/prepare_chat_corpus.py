"""Prepare a mixed general + dialogue corpus for the RetNet chat model.

All large files are kept on drive E under ``data_cache/``.  The script:

1. samples public datasets:
   * LCCC-base        - large cleaned Chinese conversation corpus
   * GLM-Open-Dialogue- generated Chinese multi-turn dialogue
   * ShareGPT-Chinese - distilled human/assistant conversations
   * filtered Chinese Wikipedia - general Chinese text
   * Wikitext-103      - general English text
2. trains one mixed byte-level BPE tokenizer,
3. encodes a token-mix of dialogue-heavy data and saves ``train.ids.npy`` /
   ``val.ids.npy`` for the training script.

Run:
    python prepare_chat_corpus.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets"))

from retnet.tokenizer import BPETokenizer  # noqa: E402

USER = "用户"
ASSISTANT = "助手"


# ----------------------------------------------------------------------
# text utilities
# ----------------------------------------------------------------------
def clean_text(s: str, max_chars: Optional[int] = None) -> str:
    """Normalize LCCC/GLM style text and remove whitespace between CJK chars."""
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ")
    # LCCC stores many Chinese sentences with a space between every CJK char.
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
    s = re.sub(r"\s+(?=[，。！？；：、,.!?;:])", "", s)
    s = re.sub(r"(?<=[，。！？；：、,.!?;:])\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars and len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s


def format_turns(utterances: List[str], start_role: str = USER) -> str:
    parts = []
    roles = (USER, ASSISTANT) if start_role == USER else (ASSISTANT, USER)
    for i, u in enumerate(utterances):
        u = clean_text(u, 600)
        if not u:
            continue
        role = roles[i % 2]
        parts.append(f"{role}：{u}")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# dataset readers (all yield formatted training strings)
# ----------------------------------------------------------------------
def iter_lccc(path: Path, stride: int = 1, limit: Optional[int] = None) -> Iterator[str]:
    n = 0
    kept = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if n % stride == 0:
                try:
                    utts = json.loads(line)
                    if isinstance(utts, list) and len(utts) >= 2:
                        text = format_turns(utts)
                        if text:
                            yield text
                            kept += 1
                except Exception:
                    pass
            n += 1
            if limit is not None and kept >= limit:
                break


def iter_glm(path: Path, stride: int = 1, limit: Optional[int] = None) -> Iterator[str]:
    import csv
    kept = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i % stride:
                continue
            src = row.get("source_text", "") or ""
            tgt = row.get("target_text", "") or ""
            m = re.search(r"上下文[:：]\s*(.*?)\s*答案[:：]", src, flags=re.S)
            if m:
                utts = [x.strip() for x in re.split(r"\[SEP\]|\n", m.group(1)) if x.strip()]
            else:
                utts = []
            if len(utts) >= 1:
                text = format_turns(utts + [tgt])
            else:
                text = f"{USER}：{clean_text(src, 400)}\n{ASSISTANT}：{clean_text(tgt, 400)}"
            if text:
                yield text
                kept += 1
            if limit is not None and kept >= limit:
                break


def iter_sharegpt(path: Path, stride: int = 1, limit: Optional[int] = None) -> Iterator[str]:
    kept = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            conv = obj.get("conversation")
            if not isinstance(conv, list):
                continue
            parts = []
            for turn in conv:
                if not isinstance(turn, dict):
                    continue
                if "human" in turn:
                    parts.append(f"{USER}：{clean_text(turn.get('human'), 800)}")
                elif "assistant" in turn:
                    parts.append(f"{ASSISTANT}：{clean_text(turn.get('assistant'), 800)}")
            text = "\n".join(p for p in parts if p)
            if text:
                yield text
                kept += 1
            if limit is not None and kept >= limit:
                break


def iter_wiki_json(path: Path, stride: int = 1, limit: Optional[int] = None) -> Iterator[str]:
    data = json.load(open(path, "r", encoding="utf-8"))
    for i, item in enumerate(data):
        if i % stride:
            continue
        text = clean_text(item.get("completion", ""), 2500)
        if text:
            yield text
        if limit is not None and i >= (limit * stride):
            break


def iter_wikitext103(stride: int = 1, limit: Optional[int] = None) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    kept = 0
    for i, row in enumerate(ds):
        if i % stride:
            continue
        text = clean_text(row.get("text", ""), 2000)
        if text:
            yield text
            kept += 1
        if limit is not None and kept >= limit:
            break


def batched(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_texts(tok: BPETokenizer, texts: Iterator[str], token_budget: int,
                 batch_size: int = 1000) -> Tuple[np.ndarray, int, int]:
    """Encode formatted documents until token_budget is reached."""
    chunks = []
    total = 0
    docs = 0
    bos, eos = tok.bos_id, tok.eos_id
    for batch in batched(texts, batch_size):
        flat = []
        for enc in tok.encode_batch(batch):
            flat.append(bos)
            flat.extend(enc)
            flat.append(eos)
        arr = np.asarray(flat, dtype=np.int64)
        chunks.append(arr)
        total += len(arr)
        docs += len(batch)
        if total >= token_budget:
            break
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=np.int64), total, docs


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=20000)
    ap.add_argument("--out_dir", type=str, default="data_cache/chat_mix_v20k")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--lccc_train_docs", type=int, default=1100000)
    ap.add_argument("--glm_train_docs", type=int, default=300000)
    ap.add_argument("--sharegpt_train_docs", type=int, default=25000)
    ap.add_argument("--wiki_train_docs", type=int, default=30000)
    ap.add_argument("--wikitext_train_docs", type=int, default=400000)
    ap.add_argument("--lccc_budget", type=int, default=30_000_000)
    ap.add_argument("--glm_budget", type=int, default=15_000_000)
    ap.add_argument("--sharegpt_budget", type=int, default=10_000_000)
    ap.add_argument("--wiki_budget", type=int, default=10_000_000)
    ap.add_argument("--wikitext_budget", type=int, default=10_000_000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = out / "bpe_tokenizer.json"
    train_path = out / "train.ids.npy"
    val_path = out / "val.ids.npy"

    files = {
        "lccc": Path("data_cache/chat_corpus/lccc_base_train.jsonl"),
        "glm": Path("data_cache/chat_corpus/train_all_df.csv"),
        "sharegpt": Path("data_cache/chat_corpus/sharegpt_jsonl/common_zh_70k.jsonl"),
        "wiki": Path("data_cache/chat_corpus/wikipedia-cn-20230720-filtered.json"),
    }

    # --------------------------------------------------------------
    # 1) sample texts for BPE training
    # --------------------------------------------------------------
    if not tok_path.exists():
        n_lccc = count_lines(files["lccc"])
        n_glm = count_lines(files["glm"])
        n_sharegpt = count_lines(files["sharegpt"])
        n_wiki = 254547  # known from json; reloaded lazily
        n_wikitext = 1801350  # number of rows in wikitext-103 train

        train_texts: List[str] = []
        train_texts.extend(iter_lccc(files["lccc"], stride=max(1, n_lccc // 250000), limit=250000))
        train_texts.extend(iter_glm(files["glm"], stride=max(1, n_glm // 250000), limit=250000))
        train_texts.extend(iter_sharegpt(files["sharegpt"], stride=max(1, n_sharegpt // 20000), limit=20000))
        train_texts.extend(iter_wiki_json(files["wiki"], stride=max(1, n_wiki // 30000), limit=30000))
        train_texts.extend(iter_wikitext103(stride=max(1, n_wikitext // 200000), limit=200000))
        print(f"Training BPE tokenizer on {len(train_texts)} texts (vocab={args.vocab_size})", flush=True)
        tok = BPETokenizer.train(train_texts, vocab_size=args.vocab_size, path=tok_path)
    else:
        tok = BPETokenizer.load(tok_path)
    print(f"Tokenizer ready, vocab_size={tok.vocab_size}", flush=True)

    # --------------------------------------------------------------
    # 2) encode training mix
    # --------------------------------------------------------------
    if not train_path.exists():
        n_lccc = count_lines(files["lccc"])
        n_glm = count_lines(files["glm"])
        n_sharegpt = count_lines(files["sharegpt"])
        n_wiki = 254547
        n_wikitext = 1801350

        sources = [
            ("lccc", iter_lccc(files["lccc"], stride=max(1, n_lccc // args.lccc_train_docs)),
             args.lccc_budget),
            ("glm", iter_glm(files["glm"], stride=max(1, n_glm // args.glm_train_docs)),
             args.glm_budget),
            ("sharegpt", iter_sharegpt(files["sharegpt"], stride=max(1, n_sharegpt // args.sharegpt_train_docs)),
             args.sharegpt_budget),
            ("wiki", iter_wiki_json(files["wiki"], stride=max(1, n_wiki // args.wiki_train_docs)),
             args.wiki_budget),
            ("wikitext", iter_wikitext103(stride=max(1, n_wikitext // args.wikitext_train_docs)),
             args.wikitext_budget),
        ]
        train_chunks = []
        total = 0
        stats = {}
        for name, gen, budget in sources:
            t0 = time.time()
            arr, n_tok, n_doc = encode_texts(tok, gen, budget)
            train_chunks.append(arr)
            total += n_tok
            stats[name] = {"tokens": n_tok, "docs": n_doc, "sec": round(time.time() - t0, 1)}
            print(f"[train] {name}: tokens={n_tok:,} docs={n_doc:,} sec={stats[name]['sec']}", flush=True)
        arr = np.concatenate(train_chunks)
        np.save(str(train_path), arr)
        print(f"Saved train tokens: {total:,} -> {train_path}", flush=True)
    else:
        print("Train cache exists:", train_path, flush=True)

    # --------------------------------------------------------------
    # 3) encode a small mixed validation set
    # --------------------------------------------------------------
    if not val_path.exists():
        val_texts: List[str] = []
        val_texts.extend(iter_lccc(files["lccc"], stride=3000, limit=2500))
        val_texts.extend(iter_glm(files["glm"], stride=3000, limit=800))
        val_texts.extend(iter_sharegpt(files["sharegpt"], stride=300, limit=300))
        val_texts.extend(iter_wiki_json(files["wiki"], stride=500, limit=600))
        val_texts.extend(iter_wikitext103(stride=5000, limit=600))
        val_arr, val_tok, val_doc = encode_texts(tok, iter(val_texts), 2_000_000)
        np.save(str(val_path), val_arr)
        print(f"Saved val tokens: {val_tok:,} docs={val_doc} -> {val_path}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
