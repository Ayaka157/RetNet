"""Stream large HF pretraining datasets and write token shards to drive E.

The script is intentionally resumable at shard level and never downloads whole
TB-scale datasets.  Run it multiple times with different --source /
--max_tokens budgets; existing complete shards are left untouched.

Examples:
    # 500M English fineweb-edu tokens, 20M-token shards
    python scripts/prepare_pretrain_shards.py --source fineweb_edu \\
        --fineweb_config sample-10BT --max_tokens 500000000

    # 500M Chinese fineweb-edu tokens
    python scripts/prepare_pretrain_shards.py --source chinese_fineweb_edu \\
        --max_tokens 500000000

Then merge with scripts/merge_shards.py and train using train.py.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets"))
import sys
sys.path.insert(0, str(ROOT))

from retnet.tokenizer import BPETokenizer  # noqa: E402

CJK_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")


def clean(text: str, min_chars: int = 80, max_chars: int = 6000) -> str:
    text = str(text or "")
    text = CJK_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < min_chars or len(text) > max_chars:
        return ""
    return text


def iter_source(source: str, fineweb_config: str):
    from datasets import load_dataset
    if source == "fineweb_edu":
        repo, config = "HuggingFaceFW/fineweb-edu", fineweb_config
    elif source == "chinese_fineweb_edu":
        repo, config = "opencsg/chinese-fineweb-edu", None
    elif source == "smollm_corpus":
        repo, config = "HuggingFaceTB/smollm-corpus", "cosmopedia-v2"
    elif source == "wikitext103":
        repo, config = "wikitext", "wikitext-103-raw-v1"
    else:
        raise ValueError(source)
    ds = load_dataset(repo, config, split="train", streaming=True)
    for ex in ds:
        text = None
        for key in ("text", "content", "completion"):
            if isinstance(ex.get(key), str):
                text = ex[key]
                break
        text = clean(text)
        if text:
            yield text


def encode_batch(tok, texts):
    flat = []
    bos, eos = tok.bos_id, tok.eos_id
    for enc in tok.encode_batch(texts):
        flat.append(bos)
        flat.extend(enc)
        flat.append(eos)
    return np.asarray(flat, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["fineweb_edu", "chinese_fineweb_edu", "smollm_corpus", "wikitext103"])
    ap.add_argument("--fineweb_config", default="sample-10BT")
    ap.add_argument("--tokenizer", default="data_cache/chat_mix_v20k/bpe_tokenizer.json")
    ap.add_argument("--out_dir", default="data_cache/pretrain_shards")
    ap.add_argument("--shard_tokens", type=int, default=20_000_000)
    ap.add_argument("--max_tokens", type=int, default=1_000_000_000)
    ap.add_argument("--batch_texts", type=int, default=512)
    ap.add_argument("--val_every", type=int, default=200,
                    help="save one validation doc every N docs")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val_path = out / f"{args.source}_val.ids.npy"
    val_chunks = []
    val_budget = 2_000_000

    tok = BPETokenizer.load(args.tokenizer)
    total = 0
    shard_idx = 0
    batch = []
    val_docs = 0
    # Find first unused shard index
    while (out / f"{args.source}_train_{shard_idx:05d}.npy").exists():
        shard_idx += 1

    shard_path = out / f"{args.source}_train_{shard_idx:05d}.npy"
    chunks = []
    t0 = time.time()
    doc_no = 0
    for text in iter_source(args.source, args.fineweb_config):
        if doc_no % args.val_every == 0 and sum(len(x) for x in val_chunks) < val_budget:
            arr = encode_batch(tok, [text])
            val_chunks.append(arr)
            val_docs += 1
        else:
            batch.append(text)
            if len(batch) >= args.batch_texts:
                arr = encode_batch(tok, batch)
                chunks.append(arr)
                total += len(arr)
                batch = []
        doc_no += 1
        if total >= args.shard_tokens:
            np.save(str(shard_path), np.concatenate(chunks))
            print(f"saved {shard_path} tokens={total:,} docs={doc_no:,} sec={time.time()-t0:.0f}", flush=True)
            chunks = []
            total = 0
            shard_idx += 1
            shard_path = out / f"{args.source}_train_{shard_idx:05d}.npy"
            if shard_idx * args.shard_tokens >= args.max_tokens:
                break
        if shard_idx * args.shard_tokens >= args.max_tokens:
            break
    if chunks:
        arr = np.concatenate(chunks)
        total += len(arr)
        np.save(str(shard_path), arr)
        print(f"saved final {shard_path} tokens={total:,}", flush=True)
    if val_chunks:
        np.save(str(val_path), np.concatenate(val_chunks))
        print(f"saved val {val_path} docs={val_docs} tokens={sum(len(x) for x in val_chunks):,}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
