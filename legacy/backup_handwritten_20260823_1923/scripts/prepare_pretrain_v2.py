"""Stream large HF pretraining corpora into compact int16 token shards.

Improvements over ``prepare_pretrain_shards.py``:

* **int16 storage** (2 bytes/token instead of 8).  The vocabulary is 20k, so
  every id fits in int16 and PyTorch can consume it directly — a 5B-token
  corpus costs 10 GB instead of 40 GB.
* **--skip_docs**: streaming datasets always start from the beginning, so a
  second pass over the same repo would re-tokenize the same documents.  Skipping
  ahead lets later runs collect *new* text.
* **CPU budget**: the Rust tokenizer is capped (default 6 threads) so a training
  run can keep the GPU fed while new data is being prepared.
* Atomic shard writes plus a manifest, so an interrupted run never leaves a
  half-written shard behind.

Examples:
    python scripts/prepare_pretrain_v2.py --source chinese_fineweb_edu \\
        --max_tokens 1500000000 --skip_docs 500000
    python scripts/prepare_pretrain_v2.py --source fineweb_edu \\
        --fineweb_config sample-100BT --max_tokens 1000000000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets"))
# Cap tokenizer threads BEFORE the tokenizers library is imported.
os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("RETNET_TOK_THREADS", "6"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from retnet.tokenizer import BPETokenizer  # noqa: E402

CJK_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")

SOURCES = {
    # name: (repo, config, split, text_keys)
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", None, "train", ("text",)),
    "chinese_fineweb_edu": ("opencsg/chinese-fineweb-edu", None, "train", ("text", "content")),
    "chinese_fineweb_edu_v2": ("opencsg/Chinese-FineWeb-Edu-V2", None, "train", ("text", "content")),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", ("text",)),
    "fineweb_edu_dedup": ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "train", ("text",)),
    "python_edu": ("HuggingFaceTB/smollm-corpus", "python-edu", "train", ("text", "content")),
    "open_web_math": ("open-web-math/open-web-math", None, "train", ("text",)),
    "wikipedia_zh": ("wikimedia/wikipedia", "20231101.zh", "train", ("text",)),
    "skypile": ("Skywork/SkyPile-150B", None, "train", ("text",)),
    "wikitext103": ("wikitext", "wikitext-103-raw-v1", "train", ("text",)),
}


def clean(text: str, min_chars: int, max_chars: int) -> str:
    text = str(text or "")
    text = CJK_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < min_chars:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def iter_source(name: str, fineweb_config: str | None, min_chars: int, max_chars: int):
    from datasets import load_dataset
    repo, config, split, keys = SOURCES[name]
    if name == "fineweb_edu" and fineweb_config:
        config = fineweb_config
    ds = load_dataset(repo, config, split=split, streaming=True)
    for ex in ds:
        raw = None
        for key in keys:
            value = ex.get(key)
            if isinstance(value, str) and value:
                raw = value
                break
        text = clean(raw, min_chars, max_chars)
        if text:
            yield text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES))
    ap.add_argument("--fineweb_config", default="sample-100BT")
    ap.add_argument("--tokenizer", default="data_cache/chat_pack_v3/bpe_tokenizer.json")
    ap.add_argument("--out_dir", default="data_cache/pretrain_v2")
    ap.add_argument("--shard_tokens", type=int, default=50_000_000)
    ap.add_argument("--max_tokens", type=int, default=1_000_000_000)
    ap.add_argument("--batch_texts", type=int, default=1024)
    ap.add_argument("--skip_docs", type=int, default=0,
                    help="skip this many documents before collecting (get fresh text)")
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=8000)
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--val_every", type=int, default=500)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.load(args.tokenizer)
    if tok.vocab_size > 32767:
        raise SystemExit(f"vocab {tok.vocab_size} does not fit in int16")
    bos, eos = tok.bos_id, tok.eos_id

    shard_idx = 0
    while (out / f"{args.source}_{shard_idx:05d}.npy").exists():
        shard_idx += 1
    print(f"[{args.source}] starting at shard {shard_idx}, "
          f"threads={os.environ.get('RAYON_NUM_THREADS')}", flush=True)

    def encode(texts):
        flat = []
        for enc in tok.encode_batch(texts):
            flat.append(bos)
            flat.extend(enc)
            flat.append(eos)
        return np.asarray(flat, dtype=np.int16)

    def save(path: Path, chunks):
        arr = np.concatenate(chunks)
        tmp = path.with_suffix(".tmp.npy")
        np.save(str(tmp), arr)
        tmp.replace(path)
        return arr.shape[0]

    val_path = out / f"{args.source}_val.npy"
    val_chunks, val_total = [], 0
    if val_path.exists():
        val_total = args.val_tokens        # already have one; do not rebuild

    chunks, shard_tokens, grand_total = [], 0, 0
    batch = []
    docs = 0
    skipped = 0
    t0 = time.time()
    t_shard = time.time()

    for text in iter_source(args.source, args.fineweb_config, args.min_chars, args.max_chars):
        if skipped < args.skip_docs:
            skipped += 1
            if skipped % 200_000 == 0:
                print(f"  skipping {skipped:,}/{args.skip_docs:,} docs "
                      f"({time.time() - t0:.0f}s)", flush=True)
            continue
        docs += 1
        if val_total < args.val_tokens and docs % args.val_every == 0:
            arr = encode([text])
            val_chunks.append(arr)
            val_total += arr.shape[0]
            continue
        batch.append(text)
        if len(batch) < args.batch_texts:
            continue
        arr = encode(batch)
        batch = []
        chunks.append(arr)
        shard_tokens += arr.shape[0]
        if shard_tokens >= args.shard_tokens:
            path = out / f"{args.source}_{shard_idx:05d}.npy"
            n = save(path, chunks)
            grand_total += n
            print(f"saved {path.name} tokens={n:,} docs={docs:,} "
                  f"total={grand_total:,} sec={time.time() - t_shard:.0f}", flush=True)
            chunks, shard_tokens = [], 0
            shard_idx += 1
            t_shard = time.time()
            if grand_total >= args.max_tokens:
                break

    if batch:
        arr = encode(batch)
        chunks.append(arr)
        shard_tokens += arr.shape[0]
    if chunks and grand_total < args.max_tokens:
        path = out / f"{args.source}_{shard_idx:05d}.npy"
        n = save(path, chunks)
        grand_total += n
        print(f"saved final {path.name} tokens={n:,}", flush=True)
    if val_chunks:
        n = save(val_path, val_chunks)
        print(f"saved val {val_path.name} tokens={n:,}", flush=True)

    manifest = out / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    entry = data.setdefault(args.source, {"runs": []})
    entry["runs"].append({
        "tokens": int(grand_total),
        "docs": int(docs),
        "skip_docs": int(args.skip_docs),
        "config": args.fineweb_config if args.source == "fineweb_edu" else None,
        "sec": round(time.time() - t0, 1),
    })
    entry["next_skip_docs"] = int(args.skip_docs + docs)
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE {args.source}: tokens={grand_total:,} docs={docs:,} "
          f"next --skip_docs {entry['next_skip_docs']:,}", flush=True)


if __name__ == "__main__":
    main()
