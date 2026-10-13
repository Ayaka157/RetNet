"""Build chat_pack_v10 = v9 packed rows + v10 new SFT sources (R1/百科/锐智吧).

Reuses chat_pack_v9's already-tokenized train/val rows as the base (avoids
re-tokenizing 280M tokens), then appends tokenized dialogue samples from:
  - r1_110k_sft.jsonl     (math/reasoning coT)  budget 250M
  - i_wonder_why.jsonl    (factual QA)          budget 130M
  - better_ruozhiba.jsonl (multi-turn/humor)    budget 25M
  - v9 rp replay          (RP preservation)     budget 200M

Output format matches chat_pack_v9: train.ids.npy (rows×seq+1 int16),
train.wcode.npy (0=pad,1=reply,2=prompt,3=plain), val.*, meta.json.

    python scripts/build_chat_mix_v10.py --base data_cache/chat_pack_v9 \
        --out_dir data_cache/chat_pack_v10 --seq_len 512
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.chat import W_PLAIN, encode_dialogues_batch  # noqa: E402
from retnet.packing import Packer, clean, iter_jsonl, messages_to_pairs  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402


def r1_reader(path: Path, strip_think: bool = False, max_chars: int = 1200):
    """R1-SFT: instruction/input/output -> (pairs, None).

    ``strip_think`` removes the ``<think>...</think>`` span *before* the length
    cap.  Without it the 1200-char cut lands inside the CoT (median R1 output is
    1912 chars), so the reply never contains the closing tag or the answer —
    the model learns to open a think block and never finish it.  Observed as a
    27.7% think-leak on chat_pack_v10 (see docs/13).

    ``max_chars`` is the second truncation point.  It must be raised together
    with ``--seq_len`` or a wide-window pack still carries chopped CoT.
    """
    for obj in iter_jsonl(path):
        ins = clean(obj.get("instruction") or obj.get("prompt") or "", 600)
        u = clean(obj.get("input") or "", 400) if obj.get("input") else ""
        raw = obj.get("output") or obj.get("response") or ""
        if strip_think:
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        a = clean(raw, max_chars)
        user = (ins + " " + u).strip() if u else ins
        if user and a:
            yield [(user[:600], a[:max_chars])], None


def wiki_reader(path: Path):
    """I_Wonder_Why: prompt(question) -> response(answer)."""
    for obj in iter_jsonl(path):
        q = clean(obj.get("prompt") or obj.get("instruction") or obj.get("question") or "", 400)
        a = clean(obj.get("response") or obj.get("output") or obj.get("answer") or "", 800)
        if q and a:
            yield [(q, a)], None


def ruozhiba_reader(path: Path):
    """Better-Ruozhiba: question/answer."""
    for obj in iter_jsonl(path):
        q = clean(obj.get("instruction") or obj.get("question") or "", 300)
        a = clean(obj.get("output") or obj.get("answer") or "", 600)
        if q and a:
            yield [(q, a)], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data_cache/chat_pack_v9")
    ap.add_argument("--out_dir", default="data_cache/chat_pack_v10")
    ap.add_argument("--tokenizer", default="data_cache/chat_pack_v9/bpe_tokenizer.json")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--ag_dir", default="data_cache/a_grade")
    ap.add_argument("--batch_docs", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=300)
    ap.add_argument("--val_max_tokens", type=int, default=2_000_000)
    # v10 budgets (tokens)
    ap.add_argument("--r1_tokens", type=int, default=250_000_000)
    ap.add_argument("--wiki_tokens", type=int, default=130_000_000)
    ap.add_argument("--ruozhiba_tokens", type=int, default=25_000_000)
    ap.add_argument("--strip_think", action="store_true",
                    help="remove <think>...</think> from R1 output before the "
                         "length cap (v12 fix; see docs/13)")
    ap.add_argument("--r1_max_chars", type=int, default=1200,
                    help="char cap on the R1 reply; raise together with --seq_len "
                         "for wide-window packs (v13 uses 4000 @ seq 2048)")
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
    stats, val_tokens = {}, 0
    ag = Path(args.ag_dir)

    # kind, name, reader, budget
    specs = [
        ("r1", "r1_110k_sft",
         r1_reader(ag / "r1_110k_sft.jsonl", args.strip_think, args.r1_max_chars),
         args.r1_tokens),
        ("wiki", "i_wonder_why", wiki_reader(ag / "i_wonder_why.jsonl"), args.wiki_tokens),
        ("multi", "better_ruozhiba", ruozhiba_reader(ag / "better_ruozhiba.jsonl"), args.ruozhiba_tokens),
    ]

    # NEW samples: R1/百科/锐智吧 tokenize + best-fit pack into fresh rows
    def add_sample(ids, codes):
        if len(ids) < 16:
            return
        docs_holder[0] += 1
        docs = docs_holder[0]
        if docs % args.val_every == 0 and val_tokens_holder[0] < args.val_max_tokens:
            val.add(ids, codes)
            val_tokens_holder[0] += len(ids)
        else:
            train.add(ids, codes)
            used_holder[0] += len(ids)

    docs_holder = [0]
    val_tokens_holder = [0]
    used_holder = [0]
    for kind, name, it, budget in specs:
        t0, used, n = time.time(), 0, 0
        docs_holder[0], val_tokens_holder[0] = 0, 0
        used_holder[0] = 0
        batch, done = [], False
        while not done and used < budget:
            batch.clear()
            try:
                for _ in range(args.batch_docs):
                    batch.append(next(it))
            except StopIteration:
                done = True
            except FileNotFoundError:
                print(f"[skip] {name}: missing file", flush=True)
                break
            if not batch:
                break
            dialogues = [b[0] for b in batch]
            systems = [b[1] for b in batch]
            for ids, codes in encode_dialogues_batch(tok, dialogues, max_len=args.seq_len,
                                                     systems=systems):
                add_sample(ids, codes)
                if len(ids) < 16:
                    continue
                n += 1
            used = used_holder[0]
        stats[name] = {"kind": kind, "tokens": used, "docs": n,
                       "sec": round(time.time() - t0, 1)}
        print(f"[{kind:6}] {name:<18} tokens={used:>10,} docs={n:>8,} "
              f"({stats[name]['sec']}s)", flush=True)

    # base v9 rows: keep as-is (already packed dialogue rows with reply/prompt codes)
    base = Path(args.base)
    base_train_ids = np.load(base / "train.ids.npy", mmap_mode="r")
    base_train_codes = np.load(base / "train.wcode.npy", mmap_mode="r")
    base_val_ids = np.load(base / "val.ids.npy", mmap_mode="r")
    base_val_codes = np.load(base / "val.wcode.npy", mmap_mode="r")
    print(f"base v9 rows: train={base_train_ids.shape[0]} val={base_val_ids.shape[0]}", flush=True)
    stats["base_v9"] = {"kind": "base", "tokens": int(base_train_ids.size),
                        "docs": int(base_train_ids.shape[0]), "sec": 0}

    # assemble: [v9 rows-as-is] + [new packed rows]
    train_ids = np.asarray(base_train_ids[:])
    train_codes = np.asarray(base_train_codes[:])
    new_ids, new_codes = train.finish()
    if len(new_ids):
        train_ids = np.concatenate([train_ids, new_ids], axis=0)
        train_codes = np.concatenate([train_codes, new_codes], axis=0)
    val_ids = np.asarray(base_val_ids[:])
    val_codes = np.asarray(base_val_codes[:])
    new_val_ids, new_val_codes = val.finish()
    if len(new_val_ids):
        val_ids = np.concatenate([val_ids, new_val_ids], axis=0)
        val_codes = np.concatenate([val_codes, new_val_codes], axis=0)

    np.save(str(out / "train.ids.npy"), train_ids)
    np.save(str(out / "train.wcode.npy"), train_codes)
    np.save(str(out / "val.ids.npy"), val_ids)
    np.save(str(out / "val.wcode.npy"), val_codes)
    shutil.copyfile(args.tokenizer, out / "bpe_tokenizer.json")

    meta = {"seq_len": args.seq_len, "train_rows": int(train_ids.shape[0]),
            "strip_think": bool(args.strip_think),
            "train_tokens": int(train_ids.size),
            "reply_tokens": int((train_codes == 1).sum()),
            "prompt_tokens": int((train_codes == 2).sum()),
            "plain_tokens": int((train_codes == 3).sum()),
            "sources": stats}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    import time
    main()
