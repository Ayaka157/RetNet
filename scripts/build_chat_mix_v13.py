"""Build chat_pack_v13 = v9 token stream re-sliced to 2048 + R1 re-encoded with FULL CoT.

Why not rebuild everything from raw: `scripts/build_chat_mix_v9.py --seq_len 2048`
grows past 13 GB RSS and dies (observed 2026-10-09).  The v9 sources are long
plain-text corpora whose readers are the memory hog.  We do not need to re-encode
them: their samples are short, so the 512-window truncation barely touched them.
What WAS destroyed by the old pipeline is the R1 CoT — and that is the one source
we re-encode here, at full length, with the 1200-char reader cap lifted.

    python scripts/build_chat_mix_v13.py --base data_cache/chat_pack_v9 ^
        --out_dir data_cache/chat_pack_v13 --seq_len 2048 --r1_max_chars 4000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from retnet.chat import encode_dialogues_batch  # noqa: E402
from retnet.packing import Packer  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402
from build_chat_mix_v10 import r1_reader, wiki_reader, ruozhiba_reader  # noqa: E402


def reslice(base: Path, split: str, row_len: int, chunk=50000):
    """Flatten packed rows (drop pad) and re-cut into row_len-wide rows."""
    idx = np.load(base / f"{split}.ids.npy", mmap_mode="r")
    cd = np.load(base / f"{split}.wcode.npy", mmap_mode="r")
    N = idx.shape[0]
    parts_i, parts_c = [], []
    for s in range(0, N, chunk):
        i = np.asarray(idx[s:s + chunk]).reshape(-1)
        c = np.asarray(cd[s:s + chunk]).reshape(-1)
        m = c != 0
        parts_i.append(i[m])
        parts_c.append(c[m])
    ids = np.concatenate(parts_i)
    codes = np.concatenate(parts_c)
    del parts_i, parts_c
    n = len(ids) // row_len
    ids = ids[: n * row_len].reshape(n, row_len)
    codes = codes[: n * row_len].reshape(n, row_len)
    print(f"  {split}: {N:,} old rows -> {len(ids):,} tokens -> {n:,} rows x {row_len}",
          flush=True)
    return ids.astype(np.int16), codes.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data_cache/chat_pack_v9")
    ap.add_argument("--out_dir", default="data_cache/chat_pack_v13")
    ap.add_argument("--tokenizer", default="data_cache/chat_pack_v9/bpe_tokenizer.json")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--ag_dir", default="data_cache/a_grade")
    ap.add_argument("--batch_docs", type=int, default=2000)
    ap.add_argument("--r1_tokens", type=int, default=250_000_000)
    ap.add_argument("--wiki_tokens", type=int, default=130_000_000)
    ap.add_argument("--ruozhiba_tokens", type=int, default=25_000_000)
    ap.add_argument("--r1_max_chars", type=int, default=4000,
                    help="R1 reply char cap.  v10/v12 used 1200, which cut the CoT "
                         "in half before packing; 4000 lets seq 2048 hold it.")
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
    ag = Path(args.ag_dir)

    print("[1/3] 重切 v9 基座", flush=True)
    b_tr_i, b_tr_c = reslice(Path(args.base), "train", row_len)
    b_va_i, b_va_c = reslice(Path(args.base), "val", row_len)

    print("\n[2/3] 用完整 CoT 重新编码 R1/wiki/锐智吧", flush=True)
    train = Packer(row_len, tok.pad_id)
    val = Packer(row_len, tok.pad_id)
    specs = [
        ("r1", "r1_110k_sft",
         r1_reader(ag / "r1_110k_sft.jsonl", False, args.r1_max_chars), args.r1_tokens),
        ("wiki", "i_wonder_why", wiki_reader(ag / "i_wonder_why.jsonl"), args.wiki_tokens),
        ("multi", "better_ruozhiba", ruozhiba_reader(ag / "better_ruozhiba.jsonl"),
         args.ruozhiba_tokens),
    ]
    stats, docs, val_tok = {}, 0, 0
    for kind, name, it, budget in specs:
        t0, used, n = time.time(), 0, 0
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
            for ids, codes in encode_dialogues_batch(
                    tok, [b[0] for b in batch], max_len=args.seq_len,
                    systems=[b[1] for b in batch]):
                if len(ids) < 16:
                    continue
                docs += 1
                n += 1
                if docs % 300 == 0 and val_tok < 2_000_000:
                    val.add(ids, codes)
                    val_tok += len(ids)
                else:
                    train.add(ids, codes)
                    used += len(ids)
        stats[name] = {"kind": kind, "tokens": used, "docs": n,
                       "sec": round(time.time() - t0, 1)}
        print(f"[{kind:6}] {name:<18} tokens={used:>10,} docs={n:>8,} "
              f"({stats[name]['sec']}s)", flush=True)

    n_i, n_c = train.finish()
    v_i, v_c = val.finish()
    print(f"\n[3/3] 合并并写出（新行 {n_i.shape[0]:,} / 基座 {b_tr_i.shape[0]:,}）", flush=True)

    train_ids = np.concatenate([b_tr_i, n_i], axis=0)
    train_codes = np.concatenate([b_tr_c, n_c], axis=0)
    val_ids = np.concatenate([b_va_i, v_i], axis=0)
    val_codes = np.concatenate([b_va_c, v_c], axis=0)

    np.save(str(out / "train.ids.npy"), train_ids)
    np.save(str(out / "train.wcode.npy"), train_codes)
    np.save(str(out / "val.ids.npy"), val_ids)
    np.save(str(out / "val.wcode.npy"), val_codes)
    shutil.copyfile(args.tokenizer, out / "bpe_tokenizer.json")

    meta = {"seq_len": args.seq_len, "train_rows": int(train_ids.shape[0]),
            "train_tokens": int(train_ids.size),
            "r1_max_chars": args.r1_max_chars, "strip_think": False,
            "reply_tokens": int((train_codes == 1).sum()),
            "prompt_tokens": int((train_codes == 2).sum()),
            "plain_tokens": int((train_codes == 3).sum()),
            "sources": stats,
            "note": "v9 base re-sliced to 2048; R1/wiki/ruozhiba fresh-encoded at "
                    "2048 with full <think> CoT (reader cap raised to "
                    f"{args.r1_max_chars} chars)"}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
