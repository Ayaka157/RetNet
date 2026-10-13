"""Build chat_pack_v11 = v10 rows re-packed to 8K-wide windows.

v10 rows are 513 wide (512 seq + 1 label).  We concatenate the whole corpus
(token stream with wcode labels), drop pad markers (wcode==0), then slice
into 8193-wide rows (8192 seq + 1 label slot maintained by Packer).

Output format matches chat_pack_v9/v10 (train.ids.npy etc.) so the same
Trainer path works with seq_len=8192.

    python scripts/build_chat_mix_v11.py --base data_cache/chat_pack_v10 \
        --out_dir data_cache/chat_pack_v11 --seq_len 8192
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.packing import Packer  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402


def flatten(ids_npy, codes_npy, row_len):
    """Flatten packed rows into (token_stream, code_stream), dropping pad.

    v10 rows have shape (N, row_len=513); last column is the label slot? No —
    the Packer rows are (seq_len+1) wide but the label is the NEXT token's
    code already inside the row; the extra column is the shift target.  We
    drop tokens whose wcode==0 (padding) and concatenate the rest.
    """
    idx = np.load(ids_npy, mmap_mode="r")
    cd = np.load(codes_npy, mmap_mode="r")
    ids = np.asarray(idx, dtype=np.int32).reshape(-1)
    codes = np.asarray(cd, dtype=np.uint8).reshape(-1)
    keep = codes != 0
    return ids[keep], codes[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data_cache/chat_pack_v10")
    ap.add_argument("--out_dir", default="data_cache/chat_pack_v11")
    ap.add_argument("--tokenizer", default="data_cache/chat_pack_v10/bpe_tokenizer.json")
    ap.add_argument("--seq_len", type=int, default=8192)
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
    base = Path(args.base)

    for split in ("train", "val"):
        ids, codes = flatten(base / f"{split}.ids.npy", base / f"{split}.wcode.npy", 513)
        print(f"{split}: flattened {len(ids):,} tokens", flush=True)
        # slice into rows of seq_len; keep a 1-token overlap for the label slot
        n_rows = len(ids) // args.seq_len
        sliced = ids[: n_rows * args.seq_len].reshape(n_rows, args.seq_len)
        sliced_codes = codes[: n_rows * args.seq_len].reshape(n_rows, args.seq_len)
        # append shifted target: rows are (seq_len) with the NEXT token as label
        # handled by dataset shift; store rows+1 wide like v9? The HF trainer
        # shifts internally; v9 stored seq_len+1 to carry the label.  We store
        # seq_len wide and let PackedChatDataset handle the +1 (it reads
        # shape[1]-1 as seq_len, so we must store seq_len+1).
        # Simplest: store seq_len+1 by appending the next token.
        label_ofs = np.roll(ids, -1)
        label_ofs[-1] = ids[-1] if len(ids) else 0
        rows_ids = np.concatenate(
            [ids[: n_rows * args.seq_len].reshape(n_rows, args.seq_len),
             np.asarray(label_ofs[: n_rows * args.seq_len]).reshape(n_rows, args.seq_len)[:, -1:]], axis=1)
        rows_codes = np.concatenate(
            [codes[: n_rows * args.seq_len].reshape(n_rows, args.seq_len),
             codes[: n_rows * args.seq_len].reshape(n_rows, args.seq_len)[:, -1:]], axis=1)
        np.save(str(out / f"{split}.ids.npy"), rows_ids.astype(np.int16))
        np.save(str(out / f"{split}.wcode.npy"), rows_codes.astype(np.uint8))
        print(f"  -> {rows_ids.shape[0]:,} rows x {rows_ids.shape[1]}", flush=True)

    shutil.copyfile(args.tokenizer, out / "bpe_tokenizer.json")
    train_ids = np.load(out / "train.ids.npy", mmap_mode="r")
    codes = np.load(out / "train.wcode.npy", mmap_mode="r")
    meta = {"seq_len": args.seq_len, "train_rows": int(train_ids.shape[0]),
            "train_tokens": int(train_ids.size),
            "reply_tokens": int((codes == 1).sum()),
            "prompt_tokens": int((codes == 2).sum()),
            "plain_tokens": int((codes == 3).sum()),
            "source": "v10 re-pack to 8K", "out_dir": str(out)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
