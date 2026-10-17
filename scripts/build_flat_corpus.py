"""把外部语料（parquet / jsonl）转成 **flat int16 token 分片**，供训练计划的 flat 模式直接读。

## 为什么需要

`data_cache/pretrain_v2` 那种 flat 语料是「一维 token 流」，训练时按
`总token // (seq_len+1)` 切行，**不需要预打包**（见 retnet_hf/data.py 的 _FlatStage）。
但从 HF 下回来的语料是 parquet/jsonl 原始文本，必须先过一遍 20k BPE。

这个脚本输出：

    <out_dir>/shard_00000.npy ...   # 每个约 --tokens-per-shard 个 int16 token
    <out_dir>/meta.json             # 来源、字段、总 token、分片清单

输出目录可以直接写进 tasks/plan.json 的 stages[].data_dir（mode=flat）。

## 用法

    python scripts/build_flat_corpus.py \
        --tokenizer data_cache/chat_pack_v13/bpe_tokenizer.json \
        --out data_cache/code_ling \
        --parquet data/hf/ling-coder-sft/*.parquet \
        --text-field prompt,response \
        --sep "\n\n" \
        --tokens-per-shard 250000000

jsonl 同理，用 `--jsonl`。`--text-field` 的多个字段按顺序用 `--sep` 拼接后编码。

## 设计取舍

- **先编码、再写盘、边写边丢**：不把全部 token 攒在内存里，只有当前分片缓冲。
- **失败要响亮**：任何一条记录编码异常都计入 `errors` 并在结尾打印；错误率 >1% 直接非零退出。
- 不用多进程：`tokenizers` 的 encode_batch 是 Rust 实现，单进程约 1–2 M tok/s，
  多进程收益有限而内存翻倍（本项目数据盘 50 GB，内存 110 GB，但没必要）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(msg, flush=True)


def iter_records(patterns, kind: str):
    """按文件顺序产出 dict 记录。"""
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"没有匹配到任何输入文件：{patterns}")
    log(f"[input] {len(files)} 个文件")
    for fi, path in enumerate(files):
        log(f"[input] ({fi + 1}/{len(files)}) {os.path.basename(path)}")
        if kind == "parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2048):
                d = batch.to_pydict()
                keys = list(d.keys())
                for i in range(len(d[keys[0]])):
                    yield {k: d[k][i] for k in keys}
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def pick_text(rec: dict, fields: list[str], sep: str) -> str | None:
    parts = []
    for f in fields:
        v = rec.get(f)
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        parts.append(v)
    if not parts:
        return None
    return sep.join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--parquet", nargs="+", help="parquet 文件通配符（可多个）")
    g.add_argument("--jsonl", nargs="+", help="jsonl 文件通配符（可多个）")
    ap.add_argument("--text-field", required=True,
                    help="逗号分隔的字段名，按顺序拼接后编码")
    ap.add_argument("--sep", default="\n\n", help="字段之间的分隔符")
    ap.add_argument("--tokens-per-shard", type=int, default=250_000_000,
                    help="每个分片的 token 数（默认 2.5e8 ≈ 500 MB int16）")
    ap.add_argument("--limit-records", type=int, default=0, help="只用前 N 条（调试）")
    ap.add_argument("--add-eos", action="store_true",
                    help="每条记录末尾追加 eos（纯文本流一般不需要，按语料决定）")
    args = ap.parse_args(argv)

    from retnet.tokenizer import BPETokenizer

    tok = BPETokenizer.load(args.tokenizer)
    log(f"[tok] vocab={tok.vocab_size}  pad={tok.pad_id} eos={tok.eos_id}")
    fields = [x.strip() for x in args.text_field.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard_*.npy"):
        old.unlink()

    buf = []
    total = 0
    errors = 0
    seen = 0
    shard_idx = 0
    shards = []
    t0 = time.time()
    kind = "parquet" if args.parquet else "jsonl"
    patterns = args.parquet or args.jsonl

    def flush(final=False):
        nonlocal buf, total, shard_idx
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.int16)
        p = out / f"shard_{shard_idx:05d}.npy"
        np.save(p, arr)
        shards.append({"file": p.name, "tokens": int(arr.shape[0])})
        log(f"[out] {p.name}  {arr.shape[0] / 1e6:.1f} M token  ({p.stat().st_size / 1e6:.0f} MB)")
        shard_idx += 1
        buf = []

    for rec in iter_records(patterns, kind):
        seen += 1
        if args.limit_records and seen > args.limit_records:
            break
        text = pick_text(rec, fields, args.sep)
        if not text:
            errors += 1
            continue
        try:
            ids = tok.encode(text)
        except Exception:                                       # noqa: BLE001
            errors += 1
            continue
        if args.add_eos:
            ids = ids + [tok.eos_id]
        buf.extend(ids)
        total += len(ids)
        while len(buf) >= args.tokens_per_shard:
            keep = buf[:args.tokens_per_shard]
            buf = buf[args.tokens_per_shard:]
            arr = np.asarray(keep, dtype=np.int16)
            p = out / f"shard_{shard_idx:05d}.npy"
            np.save(p, arr)
            shards.append({"file": p.name, "tokens": int(arr.shape[0])})
            log(f"[out] {p.name}  {arr.shape[0] / 1e6:.1f} M token  ({p.stat().st_size / 1e6:.0f} MB)")
            shard_idx += 1
        if seen % 20000 == 0:
            el = time.time() - t0
            log(f"[prog] 记录 {seen:,}  token {total / 1e6:.1f} M  "
                f"{total / max(el, 1e-9) / 1e6:.2f} M tok/s  errors={errors}")

    flush(final=True)
    meta = {
        "tokenizer": args.tokenizer,
        "tokenizer_vocab": tok.vocab_size,
        "sources": patterns,
        "text_fields": fields,
        "sep": args.sep,
        "add_eos": args.add_eos,
        "records_seen": seen,
        "records_errors": errors,
        "total_tokens": int(total),
        "shards": shards,
        "dtype": "int16",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    el = time.time() - t0
    log(f"[done] {total / 1e9:.3f} B token / {len(shards)} 分片 / 用时 {el / 60:.1f} min / "
        f"错误 {errors} 条（{errors / max(seen, 1) * 100:.2f}%）")
    if seen and errors / seen > 0.01:
        log("[fail] 错误率超过 1%，输出不可信")
        return 1
    log(f"[done] 输出目录：{out}（可直接写进 tasks/plan.json 的 data_dir，mode=flat）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
