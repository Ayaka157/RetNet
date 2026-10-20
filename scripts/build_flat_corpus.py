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


def iter_records(patterns, kind: str, skip=()):
    """按文件顺序产出 dict 记录。

    **打不开的文件跳过**：下载器可能正在写某个 parquet，footer 还没落盘，
    ``ParquetFile`` 会抛 ``Parquet magic bytes not found in footer``。
    跳过而不是崩 —— 让分片能在数据齐了之后重跑。
    """
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat, recursive=True)))
    if skip:
        n0 = len(files)
        files = [f for f in files if not any(k in os.path.basename(f) for k in skip)]
        if len(files) != n0:
            log(f"[input] 跳过 {n0 - len(files)} 个非文本文件（匹配 {skip}）")
    if not files:
        raise SystemExit(f"没有匹配到任何输入文件：{patterns}")
    log(f"[input] {len(files)} 个文件")
    skipped = 0
    for fi, path in enumerate(files):
        if kind == "parquet":
            import pyarrow.parquet as pq
            try:
                pf = pq.ParquetFile(path)
            except Exception as exc:                            # noqa: BLE001
                skipped += 1
                log(f"[input] 跳过 {os.path.basename(path)}（{type(exc).__name__}，可能正在下载）")
                continue
            log(f"[input] ({fi + 1}/{len(files)}) {os.path.basename(path)}")
            # ⚠️ 只包住 open 不够：大小校验通过、但内容损坏的文件会在 iter_batches 里
            #    抛 "Corrupt snappy compressed data"，把整次分词打断（2026-10-18 实测）。
            #    跳过它继续，让这一轮仍然产出可用分片。
            try:
                for batch in pf.iter_batches(batch_size=2048):
                    d = batch.to_pydict()
                    keys = list(d.keys())
                    for i in range(len(d[keys[0]])):
                        yield {k: d[k][i] for k in keys}
            except Exception as exc:                            # noqa: BLE001
                skipped += 1
                log(f"[input] 中途损坏，放弃 {os.path.basename(path)}（{type(exc).__name__}: {exc}）")
                continue
        else:
            log(f"[input] ({fi + 1}/{len(files)}) {os.path.basename(path)}")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def flatten_value(v) -> str:
    """把任意字段值压成文本。

    HF 数据集的字段形态很杂：纯字符串、``[{'role','content'}]`` 的对话列表、
    ``[{'explanation':...}]`` 的标注列表、dict…… 直接 ``str(v)`` 会得到 Python repr
    （带 ``{'role': ...}`` 噪声），必须按结构展开。
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts = []
        for item in v:
            if isinstance(item, dict):
                role = item.get("role") or item.get("from") or ""
                content = item.get("content")
                if content is None:
                    content = item.get("text") or item.get("explanation") or ""
                parts.append((str(role) + ": " + str(content)) if role else str(content))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(v, dict):
        for k in ("content", "text", "value"):
            if v.get(k):
                return str(v[k])
        return ""
    return str(v)


#: auto 模式下跳过的字段（ID / 链接 / 元信息，进训练是纯噪声）
_SKIP_FIELD_HINTS = ("id", "uuid", "url", "uri", "hash", "sha", "score", "count",
                     "index", "license", "source", "date", "time")


def auto_text_fields(parquet_file) -> list[str]:
    """从 parquet schema 猜要拼接的字段：留下像「正文」的，剔掉 ID/元信息。"""
    import pyarrow.parquet as pq
    names = pq.ParquetFile(parquet_file).schema_arrow.names
    keep = []
    for n in names:
        low = n.lower()
        if any(h in low for h in _SKIP_FIELD_HINTS):
            continue
        keep.append(n)
    return keep or names


def pick_text(rec: dict, fields: list[str], sep: str) -> str | None:
    parts = []
    for f in fields:
        s = flatten_value(rec.get(f))
        if s:
            parts.append(s)
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
    ap.add_argument("--skip", default="",
                    help="逗号分隔的文件名关键字，命中的文件跳过（如 stats.json）")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="产出达到 N token 就停（0=不限）。计划只需要一部分语料时用它省时间省磁盘")
    ap.add_argument("--add-eos", action="store_true", default=True,
                    help="每条记录末尾追加 eos。**默认开启**：v21 的所有 hf_pack_* 都漏了它，"
                         "78% 的训练步数跑在无文档边界的流上（docs/00 E09/E10）。")
    ap.add_argument("--no-add-eos", dest="add_eos", action="store_false",
                    help="显式关闭。只有确认下游会自己补边界时才用；关之前先读 docs/00 E09。")
    args = ap.parse_args(argv)

    # 直接用 tokenizers 库：retnet/tokenizer.py 的 BPETokenizer 只是它的薄封装，
    # 而实例上并没有上传 retnet/ 包（训练只依赖 retnet_hf/），依赖它会让分词跑不起来。
    # add_special_tokens=False 与 BPETokenizer.encode 的实现逐字一致，
    # 保证新语料与 chat_pack_v13 的编码口径相同。
    from tokenizers import Tokenizer as _Tokenizer

    _tk = _Tokenizer.from_file(str(args.tokenizer))
    _vocab = _tk.get_vocab_size()
    _eos = _tk.token_to_id("<eos>")

    def encode_one(text: str):
        return _tk.encode(text, add_special_tokens=False).ids

    log(f"[tok] vocab={_vocab}  eos={_eos}（直接走 tokenizers，不依赖 retnet 包）")
    if args.text_field.strip().lower() == "auto":
        first = None
        fields = []
        for pat in list(args.parquet or []) + list(args.jsonl or []):
            for f in sorted(glob.glob(pat, recursive=True)):
                try:
                    if f.endswith((".jsonl", ".json")):
                        # jsonl 没有 schema，取第一条记录的键
                        with open(f, encoding="utf-8") as fh:
                            for line in fh:
                                line = line.strip()
                                if line:
                                    fields = list(json.loads(line).keys())
                                    break
                        if not fields:
                            continue
                    else:
                        fields = auto_text_fields(f)
                    first = f
                    break
                except Exception:                               # noqa: BLE001
                    continue       # 正在下载的坏文件，跳过
            if first:
                break
        if first is None:
            raise SystemExit("--text-field auto 失败：没有任何可读的 parquet/jsonl 文件")
        log(f"[fields] auto 选中（来自 {os.path.basename(first)}）: {fields}")
    else:
        fields = [x.strip() for x in args.text_field.split(",") if x.strip()]
    skip_names = tuple(x.strip() for x in args.skip.split(",") if x.strip())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard_*.npy"):
        old.unlink()

    kind = "parquet" if args.parquet else "jsonl"
    patterns = args.parquet or args.jsonl

    # 2026-10-20（docs/00 E10）：显式字段要"早失败"。
    # 以前字段写错时，pick_text 对每条记录都返回空，全部计入 errors，
    # 要等分词跑完（可能几小时）才在结尾以 >1% 错误率退出；更糟的是
    # 若字段只在部分记录里缺失，会静默丢掉那部分数据。
    # 这里先读第一条记录，一个字段都对不上就立刻退出，并打印可用字段。
    _first = next(iter_records(patterns, kind, skip=skip_names), None)
    if _first is None:
        raise SystemExit(f"输入里没有任何可读记录：{patterns}")
    _missing = [f for f in fields if f not in _first]
    if len(_missing) == len(fields):
        raise SystemExit(
            f"--text-field {fields} 在第一条记录里一个都不存在。\n"
            f"  可用字段: {sorted(_first)}\n"
            f"  请显式指定正文列；不要用 auto（它会把数值/元数据列拼进正文）。")

    buf = []
    total = 0
    errors = 0
    seen = 0
    shard_idx = 0
    shards = []
    t0 = time.time()

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

    for rec in iter_records(patterns, kind, skip=skip_names):
        seen += 1
        if args.limit_records and seen > args.limit_records:
            break
        # token 上限：计划只吃 2.07B token 的代码，而整份语料分词要 2 小时、
        # 产出 7B+ token，既浪费磁盘又赶不上阶段切换。够了就停。
        if args.max_tokens and total >= args.max_tokens:
            log(f"[cap] 已达 {args.max_tokens:,} token 上限，提前结束")
            break
        text = pick_text(rec, fields, args.sep)
        if not text:
            errors += 1
            continue
        try:
            ids = encode_one(text)
        except Exception:                                       # noqa: BLE001
            errors += 1
            continue
        if args.add_eos:
            ids = ids + [_eos]
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
        "tokenizer_vocab": _vocab,
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
