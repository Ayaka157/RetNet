#!/usr/bin/env python
"""Build data_cache/index.json and data_cache/INDEX.md.

Written for plan reorg-20261024, item T2.9, which requires that
`data_cache/index.json` record each archivable pack's meta.json verbatim plus a
per-file SHA-256 before any of those bytes are relocated.

Design notes
------------
* .npy headers are parsed by hand (magic + header dict only). The array payload
  is NEVER read; a 885 MB shard costs a few hundred bytes of I/O.
* Per-file SHA-256 is computed ONLY for the DERIVABLE set (the files whose bytes
  will change location). Hashing all 43 GB would add no safety for files that do
  not move.
* `live_consumers` counts references from code that can actually run today:
  tasks/, scripts/, retnet_hf/, retnet/, tests/, serve/, run_task.py and the
  three top-level docs. The legacy/, Kaggle/, logs/ and submission snapshots are
  excluded on purpose: a reference from an archived tree is not a live consumer.

Usage:  python scripts/build_data_index.py [--hash]   (--hash forces re-hashing)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_cache")

# Classification per docs/plans/PLAN.reorg-20261024.md section 6.
CLASS = {
    "pretrain_v2": "CURRENT",
    "chat_pack_v13": "CURRENT",
    "chat_pack_v9": "BUILD_INPUT",
    "chat_pack_v10": "BUILD_INPUT",
    "chat_pack_v11": "DERIVABLE",
    "chat_pack_v11_4k": "DERIVABLE",
    "chat_pack_v12": "DERIVABLE",
    "chat_pack_v3": "SUPERSEDED",
    "chat_pack_v4": "SUPERSEDED",
    "chat_pack_v5": "SUPERSEDED",
    "chat_pack_v6": "SUPERSEDED",
    "chat_pack_v7": "SUPERSEDED",
    "chat_pack_v8": "SUPERSEDED",
    "pretrain_merged": "SUPERSEDED",
    "a_grade": "ROOT",
    "roleplay": "ROOT",
    "literature": "ROOT",
    "chat_corpus": "ROOT",
    "extra_sft": "ROOT",
    "extra_sft2": "ROOT",
    "nonhf": "ROOT",
    "wikitext103_v16k": "INCOMPATIBLE",
    "wikitext2": "INCOMPATIBLE",
    "chat_recovery_mix": "NO_MASK_CORRUPT",
    "chat_sft_mix": "NO_MASK_CORRUPT",
    "_resume_probe": "TEST_FIXTURE",
}

# Where a dataset came from, stated only where the dataset itself or a script proves it.
DERIVED = {
    "chat_pack_v11": ("chat_pack_v10", "meta.json:source = 'v10 re-pack to 8K'"),
    "chat_pack_v11_4k": ("chat_pack_v10", "meta.json:source = 'v10 re-pack to 8K'"),
    "chat_pack_v12": ("chat_pack_v9", "meta.json:sources.base_v9; strip_think=true"),
}

LIVE_SOURCES = ["tasks", "scripts", "retnet_hf", "retnet", "tests", "serve"]
LIVE_FILES = ["run_task.py", "README.md", "RUNBOOK.md", "CANONICAL.md"]


def npy_header(path: str):
    """Return (shape, dtype) from a .npy header without touching the payload."""
    try:
        with open(path, "rb") as fh:
            if fh.read(6) != b"\x93NUMPY":
                return None
            major = fh.read(1)[0]
            fh.read(1)  # minor
            hlen = int.from_bytes(fh.read(2 if major == 1 else 4), "little")
            header = ast.literal_eval(fh.read(hlen).decode("latin1"))
        return list(header["shape"]), header["descr"]
    except Exception:
        return None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def live_consumer_map(names):
    """Count references to each dataset name from code that can run today."""
    hits = {n: set() for n in names}
    texts = []
    for d in LIVE_SOURCES:
        base = os.path.join(ROOT, d)
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in filenames:
                if fn.endswith((".py", ".json", ".ps1", ".bat", ".sh", ".md")):
                    texts.append(os.path.join(dirpath, fn))
    for fn in LIVE_FILES:
        p = os.path.join(ROOT, fn)
        if os.path.exists(p):
            texts.append(p)
    for p in texts:
        try:
            if os.path.getsize(p) > 4 << 20:
                continue
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                body = fh.read()
        except OSError:
            continue
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        for n in names:
            if n in body:
                hits[n].add(rel)
    return {n: sorted(v) for n, v in hits.items()}


def main():
    force_hash = "--hash" in sys.argv
    names = sorted(
        d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))
    )
    consumers = live_consumer_map(names)
    out = {
        "generated_by": "scripts/build_data_index.py",
        "generated_for": "docs/plans/PLAN.reorg-20261024.md (T2.9 precondition)",
        "note": "Per-file SHA-256 is recorded only for DERIVABLE packs, i.e. the "
                "files whose bytes change location. Everything else is unchanged.",
        "datasets": {},
    }
    for name in names:
        d = os.path.join(DATA, name)
        files = []
        total = 0
        for dirpath, _dirnames, filenames in os.walk(d):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                total += sz
                files.append((os.path.relpath(p, d).replace("\\", "/"), sz))
        files.sort()
        rec = {
            "class": CLASS.get(name, "UNSURE"),
            "files": len(files),
            "bytes": total,
            "gb": round(total / (1 << 30), 3),
            "live_consumers": consumers.get(name, []),
        }
        meta_path = os.path.join(d, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    rec["meta"] = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                rec["meta_error"] = str(exc)
        if name in DERIVED:
            rec["derived_from"], rec["derived_evidence"] = DERIVED[name]
        # array headers (payload never read)
        arrays = {}
        for rel, _sz in files:
            if rel.endswith(".npy"):
                hdr = npy_header(os.path.join(d, rel))
                if hdr:
                    arrays[rel] = {"shape": hdr[0], "dtype": hdr[1]}
        if arrays:
            rec["arrays"] = arrays
        # per-file hashes only for the archivable set
        if rec["class"] == "DERIVABLE" or force_hash:
            rec["file_sha256"] = {
                rel: sha256(os.path.join(d, rel)) for rel, _sz in files
            }
        rec["file_sizes"] = {rel: sz for rel, sz in files}
        out["datasets"][name] = rec

    with open(os.path.join(DATA, "index.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, sort_keys=True)
    write_markdown(out)
    print(f"datasets indexed: {len(out['datasets'])}")
    print("wrote data_cache/index.json and data_cache/INDEX.md")


def write_markdown(out):
    order = ["CURRENT", "BUILD_INPUT", "DERIVABLE", "SUPERSEDED", "ROOT",
             "INCOMPATIBLE", "NO_MASK_CORRUPT", "TEST_FIXTURE", "UNSURE"]
    d = out["datasets"]
    lines = [
        "# data_cache 索引",
        "",
        "由 `scripts/build_data_index.py` 生成，**不要手工编辑**。",
        "",
        "这是 `docs/plans/PLAN.reorg-20261024.md` 第 6 节的机器可读版本：路径不变，只是把",
        "「哪一包是什么、谁在用、能不能重算」从散文变成可查询的记录。",
        "",
        "类别含义：`ROOT` 原始语料，此处不可再生成；`CURRENT` 当前在用；",
        "`BUILD_INPUT` 是别的包的构建基座；`DERIVABLE` 可由保留的基座在数分钟内重算，",
        "**只有这一类会被归档**；`SUPERSEDED` 历史对照；`INCOMPATIBLE` 词表不符；",
        "`NO_MASK_CORRUPT` 无 loss mask 或内容损坏；`TEST_FIXTURE` 测试夹具。",
        "",
        "`live_consumers` 只统计今天能跑的代码（tasks/ scripts/ retnet/ retnet_hf/",
        "tests/ serve/ 与三个顶层文档），不含 legacy/、Kaggle/、logs/ 与提交快照 ——",
        "归档树里的引用不算活引用。",
        "",
        "| 数据集 | 类别 | 文件 | GB | 活引用 | 来源 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for cls in order:
        for name in sorted(n for n, r in d.items() if r["class"] == cls):
            r = d[name]
            src = r.get("derived_from", "")
            lines.append(
                f"| `{name}` | {cls} | {r['files']} | {r['gb']} | "
                f"{len(r['live_consumers'])} | {src} |"
            )
    lines.append("")
    with open(os.path.join(DATA, "INDEX.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
