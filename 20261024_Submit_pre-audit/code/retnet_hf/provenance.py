"""数据来源（provenance）检查：验证集与训练阶段“是不是同一批记录”。

## 为什么需要这个（docs/00 E14）

上一轮的守卫只比较目录字符串（``Path(...).resolve() == ...``），它守的是
**路径相等**，而真实事故是**数据相等**：``val_dir=chat_pack_v13``，而最后
14,000 步的 ``sft_mix`` 里有 40% 就是 chat_pack_v13。两个目录名不同、
内容重叠 —— 路径守卫看不见。

这里把守卫拆成三层，从便宜到昂贵：

1. **规范化路径**（``norm_key``）：``resolve()`` + ``normcase()``，因此
   ``os.path.relpath``、尾斜杠、大小写差异都会落到同一个键上。
   这是旧注释声称、旧实现没做到的那一层。
2. **声明来源**（``declared_sources``）：读 ``<dir>/meta.json`` 的
   ``source_ids``（新产线写入）或 ``sources``+``text_fields``（兼容旧产线）。
   两个目录声明了同一个来源 -> 直接拒绝，不需要读一个字节的数据。
3. **内容指纹**（``dir_row_hashes``）：按行做 64 位 FNV 式哈希，比较集合。
   它不看文件名，因此“同一批记录被重新打包到另一个目录”也能发现。
   对大语料有 ``max_rows`` 上限（默认 200k 行 ≈ 0.8 GB），截断时返回
   ``truncated`` 供调用方如实打印，绝不假装扫全了。

三层都通过才放行。任何一层不可用（没有 meta、目录读不了）都记录在
``notes`` 里，由调用方打印，不会静默变成“无事发生”。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

#: 内容指纹默认最多扫多少行（每行 seq_len+1 个 int16）。
DEFAULT_MAX_ROWS = 200_000
_FNV_OFFSET = np.uint64(1469598103934665603)
_FNV_PRIME = np.uint64(1099511628211)


def norm_key(path) -> str:
    """目录的规范化键：绝对路径 + 大小写折叠（Windows）。

    相对路径（``os.path.relpath``）、尾斜杠、大小写变体都映射到同一个键。
    """
    return os.path.normcase(str(Path(path).resolve()))


def _read_meta(d) -> dict:
    p = Path(d) / "meta.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def declared_sources(d) -> set[str]:
    """目录声明的来源标识集合（空集 = 没有声明，不能据此判定）。"""
    m = _read_meta(d)
    out: set[str] = set()
    ids = m.get("source_ids")
    if isinstance(ids, str):
        out.add(ids)
    elif isinstance(ids, (list, tuple)):
        out.update(str(x) for x in ids)
    if out:
        return out
    # 兼容旧产线：build_flat_corpus 一直写 sources + text_fields。
    srcs = m.get("sources")
    if isinstance(srcs, str):
        srcs = [srcs]
    if srcs:
        fields = m.get("text_fields") or []
        if isinstance(fields, str):
            fields = [fields]
        tag = "|".join(sorted(str(f) for f in fields)) if fields else "*"
        out.update(f"{s}#{tag}" for s in sorted(str(x) for x in srcs))
    return out


def _row_hashes(arr: np.ndarray, width: int,
                max_rows: int) -> tuple[np.ndarray, int, bool]:
    """逐行算 64 位哈希，返回 (hashes, rows, truncated)。

    用 Numpy 逐列推进的 FNV-1a：对 2049 列做 2049 次向量化异或/乘法，
    比 Python 逐行 ``hashlib`` 快一个量级，且不需要把行变成 Python 对象。
    取 ``& 0xFFFF`` 是为了让 int16 的负值不会符号扩展成不同的 uint64。
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        n = int(a.shape[0]) // width
        rows = a[: n * width].reshape(n, width) if n else a[:0].reshape(0, width)
    elif a.ndim == 2:
        if a.shape[1] != width:
            raise ValueError(f"行宽 {a.shape[1]} 与期望的 {width} 不一致")
        rows = a
    else:
        raise ValueError(f"不支持 {a.ndim} 维数组")
    n = int(rows.shape[0])
    truncated = bool(max_rows and n > max_rows)
    if truncated:
        rows = rows[:max_rows]
        n = max_rows
    h = np.full(n, _FNV_OFFSET, dtype=np.uint64)
    for j in range(width):
        col = rows[:, j].astype(np.int64).astype(np.uint64) & np.uint64(0xFFFF)
        h = (h ^ col) * _FNV_PRIME
    return h, n, truncated


def dir_row_hashes(d, seq_len: int, max_rows: int = DEFAULT_MAX_ROWS,
                   prefer_val: bool = False) -> dict:
    """目录内 npy 的行哈希，最多 ``max_rows`` 行。

    ``prefer_val=True`` 用于**验证集目录**：优先读 ``val.ids.npy``。
    旧实现一律跳过文件名含 ``val`` 的分片，于是验证集目录里的行根本不会被
    哈希 —— 内容层永远比较两个空集合。训练侧仍是 packed 读 ``train.ids.npy``、
    flat 读全部非 val 分片（与 ``_FlatStage._scan`` 同一口径）。

    返回 ``{"hashes": np.ndarray, "rows": int, "truncated": bool,
    "files": [name]}``。
    """
    d = Path(d)
    width = int(seq_len) + 1
    if prefer_val:
        pv = d / "val.ids.npy"
        candidates = [pv] if pv.is_file() else sorted(d.glob("*.npy"))
    else:
        pt = d / "train.ids.npy"
        candidates = [pt] if pt.is_file() else \
            sorted(p for p in d.glob("*.npy") if "val" not in p.name.lower())
    parts: list[np.ndarray] = []
    files: list[str] = []
    scanned = 0
    truncated = False
    for p in candidates:
        if scanned >= max_rows:
            truncated = True
            break
        try:
            arr = np.load(p, mmap_mode="r")
        except Exception:                                   # noqa: BLE001
            continue
        if arr.ndim not in (1, 2):
            continue
        if arr.ndim == 2 and arr.shape[1] != width:
            continue
        h, n, cut = _row_hashes(np.asarray(arr), width, max_rows - scanned)
        parts.append(h)
        scanned += n
        files.append(p.name)
        truncated = truncated or cut
    hashes = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)
    return {"hashes": hashes, "rows": int(scanned), "truncated": truncated,
            "files": files}


def overlap_report(val_dir, train_dirs, seq_len: int,
                   max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    """比较验证集与每个训练目录：路径 / 声明来源 / 内容指纹。

    返回 ``{"path_hits": [...], "source_hits": [...], "content_hits": [...],
    "val_rows": n, "val_truncated": bool, "scanned": {dir: rows},
    "notes": [...]}``。命中项都是目录字符串，由调用方决定报错文案。
    """
    v_key = norm_key(val_dir)
    v_src = declared_sources(val_dir)
    v_rows = dir_row_hashes(val_dir, seq_len, max_rows, prefer_val=True)
    out = {"path_hits": [], "source_hits": [], "content_hits": [],
           "val_rows": v_rows["rows"], "val_truncated": v_rows["truncated"],
           "scanned": {}, "notes": []}
    if not v_rows["hashes"].size:
        out["notes"].append(f"验证集 {val_dir} 里没有可读的 npy 行，内容层跳过")
    if not v_src:
        out["notes"].append(f"验证集 {val_dir} 没有 meta.json 来源声明，声明层跳过")
    for d in train_dirs:
        if norm_key(d) == v_key:
            out["path_hits"].append(str(d))
        src = declared_sources(d)
        if v_src and src and (v_src & src):
            out["source_hits"].append(
                f"{d}（共同来源 {sorted(v_src & src)}）")
        h = dir_row_hashes(d, seq_len, max_rows)
        out["scanned"][str(d)] = h["rows"]
        if h["truncated"]:
            out["notes"].append(
                f"{d} 只扫了前 {h['rows']:,} 行（达到上限），内容层是不完整检查")
        if v_rows["hashes"].size and h["hashes"].size:
            common = np.intersect1d(v_rows["hashes"], h["hashes"])
            if common.size:
                out["content_hits"].append(
                    f"{d}（{int(common.size)} 行完全相同）")
    return out


def describe(rep: dict) -> str:
    """把 ``overlap_report`` 的结果写成一行人能读的结论。"""
    scanned = ", ".join(f"{Path(k).name}={v:,}" for k, v in rep["scanned"].items())
    notes = "；".join(rep["notes"])
    head = f"验证集 {rep['val_rows']:,} 行"
    if rep["val_truncated"]:
        head += "（已截断到上限）"
    tail = f"；训练侧扫描 {scanned}" if scanned else ""
    return head + tail + (f"；{notes}" if notes else "")
