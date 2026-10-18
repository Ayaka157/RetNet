"""校验 HF 下载完整性：按清单里的期望字节数逐个比对，删掉不符的（让下载器重下）。

为什么必须做：`snapshot_download` 与 `curl -C -` 混用时，前者的半成品会让后者的
断点续传从**错误的偏移**继续追加，得到一个比期望还大的文件——大小不等就是坏文件，
但 `curl -C -` 见到文件存在就不会重下，于是永远卡在那儿。
"""
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/RetNet")
MANIFEST = ROOT / "logs" / "_dl_manifest.tsv"
BASE = ROOT / "data" / "hf"

ok = bad = missing = 0
deleted = []
rows = []
with open(MANIFEST, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        url, local, fname, want = parts[0], parts[1], parts[2], int(parts[3])
        rows.append((local, fname, want))

for local, fname, want in rows:
    p = BASE / local / fname
    if not p.exists():
        missing += 1
        continue
    have = p.stat().st_size
    if have == want:
        ok += 1
    else:
        bad += 1
        deleted.append((str(p.relative_to(ROOT)), have, want))
        p.unlink()

print(f"清单 {len(rows)} 个文件：完整 {ok} / 缺失 {missing} / 不符 {bad}")
for d, have, want in deleted:
    print(f"  删除 {d}  have={have} want={want}")
sys.exit(0)
