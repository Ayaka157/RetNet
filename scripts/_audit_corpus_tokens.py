"""审计「已落地语料到底有多少 token」。

背景：docs/23 §8.6 的表写「已落地语料 3.4 B token = 415,039 步」，
但训练实际用的 data_cache/chat_pack_v13 只有 269,561 × 2048 = 552 M token = 67,390 步。
差 6.16 倍，必须先搞清 3.4 B 是怎么来的，否则报给用户的数字就是错的。

要区分两件事：
  · 各版本 pack **不能相加** —— v3..v13 是同一批文本的**逐次重建**，不是互不相交的增量
  · 所以「已落地语料」要么指 v13 本身，要么指**源 jsonl** 的总量
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np

print("=== 各 pack 的 train token ===")
rows = []
for d in sorted(glob.glob("data_cache/chat_pack*")):
    f = os.path.join(d, "train.ids.npy")
    if not os.path.exists(f):
        continue
    a = np.load(f, mmap_mode="r")
    n = int(a.shape[0]) * (int(a.shape[1]) - 1)
    rows.append((os.path.basename(d), int(a.shape[0]), int(a.shape[1]) - 1, n))
    print(f"  {os.path.basename(d):<22} {a.shape[0]:>9,} x {a.shape[1]-1:<5} = {n:>15,} token ({n/1e9:.3f} B)")

tot = sum(r[3] for r in rows)
print(f"\n  所有版本相加 = {tot:,} token ({tot/1e9:.3f} B)  ← 但这会**重复计数**同一批文本")
print(f"  仅 v13（训练实际用的）= {[r[3] for r in rows if r[0]=='chat_pack_v13'][0]:,} token")

print("\n=== chat_pack_v13/meta.json ===")
mp = "data_cache/chat_pack_v13/meta.json"
if os.path.exists(mp):
    print(json.dumps(json.load(open(mp, encoding="utf-8")), ensure_ascii=False, indent=2)[:2000])

print("\n=== 源 jsonl 的体量（data_cache 下的 raw/a_grade 等）===")
for d in sorted(glob.glob("data_cache/*")):
    if not os.path.isdir(d):
        continue
    base = os.path.basename(d)
    if base.startswith("chat_pack"):
        continue
    n = 0
    for root, _dirs, files in os.walk(d):
        for fn in files:
            if fn.endswith(".jsonl"):
                n += os.path.getsize(os.path.join(root, fn))
    if n:
        print(f"  {base:<28} jsonl 合计 {n/1e9:.3f} GB")
