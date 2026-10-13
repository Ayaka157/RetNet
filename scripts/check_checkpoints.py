"""扫描一个目录下所有 HF 检查点，报告 safetensors 键名有没有带 torch.compile 的
``_orig_mod.`` 前缀。

存在的理由：带前缀的检查点**加载不报错**，但会静默变成随机模型（见
``scripts/strip_compile_prefix.py`` 的说明）。这个检查是二值的、O(1) 的，
应该无条件跑一遍。

    python scripts/check_checkpoints.py --root checkpoints/retnet_272m_v14
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

PREFIX = "_orig_mod."


def inspect(wf: Path):
    with open(wf, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    keys = [k for k in hdr if k != "__metadata__"]
    return keys, [k for k in keys if k.startswith(PREFIX)]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="含检查点的目录")
    ap.add_argument("--recursive", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(args.root)
    cands = sorted({p.parent for p in root.rglob("model.safetensors")})
    if not cands:
        print(f"{root} 下没有找到 model.safetensors")
        return 1

    bad = 0
    print(f"{'检查点':<44}{'键数':>7}{'带前缀':>8}  判定")
    for d in cands:
        keys, pre = inspect(d / "model.safetensors")
        bkp = (d / "model.safetensors.orig").exists()
        ok = not pre
        bad += 0 if ok else 1
        rel = str(d.relative_to(root)) or "."
        print(f"{rel:<44}{len(keys):>7}{len(pre):>8}  "
              f"{'✅ 干净' if ok else '❌ 不可加载'}"
              f"{'   （有 .orig 备份）' if bkp else ''}")
    print()
    if bad:
        print(f"❌ {bad} 个检查点带 _orig_mod. 前缀 —— 用 "
              f"scripts/strip_compile_prefix.py --in-place 修复")
        return 1
    print(f"✅ 全部 {len(cands)} 个检查点键名干净，可直接 from_pretrained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
