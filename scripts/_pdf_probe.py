"""PDF 检索小工具：按关键词返回页码与上下文，供文档标注"页码+章节"。

用法：
    python scripts/_pdf_probe.py --pdf Papers/01_xxx.pdf --grep "Chunkwise" --ctx 3
    python scripts/_pdf_probe.py --pdf ... --page 4        # 打印整页
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--grep", default=None)
    ap.add_argument("--ctx", type=int, default=2)
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--page", type=int, default=None)
    args = ap.parse_args(argv)

    doc = fitz.open(args.pdf)
    if args.page is not None:
        print(doc[args.page - 1].get_text())
        return 0
    if not args.grep:
        raise SystemExit("需要 --grep 或 --page")

    pat = re.compile(args.grep, re.I)
    hits = 0
    for pno in range(doc.page_count):
        text = doc[pno].get_text()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if pat.search(ln):
                hits += 1
                lo = max(0, i - args.ctx)
                hi = min(len(lines), i + args.ctx + 1)
                print(f"--- p.{pno + 1}")
                for j in range(lo, hi):
                    print(("  >> " if j == i else "     ") + lines[j][:160])
                if hits >= args.max:
                    return 0
    if hits == 0:
        print("(no match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
