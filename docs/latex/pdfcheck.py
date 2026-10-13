"""pdfcheck.py -- extract text from both PDFs and sanity-check the content."""
import pathlib
import sys

from pypdf import PdfReader

SPECS = [
    ("zh/RetNet-272M-技术报告.pdf", True,
     ["RetNet-272M", "保留网络", "闭式", "缺陷", "参考文献", "复现命令",
      "2176", "662", "3.29", "1127", "190.5", "0.668", "_orig_mod",
      "save_total_limit", "2.92", "5.60", "552"],
     ["We present a technical report", "reference implementation runs",
      "The packed corpus holds"]),
    ("en/RetNet-272M-Technical-Report.pdf", False,
     ["RetNet-272M", "Retention", "closed-form", "Correctness Defects",
      "References", "Reproduction", "2176", "662", "3.29", "1127", "190.5",
      "0.668", "_orig_mod", "save_total_limit", "2.92", "5.60", "552",
      "arXiv:2307.08621", "arXiv:2410.10989", "Defect Summary",
      "Reference Reliability Note"],
     ["引用可靠性说明"]),
]

bad = 0
for path, is_zh, must, must_not in SPECS:
    p = pathlib.Path(path)
    reader = PdfReader(str(p))
    text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
    print(f"=== {p.name}")
    print(f"    pages      : {len(reader.pages)}")
    print(f"    bytes      : {p.stat().st_size:,}")
    print(f"    chars      : {len(text):,}")
    for needle in must:
        ok = needle in text
        if not ok:
            bad += 1
        print(("    PASS " if ok else "    FAIL ") + repr(needle))
    for needle in must_not:
        ok = needle not in text
        if not ok:
            bad += 1
        print(("    PASS " if ok else "    FAIL ") + "absent: " + repr(needle))
    print()

print(f"{bad} failures")
sys.exit(1 if bad else 0)
