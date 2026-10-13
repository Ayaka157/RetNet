"""pdfbounds.py -- check that no page's content exceeds the paper size.

TeX reports an `Overfull \\vbox` when a page's content cannot be broken to fit,
but the log line says neither which page nor by how much -- and it does not
appear in `report.py`'s hbox accounting at all.  This reads the PDFs themselves:
for every page it takes the lowest content edge (text *and* drawn rules, since a
table's rules overflow just as visibly as its glyphs) and compares it with the
page height, plus the rightmost edge against the width.

Exit status is non-zero when any page overflows, so it works as a build gate.

Run from anywhere:  python docs/latex/pdfbounds.py [report-key ...]
"""
import sys

import mktex

# sub-point rounding in the PDF writer is not an overflow
EPSILON = 0.5


def page_bounds(page) -> tuple[float, float]:
    """(max y1, max x1) of everything drawn on the page."""
    y1 = 0.0
    x1 = 0.0
    for block in page.get_text("blocks"):
        x1 = max(x1, block[2])
        y1 = max(y1, block[3])
    for path in page.get_drawings():
        rect = path["rect"]
        x1 = max(x1, rect.x1)
        y1 = max(y1, rect.y1)
    return y1, x1


def check(path, is_zh: bool) -> tuple[bool, str]:
    import fitz

    doc = fitz.open(path)
    lines = []
    worst = 0.0
    bad_pages = []
    for i, page in enumerate(doc, start=1):
        y1, x1 = page_bounds(page)
        over_y = y1 - page.rect.height
        over_x = x1 - page.rect.width
        worst = max(worst, over_y, over_x)
        if over_y > EPSILON or over_x > EPSILON:
            bad_pages.append((i, over_y, over_x))
    lines.append(f"    pages      : {doc.page_count}")
    lines.append(f"    last page  : content y1 = "
                 f"{page_bounds(doc[doc.page_count - 1])[0]:.2f} pt, "
                 f"page height = {doc[0].rect.height:.2f} pt")
    lines.append(f"    worst overflow : {worst:+.2f} pt")
    for i, oy, ox in bad_pages:
        lines.append(f"      OVERFLOW page {i}: +{oy:.2f} pt below, "
                     f"+{ox:.2f} pt right")
    ok = not bad_pages
    lines.append(f"    verdict    : {'PASS' if ok else 'FAIL'}")
    doc.close()
    return ok, "\n".join(lines)


def main(argv: list[str]) -> int:
    keys = argv or [k for k, r in mktex.REPORTS.items() if r.default]
    bad = 0
    checked = 0
    for key in keys:
        report = mktex.REPORTS[key]
        for pdf in (report.zh_dir / (report.zh_name + ".pdf"),
                    report.en_dir / (report.en_name + ".pdf")):
            if not pdf.exists():
                print(f"=== {report.key}: {pdf.name} not built yet -- skipped")
                continue
            ok, detail = check(str(pdf), pdf.parent.name == "zh")
            print(f"=== {pdf}")
            print(detail)
            print()
            checked += 1
            if not ok:
                bad += 1
    print(f"{checked - bad}/{checked} PDFs fit inside the paper")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
