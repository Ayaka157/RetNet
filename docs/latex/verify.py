"""verify.py -- structural checks on the generated LaTeX editions.

Two layers:

  * the original RetNet-272M content checks, kept verbatim, because they are
    the accepted regression evidence for that report;
  * generic checks that run for every report configured in `mktex.REPORTS`
    whose `.tex` files have been generated: language markers must not reach a
    table of contents, section numbers must be well formed, the Chinese edition
    must carry Chinese prose, and the English edition must not carry Chinese
    prose outside code (listings, `\\code{}`, `\\texttt{}`).

The table of contents is written by xelatex, so it is the only place that shows
what actually reached the reader.  Checking the .tex alone once let a
`1.1 English` heading ship, because the marker survived into the TOC while
the .tex looked clean.

Run from anywhere:  python docs/latex/verify.py
"""
import re
import sys
from pathlib import Path

import mktex

MARKER_HEADING = re.compile(
    r"\\contentsline \{(?:sub)*section\}\{[^}]*\}\{\s*(?:\d+(?:\.\d+)*\s+)?"
    r"(?:English|中文)(?:\s*\(condensed\))?\s*\}"
)

NUMBERLINE = re.compile(r"\\contentsline \{(?:sub)*section\}\{\\numberline \{([^}]*)\}")

# A well-formed number is digits/dots, or an appendix letter followed by an
# optional dotted number -- never a repeated separator and never bare.
NUMBER_OK = re.compile(r"^(?:\d+\.?|\d+(?:\.\d+)+\.?|[A-Z]\.?\d*)$")

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Code is allowed to keep its Chinese comments (they are part of the source),
# so a prose-language check has to look past the code first.
CODE_STRIP = [
    re.compile(r"\\begin\{lstlisting\}(\[[^\]]*\])?.*?\\end\{lstlisting\}", re.S),
    re.compile(r"\\code\{[^{}]*\}"),
    re.compile(r"\\texttt\{[^{}]*\}"),
    re.compile(r"\\href\{[^{}]*\}"),
    # a LaTeX comment never reaches the page, so it is not prose either
    re.compile(r"(?<!\\)%[^\n]*"),
]


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(text))


def prose(text: str) -> str:
    """The .tex with listings, inline code and link targets removed."""
    for pattern in CODE_STRIP:
        text = pattern.sub(" ", text)
    return text


def bad_numbers(text: str) -> list[str]:
    bad = []
    for num in NUMBERLINE.findall(text):
        n = num.strip()
        if n in ("",):
            continue
        if not NUMBER_OK.match(n) or ".." in n:
            bad.append(n)
    return bad


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def checks_272m(zh: str, en: str, toc_zh: str, toc_en: str) -> list[tuple[str, bool]]:
    """The original RetNet-272M checks, unchanged."""
    return [
        # --- Chinese edition must be Chinese-only in prose -------------------
        ("ZH: no English condensed block", "reference implementation runs" not in zh),
        ("ZH: no English abstract", "We present a technical report" not in zh),
        ("ZH: Chinese abstract present", "保留网络" in zh),
        ("ZH: 3.4.3 Chinese text kept", "参考实现逐块执行" in zh),
        ("ZH: Chinese 1.1 prose kept", "本项目的目标是" in zh),
        ("ZH: references present", "arXiv:2307.08621" in zh),
        ("ZH: appendix D present", "缺陷清单" in zh),
        ("ZH: no _orig_mod English prose leak", "every signature" not in zh),

        # --- English edition -------------------------------------------------
        ("EN: introduction prose", "usable Chinese dialogue model" in en),
        ("EN: background prose", "replaces attention with" in en),
        ("EN: closed-form scan covered", "Closed-form batched matmul" in en),
        ("EN: 190.5us figure present", "190.5" in en),
        ("EN: English abstract", "We present a technical report" in en),
        ("EN: supplement architecture", "Multi-Scale Retention" in en),
        ("EN: supplement data", "Measured Composition" in en),
        ("EN: supplement pipeline", "Loss Normalisation" in en),
        ("EN: supplement optimizations", "Kernel-Level Attribution" in en),
        ("EN: supplement defects", "device-side assert triggered" in en),
        ("EN: supplement evaluation", "physically impossible" in en),
        ("EN: supplement appendix D", "Defect Summary" in en),
        ("EN: references present", "arXiv:2307.08621" in en),
        ("EN: references label", r"\label{sec:refs}" in en),
        ("EN: no Chinese-only section body leaked", "本报告记述" not in en),
        ("EN: no literal placeholder left", "Placeholder" not in en),
        ("EN: no unresolved anchor", "%<<PART" not in en),

        # --- language markers must not reach either table of contents --------
        ("ZH TOC: no 中文/English heading",
         not MARKER_HEADING.search(toc_zh)),
        ("EN TOC: no 中文/English heading",
         not MARKER_HEADING.search(toc_en)),
        ("ZH TOC: has Chinese body sections",
         "引言" in toc_zh and "参考文献" in toc_zh),
        ("EN TOC: has Introduction and Architecture",
         "Introduction" in toc_en and "Architecture" in toc_en),
        ("EN TOC: no bare 'English' line", "English" not in toc_en),
        ("ZH TOC: no bare 'English' line", "English" not in toc_zh),
        ("ZH TOC: all numbers well formed", not bad_numbers(toc_zh)),
        ("EN TOC: all numbers well formed", not bad_numbers(toc_en)),
    ]


def generic_checks(tag: str, is_zh: bool, tex: str, toc: str) -> list[tuple[str, bool]]:
    """Checks any report's edition must satisfy."""
    out: list[tuple[str, bool]] = [
        (f"{tag}: document environment closed", r"\end{document}" in tex),
        (f"{tag} TOC: present", "\\contentsline" in toc),
        (f"{tag} TOC: no 中文/English heading", not MARKER_HEADING.search(toc)),
        (f"{tag} TOC: all numbers well formed", not bad_numbers(toc)),
    ]
    if is_zh:
        out.append((f"{tag}: Chinese prose present", cjk_count(prose(tex)) >= 200))
    else:
        out.append((f"{tag}: English prose kept", len(prose(tex)) > 2000))
        out.append((f"{tag}: no Chinese prose outside code",
                    cjk_count(prose(tex)) == 0))
        out.append((f"{tag}: no unresolved anchor", "%<<PART" not in tex))
    return out


def build_checks() -> list[tuple[str, bool]]:
    checks: list[tuple[str, bool]] = []
    for key, report in mktex.REPORTS.items():
        zh_path = report.zh_dir / (report.zh_name + ".tex")
        en_path = report.en_dir / (report.en_name + ".tex")
        if not zh_path.exists() and not en_path.exists():
            print(f"SKIP  {key}: neither edition has been generated yet "
                  f"({zh_path.name})")
            continue
        zh = read(zh_path)
        en = read(en_path)
        toc_zh = read(zh_path.with_suffix(".toc"))
        toc_en = read(en_path.with_suffix(".toc"))
        if key == "272m":
            checks += checks_272m(zh, en, toc_zh, toc_en)
        if zh_path.exists():
            checks += generic_checks(f"{key}/zh", True, zh, toc_zh)
        if en_path.exists():
            checks += generic_checks(f"{key}/en", False, en, toc_en)
    return checks


def main() -> int:
    checks = build_checks()
    bad = 0
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
        if not ok:
            bad += 1

    print()
    print(f"{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
