"""langcheck.py -- prove the language split from the PDFs themselves.

`verify.py` reads the generated LaTeX; this script reads the *PDF* text layer,
so it can catch something the .tex hides (a stray Chinese paragraph in the
English edition, or a whole English section that leaked into the Chinese one).

For every report configured in `mktex.REPORTS` whose PDFs exist:

  * Chinese edition -- extract the text, count CJK characters, and flag every
    line that looks like an English paragraph (12+ Latin words, no CJK).  A
    flagged line is only accepted if it is traceable to a code listing or an
    inline code span in the .tex, because code and its comments are part of the
    evidence and are allowed to be English.
  * English edition -- extract the text, and flag every line containing CJK.
    The same traceability rule applies: the accepted residue is code.

Exit status is non-zero if anything is untraceable, so the check is a gate and
not just a report.

Run from anywhere:  python docs/latex/langcheck.py [report-key ...]
"""
import re
import sys
from pathlib import Path

import mktex

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# a run of CJK characters, used to decide whether a CJK fragment in the English
# edition comes from a code span (a Chinese document filename, say)
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

LISTING_RE = re.compile(r"\\begin\{lstlisting\}(\[[^\]]*\])?(.*?)\\end\{lstlisting\}",
                        re.S)
CODE_RE = re.compile(r"\\(?:code|texttt|file)\{([^{}]*)\}")
VERB_RE = re.compile(r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}", re.S)
# block quotes are rendered as `notebox`; their content is quoted material
QUOTE_RE = re.compile(r"\\begin\{notebox\}(.*?)\\end\{notebox\}", re.S)


# XeLaTeX's TeX ligatures and the PDF text layer disagree with the source on a
# few characters (`...` comes back as an ellipsis, `'` as a curly quote, `fi` as
# a ligature glyph), so both sides are folded before comparing.
TYPO_FOLD = {
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2026": "...", "\u2013": "-", "\u2014": "--", "\u00a0": " ",
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def normalize(text: str) -> str:
    """Fold typography and collapse whitespace so extraction quirks do not matter."""
    for bad, good in TYPO_FOLD.items():
        text = text.replace(bad, good)
    return re.sub(r"\s+", "", text)


def code_strings(tex: str) -> list[str]:
    """Normalised contents of every listing, code span and file span."""
    out: list[str] = []
    for m in LISTING_RE.finditer(tex):
        out.append(normalize(m.group(2)))
    for m in VERB_RE.finditer(tex):
        out.append(normalize(m.group(1)))
    for m in CODE_RE.finditer(tex):
        out.append(normalize(m.group(1)))
    # 4, not 6: splitting a long path for wrapping can leave short tail spans
    # such as `间.md`, and a CJK fragment of one of those must still be
    # recognisable as code-derived
    return [c for c in out if len(c) >= 4]


def quoted_strings(tex: str) -> list[str]:
    """Normalised contents of every `notebox`, i.e. quoted source material.

    The Chinese edition quotes real model replies verbatim, and a reply to an
    English prompt comes back in English.  That is evidence, not the report's
    own prose, so it is exempt -- but it is counted separately so the exemption
    stays visible instead of silently widening the code rule.
    """
    out = [normalize(m.group(1)) for m in QUOTE_RE.finditer(tex)
           if "notebox" in m.group(0)]
    return [q for q in out if len(q) >= 6]


def cjk_fragments_from_code(line: str, codes: list[str]) -> bool:
    """True when every CJK fragment of a line occurs inside a code span.

    The English edition names Chinese document files inside `\\code{}`, and a
    long path is split across several code spans so that it can wrap; the PDF
    extractor can then hand back a fragment such as `...间.md` that no single
    code span contains verbatim.  Requiring *every* CJK run of the line to occur
    somewhere in a code span keeps that case exempt while still flagging real
    Chinese prose.
    """
    runs = CJK_RUN_RE.findall(line)
    if not runs:
        return False
    return all(any(run in c for c in codes) for run in runs)


def mixed_prose_strings(tex: str) -> list[str]:
    """Normalised source lines that mix Chinese with Latin letters.

    A sentence written in Chinese may quote an English reply or name an English
    artefact inline (`\u300c...\u300d` around a model output, say).  The PDF text layer breaks
    such a line wherever it likes, so the English half can surface as a line of
    its own.  Anything that occurs inside a mixed-source line is therefore an
    embedded quotation rather than a paragraph of its own; the count of such
    exemptions is reported.
    """
    out = []
    for line in tex.splitlines():
        if CJK_RE.search(line) and LATIN_WORD_RE.search(line):
            out.append(normalize(line))
    return out


def traceable(line: str, codes: list[str]) -> bool:
    """True when a PDF text line comes from code rather than from prose."""
    n = normalize(line)
    if len(n) < 4:
        return True
    for c in codes:
        # the extracted line may be a piece of a code line (or the other way
        # round, if the extractor glued two code lines together)
        if n in c or c in n:
            return True
        # long shared run: extraction can reorder a line's glyphs inside a
        # listing, so fall back to a 16-character common substring
        if len(n) >= 16:
            for i in range(0, len(n) - 16 + 1, 4):
                if n[i:i + 16] in c:
                    return True
    return False


def english_candidates(text: str) -> list[tuple[int, str]]:
    """(raw line index, text) of lines that look like an English paragraph."""
    out = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line or CJK_RE.search(line):
            continue
        if len(LATIN_WORD_RE.findall(line)) >= 12:
            out.append((i, line))
    return out


# A Chinese report legitimately carries two kinds of English text: the
# bibliography (citations keep their original language, and that is the
# convention) and display mathematics.  Both are exempted explicitly and
# counted in the output, so the exemption stays auditable.
BIB_RE = re.compile(r"^\s*(?:\[\d+\]|arXiv:|https?://|doi:)")
MATH_CHARS = set("=&^_{}\\|")


def is_math_like(line: str) -> bool:
    words = LATIN_WORD_RE.findall(line)
    if not words:
        return True
    if any(ch in MATH_CHARS for ch in line):
        return True
    return sum(len(w) for w in words) / len(words) < 3.0


def is_bib_like(line: str) -> bool:
    return BIB_RE.match(line) is not None


def check_pdf(path: Path, tex_path: Path, is_zh: bool) -> tuple[bool, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = len(reader.pages)
    text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
    tex_source = tex_path.read_text(encoding="utf-8")
    codes = code_strings(tex_source)
    quotes = quoted_strings(tex_source)
    mixed = mixed_prose_strings(tex_source)
    cjk = len(CJK_RE.findall(text))

    lines = [
        f"    pages      : {pages}",
        f"    bytes      : {path.stat().st_size:,}",
        f"    chars      : {len(text):,}",
        f"    CJK chars  : {cjk}",
    ]
    ok = True
    if is_zh:
        cands = english_candidates(text)
        bib_lines = [i for i, l in cands if is_bib_like(l)]
        checked, exempt_math, exempt_bib = [], 0, 0
        for i, line in cands:
            if is_math_like(line):
                exempt_math += 1
                continue
            if is_bib_like(line) or any(0 <= i - b <= 3 for b in bib_lines):
                exempt_bib += 1
                continue
            checked.append(line)
        leaks, exempt_quote, exempt_mixed = [], 0, 0
        for c in checked:
            if traceable(c, codes):
                continue
            if traceable(c, quotes):
                exempt_quote += 1
                continue
            if traceable(c, mixed):
                exempt_mixed += 1
                continue
            leaks.append(c)
        lines.append(f"    English-paragraph candidates : {len(cands)}")
        lines.append(f"      exempt (bibliography)        : {exempt_bib}")
        lines.append(f"      exempt (mathematics)         : {exempt_math}")
        lines.append(f"      exempt (quoted material)     : {exempt_quote}")
        lines.append(f"      exempt (inside a Chinese line): {exempt_mixed}")
        lines.append(f"      checked as prose             : {len(checked)}"
                     f"  (untraceable: {len(leaks)})")
        for c in leaks[:10]:
            lines.append(f"      LEAK  {c[:110]}")
        ok = not leaks
        if pages <= 8:
            lines.append(f"    WARNING: only {pages} pages")
    else:
        cjk_lines = [l for l in text.splitlines() if CJK_RE.search(l)]
        leaks = [l for l in cjk_lines
                 if not traceable(l, codes) and not traceable(l, quotes)
                 and not cjk_fragments_from_code(l, codes)]
        lines.append(f"    lines with CJK : {len(cjk_lines)} "
                     f"(untraceable: {len(leaks)})")
        for c in leaks[:10]:
            lines.append(f"      LEAK  {c.strip()[:110]}")
        ok = not leaks
        if pages <= 8:
            lines.append(f"    WARNING: only {pages} pages")
    return ok, "\n".join(lines)


def main(argv: list[str]) -> int:
    # extracted PDF text contains ligatures and CJK; the Windows console default
    # code page cannot encode either, so pin stdout to UTF-8 with replacement
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    keys = argv or [k for k, r in mktex.REPORTS.items() if r.default]
    bad = 0
    checked = 0
    for key in keys:
        report = mktex.REPORTS[key]
        specs = [
            (report.zh_dir / (report.zh_name + ".pdf"),
             report.zh_dir / (report.zh_name + ".tex"), True),
            (report.en_dir / (report.en_name + ".pdf"),
             report.en_dir / (report.en_name + ".tex"), False),
        ]
        for pdf, tex, is_zh in specs:
            if not pdf.exists() or not tex.exists():
                print(f"=== {report.key}: {pdf.name} not built yet -- skipped")
                continue
            ok, detail = check_pdf(pdf, tex, is_zh)
            print(f"=== {pdf}")
            print(detail)
            print(f"    verdict    : {'PASS' if ok else 'FAIL'}")
            print()
            checked += 1
            if not ok:
                bad += 1
    print(f"{checked - bad}/{checked} PDFs pass the language check")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
