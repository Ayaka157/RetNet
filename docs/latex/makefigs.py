#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
makefigs.py -- generate the report's figures as vector PDFs.

Deliberately dependency-free: matplotlib is not installed in this workspace
and is not required here, so the figures are emitted as raw PDF drawing
operators (lines, filled paths, Helvetica text).  That keeps the build
deterministic and self-contained, and the output is vector, not raster.

Every number plotted is read from the repository's own artefacts:

  * `checkpoints/retnet_272m_v14/checkpoint-10000/trainer_state.json`
        -> training loss and learning-rate schedule
  * `logs/bench_v3/report.T0.7_s1.json` and `report.c10k.json`
        -> per-category benchmark comparison (v13 vs v14)
  * `checkpoints/retnet_500m_v15b/checkpoint-67390/trainer_state.json`,
    `checkpoints/retnet_500m_v15b_ep2/trainer_state.json`,
    `checkpoints/retnet_500m_v15b_ep3/trainer_state.json`
        -> RetNet-0.5B validation loss of the three training legs
  * values measured in `docs/19_RetNet-272M_技术报告.md`
        -> step time, kernel attribution, GEMM shape ceiling,
           budget-truncation split

Usage:  python makefigs.py          # writes figures/*.pdf
"""

from __future__ import annotations

import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT = ROOT / "figures"

# ---------------------------------------------------------------------------
# Helvetica advance widths (AFM units, /1000 em) -- needed to centre text.
# ---------------------------------------------------------------------------
WIDTHS = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667,
    "'": 191, "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333,
    ".": 278, "/": 278, "0": 556, "1": 556, "2": 556, "3": 556, "4": 556,
    "5": 556, "6": 556, "7": 556, "8": 556, "9": 556, ":": 278, ";": 278,
    "<": 584, "=": 584, ">": 584, "?": 556, "@": 1015, "A": 667, "B": 667,
    "C": 722, "D": 722, "E": 667, "F": 611, "G": 778, "H": 722, "I": 278,
    "J": 500, "K": 667, "L": 556, "M": 833, "N": 722, "O": 778, "P": 667,
    "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944,
    "X": 667, "Y": 667, "Z": 611, "[": 278, "\\": 278, "]": 278, "^": 469,
    "_": 556, "`": 333, "a": 556, "b": 556, "c": 500, "d": 556, "e": 556,
    "f": 278, "g": 556, "h": 556, "i": 222, "j": 222, "k": 500, "l": 222,
    "m": 833, "n": 556, "o": 556, "p": 556, "q": 556, "r": 333, "s": 500,
    "t": 278, "u": 556, "v": 500, "w": 722, "x": 500, "y": 500, "z": 500,
    "{": 334, "|": 260, "}": 334, "~": 584,
}
BOLD_FACTOR = 1.06

# Helvetica-Bold advance widths differ from Helvetica for many glyphs (notably
# digits, which drop from 556 to 556 but letters such as 'e' rise).  Using the
# regular table with a fudge factor over-measures some strings and under-measures
# others, which shows up as text spilling out of its legend box.
BOLD_WIDTHS = {
    " ": 278, "!": 333, '"': 474, "#": 556, "$": 556, "%": 889, "&": 722,
    "'": 238, "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333,
    ".": 278, "/": 278, "0": 556, "1": 556, "2": 556, "3": 556, "4": 556,
    "5": 556, "6": 556, "7": 556, "8": 556, "9": 556, ":": 333, ";": 333,
    "<": 584, "=": 584, ">": 584, "?": 611, "@": 975, "A": 722, "B": 722,
    "C": 722, "D": 722, "E": 667, "F": 611, "G": 778, "H": 722, "I": 278,
    "J": 556, "K": 722, "L": 611, "M": 833, "N": 722, "O": 778, "P": 667,
    "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944,
    "X": 667, "Y": 667, "Z": 611, "[": 333, "\\": 278, "]": 333, "^": 584,
    "_": 556, "`": 333, "a": 556, "b": 611, "c": 556, "d": 611, "e": 556,
    "f": 333, "g": 611, "h": 611, "i": 278, "j": 278, "k": 556, "l": 278,
    "m": 889, "n": 611, "o": 611, "p": 611, "q": 611, "r": 389, "s": 556,
    "t": 333, "u": 611, "v": 556, "w": 778, "x": 556, "y": 556, "z": 500,
    "{": 389, "|": 280, "}": 389, "~": 584,
}


def text_width(s: str, size: float, bold: bool = False) -> float:
    table = BOLD_WIDTHS if bold else WIDTHS
    return sum(table.get(c, 556) for c in s) / 1000.0 * size


def esc(s: str) -> str:
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


class Canvas:
    """A minimal PDF drawing surface (PDF user space, origin bottom-left)."""

    def __init__(self, width: float, height: float):
        self.w = width
        self.h = height
        self.ops: list[str] = []

    def line(self, x1, y1, x2, y2, color=(0, 0, 0), lw=0.6, dash=None):
        d = f"[{dash[0]} {dash[1]}] 0 d\n" if dash else "[] 0 d\n"
        self.ops.append(
            f"{d}{lw:.2f} w {color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG "
            f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n"
        )

    def rect(self, x, y, w, h, fill=None, stroke=None, lw=0.6):
        op = ""
        if fill:
            op += (f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg "
                   f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re f\n")
        if stroke:
            op += (f"{lw:.2f} w {stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG "
                   f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re S\n")
        self.ops.append(op)

    def text(self, x, y, s, size=8.0, bold=False, color=(0, 0, 0),
             align="left", rotate=0.0):
        if not s:
            return
        if align == "center":
            x -= text_width(s, size, bold) / 2.0
        elif align == "right":
            x -= text_width(s, size, bold)
        font = "F2" if bold else "F1"
        if rotate:
            c, sn = math.cos(rotate), math.sin(rotate)
            tm = f"{c:.5f} {sn:.5f} {-sn:.5f} {c:.5f} {x:.2f} {y:.2f} Tm"
        else:
            tm = f"1 0 0 1 {x:.2f} {y:.2f} Tm"
        self.ops.append(
            f"BT {color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg /{font} "
            f"{size:.2f} Tf {tm} ({esc(s)}) Tj ET\n"
        )

    def polyline(self, pts, color=(0, 0, 0), lw=1.0):
        if len(pts) < 2:
            return
        body = f"{pts[0][0]:.2f} {pts[0][1]:.2f} m " + " ".join(
            f"{x:.2f} {y:.2f} l" for x, y in pts[1:])
        self.ops.append(
            f"{lw:.2f} w {color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG {body} S\n")

    def dot(self, x, y, r, color=(0, 0, 0)):
        k = 0.5523 * r
        self.ops.append(
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg {x + r:.2f} {y:.2f} m "
            f"{x + r:.2f} {y + k:.2f} {x + k:.2f} {y + r:.2f} {x:.2f} {y + r:.2f} c "
            f"{x - k:.2f} {y + r:.2f} {x - r:.2f} {y + k:.2f} {x - r:.2f} {y:.2f} c "
            f"{x - r:.2f} {y - k:.2f} {x - k:.2f} {y - r:.2f} {x:.2f} {y - r:.2f} c "
            f"{x + k:.2f} {y - r:.2f} {x + r:.2f} {y - k:.2f} {x + r:.2f} {y:.2f} c f\n")

    def save(self, path: pathlib.Path):
        content = "".join(self.ops)
        objs = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.w:.0f} {self.h:.0f}] "
             "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>"),
            f"<< /Length {len(content)} >>\nstream\n{content}endstream",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            "/Encoding /WinAnsiEncoding >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
            "/Encoding /WinAnsiEncoding >>",
        ]
        buf = bytearray(b"%PDF-1.4\n")
        offsets = []
        for i, o in enumerate(objs, start=1):
            offsets.append(len(buf))
            buf += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1")
        xref = len(buf)
        buf += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin-1")
        for off in offsets:
            buf += f"{off:010d} 00000 n \n".encode("latin-1")
        buf += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode("latin-1")
        path.write_bytes(bytes(buf))


# ---------------------------------------------------------------------------
INK = (0.13, 0.13, 0.15)
MUTED = (0.42, 0.42, 0.46)
GRID = (0.86, 0.87, 0.89)
BLUE = (0.20, 0.35, 0.60)
RED = (0.68, 0.25, 0.22)
GREEN = (0.24, 0.50, 0.35)
ORANGE = (0.80, 0.52, 0.20)
GREY = (0.62, 0.64, 0.67)
LIGHT = (0.60, 0.67, 0.78)
WHITE = (1, 1, 1)


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** int(math.floor(math.log10(raw))) if raw > 0 else 1
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.25:
            ticks.append(round(v, 10))
        v += step
    return ticks


def plot_area(cv, x0, y0, x1, y1, xlo, xhi, ylo, yhi,
              xlab, ylab, xticks=None, yticks=None,
              xfmt="{:.0f}", yfmt="{:.0f}", numeric_x=True, numeric_y=True):
    """Framed plot area with gridlines and tick labels."""
    def X(v):
        return x0 + (v - xlo) / (xhi - xlo) * (x1 - x0)

    def Y(v):
        return y0 + (v - ylo) / (yhi - ylo) * (y1 - y0)

    for t in (yticks if yticks is not None else nice_ticks(ylo, yhi)):
        yy = Y(t)
        cv.line(x0, yy, x1, yy, color=GRID, lw=0.4)
        if numeric_y:
            cv.text(x0 - 4, yy - 2.6, yfmt.format(t), size=7, color=MUTED,
                    align="right")
    if numeric_x:
        for t in (xticks if xticks is not None else nice_ticks(xlo, xhi)):
            xx = X(t)
            cv.line(xx, y0, xx, y0 - 2.2, color=MUTED, lw=0.5)
            cv.text(xx, y0 - 11.5, xfmt.format(t), size=7, color=MUTED,
                    align="center")
    cv.rect(x0, y0, x1 - x0, y1 - y0, stroke=INK, lw=0.6)
    if xlab:
        cv.text((x0 + x1) / 2, y0 - 24, xlab, size=7.5, color=INK, align="center")
    if ylab:
        cv.text(x0 - 42, (y0 + y1) / 2, ylab, size=7.5, color=INK,
                align="center", rotate=1.5708)
    return X, Y


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_history():
    p = (REPO / "checkpoints" / "retnet_272m_v14" / "checkpoint-10000"
         / "trainer_state.json")
    state = json.loads(p.read_text(encoding="utf-8"))
    pts = [(h["step"], h["loss"], h["learning_rate"])
           for h in state["log_history"]
           if "loss" in h and "learning_rate" in h]
    pts.sort()
    return pts


CAT_EN = {
    "角色扮演": "Role-play", "多轮": "Multi-turn", "一致性": "Consistency",
    "指令遵循": "Instruction", "结构化": "Structured", "鲁棒性": "Robustness",
    "双语": "Bilingual", "安全": "Safety", "填空": "Cloze", "问答": "QA",
    "语言": "Language", "续写": "Continuation", "长上下文": "Long-context",
    "编码": "Code", "数学": "Math",
}


def load_bench(rel):
    j = json.loads((REPO / rel).read_text(encoding="utf-8"))
    name = list(j["models"])[0]
    return j["models"][name]


# ---------------------------------------------------------------------------
def fig_loss():
    pts = load_history()
    cv = Canvas(430, 250)
    x0, y0, x1, y1 = 54, 48, 402, 206
    losses = [p[1] for p in pts]
    X, Y = plot_area(cv, x0, y0, x1, y1, pts[0][0], pts[-1][0], 2.05, 2.72,
                     "Optimization step", "Loss",
                     yticks=[2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7], yfmt="{:.1f}")
    cv.polyline([(X(s), Y(l)) for s, l, _ in pts], color=LIGHT, lw=0.4)
    win, ma = 25, []
    for i in range(len(pts)):
        chunk = losses[max(0, i - win + 1):i + 1]
        ma.append((X(pts[i][0]), Y(sum(chunk) / len(chunk))))
    cv.polyline(ma, color=BLUE, lw=1.3)

    # legend inside the axes, top-right, on an opaque patch
    lg = "v14 leg 1 loss (mean, w=25)"
    w = text_width(lg, 7) + 18
    cv.rect(x1 - w - 3, y1 - 15, w, 12, fill=WHITE)
    cv.line(x1 - w + 1, y1 - 9, x1 - w + 11, y1 - 9, color=BLUE, lw=1.3)
    cv.text(x1 - 6, y1 - 11.6, lg, size=7, color=INK, align="right")
    cv.text(x0 + 5, Y(2.70) - 2, f"first  {losses[0]:.4f}", size=6.8, color=MUTED)
    cv.text(x0 + 5, Y(2.66) - 2, f"step {pts[-1][0]:,}  {losses[-1]:.4f}",
            size=6.8, color=MUTED)
    cv.save(OUT / "fig-loss.pdf")


def fig_lr():
    pts = load_history()
    cv = Canvas(430, 190)
    x0, y0, x1, y1 = 66, 44, 402, 156
    lrs = [p[2] for p in pts]
    X, Y = plot_area(cv, x0, y0, x1, y1, pts[0][0], pts[-1][0],
                     min(lrs) * 0.97, max(lrs) * 1.03,
                     "Optimization step", "Learning rate",
                     yticks=[1e-6, 1.5e-6, 2e-6, 2.5e-6, 3e-6, 3.5e-6],
                     yfmt="{:.1e}")
    cv.polyline([(X(s), Y(l)) for s, _, l in pts], color=RED, lw=1.2)
    cv.text(x0 + 5, y1 - 13,
            f"peak {max(lrs):.2e} at warmup end, then cosine to min_lr",
            size=6.8, color=MUTED)
    cv.save(OUT / "fig-lr.pdf")


def fig_steps():
    labels = ["A", "B", "C", "C'"]
    subs = ["reference", "+compile", "+8-bit AdamW", "+closed-form"]
    ms = [2176, 903, 713, 662]
    tok = [3764, 8814, 11496, 12373]
    cv = Canvas(430, 200)
    x0, y0, x1, y1 = 58, 54, 396, 156
    X, Y = plot_area(cv, x0, y0, x1, y1, -0.5, 3.5, 0, 2500,
                     "", "ms per optimization step",
                     yticks=[0, 500, 1000, 1500, 2000, 2500], numeric_x=False)
    for i, m in enumerate(ms):
        col = GREEN if i == 3 else BLUE
        bx0, bx1 = X(i - 0.28), X(i + 0.28)
        cv.rect(bx0, y0, bx1 - bx0, Y(m) - y0, fill=col)
        cv.text((bx0 + bx1) / 2, Y(m) + 16, f"{m} ms", size=7.2, bold=True,
                color=INK, align="center")
        cv.text((bx0 + bx1) / 2, Y(m) + 6, f"{tok[i]:,} tok/s", size=6.4,
                color=MUTED, align="center")
        cv.text(X(i), y0 - 12, labels[i], size=7.6, bold=True, color=INK,
                align="center")
        cv.text(X(i), y0 - 20, subs[i], size=6.4, color=MUTED, align="center")
    cv.text(x1, y1 + 30, "3.29x end to end, AC power, real Trainer path",
            size=7.4, color=INK, align="right")
    cv.text(x1, y1 + 18, "measured 657 / 662 / 663 ms on repeats",
            size=6.8, color=MUTED, align="right")
    cv.save(OUT / "fig-steps.pdf")


def fig_kernels():
    parts = [("GEMM", 66.4, BLUE), ("Triton elementwise", 25.2, ORANGE),
             ("Triton reduction", 4.2, GREEN), ("elementwise", 4.2, GREY)]
    cv = Canvas(430, 150)
    x0, x1, y0, y1 = 40, 390, 44, 96
    cv.rect(x0, y0, x1 - x0, y1 - y0, stroke=INK, lw=0.6)
    acc = 0.0
    for name, pct, col in parts:
        w = (x1 - x0) * pct / 100.0
        cv.rect(x0 + acc, y0, w, y1 - y0, fill=col)
        if pct >= 20:
            cv.text(x0 + acc + w / 2, y0 + 11, f"{name}  {pct}%", size=7.2,
                    bold=True, color=WHITE, align="center")
        acc += w
    lx = x0
    for name, pct, col in parts:
        if pct >= 20:
            continue
        cv.rect(lx, y0 - 16, 7, 7, fill=col)
        lab = f"{name}  {pct}%"
        cv.text(lx + 10, y0 - 15.4, lab, size=7, color=INK)
        lx += 10 + text_width(lab, 7) + 16
    cv.text(x0, y1 + 26, "470 ms GPU busy over 3 micro-steps (166 ms wall)",
            size=7.6, bold=True, color=INK)
    cv.text(x0, y1 + 14, "94-95% occupancy; the GEMM half runs at its shape ceiling",
            size=7, color=MUTED)
    cv.save(OUT / "fig-kernels.pdf")


def fig_gemm():
    # (label, invocations, achieved TFLOPs, family)
    pts = [
        ("2048x1024x2048", 160, 26.5, "mm"),
        ("2048x2048x1024", 100, 33.4, "mm"),
        ("2048x1024x1024", 80, 31.6, "mm"),
        ("1024x2048x2048", 40, 37.1, "mm"),
        ("1024x1024x2048", 40, 33.9, "mm"),
        ("LM head (fwd+bwd)", 3, 41.0, "mm"),
        ("512x64x128x64", 120, 6.7, "bmm"),
        ("512x64x64x64", 60, 6.8, "bmm"),
        ("512x64x64x128", 60, 8.8, "bmm"),
    ]
    cv = Canvas(430, 250)
    x0, y0, x1, y1 = 62, 70, 398, 190
    X, Y = plot_area(cv, x0, y0, x1, y1, 0, 175, 0, 46,
                     "Invocations of the shape in the compiled graph",
                     "Achieved TFLOPs",
                     xticks=[0, 25, 50, 75, 100, 125, 150, 175],
                     yticks=[0, 10, 20, 30, 40])
    for val, col, lab in ((33.2, RED, "achieved 33.2 TFLOPs overall"),
                          (27.7, GREEN, "isolated ceiling 27.7 (weighted)")):
        cv.line(x0, Y(val), x1, Y(val), color=col, lw=1.0, dash=(3, 2))
        cv.text(x1 - 3, Y(val) + 4, lab, size=6.6, color=col, align="right")
    # alternate the labels above and below the markers so dense clusters read
    for k, (name, cnt, fl, fam) in enumerate(pts):
        col = BLUE if fam == "mm" else ORANGE
        cv.dot(X(cnt), Y(fl), 2.6, color=col)
        up = (k % 2 == 0)
        cv.text(X(cnt), Y(fl) + (7 if up else -11), name, size=5.9,
                color=MUTED, align="center")
    cv.rect(x0, y0 - 40, 7, 7, fill=BLUE)
    cv.text(x0 + 10, y0 - 39.4, "aten::mm", size=7, color=INK)
    cv.rect(x0 + 64, y0 - 40, 7, 7, fill=ORANGE)
    cv.text(x0 + 74, y0 - 39.4, "aten::bmm (batch = B*H*n_c)", size=7, color=INK)
    cv.save(OUT / "fig-gemm.pdf")


def fig_bench():
    v13 = load_bench("logs/bench_v3/report.T0.7_s1.json")
    v14 = load_bench("logs/bench_v3/report.c10k.json")
    c13, c14 = v13["cats"], v14["cats"]
    rows = sorted(((CAT_EN.get(k, k), c13[k]["score"], c14[k]["score"])
                   for k in c13), key=lambda r: -(r[1] + r[2]))
    n = len(rows)
    cv = Canvas(430, 340)
    x0, y0, x1, y1 = 86, 62, 396, 286
    X, Y = plot_area(cv, x0, y0, x1, y1, 0, 0.26, -0.5, n - 0.5,
                     "", "",
                     xticks=[0, 0.05, 0.10, 0.15, 0.20, 0.25], xfmt="{:.2f}",
                     numeric_y=False)
    for i, (name, s13, s14) in enumerate(rows):
        yy = Y(i)
        cv.rect(x0, yy + 0.06, X(s13) - x0, 0.32, fill=GREY)
        cv.rect(x0, yy - 0.38, X(s14) - x0, 0.32, fill=BLUE)
        cv.text(x0 - 5, yy - 2.4, name, size=6.8, color=INK, align="right")
    # x label and legend are stacked below the frame so nothing overlaps
    cv.text((x0 + x1) / 2, y0 - 14, "Weighted category score", size=7.5,
            color=INK, align="center")
    cv.rect(x0, y0 - 32, 7, 7, fill=GREY)
    cv.text(x0 + 10, y0 - 31.4, "v13  0.1049", size=7, color=INK)
    cv.rect(x0 + 76, y0 - 32, 7, 7, fill=BLUE)
    cv.text(x0 + 86, y0 - 31.4, "v14 checkpoint-10000  0.0829", size=7, color=INK)
    cv.save(OUT / "fig-bench-cats.pdf")


def load_05b_legs():
    """The three 0.5B training legs, straight from their `trainer_state.json`.

    Returns [(label, start M tokens, end M tokens, train, evals)] where

      * `train` is every logging step as (cumulative M tokens, loss, learning
        rate) -- 3369 points for ep1, 842 for ep2, 842 for ep3;
      * `evals` is every evaluation as (cumulative M tokens, eval_loss) -- 13,
        8 and 8 points respectively.

    Each leg restarts a full cosine schedule, so the x axis is *cumulative
    training tokens*: the per-leg step counter resets to zero at every restart
    and would otherwise draw three curves on top of each other.
    """
    legs = [
        ("ep1 (RTX 3090 24 GB)",
         "checkpoints/retnet_500m_v15b/checkpoint-67390/trainer_state.json", 8192, 67390),
        ("ep2 (RTX PRO 6000)",
         "checkpoints/retnet_500m_v15b_ep2/trainer_state.json", 32768, 16848),
        ("ep3 (RTX PRO 6000, online)",
         "checkpoints/retnet_500m_v15b_ep3/trainer_state.json", 32768, 16848),
    ]
    out = []
    offset = 0.0
    for name, rel, tok_per_step, max_steps in legs:
        state = json.loads((REPO / rel).read_text(encoding="utf-8"))
        train = sorted((h["step"], h["loss"], h["learning_rate"])
                       for h in state["log_history"]
                       if "loss" in h and "learning_rate" in h)
        evals = sorted((h["step"], h["eval_loss"])
                       for h in state["log_history"] if "eval_loss" in h)
        start = offset
        end = offset + max_steps * tok_per_step / 1e6
        out.append((
            name, start, end,
            [(offset + s * tok_per_step / 1e6, l, lr) for s, l, lr in train],
            [(offset + s * tok_per_step / 1e6, l) for s, l in evals],
        ))
        offset = end
    return out


def fig_0p5b_loss():
    """Per-step training loss and the evaluation points of the three 0.5B legs.

    The thin light line is every logged step; the coloured line is a 25-step
    rolling mean of it; the dots are the evaluations.  Both sets of points come
    from the three real `trainer_state.json` files.
    """
    legs = load_05b_legs()
    cv = Canvas(430, 312)
    x0, y0, x1, y1 = 54, 86, 402, 238
    X, Y = plot_area(cv, x0, y0, x1, y1, 0, 1700, 1.75, 2.90,
                     "Cumulative training tokens (M)", "Training loss",
                     xticks=[0, 250, 500, 750, 1000, 1250, 1500],
                     yticks=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8], yfmt="{:.1f}")
    cols = [BLUE, ORANGE, GREEN]
    for i, (name, start, end, train, evals) in enumerate(legs):
        if i:
            cv.line(X(start), y0, X(start), y1, color=GREY, lw=0.5, dash=(2, 2))
        cv.polyline([(X(t), Y(l)) for t, l, _ in train], color=LIGHT, lw=0.25)
        mean, win = [], 25
        for k in range(len(train)):
            chunk = [p[1] for p in train[max(0, k - win + 1):k + 1]]
            mean.append((X(train[k][0]), Y(sum(chunk) / len(chunk))))
        cv.polyline(mean, color=cols[i], lw=1.1)
        for t, l in evals:
            cv.dot(X(t), Y(l), 1.7, color=cols[i])
        # the best evaluation of each leg, called out in the empty top-right
        t, l = min(evals, key=lambda p: p[1])
        cv.text(X(t) + 4, Y(l) + 3, f"{l:.4f}", size=6.6, color=cols[i])
    # the two hot restarts: last eval of the previous leg -> first of the next
    jumps = []
    for i in range(1, len(legs)):
        prev, cur = legs[i - 1][4][-1], legs[i][4][0]
        jumps.append((cur[0], cur[1] - prev[1]))
        cv.line(X(cur[0]), Y(prev[1]), X(cur[0]), Y(cur[1]), color=RED, lw=0.8)
    cv.text(x1 - 4, y1 - 11,
            f"hot restart +{jumps[0][1]:.4f} at {jumps[0][0]:.0f} M tokens",
            size=6.8, color=RED, align="right")
    cv.text(x1 - 4, y1 - 21,
            f"hot restart +{jumps[1][1]:.4f} at {jumps[1][0]:.0f} M tokens",
            size=6.8, color=RED, align="right")
    cv.text(x0 + 3, y1 - 11, "thin: every logged step   thick: 25-step mean",
            size=6.6, color=MUTED)
    cv.text(x0 + 3, y1 - 21, "dots: evaluations", size=6.6, color=MUTED)
    # two legend rows so the third leg's label stays inside the frame width
    rows = [(legs[:2], cols[:2]), (legs[2:], cols[2:])]
    for r, (row, rowcols) in enumerate(rows):
        lx = x0
        ly = y0 - 42 - r * 13
        for (name, _, _, _, evals), col in zip(row, rowcols):
            cv.rect(lx, ly, 7, 7, fill=col)
            label = f"{name}  best {min(l for _, l in evals):.4f}"
            cv.text(lx + 10, ly + 0.6, label, size=6.9, color=INK)
            lx += 12 + text_width(label, 6.9) + 16
    print("    points: " + ", ".join(
        f"{l[0].split()[0]} {len(l[3])} train / {len(l[4])} eval" for l in legs))
    cv.save(OUT / "fig-0p5b-loss.pdf")


def fig_0p5b_lr():
    """Learning rate of the three 0.5B legs: three restarts of one cosine."""
    legs = load_05b_legs()
    cv = Canvas(430, 250)
    x0, y0, x1, y1 = 62, 58, 402, 178
    X, Y = plot_area(cv, x0, y0, x1, y1, 0, 1700, 0, 1.25e-4,
                     "Cumulative training tokens (M)", "Learning rate",
                     xticks=[0, 250, 500, 750, 1000, 1250, 1500],
                     yticks=[0, 2.5e-5, 5e-5, 7.5e-5, 1e-4], yfmt="{:.1e}")
    cols = [BLUE, ORANGE, GREEN]
    for i, (name, start, end, train, _) in enumerate(legs):
        cv.polyline([(X(t), Y(lr)) for t, _, lr in train], color=cols[i], lw=0.9)
        if i:
            cv.line(X(start), y0, X(start), y1, color=GREY, lw=0.5, dash=(2, 2))
        # the leg's own label, placed where that leg's curve is still flat
        cv.text(X(start) + 6, Y(1.16e-4), name.split(" (")[0] + " peak",
                size=6.6, color=cols[i])
    cv.text(x0 + 3, y1 + 12,
            "every leg restarts a full cosine: peak 1.1e-4, 500 warm-up steps, "
            "decay to min_lr 1.1e-5",
            size=6.8, color=INK)
    cv.text(x0 + 3, y1 + 3,
            "horizontal gap between restarts = one epoch of the same corpus",
            size=6.6, color=MUTED)
    cv.save(OUT / "fig-0p5b-lr.pdf")


def fig_0p5b_step():
    """Where one 575 ms optimization step goes (measured kernel bucketing)."""
    parts = [("GEMM (cuBLAS, fwd+bwd)", 354.6, BLUE),
             ("Triton elementwise", 160.5, ORANGE),
             ("Triton reductions", 48.5, GREEN),
             ("other (clip, optimizer step)", 11.6, GREY)]
    total = sum(ms for _, ms, _ in parts)
    cv = Canvas(430, 168)
    x0, x1, y0, y1 = 40, 390, 62, 114
    cv.rect(x0, y0, x1 - x0, y1 - y0, stroke=INK, lw=0.6)
    acc = 0.0
    for name, ms, col in parts:
        w = (x1 - x0) * ms / total
        cv.rect(x0 + acc, y0, w, y1 - y0, fill=col)
        if ms >= 45:
            cv.text(x0 + acc + w / 2, y0 + 30,
                    f"{ms:.1f} ms", size=7.4, bold=True, color=WHITE,
                    align="center")
            cv.text(x0 + acc + w / 2, y0 + 19,
                    f"{100 * ms / total:.1f}%", size=6.8, color=WHITE,
                    align="center")
        acc += w
    lx = x0
    for name, ms, col in parts:
        if ms >= 45:
            continue
        cv.rect(lx, y0 - 18, 7, 7, fill=col)
        lab = f"{name}  {ms:.1f} ms"
        cv.text(lx + 10, y0 - 17.4, lab, size=6.9, color=INK)
        lx += 12 + text_width(lab, 6.9) + 16
    cv.text(x0, y1 + 34,
            f"{total:.0f} ms steady-state optimization step", size=7.8,
            bold=True, color=INK)
    cv.text(x0, y1 + 23,
            "RTX PRO 6000, batch 16, seq 2048, chunk 128, bf16; "
            "SM occupancy 81-100%",
            size=6.7, color=MUTED)
    cv.text(x0, y1 + 12,
            "GEMM and elementwise are measured by kernel name (docs/25 SS3); "
            "22-24 ms (3.8-4.3%) is still recoverable",
            size=6.7, color=MUTED)
    cv.save(OUT / "fig-0p5b-step.pdf")


def fig_truncation():
    cv = Canvas(430, 205)
    x0, y0, x1, y1 = 62, 56, 396, 158
    X, Y = plot_area(cv, x0, y0, x1, y1, -0.5, 1.5, 0, 100,
                     "", "Percent of single-turn items",
                     yticks=[0, 20, 40, 60, 80, 100], numeric_x=False)
    groups = [("v13", 740, 280, 72.5, 0.0),
              ("v14 checkpoint-10000", 665, 355, 65.2, 0.6)]
    for i, (name, trunc, clean, pct, clean_leak) in enumerate(groups):
        bx0, bx1 = X(i - 0.22), X(i + 0.22)
        cv.rect(bx0, y0, bx1 - bx0, Y(pct) - y0, fill=RED)
        cv.text((bx0 + bx1) / 2, Y(pct) + 5, f"{pct}%", size=7.6, bold=True,
                color=INK, align="center")
        cv.text(X(i), y0 - 12, name, size=7.2, color=INK, align="center")
        cv.text(X(i), y0 - 20, f"{trunc} truncated / {clean} clean", size=6.4,
                color=MUTED, align="center")
        cv.text(X(i), y0 - 28, f"clean leak {clean_leak}%", size=6.4,
                color=GREEN, align="center")
    cv.text(x0, y1 + 32, "Budget truncation, not model behaviour, drives the leak",
            size=7.6, bold=True, color=INK)
    cv.text(x0, y1 + 20,
            "raw leak 52.0% -> 46.8%, but never-truncated items leak ~0%",
            size=7, color=MUTED)
    cv.save(OUT / "fig-truncation.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.pdf"):
        f.unlink()
    fig_loss()
    fig_lr()
    fig_steps()
    fig_kernels()
    fig_gemm()
    fig_bench()
    fig_truncation()
    fig_0p5b_loss()
    fig_0p5b_lr()
    fig_0p5b_step()
    for p in sorted(OUT.glob("*.pdf")):
        print(f"  {p.name:26} {p.stat().st_size:>7,} bytes")


if __name__ == "__main__":
    main()
