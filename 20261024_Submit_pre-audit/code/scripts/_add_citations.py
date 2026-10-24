"""给 docs/00 与 docs/02 加文内引用（按段），并在末尾附参考文献表。

01 与 00F1 由各自的生成器加引用，这个脚本只管手写文档，避免两套逻辑打架。
重复执行是幂等的：已有"（参 …）"的段落不会再加。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _refs import refs_md  # noqa: E402

# 00：每个 E/N 小节末尾（以 --- 结束）加一行引用
DOC00 = ROOT / "docs" / "00_错误总表.md"
MAP00 = {
    "E01": ["07", "08", "09", "10"],
    "E02": ["08", "09", "13"],
    "E03": ["07", "09"],
    "E04": ["10", "11"],
    "E05": ["08", "13"],
    "E09/E10/E11": ["16"],
    "E12": ["16"],
    "E16": ["01", "04"],
    "N01": ["03"],
    "N02": ["03"],
}

# 02：每个小节末尾（下一个同级或更高级标题之前）加一行引用
DOC02 = ROOT / "docs" / "02_续训恢复计划_220元.md"
MAP02 = [
    ("## 0.", ["01", "03"]),
    ("## 2.", ["07", "08", "09", "10", "13"]),
    ("### 3.1", ["16", "17"]),
    ("## 4.", ["15"]),
    ("## 5.", ["15"]),
    ("## 6.", ["03"]),
]


def _line(refs: list[str]) -> str:
    return "（参 " + "".join(f"[{r}]" for r in refs) + "）"


def _safe_insert(text: str, pos: int) -> int:
    """把插入点挪出**围栏代码块**（P2-4）。

    旧实现直接把引用插在“小节最后一个非空字符之后”。如果小节以未闭合/
    刚打开的 ```python 结尾，插入点就落在代码块内部 —— docs/02 就出现过
    “```python / 空行 / （参 [03]）/ # S1 ...”这种被引用标记切开的伪代码，
    渲染成散文，py 语法也坏了。

    规则：统计 ``pos`` 之前所有以 ``` 开头的行；奇数说明还在围栏里，
    就把插入点移到那段围栏的**开头之前**（引用落在代码块前面，读起来
    也是对的）。闭合的围栏（偶数）不受影响。
    """
    lines = text[:pos].splitlines(keepends=True)
    fence_start = None
    inside = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            if not inside:
                fence_start = i
            inside = not inside
    if not inside or fence_start is None:
        return pos
    return sum(len(x) for x in lines[:fence_start])


def _section_end(text: str, start: int) -> int:
    """小节结束位置：下一个 `---` 分隔符或下一个 `### ` 标题，取更早者。"""
    a = text.find("\n---", start)
    m = re.search(r"^### ", text[start + 1:], re.M)
    b = start + 1 + m.start() if m else -1
    cands = [x for x in (a, b) if x >= 0]
    return min(cands) if cands else len(text)


def add_00() -> None:
    text = DOC00.read_text(encoding="utf-8")
    for eid, refs in sorted(MAP00.items(), key=lambda kv: -text.find(f"### {kv[0]} ")):
        start = text.find(f"### {eid} ")
        if start < 0:
            raise SystemExit(f"[00] 找不到小节 {eid}")
        end = _section_end(text, start)
        block = text[start:end]
        if "（参 " in block:
            continue
        chunk = block.rstrip()
        insert_at = _safe_insert(text, start + len(chunk))
        text = text[:insert_at] + "\n\n" + _line(refs) + text[insert_at:]
    if "## 参考文献" not in text:
        text = text.rstrip() + "\n\n---\n\n" + refs_md() + "\n"
    DOC00.write_text(text, encoding="utf-8")
    print("[00] 引用已写入")


def add_02() -> None:
    text = DOC02.read_text(encoding="utf-8")
    for anchor, refs in MAP02:
        start = text.find(anchor)
        if start < 0:
            raise SystemExit(f"[02] 找不到标题 {anchor}")
        level = len(anchor) - len(anchor.lstrip("#"))
        pat = re.compile(r"^#{1,%d} " % level, re.M)
        m = pat.search(text, start + len(anchor))
        end = m.start() if m else len(text)
        block = text[start:end]
        if "（参 " in block:
            continue
        # 插到该小节最后一个非空行之后，避免落在下一个标题的空行里；
        # 若末尾是代码围栏，_safe_insert 会挪到围栏之前（P2-4）。
        chunk = text[start:end].rstrip()
        insert_at = _safe_insert(text, start + len(chunk))
        text = text[:insert_at] + "\n\n" + _line(refs) + text[insert_at:]
    if "## 参考文献" not in text:
        text = text.rstrip() + "\n\n---\n\n" + refs_md() + "\n"
    DOC02.write_text(text, encoding="utf-8")
    print("[02] 引用已写入")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    add_00()
    add_02()
