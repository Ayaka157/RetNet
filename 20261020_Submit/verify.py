"""独立校验脚本：审核 `20261020_Submit` 里的文档与代码是否严丝合缝。

不依赖仓库、不依赖网络、不修改任何文件。直接运行：

    python verify.py

四项检查：
  A. 文档里每个代码块，与 `code/<路径>` 的同区间原文逐字节一致；
  B. 每个函数/方法都在文档里有独立章节（按 AST 枚举，漏一个就失败）；
  C. 每条解释 >= 100 字，且不含"同上/简略/略"这类占位词；
  D. 文档引用的 00 编号，在 00 文档里都真实存在。

任何一项失败，退出码为 1。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC01 = ROOT / "docs" / "01_训练代码逐函数解释.md"
DOC00 = ROOT / "docs" / "00_错误总表.md"
CODE = ROOT / "code"

FILES = [
    "retnet_hf/train.py",
    "retnet_hf/modeling_retnet.py",
    "retnet_hf/retention_fast.py",
    "retnet_hf/data.py",
    "retnet_hf/fused_ce.py",
    "run_task.py",
    "scripts/grow_500m.py",
]

HEAD_RE = re.compile(r"^### `(.+)`（L(\d+)–L(\d+)）$")
FILE_RE = re.compile(r"^## `(.+)`$")
BAD_PHRASES = ("同上", "简略", "略。", "见上")


def parse_doc(text: str):
    """返回 [(file, qual, start, end, code, explanation)]。"""
    lines = text.splitlines()
    cur_file = None
    out = []
    i = 0
    while i < len(lines):
        m = FILE_RE.match(lines[i])
        if m:
            cur_file = m.group(1)
            i += 1
            continue
        m = HEAD_RE.match(lines[i])
        if not m:
            i += 1
            continue
        qual, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        j = i + 1
        while j < len(lines) and not lines[j].startswith("```python"):
            j += 1
        k = j + 1
        while k < len(lines) and lines[k] != "```":
            k += 1
        code = "\n".join(lines[j + 1:k])
        # 解释：从代码块结束到下一个 ### 或 ---
        e = k + 1
        buf = []
        while e < len(lines) and not lines[e].startswith("### ") and lines[e] != "---":
            buf.append(lines[e])
            e += 1
        expl = "\n".join(buf).strip()
        out.append((cur_file, qual, a, b, code, expl))
        i = e
    return out


def collect_funcs(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []

    def rec(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(prefix + child.name)
                rec(child, prefix + child.name + ".")
            elif isinstance(child, ast.ClassDef):
                rec(child, prefix + child.name + ".")

    rec(tree, "")
    return out


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    fails = []
    doc = DOC01.read_text(encoding="utf-8")
    doc00 = DOC00.read_text(encoding="utf-8")
    sections = parse_doc(doc)
    print(f"[A] 文档解析出 {len(sections)} 个函数章节")

    # A + C
    for f, q, a, b, code, expl in sections:
        src_path = CODE / f
        if not src_path.exists():
            fails.append(f"[A] 代码文件不存在: {src_path}")
            continue
        src = src_path.read_text(encoding="utf-8").splitlines()
        if a < 1 or b > len(src):
            fails.append(f"[A] {f}::{q} 行号越界 L{a}-L{b}（文件 {len(src)} 行）")
            continue
        if code != "\n".join(src[a - 1:b]):
            fails.append(f"[A] 代码块与源文件不一致: {f}::{q}")
        if len(expl) < 100:
            fails.append(f"[C] 解释不足 100 字（{len(expl)}）: {f}::{q}")
        for bad in BAD_PHRASES:
            if bad in expl:
                fails.append(f"[C] 含占位词「{bad}」: {f}::{q}")

    # B：函数覆盖
    covered = {(f, q) for f, q, *_ in sections}
    total_funcs = 0
    for rel in FILES:
        p = CODE / rel
        if not p.exists():
            fails.append(f"[B] 缺少文件: {rel}")
            continue
        for q in collect_funcs(p):
            total_funcs += 1
            if (rel, q) not in covered:
                fails.append(f"[B] 文档漏函数: {rel}::{q}")
    print(f"[B] 代码共 {total_funcs} 个函数/方法，文档覆盖 {len(covered)} 个")

    # D：E 编号
    ids = set(re.findall(r"E\d{2}", doc00))
    refs = set()
    for m in re.finditer(r"\*\*00 关联\*\*：(.+)", doc):
        refs |= set(re.findall(r"E\d{2}", m.group(1)))
    dangling = sorted(refs - ids)
    if dangling:
        fails.append(f"[D] 文档引用了 00 里不存在的编号: {dangling}")
    print(f"[D] 00 编号 {len(ids)} 个，文档引用 {len(refs)} 个，悬空 {len(dangling)} 个")

    if fails:
        print(f"\n❌ {len(fails)} 项失败")
        for x in fails[:40]:
            print("   ", x)
        return 1
    print("\n✅ 全部通过：代码块逐字节一致、函数全覆盖、解释均 >=100 字、编号无悬空")
    return 0


if __name__ == "__main__":
    sys.exit(main())
