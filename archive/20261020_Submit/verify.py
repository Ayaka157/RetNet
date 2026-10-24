"""独立校验脚本：审核 `20261020_Submit` 里的文档与代码是否严丝合缝。

不联网、不依赖仓库、不修改任何文件。直接运行：

    python verify.py

检查项：
  A. 01 文档里每个代码块，与 `code/<路径>` 同区间逐字节一致；
  B. 01 覆盖 `code/` 下 7 个训练文件的**全部**函数/方法（AST 枚举，漏一个就失败）；
  C. 01 每条解释 >=100 字，且不含"同上/简略"；
  D. 01 引用的 00 编号都真实存在；
  E. 00F1 每个代码块同样逐字节一致；
  F. 00F1 每条说明 >=500 字，且 00 里的每个编号都有对应条目。

任何一项失败，退出码为 1。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
DOC00 = ROOT / "docs" / "00_错误总表.md"
DOC01 = ROOT / "docs" / "01_训练代码逐函数解释.md"
DOCF1 = ROOT / "docs" / "00F1_修复对照.md"

FILES = [
    "retnet_hf/train.py",
    "retnet_hf/modeling_retnet.py",
    "retnet_hf/retention_fast.py",
    "retnet_hf/data.py",
    "retnet_hf/fused_ce.py",
    "run_task.py",
    "scripts/grow_500m.py",
]

HEAD01 = re.compile(r"^### `(.+)`（L(\d+)–L(\d+)）$")
FILE01 = re.compile(r"^## `(.+)`$")
REF_F1 = re.compile(r"^`(.+?)` L(\d+)–L(\d+)$")
HEAD_F1 = re.compile(r"^## (E\d{2}|N\d{2}) ")


def parse_01(text: str):
    lines = text.splitlines()
    cur = None
    out = []
    i = 0
    while i < len(lines):
        m = FILE01.match(lines[i])
        if m:
            cur = m.group(1)
            i += 1
            continue
        m = HEAD01.match(lines[i])
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
        e = k + 1
        buf = []
        while e < len(lines) and not lines[e].startswith("### ") and lines[e] != "---":
            buf.append(lines[e])
            e += 1
        out.append((cur, qual, a, b, code, "\n".join(buf).strip()))
        i = e
    return out


def parse_f1(text: str):
    lines = text.splitlines()
    cur = None
    blocks = []
    texts = {}
    i = 0
    while i < len(lines):
        m = HEAD_F1.match(lines[i])
        if m:
            cur = m.group(1)
            texts[cur] = []
            i += 1
            continue
        if cur:
            m = REF_F1.match(lines[i])
            if m:
                k = i + 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if k < len(lines) and lines[k].startswith("```python"):
                    j = k + 1
                    buf = []
                    while j < len(lines) and lines[j] != "```":
                        buf.append(lines[j])
                        j += 1
                    blocks.append((cur, m.group(1), int(m.group(2)), int(m.group(3)),
                                   "\n".join(buf)))
                    i = j + 1
                    continue
            texts[cur].append(lines[i])
        i += 1
    return blocks, {k: "\n".join(v).strip() for k, v in texts.items()}


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


def read_src(rel: str):
    p = CODE / rel
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").splitlines()


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    fails = []
    doc00 = DOC00.read_text(encoding="utf-8")
    ids = set(re.findall(r"\b(E\d{2}|N\d{2})\b", doc00))

    # ---------------- 01 ----------------
    secs = parse_01(DOC01.read_text(encoding="utf-8"))
    print(f"[A] 01 解析出 {len(secs)} 个函数章节")
    for f, q, a, b, code, expl in secs:
        src = read_src(f)
        if src is None:
            fails.append(f"[A] 代码文件不存在: {f}")
            continue
        if not (1 <= a <= b <= len(src)):
            fails.append(f"[A] {f}::{q} 行号越界 L{a}-L{b}")
            continue
        if code != "\n".join(src[a - 1:b]):
            fails.append(f"[A] 代码块与源文件不一致: {f}::{q}")
        if len(expl) < 100:
            fails.append(f"[C] 01 解释不足 100 字（{len(expl)}）: {f}::{q}")
        for bad in ("同上", "简略"):
            if bad in expl:
                fails.append(f"[C] 01 含占位词「{bad}」: {f}::{q}")

    covered = {(f, q) for f, q, *_ in secs}
    total = 0
    for rel in FILES:
        p = CODE / rel
        if not p.exists():
            fails.append(f"[B] 缺少文件: {rel}")
            continue
        for q in collect_funcs(p):
            total += 1
            if (rel, q) not in covered:
                fails.append(f"[B] 01 漏函数: {rel}::{q}")
    print(f"[B] 代码 {total} 个函数，文档覆盖 {len(covered)}")

    refs = set()
    for m in re.finditer(r"\*\*00 关联\*\*：(.+)", DOC01.read_text(encoding="utf-8")):
        refs |= set(re.findall(r"E\d{2}", m.group(1)))
    dangling = sorted(refs - ids)
    if dangling:
        fails.append(f"[D] 01 引用了 00 里不存在的编号: {dangling}")
    print(f"[D] 00 编号 {len(ids)} 个，01 引用 {len(refs)} 个，悬空 {len(dangling)}")

    # ---------------- 00F1 ----------------
    f1_blocks, f1_text = parse_f1(DOCF1.read_text(encoding="utf-8"))
    print(f"[E] 00F1 解析出 {len(f1_text)} 条、{len(f1_blocks)} 个代码块")
    for eid, rel, a, b, code in f1_blocks:
        src = read_src(rel)
        if src is None:
            fails.append(f"[E] 00F1 引用了不存在的文件: {rel}")
            continue
        if not (1 <= a <= b <= len(src)) or code != "\n".join(src[a - 1:b]):
            fails.append(f"[E] 00F1 代码块不一致: {eid} {rel} L{a}-{b}")
    for eid, txt in f1_text.items():
        if len(txt) < 500:
            fails.append(f"[F] 00F1 {eid} 说明只有 {len(txt)} 字")
        for bad in ("省略", "同上"):
            if bad in txt:
                fails.append(f"[F] 00F1 {eid} 含占位词「{bad}」")
    missing = sorted(ids - set(f1_text))
    if missing:
        fails.append(f"[F] 00F1 缺条目: {missing}")
    print(f"[F] 00F1 覆盖 {len(set(f1_text) & ids)}/{len(ids)} 个编号")

    if fails:
        print(f"\n❌ {len(fails)} 项失败")
        for x in fails[:40]:
            print("   ", x)
        return 1
    print("\n✅ 全部通过：01/00F1 代码块逐字节一致、函数全覆盖、说明长度达标、编号无悬空")
    return 0


if __name__ == "__main__":
    sys.exit(main())
