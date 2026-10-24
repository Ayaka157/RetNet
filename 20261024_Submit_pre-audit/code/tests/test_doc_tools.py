"""P2-4：引用插入器必须识别围栏代码块。

旧 `_add_citations.py` 把“（参 [03]）”插在小节最后一个非空字符之后。
当小节以未闭合的 ```python 结尾时，插入点落在代码块内部 —— docs/02 里就出现过
“```python / 空行 / （参 [03]）/ # S1 ...”这种被引用标记切开的伪代码，
渲染成散文、py 语法也坏了。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _add_citations import _safe_insert  # noqa: E402


def test_open_fence_moves_insert_point_before_the_fence():
    text = "## 1. 小节\n\n正文。\n\n```python\n# S1 停机规则\nif x:\n    pass\n"
    before = _safe_insert(text, len(text))
    # 插入点即围栏起点：引用落在代码块“之前”而不是内部
    assert before <= text.index("```python"), before
    new = text[:before] + "\n\n（参 [03]）" + text[before:]
    assert new.index("（参 [03]）") < new.index("```python")
    # 标记之后代码块仍然完整
    assert new.rstrip().endswith("pass")


def test_closed_fence_keeps_insert_point():
    text = "## 1. 小节\n\n```python\nx = 1\n```\n\n正文结束。\n"
    assert _safe_insert(text, len(text.rstrip())) == len(text.rstrip())


def test_no_fence_is_unchanged():
    text = "## 1. 小节\n\n只有正文。\n"
    assert _safe_insert(text, len(text.rstrip())) == len(text.rstrip())
