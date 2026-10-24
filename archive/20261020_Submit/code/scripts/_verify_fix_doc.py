"""独立校验 00F1：每个代码块必须与源文件同区间逐字节一致；说明 >=500 字。

用法：python scripts/_verify_fix_doc.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "00F1_修复对照.md"
DOC00 = ROOT / "docs" / "00_错误总表.md"

HEAD = re.compile(r"^## (E\d{2}|N\d{2}) ")
REFP = re.compile(r"^`(.+?)` L(\d+)–L(\d+)$")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    lines = DOC.read_text(encoding="utf-8").splitlines()
    wanted = set(re.findall(r"\b(E\d{2}|N\d{2})\b", DOC00.read_text(encoding="utf-8")))

    cur = None
    i = 0
    entries = {}
    while i < len(lines):
        m = HEAD.match(lines[i])
        if m:
            cur = m.group(1)
            entries[cur] = {"blocks": 0, "text": []}
            i += 1
            continue
        if cur:
            m = REFP.match(lines[i])
            if m:
                k = i + 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
            else:
                k = -1
            if m and k >= 0 and k < len(lines) and lines[k].startswith("```python"):
                rel, a, b = m.group(1), int(m.group(2)), int(m.group(3))
                j = k + 1
                buf = []
                while j < len(lines) and lines[j] != "```":
                    buf.append(lines[j])
                    j += 1
                src = (ROOT / rel).read_text(encoding="utf-8").splitlines()
                if "\n".join(src[a - 1:b]) != "\n".join(buf):
                    print(f"❌ {rel} L{a}-{b} 代码块与源文件不一致")
                    return 1
                entries[cur]["blocks"] += 1
                i = j + 1
                continue
            entries[cur]["text"].append(lines[i])
        i += 1

    bad = []
    for k, v in entries.items():
        text = "\n".join(v["text"]).strip()
        if v["blocks"] == 0 and k not in ("N03",):
            bad.append(f"{k} 没有代码块")
        if len(text) < 500:
            bad.append(f"{k} 说明只有 {len(text)} 字")
        for w in ("省略", "同上"):
            if w in text:
                bad.append(f"{k} 含占位词 {w}")

    missing = sorted(wanted - set(entries))
    if missing:
        bad.append(f"00 里有、00F1 缺失: {missing}")

    if bad:
        print(f"❌ {len(bad)} 项失败")
        for b in bad[:30]:
            print("   ", b)
        return 1
    print(f"✅ 00F1 通过：{len(entries)} 条，代码块与源文件逐字节一致，说明均 >=500 字，"
          f"00 编号全覆盖（{len(wanted)} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
