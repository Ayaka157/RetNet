"""08b - drift for the small append variants (completes the gain-vs-drift curve)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, TOK_PATH, fmt_table  # noqa: E402
from measure import build_testsets  # noqa: E402

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

O = OUT / "out"


def main() -> None:
    hold = json.loads((O / "holdout_rows.json").read_text(encoding="utf-8"))
    mig = json.loads((O / "06_migration.json").read_text(encoding="utf-8"))
    prod = Tokenizer.from_file(str(TOK_PATH))
    import difflib

    def spans(tok, t):
        return [tuple(o) for o in tok.encode(t, add_special_tokens=False).offsets]

    ps = [spans(prod, t) for t in hold[:300]]
    rows = []
    for name in ("app_top500", "app_top2000", "app_top8000", "app_ctl_v13_20k",
                 "app_ctl_v13_48k", "ctl_v13_20k", "ctl_v13_32k", "ctl_v13_48k",
                 "ref_qwen2.5-0.5b", "ref_deepseek-llm"):
        p = O / "tok" / f"{name}.json"
        if not p.exists():
            p = O / "ref" / f"{name.replace('ref_', '')}.json"
        if not p.exists():
            continue
        t2 = Tokenizer.from_file(str(p))
        vals = []
        for t, a in zip(hold[:300], ps):
            sm = difflib.SequenceMatcher(a=a, b=spans(t2, t), autojunk=False)
            vals.append(sum(b.size for b in sm.get_matching_blocks()) / max(1, len(a)))
        gain = mig["holdout_scores"].get(name, {}).get("prod_over_this")
        rows.append([name, t2.get_vocab_size(),
                     f"{np.mean(vals):.4f}", f"{np.median(vals):.4f}",
                     f"{(gain - 1) * 100:.2f}%" if gain else "-",
                     f"{gain:.4f}" if gain else "-"])
    L = ["# 08b 收益 vs 漂移：换分词器的取舍曲线\n",
         "留出集 300 条训练流文本上，prod 的 token span 有多少在候选里原样保留（LCS 一致率），"
         "以及同一批文本上的 token 节省比例。\n",
         fmt_table(rows, ["候选", "vocab", "平均 span 一致率", "中位", "省下的 token", "内容倍数"]),
         "",
         "> 一致率 = prod 的 token span 中能在候选编码里作为对齐 span 保留下来的比例。",
         "> 一致率越低 = 已有表示漂移越大。任何一个换法都做不到「一致率 100%」。"]
    (O / "08b_gain_vs_drift.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
