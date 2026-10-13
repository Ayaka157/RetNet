"""09 - side-by-side tokenisation examples (attribution evidence).

Shows, on the same strings, how production / a distribution-matched 20k refit /
an English-only 20k refit / a Chinese-only 20k refit / Qwen2.5-0.5B / DeepSeek-LLM
split the text.  This is the direct evidence for "is English shredded because the
vocab is Chinese-dominated" / "is the vocab mismatched to the data".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, TOK_PATH, fmt_table  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

O = OUT / "out"

EN = [" the", "hello", " world", "understanding", "implementation",
      "consequently", "the quick brown fox jumps over the lazy dog",
      "Let me think step by step.", "We can solve this by substitution.",
      "Firstly, note that 2x + 3 = 11.", "\\frac{1}{2}", "\\boxed{42}",
      "def solve(n):\n    return n * 2", "answer is 1/2"]
ZH = ["人工智能", "机器学习模型", "你好，世界", "中华人民共和国", "请一步一步地推理",
      "答案是 42。", "设 x 为未知数", "我们需要先化简这个表达式"]
MIX = ["用户：什么是梯度下降？\n助手：梯度下降是一种优化算法。",
       "Q: What is 2+2?\nA: 4",
       "id=A7-2291 数量 3480 只，占比 62%"]


def main() -> None:
    cands = {"prod_v13": O / ".." / ".." / ".." / \
             "data_cache" / "chat_pack_v13" / "bpe_tokenizer.json"}
    cands = {"prod_v13": Path(TOK_PATH)}
    for n in ("ctl_v13_20k", "ctl_v13_48k", "ctl_v13en_20k", "ctl_v13zh_20k",
              "ctl_orig_20k", "app_top500", "app_top2000"):
        p = O / "tok" / f"{n}.json"
        if p.exists():
            cands[n] = p
    for n in ("qwen2.5-0.5b", "deepseek-llm", "gpt2"):
        p = O / "ref" / f"{n}.json"
        if p.exists():
            cands["ref_" + n] = p
    toks = {n: Tokenizer.from_file(str(p)) for n, p in cands.items()}

    L = ["# 09 同串切分对照（归因证据）\n"]
    for title, group in (("英文", EN), ("中文", ZH), ("中英混排", MIX)):
        L.append(f"## {title}\n")
        for s in group:
            L.append(f"**输入** `{s}`  （{len(s)} 字符 / {len(s.encode('utf-8'))} 字节）\n")
            rows = []
            for n, t in toks.items():
                ids = t.encode(s, add_special_tokens=False).ids
                pieces = [t.id_to_token(i) for i in ids]
                rows.append([n, len(ids),
                             f"{len(s) / max(1, len(ids)):.3f}",
                             " · ".join(p.replace("Ġ", "␣") for p in pieces)])
            L.append(fmt_table(rows, ["候选", "token", "chars/token", "切分"]))
            L.append("")

    # stats on a small English word list
    L.append("## 常见英文词的单 token 率（word starts with a space）\n")
    words = ("the of and to in a is that it for was as with be by on not he this "
             "but from they we say her she or an will my one all would there their "
             "what so up out if about who get which go me when make can like time no "
             "just him know take people into year your good some could them see other "
             "than then now look only come its over think also back after use two how "
             "our work first well way even new want because any these give day most us "
             "machine learning model token language model training data context window "
             "attention gradient descent transformer architecture").split()
    rows = []
    for n, t in toks.items():
        ntok = 0
        one = 0
        for w in words:
            ids = t.encode(" " + w, add_special_tokens=False).ids
            ntok += len(ids)
            one += (len(ids) == 1)
        rows.append([n, f"{one}/{len(words)}", f"{one / len(words) * 100:.1f}%",
                     f"{ntok / len(words):.3f}"])
    L.append(fmt_table(rows, ["候选", "单 token 词数", "占比", "平均 token/词"]))
    (O / "09_examples.md").write_text("\n".join(L), encoding="utf-8")
    json.dump({"english_words": words, "per_candidate": {
        n: {"one_token": sum(1 for w in words
                             if len(t.encode(' ' + w, add_special_tokens=False).ids) == 1),
            "n_words": len(words)} for n, t in toks.items()}},
        open(O / "09_examples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote out/09_examples.md")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
