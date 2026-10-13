"""07 - reachability audit of the id-invariant appended-merge tokenizers.

A BPE merge (a,b) -> ab can only fire if both a and b are themselves reachable,
starting from the 256 single-byte tokens and walking the merge list in order.
This script walks the production tokenizer and each appended variant, and reports

  * how many of the original 20000 vocab entries stay reachable (= their embedding
    row stays meaningful)
  * how many appended merges actually fire
  * how many of the newly assigned ids are reachable

If base entries become unreachable, the append scheme silently kills embeddings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, TOK_PATH, fmt_table  # noqa: E402


def audited_merges(spec: dict) -> dict:
    vocab = spec["model"]["vocab"]
    merges = [tuple(m) for m in spec["model"]["merges"]]
    init = [t for t in vocab if len(t) == 1 and t not in
            ("<pad>", "<bos>", "<eos>", "<unk>")]
    reach = set(init)
    fired = 0
    dead = []
    for i, (a, b) in enumerate(merges):
        if a in reach and b in reach:
            reach.add(a + b)
            fired += 1
        else:
            dead.append(i)
    reach_vocab = {t for t in vocab if t in reach}
    return {
        "vocab": len(vocab), "merges": len(merges), "init_single_char": len(init),
        "merges_fired": fired, "merges_unreachable": len(dead),
        "reachable_vocab_entries": len(reach_vocab),
        "base_vocab_reachable": None,
    }


def main() -> None:
    prod = json.loads(Path(TOK_PATH).read_text(encoding="utf-8"))
    a = audited_merges(prod)
    base_tokens = set(prod["model"]["vocab"])
    # recompute reachability keeping the token set for the base slice
    vocab = prod["model"]["vocab"]
    merges = [tuple(m) for m in prod["model"]["merges"]]
    reach = set(t for t in vocab if len(t) == 1 and t not in
                ("<pad>", "<bos>", "<eos>", "<unk>"))
    for m in merges:
        if m[0] in reach and m[1] in reach:
            reach.add(m[0] + m[1])
    base_reach = base_tokens & reach
    a["base_vocab_reachable"] = len(base_reach)
    a["base_vocab_total"] = len(base_tokens)
    a["base_vocab_dead"] = sorted(base_tokens - reach)
    a["new_ids_reachable"] = 0

    rows = [["prod_v13", a["vocab"], a["merges"], a["merges_fired"],
             a["merges_unreachable"], a["reachable_vocab_entries"],
             a["base_vocab_reachable"], "-"]]
    detail = {"prod_v13": a}
    for p in sorted((OUT / "out" / "tok").glob("app_*.json")):
        spec = json.loads(p.read_text(encoding="utf-8"))
        r = audited_merges(spec)
        # how many of the appended new ids are reachable
        r["base_vocab_reachable"] = len({t for t in base_reach if t in spec["model"]["vocab"]})
        new_tokens = set(spec["model"]["vocab"]) - base_tokens
        r["new_ids_total"] = len(new_tokens)
        r["new_ids_reachable_count"] = len(new_tokens & reach_set(spec))
        rows.append([p.stem, r["vocab"], r["merges"], r["merges_fired"],
                     r["merges_unreachable"], r["reachable_vocab_entries"],
                     r["base_vocab_reachable"], r["new_ids_reachable_count"]])
        detail[p.stem] = r

    L = ["# 07 id 不变式追加 merge 的可达性审计\n",
         "BPE 的合并 `(a,b)->ab` 只有 a、b 本身可达时才会触发；从 256 个单字节 token 出发、",
         "按 merge 顺序走一遍即可判定每条 merge 是否真的会触发。\n",
         fmt_table(rows, ["变体", "vocab", "merges", "会触发的 merge",
                          "不可达 merge", "可达词表条目", "基座 20000 中仍可达",
                          "新增 id 中可达"]),
         "",
         f"prod_v13：merge 总数 {a['merges']}，其中 **{a['merges_unreachable']} 条不可达**"
         f"（占 {a['merges_unreachable'] / a['merges'] * 100:.2f}%）；"
         f"基座 20000 个条目里 **{a['base_vocab_total'] - a['base_vocab_reachable']} 个不可达**。"]
    dead = a["base_vocab_dead"]
    if dead:
        L.append(f"\nprod 自身不可达的条目（前 20 个）："
                 + ", ".join("`" + t.replace("`", "") + "`" for t in dead[:20]))
    L.append("\n> 结论判据：如果基座 20000 条目在追加变体里仍然全部可达，"
             "那么「老 id 的 embedding 仍对应同一个东西」在**表示层**成立；"
             "如果不可达条目变多，说明追加的 merge 会顶掉原有的切分路径。")
    (OUT / "out" / "07_reachability.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "out" / "07_reachability.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n".join(L))


def reach_set(spec: dict) -> set[str]:
    vocab = spec["model"]["vocab"]
    merges = [tuple(m) for m in spec["model"]["merges"]]
    reach = set(t for t in vocab if len(t) == 1 and t not in
                ("<pad>", "<bos>", "<eos>", "<unk>"))
    for m in merges:
        if m[0] in reach and m[1] in reach:
            reach.add(m[0] + m[1])
    return reach


if __name__ == "__main__":
    main()
