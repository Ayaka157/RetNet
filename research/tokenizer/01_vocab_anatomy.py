"""01 - vocabulary anatomy of data_cache/chat_pack_v13/bpe_tokenizer.json (vocab=20000).

Outputs a JSON dump + a markdown fragment, both under research/tokenizer/out/.
Pure CPU, no GPU.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, TOK_PATH, char_class, fmt_table, token_to_bytes  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

SPECIAL = {"<pad>", "<bos>", "<eos>", "<unk>"}


def main() -> None:
    (OUT / "out").mkdir(exist_ok=True)
    tok = Tokenizer.from_file(str(TOK_PATH))
    vocab = tok.get_vocab()
    V = tok.get_vocab_size()
    id2tok = [tok.id_to_token(i) for i in range(V)]

    # ---------------------------------------------------------------- merges
    raw = json.loads(TOK_PATH.read_text(encoding="utf-8"))
    n_merges = len(raw["model"]["merges"])
    merges_list = raw["model"]["merges"]

    rows = []
    classes = collections.Counter()
    byte_len_hist = collections.Counter()
    char_len_hist = collections.Counter()
    pure_cjk_len = collections.Counter()
    n_special = 0
    n_single_byte = 0
    n_pure_cjk = 0
    n_pure_latinish = 0
    n_pure_digit = 0
    multi_char_any = 0
    multi_char_cjk = 0
    multi_char_latin = 0
    cjk_repr: list[tuple[int, str, bytes]] = []
    latin_repr: list[tuple[int, str, bytes]] = []
    digit_repr: list[tuple[int, str, bytes]] = []

    for i, t in enumerate(id2tok):
        if t in SPECIAL:
            n_special += 1
            classes["special"] += 1
            continue
        b = token_to_bytes(t)
        if b is None:
            classes["unknown-non-bytelevel"] += 1
            continue
        byte_len_hist[len(b)] += 1
        if len(b) == 1:
            n_single_byte += 1
        s = b.decode("utf-8", errors="replace")
        char_len_hist[len(s)] += 1
        if len(s) > 1:
            multi_char_any += 1
        # class of the decoded surface form
        cc = collections.Counter(char_class(c) for c in s)
        body = {k: v for k, v in cc.items() if k != "space"}
        if not body:
            cls = "space"
        elif len(body) == 1:
            cls = next(iter(body))
        else:
            cls = "mixed"
        classes[cls] += 1
        if cls == "cjk":
            n_pure_cjk += 1
            pure_cjk_len[len(s)] += 1
            if len(s) > 1:
                multi_char_cjk += 1
            if len(cjk_repr) < 25:
                cjk_repr.append((i, t, b))
        if cls == "latin":
            n_pure_latinish += 1
            if len(s) > 1:
                multi_char_latin += 1
            if len(latin_repr) < 25:
                latin_repr.append((i, t, b))
        if cls == "digit":
            n_pure_digit += 1
            if len(digit_repr) < 25:
                digit_repr.append((i, t, b))

    total_bytes = sum(k * v for k, v in byte_len_hist.items())
    n_non_special = V - n_special

    # ---------------------------------------------------------------- examples
    probes = [
        " the", "hello", " world", "computer", "understanding", "transformer",
        "architecture", "government", "important", "different", "problem",
        "because", "information", "development", "university", "language",
        " the quick brown fox jumps over the lazy dog",
        "implementation", "consequently", "nevertheless", "The United States",
        "人工智能", "机器学习", "你好，世界", "中华人民共和国", "深度学习模型",
        "def ", "return", "print(", "import ", "    for i in range(",
        "2024", "3.14159", "1234567890",
    ]
    probe_rows = []
    for p in probes:
        enc = tok.encode(p, add_special_tokens=False)
        pieces = [tok.id_to_token(x) for x in enc.ids]
        probe_rows.append([
            p, len(p), len(p.encode("utf-8")), len(enc.ids),
            round(len(p.encode("utf-8")) / max(1, len(enc.ids)), 3),
            " ".join(repr(x) for x in pieces),
        ])

    # most / least used merges?  we cannot know training frequency, but we can
    # report merge-list position (BPE merges are ordered by learned priority).
    first_merges = merges_list[:15]
    merges_of_ascii_words = [m for m in merges_list if all(len(x) == 2 for x in m)][:0]

    # how many merges involve at least one non-single-byte side (i.e. length>=2)
    n_merge_multi = 0
    for a, b in merges_list:
        if len(a) > 1 or len(b) > 1:
            n_merge_multi += 1

    summary = {
        "tokenizer_path": str(TOK_PATH.relative_to(OUT.parent.parent.parent)),
        "vocab_size": V,
        "n_special": n_special,
        "n_merges": n_merges,
        "n_non_special": n_non_special,
        "single_byte_tokens": n_single_byte,
        "class_counts": dict(classes),
        "bytes_per_token_hist": dict(sorted(byte_len_hist.items())),
        "chars_per_token_hist": dict(sorted(char_len_hist.items())),
        "avg_bytes_per_token_over_vocab": round(total_bytes / n_non_special, 3),
        "pure_cjk_tokens": n_pure_cjk,
        "pure_cjk_charlen_hist": dict(sorted(pure_cjk_len.items())),
        "pure_cjk_multi_char": multi_char_cjk,
        "pure_latin_tokens": n_pure_latinish,
        "pure_latin_multi_char": multi_char_latin,
        "pure_digit_tokens": n_pure_digit,
        "multi_char_tokens_any": multi_char_any,
        "merges_with_side_len_gt1": n_merge_multi,
        "first_15_merges": first_merges,
        "model_byte_fallback": bool(getattr(tok.model, "byte_fallback", False)),
        "pre_tokenizer": str(tok.pre_tokenizer),
        "probes": probe_rows,
        "sum_chars_all_tokens": sum(k * v for k, v in char_len_hist.items()),
        "sum_bytes_all_tokens": total_bytes,
    }

    (OUT / "out" / "01_vocab_anatomy.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- md
    L: list[str] = []
    L.append("# 01 词表解剖\n")
    L.append(f"tokenizer = `{summary['tokenizer_path']}`（v9/v13 同一文件，MD5 相同）\n")
    L.append(f"- vocab_size = **{V}** = 4 special + {n_non_special} byte-level 条目")
    L.append(f"- merges = **{n_merges}**（其中 {n_merge_multi} 条至少一侧长度>1）")
    L.append(f"- `byte_fallback` = **{summary['model_byte_fallback']}**（即没有独立 byte-fallback 段，"
             f"256 个单字节 token 就是字节表本身）")
    L.append(f"- pre_tokenizer = `{summary['pre_tokenizer']}`\n")
    L.append("## 按解码表面对 Token 分类\n")
    rows_md = [[k, v, f"{v / n_non_special * 100:.2f}%"] for k, v in classes.most_common()]
    L.append(fmt_table(rows_md, ["类别", "token 数", "占非 special 比例"]))
    L.append("\n**类别定义**：把 token 的字节串按 UTF-8 解码后，看其中非空白字符都属于哪一类；"
             "只有单一类别才算 pure，混排记为 mixed。cjk 包含中日韩汉字/假名/谚文/全角与 CJK 标点。\n")
    L.append("## token 长度分布（按解码后的字节数 / 字符数）\n")
    L.append(fmt_table(
        [[k, v] for k, v in sorted(byte_len_hist.items())],
        ["字节长度", "token 数"]))
    L.append("")
    L.append(fmt_table(
        [[k, v] for k, v in sorted(char_len_hist.items())],
        ["字符长度", "token 数"]))
    L.append("")
    L.append(f"- 全部非 special token 的字节数总和 = {total_bytes:,}，"
             f"平均 **{summary['avg_bytes_per_token_over_vocab']} 字节/token**（词表容量视角，"
             f"不等于语料上的实际利用率）")
    L.append(f"- 纯 CJK token = **{n_pure_cjk}**（占 {n_pure_cjk / n_non_special * 100:.1f}%），"
             f"其中多字符的只有 **{multi_char_cjk}** 个")
    L.append(f"- 纯 Latin token = **{n_pure_latinish}**，其中多字符 **{multi_char_latin}** 个")
    L.append(f"- 纯数字 token = **{n_pure_digit}**")
    L.append(f"- 所有类别里多字符（len>1）的 token 共 **{multi_char_any}** 个，"
             f"占非 special 的 {multi_char_any / n_non_special * 100:.1f}%\n")
    L.append("### 纯 CJK token 的字符长度分布\n")
    L.append(fmt_table([[k, v] for k, v in sorted(pure_cjk_len.items())], ["字符数", "token 数"]))
    L.append("\n### merges 列表开头 15 条（BPE 学习优先级顺序）\n")
    L.append("```")
    for m in first_merges:
        L.append(f"{m[0]!r} + {m[1]!r}")
    L.append("```")
    L.append("\n## 探针：常见字符串的实际切分\n")
    L.append(fmt_table(
        [["`" + str(r[0]) + "`", r[1], r[2], r[3], r[4], "`" + r[5] + "`"] for r in probe_rows],
        ["文本", "字符", "字节", "token", "字节/token", "切分"]))
    (OUT / "out" / "01_vocab_anatomy.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote out/01_vocab_anatomy.{json,md}")
    print(json.dumps({k: summary[k] for k in (
        "vocab_size", "n_merges", "single_byte_tokens", "pure_cjk_tokens",
        "pure_latin_tokens", "pure_latin_multi_char", "pure_cjk_multi_char",
        "multi_char_tokens_any", "avg_bytes_per_token_over_vocab")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
