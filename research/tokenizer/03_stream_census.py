"""03 - census of the real training token stream (data_cache/chat_pack_v13/train.ids.npy).

Everything here reads the .npy directly, so there is no decode/encode bias.

Outputs:
  * id-usage census: how many of the 20000 ids ever appear, token-mass concentration
  * exact per-class token budget (every token classified by its own bytes)
  * pad / bos / eos / unk counts  (independently re-derives meta.json's pad number)
  * round-trip check: decode(ids) -> encode -> compare length (validates that
    decoding rows back to text is lossless up to dropped special tokens)
Pure CPU, streams the 1.1 GB file in chunks.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, ROOT, TOK_PATH, token_to_bytes  # noqa: E402

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

IDS = ROOT / "data_cache" / "chat_pack_v13" / "train.ids.npy"
WCODE = ROOT / "data_cache" / "chat_pack_v13" / "train.wcode.npy"


def class_of_token(tb: bytes) -> str:
    from common import char_class
    s = tb.decode("utf-8", errors="replace")
    cc = collections.Counter(char_class(c) for c in s)
    body = {k: v for k, v in cc.items() if k != "space"}
    if not body:
        return "space"
    if len(body) == 1:
        return next(iter(body))
    return "mixed"


def main() -> None:
    tok = Tokenizer.from_file(str(TOK_PATH))
    V = tok.get_vocab_size()
    PAD, BOS, EOS, UNK = (tok.token_to_id(x) for x in ("<pad>", "<bos>", "<eos>", "<unk>"))

    arr = np.load(IDS, mmap_mode="r")
    n_rows, width = arr.shape
    total = arr.size
    print(f"shape={arr.shape} total={total:,}", flush=True)

    counts = np.zeros(V, dtype=np.int64)
    pad = bos = eos = unk = 0
    CH = 20000
    cache = OUT / "out" / "03_counts.npz"
    if cache.exists():
        z = np.load(cache)
        counts = z["counts"]
        pad, bos, eos, unk = (int(x) for x in z["specials"])
        print("loaded cached counts", flush=True)
    else:
        for s in range(0, n_rows, CH):
            blk = np.asarray(arr[s:s + CH]).reshape(-1)
            counts += np.bincount(blk.astype(np.int64), minlength=V)
            pad += int((blk == PAD).sum())
            bos += int((blk == BOS).sum())
            eos += int((blk == EOS).sum())
            unk += int((blk == UNK).sum())
            if s % 100000 == 0:
                print(f"  {s:,}/{n_rows:,}", flush=True)
        np.savez(cache, counts=counts, specials=np.array([pad, bos, eos, unk]))

    used_ids = int((counts > 0).sum())
    order = np.argsort(-counts)
    cum = np.cumsum(counts[order])
    def mass(k: int) -> float:
        return float(cum[min(k, V) - 1] / total)

    # per-class token budget, exact
    cls_tok = collections.Counter()
    cls_bytes = collections.Counter()
    cls_chars = collections.Counter()
    SPECIAL_IDS = {PAD: "pad", BOS: "bos", EOS: "eos", UNK: "unk"}
    for i in range(V):
        c = int(counts[i])
        if c == 0:
            continue
        if i in SPECIAL_IDS:
            cls_tok[SPECIAL_IDS[i]] += c
            continue
        t = tok.id_to_token(i)
        b = token_to_bytes(t)
        if b is None:
            cls_tok["unknown-non-bytelevel"] += c
            continue
        k = class_of_token(b)
        cls_tok[k] += c
        cls_bytes[k] += len(b) * c
        cls_chars[k] += len(b.decode("utf-8", errors="replace")) * c

    # unused ids and their identity
    unused = [i for i in range(V) if counts[i] == 0]
    unused_by_cls = collections.Counter()
    unused_examples = collections.defaultdict(list)
    for i in unused:
        if i in SPECIAL_IDS:
            unused_by_cls["special"] += 1
            continue
        t = tok.id_to_token(i)
        b = token_to_bytes(t)
        if b is None:
            unused_by_cls["unknown-non-bytelevel"] += 1
            continue
        k = class_of_token(b)
        unused_by_cls[k] += 1
        if len(unused_examples[k]) < 12:
            unused_examples[k].append(t)

    rare = [(int(i), int(counts[i])) for i in order[::-1][:40]]
    rare_rows = [[i, tok.id_to_token(i), tok.decode([i]), c] for i, c in rare]

    # ------------------------------------------------------------ round trip
    rng = np.random.default_rng(7)
    rows = np.sort(rng.choice(n_rows, size=200, replace=False))
    diffs = collections.Counter()
    for ri in rows:
        row = np.asarray(arr[int(ri)])
        ids = [int(x) for x in row]
        while ids and ids[-1] == PAD:
            ids.pop()
        text = tok.decode(ids, skip_special_tokens=True)
        re = tok.encode(text, add_special_tokens=False).ids
        diffs[len(re) - len(ids)] += 1

    wc = np.load(WCODE, mmap_mode="r")
    code_counts = np.bincount(np.asarray(wc[::97]).reshape(-1).astype(np.int64), minlength=4)
    print("scan done, writing", flush=True)

    summary = {
        "shape": list(arr.shape),
        "total_slots": int(total),
        "pad": pad, "pad_pct": pad / total * 100,
        "bos": bos, "eos": eos, "unk": unk,
        "n_special_total": bos + eos + unk + pad,
        "used_ids": used_ids,
        "used_ids_pct": used_ids / V * 100,
        "unused_ids": V - used_ids,
        "unused_by_class": dict(unused_by_cls),
        "unused_examples": {k: v for k, v in unused_examples.items()},
        "mass_top1k": mass(1000), "mass_top2k": mass(2000),
        "mass_top5k": mass(5000), "mass_top10k": mass(10000),
        "mass_top16k": mass(16384),
        "rarest_used": rare_rows,
        "class_tokens": dict(cls_tok),
        "class_bytes": dict(cls_bytes),
        "class_chars": dict(cls_chars),
        "top30_ids": [[int(order[i]), tok.id_to_token(int(order[i])), int(counts[order[i]])]
                      for i in range(30)],
        "roundtrip_diff_hist": {str(k): v for k, v in sorted(diffs.items())},
        "wcode_slice_counts": {str(i): int(code_counts[i]) for i in range(4)},
    }
    (OUT / "out" / "03_stream_census.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# 03 真实训练流普查（直接读 train.ids.npy，无 decode 偏差）\n"]
    L.append(f"- 形状 {arr.shape}，槽位 {total:,}")
    L.append(f"- pad = **{pad:,}**（{pad / total * 100:.2f}%）；bos={bos:,} eos={eos:,} unk=**{unk:,}**")
    L.append(f"- 出现过的 id = **{used_ids} / {V}**（{used_ids / V * 100:.2f}%）；"
             f"**从未出现 {V - used_ids} 个**")
    L.append(f"- token 质量集中度：top1000 = {mass(1000) * 100:.2f}%，top2000 = {mass(2000) * 100:.2f}%，"
             f"top5000 = {mass(5000) * 100:.2f}%，top10000 = {mass(10000) * 100:.2f}%")
    L.append("\n## 精确 token 预算（按 token 自身字节分类，一个 token 只归一类）\n")
    rows = []
    for k in sorted(cls_tok, key=lambda x: -cls_tok[x]):
        c = cls_tok[k]
        rows.append([k, f"{c:,}", f"{c / total * 100:.2f}%",
                     f"{cls_chars[k]:,}", f"{cls_chars[k] / c:.3f}" if c else "-",
                     f"{cls_bytes[k] / c:.3f}" if c else "-"])
    L.append("| 类别 | token 数 | 占全部 token | 该类别字符数 | chars/token | bytes/token |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    L.append("\n> 上表把 id 0/1/2/3 单列为 pad/bos/eos/unk；"
             f"净算式：special 合计 = pad+bos+eos+unk = {pad + bos + eos + unk:,}，"
             f"与 bincount 总数 {total:,} 的关系见 json。")
    L.append("\n## 未使用 id 的构成\n")
    L.append("| 类别 | 未使用个数 |")
    L.append("|---|---|")
    for k, v in sorted(unused_by_cls.items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v} |")
    L.append("\n示例：")
    for k, ex in unused_examples.items():
        L.append(f"- `{k}`: " + ", ".join("`" + e.replace("`", "") + "`" for e in ex))
    L.append("\n## top 30 高频 id\n")
    L.append("| id | token | 次数 |")
    L.append("|---|---|---|")
    for i, t, c in summary["top30_ids"]:
        L.append(f"| {i} | `{t}` | {c:,} |")
    L.append("\n## 最罕见的 40 个仍在用的 id\n")
    L.append("| id | token | 解码 | 次数 |")
    L.append("|---|---|---|---|")
    for i, t, d, c in rare_rows:
        L.append(f"| {i} | `{t}` | {d!r} | {c} |")
    L.append(f"\n## 往返一致性（200 行）\n")
    L.append("decode(row) → encode 后的 token 数 − 原 token 数："
             + ", ".join(f"{k}: {v} 行" for k, v in sorted(diffs.items())))
    L.append("\n> 原行内每个文档都以 <eos> 结尾，`skip_special_tokens=True` 把它去掉，"
             "所以差值基本等于该行内的文档数（−N），不是分词器不可逆。")
    (OUT / "out" / "03_stream_census.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote out/03_stream_census.{json,md}")
    print(f"used_ids={used_ids}/{V} pad={pad:,}({pad / total * 100:.2f}%) "
          f"mass_top5000={mass(5000) * 100:.2f}%")
    print("class token budget:", dict(cls_tok))


if __name__ == "__main__":
    main()
