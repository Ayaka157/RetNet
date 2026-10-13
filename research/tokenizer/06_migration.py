"""06 - held-out measurement on the real training stream, and the migration experiments.

Part A: holdout.  1500 rows of train.ids.npy decoded back to text, drawn from
        `rng(20260418).permutation(N)[9044:]` — strictly disjoint from every row the
        04_train_controls.py corpora consumed (their samplers only ever touched
        permutation[:9044]).  Answers "how many tokens would each candidate produce
        for the same training content".

Part B: id-invariant vocab expansion.  Production ids 0..19999 are kept bit-for-bit;
        merges learned on the v13 corpus are appended after the production merge list
        with fresh ids 20000+.  Measures whether appending recovers the efficiency of
        a fresh retrain at the same final vocab size.

Part C: segmentation drift.  LCS-based agreement between production token spans and
        candidate token spans on the same text — quantifies how much of the existing
        representation would move.

Part D: embedding-row statistics of the real v13 checkpoint (for initialising new
        rows), incl. the byte-length/composition composition of each row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, ROOT, TOK_PATH, fmt_table, token_to_bytes  # noqa: E402
from measure import build_testsets, score  # noqa: E402

import numpy as np  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402

IDS = ROOT / "data_cache" / "chat_pack_v13" / "train.ids.npy"
CKPT = ROOT / "checkpoints" / "retnet_272m_v13" / "model.safetensors"
SEED = 20260418
SKIP = 9044          # every row used by the control corpora lives in permutation[:9044]
N_HOLD = 1500


# ------------------------------------------------------------------ part A
def holdout_rows(n: int = N_HOLD) -> list[str]:
    cache = OUT / "out" / "holdout_rows.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(TOK_PATH))
    arr = np.load(IDS, mmap_mode="r")
    order = np.random.default_rng(SEED).permutation(arr.shape[0])[SKIP:]
    rng = np.random.default_rng(4242)
    pick = np.sort(rng.choice(len(order), size=n, replace=False))
    texts = []
    for i in pick:
        row = np.asarray(arr[int(order[int(i)])])
        ids = [int(x) for x in row]
        while ids and ids[-1] == 0:
            ids.pop()
        if not ids:
            continue
        t = tok.decode(ids, skip_special_tokens=True)
        if t.strip():
            texts.append(t)
    cache.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return texts


# ------------------------------------------------------------------ part B
def build_appended(base_path: Path, donor_path: Path, kmax: int | None, out_path: Path) -> dict:
    """Append the donor's merges (in donor order) that the base does not already have."""
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    donor = json.loads(Path(donor_path).read_text(encoding="utf-8"))
    vocab: dict[str, int] = dict(base["model"]["vocab"])
    merges: list[list[str]] = [list(m) for m in base["model"]["merges"]]
    bset = {tuple(m) for m in merges}
    new_pairs = [list(m) for m in donor["model"]["merges"] if tuple(m) not in bset]
    if kmax is not None:
        new_pairs = new_pairs[:kmax]
    added = 0
    for a, b in new_pairs:
        merges.append([a, b])
        prod = a + b
        if prod not in vocab:
            vocab[prod] = len(vocab)
            added += 1
    spec = {
        "version": base.get("version", "1.0"),
        "truncation": base.get("truncation"),
        "padding": base.get("padding"),
        "added_tokens": base.get("added_tokens", []),
        "normalizer": base.get("normalizer"),
        "pre_tokenizer": base.get("pre_tokenizer"),
        "post_processor": base.get("post_processor"),
        "decoder": base.get("decoder"),
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>",
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False,
                  "ignore_merges": False, "vocab": vocab, "merges": merges},
    }
    out_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return {"base_vocab": len(base["model"]["vocab"]), "final_vocab": len(vocab),
            "merges_added": len(new_pairs), "new_ids": added}


# ------------------------------------------------------------------ part C
def lcs_agreement(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    """Fraction of production token spans that survive as an aligned span in b."""
    import difflib
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    match = sum(bl.size for bl in sm.get_matching_blocks())
    return match / max(1, len(a))


def spans(tok, text: str) -> list[tuple[int, int]]:
    enc = tok.encode(text, add_special_tokens=False)
    return [tuple(o) for o in enc.offsets]


# ------------------------------------------------------------------ part D
def embed_stats() -> dict:
    from safetensors import safe_open
    with safe_open(str(CKPT), framework="pt") as f:
        E = f.get_tensor("embedding.weight").float().numpy()
        names = None
    tok = Tokenizer.from_file(str(TOK_PATH))
    counts = np.load(OUT / "out" / "03_counts.npz")["counts"]
    v, d = E.shape
    std_all = float(E.std())
    row_norm = np.linalg.norm(E, axis=1)

    def grp(sel):
        if not sel.any():
            return {}
        return {"n": int(sel.sum()), "mean": float(E[sel].mean()),
                "std": float(E[sel].std()),
                "mean_row_norm": float(row_norm[sel].mean())}

    from common import char_class as _cc
    cls = []
    for i in range(v):
        if i < 4:
            cls.append("special")
            continue
        b = token_to_bytes(tok.id_to_token(i))
        s = b.decode("utf-8", errors="replace") if b else ""
        body = [c for c in s if not c.isspace()]
        k = "space" if not body else _cc(body[0])
        cls.append(k)
    cls = np.array(cls)
    by_class = {k: grp(cls == k) for k in sorted(set(cls))}

    # by training frequency bucket
    freq = counts[:v].copy()
    freq[0:4] = 0
    buckets = [(0, 0, "never used"), (1, 10, "1-10"), (11, 1000, "11-1k"),
               (1001, 100_000, "1k-100k"), (100_001, 10**12, ">100k")]
    by_freq = {}
    for lo, hi, name in buckets:
        sel = (freq >= lo) & (freq <= hi)
        by_freq[name] = grp(sel)
    # theoretical init stds
    init = {
        "N(0,1/d) std": (1.0 / d) ** 0.5,
        "N(0,1) std": 1.0,
        "N(0,1/sqrt(d)) std": (1.0 / d ** 0.5) ** 0.5,
        "empirical std": std_all,
        "empirical std of >100k bucket": by_freq[">100k"].get("std"),
        "sqrt(1/1024)": 1024 ** -0.5,
    }
    # mean-of-components init: for a few real 2-char CJK tokens, compare norm
    examples = []
    for i in range(4, v):
        t = tok.id_to_token(i)
        b = token_to_bytes(t)
        if not b:
            continue
        s = b.decode("utf-8", errors="replace")
        if len(s) == 2:
            parts = [tok.encode(ch, add_special_tokens=False).ids for ch in s]
            if all(len(p) == 1 for p in parts):
                m = E[parts[0]] + E[parts[1]]      # sum init
                a = 0.5 * (E[parts[0]] + E[parts[1]])
                examples.append({
                    "token": s,
                    "component_norms": [float(np.linalg.norm(E[p[0]])) for p in parts],
                    "sum_init_norm": float(np.linalg.norm(m)),
                    "mean_init_norm": float(np.linalg.norm(a)),
                    "own_row_norm": float(row_norm[i]),
                })
        if len(examples) >= 12:
            break
    return {"shape": [v, d], "global_mean": float(E.mean()), "global_std": std_all,
            "row_norm_mean": float(row_norm.mean()), "row_norm_std": float(row_norm.std()),
            "row_norm_min": float(row_norm.min()), "row_norm_max": float(row_norm.max()),
            "by_class": by_class, "by_freq": by_freq, "init_reference": init,
            "mean_or_sum_examples": examples}


def main() -> None:
    ts = build_testsets()
    hold = holdout_rows()
    print(f"holdout: {len(hold)} rows, {sum(len(t) for t in hold):,} chars", flush=True)

    cands = {"prod_v13": Path(TOK_PATH)}
    for p in sorted((OUT / "out" / "tok").glob("*.json")):
        if p.name == "train_log.json":
            continue
        cands[p.stem] = p
    for p in sorted((OUT / "out" / "ref").glob("*.json")):
        cands["ref_" + p.stem] = p

    # ---- part B: build appended variants
    tokdir = OUT / "out" / "tok"
    appended = {}
    for donor in ("ctl_v13_20k", "ctl_v13_48k"):
        info = build_appended(TOK_PATH, tokdir / f"{donor}.json", None,
                              tokdir / f"app_{donor}.json")
        appended[f"app_{donor}"] = info
        print("built", donor, info, flush=True)
    for k in (500, 2000, 8000):
        info = build_appended(TOK_PATH, tokdir / "ctl_v13_48k.json", k,
                              tokdir / f"app_top{k}.json")
        appended[f"app_top{k}"] = info
        print("built top", k, info, flush=True)
    for name in appended:
        cands[name] = tokdir / f"{name}.json"

    # ---- score everything on the holdout
    rows = []
    base_tokens = None
    results = {}
    for name, path in cands.items():
        tok = Tokenizer.from_file(str(path))
        s = score(tok, hold)
        results[name] = {"vocab_size": tok.get_vocab_size(), **{
            k: s[k] for k in ("docs", "chars", "bytes", "tokens", "chars_per_token",
                              "bytes_per_token")}}
        if name == "prod_v13":
            base_tokens = s["tokens"]
        print(f"  {name:22s} vocab={tok.get_vocab_size():>7} tokens={s['tokens']:>12,} "
              f"chars/tok={s['chars_per_token']:.4f}", flush=True)
    for name, r in results.items():
        r["prod_over_this"] = base_tokens / r["tokens"]

    # ---- part C: segmentation drift on 300 holdout docs
    prod = Tokenizer.from_file(str(TOK_PATH))
    drift = {}
    for name in ("ctl_v13_20k", "ctl_v13_48k", "app_ctl_v13_48k", "ref_qwen2.5-0.5b"):
        if name not in cands:
            continue
        other = Tokenizer.from_file(str(cands[name]))
        vals = []
        for t in hold[:300]:
            a, b = spans(prod, t), spans(other, t)
            vals.append(lcs_agreement(a, b))
        drift[name] = {"mean_token_agreement": float(np.mean(vals)),
                       "median": float(np.median(vals)),
                       "min": float(np.min(vals)), "max": float(np.max(vals))}
        print(f"drift {name}: agreement={drift[name]['mean_token_agreement']:.3f}",
              flush=True)

    es = embed_stats()
    print("embedding std", es["global_std"], "row norm mean", es["row_norm_mean"], flush=True)

    out = {"holdout": {"rows": len(hold), "chars": sum(len(t) for t in hold),
                       "source": f"train.ids.npy rng({SEED}).permutation(N)[{SKIP}:]"},
           "holdout_scores": results, "appended_variants": appended,
           "segmentation_drift": drift, "embedding_stats": es}
    (OUT / "out" / "06_migration.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# 06 迁移实验与嵌入统计\n",
         f"## Part A 真实训练流的留出行（{len(hold)} 行，"
         f"{sum(len(t) for t in hold):,} 字符）\n",
         "取自 `train.ids.npy` 中 `rng(20260418).permutation(N)[9044:]` 的随机 1500 行——"
         "与 `04_train_controls.py` 训练语料用过的行**完全不重叠**（那些采样器只碰过 "
         "`permutation[:9044]`）。\n"]
    L.append(fmt_table(
        [[n, r["vocab_size"], f"{r['tokens']:,}", f"{r['chars_per_token']:.4f}",
          f"{r['bytes_per_token']:.4f}", f"{r['prod_over_this']:.4f}"]
         for n, r in sorted(results.items(), key=lambda x: x[1]['tokens'])],
        ["候选", "vocab", "token", "chars/token", "bytes/token",
         "prod token 数 / 本候选"]))
    L.append("\n## Part B id 不变式词表扩张（保留 0–19999，只追加 20000+）\n")
    L.append(fmt_table([[n, i["base_vocab"], i["merges_added"], i["new_ids"],
                         i["final_vocab"]] for n, i in appended.items()],
                       ["变体", "基座 vocab", "追加 merge 数", "新 id 数", "最终 vocab"]))
    L.append("\n## Part C 分词边界漂移（prod 的 token span 有多少在候选里原样保留，LCS）\n")
    L.append(fmt_table([[n, f"{v['mean_token_agreement']:.4f}", f"{v['median']:.4f}",
                         f"{v['min']:.4f}", f"{v['max']:.4f}"] for n, v in drift.items()],
                       ["候选", "平均一致率", "中位", "最小", "最大"]))
    L.append("\n## Part D v13 checkpoint 的 embedding 行统计\n")
    L.append(f"- `embedding.weight` 形状 {es['shape']}，全局 mean={es['global_mean']:.5f}，"
             f"std={es['global_std']:.5f}")
    L.append(f"- 行范数 mean={es['row_norm_mean']:.4f}，std={es['row_norm_std']:.4f}，"
             f"min={es['row_norm_min']:.4f}，max={es['row_norm_max']:.4f}")
    L.append(f"- 参考初始化 std：`N(0,1/d)` = {es['init_reference']['N(0,1/d) std']:.5f}")
    L.append("\n| 类别 | 行数 | 行均值 | 行 std | 平均行范数 |")
    L.append("|---|---|---|---|---|")
    for k, v in es["by_class"].items():
        L.append(f"| {k} | {v['n']:,} | {v['mean']:+.5f} | {v['std']:.5f} | "
                 f"{v['mean_row_norm']:.4f} |")
    L.append("\n| 训练频率桶 | 行数 | 行 std | 平均行范数 |")
    L.append("|---|---|---|---|")
    for k, v in es["by_freq"].items():
        L.append(f"| {k} | {v['n']:,} | {v['std']:.5f} | {v['mean_row_norm']:.4f} |")
    L.append("\n### 「组成片段求和/取均值」初始化 vs 该 token 自己的真实行\n")
    L.append("| token | 片段行范数 | sum 初始化范数 | mean 初始化范数 | 真实行范数 |")
    L.append("|---|---|---|---|---|")
    for e in es["mean_or_sum_examples"]:
        L.append(f"| {e['token']} | "
                 f"{'/'.join(f'{x:.3f}' for x in e['component_norms'])} | "
                 f"{e['sum_init_norm']:.3f} | {e['mean_init_norm']:.3f} | "
                 f"{e['own_row_norm']:.3f} |")
    (OUT / "out" / "06_migration.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote out/06_migration.{md,json}")


if __name__ == "__main__":
    main()
