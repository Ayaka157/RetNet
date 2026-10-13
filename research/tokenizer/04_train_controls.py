"""04 - build control-tokenizer training corpora and train the controls.

Design (see research/tokenizer/report.md section 3):
  * corpora are CACHED as .txt under out/corpus/ so training is reproducible
  * every control uses the production config: BPE + ByteLevel(add_prefix_space=True)
    pre-tokenizer, ByteLevel decoder, initial_alphabet = ByteLevel.alphabet(),
    specials = <pad>/<bos>/<eos>/<unk>, min_frequency=2   (retnet/tokenizer.py)

Corpora
  v13_4M     4M chars decoded from data_cache/chat_pack_v13/train.ids.npy, uniform
             row stride over the whole file  == the actual training distribution
  v13_20M    20M chars, same sampler, to bound the "corpus size" confound
  v13zh_4M   rows of the v13 stream whose CJK ratio >= 0.9
  v13en_4M   rows of the v13 stream whose CJK ratio <= 0.02
  orig_4M    the *original* tokenizer's corpus (legacy/prepare_chat_corpus.py
             readers, same file list and same per-source proportion), truncated to
             the matched 4M-char budget -> isolates corpus DISTRIBUTION from SIZE

Usage:
  python 04_train_controls.py corpus      # build+cache corpora
  python 04_train_controls.py train       # train the tokenizers
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "legacy"))

import numpy as np  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from common import OUT, ROOT, TOK_PATH, cjk_ratio  # noqa: E402

CORP = OUT / "out" / "corpus"
TOKDIR = OUT / "out" / "tok"
IDS = ROOT / "data_cache" / "chat_pack_v13" / "train.ids.npy"
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
SEED = 20260418


# ------------------------------------------------------------------ corpora
def v13_rows(n_chars: int, lang: str = "any", seed: int = SEED) -> str:
    """Decode rows out of the v13 training stream until ~n_chars collected."""
    tok = Tokenizer.from_file(str(TOK_PATH))
    arr = np.load(IDS, mmap_mode="r")
    n = arr.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    parts: list[str] = []
    total = 0
    kept = scanned = 0
    for ri in order:
        scanned += 1
        row = np.asarray(arr[int(ri)])
        ids = [int(x) for x in row]
        while ids and ids[-1] == 0:
            ids.pop()
        if not ids:
            continue
        text = tok.decode(ids, skip_special_tokens=True)
        r = cjk_ratio(text)
        if lang == "zh" and r < 0.9:
            continue
        if lang == "en" and r > 0.02:
            continue
        parts.append(text)
        total += len(text)
        kept += 1
        if total >= n_chars:
            break
    print(f"  v13[{lang}]: scanned {scanned} rows, kept {kept}, {total:,} chars", flush=True)
    return "\n".join(parts)[:n_chars]


def _orig_sources():
    """The exact file list of legacy/prepare_chat_corpus.py, plus its wikitext103
    slice (fetched by 04b_fetch_wikitext.py)."""
    import prepare_chat_corpus as pcc  # legacy/
    cc = ROOT / "data_cache" / "chat_corpus"
    out = [
        ("lccc", lambda lim: pcc.iter_lccc(cc / "lccc_base_train.jsonl", limit=lim)),
        ("glm", lambda lim: pcc.iter_glm(cc / "train_all_df.csv", limit=lim)),
        ("sharegpt", lambda lim: pcc.iter_sharegpt(
            cc / "sharegpt_jsonl/common_zh_70k.jsonl", limit=lim)),
        ("wiki", lambda lim: pcc.iter_wiki_json(
            cc / "wikipedia-cn-20230720-filtered.json", limit=lim)),
    ]
    wt = CORP / "wikitext103.txt"
    if wt.exists():
        def _wt(lim, _p=wt):
            with _p.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= lim:
                        break
                    line = line.strip()
                    if line:
                        yield line[:2000]
        out.append(("wikitext", _wt))
    return out


# text-count weights = exactly what prepare_chat_corpus.py sampled
ORIG_WEIGHT = {"lccc": 250_000, "glm": 250_000, "sharegpt": 20_000,
               "wiki": 30_000, "wikitext": 200_000}


def orig_corpus(n_chars: int) -> str:
    """Original distribution, matched to n_chars.

    Pass 1 measures chars/text per source on a 400-text probe; pass 2 samples each
    source to the char budget implied by its original text-count weight.
    """
    sources = _orig_sources()
    probe = {}
    for name, mk in sources:
        got, cnt = 0, 0
        for t in mk(400):
            got += len(t)
            cnt += 1
        probe[name] = got / max(1, cnt)
        print(f"  orig probe[{name}]: {cnt} texts, avg {probe[name]:.0f} chars", flush=True)
    # allocate: source char share = weight * avg_chars; solve iteratively by scaling
    raw = {n: ORIG_WEIGHT[n] * probe[n] for n, _ in sources}
    tot = sum(raw.values())
    parts, stats = [], {}
    for name, mk in sources:
        budget = int(n_chars * raw[name] / tot)
        limit = max(1, int(budget / max(1.0, probe[name]) * 1.4) + 10)
        got, cnt = 0, 0
        for t in mk(limit):
            if got >= budget:
                break
            parts.append(t)
            got += len(t)
            cnt += 1
        stats[name] = [cnt, got]
        print(f"  orig[{name}]: {cnt} texts, {got:,} chars (budget {budget:,})", flush=True)
    return "\n".join(parts)[:n_chars]


def build_corpora() -> dict[str, int]:
    CORP.mkdir(parents=True, exist_ok=True)
    jobs = [("v13_4M", lambda: v13_rows(4_000_000)),
            ("v13_20M", lambda: v13_rows(20_000_000)),
            ("v13zh_4M", lambda: v13_rows(4_000_000, "zh")),
            ("v13en_4M", lambda: v13_rows(4_000_000, "en")),
            ("orig_4M", lambda: orig_corpus(4_000_000))]
    info = {}
    for name, fn in jobs:
        p = CORP / f"{name}.txt"
        if p.exists():
            info[name] = len(p.read_text(encoding="utf-8"))
            print(f"[skip] {name} {info[name]:,} chars", flush=True)
            continue
        t0 = time.time()
        txt = fn()
        p.write_text(txt, encoding="utf-8")
        info[name] = len(txt)
        print(f"[ok] {name} {len(txt):,} chars in {time.time() - t0:.1f}s "
              f"({len(txt.encode('utf-8')):,} bytes)", flush=True)
    (CORP / "index.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ------------------------------------------------------------------ training
def train_bpe(texts: str, vocab_size: int, path: Path, min_frequency: int = 2) -> None:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator([texts], trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


PLAN = [
    ("ctl_v13_20k", "v13_4M", 20000),
    ("ctl_v13_32k", "v13_4M", 32000),
    ("ctl_v13_48k", "v13_4M", 48000),
    ("ctl_v13big_20k", "v13_20M", 20000),
    ("ctl_v13zh_20k", "v13zh_4M", 20000),
    ("ctl_v13en_20k", "v13en_4M", 20000),
    ("ctl_orig_20k", "orig_4M", 20000),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["corpus", "train", "all"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    if args.stage in ("corpus", "all"):
        build_corpora()
    if args.stage in ("train", "all"):
        TOKDIR.mkdir(parents=True, exist_ok=True)
        results = {}
        for name, corp, vs in PLAN:
            if args.only and args.only not in name:
                continue
            out = TOKDIR / f"{name}.json"
            if out.exists():
                print(f"[skip] {name}", flush=True)
                continue
            p = CORP / f"{corp}.txt"
            assert p.exists(), f"missing corpus {p}; run the corpus stage first"
            txt = p.read_text(encoding="utf-8")
            t0 = time.time()
            train_bpe(txt, vs, out)
            dt = time.time() - t0
            results[name] = {"corpus": corp, "vocab": vs, "chars": len(txt),
                             "sec": round(dt, 1), "path": str(out)}
            print(f"[train] {name} vocab={vs} corpus={corp} {len(txt):,} chars "
                  f"-> {dt:.1f}s", flush=True)
            (TOKDIR / "train_log.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
        print("done", flush=True)


if __name__ == "__main__":
    main()
