"""Build chat_pack_v9 = chat_pack_v8 sources + A-grade datasets.

Reuses build_chat_mix_v4's specs (v8 = last pack with v8 sources) and adds the
downloaded A-grade corpora.  Plain long texts (wiki/anime/novels) are tokenized
and cut into fixed-length W_PLAIN blocks so nothing is wasted.

    python scripts/build_chat_mix_v9.py --out_dir data_cache/chat_pack_v9 --seq_len 512
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.chat import W_PLAIN, encode_dialogues_batch  # noqa: E402
from retnet.packing import Packer, clean, iter_jsonl, messages_to_pairs  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402

Pairs = List[tuple]
Sample = tuple   # (pairs, system) or ("TEXT", text) for plain blocks


# ------------------------------------------------------------- A-grade readers
def rp_nana(path: Path):
    """system 角色卡 + user/assistant 多轮。"""
    for obj in iter_jsonl(path):
        msgs = obj.get("messages") or []
        system = ""
        rest = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "system":
                system = clean(m.get("content") or "", 500)
            else:
                rest.append(m)
        pairs = messages_to_pairs(rest, 500)
        if pairs:
            yield pairs, (system[:500] if system else None)


def rp_singleturn(path: Path):
    """instruction=角色设定, input=用户, output=角色回复。"""
    for obj in iter_jsonl(path):
        ins = clean(obj.get("instruction") or "", 400)
        u = clean(obj.get("input") or "", 400)
        a = clean(obj.get("output") or "", 600)
        if u and a:
            yield [(u, a)], ins or None


def rp_meow(path: Path):
    """猫娘 R1 格式：prompt + reasoning + response（用 response 作回答）。"""
    for obj in iter_jsonl(path):
        u = clean(obj.get("prompt") or "", 400)
        r = clean(obj.get("response") or "", 800)
        reas = clean(obj.get("reasoning") or "", 300)
        if u and r:
            a = (reas + "\n" + r) if reas else r
            yield [(u, a[:800])], "你是一只可爱的猫娘，说话带“喵”口吻，喜欢用拟声词和撒娇语气。"


def rp_longevo(path: Path):
    """LongEvoRoleBench: role + profile_text + input/output（剥离 input 里的角色前缀）。"""
    for obj in iter_jsonl(path):
        profile = clean(obj.get("profile_text") or "", 300)
        u = clean(obj.get("input") or "", 400)
        a = clean(obj.get("output") or "", 600)
        role = clean(obj.get("role") or "", 40)
        if role and u.startswith(role + ":"):
            u = u[len(role) + 1:].strip()
        if u and a:
            persona = f"你是{role}。{profile}" if role else (profile or None)
            yield [(u, a)], persona[:400] if persona else None


def rp_novelist(path: Path):
    """英文写作 CoT：instruction/input/output。"""
    for obj in iter_jsonl(path):
        ins = clean(obj.get("instruction") or "", 300)
        u = clean(obj.get("input") or "", 300)
        a = clean(obj.get("output") or "", 1200)
        user = (ins + " " + u).strip() if u else ins
        if user and a:
            yield [(user, a)], None


def casual_hundredcv(path: Path):
    """dialog 是 user1：/user2： 字符串交替 → 轮流当用户/助手。"""
    for obj in iter_jsonl(path):
        lines = [clean(x.split("：", 1)[-1], 300) for x in (obj.get("dialog") or [])
                 if isinstance(x, str) and "：" in x]
        lines = [x for x in lines if x]
        if len(lines) >= 2:
            pairs = [(lines[j], lines[j + 1]) for j in range(0, len(lines) - 1, 2)]
            yield pairs, None


def plain_texts(path: Path, key: str = "text", min_len: int = 200):
    for obj in iter_jsonl(path):
        t = obj.get(key) or ""
        if isinstance(t, str) and len(t) >= min_len:
            yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_cache/chat_pack_v9")
    ap.add_argument("--tokenizer", default="data_cache/chat_pack_v4/bpe_tokenizer.json")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--ag_dir", default="data_cache/a_grade")
    ap.add_argument("--batch_docs", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=300)
    ap.add_argument("--val_max_tokens", type=int, default=2_000_000)
    # A-grade budgets (tokens)
    ap.add_argument("--nana_tokens", type=int, default=15_000_000)
    ap.add_argument("--hundredcv_tokens", type=int, default=8_000_000)
    ap.add_argument("--singleturn_tokens", type=int, default=3_000_000)
    ap.add_argument("--meow_tokens", type=int, default=10_000_000)
    ap.add_argument("--longevo_tokens", type=int, default=8_000_000)
    ap.add_argument("--novelist_tokens", type=int, default=3_000_000)
    ap.add_argument("--anime_plain_tokens", type=int, default=15_000_000)
    ap.add_argument("--moegirl_plain_tokens", type=int, default=12_000_000)
    ap.add_argument("--novels_plain_tokens", type=int, default=10_000_000)
    ap.add_argument("--cc0_plain_tokens", type=int, default=5_000_000)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    tok = BPETokenizer.load(args.tokenizer)
    row_len = args.seq_len + 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = Packer(row_len, tok.pad_id)
    val = Packer(row_len, tok.pad_id)
    stats, val_tokens = {}, 0
    ag = Path(args.ag_dir)

    # (name, kind, iterator, budget)
    specs = [
        ("a_nana", "rp", rp_nana(ag / "nana_catgirl.jsonl"), args.nana_tokens),
        ("a_singleturn", "rp", rp_singleturn(ag / "roleplay_singleturn.jsonl"), args.singleturn_tokens),
        ("a_meow", "rp", rp_meow(ag / "meow_reasoning.jsonl"), args.meow_tokens),
        ("a_longevo", "rp", rp_longevo(ag / "longevorolebench.jsonl"), args.longevo_tokens),
        ("a_novelist", "instruct", rp_novelist(ag / "novelist_cot.jsonl"), args.novelist_tokens),
        ("a_hundredcv", "casual", casual_hundredcv(ag / "hundredcv_chat.jsonl"), args.hundredcv_tokens),
    ]

    for name, kind, it, budget in specs:
        t0, used, docs = time.time(), 0, 0
        batch = []
        done = False
        while not done and used < budget:
            batch.clear()
            try:
                for _ in range(args.batch_docs):
                    batch.append(next(it))
            except StopIteration:
                done = True
            except FileNotFoundError:
                print(f"[skip] {name}: missing file", flush=True)
                break
            if not batch:
                break
            dialogues = [b[0] for b in batch]
            systems = [b[1] for b in batch]
            for ids, codes in encode_dialogues_batch(tok, dialogues, max_len=args.seq_len,
                                                     systems=systems):
                if len(ids) < 16:
                    continue
                docs += 1
                if docs % args.val_every == 0 and val_tokens < args.val_max_tokens:
                    val.add(ids, codes)
                    val_tokens += len(ids)
                else:
                    train.add(ids, codes)
                    used += len(ids)
        stats[name] = {"kind": kind, "tokens": used, "docs": docs,
                       "sec": round(time.time() - t0, 1)}
        print(f"[{kind:8}] {name:<16} tokens={used:>10,} docs={docs:>8,} ({stats[name]['sec']}s)", flush=True)

    # plain text corpora -> W_PLAIN blocks
    plain_specs = [
        ("a_refined_anime", plain_texts(ag / "refined_anime.jsonl"), args.anime_plain_tokens),
        ("a_moegirl", plain_texts(ag / "moegirlpedia.jsonl"), args.moegirl_plain_tokens),
        ("a_scholar_novels", plain_texts(ag / "scholar_novels.jsonl"), args.novels_plain_tokens),
        ("a_cc0_novel", plain_texts(ag / "chinese_novel_cc0.jsonl"), args.cc0_plain_tokens),
    ]
    for name, it, budget in plain_specs:
        t0, used, docs = time.time(), 0, 0
        buf: List[int] = []
        try:
            for text in it:
                enc = tok.encode(clean(text, 40000))
                if len(enc) < 40:
                    continue
                buf.extend([tok.bos_id] + enc + [tok.eos_id])
                while len(buf) >= args.seq_len and used < budget:
                    blk = buf[:args.seq_len]
                    train.add(blk, [W_PLAIN] * len(blk))
                    used += len(blk)
                    docs += 1
                    buf = buf[args.seq_len:]
        except FileNotFoundError:
            print(f"[skip] {name}: missing file", flush=True)
            continue
        stats[name] = {"kind": "plain", "tokens": used, "docs": docs,
                       "sec": round(time.time() - t0, 1)}
        print(f"[plain   ] {name:<16} tokens={used:>10,} docs={docs:>8,} ({stats[name]['sec']}s)", flush=True)

    # base v8 sources (reuse build specs logic from build_chat_mix_v4)
    from build_chat_mix_v4 import build_specs, Sample as _S
    base_args = argparse.Namespace(
        rp_dir=Path("data_cache/roleplay"), sft_dir=Path("data_cache/extra_sft"),
        extra_sft_dir=Path("data_cache/extra_sft2"), corpus_dir=Path("data_cache/chat_corpus"),
        lit_dir=Path("data_cache/literature"), rp_expand_tokens=120_000_000,
        rp_repeat=3, instruct_scale=0.45, style_tokens=25_000_000,
        cornell_tokens=20_000_000, wikisource_zh_tokens=8_000_000,
        wikisource_en_tokens=3_000_000, gutenberg_tokens=3_000_000,
        novel_tokens=20_000_000, lccc_tokens=20_000_000, lccc_stride=5,
        lccc_min_turn=8, glm_tokens=10_000_000)
    for name, kind, factory, budget in build_specs(base_args):
        t0, used, docs = time.time(), 0, 0
        batch: List[Sample] = []
        done = False
        gen = factory()          # 只创建一次：生成器是惰性的，重复创建会从头重放
        while not done and used < budget:
            batch.clear()
            try:
                for _ in range(args.batch_docs):
                    batch.append(next(gen))
            except StopIteration:
                done = True
            except FileNotFoundError:
                print(f"[skip] {name}: missing file", flush=True)
                break
            if not batch:
                break
            from retnet.packing import is_fragment
            batch = [b for b in batch if not any(is_fragment(a) for _, a in b[0])]
            if not batch:
                continue
            dialogues = [b[0] for b in batch]
            systems = [b[1] for b in batch]
            for ids, codes in encode_dialogues_batch(tok, dialogues, max_len=args.seq_len,
                                                     systems=systems):
                if len(ids) < 16:
                    continue
                docs += 1
                if docs % args.val_every == 0 and val_tokens < args.val_max_tokens:
                    val.add(ids, codes)
                    val_tokens += len(ids)
                else:
                    train.add(ids, codes)
                    used += len(ids)
        stats[name] = {"kind": kind, "tokens": used, "docs": docs,
                       "sec": round(time.time() - t0, 1)}
        print(f"[{kind:8}] {name:<16} tokens={used:>10,} docs={docs:>8,} ({stats[name]['sec']}s)", flush=True)

    # replay
    t0 = time.time()
    files = []
    for d in ["data_cache/pretrain_v2", "data_cache/pretrain_merged"]:
        p = Path(d)
        if p.exists():
            files.extend(sorted(x for x in p.glob("*.npy") if "_val" not in x.name))
            if (p / "train.ids.npy").exists():
                files.append(p / "train.ids.npy")
    taken, chunk = 0, args.seq_len
    for path in files[:30]:
        if taken >= 30_000_000:
            break
        src = np.load(str(path), mmap_mode="r")
        for start in range(0, src.shape[0] - chunk, chunk * 9973):
            piece = np.asarray(src[start:start + chunk], dtype=np.int32)
            train.add(piece.tolist(), [W_PLAIN] * chunk)
            taken += chunk
            if taken >= 30_000_000:
                break
    stats["replay"] = {"kind": "plain", "tokens": taken, "docs": 0,
                       "sec": round(time.time() - t0, 1)}
    print(f"[plain   ] replay              tokens={taken:>10,}", flush=True)

    train_ids, train_codes = train.finish()
    val_ids, val_codes = val.finish()
    np.save(str(out / "train.ids.npy"), train_ids)
    np.save(str(out / "train.wcode.npy"), train_codes)
    np.save(str(out / "val.ids.npy"), val_ids)
    np.save(str(out / "val.wcode.npy"), val_codes)
    shutil.copyfile(args.tokenizer, out / "bpe_tokenizer.json")
    by_kind = {}
    for name, s in stats.items():
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + s["tokens"]
    meta = {"seq_len": args.seq_len, "train_rows": int(train_ids.shape[0]),
            "train_tokens": int(train_ids.size),
            "reply_tokens": int((train_codes == 1).sum()),
            "prompt_tokens": int((train_codes == 2).sum()),
            "plain_tokens": int((train_codes == 3).sum()),
            "by_kind": by_kind, "sources": stats}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
