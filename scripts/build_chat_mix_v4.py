"""Build the role-play focused chat dataset (packed rows, int16, loss-masked).

Target model: a Chinese/English **role-play** chat model.  Mixture priorities:

1. ``rp``      – in-character dialogue with a 设定 persona block (highest weight)
2. ``instruct``– instruction following, so the model stays useful
3. ``casual``  – everyday chit-chat (quality filtered; raw LCCC is very noisy)
4. ``novel``   – Chinese narration + quoted dialogue, gives role-play "voice"
5. ``plain``   – pretrain replay so world knowledge does not wash out

The ChatHaruhi-Expand contexts contain whole scene transcripts from the source
novels; ``mine_haruhi_expand`` slices them into de-duplicated multi-turn samples,
which is where most of the Chinese role-play tokens come from.

    python scripts/build_chat_mix_v4.py --out_dir data_cache/chat_pack_v4 --seq_len 512
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.chat import W_PLAIN, encode_dialogues_batch  # noqa: E402
from retnet.packing import (Packer, clean, cjk_ratio, is_fragment, iter_glm,  # noqa: E402
                            iter_jsonl, iter_lccc, iter_sft_file, messages_to_pairs)
from retnet.tokenizer import BPETokenizer  # noqa: E402

Pairs = List[Tuple[str, str]]
Sample = Tuple[Pairs, str | None]          # (dialogue pairs, system/persona)

ROLE_LINE = re.compile(r"^([^\s:：]{1,24})\s*[:：]\s*(.+)$")
QUOTES = "「」“”\"'《》"


# ------------------------------------------------------------ role-play tools
def strip_quotes(s: str) -> str:
    s = s.strip()
    while s and s[0] in QUOTES:
        s = s[1:]
    while s and s[-1] in QUOTES:
        s = s[:-1]
    return s.strip()


def persona_text(role: str, source: str = "", traits: List[str] | None = None,
                 max_chars: int = 400) -> str:
    bits = [f"你是{role}" + (f"（出自{source}）" if source else "") + "。"]
    for t in (traits or []):
        t = clean(t, 160)
        if t and t not in bits:
            bits.append(t)
    text = "".join(bits)
    return text[:max_chars]


def lines_to_pairs(entries: List[Tuple[str, str]], role: str,
                   max_chars: int = 300) -> Pairs:
    """Turn ``[(speaker, text), ...]`` into (user, assistant) pairs for ``role``.

    Lines by other speakers (and ``scene:`` narration) accumulate into the user
    turn; lines by ``role`` become the assistant reply.
    """
    pairs: Pairs = []
    buffer: List[str] = []
    for speaker, text in entries:
        text = clean(text, max_chars)
        if not text:
            continue
        if speaker == "scene":
            buffer.append(f"（{text}）")
        elif speaker == role:
            if buffer:
                pairs.append((" ".join(buffer)[:max_chars * 2], text))
                buffer = []
            elif pairs:
                # consecutive lines by the role: merge into the last reply
                prev_u, prev_a = pairs[-1]
                pairs[-1] = (prev_u, f"{prev_a} {text}"[:max_chars * 2])
        else:
            buffer.append(f"{speaker}：{text}")
    return pairs


def mine_haruhi_expand(path: Path, seen: set, min_pairs: int = 1,
                       max_pairs: int = 6) -> Iterator[Sample]:
    """ChatHaruhi-Expand: persona card + novel scene transcripts -> RP samples."""
    for obj in iter_jsonl(path):
        context = obj.get("context") or ""
        target = clean(obj.get("target") or "", 300)
        if not context:
            continue
        m = re.search(r"act like (.+?) from (.+?)[\.\n]", context)
        if not m:
            continue
        role, source = m.group(1).strip(), m.group(2).strip()
        head = context.split("Classic scenes", 1)[0]
        traits = [l.strip() for l in head.split("\n")
                  if cjk_ratio(l) > 0.5 and len(l.strip()) > 6]
        persona = persona_text(role, source, traits[:4])

        body = context.split("Classic scenes for the role are as follows:", 1)[-1]
        segments: List[List[Tuple[str, str]]] = [[]]
        for raw in body.split("\n"):
            s = raw.strip()
            if not s:
                continue
            if set(s) <= {"#"}:
                segments.append([])
                continue
            if s.startswith("scene:"):
                segments[-1].append(("scene", s[len("scene:"):].strip()))
                continue
            mm = ROLE_LINE.match(s)
            if mm:
                segments[-1].append((mm.group(1).strip(), strip_quotes(mm.group(2))))

        for seg in segments:
            if not seg:
                continue
            pairs = lines_to_pairs(seg, role)
            if len(pairs) < min_pairs:
                continue
            pairs = pairs[:max_pairs]
            key = hashlib.blake2b(
                (role + "|" + "|".join(u + a for u, a in pairs)).encode("utf-8"),
                digest_size=12).digest()
            if key in seen:
                continue
            seen.add(key)
            yield pairs, persona

        # the reply the dataset actually asks for, with the tail as context
        if target:
            mm = ROLE_LINE.match(target)
            reply = strip_quotes(mm.group(2)) if mm else target
            tail = segments[-1] if segments else []
            pairs = lines_to_pairs(tail, role)[-3:]
            user_bits = [f"{sp}：{clean(tx, 200)}" for sp, tx in tail[-2:]
                         if sp not in {role, "scene"}]
            if user_bits:
                pairs = pairs + [(" ".join(user_bits), reply)]
            elif pairs:
                pairs[-1] = (pairs[-1][0], reply)
            else:
                continue
            key = hashlib.blake2b((role + "|T|" + reply).encode("utf-8"),
                                  digest_size=12).digest()
            if key not in seen:
                seen.add(key)
                yield pairs, persona


def mine_haruhi_54k(path: Path) -> Iterator[Sample]:
    for obj in iter_jsonl(path):
        agent = clean(obj.get("agent_role"), 40)
        user_role = clean(obj.get("user_role"), 40)
        q = strip_quotes(clean(obj.get("user_question"), 300))
        a = strip_quotes(clean(obj.get("agent_response"), 400))
        if not (agent and q and a):
            continue
        persona = persona_text(agent, "", [f"你正在与{user_role}对话。"] if user_role else None)
        pairs: Pairs = [(q, a)]
        more = obj.get("more_dialogues")
        if isinstance(more, list):
            for extra in more[:3]:
                if isinstance(extra, dict):
                    q2 = strip_quotes(clean(extra.get("user_question"), 300))
                    a2 = strip_quotes(clean(extra.get("agent_response"), 400))
                    if q2 and a2:
                        pairs.append((q2, a2))
        yield pairs, persona


def mine_roleplay_zh_gpt4(path: Path) -> Iterator[Sample]:
    for obj in iter_jsonl(path):
        raw = obj.get("system_prompt") or ""
        # Read the name from the RAW field: cleaning collapses the newlines that
        # separate 角色名称 from 性格特征.
        m = re.search(r"角色名称[:：]\s*([^\n]{1,20})", raw)
        system = clean(raw, 450)
        pairs = messages_to_pairs(obj.get("conversations") or [], 500)
        if system and pairs:
            head = f"你是{m.group(1).strip()}。" if m else ""
            yield pairs, (head + system)[:500]


def mine_haruhi_zero_pippa(path: Path) -> Iterator[Sample]:
    for obj in iter_jsonl(path):
        prompt = obj.get("prompt_zh") or obj.get("prompt") or ""
        if not prompt:
            continue
        head, _, body = prompt.partition("###")
        m = re.search(r"你扮演\s*(.+?)\s*中的\s*(\S+)", head)
        if m:
            role, source = m.group(2), m.group(1)
        else:
            role, source = clean(obj.get("bot_name"), 30), ""
        if not role:
            continue
        traits = [l.strip() for l in head.split("\n") if cjk_ratio(l) > 0.4]
        persona = persona_text(role, source, traits[:4])
        entries: List[Tuple[str, str]] = []
        for raw in body.split("\n"):
            s = raw.strip()
            if not s or set(s) <= {"#"}:
                continue
            mm = ROLE_LINE.match(s)
            if not mm:
                continue
            speaker = mm.group(1).strip().replace("{{user}}", "对方")
            entries.append((speaker, strip_quotes(mm.group(2))))
        pairs = lines_to_pairs(entries, role)
        if pairs:
            yield pairs, persona


def mine_conversations_rp(path: Path, must_contain: str | None = None,
                          max_chars: int = 600) -> Iterator[Sample]:
    """Generic ``conversations`` role-play file; the system turn becomes 设定."""
    for obj in iter_jsonl(path):
        conv = obj.get("conversations")
        if not isinstance(conv, list) or not conv:
            continue
        system = ""
        rest = []
        for m in conv:
            if not isinstance(m, dict):
                continue
            if str(m.get("from", "")).lower() == "system":
                system = clean(m.get("value"), 600)
            else:
                rest.append(m)
        if must_contain and must_contain not in system:
            continue
        pairs = messages_to_pairs(rest, max_chars)
        if not pairs:
            continue
        if not system:
            # LimaRP puts the RP setup inside the first human turn
            first_user = pairs[0][0]
            if len(first_user) > 200:
                system = first_user[:600]
                pairs = pairs[1:] if len(pairs) > 1 else [("（继续）", pairs[0][1])]
        yield pairs, (system or None)


def iter_novel_pairs(path: Path, max_chars: int = 600) -> Iterator[Sample]:
    """webnovel_cn style instruction data -> plain (user, assistant) pairs."""
    for pairs in iter_sft_file(path, max_chars):
        yield pairs, None


def iter_literature_rp(path: Path) -> Iterator[Sample]:
    """Mined literature dialogue: {"title": ..., "pairs": [[u, a], ...]}.

    The work title becomes the persona so the model learns in-character speech
    from classical drama and movie dialogue alike.
    """
    if not path.exists():
        return
    for obj in iter_jsonl(path):
        pairs = obj.get("pairs")
        title = obj.get("title") or ""
        if not isinstance(pairs, list) or not pairs:
            continue
        clean_pairs: Pairs = []
        for p in pairs:
            if isinstance(p, list) and len(p) == 2 and p[0] and p[1]:
                clean_pairs.append((clean(p[0], 400), clean(p[1], 400)))
        if len(clean_pairs) >= 1:
            persona = f"你正在扮演{title}中的角色，用该作品的语言风格说话。" if title else None
            yield clean_pairs, persona


# --------------------------------------------------------------------- specs
def repeat(factory, times: int):
    """Cycle a source ``times`` times (RP data is small but high value).

    Each pass gets a fresh dedupe set so repeats are not filtered out.
    """
    def gen():
        for _ in range(max(1, times)):
            yield from factory()
    return gen


def build_specs(args):
    rp = Path(args.rp_dir)
    sft = Path(args.sft_dir)
    sft2 = Path(args.extra_sft_dir)
    corpus = Path(args.corpus_dir)
    ins = args.instruct_scale
    rep = args.rp_repeat
    firefly_path = (rp / "firefly_full.jsonl") if (rp / "firefly_full.jsonl").exists() \
        else (sft2 / "firefly.jsonl")

    specs = [
        # ---------------- role-play (highest priority, repeated) ----------------
        ("haruhi_expand", "rp",
         repeat(lambda: mine_haruhi_expand(rp / "haruhi_expand.jsonl", set()), rep),
         args.rp_expand_tokens),
        ("haruhi_54k", "rp",
         repeat(lambda: mine_haruhi_54k(rp / "haruhi_54k.jsonl"), rep), 30_000_000),
        ("roleplay_zh_gpt4", "rp",
         repeat(lambda: mine_roleplay_zh_gpt4(rp / "roleplay_zh_gpt4.jsonl"), rep), 8_000_000),
        ("haruhi_zero_pippa", "rp",
         repeat(lambda: mine_haruhi_zero_pippa(rp / "haruhi_zero_pippa.jsonl"), rep), 4_000_000),
        ("coser_rp", "rp",
         repeat(lambda: mine_conversations_rp(rp / "coser.jsonl", must_contain="portraying"), rep),
         8_000_000),
        ("limarp_aug", "rp",
         repeat(lambda: mine_conversations_rp(rp / "limarp_aug.jsonl"), rep), 4_000_000),
        # 风格/口癖约束合成数据：教「照设定里写的格式要求说话」这个元能力
        ("style_rp", "rp",
         lambda: mine_roleplay_zh_gpt4(rp / "style_rp.jsonl"), args.style_tokens),
        # ---------------- literature: movie dialogue + classical drama RP ----------------
        ("cornell_rp", "rp",
         lambda: iter_literature_rp(Path(args.lit_dir) / "cornell_rp.jsonl"),
         args.cornell_tokens),
        ("wikisource_zh_rp", "rp",
         lambda: iter_literature_rp(Path(args.lit_dir) / "wikisource_zh_rp.jsonl"),
         args.wikisource_zh_tokens),
        ("wikisource_en_rp", "rp",
         lambda: iter_literature_rp(Path(args.lit_dir) / "wikisource_en_rp.jsonl"),
         args.wikisource_en_tokens),
        # ---------------- Shakespeare: 400-year-old stage dialogue ----------------
        ("gutenberg_rp", "rp",
         lambda: iter_literature_rp(Path(args.lit_dir) / "gutenberg_rp.jsonl"),
         args.gutenberg_tokens),
        # ---------------- novels: role-play voice ----------------
        ("webnovel_cn", "novel",
         lambda: iter_novel_pairs(rp / "webnovel_cn.jsonl"), args.novel_tokens),
        # ---------------- casual chit-chat (quality filtered) ----------------
        ("lccc", "casual",
         lambda: ((p, None) for p in iter_lccc(corpus / "lccc_base_train.jsonl",
                                               args.lccc_stride, 300,
                                               min_turn_chars=args.lccc_min_turn)),
         args.lccc_tokens),
        ("glm_dialogue", "casual",
         lambda: ((p, None) for p in iter_glm(corpus / "train_all_df.csv", 1, 300)),
         args.glm_tokens),
        ("coig_douban", "casual",
         lambda: ((p, None) for p in iter_sft_file(sft / "coig_douban.jsonl", 600)), 3_000_000),
        ("coig_xhs", "casual",
         lambda: ((p, None) for p in iter_sft_file(sft / "coig_xhs.jsonl", 600)), 2_000_000),
        ("smoltalk_everyday", "casual",
         lambda: ((p, None) for p in iter_sft_file(sft / "smoltalk_everyday-conversations.jsonl", 600)),
         2_000_000),
        # ---------------- instruction (scaled by --instruct_scale) ----------------
        ("firefly", "instruct",
         lambda: ((p, None) for p in iter_sft_file(firefly_path, 800)), int(18e6 * ins)),
        ("belle", "instruct",
         lambda: ((p, None) for p in iter_sft_file(sft / "belle.jsonl", 800)), int(10e6 * ins)),
        ("belle_3_5m", "instruct",
         lambda: ((p, None) for p in iter_sft_file(rp / "belle_3_5m.jsonl", 800)), int(12e6 * ins)),
        ("moss_003", "instruct",
         lambda: ((p, None) for p in iter_sft_file(rp / "moss_003.jsonl", 800)), int(12e6 * ins)),
        ("sharegpt_zh_90k", "instruct",
         lambda: ((p, None) for p in iter_sft_file(rp / "sharegpt_zh_90k.jsonl", 900)), int(12e6 * ins)),
        ("smoltalk_zh_full", "instruct",
         lambda: ((p, None) for p in iter_sft_file(rp / "smoltalk_zh_full.jsonl", 900)), int(8e6 * ins)),
        ("ultrachat", "instruct",
         lambda: ((p, None) for p in iter_sft_file(sft / "ultrachat.jsonl", 1000)), int(8e6 * ins)),
        ("sharegpt4", "instruct",
         lambda: ((p, None) for p in iter_sft_file(sft / "sharegpt4.jsonl", 1000)), int(10e6 * ins)),
        ("zhihu_kol", "instruct",
         lambda: ((p, None) for p in iter_sft_file(sft2 / "zhihu_kol.jsonl", 900)), int(6e6 * ins)),
        ("coig_zhihu", "instruct",
         lambda: ((p, None) for p in iter_sft_file(sft / "coig_zhihu.jsonl", 1000)), int(4e6 * ins)),
        ("coig_pc", "instruct",
         lambda: ((p, None) for p in iter_sft_file(sft / "coig_coig_pc.jsonl", 1000)), int(3e6 * ins)),
        ("ruozhiba", "instruct",
         lambda: ((p, None) for p in iter_sft_file(sft / "ruozhiba.jsonl", 800)), int(2e6 * ins)),
        ("claude_multiround", "instruct",
         lambda: mine_conversations_rp(rp / "claude_rp_30k.jsonl"), int(6e6 * ins)),
    ]
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_cache/chat_pack_v4")
    ap.add_argument("--tokenizer", default="data_cache/chat_pack_v3/bpe_tokenizer.json")
    ap.add_argument("--rp_dir", default="data_cache/roleplay")
    ap.add_argument("--sft_dir", default="data_cache/extra_sft")
    ap.add_argument("--extra_sft_dir", default="data_cache/extra_sft2")
    ap.add_argument("--corpus_dir", default="data_cache/chat_corpus")
    ap.add_argument("--lit_dir", default="data_cache/literature",
                    help="mined literature outputs (scripts/mine_literature.py)")
    ap.add_argument("--pretrain_dirs", nargs="*",
                    default=["data_cache/pretrain_v2", "data_cache/pretrain_merged"])
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_docs", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=300)
    ap.add_argument("--val_max_tokens", type=int, default=2_000_000)
    ap.add_argument("--rp_expand_tokens", type=int, default=120_000_000)
    ap.add_argument("--style_tokens", type=int, default=25_000_000,
                    help="风格/口癖约束合成样本的 token 预算（scripts/build_style_rp.py 产出）")
    ap.add_argument("--cornell_tokens", type=int, default=20_000_000)
    ap.add_argument("--wikisource_zh_tokens", type=int, default=8_000_000)
    ap.add_argument("--wikisource_en_tokens", type=int, default=3_000_000)
    ap.add_argument("--gutenberg_tokens", type=int, default=3_000_000,
                    help="Shakespeare/movie-style stage dialogue RP budget")
    ap.add_argument("--rp_repeat", type=int, default=3,
                    help="how many times to cycle the (small) role-play sources")
    ap.add_argument("--instruct_scale", type=float, default=0.45,
                    help="multiplier on every instruction-data budget; <1 keeps "
                         "role-play the dominant slice")
    ap.add_argument("--novel_tokens", type=int, default=20_000_000)
    ap.add_argument("--lccc_tokens", type=int, default=20_000_000)
    ap.add_argument("--lccc_stride", type=int, default=5)
    ap.add_argument("--lccc_min_turn", type=int, default=8)
    ap.add_argument("--glm_tokens", type=int, default=10_000_000)
    ap.add_argument("--replay_tokens", type=int, default=30_000_000)
    ap.add_argument("--replay_spans", type=int, default=3000)
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

    for name, kind, factory, budget in build_specs(args):
        t0, used, docs = time.time(), 0, 0
        try:
            it = factory()
            first_probe = True
        except FileNotFoundError:
            print(f"[skip] {name}: missing file", flush=True)
            continue
        batch: List[Sample] = []
        stop = False
        while not stop and used < budget:
            batch.clear()
            try:
                for _ in range(args.batch_docs):
                    batch.append(next(it))
            except StopIteration:
                stop = True
            except FileNotFoundError:
                if first_probe:
                    print(f"[skip] {name}: missing file", flush=True)
                break
            except Exception as exc:
                print(f"[warn] {name}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                break
            first_probe = False
            if not batch:
                break
            # 碎句过滤：任何一条回复像半截话就整条丢弃
            batch = [b for b in batch
                     if not any(is_fragment(a) for _, a in b[0])]
            if not batch:
                continue
            dialogues = [b[0] for b in batch]
            systems = [b[1] for b in batch]
            for ids, codes in encode_dialogues_batch(tok, dialogues,
                                                     max_len=args.seq_len,
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
        print(f"[{kind:8}] {name:<20} tokens={used:>12,} docs={docs:>8,} "
              f"({stats[name]['sec']}s)", flush=True)

    # ------------------- plain-text replay (int16 or int64 shards) -------------------
    if args.replay_tokens > 0:
        t0 = time.time()
        files: List[Path] = []
        for d in args.pretrain_dirs:
            p = Path(d)
            if not p.exists():
                continue
            files.extend(sorted(x for x in p.glob("*.npy") if "_val" not in x.name))
            if (p / "train.ids.npy").exists():
                files.append(p / "train.ids.npy")
        taken = 0
        if files:
            per_file = args.replay_tokens // len(files) + 1
            chunk = args.seq_len
            for path in files:
                if taken >= args.replay_tokens:
                    break
                src = np.load(str(path), mmap_mode="r")
                spans = max(1, min(args.replay_spans // len(files) + 1, per_file // chunk))
                span_tokens = max(chunk, (per_file // spans) // chunk * chunk)
                starts = np.linspace(0, max(0, src.shape[0] - span_tokens - 1), spans,
                                     dtype=np.int64)
                got = 0
                for start in starts:
                    if taken >= args.replay_tokens or got >= per_file:
                        break
                    span = np.asarray(src[start:start + span_tokens], dtype=np.int32)
                    for off in range(0, span.shape[0] - chunk + 1, chunk):
                        train.add(span[off:off + chunk].tolist(), [W_PLAIN] * chunk)
                        taken += chunk
                        got += chunk
                        if taken >= args.replay_tokens or got >= per_file:
                            break
        stats["pretrain_replay"] = {"kind": "plain", "tokens": int(taken),
                                    "docs": len(files), "sec": round(time.time() - t0, 1)}
        print(f"[plain   ] pretrain_replay      tokens={taken:>12,} "
              f"files={len(files)} ({stats['pretrain_replay']['sec']}s)", flush=True)

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
    meta = {
        "seq_len": args.seq_len,
        "dtype": str(train_ids.dtype),
        "train_rows": int(train_ids.shape[0]),
        "val_rows": int(val_ids.shape[0]),
        "train_tokens": int(train_ids.size),
        "reply_tokens": int((train_codes == 1).sum()),
        "prompt_tokens": int((train_codes == 2).sum()),
        "plain_tokens": int((train_codes == 3).sum()),
        "pad_tokens": int((train_codes == 0).sum()),
        "by_kind": by_kind,
        "sources": stats,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
