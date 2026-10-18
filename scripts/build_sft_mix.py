"""构建新旧混合的 SFT 语料（packed 格式，保留对话段的 reply/prompt 掩码）。

为什么要"混合"而不是"先训新数据、再训旧对话"：
  顺序训练的最后一段会主导模型的行为。如果只拿旧的 chat_pack_v13 做退火，
  模型会重新过拟合到那 552M token（前五轮的老毛病）；如果只拿新语料，对话
  能力又拉不回来。混合采样能让每个 batch 都同时看到两种分布。

为什么用 packed 而不是 flat：
  flat 分片的 wcode 恒为 1，会把用户提问也当回答来算 loss。packed 保留了
  chat_pack_v13 原有的 wcode（回复/prompt/纯文本），新语料则标成 1（与 pretrain
  阶段已有的处理一致）。

比例：旧对话 40% / 代码 35% / 教材 15% / 数学 10%。

用法：
    python scripts/build_sft_mix.py --out data_cache/hf_pack_sft --rows 300000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

W = 2049                     # 行宽 = seq_len + 1


def log(m: str) -> None:
    print(m, flush=True)


def flat_rows(shards: list[Path], need: int, rng: np.random.Generator) -> np.ndarray:
    """从 flat 分片里随机切出 need 行（每行 W 个连续 token）。"""
    if not shards or need <= 0:
        return np.zeros((0, W), dtype=np.int16)
    # 先把每个分片的总行数算出来，再按长度加权分配
    lens = []
    for p in shards:
        a = np.load(p, mmap_mode="r")
        lens.append(len(a) // W)
    total = sum(lens)
    if total == 0:
        return np.zeros((0, W), dtype=np.int16)
    out = np.empty((need, W), dtype=np.int16)
    filled = 0
    while filled < need:
        k = min(need - filled, 4096)
        # 按分片长度加权随机选分片，再在分片内随机选行
        picks = rng.choice(len(shards), size=k, p=np.asarray(lens) / total)
        for i, pi in enumerate(picks):
            if filled + i >= need:
                break
            a = np.load(shards[pi], mmap_mode="r")
            n = lens[pi]
            r = int(rng.integers(0, n))
            out[filled + i] = np.asarray(a[r * W:(r + 1) * W], dtype=np.int16)
        filled += k
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_cache/hf_pack_sft")
    ap.add_argument("--rows", type=int, default=300000, help="总行数（每行 2049 token）")
    ap.add_argument("--chat", default="data_cache/chat_pack_v13")
    ap.add_argument("--code", default="data_cache/hf_pack_code2")
    ap.add_argument("--textbook", default="data_cache/hf_pack_textbook")
    ap.add_argument("--math", default="data_cache/hf_pack_math")
    ap.add_argument("--w-chat", type=float, default=0.40)
    ap.add_argument("--w-code", type=float, default=0.35)
    ap.add_argument("--w-textbook", type=float, default=0.15)
    ap.add_argument("--w-math", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    shares = {"chat": args.w_chat, "code": args.w_code,
              "textbook": args.w_textbook, "math": args.w_math}
    s = sum(shares.values())
    shares = {k: v / s for k, v in shares.items()}
    counts = {k: int(args.rows * v) for k, v in shares.items()}
    log(f"[mix] 目标 {args.rows} 行；比例 " +
        ", ".join(f"{k}={v:.2f}({counts[k]}行)" for k, v in shares.items()))

    parts_ids, parts_wc, used = [], [], []

    # --- 旧对话：直接取 packed 行，wcode 原样保留 ---
    chat = ROOT / args.chat
    if (chat / "train.ids.npy").exists() and counts["chat"] > 0:
        ids = np.load(chat / "train.ids.npy", mmap_mode="r")
        wc = np.load(chat / "train.wcode.npy", mmap_mode="r")
        n = ids.shape[0]
        take = min(counts["chat"], n)
        sel = rng.choice(n, size=take, replace=False)
        sel.sort()
        parts_ids.append(np.asarray(ids[sel], dtype=np.int16))
        used.append("chat")
        parts_wc.append(np.asarray(wc[sel], dtype=np.uint8))
        log(f"[chat] 取 {take} 行（共 {n} 行，wcode 保留）")
    else:
        log(f"[chat] 跳过（{chat} 不存在）")

    # --- 新语料：flat 分片切成行，wcode=1（与 pretrain 阶段一致）---
    for key in ("code", "textbook", "math"):
        d = ROOT / {"code": args.code, "textbook": args.textbook, "math": args.math}[key]
        shards = sorted(d.glob("shard_*.npy")) if d.exists() else []
        if not shards or counts[key] <= 0:
            log(f"[{key}] 跳过（{d} 没有分片）")
            continue
        rows = flat_rows(shards, counts[key], rng)
        parts_ids.append(rows)
        parts_wc.append(np.ones(rows.shape, dtype=np.uint8))
        used.append(key)
        log(f"[{key}] 取 {rows.shape[0]} 行（来自 {len(shards)} 个分片）")

    if not parts_ids:
        raise SystemExit("没有任何数据源可用")

    ids = np.concatenate(parts_ids, axis=0)
    wc = np.concatenate(parts_wc, axis=0)
    perm = rng.permutation(ids.shape[0])          # 打散，避免同源连续
    ids = ids[perm]
    wc = wc[perm]
    log(f"[out] 合计 {ids.shape[0]} 行 = {ids.shape[0] * 2048 / 1e9:.3f}B token")

    np.save(out / "train.ids.npy", ids)
    np.save(out / "train.wcode.npy", wc)
    # 验证集沿用 chat 的（只用于观察对话能力，不进训练）
    for name in ("val.ids.npy", "val.wcode.npy"):
        src = chat / name
        if src.exists():
            np.save(out / name, np.load(src))

    meta = {"rows": int(ids.shape[0]), "seq_len": 2048,
            "tokens": int(ids.shape[0] * 2048),
            "mix": {k: round(v, 3) for k, v in shares.items()},
            # 用实际用到的名字，而不是按固定顺序猜——缺源时标签会错位
            "sources": {k: int(parts_ids[i].shape[0]) for i, k in enumerate(used)}}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    log(f"[out] 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
