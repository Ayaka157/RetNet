"""构造 v22 的 16K 上下文扩展用混合语料。

为什么需要它：
  v22 原本只用 `hf_pack_code2`（纯代码），问题是模型刚跑完 68,557 步代码，
  再用纯代码做长上下文扩展会把它进一步推向代码、把 SFT 段刚恢复的对话能力推远。
  混入 25% 对话 + 少量数学/编程，可以边学长窗口边把对话分布锚住。

配比（按 token）：
  code2 60%  —— 有跨行/跨函数的真实长程结构
  sft   25%  —— 对话 + 代码 + 教材 + 数学，保住 SFT 段的成果
  math  10%
  prog   5%

关于文档边界：flat 分片的原始边界在分词时就丢了（`--add-eos` 默认关闭），
无法恢复；但**从 SFT packed 包转过来的部分可以插入 eos**（它有 wcode，
pad=0 的行尾可以直接剥掉并补 eos）。所以这个包比纯 flat 略好一点。

用法：
    python scripts/build_ctx_mix.py --out data_cache/hf_pack_ctx16k --tokens 500000000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

SHARD_TOKENS = 250_000_000
EOS = 2


def log(msg: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iter_flat(d: Path):
    """逐个 flat 分片产出 int16 数组。"""
    for f in sorted(d.glob("shard_*.npy")):
        arr = np.load(f, mmap_mode="r")
        yield np.asarray(arr, dtype=np.int16)


def iter_packed_sentences(d: Path):
    """从 packed 包产出单句 token（剥掉 pad，末尾补 eos）。

    packed 的结构是 train.ids.npy [rows, seq] + train.wcode.npy [rows, seq]，
    wcode=0 表示 pad。
    """
    ids = np.load(d / "train.ids.npy", mmap_mode="r")
    wc = np.load(d / "train.wcode.npy", mmap_mode="r")
    for r in range(ids.shape[0]):
        row = np.asarray(ids[r], dtype=np.int16)
        w = np.asarray(wc[r])
        n = int((w != 0).sum())
        if n <= 1:
            continue
        yield row[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code2", default="data_cache/hf_pack_code2")
    ap.add_argument("--sft", default="data_cache/hf_pack_sft")
    ap.add_argument("--math", default="data_cache/hf_pack_math")
    ap.add_argument("--prog", default="data_cache/hf_pack_prog")
    ap.add_argument("--out", default="data_cache/hf_pack_ctx16k")
    ap.add_argument("--tokens", type=int, default=500_000_000)
    ap.add_argument("--mix", default="code2:0.60,sft:0.25,math:0.10,prog:0.05")
    args = ap.parse_args()

    ratios = {}
    for part in args.mix.split(","):
        k, v = part.split(":")
        ratios[k.strip()] = float(v)
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "配比之和必须是 1"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard_*.npy"):
        old.unlink()

    budget = {k: int(args.tokens * v) for k, v in ratios.items()}
    log(f"目标 {args.tokens/1e9:.2f}B token，配比 {ratios}")
    for k, v in budget.items():
        log(f"  {k}: {v/1e9:.3f}B")

    # ---- 各来源的迭代器（循环供料，取够预算为止）
    def code2_src():
        while True:
            got = False
            for arr in iter_flat(Path(args.code2)):
                got = True
                yield arr
            if not got:
                return

    def sft_src():
        while True:
            got = False
            for row in iter_packed_sentences(Path(args.sft)):
                got = True
                yield np.append(row, np.int16(EOS))
            if not got:
                return

    def flat_src(d):
        def gen():
            while True:
                got = False
                for arr in iter_flat(Path(d)):
                    got = True
                    yield arr
                if not got:
                    return
        return gen()

    srcs = {
        "code2": code2_src(),
        "sft": sft_src(),
        "math": flat_src(args.math),
        "prog": flat_src(args.prog),
    }
    used = {k: 0 for k in ratios}
    shard, shard_n, n_shards = [], 0, 0
    total = 0

    def flush() -> None:
        nonlocal shard, shard_n, n_shards
        if not shard:
            return
        arr = np.concatenate(shard).astype(np.int16)
        p = out / f"shard_{n_shards:05d}.npy"
        np.save(p, arr)
        log(f"  [out] {p.name}  {len(arr)/1e6:.0f} M token  ({p.stat().st_size/1e6:.0f} MB)")
        n_shards += 1
        shard, shard_n = [], 0

    # 按比例轮流取，保证混合均匀（不是先塞完 code2 再塞对话）
    while total < args.tokens:
        progressed = False
        for name, it in srcs.items():
            if used[name] >= budget[name]:
                continue
            try:
                chunk = next(it)
            except StopIteration:
                continue
            progressed = True
            take = min(len(chunk), budget[name] - used[name], args.tokens - total)
            seg = np.asarray(chunk[:take], dtype=np.int16)
            shard.append(seg)
            shard_n += len(seg)
            used[name] += len(seg)
            total += len(seg)
            if shard_n >= SHARD_TOKENS:
                flush()
        if not progressed:
            break

    flush()
    meta = {
        "total_tokens": int(total),
        "shards": [f"shard_{i:05d}.npy" for i in range(n_shards)],
        "mode": "flat",
        "mix": ratios,
        "used": {k: int(v) for k, v in used.items()},
        "note": "v22 16K 上下文扩展用；含 25% 对话以锁住 SFT 段成果。"
                "flat 来源无文档边界（原始 --add-eos 未开），SFT 来源已补 eos。",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    log(f"完成：{total/1e9:.2f}B token，{n_shards} 个分片 -> {out}")
    for k, v in used.items():
        log(f"  {k}: {v/1e9:.3f}B ({v/max(total,1)*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
