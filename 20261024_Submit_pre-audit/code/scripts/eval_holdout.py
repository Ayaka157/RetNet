"""在 pretrain_v2 的 held-out 分片上测损失——这是本轮「真的学到新东西」的判据。

为什么不能只看 chat-val：主验证集是**对话分布**，而 v21 的 Stage 0 训的是网页/百科，
chat-val 会先变差。域内 held-out 才是"模型有没有学会新语料"的直接证据。

held-out 来源：`data_cache/pretrain_v2/*val*.npy`（6 个分片、约 5.47M token）。
训练时 `_FlatStage` 按文件名把含 "val" 的分片排除，所以它们没进过梯度。

用法：
    # 基线（训练前的 ep3）
    python scripts/eval_holdout.py --model checkpoints/retnet_500m_v15b_ep3 --tag ep3

    # 训完的 v21
    python scripts/eval_holdout.py --model checkpoints/retnet_500m_v21 --tag v21

输出：每个分片的 loss、合计 token 数、加权平均 loss。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _token_ce(model, x, y, pos_chunk: int = 256):
    """逐 token 交叉熵，shape ``[B, L]``。

    ``model(x)`` 会返回完整 ``[B,L,V]`` logits；这里按**位置**分块转 fp32 再求
    CE，峰值是 ``pos_chunk × V × 4`` 而不是整段。求的是**不带权重码**的普通
    CE —— 探针要比较的是"同一个 token 在不同上下文下的可预测性"，
    权重码（prompt/plain）会把两种上下文之间的差掩盖掉。
    """
    import torch
    import torch.nn.functional as F
    out = model(input_ids=x)
    logits = out.logits
    B, L, V = logits.shape
    parts = []
    for s in range(0, L, pos_chunk):
        lg = logits[:, s:s + pos_chunk].reshape(-1, V).float()
        tgt = y[:, s:s + pos_chunk].reshape(-1)
        parts.append(F.cross_entropy(lg, tgt, reduction="none").view(B, -1))
    return torch.cat(parts, dim=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 格式的模型目录")
    ap.add_argument("--val-glob", default="data_cache/pretrain_v2/*val*.npy")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=128,
                    help="chunkwise 的块大小，必须与训练时一致（docs/00 N04）")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = 全部")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="logs/holdout_eval.jsonl")
    # ---- 两个探针（docs/02 §5.3）----
    ap.add_argument("--by-position", action="store_true",
                    help="把序列按位置分桶，各桶单独报 loss（看是否越往后越省）")
    ap.add_argument("--ctx-ablation", action="store_true",
                    help="对每个窗口再算一次 L(ctx)（只喂最后 ctx 个 token），"
                         "与 L(seq_len) 在最后 ctx 个位置上的 loss 相减，得到 Δ_ctx")
    ap.add_argument("--ctx", type=int, default=256,
                    help="--ctx-ablation 的短上下文长度（默认 256）")
    ap.add_argument("--pos-buckets", type=int, default=8,
                    help="--by-position 的分桶数（默认 8，对应 2048/8=256 一桶）")
    args = ap.parse_args(argv)
    # chunkwise 要求序列长度是 chunk_size 的整数倍（内核的硬约束）。
    # 两个探针会额外构造“短窗口”，所以提前把两条都校验掉，不要跑到一半才崩。
    if args.chunk_size <= 0 or args.seq_len % args.chunk_size:
        raise SystemExit(
            f"--seq-len {args.seq_len} 必须是 --chunk-size {args.chunk_size} 的整数倍")
    if args.ctx_ablation and args.ctx % args.chunk_size:
        raise SystemExit(
            f"--ctx {args.ctx} 必须是 --chunk-size {args.chunk_size} 的整数倍"
            f"（短窗口同样走 chunkwise 内核）")

    import numpy as np
    import torch

    # 本地 checkpoint 的 config.json 里没有 auto_map（只有上传到 HF 的那几份加过），
    # 所以不能走 AutoModelForCausalLM；直接用仓库里的类加载。
    from retnet_hf import RetNetForCausalLM as _ModelCls

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    t0 = time.time()
    try:
        model = _ModelCls.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    except TypeError:                     # 老版本 transformers 用 torch_dtype
        model = _ModelCls.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    # 2026-10-20（docs/00 N04/E16）：评测必须走**训练用的那条路径**。
    # 旧代码强制 mode="parallel"，理由是"parallel 与 chunkwise 数学等价"——
    # 但 E16 证明 parallel 当时多做两次归一化，两者并不等价。
    # 那条注释里还提到 chunkwise 参考实现在 bf16 下有 dtype 冲突；
    # 这里显式切到 chunkwise 并把 chunk_size 固定成训练值，
    # 若真遇到 dtype 冲突，应当修内核而不是换一条不等价的路径来测。
    model.config.mode = "chunkwise"
    model.config.chunk_size = int(args.chunk_size)
    print(f"[load] {args.model} -> {dev}/{dtype}  "
          f"mode=chunkwise chunk={args.chunk_size}（{time.time() - t0:.1f}s）",
          flush=True)

    files = sorted(glob.glob(str(ROOT / args.val_glob)))
    if not files:
        raise SystemExit(f"没有匹配到 held-out 分片：{args.val_glob}")
    print(f"[val] {len(files)} 个分片", flush=True)

    width = args.seq_len + 1
    want_probe = bool(args.by_position or args.ctx_ablation)
    tot_loss = 0.0
    tot_tok = 0
    per_file = []
    # 位置桶：pos_sums[i] / pos_cnt[i] 是所有文件累计到第 i 桶的 CE 和与 token 数
    pos_sums = torch.zeros(args.seq_len, dtype=torch.float64)
    pos_cnt = torch.zeros(args.seq_len, dtype=torch.float64)
    # Δ_ctx = mean(L(ctx) − L(seq_len))，只统计最后 ctx 个位置
    ctx_sum = 0.0
    ctx_tok = 0
    ctx_short = 0.0
    ctx_full = 0.0
    for f in files:
        arr = np.load(f, mmap_mode="r")
        n_rows = int(arr.shape[0]) // width
        if n_rows == 0:
            continue
        rows = np.arange(n_rows)
        if args.max_tokens:
            rows = rows[: max(1, args.max_tokens // args.seq_len)]
        fl = ft = 0
        with torch.inference_mode():
            for i in range(0, len(rows), args.batch):
                idx = rows[i:i + args.batch]
                chunk = np.stack([np.asarray(arr[r * width:(r + 1) * width], dtype=np.int64)
                                  for r in idx])
                x = torch.from_numpy(chunk[:, :-1]).to(dev)
                y = torch.from_numpy(chunk[:, 1:]).to(dev)
                n = y.numel()
                if not want_probe:
                    out = model(input_ids=x, labels=y)
                    fl += float(out.loss) * n
                    ft += n
                    continue
                ce = _token_ce(model, x, y)          # [B, L]
                fl += float(ce.double().sum())
                ft += n
                pos_sums += ce.double().sum(dim=0).cpu()
                pos_cnt += float(ce.shape[0])
                if args.ctx_ablation and ce.shape[1] > args.ctx:
                    c = int(args.ctx)
                    # 短窗口：只喂最后 c 个 token（位置编号从 0 重新开始，
                    # 这是 docs/02 §5.3 定义的探针，不是等价的推理方式）
                    ce_short = _token_ce(model, x[:, -c:], y[:, -c:])
                    ce_full_tail = ce[:, -c:]
                    ctx_sum += float((ce_short.double() - ce_full_tail.double()).sum())
                    ctx_short += float(ce_short.double().sum())
                    ctx_full += float(ce_full_tail.double().sum())
                    ctx_tok += int(ce_short.numel())
        if ft:
            loss = fl / ft
            per_file.append({"file": Path(f).name, "tokens": ft, "loss": loss})
            tot_loss += fl
            tot_tok += ft
            print(f"  {Path(f).name:<28} {ft:>10,} token  loss={loss:.4f}", flush=True)

    if not tot_tok:
        raise SystemExit("held-out 上一个 token 都没测到")
    avg = tot_loss / tot_tok
    print(f"\n[{args.tag}] held-out 加权平均 loss = {avg:.4f}（{tot_tok:,} token）", flush=True)

    rec = {"tag": args.tag, "model": args.model, "loss": avg, "tokens": tot_tok,
           "per_file": per_file, "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    if args.by_position:
        L = args.seq_len
        n_b = max(1, int(args.pos_buckets))
        edges = [round(i * L / n_b) for i in range(n_b + 1)]
        buckets = []
        print(f"[by-position] {n_b} 桶：", flush=True)
        for b in range(n_b):
            a, z = edges[b], edges[b + 1]
            c = float(pos_cnt[a:z].sum())
            if c <= 0:
                continue
            loss = float(pos_sums[a:z].sum()) / c
            buckets.append({"range": [a, z - 1], "tokens": int(c), "loss": loss})
            print(f"  [{a:>4}-{z - 1:<4}] {int(c):>10,} token  loss={loss:.4f}",
                  flush=True)
        rec["by_position"] = buckets

    if args.ctx_ablation:
        if not ctx_tok:
            print(f"[ctx-ablation] 序列长度 {args.seq_len} 不大于 --ctx {args.ctx}，跳过",
                  flush=True)
        else:
            delta = ctx_sum / ctx_tok
            l_short = ctx_short / ctx_tok
            l_full = ctx_full / ctx_tok
            rec["ctx_ablation"] = {"ctx": int(args.ctx), "seq_len": args.seq_len,
                                   "loss_short": l_short, "loss_full_tail": l_full,
                                   "delta_ctx": delta, "tokens": ctx_tok}
            print(f"[ctx-ablation] L({args.seq_len}) 尾部={l_full:.4f}  "
                  f"L({args.ctx})={l_short:.4f}  "
                  f"Δ_ctx = {delta:+.4f}（{ctx_tok:,} token；正数=远距上下文有用）",
                  flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[out] 追加到 {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
