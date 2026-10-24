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
    args = ap.parse_args(argv)

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
    tot_loss = 0.0
    tot_tok = 0
    per_file = []
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
                out = model(input_ids=x, labels=y)
                n = y.numel()
                fl += float(out.loss) * n
                ft += n
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
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[out] 追加到 {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
