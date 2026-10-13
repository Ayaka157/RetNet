"""Decode-path optimisation benchmark.

Single-sequence decode is latency bound: one 272M fp32 weight read (~1.09 GB)
is ~2.4 ms of real work spread over ~200 tiny kernels per token, none of which
fill the GPU.  ``bench/fastgen.py`` shows batching fixes that for the battery,
but interactive use (``play.py``) is single-sequence by nature.  This script
measures what is left for that case:

    python scripts/bench_decode_opt.py --model_dir checkpoints/retnet_272m_rp ^
        --tokenizer data_cache/chat_pack_v13/bpe_tokenizer.json
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402
from retnet.tokenizer import BPETokenizer                # noqa: E402


def timed(fn, reps: int = 3):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="你好，请用三句话介绍一下你自己。")
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch_sizes", default="1,2,4,8,16")
    args = ap.parse_args()

    dev = "cuda"
    model = RetNetForCausalLM.from_pretrained(args.model_dir).to(dev).eval()
    tok = BPETokenizer.load(args.tokenizer)

    _bos = tok.bos_id if isinstance(tok.bos_id, int) else tok.bos_id()
    ids = [_bos] + tok.encode(args.prompt)
    x = torch.tensor([ids], device=dev, dtype=torch.long)
    print(f"模型 272M · prompt {len(ids)} token · 生成长度 {args.max_new}")

    # ---------- 1. 参考 generate（fp32，逐 token recurrent） ----------
    def ref():
        return model.generate(x, max_new_tokens=args.max_new, temperature=0.7,
                              top_k=40, top_p=0.9, repetition_penalty=1.1,
                              no_repeat_ngram_size=4, eos_id=None)

    t = timed(ref, args.reps)
    n = args.max_new
    print(f"\n[1] 参考 generate            {t*1000:8.1f} ms  "
          f"{n/t:7.1f} tok/s")

    # ---------- 2. 只测 forward 部分（不计采样 Python 开销） ----------
    states = [None] * len(model.blocks)
    with torch.inference_mode():
        logits, states = model._forward_hidden(x, mode="parallel", return_state=True)
    one = torch.tensor([[ids[-1]]], device=dev)
    st = [s.clone() for s in states]

    def fwd_only():
        tt = one
        ss = [s.clone() for s in st]
        off = len(ids)
        for _ in range(n):
            lg, ss = model._forward_hidden(tt, mode="recurrent", states=ss,
                                           offset=off, return_state=True)
            off += 1
        return lg

    t_fwd = timed(fwd_only, args.reps)
    print(f"[2] 纯 recurrent 前向        {t_fwd*1000:8.1f} ms  {n/t_fwd:7.1f} tok/s"
          f"   （占参考的 {t_fwd/t*100:.0f}%）")

    # ---------- 3. torch.compile 单序列前向 ----------
    print()
    try:
        cm = torch.compile(model, backend="inductor")
        with torch.inference_mode():
            cm._orig_mod  # noqa: B018
        def comp():
            return cm.generate(x, max_new_tokens=n, temperature=0.7, top_k=40,
                               top_p=0.9, repetition_penalty=1.1,
                               no_repeat_ngram_size=4, eos_id=None)
        t_c = timed(comp, max(2, args.reps))
        print(f"[3] + torch.compile          {t_c*1000:8.1f} ms  {n/t_c:7.1f} tok/s"
              f"   加速 {t/t_c:.2f}x")
    except Exception as exc:
        print(f"[3] torch.compile 失败: {type(exc).__name__}: {str(exc)[:200]}")

    # ---------- 3b. CUDA Graphs（reduce-overhead）----------
    print()
    try:
        cg = torch.compile(model, mode="reduce-overhead")
        def cgf():
            return cg.generate(x, max_new_tokens=n, temperature=0.7, top_k=40,
                               top_p=0.9, repetition_penalty=1.1,
                               no_repeat_ngram_size=4, eos_id=None)
        t_g = timed(cgf, max(2, args.reps))
        print(f"[3b] reduce-overhead(CUDA G)  {t_g*1000:8.1f} ms  {n/t_g:7.1f} tok/s"
              f"   加速 {t/t_g:.2f}x")
    except Exception as exc:
        print(f"[3b] CUDA Graphs 失败: {type(exc).__name__}: {str(exc)[:200]}")

    # ---------- 4. bf16 权重推理 ----------
    print()
    try:
        m16 = RetNetForCausalLM.from_pretrained(args.model_dir).to(dev).eval()
        m16 = m16.to(torch.bfloat16)
        def ref16():
            return m16.generate(x, max_new_tokens=n, temperature=0.7, top_k=40,
                                top_p=0.9, repetition_penalty=1.1,
                                no_repeat_ngram_size=4, eos_id=None)
        t16 = timed(ref16, args.reps)
        print(f"[4] bf16 权重                {t16*1000:8.1f} ms  {n/t16:7.1f} tok/s"
              f"   加速 {t/t16:.2f}x")
        out32 = model.generate(x, max_new_tokens=16, temperature=0, top_k=None,
                               top_p=None, repetition_penalty=1.0, eos_id=None)
        out16 = m16.generate(x, max_new_tokens=16, temperature=0, top_k=None,
                             top_p=None, repetition_penalty=1.0, eos_id=None)
        same = int((out32[0] == out16[0]).sum()), out32.shape[1]
        print(f"    贪婪解码前 16 token 一致率: {same[0]}/{same[1]}")
        del m16
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"[4] bf16 推理失败: {type(exc).__name__}: {str(exc)[:200]}")

    # ---------- 5. 批处理扩展 ----------
    print(f"\n[5] 批处理（独立序列，共享一次前向）")
    try:
        from bench.fastgen import BatchedDecoder, Request
    except ModuleNotFoundError as exc:
        print(f"    跳过（{exc}）")
        return 0

    def fresh(B):
        # Request 是有状态的（run() 会把 out 填满、finished 置位），
        # 复用同一批会让第二次计时几乎什么都不做 —— 必须每次重建。
        return [Request(ids=list(ids), max_new=n, temperature=0.7, top_k=40,
                        top_p=0.9, repetition_penalty=1.1, no_repeat_ngram_size=4,
                        rng_seed=1000 + i) for i in range(B)]

    for B in [int(v) for v in args.batch_sizes.split(",")]:
        try:
            dec = BatchedDecoder(model, device=dev)
            t_b = timed(lambda: dec.run(fresh(B)), max(2, args.reps))
            print(f"    B={B:<3} {t_b*1000:8.1f} ms  {B*n/t_b:8.1f} tok/s   "
                  f"单序列等效 {n/t_b:6.1f} tok/s，相对单序列 {t/t_b:5.2f}x")
        except Exception as exc:
            print(f"    B={B:<3} 失败: {type(exc).__name__}: {str(exc)[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
