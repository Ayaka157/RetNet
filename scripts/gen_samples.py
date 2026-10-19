"""对最终模型做生成采样，留下"能不能聊天"的直接证据。

为什么需要：整个过程只有 loss / 困惑度这类**代理指标**，没有任何人工可读的
生成样本。loss 降低不等于会聊天——尤其这一轮前 145,857 步完全没有对话数据。

用法：
    python scripts/gen_samples.py --model checkpoints/retnet_500m_v21 --tag v21_final
    python scripts/gen_samples.py --model checkpoints/retnet_500m_v22 --tag v22_ctx16k

输出：logs/samples_<tag>.txt（同时追加一份到 logs/samples_all.txt）
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
# 从 scripts/ 下运行时，仓库根不在 sys.path 里，补上
_ROOT = _P(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import time
from pathlib import Path

import torch

PROMPTS = [
    # 对话
    ("对话·常识", "用户：用一句话解释为什么天空是蓝色的。\n助手："),
    ("对话·建议", "用户：我最近总是睡不好，有什么办法？\n助手："),
    # 代码
    ("代码·Python", "用户：写一个 Python 函数，判断一个字符串是不是回文。\n助手："),
    ("代码·SQL", "用户：写一条 SQL，查出每个部门工资最高的员工。\n助手："),
    # 数学
    ("数学", "用户：一个数的三倍加上 7 等于 25，这个数是多少？请写出过程。\n助手："),
    # 中文常识
    ("知识", "用户：介绍一下长江。\n助手："),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokenizer", default="data_cache/chat_pack_v13/bpe_tokenizer.json")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    from retnet_hf import RetNetForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)

    t0 = time.time()
    model = RetNetForCausalLM.from_pretrained(args.model).to(device).eval()
    load_s = time.time() - t0
    # 某些导出会带 bf16 权重；生成用 fp32 更稳，且不占多少显存
    if next(model.parameters()).dtype != torch.float32:
        model = model.float()

    out = [f"模型: {args.model}",
           f"标签: {args.tag}",
           f"参数: max_new={args.max_new} T={args.temperature} top_p={args.top_p} rp={args.repetition_penalty}",
           f"加载耗时: {load_s:.1f}s",
           "=" * 72, ""]

    # 这个 tokenizer 没有 <|endoftext|> 这个特殊 token，训练时的 eos 是 id=2
    # （见 build_flat_corpus 的日志：[tok] vocab=20000 eos=2）
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = 2
    for i, (cat, prompt) in enumerate(PROMPTS, 1):
        ids = tok.encode(prompt).ids
        ids = ids[-2048:] if len(ids) > 2048 else ids
        inp = torch.tensor([ids], device=device)
        t = time.time()
        with torch.no_grad():
            seq = model.generate(
                inp, max_new_tokens=args.max_new,
                temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_id=eos_id, device=device,
            )
        gen = tok.decode(seq[0, inp.shape[1]:].tolist())
        # 模型会一直编造后续的 用户/助手 轮，截到第一个轮次边界
        for stop in ("\n用户：", "用户：", "\n助手："):
            k = gen.find(stop)
            if k > 0:
                gen = gen[:k]
        dt = time.time() - t
        out.append(f"[{i}/{len(PROMPTS)}] {cat}")
        out.append(f"PROMPT> {prompt}")
        out.append(f"OUTPUT> {gen.strip()}")
        out.append(f"({dt:.1f}s, {args.max_new / max(dt, 1e-6):.1f} tok/s)")
        out.append("-" * 72)
        print(f"  [{i}/{len(PROMPTS)}] {cat}  {dt:.1f}s")

    out.append("")
    Path("logs").mkdir(exist_ok=True)
    p = Path(f"logs/samples_{args.tag}.txt")
    p.write_text("\n".join(out), encoding="utf-8")
    with open("logs/samples_all.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n\n")
    print(f"  已写 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
