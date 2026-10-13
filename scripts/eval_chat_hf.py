"""HF-format chat eval (fixed battery) for retnet_hf checkpoints."""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402
from retnet.chat import encode_prompt  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402

CASUAL = ["你真好看", "早上好！", "你喜欢什么颜色？", "我帅吗？", "我今天心情不太好"]
INSTRUCT = ["用一句话解释什么是人工智能。", "番茄炒蛋怎么做？", "北京有哪些著名的景点？",
            "把“今天天气很好”翻译成英文。", "写一首关于秋天的四行小诗。"]
ENGLISH = ["Hello! How are you?", "What is machine learning?"]
MULTITURN = [["你好", "你叫什么名字？"], ["我想去旅游", "有什么推荐的地方吗？"]]
ROLEPLAY = [
    ("你是令狐冲（出自笑傲江湖）。你性格豪爽不羁，不在意规矩，重情重义，说话直来直去。",
     ["你是谁？", "你师父对你如何？"]),
    ("你是猫娘小白，说话可爱亲切，**每句话结尾都要加“喵”**，不要忘记这个口癖。",
     ["你在干什么呀？"]),
    ("你是一位经验丰富的中医大夫，说话温和稳重，喜欢讲道理。", ["我最近老是失眠怎么办？"]),
]


def cut(text: str) -> str:
    return text.split("用户：")[0].split("助手：")[0].strip()


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tag", default="hf")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = RetNetForCausalLM.from_pretrained(args.model_dir).cuda().eval()
    tok = BPETokenizer.load(args.tokenizer)
    t0 = time.time()

    print(f"===== {args.tag} (144M RP-SFT, HF) =====")
    for title, prompts in (("闲聊", CASUAL), ("指令", INSTRUCT), ("英文", ENGLISH)):
        print(f"## {title}")
        for p in prompts:
            ids = encode_prompt(tok, [], p)
            with torch.inference_mode():
                out = model.generate(torch.tensor([ids], device="cuda"),
                                     max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature, top_k=args.top_k,
                                     top_p=args.top_p, repetition_penalty=1.1,
                                     no_repeat_ngram_size=4, eos_id=tok.eos_id)
            r = cut(tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True))
            print(f"  用户：{p}\n  助手：{r}", flush=True)
    print("## 多轮")
    for convo in MULTITURN:
        turns = []
        for u in convo:
            ids = encode_prompt(tok, turns, u)
            with torch.inference_mode():
                out = model.generate(torch.tensor([ids], device="cuda"),
                                     max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature, top_k=args.top_k,
                                     top_p=args.top_p, repetition_penalty=1.1,
                                     no_repeat_ngram_size=4, eos_id=tok.eos_id)
            r = cut(tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True))
            print(f"  用户：{u}\n  助手：{r}", flush=True)
            turns.append((u, r))
    print("## 角色扮演")
    for persona, convo in ROLEPLAY:
        turns = []
        print(f"  设定：{persona[:50]}…")
        for u in convo:
            ids = encode_prompt(tok, turns, u, system=persona)
            with torch.inference_mode():
                out = model.generate(torch.tensor([ids], device="cuda"),
                                     max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature, top_k=args.top_k,
                                     top_p=args.top_p, repetition_penalty=1.1,
                                     no_repeat_ngram_size=4, eos_id=tok.eos_id)
            r = cut(tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True))
            print(f"  用户：{u}\n  助手：{r}", flush=True)
            turns.append((u, r))
    print(f"(elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
