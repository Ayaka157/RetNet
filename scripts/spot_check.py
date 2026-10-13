"""Mid-training spot check: small fixed battery, runs on CPU to avoid stealing
VRAM from a running training process.

    python scripts/spot_check.py \
        --checkpoint checkpoints/retnet_100m_pretrain_v4/last_model.pt \
        --tokenizer data_cache/chat_pack_v4/bpe_tokenizer.json \
        --tag pretrain_v4_step115000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.chat import encode_prompt  # noqa: E402
from retnet.model import RetNetLM, strip_optimized_prefix  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402

CASUAL = ["你真好看", "早上好！", "我最近在学做菜，有什么简单的菜推荐？"]
INSTRUCT = ["番茄炒蛋怎么做？", "把“今天天气很好”翻译成英文。"]
ROLEPLAY = [
    ("你是令狐冲（出自笑傲江湖）。你性格豪爽不羁，不在意规矩，重情重义，说话直来直去。",
     "你是谁？"),
    ("你是猫娘小白，说话可爱亲切，**每句话结尾都要加“喵”**，不要忘记这个口癖。",
     "你在干什么呀？"),
]


def fix_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def extract(text: str) -> str:
    return text.split("用户：")[0].split("助手：")[0].strip()


def main():
    fix_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tag", default="spot")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RetNetLM(ck["config"]).to(device)
    model.load_state_dict(strip_optimized_prefix(ck.get("model_state", ck)))
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)
    step = ck.get("step", "?")

    t0 = time.time()
    print(f"===== {args.tag} (step={step}, device=cpu) =====")
    for label, prompts in (("闲聊", CASUAL), ("指令", INSTRUCT)):
        for p in prompts:
            ids = encode_prompt(tok, [], p)
            x = torch.tensor([ids], device=device)
            with torch.inference_mode():
                out = model.generate(x, max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature, top_k=args.top_k,
                                     top_p=args.top_p,
                                     repetition_penalty=args.repetition_penalty,
                                     no_repeat_ngram_size=4, eos_id=tok.eos_id)
            r = extract(tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True))
            print(f"[{label}] 用户：{p}")
            print(f"  助手：{r}")
            print("", flush=True)
    for persona, q in ROLEPLAY:
        ids = encode_prompt(tok, [], q, system=persona)
        x = torch.tensor([ids], device=device)
        with torch.inference_mode():
            out = model.generate(x, max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 top_p=args.top_p,
                                 repetition_penalty=args.repetition_penalty,
                                 no_repeat_ngram_size=4, eos_id=tok.eos_id)
        r = extract(tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True))
        print(f"[角色] 设定：{persona[:40]}…")
        print(f"  用户：{q}")
        print(f"  助手：{r}")
        print("", flush=True)
    print(f"(elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
