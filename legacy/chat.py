"""Interactive chat with the RetNet dialogue model.

The model was pretrained on dialogue data formatted as:

    用户：你好
    助手：你好呀

Run:
    python chat.py
    python chat.py --prompt "你真好看"
"""
from __future__ import annotations

import argparse

import torch

from retnet.model import RetNetLM, strip_optimized_prefix
from retnet.tokenizer import BPETokenizer


def extract_assistant_reply(text: str) -> str:
    marker = "助手："
    if marker in text:
        text = text.split(marker, 1)[1]
    # Stop if the model starts a new user turn.
    if "用户：" in text:
        text = text.split("用户：", 1)[0]
    return text.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/retnet_100m_chat_v2/model.pt")
    p.add_argument("--tokenizer", type=str, default="data_cache/chat_recovery_mix/bpe_tokenizer.json")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else \
             ("cpu" if args.device == "auto" else args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RetNetLM(ckpt["config"]).to(device)
    model.load_state_dict(strip_optimized_prefix(ckpt["model_state"]))
    model.eval()
    tokenizer = BPETokenizer.load(args.tokenizer)

    def respond(user: str, history):
        history = history + [f"用户：{user}"]
        prompt = "\n".join(history) + "\n助手："
        ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        with torch.inference_mode():
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 eos_id=tokenizer.eos_id)
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        reply = extract_assistant_reply(text)
        return reply

    history = []
    if args.prompt:
        print("用户：", args.prompt)
        reply = respond(args.prompt, history)
        print("助手：", reply)
        return

    print("RetNet chat. 输入 quit / exit 退出。")
    while True:
        try:
            user = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue
        reply = respond(user, history)
        history.append(f"用户：{user}")
        history.append(f"助手：{reply}")
        print(f"助手：{reply}")


if __name__ == "__main__":
    main()
