"""RetNet 推理小玩具：支持中文聊天 / 英文续写，命令行交互。

用法示例（PowerShell）:

    # 中文聊天（默认加载 100M 对话模型）
    python play.py --mode chat
    python play.py --mode chat --prompt "你真好看"

    # 英文续写（默认加载 97.6M Wikitext-103 模型）
    python play.py --mode text --preset en
    python play.py --mode text --preset en --prompt "The history of the English language"

参数都可不写，脚本会给出交互提示。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from retnet.chat import encode_prompt
from retnet.model import RetNetLM, strip_optimized_prefix
from retnet.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent

PRESETS = {
    "chat": {
        "checkpoint": ROOT / "checkpoints/retnet_100m_rp_v2/model.pt",
        "tokenizer": ROOT / "data_cache/chat_pack_v5/bpe_tokenizer.json",
        "name": "RetNet-100M 角色扮演对话模型 v2 (rp_v2)",
        "chat": True,
    },
    "rp_v1": {
        "checkpoint": ROOT / "checkpoints/retnet_100m_rp/model.pt",
        "tokenizer": ROOT / "data_cache/chat_pack_v4/bpe_tokenizer.json",
        "name": "RetNet-100M 角色扮演对话模型 v1",
        "chat": True,
    },
    "chat_v3": {
        "checkpoint": ROOT / "checkpoints/retnet_100m_chat_v3/model.pt",
        "tokenizer": ROOT / "data_cache/chat_pack_v3/bpe_tokenizer.json",
        "name": "RetNet-100M 通用对话模型 v3 (packed SFT)",
        "chat": True,
    },
    "sft": {
        "checkpoint": ROOT / "checkpoints/retnet_100m_chat_sft/model.pt",
        "tokenizer": ROOT / "data_cache/chat_sft_mix/bpe_tokenizer.json",
        "name": "RetNet-100M 对话模型 (第一版 SFT)",
        "chat": True,
    },
    "chat_v2": {
        "checkpoint": ROOT / "checkpoints/retnet_100m_chat_v2/model.pt",
        "tokenizer": ROOT / "data_cache/chat_recovery_mix/bpe_tokenizer.json",
        "name": "RetNet-100M 对话恢复模型 v2",
        "chat": True,
    },
    "en": {
        "checkpoint": ROOT / "checkpoints/retnet_97m_wikitext103/model.pt",
        "tokenizer": ROOT / "data_cache/wikitext103_v16k/bpe_tokenizer.json",
        "name": "RetNet-97.6M 英文基础模型",
        "chat": False,
    },
}


def fix_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def extract_assistant_reply(text: str) -> str:
    """Cut a fresh continuation down to the first assistant turn."""
    text = text.split("用户：")[0]
    text = text.split("助手：")[0]
    return text.strip()


def load_model(checkpoint, device):
    if not Path(checkpoint).exists():
        raise SystemExit(f"找不到 checkpoint: {checkpoint}\n先跑训练，或手动指定 --checkpoint。")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state = ckpt.get("model_state", ckpt)
    model = RetNetLM(cfg).to(device)
    # 允许从完整训练 checkpoint 或纯权重 checkpoint 加载
    if isinstance(state, dict):
        model.load_state_dict(strip_optimized_prefix(state))
    model.eval()
    return model, cfg


def sample_text(model, tokenizer, prompt, args, device):
    ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    return _generate(model, tokenizer, ids, args, device)


def _generate(model, tokenizer, ids, args, device):
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            penalize_prompt=args.penalize_prompt,
            eos_id=tokenizer.eos_id,
        )
    return tokenizer.decode(out[0].tolist()[len(ids):], skip_special_tokens=True)


def chat_reply(model, tokenizer, history, user, args, device, system=None):
    """history: list of (user, assistant) pairs, encoded exactly like training."""
    ids = encode_prompt(tokenizer, history, user, system=system)
    return extract_assistant_reply(_generate(model, tokenizer, ids, args, device))


def main():
    fix_console()
    ap = argparse.ArgumentParser(description="RetNet 推理小玩具")
    ap.add_argument("--mode", choices=["chat", "text"], default="chat")
    ap.add_argument("--preset", choices=list(PRESETS), default="chat",
                    help="chat=角色扮演对话模型；chat_v3=通用对话；en=英文基础模型")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--system", default=None,
                    help="角色设定，例如「你是令狐冲（出自笑傲江湖），豪爽不羁、重情重义」；"
                         "交互模式下也可以随时输入 /set <设定> 更换角色")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9,
                    help="nucleus sampling；设 1.0 表示关闭")
    ap.add_argument("--repetition_penalty", type=float, default=1.1,
                    help="1.0 表示不惩罚重复")
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4,
                    help="禁止重复的 n-gram 长度，0 表示关闭")
    ap.add_argument("--penalize_prompt", action="store_true",
                    help="同时惩罚提问里的词（默认关闭：会让回答躲开问题本身的词）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    if args.mode == "text" and args.preset == "chat":
        # 文本模式也可以强行使用聊天模型，不报错，仅提示。
        pass
    checkpoint = Path(args.checkpoint) if args.checkpoint else preset["checkpoint"]
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else preset["tokenizer"]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else \
             ("cpu" if args.device == "auto" else args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if not tokenizer_path.exists():
        raise SystemExit(f"找不到 tokenizer: {tokenizer_path}")
    model, cfg = load_model(checkpoint, device)
    tokenizer = BPETokenizer.load(tokenizer_path)

    print(f"模型：{preset['name']}")
    print(f"配置：{cfg.num_layers} 层, d_model={cfg.d_model}, heads={cfg.num_heads}, "
          f"vocab={cfg.vocab_size}")
    print(f"设备：{device} | temperature={args.temperature} | top_k={args.top_k} | "
          f"top_p={args.top_p} | rep_penalty={args.repetition_penalty} | "
          f"max_new_tokens={args.max_new_tokens}")
    print("-" * 50)

    if args.mode == "text":
        if args.prompt:
            print(sample_text(model, tokenizer, args.prompt, args, device))
            return
        print("英文续写模式。输入 quit / exit 退出。")
        while True:
            try:
                prompt = input("Prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in {"quit", "exit"}:
                break
            if not prompt:
                continue
            print("-" * 50)
            print(sample_text(model, tokenizer, prompt, args, device))
            print("-" * 50)
        return

    # chat mode
    system = args.system
    if args.prompt:
        if system:
            print(f"设定：{system}")
        print(f"用户：{args.prompt}")
        print(f"助手：{chat_reply(model, tokenizer, [], args.prompt, args, device, system=system)}")
        return

    print("聊天模式。quit/exit 退出，clear 清空上下文，/set <设定> 设置角色，/set 清空角色。")
    if system:
        print(f"当前设定：{system}")
    history = []
    while True:
        try:
            user = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        if user.lower() in {"clear", "reset"}:
            history = []
            print("(上下文已清空)")
            continue
        if user.startswith("/set"):
            system = user[4:].strip() or None
            history = []
            print(f"(角色设定已{'更新为：' + system if system else '清空'})")
            continue
        if not user:
            continue
        reply = chat_reply(model, tokenizer, history, user, args, device, system=system)
        print(f"助手：{reply}")
        history.append((user, reply))
        # 只保留最近 4 轮，避免超出模型训练时的 512 token 上下文
        history = history[-4:]


if __name__ == "__main__":
    main()
