"""Fixed-battery chat evaluation for RetNet checkpoints.

Runs the same prompt set against one or more checkpoints and writes a Markdown
report so different training stages can be compared side by side.

Example:
    python scripts/eval_chat.py \
        --checkpoint checkpoints/retnet_100m_chat_sft/model.pt \
        --tokenizer data_cache/chat_sft_mix/bpe_tokenizer.json \
        --tag chat_sft
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.chat import encode_prompt  # noqa: E402
from retnet.model import RetNetLM, strip_optimized_prefix  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402

# ---------------------------------------------------------------- prompt sets
CASUAL = [
    "你真好看",
    "早上好！",
    "你喜欢什么颜色？",
    "你爱看书吗？",
    "在干什么呢",
    "我今天心情不太好",
    "晚安",
    "你是谁？",
    "哈哈哈哈笑死我了",
    "我帅吗？",
]

INSTRUCT = [
    "用一句话解释什么是人工智能。",
    "推荐三本适合入门编程的书。",
    "番茄炒蛋怎么做？",
    "帮我写一句祝朋友生日快乐的话。",
    "北京有哪些著名的景点？",
    "为什么天空是蓝色的？",
    "把“今天天气很好”翻译成英文。",
    "写一首关于秋天的四行小诗。",
    "感冒了应该注意什么？",
    "怎么才能早睡早起？",
]

ENGLISH = [
    "Hello! How are you?",
    "What is machine learning?",
    "Tell me a short joke.",
]

MULTITURN = [
    ["你好", "你叫什么名字？"],
    ["我想去旅游", "有什么推荐的地方吗？"],
    ["我最近在学做菜", "有什么简单的菜推荐？"],
]

# 角色扮演：设定 + 连续对话，检查是否「按人设说话」
ROLEPLAY = [
    ("你是令狐冲（出自笑傲江湖）。你性格豪爽不羁，不在意规矩，重情重义，说话直来直去。",
     ["你是谁？", "你师父对你如何？"]),
    ("你是雪之下雪乃，性格高冷、傲娇、说话直接带刺，但其实很关心朋友。",
     ["早上好！", "你觉得我怎么样？"]),
    ("你是猫娘小白，说话可爱、句尾常带“喵”，喜欢撒娇。",
     ["你在干什么呀？", "陪我聊聊天好吗？"]),
    ("你是一位经验丰富的中医大夫，说话温和稳重，喜欢讲道理。",
     ["我最近老是失眠怎么办？"]),
]


def fix_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def extract_reply(text: str) -> str:
    """Cut a freshly generated continuation down to the first assistant turn.

    The model happily keeps inventing further 用户/助手 turns, so everything from
    the next role marker onwards is dropped.
    """
    text = text.split("用户：")[0]
    text = text.split("助手：")[0]
    return text.strip()


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state = strip_optimized_prefix(ckpt.get("model_state", ckpt))
    model = RetNetLM(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, int(ckpt.get("step", -1)), ckpt.get("val_loss")


@torch.inference_mode()
def reply(model, tok, history, user, args, device, system=None):
    ids = encode_prompt(tok, history, user, system=system)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    out = model.generate(x, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k,
                         top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                         no_repeat_ngram_size=args.no_repeat_ngram_size,
                         penalize_prompt=args.penalize_prompt,
                         eos_id=tok.eos_id)
    new_tokens = out[0].tolist()[len(ids):]
    return extract_reply(tok.decode(new_tokens, skip_special_tokens=True))


def main():
    fix_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tag", default=None, help="name used in the report file")
    ap.add_argument("--out_dir", default="logs/chat_eval")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--penalize_prompt", action="store_true",
                    help="also penalize prompt tokens (off by default: it pushes "
                         "replies away from the words of the question)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, cfg, step, val_loss = load_model(args.checkpoint, device)
    tok = BPETokenizer.load(args.tokenizer)

    tag = args.tag or Path(args.checkpoint).parent.name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.md"

    lines = [f"# Chat eval: {tag}", ""]
    lines.append(f"- checkpoint: `{args.checkpoint}` (step={step}, val_loss={val_loss})")
    lines.append(f"- tokenizer: `{args.tokenizer}`")
    lines.append(f"- model: {cfg.num_layers} layers, d_model={cfg.d_model}, heads={cfg.num_heads}")
    lines.append(f"- sampling: T={args.temperature} top_k={args.top_k} top_p={args.top_p} "
                 f"rep={args.repetition_penalty} max_new={args.max_new_tokens}")
    lines.append("")

    t0 = time.time()
    for title, prompts in (("闲聊 (casual)", CASUAL),
                           ("指令 (instruct)", INSTRUCT),
                           ("English", ENGLISH)):
        lines.append(f"## {title}")
        lines.append("")
        for p in prompts:
            r = reply(model, tok, [], p, args, device)
            print(f"[{title}] 用户：{p}\n  助手：{r}", flush=True)
            lines.append(f"**用户：** {p}")
            lines.append("")
            lines.append(f"**助手：** {r}")
            lines.append("")

    lines.append("## 多轮 (multi-turn)")
    lines.append("")
    for convo in MULTITURN:
        turns = []
        block = []
        for user in convo:
            r = reply(model, tok, turns, user, args, device)
            turns.append((user, r))
            block.append(f"- **用户：** {user}")
            block.append(f"- **助手：** {r}")
        print("\n".join(block), flush=True)
        lines.extend(block)
        lines.append("")

    lines.append("## 角色扮演 (role-play, 带设定)")
    lines.append("")
    for persona, convo in ROLEPLAY:
        turns = []
        block = [f"**设定：** {persona}", ""]
        for user in convo:
            r = reply(model, tok, turns, user, args, device, system=persona)
            turns.append((user, r))
            block.append(f"- **用户：** {user}")
            block.append(f"- **助手：** {r}")
        print(f"[RP] 设定：{persona}\n" + "\n".join(block[2:]), flush=True)
        lines.extend(block)
        lines.append("")

    lines.append(f"_generated in {time.time() - t0:.1f}s_")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
