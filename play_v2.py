"""RetNet 对话脚本 v2 —— 直接跑 retnet_hf / safetensors 检查点。

为什么需要新脚本：旧 ``play.py`` 只认手写实现保存的 ``model.pt``（100M 系列），
而 v10 / v13 / v14 全是 ``retnet_hf`` 的 HF safetensors 目录，两者不通用。

相对旧脚本补上的几件事：

* **CoT 分段展示**。训练数据里助手回复的形状是
  ``助手：<think>推理</think>正文``（从 train.ids.npy 实际解出来的，不是猜的）。
  脚本把推理段压暗，并在 ``<think>`` 没有 ``</think>`` 收尾时明确标出
  「未闭合」—— 这正是 v13 的主要失败模式。
* **流式输出**。逐 token 打印，不用等整段生成完。
* **历史存 token id，不存文本**。``<`` ``>`` ``/`` 在词表里是独立特殊 token，
  ``decode()`` 会在它们两侧插空格，所以 ``encode(decode(ids)) != ids``。
  多轮上下文直接回灌 id，才能保证与训练时的切分完全一致。
* **回合切断**。模型续写出下一轮的「用户：」/「助手：」时立即停止，
  不会把两轮话混在一起。

用法::

    python play_v2.py --preset v13                 # 交互对话
    python play_v2.py --preset v13 --prompt "你好"  # 单轮
    python play_v2.py --preset v14                 # v14 训练中，自动取最新 checkpoint-N
    python play_v2.py --list                       # 列出预设

交互中可用：/set 角色设定 · /clear 清空 · /think · /temp · /topp · /topk
· /rep · /ngram · /maxtok · /stats · /save · /load · /help · /quit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from retnet.chat import ASSISTANT, USER, system_prefix, turn_prefix  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402

# 所有 chat_pack 的 bpe_tokenizer.json 都是同一份（SHA256 abc0eafa2dfe8067741f…）
TOKENIZER = ROOT / "data_cache/chat_pack_v13/bpe_tokenizer.json"

PRESETS = {
    "v13": ("checkpoints/retnet_272m_v13", "272M · v13 终版（12500 步，2k 窗口，CoT 闭合约 86%）"),
    "v14": ("checkpoints/retnet_272m_v14", "272M · v13 续训（训练中，自动取最新 checkpoint-N）"),
    "v14_50k": ("checkpoints/retnet_v14_50k",
                "272M · v14 续训**跑完 50000 步的终版**（从笔记本拉回，基准对照用）"),
    "v10": ("checkpoints/retnet_272m_v10", "272M · v10（CoT 被剥离那版）"),
    "rp": ("checkpoints/retnet_272m_rp", "272M · v13 的起点，对话 SFT 基座"),
    "272m": ("checkpoints/retnet_272m_hf", "272M · 预训练直出，未做对话微调"),
    "143m": ("checkpoints/retnet_143m_hf", "143M · 早期对话模型"),
    # --- 0.5B（504,179,896 参数，由 v13 生长而来）---
    # resolve_model_dir 会自动退到最新的 checkpoint-N，所以训练一开始就能玩。
    "500m": ("checkpoints/retnet_500m",
             "0.5B · 生长产物（未训练，仅函数保真，输出≈272M）"),
    "500m_v15": ("checkpoints/retnet_500m_v15",
                 "0.5B · v15 训练中（自动取最新 checkpoint-N）"),
    # --- 0.5B / v15b：RTX 3090 在 AutoDL 上跑满**整个 epoch**（67,390 步 / 552 M token /
    #     11.21 h / ¥16.6）的产物。顶层 model.safetensors 就是 val-best（step 65000，
    #     sha256 21114ecc…e20ffa），checkpoint-67390 是终点步（val 次优但步数更远）。
    "500m_v15b": ("checkpoints/retnet_500m_v15b",
                  "0.5B · v15b **val-best**（step 65000，一个 epoch 的检查点里 val loss 最低）"),
    "500m_v15b_last": ("checkpoints/retnet_500m_v15b/checkpoint-67390",
                       "0.5B · v15b 终点（step 67390，即那一轮的最后一秒）"),
}

STOPS = ("\n用户：", "\n助手：", "\n用户:", "\n助手:")
OPEN, CLOSE = "<think>", "</think>"

DIM, RESET, RED, YEL = "\x1b[2m", "\x1b[0m", "\x1b[31m", "\x1b[33m"


# --------------------------------------------------------------------------- 编码

def encode_prompt_ids(tok, history, user_text, system=None):
    """与 :func:`retnet.chat.encode_prompt` 同构，但历史的回复用原始 token id。

    ``history`` 是 ``[(user_text, reply_ids), ...]``。
    """
    ids = [tok.bos_id]
    if system:
        ids.extend(tok.encode(system_prefix(system)))
    first = not system
    for prev_user, prev_ids in history:
        ids.extend(tok.encode(turn_prefix(prev_user, first=first)))
        ids.extend(prev_ids)
        first = False
    ids.extend(tok.encode(turn_prefix(user_text, first=first)))
    return ids


def find_stop(text: str) -> int:
    """返回第一个回合边界的位置，没有则 -1。"""
    best = -1
    for marker in STOPS:
        i = text.find(marker)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def split_think(text: str):
    """切成 ``(推理, 正文, 是否闭合)``；没有 <think> 时推理为 None。"""
    i = text.find(OPEN)
    if i == -1:
        return None, text.strip(), True
    j = text.find(CLOSE, i)
    if j == -1:
        return text[i + len(OPEN):].strip(), text[:i].strip(), False
    return (text[i + len(OPEN):j].strip(),
            (text[:i] + text[j + len(CLOSE):]).strip(),
            True)


# --------------------------------------------------------------------------- 模型

def resolve_model_dir(path: Path) -> Path:
    """目录里没有顶层权重时，退到最新的 ``checkpoint-N``（训练中常用）。"""
    def has_weights(d: Path) -> bool:
        return any((d / n).exists() for n in
                   ("model.safetensors", "pytorch_model.bin",
                    "model.safetensors.index.json"))

    if has_weights(path):
        return path
    subs = []
    for d in path.glob("checkpoint-*"):
        try:
            subs.append((int(d.name.rsplit("-", 1)[-1]), d))
        except ValueError:
            pass
    if subs:
        subs.sort()
        latest = subs[-1][1]
        if has_weights(latest):
            print(f"[i] {path.name} 无顶层权重，自动使用 {latest.name}")
            return latest
    return path


def load_model(model_dir: Path, device: str, dtype: str):
    if not model_dir.is_dir():
        raise SystemExit(f"找不到模型目录：{model_dir}")
    model = RetNetForCausalLM.from_pretrained(str(model_dir))
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    elif dtype == "fp32":
        model = model.to(torch.float32)
    model = model.to(device).eval()
    return model


# --------------------------------------------------------------------------- 生成

class Streamer:
    """边生成边打印：按 <think> 分段着色，遇到回合边界即停。"""

    def __init__(self, tok, out, color=True):
        self.tok = tok
        self.out = out
        self.color = color
        self.ids: list[int] = []
        self.texts: list[str] = []      # texts[i] = decode(ids[:i+1])，用于把字符位置映射回 token
        self.prev = ""
        self.inside = False
        self.cut: int | None = None

    # -- 着色输出 ----------------------------------------------------------
    def _w(self, s: str, dim: bool):
        if not s:
            return
        if self.color and dim:
            self.out.write(DIM + s + RESET)
        else:
            self.out.write(s)
        self.out.flush()

    def _emit(self, chunk: str):
        i = 0
        while i < len(chunk):
            if not self.inside:
                k = chunk.find(OPEN, i)
                if k == -1:
                    self._w(chunk[i:], False)
                    return
                self._w(chunk[i:k], False)
                self._w(OPEN, True)
                self.inside = True
                i = k + len(OPEN)
            else:
                k = chunk.find(CLOSE, i)
                if k == -1:
                    self._w(chunk[i:], True)
                    return
                self._w(chunk[i:k], True)
                self._w(CLOSE, False)
                self.inside = False
                i = k + len(CLOSE)

    # -- 回调 --------------------------------------------------------------
    def __call__(self, token_id: int) -> bool:
        self.ids.append(token_id)
        cur = self.tok.decode(list(self.ids), skip_special_tokens=False)
        self.texts.append(cur)
        cut = find_stop(cur)
        visible = cur if cut == -1 else cur[:cut]
        # 字节级 BPE 会把一个汉字劈到相邻两个 token 上，只解码前缀会得到一段
        # 未完成的字节序列，tokenizer 拿 U+FFFD 顶上。先扣住不打印，
        # 等下一个 token 把字符补齐再吐。
        while cut == -1 and visible.endswith("\ufffd"):
            visible = visible[:-1]
        if len(visible) > len(self.prev):
            self._emit(visible[len(self.prev):])
            self.prev = visible
        if cut != -1:
            self.cut = cut
            return True
        return False

    # -- 收尾 --------------------------------------------------------------
    def flush(self):
        """生成结束时补打被扣住的尾巴（最后一个字符可能一直等着下一 token）。"""
        if not self.ids:
            return
        cur = self.tok.decode(list(self.ids), skip_special_tokens=False)
        visible = cur if self.cut is None else cur[:self.cut]
        if len(visible) > len(self.prev):
            self._emit(visible[len(self.prev):])
            self.prev = visible

    def reply_ids(self) -> list[int]:
        """切掉回合边界之后的 token，得到可回灌历史的回复 id。"""
        if self.cut is None:
            return list(self.ids)
        keep = 0
        for i, s in enumerate(self.texts):
            if len(s) <= self.cut:
                keep = i + 1
            else:
                break
        return self.ids[:keep]

    @property
    def unclosed(self) -> bool:
        return self.inside


def chat_once(model, tok, history, user_text, system, args, device, out):
    budget = args.context - args.max_new_tokens
    trimmed = list(history)
    while True:
        ids = encode_prompt_ids(tok, trimmed, user_text, system=system)
        if len(ids) <= budget or not trimmed:
            break
        trimmed = trimmed[1:]          # 丢掉最老的一轮
    if len(ids) > budget:
        ids = ids[-budget:]            # 单轮就超长，只能掐头

    streamer = Streamer(tok, out, color=args.color)
    t0 = time.time()
    model.generate(
        torch.tensor([ids], device=device, dtype=torch.long),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        penalize_prompt=args.penalize_prompt,
        eos_id=tok.eos_id, device=device, on_token=streamer,
    )
    dt = time.time() - t0
    streamer.flush()
    if streamer.unclosed:
        out.write(f"  {YEL if args.color else ''}⟪<think> 未闭合⟫{RESET if args.color else ''}\n")
    out.write("\n")
    out.flush()

    raw = streamer.prev
    think, answer, closed = split_think(raw)
    n_gen = len(streamer.ids)
    return {
        "ids": streamer.reply_ids(),
        "raw": raw,
        "think": think,
        "answer": answer,
        "closed": closed,
        "prompt_tokens": len(ids),
        "gen_tokens": n_gen,
        "seconds": dt,
        "trimmed": len(history) - len(trimmed),
    }


# --------------------------------------------------------------------------- 显示

def show_result(res, args, out=sys.stdout):
    """流式已经打过一遍了；这里只在需要时补一张统计卡。"""
    if args.quiet:
        return
    tps = res["gen_tokens"] / res["seconds"] if res["seconds"] > 0 else 0.0
    bits = [f"prompt {res['prompt_tokens']} tok", f"gen {res['gen_tokens']} tok",
            f"{res['seconds']:.1f}s", f"{tps:.1f} tok/s"]
    if res["trimmed"]:
        bits.append(f"丢弃 {res['trimmed']} 轮历史")
    if not res["closed"]:
        bits.append("CoT 未闭合")
    out.write(f"  [{YEL if args.color else ''}{' · '.join(bits)}{RESET if args.color else ''}]\n\n")
    out.flush()


# --------------------------------------------------------------------------- 交互

HELP = """可用命令：
  /set <设定>          设置角色/场景（不带参数则清空），会清空上下文
  /clear               清空上下文
  /think on|off        是否显示 <think> 推理段
  /temp <float>        温度              /topp <float>   nucleus
  /topk <int>          top-k             /rep <float>    重复惩罚
  /ngram <int>         禁重复 n-gram     /maxtok <int>   单轮上限
  /stats               当前采样参数
  /save <文件>         存对话（含 token id）  /load <文件>
  /raw on|off          显示原始输出（不切回合边界）
  /help  /quit"""


def dump_state(path, history, system):
    path = Path(path)
    path.write_text(json.dumps(
        {"system": system,
         "history": [{"user": u, "reply_ids": r} for u, r in history]},
        ensure_ascii=False), encoding="utf-8")
    print(f"(已保存 {len(history)} 轮到 {path})")


def load_state(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    history = [(h["user"], list(h["reply_ids"])) for h in data.get("history", [])]
    print(f"(已载入 {len(history)} 轮，设定：{data.get('system') or '无'})")
    return history, data.get("system")


def repl(model, tok, args, device):
    system = args.system
    history = []
    print("输入内容开始对话。/help 看命令，/quit 退出。")
    if system:
        print(f"设定：{system}")
    print()

    while True:
        try:
            line = input(f"{USER}：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            cmd, rest = cmd.lower(), rest.strip()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "set":
                system = rest or None
                history = []
                print(f"(设定{'已更新：' + system if system else '已清空'}，上下文已重置)")
            elif cmd == "clear":
                history = []
                print("(上下文已清空)")
            elif cmd == "think":
                args.show_think = rest.lower() not in ("off", "0", "false", "no")
                print(f"(推理段显示：{'开' if args.show_think else '关'})")
            elif cmd == "raw":
                args.raw = rest.lower() in ("on", "1", "true", "yes")
                print(f"(原始输出：{'开' if args.raw else '关'})")
            elif cmd in ("temp", "topp", "topk", "rep", "ngram", "maxtok"):
                attr = {"temp": "temperature", "topp": "top_p", "topk": "top_k",
                        "rep": "repetition_penalty", "ngram": "no_repeat_ngram_size",
                        "maxtok": "max_new_tokens"}[cmd]
                try:
                    val = float(rest) if cmd in ("temp", "topp", "rep") else int(rest)
                    setattr(args, attr, val)
                    print(f"({attr} = {val})")
                except ValueError:
                    print("(需要一个数字)")
            elif cmd == "stats":
                print(f"(temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
                      f"rep={args.repetition_penalty} ngram={args.no_repeat_ngram_size} "
                      f"maxtok={args.max_new_tokens} 上下文={args.context})")
            elif cmd == "save":
                if rest:
                    dump_state(rest, history, system)
                else:
                    print("(用法：/save 文件名.json)")
            elif cmd == "load":
                try:
                    history, system = load_state(rest)
                except Exception as exc:                       # noqa: BLE001
                    print(f"(载入失败：{exc})")
            else:
                print(f"(未知命令 /{cmd}，试试 /help)")
            print()
            continue

        print(f"{ASSISTANT}：", end="", flush=True)
        res = chat_once(model, tok, history, line, system, args, device, sys.stdout)
        show_result(res, args)

        if not args.show_think and res["think"]:
            ans = res["answer"]
            print(f"  → {ans}\n" if ans else "  → (推理未收尾，正文为空)\n")

        history.append((line, res["ids"]))


# --------------------------------------------------------------------------- main

def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="RetNet 对话脚本 v2（retnet_hf 检查点）")
    ap.add_argument("--preset", default="v13", help=f"预设之一：{', '.join(PRESETS)}")
    ap.add_argument("--model_dir", default=None, help="直接指定模型目录，覆盖 --preset")
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--list", action="store_true", help="列出预设后退出")
    ap.add_argument("--prompt", default=None, help="单轮模式：只回答这一句")
    ap.add_argument("--system", default=None, help="角色设定")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--penalize_prompt", action="store_true",
                    help="同时惩罚提问里的词（默认关，否则回答会躲开问题本身的词）")
    ap.add_argument("--context", type=int, default=2048, help="模型训练窗口")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "keep"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--show_think", action="store_true", default=True)
    ap.add_argument("--hide_think", dest="show_think", action="store_false")
    ap.add_argument("--raw", action="store_true", help="不做回合边界切分")
    ap.add_argument("--quiet", action="store_true", help="不打印统计行")
    ap.add_argument("--no_color", dest="color", action="store_false", default=True)
    args = ap.parse_args()

    if args.list:
        for k, (d, note) in PRESETS.items():
            exists = "" if (ROOT / d).exists() else "  [缺失]"
            print(f"{k:<6} {d:<34} {note}{exists}")
        return

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             ("cpu" if args.device == "auto" else args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.model_dir:
        model_dir, note = Path(args.model_dir), "自定义目录"
    else:
        if args.preset not in PRESETS:
            raise SystemExit(f"未知预设 {args.preset}；可选：{', '.join(PRESETS)}")
        rel, note = PRESETS[args.preset]
        model_dir = ROOT / rel
    model_dir = resolve_model_dir(model_dir)

    if not Path(args.tokenizer).exists():
        raise SystemExit(f"找不到 tokenizer：{args.tokenizer}")
    tok = BPETokenizer.load(args.tokenizer)

    t0 = time.time()
    model = load_model(model_dir, device, args.dtype)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"模型：{note}  ({model_dir})")
    print(f"       {n_param / 1e6:.2f}M 参数 · {model.config.num_layers} 层 · "
          f"d_model={model.config.d_model} · vocab={model.config.vocab_size} · "
          f"窗口 {args.context}")
    print(f"设备：{device} · dtype={args.dtype} · "
          f"temp={args.temperature} top_k={args.top_k} top_p={args.top_p} "
          f"rep={args.repetition_penalty} ngram={args.no_repeat_ngram_size} "
          f"maxtok={args.max_new_tokens}")
    print(f"tokenizer：{Path(args.tokenizer).name} · 加载耗时 {time.time() - t0:.1f}s")
    print("-" * 72)

    if args.prompt:
        if args.system:
            print(f"设定：{args.system}")
        print(f"{USER}：{args.prompt}")
        print(f"{ASSISTANT}：", end="", flush=True)
        res = chat_once(model, tok, [], args.prompt, args.system, args, device, sys.stdout)
        show_result(res, args)
        if not args.show_think and res["think"]:
            print(f"  → {res['answer']}\n")
        return

    repl(model, tok, args, device)


if __name__ == "__main__":
    main()
