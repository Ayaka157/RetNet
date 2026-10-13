"""加载一个 HF 检查点并真的跑一次前向，回答「它到底是不是一个可用的模型」。

比 ``check_checkpoints.py`` 更进一步：那个只查 safetensors 键名（O(1)、二值），
这个真的把权重读进来跑一遍。

为什么两步都要：带 ``_orig_mod.`` 前缀的检查点 ``from_pretrained`` **不报错**，
它把带前缀的键当"多余"丢掉、把不带前缀的当"缺失"随机初始化。只有真跑一次
才看得见 ``loss=nan``。这个坑已经在 v14 上踩过一次
（评测侧的表现为 ``torch.multinomial`` 的 device-side error，极难追根因）。

    python scripts/verify_checkpoint_loads.py --ckpt checkpoints/retnet_272m_v14/checkpoint-10000
    python scripts/verify_checkpoint_loads.py --ckpt a --ckpt b --expect 2.606464
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 随机初始化的 272M 语言模型在真实中文数据上的 loss 约为 ln(20000) ≈ 9.9；
# 训练过的模型在这个数据上是 2.5~3.5。用 8.0 当分界线足够安全。
RANDOM_LOSS_FLOOR = 8.0


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--data_dir", default=str(ROOT / "data_cache" / "chat_pack_v13"))
    ap.add_argument("--rows", type=int, default=1, help="用前 N 行做前向")
    ap.add_argument("--chunk_size", type=int, default=128)
    ap.add_argument("--expect", type=float, default=None,
                    help="期望 loss（给个参考值，差太多会标出来）")
    ap.add_argument("--data_only", action="store_true",
                    help="跳过模型，只打印数据参考 loss（用不到）")
    args = ap.parse_args()
    del args.data_only

    from retnet_hf.modeling_retnet import RetNetForCausalLM
    from retnet_hf.data import PackedChatDataset, ChatDataCollator

    dd = Path(args.data_dir)
    ds = PackedChatDataset(str(dd / "train.ids.npy"), str(dd / "train.wcode.npy"), 2048)
    batch = ChatDataCollator()([ds[i] for i in range(args.rows)])
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    fails = 0
    print(f"{'检查点':<46}{'loss':>14}{'logits':>10}  判定")
    for ck in args.ckpt:
        p = Path(ck)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            print(f"{str(ck):<46}{'—':>14}{'—':>10}  ❌ 不存在")
            fails += 1
            continue
        try:
            model = RetNetForCausalLM.from_pretrained(str(p))
        except Exception as exc:                               # noqa: BLE001
            print(f"{str(ck):<46}{'—':>14}{'—':>10}  ❌ 加载失败 {type(exc).__name__}")
            fails += 1
            continue
        model = model.cuda().eval()
        model.config.mode = "chunkwise"
        with torch.no_grad():
            out = model(chunk_size=args.chunk_size, **batch)
        loss = out.loss.item()
        finite = bool(torch.isfinite(out.logits).all())

        if not finite or loss != loss:
            verdict = "❌ 输出 NaN/Inf —— 权重没被真正加载"
            fails += 1
        elif loss > RANDOM_LOSS_FLOOR:
            verdict = (f"❌ loss={loss:.2f} > {RANDOM_LOSS_FLOOR} —— 像是随机初始化，"
                       f"检查是不是带了 _orig_mod. 前缀")
            fails += 1
        elif args.expect is not None and abs(loss - args.expect) > 0.5:
            verdict = f"⚠️ 与期望 {args.expect:.4f} 差 {abs(loss - args.expect):.4f}"
            fails += 1
        else:
            verdict = "✅ 可用"
        print(f"{str(ck):<46}{loss:>14.6f}{'finite' if finite else 'NaN':>10}  {verdict}")

        del model
        torch.cuda.empty_cache()

    print()
    print("全部通过" if not fails else f"{fails} 个检查点有问题")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
