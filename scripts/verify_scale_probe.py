"""在**没有 GPU** 的情况下，把 `scripts/_run_scale_probe.ps1` 要在 5 小时后跑的
那条链路预先走一遍，把"白等一夜"的风险提前排掉。

    python scripts/verify_scale_probe.py

逐项镜像 `scripts/bench_trainer.py` 的加载路径（`from_pretrained` 不带 dtype →
fp32、`config.mode='chunkwise'`、`set_checkpoint_layers`、`chunk_size`），
只把 device 换成 cpu、把 seq 缩短。

**不覆盖**：CUDA 显存、torch.compile 的图捕获、OOM/颠簸。
这三样必须有真卡才能测，正是驱动要等 GPU 的原因。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402
from retnet_hf.data import PackedChatDataset, ChatDataCollator  # noqa: E402

DATA = ROOT / "data_cache" / "chat_pack_v13"

# (标签, 目录, 驱动会在它上面用的 checkpoint_layers, 驱动会用的 chunk_size)
CASES = [
    ("E1  272M v13", "checkpoints/retnet_272m_v13", None, [128]),
    ("E2a/E3a 500M", "checkpoints/probe_500m", [16, 12, 10, 8], [128, 256]),
    ("E2b/E3b 600M", "checkpoints/probe_600m", [20, 16, 12], [128, 256]),
]

fails = []


def check(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        fails.append(msg)
    return cond


def main():
    print("=" * 74)
    print("预演 _run_scale_probe.ps1 的加载链路（CPU，不含 CUDA/compile）")
    print("=" * 74)

    # ---- 数据集（bench_trainer 用 mmap 读第 2 维当 seq_len）----
    ids = DATA / "train.ids.npy"
    check(ids.exists(), f"数据集存在 {ids}")
    if not ids.exists():
        return 1
    import numpy as np
    probe = np.load(ids, mmap_mode="r")
    seq_len = probe.shape[1] - 1
    del probe
    check(seq_len == 2048, f"seq_len = {seq_len}（驱动的前提是 2048）")
    ds = PackedChatDataset(str(ids), str(DATA / "train.wcode.npy"), seq_len)
    col = ChatDataCollator()
    batch = col([ds[0]])
    check(batch["input_ids"].shape == (1, 2048),
          f"一个样本 = {tuple(batch['input_ids'].shape)}")

    # ---- 逐个模型 ----
    for label, rel, ckpt_layers, chunks in CASES:
        print(f"\n--- {label}  [{rel}]")
        p = ROOT / rel
        if not check(p.exists(), "目录存在"):
            continue
        try:
            model = RetNetForCausalLM.from_pretrained(str(p))
        except Exception as e:                                  # noqa: BLE001
            check(False, f"from_pretrained 失败: {type(e).__name__}: {e}")
            continue
        n_par = sum(x.numel() for x in model.parameters())
        L = len(model.blocks)
        check(True, f"from_pretrained OK  {n_par:,} 参数  {L} 层  d={model.config.d_model}")

        # 权重 dtype 必须是 fp32（docs/19 的关键事实，也是 10.004 B/参数的前提）
        dts = {x.dtype for x in model.parameters()}
        check(dts == {torch.float32}, f"权重 dtype = {dts}（10.004 B/参数 的前提）")

        model.config.mode = "chunkwise"

        # 分层检查点：驱动用到的每一个值都要被接受，且返回 0<=n<=L
        for n in (ckpt_layers or []):
            ret = model.set_checkpoint_layers(n)
            check(0 <= ret <= L, f"set_checkpoint_layers({n}) -> {ret}（0<=n<={L}）")
        if ckpt_layers:
            model.set_checkpoint_layers(-1)      # 复位成"全检查点"

        # 前向：驱动会用 chunk 128 与 256，两者都必须能算
        model.eval()
        with torch.no_grad():
            for C in chunks:
                model.config.chunk_size = C
                s = 512                                  # 512 同时是 128/256 的倍数
                b = {k: (v[:, :s] if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] >= s
                         else v) for k, v in batch.items()}
                try:
                    out = model(chunk_size=C, **b)
                    ok = torch.isfinite(out.loss).item()
                except Exception as e:                  # noqa: BLE001
                    check(False, f"chunk={C} 前向抛异常: {type(e).__name__}: {e}")
                    continue
                check(ok, f"chunk={C} 前向 OK  loss={out.loss.item():.4f}  "
                          f"logits={tuple(out.logits.shape)}")
        del model

    print("\n" + "=" * 74)
    if fails:
        print(f"❌ {len(fails)} 项失败：")
        for f in fails:
            print("   - " + f)
        print("**不要让它无人值守跑**，先修。")
        return 1
    print("✅ 全部通过。链路在 CPU 上可走通；剩余风险只有 CUDA 显存/compile/OOM，")
    print("   那三样必须等真卡，正是驱动要等的原因。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
