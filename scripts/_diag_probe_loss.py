"""为什么 probe_500m/600m 在 GPU 上 loss=0.0 且 grad_norm=nan，而在 CPU 上是 11.73？

这个问题必须回答，因为它决定了 E3/E4/E5 的步时是否可信：
  - 若 loss 是「有图但恒为 0」→ 反向仍然走完整计算图，FLOPs 不变，步时**可信**
  - 若 loss 是「detached 常数」→ 反向极便宜，步时**严重偏乐观**，整套测量作废

判据：反向之后，transformer 内部参数是否拿到了**非零**梯度。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.data import PackedChatDataset, ChatDataCollator          # noqa: E402
from retnet_hf.modeling_retnet import RetNetForCausalLM                 # noqa: E402

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def batch(data_dir: str, seq_len: int):
    ds = PackedChatDataset(str(Path(data_dir) / "train.ids.npy"),
                           str(Path(data_dir) / "train.wcode.npy"), seq_len)
    coll = ChatDataCollator()
    items = [ds[i] for i in range(4)]
    return coll(items)


def run(model_dir: str, data_dir: str, autocast: bool, tag: str):
    model = RetNetForCausalLM.from_pretrained(model_dir)
    cfg = model.config
    model.config.mode = "chunkwise"
    model.config.chunk_size = 128
    model.train()
    dev = "cuda" if autocast else "cpu"
    model = model.to(dev)

    probe = np.load(Path(data_dir) / "train.ids.npy", mmap_mode="r")
    seq_len = probe.shape[1] - 1
    del probe
    b = batch(data_dir, seq_len)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

    print(f"\n===== {tag}  (dtype={cfg.dtype} device={dev}) =====")
    print(f"  embed_scale={getattr(cfg, 'embed_scale', 'MISSING')}  "
          f"growth_old_dim={getattr(cfg, 'growth_old_dim', 'MISSING')}  "
          f"prompt_loss_weight={cfg.prompt_loss_weight}")

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = model(**b)
    loss = out.loss
    print(f"  loss={loss.item():.6f}  logits dtype={out.logits.dtype}  "
          f"logits absmax={out.logits.abs().max().item():.4g}  "
          f"finite={bool(torch.isfinite(out.logits).all())}")
    print(f"  loss.requires_grad={loss.requires_grad}  grad_fn={type(loss.grad_fn).__name__}")

    # wcode / labels 分布：prompt_loss_weight 会不会把所有 token 都降到 0
    lab = b["labels"]
    wc = b.get("wcode")
    print(f"  labels: shape={tuple(lab.shape)}  ignore={(lab == -100).sum().item()}  "
          f"valid={(lab != -100).sum().item()}")
    if wc is not None:
        u = torch.unique(wc)
        print(f"  wcode 取值={u.tolist()[:8]}  (共 {u.numel()} 种)")

    loss.backward()
    tot, nz, nan = 0, 0, 0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        tot += 1
        g = p.grad
        if not torch.isfinite(g).all():
            nan += 1
        elif g.abs().max().item() > 0:
            nz += 1
    print(f"  反向：有梯度的参数 {tot} 个，其中非零 {nz}，非有限 {nan}")
    # 只看编码器/块内部，排除 embed 与 lm_head
    inner = [(n, p) for n, p in model.named_parameters()
             if "blocks" in n and p.grad is not None]
    if inner:
        mx = max(p.grad.abs().max().item() for _, p in inner)
        print(f"  blocks.* 梯度最大绝对值 {mx:.4g}（{len(inner)} 个张量）")
    del model
    torch.cuda.empty_cache() if autocast else None


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/probe_500m"
    dd = sys.argv[2] if len(sys.argv) > 2 else "data_cache/chat_pack_v13"
    run(md, dd, False, f"{md} CPU/fp32")
    run(md, dd, True, f"{md} GPU/bf16-autocast")
