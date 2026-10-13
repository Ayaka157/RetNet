"""NaN 是「检查点本身就坏」还是「长序列才发散」？

probe_500m 在 CPU/fp32、seq 2048 下 logits 已是 NaN。但 verify_scale_probe.py
只用 seq 512 预演过并报通过。所以要么权重本身非有限，要么 NaN 从某个长度才开始。

判据：逐步加长序列，看 logits 的 absmax 从哪一步开始变成非有限。
  - seq=128 就 NaN  → 权重/初始化坏了
  - 短序列有限、长序列 NaN → 数值发散（长序列上状态累积爆炸）
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

MD = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/probe_500m"
DD = "data_cache/chat_pack_v13"

m = RetNetForCausalLM.from_pretrained(MD)
m.config.mode = "chunkwise"
m.eval()
m = m.to("cuda")

# ---- 权重本身是否有限 ----
tot = nbad = 0
worst = ("", 0.0)
for n, p in m.named_parameters():
    tot += 1
    if not torch.isfinite(p).all().item():
        nbad += 1
        print(f"  非有限权重: {n}")
    else:
        mx = p.abs().max().item()
        if mx > worst[1]:
            worst = (n, mx)
print(f"权重扫描：{tot} 个张量，非有限 {nbad} 个，最大绝对值 {worst[1]:.4g}  ({worst[0]})")

probe = np.load(Path(DD) / "train.ids.npy", mmap_mode="r")
seq_len = probe.shape[1] - 1
del probe
ds = PackedChatDataset(str(Path(DD) / "train.ids.npy"),
                       str(Path(DD) / "train.wcode.npy"), seq_len)
col = ChatDataCollator()
b = col([ds[0]])

print(f"\n序列长度扫描（chunk=128, eval, fp32 权重 + bf16 autocast 两种都试）")
for amp in (False, True):
    tag = "bf16-autocast" if amp else "fp32"
    print(f"\n  --- {tag} ---")
    for s in (128, 256, 512, 1024, 2048):
        bb = {k: (v[:, :s] if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] >= s else v)
              for k, v in b.items()}
        bb = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in bb.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = m(**bb)
        lg = out.logits
        fin = bool(torch.isfinite(lg).all())
        print(f"    seq={s:5d}  logits absmax={lg.abs().max().item():12.4g}  "
              f"finite={fin}  loss={out.loss.item():10.4f}")
        del out, bb
        torch.cuda.empty_cache()
