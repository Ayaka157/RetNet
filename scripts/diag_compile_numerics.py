"""torch.compile 数值校验：eager vs compiled，逐位/逐步对比。

装好 triton 后 compile 提速 1.65x，但 bench_step 里出现了 loss=nan。
本脚本定位到底是哪一步开始 NaN、以及编译前后输出是否一致。
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RETNET_FAST_CHUNKWISE", "1")

from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402
from retnet_hf.data import PackedChatDataset, ChatDataCollator  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
base = RetNetForCausalLM.from_pretrained(str(ROOT / "checkpoints/retnet_272m_rp")).to(dev)
base.gradient_checkpointing_enable()
base.train()

ds = PackedChatDataset(str(ROOT / "data_cache/chat_pack_v13/train.ids.npy"),
                       str(ROOT / "data_cache/chat_pack_v13/wcode.npy"
                           ).replace("wcode.npy", "train.wcode.npy"), 2048)
col = ChatDataCollator()
batches = [{k: v.to(dev) for k, v in col([ds[i * 4 + j] for j in range(4)]).items()}
           for i in range(3)]


def forward_loss(model, b):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=b["input_ids"], labels=b["labels"],
                    wcodes=b["wcodes"], mode="chunkwise", chunk_size=256)
    return out


print("=" * 76)
print("A. 纯前向：eager vs compiled 的 loss 与 logits")
print("=" * 76)
with torch.no_grad():
    base.eval()
    eager = [forward_loss(base, b) for b in batches]
    eager_logits = [o.logits.float().clone() for o in eager]
    eager_loss = [o.loss.item() for o in eager]

cm = torch.compile(base, backend="inductor")
with torch.no_grad():
    comp = [forward_loss(cm, b) for b in batches]
    comp_logits = [o.logits.float().clone() for o in comp]
    comp_loss = [o.loss.item() for o in comp]

for i, (a, b) in enumerate(zip(eager_loss, comp_loss)):
    d = (comp_logits[i] - eager_logits[i]).abs().max().item()
    print(f"  batch{i}: eager loss={a:.6f}  compiled loss={b:.6f}  "
          f"logits max|d|={d:.3e}")

print()
print("=" * 76)
print("B. 训练 8 步：每步的 loss 与 grad_norm")
print("=" * 76)


def train_steps(model, n=8, lr=1e-6):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rows = []
    for i in range(n):
        b = batches[i % len(batches)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=b["input_ids"], labels=b["labels"],
                        wcodes=b["wcodes"], mode="chunkwise", chunk_size=256)
        loss = out.loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        rows.append((i, loss.item(), gn))
    return rows


base2 = RetNetForCausalLM.from_pretrained(str(ROOT / "checkpoints/retnet_272m_rp")).to(dev)
base2.gradient_checkpointing_enable()
base2.train()
print("  eager:")
for i, l, g in train_steps(base2):
    flag = "  <-- NaN/Inf" if not (l == l and abs(l) != float("inf")) else ""
    print(f"    step {i}  loss={l:.6f}  grad_norm={g:.4f}{flag}")

del base2
torch.cuda.empty_cache()

base3 = RetNetForCausalLM.from_pretrained(str(ROOT / "checkpoints/retnet_272m_rp")).to(dev)
base3.gradient_checkpointing_enable()
base3.train()
cm2 = torch.compile(base3, backend="inductor")
print("  compiled:")
for i, l, g in train_steps(cm2):
    flag = "  <-- NaN/Inf" if not (l == l and abs(l) != float("inf")) else ""
    print(f"    step {i}  loss={l:.6f}  grad_norm={g:.4f}{flag}")
