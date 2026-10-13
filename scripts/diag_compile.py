"""Is torch.compile actually doing anything for this model?"""
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RETNET_FAST_CHUNKWISE", "1")

from retnet_hf.modeling_retnet import RetNetForCausalLM, FAST_CHUNKWISE  # noqa: E402
from retnet_hf.data import PackedChatDataset, ChatDataCollator          # noqa: E402

print("FAST_CHUNKWISE =", FAST_CHUNKWISE)
dev = "cuda"
m = RetNetForCausalLM.from_pretrained(str(ROOT / "checkpoints/retnet_272m_rp")).to(dev).train()
m.gradient_checkpointing_enable()
ds = PackedChatDataset(str(ROOT / "data_cache/chat_pack_v13/train.ids.npy"),
                       str(ROOT / "data_cache/chat_pack_v13/train.wcode.npy"), 2048)
col = ChatDataCollator()
b = col([ds[i] for i in range(4)])
b = {k: v.to(dev) for k, v in b.items()}

import torch._dynamo as dynamo
dynamo.config.suppress_errors = False


def step(model, autocast=True):
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
           else torch.autocast("cuda", enabled=False))
    with ctx:
        logits, _ = model._forward_hidden(b["input_ids"], mode="chunkwise",
                                          return_state=True, chunk_size=256)
    import torch.nn.functional as F
    lf = logits.float()
    loss = F.cross_entropy(lf.reshape(-1, lf.shape[-1]), b["labels"].reshape(-1))
    loss.backward()
    model.zero_grad(set_to_none=True)
    return loss.detach()


print("\n--- eager ---")
for _ in range(3):
    step(m)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    step(m)
torch.cuda.synchronize()
t_eager = (time.perf_counter() - t0) / 5
print(f"  {t_eager*1000:.1f} ms/step")

print("\n--- compiled ---")
cm = torch.compile(m, backend="inductor", fullgraph=False)
try:
    for _ in range(3):
        step(cm)
    torch.cuda.synchronize()
except Exception as exc:
    print("  编译/执行失败:", type(exc).__name__, exc)
    raise SystemExit(1)

# inductor 是否真的产出了内核？
from torch._inductor.utils import fresh_inductor_cache  # noqa: F401
try:
    from torch._dynamo.utils import counters
    print("  dynamo counters:", {k: v for k, v in counters.items() if v})
except Exception as exc:
    print("  counters 不可用:", exc)

t0 = time.perf_counter()
for _ in range(5):
    step(cm)
torch.cuda.synchronize()
t_comp = (time.perf_counter() - t0) / 5
print(f"  {t_comp*1000:.1f} ms/step   speedup {t_eager/t_comp:.3f}x")

import torch._inductor.config as ic
print(f"\n  inductor cache dir: {ic.cache_dir}")
cd = Path(ic.cache_dir)
if cd.exists():
    for sub in sorted(cd.rglob("*.py"))[:5]:
        print("   ", sub.relative_to(cd))
    n_py = len(list(cd.rglob("*.py")))
    print(f"  生成的 .py 内核文件数: {n_py}")
else:
    print("  缓存目录不存在 —— inductor 没有产出任何东西")
