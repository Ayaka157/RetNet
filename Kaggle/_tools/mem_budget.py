"""Analytic HBM budget for the configured step (v5e-8, 16 GB per chip).

The original run died on/around the first `mark_step` with no Python
traceback, which is the signature of a compile-time or device-side failure
rather than an exception in user code.  This computes the per-chip footprint of
every tensor the config creates, so we can say whether step 1 is even feasible
at PER_DEVICE_BATCH=4, SEQ_LEN=8192.

Run:  python mem_budget.py
"""
from __future__ import annotations

B = 4            # PER_DEVICE_BATCH
T = 8192         # SEQ_LEN
V = 20000        # vocab_size
D = 1024         # d_model
L = 20           # num_layers
H = 16           # num_heads
KD = D // H      # 64 key_dim
VD = 2048        # value_dim
VH = VD // H     # 128 value_head_dim
FFN = 2048       # ffn_dim
C = 64           # chunk_size

GB = 1024 ** 3


def mb(x):
    return f"{x / GB:8.3f} GB"


def row(name, n_elems, dtype=2, note=""):
    return (name, n_elems * dtype, note)


rows = [
    row("logits [B,T,V] bf16", B * T * V, 2, "lm_head output, materialised"),
    row("logits.float() [B,T,V] fp32", B * T * V, 4, "F.cross_entropy input cast"),
    row("token_loss [B*T] fp32", B * T, 4, "reduction='none' output"),
    row("w / masked copies [B,T] fp32 x4", B * T * 4, 4, "wcode weight tensor + 3 where() temps"),
    row("labels/wcodes ids [B,T] i64 x2", B * T * 2, 8, "collated on device"),
    row("embed out [B,T,D] bf16", B * T * D, 2, "block 0 input"),
    row("per-block residual [B,T,D] bf16", B * T * D, 2, "one block's live activation"),
    row("retention q/k/v/g [B,T,D] bf16 x4", 4 * B * T * D, 2, "MSR projections"),
    row("chunk inner scores [B,H,N,C,C] bf16", B * H * (T // C) * C * C, 2, "one chunk at a time"),
    row("state + output carry [B,H,KD,VH] bf16 x2", 2 * B * H * KD * VH, 2, "recurrence carry"),
]

print(f"config: B={B} T={T} V={V} D={D} L={L} H={H} VD={VD} FFN={FFN} C={C}")
print(f"tokens per step (global, 8 chips) = {B * T * 8:,}")
print()
total_act = 0
print(f"{'tensor':44s} {'bytes':>12s}  note")
print("-" * 90)
for name, nbytes, note in rows:
    print(f"{name:44s} {mb(nbytes)}  {note}")
    total_act += nbytes
print("-" * 90)
print(f"{'sum of the above':44s} {mb(total_act)}")

params = L * (4 * D * D + 2 * D * VD + 2 * D * FFN) + V * D
p_bytes = params * 2
opt_bytes = params * 4 * 2          # AdamW m and v in fp32
grad_bytes = params * 2             # bf16 grads
print()
print(f"{'model params':44s} {params / 1e6:8.1f} M")
print(f"{'params bf16':44s} {mb(p_bytes)}")
print(f"{'grads bf16':44s} {mb(grad_bytes)}")
print(f"{'AdamW m+v fp32':44s} {mb(opt_bytes)}")
static = p_bytes + grad_bytes + opt_bytes
print(f"{'static total':44s} {mb(static)}")
print()
print(f"{'HBM per v5e chip':44s} {mb(16 * GB)}")
print(f"{'worst-case sum (static + activations)':44s} {mb(static + total_act)}")
print()
print("NOTE: gradient checkpointing recomputes each block, so per-block")
print("activations are not all live at once, but the logits / loss path is")
print("NOT checkpointed and is fully materialised, twice (bf16 + fp32).")
