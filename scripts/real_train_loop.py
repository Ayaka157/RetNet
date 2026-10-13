"""Real training loop (mirrors the HF Trainer exactly): bf16 autocast,
chunkwise retention, gradient checkpointing, AdamW, weighted-CE via the model's
own forward(labels, wcodes).  Reports per-step torch memory AND nvidia-smi
(dedicated vs the 8GiB ceiling) so shared-memory spill is visible."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retnet_hf.modeling_retnet import RetNetForCausalLM  # noqa: E402


def gb(n: float) -> str:
    return f"{n / 1e9:.2f}G"


def nvidia_smi():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"]).decode().strip()
    used, total = out.split(", ")
    return int(used), int(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--no-float", action="store_true",
                    help="drop logits.float() (compute CE on bf16 logits)")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    amp = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    model = RetNetForCausalLM.from_pretrained("checkpoints/retnet_272m_rp").cuda().train()
    model.config.mode = "chunkwise"
    model.config.chunk_size = 64
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=1.1e-4, weight_decay=0.1)
    V = model.config.vocab_size
    B, L = args.batch, args.seq

    if args.no_float:
        # patch: replace the model's loss path to skip logits.float()
        import torch.nn.functional as F
        orig_forward = model.forward
        def patched(input_ids, labels=None, wcodes=None, mode=None, states=None,
                    offset=0, return_state=False, chunk_size=64, **_k):
            logits = model._forward_hidden(input_ids, mode="chunkwise",
                                           return_state=True, chunk_size=chunk_size)[0]
            token_loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                                         reduction="none")
            w = torch.ones_like(wcodes, dtype=torch.float32)
            w = torch.where(wcodes == 1, torch.ones_like(w), w)
            w = torch.where(wcodes == 2, torch.full_like(w, 0.1), w)
            w = torch.where(wcodes == 3, torch.ones_like(w), w)
            loss = (token_loss * w.reshape(-1)).sum() / w.sum().clamp_min(1.0)
            from transformers.modeling_outputs import CausalLMOutput
            return CausalLMOutput(loss=loss, logits=logits)
        model.forward = patched

    print(f"batch={B} seq={L} dtype={args.dtype} no_float={args.no_float}")
    print(f"{'step':>4} | {'loss':>8} | {'alloc':>7} | {'peak':>7} | {'resv':>7} | "
          f"{'smi_used':>8} | {'smi_total':>9} | {'sec/it':>7}")
    print("-" * 82)

    for step in range(args.steps):
        x = torch.randint(0, V, (B, L), device="cuda")
        labels = torch.randint(0, V, (B, L), device="cuda")
        wc = torch.ones((B, L), device="cuda", dtype=torch.long)

        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.autocast("cuda", dtype=amp):
            out = model(input_ids=x, labels=labels, wcodes=wc)
            loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.time() - t0

        used, total = nvidia_smi()
        alloc = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        resv = torch.cuda.memory_reserved()
        print(f"{step:4d} | {loss.item():8.4f} | {gb(alloc):>7} | {gb(peak):>7} | "
              f"{gb(resv):>7} | {used:6d}MiB | {total:7d}MiB | {dt:7.2f}")

    print(f"\n8GiB ceiling = {8192} MiB. smi_used > 8192 means spill to shared RAM.")


if __name__ == "__main__":
    main()
