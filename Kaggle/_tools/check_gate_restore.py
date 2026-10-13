"""Does `model.to(dtype=bfloat16)` irreversibly damage the fp32 decay gates?

A teammate claims: `model.to(dtype=bfloat16)` rounds the gates BEFORE the
`.data = .data.to(torch.float32)` restore can see the originals, so the fp32
master is lost (not merely re-typed).  If true, the shipped training run starts
from rounded gates; if false, the restore plus the checkpoint loaded later in
`_load_checkpoint` recovers the exact fp32 values.

Tested against the REAL initial weights:
  Kaggle/archived/retnet_272m_init_weights/retnet_272m_rp/model.safetensors

Run:  python check_gate_restore.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(r"E:\Workspaces\RetNet")
CKPT = REPO / "Kaggle" / "archived" / "retnet_272m_init_weights" / "retnet_272m_rp"


def main() -> int:
    import torch
    from safetensors.torch import load_file

    print(f"[ckpt] {CKPT}")
    cfg = json.loads((CKPT / "config.json").read_text(encoding="utf-8"))
    print(f"[ckpt] use_gla={cfg['use_gla']} num_heads={cfg['num_heads']} "
          f"layers={cfg['num_layers']} value_dim={cfg['value_dim']}")

    sd = load_file(str(CKPT / "model.safetensors"))
    gate_keys = sorted(k for k in sd if k.endswith("msr.gamma"))
    print(f"[ckpt] gamma tensors: {len(gate_keys)}  dtype={sd[gate_keys[0]].dtype}  "
          f"shape={tuple(sd[gate_keys[0]].shape)}")
    orig = {k: sd[k].clone() for k in gate_keys}

    for k in gate_keys[:2] + gate_keys[-1:]:
        v = orig[k]
        print(f"       {k}: min={v.min().item():.7f} max={v.max().item():.7f}")

    # ---- simulate the shipped order: fp32 state -> .to(bf16) -> restore fp32
    obj = {k: v.clone() for k, v in orig.items()}
    def to_dtype(d, dt):
        return {k: v.to(dt) for k, v in d.items()}

    bf16 = to_dtype(obj, torch.bfloat16)          # this is what model.to(bf16) does
    restored = to_dtype(bf16, torch.float32)      # this is what the restore loop does

    exact = sum(
        int(torch.equal(restored[k], orig[k])) for k in gate_keys
    )
    maxdev = max((restored[k] - orig[k]).abs().max().item() for k in gate_keys)
    print(f"\n[order A: fp32 -> bf16 -> fp32]  gates exactly recovered: {exact}/{len(gate_keys)}  "
          f"max deviation {maxdev:.3e}")

    # Which heads merged?
    a = orig[gate_keys[0]]
    b = restored[gate_keys[0]]
    print(f"       head values      orig : {[f'{v:.7f}' for v in a.tolist()]}")
    print(f"       head values restored : {[f'{v:.7f}' for v in b.tolist()]}")
    n_one = int((b == 1.0).sum().item())
    print(f"       heads restored to exactly 1.0: {n_one}/{a.numel()}")

    # gamma**64 is the cross-chunk decay: how much error does the rounding cost?
    dec_o = (orig[gate_keys[0]].double() ** 64)
    dec_r = (restored[gate_keys[0]].double() ** 64)
    rel = ((dec_r - dec_o).abs() / dec_o.clamp_min(1e-300)).max().item()
    print(f"       cross-chunk decay gamma**64: max relative error {rel:.3e}")
    print(f"       (orig gamma**64 range {dec_o.min().item():.6f} .. {dec_o.max().item():.6f})")

    print("\nVERDICT:", "the restore CANNOT recover the fp32 master (claim CONFIRMED)"
          if exact == 0 else f"exact recovery on {exact} gates (claim REFUTED)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
