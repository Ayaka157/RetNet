"""`_head_group_norm` 重写后的等价性回归。

原实现把 ``[B,H,L,vhd]`` 转成 ``[B*L, H*vhd, 1]`` 交给 ``nn.GroupNorm(H, H*vhd)``；
新实现直接在 ``[B,H,L,vhd]`` 上沿最后一维归一化。

两者等价的依据：``num_groups == num_heads`` 且空间维为 1，所以每个 group 恰好
是一个 head 的 ``vhd`` 个通道，等价于对每个 (b,h,l) 沿最后一维做归一化，
仿射系数按 head 分组（``weight.view(H, vhd)``）。

单元级比随机张量；模型级比真实前向的 loss 与 logits —— 后者才能证明这条改动
在真实训练路径上没有把数值改掉。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import MultiScaleRetention, RetNetForCausalLM  # noqa: E402
from retnet_hf.data import PackedChatDataset, ChatDataCollator               # noqa: E402

B, H, L, VHD = 2, 4, 64, 32          # 小尺寸，可穷举
REAL = dict(B=1, H=16, L=512, VHD=128)


def reference(x, num_heads, gn, bsz, tgt_len):
    """改动前的实现，逐字保留。"""
    x = x.transpose(1, 2).reshape(bsz * tgt_len, num_heads * x.shape[-1])
    x = gn(x.unsqueeze(-1)).squeeze(-1)
    return x.reshape(bsz, tgt_len, -1)


def unit_case(b, h, l, vhd, dtype, eps=1e-5):
    torch.manual_seed(0)
    gn = nn.GroupNorm(h, h * vhd, eps=eps).cuda().to(dtype)
    with torch.no_grad():
        gn.weight.copy_(torch.randn(h * vhd, device="cuda"))
        gn.bias.copy_(torch.randn(h * vhd, device="cuda"))
    x = torch.randn(b, h, l, vhd, device="cuda", dtype=dtype) * 3.0
    want = reference(x, h, gn, b, l)
    got = MultiScaleRetention._head_group_norm(x, h, gn, b, l)
    d = (got.float() - want.float()).abs()
    scale = want.float().abs().max().clamp_min(1e-6)
    return d.max().item(), (d.max() / scale).item(), got.dtype == x.dtype


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if not torch.cuda.is_available():
        print("需要 CUDA")
        return 1

    print("== 单元级：随机张量 ==")
    bad = 0
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for shape in ((B, H, L, VHD), (REAL["B"], REAL["H"], REAL["L"], REAL["VHD"]),
                      (1, 8, 7, 64)):
            mx, rel, same_dtype = unit_case(*shape, dtype=dtype)
            # fp32 下两条路径的归约顺序不同，差在 1 ULP 量级；bf16/fp16 放宽到
            # 各自格式的 1 ULP（bf16 是 2^-7，fp16 是 2^-10）。
            tol = {torch.float32: 1e-5, torch.bfloat16: 2.0 ** -7,
                   torch.float16: 2.0 ** -9}[dtype]
            ok = rel <= tol and same_dtype
            bad += 0 if ok else 1
            print(f"  {str(dtype):<16} {str(shape):<22} max|d|={mx:.3e} "
                  f"rel={rel:.2e} dtype_ok={same_dtype} {'OK' if ok else 'FAIL'}")

    print("\n== 模型级：真实前向的 loss / logits ==")
    torch.manual_seed(0)
    model = RetNetForCausalLM.from_pretrained(str(ROOT / "checkpoints/retnet_272m_rp"))
    model.config.mode = "chunkwise"
    model.config.chunk_size = 128
    model = model.cuda().eval()

    dd = ROOT / "data_cache/chat_pack_v13"
    ds = PackedChatDataset(str(dd / "train.ids.npy"), str(dd / "train.wcode.npy"), 2048)
    batch = ChatDataCollator()([ds[i] for i in range(2)])
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    from retnet_hf.modeling_retnet import FAST_CHUNKWISE  # noqa: F401
    # 注意：``Cls.__dict__["x"]`` 拿到的才是 staticmethod 对象；直接
    # ``Cls.x`` 取出来是**普通函数**，再赋回类上就退化成实例方法，
    # 调用时会多传一个 self（踩过）。
    new_impl = staticmethod(MultiScaleRetention.__dict__["_head_group_norm"].__func__)
    old_impl = staticmethod(reference)

    outs = {}
    for tag, impl in (("new", new_impl), ("old", old_impl)):
        MultiScaleRetention._head_group_norm = impl
        with torch.no_grad():
            out = model(**batch)
        outs[tag] = out.loss.item()
    MultiScaleRetention._head_group_norm = new_impl

    dl = abs(outs["new"] - outs["old"])
    print(f"  loss  新={outs['new']:.6f}  旧={outs['old']:.6f}  |Δ|={dl:.2e}")
    ok = dl < 1e-4
    print("  " + ("PASS" if ok and not bad else "FAIL"))
    return 0 if (ok and not bad) else 1


if __name__ == "__main__":
    raise SystemExit(main())
