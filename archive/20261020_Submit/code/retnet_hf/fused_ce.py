"""融合线性 + 交叉熵：只保存 logsumexp，不保存 [N, V]。

标准 ``F.cross_entropy`` 会保存 ``log_softmax`` 的结果用于反向，那是
``[N, vocab]`` 的 fp32 张量 —— batch 1/seq 2048/vocab 20000 下 164 MB，
batch 2 下 328 MB；再叠上调用点显式 ``.float()`` 的那一份，batch 2 合计约 656 MB。

本项目当前卡在显存墙上：batch 2 关掉梯度检查点会颠簸到 11,000 ms/微步，
而 batch 2 能把微步数从 4 降到 2、内核启动数从 10,116 降到约 5,000，
正好对上实测 20.8% 的 GPU 空闲。**所以这里省下的不是时间，是「能不能上 batch 2」的入场券。**

本实现：前向按块算 logits 得到逐 token 的 CE，只保存 ``lse``（``[N]``）；
反向按块重算 logits 再求 ``softmax - onehot``，从而完全不需要 ``[N, V]`` 常驻。
代价是反向多一遍 LM head 的 GEMM（batch 1 下约 2.5 ms/微步），可忽略。

权重码 ``w`` 的语义与 ``modeling_retnet.py`` 原有的损失完全一致：
``loss = Σ ce_i·w_i / Σ w_i``。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: 一次处理多少个 token 的行；峰值显存 ≈ CHUNK × vocab × 4 B。
LOGIT_CHUNK = 2048


@torch._dynamo.disable
def _fused_forward(hidden, weight, target, w):
    """前向拆出来并禁用 Dynamo。

    分块循环里的 ``h[s:e]`` 是动态切片，Dynamo 追踪不了，会在此处图断裂并
    让整段退回 eager（实测直接卡住）。用 ``torch._dynamo.disable`` 明确告诉
    它「这段别管」，图断裂变成显式的边界而不是异常。
    """
    N, D = hidden.shape
    h = hidden.reshape(N, D)
    lse = torch.empty(N, device=hidden.device, dtype=torch.float32)
    tgt_logit = torch.empty(N, device=hidden.device, dtype=torch.float32)
    for s in range(0, N, LOGIT_CHUNK):
        e = min(s + LOGIT_CHUNK, N)
        lg = (h[s:e].float() @ weight.float().t())            # [c, V]
        lse[s:e] = torch.logsumexp(lg, dim=-1)
        tgt_logit[s:e] = lg.gather(1, target[s:e].unsqueeze(1)).squeeze(1)
        del lg
    return lse, tgt_logit


@torch._dynamo.disable
def _fused_backward(h, weight, target, w, grad_out, wsum):
    N, D = h.shape
    grad_h = torch.zeros_like(h)
    grad_w = torch.zeros_like(weight)
    scale = (grad_out / wsum).to(torch.float32)
    for s in range(0, N, LOGIT_CHUNK):
        e = min(s + LOGIT_CHUNK, N)
        hc = h[s:e].float()
        lg = hc @ weight.float().t()                          # [c, V]
        p = torch.softmax(lg, dim=-1)
        p.scatter_(1, target[s:e].unsqueeze(1), -1.0)         # p - onehot
        p.mul_(w[s:e].unsqueeze(1) * scale)                   # 权重码 + 上游梯度
        grad_h[s:e] = (p @ weight.float()).to(h.dtype)
        grad_w += (p.t() @ hc).to(weight.dtype)
        del lg, p, hc
    return grad_h, grad_w


class FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, w):
        """hidden [N,D] · weight [V,D] · target [N] · w [N]（已按权重码展开）。"""
        lse, tgt_logit = _fused_forward(hidden, weight, target, w)
        wsum = w.sum().clamp_min(1.0)
        loss = ((lse - tgt_logit) * w).sum() / wsum
        ctx.save_for_backward(hidden, weight, target, w)
        ctx.wsum = wsum.detach()
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        h, weight, target, w = ctx.saved_tensors
        grad_h, grad_w = _fused_backward(h, weight, target, w, grad_out, ctx.wsum)
        return grad_h, grad_w, None, None


def fused_linear_cross_entropy(hidden, weight, target, w):
    return FusedLinearCrossEntropy.apply(hidden, weight, target, w)
