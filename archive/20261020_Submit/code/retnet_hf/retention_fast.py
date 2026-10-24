"""Batched-chunk chunkwise retention — mathematically identical to the reference
per-chunk loop in ``retnet_hf/modeling_retnet.py``, but with ~30x fewer kernel
launches.

Reference (per-chunk loop, 4 matmuls per chunk):

    for i in chunks:
        inner  = (q_i @ k_i.T * inner_mask) @ v_i
        cross  = (q_i * cross_decay) @ state
        out_i  = inner + cross
        state  = chunk_decay * state + k_i.T @ (v_i * value_decay)

Batched form (3 matmuls total, independent of the chunk count):

    1. U_i      = k_i.T @ (v_i * value_decay)          one batched matmul
    2. S_prev_i = scan(U, chunk_decay, state)          tiny sequential loop on
                                                       [B,H,kd,vd] tensors only
    3. inner    = (q_r @ k_r.T * inner_mask) @ v_r     one batched matmul
       cross    = (q_r * cross_decay) @ S_prev         one batched matmul

The scan in step 2 is deliberately *not* replaced by a cumsum: with
``chunk_decay = gamma^C`` and C=256, ``gamma^-j`` overflows fp32 for the fast
heads, while the log-depth pairwise scan costs more than it saves at 8 chunks.
"""
from __future__ import annotations

import torch


_INDEX_CACHE: dict = {}


def _index_tables(chunk_size: int, device):
    """``causal`` / ``expo`` / ``rel`` 只依赖 chunk_size，缓存掉。

    原实现每次调用都重算 ``arange``/``sub``/``ge``/``clamp``，是 4 次无用的小
    内核发射；而本算子的成本**几乎全是发射延迟**（见下）。
    """
    key = (int(chunk_size), str(device))
    tables = _INDEX_CACHE.get(key)
    if tables is None:
        rel = torch.arange(chunk_size, device=device)
        delta = rel[:, None] - rel[None, :]
        causal = delta >= 0
        expo = delta.clamp_min(0).to(torch.float32)
        tables = (causal, expo, rel.to(torch.float32))
        _INDEX_CACHE[key] = tables
    return tables


def _decay_tables(gamma: torch.Tensor, chunk_size: int, dtype: torch.dtype,
                  device: torch.device):
    """gamma: [H] (fp32)。返回 inner_mask [H,C,C], cross_decay [H,C],
    value_decay [H,C], chunk_decay [H]，均为 ``dtype``。

    块内掩码用**对数空间** ``exp(log γ · delta)``，这是唯一在本模型门控范围
    两端都安全的写法：

    * 因式分解的 ``γ^i · γ^-j`` 一旦需要 ``γ^-255`` 就溢出 fp32。
      ``retnet_272m_rp`` 的 GLA 门控最小到 0.668，``0.668**-255 = 1e44 > 3.4e38``
      → Inf → 前向 NaN（``torch.where`` 只丢结果，不阻止 Inf 被算出来）。
    * 朴素 ``γ ** delta`` 对 γ<1 安全（下溢到 0，那里本来就该是 0），
      但对 γ>1 的门控不成立；对数形式两端都对。

    NOTE(2026-10-10): 这个算子只有 26 万个元素却要花 **410 µs** —— 成本几乎全是
    「22 个相互依赖的小内核」的发射延迟（Windows 上约 18 µs/次）。所以这里的
    优化目标一律是**减少发射次数**，而不是减少计算量：
      * 索引张量进缓存（省 4 次发射）；
      * 三张一维表复用同一个 ``γ^j``：[H,C]：``cross = γ^j·γ``、
        ``value = flip(γ^j)``、``chunk = γ^j[-1]·γ``，
        把原来的 3 次 ``exp`` 压成 1 次。
    """
    causal, expo, relf = _index_tables(chunk_size, device)
    g = gamma.to(device=device, dtype=torch.float32)
    # γ 可以合法地取到 0（门控被训练成「完全遗忘」），此时 log(0) = -inf，
    # 而对角线 delta=0 会让 exp(-inf * 0) = exp(NaN) —— torch.where 只丢结果、
    # 不阻止 NaN 被算出来，于是整层 NaN。夹一个正下限后
    # exp(logγ·0) = 1、exp(logγ·k) 下溢到 0，正好是参考实现 gamma**delta 的
    # 语义（0**0 = 1、0**k = 0），两条路径在 γ=0 处重新等价。
    log_gamma = torch.log(g.clamp_min(torch.finfo(torch.float32).tiny))

    inner_mask = torch.exp(log_gamma[:, None, None] * expo[None])
    inner_mask = torch.where(causal, inner_mask,
                             torch.zeros((), device=device, dtype=torch.float32))

    pow1d = torch.exp(log_gamma[:, None] * relf[None, :])     # [H,C] = γ^j
    cross_decay = pow1d * g[:, None]                          # γ^(j+1)
    value_decay = pow1d.flip(-1)                              # γ^(C-1-j)
    chunk_decay = pow1d[:, -1] * g                            # γ^C

    return (inner_mask.to(dtype), cross_decay.to(dtype),
            value_decay.to(dtype), chunk_decay.to(dtype))


def chunkwise_batched(q, k, v, gamma, state, chunk_size):
    """Drop-in replacement for ``MultiScaleRetention._chunkwise_retention``.

    q, k: [B,H,L,kd]   v: [B,H,L,vd]   gamma: [H] fp32
    state: [B,H,kd,vd] or None    ->  (output [B,H,L,vd], new_state)
    """
    B, H, L, kd = q.shape
    vd = v.shape[-1]
    if L % chunk_size != 0:
        raise ValueError(f"seq_len {L} not a multiple of chunk_size {chunk_size}")
    nc = L // chunk_size
    C = chunk_size
    dev = q.device

    inner_mask, cross_decay, value_decay, chunk_decay = _decay_tables(
        gamma.to(device=dev, dtype=torch.float32), C, q.dtype, dev)

    has_state = state is not None
    if state is None:
        state = torch.zeros(B, H, kd, vd, device=dev, dtype=q.dtype)

    # [B,H,nc,C,*] views — no copy (L is contiguous)
    qr = q.view(B, H, nc, C, kd)
    kr = k.view(B, H, nc, C, kd)
    vr = v.view(B, H, nc, C, vd)

    # (1) per-chunk state contributions, one batched matmul over all chunks
    U = torch.matmul(kr.transpose(-1, -2),
                     vr * value_decay[None, :, None, :, None])   # [B,H,nc,kd,vd]

    # (2) exclusive scan -> S_prev[i] = state before chunk i
    #
    # NOTE(2026-10-10): 原来的写法是 nc 次串行 ``s = decay*s + U[:,:,i]``
    # 外加一次 ``torch.stack`` —— nc=16 时 33 次内核发射、实测 **591 µs**，
    # 而每个参与的张量只有 0.5 MB：几乎全部是发射延迟。
    #
    # 关键观察是 ``decay = chunk_decay`` 在整个循环里**是常量**（它是 [H] 的
    # 每头常数，不随 chunk 变），于是有闭式解
    #
    #     S_prev[i] = d^i · state + Σ_{j<i} d^(i-1-j) · U_j
    #
    # 这正是对 ``T[h,i,j] = d_h^(i-1-j)``（下三角）的一次批量矩阵乘：
    #
    #     S_all = T[None] @ U.reshape(B,H,nc,kd*vd)          [B,H,nc+1,kd*vd]
    #
    # 多出来的一行 i=nc 恰好就是 new_state。**所有指数非负**，因此不存在
    # §2.1 那种 ``γ^-j`` 溢出——d 再小也只是下溢到 0，那本就代表历史被忘光。
    #
    # NOTE(2026-10-10): 之前试过对数深度（Hillis-Steele）扫描把启动数从 2·nc
    # 降到 ~2·log2(nc)，**反而慢 2.8 倍**（3004 vs 1078 ms），内核数还从
    # 10116 涨到 11472 —— 因为每轮都要 ``torch.cat`` 物化一个
    # [B,H,nc,kd,vd] 张量，既贵又破坏 inductor 的融合。这里的闭式解不物化任何
    # 额外的大张量（``T`` 只有 H·(nc+1)·nc 个元素），所以是另一回事。
    #     Collected in a list and stacked: writing into a torch.empty() buffer
    #     with slice assignment makes autograd emit CopySlices and leaves the
    #     untouched-in-backward buffer as uninitialised memory.
    # 同样要夹下限：expo_t 在 (i=0, j=0) 处为 -1，clamp_min(0) 后是 0，
    # 若 log_d 是 -inf 则 -inf*0 = NaN 会留在 T 的对角线上。
    log_d = (torch.log(gamma.to(device=dev, dtype=torch.float32)
                       .clamp_min(torch.finfo(torch.float32).tiny))
             * float(chunk_size))
    idx = torch.arange(nc + 1, device=dev)
    expo_t = idx[:, None] - torch.arange(nc, device=dev)[None, :] - 1
    valid_t = expo_t >= 0
    T = torch.exp(log_d[:, None, None] * expo_t.clamp_min(0).to(torch.float32)[None])
    T = torch.where(valid_t[None], T, torch.zeros((), device=dev, dtype=torch.float32))
    T = T.to(q.dtype)                                          # [H,nc+1,nc]

    U_flat = U.reshape(B, H, nc, kd * vd)
    S_all = torch.matmul(T[None], U_flat)                      # [B,H,nc+1,kd*vd]
    if has_state:
        t_state = torch.exp(log_d[:, None] * idx.to(torch.float32)[None]).to(q.dtype)
        S_all = S_all + t_state[None, :, :, None] * state.reshape(B, H, 1, kd * vd)

    S_prev = S_all[:, :, :nc].reshape(B, H, nc, kd, vd)
    new_state = S_all[:, :, nc].reshape(B, H, kd, vd)

    # (3) inner chunk, one batched matmul
    inner = torch.matmul(
        torch.matmul(qr, kr.transpose(-1, -2)) * inner_mask[None, :, None, :, :], vr)

    # (4) cross chunk, one batched matmul
    cross = torch.matmul(qr * cross_decay[None, :, None, :, None], S_prev)

    out = (inner + cross).reshape(B, H, L, vd)
    return out, new_state
