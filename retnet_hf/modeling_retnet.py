"""HuggingFace-style RetNet implementation — algorithm identical to the
handwritten retnet/model.py, wrapped for the mature ecosystem (Trainer,
Accelerate, safetensors, PreTrainedModel).

Parameter names are kept byte-for-byte identical to the handwritten model so
that ``load_state_dict(old_checkpoint["model_state"], strict=True)`` works
without any renaming.

    model = RetNetForCausalLM(config)
    model.load_state_dict(strip_optimized_prefix(old["model_state"]))
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

#: Opt-in batched-chunk chunkwise retention (see ``retention_fast.py``).
#: Mathematically identical to the reference per-chunk loop but replaces the
#: Python loop's per-chunk slicing + small matmuls with 3 batched matmuls and a
#: tiny state scan.  Verified equivalent by ``scripts/test_fast_chunkwise.py``.
FAST_CHUNKWISE = os.environ.get("RETNET_FAST_CHUNKWISE", "0") == "1"

# NOTE(2026-10-10): 试过给 ``MultiScaleRetention._rotations`` 加一层
# ``(length, offset, device, dtype)`` 键的 sin/cos 表缓存，理由是 profile 里
# ``triton_poi_fused__to_copy*`` 合计 42.9 ms（占一个优化步 5.9%），而这 20 层
# 每次前向算的都是同一张表。**实测没有收益**：缓存开 1080 ms / 关 1084 ms
# （同一供电状态下背靠背），逐字符还原成原始实现也仍是 1078 ms。
# 结论是 inductor 已经把这部分处理掉了，Python 侧的缓存只是在编译区域里
# 多插一层间接。**不要再加这个缓存。**


class RetNetHFConfig(PretrainedConfig):
    model_type = "retnet_hf"

    def __init__(self, vocab_size=8192, d_model=512, num_layers=6, num_heads=8,
                 value_dim=None, ffn_dim=None, dropout=0.1, norm_eps=1e-5,
                 tie_weights=True, embed_scale=None, prompt_loss_weight=0.1,
                 use_gla=False, use_swiglu=False, use_growth_ln=False,
                 growth_old_dim=None, checkpoint_layers=-1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.value_dim = value_dim if value_dim is not None else 2 * d_model
        self.ffn_dim = ffn_dim if ffn_dim is not None else 2 * d_model
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.tie_weights = tie_weights
        self.embed_scale = embed_scale
        self.prompt_loss_weight = prompt_loss_weight
        self.use_gla = use_gla
        self.use_swiglu = use_swiglu
        self.use_growth_ln = use_growth_ln
        self.growth_old_dim = growth_old_dim
        #: 只对前 N 层做梯度检查点；-1 = 全部（沿用 HF 的全局开关）。
        #: 分层取舍的依据：重算开销与显存占用都随被检查点的层数线性变化，
        #: 而激活值总量也是。取中间值可以在同样的 8 GiB 里换到更大的物理 batch，
        #: 而更大的 batch 又能摊薄内核启动数（见 docs/15 §0.2）。
        self.checkpoint_layers = checkpoint_layers

    @property
    def key_dim(self) -> int:
        return self.d_model // self.num_heads


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)


def theta_shift(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_every_two(x) * sin


class MultiScaleRetention(nn.Module):
    def __init__(self, config: RetNetHFConfig):
        super().__init__()
        d_model, num_heads, value_dim = config.d_model, config.num_heads, config.value_dim
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if value_dim % num_heads != 0:
            raise ValueError("value_dim must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = d_model // num_heads
        self.value_dim = value_dim
        self.value_head_dim = value_dim // num_heads
        self.scale = self.key_dim ** -0.5

        gamma = 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads, dtype=torch.float32))
        if config.use_gla:
            # GLA: 可学习逐头门控（初始值 = RetNet 固定公式，训练中自适应）
            self.gamma = nn.Parameter(gamma.clone())
        else:
            self.register_buffer("gamma", gamma, persistent=True)
        inv_freq = 1.0 / (10000.0 ** torch.linspace(0.0, 1.0, steps=self.key_dim // 2,
                                                     dtype=torch.float32))
        self.register_buffer("inv_freq", inv_freq.repeat_interleave(2), persistent=False)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, value_dim, bias=False)
        self.g_proj = nn.Linear(d_model, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.group_norm = nn.GroupNorm(num_heads, value_dim, eps=config.norm_eps)

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name.endswith("_proj.weight"):
                if name.split(".")[-1].startswith(("q_", "k_", "v_", "g_")):
                    nn.init.xavier_uniform_(p, gain=2 ** -2.5)
                else:
                    nn.init.xavier_uniform_(p, gain=1.0)
            elif name == "gamma":
                # GLA 的逐头门控，初值必须是 __init__ 里按 RetNet 固定公式算出的
                # 1 - 2^(-5-h)，训练中再自适应。**绝不能置零**：
                # 之前这里写的是「1 维 Parameter 一律 zeros_」兜底 bias，可本模块
                # 所有 _proj 都是 bias=False，于是唯一的 1 维 Parameter 就是 gamma，
                # 门控被置零 → log(0) = -inf → 掩码对角线 exp(-inf*0) = NaN，
                # 批量 chunkwise 前向整层 NaN（use_gla=True 时每一条路径都中招）。
                # 参考实现用 gamma ** delta，0**0 = 1 良定义，所以两条路径在
                # gamma=0 处还**不等价**。详见 tests/test_gla_gamma_init.py。
                continue
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def _rotations(self, length, offset, device, dtype):
        position = torch.arange(offset, offset + length, device=device, dtype=torch.float32)
        freqs = torch.outer(position, self.inv_freq.to(device=device, dtype=torch.float32))
        return torch.sin(freqs).to(dtype=dtype), torch.cos(freqs).to(dtype=dtype)

    def _project_and_rotate(self, x, offset):
        bsz, tgt_len, _ = x.shape
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, tgt_len, self.num_heads, self.value_head_dim).transpose(1, 2)
        g = self.g_proj(x)
        k = k * self.scale
        sin, cos = self._rotations(tgt_len, offset, x.device, q.dtype)
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)
        return q, k, v, g

    @staticmethod
    def _head_group_norm(x, num_heads, group_norm, bsz, tgt_len):
        """GroupNorm over ``[B,H,L,vhd]``，输出 ``[B,L,H*vhd]``。

        NOTE(2026-10-10): 试过改写成「先在 ``[B,H,L,vhd]`` 上手工算 mean/var
        归一化、最后再 permute 一次」，理由是省掉参考实现里那次物化拷贝。
        **实测反而慢 9.2%**（190 vs 174 ms/微步，真实 Trainer 路径，两轮一致）：
        参考实现把 permute 放在**归一化之前**，那次拷贝随即被逐元素归一化核吸收；
        而放到**之后**，它直接成为 ``out_proj`` 这个外部 cuBLAS 调用的输入，
        再也融不进任何东西。等价性没问题（fp32 相对差 1.6e-7，bf16 5.1e-3 均
        在 1 ULP 量级，见 ``tests/test_head_group_norm.py``），但收益是负的。
        **不要再动这个布局。**
        """
        x = x.transpose(1, 2).reshape(bsz * tgt_len, num_heads * x.shape[-1])
        x = group_norm(x.unsqueeze(-1)).squeeze(-1)
        return x.reshape(bsz, tgt_len, -1)

    def _decay_mask(self, length, device, dtype):
        idx = torch.arange(length, device=device)
        delta = idx[:, None] - idx[None, :]
        causal = delta >= 0
        exp = delta.masked_fill(~causal, 0)
        mask = self.gamma.to(device=device, dtype=torch.float32)[:, None, None] ** exp
        mask = mask.masked_fill(~causal, 0.0)
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        return mask.to(dtype=dtype)

    def _parallel_retention(self, q, k, v):
        bsz, _, tgt_len, _ = q.shape
        mask = self._decay_mask(tgt_len, q.device, q.dtype)
        scores = torch.matmul(q, k.transpose(-1, -2)) * mask
        scores = scores / scores.detach().abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(scores, v)

    def _parallel_final_state(self, k, v, state):
        bsz, _, tgt_len, key_dim = k.shape
        idx = torch.arange(tgt_len, device=k.device)
        weights = self.gamma.to(device=k.device, dtype=torch.float32)[:, None] ** (tgt_len - 1 - idx)
        new_state = torch.einsum("bhtk,bhtv,ht->bhkv", k, v, weights.to(k.dtype))
        if state is not None:
            gamma_pow = self.gamma.to(device=k.device, dtype=k.dtype) ** tgt_len
            new_state = gamma_pow.view(1, -1, 1, 1) * state + new_state
        return new_state

    def _recurrent_retention(self, q, k, v, state):
        bsz, _, tgt_len, key_dim = q.shape
        value_head_dim = v.shape[-1]
        if state is None:
            state = torch.zeros(bsz, self.num_heads, key_dim, value_head_dim,
                                device=q.device, dtype=q.dtype)
        outputs = []
        gamma = self.gamma.to(device=q.device, dtype=q.dtype)
        for t in range(tgt_len):
            state = gamma.view(1, -1, 1, 1) * state + \
                k[:, :, t, :].unsqueeze(-1) * v[:, :, t, :].unsqueeze(-2)
            out_t = torch.sum(q[:, :, t, :].unsqueeze(-1) * state, dim=-2)
            outputs.append(out_t)
        output = torch.stack(outputs, dim=2)
        return output, state

    def _chunkwise_retention(self, q, k, v, state, chunk_size):
        """Equation (7): inner-chunk parallel + cross-chunk recurrent state."""
        if FAST_CHUNKWISE:
            from .retention_fast import chunkwise_batched
            return chunkwise_batched(q, k, v, self.gamma, state, chunk_size)
        bsz, _, tgt_len, key_dim = q.shape
        if tgt_len % chunk_size != 0:
            raise ValueError(f"chunkwise mode requires seq_len ({tgt_len}) to be a "
                             f"multiple of chunk_size ({chunk_size})")
        if state is None:
            state = torch.zeros(bsz, self.num_heads, key_dim, v.shape[-1],
                                device=q.device, dtype=q.dtype)
        gamma = self.gamma.to(device=q.device, dtype=q.dtype)
        outputs = []
        rel = torch.arange(chunk_size, device=q.device)
        delta = rel[:, None] - rel[None, :]
        causal = delta >= 0
        inner_mask = torch.where(causal, gamma[:, None, None] ** delta.masked_fill(~causal, 0),
                                 torch.zeros_like(delta, dtype=torch.float32))
        cross_decay = gamma[:, None] ** (rel + 1)[None, :]
        value_decay = gamma[:, None] ** (chunk_size - 1 - rel)[None, :]
        chunk_decay = gamma ** chunk_size

        for i in range(tgt_len // chunk_size):
            sl = slice(i * chunk_size, (i + 1) * chunk_size)
            qi, ki, vi = q[:, :, sl], k[:, :, sl], v[:, :, sl]
            inner_scores = torch.matmul(qi, ki.transpose(-1, -2)) * inner_mask
            inner = torch.matmul(inner_scores, vi)
            cross = torch.matmul(qi * cross_decay[None, :, :, None], state)
            outputs.append(inner + cross)
            weighted_v = vi * value_decay[None, :, :, None]
            state = chunk_decay.view(1, -1, 1, 1) * state + \
                torch.matmul(ki.transpose(-1, -2), weighted_v)

        output = torch.cat(outputs, dim=2)
        return output, state

    def forward(self, x, mode="parallel", state=None, offset=0, return_state=False,
                chunk_size=64):
        bsz, tgt_len, _ = x.shape
        q, k, v, g = self._project_and_rotate(x, offset)

        if mode == "parallel":
            output, new_state = self._parallel_retention(q, k, v), None
            if return_state:
                new_state = self._parallel_final_state(k, v, state)
        elif mode == "recurrent":
            output, new_state = self._recurrent_retention(q, k, v, state)
        elif mode == "chunkwise":
            output, new_state = self._chunkwise_retention(q, k, v, state, chunk_size)
        else:
            raise ValueError(f"unknown mode: {mode}")

        output = self._head_group_norm(output, self.num_heads, self.group_norm, bsz, tgt_len)
        output = F.silu(g) * output
        output = self.dropout(output)
        output = self.out_proj(output)
        if return_state:
            return output, new_state
        return output


class GrowthLayerNorm(nn.Module):
    """Two-segment LayerNorm for function-preserving width growth.

    The old ``old_dim`` channels are normalized exactly as before (statistics
    unaffected by new channels); the new channels are scaled independently with
    ``gamma_new``/``beta_new`` and stay a zero side branch while
    ``gamma_new=0, beta_new=0``.  This is what makes pre-LN width expansion
    strictly function-preserving (BERT2BERT-style FPI for LayerNorm).

    state_dict keys: ``ln_old.weight``, ``ln_old.bias``, ``gamma_new``, ``beta_new``.
    """

    def __init__(self, old_dim: int, new_dim: int, eps: float = 1e-5,
                 gamma_new: float = 1.0, beta_new: float = 0.0):
        super().__init__()
        self.ln_old = nn.LayerNorm(old_dim, eps=eps)
        self.gamma_new = nn.Parameter(torch.full((new_dim - old_dim,), gamma_new))
        self.beta_new = nn.Parameter(torch.full((new_dim - old_dim,), beta_new))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old = self.ln_old(x[..., : self.ln_old.normalized_shape[0]])
        new = x[..., self.ln_old.normalized_shape[0]:] * self.gamma_new + self.beta_new
        return torch.cat([old, new], dim=-1)


class RetNetBlock(nn.Module):
    def _norm(self, config: RetNetHFConfig, dim: int):
        if config.use_growth_ln:
            return GrowthLayerNorm(config.growth_old_dim, dim, eps=config.norm_eps,
                                   gamma_new=1.0, beta_new=0.0)
        return nn.LayerNorm(dim, eps=config.norm_eps)

    def __init__(self, config: RetNetHFConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        #: 该层是否做梯度检查点。-1 表示「跟随 HF 的全局开关」。
        self.checkpoint_layers = getattr(config, "checkpoint_layers", -1)
        self.msr = MultiScaleRetention(config)
        self.norm1 = self._norm(config, config.d_model)
        self.gradient_checkpointing = False
        if config.use_swiglu:
            # SwiGLU FFN：y = silu(xW_gate) * (xW_up) -> W_down
            self.ffn = nn.Sequential()  # placeholder; real modules below
            self.gate_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
            self.up_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
            self.down_proj = nn.Linear(config.ffn_dim, config.d_model, bias=False)
            self.swiglu = True
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.ffn_dim, bias=False),
                nn.GELU(),
                nn.Linear(config.ffn_dim, config.d_model, bias=False),
            )
            self.swiglu = False
        self.norm2 = self._norm(config, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def _block_forward(self, x, mode, state, offset, return_state, chunk_size=64):
        msr_out = self.msr(self.norm1(x), mode=mode, state=state, offset=offset,
                           return_state=return_state, chunk_size=chunk_size)
        if return_state:
            y, new_state = msr_out
        else:
            y, new_state = msr_out, None
        y = x + y
        if self.swiglu:
            ffn_out = F.silu(self.gate_proj(self.norm2(y))) * self.up_proj(self.norm2(y))
            ffn_out = self.down_proj(ffn_out)
        else:
            ffn_out = self.ffn(self.norm2(y))
        y = y + self.dropout(ffn_out)
        if return_state:
            return y, new_state
        return y

    def forward(self, x, mode="parallel", state=None, offset=0, return_state=False,
                chunk_size=64):
        do_ckpt = self.gradient_checkpointing and self.training
        if do_ckpt and 0 <= self.checkpoint_layers <= self.layer_idx:
            do_ckpt = False          # 分层策略：只检查点前 checkpoint_layers 层
        if do_ckpt:
            return torch.utils.checkpoint.checkpoint(
                self._block_forward, x, mode, state, offset, return_state, chunk_size,
                use_reentrant=False)
        return self._block_forward(x, mode, state, offset, return_state, chunk_size)


class RetNetForCausalLM(PreTrainedModel):
    """RetNet language model with a from-scratch token embedding.

    Forward signature is Trainer-compatible:
        forward(input_ids, labels=None, wcodes=None, ...) -> CausalLMOutput
    When labels+wcodes are given the loss is the wcode-weighted cross entropy
    (wcode: 0=pad, 1=reply, 2=prompt, 3=plain LM) exactly like train.py.
    """
    config_class = RetNetHFConfig
    base_model_prefix = "retnet"
    supports_gradient_checkpointing = True

    #: 本模型**不使用** ``num_items_in_batch`` 做损失归一化（loss 已按权重码
    #: ``w.sum()`` 归一），但 ``forward`` 签名末尾的 ``**_kwargs`` 会让 HF 的启发式
    #: 判定「模型自己会处理归一化」，从而**跳过** ``loss = loss / grad_accum_steps``。
    #: 后果是梯度被放大 accum 倍、等效学习率成倍上升 —— 静默，且只在 accum>1 时发作。
    #: HF 按 **``accepts_loss_kwargs``**（不是 ``model_accepts_loss_kwargs``）查找；
    #: 找不到时它会去猜 forward 签名，本项目会被误判为 True。
    accepts_loss_kwargs = False
    model_accepts_loss_kwargs = False

    def __init__(self, config: RetNetHFConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.embed_scale = config.embed_scale if config.embed_scale is not None \
            else config.d_model ** 0.5
        self.blocks = nn.ModuleList([RetNetBlock(config, i)
                                     for i in range(config.num_layers)])
        if config.use_growth_ln:
            self.final_norm = GrowthLayerNorm(config.growth_old_dim, config.d_model,
                                              eps=config.norm_eps,
                                              gamma_new=1.0, beta_new=0.0)
        else:
            self.final_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight
            self._tied_weights_keys = ["lm_head.weight"]
        self._init_weights_local()

    def _init_weights_local(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            if block.swiglu:
                nn.init.xavier_uniform_(block.gate_proj.weight)
                nn.init.xavier_uniform_(block.up_proj.weight)
                nn.init.xavier_uniform_(block.down_proj.weight)
            else:
                for name, p in block.ffn.named_parameters():
                    if name.endswith("weight"):
                        nn.init.xavier_uniform_(p)
            block.msr._init_weights()
        if not self.config.tie_weights:
            nn.init.xavier_uniform_(self.lm_head.weight)

    def tie_weights(self):
        """共享 lm_head 与 embedding（from_pretrained 后恢复 tie）。"""
        if self.config.tie_weights:
            self.lm_head.weight = self.embedding.weight

    def set_checkpoint_layers(self, n: int):
        """只对前 ``n`` 层做梯度检查点（n<0 = 全部）。

        必须在 HF 启用梯度检查点**之后**调用，因为它会把所有层的
        ``gradient_checkpointing`` 置 True。
        """
        self.config.checkpoint_layers = n
        for b in self.blocks:
            b.checkpoint_layers = n
        return n

    def _forward_hidden(self, input_ids, mode="parallel", states=None, offset=0,
                        return_state=False, chunk_size=64, return_hidden=False):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, seq]")
        x = self.embedding(input_ids) * self.embed_scale
        x = self.embed_dropout(x)
        if states is None:
            states = [None] * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("states must contain one entry per block")
        new_states: List[Optional[torch.Tensor]] = []
        for block, block_state in zip(self.blocks, states):
            if return_state or (mode == "recurrent" and block_state is not None):
                x, ns = block(x, mode=mode, state=block_state, offset=offset,
                              return_state=True, chunk_size=chunk_size)
                new_states.append(ns)
            else:
                x = block(x, mode=mode, state=block_state, offset=offset,
                          chunk_size=chunk_size)
                new_states.append(block_state)
        x = self.final_norm(x)
        if return_hidden:
            # 融合交叉熵需要的是 LM head 的**输入**（hidden），不是输出。
            # 这里提前返回可以完全不物化 [B,L,V]。
            if return_state or mode == "recurrent":
                return x, new_states
            return x
        logits = self.lm_head(x)
        if return_state or mode == "recurrent":
            return logits, new_states
        return logits

    def forward(self, input_ids, labels=None, wcodes=None, mode=None,
                states=None, offset=0, return_state=False, chunk_size=None, **_kwargs):
        # NOTE(2026-10-10): 试过在这里调 torch.compiler.cudagraph_mark_step_begin()
        # 以启用 torch.compile(mode="reduce-overhead") 的 CUDA Graphs。**无效**——
        # 错误从损失移到 _forward_hidden，仍是 "accessing tensor output of
        # CUDAGraphs that has been overwritten / Gradient addition node due to
        # multiple use of tensor"：这是 CUDA Graphs 与梯度累积（同一张图跑 4 次、
        # 各自反向）的硬冲突，不是漏调一个 API 能解决的。已回滚。
        # NOTE(2026-10-10): 上面这个默认值以前写的是 64，而 HF Trainer 调用模型时
        # **不会**传这个关键字，于是 ``config.chunk_size`` 和任务 JSON 里的
        # ``chunk_size`` 一直是**死字段**——项目以为在跑 128，实际永远 64。
        # 实测：``config.chunk_size`` 设 64/128/256 的 loss **逐位相同**，而显式
        # 传 ``chunk_size=`` 会改变 loss。``bench_step.py`` 的
        # ``--sweep chunk_size=...`` 因此扫的是空变量，``docs/15`` §6 的
        # 「chunk 512/1024 反而更慢，最优是 128」不成立。
        # 现改成 None 再回落到 config，任务配置才真正生效。
        # ⚠️ 对已经在跑的任务，这会让 chunk 从 64 跳到配置值 —— 数值会跳变，
        # 只能在**新开一轮**时启用，不要热切。
        if chunk_size is None:
            chunk_size = int(getattr(self.config, "chunk_size", 64) or 64)
        if mode is None:
            mode = getattr(self.config, "mode", "parallel")
        _fused = (os.environ.get("RETNET_FUSED_CE") == "1"
                  and labels is not None and wcodes is not None)
        logits = None
        # `loss` 必须在所有分支之前初始化：不传 labels 时（普通推理、prefill、
        # 以及 HF 的 generate 路径）上面几个分支都不会赋值，line 536 的
        # CausalLMOutput(loss=loss, ...) 会直接 UnboundLocalError。
        loss = None
        if _fused:
            # 融合路径：LM head 的输入直接进 fused CE，完全不物化 [B,L,V]
            hidden, new_states = self._forward_hidden(
                input_ids, mode=mode, states=states, offset=offset,
                return_state=True, chunk_size=chunk_size, return_hidden=True)
            B_, L_, D_ = hidden.shape
            w = torch.zeros_like(wcodes, dtype=torch.float32)
            w = torch.where(wcodes == 1, torch.ones_like(w), w)
            w = torch.where(wcodes == 2,
                            torch.full_like(w, self.config.prompt_loss_weight), w)
            w = torch.where(wcodes == 3, torch.ones_like(w), w)
            from .fused_ce import fused_linear_cross_entropy
            loss = fused_linear_cross_entropy(
                hidden.reshape(B_ * L_, D_), self.lm_head.weight,
                labels.reshape(-1), w.reshape(-1))
        elif return_state or mode == "recurrent" or mode == "chunkwise":
            logits, new_states = self._forward_hidden(input_ids, mode=mode, states=states,
                                                      offset=offset, return_state=True,
                                                      chunk_size=chunk_size)
        else:
            logits = self._forward_hidden(input_ids, mode=mode, states=states, offset=offset)
            new_states = None

        if not _fused:
            # 非融合路径：沿用原来的显式 fp32 上采样。
            # NOTE(2026-10-10): 试过去掉 ``logits.float()``（让 ``F.cross_entropy``
            # 直接吃 bf16）。loss 逐位一致但**步时 0% 变化**——profile 把这两个
            # 融合核记成 3.9%，是归因假象。故保留原状。
            if labels is not None and wcodes is not None:
                logits_f = logits.float()
                token_loss = F.cross_entropy(
                    logits_f.reshape(-1, logits_f.shape[-1]), labels.reshape(-1),
                    reduction="none")
                w = torch.zeros_like(wcodes, dtype=torch.float32)
                w = torch.where(wcodes == 1, torch.ones_like(w), w)
                w = torch.where(wcodes == 2,
                                torch.full_like(w, self.config.prompt_loss_weight), w)
                w = torch.where(wcodes == 3, torch.ones_like(w), w)
                loss = (token_loss * w.reshape(-1)).sum() / w.sum().clamp_min(1.0)
            elif labels is not None:
                logits_f = logits.float()
                loss = F.cross_entropy(logits_f.reshape(-1, logits_f.shape[-1]),
                                       labels.reshape(-1))

        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, repetition_penalty: float = 1.0,
                 eos_id: Optional[int] = None, device: Optional[torch.device] = None,
                 penalize_prompt: bool = False, no_repeat_ngram_size: int = 0,
                 on_token=None) -> torch.Tensor:
        """Autoregressive generation (same algorithm/sampling as handwritten).

        ``on_token(token_id)`` is called for every accepted token, in order, as
        soon as it is sampled.  Returning a truthy value stops generation
        immediately (the token is still part of the returned sequence).  Used by
        the interactive play script to stream output and to cut a reply at the
        first ``用户：`` / ``助手：`` turn boundary.
        """
        device = device or next(self.parameters()).device
        ids = input_ids.to(device)
        states = [None] * len(self.blocks)
        if ids.shape[1] > 1:
            logits, states = self._forward_hidden(ids, mode="parallel", return_state=True)
            next_logits = logits[:, -1]
        else:
            next_logits, states = self._forward_hidden(ids, mode="recurrent", states=states,
                                                       offset=0, return_state=True)
            next_logits = next_logits[:, -1]
        generated = list(ids[0].tolist())
        prompt_len = ids.shape[1]
        offset = ids.shape[1]

        for _ in range(max_new_tokens):
            penalty_source = generated if penalize_prompt else generated[prompt_len:]
            banned = []
            if no_repeat_ngram_size and len(generated) >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                suffix = tuple(generated[-(n - 1):]) if n > 1 else ()
                for i in range(len(generated) - n + 1):
                    if tuple(generated[i:i + n - 1]) == suffix:
                        banned.append(generated[i + n - 1])
            if temperature > 0:
                logits = next_logits / temperature
                if repetition_penalty and repetition_penalty != 1.0:
                    for token_id in set(penalty_source):
                        score = logits[0, token_id]
                        logits[0, token_id] = score / repetition_penalty if score >= 0 \
                            else score * repetition_penalty
                for token_id in banned:
                    logits[0, token_id] = -torch.inf
                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    top_values, _ = torch.topk(logits, k)
                    logits[logits < top_values[:, -1:]] = -torch.inf
                probs = F.softmax(logits, dim=-1)
                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum = sorted_probs.cumsum(dim=-1)
                    remove = cumsum > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    filtered = torch.zeros_like(probs)
                    filtered.scatter_(1, sorted_indices[~remove].unsqueeze(0),
                                      sorted_probs[~remove].unsqueeze(0))
                    probs = filtered / filtered.sum(dim=-1, keepdim=True)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                logits = next_logits.clone()
                for token_id in banned:
                    logits[0, token_id] = -torch.inf
                next_id = logits.argmax(dim=-1).item()
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            if on_token is not None and on_token(next_id):
                break
            next_token = torch.tensor([[next_id]], device=device, dtype=torch.long)
            next_logits, states = self._forward_hidden(next_token, mode="recurrent",
                                                       states=states, offset=offset,
                                                       return_state=True)
            next_logits = next_logits[:, -1]
            offset += 1
        return torch.tensor([generated], device=device, dtype=torch.long)
