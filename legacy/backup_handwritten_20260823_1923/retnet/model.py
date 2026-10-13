"""RetNet (Retentive Network) implemented from scratch with basic PyTorch ops.

Paper: Retentive Network: A Successor to Transformer for Large Language Models
       (arXiv:2307.08621v4)

The implementation deliberately avoids high-level/complete modules such as
``nn.Transformer`` or HuggingFace model classes.  Only elementary building
blocks (``nn.Linear``, ``nn.Embedding``, ``nn.LayerNorm``, ``nn.GroupNorm``,
``nn.Dropout``) are used.  The retention mechanism, the multi-scale retention
module, the RetNet block, the xPos-style relative-position rotation, and the
three retention computation paradigms are all written manually.

Supported retention modes
-------------------------
* ``parallel``  : Equation (5) of the paper, used for training.
* ``recurrent`` : Equation (6) of the paper, O(1) state inference.
* ``chunkwise`` : Equation (7) of the paper, linear-complexity long-sequence
                  training/inference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def strip_optimized_prefix(state_dict):
    """Remove ``_orig_mod.`` prefixes saved by ``torch.compile`` wrappers."""
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state_dict.items()}


@dataclass
class RetNetConfig:
    """Configuration of a RetNet language model."""

    vocab_size: int = 8192
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    value_dim: Optional[int] = None  # default: 2 * d_model (paper parameter allocation)
    ffn_dim: Optional[int] = None     # default: 2 * d_model (paper parameter allocation)
    dropout: float = 0.1
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    tie_weights: bool = True
    embed_scale: Optional[float] = None  # default: sqrt(d_model)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs: (x0, x1) -> (-x1, x0)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)


def theta_shift(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply the real-valued xPos rotation of the paper (Equation 5).

    The paper writes the rotation in complex form, ``e^{i n theta}``.  We use
    the equivalent real 2x2 block rotation so that all tensors remain real and
    ``GroupNorm`` can be applied.
    """
    return x * cos + rotate_every_two(x) * sin


class MultiScaleRetention(nn.Module):
    """Equation (5)-(8): multi-scale retention (MSR).

    Shapes
    ------
    x     : [batch, seq, d_model]
    state : [batch, heads, key_dim, value_head_dim] or ``None``
    """

    def __init__(self, d_model: int, num_heads: int, value_dim: Optional[int] = None,
                 dropout: float = 0.0, norm_eps: float = 1e-5):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = d_model // num_heads
        value_dim = value_dim or (2 * d_model)
        if value_dim % num_heads != 0:
            raise ValueError("value_dim must be divisible by num_heads")
        self.value_dim = value_dim
        self.value_head_dim = value_dim // num_heads
        self.scale = self.key_dim ** -0.5

        # Equation (8): gamma = 1 - 2^{-5 - arange(0, h)}.  The decay rates
        # differ across heads and are fixed (not learnable).
        gamma = 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads, dtype=torch.float32))
        self.register_buffer("gamma", gamma, persistent=True)  # [heads]

        # xPos rotation base, paired for the real representation.
        inv_freq = 1.0 / (10000.0 ** torch.linspace(0.0, 1.0, steps=self.key_dim // 2,
                                                     dtype=torch.float32))
        self.register_buffer("inv_freq", inv_freq.repeat_interleave(2), persistent=False)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, value_dim, bias=False)
        self.g_proj = nn.Linear(d_model, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Paper Equation (8): GroupNorm_h, one normalization group per head.
        # We apply it per token (spatial size = 1), which is the Sub-LN
        # behaviour used in the paper and is scale-invariant under the row-wise
        # retention-score normalization.
        self.group_norm = nn.GroupNorm(num_heads, value_dim, eps=norm_eps)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name.endswith("_proj.weight"):
                if name.startswith("q_") or name.startswith("k_") or name.startswith("v_") or name.startswith("g_"):
                    nn.init.xavier_uniform_(p, gain=2 ** -2.5)
                else:
                    nn.init.xavier_uniform_(p, gain=1.0)
            elif isinstance(p, nn.Parameter) and p.dim() == 1:
                nn.init.zeros_(p)

    def _rotations(self, length: int, offset: int, device: torch.device, dtype: torch.dtype):
        position = torch.arange(offset, offset + length, device=device, dtype=torch.float32)
        freqs = torch.outer(position, self.inv_freq.to(device=device, dtype=torch.float32))
        return torch.sin(freqs).to(dtype=dtype), torch.cos(freqs).to(dtype=dtype)

    def _project_and_rotate(self, x: torch.Tensor, offset: int):
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
    def _head_group_norm(x: torch.Tensor, num_heads: int, group_norm: nn.GroupNorm,
                         bsz: int, tgt_len: int):
        # x: [batch, heads, tgt_len, value_head_dim]
        x = x.transpose(1, 2).reshape(bsz * tgt_len, num_heads * (x.shape[-1]))
        x = group_norm(x.unsqueeze(-1)).squeeze(-1)
        return x.reshape(bsz, tgt_len, -1)

    def _decay_mask(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Causal decay matrix D, row-normalized as described in the paper."""
        idx = torch.arange(length, device=device)
        delta = idx[:, None] - idx[None, :]           # [length, length]
        causal = delta >= 0
        exp = delta.masked_fill(~causal, 0)
        mask = self.gamma.to(device=device, dtype=torch.float32)[:, None, None] ** exp  # [heads,L,L]
        mask = mask.masked_fill(~causal, 0.0)
        # Paper: D~_nm = D_nm / sqrt(sum_i D_ni).  This row scalar cancels in
        # GroupNorm, but improves numerical stability.
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        return mask.to(dtype=dtype)

    def _parallel_retention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        bsz, _, tgt_len, _ = q.shape
        mask = self._decay_mask(tgt_len, q.device, q.dtype)  # [heads,L,L]
        scores = torch.matmul(q, k.transpose(-1, -2)) * mask
        # Paper: R~_nm = R_nm / max(sum_i |R_ni|, 1).  Again this per-token
        # scalar is absorbed by GroupNorm, but stabilizes training.
        scores = scores / scores.detach().abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(scores, v)

    def _parallel_final_state(self, k: torch.Tensor, v: torch.Tensor,
                              state: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, _, tgt_len, key_dim = k.shape
        idx = torch.arange(tgt_len, device=k.device)
        # State after a length-T prefix: S_T = gamma^T S_0 + sum_t gamma^{T-1-t} k_t^T v_t
        weights = self.gamma.to(device=k.device, dtype=torch.float32)[:, None] ** (tgt_len - 1 - idx)
        new_state = torch.einsum("bhtk,bhtv,ht->bhkv", k, v, weights.to(k.dtype))
        if state is not None:
            gamma_pow = self.gamma.to(device=k.device, dtype=k.dtype) ** tgt_len
            new_state = gamma_pow.view(1, -1, 1, 1) * state + new_state
        return new_state

    def _recurrent_retention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             state: Optional[torch.Tensor]):
        bsz, _, tgt_len, key_dim = q.shape
        value_head_dim = v.shape[-1]
        if state is None:
            state = torch.zeros(bsz, self.num_heads, key_dim, value_head_dim,
                                device=q.device, dtype=q.dtype)
        outputs = []
        gamma = self.gamma.to(device=q.device, dtype=q.dtype)
        for t in range(tgt_len):
            # Equation (6): S_n = gamma * S_{n-1} + K_n^H V_n
            state = gamma.view(1, -1, 1, 1) * state + \
                k[:, :, t, :].unsqueeze(-1) * v[:, :, t, :].unsqueeze(-2)
            # Retention(X_n) = Q_n S_n
            out_t = torch.sum(q[:, :, t, :].unsqueeze(-1) * state, dim=-2)
            outputs.append(out_t)
        output = torch.stack(outputs, dim=2)  # [batch, heads, tgt_len, value_head_dim]
        return output, state

    def _chunkwise_retention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             state: Optional[torch.Tensor], chunk_size: int):
        bsz, _, tgt_len, key_dim = q.shape
        if tgt_len % chunk_size != 0:
            raise ValueError(f"chunkwise mode requires seq_len ({tgt_len}) to be a multiple "
                             f"of chunk_size ({chunk_size})")
        if state is None:
            state = torch.zeros(bsz, self.num_heads, key_dim, v.shape[-1],
                                device=q.device, dtype=q.dtype)
        gamma = self.gamma.to(device=q.device, dtype=q.dtype)
        outputs = []
        rel = torch.arange(chunk_size, device=q.device)
        delta = rel[:, None] - rel[None, :]
        causal = delta >= 0
        # Un-normalized inner-chunk decay mask; per-token normalization is not
        # required because GroupNorm is scale invariant (see text below Equation 7).
        inner_mask = torch.where(causal, gamma[:, None, None] ** delta.masked_fill(~causal, 0),
                                 torch.zeros_like(delta, dtype=torch.float32))
        # Cross-chunk weight for local position j: gamma^{j+1}
        cross_decay = gamma[:, None] ** (rel + 1)[None, :]   # [heads, chunk]
        # Value weight used when updating the chunk state: gamma^{B-1-j}
        value_decay = gamma[:, None] ** (chunk_size - 1 - rel)[None, :]  # [heads, chunk]
        chunk_decay = gamma ** chunk_size

        for i in range(tgt_len // chunk_size):
            sl = slice(i * chunk_size, (i + 1) * chunk_size)
            qi, ki, vi = q[:, :, sl], k[:, :, sl], v[:, :, sl]

            # Inner chunk, parallel form
            inner_scores = torch.matmul(qi, ki.transpose(-1, -2)) * inner_mask
            inner = torch.matmul(inner_scores, vi)  # [batch, heads, chunk, v_dim]

            # Cross chunk, recurrent form
            cross = torch.matmul(qi * cross_decay[None, :, :, None], state)

            outputs.append(inner + cross)

            # Equation (7): R_i = K_i^H (V_i * zeta) + gamma^B R_{i-1}
            weighted_v = vi * value_decay[None, :, :, None]
            state = chunk_decay.view(1, -1, 1, 1) * state + \
                torch.matmul(ki.transpose(-1, -2), weighted_v)

        output = torch.cat(outputs, dim=2)  # [batch, heads, tgt_len, v_dim]
        return output, state

    def forward(self, x: torch.Tensor, mode: str = "parallel", chunk_size: int = 64,
                state: Optional[torch.Tensor] = None, offset: int = 0,
                return_state: bool = False):
        """Forward pass of one MSR layer.

        Args:
            x: [batch, seq, d_model]
            mode: one of ``parallel``, ``recurrent``, ``chunkwise``
            chunk_size: chunk length for ``chunkwise`` mode
            state: [batch, heads, key_dim, value_head_dim] recurrent state
            offset: absolute position of the first token in ``x``
            return_state: also return the state after processing ``x``
        """
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

        # Y = GroupNorm_h(Concat(head_1, ..., head_h));  MSR = (swish(X W_G) * Y) W_O
        output = self._head_group_norm(output, self.num_heads, self.group_norm, bsz, tgt_len)
        output = F.silu(g) * output
        output = self.dropout(output)
        output = self.out_proj(output)
        if return_state:
            return output, new_state
        return output


class RetNetBlock(nn.Module):
    """Equation (9): pre-LN MSR block + pre-LN FFN block."""

    def __init__(self, config: RetNetConfig, layer_idx: int = 0):
        super().__init__()
        self.msr = MultiScaleRetention(config.d_model, config.num_heads,
                                       value_dim=config.value_dim,
                                       dropout=config.dropout,
                                       norm_eps=config.norm_eps)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        ffn_dim = config.ffn_dim or (2 * config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, config.d_model, bias=False),
        )
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mode: str = "parallel", chunk_size: int = 64,
                state: Optional[torch.Tensor] = None, offset: int = 0,
                return_state: bool = False):
        msr_out = self.msr(self.norm1(x), mode=mode, chunk_size=chunk_size,
                           state=state, offset=offset, return_state=return_state)
        if return_state:
            y, new_state = msr_out
        else:
            y, new_state = msr_out, None
        y = x + y
        y = y + self.dropout(self.ffn(self.norm2(y)))
        if return_state:
            return y, new_state
        return y


class RetNetLM(nn.Module):
    """Autoregressive RetNet language model with a from-scratch token embedding."""

    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.embed_scale = config.embed_scale if config.embed_scale is not None else math.sqrt(config.d_model)
        self.blocks = nn.ModuleList([RetNetBlock(config, i) for i in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight
        self._init_embedding()

    def _init_embedding(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            for name, p in block.ffn.named_parameters():
                if name.endswith("weight"):
                    nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.lm_head.weight) if not self.config.tie_weights else None

    def forward(self, input_ids: torch.Tensor, mode: str = "parallel", chunk_size: int = 64,
                states: Optional[List[Optional[torch.Tensor]]] = None,
                offset: int = 0, return_state: bool = False):
        """Forward over token ids.

        Args:
            input_ids: [batch, seq] long tensor
            states: list of length ``num_layers`` containing each layer's
                recurrent state, used/returned when ``return_state`` or
                ``mode == 'recurrent'``.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, seq]")
        x = self.embedding(input_ids) * self.embed_scale
        x = self.embed_dropout(x)

        if states is None:
            states = [None] * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("states must contain one entry per RetNet block")

        new_states: List[Optional[torch.Tensor]] = []
        for block, block_state in zip(self.blocks, states):
            if return_state or (mode == "recurrent" and block_state is not None):
                x, ns = block(x, mode=mode, chunk_size=chunk_size, state=block_state,
                              offset=offset, return_state=True)
                new_states.append(ns)
            else:
                x = block(x, mode=mode, chunk_size=chunk_size, state=block_state, offset=offset)
                new_states.append(block_state)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if return_state or mode == "recurrent":
            return logits, new_states
        return logits

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, repetition_penalty: float = 1.0,
                 eos_id: Optional[int] = None, device: Optional[torch.device] = None,
                 penalize_prompt: bool = False, no_repeat_ngram_size: int = 0) -> torch.Tensor:
        """Autoregressive generation using the O(1) recurrent representation.

        ``top_p`` is nucleus sampling (applied after ``top_k``).
        ``repetition_penalty`` discourages repeating tokens; by default only the
        tokens this call generated are penalized, because penalizing the prompt
        pushes chat replies away from the words of the question.
        ``no_repeat_ngram_size`` hard-blocks repeating an n-gram already emitted.
        """
        device = device or next(self.parameters()).device
        ids = input_ids.to(device)
        states = [None] * len(self.blocks)
        # Encode the prompt with the parallel representation, then continue
        # recurrently.  This is much faster than feeding the prompt one token
        # at a time.
        if ids.shape[1] > 1:
            logits, states = self(ids, mode="parallel", return_state=True)
            next_logits = logits[:, -1]
        else:
            next_logits, states = self(ids, mode="recurrent", states=states,
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
                        logits[0, token_id] = score / repetition_penalty if score >= 0 else score * repetition_penalty
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
            next_token = torch.tensor([[next_id]], device=device, dtype=torch.long)
            next_logits, states = self(next_token, mode="recurrent", states=states,
                                       offset=offset, return_state=True)
            next_logits = next_logits[:, -1]
            offset += 1
        return torch.tensor([generated], device=device, dtype=torch.long)
