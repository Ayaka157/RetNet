from __future__ import annotations



import json

import os

from typing import List, Optional, Tuple



import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset

from transformers import PreTrainedModel, PretrainedConfig



import torch_xla

# torch_xla's own gradient checkpointing. Upstream torch.utils.checkpoint is

# unusable on this stack: torch<2.9 has no torch.xla module (crashes in

# _get_device_module), and XLA's CSE pass would merge the backward recompute

# with the original forward, silently undoing the memory saving

# (pytorch/xla#7024). torch_xla's version adds xm.optimization_barrier_ on the

# recompute inputs and handles the XLA RNG seed (reentrant mode only, #8729).

from torch_xla.utils.checkpoint import checkpoint as _xla_checkpoint



torch.set_default_dtype(torch.bfloat16)





# ==================== 模型定义（与 checkpoint 状态字典键逐字一致）====================

class RetNetHFConfig(PretrainedConfig):

    model_type = "retnet_hf"



    def __init__(self, vocab_size=8192, d_model=512, num_layers=6, num_heads=8,

                 value_dim=None, ffn_dim=None, dropout=0.1, norm_eps=1e-5,

                 tie_weights=True, embed_scale=None, prompt_loss_weight=0.1,

                 use_gla=False, use_swiglu=False, use_growth_ln=False,

                 growth_old_dim=None, **kwargs):

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

        # The decay gates live extremely close to 1.0 (up to 1 - 2**-20). bf16

        # only resolves ~2**-8 there, so a bf16 gate collapses 12 of 16 heads

        # to exactly 1.0 (no decay at all) and quantizes away gradient updates

        # -- the GLA gate would be registered but numerically dead. Keep gamma

        # in fp32 (manual mixed precision); train_worker restores the fp32

        # master weight after model.to(dtype=bfloat16).

        if config.use_gla:

            # GLA: 可学习逐头门控（初始值 = RetNet 固定公式，fp32 主权重，训练中自适应）

            self.gamma = nn.Parameter(gamma.clone())

        else:

            self.register_buffer("gamma", gamma.clone(), persistent=True)

        inv_freq = 1.0 / (10000.0 ** torch.linspace(0.0, 1.0, steps=self.key_dim // 2,

                                                     dtype=torch.float32))

        self.register_buffer("inv_freq", inv_freq.repeat_interleave(2).to(torch.bfloat16), persistent=False)



        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        self.v_proj = nn.Linear(d_model, value_dim, bias=False)

        self.g_proj = nn.Linear(d_model, value_dim, bias=False)

        self.out_proj = nn.Linear(value_dim, d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        self.group_norm = nn.GroupNorm(num_heads, value_dim, eps=config.norm_eps)



    def _init_weights(self):

        # Keep GroupNorm defaults (weight=1, bias=0) and preserve the

        # RetNet gamma initialization. Only initialize the projection matrices.

        for name, p in self.named_parameters():

            if not name.endswith(".weight"):

                continue

            if name.startswith(("q_proj.", "k_proj.", "v_proj.", "g_proj.")):

                nn.init.xavier_uniform_(p, gain=2 ** -2.5)

            elif name.startswith("out_proj."):

                nn.init.xavier_uniform_(p, gain=1.0)



    def _rotations(self, length, offset, device, dtype):

        # Keep the whole rotary-frequency path in the surrounding compute dtype.

        position = torch.arange(offset, offset + length, device=device, dtype=dtype)

        inv_freq = self.inv_freq.to(device=device, dtype=dtype)

        freqs = torch.outer(position, inv_freq)

        return torch.sin(freqs), torch.cos(freqs)



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

        # Normalization in fp32 (manual mixed precision); cast back to bf16.

        dtype = x.dtype

        x = x.transpose(1, 2).reshape(bsz * tgt_len, num_heads * x.shape[-1])

        x = F.group_norm(x.unsqueeze(-1).float(), group_norm.num_groups,

                         group_norm.weight.float(), group_norm.bias.float(),

                         group_norm.eps).squeeze(-1)

        return x.to(dtype).reshape(bsz, tgt_len, -1)



    def _decay_mask(self, length, device, dtype):

        idx = torch.arange(length, device=device)

        delta = idx[:, None] - idx[None, :]

        causal = delta >= 0

        exp = delta.masked_fill(~causal, 0).to(torch.float32)

        gamma = self.gamma.to(device=device, dtype=torch.float32)

        # Exponentiate the fp32 gate in fp32; cast to compute dtype at the end.

        mask = gamma[:, None, None] ** exp

        mask = mask.masked_fill(~causal, 0.0)

        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)

        return mask.to(dtype)



    def _parallel_retention(self, q, k, v):

        bsz, _, tgt_len, _ = q.shape

        mask = self._decay_mask(tgt_len, q.device, q.dtype)

        scores = torch.matmul(q, k.transpose(-1, -2)) * mask

        scores = scores / scores.detach().abs().sum(dim=-1, keepdim=True).clamp_min(1.0)

        return torch.matmul(scores, v)



    def _parallel_final_state(self, k, v, state):

        bsz, _, tgt_len, key_dim = k.shape

        idx = torch.arange(tgt_len, device=k.device)

        gamma32 = self.gamma.to(device=k.device, dtype=torch.float32)

        weights = (gamma32[:, None] ** (tgt_len - 1 - idx).to(torch.float32)).to(k.dtype)

        new_state = torch.einsum("bhtk,bhtv,ht->bhkv", k, v, weights)

        if state is not None:

            gamma_pow = (gamma32 ** tgt_len).to(k.dtype)

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

        """Equation (7), as an unrolled chunk loop (numerically identical to scan).



        The loop is a recurrent state update per chunk. The module parameter

        set is unchanged, so the resulting state_dict remains compatible with

        the ordinary PyTorch/HF model and checkpoints.

        """

        # q/k/v come straight from bf16 projections; no cast needed.

 

        bsz, _, tgt_len, key_dim = q.shape

        if tgt_len % chunk_size != 0:

            raise ValueError(f"chunkwise mode requires seq_len ({tgt_len}) to be a "

                             f"multiple of chunk_size ({chunk_size})")

        if state is None:

            state = torch.zeros(

                bsz, self.num_heads, key_dim, v.shape[-1],

                device=q.device, dtype=torch.bfloat16)

        else:

            state = state.to(device=q.device, dtype=torch.bfloat16)



        n_chunks = tgt_len // chunk_size

        # Decay math in fp32 (bf16 cannot resolve gamma this close to 1.0);

        # cast to bf16 once, at the boundary into the bf16 scan.

        gamma32 = self.gamma.to(device=q.device, dtype=torch.float32)



        rel = torch.arange(chunk_size, device=q.device, dtype=torch.float32)

        delta = rel[:, None] - rel[None, :]

        causal = delta >= 0

        exp = delta.masked_fill(~causal, 0)

        inner_mask = (gamma32[:, None, None] ** exp).to(torch.bfloat16)

        inner_mask = inner_mask.masked_fill(~causal, 0.0)

        cross_decay = (gamma32[:, None] ** (rel + 1)).to(torch.bfloat16)

        value_decay = (gamma32[:, None] ** (chunk_size - 1 - rel)).to(torch.bfloat16)

        chunk_decay = (gamma32 ** chunk_size).to(torch.bfloat16)



        # Unrolled chunk recurrence. Numerically identical to the previous

        # torch_xla.experimental.scan formulation (scan is just a While loop

        # over this same body), but traced by the standard XLA lazy-tensor

        # path: scan's AOT/HLO-While machinery mis-types one fn_computation

        # parameter (f32 vs bf16) on this stack, which is the crash root cause.

        # Activation memory is bounded by block-level gradient checkpointing

        # (enabled in train_worker).

        q_chunks = q.reshape(bsz, self.num_heads, n_chunks, chunk_size, key_dim)

        k_chunks = k.reshape(bsz, self.num_heads, n_chunks, chunk_size, key_dim)

        v_chunks = v.reshape(

            bsz, self.num_heads, n_chunks, chunk_size, v.shape[-1])



        cur_state = state

        outputs = []

        for i in range(n_chunks):

            qi = q_chunks[:, :, i]

            ki = k_chunks[:, :, i]

            vi = v_chunks[:, :, i]

            inner_scores = torch.matmul(qi, ki.transpose(-1, -2)) * inner_mask

            inner = torch.matmul(inner_scores, vi)

            cross = torch.matmul(

                qi * cross_decay[None, :, :, None], cur_state)

            outputs.append(inner + cross)

            weighted_v = vi * value_decay[None, :, :, None]

            cur_state = (

                chunk_decay.view(1, -1, 1, 1) * cur_state

                + torch.matmul(ki.transpose(-1, -2), weighted_v)

            )

        final_state = cur_state



        # [B, H, N, C, V] -> [B, H, T, V]

        output = torch.stack(outputs, dim=2).reshape(

            bsz, self.num_heads, tgt_len, v.shape[-1])

        return output, final_state



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

        # Normalization in fp32 (manual mixed precision); cast back to bf16.

        dtype = x.dtype

        x = x.float()

        old = F.layer_norm(x[..., : self.ln_old.normalized_shape[0]],

                           self.ln_old.normalized_shape,

                           self.ln_old.weight.float(), self.ln_old.bias.float(),

                           self.ln_old.eps)

        new = (x[..., self.ln_old.normalized_shape[0]:] * self.gamma_new.float()

               + self.beta_new.float())

        return torch.cat([old, new], dim=-1).to(dtype)





class RetNetBlock(nn.Module):

    def _norm(self, config: RetNetHFConfig, dim: int):

        if config.use_growth_ln:

            return GrowthLayerNorm(config.growth_old_dim, dim, eps=config.norm_eps,

                                   gamma_new=1.0, beta_new=0.0)

        return nn.LayerNorm(dim, eps=config.norm_eps)



    def __init__(self, config: RetNetHFConfig):

        super().__init__()

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

        if self.gradient_checkpointing and self.training:

            return _xla_checkpoint(

                self._block_forward, x, mode, state, offset, return_state, chunk_size)

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



    def __init__(self, config: RetNetHFConfig):

        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.embed_dropout = nn.Dropout(config.dropout)

        self.embed_scale = config.embed_scale if config.embed_scale is not None \

            else config.d_model ** 0.5

        self.blocks = nn.ModuleList([RetNetBlock(config) for _ in range(config.num_layers)])

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



    def _forward_hidden(self, input_ids, mode="parallel", states=None, offset=0,

                        return_state=False, chunk_size=64):

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

        logits = self.lm_head(x)

        if return_state or mode == "recurrent":

            return logits, new_states

        return logits



    def forward(self, input_ids, labels=None, wcodes=None, mode=None,

                states=None, offset=0, return_state=False, chunk_size=64, **_kwargs):

        if mode is None:

            mode = getattr(self.config, "mode", "parallel")

        if return_state or mode == "recurrent" or mode == "chunkwise":

            logits, new_states = self._forward_hidden(input_ids, mode=mode, states=states,

                                                      offset=offset, return_state=True,

                                                      chunk_size=chunk_size)

        else:

            logits = self._forward_hidden(input_ids, mode=mode, states=states, offset=offset)

            new_states = None



        loss = None

        if labels is not None and wcodes is not None:

            logits_f = logits.float()

            token_loss = F.cross_entropy(

                logits_f.reshape(-1, logits_f.shape[-1]), labels.reshape(-1), reduction="none")

            w = torch.zeros_like(wcodes, dtype=torch.float32)

            w = torch.where(wcodes == 1, torch.ones_like(w), w)

            w = torch.where(wcodes == 2, torch.full_like(w, self.config.prompt_loss_weight), w)

            w = torch.where(wcodes == 3, torch.ones_like(w), w)

            loss = (token_loss * w.reshape(-1)).sum() / w.sum().clamp_min(1.0)

        elif labels is not None:

            logits_f = logits.float()

            loss = F.cross_entropy(logits_f.reshape(-1, logits_f.shape[-1]), labels.reshape(-1))



        from transformers.modeling_outputs import CausalLMOutput

        return CausalLMOutput(loss=loss, logits=logits)



    @torch.inference_mode()

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32,

                 temperature: float = 1.0, top_k: Optional[int] = None,

                 top_p: Optional[float] = None, repetition_penalty: float = 1.0,

                 eos_id: Optional[int] = None, device: Optional[torch.device] = None,

                 penalize_prompt: bool = False, no_repeat_ngram_size: int = 0) -> torch.Tensor:

        """Autoregressive generation (same algorithm/sampling as handwritten)."""

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

            next_token = torch.tensor([[next_id]], device=device, dtype=torch.long)

            next_logits, states = self._forward_hidden(next_token, mode="recurrent",

                                                       states=states, offset=offset,

                                                       return_state=True)

            next_logits = next_logits[:, -1]

            offset += 1

        return torch.tensor([generated], device=device, dtype=torch.long)





# ==================== 数据 ====================

class PackedChatDataset(Dataset):

    """行布局 (seq_len+1)：input_ids=row[:-1], labels=row[1:], wcodes=codes[1:]"""

    def __init__(self, ids_path, wcode_path, seq_len):

        self.ids = np.load(ids_path, mmap_mode="r")

        self.codes = np.load(wcode_path, mmap_mode="r")

        assert self.ids.ndim == 2 and self.ids.shape == self.codes.shape

        assert self.ids.shape[1] == seq_len + 1, \

            f"rows={self.ids.shape[1]} != seq_len+1={seq_len + 1}"

        self.seq_len = seq_len



    def __len__(self):

        return self.ids.shape[0]



    def __getitem__(self, idx):

        row = self.ids[idx]

        return {

            "input_ids": torch.from_numpy(row[:-1].astype(np.int64)),

            "labels": torch.from_numpy(row[1:].astype(np.int64)),

            "wcodes": torch.from_numpy(self.codes[idx][1:].astype(np.int64)),

        }





class ChatDataCollator:

    def __call__(self, batch):

        return {

            "input_ids": torch.stack([b["input_ids"] for b in batch]),

            "labels": torch.stack([b["labels"] for b in batch]),

            "wcodes": torch.stack([b["wcodes"] for b in batch]),

        }









# ==================== torch_xla / distributed training ====================

from dataclasses import asdict, dataclass

from pathlib import Path

import math

import random

import time



import torch

from torch.optim import AdamW

from torch.utils.data import DataLoader, DistributedSampler





@dataclass

class TrainConfig:

    dataset_dir: str

    model_dir: str

    output_dir: str

    model_mode: str = "chunkwise"



    seq_len: int = 8192

    per_device_batch: int = 4

    grad_accum: int = 1



    max_steps: int = 30_000

    lr: float = 1.1e-4

    min_lr: float = 1.1e-5

    warmup_steps: int = 300

    weight_decay: float = 0.1

    max_grad_norm: float = 1.0

    chunk_size: int = 64



    save_steps: int = 500

    eval_steps: int = 500

    save_total_limit: int = 3

    seed: int = 1234





CFG = TrainConfig(

    dataset_dir=DATASET_DIR,

    model_dir=MODEL_DIR,

    output_dir=OUTPUT_DIR,

    model_mode=MODEL_MODE,

    seq_len=SEQ_LEN,

    per_device_batch=PER_DEVICE_BATCH,

    grad_accum=GRAD_ACCUM,

    max_steps=MAX_STEPS,

    lr=LR,

    min_lr=MIN_LR,

    warmup_steps=WARMUP,

    weight_decay=WEIGHT_DECAY,

    max_grad_norm=MAX_GRAD_NORM,

    chunk_size=CHUNK_SIZE,

    save_steps=SAVE_STEPS,

    eval_steps=EVAL_STEPS,

    save_total_limit=SAVE_LIMIT,

)



# A JSON copy is useful for reproducibility. It contains no runtime objects.

with open("/kaggle/working/train_config.json", "w", encoding="utf-8") as f:

    json.dump(asdict(CFG), f, ensure_ascii=False, indent=2)



def _seed_everything(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)





def _latest_checkpoint(output_dir: str):

    paths = []

    for p in Path(output_dir).glob("checkpoint-*"):

        m = re.fullmatch(r"checkpoint-(\d+)", p.name)

        if m:

            paths.append((int(m.group(1)), p))

    return max(paths, default=(None, None), key=lambda x: x[0] if x[0] is not None else -1)





def _save_checkpoint(model, optimizer, scheduler, step: int, best_eval_loss: float,

                     output_dir: str, keep: int, xm, xr) -> None:

    """Save a single rank-0 checkpoint and synchronize all workers."""

    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"

    if xm.is_master_ordinal():

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {

            "model": model.state_dict(),

            "optimizer": optimizer.state_dict(),

            "scheduler": scheduler.state_dict(),

            "step": step,

            "best_eval_loss": best_eval_loss,

            "config": asdict(CFG),

        }

        # xm.save handles tensors that live on the XLA device.

        xm.save(state, str(ckpt_dir / "training.pt"), master_only=True)



        checkpoints = sorted(

            (p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir()),

            key=lambda p: int(p.name.split("-")[-1]),

        )

        while len(checkpoints) > keep:

            old = checkpoints.pop(0)

            for child in old.iterdir():

                child.unlink()

            old.rmdir()

    xm.rendezvous(f"checkpoint-{step}")





def _load_checkpoint(model, optimizer, scheduler, checkpoint_dir: Path):

    checkpoint_file = checkpoint_dir / "training.pt"

    if not checkpoint_file.is_file():

        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_file}")



    # The checkpoint was serialized by xm.save from rank 0. Loading happens

    # independently on every worker; optimizer tensors are cast to the

    # parameter device by PyTorch's optimizer load_state_dict.

    state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    model.load_state_dict(state["model"], strict=True)

    optimizer.load_state_dict(state["optimizer"])

    scheduler.load_state_dict(state["scheduler"])

    return int(state["step"]), float(state["best_eval_loss"])





class CosineWithMinLRScheduler(torch.optim.lr_scheduler.LambdaLR):

    """Warmup + cosine decay to a non-zero floor."""



    def __init__(self, optimizer, warmup_steps, total_steps, min_lr):

        self.warmup_steps = int(warmup_steps)

        self.total_steps = int(total_steps)

        self.min_lr_ratio = float(min_lr / optimizer.defaults["lr"])

        super().__init__(optimizer, lr_lambda=self._lr_lambda)



    def _lr_lambda(self, step):

        if step < self.warmup_steps:

            return float(step + 1) / max(1, self.warmup_steps)

        progress = (step - self.warmup_steps) / max(

            1, self.total_steps - self.warmup_steps

        )

        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine





@torch.no_grad()

def evaluate(model, loader, device, xm, xr):

    model.eval()

    total_loss = torch.zeros((), device=device, dtype=torch.float32)

    batches = 0



    for batch in loader:

        # autocast removed: manual mixed precision is done explicitly

        # (XLA_USE_BF16=1 + bf16 weights + fp32 islands for gate/norm/loss).

        outputs = model(

            input_ids=batch["input_ids"].to(device),

            labels=batch["labels"].to(device),

            wcodes=batch["wcodes"].to(device),

        )

        total_loss = total_loss + outputs.loss.float()

        batches += 1



    # DistributedSampler(drop_last=True) gives every worker the same number

    # of batches, so a sum/mean across replicas is deterministic.

    if batches == 0:

        mean_loss = torch.tensor(float("nan"), device=device)

    else:

        mean_loss = total_loss / batches



    mean_loss = xm.all_reduce(xm.REDUCE_SUM, mean_loss) / xr.world_size()

    xm.mark_step()

    model.train()

    return float(mean_loss.cpu())





def train_worker():

    # Import/access XLA only inside the spawned worker. This is a key PJRT/XLA

    # multiprocessing invariant.

    import torch_xla.core.xla_model as xm

    import torch_xla.runtime as xr

    import torch_xla.distributed.parallel_loader as pl



    rank = xr.global_ordinal()

    world_size = xr.world_size()

    # device = xm.xla_device()

    device = torch_xla.device()



    _seed_everything(CFG.seed + rank)



    if xm.is_master_ordinal():

        print(

            f"[XLA] world_size={world_size} device={device} "

            f"model_mode={CFG.model_mode} batch/device={CFG.per_device_batch}",

            flush=True,

        )



    # Dataset metadata is inexpensive; every worker opens its own mmap view.

    ds = Path(CFG.dataset_dir)

    train_ds = PackedChatDataset(

        ds / "train.ids.npy",

        ds / "train.wcode.npy",

        CFG.seq_len,

    )

    val_ds = PackedChatDataset(

        ds / "val.ids.npy",

        ds / "val.wcode.npy",

        CFG.seq_len,

    )



    # DistributedSampler performs data parallel sharding without requiring a

    # manually initialized torch.distributed process group.

    train_sampler = DistributedSampler(

        train_ds,

        num_replicas=world_size,

        rank=rank,

        shuffle=True,

        seed=CFG.seed,

        drop_last=True,

    )

    val_sampler = DistributedSampler(

        val_ds,

        num_replicas=world_size,

        rank=rank,

        shuffle=False,

        drop_last=True,

    )



    train_loader_cpu = DataLoader(

        train_ds,

        batch_size=CFG.per_device_batch,

        sampler=train_sampler,

        num_workers=0,

        pin_memory=False,

        drop_last=True,

    )

    val_loader_cpu = DataLoader(

        val_ds,

        batch_size=CFG.per_device_batch,

        sampler=val_sampler,

        num_workers=0,

        pin_memory=False,

        drop_last=True,

    )



    # MpDeviceLoader overlaps host->TPU input transfer with execution and

    # inserts mark_step() for batch boundaries.

    train_loader = pl.MpDeviceLoader(train_loader_cpu, device)

    val_loader = pl.MpDeviceLoader(val_loader_cpu, device)



    # Load weights on CPU first, then transfer exactly once to this worker's TPU.

    model = RetNetForCausalLM.from_pretrained(CFG.model_dir)

    model.config.mode = CFG.model_mode

    model.config.chunk_size = CFG.chunk_size

    model.to(device=device, dtype=torch.bfloat16)

    # Manual mixed precision: restore the decay gates to fp32 master weights

    # (model.to(bfloat16) casts them too). bf16 resolves only ~2**-8 near 1.0,

    # which collapses 12/16 heads' decay to exactly 1.0 and quantizes away

    # gate learning -- this restore is what actually enables the GLA gate.

    for _block in model.blocks:

        _gate = _block.msr.gamma

        if isinstance(_gate, nn.Parameter):

            _gate.data = _gate.data.to(torch.float32)

    # The unrolled chunk loop produces large per-block activation graphs, so

    # block-level gradient checkpointing is REQUIRED to fit TPU HBM (each

    # block's activations are recomputed in backward instead of stored).

    # Disable only if memory turns out to be ample.

    for _block in model.blocks:

        _block.gradient_checkpointing = True

    model.train()



    # No weight decay on 1-D tensors (norm scales, biases, decay gates):

    # decaying the fp32 gate gamma toward 0 would kill long-range memory.

    _wd = [p for p in model.parameters() if p.ndim >= 2]

    _nowd = [p for p in model.parameters() if p.ndim < 2]

    optimizer = AdamW(

        [

            {"params": _wd, "weight_decay": CFG.weight_decay},

            {"params": _nowd, "weight_decay": 0.0},

        ],

        lr=CFG.lr,

        betas=(0.9, 0.95),

        eps=1e-8,

    )

    scheduler = CosineWithMinLRScheduler(

        optimizer,

        warmup_steps=CFG.warmup_steps,

        total_steps=CFG.max_steps,

        min_lr=CFG.min_lr,

    )



    start_step = 0

    best_eval_loss = float("inf")



    latest_step, latest_dir = _latest_checkpoint(CFG.output_dir)

    if latest_dir is not None:

        start_step, best_eval_loss = _load_checkpoint(

            model, optimizer, scheduler, latest_dir

        )

        if xm.is_master_ordinal():

            print(f"[resume] checkpoint={latest_dir.name} step={start_step}", flush=True)

        xm.rendezvous(f"resume-{start_step}")



    data_iter = None

    current_epoch = 0



    def next_batch():

        nonlocal data_iter, current_epoch

        if data_iter is None:

            train_sampler.set_epoch(current_epoch)

            data_iter = iter(train_loader)



        try:

            return next(data_iter)

        except StopIteration:

            current_epoch += 1

            train_sampler.set_epoch(current_epoch)

            data_iter = iter(train_loader)

            return next(data_iter)



    optimizer.zero_grad(set_to_none=True)

    t0 = time.perf_counter()



    for step in range(start_step + 1, CFG.max_steps + 1):

        step_loss = torch.zeros((), device=device, dtype=torch.float32)



        for _micro_step in range(CFG.grad_accum):

            batch = next_batch()

            # autocast removed: manual mixed precision is done explicitly

            # (XLA_USE_BF16=1 + bf16 weights + fp32 islands for gate/norm/loss).

            outputs = model(

                input_ids=batch["input_ids"],

                labels=batch["labels"],

                wcodes=batch["wcodes"],

            )

            loss = outputs.loss / CFG.grad_accum

            loss.backward()

            step_loss = step_loss + outputs.loss.detach().float()



        if CFG.max_grad_norm is not None:

            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)



        # optimizer_step = gradient all-reduce + optimizer.step() + mark_step()

        xm.optimizer_step(optimizer, barrier=False)

        optimizer.zero_grad(set_to_none=True)

        scheduler.step()



        if (

            step == 1

            or step % 20 == 0

            or step == CFG.max_steps

        ):

            reduced_loss = xm.all_reduce(

                xm.REDUCE_SUM, step_loss / CFG.grad_accum

            ) / world_size

            xm.mark_step()

            if xm.is_master_ordinal():

                lr = scheduler.get_last_lr()[0]

                elapsed = time.perf_counter() - t0

                print(

                    f"[train] step={step:6d} "

                    f"loss={float(reduced_loss.cpu()):.5f} "

                    f"lr={lr:.3e} "

                    f"elapsed={elapsed/60:.1f}m",

                    flush=True,

                )



        should_eval = CFG.eval_steps and step % CFG.eval_steps == 0

        should_save = CFG.save_steps and step % CFG.save_steps == 0



        if should_eval:

            eval_loss = evaluate(model, val_loader, device, xm, xr)

            if xm.is_master_ordinal():

                print(

                    f"[eval] step={step:6d} loss={eval_loss:.5f}",

                    flush=True,

                )

            if eval_loss < best_eval_loss:

                best_eval_loss = eval_loss



        if should_save or should_eval:

            _save_checkpoint(

                model,

                optimizer,

                scheduler,

                step,

                best_eval_loss,

                CFG.output_dir,

                CFG.save_total_limit,

                xm,

                xr,

            )



    # Final rank-0 HF-format export. Convert tensors to CPU before asking

    # Transformers to write safetensors, avoiding XLA-device serialization.

    xm.mark_step()

    if xm.is_master_ordinal():

        final_state = {

            k: v.detach().cpu()

            for k, v in model.state_dict().items()

        }

        model.save_pretrained(

            CFG.output_dir,

            state_dict=final_state,

            safe_serialization=True,

        )

        print(

            f"[done] saved Hugging Face model to {CFG.output_dir} "

            f"(best_eval_loss={best_eval_loss:.5f})",

            flush=True,

        )



    xm.rendezvous("finalize")
