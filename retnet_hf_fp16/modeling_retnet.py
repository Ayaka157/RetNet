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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


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
            elif isinstance(p, nn.Parameter) and p.dim() == 1:
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
        # fp16 safety: LayerNorm stats in fp32; output cast back to input dtype
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x = x.transpose(1, 2).reshape(bsz * tgt_len, num_heads * x.shape[-1])
        x = group_norm(x.unsqueeze(-1)).squeeze(-1)
        x = x.reshape(bsz, tgt_len, -1)
        return x.to(orig_dtype)

    def _decay_mask(self, length, device, dtype):
        idx = torch.arange(length, device=device)
        delta = idx[:, None] - idx[None, :]
        causal = delta >= 0
        exp = delta.masked_fill(~causal, 0)
        # fp16 safety: gamma^delta for large delta (e.g. 2^-511) underflows to 0
        # in fp16 but is exact in fp32. Compute in fp32, only cast the final mask.
        mask = self.gamma.to(device=device, dtype=torch.float32)[:, None, None] ** exp
        mask = mask.masked_fill(~causal, 0.0)
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        return mask.to(dtype=dtype)

    def _parallel_retention(self, q, k, v):
        bsz, _, tgt_len, _ = q.shape
        mask = self._decay_mask(tgt_len, q.device, q.dtype)
        # fp16 safety: matmul in fp16 (Tensor Core) but normalize in fp32;
        # output in fp32 so GroupNorm gets real values (fp16 overflow risk).
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores * mask
        norm = scores.detach().float().abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        scores = (scores.float() / norm).to(scores.dtype)
        return torch.matmul(scores, v).float()

    def _parallel_final_state(self, k, v, state):
        bsz, _, tgt_len, key_dim = k.shape
        idx = torch.arange(tgt_len, device=k.device)
        # fp16 safety: state accumulation in fp32
        weights = self.gamma.to(device=k.device, dtype=torch.float32)[:, None] ** (tgt_len - 1 - idx)
        new_state = torch.einsum("bhtk,bhtv,ht->bhkv", k.float(), v.float(), weights)
        if state is not None:
            gamma_pow = self.gamma.to(device=k.device, dtype=torch.float32) ** tgt_len
            new_state = gamma_pow.float().view(1, -1, 1, 1) * state.float() + new_state
        return new_state

    def _recurrent_retention(self, q, k, v, state):
        bsz, _, tgt_len, key_dim = q.shape
        value_head_dim = v.shape[-1]
        if state is None:
            state = torch.zeros(bsz, self.num_heads, key_dim, value_head_dim,
                                device=q.device, dtype=torch.float32)
        else:
            state = state.to(torch.float32)
        outputs = []
        gamma = self.gamma.to(device=q.device, dtype=q.dtype)
        for t in range(tgt_len):
            state = gamma.float().view(1, -1, 1, 1) * state + \
                (k[:, :, t, :].unsqueeze(-1) * v[:, :, t, :].unsqueeze(-2)).float()
            out_t = torch.sum(q[:, :, t, :].float().unsqueeze(-1) * state, dim=-2)
            outputs.append(out_t)
        output = torch.stack(outputs, dim=2)
        return output, state

    def _chunkwise_retention(self, q, k, v, state, chunk_size):
        """Equation (7): inner-chunk parallel + cross-chunk recurrent state.

        fp16 safety (GLA model: gamma can be ~0.79-0.99, state |values| reach
        ~3e4, near fp16's 65504 ceiling):
          * ALL decay powers computed in fp32 (gamma^delta underflows to 0 in
            fp16 for large delta, and cross_decay 0.786^64 = 3e-7 would be
            silently lost).
          * cross path (chunk state) entirely in fp32.
          * inner-chunk matmuls (q·k, score·v) stay fp16 for Tensor Core.
          * output returned fp32 so GroupNorm sees true values before any
            fp16 cast downstream.
        """
        bsz, _, tgt_len, key_dim = q.shape
        if tgt_len % chunk_size != 0:
            raise ValueError(f"chunkwise mode requires seq_len ({tgt_len}) to be a "
                             f"multiple of chunk_size ({chunk_size})")
        if state is None:
            state = torch.zeros(bsz, self.num_heads, key_dim, v.shape[-1],
                                device=q.device, dtype=torch.float32)
        else:
            state = state.to(torch.float32)
        gamma = self.gamma.to(device=q.device, dtype=torch.float32)
        dtype = q.dtype
        outputs = []
        rel = torch.arange(chunk_size, device=q.device)
        delta = rel[:, None] - rel[None, :]
        causal = delta >= 0
        # decay matrices STAY fp32 (never cast to fp16)
        inner_mask = torch.where(causal, gamma[:, None, None] ** delta.masked_fill(~causal, 0),
                                 torch.zeros_like(delta, dtype=torch.float32))
        cross_decay = gamma.float()[:, None] ** (rel + 1)[None, :]
        value_decay = gamma.float()[:, None] ** (chunk_size - 1 - rel)[None, :]
        chunk_decay = gamma.float() ** chunk_size

        for i in range(tgt_len // chunk_size):
            sl = slice(i * chunk_size, (i + 1) * chunk_size)
            qi, ki, vi = q[:, :, sl], k[:, :, sl], v[:, :, sl]
            # inner path also fp32 at 8K: fp16 matmul accumulation overflows
            # the long-context values (verified: 4K OK, 8K NaN in fp16).
            inner_scores = torch.matmul(qi.float(), ki.float().transpose(-1, -2)) * inner_mask
            inner = torch.matmul(inner_scores, vi.float())
            # cross path in fp32 with EXPLICIT broadcast sum: torch.matmul/einsum
            # on this GLA model (state ~3e4, q*decay ~5) can return inf spuriously
            # (verified: manual sum = 3.3e5, matmul = inf). Avoid the kernel bug.
            qc = qi.float() * cross_decay[None, :, :, None]
            cross = (qc.unsqueeze(-1) * state.unsqueeze(2)).sum(dim=-2)
            outputs.append(inner + cross)
            weighted_v = vi.float() * value_decay[None, :, :, None]
            state = chunk_decay.view(1, -1, 1, 1) * state + \
                torch.matmul(ki.float().transpose(-1, -2), weighted_v).float()

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

        # fp16 safety: retention output can reach ~3e4 (GLA state) which
        # overflows fp16. Do GroupNorm in fp32, then cast down for the gate.
        output = self._head_group_norm(output, self.num_heads, self.group_norm, bsz, tgt_len)
        output = output.to(x.dtype)
        # fp16 safety: silu + gate multiply stays fp16 (gate values bounded),
        # but GroupNorm statistics already computed safely above.
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
        # fp16 safety: LayerNorm statistics in fp32 (fp16 variance overflow for
        # large hidden states can NaN); output cast back to input dtype.
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        old = self.ln_old(x[..., : self.ln_old.normalized_shape[0]])
        new = x[..., self.ln_old.normalized_shape[0]:] * self.gamma_new.to(torch.float32) + self.beta_new.to(torch.float32)
        out = torch.cat([old, new], dim=-1)
        return out.to(orig_dtype)


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
            # use_reentrant=True: preserves the fp16 autocast re-computation
            # context correctly; reentrant=False breaks fp16 at 8K (NaN).
            return torch.utils.checkpoint.checkpoint(
                self._block_forward, x, mode, state, offset, return_state, chunk_size,
                use_reentrant=True)
        return self._block_forward(x, mode, state, offset, return_state, chunk_size)

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable per-block gradient checkpointing (Trainer calls this)."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False


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
    _supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False, **kwargs):
        """HF Trainer calls this on the model; propagate to each block."""
        if isinstance(module, RetNetBlock):
            module.gradient_checkpointing = value

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
