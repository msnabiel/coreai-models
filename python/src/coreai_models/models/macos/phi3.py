# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Core AI export model for Hugging Face Phi-3 causal language models."""

import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM as HFPhi3ForCausalLM
from typing_extensions import Self, override

from coreai_models._hf import is_default_rope_scaling, resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA


class Phi3MLP(nn.Module):
    """Phi-3's fused gate/up projection and SiLU gated activation."""

    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(up * torch.nn.functional.silu(gate))


class Phi3LongRoPE(nn.Module):
    """Static LongRoPE lowering for the exported context length.

    Phi-3.5 selects ``short_factor`` through the pretrained context window and
    ``long_factor`` beyond it. Core AI export shapes are fixed by the selected
    max context, so selecting the matching factor list at construction time is
    both exportable and numerically equivalent for that asset.
    """

    def __init__(self, config: Phi3Config, dims: int) -> None:
        super().__init__()
        scaling = config.rope_scaling
        assert isinstance(scaling, dict)
        original = getattr(config, "original_max_position_embeddings", config.max_position_embeddings)
        if config.max_position_embeddings > original:
            raise ValueError(
                "Phi-3 LongRoPE exports are currently limited to the original "
                f"{original}-token context; choose --max-context-length <= {original}."
            )
        factor_key = "short_factor"
        factors = torch.tensor(scaling[factor_key], dtype=torch.float32)
        base = float(resolve_rope_theta(config))
        inv_freq = 1.0 / (factors * base ** (torch.arange(0, dims, 2, dtype=torch.float32) / dims))
        factor = config.max_position_embeddings / original
        attention_scale = 1.0 if factor <= 1 else (1 + torch.log(torch.tensor(factor)) / torch.log(torch.tensor(float(original)))) ** 0.5
        self.register_buffer("freqs", inv_freq, persistent=False)
        self.rope = RoPE(scale=float(attention_scale), dims=dims)

    def forward(self, x: torch.Tensor, position_ids: torch.IntTensor) -> torch.Tensor:
        return self.rope(x, position_ids=position_ids, freqs=self.freqs.to(x.device))


class Attention(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", None) or dim // n_heads
        self.qkv_proj = nn.Linear(
            dim, n_heads * head_dim + 2 * n_kv_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.sdpa = SDPA(is_causal=True)
        rotary_dims = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        if is_default_rope_scaling(config):
            self.rope = initialize_rope(
                dims=rotary_dims if rotary_dims != self.head_dim else None,
                base=resolve_rope_theta(config),
            )
        else:
            self.rope = Phi3LongRoPE(config, rotary_dims)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads
        qkv = self.qkv_proj(x).reshape(
            batch_size, query_len, n_heads + 2 * n_kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)
        query_key = self.rope(qkv.narrow(1, 0, n_heads + n_kv_heads), position_ids=rope_positions)
        query = query_key.narrow(1, 0, n_heads)
        key = query_key.narrow(1, n_heads, n_kv_heads)
        value = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)
        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
            )
        return self.o_proj(
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, n_heads * self.head_dim)
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, position_ids, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), position_ids, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Phi3Model(nn.Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, position_ids, cache=None):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class Phi3ForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFPhi3ForCausalLM

    @override
    def _init_model(self, config: Phi3Config) -> None:
        self.model = Phi3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(self, input_ids, position_ids, k_cache, v_cache):
        return self.lm_head(self.model(input_ids, position_ids, KVCache(k_cache, v_cache)))

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # HF Phi-3 already stores qkv_proj and gate_up_proj in the fused layout
        # used by this export model. Keep this hook explicit for BaseForCausalLM.
        del state_dict
