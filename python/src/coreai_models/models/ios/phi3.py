# Copyright 2026 Apple Inc.
"""Neural Engine/iOS Phi-3 export model."""

import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM as HFPhi3ForCausalLM

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.quantization import dequantize_per_tensor, quantize_per_tensor
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import RoPECache, apply_rope
from coreai_models.primitives.ios.sdpa import SDPA


class Phi3Attention(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.n_heads)
        self.q_proj = nn.Conv2d(config.hidden_size, self.n_heads * self.head_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(config.hidden_size, self.n_kv_heads * self.head_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(config.hidden_size, self.n_kv_heads * self.head_dim, 1, bias=False)
        self.o_proj = nn.Conv2d(self.n_heads * self.head_dim, config.hidden_size, 1, bias=False)
        self.sdpa = SDPA(head_dim=self.head_dim)

    def forward(self, x, rope_cos, rope_sin, in_step, causal_mask, cache=None):
        b, s, _, _ = x.shape
        x = x.transpose(-3, -1)
        q = self.q_proj(x).transpose(-3, -1).reshape(b, s, self.n_heads, self.head_dim).transpose(-2, -3)
        k = self.k_proj(x).transpose(-3, -1).reshape(b, s, self.n_kv_heads, self.head_dim).transpose(-2, -3)
        v = self.v_proj(x).transpose(-3, -1).reshape(b, s, self.n_kv_heads, self.head_dim).transpose(-2, -3)
        q, k = apply_rope(q, rope_cos, rope_sin), apply_rope(k, rope_cos, rope_sin)
        q = q.transpose(-2, -3).reshape(b, s, 1, -1).transpose(-3, -1)
        k = k.transpose(-2, -3).reshape(b, s, 1, -1).transpose(-3, -1)
        v = v.transpose(-2, -3).reshape(b, s, 1, -1).transpose(-3, -1)
        if cache is not None:
            k, v = cache.update_and_fetch(self.layer_idx, in_step, k, v, s)
        return self.o_proj(self.sdpa(q, k, v, causal_mask)).transpose(-3, -1)


class Phi3MLP(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.gate_up_proj = nn.Conv2d(config.hidden_size, 2 * config.intermediate_size, 1, bias=False)
        self.down_proj = nn.Conv2d(config.intermediate_size, config.hidden_size, 1, bias=False)

    def forward(self, x):
        b, s, _, d = x.shape
        y = self.gate_up_proj(x.reshape(b * s, d, 1, 1))
        gate, up = y.chunk(2, dim=1)
        return self.down_proj(up * torch.nn.functional.silu(gate)).reshape(b, s, 1, d)


class Phi3Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = Phi3Attention(config, layer_idx)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, rope_cos, rope_sin, in_step, mask, cache):
        x = x + self.attn(self.input_layernorm(x), rope_cos, rope_sin, in_step, mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Phi3Extend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Phi3Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)
        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.lm_head = None if config.tie_word_embeddings else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.kv_cache = KVCacheHandler(config.num_hidden_layers, config.hidden_size)
        factors = None
        if isinstance(config.rope_scaling, dict):
            original = getattr(config, "original_max_position_embeddings", config.max_position_embeddings)
            if config.max_position_embeddings > original:
                raise ValueError(f"Phi-3 iOS export is limited to {original}-token context")
            factors = torch.tensor(config.rope_scaling["short_factor"], dtype=torch.float32)
        self.rope = RoPECache(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            config.max_position_embeddings,
            resolve_rope_theta(config),
            freq_factors=factors,
        )
        self.prefill_mode = False

    def forward(self, x, position_ids, in_step, mask, key_cache, value_cache, embedding_table=None):
        self.kv_cache.register_kv_cache(key_cache, value_cache)
        cos, sin = self.rope.gather_cos_sin(position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, in_step, mask, self.kv_cache)
        x = self.norm(x)
        if self.prefill_mode:
            return self.kv_cache.k_cache[0, 0, 0, 0, 0] + self.kv_cache.v_cache[0, 0, 0, 0, 0]
        if self.lm_head is not None:
            return self.lm_head(x.transpose(-2, -3))
        if embedding_table.dtype == torch.int8:
            embedding_table = dequantize_per_tensor(embedding_table, self.emb_scale, self.emb_zero_point, x.dtype)
        table = embedding_table.reshape(embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2])
        b, s, _, d = x.shape
        return (table @ x.transpose(-3, -1).reshape(b, 1, d, s)).permute(0, 3, 1, 2)


class Phi3ForCausalLMForiOS(BaseForCausalLMForiOS):
    _HF_MODEL_CLASS = HFPhi3ForCausalLM

    def _init_model(self, config):
        self.extend = Phi3Extend(config)

    def forward(self, input_ids, position_ids, in_step, causal_mask, key_cache, value_cache):
        table = self.load_embeddings.embedding_table
        return self.extend(self.gather_embeddings(input_ids, table), position_ids, in_step, causal_mask, key_cache, value_cache, table)

    def _mutate_state_dict(self, state_dict):
        for i in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{i}."
            qkv = state_dict.pop(prefix + "self_attn.qkv_proj.weight")
            q, k, v = qkv.split([
                self.config.num_attention_heads * self.extend.layers[i].attn.head_dim,
                self.config.num_key_value_heads * self.extend.layers[i].attn.head_dim,
                self.config.num_key_value_heads * self.extend.layers[i].attn.head_dim,
            ])
            for name, value in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                state_dict[prefix + "attn." + name + ".weight"] = value.unsqueeze(-1).unsqueeze(-1)
            state_dict[prefix + "attn.o_proj.weight"] = state_dict.pop(prefix + "self_attn.o_proj.weight").unsqueeze(-1).unsqueeze(-1)
            gate_up = state_dict.pop(prefix + "mlp.gate_up_proj.weight")
            state_dict[prefix + "mlp.gate_up_proj.weight"] = gate_up.unsqueeze(-1).unsqueeze(-1)
            state_dict[prefix + "mlp.down_proj.weight"] = state_dict.pop(prefix + "mlp.down_proj.weight").unsqueeze(-1).unsqueeze(-1)
        table = state_dict.pop("model.embed_tokens.weight").unsqueeze(1)
        if not self.disable_embedding_quantization:
            table, scale, zero = quantize_per_tensor(table, nbits=8, symmetric=True)
        else:
            scale, zero = torch.tensor(1.0, dtype=table.dtype), torch.tensor(0, dtype=torch.int8)
        state_dict["load_embeddings.embedding_table"] = table
        state_dict["gather_embeddings.scale"] = scale
        state_dict["gather_embeddings.zero_point"] = zero
        state_dict["extend.emb_scale"] = scale
        state_dict["extend.emb_zero_point"] = zero
        moved = {}
        for key in list(state_dict):
            if key.startswith("model.layers.") or key == "model.norm.weight":
                moved["extend." + key.removeprefix("model.")] = state_dict.pop(key)
        state_dict.update(moved)
        if not self.config.tie_word_embeddings:
            state_dict["extend.lm_head.weight"] = state_dict.pop("lm_head.weight")
