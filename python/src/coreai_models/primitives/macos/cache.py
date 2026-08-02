# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
from typing_extensions import Self

from coreai_models.primitives._ops import mutable_cache_update_and_fetch, mutable_slice_update


class KVCache:
    # consts for the HF source model.
    # names start with _ to make sure that the users should ALWAYS
    # interact the caches from the KVCache class APIs.
    HF_K_BUFFER_NAME = "_full_cached_k"
    HF_V_BUFFER_NAME = "_full_cached_v"

    def __init__(
        self: Self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ):
        """
        Args:
            k_cache, v_cache: KV cache tensors, float16 or (when k_scale/v_scale
                are also given) int8.
            k_scale, v_scale: Optional per-(layer, kv_head, token) float16 scale
                tensors of shape (n_layers, 1, n_kv_heads, max_seq_len, 1). When
                given, k_cache/v_cache are treated as int8 and dequantized on
                read; when omitted, k_cache/v_cache are used directly as float16.
        """
        self._k_cache = k_cache
        self._v_cache = v_cache
        self._k_scale = k_scale
        self._v_scale = v_scale

    @classmethod
    def seq_len_dim(cls) -> int:
        """
        Get the dimension index for sequence length in the KVCache.
        """
        return 3

    @classmethod
    def create_cache_tensors(
        cls,
        config,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized KV cache tensors from a model config.

        Returns:
            (k_cache, v_cache) tensors of shape (n_layers, 1, n_kv_heads, max_seq_len, head_dim).
        """
        n_kv_heads = config.num_key_value_heads
        n_layers = config.num_hidden_layers
        max_seq_len = config.max_position_embeddings
        if hasattr(config, "head_dim") and config.head_dim is not None:
            head_dim = config.head_dim
        else:
            head_dim = config.hidden_size // config.num_attention_heads
        k_cache = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim, dtype=dtype)
        v_cache = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim, dtype=dtype)
        return k_cache, v_cache

    @classmethod
    def create_quantized_cache_tensors(
        cls,
        config,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create zero-initialized int8 KV cache tensors + float16 per-token-per-head scales.

        Returns:
            (k_cache, v_cache, k_scale, v_scale) where k_cache/v_cache have shape
            (n_layers, 1, n_kv_heads, max_seq_len, head_dim) dtype int8, and
            k_scale/v_scale have shape (n_layers, 1, n_kv_heads, max_seq_len, 1)
            dtype float16.
        """
        n_kv_heads = config.num_key_value_heads
        n_layers = config.num_hidden_layers
        max_seq_len = config.max_position_embeddings
        if hasattr(config, "head_dim") and config.head_dim is not None:
            head_dim = config.head_dim
        else:
            head_dim = config.hidden_size // config.num_attention_heads
        k_cache = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim, dtype=torch.int8)
        v_cache = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim, dtype=torch.int8)
        k_scale = torch.ones(n_layers, 1, n_kv_heads, max_seq_len, 1, dtype=torch.float16)
        v_scale = torch.ones(n_layers, 1, n_kv_heads, max_seq_len, 1, dtype=torch.float16)
        return k_cache, v_cache, k_scale, v_scale

    @classmethod
    def from_dimensions(
        cls,
        n_layers: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
    ) -> Self:
        """
        Create a KVCache object with specified dimensions.

        This method creates a standalone KV cache with custom dimensions.
        """
        k_cache = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim)
        v_cache = torch.zeros(n_layers, 1, n_kv_heads, max_seq_len, head_dim)
        return cls(k_cache, v_cache)

    def update_and_fetch(
        self: Self,
        layer_idx: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int | None = None,
        query_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # check query size
        if query_len is None:
            query_len: int = k.shape[-2]
        torch._check_is_size(query_len, message="int query length >= 0")
        torch._check(query_len <= self._k_cache.size(-2), message="query length <= context size")
        torch._check(query_len <= self._v_cache.size(-2), message="query length <= context size")

        # check offset
        torch._check_is_size(offset, message="int offset >= 0")
        torch._check(offset < self._k_cache.size(-2), message="offset < context size")
        torch._check(offset < self._v_cache.size(-2), message="offset < context size")

        # check layer index
        torch._check_is_size(layer_idx, message="int layer index >= 0")
        torch._check(
            layer_idx < self._k_cache.size(0),
            message="layer index < number of transformer layers",
        )
        torch._check(
            layer_idx < self._v_cache.size(0),
            message="layer index < number of transformer layers",
        )

        if seq_len is None:
            seq_len = offset + query_len

        torch._check_is_size(seq_len)
        device = self._k_cache.device

        compute_device = k.device
        cross_device = compute_device != device
        if cross_device:
            k = k.to(device)
            v = v.to(device)

        layer_index = torch.tensor((layer_idx,), dtype=torch.int32, device=device)
        layer_index_end = torch.tensor((layer_idx + 1,), dtype=torch.int32, device=device)

        if self._k_scale is not None:
            return self._update_and_fetch_quantized(
                layer_idx,
                layer_index,
                layer_index_end,
                offset,
                seq_len,
                k,
                v,
                cross_device,
                compute_device,
            )

        # Float16 path - use upstream's fused op (macOS layout: seq_dim=-2)
        k_out = mutable_cache_update_and_fetch(
            x=self._k_cache,
            update=k,
            begin=torch.concatenate(
                [
                    layer_index,
                    torch.tensor((0,), dtype=torch.int32, device=device),
                    torch.tensor((0,), dtype=torch.int32, device=device),
                    torch.tensor((offset,), dtype=torch.int32, device=device),
                    torch.tensor((0,), dtype=torch.int32, device=device),
                ]
            ),
            end=torch.cat(
                [
                    layer_index_end,
                    torch.tensor((self._k_cache.size(1),), dtype=torch.int32, device=device),
                    torch.tensor((self._k_cache.size(2),), dtype=torch.int32, device=device),
                    torch.tensor((offset + k.size(2),), dtype=torch.int32, device=device),
                    torch.tensor((self._k_cache.size(4),), dtype=torch.int32, device=device),
                ]
            ),
            layer_idx=layer_idx,
            seq_dim=-2,
            seq_len=seq_len,
        )
        v_out = mutable_cache_update_and_fetch(
            x=self._v_cache,
            update=v,
            begin=torch.cat(
                [
                    layer_index,
                    torch.tensor((0,), dtype=torch.int32, device=device),
                    torch.tensor((0,), dtype=torch.int32, device=device),
                    torch.tensor((offset,), dtype=torch.int32, device=device),
                    torch.tensor((0,), dtype=torch.int32, device=device),
                ]
            ),
            end=torch.cat(
                [
                    layer_index_end,
                    torch.tensor((int(self._v_cache.size(1)),), dtype=torch.int32, device=device),
                    torch.tensor((int(self._v_cache.size(2)),), dtype=torch.int32, device=device),
                    torch.tensor((offset + v.size(2),), dtype=torch.int32, device=device),
                    torch.tensor((int(self._v_cache.size(4)),), dtype=torch.int32, device=device),
                ]
            ),
            layer_idx=layer_idx,
            seq_dim=-2,
            seq_len=seq_len,
        )
        if cross_device:
            return k_out.to(compute_device), v_out.to(compute_device)
        return k_out, v_out

    def _update_and_fetch_quantized(
        self: Self,
        layer_idx: int,
        layer_index: torch.Tensor,
        layer_index_end: torch.Tensor,
        offset: int,
        seq_len: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cross_device: bool,
        compute_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Int8 KV cache path: quantize per (token, kv_head) before writing, dequantize on read.

        k, v shape: (batch=1, n_kv_heads, query_len, head_dim). One scale is
        computed per (kv_head, token) by reducing over batch and head_dim, so
        each head keeps its own dynamic range instead of sharing one scale
        across all heads.
        """
        device = self._k_cache.device

        k_scale_val = k.float().abs().amax(dim=(0, 3), keepdim=True).clamp(min=1e-6) / 127.0
        v_scale_val = v.float().abs().amax(dim=(0, 3), keepdim=True).clamp(min=1e-6) / 127.0
        q_k = (k.float() / k_scale_val).clamp(-127, 127).to(torch.int8)
        q_v = (v.float() / v_scale_val).clamp(-127, 127).to(torch.int8)

        zero = torch.tensor((0,), dtype=torch.int32, device=device)
        offset_t = torch.tensor((offset,), dtype=torch.int32, device=device)
        cache_begin = torch.cat([layer_index, zero, zero, offset_t, zero])
        cache_end = torch.cat(
            [
                layer_index_end,
                torch.tensor((self._k_cache.size(1),), dtype=torch.int32, device=device),
                torch.tensor((self._k_cache.size(2),), dtype=torch.int32, device=device),
                torch.tensor((offset + k.size(2),), dtype=torch.int32, device=device),
                torch.tensor((self._k_cache.size(4),), dtype=torch.int32, device=device),
            ]
        )
        mutable_slice_update(x=self._k_cache, update=q_k.unsqueeze(0), begin=cache_begin, end=cache_end)
        mutable_slice_update(x=self._v_cache, update=q_v.unsqueeze(0), begin=cache_begin, end=cache_end)

        scale_begin = torch.cat([layer_index, zero, zero, offset_t, zero])
        scale_end = torch.cat(
            [
                layer_index_end,
                torch.tensor((self._k_scale.size(1),), dtype=torch.int32, device=device),
                torch.tensor((self._k_scale.size(2),), dtype=torch.int32, device=device),
                torch.tensor((offset + k.size(2),), dtype=torch.int32, device=device),
                torch.tensor((self._k_scale.size(4),), dtype=torch.int32, device=device),
            ]
        )
        mutable_slice_update(
            x=self._k_scale, update=k_scale_val.to(torch.float16).unsqueeze(0), begin=scale_begin, end=scale_end
        )
        mutable_slice_update(
            x=self._v_scale, update=v_scale_val.to(torch.float16).unsqueeze(0), begin=scale_begin, end=scale_end
        )

        k_q = self._k_cache.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len)
        v_q = self._v_cache.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len)
        k_s = self._k_scale.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len)
        v_s = self._v_scale.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len)

        k_out = (k_q.to(torch.float16) * k_s).squeeze(0)
        v_out = (v_q.to(torch.float16) * v_s).squeeze(0)
        if cross_device:
            return k_out.to(compute_device), v_out.to(compute_device)
        return k_out, v_out


class SSMState:
    """
    State Space Model (SSM) state cache for managing hidden states across layers.

    This class provides a mechanism to store and update SSM states (e.g., Mamba states)
    across multiple layers in a neural network. It uses a mutable slice update operation
    to efficiently update states for specific layers while maintaining the full state tensor.

    Attributes:
        _states (torch.Tensor): Internal tensor storing SSM states for all layers.
            Shape: (num_layers, batch_size, *state_dims) where:
                - num_layers: Number of transformer layers
                - batch_size: Batch size (typically 1 for inference)
                - *state_dims: Model-specific state dimensions (e.g., state_size, d_inner, etc.)
    """

    def __init__(
        self: Self,
        states: torch.Tensor,
    ) -> None:
        """
        Initialize the SSMState with a pre-allocated state tensor.

        Args:
            states (torch.Tensor): Pre-allocated tensor to store SSM states across layers.
                Shape: (num_layers, batch_size, *state_dims)
                - First dimension must correspond to the number of layers
                - Second dimension is typically batch_size (usually 1 for inference)
                - Remaining dimensions are model-specific state dimensions
        """
        self._states = states

    @property
    def states(self) -> torch.Tensor:
        """
        Get the full SSM state tensor.

        Returns:
            torch.Tensor: The complete state tensor containing states for all layers.
                Shape: (num_layers, batch_size, *state_dims)
        """
        return self._states

    def update_states(
        self: Self,
        layer_idx: int,
        new_state: torch.Tensor,
    ) -> None:
        """
        Update the SSM state for a specific layer.

        This method updates the state cache for a given layer using a mutable slice
        update operation. The update is performed in-place (conceptually) on the
        internal state tensor.

        Args:
            layer_idx (int): Index of the layer to update. Must be >= 0 and < num_layers.
            new_state (torch.Tensor): New state tensor for the specified layer.
                Shape: (batch_size, *state_dims)
                Should match the state dimensions excluding the layer dimension.

        Raises:
            RuntimeError: If layer_idx is out of bounds (>= number of layers).

        Note:
            The update operation uses torch.export-compatible size checking to ensure
            the layer index is valid. The new_state is automatically unsqueezed to add
            the layer dimension before updating.
        """
        cache = self._states

        # size checking for torch.export
        torch._check_is_size(layer_idx)
        torch._check(
            layer_idx < self._states.size(0),
        )

        # use the slice_update to update the cache
        layer_index = torch.tensor((layer_idx,), dtype=torch.int32)
        layer_index_end = torch.tensor((layer_idx + 1,), dtype=torch.int32)

        mutable_slice_update(
            x=cache,
            update=new_state.unsqueeze(0),
            begin=torch.concatenate(
                [
                    layer_index,
                    *[torch.tensor((0,), dtype=torch.int32) for _ in range(cache.dim() - 1)],
                ]
            ),
            end=torch.cat(
                [
                    layer_index_end,
                    *[
                        torch.tensor((cache.size(i),), dtype=torch.int32)
                        for i in range(1, 1 + cache.dim() - 2)
                    ],
                ]
            ),
        )
