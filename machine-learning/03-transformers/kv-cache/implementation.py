"""
KV Cache -- From-scratch NumPy implementation.

Stores key and value tensors from previous positions so they are not recomputed
during autoregressive generation. Without caching, generating n tokens requires
O(n^2 * d) projection FLOPs because the full sequence is re-projected at every
step. With KV cache the projection cost drops to O(n * d) -- each step projects
only the new token, appends K and V to the cache, and attends over the full
history. This module implements the KVCache data structure, cache-aware
transformer forward passes, and side-by-side generation with and without caching
to verify output equivalence.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from importlib import import_module

import numpy as np

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

_norm_mod = import_module("02-neural-networks.normalization.implementation")
RMSNorm = _norm_mod.RMSNorm

_gqa_mod = import_module("03-transformers.grouped-query-attention.implementation")
softmax = _gqa_mod.softmax
create_causal_mask = _gqa_mod.create_causal_mask
repeat_kv = _gqa_mod.repeat_kv

_rope_mod = import_module("03-transformers.rope.implementation")
apply_rope = _rope_mod.apply_rope

_block_mod = import_module("03-transformers.transformer-block.implementation")
TransformerBlock = _block_mod.TransformerBlock

_causal_mod = import_module("03-transformers.causal-decoding.implementation")
CausalLM = _causal_mod.CausalLM
sample_token = _causal_mod.sample_token


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------

class KVCache:
    """Per-layer key/value cache for autoregressive generation.

    Stores K and V tensors for each transformer layer. Supports both
    pre-allocated (fixed max_seq_len) and dynamic growth modes.

    Shapes stored per layer:
        K: (batch, n_kv_heads, seq_len, d_k)
        V: (batch, n_kv_heads, seq_len, d_k)
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        d_k: int,
        max_seq_len: int = 0,
        dtype: np.dtype = np.float64,
    ):
        """
        Args:
            n_layers: Number of transformer layers
            batch_size: Batch dimension B
            n_kv_heads: Number of KV heads
            d_k: Head dimension
            max_seq_len: If > 0, pre-allocate buffers; otherwise grow dynamically
            dtype: Element dtype for memory calculations
        """
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        if max_seq_len > 0:
            self._k = [
                np.zeros((batch_size, n_kv_heads, max_seq_len, d_k), dtype=dtype)
                for _ in range(n_layers)
            ]
            self._v = [
                np.zeros((batch_size, n_kv_heads, max_seq_len, d_k), dtype=dtype)
                for _ in range(n_layers)
            ]
            self._seq_lens = [0] * n_layers
            self._preallocated = True
        else:
            self._k: List[Optional[np.ndarray]] = [None] * n_layers
            self._v: List[Optional[np.ndarray]] = [None] * n_layers
            self._seq_lens = [0] * n_layers
            self._preallocated = False

    @property
    def seq_len(self) -> int:
        """Current cached sequence length (assumes all layers are in sync)."""
        if self.n_layers == 0:
            return 0
        return self._seq_lens[0]

    def append(
        self, layer_idx: int, k: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Append new K, V tokens to the cache for a given layer.

        Args:
            layer_idx: Which transformer layer
            k: New keys, shape (batch, n_kv_heads, new_tokens, d_k)
            v: New values, shape (batch, n_kv_heads, new_tokens, d_k)

        Returns:
            (K_cache, V_cache) containing all cached tokens up to and including
            the newly appended ones.
        """
        if k.shape[0] != self.batch_size:
            raise ValueError(
                f"Batch mismatch: cache has {self.batch_size}, got {k.shape[0]}"
            )
        if k.shape[1] != self.n_kv_heads or k.shape[3] != self.d_k:
            raise ValueError(
                f"Shape mismatch: expected (B, {self.n_kv_heads}, *, {self.d_k}), "
                f"got {k.shape}"
            )

        new_tokens = k.shape[2]
        cur = self._seq_lens[layer_idx]

        if self._preallocated:
            if cur + new_tokens > self.max_seq_len:
                raise ValueError(
                    f"Cache overflow: {cur} + {new_tokens} > {self.max_seq_len}"
                )
            self._k[layer_idx][:, :, cur:cur + new_tokens, :] = k
            self._v[layer_idx][:, :, cur:cur + new_tokens, :] = v
            self._seq_lens[layer_idx] = cur + new_tokens
            return (
                self._k[layer_idx][:, :, :cur + new_tokens, :],
                self._v[layer_idx][:, :, :cur + new_tokens, :],
            )
        else:
            if self._k[layer_idx] is None:
                self._k[layer_idx] = k.copy()
                self._v[layer_idx] = v.copy()
            else:
                self._k[layer_idx] = np.concatenate(
                    [self._k[layer_idx], k], axis=2
                )
                self._v[layer_idx] = np.concatenate(
                    [self._v[layer_idx], v], axis=2
                )
            self._seq_lens[layer_idx] = self._k[layer_idx].shape[2]
            return self._k[layer_idx], self._v[layer_idx]

    def get_kv(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return cached (K, V) for a layer.

        Returns:
            (K, V) each of shape (batch, n_kv_heads, seq_len, d_k).
            If empty, seq_len dimension is 0.
        """
        cur = self._seq_lens[layer_idx]
        if cur == 0:
            empty_shape = (self.batch_size, self.n_kv_heads, 0, self.d_k)
            return (
                np.zeros(empty_shape, dtype=self.dtype),
                np.zeros(empty_shape, dtype=self.dtype),
            )
        if self._preallocated:
            return (
                self._k[layer_idx][:, :, :cur, :],
                self._v[layer_idx][:, :, :cur, :],
            )
        return self._k[layer_idx], self._v[layer_idx]

    def memory_bytes(self) -> int:
        """Total bytes currently used by cached K and V tensors."""
        element_size = np.dtype(self.dtype).itemsize
        total_elements = 0
        for i in range(self.n_layers):
            sl = self._seq_lens[i]
            total_elements += 2 * self.batch_size * self.n_kv_heads * sl * self.d_k
        return total_elements * element_size

    def reset(self) -> None:
        """Clear all cached entries."""
        for i in range(self.n_layers):
            self._seq_lens[i] = 0
            if not self._preallocated:
                self._k[i] = None
                self._v[i] = None


# ---------------------------------------------------------------------------
# Cache-aware transformer block forward
# ---------------------------------------------------------------------------

def block_forward_with_cache(
    block: TransformerBlock,
    x: np.ndarray,
    positions: np.ndarray,
    kv_cache: Optional[KVCache] = None,
    layer_idx: int = 0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Forward pass through a TransformerBlock with optional KV cache.

    During prefill, x has shape (B, L, d_model) and the full K, V are cached.
    During decode, x has shape (B, 1, d_model) and only the new token's K, V
    are appended to the cache.

    Args:
        block: TransformerBlock instance (uses its weights)
        x: Input embeddings, (B, L, d_model)
        positions: Integer position indices, (L,)
        kv_cache: If provided, use/update the cache
        layer_idx: Layer index into the cache
        mask: Additive attention mask. If None and no cache, uses standard
              causal mask. When using cache during decode, builds the correct
              mask automatically.

    Returns:
        Output tensor, (B, L, d_model)
    """
    x = np.asarray(x, dtype=np.float64)
    B, L, _ = x.shape

    x_norm = block.norm1.forward(x)

    # (B, L, d_model) @ (d_model, h*d_k) -> (B, L, h*d_k) -> (B, h, L, d_k)
    Q = (x_norm @ block.W_Q).reshape(B, L, block.num_heads, block.d_k).transpose(0, 2, 1, 3)
    K = (x_norm @ block.W_K).reshape(B, L, block.num_kv_heads, block.d_k).transpose(0, 2, 1, 3)
    V = (x_norm @ block.W_V).reshape(B, L, block.num_kv_heads, block.d_k).transpose(0, 2, 1, 3)

    cos_cache = block.rope.cos_cache
    sin_cache = block.rope.sin_cache

    Q_rot = apply_rope(Q, cos_cache, sin_cache, positions)
    K_rot = apply_rope(K, cos_cache, sin_cache, positions)

    if kv_cache is not None:
        K_full, V_full = kv_cache.append(layer_idx, K_rot, V)
    else:
        K_full = K_rot
        V_full = V

    K_exp = repeat_kv(K_full, block.group_size)
    V_exp = repeat_kv(V_full, block.group_size)

    kv_len = K_exp.shape[2]

    # (B, h, L, d_k) @ (B, h, d_k, kv_len) -> (B, h, L, kv_len)
    scores = Q_rot @ K_exp.transpose(0, 1, 3, 2) / np.sqrt(block.d_k)

    if mask is not None:
        scores = scores + mask
    else:
        if kv_cache is not None and L == 1:
            pass  # single query attending to full cache -- no masking needed
        else:
            causal = create_causal_mask(kv_len)
            if L < kv_len:
                # During prefill with partial cache (shouldn't happen normally),
                # slice the mask to (1, 1, L, kv_len)
                causal = causal[:, :, -L:, :]
            scores = scores + causal

    A = softmax(scores, axis=-1)

    # (B, h, L, kv_len) @ (B, h, kv_len, d_k) -> (B, h, L, d_k)
    attn_output = A @ V_exp

    concat = attn_output.transpose(0, 2, 1, 3).reshape(B, L, block.d_model)
    attn_out = concat @ block.W_O

    h = x + attn_out

    h_norm = block.norm2.forward(h)
    ffn_out = block.ffn.forward(h_norm)
    output = h + ffn_out

    return output


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_without_cache(
    model: CausalLM,
    prompt_tokens: np.ndarray,
    n_tokens: int,
    greedy: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Naive autoregressive generation -- full recomputation at every step.

    Args:
        model: CausalLM instance
        prompt_tokens: Shape (B, P) or (P,)
        n_tokens: Number of new tokens to generate
        greedy: Use argmax sampling
        seed: Random seed

    Returns:
        (tokens, projection_flops) where tokens is (B, P + n_tokens)
        and projection_flops counts Q/K/V/O projection multiply-adds.
    """
    prompt_tokens = np.asarray(prompt_tokens)
    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens[np.newaxis, :]

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    tokens = prompt_tokens.copy()
    d_model = model.d_model
    proj_flops = 0

    for _ in range(n_tokens):
        L = tokens.shape[1]
        # Full forward recomputes all projections for all L tokens per layer
        proj_flops += model.num_layers * 4 * 2 * L * d_model * d_model
        logits = model.forward(tokens)
        next_logits = logits[:, -1, :]
        next_tok = sample_token(next_logits, greedy=greedy, rng=rng)
        if next_tok.ndim == 0:
            next_tok = next_tok.reshape(1)
        tokens = np.concatenate([tokens, next_tok.reshape(-1, 1)], axis=1)

    return tokens, proj_flops


def generate_with_cache(
    model: CausalLM,
    prompt_tokens: np.ndarray,
    n_tokens: int,
    greedy: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    KV-cached autoregressive generation.

    Prefill phase: process the full prompt in one pass, populate cache.
    Decode phase: process one new token per step using cached K, V.

    Args:
        model: CausalLM instance
        prompt_tokens: Shape (B, P) or (P,)
        n_tokens: Number of new tokens to generate
        greedy: Use argmax sampling
        seed: Random seed

    Returns:
        (tokens, projection_flops) where tokens is (B, P + n_tokens)
        and projection_flops counts Q/K/V/O projection multiply-adds.
    """
    prompt_tokens = np.asarray(prompt_tokens)
    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens[np.newaxis, :]

    B, P = prompt_tokens.shape
    d_model = model.d_model
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    d_k = model.d_model // model.num_heads
    cache = KVCache(
        n_layers=model.num_layers,
        batch_size=B,
        n_kv_heads=model.num_kv_heads,
        d_k=d_k,
        max_seq_len=model.max_seq_len,
        dtype=np.float64,
    )

    proj_flops = 0

    # --- Prefill: process all prompt tokens at once ---
    x = model.embedding[prompt_tokens]  # (B, P, d_model)
    positions = np.arange(P)

    for layer_idx, block in enumerate(model.blocks):
        x = block_forward_with_cache(block, x, positions, cache, layer_idx)

    x = model.final_norm.forward(x)
    logits = x @ model.W_out  # (B, P, V)

    proj_flops += model.num_layers * 4 * 2 * P * d_model * d_model

    next_logits = logits[:, -1, :]
    next_tok = sample_token(next_logits, greedy=greedy, rng=rng)
    if next_tok.ndim == 0:
        next_tok = next_tok.reshape(1)
    next_tok = next_tok.reshape(-1, 1)

    tokens = np.concatenate([prompt_tokens, next_tok], axis=1)

    # --- Decode: one token at a time ---
    for step in range(1, n_tokens):
        tok = tokens[:, -1:]  # (B, 1)
        pos = np.array([P + step - 1])

        x = model.embedding[tok]  # (B, 1, d_model)

        for layer_idx, block in enumerate(model.blocks):
            x = block_forward_with_cache(block, x, pos, cache, layer_idx)

        x = model.final_norm.forward(x)
        logits = x @ model.W_out  # (B, 1, V)

        proj_flops += model.num_layers * 4 * 2 * 1 * d_model * d_model

        next_logits = logits[:, -1, :]
        next_tok = sample_token(next_logits, greedy=greedy, rng=rng)
        if next_tok.ndim == 0:
            next_tok = next_tok.reshape(1)
        tokens = np.concatenate([tokens, next_tok.reshape(-1, 1)], axis=1)

    return tokens, proj_flops


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def memory_usage(cache: KVCache) -> Dict[str, object]:
    """
    Detailed memory breakdown for a KVCache instance.

    Returns:
        Dict with total_bytes, per_layer_bytes, bytes_per_token, and human-readable sizes.
    """
    element_size = np.dtype(cache.dtype).itemsize
    per_token = 2 * cache.n_layers * cache.n_kv_heads * cache.d_k * element_size

    total = cache.memory_bytes()

    return {
        "total_bytes": total,
        "total_mb": total / (1024 ** 2),
        "bytes_per_token": per_token,
        "mb_per_token": per_token / (1024 ** 2),
        "seq_len": cache.seq_len,
        "n_layers": cache.n_layers,
        "n_kv_heads": cache.n_kv_heads,
        "d_k": cache.d_k,
        "element_size": element_size,
    }


def flops_comparison(
    prompt_len: int,
    n_tokens: int,
    n_layers: int,
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
) -> Dict[str, object]:
    """
    Theoretical projection FLOP comparison: with vs. without KV cache.

    Only counts Q/K/V/O projection multiply-adds (the part that changes from
    O(n^2) to O(n) with caching). Attention FLOPs are O(n^2) in both cases.

    Returns:
        Dict with without_cache, with_cache, speedup, and per-step breakdowns.
    """
    proj_per_token_per_layer = 4 * 2 * d_model * d_model

    without_cache = 0
    without_steps = []
    for i in range(n_tokens):
        L = prompt_len + i + 1
        step_flops = n_layers * 4 * 2 * L * d_model * d_model
        without_steps.append(step_flops)
        without_cache += step_flops

    with_cache = 0
    prefill = n_layers * 4 * 2 * prompt_len * d_model * d_model
    with_cache += prefill
    decode_steps = []
    for i in range(n_tokens):
        step_flops = n_layers * proj_per_token_per_layer
        decode_steps.append(step_flops)
        with_cache += step_flops

    speedup = without_cache / with_cache if with_cache > 0 else float("inf")

    return {
        "without_cache": without_cache,
        "with_cache": with_cache,
        "speedup": speedup,
        "prefill_flops": prefill,
        "without_steps": without_steps,
        "decode_steps": decode_steps,
    }


def model_kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    d_model: int,
    bytes_per_element: int = 2,
) -> int:
    """
    Total KV cache bytes for a full model at a given sequence length.

    Formula: 2 * n_layers * batch * n_heads * seq_len * d_k * bytes_per_element
    """
    d_k = d_model // n_heads
    return 2 * n_layers * batch_size * n_heads * seq_len * d_k * bytes_per_element
