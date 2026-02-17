"""
Causal Decoding -- From-scratch NumPy implementation.

Complete decoder-only language model: token embeddings, stacked transformer
blocks, output projection (with optional weight tying), and autoregressive
generation with temperature/top-k/top-p sampling. This is the naive version
that recomputes the full forward pass at every generation step -- no KV cache.
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

_block_mod = import_module("03-transformers.transformer-block.implementation")
TransformerBlock = _block_mod.TransformerBlock
_block_count_parameters = _block_mod.count_parameters
_block_count_flops = _block_mod.count_flops


def _xavier(shape: Tuple[int, ...]) -> np.ndarray:
    std = np.sqrt(2.0 / (shape[0] + shape[1]))
    return np.random.randn(*shape).astype(np.float64) * std


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------

def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Scale logits by temperature.

    Args:
        logits: Raw logits, shape (..., V)
        temperature: Temperature > 0. Values near 0 sharpen the distribution.

    Returns:
        Scaled logits, same shape.
    """
    if temperature == 0.0:
        return logits
    return logits / temperature


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only the top-k logits; set the rest to -inf.

    Args:
        logits: Shape (..., V)
        k: Number of top entries to keep (>= 1)

    Returns:
        Filtered logits, same shape.
    """
    if k <= 0:
        raise ValueError(f"top_k must be >= 1, got {k}")
    if k >= logits.shape[-1]:
        return logits.copy()

    sorted_logits = np.sort(logits, axis=-1)
    threshold = sorted_logits[..., -k:][..., :1]
    filtered = np.where(logits >= threshold, logits, -np.inf)
    return filtered


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus (top-p) sampling filter.

    Keeps the smallest set of tokens whose cumulative probability >= p.

    Args:
        logits: Shape (..., V)
        p: Cumulative probability threshold in (0, 1]

    Returns:
        Filtered logits with masked positions set to -inf.
    """
    if p <= 0.0 or p > 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    if p == 1.0:
        return logits.copy()

    probs = softmax(logits, axis=-1)
    sorted_indices = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = np.cumsum(sorted_probs, axis=-1)

    # Mask tokens where cumulative prob (excluding current token) >= p
    sorted_mask = (cumulative - sorted_probs) >= p

    # Scatter mask back to original positions
    mask = np.zeros_like(sorted_mask)
    np.put_along_axis(mask, sorted_indices, sorted_mask, axis=-1)

    filtered = np.where(mask, -np.inf, logits)
    return filtered


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Full sampling pipeline: temperature -> top-k -> top-p -> softmax -> sample.

    Args:
        logits: Shape (B, V) or (V,)
        temperature: Sampling temperature (> 0 unless greedy)
        top_k: If > 0, apply top-k filtering
        top_p: If < 1.0, apply nucleus filtering
        greedy: If True, return argmax (ignores temperature/top_k/top_p)
        rng: NumPy RandomState for reproducibility

    Returns:
        Sampled token IDs, shape (B,) or scalar.
    """
    if greedy:
        return np.argmax(logits, axis=-1)

    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0 for non-greedy sampling, got {temperature}")
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    if rng is None:
        rng = np.random.RandomState()

    scaled = temperature_scale(logits, temperature)

    if top_k > 0:
        scaled = top_k_filter(scaled, top_k)

    if top_p < 1.0:
        scaled = top_p_filter(scaled, top_p)

    probs = softmax(scaled, axis=-1)

    if logits.ndim == 1:
        return rng.choice(len(probs), p=probs)

    # Batch sampling
    B = probs.shape[0]
    tokens = np.empty(B, dtype=np.int64)
    for i in range(B):
        tokens[i] = rng.choice(probs.shape[-1], p=probs[i])
    return tokens


# ---------------------------------------------------------------------------
# CausalLM
# ---------------------------------------------------------------------------

class CausalLM:
    """Decoder-only language model with autoregressive generation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        tie_weights: bool = True,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

        self.embedding = _xavier((vocab_size, d_model))

        self.blocks: List[TransformerBlock] = []
        for _ in range(num_layers):
            self.blocks.append(
                TransformerBlock(d_model, num_heads, num_kv_heads, d_ff, max_seq_len, rope_theta)
            )

        self.final_norm = RMSNorm(d_model)

        if tie_weights:
            self.W_out = self.embedding.T
        else:
            self.W_out = _xavier((d_model, vocab_size))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Full forward pass: embedding -> N blocks -> final norm -> output logits.

        Args:
            token_ids: Integer token IDs, shape (B, L)

        Returns:
            Logits, shape (B, L, V)
        """
        token_ids = np.asarray(token_ids)
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        B, L = token_ids.shape

        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}"
            )
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            raise ValueError(
                f"Token IDs must be in [0, {self.vocab_size}), "
                f"got min={token_ids.min()}, max={token_ids.max()}"
            )

        x = self.embedding[token_ids]

        mask = create_causal_mask(L)
        positions = np.arange(L)

        for block in self.blocks:
            x = block.forward(x, mask=mask, positions=positions)

        x = self.final_norm.forward(x)

        # (B, L, d_model) @ (d_model, V) -> (B, L, V)
        logits = x @ self.W_out

        return logits

    def generate(
        self,
        prompt_tokens: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        greedy: bool = False,
        eos_token_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Autoregressive generation (naive, no KV cache).

        Args:
            prompt_tokens: Shape (B, P) or (P,) prompt token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus filtering (1.0 = disabled)
            greedy: If True, always pick argmax
            eos_token_id: Stop generation when this token is produced
            seed: Random seed for reproducibility

        Returns:
            Full sequence including prompt, shape (B, P + n_generated)
        """
        prompt_tokens = np.asarray(prompt_tokens)
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[np.newaxis, :]

        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        tokens = prompt_tokens.copy()

        for _ in range(max_new_tokens):
            if tokens.shape[1] >= self.max_seq_len:
                break

            logits = self.forward(tokens)
            next_logits = logits[:, -1, :]

            next_token = sample_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                rng=rng,
            )

            if next_token.ndim == 0:
                next_token = next_token.reshape(1)
            next_token = next_token.reshape(-1, 1)

            tokens = np.concatenate([tokens, next_token], axis=1)

            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break

        return tokens


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def count_model_parameters(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    d_ff: int,
    tie_weights: bool = True,
) -> Dict[str, object]:
    """
    Full model parameter breakdown.

    Returns:
        Dict with component counts, percentages, and total.
    """
    embedding_params = vocab_size * d_model

    block_params = _block_count_parameters(d_model, num_heads, num_kv_heads, d_ff)
    per_block = block_params["total"]
    total_blocks = num_layers * per_block

    final_norm_params = d_model

    output_proj_params = 0 if tie_weights else d_model * vocab_size

    total = embedding_params + total_blocks + final_norm_params + output_proj_params

    return {
        "embedding": embedding_params,
        "per_block": per_block,
        "total_blocks": total_blocks,
        "final_norm": final_norm_params,
        "output_proj": output_proj_params,
        "total": total,
        "embedding_pct": 100.0 * embedding_params / total,
        "blocks_pct": 100.0 * total_blocks / total,
        "final_norm_pct": 100.0 * final_norm_params / total,
        "output_proj_pct": 100.0 * output_proj_params / total,
        "tie_weights": tie_weights,
    }


def generation_flops(
    prompt_len: int,
    num_new_tokens: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    d_ff: int,
    vocab_size: int,
) -> Dict[str, object]:
    """
    Total FLOPs for naive generation (full recompute each step, no KV cache).

    Returns:
        Dict with total FLOPs, per-step breakdown, and metadata.
    """
    per_step_flops = []
    total = 0

    for i in range(num_new_tokens):
        seq_len = prompt_len + i + 1

        block_flops = _block_count_flops(1, seq_len, d_model, num_heads, num_kv_heads, d_ff)
        blocks_total = num_layers * block_flops["total"]

        output_proj = 2 * seq_len * d_model * vocab_size
        final_norm = 2 * seq_len * d_model

        step_total = blocks_total + output_proj + final_norm
        per_step_flops.append(step_total)
        total += step_total

    return {
        "total": total,
        "per_step": per_step_flops,
        "num_steps": num_new_tokens,
        "prompt_len": prompt_len,
    }


def generation_flops_with_cache(
    prompt_len: int,
    num_new_tokens: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    d_ff: int,
    vocab_size: int,
) -> Dict[str, object]:
    """
    Theoretical FLOPs with KV cache for comparison.

    Prefill processes all prompt tokens at once. Each decode step projects
    only 1 new token and attends over the growing cache.
    """
    prefill_block = _block_count_flops(1, prompt_len, d_model, num_heads, num_kv_heads, d_ff)
    prefill_blocks = num_layers * prefill_block["total"]
    prefill_output_proj = 2 * prompt_len * d_model * vocab_size
    prefill_norm = 2 * prompt_len * d_model
    prefill_total = prefill_blocks + prefill_output_proj + prefill_norm

    d_k = d_model // num_heads
    decode_total = 0
    per_step_flops = []

    for i in range(num_new_tokens):
        cache_len = prompt_len + i

        # Q/K/V/O projections for 1 token
        proj_q = 2 * d_model * d_model
        proj_k = 2 * d_model * (num_kv_heads * d_k)
        proj_v = 2 * d_model * (num_kv_heads * d_k)
        proj_o = 2 * d_model * d_model
        proj_total = proj_q + proj_k + proj_v + proj_o

        # Attention: Q (1 token) attends to K/V (cache_len + 1 tokens)
        kv_len = cache_len + 1
        attn_qk = 2 * num_heads * kv_len * d_k
        attn_av = 2 * num_heads * kv_len * d_k
        attn_softmax = 5 * num_heads * kv_len
        attn_total = attn_qk + attn_av + attn_softmax

        # RoPE for 1 token
        rope = 6 * num_heads * d_k

        # FFN for 1 token
        ffn_total = 2 * d_model * d_ff * 3

        # Norms for 1 token
        norm = 4 * d_model

        step_blocks = num_layers * (proj_total + attn_total + rope + ffn_total + norm)
        step_output_proj = 2 * d_model * vocab_size
        step_norm = 2 * d_model
        step_total = step_blocks + step_output_proj + step_norm

        per_step_flops.append(step_total)
        decode_total += step_total

    return {
        "total": prefill_total + decode_total,
        "prefill": prefill_total,
        "decode": decode_total,
        "per_step": per_step_flops,
        "num_steps": num_new_tokens,
        "prompt_len": prompt_len,
    }
