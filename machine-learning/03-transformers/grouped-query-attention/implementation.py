"""
Grouped-Query Attention -- From-scratch NumPy implementation.

GQA reduces the number of key/value heads while keeping the full number of query
heads. Multiple query heads share a single KV head, forming groups. This unifies
MHA (num_kv_heads == num_heads), GQA (1 < num_kv_heads < num_heads), and MQA
(num_kv_heads == 1) along a single axis. The KV cache shrinks by factor
num_heads / num_kv_heads, which is the key architectural decision enabling
long-context inference in Llama 2, Mistral, and all modern production LLMs.
"""

import numpy as np
from typing import Optional, Tuple, Dict


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax via the subtract-max trick."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(grad_output: np.ndarray, softmax_output: np.ndarray) -> np.ndarray:
    """Backward pass through softmax: dS = A * (g - rowsum(g * A))."""
    dot = np.sum(grad_output * softmax_output, axis=-1, keepdims=True)
    return softmax_output * (grad_output - dot)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Additive causal mask with shape (1, 1, L, L) for broadcasting over batch and heads.

    Position i can attend to positions j <= i. Masked positions are -1e9.
    """
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -1e9
    return mask.reshape(1, 1, seq_len, seq_len)


def repeat_kv(x: np.ndarray, num_repeats: int) -> np.ndarray:
    """
    Expand KV heads to match query heads.

    Args:
        x: KV tensor, shape (B, h_kv, L, d)
        num_repeats: Group size g = num_heads // num_kv_heads

    Returns:
        Expanded tensor, shape (B, h_kv * num_repeats, L, d)
    """
    if num_repeats == 1:
        return x
    return np.repeat(x, repeats=num_repeats, axis=1)


def reduce_kv_grad(grad_expanded: np.ndarray, num_kv_heads: int, group_size: int) -> np.ndarray:
    """
    Sum gradients from query head groups back to shared KV heads.

    Args:
        grad_expanded: Gradient w.r.t. expanded KV, shape (B, h, L, d)
        num_kv_heads: Number of KV heads h_kv
        group_size: Number of query heads per KV head g

    Returns:
        Reduced gradient, shape (B, h_kv, L, d)
    """
    if group_size == 1:
        return grad_expanded
    B, _, L, d = grad_expanded.shape
    return grad_expanded.reshape(B, num_kv_heads, group_size, L, d).sum(axis=2)


class GroupedQueryAttention:
    """Grouped-query attention with configurable KV head count."""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of query heads h
            num_kv_heads: Number of KV heads h_kv (must divide num_heads)
        """
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.group_size = num_heads // num_kv_heads

        self.W_Q = self._xavier((d_model, d_model))
        self.W_K = self._xavier((d_model, num_kv_heads * self.d_k))
        self.W_V = self._xavier((d_model, num_kv_heads * self.d_v))
        self.W_O = self._xavier((d_model, d_model))

        self.b_Q = np.zeros(d_model, dtype=np.float64)
        self.b_K = np.zeros(num_kv_heads * self.d_k, dtype=np.float64)
        self.b_V = np.zeros(num_kv_heads * self.d_v, dtype=np.float64)
        self.b_O = np.zeros(d_model, dtype=np.float64)

        self._cache: Optional[dict] = None

        self.grad_W_Q: Optional[np.ndarray] = None
        self.grad_W_K: Optional[np.ndarray] = None
        self.grad_W_V: Optional[np.ndarray] = None
        self.grad_W_O: Optional[np.ndarray] = None
        self.grad_b_Q: Optional[np.ndarray] = None
        self.grad_b_K: Optional[np.ndarray] = None
        self.grad_b_V: Optional[np.ndarray] = None
        self.grad_b_O: Optional[np.ndarray] = None

    def _xavier(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier/Glorot normal initialization."""
        std = np.sqrt(2.0 / (shape[0] + shape[1]))
        return np.random.randn(*shape).astype(np.float64) * std

    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        GQA forward pass.

        Args:
            X: Input tensor, shape (B, L, d_model)
            mask: Optional additive mask, broadcastable to (B, h, L, L)

        Returns:
            Output tensor, shape (B, L, d_model)
        """
        X = np.asarray(X, dtype=np.float64)
        B, L, _ = X.shape

        # (B, L, d_model) @ (d_model, h*d_k) -> (B, L, h*d_k)
        Q = X @ self.W_Q + self.b_Q
        # (B, L, d_model) @ (d_model, h_kv*d_k) -> (B, L, h_kv*d_k)
        K = X @ self.W_K + self.b_K
        V = X @ self.W_V + self.b_V

        # (B, L, h*d_k) -> (B, h, L, d_k)
        Q = Q.reshape(B, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        # (B, L, h_kv*d_k) -> (B, h_kv, L, d_k)
        K = K.reshape(B, L, self.num_kv_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, L, self.num_kv_heads, self.d_v).transpose(0, 2, 1, 3)

        # (B, h_kv, L, d_k) -> (B, h, L, d_k)
        K_exp = repeat_kv(K, self.group_size)
        V_exp = repeat_kv(V, self.group_size)

        # (B, h, L, d_k) @ (B, h, d_k, L) -> (B, h, L, L)
        scores = Q @ K_exp.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        A = softmax(scores, axis=-1)

        # (B, h, L, L) @ (B, h, L, d_v) -> (B, h, L, d_v)
        attn_output = A @ V_exp

        # (B, h, L, d_v) -> (B, L, h, d_v) -> (B, L, d_model)
        concat = attn_output.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)

        # (B, L, d_model) @ (d_model, d_model) -> (B, L, d_model)
        output = concat @ self.W_O + self.b_O

        self._cache = {
            "X": X, "Q": Q, "K_exp": K_exp, "V_exp": V_exp,
            "A": A, "attn_output": attn_output, "concat": concat, "mask": mask,
        }
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Full backward pass through grouped-query attention.

        Args:
            grad_output: Upstream gradient dL/d(output), shape (B, L, d_model)

        Returns:
            Gradient dL/dX, shape (B, L, d_model)
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward().")

        grad_output = np.asarray(grad_output, dtype=np.float64)
        X = self._cache["X"]
        Q = self._cache["Q"]
        K_exp = self._cache["K_exp"]
        V_exp = self._cache["V_exp"]
        A = self._cache["A"]
        concat = self._cache["concat"]

        B, L, _ = X.shape
        scale = np.sqrt(self.d_k)

        # Gradient through output projection: output = concat @ W_O + b_O
        grad_concat = grad_output @ self.W_O.T
        self.grad_W_O = np.einsum("blm,bln->mn", concat, grad_output)
        self.grad_b_O = grad_output.sum(axis=(0, 1))

        # Gradient through head merge: (B, L, d_model) -> (B, h, L, d_v)
        grad_attn_output = grad_concat.reshape(B, L, self.num_heads, self.d_v).transpose(0, 2, 1, 3)

        # Gradient through value weighting: attn_output = A @ V_exp
        grad_A = grad_attn_output @ V_exp.transpose(0, 1, 3, 2)
        grad_V_exp = A.transpose(0, 1, 3, 2) @ grad_attn_output

        # Gradient through softmax
        grad_scores = softmax_backward(grad_A, A)

        # Gradient through scaling and QK^T
        grad_raw = grad_scores / scale
        grad_Q = grad_raw @ K_exp
        grad_K_exp = grad_raw.transpose(0, 1, 3, 2) @ Q

        # Gradient through repeat-interleave: (B, h, L, d) -> (B, h_kv, L, d)
        grad_K = reduce_kv_grad(grad_K_exp, self.num_kv_heads, self.group_size)
        grad_V = reduce_kv_grad(grad_V_exp, self.num_kv_heads, self.group_size)

        # Gradient through head split: reverse transpose + reshape
        grad_Q_flat = grad_Q.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        grad_K_flat = grad_K.transpose(0, 2, 1, 3).reshape(B, L, self.num_kv_heads * self.d_k)
        grad_V_flat = grad_V.transpose(0, 2, 1, 3).reshape(B, L, self.num_kv_heads * self.d_v)

        # Gradient through projections
        self.grad_W_Q = np.einsum("blm,bld->md", X, grad_Q_flat)
        self.grad_W_K = np.einsum("blm,bld->md", X, grad_K_flat)
        self.grad_W_V = np.einsum("blm,bld->md", X, grad_V_flat)

        self.grad_b_Q = grad_Q_flat.sum(axis=(0, 1))
        self.grad_b_K = grad_K_flat.sum(axis=(0, 1))
        self.grad_b_V = grad_V_flat.sum(axis=(0, 1))

        grad_X = grad_Q_flat @ self.W_Q.T + grad_K_flat @ self.W_K.T + grad_V_flat @ self.W_V.T
        return grad_X


def count_parameters(d_model: int, num_heads: int, num_kv_heads: int) -> Dict[str, int]:
    """
    Parameter counts for each projection matrix and total.

    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads

    Returns:
        Dict with keys 'W_Q', 'W_K', 'W_V', 'W_O', 'total', and bias counts
    """
    d_k = d_model // num_heads
    wq = d_model * d_model
    wk = d_model * (num_kv_heads * d_k)
    wv = d_model * (num_kv_heads * d_k)
    wo = d_model * d_model

    bq = d_model
    bk = num_kv_heads * d_k
    bv = num_kv_heads * d_k
    bo = d_model

    total_weights = wq + wk + wv + wo
    total_biases = bq + bk + bv + bo

    return {
        "W_Q": wq, "W_K": wk, "W_V": wv, "W_O": wo,
        "b_Q": bq, "b_K": bk, "b_V": bv, "b_O": bo,
        "total_weights": total_weights,
        "total_biases": total_biases,
        "total": total_weights + total_biases,
    }


def kv_cache_size(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    d_k: int,
    dtype: str = "float16",
) -> int:
    """
    KV cache bytes for a single layer.

    Args:
        batch_size: B
        seq_len: L
        num_kv_heads: h_kv
        d_k: Head dimension
        dtype: 'float16' (2 bytes) or 'float32' (4 bytes)

    Returns:
        Total bytes for K and V caches in one layer
    """
    bytes_per_element = 2 if dtype == "float16" else 4
    return 2 * batch_size * num_kv_heads * seq_len * d_k * bytes_per_element


def kv_cache_size_model(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    d_k: int,
    dtype: str = "float16",
) -> int:
    """
    Total KV cache bytes across all layers.

    Args:
        batch_size: B
        seq_len: L
        num_layers: Number of transformer layers
        num_kv_heads: h_kv
        d_k: Head dimension
        dtype: 'float16' or 'float32'

    Returns:
        Total bytes
    """
    return num_layers * kv_cache_size(batch_size, seq_len, num_kv_heads, d_k, dtype)


def count_flops(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
) -> Dict[str, int]:
    """
    Forward-pass FLOPs breakdown for GQA.

    Counts multiply-accumulate as 2 FLOPs per element.

    Args:
        batch_size: B
        seq_len: L
        d_model: Model dimension
        num_heads: Number of query heads h
        num_kv_heads: Number of KV heads h_kv

    Returns:
        Dict with 'proj_q', 'proj_k', 'proj_v', 'proj_o', 'attn_qk', 'attn_av',
        'attn_softmax', 'proj_total', 'attn_total', 'total'
    """
    B, L, h = batch_size, seq_len, num_heads
    d_k = d_model // h

    proj_q = 2 * B * L * d_model * d_model
    proj_k = 2 * B * L * d_model * (num_kv_heads * d_k)
    proj_v = 2 * B * L * d_model * (num_kv_heads * d_k)
    proj_o = 2 * B * L * d_model * d_model

    attn_qk = 2 * B * h * L * L * d_k
    attn_av = 2 * B * h * L * L * d_k
    attn_softmax = 5 * B * h * L * L

    proj_total = proj_q + proj_k + proj_v + proj_o
    attn_total = attn_qk + attn_av + attn_softmax

    return {
        "proj_q": proj_q, "proj_k": proj_k, "proj_v": proj_v, "proj_o": proj_o,
        "attn_qk": attn_qk, "attn_av": attn_av, "attn_softmax": attn_softmax,
        "proj_total": proj_total, "attn_total": attn_total,
        "total": proj_total + attn_total,
    }
