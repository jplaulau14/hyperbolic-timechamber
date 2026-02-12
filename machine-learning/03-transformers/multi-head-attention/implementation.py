"""
Multi-Head Attention -- From-scratch NumPy implementation.

Runs h parallel attention heads using fused weight matrices and reshape/transpose
operations instead of per-head loops. Each head independently attends to different
representation subspaces, then results are concatenated and projected. This is the
core building block of the Transformer and the natural axis for tensor parallelism.
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax via the subtract-max trick."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(grad_output: np.ndarray, softmax_output: np.ndarray) -> np.ndarray:
    """
    Backward pass through softmax: dS = A * (g - rowsum(g * A)).

    Args:
        grad_output: Upstream gradient, same shape as softmax_output
        softmax_output: Cached softmax output A

    Returns:
        Gradient w.r.t. softmax input, same shape as input
    """
    dot = np.sum(grad_output * softmax_output, axis=-1, keepdims=True)
    return softmax_output * (grad_output - dot)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Additive causal mask with shape (1, 1, L, L) for broadcasting over batch and heads.

    Position i can attend to positions j <= i. Masked positions are -inf.
    """
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask.reshape(1, 1, seq_len, seq_len)


class MultiHeadAttention:
    """Multi-head self-attention with fused Q/K/V projections and output projection."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_bias: bool = True,
    ):
        """
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            use_bias: Whether to include bias terms in projections
        """
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.use_bias = use_bias

        self.W_Q = self._xavier((d_model, d_model))
        self.W_K = self._xavier((d_model, d_model))
        self.W_V = self._xavier((d_model, d_model))
        self.W_O = self._xavier((d_model, d_model))

        if use_bias:
            self.b_Q = np.zeros(d_model, dtype=np.float64)
            self.b_K = np.zeros(d_model, dtype=np.float64)
            self.b_V = np.zeros(d_model, dtype=np.float64)
            self.b_O = np.zeros(d_model, dtype=np.float64)
        else:
            self.b_Q = None
            self.b_K = None
            self.b_V = None
            self.b_O = None

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

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (B, L, d_model) -> (B, h, L, d_k) via reshape + transpose."""
        B, L, _ = x.shape
        # (B, L, d_model) -> (B, L, h, d_k) -> (B, h, L, d_k)
        return x.reshape(B, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (B, h, L, d_v) -> (B, L, d_model) via transpose + reshape."""
        B, _, L, _ = x.shape
        # (B, h, L, d_v) -> (B, L, h, d_v) -> (B, L, d_model)
        return x.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)

    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Multi-head attention forward pass.

        Args:
            X: Input tensor, shape (B, L, d_model)
            mask: Optional additive mask, broadcastable to (B, h, L, L)

        Returns:
            Output tensor, shape (B, L, d_model)
        """
        X = np.asarray(X, dtype=np.float64)
        B, L, _ = X.shape

        # (B, L, d_model) @ (d_model, d_model) -> (B, L, d_model)
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        if self.use_bias:
            Q = Q + self.b_Q
            K = K + self.b_K
            V = V + self.b_V

        # (B, L, d_model) -> (B, h, L, d_k)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # (B, h, L, d_k) @ (B, h, d_k, L) -> (B, h, L, L)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        A = softmax(scores, axis=-1)

        # (B, h, L, L) @ (B, h, L, d_v) -> (B, h, L, d_v)
        attn_output = A @ V

        # (B, h, L, d_v) -> (B, L, d_model)
        concat = self._merge_heads(attn_output)

        # (B, L, d_model) @ (d_model, d_model) -> (B, L, d_model)
        output = concat @ self.W_O
        if self.use_bias:
            output = output + self.b_O

        self._cache = {
            "X": X, "Q": Q, "K": K, "V": V, "A": A,
            "attn_output": attn_output, "concat": concat, "mask": mask,
        }
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Full backward pass through multi-head attention.

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
        K = self._cache["K"]
        V = self._cache["V"]
        A = self._cache["A"]
        concat = self._cache["concat"]

        B, L, _ = X.shape
        scale = np.sqrt(self.d_k)

        # Step 1: gradient through output projection (output = concat @ W_O + b_O)
        # (B, L, d_model) @ (d_model, d_model) -> (B, L, d_model)
        grad_concat = grad_output @ self.W_O.T
        self.grad_W_O = np.einsum("blm,bln->mn", concat, grad_output)
        if self.use_bias:
            self.grad_b_O = grad_output.sum(axis=(0, 1))

        # Step 2: gradient through head merge (reverse: reshape then transpose)
        # (B, L, d_model) -> (B, L, h, d_v) -> (B, h, L, d_v)
        grad_attn_output = grad_concat.reshape(B, L, self.num_heads, self.d_v).transpose(0, 2, 1, 3)

        # Step 3: gradient through value weighting (attn_output = A @ V)
        # (B, h, L, d_v) @ (B, h, d_v, L) -> (B, h, L, L)
        grad_A = grad_attn_output @ V.transpose(0, 1, 3, 2)
        # (B, h, L, L) @ (B, h, L, d_v) -> (B, h, L, d_v)
        grad_V = A.transpose(0, 1, 3, 2) @ grad_attn_output

        # Step 4: gradient through softmax
        grad_scores = softmax_backward(grad_A, A)

        # Step 5: gradient through scaling and QK^T
        grad_raw = grad_scores / scale
        # (B, h, L, L) @ (B, h, L, d_k) -> (B, h, L, d_k)
        grad_Q = grad_raw @ K
        # (B, h, L, L)^T @ (B, h, L, d_k) -> (B, h, L, d_k)
        grad_K = grad_raw.transpose(0, 1, 3, 2) @ Q

        # Step 6: gradient through head split (reverse: transpose then reshape)
        # (B, h, L, d_k) -> (B, L, h, d_k) -> (B, L, d_model)
        grad_Q_flat = grad_Q.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        grad_K_flat = grad_K.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        grad_V_flat = grad_V.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)

        # Step 7: gradient through Q/K/V projections
        self.grad_W_Q = np.einsum("blm,bld->md", X, grad_Q_flat)
        self.grad_W_K = np.einsum("blm,bld->md", X, grad_K_flat)
        self.grad_W_V = np.einsum("blm,bld->md", X, grad_V_flat)

        if self.use_bias:
            self.grad_b_Q = grad_Q_flat.sum(axis=(0, 1))
            self.grad_b_K = grad_K_flat.sum(axis=(0, 1))
            self.grad_b_V = grad_V_flat.sum(axis=(0, 1))

        # dX accumulates from all three projection paths
        grad_X = grad_Q_flat @ self.W_Q.T + grad_K_flat @ self.W_K.T + grad_V_flat @ self.W_V.T

        return grad_X


def count_flops(batch_size: int, seq_len: int, d_model: int, n_heads: int) -> int:
    """
    Total forward-pass FLOPs for multi-head attention.

    Counts multiply-accumulate as 2 FLOPs per element.

    Args:
        batch_size: B
        seq_len: L
        d_model: Model dimension
        n_heads: Number of heads

    Returns:
        Total FLOPs (integer)
    """
    B, L, h = batch_size, seq_len, n_heads
    d_k = d_model // h

    proj_qkv = 3 * 2 * B * L * d_model * d_model
    proj_o = 2 * B * L * d_model * d_model

    # Attention core: h heads each with d_k dimensions
    qk = 2 * B * h * L * L * d_k      # Q @ K^T per head
    av = 2 * B * h * L * L * d_k      # A @ V per head (d_v = d_k)
    sm = 5 * B * h * L * L

    return proj_qkv + proj_o + qk + av + sm


def count_memory_bytes(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
    dtype: str = "float32",
) -> int:
    """
    Bytes for key intermediate tensors in the forward pass.

    Counts Q, K, V (after split), attention matrices (per head), attention output,
    and concatenated output.

    Args:
        batch_size: B
        seq_len: L
        d_model: Model dimension
        n_heads: Number of heads
        dtype: Data type string ('float32' or 'float16')

    Returns:
        Total bytes (integer)
    """
    bytes_per_element = 4 if dtype == "float32" else 2
    B, L, h = batch_size, seq_len, n_heads
    d_k = d_model // h

    q_bytes = B * h * L * d_k * bytes_per_element
    k_bytes = B * h * L * d_k * bytes_per_element
    v_bytes = B * h * L * d_k * bytes_per_element
    attn_bytes = B * h * L * L * bytes_per_element
    attn_output_bytes = B * h * L * d_k * bytes_per_element
    concat_bytes = B * L * d_model * bytes_per_element

    return q_bytes + k_bytes + v_bytes + attn_bytes + attn_output_bytes + concat_bytes
