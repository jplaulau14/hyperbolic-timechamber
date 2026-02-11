"""
Self-Attention -- From-scratch NumPy implementation.

Single-head scaled dot-product attention as described in "Attention Is All You
Need" (Vaswani et al., 2017). Each position in a sequence attends to all other
positions, computing a weighted sum of value vectors where the weights come from
query-key dot products. This is the O(n^2) operation that every inference
optimization (Flash Attention, KV caching, GQA) exists to accelerate.
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax via the subtract-max trick.

    Args:
        x: Input logits, arbitrary shape
        axis: Axis along which to apply softmax

    Returns:
        Probability distribution along the given axis, same shape as x
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(grad_output: np.ndarray, softmax_output: np.ndarray) -> np.ndarray:
    """
    Backward pass through softmax.

    Implements: dS = A * (g - rowsum(g * A))
    where A is the softmax output and g is the upstream gradient.

    Args:
        grad_output: Upstream gradient, same shape as softmax_output
        softmax_output: Cached output from the softmax forward pass

    Returns:
        Gradient with respect to softmax input, same shape as input
    """
    dot = np.sum(grad_output * softmax_output, axis=-1, keepdims=True)
    return softmax_output * (grad_output - dot)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Generate an additive causal mask for autoregressive attention.

    Position i can attend to positions j <= i but not j > i. Masked positions
    are set to -inf so they become zero after softmax.

    Args:
        seq_len: Sequence length n

    Returns:
        Mask of shape (n, n) with 0 on/below diagonal, -inf above
    """
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask


def create_padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """
    Generate an additive padding mask from sequence lengths.

    Columns corresponding to padded positions are set to -inf so those
    positions receive zero attention weight after softmax.

    Args:
        lengths: Actual sequence lengths per batch element, shape (B,)
        max_len: Maximum sequence length (total padded length)

    Returns:
        Mask of shape (B, 1, max_len), broadcastable to (B, n, n)
    """
    # (B, max_len): True for valid positions
    positions = np.arange(max_len)[None, :]
    valid = positions < lengths[:, None]
    mask = np.where(valid[:, None, :], 0.0, -np.inf)
    return mask


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries, shape (B, n, d_k)
        K: Keys, shape (B, n, d_k)
        V: Values, shape (B, n, d_v)
        mask: Optional additive mask, broadcastable to (B, n, n)

    Returns:
        output: Weighted sum of values, shape (B, n, d_v)
        attention_weights: Attention distribution, shape (B, n, n)
    """
    d_k = Q.shape[-1]

    # (B, n, d_k) @ (B, d_k, n) -> (B, n, n)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    A = softmax(scores, axis=-1)

    # (B, n, n) @ (B, n, d_v) -> (B, n, d_v)
    output = A @ V
    return output, A


class SelfAttention:
    """Single-head self-attention with learned Q/K/V/O projections."""

    def __init__(
        self,
        d_model: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_out: Optional[int] = None,
        use_bias: bool = True,
    ):
        """
        Args:
            d_model: Input/output model dimension
            d_k: Key/query dimension (defaults to d_model)
            d_v: Value dimension (defaults to d_k)
            d_out: Output dimension (defaults to d_model)
            use_bias: Whether to include bias terms in projections
        """
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else self.d_k
        self.d_out = d_out if d_out is not None else d_model
        self.use_bias = use_bias

        self.W_Q = self._xavier((d_model, self.d_k))
        self.W_K = self._xavier((d_model, self.d_k))
        self.W_V = self._xavier((d_model, self.d_v))
        self.W_O = self._xavier((self.d_v, self.d_out))

        if use_bias:
            self.b_Q = np.zeros(self.d_k, dtype=np.float64)
            self.b_K = np.zeros(self.d_k, dtype=np.float64)
            self.b_V = np.zeros(self.d_v, dtype=np.float64)
            self.b_O = np.zeros(self.d_out, dtype=np.float64)
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

    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full self-attention forward pass with output projection.

        Args:
            X: Input tensor, shape (B, n, d_model)
            mask: Optional additive mask, broadcastable to (B, n, n)

        Returns:
            Output tensor, shape (B, n, d_out)
        """
        X = np.asarray(X, dtype=np.float64)

        # (B, n, d_model) @ (d_model, d_k) -> (B, n, d_k)
        Q = X @ self.W_Q
        K = X @ self.W_K
        # (B, n, d_model) @ (d_model, d_v) -> (B, n, d_v)
        V = X @ self.W_V

        if self.use_bias:
            Q = Q + self.b_Q
            K = K + self.b_K
            V = V + self.b_V

        O, A = scaled_dot_product_attention(Q, K, V, mask)

        # (B, n, d_v) @ (d_v, d_out) -> (B, n, d_out)
        output = O @ self.W_O
        if self.use_bias:
            output = output + self.b_O

        self._cache = {"X": X, "Q": Q, "K": K, "V": V, "A": A, "O": O, "mask": mask}
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Full backward pass through self-attention.

        Follows the 6-step derivation: output projection -> value weighting ->
        softmax -> scaling -> QK^T -> input projections.

        Args:
            grad_output: Upstream gradient dL/d(output), shape (B, n, d_out)

        Returns:
            Gradient dL/dX, shape (B, n, d_model)
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward().")

        grad_output = np.asarray(grad_output, dtype=np.float64)
        X = self._cache["X"]
        Q = self._cache["Q"]
        K = self._cache["K"]
        V = self._cache["V"]
        A = self._cache["A"]
        O = self._cache["O"]

        B, n, _ = X.shape
        d_k = self.d_k
        scale = np.sqrt(d_k)

        # Step 1: gradient through output projection (out = O @ W_O + b_O)
        # (B, n, d_out) @ (d_out, d_v) -> (B, n, d_v)
        grad_O = grad_output @ self.W_O.T
        # (d_v, B*n) @ (B*n, d_out) -> (d_v, d_out)
        self.grad_W_O = np.einsum("biv,bio->vo", O, grad_output)
        if self.use_bias:
            self.grad_b_O = grad_output.sum(axis=(0, 1))

        # Step 2: gradient through value weighting (O = A @ V)
        # (B, n, d_v) @ (B, d_v, n) -> (B, n, n)
        grad_A = grad_O @ V.transpose(0, 2, 1)
        # (B, n, n) @ (B, n, d_v) -> (B, n, d_v)
        grad_V = A.transpose(0, 2, 1) @ grad_O

        # Step 3: gradient through softmax
        grad_scores = softmax_backward(grad_A, A)

        # Step 4: gradient through scaling (scores = raw_scores / sqrt(d_k))
        grad_raw_scores = grad_scores / scale

        # Step 5: gradient through QK^T
        # (B, n, n) @ (B, n, d_k) -> (B, n, d_k)
        grad_Q = grad_raw_scores @ K
        # (B, n, n)^T @ (B, n, d_k) -> (B, n, d_k)
        grad_K = grad_raw_scores.transpose(0, 2, 1) @ Q

        # Step 6: gradient through Q/K/V projections
        # (d_model, B*n) @ (B*n, d_k) -> (d_model, d_k)
        self.grad_W_Q = np.einsum("bnd,bnk->dk", X, grad_Q)
        self.grad_W_K = np.einsum("bnd,bnk->dk", X, grad_K)
        self.grad_W_V = np.einsum("bnd,bnv->dv", X, grad_V)

        if self.use_bias:
            self.grad_b_Q = grad_Q.sum(axis=(0, 1))
            self.grad_b_K = grad_K.sum(axis=(0, 1))
            self.grad_b_V = grad_V.sum(axis=(0, 1))

        # dX accumulates from all three projection paths
        # (B, n, d_k) @ (d_k, d_model) -> (B, n, d_model)
        grad_X = grad_Q @ self.W_Q.T + grad_K @ self.W_K.T + grad_V @ self.W_V.T

        return grad_X


def count_flops(
    batch_size: int,
    seq_len: int,
    d_model: int,
    d_k: int,
    d_v: int,
) -> int:
    """
    Total forward-pass FLOPs for single-head self-attention.

    Counts multiply-accumulate as 2 FLOPs per element (one multiply, one add).

    Args:
        batch_size: B
        seq_len: n
        d_model: Input dimension
        d_k: Key/query dimension
        d_v: Value dimension

    Returns:
        Total FLOPs (integer)
    """
    B, n = batch_size, seq_len

    proj_q = 2 * B * n * d_model * d_k
    proj_k = 2 * B * n * d_model * d_k
    proj_v = 2 * B * n * d_model * d_v
    proj_o = 2 * B * n * d_v * d_model

    qk = 2 * B * n * n * d_k
    av = 2 * B * n * n * d_v

    # softmax: exp + sum + div per element ~ 5 ops per element
    sm = 5 * B * n * n

    return proj_q + proj_k + proj_v + proj_o + qk + av + sm


def count_memory_bytes(
    batch_size: int,
    seq_len: int,
    d_k: int,
    d_v: int,
    dtype: str = "float32",
) -> int:
    """
    Bytes for key intermediate tensors in the forward pass.

    Counts Q, K, V, attention matrix, and output.

    Args:
        batch_size: B
        seq_len: n
        d_k: Key/query dimension
        d_v: Value dimension
        dtype: Data type string ('float32' or 'float16')

    Returns:
        Total bytes (integer)
    """
    bytes_per_element = 4 if dtype == "float32" else 2
    B, n = batch_size, seq_len

    q_bytes = B * n * d_k * bytes_per_element
    k_bytes = B * n * d_k * bytes_per_element
    v_bytes = B * n * d_v * bytes_per_element
    attn_bytes = B * n * n * bytes_per_element
    output_bytes = B * n * d_v * bytes_per_element

    return q_bytes + k_bytes + v_bytes + attn_bytes + output_bytes
