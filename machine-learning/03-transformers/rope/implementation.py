"""
Rotary Position Embeddings (RoPE) -- From-scratch NumPy implementation.

Encodes relative position directly into attention dot products through rotation.
Each pair of consecutive dimensions (2i, 2i+1) in the head dimension defines a 2D
subspace, and RoPE applies a position-dependent rotation in each subspace. The key
property is that the dot product between rotated Q at position m and rotated K at
position n depends only on the relative distance (m - n), making RoPE a true relative
position encoding despite using absolute position information in the rotation itself.
Used by LLaMA, Mistral, Qwen, and virtually all modern open-weight LLMs.
"""

import numpy as np
from typing import Optional, Tuple


def precompute_freqs(
    d: int, max_seq_len: int, theta_base: float = 10000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute cos and sin caches for RoPE.

    Args:
        d: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        theta_base: Base frequency (10000 for LLaMA 1/2, 500000 for LLaMA 3)

    Returns:
        (cos_cache, sin_cache) each of shape (max_seq_len, d // 2)
    """
    if d % 2 != 0:
        raise ValueError(f"d must be even, got {d}")

    # (d/2,) -- log-space for numerical stability
    i = np.arange(d // 2, dtype=np.float64)
    inv_freq = np.exp(-2.0 * i / d * np.log(theta_base))

    # (max_seq_len, d/2) via outer product
    positions = np.arange(max_seq_len, dtype=np.float64)
    angles = positions[:, np.newaxis] * inv_freq[np.newaxis, :]

    return np.cos(angles), np.sin(angles)


def rotate_half(x: np.ndarray) -> np.ndarray:
    """
    Swap and negate dimension pairs: [-x1, x0, -x3, x2, ...].

    Args:
        x: Tensor of shape (..., d) where d is even

    Returns:
        Rotated tensor of same shape
    """
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2)
    x = np.stack([-x[..., 1], x[..., 0]], axis=-1)
    return x.reshape(*x.shape[:-2], d)


def rotate_half_backward(x: np.ndarray) -> np.ndarray:
    """
    Inverse of rotate_half: [x1, -x0, x3, -x2, ...].

    For the backward pass, the transpose rotation swaps and negates the second
    element in each pair (opposite of forward rotate_half).

    Args:
        x: Tensor of shape (..., d) where d is even

    Returns:
        Inverse-rotated tensor of same shape
    """
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2)
    x = np.stack([x[..., 1], -x[..., 0]], axis=-1)
    return x.reshape(*x.shape[:-2], d)


def _broadcast_cos_sin(
    cos_cache: np.ndarray,
    sin_cache: np.ndarray,
    seq_len: int,
    d: int,
    positions: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice, repeat-interleave, and reshape cos/sin for broadcasting with (B, H, L, d).

    Args:
        cos_cache: Shape (max_seq_len, d // 2)
        sin_cache: Shape (max_seq_len, d // 2)
        seq_len: Current sequence length L
        d: Head dimension
        positions: Optional (L,) integer positions for non-contiguous indexing

    Returns:
        (cos, sin) each of shape (1, 1, L, d)
    """
    if positions is not None:
        cos = cos_cache[positions]
        sin = sin_cache[positions]
    else:
        cos = cos_cache[:seq_len]
        sin = sin_cache[:seq_len]

    # (L, d/2) -> (L, d) by repeating each freq for both dims in pair
    cos = np.repeat(cos, 2, axis=-1)
    sin = np.repeat(sin, 2, axis=-1)

    # (1, 1, L, d) for broadcasting over batch and heads
    return cos[np.newaxis, np.newaxis, :, :], sin[np.newaxis, np.newaxis, :, :]


def apply_rope(
    x: np.ndarray,
    cos_cache: np.ndarray,
    sin_cache: np.ndarray,
    positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply RoPE using the efficient element-wise form: x' = x * cos + rotate_half(x) * sin.

    Args:
        x: Input tensor of shape (B, H, L, d)
        cos_cache: Precomputed cosines, shape (max_seq_len, d // 2)
        sin_cache: Precomputed sines, shape (max_seq_len, d // 2)
        positions: Optional (L,) integer array for non-contiguous positions (KV cache)

    Returns:
        Rotated tensor of same shape as x
    """
    seq_len = x.shape[2]
    d = x.shape[-1]
    cos, sin = _broadcast_cos_sin(cos_cache, sin_cache, seq_len, d, positions)
    return x * cos + rotate_half(x) * sin


def apply_rope_complex(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Apply RoPE using complex number multiplication.

    Pairs of real dims are viewed as complex numbers, multiplied by e^{i*m*theta},
    then converted back to real. Serves as a correctness check against rotate_half.

    Args:
        x: Input tensor of shape (..., d) where d is even
        freqs: Complex exponentials e^{i*m*theta} broadcastable to (..., d // 2)

    Returns:
        Rotated tensor of same shape as x
    """
    d = x.shape[-1]
    x_pairs = x.reshape(*x.shape[:-1], d // 2, 2)
    x_complex = x_pairs[..., 0] + 1j * x_pairs[..., 1]
    x_rotated = x_complex * freqs
    result = np.stack([x_rotated.real, x_rotated.imag], axis=-1)
    return result.reshape(*x.shape)


class RoPE:
    """Rotary Position Embeddings with precomputed frequency caches."""

    def __init__(self, d_head: int, max_seq_len: int, theta_base: float = 10000.0):
        """
        Args:
            d_head: Head dimension (must be even)
            max_seq_len: Maximum sequence length for cache
            theta_base: Base frequency
        """
        if d_head % 2 != 0:
            raise ValueError(f"d_head must be even, got {d_head}")

        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.theta_base = theta_base

        i = np.arange(d_head // 2, dtype=np.float64)
        self.inv_freq = np.exp(-2.0 * i / d_head * np.log(theta_base))

        self.cos_cache, self.sin_cache = precompute_freqs(
            d_head, max_seq_len, theta_base
        )

        self._cache: Optional[dict] = None

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        positions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply RoPE to both Q and K.

        Args:
            q: Query tensor, shape (B, num_heads, L, d_head)
            k: Key tensor, shape (B, num_kv_heads, L, d_head)
            positions: Optional (L,) integer positions for KV cache scenarios

        Returns:
            (q_rotated, k_rotated) with same shapes as inputs
        """
        seq_len = q.shape[2]
        d = q.shape[-1]

        cos, sin = _broadcast_cos_sin(
            self.cos_cache, self.sin_cache, seq_len, d, positions
        )

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

        self._cache = {
            "q": q, "k": k, "cos": cos, "sin": sin, "positions": positions,
        }

        return q_rot, k_rot

    def backward(
        self, grad_q_rot: np.ndarray, grad_k_rot: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass: apply inverse rotation R(-m) to upstream gradients.

        The inverse rotation negates the sine term (since R(-m) = R(m)^T for
        orthogonal rotation matrices).

        Args:
            grad_q_rot: Gradient w.r.t. rotated Q, shape (B, num_heads, L, d_head)
            grad_k_rot: Gradient w.r.t. rotated K, shape (B, num_kv_heads, L, d_head)

        Returns:
            (grad_q, grad_k) gradients w.r.t. original Q and K
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")

        cos = self._cache["cos"]
        sin = self._cache["sin"]

        grad_q = grad_q_rot * cos + rotate_half_backward(grad_q_rot) * sin
        grad_k = grad_k_rot * cos + rotate_half_backward(grad_k_rot) * sin

        return grad_q, grad_k


def verify_relative_position_property(
    q: np.ndarray,
    k: np.ndarray,
    rope: RoPE,
    positions_m: np.ndarray,
    positions_n: np.ndarray,
) -> np.ndarray:
    """
    Empirically verify that RoPE(q,m)^T RoPE(k,n) depends only on (m-n).

    Rotates q and k at given positions, then shifts both positions by several
    offsets delta and verifies the dot products remain the same.

    Args:
        q: Query vector, shape (1, 1, 1, d)
        k: Key vector, shape (1, 1, 1, d)
        rope: RoPE instance
        positions_m: (1,) integer array for q position
        positions_n: (1,) integer array for k position

    Returns:
        Array of absolute differences between base dot product and shifted dot products
    """
    q_rot_base, _ = rope.forward(q, q, positions_m)
    _, k_rot_base = rope.forward(k, k, positions_n)
    base_dot = np.sum(q_rot_base * k_rot_base)

    deltas = np.array([1, 10, 50, 100, 500])
    diffs = []
    for delta in deltas:
        shifted_m = positions_m + delta
        shifted_n = positions_n + delta
        if shifted_m[0] >= rope.max_seq_len or shifted_n[0] >= rope.max_seq_len:
            continue
        q_rot, _ = rope.forward(q, q, shifted_m)
        _, k_rot = rope.forward(k, k, shifted_n)
        shifted_dot = np.sum(q_rot * k_rot)
        diffs.append(abs(base_dot - shifted_dot))

    return np.array(diffs)


def rotation_is_orthogonal(
    cos_cache: np.ndarray, sin_cache: np.ndarray, pos: int
) -> Tuple[np.ndarray, float, float]:
    """
    Construct the full block-diagonal rotation matrix at a given position and verify orthogonality.

    Args:
        cos_cache: Shape (max_seq_len, d // 2)
        sin_cache: Shape (max_seq_len, d // 2)
        pos: Position index

    Returns:
        (R, frob_error, det) where R is the rotation matrix (d, d),
        frob_error is ||R R^T - I||_F, and det is det(R)
    """
    d_half = cos_cache.shape[1]
    d = d_half * 2
    R = np.zeros((d, d), dtype=np.float64)

    cos_vals = cos_cache[pos]
    sin_vals = sin_cache[pos]

    for i in range(d_half):
        idx = 2 * i
        R[idx, idx] = cos_vals[i]
        R[idx, idx + 1] = -sin_vals[i]
        R[idx + 1, idx] = sin_vals[i]
        R[idx + 1, idx + 1] = cos_vals[i]

    frob_error = np.linalg.norm(R @ R.T - np.eye(d))
    det = np.linalg.det(R)

    return R, frob_error, det


def compare_with_sinusoidal(d: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare RoPE frequencies with sinusoidal PE frequencies.

    Both use theta_i = 10000^{-2i/d}. Returns both frequency arrays for comparison.

    Args:
        d: Dimension (must be even)
        seq_len: Sequence length (unused but kept for API symmetry)

    Returns:
        (rope_freqs, sinusoidal_freqs) each of shape (d // 2,)
    """
    i = np.arange(d // 2, dtype=np.float64)

    rope_freqs = np.exp(-2.0 * i / d * np.log(10000.0))
    sinusoidal_freqs = np.exp(-2.0 * i / d * np.log(10000.0))

    return rope_freqs, sinusoidal_freqs
