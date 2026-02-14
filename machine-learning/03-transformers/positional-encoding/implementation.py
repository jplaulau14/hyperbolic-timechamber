"""
Positional Encoding -- From-scratch NumPy implementation.

Implements sinusoidal (fixed) and learned (trainable) absolute positional encodings
as described in "Attention Is All You Need" (Vaswani et al., 2017). Sinusoidal
encodings use geometrically spaced frequencies so that relative position offsets
correspond to linear transformations (rotations) of the encoding vectors. Learned
encodings are a trainable lookup table indexed by position (used in BERT, GPT-2).
Both are absolute encodings -- the precursors to RoPE.
"""

import numpy as np
from typing import Dict, Tuple


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Generate the sinusoidal positional encoding matrix.

    PE(pos, 2i)   = sin(pos * omega_i)
    PE(pos, 2i+1) = cos(pos * omega_i)
    where omega_i = exp(-2i * ln(10000) / d_model)

    Args:
        seq_len: Sequence length L
        d_model: Model dimension (must be even)

    Returns:
        PE matrix of shape (L, d_model)
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    # (d_model/2,)
    i = np.arange(d_model // 2, dtype=np.float64)
    omega = np.exp(-2.0 * i / d_model * np.log(10000.0))

    # (L, 1) * (1, d_model/2) -> (L, d_model/2)
    pos = np.arange(seq_len, dtype=np.float64)[:, np.newaxis]
    angles = pos * omega[np.newaxis, :]

    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


class SinusoidalPositionalEncoding:
    """Fixed sinusoidal positional encoding with precomputed PE matrix."""

    def __init__(self, max_seq_len: int, d_model: int):
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pe = sinusoidal_positional_encoding(max_seq_len, d_model)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.

        Args:
            X: Input tensor of shape (B, L, d_model) where L <= max_seq_len

        Returns:
            X + PE[:L, :] of shape (B, L, d_model)
        """
        L = X.shape[1]
        return X + self.pe[:L, :]

    def get_encoding(self, seq_len: int) -> np.ndarray:
        """
        Return the PE matrix for a given sequence length.

        Args:
            seq_len: Number of positions to return

        Returns:
            PE[:seq_len, :] of shape (seq_len, d_model)
        """
        return self.pe[:seq_len, :]


class LearnedPositionalEncoding:
    """Trainable positional embedding lookup table."""

    def __init__(self, max_seq_len: int, d_model: int):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = np.random.randn(max_seq_len, d_model).astype(np.float64) * 0.02
        self._cache: dict = {}
        self.grad_embedding: np.ndarray = np.zeros_like(self.embedding)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Add learned positional embeddings to input.

        Args:
            X: Input tensor of shape (B, L, d_model) where L <= max_seq_len

        Returns:
            X + embedding[:L, :] of shape (B, L, d_model)
        """
        L = X.shape[1]
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}"
            )
        self._cache = {"X": X, "L": L}
        return X + self.embedding[:L, :]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through learned positional encoding.

        Args:
            grad_output: Upstream gradient dL/dX' of shape (B, L, d_model)

        Returns:
            Gradient dL/dX of shape (B, L, d_model) (pass-through)
        """
        L = self._cache["L"]
        self.grad_embedding = np.zeros_like(self.embedding)
        # Sum upstream gradient across batch dimension
        self.grad_embedding[:L, :] = grad_output.sum(axis=0)
        return grad_output.copy()


def relative_position_matrix(
    pe: np.ndarray, offset: int
) -> Tuple[np.ndarray, float]:
    """
    Verify the linear transformation property of sinusoidal encodings.

    For each dimension pair (2i, 2i+1), the rotation matrix R_i(k) maps
    PE[pos] to PE[pos+k]. The full transformation M_k is block-diagonal
    with d_model/2 such 2x2 rotation blocks.

    Args:
        pe: Positional encoding matrix of shape (L, d_model)
        offset: Position offset k

    Returns:
        M_k: Block-diagonal rotation matrix of shape (d_model, d_model)
        max_error: Maximum reconstruction error across all valid positions
    """
    L, d_model = pe.shape
    d_half = d_model // 2

    M_k = np.zeros((d_model, d_model), dtype=np.float64)

    for i in range(d_half):
        omega_i = np.exp(-2.0 * i / d_model * np.log(10000.0))
        cos_val = np.cos(omega_i * offset)
        sin_val = np.sin(omega_i * offset)
        idx = 2 * i
        M_k[idx, idx] = cos_val
        M_k[idx, idx + 1] = sin_val
        M_k[idx + 1, idx] = -sin_val
        M_k[idx + 1, idx + 1] = cos_val

    max_error = 0.0
    for pos in range(L - offset):
        reconstructed = M_k @ pe[pos]
        error = np.linalg.norm(reconstructed - pe[pos + offset])
        max_error = max(max_error, error)

    return M_k, max_error


def dot_product_distance(pe: np.ndarray) -> np.ndarray:
    """
    Compute pairwise dot products of positional encoding vectors.

    For sinusoidal encodings, D[i,j] depends only on |i - j| (Toeplitz structure).

    Args:
        pe: Positional encoding matrix of shape (L, d_model)

    Returns:
        Dot product matrix of shape (L, L) where D[i,j] = PE[i] . PE[j]
    """
    return pe @ pe.T


def encoding_statistics(pe: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute statistics of the positional encoding matrix.

    Args:
        pe: Positional encoding matrix of shape (L, d_model)

    Returns:
        Dictionary with:
            position_norms: L2 norm per position, shape (L,)
            mean_per_dim: Mean across positions per dimension, shape (d_model,)
            var_per_dim: Variance across positions per dimension, shape (d_model,)
            min_val: Global minimum value
            max_val: Global maximum value
    """
    return {
        "position_norms": np.linalg.norm(pe, axis=1),
        "mean_per_dim": pe.mean(axis=0),
        "var_per_dim": pe.var(axis=0),
        "min_val": pe.min(),
        "max_val": pe.max(),
    }
