"""
Flash Attention Concept -- From-scratch NumPy implementation.

Demonstrates the algorithmic core of flash attention: tiled attention with
online softmax that avoids materializing the full N x N attention matrix.
Standard attention requires O(N^2) memory for the score and probability
matrices; tiled attention achieves O(N) memory by processing Q/K/V in blocks
and incrementally updating output rows using the online softmax trick --
rescaling previous accumulations when a new block reveals a larger maximum.
The output is numerically equivalent to standard attention.
"""

import numpy as np
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Online softmax
# ---------------------------------------------------------------------------

def online_softmax(
    x: np.ndarray, chunk_size: int = 0
) -> Tuple[np.ndarray, float, float]:
    """
    Streaming softmax in a single pass over configurable chunks.

    Args:
        x: Input vector, shape (N,)
        chunk_size: Elements per chunk. 0 means process all at once.

    Returns:
        (softmax_output, m, ell) where m is the running max and ell is the
        running sum of exp(x - m).
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    if chunk_size <= 0:
        chunk_size = N

    m = -np.inf
    ell = 0.0

    for start in range(0, N, chunk_size):
        chunk = x[start:start + chunk_size]
        m_chunk = float(np.max(chunk))

        m_new = max(m, m_chunk)
        ell = ell * np.exp(m - m_new) + float(np.sum(np.exp(chunk - m_new)))
        m = m_new

    result = np.exp(x - m) / ell
    return result, m, ell


def online_softmax_2d(
    x: np.ndarray, chunk_size: int = 0
) -> np.ndarray:
    """
    Row-wise online softmax on a 2D matrix.

    Args:
        x: Input matrix, shape (N, M)
        chunk_size: Columns per chunk for each row. 0 means full row at once.

    Returns:
        Matrix of shape (N, M) with each row summing to 1.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i], _, _ = online_softmax(x[i], chunk_size)
    return out


# ---------------------------------------------------------------------------
# Standard attention (baseline)
# ---------------------------------------------------------------------------

def standard_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Full N^2 attention with explicit score materialization.

    Args:
        Q: Queries, shape (N, d)
        K: Keys, shape (N, d)
        V: Values, shape (N, d)

    Returns:
        (O, P, peak_bytes) where O is the output (N, d), P is the attention
        weight matrix (N, N), and peak_bytes counts the bytes allocated for
        the two largest intermediates (S and P).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    N, d = Q.shape
    scale = np.sqrt(d)

    # (N, d) @ (d, N) -> (N, N)
    S = Q @ K.T / scale
    peak_bytes = S.nbytes

    S_max = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    P = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    peak_bytes += P.nbytes

    # (N, N) @ (N, d) -> (N, d)
    O = P @ V
    return O, P, peak_bytes


# ---------------------------------------------------------------------------
# Tiled attention (flash attention algorithm)
# ---------------------------------------------------------------------------

def tiled_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size_q: int = 32,
    block_size_kv: int = 32,
    causal: bool = False,
) -> np.ndarray:
    """
    Flash attention: tiled computation with online softmax, O(N) memory.

    Processes Q in blocks of block_size_q rows and K/V in blocks of
    block_size_kv rows. Per-row running statistics (m, ell) allow exact
    softmax computation without materializing the full N x N score matrix.

    Args:
        Q: Queries, shape (N, d)
        K: Keys, shape (N, d)
        V: Values, shape (N, d)
        block_size_q: Number of query rows per tile
        block_size_kv: Number of key/value rows per tile
        causal: If True, apply causal masking (query i attends only to keys j <= i)

    Returns:
        Output O of shape (N, d), numerically equivalent to standard attention.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    N, d = Q.shape
    scale = np.sqrt(d)

    O = np.zeros((N, d), dtype=np.float64)
    m = np.full(N, -np.inf, dtype=np.float64)
    ell = np.zeros(N, dtype=np.float64)

    for j_start in range(0, N, block_size_kv):
        j_end = min(j_start + block_size_kv, N)
        K_j = K[j_start:j_end]  # (Bc, d)
        V_j = V[j_start:j_end]  # (Bc, d)

        for i_start in range(0, N, block_size_q):
            i_end = min(i_start + block_size_q, N)

            if causal and j_start >= i_end:
                continue

            Q_i = Q[i_start:i_end]  # (Br, d)

            # (Br, d) @ (d, Bc) -> (Br, Bc)
            S_ij = Q_i @ K_j.T / scale

            if causal:
                Br = i_end - i_start
                Bc = j_end - j_start
                row_idx = np.arange(i_start, i_end)[:, None]  # (Br, 1)
                col_idx = np.arange(j_start, j_end)[None, :]  # (1, Bc)
                S_ij = np.where(col_idx <= row_idx, S_ij, -np.inf)

            m_slice = m[i_start:i_end]       # (Br,)
            ell_slice = ell[i_start:i_end]   # (Br,)
            O_slice = O[i_start:i_end]       # (Br, d)

            m_block = np.max(S_ij, axis=-1)  # (Br,)

            # Rows where all scores are -inf (fully masked) contribute nothing
            valid_row = np.isfinite(m_block)
            m_block_safe = np.where(valid_row, m_block, -np.inf)

            m_new = np.maximum(m_slice, m_block_safe)

            alpha = np.exp(m_slice - m_new)            # (Br,)
            beta = np.where(valid_row, np.exp(m_block_safe - m_new), 0.0)
            P_block = np.where(
                valid_row[:, None],
                np.exp(S_ij - np.where(valid_row, m_block, 0.0)[:, None]),
                0.0,
            )  # (Br, Bc)
            ell_block = np.sum(P_block, axis=-1)       # (Br,)

            ell_new = ell_slice * alpha + ell_block * beta  # (Br,)

            safe_ell = np.where(ell_new == 0.0, 1.0, ell_new)

            # Rescale previous output and add new contribution
            O[i_start:i_end] = (
                O_slice * (alpha * ell_slice)[:, None]
                + (P_block * beta[:, None]) @ V_j  # (Br, Bc) @ (Bc, d) -> (Br, d)
            ) / safe_ell[:, None]

            m[i_start:i_end] = m_new
            ell[i_start:i_end] = ell_new

    return O


# ---------------------------------------------------------------------------
# Memory analysis
# ---------------------------------------------------------------------------

def memory_analysis(
    N: int, d: int, block_size: int = 32, dtype: str = "float32"
) -> Dict[str, object]:
    """
    Compare peak memory requirements for standard vs tiled attention.

    Args:
        N: Sequence length
        d: Head dimension
        block_size: Block size for tiled attention (used for both Br and Bc)
        dtype: 'float32' (4 bytes) or 'float16' (2 bytes)

    Returns:
        Dictionary with byte counts and human-readable sizes for both methods.
    """
    bpe = 4 if dtype == "float32" else 2

    std_S = N * N * bpe
    std_P = N * N * bpe
    std_O = N * d * bpe
    std_total = std_S + std_P + std_O

    Br = min(block_size, N)
    Bc = min(block_size, N)
    tile_S = Br * Bc * bpe
    tile_P = Br * Bc * bpe
    tile_m = N * 8  # float64 statistics
    tile_ell = N * 8
    tile_O = N * d * bpe
    tile_total = tile_S + tile_P + tile_m + tile_ell + tile_O

    def _fmt(b: int) -> str:
        if b < 1024:
            return f"{b} B"
        if b < 1024 ** 2:
            return f"{b / 1024:.1f} KB"
        return f"{b / 1024 ** 2:.2f} MB"

    return {
        "standard": {
            "S_bytes": std_S,
            "P_bytes": std_P,
            "O_bytes": std_O,
            "total_bytes": std_total,
            "total_human": _fmt(std_total),
        },
        "tiled": {
            "block_S_bytes": tile_S,
            "block_P_bytes": tile_P,
            "statistics_bytes": tile_m + tile_ell,
            "O_bytes": tile_O,
            "total_bytes": tile_total,
            "total_human": _fmt(tile_total),
        },
        "ratio": std_total / tile_total if tile_total > 0 else float("inf"),
        "N": N,
        "d": d,
        "block_size": block_size,
        "dtype": dtype,
    }


# ---------------------------------------------------------------------------
# Instrumented verification
# ---------------------------------------------------------------------------

def verify_no_full_materialization(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 32,
) -> Tuple[np.ndarray, int]:
    """
    Tiled attention instrumented to track the largest intermediate tensor.

    Runs the same algorithm as tiled_attention but records the element count
    of every block-level intermediate to prove no O(N^2) tensor is created.

    Args:
        Q: Queries, shape (N, d)
        K: Keys, shape (N, d)
        V: Values, shape (N, d)
        block_size: Block size for both Br and Bc

    Returns:
        (O, max_tensor_elements) where max_tensor_elements is the largest
        intermediate tensor's element count (should be O(block_size^2), not O(N^2)).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    N, d = Q.shape
    scale = np.sqrt(d)

    O = np.zeros((N, d), dtype=np.float64)
    m = np.full(N, -np.inf, dtype=np.float64)
    ell = np.zeros(N, dtype=np.float64)
    max_tensor_elements = 0

    for j_start in range(0, N, block_size):
        j_end = min(j_start + block_size, N)
        K_j = K[j_start:j_end]
        V_j = V[j_start:j_end]
        max_tensor_elements = max(max_tensor_elements, K_j.size, V_j.size)

        for i_start in range(0, N, block_size):
            i_end = min(i_start + block_size, N)
            Q_i = Q[i_start:i_end]
            max_tensor_elements = max(max_tensor_elements, Q_i.size)

            S_ij = Q_i @ K_j.T / scale
            max_tensor_elements = max(max_tensor_elements, S_ij.size)

            m_slice = m[i_start:i_end]
            ell_slice = ell[i_start:i_end]
            O_slice = O[i_start:i_end]

            m_block = np.max(S_ij, axis=-1)
            m_new = np.maximum(m_slice, m_block)

            alpha = np.exp(m_slice - m_new)
            beta = np.exp(m_block - m_new)
            P_block = np.exp(S_ij - m_block[:, None])
            max_tensor_elements = max(max_tensor_elements, P_block.size)

            ell_block = np.sum(P_block, axis=-1)
            ell_new = ell_slice * alpha + ell_block * beta

            safe_ell = np.where(ell_new == 0.0, 1.0, ell_new)

            PV = (P_block * beta[:, None]) @ V_j
            max_tensor_elements = max(max_tensor_elements, PV.size)

            O[i_start:i_end] = (
                O_slice * (alpha * ell_slice)[:, None] + PV
            ) / safe_ell[:, None]

            m[i_start:i_end] = m_new
            ell[i_start:i_end] = ell_new

    return O, max_tensor_elements
