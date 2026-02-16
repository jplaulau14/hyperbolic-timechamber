"""
Transformer Block -- From-scratch NumPy implementation.

The fundamental repeated unit of every modern LLM. Wires together RMSNorm,
grouped-query attention with RoPE, and a SwiGLU FFN into the pre-norm
architecture used by Llama, Mistral, and all modern open-weight models.
This is an integration module that imports and reuses existing component
implementations.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from importlib import import_module

_norm_mod = import_module("02-neural-networks.normalization.implementation")
RMSNorm = _norm_mod.RMSNorm

_act_mod = import_module("02-neural-networks.activations.implementation")
_stable_sigmoid = _act_mod._stable_sigmoid

_gqa_mod = import_module("03-transformers.grouped-query-attention.implementation")
softmax = _gqa_mod.softmax
softmax_backward = _gqa_mod.softmax_backward
create_causal_mask = _gqa_mod.create_causal_mask
repeat_kv = _gqa_mod.repeat_kv
reduce_kv_grad = _gqa_mod.reduce_kv_grad

_rope_mod = import_module("03-transformers.rope.implementation")
RoPE = _rope_mod.RoPE
apply_rope = _rope_mod.apply_rope
rotate_half = _rope_mod.rotate_half
rotate_half_backward = _rope_mod.rotate_half_backward


def _xavier(shape: Tuple[int, ...]) -> np.ndarray:
    std = np.sqrt(2.0 / (shape[0] + shape[1]))
    return np.random.randn(*shape).astype(np.float64) * std


class SwiGLUFFN:
    """SwiGLU feed-forward network: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down."""

    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff

        self.W_gate = _xavier((d_model, d_ff))
        self.W_up = _xavier((d_model, d_ff))
        self.W_down = _xavier((d_ff, d_model))

        self._cache: Optional[dict] = None
        self.grad_W_gate: Optional[np.ndarray] = None
        self.grad_W_up: Optional[np.ndarray] = None
        self.grad_W_down: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (B, L, d_model)

        Returns:
            Output tensor, shape (B, L, d_model)
        """
        gate = x @ self.W_gate
        up = x @ self.W_up
        sig_gate = _stable_sigmoid(gate)
        silu_gate = gate * sig_gate
        hidden = silu_gate * up
        output = hidden @ self.W_down

        self._cache = {
            "x": x, "gate": gate, "up": up,
            "sig_gate": sig_gate, "silu_gate": silu_gate, "hidden": hidden,
        }
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through SwiGLU.

        Args:
            grad_output: Upstream gradient, shape (B, L, d_model)

        Returns:
            Gradient w.r.t. input x, shape (B, L, d_model)
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward().")

        x = self._cache["x"]
        gate = self._cache["gate"]
        up = self._cache["up"]
        sig_gate = self._cache["sig_gate"]
        silu_gate = self._cache["silu_gate"]
        hidden = self._cache["hidden"]

        # (B, L, d_model) @ (d_model, d_ff) -> (B, L, d_ff)
        grad_hidden = grad_output @ self.W_down.T
        self.grad_W_down = np.einsum("bld,blm->dm", hidden, grad_output)

        grad_silu_gate = grad_hidden * up
        grad_up = grad_hidden * silu_gate

        # SiLU derivative: sigma(z) * (1 + z * (1 - sigma(z)))
        silu_deriv = sig_gate * (1.0 + gate * (1.0 - sig_gate))
        grad_gate = grad_silu_gate * silu_deriv

        self.grad_W_gate = np.einsum("blm,bld->md", x, grad_gate)
        self.grad_W_up = np.einsum("blm,bld->md", x, grad_up)

        grad_x = grad_gate @ self.W_gate.T + grad_up @ self.W_up.T
        return grad_x


class TransformerBlock:
    """Pre-norm decoder transformer block with GQA, RoPE, and SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
    ):
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
        self.group_size = num_heads // num_kv_heads
        self.d_ff = d_ff

        if self.d_k % 2 != 0:
            raise ValueError(
                f"d_k ({self.d_k}) must be even for RoPE"
            )

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.rope = RoPE(self.d_k, max_seq_len, rope_theta)

        self.W_Q = _xavier((d_model, d_model))
        self.W_K = _xavier((d_model, num_kv_heads * self.d_k))
        self.W_V = _xavier((d_model, num_kv_heads * self.d_k))
        self.W_O = _xavier((d_model, d_model))

        self.ffn = SwiGLUFFN(d_model, d_ff)

        self._cache: Optional[dict] = None

        self.grad_W_Q: Optional[np.ndarray] = None
        self.grad_W_K: Optional[np.ndarray] = None
        self.grad_W_V: Optional[np.ndarray] = None
        self.grad_W_O: Optional[np.ndarray] = None

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Full pre-norm decoder block forward pass.

        Args:
            x: Input tensor, shape (B, L, d_model)
            mask: Optional additive attention mask, broadcastable to (B, h, L, L)
            positions: Optional (L,) integer positions for RoPE

        Returns:
            Output tensor, shape (B, L, d_model)
        """
        x = np.asarray(x, dtype=np.float64)
        B, L, _ = x.shape

        # Step 1: Pre-norm for attention
        x_norm = self.norm1.forward(x)

        # Step 2: Q/K/V projections
        # (B, L, d_model) @ (d_model, h*d_k) -> (B, L, h*d_k)
        Q = x_norm @ self.W_Q
        # (B, L, d_model) @ (d_model, h_kv*d_k) -> (B, L, h_kv*d_k)
        K = x_norm @ self.W_K
        V = x_norm @ self.W_V

        # Step 3: Reshape to head layout
        # (B, L, h*d_k) -> (B, h, L, d_k)
        Q = Q.reshape(B, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        # (B, L, h_kv*d_k) -> (B, h_kv, L, d_k)
        K = K.reshape(B, L, self.num_kv_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, L, self.num_kv_heads, self.d_k).transpose(0, 2, 1, 3)

        # Step 4: Apply RoPE to Q and K
        Q_rot, K_rot = self.rope.forward(Q, K, positions)

        # Step 5: Expand KV heads
        K_exp = repeat_kv(K_rot, self.group_size)
        V_exp = repeat_kv(V, self.group_size)

        # Step 6: Attention scores with causal mask
        # (B, h, L, d_k) @ (B, h, d_k, L) -> (B, h, L, L)
        scores = Q_rot @ K_exp.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is None:
            mask = create_causal_mask(L)
        scores = scores + mask

        A = softmax(scores, axis=-1)

        # (B, h, L, L) @ (B, h, L, d_k) -> (B, h, L, d_k)
        attn_output = A @ V_exp

        # Step 7: Merge heads and output projection
        # (B, h, L, d_k) -> (B, L, h, d_k) -> (B, L, d_model)
        concat = attn_output.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        attn_out = concat @ self.W_O

        # Step 8: First residual connection
        h = x + attn_out

        # Step 9: Pre-norm for FFN
        h_norm = self.norm2.forward(h)

        # Step 10: SwiGLU FFN
        ffn_out = self.ffn.forward(h_norm)

        # Step 11: Second residual connection
        output = h + ffn_out

        self._cache = {
            "x": x, "x_norm": x_norm,
            "Q": Q, "K": K, "V": V,
            "Q_rot": Q_rot, "K_rot": K_rot,
            "K_exp": K_exp, "V_exp": V_exp,
            "A": A, "attn_output": attn_output, "concat": concat,
            "attn_out": attn_out, "h": h, "h_norm": h_norm,
            "mask": mask,
        }
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Full backward pass through the transformer block.

        Args:
            grad_output: Upstream gradient, shape (B, L, d_model)

        Returns:
            Gradient w.r.t. input x, shape (B, L, d_model)
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward().")

        grad_output = np.asarray(grad_output, dtype=np.float64)
        c = self._cache
        B, L, _ = c["x"].shape

        # Residual 2: output = h + ffn_out
        grad_h_res2 = grad_output.copy()
        grad_ffn_out = grad_output.copy()

        # Backward through SwiGLU FFN
        grad_h_norm = self.ffn.backward(grad_ffn_out)

        # Backward through RMSNorm_2
        grad_h_from_ffn = self.norm2.backward(grad_h_norm)

        # Accumulate gradients into d_h
        grad_h = grad_h_res2 + grad_h_from_ffn

        # Residual 1: h = x + attn_out
        grad_x_res1 = grad_h.copy()
        grad_attn_out = grad_h.copy()

        # Backward through output projection: attn_out = concat @ W_O
        grad_concat = grad_attn_out @ self.W_O.T
        self.grad_W_O = np.einsum("blm,bln->mn", c["concat"], grad_attn_out)

        # Backward through head merge: (B, L, d_model) -> (B, h, L, d_k)
        grad_attn_output = grad_concat.reshape(
            B, L, self.num_heads, self.d_k
        ).transpose(0, 2, 1, 3)

        # Backward through value weighting: attn_output = A @ V_exp
        grad_A = grad_attn_output @ c["V_exp"].transpose(0, 1, 3, 2)
        grad_V_exp = c["A"].transpose(0, 1, 3, 2) @ grad_attn_output

        # Backward through softmax
        grad_scores = softmax_backward(grad_A, c["A"])

        # Backward through scaled dot-product
        scale = np.sqrt(self.d_k)
        grad_raw = grad_scores / scale
        # (B, h, L, L) @ (B, h, L, d_k) -> (B, h, L, d_k)
        grad_Q_rot = grad_raw @ c["K_exp"]
        # (B, h, L, L)^T @ (B, h, L, d_k) -> (B, h, L, d_k)
        grad_K_exp = grad_raw.transpose(0, 1, 3, 2) @ c["Q_rot"]

        # Backward through KV repeat-interleave
        grad_K_rot = reduce_kv_grad(grad_K_exp, self.num_kv_heads, self.group_size)
        grad_V = reduce_kv_grad(grad_V_exp, self.num_kv_heads, self.group_size)

        # Backward through RoPE
        grad_Q, grad_K = self.rope.backward(grad_Q_rot, grad_K_rot)

        # Backward through head split: reverse transpose + reshape
        grad_Q_flat = grad_Q.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        grad_K_flat = grad_K.transpose(0, 2, 1, 3).reshape(
            B, L, self.num_kv_heads * self.d_k
        )
        grad_V_flat = grad_V.transpose(0, 2, 1, 3).reshape(
            B, L, self.num_kv_heads * self.d_k
        )

        # Backward through Q/K/V projections
        x_norm = c["x_norm"]
        self.grad_W_Q = np.einsum("blm,bld->md", x_norm, grad_Q_flat)
        self.grad_W_K = np.einsum("blm,bld->md", x_norm, grad_K_flat)
        self.grad_W_V = np.einsum("blm,bld->md", x_norm, grad_V_flat)

        grad_x_norm = (
            grad_Q_flat @ self.W_Q.T
            + grad_K_flat @ self.W_K.T
            + grad_V_flat @ self.W_V.T
        )

        # Backward through RMSNorm_1
        grad_x_from_attn = self.norm1.backward(grad_x_norm)

        # Accumulate gradients from both paths
        grad_x = grad_x_res1 + grad_x_from_attn

        return grad_x


def count_parameters(
    d_model: int, num_heads: int, num_kv_heads: int, d_ff: int
) -> Dict[str, object]:
    """
    Parameter count breakdown for a single transformer block.

    Returns:
        Dict with per-component counts, percentages, and total.
    """
    d_k = d_model // num_heads

    wq = d_model * d_model
    wk = d_model * (num_kv_heads * d_k)
    wv = d_model * (num_kv_heads * d_k)
    wo = d_model * d_model
    attn_total = wq + wk + wv + wo

    w_gate = d_model * d_ff
    w_up = d_model * d_ff
    w_down = d_ff * d_model
    ffn_total = w_gate + w_up + w_down

    norm_total = 2 * d_model

    total = attn_total + ffn_total + norm_total

    return {
        "W_Q": wq, "W_K": wk, "W_V": wv, "W_O": wo,
        "attn_total": attn_total,
        "W_gate": w_gate, "W_up": w_up, "W_down": w_down,
        "ffn_total": ffn_total,
        "norm_total": norm_total,
        "total": total,
        "attn_pct": 100.0 * attn_total / total,
        "ffn_pct": 100.0 * ffn_total / total,
        "norm_pct": 100.0 * norm_total / total,
    }


def count_flops(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    d_ff: int,
) -> Dict[str, int]:
    """
    Forward-pass FLOPs breakdown for a single transformer block.

    Counts multiply-accumulate as 2 FLOPs per element.
    """
    B, L, h = batch_size, seq_len, num_heads
    d_k = d_model // h

    proj_q = 2 * B * L * d_model * d_model
    proj_k = 2 * B * L * d_model * (num_kv_heads * d_k)
    proj_v = 2 * B * L * d_model * (num_kv_heads * d_k)
    proj_o = 2 * B * L * d_model * d_model
    attn_proj = proj_q + proj_k + proj_v + proj_o

    attn_qk = 2 * B * h * L * L * d_k
    attn_softmax = 5 * B * h * L * L
    attn_av = 2 * B * h * L * L * d_k
    attn_core = attn_qk + attn_softmax + attn_av

    rope_flops = 6 * B * h * L * d_k

    ffn_gate = 2 * B * L * d_model * d_ff
    ffn_up = 2 * B * L * d_model * d_ff
    ffn_down = 2 * B * L * d_ff * d_model
    ffn_total = ffn_gate + ffn_up + ffn_down

    norm_flops = 4 * B * L * d_model

    total = attn_proj + attn_core + rope_flops + ffn_total + norm_flops

    return {
        "attn_proj": attn_proj,
        "attn_core": attn_core,
        "rope": rope_flops,
        "ffn_total": ffn_total,
        "norm": norm_flops,
        "total": total,
        "per_token": total // (B * L) if B * L > 0 else 0,
    }


def memory_footprint(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    d_ff: int,
    bytes_per_param: int = 8,
) -> Dict[str, object]:
    """
    Memory analysis for a single transformer block (parameters + activations).

    Args:
        bytes_per_param: Bytes per element (8 for float64, 4 for float32, 2 for float16)

    Returns:
        Dict with parameter bytes, activation bytes, peak tensor, and total.
    """
    B, L = batch_size, seq_len
    d_k = d_model // num_heads
    bpe = bytes_per_param

    params = count_parameters(d_model, num_heads, num_kv_heads, d_ff)
    param_bytes = params["total"] * bpe

    activations = {
        "x_norm": B * L * d_model * bpe,
        "Q": B * num_heads * L * d_k * bpe,
        "K": B * num_kv_heads * L * d_k * bpe,
        "V": B * num_kv_heads * L * d_k * bpe,
        "Q_rot": B * num_heads * L * d_k * bpe,
        "K_rot": B * num_kv_heads * L * d_k * bpe,
        "K_exp": B * num_heads * L * d_k * bpe,
        "V_exp": B * num_heads * L * d_k * bpe,
        "attn_scores": B * num_heads * L * L * bpe,
        "attn_weights": B * num_heads * L * L * bpe,
        "attn_output": B * num_heads * L * d_k * bpe,
        "concat": B * L * d_model * bpe,
        "attn_out": B * L * d_model * bpe,
        "h": B * L * d_model * bpe,
        "h_norm": B * L * d_model * bpe,
        "ffn_gate": B * L * d_ff * bpe,
        "ffn_up": B * L * d_ff * bpe,
        "ffn_hidden": B * L * d_ff * bpe,
        "ffn_out": B * L * d_model * bpe,
    }

    activation_bytes = sum(activations.values())
    largest_tensor_name = max(activations, key=activations.get)
    largest_tensor_bytes = activations[largest_tensor_name]

    return {
        "param_bytes": param_bytes,
        "activation_bytes": activation_bytes,
        "activation_breakdown": activations,
        "largest_tensor": largest_tensor_name,
        "largest_tensor_bytes": largest_tensor_bytes,
        "total_bytes": param_bytes + activation_bytes,
    }
