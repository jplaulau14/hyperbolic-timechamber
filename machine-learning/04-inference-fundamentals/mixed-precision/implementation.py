"""
Mixed Precision -- From-scratch NumPy implementation.

Explores IEEE 754 floating point formats (FP32, FP16, BF16), their bit layouts,
range/precision tradeoffs, and mixed-precision computation patterns used in modern
inference. Since memory bandwidth is the primary bottleneck for transformer inference,
halving precision from FP32 to FP16/BF16 roughly doubles throughput. This module
implements precision format utilities, range/precision analysis, mixed-precision
forward passes simulating tensor core behavior, dynamic loss scaling for FP16
training, gradient analysis, and memory calculations.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Precision format utilities
# ---------------------------------------------------------------------------

_FORMAT_INFO = {
    np.float32: {
        "sign_bits": 1,
        "exponent_bits": 8,
        "mantissa_bits": 23,
        "bias": 127,
        "total_bits": 32,
    },
    np.float16: {
        "sign_bits": 1,
        "exponent_bits": 5,
        "mantissa_bits": 10,
        "bias": 15,
        "total_bits": 16,
    },
    "bfloat16": {
        "sign_bits": 1,
        "exponent_bits": 8,
        "mantissa_bits": 7,
        "bias": 127,
        "total_bits": 16,
    },
}


def float_info(dtype) -> Dict[str, object]:
    """
    Return format information for a floating point dtype.

    Args:
        dtype: np.float32, np.float16, or the string "bfloat16"

    Returns:
        Dict with sign_bits, exponent_bits, mantissa_bits, bias,
        max_value, min_positive, precision_decimal_digits.
    """
    if dtype == "bfloat16":
        info = dict(_FORMAT_INFO["bfloat16"])
        info["max_value"] = np.finfo(np.float32).max
        info["min_positive"] = float(np.finfo(np.float32).tiny)
        info["precision_decimal_digits"] = np.log10(2 ** 7)
        return info

    dtype = np.dtype(dtype)
    finfo = np.finfo(dtype)

    if dtype == np.float32:
        base = dict(_FORMAT_INFO[np.float32])
    elif dtype == np.float16:
        base = dict(_FORMAT_INFO[np.float16])
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    base["max_value"] = float(finfo.max)
    base["min_positive"] = float(finfo.tiny)
    base["precision_decimal_digits"] = np.log10(2 ** base["mantissa_bits"])
    return base


def visualize_bits(value: float, dtype=np.float32) -> str:
    """
    Return a string showing sign | exponent | mantissa bit breakdown.

    Args:
        value: Scalar float value
        dtype: np.float32 or np.float16

    Returns:
        String like "0 | 01111111 | 00000000000000000000000" for 1.0 in FP32
    """
    dtype = np.dtype(dtype)

    if dtype == np.float32:
        bits = np.float32(value).view(np.uint32)
        sign = format((bits >> 31) & 1, "01b")
        exponent = format((bits >> 23) & 0xFF, "08b")
        mantissa = format(bits & 0x7FFFFF, "023b")
    elif dtype == np.float16:
        bits = np.float16(value).view(np.uint16)
        sign = format((bits >> 15) & 1, "01b")
        exponent = format((bits >> 10) & 0x1F, "05b")
        mantissa = format(bits & 0x3FF, "010b")
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return f"{sign} | {exponent} | {mantissa}"


def extract_fp32_parts(value: float) -> Tuple[int, int, int]:
    """Extract (sign, exponent, mantissa) fields from an FP32 value."""
    bits = np.float32(value).view(np.uint32)
    sign = int((bits >> 31) & 1)
    exponent = int((bits >> 23) & 0xFF)
    mantissa = int(bits & 0x7FFFFF)
    return sign, exponent, mantissa


def extract_fp16_parts(value: float) -> Tuple[int, int, int]:
    """Extract (sign, exponent, mantissa) fields from an FP16 value."""
    bits = np.float16(value).view(np.uint16)
    sign = int((bits >> 15) & 1)
    exponent = int((bits >> 10) & 0x1F)
    mantissa = int(bits & 0x3FF)
    return sign, exponent, mantissa


def bf16_simulate(x: np.ndarray) -> np.ndarray:
    """
    Simulate BF16 by truncating the lower 16 bits of FP32 mantissa.

    NumPy lacks native BF16 support. BF16 keeps the FP32 sign + exponent
    (8 bits) but reduces the mantissa from 23 to 7 bits.

    Args:
        x: Input array (any dtype, will be cast to float32)

    Returns:
        Float32 array with BF16 precision (lower 16 mantissa bits zeroed)
    """
    x = np.asarray(x, dtype=np.float32)
    x_int = x.view(np.uint32)
    x_int = x_int & np.uint32(0xFFFF0000)
    return x_int.view(np.float32)


# ---------------------------------------------------------------------------
# Range and precision analysis
# ---------------------------------------------------------------------------

def demonstrate_overflow(dtype=np.float16) -> Dict[str, object]:
    """
    Show values that overflow to inf in the given dtype.

    Returns:
        Dict with test_values and their representations in the target dtype.
    """
    test_values = [65000.0, 65500.0, 65504.0, 65505.0, 65536.0, 100000.0]
    results = {}
    for v in test_values:
        if dtype == np.float16:
            converted = np.float16(v)
        elif dtype == np.float32:
            converted = np.float32(v)
        else:
            converted = np.float16(v)
        results[v] = {
            "converted": float(converted),
            "is_inf": bool(np.isinf(converted)),
        }
    return {"dtype": str(dtype), "values": results}


def demonstrate_underflow(dtype=np.float16) -> Dict[str, object]:
    """
    Show small values that underflow to zero in the given dtype.

    Returns:
        Dict with test_values and their representations in the target dtype.
    """
    test_values = [1e-4, 1e-5, 6.1e-5, 5.96e-8, 1e-8, 1e-10]
    results = {}
    for v in test_values:
        converted = np.array(v, dtype=np.float32).astype(dtype)
        results[v] = {
            "converted": float(converted),
            "is_zero": float(converted) == 0.0,
        }
    return {"dtype": str(dtype), "values": results}


def precision_loss(x: np.ndarray, dtype=np.float16) -> np.ndarray:
    """
    Compute relative error when converting FP32 values to lower precision.

    Args:
        x: FP32 input array
        dtype: Target precision (np.float16 or "bfloat16")

    Returns:
        Array of relative errors |x_fp32 - x_low| / |x_fp32|
    """
    x = np.asarray(x, dtype=np.float32)

    if dtype == "bfloat16":
        x_low = bf16_simulate(x)
    else:
        x_low = x.astype(dtype).astype(np.float32)

    abs_x = np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(abs_x > 0, np.abs(x - x_low) / abs_x, 0.0)
    return rel_err


def consecutive_value_gap(value: float, dtype=np.float16) -> float:
    """
    Compute the gap to the next representable number above value.

    Args:
        value: The floating point value to check
        dtype: np.float16, np.float32, or "bfloat16"

    Returns:
        The gap (next_representable - value)
    """
    if dtype == "bfloat16":
        v = bf16_simulate(np.array(np.abs(value), dtype=np.float32))
        v_scalar = float(v)
        # Increment the BF16 mantissa by 1 ULP (bit 16 in the uint32 view)
        bits = np.float32(v_scalar).view(np.uint32)
        next_bits = np.uint32(int(bits) + (1 << 16))
        next_val = next_bits.view(np.float32)
        return float(next_val) - v_scalar

    v = np.abs(np.array(value, dtype=np.float32).astype(dtype))
    v_f = float(v)

    if dtype == np.float16:
        bits = np.float16(v_f).view(np.uint16)
        next_bits = np.uint16(int(bits) + 1)
        next_val = next_bits.view(np.float16)
    elif dtype == np.float32:
        bits = np.float32(v_f).view(np.uint32)
        next_bits = np.uint32(int(bits) + 1)
        next_val = next_bits.view(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return float(next_val) - v_f


# ---------------------------------------------------------------------------
# Mixed precision forward pass
# ---------------------------------------------------------------------------

def mixed_precision_linear(
    x: np.ndarray,
    W: np.ndarray,
    b: Optional[np.ndarray] = None,
    compute_dtype: np.dtype = np.float16,
    accumulate_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Linear layer with configurable compute and accumulation precision.

    Casts inputs and weights to compute_dtype, performs matmul with
    accumulation in accumulate_dtype, then casts output back to compute_dtype.

    Args:
        x: Input activations, shape (..., in_features)
        W: Weight matrix, shape (in_features, out_features)
        b: Optional bias, shape (out_features,)
        compute_dtype: Precision for inputs/weights
        accumulate_dtype: Precision for accumulation

    Returns:
        Output in compute_dtype, shape (..., out_features)
    """
    x_low = x.astype(compute_dtype)
    W_low = W.astype(compute_dtype)

    result = mixed_precision_matmul(x_low, W_low, compute_dtype, accumulate_dtype)

    if b is not None:
        b_acc = b.astype(accumulate_dtype)
        result = result.astype(accumulate_dtype) + b_acc
        result = result.astype(compute_dtype)

    return result


def mixed_precision_matmul(
    A: np.ndarray,
    B: np.ndarray,
    compute_dtype: np.dtype = np.float16,
    accumulate_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Matrix multiply simulating tensor core behavior: compute in low precision,
    accumulate dot products in high precision.

    Element-wise products are computed in compute_dtype, then cast to
    accumulate_dtype for summation. When accumulate_dtype == compute_dtype,
    accumulation happens entirely in low precision, demonstrating precision loss.

    Args:
        A: Left matrix, shape (..., M, K)
        B: Right matrix, shape (..., K, N)
        compute_dtype: Precision for element-wise multiplications
        accumulate_dtype: Precision for summation (accumulator)

    Returns:
        Result in compute_dtype, shape (..., M, N)
    """
    A_low = A.astype(compute_dtype)
    B_low = B.astype(compute_dtype)

    if accumulate_dtype == compute_dtype and compute_dtype == np.float16:
        # Simulate true low-precision accumulation by summing element-wise
        # products in compute_dtype. NumPy's matmul internally promotes to
        # higher precision, so we must do this manually.
        M = A_low.shape[-2]
        K = A_low.shape[-1]
        N = B_low.shape[-1]
        batch_shape = np.broadcast_shapes(A_low.shape[:-2], B_low.shape[:-2])
        result = np.zeros(batch_shape + (M, N), dtype=compute_dtype)
        for k in range(K):
            # Each product and addition stays in compute_dtype
            product = (A_low[..., k:k+1] * B_low[..., k:k+1, :]).astype(compute_dtype)
            result = (result + product).astype(compute_dtype)
        return result

    result = np.matmul(
        A_low.astype(accumulate_dtype), B_low.astype(accumulate_dtype)
    )
    return result.astype(compute_dtype)


def stable_softmax(x: np.ndarray, dtype: np.dtype = np.float16) -> np.ndarray:
    """
    Numerically stable softmax that works in low precision.

    Subtracts the max before exp to prevent overflow, and accumulates
    the denominator sum in FP32 for precision.

    Args:
        x: Input logits, shape (..., n_classes)
        dtype: Working precision

    Returns:
        Probabilities in the given dtype, shape (..., n_classes)
    """
    x = x.astype(dtype)
    x_max = np.max(x, axis=-1, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    # Accumulate sum in FP32 for precision
    sum_exp = np.sum(exp_x.astype(np.float32), axis=-1, keepdims=True)
    return (exp_x.astype(np.float32) / sum_exp).astype(dtype)


# ---------------------------------------------------------------------------
# Loss scaling
# ---------------------------------------------------------------------------

def detect_overflow(tensor: np.ndarray) -> bool:
    """Check if a tensor contains any inf or nan values."""
    return bool(np.any(np.isinf(tensor)) or np.any(np.isnan(tensor)))


class LossScaler:
    """
    Dynamic loss scaler for mixed-precision training.

    Scales the loss before backward pass so that small gradients survive
    FP16 conversion. Dynamically adjusts the scale factor: increases after
    N consecutive successful steps, halves on overflow.
    """

    def __init__(
        self,
        init_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2**24,
    ):
        """
        Args:
            init_scale: Initial loss scale factor
            growth_factor: Multiplier when increasing scale
            backoff_factor: Multiplier when decreasing scale (on overflow)
            growth_interval: Steps of no overflow before increasing scale
            min_scale: Minimum allowed scale
            max_scale: Maximum allowed scale
        """
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._steps_since_overflow = 0

    def get_scale(self) -> float:
        """Return current scale factor."""
        return self._scale

    def scale_loss(self, loss: np.ndarray) -> np.ndarray:
        """Multiply loss by current scale factor."""
        return loss * self._scale

    def unscale_grads(self, grads: np.ndarray) -> np.ndarray:
        """Divide gradients by current scale factor, returning FP32."""
        return grads.astype(np.float32) / self._scale

    def update(self, has_overflow: bool) -> None:
        """
        Update scale factor based on overflow detection.

        Args:
            has_overflow: True if gradients contained inf/nan
        """
        if has_overflow:
            self._scale = max(self._scale * self._backoff_factor, self._min_scale)
            self._steps_since_overflow = 0
        else:
            self._steps_since_overflow += 1
            if self._steps_since_overflow >= self._growth_interval:
                self._scale = min(self._scale * self._growth_factor, self._max_scale)
                self._steps_since_overflow = 0


def dynamic_loss_scaling_demo(
    n_steps: int = 20,
    d_in: int = 16,
    d_out: int = 8,
    seed: int = 42,
) -> Dict[str, List]:
    """
    Demonstrate dynamic loss scaling with a simple linear model.

    Simulates a training loop: FP16 forward, scaled loss, FP16 backward,
    unscale gradients, update FP32 master weights.

    Args:
        n_steps: Number of training steps
        d_in: Input dimension
        d_out: Output dimension
        seed: Random seed

    Returns:
        Dict with per-step scale, loss, overflow, and gradient stats.
    """
    rng = np.random.RandomState(seed)

    W_master = rng.randn(d_in, d_out).astype(np.float32) * 0.1
    b_master = np.zeros(d_out, dtype=np.float32)
    lr = 0.01
    scaler = LossScaler(init_scale=2**10, growth_interval=5)

    history: Dict[str, List] = {
        "step": [], "scale": [], "loss": [], "overflow": [],
        "grad_mean": [], "grad_max": [],
    }

    for step in range(n_steps):
        x = rng.randn(4, d_in).astype(np.float32)
        y_true = rng.randn(4, d_out).astype(np.float32)

        x_fp16 = x.astype(np.float16)
        W_fp16 = W_master.astype(np.float16)
        b_fp16 = b_master.astype(np.float16)

        y_pred = (x_fp16 @ W_fp16 + b_fp16).astype(np.float16)
        diff = (y_pred.astype(np.float32) - y_true)
        loss = np.mean(diff ** 2)

        scaled_loss = scaler.scale_loss(loss)

        # Simulated scaled gradients (dL/dW = 2/N * X^T @ (y_pred - y_true) * scale)
        N = x.shape[0]
        scaled_grad_W = (2.0 / N) * (x_fp16.astype(np.float32).T @ diff) * scaler.get_scale()
        scaled_grad_W_fp16 = scaled_grad_W.astype(np.float16)

        grad_W = scaler.unscale_grads(scaled_grad_W_fp16)
        overflow = detect_overflow(grad_W)
        scaler.update(overflow)

        history["step"].append(step)
        history["scale"].append(scaler.get_scale())
        history["loss"].append(float(loss))
        history["overflow"].append(overflow)

        if not overflow:
            W_master -= lr * grad_W
            history["grad_mean"].append(float(np.mean(np.abs(grad_W))))
            history["grad_max"].append(float(np.max(np.abs(grad_W))))
        else:
            history["grad_mean"].append(float("nan"))
            history["grad_max"].append(float("nan"))

    return history


# ---------------------------------------------------------------------------
# Gradient analysis
# ---------------------------------------------------------------------------

def gradient_histogram(
    grads: np.ndarray, n_bins: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute histogram of gradient magnitudes on a log scale.

    Args:
        grads: Gradient array (any shape)
        n_bins: Number of histogram bins

    Returns:
        Dict with bin_edges (log10 scale), counts, and magnitude stats.
    """
    magnitudes = np.abs(grads.ravel())
    nonzero = magnitudes[magnitudes > 0]

    if len(nonzero) == 0:
        return {
            "bin_edges": np.array([]),
            "counts": np.array([]),
            "min_magnitude": 0.0,
            "max_magnitude": 0.0,
            "mean_magnitude": 0.0,
            "num_zeros": int(len(magnitudes)),
        }

    log_mags = np.log10(nonzero)
    counts, bin_edges = np.histogram(log_mags, bins=n_bins)

    return {
        "bin_edges": bin_edges,
        "counts": counts,
        "min_magnitude": float(np.min(nonzero)),
        "max_magnitude": float(np.max(nonzero)),
        "mean_magnitude": float(np.mean(magnitudes)),
        "num_zeros": int(np.sum(magnitudes == 0)),
    }


def count_underflow(grads: np.ndarray, dtype=np.float16) -> Dict[str, object]:
    """
    Count how many gradient values would underflow to zero in the target dtype.

    Args:
        grads: Gradient array in FP32
        dtype: Target precision to check against

    Returns:
        Dict with total, underflow_count, underflow_fraction, threshold.
    """
    grads = np.asarray(grads, dtype=np.float32)
    threshold = effective_zero_threshold(dtype)
    magnitudes = np.abs(grads.ravel())
    nonzero_mask = magnitudes > 0
    underflow_mask = nonzero_mask & (magnitudes < threshold)

    return {
        "total": int(grads.size),
        "nonzero": int(np.sum(nonzero_mask)),
        "underflow_count": int(np.sum(underflow_mask)),
        "underflow_fraction": float(np.sum(underflow_mask)) / max(int(np.sum(nonzero_mask)), 1),
        "threshold": threshold,
    }


def effective_zero_threshold(dtype=np.float16) -> float:
    """
    Compute the smallest positive value that survives conversion to dtype.

    Values below this threshold underflow to zero. This is the smallest
    subnormal, not the smallest normal (which is finfo.tiny).

    Args:
        dtype: np.float16 or "bfloat16"

    Returns:
        Threshold value as float
    """
    if dtype == "bfloat16":
        # BF16 smallest subnormal: 2^(-126-7) = 2^-133
        return float(2.0 ** -133)
    dt = np.dtype(dtype)
    return float(np.finfo(dt).smallest_subnormal)


# ---------------------------------------------------------------------------
# Memory analysis
# ---------------------------------------------------------------------------

def model_memory_usage(n_params: int, dtype=np.float16) -> Dict[str, float]:
    """
    Compute memory requirements for model weights.

    Args:
        n_params: Number of parameters
        dtype: Storage precision (np.float32, np.float16, or string like "int8", "int4")

    Returns:
        Dict with bytes, megabytes, and gigabytes.
    """
    bytes_per_element = _bytes_per_element(dtype)
    total_bytes = n_params * bytes_per_element

    return {
        "bytes": total_bytes,
        "megabytes": total_bytes / (1024 ** 2),
        "gigabytes": total_bytes / (1024 ** 3),
        "n_params": n_params,
        "bytes_per_element": bytes_per_element,
    }


def kv_cache_memory(
    n_layers: int,
    n_heads: int,
    d_head: int,
    seq_len: int,
    dtype=np.float16,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Compute KV cache memory for a transformer model.

    Formula: 2 * batch * n_layers * n_heads * seq_len * d_head * bytes_per_element

    Args:
        n_layers: Number of transformer layers
        n_heads: Number of KV heads
        d_head: Head dimension
        seq_len: Sequence length
        dtype: Storage precision
        batch_size: Batch size

    Returns:
        Dict with bytes, megabytes, per_token_bytes.
    """
    bpe = _bytes_per_element(dtype)
    total = 2 * batch_size * n_layers * n_heads * seq_len * d_head * bpe
    per_token = 2 * batch_size * n_layers * n_heads * d_head * bpe

    return {
        "bytes": total,
        "megabytes": total / (1024 ** 2),
        "gigabytes": total / (1024 ** 3),
        "per_token_bytes": per_token,
        "per_token_mb": per_token / (1024 ** 2),
    }


def memory_bandwidth_analysis(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int = 32,
) -> Dict[str, object]:
    """
    Compare memory bandwidth requirements for FP32 vs FP16 inference.

    Computes bytes loaded for weight matrices in a transformer and the
    resulting arithmetic intensity.

    Args:
        batch_size: Batch dimension
        seq_len: Sequence length (typically 1 for decode)
        d_model: Model dimension
        n_layers: Number of transformer layers

    Returns:
        Dict with FP32 and FP16 stats: bytes_loaded, flops, arithmetic_intensity.
    """
    # Per-layer: Q, K, V, O projections = 4 * d_model^2 weights
    # FFN: typically 2 * d_model * 4*d_model = 8 * d_model^2 weights
    weights_per_layer = 4 * d_model * d_model + 8 * d_model * d_model
    total_weights = n_layers * weights_per_layer

    # FLOPs per token per layer: each weight is used once per token
    flops_per_token = 2 * total_weights
    total_flops = batch_size * seq_len * flops_per_token

    results = {}
    for label, bpe in [("fp32", 4), ("fp16", 2)]:
        bytes_loaded = total_weights * bpe
        # Also load activations
        activation_bytes = batch_size * seq_len * d_model * bpe
        total_bytes = bytes_loaded + activation_bytes
        intensity = total_flops / total_bytes if total_bytes > 0 else 0.0

        results[label] = {
            "weight_bytes": bytes_loaded,
            "activation_bytes": activation_bytes,
            "total_bytes": total_bytes,
            "flops": total_flops,
            "arithmetic_intensity": intensity,
        }

    results["memory_savings_ratio"] = results["fp32"]["total_bytes"] / results["fp16"]["total_bytes"]

    return results


def _bytes_per_element(dtype) -> float:
    """Return bytes per element for a given dtype specification."""
    if isinstance(dtype, str):
        dtype_lower = dtype.lower()
        if dtype_lower in ("int8", "uint8"):
            return 1
        elif dtype_lower in ("int4", "uint4"):
            return 0.5
        elif dtype_lower == "bfloat16":
            return 2
        elif dtype_lower in ("float16", "fp16"):
            return 2
        elif dtype_lower in ("float32", "fp32"):
            return 4
        else:
            raise ValueError(f"Unknown dtype string: {dtype}")
    return np.dtype(dtype).itemsize
