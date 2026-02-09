"""
Normalization Layers -- From-scratch NumPy implementation.

LayerNorm and RMSNorm for stabilizing activations in deep networks. LayerNorm
normalizes to zero mean and unit variance across the feature dimension, then
applies learnable scale (gamma) and shift (beta). RMSNorm simplifies this by
removing mean subtraction entirely -- only rescaling by the root mean square.
Both behave identically at training and inference time, unlike BatchNorm.
"""

import numpy as np
from typing import Optional


class LayerNorm:
    """Layer Normalization (Ba et al., 2016).

    Normalizes across the last dimension to zero mean and unit variance,
    then applies learnable affine transform: y = gamma * x_hat + beta.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape, dtype=np.float64)
        self.beta = np.zeros(normalized_shape, dtype=np.float64)

        self._cache: Optional[dict] = None
        self.grad_gamma: Optional[np.ndarray] = None
        self.grad_beta: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: normalize, then scale and shift.

        Args:
            x: Input tensor, shape (B, D) or (B, L, D)

        Returns:
            Normalized output, same shape as x
        """
        x = np.asarray(x, dtype=np.float64)
        D = self.normalized_shape

        mu = np.mean(x, axis=-1, keepdims=True)
        x_centered = x - mu
        var = np.mean(x_centered ** 2, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = x_centered * std_inv
        y = self.gamma * x_hat + self.beta

        self._cache = {
            "x_hat": x_hat,
            "std_inv": std_inv,
            "D": D,
        }
        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass using the simplified LayerNorm gradient formula.

        Args:
            grad_output: Upstream gradient dL/dy, same shape as forward output

        Returns:
            Gradient dL/dx, same shape as input
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward().")

        grad_output = np.asarray(grad_output, dtype=np.float64)
        x_hat = self._cache["x_hat"]
        std_inv = self._cache["std_inv"]
        D = self._cache["D"]

        sum_axes = tuple(range(grad_output.ndim - 1))
        self.grad_gamma = np.sum(grad_output * x_hat, axis=sum_axes)
        self.grad_beta = np.sum(grad_output, axis=sum_axes)

        # g = dL/dy * gamma
        g = grad_output * self.gamma

        # dx = std_inv/D * (D*g - sum(g) - x_hat * sum(g * x_hat))
        sum_g = np.sum(g, axis=-1, keepdims=True)
        sum_g_xhat = np.sum(g * x_hat, axis=-1, keepdims=True)
        dx = (std_inv / D) * (D * g - sum_g - x_hat * sum_g_xhat)

        return dx


class RMSNorm:
    """RMS Normalization (Zhang & Sennrich, 2019).

    Normalizes by root mean square without mean subtraction.
    Simpler and faster than LayerNorm with comparable performance.
    Used in LLaMA, Mistral, and Gemma.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape, dtype=np.float64)

        self._cache: Optional[dict] = None
        self.grad_gamma: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: normalize by RMS, then scale.

        Args:
            x: Input tensor, shape (B, D) or (B, L, D)

        Returns:
            Normalized output, same shape as x
        """
        x = np.asarray(x, dtype=np.float64)
        D = self.normalized_shape

        ms = np.mean(x ** 2, axis=-1, keepdims=True)
        rms_inv = 1.0 / np.sqrt(ms + self.eps)
        x_hat = x * rms_inv
        y = x_hat * self.gamma

        self._cache = {
            "x_hat": x_hat,
            "rms_inv": rms_inv,
            "D": D,
        }
        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for RMSNorm.

        Args:
            grad_output: Upstream gradient dL/dy, same shape as forward output

        Returns:
            Gradient dL/dx, same shape as input
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward().")

        grad_output = np.asarray(grad_output, dtype=np.float64)
        x_hat = self._cache["x_hat"]
        rms_inv = self._cache["rms_inv"]
        D = self._cache["D"]

        sum_axes = tuple(range(grad_output.ndim - 1))
        self.grad_gamma = np.sum(grad_output * x_hat, axis=sum_axes)

        # g = dL/dy * gamma
        g = grad_output * self.gamma

        # dx = rms_inv * (g - x_hat * (1/D) * sum(g * x_hat))
        sum_g_xhat = np.sum(g * x_hat, axis=-1, keepdims=True)
        dx = rms_inv * (g - x_hat * (sum_g_xhat / D))

        return dx


def gradient_check(
    norm_layer,
    x: np.ndarray,
    h: float = 1e-5,
) -> dict:
    """
    Verify analytical gradients against numerical finite differences.

    Checks dx, dgamma, and dbeta (if applicable) using central differences.

    Args:
        norm_layer: LayerNorm or RMSNorm instance
        x: Input array to test at
        h: Step size for finite differences

    Returns:
        Dict with per-gradient max relative errors and overall pass/fail
    """
    x = np.asarray(x, dtype=np.float64)

    y = norm_layer.forward(x)
    grad_output = np.random.randn(*y.shape)
    dx_analytical = norm_layer.backward(grad_output)

    loss_fn = lambda out: np.sum(out * grad_output)

    dx_numerical = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += h
        x_minus[idx] -= h
        dx_numerical[idx] = (loss_fn(norm_layer.forward(x_plus)) - loss_fn(norm_layer.forward(x_minus))) / (2.0 * h)
        it.iternext()

    norm_layer.forward(x)
    norm_layer.backward(grad_output)

    results = {}
    results["dx"] = _relative_error(dx_analytical, dx_numerical)

    dgamma_numerical = np.zeros_like(norm_layer.gamma)
    original_gamma = norm_layer.gamma.copy()
    it = np.nditer(norm_layer.gamma, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        norm_layer.gamma = original_gamma.copy()
        norm_layer.gamma[idx] += h
        y_plus = norm_layer.forward(x)
        norm_layer.gamma = original_gamma.copy()
        norm_layer.gamma[idx] -= h
        y_minus = norm_layer.forward(x)
        dgamma_numerical[idx] = (loss_fn(y_plus) - loss_fn(y_minus)) / (2.0 * h)
        it.iternext()
    norm_layer.gamma = original_gamma.copy()
    norm_layer.forward(x)
    norm_layer.backward(grad_output)
    results["dgamma"] = _relative_error(norm_layer.grad_gamma, dgamma_numerical)

    if hasattr(norm_layer, "beta"):
        dbeta_numerical = np.zeros_like(norm_layer.beta)
        original_beta = norm_layer.beta.copy()
        it = np.nditer(norm_layer.beta, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            norm_layer.beta = original_beta.copy()
            norm_layer.beta[idx] += h
            y_plus = norm_layer.forward(x)
            norm_layer.beta = original_beta.copy()
            norm_layer.beta[idx] -= h
            y_minus = norm_layer.forward(x)
            dbeta_numerical[idx] = (loss_fn(y_plus) - loss_fn(y_minus)) / (2.0 * h)
            it.iternext()
        norm_layer.beta = original_beta.copy()
        norm_layer.forward(x)
        norm_layer.backward(grad_output)
        results["dbeta"] = _relative_error(norm_layer.grad_beta, dbeta_numerical)

    passed = all(v < 1e-5 for v in results.values())
    results["passed"] = passed
    return results


def _relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    """Max relative error between analytical and numerical gradients."""
    abs_err = np.abs(analytical - numerical)
    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-8)
    return float(np.max(abs_err / denom))
