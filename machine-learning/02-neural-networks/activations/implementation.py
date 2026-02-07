"""
Activation Functions -- From-scratch NumPy implementation.

Element-wise nonlinearities that enable neural networks to approximate arbitrary
functions. Includes ReLU, Leaky ReLU, Sigmoid, Tanh, GELU, and SiLU/Swish, each
with forward and backward passes. These are the first operations you fuse into
CUDA kernels for inference optimization.
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

_erf = np.vectorize(math.erf, otypes=[np.float64])


class Activation(ABC):
    """Base class for activation functions with forward/backward interface."""

    def __init__(self):
        self._cache: Optional[dict] = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute activation and cache values needed for backward."""
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradient using cached values from forward pass."""
        pass

    def _check_cache(self) -> None:
        if self._cache is None:
            raise RuntimeError(
                "backward() called before forward(). Run forward() first."
            )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.

    For x >= 0: 1 / (1 + exp(-x))
    For x < 0:  exp(x) / (1 + exp(x))
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    pos = x >= 0
    neg = ~pos
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    result[neg] = exp_x / (1.0 + exp_x)
    return result


class ReLU(Activation):
    """Rectified Linear Unit: max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self._cache = {"x": x.copy()}
        return np.maximum(0.0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        x = self._cache["x"]
        return grad_output * (x > 0).astype(np.float64)


class LeakyReLU(Activation):
    """Leaky ReLU with configurable negative slope."""

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self._cache = {"x": x.copy()}
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        x = self._cache["x"]
        return grad_output * np.where(x > 0, 1.0, self.alpha)


class Sigmoid(Activation):
    """Numerically stable sigmoid activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        s = _stable_sigmoid(x)
        self._cache = {"s": s}
        return s

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        s = self._cache["s"]
        return grad_output * s * (1.0 - s)


class Tanh(Activation):
    """Hyperbolic tangent activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        t = np.tanh(x)
        self._cache = {"t": t}
        return t

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        t = self._cache["t"]
        return grad_output * (1.0 - t ** 2)


class GELU(Activation):
    """
    Gaussian Error Linear Unit.

    Supports exact (erf-based) and tanh approximation modes.
    Default is tanh approximation, matching common practice in GPT/BERT.
    """

    SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)  # ~0.7978845608

    def __init__(self, approximate: bool = True):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self._cache = {"x": x.copy()}

        if self.approximate:
            inner = self.SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)
            t = np.tanh(inner)
            self._cache["t"] = t
            return 0.5 * x * (1.0 + t)
        else:
            phi = 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
            self._cache["phi"] = phi
            return x * phi

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        x = self._cache["x"]

        if self.approximate:
            t = self._cache["t"]
            sech2 = 1.0 - t ** 2
            d_inner = self.SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x ** 2)
            local_grad = 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
        else:
            phi = self._cache["phi"]
            # pdf of standard normal: (1/sqrt(2*pi)) * exp(-x^2/2)
            pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
            local_grad = phi + x * pdf

        return grad_output * local_grad


class SiLU(Activation):
    """Sigmoid Linear Unit (Swish): x * sigmoid(x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        s = _stable_sigmoid(x)
        self._cache = {"x": x.copy(), "s": s}
        return x * s

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        x = self._cache["x"]
        s = self._cache["s"]
        local_grad = s * (1.0 + x * (1.0 - s))
        return grad_output * local_grad


Swish = SiLU


def gradient_check(
    activation: Activation,
    x: np.ndarray,
    h: float = 1e-5,
) -> dict:
    """
    Verify analytical backward against numerical finite differences.

    Uses central differences: f'(x) ~ (f(x+h) - f(x-h)) / (2h)

    Args:
        activation: Activation instance to check
        x: Input array to test at
        h: Step size for finite differences

    Returns:
        Dict with max_abs_error, max_rel_error, and whether check passed
    """
    x = np.asarray(x, dtype=np.float64)

    activation.forward(x)
    grad_output = np.ones_like(x)
    analytical = activation.backward(grad_output)

    numerical = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += h
        x_minus[idx] -= h

        f_plus = activation.forward(x_plus)
        f_minus = activation.forward(x_minus)
        numerical[idx] = (f_plus[idx] - f_minus[idx]) / (2.0 * h)
        it.iternext()

    # Restore the cache to the original input
    activation.forward(x)

    abs_error = np.abs(analytical - numerical)
    max_abs_error = float(np.max(abs_error)) if abs_error.size > 0 else 0.0

    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-8)
    rel_error = abs_error / denom
    max_rel_error = float(np.max(rel_error)) if rel_error.size > 0 else 0.0

    return {
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "passed": max_rel_error < 1e-5,
    }
