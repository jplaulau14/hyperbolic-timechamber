"""
Optimizers -- From-scratch NumPy implementation.

SGD, SGD with Momentum, Adam, and AdamW, plus learning rate schedulers.
These are the update rules that convert gradients into weight updates. AdamW is
the default optimizer for every modern LLM (GPT-3, LLaMA, Mistral). The
progression from SGD to AdamW represents decades of research into why adaptive
per-parameter learning rates and decoupled weight decay produce better models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


ParamDict = Dict[str, np.ndarray]
ParamList = List[ParamDict]


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, params: ParamList, lr: float):
        """
        Args:
            params: List of dicts, each with "params" (np.ndarray) and "grad" (np.ndarray)
            lr: Learning rate
        """
        self.params = params
        self.lr = lr
        self.state: Dict[int, dict] = {}

    @abstractmethod
    def step(self) -> None:
        """Apply one update to all parameters using their gradients."""
        pass

    def zero_grad(self) -> None:
        """Set all gradients to zero."""
        for p in self.params:
            p["grad"] = np.zeros_like(p["params"])


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and Nesterov acceleration.

    Without momentum: theta -= lr * g
    With momentum:    v = beta*v + g; theta -= lr * v
    Nesterov:         theta -= lr * (beta*v + g)  (after v update)
    """

    def __init__(
        self,
        params: ParamList,
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.nesterov = nesterov
        if nesterov and momentum == 0.0:
            raise ValueError("Nesterov momentum requires momentum > 0")

    def step(self) -> None:
        for i, p in enumerate(self.params):
            grad = p["grad"]

            if self.momentum == 0.0:
                p["params"] -= self.lr * grad
            else:
                if i not in self.state:
                    self.state[i] = {"velocity": np.zeros_like(p["params"])}

                v = self.state[i]["velocity"]
                v[:] = self.momentum * v + grad

                if self.nesterov:
                    p["params"] -= self.lr * (self.momentum * v + grad)
                else:
                    p["params"] -= self.lr * v


class Adam(Optimizer):
    """
    Adam optimizer with bias-corrected first and second moment estimates.

    Maintains per-parameter adaptive learning rates via exponential moving
    averages of gradients (first moment m) and squared gradients (second moment v).
    """

    def __init__(
        self,
        params: ParamList,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

    def step(self) -> None:
        self.t += 1

        for i, p in enumerate(self.params):
            grad = p["grad"]

            if i not in self.state:
                self.state[i] = {
                    "m": np.zeros_like(p["params"]),
                    "v": np.zeros_like(p["params"]),
                }

            m = self.state[i]["m"]
            v = self.state[i]["v"]

            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * grad ** 2

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            p["params"] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    Identical to Adam except weight decay is applied directly to parameters
    rather than being folded into the gradient. This ensures weight decay
    strength is uniform across all parameters regardless of gradient statistics.
    """

    def __init__(
        self,
        params: ParamList,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

    def step(self) -> None:
        self.t += 1

        for i, p in enumerate(self.params):
            grad = p["grad"]

            if i not in self.state:
                self.state[i] = {
                    "m": np.zeros_like(p["params"]),
                    "v": np.zeros_like(p["params"]),
                }

            m = self.state[i]["m"]
            v = self.state[i]["v"]

            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * grad ** 2

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Decoupled weight decay: applied to params directly, not through the gradient
            p["params"] -= self.lr * (
                m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p["params"]
            )


class AdamL2(Optimizer):
    """
    Adam with L2 regularization (the wrong way).

    Weight decay is injected into the gradient BEFORE Adam processes it, so it
    gets scaled by the adaptive learning rate. Provided for comparison with AdamW
    to demonstrate why decoupled weight decay matters.
    """

    def __init__(
        self,
        params: ParamList,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

    def step(self) -> None:
        self.t += 1

        for i, p in enumerate(self.params):
            # L2: fold penalty into the gradient before moment computation
            grad = p["grad"] + self.weight_decay * p["params"]

            if i not in self.state:
                self.state[i] = {
                    "m": np.zeros_like(p["params"]),
                    "v": np.zeros_like(p["params"]),
                }

            m = self.state[i]["m"]
            v = self.state[i]["v"]

            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * grad ** 2

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            p["params"] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Learning Rate Schedulers
# ---------------------------------------------------------------------------


class LRScheduler(ABC):
    """Base class for learning rate schedulers."""

    @abstractmethod
    def get_lr(self, step: int) -> float:
        """Return the learning rate for the given step."""
        pass

    def step(self, optimizer: Optimizer, current_step: int) -> None:
        """Update the optimizer's learning rate for the current step."""
        optimizer.lr = self.get_lr(current_step)


class StepDecayScheduler(LRScheduler):
    """Multiply learning rate by gamma every step_size steps."""

    def __init__(self, initial_lr: float, decay_factor: float, step_size: int):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.step_size = step_size

    def get_lr(self, step: int) -> float:
        return self.initial_lr * self.decay_factor ** (step // self.step_size)


class CosineScheduler(LRScheduler):
    """Cosine annealing from max_lr to min_lr over total_steps."""

    def __init__(self, max_lr: float, min_lr: float, total_steps: int):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        step = min(step, self.total_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + np.cos(np.pi * step / self.total_steps)
        )


class WarmupCosineScheduler(LRScheduler):
    """Linear warmup from 0 to max_lr, then cosine decay to min_lr."""

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps

        decay_steps = self.total_steps - self.warmup_steps
        decay_step = min(step - self.warmup_steps, decay_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + np.cos(np.pi * decay_step / decay_steps)
        )
