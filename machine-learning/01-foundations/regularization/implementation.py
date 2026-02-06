"""
Regularization (L1, L2, Elastic Net) — From-scratch NumPy implementation.

Adds penalty terms to the loss function to prevent overfitting by discouraging
large weights. L2 (Ridge) shrinks weights toward zero smoothly, L1 (Lasso) drives
weights to exactly zero for sparsity, and Elastic Net combines both. Also
demonstrates weight decay and its equivalence to L2 regularization under SGD.
"""

import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core penalty and gradient functions
# ---------------------------------------------------------------------------

def l2_penalty(w: np.ndarray, lambda_: float) -> float:
    """
    Compute L2 regularization penalty: (lambda / 2) * sum(w_i^2).

    Args:
        w: Weight vector, shape (d,) or (d, k)
        lambda_: Regularization strength

    Returns:
        Scalar penalty value
    """
    return 0.5 * lambda_ * np.sum(w ** 2)


def l2_gradient(w: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Compute L2 regularization gradient: lambda * w.

    Args:
        w: Weight vector, shape (d,) or (d, k)
        lambda_: Regularization strength

    Returns:
        Gradient with same shape as w
    """
    return lambda_ * w


def l1_penalty(w: np.ndarray, lambda_: float) -> float:
    """
    Compute L1 regularization penalty: lambda * sum(|w_i|).

    Args:
        w: Weight vector, shape (d,) or (d, k)
        lambda_: Regularization strength

    Returns:
        Scalar penalty value
    """
    return lambda_ * np.sum(np.abs(w))


def l1_gradient(w: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Compute L1 regularization subgradient: lambda * sign(w).

    Uses the subgradient convention sign(0) = 0 at the non-differentiable point.

    Args:
        w: Weight vector, shape (d,) or (d, k)
        lambda_: Regularization strength

    Returns:
        Subgradient with same shape as w
    """
    return lambda_ * np.sign(w)


def elastic_net_penalty(w: np.ndarray, lambda_: float, l1_ratio: float) -> float:
    """
    Compute Elastic Net penalty: lambda * (l1_ratio * ||w||_1 + (1 - l1_ratio)/2 * ||w||_2^2).

    Args:
        w: Weight vector, shape (d,) or (d, k)
        lambda_: Overall regularization strength
        l1_ratio: Mixing parameter in [0, 1]. 1.0 = pure L1, 0.0 = pure L2.

    Returns:
        Scalar penalty value
    """
    l1_term = l1_ratio * np.sum(np.abs(w))
    l2_term = 0.5 * (1.0 - l1_ratio) * np.sum(w ** 2)
    return lambda_ * (l1_term + l2_term)


def elastic_net_gradient(w: np.ndarray, lambda_: float, l1_ratio: float) -> np.ndarray:
    """
    Compute Elastic Net gradient: lambda * (l1_ratio * sign(w) + (1 - l1_ratio) * w).

    Args:
        w: Weight vector, shape (d,) or (d, k)
        lambda_: Overall regularization strength
        l1_ratio: Mixing parameter in [0, 1]

    Returns:
        Gradient with same shape as w
    """
    return lambda_ * (l1_ratio * np.sign(w) + (1.0 - l1_ratio) * w)


# ---------------------------------------------------------------------------
# Weight decay
# ---------------------------------------------------------------------------

def sgd_with_weight_decay(
    w: np.ndarray, gradient: np.ndarray, lr: float, lambda_: float
) -> np.ndarray:
    """
    Single SGD update step with weight decay.

    w <- w - lr * gradient - lr * lambda_ * w

    This is mathematically equivalent to L2 regularization for vanilla SGD.

    Args:
        w: Current weights, shape (d,)
        gradient: Data loss gradient (not including regularization), shape (d,)
        lr: Learning rate
        lambda_: Weight decay coefficient

    Returns:
        Updated weights, same shape as w
    """
    return w - lr * gradient - lr * lambda_ * w


# ---------------------------------------------------------------------------
# Helper: MSE loss (same convention as linear regression module)
# ---------------------------------------------------------------------------

def _mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE loss with 1/(2n) factor."""
    n = y_true.shape[0]
    return float(np.sum((y_pred - y_true) ** 2) / (2 * n))


# ---------------------------------------------------------------------------
# Regularized regression classes
# ---------------------------------------------------------------------------

class RidgeRegression:
    """Linear regression with L2 (Ridge) regularization."""

    def __init__(
        self,
        lambda_: float = 1.0,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-7,
    ):
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance

        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """
        Fit with L2-regularized gradient descent.

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)
            y: Target values, shape (n_samples,)

        Returns:
            Self for method chaining
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history = []

        prev_loss = float("inf")

        for _ in range(self.n_iterations):
            y_pred = X @ self.w + self.b
            data_loss = _mse_loss(y, y_pred)
            total_loss = data_loss + l2_penalty(self.w, self.lambda_)
            self.history.append(total_loss)

            if abs(prev_loss - total_loss) < self.tolerance:
                break
            prev_loss = total_loss

            error = y_pred - y
            # (n_features, n_samples) @ (n_samples,) -> (n_features,)
            dw = X.T @ error / n_samples + l2_gradient(self.w, self.lambda_)
            db = np.mean(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self

    def fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """
        Fit using the Ridge closed-form solution.

        The GD objective is (1/2n)||Xw - y||^2 + (lambda/2)||w||^2, so the
        normal equation is (X^T X + n*lambda*I) w = X^T y. We scale by n to
        match the per-sample gradient used in fit().

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)
            y: Target values, shape (n_samples,)

        Returns:
            Self for method chaining
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        X_aug = np.column_stack([np.ones(n_samples), X])

        # Scale lambda by n to match the 1/n factor in the data loss gradient
        reg_matrix = n_samples * self.lambda_ * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0.0

        # (d+1, d+1) solve
        A = X_aug.T @ X_aug + reg_matrix
        b_vec = X_aug.T @ y
        w_aug = np.linalg.solve(A, b_vec)

        self.b = w_aug[0]
        self.w = w_aug[1:]
        self.history = []

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions, shape (n_samples,)."""
        if self.w is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.w + self.b

    def get_weights(self) -> Tuple[np.ndarray, float]:
        """Return (weights, bias)."""
        if self.w is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self.w.copy(), self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R^2 (coefficient of determination)."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-15:
            return 1.0 if ss_res < 1e-15 else 0.0
        return float(1.0 - ss_res / ss_tot)


class LassoRegression:
    """Linear regression with L1 (Lasso) regularization."""

    def __init__(
        self,
        lambda_: float = 1.0,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-7,
    ):
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance

        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        """
        Fit with L1-regularized gradient descent (subgradient method).

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)
            y: Target values, shape (n_samples,)

        Returns:
            Self for method chaining
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history = []

        prev_loss = float("inf")

        for _ in range(self.n_iterations):
            y_pred = X @ self.w + self.b
            data_loss = _mse_loss(y, y_pred)
            total_loss = data_loss + l1_penalty(self.w, self.lambda_)
            self.history.append(total_loss)

            if abs(prev_loss - total_loss) < self.tolerance:
                break
            prev_loss = total_loss

            error = y_pred - y
            dw = X.T @ error / n_samples + l1_gradient(self.w, self.lambda_)
            db = np.mean(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions, shape (n_samples,)."""
        if self.w is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.w + self.b

    def get_weights(self) -> Tuple[np.ndarray, float]:
        """Return (weights, bias)."""
        if self.w is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self.w.copy(), self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R^2 (coefficient of determination)."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-15:
            return 1.0 if ss_res < 1e-15 else 0.0
        return float(1.0 - ss_res / ss_tot)


class ElasticNetRegression:
    """Linear regression with Elastic Net (L1 + L2) regularization."""

    def __init__(
        self,
        lambda_: float = 1.0,
        l1_ratio: float = 0.5,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-7,
    ):
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance

        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetRegression":
        """
        Fit with Elastic Net regularized gradient descent.

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)
            y: Target values, shape (n_samples,)

        Returns:
            Self for method chaining
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history = []

        prev_loss = float("inf")

        for _ in range(self.n_iterations):
            y_pred = X @ self.w + self.b
            data_loss = _mse_loss(y, y_pred)
            total_loss = data_loss + elastic_net_penalty(
                self.w, self.lambda_, self.l1_ratio
            )
            self.history.append(total_loss)

            if abs(prev_loss - total_loss) < self.tolerance:
                break
            prev_loss = total_loss

            error = y_pred - y
            dw = X.T @ error / n_samples + elastic_net_gradient(
                self.w, self.lambda_, self.l1_ratio
            )
            db = np.mean(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions, shape (n_samples,)."""
        if self.w is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.w + self.b

    def get_weights(self) -> Tuple[np.ndarray, float]:
        """Return (weights, bias)."""
        if self.w is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self.w.copy(), self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R^2 (coefficient of determination)."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-15:
            return 1.0 if ss_res < 1e-15 else 0.0
        return float(1.0 - ss_res / ss_tot)
