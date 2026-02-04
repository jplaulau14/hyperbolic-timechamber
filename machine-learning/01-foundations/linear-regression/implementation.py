"""
Linear Regression - From-scratch NumPy implementation.

Implements linear regression with both gradient descent and normal equation
(closed-form) solutions. This is the foundation of supervised learning:
forward pass, loss computation, backward pass, and parameter update.
"""

import numpy as np
from typing import Literal, Tuple, List, Optional


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error loss with 1/2n factor.

    The 1/2 factor cancels with the exponent during differentiation.

    Args:
        y_true: Ground truth values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        Scalar loss value
    """
    n = y_true.shape[0]
    return np.sum((y_pred - y_true) ** 2) / (2 * n)


class LinearRegression:
    """Linear regression with gradient descent or normal equation."""

    def __init__(
        self,
        method: Literal["gradient_descent", "normal_equation"] = "gradient_descent",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        """
        Initialize linear regression model.

        Args:
            method: Training method, either 'gradient_descent' or 'normal_equation'
            learning_rate: Step size for gradient descent
            n_iterations: Maximum iterations for gradient descent
            tolerance: Convergence threshold for gradient descent
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance

        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.history: List[float] = []

    def _compute_gradients(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of MSE loss with respect to weights and bias.

        Args:
            X: Input features, shape (n_samples, n_features)
            y: True targets, shape (n_samples,)
            y_pred: Predictions, shape (n_samples,)

        Returns:
            Tuple of (dw, db) where dw has shape (n_features,) and db is scalar
        """
        n = X.shape[0]
        error = y_pred - y
        # (n_features, n_samples) @ (n_samples,) -> (n_features,)
        dw = X.T @ error / n
        db = np.mean(error)
        return dw, db

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit model using gradient descent optimization.

        Args:
            X: Input features, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)
        """
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history = []

        prev_loss = float("inf")

        for _ in range(self.n_iterations):
            y_pred = X @ self.w + self.b
            loss = mse_loss(y, y_pred)
            self.history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss

            dw, db = self._compute_gradients(X, y, y_pred)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit model using the normal equation (closed-form solution).

        Uses np.linalg.lstsq for numerical stability and handling of
        singular/underdetermined systems.

        Args:
            X: Input features, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)
        """
        n_samples = X.shape[0]
        X_aug = np.column_stack([np.ones(n_samples), X])
        w_aug, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        self.b = w_aug[0]
        self.w = w_aug[1:]
        self.history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the linear regression model.

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)
            y: Target values, shape (n_samples,)

        Returns:
            Self for method chaining
        """
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.method == "gradient_descent":
            self._fit_gradient_descent(X, y)
        else:
            self._fit_normal_equation(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)

        Returns:
            Predictions, shape (n_samples,)
        """
        if self.w is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.w + self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R^2 (coefficient of determination).

        R^2 = 1 - SS_res / SS_tot

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)
            y: True target values, shape (n_samples,)

        Returns:
            R^2 score (1.0 is perfect, 0.0 means predicting mean, can be negative)
        """
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot < 1e-15:
            return 1.0 if ss_res < 1e-15 else 0.0

        return 1 - ss_res / ss_tot
