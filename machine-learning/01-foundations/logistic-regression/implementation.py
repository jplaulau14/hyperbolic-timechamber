"""
Logistic Regression - From-scratch NumPy implementation.

Binary classification via the sigmoid function. Transforms linear output into
probabilities using the sigmoid activation, then applies binary cross-entropy
loss for training. This is the building block for neural network output layers.
"""

import numpy as np
from typing import Tuple, List, Optional


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid activation function.

    Uses different formulas for positive and negative inputs to avoid overflow:
    - For z >= 0: 1 / (1 + exp(-z))
    - For z < 0: exp(z) / (1 + exp(z))

    Args:
        z: Input array, any shape

    Returns:
        Array of same shape with values in (0, 1)
    """
    z = np.asarray(z)
    result = np.zeros_like(z, dtype=np.float64)
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[neg_mask])
    result[neg_mask] = exp_z / (1.0 + exp_z)
    return result


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z)).

    Args:
        z: Input array, any shape

    Returns:
        Array of same shape with derivative values
    """
    s = sigmoid(z)
    return s * (1.0 - s)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary cross-entropy loss with numerical stability.

    Args:
        y_true: Ground truth labels (0 or 1), shape (n_samples,)
        y_pred: Predicted probabilities, shape (n_samples,)
        eps: Small constant to prevent log(0)

    Returns:
        Scalar loss value
    """
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    n = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)) / n
    return float(loss)


class LogisticRegression:
    """Logistic regression with gradient descent optimization."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        """
        Initialize logistic regression model.

        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Maximum iterations for gradient descent
            tolerance: Convergence threshold (stop if loss change < tolerance)
        """
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
        Compute gradients of BCE loss with respect to weights and bias.

        Args:
            X: Input features, shape (n_samples, n_features)
            y: True labels, shape (n_samples,)
            y_pred: Predicted probabilities, shape (n_samples,)

        Returns:
            Tuple of (dw, db) where dw has shape (n_features,) and db is scalar
        """
        n = X.shape[0]
        error = y_pred - y
        # (n_features, n_samples) @ (n_samples,) -> (n_features,)
        dw = X.T @ error / n
        db = float(np.mean(error))
        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit the logistic regression model using gradient descent.

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)
            y: Binary labels (0 or 1), shape (n_samples,)

        Returns:
            Self for method chaining
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y.ndim == 2:
            y = y.ravel()

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history = []

        prev_loss = float("inf")

        for _ in range(self.n_iterations):
            z = X @ self.w + self.b
            y_pred = sigmoid(z)

            loss = binary_cross_entropy(y, y_pred)
            self.history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss

            dw, db = self._compute_gradients(X, y, y_pred)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted probabilities P(y=1|X).

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)

        Returns:
            Probabilities, shape (n_samples,)
        """
        if self.w is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        z = X @ self.w + self.b
        return sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary predictions.

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)
            threshold: Decision threshold (default 0.5)

        Returns:
            Binary predictions (0 or 1), shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Input features, shape (n_samples,) or (n_samples, n_features)
            y: True labels (0 or 1), shape (n_samples,)

        Returns:
            Accuracy score between 0 and 1
        """
        y_pred = self.predict(X)
        if y.ndim == 2:
            y = y.ravel()
        return float(np.mean(y_pred == y))

    def decision_boundary_params(self) -> Tuple[float, float]:
        """
        Get slope and intercept for 2D decision boundary visualization.

        For 2D features, the decision boundary is where z = w1*x1 + w2*x2 + b = 0.
        Solving for x2: x2 = -(w1/w2)*x1 - b/w2

        Returns:
            Tuple of (slope, intercept) where slope = -w1/w2 and intercept = -b/w2

        Raises:
            ValueError: If model has not been fitted or doesn't have exactly 2 features
        """
        if self.w is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        if len(self.w) != 2:
            raise ValueError("decision_boundary_params requires exactly 2 features")
        if abs(self.w[1]) < 1e-10:
            raise ValueError("w2 is too close to zero; decision boundary is vertical")

        slope = -self.w[0] / self.w[1]
        intercept = -self.b / self.w[1]
        return float(slope), float(intercept)
