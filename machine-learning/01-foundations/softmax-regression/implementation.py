"""
Softmax Regression - From-scratch NumPy implementation.

Multiclass extension of logistic regression that produces probability distributions
over K classes. Uses the softmax function to convert logits to probabilities and
cross-entropy loss for training. This is the foundation for every transformer
output layer and attention mechanism.
"""

import numpy as np
from typing import Tuple, List, Optional


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax function using the subtract-max trick.

    Args:
        z: Logits array, shape (n, K) or (K,)

    Returns:
        Probabilities of same shape, rows sum to 1
    """
    z = np.asarray(z)

    if z.ndim == 1:
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)

    # (n, K) case: subtract max per row for numerical stability
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(P: np.ndarray, Y: np.ndarray, eps: float = 1e-15) -> float:
    """
    Cross-entropy loss between predicted probabilities and one-hot labels.

    Args:
        P: Predicted probabilities, shape (n, K)
        Y: One-hot encoded labels, shape (n, K)
        eps: Small constant to prevent log(0)

    Returns:
        Scalar loss value
    """
    P_clipped = np.clip(P, eps, 1.0 - eps)
    n = Y.shape[0]
    return float(-np.sum(Y * np.log(P_clipped)) / n)


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class indices to one-hot encoding.

    Args:
        y: Class indices, shape (n,)
        num_classes: Number of classes K

    Returns:
        One-hot matrix, shape (n, K)
    """
    y = np.asarray(y)
    n = y.shape[0]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y] = 1.0
    return one_hot


class SoftmaxRegression:
    """Softmax regression (multinomial logistic regression) with gradient descent."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        learning_rate: float = 0.01,
    ):
        """
        Initialize softmax regression model.

        Args:
            num_features: Number of input features d
            num_classes: Number of output classes K
            learning_rate: Step size for gradient descent
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.W: np.ndarray = np.random.randn(num_features, num_classes) * 0.01
        self.b: np.ndarray = np.zeros(num_classes)
        self.history: List[float] = []

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute logits and probabilities.

        Args:
            X: Input features, shape (n, d)

        Returns:
            Tuple of (Z, P) where Z is logits (n, K) and P is probabilities (n, K)
        """
        # (n, d) @ (d, K) -> (n, K)
        Z = X @ self.W + self.b
        P = softmax(Z)
        return Z, P

    def backward(
        self, X: np.ndarray, P: np.ndarray, Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of cross-entropy loss.

        Args:
            X: Input features, shape (n, d)
            P: Predicted probabilities, shape (n, K)
            Y: One-hot labels, shape (n, K)

        Returns:
            Tuple of (dW, db) where dW has shape (d, K) and db has shape (K,)
        """
        n = X.shape[0]
        error = P - Y
        # (d, n) @ (n, K) -> (d, K)
        dW = X.T @ error / n
        # (K,)
        db = np.mean(error, axis=0)
        return dW, db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        verbose: bool = False,
    ) -> "SoftmaxRegression":
        """
        Fit the model using gradient descent.

        Args:
            X: Input features, shape (n, d)
            y: Class labels, shape (n,) with values in [0, K-1]
            epochs: Number of training iterations
            verbose: Print loss every 100 epochs if True

        Returns:
            Self for method chaining
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        Y = one_hot_encode(y, self.num_classes)
        self.history = []

        for epoch in range(epochs):
            Z, P = self.forward(X)
            loss = cross_entropy_loss(P, Y)
            self.history.append(loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss:.6f}")

            dW, db = self.backward(X, P, Y)
            self.W = self.W - self.learning_rate * dW
            self.b = self.b - self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted probability distributions.

        Args:
            X: Input features, shape (n, d)

        Returns:
            Probabilities, shape (n, K)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        _, P = self.forward(X)
        return P

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted class labels.

        Args:
            X: Input features, shape (n, d)

        Returns:
            Class predictions, shape (n,)
        """
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Input features, shape (n, d)
            y: True class labels, shape (n,)

        Returns:
            Accuracy score between 0 and 1
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
