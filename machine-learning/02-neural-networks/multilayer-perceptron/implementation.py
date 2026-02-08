"""
Multilayer Perceptron -- From-scratch NumPy implementation.

Feedforward neural network with configurable hidden layers, activation functions,
and loss functions. Implements forward pass, backpropagation via chain rule, and
mini-batch gradient descent. The MLP is the building block of all deep learning:
the FFN in every transformer layer is literally a two-layer MLP.
"""

import importlib.util
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

_activations_path = Path(__file__).parent.parent / "activations" / "implementation.py"
_spec = importlib.util.spec_from_file_location("activations_module", _activations_path)
_activations_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_activations_mod)

Activation = _activations_mod.Activation
ReLU = _activations_mod.ReLU
LeakyReLU = _activations_mod.LeakyReLU
Sigmoid = _activations_mod.Sigmoid
Tanh = _activations_mod.Tanh
GELU = _activations_mod.GELU
SiLU = _activations_mod.SiLU


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax using the subtract-max trick.

    Args:
        z: Logits, shape (n, K) or (K,)

    Returns:
        Probabilities of same shape, rows sum to 1
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-15) -> float:
    """
    Cross-entropy loss between predicted probabilities and one-hot labels.

    Args:
        predictions: Predicted probabilities, shape (n, K)
        targets: One-hot encoded labels, shape (n, K)
        eps: Small constant to prevent log(0)

    Returns:
        Scalar loss value
    """
    P_clipped = np.clip(predictions, eps, 1.0 - eps)
    n = targets.shape[0]
    return float(-np.sum(targets * np.log(P_clipped)) / n)


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean squared error loss with 1/(2n) scaling.

    Args:
        predictions: Predicted values, shape (n, d_out)
        targets: Target values, shape (n, d_out)

    Returns:
        Scalar loss value
    """
    n = targets.shape[0]
    return float(0.5 * np.sum((predictions - targets) ** 2) / n)


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class indices to one-hot encoding.

    Args:
        y: Class indices, shape (n,)
        num_classes: Number of classes K

    Returns:
        One-hot matrix, shape (n, K)
    """
    y = np.asarray(y, dtype=int)
    n = y.shape[0]
    one_hot = np.zeros((n, num_classes), dtype=np.float64)
    one_hot[np.arange(n), y] = 1.0
    return one_hot


class Layer:
    """Single layer: linear transform followed by optional activation."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: Union[Activation, str, None] = None,
        init_method: str = "he",
    ):
        """
        Args:
            n_in: Number of input features
            n_out: Number of output features
            activation: Activation instance, "softmax" string, or None for linear
            init_method: "he" for ReLU variants, "xavier" for sigmoid/tanh
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        if init_method == "he":
            self.W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
        elif init_method == "xavier":
            self.W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / (n_in + n_out))
        else:
            raise ValueError(f"Unknown init method: {init_method}")

        self.b = np.zeros(n_out)

        self.z: Optional[np.ndarray] = None
        self.a: Optional[np.ndarray] = None
        self.a_prev: Optional[np.ndarray] = None

        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: z = x @ W.T + b, then apply activation.

        Args:
            x: Input, shape (batch, n_in)

        Returns:
            Output activations, shape (batch, n_out)
        """
        self.a_prev = x.copy()

        # (batch, n_in) @ (n_in, n_out) -> (batch, n_out)
        self.z = x @ self.W.T + self.b

        if self.activation is None:
            self.a = self.z.copy()
        elif isinstance(self.activation, str) and self.activation == "softmax":
            self.a = softmax(self.z)
        else:
            self.a = self.activation.forward(self.z)

        return self.a

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute parameter gradients and return delta for previous layer.

        Args:
            delta: Error signal for this layer, shape (batch, n_out).
                   For output layers this is already dL/dz.
                   For hidden layers this is dL/da (before activation backward).

        Returns:
            Delta to propagate to the previous layer (dL/da_prev), shape (batch, n_in)
        """
        # (n_out, batch) @ (batch, n_in) -> (n_out, n_in)
        self.dW = delta.T @ self.a_prev
        # (n_out,)
        self.db = np.sum(delta, axis=0)

        # (batch, n_out) @ (n_out, n_in) -> (batch, n_in)
        return delta @ self.W


class MLP:
    """Multilayer perceptron with configurable architecture, activations, and loss."""

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[Union[Activation, str, None]],
        init_method: str = "he",
    ):
        """
        Args:
            layer_sizes: List of layer widths, e.g. [784, 256, 128, 10].
                         Length L+1 where L is number of layers.
            activations: List of activations, one per layer (length L).
                         Each is an Activation instance, "softmax", or None.
            init_method: "he" (default) or "xavier"
        """
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"Need {len(layer_sizes) - 1} activations for {len(layer_sizes)} layer sizes, "
                f"got {len(activations)}"
            )

        self.layer_sizes = layer_sizes
        self.layers: List[Layer] = []

        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i + 1], activations[i], init_method)
            )

        last_act = self.layers[-1].activation
        self._is_classification = isinstance(last_act, str) and last_act == "softmax"

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.

        Args:
            X: Input, shape (batch, n_features)

        Returns:
            Output activations, shape (batch, n_out)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, y: np.ndarray) -> float:
        """
        Backpropagation through all layers. Must be called after forward().

        Args:
            y: Targets -- one-hot (n, K) for classification, (n, d_out) for regression

        Returns:
            Loss value (cross-entropy or MSE)
        """
        output_layer = self.layers[-1]
        a_out = output_layer.a
        n = y.shape[0]

        if self._is_classification:
            loss = cross_entropy_loss(a_out, y)
            # Softmax + cross-entropy combined gradient is dL/dz directly
            delta = (a_out - y) / n
        else:
            loss = mse_loss(a_out, y)
            # dL/da for MSE
            da_out = (a_out - y) / n
            # Propagate through output activation to get dL/dz
            if output_layer.activation is not None and not isinstance(output_layer.activation, str):
                delta = output_layer.activation.backward(da_out)
            else:
                delta = da_out

        da_prev = output_layer.backward(delta)

        # Backward through hidden layers
        for layer in reversed(self.layers[:-1]):
            # da_prev is dL/da for this layer; propagate through activation to get dL/dz
            if layer.activation is not None and not isinstance(layer.activation, str):
                delta = layer.activation.backward(da_prev)
            else:
                delta = da_prev
            da_prev = layer.backward(delta)

        return loss

    def update(self, learning_rate: float) -> None:
        """Apply vanilla gradient descent to all parameters."""
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> List[float]:
        """
        Train the network with mini-batch gradient descent.

        Args:
            X: Training inputs, shape (n, n_features)
            y: Labels -- integer class indices (n,) for classification,
               or float targets (n,) or (n, d_out) for regression
            epochs: Number of training epochs
            learning_rate: Step size for gradient descent
            batch_size: Mini-batch size, None for full-batch
            verbose: Print loss every 10% of epochs

        Returns:
            Loss history (average loss per epoch)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = X.shape[0]

        if self._is_classification:
            y_int = y.astype(int)
            num_classes = self.layers[-1].n_out
            Y = one_hot_encode(y_int, num_classes)
        else:
            Y = y.copy()
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)

        if batch_size is None:
            batch_size = n

        history: List[float] = []
        verbose_interval = max(1, epochs // 10)

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            epoch_losses: List[float] = []

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch = X[batch_idx]
                Y_batch = Y[batch_idx]

                self.forward(X_batch)
                loss = self.backward(Y_batch)
                self.update(learning_rate)
                epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses))
            history.append(avg_loss)

            if verbose and epoch % verbose_interval == 0:
                print(f"Epoch {epoch}/{epochs}: loss = {avg_loss:.6f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predictions.

        Args:
            X: Input, shape (n, n_features)

        Returns:
            Class indices (n,) for classification, raw output (n, d_out) for regression
        """
        output = self.forward(X)
        if self._is_classification:
            return np.argmax(output, axis=1)
        return output

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability distributions (softmax output).

        Args:
            X: Input, shape (n, n_features)

        Returns:
            Probabilities, shape (n, K)
        """
        return self.forward(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Input, shape (n, n_features)
            y: True class labels, shape (n,)

        Returns:
            Accuracy between 0 and 1
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y, dtype=int)))
