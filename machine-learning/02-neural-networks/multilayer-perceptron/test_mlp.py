"""Tests for multilayer perceptron."""

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from implementation import (
    MLP, Layer, softmax, cross_entropy_loss, mse_loss, one_hot_encode,
    ReLU, LeakyReLU, Sigmoid, Tanh, GELU, SiLU,
)


class TestSoftmax(unittest.TestCase):

    def test_sums_to_one(self):
        z = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        p = softmax(z)
        np.testing.assert_allclose(np.sum(p, axis=1), [1.0, 1.0])

    def test_1d(self):
        z = np.array([1.0, 2.0, 3.0])
        p = softmax(z)
        self.assertAlmostEqual(np.sum(p), 1.0)

    def test_large_logits(self):
        """Softmax should handle large inputs without overflow."""
        z = np.array([1000.0, 1001.0, 999.0])
        p = softmax(z)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(np.sum(p), 1.0)

    def test_uniform(self):
        z = np.array([5.0, 5.0, 5.0])
        p = softmax(z)
        np.testing.assert_allclose(p, [1/3, 1/3, 1/3])


class TestLossFunctions(unittest.TestCase):

    def test_cross_entropy_perfect(self):
        """Perfect predictions should have near-zero loss."""
        Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        P = np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005], [0.005, 0.005, 0.99]])
        loss = cross_entropy_loss(P, Y)
        self.assertLess(loss, 0.05)

    def test_cross_entropy_bad(self):
        """Bad predictions should have high loss."""
        Y = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        P = np.array([[0.01, 0.49, 0.50], [0.50, 0.01, 0.49]])
        loss = cross_entropy_loss(P, Y)
        self.assertGreater(loss, 1.0)

    def test_cross_entropy_log_zero(self):
        """Should handle zero predictions without NaN."""
        Y = np.array([[1, 0], [0, 1]], dtype=float)
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        loss = cross_entropy_loss(P, Y)
        self.assertTrue(np.isfinite(loss))

    def test_mse_zero(self):
        y = np.array([[1.0], [2.0]])
        loss = mse_loss(y, y)
        self.assertAlmostEqual(loss, 0.0)

    def test_mse_known(self):
        pred = np.array([[1.0], [2.0]])
        target = np.array([[2.0], [4.0]])
        # 0.5 * (1 + 4) / 2 = 1.25
        loss = mse_loss(pred, target)
        self.assertAlmostEqual(loss, 1.25)


class TestOneHotEncode(unittest.TestCase):

    def test_basic(self):
        y = np.array([0, 1, 2])
        Y = one_hot_encode(y, 3)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_array_equal(Y, expected)


class TestLayer(unittest.TestCase):

    def test_forward_shape(self):
        layer = Layer(4, 3, ReLU())
        x = np.random.randn(5, 4)
        out = layer.forward(x)
        self.assertEqual(out.shape, (5, 3))

    def test_softmax_output(self):
        layer = Layer(3, 2, "softmax")
        x = np.random.randn(4, 3)
        out = layer.forward(x)
        np.testing.assert_allclose(np.sum(out, axis=1), np.ones(4), atol=1e-10)

    def test_linear_output(self):
        layer = Layer(3, 2, None)
        x = np.random.randn(4, 3)
        out = layer.forward(x)
        expected = x @ layer.W.T + layer.b
        np.testing.assert_allclose(out, expected)


class TestWeightInitialization(unittest.TestCase):

    def test_he_variance(self):
        """He-initialized weights should have variance ~2/n_in."""
        np.random.seed(42)
        n_in = 1000
        layer = Layer(n_in, 500, ReLU(), init_method="he")
        expected_var = 2.0 / n_in
        actual_var = np.var(layer.W)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.15)

    def test_xavier_variance(self):
        """Xavier-initialized weights should have variance ~2/(n_in + n_out)."""
        np.random.seed(42)
        n_in, n_out = 1000, 500
        layer = Layer(n_in, n_out, Sigmoid(), init_method="xavier")
        expected_var = 2.0 / (n_in + n_out)
        actual_var = np.var(layer.W)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.15)

    def test_bias_zero(self):
        layer = Layer(10, 5, ReLU())
        np.testing.assert_array_equal(layer.b, np.zeros(5))

    def test_he_activation_stability(self):
        """He-initialized deep ReLU network should have stable activation variance."""
        np.random.seed(42)
        sizes = [256, 256, 256, 256, 256, 256]
        activations = [ReLU() for _ in range(len(sizes) - 1)]
        mlp = MLP(sizes, activations, init_method="he")

        x = np.random.randn(100, 256)
        a = x
        variances = []
        for layer in mlp.layers:
            a = layer.forward(a)
            variances.append(np.var(a))

        for v in variances:
            self.assertGreater(v, 0.01, "Activation variance too small (vanishing)")
            self.assertLess(v, 100.0, "Activation variance too large (exploding)")

    def test_bad_init_explodes(self):
        """Standard normal init on deep network should have exploding activations."""
        np.random.seed(42)
        n_layers = 5
        width = 256
        sizes = [width] * (n_layers + 1)

        layers = []
        for i in range(n_layers):
            layer = Layer(width, width, ReLU())
            layer.W = np.random.randn(width, width)  # var=1 instead of 2/n_in
            layers.append(layer)

        x = np.random.randn(10, width)
        a = x
        for layer in layers:
            a = layer.forward(a)

        final_var = np.var(a)
        self.assertGreater(final_var, 1e6, "Bad init should cause exploding activations")


class TestGradientCheck(unittest.TestCase):
    """Numerical gradient checking -- the most critical tests."""

    def _numerical_gradient_check(self, mlp, X, Y, eps=1e-5, tol=1e-5):
        """Check analytical gradients against numerical for all parameters."""
        mlp.forward(X)
        mlp.backward(Y)

        for l_idx, layer in enumerate(mlp.layers):
            for param_name in ["W", "b"]:
                param = getattr(layer, param_name)
                grad = getattr(layer, f"d{param_name}")

                it = np.nditer(param, flags=["multi_index"])
                max_rel_error = 0.0
                while not it.finished:
                    idx = it.multi_index
                    old_val = param[idx]

                    param[idx] = old_val + eps
                    mlp.forward(X)
                    if mlp._is_classification:
                        loss_plus = cross_entropy_loss(mlp.layers[-1].a, Y)
                    else:
                        loss_plus = mse_loss(mlp.layers[-1].a, Y)

                    param[idx] = old_val - eps
                    mlp.forward(X)
                    if mlp._is_classification:
                        loss_minus = cross_entropy_loss(mlp.layers[-1].a, Y)
                    else:
                        loss_minus = mse_loss(mlp.layers[-1].a, Y)

                    param[idx] = old_val

                    numerical = (loss_plus - loss_minus) / (2 * eps)
                    analytical = grad[idx]

                    # Skip elements where both gradients are negligible
                    # (dead ReLU neurons cause non-differentiable points)
                    if abs(analytical) < 1e-7 and abs(numerical) < 1e-7:
                        it.iternext()
                        continue

                    denom = abs(analytical) + abs(numerical) + 1e-8
                    rel_error = abs(analytical - numerical) / denom
                    max_rel_error = max(max_rel_error, rel_error)

                    it.iternext()

                self.assertLess(
                    max_rel_error, tol,
                    f"Gradient check failed for layer {l_idx} {param_name}: "
                    f"max_rel_error={max_rel_error:.2e}"
                )

        # Restore forward cache
        mlp.forward(X)

    def test_gradient_relu_classification(self):
        np.random.seed(42)
        mlp = MLP([3, 4, 2], [ReLU(), "softmax"])
        X = np.random.randn(5, 3)
        Y = one_hot_encode(np.array([0, 1, 0, 1, 0]), 2)
        self._numerical_gradient_check(mlp, X, Y)

    def test_gradient_sigmoid_classification(self):
        np.random.seed(42)
        mlp = MLP([3, 4, 2], [Sigmoid(), "softmax"], init_method="xavier")
        X = np.random.randn(5, 3)
        Y = one_hot_encode(np.array([0, 1, 0, 1, 0]), 2)
        self._numerical_gradient_check(mlp, X, Y)

    def test_gradient_tanh_classification(self):
        np.random.seed(42)
        mlp = MLP([3, 4, 2], [Tanh(), "softmax"], init_method="xavier")
        X = np.random.randn(5, 3)
        Y = one_hot_encode(np.array([0, 1, 0, 1, 0]), 2)
        self._numerical_gradient_check(mlp, X, Y)

    def test_gradient_gelu_classification(self):
        np.random.seed(42)
        mlp = MLP([3, 4, 2], [GELU(), "softmax"])
        X = np.random.randn(5, 3)
        Y = one_hot_encode(np.array([0, 1, 0, 1, 0]), 2)
        self._numerical_gradient_check(mlp, X, Y)

    def test_gradient_silu_classification(self):
        np.random.seed(42)
        mlp = MLP([3, 4, 2], [SiLU(), "softmax"])
        X = np.random.randn(5, 3)
        Y = one_hot_encode(np.array([0, 1, 0, 1, 0]), 2)
        self._numerical_gradient_check(mlp, X, Y)

    def test_gradient_leaky_relu_classification(self):
        np.random.seed(42)
        mlp = MLP([3, 4, 2], [LeakyReLU(0.1), "softmax"])
        X = np.random.randn(5, 3)
        Y = one_hot_encode(np.array([0, 1, 0, 1, 0]), 2)
        self._numerical_gradient_check(mlp, X, Y)

    def test_gradient_deep_network(self):
        """Deep network gradient check uses Tanh (smooth) to avoid ReLU kink artifacts."""
        np.random.seed(42)
        mlp = MLP([3, 5, 4, 3, 2], [Tanh(), Tanh(), Tanh(), "softmax"], init_method="xavier")
        X = np.random.randn(5, 3)
        Y = one_hot_encode(np.array([0, 1, 0, 1, 0]), 2)
        self._numerical_gradient_check(mlp, X, Y)

    def test_gradient_mse_regression(self):
        np.random.seed(42)
        mlp = MLP([3, 4, 1], [ReLU(), None])
        X = np.random.randn(5, 3)
        Y = np.random.randn(5, 1)
        self._numerical_gradient_check(mlp, X, Y)

    def test_gradient_sigmoid_output_mse(self):
        """Non-linear output activation with MSE must propagate through activation backward."""
        np.random.seed(42)
        mlp = MLP([3, 4, 1], [ReLU(), Sigmoid()])
        X = np.random.randn(5, 3)
        Y = np.random.rand(5, 1) * 0.8 + 0.1  # targets in sigmoid range
        self._numerical_gradient_check(mlp, X, Y)

    def test_gradient_tanh_output_mse(self):
        """Tanh output with MSE regression."""
        np.random.seed(42)
        mlp = MLP([3, 4, 1], [Tanh(), Tanh()], init_method="xavier")
        X = np.random.randn(5, 3)
        Y = np.random.randn(5, 1) * 0.5
        self._numerical_gradient_check(mlp, X, Y)

    def test_gradient_mixed_activations(self):
        np.random.seed(42)
        mlp = MLP([3, 4, 5, 2], [ReLU(), GELU(), "softmax"])
        X = np.random.randn(5, 3)
        Y = one_hot_encode(np.array([0, 1, 0, 1, 0]), 2)
        self._numerical_gradient_check(mlp, X, Y)


class TestXOR(unittest.TestCase):

    def test_xor_classification(self):
        """XOR is the simplest non-linearly-separable problem."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])

        np.random.seed(42)
        mlp = MLP([2, 8, 2], [ReLU(), "softmax"])
        mlp.fit(X, y, epochs=1000, learning_rate=0.5, verbose=False)

        preds = mlp.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_xor_sigmoid_output(self):
        """XOR with sigmoid output (binary)."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y_targets = np.array([[0], [1], [1], [0]], dtype=float)

        np.random.seed(0)
        mlp = MLP([2, 8, 1], [ReLU(), Sigmoid()])

        for _ in range(2000):
            mlp.forward(X)
            mlp.backward(y_targets)
            mlp.update(learning_rate=1.0)

        output = mlp.forward(X)
        preds = (output > 0.5).astype(int).flatten()
        np.testing.assert_array_equal(preds, [0, 1, 1, 0])


class TestShapes(unittest.TestCase):

    def test_forward_shapes(self):
        mlp = MLP([10, 8, 6, 3], [ReLU(), ReLU(), "softmax"])
        X = np.random.randn(20, 10)
        out = mlp.forward(X)
        self.assertEqual(out.shape, (20, 3))

    def test_single_sample(self):
        mlp = MLP([5, 3, 2], [ReLU(), "softmax"])
        X = np.random.randn(1, 5)
        out = mlp.forward(X)
        self.assertEqual(out.shape, (1, 2))

    def test_large_batch(self):
        mlp = MLP([5, 3, 2], [ReLU(), "softmax"])
        X = np.random.randn(1000, 5)
        out = mlp.forward(X)
        self.assertEqual(out.shape, (1000, 2))

    def test_gradient_shapes(self):
        mlp = MLP([4, 5, 3], [ReLU(), "softmax"])
        X = np.random.randn(10, 4)
        Y = one_hot_encode(np.random.randint(0, 3, 10), 3)
        mlp.forward(X)
        mlp.backward(Y)

        self.assertEqual(mlp.layers[0].dW.shape, (5, 4))
        self.assertEqual(mlp.layers[0].db.shape, (5,))
        self.assertEqual(mlp.layers[1].dW.shape, (3, 5))
        self.assertEqual(mlp.layers[1].db.shape, (3,))


class TestConvergence(unittest.TestCase):

    def test_linearly_separable(self):
        """MLP should easily solve linearly separable problem."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        mlp = MLP([2, 4, 2], [ReLU(), "softmax"])
        mlp.fit(X, y, epochs=200, learning_rate=0.1)

        accuracy = mlp.score(X, y)
        self.assertGreater(accuracy, 0.95)

    def test_loss_decreases(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)

        mlp = MLP([3, 8, 2], [ReLU(), "softmax"])
        history = mlp.fit(X, y, epochs=100, learning_rate=0.05)

        self.assertLess(history[-1], history[0])

    def test_spiral_dataset(self):
        """Spiral requires nonlinear decision boundary."""
        np.random.seed(42)
        n_per_class = 100
        t = np.linspace(0, 4 * np.pi, n_per_class)

        r1 = t / (4 * np.pi)
        x1 = np.column_stack([r1 * np.cos(t) + np.random.randn(n_per_class) * 0.05,
                               r1 * np.sin(t) + np.random.randn(n_per_class) * 0.05])

        r2 = t / (4 * np.pi)
        x2 = np.column_stack([r2 * np.cos(t + np.pi) + np.random.randn(n_per_class) * 0.05,
                               r2 * np.sin(t + np.pi) + np.random.randn(n_per_class) * 0.05])

        X = np.vstack([x1, x2])
        y = np.array([0] * n_per_class + [1] * n_per_class)

        mlp = MLP([2, 64, 32, 2], [ReLU(), ReLU(), "softmax"])
        mlp.fit(X, y, epochs=1000, learning_rate=0.5, batch_size=32)

        accuracy = mlp.score(X, y)
        self.assertGreater(accuracy, 0.85)

    def test_regression_sinx(self):
        """MLP with linear output should approximate sin(x)."""
        np.random.seed(42)
        X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
        y = np.sin(X)

        mlp = MLP([1, 32, 16, 1], [ReLU(), ReLU(), None])
        mlp.fit(X, y, epochs=500, learning_rate=0.01, batch_size=32)

        y_pred = mlp.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        self.assertLess(mse, 0.1)


class TestNumericalStability(unittest.TestCase):

    def test_softmax_large_logits_in_network(self):
        mlp = MLP([2, 3], ["softmax"])
        # W shape is (n_out, n_in) = (3, 2)
        mlp.layers[0].W = np.array([[100, 100], [200, 200], [300, 300]], dtype=float)
        X = np.array([[10.0, 10.0]])
        out = mlp.forward(X)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_zero_input(self):
        mlp = MLP([3, 4, 2], [ReLU(), "softmax"])
        X = np.zeros((5, 3))
        out = mlp.forward(X)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_deep_network_no_nan(self):
        np.random.seed(42)
        mlp = MLP([3, 8, 8, 8, 8, 8, 2], [ReLU()] * 5 + ["softmax"])
        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, 10)
        history = mlp.fit(X, y, epochs=50, learning_rate=0.01)
        self.assertTrue(all(np.isfinite(l) for l in history))


class TestMiniBatch(unittest.TestCase):

    def test_batch_size_1(self):
        """SGD (batch_size=1) should converge."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = (X[:, 0] > 0).astype(int)

        mlp = MLP([2, 4, 2], [ReLU(), "softmax"])
        history = mlp.fit(X, y, epochs=200, learning_rate=0.01, batch_size=1)

        self.assertLess(history[-1], history[0])

    def test_full_batch(self):
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = (X[:, 0] > 0).astype(int)

        mlp = MLP([2, 4, 2], [ReLU(), "softmax"])
        history = mlp.fit(X, y, epochs=100, learning_rate=0.1, batch_size=None)

        self.assertLess(history[-1], history[0])

    def test_indivisible_batch_size(self):
        """Last batch can be smaller than batch_size."""
        np.random.seed(42)
        X = np.random.randn(17, 2)
        y = np.random.randint(0, 2, 17)

        mlp = MLP([2, 4, 2], [ReLU(), "softmax"])
        history = mlp.fit(X, y, epochs=50, learning_rate=0.1, batch_size=5)
        self.assertEqual(len(history), 50)


class TestEdgeCases(unittest.TestCase):

    def test_single_hidden_layer(self):
        mlp = MLP([4, 3, 2], [ReLU(), "softmax"])
        X = np.random.randn(5, 4)
        out = mlp.forward(X)
        self.assertEqual(out.shape, (5, 2))

    def test_five_hidden_layers(self):
        mlp = MLP([4, 8, 8, 8, 8, 8, 2], [ReLU()] * 5 + ["softmax"])
        X = np.random.randn(5, 4)
        out = mlp.forward(X)
        self.assertEqual(out.shape, (5, 2))

    def test_wide_layer(self):
        mlp = MLP([4, 1024, 2], [ReLU(), "softmax"])
        X = np.random.randn(5, 4)
        out = mlp.forward(X)
        self.assertEqual(out.shape, (5, 2))

    def test_narrow_bottleneck(self):
        mlp = MLP([100, 2, 100], [ReLU(), None])
        X = np.random.randn(5, 100)
        out = mlp.forward(X)
        self.assertEqual(out.shape, (5, 100))

    def test_single_class(self):
        """All samples same class -- loss should go to zero."""
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = np.zeros(20, dtype=int)

        mlp = MLP([3, 4, 2], [ReLU(), "softmax"])
        history = mlp.fit(X, y, epochs=200, learning_rate=0.1)
        preds = mlp.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_two_classes_softmax(self):
        """Two-class softmax should work like logistic regression."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)

        mlp = MLP([2, 2], ["softmax"])
        mlp.fit(X, y, epochs=200, learning_rate=0.5)

        accuracy = mlp.score(X, y)
        self.assertGreater(accuracy, 0.9)


class TestActivationsIntegration(unittest.TestCase):

    def test_all_activations_train(self):
        """Every activation from the module should work in an MLP."""
        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.random.randint(0, 2, 30)

        activation_classes = [ReLU, LeakyReLU, Sigmoid, Tanh, GELU, SiLU]
        for act_cls in activation_classes:
            act = act_cls() if act_cls not in (LeakyReLU,) else act_cls(0.01)
            init = "xavier" if act_cls in (Sigmoid, Tanh) else "he"
            mlp = MLP([3, 4, 2], [act, "softmax"], init_method=init)
            history = mlp.fit(X, y, epochs=50, learning_rate=0.05)
            self.assertTrue(
                all(np.isfinite(l) for l in history),
                f"NaN loss with activation {act_cls.__name__}"
            )

    def test_mixed_activations_per_layer(self):
        np.random.seed(42)
        mlp = MLP([3, 4, 5, 2], [ReLU(), GELU(), "softmax"])
        X = np.random.randn(10, 3)
        Y = one_hot_encode(np.random.randint(0, 2, 10), 2)
        mlp.forward(X)
        loss = mlp.backward(Y)
        self.assertTrue(np.isfinite(loss))


class TestPredictAndScore(unittest.TestCase):

    def test_predict_proba_sums_to_one(self):
        mlp = MLP([3, 4, 3], [ReLU(), "softmax"])
        X = np.random.randn(10, 3)
        proba = mlp.predict_proba(X)
        np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(10), atol=1e-10)

    def test_predict_returns_integers(self):
        mlp = MLP([3, 4, 3], [ReLU(), "softmax"])
        X = np.random.randn(10, 3)
        preds = mlp.predict(X)
        self.assertEqual(preds.shape, (10,))
        self.assertTrue(all(p in [0, 1, 2] for p in preds))

    def test_score_perfect(self):
        """Score should be 1.0 when predictions match labels."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = (X[:, 0] > 0).astype(int)

        mlp = MLP([2, 8, 2], [ReLU(), "softmax"])
        mlp.fit(X, y, epochs=500, learning_rate=0.1)

        self.assertGreater(mlp.score(X, y), 0.95)


class TestConstructorValidation(unittest.TestCase):

    def test_wrong_activation_count(self):
        with self.assertRaises(ValueError):
            MLP([3, 4, 2], [ReLU()])  # needs 2 activations, got 1


if __name__ == "__main__":
    unittest.main()
