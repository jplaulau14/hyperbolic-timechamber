"""Tests for softmax regression."""

import unittest
import numpy as np
from implementation import softmax, cross_entropy_loss, one_hot_encode, SoftmaxRegression


class TestSoftmax(unittest.TestCase):
    """Tests for the softmax function."""

    def test_large_logits_no_overflow(self):
        """softmax([1000, 1000, 1000]) should not overflow."""
        z = np.array([1000.0, 1000.0, 1000.0])
        result = softmax(z)
        np.testing.assert_array_almost_equal(result, [1/3, 1/3, 1/3], decimal=5)

    def test_mixed_extreme_logits(self):
        """softmax([1000, 0, 0]) should return approximately [1, 0, 0]."""
        z = np.array([1000.0, 0.0, 0.0])
        result = softmax(z)
        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[1], 0.0, places=5)
        self.assertAlmostEqual(result[2], 0.0, places=5)

    def test_negative_logits_no_underflow(self):
        """softmax([-1000, -999, -998]) should work without underflow."""
        z = np.array([-1000.0, -999.0, -998.0])
        result = softmax(z)
        self.assertTrue(np.all(result > 0))
        self.assertAlmostEqual(np.sum(result), 1.0, places=10)

    def test_zero_logits(self):
        """softmax([0, 0, 0]) should return [1/3, 1/3, 1/3]."""
        z = np.array([0.0, 0.0, 0.0])
        result = softmax(z)
        np.testing.assert_array_almost_equal(result, [1/3, 1/3, 1/3], decimal=10)

    def test_sum_to_one_1d(self):
        """1D softmax output should sum to 1."""
        z = np.array([1.0, 2.0, 3.0, 4.0])
        result = softmax(z)
        self.assertAlmostEqual(np.sum(result), 1.0, places=10)

    def test_sum_to_one_2d(self):
        """2D softmax output rows should sum to 1."""
        z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = softmax(z)
        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0, 1.0], decimal=10)

    def test_positive_outputs(self):
        """All softmax outputs should be strictly positive."""
        z = np.array([-5.0, 0.0, 5.0])
        result = softmax(z)
        self.assertTrue(np.all(result > 0))

    def test_monotonicity(self):
        """Larger logits should produce larger probabilities."""
        z = np.array([1.0, 2.0, 3.0])
        result = softmax(z)
        self.assertLess(result[0], result[1])
        self.assertLess(result[1], result[2])

    def test_invariance_to_constant_shift(self):
        """softmax(z) should equal softmax(z + c) for any constant c."""
        z = np.array([1.0, 2.0, 3.0])
        result1 = softmax(z)
        result2 = softmax(z + 100.0)
        result3 = softmax(z - 100.0)
        np.testing.assert_array_almost_equal(result1, result2, decimal=10)
        np.testing.assert_array_almost_equal(result1, result3, decimal=10)


class TestCrossEntropyLoss(unittest.TestCase):
    """Tests for cross-entropy loss."""

    def test_perfect_prediction(self):
        """Perfect prediction should have loss close to 0."""
        P = np.array([[0.99999, 0.00001, 0.00000]])
        Y = np.array([[1.0, 0.0, 0.0]])
        loss = cross_entropy_loss(P, Y)
        self.assertLess(loss, 0.001)

    def test_confident_wrong_prediction(self):
        """Confident wrong prediction should have high loss."""
        P = np.array([[0.01, 0.98, 0.01]])
        Y = np.array([[1.0, 0.0, 0.0]])
        loss = cross_entropy_loss(P, Y)
        self.assertGreater(loss, 4.0)

    def test_uniform_prediction(self):
        """Uniform prediction P=[1/3, 1/3, 1/3] should give loss = log(3)."""
        P = np.array([[1/3, 1/3, 1/3]])
        Y = np.array([[1.0, 0.0, 0.0]])
        loss = cross_entropy_loss(P, Y)
        expected = np.log(3)
        self.assertAlmostEqual(loss, expected, places=5)

    def test_batch_loss(self):
        """Test loss computation with multiple samples."""
        P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        Y = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        loss = cross_entropy_loss(P, Y)
        expected = -(np.log(0.7) + np.log(0.8) + np.log(0.8)) / 3
        self.assertAlmostEqual(loss, expected, places=5)


class TestOneHotEncode(unittest.TestCase):
    """Tests for one-hot encoding."""

    def test_basic_encoding(self):
        """Test basic one-hot encoding."""
        y = np.array([0, 1, 2])
        result = one_hot_encode(y, 3)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_repeated_classes(self):
        """Test encoding with repeated class labels."""
        y = np.array([0, 0, 1, 1, 2])
        result = one_hot_encode(y, 3)
        expected = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_single_sample(self):
        """Test encoding with single sample."""
        y = np.array([2])
        result = one_hot_encode(y, 4)
        expected = np.array([[0, 0, 1, 0]])
        np.testing.assert_array_equal(result, expected)


class TestSoftmaxRegression(unittest.TestCase):
    """Tests for SoftmaxRegression class."""

    def test_forward_shapes(self):
        """Test that forward pass produces correct shapes."""
        model = SoftmaxRegression(num_features=4, num_classes=3)
        X = np.random.randn(10, 4)
        Z, P = model.forward(X)
        self.assertEqual(Z.shape, (10, 3))
        self.assertEqual(P.shape, (10, 3))

    def test_forward_probabilities_valid(self):
        """Test that forward pass produces valid probabilities."""
        model = SoftmaxRegression(num_features=4, num_classes=3)
        X = np.random.randn(10, 4)
        _, P = model.forward(X)
        self.assertTrue(np.all(P > 0))
        np.testing.assert_array_almost_equal(np.sum(P, axis=1), np.ones(10), decimal=10)

    def test_backward_shapes(self):
        """Test that backward pass produces correct gradient shapes."""
        model = SoftmaxRegression(num_features=4, num_classes=3)
        X = np.random.randn(10, 4)
        _, P = model.forward(X)
        Y = one_hot_encode(np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]), 3)
        dW, db = model.backward(X, P, Y)
        self.assertEqual(dW.shape, (4, 3))
        self.assertEqual(db.shape, (3,))

    def test_numerical_gradient_check(self):
        """Compare analytical gradient to numerical gradient."""
        np.random.seed(42)
        model = SoftmaxRegression(num_features=3, num_classes=3, learning_rate=0.1)
        X = np.random.randn(5, 3)
        y = np.array([0, 1, 2, 0, 1])
        Y = one_hot_encode(y, 3)

        _, P = model.forward(X)
        dW_analytical, db_analytical = model.backward(X, P, Y)

        eps = 1e-5
        dW_numerical = np.zeros_like(model.W)

        for i in range(model.W.shape[0]):
            for j in range(model.W.shape[1]):
                model.W[i, j] += eps
                _, P_plus = model.forward(X)
                loss_plus = cross_entropy_loss(P_plus, Y)

                model.W[i, j] -= 2 * eps
                _, P_minus = model.forward(X)
                loss_minus = cross_entropy_loss(P_minus, Y)

                dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)
                model.W[i, j] += eps

        np.testing.assert_array_almost_equal(dW_analytical, dW_numerical, decimal=5)

        db_numerical = np.zeros_like(model.b)
        for j in range(model.b.shape[0]):
            model.b[j] += eps
            _, P_plus = model.forward(X)
            loss_plus = cross_entropy_loss(P_plus, Y)

            model.b[j] -= 2 * eps
            _, P_minus = model.forward(X)
            loss_minus = cross_entropy_loss(P_minus, Y)

            db_numerical[j] = (loss_plus - loss_minus) / (2 * eps)
            model.b[j] += eps

        np.testing.assert_array_almost_equal(db_analytical, db_numerical, decimal=5)

    def test_linearly_separable_data(self):
        """Test on linearly separable 3-class problem."""
        np.random.seed(42)
        n_per_class = 50
        X0 = np.random.randn(n_per_class, 2) + np.array([0, 3])
        X1 = np.random.randn(n_per_class, 2) + np.array([-3, -1])
        X2 = np.random.randn(n_per_class, 2) + np.array([3, -1])

        X = np.vstack([X0, X1, X2])
        y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

        model = SoftmaxRegression(num_features=2, num_classes=3, learning_rate=0.1)
        model.fit(X, y, epochs=500)

        accuracy = model.score(X, y)
        self.assertGreater(accuracy, 0.95)

    def test_loss_decreases(self):
        """Test that loss decreases during training."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)

        model = SoftmaxRegression(num_features=4, num_classes=3, learning_rate=0.1)
        model.fit(X, y, epochs=100)

        self.assertLess(model.history[-1], model.history[0])

    def test_predict_vs_predict_proba(self):
        """argmax(predict_proba(X)) should equal predict(X)."""
        np.random.seed(42)
        model = SoftmaxRegression(num_features=4, num_classes=3, learning_rate=0.1)
        X = np.random.randn(20, 4)
        y = np.random.randint(0, 3, 20)
        model.fit(X, y, epochs=100)

        proba = model.predict_proba(X)
        pred = model.predict(X)
        pred_from_proba = np.argmax(proba, axis=1)

        np.testing.assert_array_equal(pred, pred_from_proba)

    def test_single_sample(self):
        """Training with n=1 should work."""
        model = SoftmaxRegression(num_features=2, num_classes=3, learning_rate=0.1)
        X = np.array([[1.0, 2.0]])
        y = np.array([1])
        model.fit(X, y, epochs=100)
        pred = model.predict(X)
        self.assertEqual(pred.shape, (1,))

    def test_two_classes_like_logistic(self):
        """Two classes should behave reasonably."""
        np.random.seed(42)
        X0 = np.random.randn(30, 2) + np.array([-2, 0])
        X1 = np.random.randn(30, 2) + np.array([2, 0])
        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)

        model = SoftmaxRegression(num_features=2, num_classes=2, learning_rate=0.5)
        model.fit(X, y, epochs=500)

        accuracy = model.score(X, y)
        self.assertGreater(accuracy, 0.90)

    def test_single_feature(self):
        """d=1 (single feature) should work."""
        np.random.seed(42)
        X = np.concatenate([
            np.random.randn(20, 1) - 3,
            np.random.randn(20, 1),
            np.random.randn(20, 1) + 3,
        ])
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)

        model = SoftmaxRegression(num_features=1, num_classes=3, learning_rate=0.5)
        model.fit(X, y, epochs=500)

        accuracy = model.score(X, y)
        self.assertGreater(accuracy, 0.80)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_softmax_1d_vs_2d_consistency(self):
        """1D and 2D softmax should produce consistent results."""
        z_1d = np.array([1.0, 2.0, 3.0])
        z_2d = np.array([[1.0, 2.0, 3.0]])

        result_1d = softmax(z_1d)
        result_2d = softmax(z_2d)

        np.testing.assert_array_almost_equal(result_1d, result_2d[0], decimal=10)

    def test_cross_entropy_epsilon_prevents_nan(self):
        """Cross-entropy should not produce NaN with near-zero probabilities."""
        P = np.array([[1e-20, 1.0 - 1e-20, 0.0]])
        Y = np.array([[1.0, 0.0, 0.0]])
        loss = cross_entropy_loss(P, Y)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))


if __name__ == "__main__":
    unittest.main()
