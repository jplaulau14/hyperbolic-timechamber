"""Tests for linear regression implementation."""

import unittest
import numpy as np
from implementation import LinearRegression, mse_loss


class TestMSELoss(unittest.TestCase):
    """Tests for MSE loss function."""

    def test_perfect_predictions(self):
        """MSE is zero when predictions match targets exactly."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mse_loss(y_true, y_pred), 0.0)

    def test_known_value(self):
        """MSE with known hand-computed value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # Errors: [1, 1, 1], squared: [1, 1, 1], sum: 3, loss: 3 / (2 * 3) = 0.5
        self.assertAlmostEqual(mse_loss(y_true, y_pred), 0.5)


class TestLinearRegressionBasic(unittest.TestCase):
    """Basic correctness tests."""

    def test_perfect_linear_fit_1d(self):
        """Recover known weights for y = 2x + 3."""
        np.random.seed(42)
        X = np.random.randn(100)
        y = 2 * X + 3

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        self.assertAlmostEqual(model.w[0], 2.0, places=5)
        self.assertAlmostEqual(model.b, 3.0, places=5)

    def test_perfect_linear_fit_gradient_descent(self):
        """Gradient descent recovers known weights."""
        np.random.seed(42)
        X = np.random.randn(100)
        y = 2 * X + 3

        model = LinearRegression(
            method="gradient_descent", learning_rate=0.1, n_iterations=5000, tolerance=1e-12
        )
        model.fit(X, y)

        self.assertAlmostEqual(model.w[0], 2.0, places=2)
        self.assertAlmostEqual(model.b, 3.0, places=2)

    def test_multiple_features(self):
        """Recover coefficients for y = x1 + 2*x2 + 3*x3 + 4."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2] + 4

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        np.testing.assert_array_almost_equal(model.w, [1.0, 2.0, 3.0], decimal=5)
        self.assertAlmostEqual(model.b, 4.0, places=5)

    def test_gradient_descent_vs_normal_equation(self):
        """Both methods produce same weights within tolerance."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + 2.0

        model_gd = LinearRegression(
            method="gradient_descent", learning_rate=0.1, n_iterations=10000, tolerance=1e-12
        )
        model_gd.fit(X, y)

        model_ne = LinearRegression(method="normal_equation")
        model_ne.fit(X, y)

        np.testing.assert_array_almost_equal(model_gd.w, model_ne.w, decimal=2)
        self.assertAlmostEqual(model_gd.b, model_ne.b, places=2)


class TestPredictionShapes(unittest.TestCase):
    """Tests for prediction output shapes."""

    def test_predict_1d_input(self):
        """Prediction shape is correct for 1D input."""
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        y_pred = model.predict(X)
        self.assertEqual(y_pred.shape, (3,))

    def test_predict_2d_input(self):
        """Prediction shape is correct for 2D input."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        y_pred = model.predict(X)
        self.assertEqual(y_pred.shape, (50,))

    def test_predict_single_sample(self):
        """Prediction works for single sample."""
        X_train = np.random.randn(50, 2)
        y_train = np.random.randn(50)

        model = LinearRegression(method="normal_equation")
        model.fit(X_train, y_train)

        X_test = np.array([[1.0, 2.0]])
        y_pred = model.predict(X_test)
        self.assertEqual(y_pred.shape, (1,))


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_single_sample(self):
        """Fitting with single sample works (underdetermined system)."""
        X = np.array([[1.0, 2.0]])
        y = np.array([3.0])

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        y_pred = model.predict(X)
        np.testing.assert_array_almost_equal(y_pred, y, decimal=5)

    def test_single_feature(self):
        """1D feature vector input works."""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        self.assertAlmostEqual(model.w[0], 2.0, places=5)
        self.assertAlmostEqual(model.b, 0.0, places=5)

    def test_negative_targets(self):
        """Model handles negative target values."""
        np.random.seed(42)
        X = np.random.randn(100)
        y = -2 * X - 5

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        self.assertAlmostEqual(model.w[0], -2.0, places=5)
        self.assertAlmostEqual(model.b, -5.0, places=5)

    def test_large_feature_values(self):
        """Model handles large feature values (numerical stability)."""
        np.random.seed(42)
        X = np.random.randn(100) * 1e6 + 5e6
        y = 2 * X + 3e6

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        self.assertAlmostEqual(model.w[0], 2.0, places=3)
        self.assertAlmostEqual(model.b / 1e6, 3.0, places=3)

    def test_zero_variance_feature_with_others(self):
        """Model handles constant feature column mixed with varying features."""
        np.random.seed(42)
        X = np.column_stack([np.ones(100), np.random.randn(100)])
        y = 3 * X[:, 1] + 2

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        y_pred = model.predict(X)
        r2 = model.score(X, y)
        self.assertGreater(r2, 0.99)


class TestConvergenceAndHistory(unittest.TestCase):
    """Tests for convergence behavior and training history."""

    def test_loss_decreases_monotonically(self):
        """Loss decreases (mostly) monotonically with appropriate learning rate."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + 2 * X[:, 1] + 1

        model = LinearRegression(
            method="gradient_descent", learning_rate=0.1, n_iterations=100
        )
        model.fit(X, y)

        # Check that final loss is much less than initial
        self.assertLess(model.history[-1], model.history[0] * 0.01)

    def test_early_stopping(self):
        """Model converges before max iterations with tolerance."""
        np.random.seed(42)
        X = np.random.randn(100)
        y = 2 * X + 3

        model = LinearRegression(
            method="gradient_descent",
            learning_rate=0.5,
            n_iterations=10000,
            tolerance=1e-10,
        )
        model.fit(X, y)

        # Should converge well before 10000 iterations
        self.assertLess(len(model.history), 500)

    def test_training_history_recorded(self):
        """Training history has correct length."""
        np.random.seed(42)
        X = np.random.randn(50)
        y = 2 * X + 1

        model = LinearRegression(
            method="gradient_descent",
            learning_rate=0.01,
            n_iterations=100,
            tolerance=0,
        )
        model.fit(X, y)

        self.assertEqual(len(model.history), 100)

    def test_normal_equation_no_history(self):
        """Normal equation has empty history (no iterations)."""
        X = np.random.randn(50)
        y = 2 * X + 1

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        self.assertEqual(len(model.history), 0)


class TestNumericalPrecision(unittest.TestCase):
    """Tests for numerical precision and stability."""

    def test_float32_consistency(self):
        """Results are reasonable with float32 inputs."""
        np.random.seed(42)
        X = np.random.randn(100).astype(np.float32)
        y = (2 * X + 3).astype(np.float32)

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        self.assertAlmostEqual(model.w[0], 2.0, places=4)
        self.assertAlmostEqual(model.b, 3.0, places=4)

    def test_noisy_data(self):
        """With Gaussian noise, R^2 is reasonable."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        noise = np.random.randn(200) * 0.5
        y = X[:, 0] + 2 * X[:, 1] + 3 + noise

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        r2 = model.score(X, y)
        self.assertGreater(r2, 0.9)

    def test_hand_computed_2x2_system(self):
        """Verify against hand-computed 2x2 system."""
        # X = [[1, 1], [1, 2]], y = [3, 5]
        # With bias: X_aug = [[1, 1, 1], [1, 1, 2]]
        # Solution: y = 1 + 0*x1 + 2*x2 -> w=[0, 2], b=1
        X = np.array([[1.0, 1.0], [1.0, 2.0]])
        y = np.array([3.0, 5.0])

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        # y = b + w1*x1 + w2*x2
        # For [1,1]: b + w1 + w2 = 3
        # For [1,2]: b + w1 + 2*w2 = 5
        # Difference: w2 = 2
        # Then: b + w1 + 2 = 3 -> b + w1 = 1
        y_pred = model.predict(X)
        np.testing.assert_array_almost_equal(y_pred, y, decimal=10)


class TestR2Score(unittest.TestCase):
    """Tests for R^2 metric."""

    def test_perfect_fit_r2(self):
        """R^2 = 1.0 for perfect predictions."""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * X + 1

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        r2 = model.score(X, y)
        self.assertAlmostEqual(r2, 1.0, places=10)

    def test_mean_prediction_r2(self):
        """R^2 = 0.0 when predicting mean for all."""
        np.random.seed(42)
        X = np.random.randn(100)
        y = np.random.randn(100)

        # Force model to predict approximately mean by using unrelated X
        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        # Manually set weights to predict mean
        model.w = np.zeros_like(model.w)
        model.b = np.mean(y)

        r2 = model.score(X, y)
        self.assertAlmostEqual(r2, 0.0, places=10)

    def test_worse_than_mean_r2(self):
        """R^2 can be negative for bad predictions."""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        # Force bad predictions
        model.w = np.array([-10.0])
        model.b = 100.0

        r2 = model.score(X, y)
        self.assertLess(r2, 0)

    def test_constant_target_r2(self):
        """R^2 handles constant target values."""
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 5.0])

        model = LinearRegression(method="normal_equation")
        model.fit(X, y)

        r2 = model.score(X, y)
        self.assertEqual(r2, 1.0)


class TestSklearnComparison(unittest.TestCase):
    """Comparison tests with sklearn (optional, requires sklearn)."""

    def test_matches_sklearn(self):
        """Our implementation matches sklearn."""
        try:
            from sklearn.linear_model import LinearRegression as SklearnLR
        except ImportError:
            self.skipTest("sklearn not available")

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + 3.0

        our_model = LinearRegression(method="normal_equation")
        our_model.fit(X, y)

        sklearn_model = SklearnLR()
        sklearn_model.fit(X, y)

        np.testing.assert_array_almost_equal(our_model.w, sklearn_model.coef_, decimal=10)
        self.assertAlmostEqual(our_model.b, sklearn_model.intercept_, places=10)

        our_r2 = our_model.score(X, y)
        sklearn_r2 = sklearn_model.score(X, y)
        self.assertAlmostEqual(our_r2, sklearn_r2, places=10)


class TestFitReturnsself(unittest.TestCase):
    """Test that fit returns self for chaining."""

    def test_fit_returns_self(self):
        """fit() returns self for method chaining."""
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        model = LinearRegression()
        result = model.fit(X, y)

        self.assertIs(result, model)


class TestUnfittedModel(unittest.TestCase):
    """Test behavior when model is not fitted."""

    def test_predict_before_fit_raises(self):
        """predict() raises error before fit()."""
        model = LinearRegression()
        X = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            model.predict(X)


if __name__ == "__main__":
    unittest.main()
