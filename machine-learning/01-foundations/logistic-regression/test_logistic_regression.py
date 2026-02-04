"""Tests for logistic regression implementation."""

import unittest
import numpy as np
from implementation import LogisticRegression, sigmoid, sigmoid_derivative, binary_cross_entropy


class TestSigmoid(unittest.TestCase):
    """Tests for sigmoid function."""

    def test_sigmoid_at_zero(self):
        """sigmoid(0) = 0.5."""
        self.assertAlmostEqual(sigmoid(np.array([0.0]))[0], 0.5)

    def test_sigmoid_large_positive(self):
        """sigmoid(large) approaches 1."""
        result = sigmoid(np.array([100.0]))[0]
        self.assertGreater(result, 0.99999)
        self.assertLessEqual(result, 1.0)

    def test_sigmoid_large_negative(self):
        """sigmoid(-large) approaches 0."""
        result = sigmoid(np.array([-100.0]))[0]
        self.assertLess(result, 0.00001)
        self.assertGreaterEqual(result, 0.0)

    def test_sigmoid_symmetry(self):
        """sigmoid(-z) = 1 - sigmoid(z)."""
        np.random.seed(42)
        z = np.random.randn(100)
        np.testing.assert_array_almost_equal(sigmoid(-z), 1 - sigmoid(z))

    def test_sigmoid_vectorized(self):
        """sigmoid works on arrays."""
        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = sigmoid(z)
        self.assertEqual(result.shape, z.shape)
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(result < 1))

    def test_sigmoid_extreme_values(self):
        """sigmoid handles extreme values without overflow."""
        z_extreme = np.array([1000.0, -1000.0])
        result = sigmoid(z_extreme)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))


class TestSigmoidDerivative(unittest.TestCase):
    """Tests for sigmoid derivative."""

    def test_derivative_at_zero(self):
        """Maximum derivative at z=0: sigmoid(0)*(1-sigmoid(0)) = 0.25."""
        result = sigmoid_derivative(np.array([0.0]))[0]
        self.assertAlmostEqual(result, 0.25)

    def test_derivative_shape(self):
        """Derivative has same shape as input."""
        z = np.random.randn(5, 3)
        result = sigmoid_derivative(z)
        self.assertEqual(result.shape, z.shape)

    def test_derivative_numerical(self):
        """Numerical gradient check for sigmoid derivative."""
        z = np.array([0.5, -0.5, 1.0, -1.0])
        h = 1e-7
        numerical_grad = (sigmoid(z + h) - sigmoid(z - h)) / (2 * h)
        analytical_grad = sigmoid_derivative(z)
        np.testing.assert_array_almost_equal(numerical_grad, analytical_grad, decimal=6)


class TestBinaryCrossEntropy(unittest.TestCase):
    """Tests for binary cross-entropy loss."""

    def test_perfect_predictions(self):
        """BCE is near zero for perfect predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.0001, 0.9999, 0.9999, 0.0001])
        loss = binary_cross_entropy(y_true, y_pred)
        self.assertLess(loss, 0.001)

    def test_worst_predictions(self):
        """BCE is large for opposite predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.9999, 0.0001, 0.0001, 0.9999])
        loss = binary_cross_entropy(y_true, y_pred)
        self.assertGreater(loss, 5.0)

    def test_numerical_stability_pred_zero(self):
        """BCE handles y_pred=0 without inf/nan (epsilon clipping)."""
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 0.0])
        loss = binary_cross_entropy(y_true, y_pred)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))

    def test_numerical_stability_pred_one(self):
        """BCE handles y_pred=1 without inf/nan (epsilon clipping)."""
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([1.0, 1.0])
        loss = binary_cross_entropy(y_true, y_pred)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))

    def test_known_value(self):
        """BCE with hand-computed value."""
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.8, 0.3])
        expected = -0.5 * (np.log(0.8) + np.log(0.7))
        loss = binary_cross_entropy(y_true, y_pred)
        self.assertAlmostEqual(loss, expected, places=10)


class TestLogisticRegressionBasic(unittest.TestCase):
    """Basic correctness tests."""

    def test_perfect_separation(self):
        """Model achieves 100% accuracy on linearly separable data."""
        np.random.seed(42)
        X_class0 = np.random.randn(50, 2) - 3
        X_class1 = np.random.randn(50, 2) + 3
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 50 + [1] * 50)

        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertEqual(accuracy, 1.0)

    def test_known_weights_recovery(self):
        """Model approximately recovers known weights."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2) * 2
        true_w = np.array([1.5, -2.0])
        true_b = 0.5
        z = X @ true_w + true_b
        proba = sigmoid(z)
        y = (np.random.rand(n) < proba).astype(int)

        model = LogisticRegression(learning_rate=0.1, n_iterations=5000, tolerance=1e-10)
        model.fit(X, y)

        np.testing.assert_array_almost_equal(model.w, true_w, decimal=0)
        self.assertAlmostEqual(model.b, true_b, places=0)

    def test_single_sample(self):
        """Fitting with single sample doesn't crash."""
        X = np.array([[1.0, 2.0]])
        y = np.array([1])

        model = LogisticRegression(n_iterations=10)
        model.fit(X, y)

        pred = model.predict(X)
        self.assertEqual(pred.shape, (1,))

    def test_single_feature(self):
        """1D feature input works."""
        np.random.seed(42)
        X = np.linspace(-3, 3, 100)
        y = (X > 0).astype(int)

        model = LogisticRegression(learning_rate=0.5, n_iterations=1000)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertGreater(accuracy, 0.95)


class TestGradientCorrectness(unittest.TestCase):
    """Tests for gradient computation."""

    def test_numerical_gradient_check(self):
        """Analytical gradient matches numerical gradient."""
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = (np.random.rand(20) > 0.5).astype(float)

        model = LogisticRegression()
        model.w = np.random.randn(3)
        model.b = np.random.randn()

        h = 1e-5

        z = X @ model.w + model.b
        y_pred = sigmoid(z)
        dw_analytical, db_analytical = model._compute_gradients(X, y, y_pred)

        dw_numerical = np.zeros_like(model.w)
        for i in range(len(model.w)):
            w_plus = model.w.copy()
            w_plus[i] += h
            z_plus = X @ w_plus + model.b
            loss_plus = binary_cross_entropy(y, sigmoid(z_plus))

            w_minus = model.w.copy()
            w_minus[i] -= h
            z_minus = X @ w_minus + model.b
            loss_minus = binary_cross_entropy(y, sigmoid(z_minus))

            dw_numerical[i] = (loss_plus - loss_minus) / (2 * h)

        np.testing.assert_array_almost_equal(dw_analytical, dw_numerical, decimal=5)

        loss_plus_b = binary_cross_entropy(y, sigmoid(X @ model.w + model.b + h))
        loss_minus_b = binary_cross_entropy(y, sigmoid(X @ model.w + model.b - h))
        db_numerical = (loss_plus_b - loss_minus_b) / (2 * h)
        self.assertAlmostEqual(db_analytical, db_numerical, places=5)

    def test_gradient_shape(self):
        """Gradient has same shape as weights."""
        np.random.seed(42)
        X = np.random.randn(30, 5)
        y = (np.random.rand(30) > 0.5).astype(float)

        model = LogisticRegression()
        model.w = np.zeros(5)
        model.b = 0.0

        z = X @ model.w + model.b
        y_pred = sigmoid(z)
        dw, db = model._compute_gradients(X, y, y_pred)

        self.assertEqual(dw.shape, (5,))
        self.assertIsInstance(db, float)

    def test_gradient_small_at_convergence(self):
        """Gradient magnitude is small after convergence."""
        np.random.seed(42)
        X_class0 = np.random.randn(30, 2) - 2
        X_class1 = np.random.randn(30, 2) + 2
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 30 + [1] * 30).astype(float)

        model = LogisticRegression(learning_rate=0.5, n_iterations=2000, tolerance=1e-12)
        model.fit(X, y)

        z = X @ model.w + model.b
        y_pred = sigmoid(z)
        dw, db = model._compute_gradients(X, y, y_pred)

        grad_norm = np.sqrt(np.sum(dw**2) + db**2)
        self.assertLess(grad_norm, 0.1)


class TestConvergence(unittest.TestCase):
    """Tests for convergence behavior."""

    def test_loss_decreases(self):
        """Loss decreases during training."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(learning_rate=0.1, n_iterations=100, tolerance=0)
        model.fit(X, y)

        self.assertLess(model.history[-1], model.history[0])

    def test_early_stopping(self):
        """Model stops early when converged."""
        np.random.seed(42)
        X_class0 = np.random.randn(50, 2) - 3
        X_class1 = np.random.randn(50, 2) + 3
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 50 + [1] * 50)

        model = LogisticRegression(
            learning_rate=0.5, n_iterations=10000, tolerance=1e-6
        )
        model.fit(X, y)

        self.assertLess(len(model.history), 5000)

    def test_training_history_recorded(self):
        """Training history has correct length when no early stopping."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(learning_rate=0.01, n_iterations=100, tolerance=0)
        model.fit(X, y)

        self.assertEqual(len(model.history), 100)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_all_same_class_ones(self):
        """Model predicts all 1s when trained on all 1s."""
        X = np.random.randn(50, 2)
        y = np.ones(50)

        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)

        proba = model.predict_proba(X)
        self.assertTrue(np.all(proba > 0.9))

    def test_all_same_class_zeros(self):
        """Model predicts all 0s when trained on all 0s."""
        X = np.random.randn(50, 2)
        y = np.zeros(50)

        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)

        proba = model.predict_proba(X)
        self.assertTrue(np.all(proba < 0.1))

    def test_imbalanced_data(self):
        """Model handles imbalanced data (90-10 split)."""
        np.random.seed(42)
        X_class0 = np.random.randn(90, 2) - 2
        X_class1 = np.random.randn(10, 2) + 2
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 90 + [1] * 10)

        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertGreater(accuracy, 0.9)

    def test_zero_features(self):
        """Model works when all X values are zero (bias only)."""
        X = np.zeros((50, 2))
        y = np.array([0] * 25 + [1] * 25)

        model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        model.fit(X, y)

        proba = model.predict_proba(X)
        self.assertTrue(np.all(proba > 0.4))
        self.assertTrue(np.all(proba < 0.6))

    def test_collinear_features(self):
        """Model handles collinear features (identical columns)."""
        np.random.seed(42)
        x1 = np.random.randn(100)
        X = np.column_stack([x1, x1])
        y = (x1 > 0).astype(int)

        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertGreater(accuracy, 0.9)

    def test_2d_y_input(self):
        """Model handles 2D y input of shape (n, 1)."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, (50, 1))

        model = LogisticRegression(n_iterations=10)
        model.fit(X, y)

        pred = model.predict(X)
        self.assertEqual(pred.shape, (50,))


class TestShapeValidation(unittest.TestCase):
    """Tests for shape validation."""

    def test_predict_proba_shape(self):
        """predict_proba output shape matches input samples."""
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randint(0, 2, 50)

        model = LogisticRegression(n_iterations=10)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 3)
        proba = model.predict_proba(X_test)
        self.assertEqual(proba.shape, (20,))

    def test_predict_shape(self):
        """predict output shape matches input samples."""
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randint(0, 2, 50)

        model = LogisticRegression(n_iterations=10)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 3)
        pred = model.predict(X_test)
        self.assertEqual(pred.shape, (20,))

    def test_1d_input_handling(self):
        """1D X input is automatically reshaped."""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([0, 0, 1, 1, 1])

        model = LogisticRegression(n_iterations=100)
        model.fit(X, y)

        pred = model.predict(X)
        self.assertEqual(pred.shape, (5,))


class TestPredictThreshold(unittest.TestCase):
    """Tests for prediction threshold."""

    def test_default_threshold(self):
        """Default threshold is 0.5."""
        np.random.seed(42)
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, 10)

        model = LogisticRegression(n_iterations=10)
        model.fit(X, y)

        proba = model.predict_proba(X)
        pred_default = model.predict(X)
        pred_explicit = model.predict(X, threshold=0.5)

        np.testing.assert_array_equal(pred_default, pred_explicit)

    def test_custom_threshold(self):
        """Custom threshold changes predictions."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(learning_rate=0.5, n_iterations=500)
        model.fit(X, y)

        pred_low = model.predict(X, threshold=0.3)
        pred_high = model.predict(X, threshold=0.7)

        self.assertGreaterEqual(np.sum(pred_low), np.sum(pred_high))


class TestDecisionBoundary(unittest.TestCase):
    """Tests for decision boundary parameters."""

    def test_decision_boundary_2d(self):
        """Decision boundary params for 2D data."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(learning_rate=0.5, n_iterations=1000)
        model.fit(X, y)

        slope, intercept = model.decision_boundary_params()
        self.assertIsInstance(slope, float)
        self.assertIsInstance(intercept, float)

    def test_decision_boundary_wrong_dimensions(self):
        """decision_boundary_params raises for non-2D data."""
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)

        model = LogisticRegression(n_iterations=10)
        model.fit(X, y)

        with self.assertRaises(ValueError):
            model.decision_boundary_params()

    def test_decision_boundary_unfitted(self):
        """decision_boundary_params raises for unfitted model."""
        model = LogisticRegression()

        with self.assertRaises(ValueError):
            model.decision_boundary_params()


class TestNumericalStability(unittest.TestCase):
    """Tests for numerical stability."""

    def test_gradient_with_saturated_sigmoid(self):
        """Gradients are finite when sigmoid saturates."""
        np.random.seed(42)
        X = np.random.randn(50, 2) * 10
        y = (X[:, 0] > 0).astype(float)

        model = LogisticRegression()
        model.w = np.array([10.0, 0.0])
        model.b = 0.0

        z = X @ model.w + model.b
        y_pred = sigmoid(z)
        dw, db = model._compute_gradients(X, y, y_pred)

        self.assertFalse(np.any(np.isnan(dw)))
        self.assertFalse(np.any(np.isinf(dw)))
        self.assertFalse(np.isnan(db))
        self.assertFalse(np.isinf(db))


class TestUnfittedModel(unittest.TestCase):
    """Test behavior when model is not fitted."""

    def test_predict_before_fit_raises(self):
        """predict() raises error before fit()."""
        model = LogisticRegression()
        X = np.array([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            model.predict(X)

    def test_predict_proba_before_fit_raises(self):
        """predict_proba() raises error before fit()."""
        model = LogisticRegression()
        X = np.array([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            model.predict_proba(X)

    def test_score_before_fit_raises(self):
        """score() raises error before fit()."""
        model = LogisticRegression()
        X = np.array([[1.0, 2.0]])
        y = np.array([1])

        with self.assertRaises(ValueError):
            model.score(X, y)


class TestFitReturnsSelf(unittest.TestCase):
    """Test that fit returns self for chaining."""

    def test_fit_returns_self(self):
        """fit() returns self for method chaining."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])

        model = LogisticRegression(n_iterations=10)
        result = model.fit(X, y)

        self.assertIs(result, model)


class TestSklearnComparison(unittest.TestCase):
    """Comparison tests with sklearn (optional, requires sklearn)."""

    def test_matches_sklearn(self):
        """Our implementation produces similar results to sklearn."""
        try:
            from sklearn.linear_model import LogisticRegression as SklearnLR
        except ImportError:
            self.skipTest("sklearn not available")

        np.random.seed(42)
        X_class0 = np.random.randn(100, 2) - 1
        X_class1 = np.random.randn(100, 2) + 1
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 100 + [1] * 100)

        our_model = LogisticRegression(learning_rate=0.1, n_iterations=5000, tolerance=1e-10)
        our_model.fit(X, y)

        sklearn_model = SklearnLR(penalty=None, solver='lbfgs', max_iter=5000)
        sklearn_model.fit(X, y)

        our_acc = our_model.score(X, y)
        sklearn_acc = sklearn_model.score(X, y)
        self.assertAlmostEqual(our_acc, sklearn_acc, places=1)

        our_proba = our_model.predict_proba(X)
        sklearn_proba = sklearn_model.predict_proba(X)[:, 1]
        correlation = np.corrcoef(our_proba, sklearn_proba)[0, 1]
        self.assertGreater(correlation, 0.95)


if __name__ == "__main__":
    unittest.main()
