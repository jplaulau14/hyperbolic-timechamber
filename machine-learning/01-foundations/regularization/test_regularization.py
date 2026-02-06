"""Tests for regularization (L1, L2, Elastic Net)."""

import unittest
import numpy as np
from implementation import (
    l2_penalty,
    l2_gradient,
    l1_penalty,
    l1_gradient,
    elastic_net_penalty,
    elastic_net_gradient,
    sgd_with_weight_decay,
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression,
)


class TestL2Penalty(unittest.TestCase):

    def test_known_values(self):
        """w = [1, 2, 3], lambda = 0.1 -> penalty = 0.1/2 * (1+4+9) = 0.7"""
        w = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(l2_penalty(w, 0.1), 0.7)

    def test_zero_weights(self):
        w = np.zeros(5)
        self.assertEqual(l2_penalty(w, 1.0), 0.0)

    def test_zero_lambda(self):
        w = np.array([10.0, -20.0])
        self.assertEqual(l2_penalty(w, 0.0), 0.0)

    def test_large_weights(self):
        w = np.array([100.0, -200.0])
        expected = 0.5 * 0.01 * (100**2 + 200**2)
        self.assertAlmostEqual(l2_penalty(w, 0.01), expected)


class TestL2Gradient(unittest.TestCase):

    def test_known_values(self):
        """w = [1, 2, 3], lambda = 0.1 -> gradient = [0.1, 0.2, 0.3]"""
        w = np.array([1.0, 2.0, 3.0])
        grad = l2_gradient(w, 0.1)
        np.testing.assert_allclose(grad, [0.1, 0.2, 0.3])

    def test_zero_weights(self):
        w = np.zeros(3)
        grad = l2_gradient(w, 1.0)
        np.testing.assert_array_equal(grad, np.zeros(3))

    def test_negative_weights(self):
        w = np.array([-1.0, -2.0])
        grad = l2_gradient(w, 0.5)
        np.testing.assert_allclose(grad, [-0.5, -1.0])


class TestL1Penalty(unittest.TestCase):

    def test_known_values(self):
        """w = [1, -2, 3], lambda = 0.1 -> penalty = 0.1 * (1+2+3) = 0.6"""
        w = np.array([1.0, -2.0, 3.0])
        self.assertAlmostEqual(l1_penalty(w, 0.1), 0.6)

    def test_zero_weights(self):
        w = np.zeros(5)
        self.assertEqual(l1_penalty(w, 1.0), 0.0)

    def test_zero_lambda(self):
        w = np.array([10.0, -20.0])
        self.assertEqual(l1_penalty(w, 0.0), 0.0)


class TestL1Gradient(unittest.TestCase):

    def test_known_values(self):
        """w = [1, -2, 0], lambda = 0.1 -> gradient = [0.1, -0.1, 0.0]"""
        w = np.array([1.0, -2.0, 0.0])
        grad = l1_gradient(w, 0.1)
        np.testing.assert_allclose(grad, [0.1, -0.1, 0.0])

    def test_zero_at_origin(self):
        """sign(0) should be 0 (subgradient convention)."""
        w = np.array([0.0, 0.0, 0.0])
        grad = l1_gradient(w, 1.0)
        np.testing.assert_array_equal(grad, np.zeros(3))

    def test_mixed_signs(self):
        w = np.array([3.0, -0.5, 0.0, 1.0, -7.0])
        grad = l1_gradient(w, 2.0)
        np.testing.assert_allclose(grad, [2.0, -2.0, 0.0, 2.0, -2.0])


class TestElasticNetPenalty(unittest.TestCase):

    def test_pure_l1(self):
        """l1_ratio = 1.0 should equal pure L1 penalty."""
        w = np.array([1.0, -2.0, 3.0])
        lambda_ = 0.5
        en = elastic_net_penalty(w, lambda_, l1_ratio=1.0)
        l1 = l1_penalty(w, lambda_)
        self.assertAlmostEqual(en, l1)

    def test_pure_l2(self):
        """l1_ratio = 0.0 should equal pure L2 penalty."""
        w = np.array([1.0, -2.0, 3.0])
        lambda_ = 0.5
        en = elastic_net_penalty(w, lambda_, l1_ratio=0.0)
        l2 = l2_penalty(w, lambda_)
        self.assertAlmostEqual(en, l2)

    def test_mixed(self):
        w = np.array([1.0, 2.0])
        lambda_ = 1.0
        l1_ratio = 0.5
        expected = 1.0 * (0.5 * (1 + 2) + 0.5 * 0.5 * (1 + 4))
        self.assertAlmostEqual(
            elastic_net_penalty(w, lambda_, l1_ratio), expected
        )


class TestElasticNetGradient(unittest.TestCase):

    def test_pure_l1(self):
        w = np.array([1.0, -2.0, 0.0])
        lambda_ = 0.3
        en_grad = elastic_net_gradient(w, lambda_, l1_ratio=1.0)
        l1_grad = l1_gradient(w, lambda_)
        np.testing.assert_allclose(en_grad, l1_grad)

    def test_pure_l2(self):
        w = np.array([1.0, -2.0, 0.0])
        lambda_ = 0.3
        en_grad = elastic_net_gradient(w, lambda_, l1_ratio=0.0)
        l2_grad = l2_gradient(w, lambda_)
        np.testing.assert_allclose(en_grad, l2_grad)

    def test_mixed(self):
        w = np.array([2.0, -1.0])
        lambda_ = 0.4
        l1_ratio = 0.6
        expected = 0.4 * (0.6 * np.sign(w) + 0.4 * w)
        np.testing.assert_allclose(
            elastic_net_gradient(w, lambda_, l1_ratio), expected
        )


class TestWeightDecay(unittest.TestCase):

    def test_single_step(self):
        w = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        lr = 0.01
        lambda_ = 0.1
        updated = sgd_with_weight_decay(w, grad, lr, lambda_)
        expected = w - lr * grad - lr * lambda_ * w
        np.testing.assert_allclose(updated, expected)

    def test_equivalence_to_l2_sgd(self):
        """L2 regularization and weight decay produce identical updates for SGD."""
        np.random.seed(42)
        w = np.random.randn(10)
        data_grad = np.random.randn(10)
        lr = 0.01
        lambda_ = 0.1

        # Weight decay update
        w_wd = sgd_with_weight_decay(w, data_grad, lr, lambda_)

        # L2 regularization update: w - lr * (data_grad + lambda * w)
        w_l2 = w - lr * (data_grad + l2_gradient(w, lambda_))

        np.testing.assert_allclose(w_wd, w_l2, atol=1e-15)

    def test_multi_step_trajectory(self):
        """Over multiple steps, L2 reg and weight decay remain equivalent."""
        np.random.seed(7)
        w_wd = np.random.randn(5)
        w_l2 = w_wd.copy()
        lr = 0.005
        lambda_ = 0.05

        for _ in range(100):
            data_grad = np.random.randn(5)
            w_wd = sgd_with_weight_decay(w_wd, data_grad, lr, lambda_)
            w_l2 = w_l2 - lr * (data_grad + l2_gradient(w_l2, lambda_))

        np.testing.assert_allclose(w_wd, w_l2, atol=1e-12)


class TestRidgeRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 5
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.true_w = np.array([3.0, -1.0, 0.5, 0.0, 0.0])
        self.y = self.X @ self.true_w + 2.0 + 0.1 * np.random.randn(self.n_samples)

    def test_fit_predict(self):
        model = RidgeRegression(lambda_=0.01, learning_rate=0.01, n_iterations=2000)
        model.fit(self.X, self.y)
        r2 = model.score(self.X, self.y)
        self.assertGreater(r2, 0.95)

    def test_shrinkage(self):
        """L2 regularized weights should have smaller norm than unregularized."""
        low_reg = RidgeRegression(lambda_=0.001, learning_rate=0.01, n_iterations=2000)
        high_reg = RidgeRegression(lambda_=1.0, learning_rate=0.01, n_iterations=2000)

        low_reg.fit(self.X, self.y)
        high_reg.fit(self.X, self.y)

        w_low, _ = low_reg.get_weights()
        w_high, _ = high_reg.get_weights()

        self.assertGreater(np.linalg.norm(w_low), np.linalg.norm(w_high))

    def test_no_exact_sparsity(self):
        """L2 should not drive weights to exactly zero."""
        model = RidgeRegression(lambda_=0.1, learning_rate=0.01, n_iterations=2000)
        model.fit(self.X, self.y)
        w, _ = model.get_weights()
        self.assertTrue(np.all(np.abs(w) > 1e-10))

    def test_closed_form_matches_gd(self):
        """Ridge closed-form and gradient descent should agree."""
        # Small problem with standardized features for fast GD convergence
        np.random.seed(99)
        X = np.random.randn(100, 3)
        y = X @ np.array([1.0, -2.0, 0.5]) + 0.5 + 0.1 * np.random.randn(100)

        model_gd = RidgeRegression(
            lambda_=0.1, learning_rate=0.05, n_iterations=20000, tolerance=1e-14
        )
        model_cf = RidgeRegression(lambda_=0.1)

        model_gd.fit(X, y)
        model_cf.fit_closed_form(X, y)

        w_gd, b_gd = model_gd.get_weights()
        w_cf, b_cf = model_cf.get_weights()

        np.testing.assert_allclose(w_gd, w_cf, atol=0.01)
        self.assertAlmostEqual(b_gd, b_cf, places=2)

    def test_closed_form_condition_number(self):
        """Adding lambda*I should improve condition number of X^T X."""
        X = self.X
        XtX = X.T @ X
        lambda_ = 1.0
        XtX_reg = XtX + lambda_ * np.eye(self.n_features)

        cond_unreg = np.linalg.cond(XtX)
        cond_reg = np.linalg.cond(XtX_reg)
        self.assertLess(cond_reg, cond_unreg)

    def test_zero_lambda(self):
        """lambda=0 should behave like unregularized regression."""
        model = RidgeRegression(lambda_=0.0, learning_rate=0.01, n_iterations=2000)
        model.fit(self.X, self.y)
        r2 = model.score(self.X, self.y)
        self.assertGreater(r2, 0.95)

    def test_single_feature(self):
        X = np.random.randn(50, 1)
        y = 3.0 * X.ravel() + 1.0 + 0.05 * np.random.randn(50)
        model = RidgeRegression(lambda_=0.01, learning_rate=0.01, n_iterations=1000)
        model.fit(X, y)
        self.assertGreater(model.score(X, y), 0.9)

    def test_unfitted_raises(self):
        model = RidgeRegression()
        with self.assertRaises(ValueError):
            model.predict(np.zeros((5, 3)))
        with self.assertRaises(ValueError):
            model.get_weights()

    def test_1d_input(self):
        X = np.random.randn(50)
        y = 2.0 * X + 1.0
        model = RidgeRegression(lambda_=0.01, learning_rate=0.01, n_iterations=1000)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(preds.shape, (50,))

    def test_high_lambda_underfitting(self):
        """Very high lambda should cause underfitting (low R^2)."""
        model = RidgeRegression(lambda_=1000.0)
        model.fit_closed_form(self.X, self.y)
        r2 = model.score(self.X, self.y)
        self.assertLess(r2, 0.5)


class TestLassoRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features)
        # Only first 3 features matter
        self.true_w = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        self.y = self.X @ self.true_w + 1.0 + 0.1 * np.random.randn(self.n_samples)

    def test_fit_predict(self):
        model = LassoRegression(lambda_=0.01, learning_rate=0.005, n_iterations=3000)
        model.fit(self.X, self.y)
        r2 = model.score(self.X, self.y)
        self.assertGreater(r2, 0.9)

    def test_sparsity(self):
        """L1 should drive irrelevant feature weights near zero."""
        model = LassoRegression(lambda_=0.05, learning_rate=0.005, n_iterations=5000)
        model.fit(self.X, self.y)
        w, _ = model.get_weights()

        # Features 3-9 should have small weights
        irrelevant = np.abs(w[3:])
        relevant = np.abs(w[:3])
        self.assertGreater(np.mean(relevant), 5 * np.mean(irrelevant))

    def test_sparsity_increases_with_lambda(self):
        """Higher lambda should produce more near-zero weights."""
        threshold = 0.1
        sparsity_counts = []

        for lam in [0.001, 0.05, 0.2]:
            model = LassoRegression(
                lambda_=lam, learning_rate=0.005, n_iterations=5000
            )
            model.fit(self.X, self.y)
            w, _ = model.get_weights()
            n_near_zero = np.sum(np.abs(w) < threshold)
            sparsity_counts.append(n_near_zero)

        # Sparsity should be non-decreasing with lambda
        for i in range(len(sparsity_counts) - 1):
            self.assertGreaterEqual(sparsity_counts[i + 1], sparsity_counts[i])

    def test_unfitted_raises(self):
        model = LassoRegression()
        with self.assertRaises(ValueError):
            model.predict(np.zeros((5, 3)))

    def test_single_feature(self):
        np.random.seed(0)
        X = np.random.randn(50, 1)
        y = 3.0 * X.ravel() + 1.0 + 0.05 * np.random.randn(50)
        model = LassoRegression(lambda_=0.01, learning_rate=0.01, n_iterations=2000)
        model.fit(X, y)
        self.assertGreater(model.score(X, y), 0.9)


class TestElasticNetRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.true_w = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        self.y = self.X @ self.true_w + 1.0 + 0.1 * np.random.randn(self.n_samples)

    def test_fit_predict(self):
        model = ElasticNetRegression(
            lambda_=0.01, l1_ratio=0.5, learning_rate=0.005, n_iterations=3000
        )
        model.fit(self.X, self.y)
        r2 = model.score(self.X, self.y)
        self.assertGreater(r2, 0.9)

    def test_pure_l1_matches_lasso(self):
        """l1_ratio=1.0 should behave like Lasso."""
        en = ElasticNetRegression(
            lambda_=0.05, l1_ratio=1.0, learning_rate=0.005, n_iterations=3000
        )
        lasso = LassoRegression(
            lambda_=0.05, learning_rate=0.005, n_iterations=3000
        )
        en.fit(self.X, self.y)
        lasso.fit(self.X, self.y)

        w_en, b_en = en.get_weights()
        w_lasso, b_lasso = lasso.get_weights()
        np.testing.assert_allclose(w_en, w_lasso, atol=1e-6)
        self.assertAlmostEqual(b_en, b_lasso, places=5)

    def test_pure_l2_matches_ridge(self):
        """l1_ratio=0.0 should behave like Ridge."""
        en = ElasticNetRegression(
            lambda_=0.05, l1_ratio=0.0, learning_rate=0.005, n_iterations=3000
        )
        ridge = RidgeRegression(
            lambda_=0.05, learning_rate=0.005, n_iterations=3000
        )
        en.fit(self.X, self.y)
        ridge.fit(self.X, self.y)

        w_en, b_en = en.get_weights()
        w_ridge, b_ridge = ridge.get_weights()
        np.testing.assert_allclose(w_en, w_ridge, atol=1e-6)
        self.assertAlmostEqual(b_en, b_ridge, places=5)

    def test_unfitted_raises(self):
        model = ElasticNetRegression()
        with self.assertRaises(ValueError):
            model.predict(np.zeros((5, 3)))

    def test_l1_ratio_effect(self):
        """Higher l1_ratio should produce sparser weights."""
        threshold = 0.1
        sparsity_by_ratio = []

        for ratio in [0.0, 0.5, 1.0]:
            model = ElasticNetRegression(
                lambda_=0.1, l1_ratio=ratio, learning_rate=0.005, n_iterations=5000
            )
            model.fit(self.X, self.y)
            w, _ = model.get_weights()
            n_near_zero = np.sum(np.abs(w) < threshold)
            sparsity_by_ratio.append(n_near_zero)

        # Pure L1 should be at least as sparse as pure L2
        self.assertGreaterEqual(sparsity_by_ratio[2], sparsity_by_ratio[0])


class TestWeightDistribution(unittest.TestCase):
    """Compare L1 vs L2 weight distributions."""

    def test_l1_sparser_than_l2(self):
        """L1 should produce more near-zero weights than L2."""
        np.random.seed(42)
        n = 200
        d = 10
        X = np.random.randn(n, d)
        true_w = np.zeros(d)
        true_w[:3] = [3.0, -2.0, 1.0]
        y = X @ true_w + 0.1 * np.random.randn(n)

        ridge = RidgeRegression(lambda_=0.1, learning_rate=0.005, n_iterations=5000)
        lasso = LassoRegression(lambda_=0.1, learning_rate=0.005, n_iterations=5000)

        ridge.fit(X, y)
        lasso.fit(X, y)

        w_ridge, _ = ridge.get_weights()
        w_lasso, _ = lasso.get_weights()

        threshold = 0.1
        n_sparse_ridge = np.sum(np.abs(w_ridge) < threshold)
        n_sparse_lasso = np.sum(np.abs(w_lasso) < threshold)
        self.assertGreaterEqual(n_sparse_lasso, n_sparse_ridge)


class TestLambdaSweep(unittest.TestCase):
    """Test hyperparameter sensitivity."""

    def test_underfitting_with_high_lambda(self):
        """Very high lambda should cause high training error."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1

        model = RidgeRegression(lambda_=1000.0)
        model.fit_closed_form(X, y)
        r2 = model.score(X, y)
        self.assertLess(r2, 0.5)

    def test_overfitting_detection(self):
        """Unregularized model on overparameterized problem should overfit:
        train R^2 >> val R^2. Regularization should narrow this gap."""
        np.random.seed(42)
        n_train, n_features = 25, 20
        X = np.random.randn(n_train, n_features)
        true_w = np.zeros(n_features)
        true_w[:3] = [2.0, -1.0, 0.5]
        y = X @ true_w + np.random.randn(n_train) * 0.5

        model_unreg = RidgeRegression(lambda_=1e-6)
        model_unreg.fit_closed_form(X, y)

        train_r2 = model_unreg.score(X, y)
        X_val = np.random.randn(200, n_features)
        y_val = X_val @ true_w + np.random.randn(200) * 0.5
        val_r2 = model_unreg.score(X_val, y_val)

        # Overfit: train much better than val
        self.assertGreater(train_r2, val_r2 + 0.1)


class TestNumericalStability(unittest.TestCase):

    def test_gradient_magnitude_bounded(self):
        """Regularization gradients should be finite and bounded."""
        w = np.array([1e6, -1e6, 1e-8])
        grad_l2 = l2_gradient(w, 0.01)
        grad_l1 = l1_gradient(w, 0.01)
        self.assertTrue(np.all(np.isfinite(grad_l2)))
        self.assertTrue(np.all(np.isfinite(grad_l1)))

        # L1 gradient magnitude is bounded by lambda regardless of weight size
        self.assertTrue(np.all(np.abs(grad_l1) <= 0.01 + 1e-15))

    def test_condition_number_improvement(self):
        """Ridge regularization improves conditioning of ill-conditioned matrices."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        # Make nearly collinear columns
        X[:, 9] = X[:, 0] + 1e-8 * np.random.randn(50)

        XtX = X.T @ X
        cond_unreg = np.linalg.cond(XtX)

        for lam in [0.01, 0.1, 1.0]:
            cond_reg = np.linalg.cond(XtX + lam * np.eye(10))
            self.assertLess(cond_reg, cond_unreg)


class TestEdgeCases(unittest.TestCase):

    def test_zero_lambda_reduces_to_unregularized(self):
        """With lambda=0, Ridge/Lasso/ElasticNet should match unregularized."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + 0.5

        ridge = RidgeRegression(lambda_=0.0, learning_rate=0.01, n_iterations=2000)
        lasso = LassoRegression(lambda_=0.0, learning_rate=0.01, n_iterations=2000)

        ridge.fit(X, y)
        lasso.fit(X, y)

        w_ridge, b_ridge = ridge.get_weights()
        w_lasso, b_lasso = lasso.get_weights()

        np.testing.assert_allclose(w_ridge, w_lasso, atol=0.05)
        self.assertAlmostEqual(b_ridge, b_lasso, places=1)

    def test_2d_weight_matrix(self):
        """Penalty functions should work on 2D weight matrices."""
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        penalty = l2_penalty(W, 0.1)
        expected = 0.5 * 0.1 * (1 + 4 + 9 + 16)
        self.assertAlmostEqual(penalty, expected)

        grad = l2_gradient(W, 0.1)
        np.testing.assert_allclose(grad, 0.1 * W)


if __name__ == "__main__":
    unittest.main()
