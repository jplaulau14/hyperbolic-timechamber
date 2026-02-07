"""Tests for activation functions."""

import unittest
import numpy as np
from implementation import (
    Activation,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
    Swish,
    gradient_check,
    _stable_sigmoid,
)


class TestReLU(unittest.TestCase):

    def setUp(self):
        self.act = ReLU()

    def test_known_values(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(self.act(x), expected)

    def test_backward_known(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.act.forward(x)
        grad = np.ones_like(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(self.act.backward(grad), expected)

    def test_zero_gradient_at_zero(self):
        x = np.array([0.0])
        self.act.forward(x)
        grad = self.act.backward(np.ones(1))
        self.assertEqual(grad[0], 0.0)

    def test_gradient_check(self):
        x = np.array([0.5, 1.0, -0.5, 2.0, -2.0])
        result = gradient_check(self.act, x)
        self.assertLess(result["max_abs_error"], 1e-4)


class TestLeakyReLU(unittest.TestCase):

    def setUp(self):
        self.act = LeakyReLU(alpha=0.01)

    def test_known_values_default_alpha(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([-0.02, -0.01, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(self.act(x), expected)

    def test_backward_known(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.act.forward(x)
        grad = np.ones_like(x)
        expected = np.array([0.01, 0.01, 0.01, 1.0, 1.0])
        np.testing.assert_allclose(self.act.backward(grad), expected)

    def test_custom_alpha(self):
        act = LeakyReLU(alpha=0.2)
        x = np.array([-1.0, 0.0, 1.0])
        expected = np.array([-0.2, 0.0, 1.0])
        np.testing.assert_allclose(act(x), expected)

    def test_alpha_zero_matches_relu(self):
        act_leaky = LeakyReLU(alpha=0.0)
        act_relu = ReLU()
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        np.testing.assert_array_equal(act_leaky(x), act_relu(x))

    def test_alpha_one_is_identity(self):
        act = LeakyReLU(alpha=1.0)
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        np.testing.assert_allclose(act(x), x)

    def test_gradient_check(self):
        x = np.array([0.5, 1.0, -0.5, 2.0, -2.0])
        result = gradient_check(self.act, x)
        self.assertLess(result["max_abs_error"], 1e-4)


class TestSigmoid(unittest.TestCase):

    def setUp(self):
        self.act = Sigmoid()

    def test_at_zero(self):
        x = np.array([0.0])
        self.assertAlmostEqual(self.act(x)[0], 0.5)

    def test_large_positive(self):
        x = np.array([100.0, 1000.0])
        result = self.act(x)
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-10)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_large_negative(self):
        x = np.array([-100.0, -1000.0])
        result = self.act(x)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_backward_at_zero(self):
        x = np.array([0.0])
        self.act.forward(x)
        grad = self.act.backward(np.ones(1))
        self.assertAlmostEqual(grad[0], 0.25)

    def test_gradient_check(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = gradient_check(self.act, x)
        self.assertLess(result["max_rel_error"], 1e-5)

    def test_numerical_stability_no_nan_inf(self):
        x = np.linspace(-1000, 1000, 1000)
        result = self.act(x)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))


class TestTanh(unittest.TestCase):

    def setUp(self):
        self.act = Tanh()

    def test_at_zero(self):
        x = np.array([0.0])
        self.assertAlmostEqual(self.act(x)[0], 0.0)

    def test_bounded(self):
        x = np.array([-100.0, 100.0])
        result = self.act(x)
        np.testing.assert_allclose(result, [-1.0, 1.0], atol=1e-10)

    def test_backward_at_zero(self):
        x = np.array([0.0])
        self.act.forward(x)
        grad = self.act.backward(np.ones(1))
        self.assertAlmostEqual(grad[0], 1.0)

    def test_gradient_check(self):
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = gradient_check(self.act, x)
        self.assertLess(result["max_rel_error"], 1e-5)

    def test_saturation(self):
        x = np.array([100.0, -100.0])
        result = self.act(x)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], -1.0)


class TestGELU(unittest.TestCase):

    def test_at_zero_approximate(self):
        act = GELU(approximate=True)
        x = np.array([0.0])
        self.assertAlmostEqual(act(x)[0], 0.0)

    def test_at_zero_exact(self):
        act = GELU(approximate=False)
        x = np.array([0.0])
        self.assertAlmostEqual(act(x)[0], 0.0)

    def test_large_positive(self):
        act = GELU(approximate=True)
        x = np.array([100.0])
        result = act(x)
        self.assertTrue(np.isfinite(result[0]))
        np.testing.assert_allclose(result, [100.0], atol=0.1)

    def test_large_negative(self):
        act = GELU(approximate=True)
        x = np.array([-100.0])
        result = act(x)
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_approx_vs_exact(self):
        """Tanh approximation should be within 0.005 of exact for x in [-10, 10]."""
        x = np.linspace(-10, 10, 1000)
        approx = GELU(approximate=True)(x)
        exact = GELU(approximate=False)(x)
        max_diff = np.max(np.abs(approx - exact))
        self.assertLess(max_diff, 0.005)

    def test_gradient_check_approximate(self):
        act = GELU(approximate=True)
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = gradient_check(act, x)
        self.assertLess(result["max_rel_error"], 1e-5)

    def test_gradient_check_exact(self):
        act = GELU(approximate=False)
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = gradient_check(act, x)
        self.assertLess(result["max_rel_error"], 1e-5)

    def test_no_nan_inf(self):
        x = np.linspace(-1000, 1000, 2000)
        for approx in [True, False]:
            act = GELU(approximate=approx)
            result = act(x)
            self.assertTrue(np.all(np.isfinite(result)), f"approx={approx}")


class TestSiLU(unittest.TestCase):

    def setUp(self):
        self.act = SiLU()

    def test_at_zero(self):
        x = np.array([0.0])
        self.assertAlmostEqual(self.act(x)[0], 0.0)

    def test_large_positive(self):
        x = np.array([100.0])
        result = self.act(x)
        self.assertTrue(np.isfinite(result[0]))
        np.testing.assert_allclose(result, [100.0], atol=0.1)

    def test_large_negative(self):
        x = np.array([-100.0])
        result = self.act(x)
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_gradient_check(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = gradient_check(self.act, x)
        self.assertLess(result["max_rel_error"], 1e-5)

    def test_swish_alias(self):
        self.assertIs(Swish, SiLU)

    def test_no_nan_inf(self):
        x = np.linspace(-1000, 1000, 2000)
        result = self.act(x)
        self.assertTrue(np.all(np.isfinite(result)))


class TestShapePreservation(unittest.TestCase):
    """All activations must preserve input shape."""

    ACTIVATIONS = [
        ReLU(),
        LeakyReLU(),
        Sigmoid(),
        Tanh(),
        GELU(approximate=True),
        GELU(approximate=False),
        SiLU(),
    ]

    def test_1d(self):
        x = np.random.randn(10)
        for act in self.ACTIVATIONS:
            result = act(x)
            self.assertEqual(result.shape, x.shape, f"{act.__class__.__name__}")

    def test_2d(self):
        x = np.random.randn(32, 784)
        for act in self.ACTIVATIONS:
            result = act(x)
            self.assertEqual(result.shape, x.shape, f"{act.__class__.__name__}")

    def test_3d(self):
        x = np.random.randn(16, 128, 512)
        for act in self.ACTIVATIONS:
            result = act(x)
            self.assertEqual(result.shape, x.shape, f"{act.__class__.__name__}")

    def test_4d(self):
        x = np.random.randn(2, 3, 4, 5)
        for act in self.ACTIVATIONS:
            result = act(x)
            self.assertEqual(result.shape, x.shape, f"{act.__class__.__name__}")

    def test_gradient_shape_matches(self):
        x = np.random.randn(4, 8, 16)
        grad = np.ones_like(x)
        for act in self.ACTIVATIONS:
            act.forward(x)
            grad_out = act.backward(grad)
            self.assertEqual(grad_out.shape, x.shape, f"{act.__class__.__name__}")


class TestEmptyInput(unittest.TestCase):

    ACTIVATIONS = [
        ReLU(),
        LeakyReLU(),
        Sigmoid(),
        Tanh(),
        GELU(approximate=True),
        GELU(approximate=False),
        SiLU(),
    ]

    def test_empty_array(self):
        x = np.array([])
        for act in self.ACTIVATIONS:
            result = act(x)
            self.assertEqual(result.shape, (0,), f"{act.__class__.__name__}")


class TestSingleElement(unittest.TestCase):

    ACTIVATIONS = [
        ReLU(),
        LeakyReLU(),
        Sigmoid(),
        Tanh(),
        GELU(approximate=True),
        SiLU(),
    ]

    def test_single_element(self):
        x = np.array([0.5])
        for act in self.ACTIVATIONS:
            result = act(x)
            self.assertEqual(result.shape, (1,), f"{act.__class__.__name__}")
            self.assertTrue(np.isfinite(result[0]), f"{act.__class__.__name__}")


class TestBackwardWithoutForward(unittest.TestCase):

    ACTIVATIONS = [
        ReLU(),
        LeakyReLU(),
        Sigmoid(),
        Tanh(),
        GELU(approximate=True),
        GELU(approximate=False),
        SiLU(),
    ]

    def test_raises_error(self):
        grad = np.ones(5)
        for act in self.ACTIVATIONS:
            with self.assertRaises(RuntimeError, msg=f"{act.__class__.__name__}"):
                act.backward(grad)


class TestCacheIndependence(unittest.TestCase):
    """Modifying input after forward should not affect backward."""

    def test_relu_cache_independence(self):
        act = ReLU()
        x = np.array([1.0, -1.0, 2.0])
        act.forward(x)
        x[:] = 0.0  # mutate original
        grad = act.backward(np.ones(3))
        expected = np.array([1.0, 0.0, 1.0])
        np.testing.assert_array_equal(grad, expected)

    def test_sigmoid_cache_independence(self):
        act = Sigmoid()
        x = np.array([0.0])
        act.forward(x)
        x[:] = 100.0  # mutate original
        grad = act.backward(np.ones(1))
        self.assertAlmostEqual(grad[0], 0.25)

    def test_silu_cache_independence(self):
        act = SiLU()
        x = np.array([1.0, -1.0])
        act.forward(x)
        expected_grad = act.backward(np.ones(2)).copy()
        x[:] = 999.0  # mutate original
        act2 = SiLU()
        x2 = np.array([1.0, -1.0])
        act2.forward(x2)
        grad2 = act2.backward(np.ones(2))
        np.testing.assert_allclose(expected_grad, grad2)


class TestVanishingGradients(unittest.TestCase):
    """Demonstrate vanishing gradient in sigmoid for large |x|."""

    def test_sigmoid_gradient_shrinks(self):
        act = Sigmoid()
        for val in [0.0, 2.0, 5.0, 10.0, 50.0]:
            x = np.array([val])
            act.forward(x)
            grad = act.backward(np.ones(1))
            if val == 0.0:
                max_grad = grad[0]
            else:
                self.assertLess(grad[0], max_grad)

    def test_dying_relu(self):
        """ReLU has zero gradient for persistently negative input."""
        act = ReLU()
        x = np.array([-5.0, -3.0, -1.0, -0.1])
        act.forward(x)
        grad = act.backward(np.ones(4))
        np.testing.assert_array_equal(grad, np.zeros(4))

    def test_leaky_relu_survives(self):
        """Leaky ReLU maintains non-zero gradient for negative input."""
        act = LeakyReLU(alpha=0.01)
        x = np.array([-5.0, -3.0, -1.0, -0.1])
        act.forward(x)
        grad = act.backward(np.ones(4))
        self.assertTrue(np.all(grad > 0))


if __name__ == "__main__":
    unittest.main()
