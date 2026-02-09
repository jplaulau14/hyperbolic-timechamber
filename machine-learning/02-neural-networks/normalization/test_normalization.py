"""Tests for normalization layers (LayerNorm, RMSNorm)."""

import unittest
import numpy as np
from implementation import LayerNorm, RMSNorm, gradient_check


class TestLayerNormForward(unittest.TestCase):

    def test_output_has_zero_mean_unit_variance(self):
        """After normalization (before gamma/beta), mean ~ 0 and var ~ 1."""
        ln = LayerNorm(64)
        x = np.random.randn(4, 64)
        y = ln.forward(x)
        np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-10)
        # Biased variance of normalized output is (D-1)/D due to the normalization
        # constraint, so we use a looser tolerance
        np.testing.assert_allclose(y.var(axis=-1), 1.0, atol=0.02)

    def test_shape_preservation_2d(self):
        ln = LayerNorm(32)
        x = np.random.randn(8, 32)
        y = ln.forward(x)
        self.assertEqual(y.shape, x.shape)

    def test_shape_preservation_3d(self):
        ln = LayerNorm(64)
        x = np.random.randn(2, 10, 64)
        y = ln.forward(x)
        self.assertEqual(y.shape, x.shape)

    def test_default_parameters_give_normalized_output(self):
        """With gamma=1, beta=0, output should equal x_hat."""
        ln = LayerNorm(16)
        x = np.random.randn(3, 16)
        y = ln.forward(x)
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(y, x_hat, atol=1e-12)

    def test_hand_calculated_example(self):
        """Verify against manually computed values for x = [1, 2, 3, 4]."""
        ln = LayerNorm(4)
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        mu = 2.5
        var = np.mean([(1-2.5)**2, (2-2.5)**2, (3-2.5)**2, (4-2.5)**2])
        std_inv = 1.0 / np.sqrt(var + 1e-5)
        expected = (x - mu) * std_inv
        y = ln.forward(x)
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_gamma_beta_affect_output(self):
        ln = LayerNorm(4)
        ln.gamma = np.array([2.0, 2.0, 2.0, 2.0])
        ln.beta = np.array([1.0, 1.0, 1.0, 1.0])
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        y = ln.forward(x)
        mu = 2.5
        var = np.mean([(1-2.5)**2, (2-2.5)**2, (3-2.5)**2, (4-2.5)**2])
        std_inv = 1.0 / np.sqrt(var + 1e-5)
        x_hat = (x - mu) * std_inv
        expected = 2.0 * x_hat + 1.0
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_3d_normalization_over_last_dim(self):
        """Each position in (B, L) should be independently normalized."""
        ln = LayerNorm(32)
        x = np.random.randn(2, 5, 32)
        y = ln.forward(x)
        for b in range(2):
            for l in range(5):
                np.testing.assert_allclose(y[b, l].mean(), 0.0, atol=1e-10)
                np.testing.assert_allclose(y[b, l].var(), 1.0, atol=0.04)


class TestLayerNormBackward(unittest.TestCase):

    def test_gradient_check_2d(self):
        np.random.seed(42)
        ln = LayerNorm(16)
        ln.gamma = np.random.randn(16) * 0.5 + 1.0
        ln.beta = np.random.randn(16) * 0.1
        x = np.random.randn(4, 16)
        result = gradient_check(ln, x)
        self.assertTrue(result["passed"], f"Gradient check failed: {result}")

    def test_gradient_check_3d(self):
        np.random.seed(43)
        ln = LayerNorm(8)
        ln.gamma = np.random.randn(8) * 0.5 + 1.0
        ln.beta = np.random.randn(8) * 0.1
        x = np.random.randn(2, 3, 8)
        result = gradient_check(ln, x)
        self.assertTrue(result["passed"], f"Gradient check failed: {result}")

    def test_gradient_shapes(self):
        ln = LayerNorm(64)
        x = np.random.randn(4, 64)
        ln.forward(x)
        grad = np.random.randn(4, 64)
        dx = ln.backward(grad)
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(ln.grad_gamma.shape, (64,))
        self.assertEqual(ln.grad_beta.shape, (64,))

    def test_gradient_shapes_3d(self):
        ln = LayerNorm(32)
        x = np.random.randn(2, 5, 32)
        ln.forward(x)
        grad = np.random.randn(2, 5, 32)
        dx = ln.backward(grad)
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(ln.grad_gamma.shape, (32,))
        self.assertEqual(ln.grad_beta.shape, (32,))

    def test_backward_before_forward_raises(self):
        ln = LayerNorm(8)
        with self.assertRaises(RuntimeError):
            ln.backward(np.ones((2, 8)))

    def test_gradient_check_various_shapes(self):
        """Test gradient correctness across multiple input shapes."""
        shapes = [(4, 16), (2, 5, 8), (1, 1, 4), (8, 4)]
        for shape in shapes:
            D = shape[-1]
            np.random.seed(44)
            ln = LayerNorm(D)
            ln.gamma = np.random.randn(D) * 0.5 + 1.0
            ln.beta = np.random.randn(D) * 0.1
            x = np.random.randn(*shape)
            result = gradient_check(ln, x)
            self.assertTrue(result["passed"], f"Failed for shape {shape}: {result}")


class TestRMSNormForward(unittest.TestCase):

    def test_output_has_unit_rms(self):
        """After RMSNorm (before gamma), RMS ~ 1."""
        rn = RMSNorm(64)
        x = np.random.randn(4, 64)
        y = rn.forward(x)
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-5)

    def test_shape_preservation_2d(self):
        rn = RMSNorm(32)
        x = np.random.randn(8, 32)
        y = rn.forward(x)
        self.assertEqual(y.shape, x.shape)

    def test_shape_preservation_3d(self):
        rn = RMSNorm(64)
        x = np.random.randn(2, 10, 64)
        y = rn.forward(x)
        self.assertEqual(y.shape, x.shape)

    def test_default_parameters_give_normalized_output(self):
        rn = RMSNorm(16)
        x = np.random.randn(3, 16)
        y = rn.forward(x)
        ms = np.mean(x ** 2, axis=-1, keepdims=True)
        rms_inv = 1.0 / np.sqrt(ms + 1e-6)
        expected = x * rms_inv
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_hand_calculated_example(self):
        rn = RMSNorm(4)
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        ms = (1 + 4 + 9 + 16) / 4.0  # 7.5
        rms_inv = 1.0 / np.sqrt(ms + 1e-6)
        expected = x * rms_inv
        y = rn.forward(x)
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_no_beta_parameter(self):
        rn = RMSNorm(8)
        self.assertFalse(hasattr(rn, "beta"))

    def test_3d_normalization_over_last_dim(self):
        rn = RMSNorm(32)
        x = np.random.randn(2, 5, 32)
        y = rn.forward(x)
        for b in range(2):
            for l in range(5):
                rms = np.sqrt(np.mean(y[b, l] ** 2))
                np.testing.assert_allclose(rms, 1.0, atol=1e-5)


class TestRMSNormBackward(unittest.TestCase):

    def test_gradient_check_2d(self):
        np.random.seed(42)
        rn = RMSNorm(16)
        rn.gamma = np.random.randn(16) * 0.5 + 1.0
        x = np.random.randn(4, 16)
        result = gradient_check(rn, x)
        self.assertTrue(result["passed"], f"Gradient check failed: {result}")

    def test_gradient_check_3d(self):
        np.random.seed(43)
        rn = RMSNorm(8)
        rn.gamma = np.random.randn(8) * 0.5 + 1.0
        x = np.random.randn(2, 3, 8)
        result = gradient_check(rn, x)
        self.assertTrue(result["passed"], f"Gradient check failed: {result}")

    def test_gradient_shapes(self):
        rn = RMSNorm(64)
        x = np.random.randn(4, 64)
        rn.forward(x)
        grad = np.random.randn(4, 64)
        dx = rn.backward(grad)
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(rn.grad_gamma.shape, (64,))

    def test_backward_before_forward_raises(self):
        rn = RMSNorm(8)
        with self.assertRaises(RuntimeError):
            rn.backward(np.ones((2, 8)))

    def test_gradient_check_various_shapes(self):
        shapes = [(4, 16), (2, 5, 8), (1, 1, 4), (8, 4)]
        for shape in shapes:
            D = shape[-1]
            np.random.seed(44)
            rn = RMSNorm(D)
            rn.gamma = np.random.randn(D) * 0.5 + 1.0
            x = np.random.randn(*shape)
            result = gradient_check(rn, x)
            self.assertTrue(result["passed"], f"Failed for shape {shape}: {result}")


class TestNumericalStability(unittest.TestCase):

    def test_layernorm_identical_inputs(self):
        """Zero variance -- epsilon prevents NaN/Inf."""
        ln = LayerNorm(4)
        x = np.full((2, 4), 5.0)
        y = ln.forward(x)
        self.assertFalse(np.any(np.isnan(y)))
        self.assertFalse(np.any(np.isinf(y)))
        dx = ln.backward(np.ones_like(y))
        self.assertFalse(np.any(np.isnan(dx)))
        self.assertFalse(np.any(np.isinf(dx)))

    def test_rmsnorm_identical_inputs(self):
        rn = RMSNorm(4)
        x = np.full((2, 4), 5.0)
        y = rn.forward(x)
        self.assertFalse(np.any(np.isnan(y)))
        self.assertFalse(np.any(np.isinf(y)))
        dx = rn.backward(np.ones_like(y))
        self.assertFalse(np.any(np.isnan(dx)))
        self.assertFalse(np.any(np.isinf(dx)))

    def test_layernorm_all_zeros(self):
        ln = LayerNorm(4)
        x = np.zeros((2, 4))
        y = ln.forward(x)
        self.assertFalse(np.any(np.isnan(y)))
        self.assertFalse(np.any(np.isinf(y)))
        expected = np.broadcast_to(ln.beta, y.shape)
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_rmsnorm_all_zeros(self):
        rn = RMSNorm(4)
        x = np.zeros((2, 4))
        y = rn.forward(x)
        self.assertFalse(np.any(np.isnan(y)))
        self.assertFalse(np.any(np.isinf(y)))
        np.testing.assert_allclose(y, 0.0, atol=1e-12)

    def test_layernorm_very_small_inputs(self):
        ln = LayerNorm(8)
        x = np.random.randn(2, 8) * 1e-10
        y = ln.forward(x)
        self.assertFalse(np.any(np.isnan(y)))
        self.assertFalse(np.any(np.isinf(y)))
        dx = ln.backward(np.ones_like(y))
        self.assertFalse(np.any(np.isnan(dx)))
        self.assertFalse(np.any(np.isinf(dx)))

    def test_layernorm_very_large_inputs(self):
        ln = LayerNorm(8)
        x = np.random.randn(2, 8) * 1e10
        y = ln.forward(x)
        self.assertFalse(np.any(np.isnan(y)))
        self.assertFalse(np.any(np.isinf(y)))
        dx = ln.backward(np.ones_like(y))
        self.assertFalse(np.any(np.isnan(dx)))
        self.assertFalse(np.any(np.isinf(dx)))

    def test_rmsnorm_very_small_inputs(self):
        rn = RMSNorm(8)
        x = np.random.randn(2, 8) * 1e-10
        y = rn.forward(x)
        self.assertFalse(np.any(np.isnan(y)))
        self.assertFalse(np.any(np.isinf(y)))
        dx = rn.backward(np.ones_like(y))
        self.assertFalse(np.any(np.isnan(dx)))
        self.assertFalse(np.any(np.isinf(dx)))

    def test_rmsnorm_very_large_inputs(self):
        rn = RMSNorm(8)
        x = np.random.randn(2, 8) * 1e10
        y = rn.forward(x)
        self.assertFalse(np.any(np.isnan(y)))
        self.assertFalse(np.any(np.isinf(y)))
        dx = rn.backward(np.ones_like(y))
        self.assertFalse(np.any(np.isnan(dx)))
        self.assertFalse(np.any(np.isinf(dx)))

    def test_layernorm_mixed_scales(self):
        """Catastrophic cancellation test: values at vastly different scales."""
        ln = LayerNorm(2)
        x = np.array([[1e-8, 1e8]])
        y = ln.forward(x)
        self.assertFalse(np.any(np.isnan(y)))
        self.assertFalse(np.any(np.isinf(y)))


class TestEdgeCases(unittest.TestCase):

    def test_batch_size_1_2d(self):
        ln = LayerNorm(16)
        x = np.random.randn(1, 16)
        y = ln.forward(x)
        self.assertEqual(y.shape, (1, 16))

    def test_batch_size_1_3d(self):
        ln = LayerNorm(16)
        x = np.random.randn(1, 5, 16)
        y = ln.forward(x)
        self.assertEqual(y.shape, (1, 5, 16))

    def test_sequence_length_1(self):
        ln = LayerNorm(32)
        x = np.random.randn(4, 1, 32)
        y = ln.forward(x)
        self.assertEqual(y.shape, (4, 1, 32))

    def test_feature_dim_1_layernorm(self):
        """D=1: variance is always 0, so x_hat = 0, output = beta."""
        ln = LayerNorm(1)
        x = np.array([[3.0], [7.0]])
        y = ln.forward(x)
        expected = np.broadcast_to(ln.beta, y.shape)
        np.testing.assert_allclose(y, expected, atol=1e-5)

    def test_single_element(self):
        ln = LayerNorm(1)
        x = np.array([[[5.0]]])
        y = ln.forward(x)
        self.assertEqual(y.shape, (1, 1, 1))
        self.assertFalse(np.any(np.isnan(y)))

    def test_rmsnorm_batch_size_1(self):
        rn = RMSNorm(16)
        x = np.random.randn(1, 16)
        y = rn.forward(x)
        self.assertEqual(y.shape, (1, 16))


class TestLearnableParameters(unittest.TestCase):

    def test_layernorm_gamma_initialized_ones(self):
        ln = LayerNorm(32)
        np.testing.assert_array_equal(ln.gamma, np.ones(32))

    def test_layernorm_beta_initialized_zeros(self):
        ln = LayerNorm(32)
        np.testing.assert_array_equal(ln.beta, np.zeros(32))

    def test_rmsnorm_gamma_initialized_ones(self):
        rn = RMSNorm(32)
        np.testing.assert_array_equal(rn.gamma, np.ones(32))

    def test_each_backward_produces_fresh_gradients(self):
        """Gradients should not accumulate across backward calls."""
        ln = LayerNorm(8)
        x1 = np.random.randn(2, 8)
        x2 = np.random.randn(2, 8)

        ln.forward(x1)
        ln.backward(np.ones((2, 8)))
        grad_gamma_1 = ln.grad_gamma.copy()

        ln.forward(x2)
        ln.backward(np.ones((2, 8)))
        grad_gamma_2 = ln.grad_gamma.copy()

        self.assertFalse(np.allclose(grad_gamma_1, grad_gamma_2))

    def test_parameter_update_changes_output(self):
        ln = LayerNorm(8)
        x = np.random.randn(2, 8)
        y_before = ln.forward(x).copy()
        ln.backward(np.ones_like(y_before))
        ln.gamma -= 0.1 * ln.grad_gamma
        ln.beta -= 0.1 * ln.grad_beta
        y_after = ln.forward(x)
        self.assertFalse(np.allclose(y_before, y_after))


class TestPyTorchComparison(unittest.TestCase):
    """Compare against PyTorch reference (skipped if torch not available)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
            cls.has_torch = True
        except ImportError:
            cls.has_torch = False

    def test_layernorm_forward_matches_pytorch(self):
        if not self.has_torch:
            self.skipTest("PyTorch not installed")

        torch = self.torch
        np.random.seed(42)
        x_np = np.random.randn(2, 5, 64).astype(np.float64)
        x_torch = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)

        ln_torch = torch.nn.LayerNorm(64).double()
        gamma_np = ln_torch.weight.detach().numpy().copy()
        beta_np = ln_torch.bias.detach().numpy().copy()

        y_torch = ln_torch(x_torch)
        y_torch.sum().backward()

        ln_np = LayerNorm(64)
        ln_np.gamma = gamma_np.copy()
        ln_np.beta = beta_np.copy()
        y_np = ln_np.forward(x_np)
        dx_np = ln_np.backward(np.ones_like(y_np))

        np.testing.assert_allclose(y_np, y_torch.detach().numpy(), atol=1e-10)
        np.testing.assert_allclose(dx_np, x_torch.grad.numpy(), atol=1e-10)
        np.testing.assert_allclose(
            ln_np.grad_gamma, ln_torch.weight.grad.numpy(), atol=1e-10
        )
        np.testing.assert_allclose(
            ln_np.grad_beta, ln_torch.bias.grad.numpy(), atol=1e-10
        )

    def test_layernorm_backward_matches_pytorch_nontrivial(self):
        """Test with non-default gamma/beta and random grad_output."""
        if not self.has_torch:
            self.skipTest("PyTorch not installed")

        torch = self.torch
        np.random.seed(99)
        x_np = np.random.randn(3, 4, 16).astype(np.float64)
        grad_np = np.random.randn(3, 4, 16).astype(np.float64)
        x_torch = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)

        ln_torch = torch.nn.LayerNorm(16).double()
        with torch.no_grad():
            ln_torch.weight.copy_(torch.randn(16, dtype=torch.float64) * 0.5 + 1.0)
            ln_torch.bias.copy_(torch.randn(16, dtype=torch.float64) * 0.1)

        y_torch = ln_torch(x_torch)
        y_torch.backward(torch.tensor(grad_np, dtype=torch.float64))

        ln_np = LayerNorm(16)
        ln_np.gamma = ln_torch.weight.detach().numpy().copy()
        ln_np.beta = ln_torch.bias.detach().numpy().copy()
        y_np = ln_np.forward(x_np)
        dx_np = ln_np.backward(grad_np)

        np.testing.assert_allclose(y_np, y_torch.detach().numpy(), atol=1e-10)
        np.testing.assert_allclose(dx_np, x_torch.grad.numpy(), atol=1e-10)

    def test_rmsnorm_forward_matches_reference(self):
        """Compare RMSNorm against the LLaMA-style PyTorch reference."""
        if not self.has_torch:
            self.skipTest("PyTorch not installed")

        torch = self.torch
        np.random.seed(42)
        x_np = np.random.randn(2, 5, 64).astype(np.float64)
        x_torch = torch.tensor(x_np, dtype=torch.float64)

        weight = torch.ones(64, dtype=torch.float64)
        rms = torch.sqrt(torch.mean(x_torch ** 2, dim=-1, keepdim=True) + 1e-6)
        y_ref = (x_torch / rms * weight).numpy()

        rn = RMSNorm(64)
        y_np = rn.forward(x_np)
        np.testing.assert_allclose(y_np, y_ref, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
