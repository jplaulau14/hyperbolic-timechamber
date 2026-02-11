"""Tests for self-attention."""

import unittest
import numpy as np
from implementation import (
    softmax,
    softmax_backward,
    create_causal_mask,
    create_padding_mask,
    scaled_dot_product_attention,
    SelfAttention,
    count_flops,
    count_memory_bytes,
)


class TestSoftmax(unittest.TestCase):

    def test_sums_to_one(self):
        x = np.random.randn(2, 5, 8)
        out = softmax(x)
        np.testing.assert_allclose(out.sum(axis=-1), np.ones((2, 5)), atol=1e-12)

    def test_known_values(self):
        x = np.array([[[0.0, 0.0]]])
        out = softmax(x)
        np.testing.assert_allclose(out, [[[0.5, 0.5]]], atol=1e-12)

    def test_large_values_no_overflow(self):
        x = np.array([[[1000.0, 1001.0, 999.0]]])
        out = softmax(x)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-12)

    def test_negative_inf_becomes_zero(self):
        x = np.array([[[0.0, -np.inf, 0.0]]])
        out = softmax(x)
        np.testing.assert_allclose(out[0, 0, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-12)


class TestSoftmaxBackward(unittest.TestCase):

    def test_numerical_gradient(self):
        np.random.seed(42)
        x = np.random.randn(2, 4, 6)
        A = softmax(x)
        g = np.random.randn(*A.shape)

        analytical = softmax_backward(g, A)

        h = 1e-5
        numerical = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[idx] += h
            x_minus[idx] -= h
            numerical[idx] = np.sum(g * (softmax(x_plus) - softmax(x_minus))) / (2.0 * h)

        rel_err = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
        self.assertLess(rel_err.max(), 1e-5)


class TestCausalMask(unittest.TestCase):

    def test_shape(self):
        mask = create_causal_mask(5)
        self.assertEqual(mask.shape, (5, 5))

    def test_lower_triangular_zeros(self):
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i + 1):
                self.assertEqual(mask[i, j], 0.0)

    def test_upper_triangular_neginf(self):
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertEqual(mask[i, j], -np.inf)

    def test_seq_len_1(self):
        mask = create_causal_mask(1)
        np.testing.assert_array_equal(mask, np.array([[0.0]]))


class TestPaddingMask(unittest.TestCase):

    def test_shape(self):
        lengths = np.array([3, 2])
        mask = create_padding_mask(lengths, max_len=4)
        self.assertEqual(mask.shape, (2, 1, 4))

    def test_valid_positions_are_zero(self):
        lengths = np.array([3])
        mask = create_padding_mask(lengths, max_len=5)
        np.testing.assert_array_equal(mask[0, 0, :3], [0.0, 0.0, 0.0])

    def test_padded_positions_are_neginf(self):
        lengths = np.array([3])
        mask = create_padding_mask(lengths, max_len=5)
        self.assertTrue(np.all(mask[0, 0, 3:] == -np.inf))


class TestScaledDotProductAttention(unittest.TestCase):

    def test_weights_sum_to_one(self):
        np.random.seed(0)
        Q = np.random.randn(2, 5, 8)
        K = np.random.randn(2, 5, 8)
        V = np.random.randn(2, 5, 8)
        _, A = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(A.sum(axis=-1), np.ones((2, 5)), atol=1e-6)

    def test_output_shape(self):
        Q = np.random.randn(3, 4, 8)
        K = np.random.randn(3, 4, 8)
        V = np.random.randn(3, 4, 6)
        out, A = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(out.shape, (3, 4, 6))
        self.assertEqual(A.shape, (3, 4, 4))

    def test_known_values_identity(self):
        """With identity-like Q and K, attention should focus on matching positions."""
        Q = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        K = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        V = np.array([[[10.0, 20.0], [30.0, 40.0]]])

        out, A = scaled_dot_product_attention(Q, K, V)

        self.assertTrue(A[0, 0, 0] > A[0, 0, 1])
        self.assertTrue(A[0, 1, 1] > A[0, 1, 0])

    def test_causal_mask_zeros_future(self):
        np.random.seed(1)
        Q = np.random.randn(1, 4, 8)
        K = np.random.randn(1, 4, 8)
        V = np.random.randn(1, 4, 8)
        mask = create_causal_mask(4)
        _, A = scaled_dot_product_attention(Q, K, V, mask=mask)

        for i in range(4):
            for j in range(i + 1, 4):
                self.assertAlmostEqual(A[0, i, j], 0.0, places=10)

    def test_seq_len_1(self):
        Q = np.random.randn(2, 1, 4)
        K = np.random.randn(2, 1, 4)
        V = np.random.randn(2, 1, 4)
        _, A = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(A, np.ones((2, 1, 1)), atol=1e-12)

    def test_causal_mask_seq_len_1(self):
        Q = np.random.randn(1, 1, 4)
        K = np.random.randn(1, 1, 4)
        V = np.random.randn(1, 1, 4)
        mask = create_causal_mask(1)
        _, A = scaled_dot_product_attention(Q, K, V, mask=mask)
        np.testing.assert_allclose(A, np.ones((1, 1, 1)), atol=1e-12)

    def test_padding_mask(self):
        np.random.seed(2)
        Q = np.random.randn(1, 4, 8)
        K = np.random.randn(1, 4, 8)
        V = np.random.randn(1, 4, 8)
        lengths = np.array([2])
        mask = create_padding_mask(lengths, max_len=4)
        _, A = scaled_dot_product_attention(Q, K, V, mask=mask)

        np.testing.assert_allclose(A[0, :, 2:], 0.0, atol=1e-12)
        np.testing.assert_allclose(A.sum(axis=-1), 1.0, atol=1e-6)


class TestSelfAttention(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_output_shape(self):
        sa = SelfAttention(d_model=16, d_k=8, d_v=8)
        X = np.random.randn(2, 5, 16)
        out = sa.forward(X)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_output_shape_non_square(self):
        sa = SelfAttention(d_model=16, d_k=8, d_v=12, d_out=16)
        X = np.random.randn(2, 5, 16)
        out = sa.forward(X)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_output_shape_custom_d_out(self):
        sa = SelfAttention(d_model=16, d_k=8, d_v=8, d_out=32)
        X = np.random.randn(2, 5, 16)
        out = sa.forward(X)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_defaults(self):
        sa = SelfAttention(d_model=16)
        self.assertEqual(sa.d_k, 16)
        self.assertEqual(sa.d_v, 16)
        self.assertEqual(sa.d_out, 16)

    def test_no_bias(self):
        sa = SelfAttention(d_model=8, d_k=4, d_v=4, use_bias=False)
        X = np.random.randn(1, 3, 8)
        out = sa.forward(X)
        self.assertEqual(out.shape, (1, 3, 8))
        self.assertIsNone(sa.b_Q)

    def test_batch_independence(self):
        sa = SelfAttention(d_model=8, d_k=4, d_v=4)
        x0 = np.random.randn(1, 3, 8)
        x1 = np.random.randn(1, 3, 8)

        X_batch = np.concatenate([x0, x1], axis=0)
        out_batch = sa.forward(X_batch)

        out_0 = sa.forward(x0)
        out_1 = sa.forward(x1)

        np.testing.assert_allclose(out_batch[0], out_0[0], atol=1e-12)
        np.testing.assert_allclose(out_batch[1], out_1[0], atol=1e-12)

    def test_variable_batch_sizes(self):
        sa = SelfAttention(d_model=8, d_k=4, d_v=4)
        for B in [1, 4, 16]:
            X = np.random.randn(B, 3, 8)
            out = sa.forward(X)
            self.assertEqual(out.shape, (B, 3, 8))

    def test_forward_with_causal_mask(self):
        sa = SelfAttention(d_model=8, d_k=4, d_v=4)
        X = np.random.randn(2, 5, 8)
        mask = create_causal_mask(5)
        out = sa.forward(X, mask=mask)
        self.assertEqual(out.shape, (2, 5, 8))

        A = sa._cache["A"]
        for b in range(2):
            for i in range(5):
                for j in range(i + 1, 5):
                    self.assertAlmostEqual(A[b, i, j], 0.0, places=10)

    def test_attention_weights_cached(self):
        sa = SelfAttention(d_model=8, d_k=4, d_v=4)
        X = np.random.randn(1, 3, 8)
        sa.forward(X)
        A = sa._cache["A"]
        self.assertEqual(A.shape, (1, 3, 3))
        np.testing.assert_allclose(A.sum(axis=-1), 1.0, atol=1e-6)


class TestSelfAttentionBackward(unittest.TestCase):

    def _numerical_gradient(self, sa, X, param_name, mask=None, h=1e-5):
        """Compute numerical gradient for a parameter using central differences."""
        param = getattr(sa, param_name)
        grad_num = np.zeros_like(param)
        grad_output = self._grad_output

        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]

            param[idx] = old_val + h
            out_plus = sa.forward(X, mask=mask)
            loss_plus = np.sum(out_plus * grad_output)

            param[idx] = old_val - h
            out_minus = sa.forward(X, mask=mask)
            loss_minus = np.sum(out_minus * grad_output)

            grad_num[idx] = (loss_plus - loss_minus) / (2.0 * h)
            param[idx] = old_val
            it.iternext()

        return grad_num

    def _numerical_gradient_input(self, sa, X, mask=None, h=1e-5):
        """Compute numerical gradient for the input X."""
        grad_num = np.zeros_like(X)
        grad_output = self._grad_output

        it = np.nditer(X, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old_val = X[idx]

            X[idx] = old_val + h
            out_plus = sa.forward(X, mask=mask)
            loss_plus = np.sum(out_plus * grad_output)

            X[idx] = old_val - h
            out_minus = sa.forward(X, mask=mask)
            loss_minus = np.sum(out_minus * grad_output)

            grad_num[idx] = (loss_plus - loss_minus) / (2.0 * h)
            X[idx] = old_val
            it.iternext()

        return grad_num

    def _check_gradient(self, analytical, numerical, name, tol=1e-5):
        rel_err = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
        max_err = rel_err.max()
        self.assertLess(max_err, tol, f"Gradient check failed for {name}: max relative error = {max_err:.2e}")

    def test_gradient_check_no_mask(self):
        np.random.seed(7)
        sa = SelfAttention(d_model=6, d_k=4, d_v=4, d_out=6)
        X = np.random.randn(1, 3, 6) * 0.5

        out = sa.forward(X)
        self._grad_output = np.random.randn(*out.shape)
        grad_X = sa.backward(self._grad_output)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            num = self._numerical_gradient(sa, X, param_name)
            ana = getattr(sa, f"grad_{param_name}")
            self._check_gradient(ana, num, param_name)

        for bias_name in ["b_Q", "b_K", "b_V", "b_O"]:
            num = self._numerical_gradient(sa, X, bias_name)
            ana = getattr(sa, f"grad_{bias_name}")
            self._check_gradient(ana, num, bias_name)

        num_X = self._numerical_gradient_input(sa, X)
        sa.forward(X)
        grad_X = sa.backward(self._grad_output)
        self._check_gradient(grad_X, num_X, "X")

    def test_gradient_check_with_causal_mask(self):
        np.random.seed(14)
        sa = SelfAttention(d_model=6, d_k=4, d_v=4, d_out=6)
        X = np.random.randn(1, 4, 6) * 0.5
        mask = create_causal_mask(4)

        out = sa.forward(X, mask=mask)
        self._grad_output = np.random.randn(*out.shape)
        grad_X = sa.backward(self._grad_output)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            num = self._numerical_gradient(sa, X, param_name, mask=mask)
            ana = getattr(sa, f"grad_{param_name}")
            self._check_gradient(ana, num, param_name)

        num_X = self._numerical_gradient_input(sa, X, mask=mask)
        sa.forward(X, mask=mask)
        grad_X = sa.backward(self._grad_output)
        self._check_gradient(grad_X, num_X, "X")

    def test_gradient_check_no_bias(self):
        np.random.seed(789)
        sa = SelfAttention(d_model=4, d_k=3, d_v=3, d_out=4, use_bias=False)
        X = np.random.randn(1, 3, 4)

        out = sa.forward(X)
        self._grad_output = np.random.randn(*out.shape)
        grad_X = sa.backward(self._grad_output)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            num = self._numerical_gradient(sa, X, param_name)
            ana = getattr(sa, f"grad_{param_name}")
            self._check_gradient(ana, num, param_name)

        num_X = self._numerical_gradient_input(sa, X)
        sa.forward(X)
        grad_X = sa.backward(self._grad_output)
        self._check_gradient(grad_X, num_X, "X")

    def test_gradient_check_batch_size_2(self):
        np.random.seed(101)
        sa = SelfAttention(d_model=4, d_k=3, d_v=3, d_out=4)
        X = np.random.randn(2, 3, 4)

        out = sa.forward(X)
        self._grad_output = np.random.randn(*out.shape)
        grad_X = sa.backward(self._grad_output)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            num = self._numerical_gradient(sa, X, param_name)
            ana = getattr(sa, f"grad_{param_name}")
            self._check_gradient(ana, num, param_name)

        num_X = self._numerical_gradient_input(sa, X)
        sa.forward(X)
        grad_X = sa.backward(self._grad_output)
        self._check_gradient(grad_X, num_X, "X")

    def test_causal_mask_blocks_gradient_flow(self):
        """Gradients should not flow through masked (future) positions."""
        np.random.seed(55)
        sa = SelfAttention(d_model=4, d_k=3, d_v=3, d_out=4)
        mask = create_causal_mask(3)

        X = np.random.randn(1, 3, 4)
        out = sa.forward(X, mask=mask)
        grad_out = np.zeros_like(out)
        grad_out[0, 0, :] = np.random.randn(4)

        grad_X = sa.backward(grad_out)

        # Position 0 with causal mask only attends to position 0.
        # Gradient w.r.t. positions 1 and 2 from grad_out at position 0 should
        # only come from their role as keys/values for position 0's query.
        # The key test: changing X[0,2,:] should NOT affect output at position 0
        # because position 0 cannot see position 2 through the causal mask.
        h = 1e-5
        for d in range(4):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[0, 2, d] += h
            X_minus[0, 2, d] -= h
            out_plus = sa.forward(X_plus, mask=mask)
            out_minus = sa.forward(X_minus, mask=mask)
            numerical_effect = np.abs(out_plus[0, 0, :] - out_minus[0, 0, :]).max() / (2 * h)
            self.assertLess(numerical_effect, 1e-8,
                            "Future position should not affect masked output position")

    def test_backward_raises_without_forward(self):
        sa = SelfAttention(d_model=4, d_k=3, d_v=3)
        with self.assertRaises(RuntimeError):
            sa.backward(np.zeros((1, 3, 4)))


class TestNumericalStability(unittest.TestCase):

    def test_large_dk_without_scaling_saturates(self):
        """Without sqrt(d_k) scaling, large d_k causes softmax saturation."""
        np.random.seed(0)
        d_k = 512
        Q = np.random.randn(1, 10, d_k)
        K = np.random.randn(1, 10, d_k)

        # Without scaling: dot products have variance ~ d_k = 512
        scores_unscaled = Q @ K.transpose(0, 2, 1)
        A_unscaled = softmax(scores_unscaled)
        max_weight = A_unscaled.max(axis=-1)
        # Near-binary attention: max weight should be close to 1.0
        self.assertGreater(max_weight.mean(), 0.9)

        # With proper scaling: variance ~ 1
        scores_scaled = scores_unscaled / np.sqrt(d_k)
        A_scaled = softmax(scores_scaled)
        max_weight_scaled = A_scaled.max(axis=-1)
        # Meaningful distribution: max weight well below 1.0
        self.assertLess(max_weight_scaled.mean(), 0.5)

    def test_large_input_values(self):
        np.random.seed(1)
        sa = SelfAttention(d_model=8, d_k=4, d_v=4)
        X = np.random.uniform(-100, 100, size=(2, 5, 8))
        out = sa.forward(X)
        self.assertTrue(np.all(np.isfinite(out)))

        grad_out = np.random.randn(*out.shape)
        grad_X = sa.backward(grad_out)
        self.assertTrue(np.all(np.isfinite(grad_X)))

    def test_softmax_overflow_prevention(self):
        """Values that would overflow naive exp should be handled."""
        x = np.array([[[800.0, 801.0, 799.0]]])
        out = softmax(x)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-12)

    def test_gradient_stability_near_saturation(self):
        """Backward should produce finite gradients even with near-saturated softmax."""
        np.random.seed(2)
        sa = SelfAttention(d_model=4, d_k=4, d_v=4)
        X = np.random.randn(1, 3, 4) * 50.0
        out = sa.forward(X)
        grad_out = np.random.randn(*out.shape)
        grad_X = sa.backward(grad_out)
        self.assertTrue(np.all(np.isfinite(grad_X)))
        for name in ["grad_W_Q", "grad_W_K", "grad_W_V", "grad_W_O"]:
            self.assertTrue(np.all(np.isfinite(getattr(sa, name))), f"{name} has non-finite values")


class TestCountFlops(unittest.TestCase):

    def test_known_values(self):
        B, n, d_model, d_k, d_v = 1, 4, 8, 4, 4
        flops = count_flops(B, n, d_model, d_k, d_v)

        proj_qkv = 3 * 2 * B * n * d_model * d_k  # Q, K, V all d_k=d_v here
        proj_o = 2 * B * n * d_v * d_model
        qk = 2 * B * n * n * d_k
        av = 2 * B * n * n * d_v
        sm = 5 * B * n * n

        expected = proj_qkv + proj_o + qk + av + sm
        self.assertEqual(flops, expected)

    def test_scales_with_seq_len_squared(self):
        # Use large n and small d_model so quadratic n^2 terms dominate projections
        base = count_flops(1, 128, 8, 8, 8)
        doubled = count_flops(1, 256, 8, 8, 8)
        ratio = doubled / base
        self.assertGreater(ratio, 3.5)


class TestCountMemoryBytes(unittest.TestCase):

    def test_attention_matrix_size(self):
        B, n, d_k, d_v = 1, 16, 8, 8
        mem = count_memory_bytes(B, n, d_k, d_v, dtype="float32")
        attn_matrix_bytes = B * n * n * 4
        self.assertGreater(mem, attn_matrix_bytes)

    def test_memory_scales_quadratically(self):
        mem_16 = count_memory_bytes(1, 16, 8, 8)
        mem_32 = count_memory_bytes(1, 32, 8, 8)
        mem_64 = count_memory_bytes(1, 64, 8, 8)
        mem_128 = count_memory_bytes(1, 128, 8, 8)

        # Ratio should approach 4x as n grows (quadratic dominates)
        ratio_32_16 = mem_32 / mem_16
        ratio_64_32 = mem_64 / mem_32
        ratio_128_64 = mem_128 / mem_64

        self.assertGreater(ratio_128_64, 3.0)
        self.assertGreater(ratio_64_32, ratio_32_16)

    def test_float16_half_of_float32(self):
        mem_fp32 = count_memory_bytes(1, 32, 8, 8, dtype="float32")
        mem_fp16 = count_memory_bytes(1, 32, 8, 8, dtype="float16")
        self.assertEqual(mem_fp16, mem_fp32 // 2)

    def test_attention_matrix_elements(self):
        B, n = 2, 8
        mem = count_memory_bytes(B, n, 4, 4)
        attn_elements = B * n * n
        attn_bytes = attn_elements * 4
        self.assertGreaterEqual(mem, attn_bytes)


if __name__ == "__main__":
    unittest.main()
