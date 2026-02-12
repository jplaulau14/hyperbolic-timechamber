"""Tests for multi-head attention."""

import unittest
import sys
import os
import importlib.util
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from implementation import (
    softmax,
    softmax_backward,
    create_causal_mask,
    MultiHeadAttention,
    count_flops,
    count_memory_bytes,
)

# Import self-attention module under a different name to avoid collision
_sa_path = os.path.join(os.path.dirname(__file__), "..", "self-attention", "implementation.py")
_sa_spec = importlib.util.spec_from_file_location("self_attention_impl", _sa_path)
_sa_mod = importlib.util.module_from_spec(_sa_spec)
_sa_spec.loader.exec_module(_sa_mod)
SelfAttention = _sa_mod.SelfAttention


class TestSoftmax(unittest.TestCase):

    def test_sums_to_one_4d(self):
        x = np.random.randn(2, 4, 5, 5)
        out = softmax(x)
        np.testing.assert_allclose(out.sum(axis=-1), np.ones((2, 4, 5)), atol=1e-12)

    def test_large_values_no_overflow(self):
        x = np.array([[[[1000.0, 1001.0, 999.0]]]])
        out = softmax(x)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-12)

    def test_negative_inf_becomes_zero(self):
        x = np.array([[[[0.0, -np.inf, 0.0]]]])
        out = softmax(x)
        np.testing.assert_allclose(out[0, 0, 0, 1], 0.0, atol=1e-12)


class TestSoftmaxBackward(unittest.TestCase):

    def test_numerical_gradient_4d(self):
        np.random.seed(42)
        x = np.random.randn(2, 3, 4, 4)
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

    def test_shape_4d(self):
        mask = create_causal_mask(5)
        self.assertEqual(mask.shape, (1, 1, 5, 5))

    def test_lower_triangular_zeros(self):
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i + 1):
                self.assertEqual(mask[0, 0, i, j], 0.0)

    def test_upper_triangular_neginf(self):
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertEqual(mask[0, 0, i, j], -np.inf)

    def test_broadcasts_over_batch_and_heads(self):
        mask = create_causal_mask(3)
        scores = np.random.randn(2, 4, 3, 3)
        masked = scores + mask
        self.assertEqual(masked.shape, (2, 4, 3, 3))
        for b in range(2):
            for h in range(4):
                for i in range(3):
                    for j in range(i + 1, 3):
                        self.assertEqual(masked[b, h, i, j], -np.inf)


class TestMultiHeadAttentionShapes(unittest.TestCase):

    def test_output_shape(self):
        np.random.seed(0)
        mha = MultiHeadAttention(d_model=16, num_heads=4)
        X = np.random.randn(2, 5, 16)
        out = mha.forward(X)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_various_batch_sizes(self):
        np.random.seed(1)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        for B in [1, 4, 32]:
            X = np.random.randn(B, 3, 8)
            out = mha.forward(X)
            self.assertEqual(out.shape, (B, 3, 8))

    def test_various_seq_lengths(self):
        np.random.seed(2)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        for L in [1, 16, 128]:
            X = np.random.randn(1, L, 8)
            out = mha.forward(X)
            self.assertEqual(out.shape, (1, L, 8))

    def test_intermediate_shapes(self):
        np.random.seed(3)
        mha = MultiHeadAttention(d_model=12, num_heads=3)
        X = np.random.randn(2, 5, 12)
        mha.forward(X)
        cache = mha._cache

        self.assertEqual(cache["Q"].shape, (2, 3, 5, 4))   # (B, h, L, d_k)
        self.assertEqual(cache["K"].shape, (2, 3, 5, 4))
        self.assertEqual(cache["V"].shape, (2, 3, 5, 4))
        self.assertEqual(cache["A"].shape, (2, 3, 5, 5))    # (B, h, L, L)
        self.assertEqual(cache["attn_output"].shape, (2, 3, 5, 4))  # (B, h, L, d_v)
        self.assertEqual(cache["concat"].shape, (2, 5, 12))  # (B, L, d_model)

    def test_d_model_not_divisible_raises(self):
        with self.assertRaises(AssertionError):
            MultiHeadAttention(d_model=7, num_heads=3)


class TestMultiHeadAttentionNumerical(unittest.TestCase):

    def test_attention_weights_sum_to_one(self):
        np.random.seed(10)
        mha = MultiHeadAttention(d_model=16, num_heads=4)
        X = np.random.randn(2, 5, 16)
        mha.forward(X)
        A = mha._cache["A"]
        np.testing.assert_allclose(A.sum(axis=-1), np.ones((2, 4, 5)), atol=1e-6)

    def test_single_token_attention_is_one(self):
        np.random.seed(11)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        X = np.random.randn(2, 1, 8)
        mha.forward(X)
        A = mha._cache["A"]
        np.testing.assert_allclose(A, np.ones((2, 2, 1, 1)), atol=1e-12)

    def test_batch_independence(self):
        np.random.seed(12)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        x0 = np.random.randn(1, 3, 8)
        x1 = np.random.randn(1, 3, 8)

        X_batch = np.concatenate([x0, x1], axis=0)
        out_batch = mha.forward(X_batch)

        out_0 = mha.forward(x0)
        out_1 = mha.forward(x1)

        np.testing.assert_allclose(out_batch[0], out_0[0], atol=1e-12)
        np.testing.assert_allclose(out_batch[1], out_1[0], atol=1e-12)

    def test_no_bias(self):
        np.random.seed(13)
        mha = MultiHeadAttention(d_model=8, num_heads=2, use_bias=False)
        X = np.random.randn(1, 3, 8)
        out = mha.forward(X)
        self.assertEqual(out.shape, (1, 3, 8))
        self.assertIsNone(mha.b_Q)


class TestSingleHeadEquivalence(unittest.TestCase):
    """With h=1, multi-head attention should match single-head self-attention."""

    def test_single_head_matches_self_attention(self):
        np.random.seed(42)
        d_model = 8

        mha = MultiHeadAttention(d_model=d_model, num_heads=1, use_bias=True)
        sa = SelfAttention(d_model=d_model, d_k=d_model, d_v=d_model, d_out=d_model, use_bias=True)

        sa.W_Q = mha.W_Q.copy()
        sa.W_K = mha.W_K.copy()
        sa.W_V = mha.W_V.copy()
        sa.W_O = mha.W_O.copy()
        sa.b_Q = mha.b_Q.copy()
        sa.b_K = mha.b_K.copy()
        sa.b_V = mha.b_V.copy()
        sa.b_O = mha.b_O.copy()

        X = np.random.randn(2, 5, d_model)
        out_mha = mha.forward(X)
        out_sa = sa.forward(X)

        np.testing.assert_allclose(out_mha, out_sa, atol=1e-12)

    def test_single_head_matches_with_causal_mask(self):
        np.random.seed(43)
        d_model = 6

        mha = MultiHeadAttention(d_model=d_model, num_heads=1, use_bias=False)
        sa = SelfAttention(d_model=d_model, d_k=d_model, d_v=d_model, d_out=d_model, use_bias=False)

        sa.W_Q = mha.W_Q.copy()
        sa.W_K = mha.W_K.copy()
        sa.W_V = mha.W_V.copy()
        sa.W_O = mha.W_O.copy()

        X = np.random.randn(1, 4, d_model)
        mask_mha = create_causal_mask(4)
        mask_sa = mask_mha[0, 0]  # SelfAttention expects (L, L)

        out_mha = mha.forward(X, mask=mask_mha)
        out_sa = sa.forward(X, mask=mask_sa)

        np.testing.assert_allclose(out_mha, out_sa, atol=1e-12)


class TestFusedVsSeparateHeads(unittest.TestCase):
    """Fused projection + reshape should match separate per-head projections."""

    def test_equivalence(self):
        np.random.seed(44)
        d_model = 8
        num_heads = 2
        d_k = d_model // num_heads

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, use_bias=False)
        X = np.random.randn(1, 3, d_model)

        out_fused = mha.forward(X)

        Q_full = X @ mha.W_Q
        K_full = X @ mha.W_K
        V_full = X @ mha.W_V

        head_outputs = []
        for i in range(num_heads):
            W_Q_i = mha.W_Q[:, i * d_k:(i + 1) * d_k]
            W_K_i = mha.W_K[:, i * d_k:(i + 1) * d_k]
            W_V_i = mha.W_V[:, i * d_k:(i + 1) * d_k]

            Q_i = X @ W_Q_i
            K_i = X @ W_K_i
            V_i = X @ W_V_i

            scores_i = Q_i @ K_i.transpose(0, 2, 1) / np.sqrt(d_k)
            A_i = softmax(scores_i)
            head_i = A_i @ V_i
            head_outputs.append(head_i)

        concat_separate = np.concatenate(head_outputs, axis=-1)
        out_separate = concat_separate @ mha.W_O

        np.testing.assert_allclose(out_fused, out_separate, atol=1e-12)


class TestMasking(unittest.TestCase):

    def test_causal_mask_zeros_future(self):
        np.random.seed(20)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        X = np.random.randn(2, 5, 8)
        mask = create_causal_mask(5)
        mha.forward(X, mask=mask)
        A = mha._cache["A"]

        for b in range(2):
            for h in range(2):
                for i in range(5):
                    for j in range(i + 1, 5):
                        self.assertAlmostEqual(A[b, h, i, j], 0.0, places=10)

    def test_causal_mask_weights_sum_to_one(self):
        np.random.seed(21)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        X = np.random.randn(1, 4, 8)
        mask = create_causal_mask(4)
        mha.forward(X, mask=mask)
        A = mha._cache["A"]
        np.testing.assert_allclose(A.sum(axis=-1), 1.0, atol=1e-6)

    def test_mask_broadcasts_across_heads(self):
        np.random.seed(22)
        mha = MultiHeadAttention(d_model=12, num_heads=3)
        X = np.random.randn(2, 4, 12)
        mask = create_causal_mask(4)
        mha.forward(X, mask=mask)
        A = mha._cache["A"]

        for h in range(3):
            for i in range(4):
                for j in range(i + 1, 4):
                    self.assertAlmostEqual(A[0, h, i, j], 0.0, places=10)


class TestMultiHeadAttentionBackward(unittest.TestCase):

    def _numerical_gradient(self, mha, X, param_name, mask=None, h=1e-5):
        param = getattr(mha, param_name)
        grad_num = np.zeros_like(param)
        grad_output = self._grad_output

        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]

            param[idx] = old_val + h
            out_plus = mha.forward(X, mask=mask)
            loss_plus = np.sum(out_plus * grad_output)

            param[idx] = old_val - h
            out_minus = mha.forward(X, mask=mask)
            loss_minus = np.sum(out_minus * grad_output)

            grad_num[idx] = (loss_plus - loss_minus) / (2.0 * h)
            param[idx] = old_val
            it.iternext()

        return grad_num

    def _numerical_gradient_input(self, mha, X, mask=None, h=1e-5):
        grad_num = np.zeros_like(X)
        grad_output = self._grad_output

        it = np.nditer(X, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old_val = X[idx]

            X[idx] = old_val + h
            out_plus = mha.forward(X, mask=mask)
            loss_plus = np.sum(out_plus * grad_output)

            X[idx] = old_val - h
            out_minus = mha.forward(X, mask=mask)
            loss_minus = np.sum(out_minus * grad_output)

            grad_num[idx] = (loss_plus - loss_minus) / (2.0 * h)
            X[idx] = old_val
            it.iternext()

        return grad_num

    def _check_gradient(self, analytical, numerical, name, tol=1e-5, atol=1e-7):
        abs_diff = np.abs(analytical - numerical)
        denom = np.abs(analytical) + np.abs(numerical) + 1e-8
        rel_err = abs_diff / denom
        # Skip relative check where both values are near zero (absolute error dominates)
        significant = abs_diff > atol
        if significant.any():
            max_err = rel_err[significant].max()
            self.assertLess(max_err, tol, f"Gradient check failed for {name}: max relative error = {max_err:.2e}")
        # Always check absolute error as a fallback
        max_abs = abs_diff.max()
        self.assertLess(max_abs, atol * 100, f"Gradient check failed for {name}: max absolute error = {max_abs:.2e}")

    def test_gradient_check_no_mask(self):
        np.random.seed(100)
        mha = MultiHeadAttention(d_model=6, num_heads=2)
        X = np.random.randn(1, 3, 6) * 0.5

        out = mha.forward(X)
        self._grad_output = np.random.randn(*out.shape)
        mha.backward(self._grad_output)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            num = self._numerical_gradient(mha, X, param_name)
            ana = getattr(mha, f"grad_{param_name}")
            self._check_gradient(ana, num, param_name)

        for bias_name in ["b_Q", "b_K", "b_V", "b_O"]:
            num = self._numerical_gradient(mha, X, bias_name)
            ana = getattr(mha, f"grad_{bias_name}")
            self._check_gradient(ana, num, bias_name)

        num_X = self._numerical_gradient_input(mha, X)
        mha.forward(X)
        grad_X = mha.backward(self._grad_output)
        self._check_gradient(grad_X, num_X, "X")

    def test_gradient_check_with_causal_mask(self):
        np.random.seed(101)
        mha = MultiHeadAttention(d_model=6, num_heads=2)
        X = np.random.randn(1, 4, 6) * 0.5
        mask = create_causal_mask(4)

        out = mha.forward(X, mask=mask)
        self._grad_output = np.random.randn(*out.shape)
        mha.backward(self._grad_output)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            num = self._numerical_gradient(mha, X, param_name, mask=mask)
            ana = getattr(mha, f"grad_{param_name}")
            self._check_gradient(ana, num, param_name)

        num_X = self._numerical_gradient_input(mha, X, mask=mask)
        mha.forward(X, mask=mask)
        grad_X = mha.backward(self._grad_output)
        self._check_gradient(grad_X, num_X, "X")

    def test_gradient_check_no_bias(self):
        np.random.seed(102)
        mha = MultiHeadAttention(d_model=6, num_heads=2, use_bias=False)
        X = np.random.randn(1, 3, 6) * 0.5

        out = mha.forward(X)
        self._grad_output = np.random.randn(*out.shape)
        mha.backward(self._grad_output)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            num = self._numerical_gradient(mha, X, param_name)
            ana = getattr(mha, f"grad_{param_name}")
            self._check_gradient(ana, num, param_name)

        num_X = self._numerical_gradient_input(mha, X)
        mha.forward(X)
        grad_X = mha.backward(self._grad_output)
        self._check_gradient(grad_X, num_X, "X")

    def test_gradient_check_batch_size_2(self):
        np.random.seed(103)
        mha = MultiHeadAttention(d_model=4, num_heads=2)
        X = np.random.randn(2, 3, 4) * 0.5

        out = mha.forward(X)
        self._grad_output = np.random.randn(*out.shape)
        mha.backward(self._grad_output)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            num = self._numerical_gradient(mha, X, param_name)
            ana = getattr(mha, f"grad_{param_name}")
            self._check_gradient(ana, num, param_name)

        num_X = self._numerical_gradient_input(mha, X)
        mha.forward(X)
        grad_X = mha.backward(self._grad_output)
        self._check_gradient(grad_X, num_X, "X")

    def test_gradient_check_many_heads(self):
        np.random.seed(104)
        mha = MultiHeadAttention(d_model=8, num_heads=4, use_bias=False)
        X = np.random.randn(1, 3, 8) * 0.5

        out = mha.forward(X)
        self._grad_output = np.random.randn(*out.shape)
        mha.backward(self._grad_output)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            num = self._numerical_gradient(mha, X, param_name)
            ana = getattr(mha, f"grad_{param_name}")
            self._check_gradient(ana, num, param_name)

    def test_backward_raises_without_forward(self):
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        with self.assertRaises(RuntimeError):
            mha.backward(np.zeros((1, 3, 8)))

    def test_causal_mask_blocks_gradient_flow(self):
        np.random.seed(55)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        mask = create_causal_mask(3)
        X = np.random.randn(1, 3, 8)

        out = mha.forward(X, mask=mask)
        grad_out = np.zeros_like(out)
        grad_out[0, 0, :] = np.random.randn(8)

        mha.backward(grad_out)

        h = 1e-5
        for d in range(8):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[0, 2, d] += h
            X_minus[0, 2, d] -= h
            out_plus = mha.forward(X_plus, mask=mask)
            out_minus = mha.forward(X_minus, mask=mask)
            numerical_effect = np.abs(out_plus[0, 0, :] - out_minus[0, 0, :]).max() / (2 * h)
            self.assertLess(numerical_effect, 1e-8,
                            "Future position should not affect masked output position")


class TestNumericalStability(unittest.TestCase):

    def test_large_input_values(self):
        np.random.seed(30)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        X = np.random.uniform(-100, 100, size=(2, 5, 8))
        out = mha.forward(X)
        self.assertTrue(np.all(np.isfinite(out)))

        grad_out = np.random.randn(*out.shape)
        grad_X = mha.backward(grad_out)
        self.assertTrue(np.all(np.isfinite(grad_X)))

    def test_gradient_stability_near_saturation(self):
        np.random.seed(31)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        X = np.random.randn(1, 3, 8) * 50.0
        out = mha.forward(X)
        grad_out = np.random.randn(*out.shape)
        grad_X = mha.backward(grad_out)
        self.assertTrue(np.all(np.isfinite(grad_X)))
        for name in ["grad_W_Q", "grad_W_K", "grad_W_V", "grad_W_O"]:
            self.assertTrue(np.all(np.isfinite(getattr(mha, name))), f"{name} has non-finite values")

    def test_long_sequence(self):
        np.random.seed(32)
        mha = MultiHeadAttention(d_model=16, num_heads=4)
        X = np.random.randn(1, 512, 16)
        out = mha.forward(X)
        self.assertTrue(np.all(np.isfinite(out)))

        A = mha._cache["A"]
        np.testing.assert_allclose(A.sum(axis=-1), 1.0, atol=1e-6)

    def test_large_d_model_many_heads(self):
        np.random.seed(33)
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        X = np.random.randn(1, 4, 64)
        out = mha.forward(X)
        self.assertTrue(np.all(np.isfinite(out)))
        grad_out = np.random.randn(*out.shape)
        grad_X = mha.backward(grad_out)
        self.assertTrue(np.all(np.isfinite(grad_X)))


class TestCountFlops(unittest.TestCase):

    def test_known_values(self):
        B, L, d_model, h = 1, 4, 8, 2
        d_k = d_model // h
        flops = count_flops(B, L, d_model, h)

        proj_qkv = 3 * 2 * B * L * d_model * d_model
        proj_o = 2 * B * L * d_model * d_model
        qk = 2 * B * h * L * L * d_k
        av = 2 * B * h * L * L * d_k
        sm = 5 * B * h * L * L

        expected = proj_qkv + proj_o + qk + av + sm
        self.assertEqual(flops, expected)

    def test_multi_head_same_matmul_flops_as_single_head(self):
        """Projection and attention matmul FLOPs are identical regardless of head count.

        Softmax FLOPs scale with h (more heads = more L x L softmax operations),
        but the matmul FLOPs (which dominate) are the same.
        """
        B, L, d_model = 1, 8, 16

        flops_1h = count_flops(B, L, d_model, 1)
        flops_4h = count_flops(B, L, d_model, 4)

        sm_1h = 5 * B * 1 * L * L
        sm_4h = 5 * B * 4 * L * L

        matmul_1h = flops_1h - sm_1h
        matmul_4h = flops_4h - sm_4h

        self.assertEqual(matmul_1h, matmul_4h)

    def test_scales_with_seq_len_squared(self):
        base = count_flops(1, 128, 8, 2)
        doubled = count_flops(1, 256, 8, 2)
        ratio = doubled / base
        self.assertGreater(ratio, 3.5)


class TestCountMemoryBytes(unittest.TestCase):

    def test_attention_matrix_dominates(self):
        B, L, d_model, h = 1, 64, 8, 2
        mem = count_memory_bytes(B, L, d_model, h)
        attn_matrix_bytes = B * h * L * L * 4
        self.assertGreater(mem, attn_matrix_bytes)

    def test_float16_half_of_float32(self):
        mem_fp32 = count_memory_bytes(1, 32, 16, 4, dtype="float32")
        mem_fp16 = count_memory_bytes(1, 32, 16, 4, dtype="float16")
        self.assertEqual(mem_fp16, mem_fp32 // 2)

    def test_h_heads_more_attn_memory(self):
        """More heads -> same Q/K/V memory but attention matrices scale with h."""
        mem_2h = count_memory_bytes(1, 16, 8, 2)
        mem_4h = count_memory_bytes(1, 16, 8, 4)
        self.assertGreater(mem_4h, mem_2h)


class TestSequenceLen2ManualVerification(unittest.TestCase):

    def test_two_token_attention_pattern(self):
        np.random.seed(50)
        d_model = 4
        num_heads = 2
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, use_bias=False)

        X = np.random.randn(1, 2, d_model)
        mha.forward(X)
        A = mha._cache["A"]

        self.assertEqual(A.shape, (1, 2, 2, 2))
        np.testing.assert_allclose(A.sum(axis=-1), 1.0, atol=1e-12)

        for h in range(num_heads):
            self.assertTrue(np.all(A[0, h] >= 0))
            self.assertTrue(np.all(A[0, h] <= 1.0 + 1e-12))


if __name__ == "__main__":
    unittest.main()
