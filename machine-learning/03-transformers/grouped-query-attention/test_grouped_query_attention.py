"""Tests for grouped-query attention."""

import unittest
import numpy as np
from implementation import (
    GroupedQueryAttention,
    repeat_kv,
    reduce_kv_grad,
    create_causal_mask,
    softmax,
    count_parameters,
    kv_cache_size,
    kv_cache_size_model,
    count_flops,
)


class TestRepeatKV(unittest.TestCase):

    def test_no_repeat(self):
        x = np.random.randn(2, 4, 8, 16)
        out = repeat_kv(x, 1)
        np.testing.assert_array_equal(out, x)

    def test_repeat_doubles(self):
        x = np.random.randn(2, 2, 8, 16)
        out = repeat_kv(x, 4)
        self.assertEqual(out.shape, (2, 8, 8, 16))
        for i in range(4):
            np.testing.assert_array_equal(out[:, i, :, :], x[:, 0, :, :])
        for i in range(4, 8):
            np.testing.assert_array_equal(out[:, i, :, :], x[:, 1, :, :])

    def test_roundtrip(self):
        """repeat_kv followed by reshape-and-sum recovers the original (scaled by g)."""
        B, h_kv, L, d = 2, 3, 5, 8
        g = 4
        x = np.random.randn(B, h_kv, L, d)
        expanded = repeat_kv(x, g)
        recovered = reduce_kv_grad(expanded, h_kv, g)
        np.testing.assert_allclose(recovered, x * g)

    def test_mqa_repeat(self):
        x = np.random.randn(1, 1, 4, 8)
        out = repeat_kv(x, 8)
        self.assertEqual(out.shape, (1, 8, 4, 8))
        for i in range(8):
            np.testing.assert_array_equal(out[:, i, :, :], x[:, 0, :, :])


class TestCausalMask(unittest.TestCase):

    def test_shape(self):
        mask = create_causal_mask(5)
        self.assertEqual(mask.shape, (1, 1, 5, 5))

    def test_lower_triangle_zero(self):
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i + 1):
                self.assertEqual(mask[0, 0, i, j], 0.0)

    def test_upper_triangle_neginf(self):
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertLess(mask[0, 0, i, j], -1e8)

    def test_single_token(self):
        mask = create_causal_mask(1)
        self.assertEqual(mask[0, 0, 0, 0], 0.0)


class TestGroupedQueryAttentionInit(unittest.TestCase):

    def test_valid_configs(self):
        configs = [(64, 8, 8), (64, 8, 4), (64, 8, 2), (64, 8, 1)]
        for d_model, h, h_kv in configs:
            gqa = GroupedQueryAttention(d_model, h, h_kv)
            self.assertEqual(gqa.d_model, d_model)
            self.assertEqual(gqa.num_heads, h)
            self.assertEqual(gqa.num_kv_heads, h_kv)

    def test_d_model_not_divisible_by_heads(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(100, 7, 1)

    def test_heads_not_divisible_by_kv_heads(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(64, 7, 3)

    def test_weight_shapes(self):
        gqa = GroupedQueryAttention(64, 8, 2)
        d_k = 64 // 8
        self.assertEqual(gqa.W_Q.shape, (64, 64))
        self.assertEqual(gqa.W_K.shape, (64, 2 * d_k))
        self.assertEqual(gqa.W_V.shape, (64, 2 * d_k))
        self.assertEqual(gqa.W_O.shape, (64, 64))

    def test_bias_shapes(self):
        gqa = GroupedQueryAttention(64, 8, 2)
        d_k = 64 // 8
        self.assertEqual(gqa.b_Q.shape, (64,))
        self.assertEqual(gqa.b_K.shape, (2 * d_k,))
        self.assertEqual(gqa.b_V.shape, (2 * d_k,))
        self.assertEqual(gqa.b_O.shape, (64,))


class TestForwardShapes(unittest.TestCase):

    def test_output_shape_gqa(self):
        gqa = GroupedQueryAttention(64, 8, 2)
        X = np.random.randn(2, 16, 64)
        out = gqa.forward(X)
        self.assertEqual(out.shape, (2, 16, 64))

    def test_output_shape_mha(self):
        gqa = GroupedQueryAttention(64, 8, 8)
        X = np.random.randn(2, 16, 64)
        out = gqa.forward(X)
        self.assertEqual(out.shape, (2, 16, 64))

    def test_output_shape_mqa(self):
        gqa = GroupedQueryAttention(64, 8, 1)
        X = np.random.randn(2, 16, 64)
        out = gqa.forward(X)
        self.assertEqual(out.shape, (2, 16, 64))

    def test_variable_batch_sizes(self):
        gqa = GroupedQueryAttention(64, 8, 2)
        for B in [1, 4, 16]:
            X = np.random.randn(B, 8, 64)
            out = gqa.forward(X)
            self.assertEqual(out.shape, (B, 8, 64))

    def test_variable_seq_lengths(self):
        gqa = GroupedQueryAttention(64, 8, 2)
        for L in [1, 16, 128]:
            X = np.random.randn(2, L, 64)
            out = gqa.forward(X)
            self.assertEqual(out.shape, (2, L, 64))

    def test_internal_shapes_cached(self):
        gqa = GroupedQueryAttention(64, 8, 2)
        B, L = 2, 10
        d_k = 64 // 8
        X = np.random.randn(B, L, 64)
        gqa.forward(X)

        self.assertEqual(gqa._cache["Q"].shape, (B, 8, L, d_k))
        self.assertEqual(gqa._cache["K_exp"].shape, (B, 8, L, d_k))
        self.assertEqual(gqa._cache["V_exp"].shape, (B, 8, L, d_k))
        self.assertEqual(gqa._cache["A"].shape, (B, 8, L, L))
        self.assertEqual(gqa._cache["concat"].shape, (B, L, 64))


class TestForwardCorrectness(unittest.TestCase):

    def test_attention_weights_sum_to_one(self):
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 2)
        X = np.random.randn(2, 16, 64)
        gqa.forward(X)
        A = gqa._cache["A"]
        sums = A.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_attention_weights_sum_to_one_with_mask(self):
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 2)
        X = np.random.randn(2, 16, 64)
        mask = create_causal_mask(16)
        gqa.forward(X, mask=mask)
        A = gqa._cache["A"]
        sums = A.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_batch_independence(self):
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 2)
        X0 = np.random.randn(1, 8, 64)
        X1 = np.random.randn(1, 8, 64)

        out0 = gqa.forward(X0).copy()
        out1 = gqa.forward(X1).copy()

        X_both = np.concatenate([X0, X1], axis=0)
        out_both = gqa.forward(X_both)

        np.testing.assert_allclose(out_both[0:1], out0, atol=1e-12)
        np.testing.assert_allclose(out_both[1:2], out1, atol=1e-12)

    def test_forward_determinism(self):
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 2)
        X = np.random.randn(2, 8, 64)
        out1 = gqa.forward(X).copy()
        out2 = gqa.forward(X).copy()
        np.testing.assert_array_equal(out1, out2)

    def test_mha_equivalence(self):
        """GQA with h_kv == h should match standard MHA given identical weights."""
        np.random.seed(42)
        d_model, h = 32, 4
        gqa = GroupedQueryAttention(d_model, h, h)

        from importlib import import_module
        import sys, os
        mha_path = os.path.join(os.path.dirname(__file__), "..", "multi-head-attention")
        sys.path.insert(0, mha_path)
        try:
            import importlib
            mha_mod = importlib.import_module("implementation")
            importlib.reload(mha_mod)
            MultiHeadAttention = mha_mod.MultiHeadAttention
        finally:
            sys.path.pop(0)

        mha = MultiHeadAttention(d_model, h, use_bias=True)
        mha.W_Q = gqa.W_Q.copy()
        mha.W_K = gqa.W_K.copy()
        mha.W_V = gqa.W_V.copy()
        mha.W_O = gqa.W_O.copy()
        mha.b_Q = gqa.b_Q.copy()
        mha.b_K = gqa.b_K.copy()
        mha.b_V = gqa.b_V.copy()
        mha.b_O = gqa.b_O.copy()

        X = np.random.randn(2, 8, d_model)
        out_gqa = gqa.forward(X)
        out_mha = mha.forward(X)
        np.testing.assert_allclose(out_gqa, out_mha, atol=1e-10)

    def test_mqa_works(self):
        """GQA with h_kv == 1 should work correctly."""
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 1)
        X = np.random.randn(2, 8, 64)
        out = gqa.forward(X)
        self.assertEqual(out.shape, (2, 8, 64))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_known_small_config(self):
        """Test with d_model=8, h=4, h_kv=2, L=3 using fixed weights."""
        np.random.seed(0)
        d_model, h, h_kv = 8, 4, 2
        gqa = GroupedQueryAttention(d_model, h, h_kv)

        gqa.W_Q = np.eye(8) * 0.1
        gqa.W_K = np.ones((8, 4)) * 0.05
        gqa.W_V = np.ones((8, 4)) * 0.05
        gqa.W_O = np.eye(8) * 0.1
        gqa.b_Q = np.zeros(8)
        gqa.b_K = np.zeros(4)
        gqa.b_V = np.zeros(4)
        gqa.b_O = np.zeros(8)

        X = np.ones((1, 3, 8))
        out = gqa.forward(X)
        self.assertEqual(out.shape, (1, 3, 8))
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out[0, 0], out[0, 1], atol=1e-10)


class TestCausalMasking(unittest.TestCase):

    def test_causal_mask_zero_future_attention(self):
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 2)
        X = np.random.randn(2, 16, 64)
        mask = create_causal_mask(16)
        gqa.forward(X, mask=mask)
        A = gqa._cache["A"]

        for i in range(16):
            for j in range(i + 1, 16):
                self.assertAlmostEqual(A[0, 0, i, j], 0.0, places=10)

    def test_causal_mask_broadcasts_over_heads_and_batch(self):
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 2)
        X = np.random.randn(3, 10, 64)
        mask = create_causal_mask(10)
        gqa.forward(X, mask=mask)
        A = gqa._cache["A"]

        for b in range(3):
            for head in range(8):
                for i in range(10):
                    for j in range(i + 1, 10):
                        self.assertAlmostEqual(A[b, head, i, j], 0.0, places=10)

    def test_single_token_causal(self):
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 2)
        X = np.random.randn(1, 1, 64)
        mask = create_causal_mask(1)
        gqa.forward(X, mask=mask)
        A = gqa._cache["A"]
        np.testing.assert_allclose(A, 1.0, atol=1e-10)


class TestNumericalStability(unittest.TestCase):

    def test_large_inputs_no_nan(self):
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 2)
        X = np.random.uniform(-100, 100, (2, 8, 64))
        out = gqa.forward(X)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_softmax_overflow_prevention(self):
        """Logits > 700 would overflow naive exp; verify stability."""
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 2)
        gqa.W_Q *= 100
        gqa.W_K *= 100
        X = np.random.randn(1, 4, 64) * 10
        out = gqa.forward(X)
        self.assertTrue(np.all(np.isfinite(out)))
        A = gqa._cache["A"]
        self.assertTrue(np.all(np.isfinite(A)))

    def test_gradient_stability(self):
        np.random.seed(42)
        gqa = GroupedQueryAttention(64, 8, 2)
        gqa.W_Q *= 50
        gqa.W_K *= 50
        X = np.random.randn(1, 4, 64) * 5
        out = gqa.forward(X)
        grad = np.ones_like(out)
        grad_X = gqa.backward(grad)
        self.assertTrue(np.all(np.isfinite(grad_X)))


class TestBackwardShapes(unittest.TestCase):

    def test_grad_shapes(self):
        np.random.seed(42)
        d_model, h, h_kv = 64, 8, 2
        d_k = d_model // h
        gqa = GroupedQueryAttention(d_model, h, h_kv)
        X = np.random.randn(2, 8, d_model)
        out = gqa.forward(X)
        grad = np.random.randn(*out.shape)
        grad_X = gqa.backward(grad)

        self.assertEqual(grad_X.shape, (2, 8, d_model))
        self.assertEqual(gqa.grad_W_Q.shape, (d_model, d_model))
        self.assertEqual(gqa.grad_W_K.shape, (d_model, h_kv * d_k))
        self.assertEqual(gqa.grad_W_V.shape, (d_model, h_kv * d_k))
        self.assertEqual(gqa.grad_W_O.shape, (d_model, d_model))
        self.assertEqual(gqa.grad_b_Q.shape, (d_model,))
        self.assertEqual(gqa.grad_b_K.shape, (h_kv * d_k,))
        self.assertEqual(gqa.grad_b_V.shape, (h_kv * d_k,))
        self.assertEqual(gqa.grad_b_O.shape, (d_model,))


class TestGradientCheck(unittest.TestCase):
    """Numerical gradient verification via central finite differences."""

    def _numerical_grad(self, gqa, X, param_name, mask=None, eps=1e-5):
        """Compute numerical gradient for a weight matrix."""
        param = getattr(gqa, param_name)
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]

            param[idx] = old_val + eps
            out_plus = gqa.forward(X, mask=mask)
            loss_plus = np.sum(out_plus)

            param[idx] = old_val - eps
            out_minus = gqa.forward(X, mask=mask)
            loss_minus = np.sum(out_minus)

            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            param[idx] = old_val
            it.iternext()
        return grad

    def _numerical_grad_input(self, gqa, X, mask=None, eps=1e-5):
        """Compute numerical gradient for the input X."""
        grad = np.zeros_like(X)
        it = np.nditer(X, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old_val = X[idx]

            X[idx] = old_val + eps
            loss_plus = np.sum(gqa.forward(X, mask=mask))

            X[idx] = old_val - eps
            loss_minus = np.sum(gqa.forward(X, mask=mask))

            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            X[idx] = old_val
            it.iternext()
        return grad

    def _check_grad(self, analytical, numerical, name, tol=1e-5):
        diff = np.abs(analytical - numerical)
        denom = np.abs(analytical) + np.abs(numerical) + 1e-8
        rel_err = diff / denom
        max_err = np.max(rel_err)
        self.assertLess(
            max_err, tol,
            f"Gradient check failed for {name}: max relative error {max_err:.2e}"
        )

    def test_grad_check_no_mask(self):
        np.random.seed(42)
        d_model, h, h_kv = 16, 4, 2
        gqa = GroupedQueryAttention(d_model, h, h_kv)
        X = np.random.randn(1, 3, d_model) * 0.5

        out = gqa.forward(X)
        grad_out = np.ones_like(out)
        grad_X_analytical = gqa.backward(grad_out)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            gqa.forward(X)
            gqa.backward(grad_out)
            analytical = getattr(gqa, f"grad_{param_name}")
            numerical = self._numerical_grad(gqa, X, param_name)
            self._check_grad(analytical, numerical, param_name)

        numerical_X = self._numerical_grad_input(gqa, X)
        gqa.forward(X)
        grad_X_analytical = gqa.backward(grad_out)
        self._check_grad(grad_X_analytical, numerical_X, "X")

    def test_grad_check_with_causal_mask(self):
        np.random.seed(123)
        d_model, h, h_kv = 16, 4, 2
        L = 4
        gqa = GroupedQueryAttention(d_model, h, h_kv)
        X = np.random.randn(1, L, d_model) * 0.5
        mask = create_causal_mask(L)

        out = gqa.forward(X, mask=mask)
        grad_out = np.ones_like(out)
        gqa.backward(grad_out)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            gqa.forward(X, mask=mask)
            gqa.backward(grad_out)
            analytical = getattr(gqa, f"grad_{param_name}")
            numerical = self._numerical_grad(gqa, X, param_name, mask=mask)
            self._check_grad(analytical, numerical, f"{param_name} (causal)")

        numerical_X = self._numerical_grad_input(gqa, X, mask=mask)
        gqa.forward(X, mask=mask)
        grad_X_analytical = gqa.backward(grad_out)
        self._check_grad(grad_X_analytical, numerical_X, "X (causal)")

    def test_grad_check_mha_mode(self):
        np.random.seed(7)
        d_model, h = 16, 4
        gqa = GroupedQueryAttention(d_model, h, h)
        X = np.random.randn(1, 3, d_model) * 0.5

        out = gqa.forward(X)
        grad_out = np.ones_like(out)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            gqa.forward(X)
            gqa.backward(grad_out)
            analytical = getattr(gqa, f"grad_{param_name}")
            numerical = self._numerical_grad(gqa, X, param_name)
            self._check_grad(analytical, numerical, f"{param_name} (MHA)")

    def test_grad_check_mqa_mode(self):
        np.random.seed(11)
        d_model, h = 16, 4
        gqa = GroupedQueryAttention(d_model, h, 1)
        X = np.random.randn(1, 3, d_model) * 0.5

        out = gqa.forward(X)
        grad_out = np.ones_like(out)

        for param_name in ["W_Q", "W_K", "W_V", "W_O"]:
            gqa.forward(X)
            gqa.backward(grad_out)
            analytical = getattr(gqa, f"grad_{param_name}")
            numerical = self._numerical_grad(gqa, X, param_name)
            self._check_grad(analytical, numerical, f"{param_name} (MQA)")

    def test_bias_grad_check(self):
        np.random.seed(55)
        d_model, h, h_kv = 16, 4, 2
        gqa = GroupedQueryAttention(d_model, h, h_kv)
        gqa.W_Q *= 0.1
        gqa.W_K *= 0.1
        gqa.W_V *= 0.1
        gqa.W_O *= 0.1
        X = np.random.randn(1, 3, d_model) * 0.1

        out = gqa.forward(X)
        grad_out = np.ones_like(out)

        for param_name in ["b_Q", "b_K", "b_V", "b_O"]:
            gqa.forward(X)
            gqa.backward(grad_out)
            analytical = getattr(gqa, f"grad_{param_name}")
            numerical = self._numerical_grad(gqa, X, param_name, eps=1e-6)
            self._check_grad(analytical, numerical, param_name, tol=1e-4)


class TestSharedKVGradientAccumulation(unittest.TestCase):

    def test_kv_grad_is_group_sum(self):
        """Verify grad_K accumulates from all g query heads in each group."""
        np.random.seed(42)
        d_model, h, h_kv = 32, 8, 2
        gqa = GroupedQueryAttention(d_model, h, h_kv)
        B, L = 1, 4
        X = np.random.randn(B, L, d_model) * 0.5

        out = gqa.forward(X)
        grad_out = np.ones_like(out)
        gqa.backward(grad_out)

        Q = gqa._cache["Q"]
        K_exp = gqa._cache["K_exp"]
        V_exp = gqa._cache["V_exp"]
        A = gqa._cache["A"]
        concat = gqa._cache["concat"]

        scale = np.sqrt(gqa.d_k)
        grad_concat = grad_out @ gqa.W_O.T
        grad_attn_output = grad_concat.reshape(B, L, h, gqa.d_v).transpose(0, 2, 1, 3)
        grad_V_exp = A.transpose(0, 1, 3, 2) @ grad_attn_output
        grad_A = grad_attn_output @ V_exp.transpose(0, 1, 3, 2)

        dot = np.sum(grad_A * A, axis=-1, keepdims=True)
        grad_scores = A * (grad_A - dot)
        grad_raw = grad_scores / scale
        grad_K_exp = grad_raw.transpose(0, 1, 3, 2) @ Q

        g = gqa.group_size
        d_k = gqa.d_k
        grad_K_manual = grad_K_exp.reshape(B, h_kv, g, L, d_k).sum(axis=2)
        grad_K_flat_manual = grad_K_manual.transpose(0, 2, 1, 3).reshape(B, L, h_kv * d_k)
        grad_W_K_manual = np.einsum("blm,bld->md", X, grad_K_flat_manual)

        np.testing.assert_allclose(gqa.grad_W_K, grad_W_K_manual, atol=1e-10)


class TestCountParameters(unittest.TestCase):

    def test_mha_parameters(self):
        d_model, h = 64, 8
        params = count_parameters(d_model, h, h)
        self.assertEqual(params["W_Q"], d_model * d_model)
        self.assertEqual(params["W_K"], d_model * d_model)
        self.assertEqual(params["W_V"], d_model * d_model)
        self.assertEqual(params["W_O"], d_model * d_model)

    def test_gqa_smaller_kv(self):
        d_model, h, h_kv = 64, 8, 2
        d_k = d_model // h
        params = count_parameters(d_model, h, h_kv)
        self.assertEqual(params["W_K"], d_model * (h_kv * d_k))
        self.assertEqual(params["W_V"], d_model * (h_kv * d_k))
        self.assertLess(params["W_K"], d_model * d_model)

    def test_mqa_minimal_kv(self):
        d_model, h = 64, 8
        d_k = d_model // h
        params = count_parameters(d_model, h, 1)
        self.assertEqual(params["W_K"], d_model * d_k)
        self.assertEqual(params["W_V"], d_model * d_k)


class TestKVCacheSize(unittest.TestCase):

    def test_reduction_ratio(self):
        """h_kv=8 should be exactly 8x smaller than h_kv=64."""
        B, L, d_k = 1, 4096, 128
        cache_mha = kv_cache_size(B, L, 64, d_k)
        cache_gqa = kv_cache_size(B, L, 8, d_k)
        self.assertEqual(cache_mha, 8 * cache_gqa)

    def test_llama2_70b(self):
        """Llama 2 70B: 80 layers, 8 KV heads, d_k=128, L=4096, FP16."""
        cache = kv_cache_size_model(
            batch_size=1, seq_len=4096, num_layers=80,
            num_kv_heads=8, d_k=128, dtype="float16"
        )
        expected_bytes = 80 * 2 * 8 * 4096 * 128 * 2
        self.assertEqual(cache, expected_bytes)
        cache_mha = kv_cache_size_model(
            batch_size=1, seq_len=4096, num_layers=80,
            num_kv_heads=64, d_k=128, dtype="float16"
        )
        self.assertEqual(cache_mha, 8 * cache)

    def test_mistral_7b(self):
        """Mistral 7B: 32 layers, 8 KV heads, d_k=128, L=8192, FP16."""
        cache = kv_cache_size_model(
            batch_size=1, seq_len=8192, num_layers=32,
            num_kv_heads=8, d_k=128, dtype="float16"
        )
        expected = 32 * 2 * 8 * 8192 * 128 * 2
        self.assertEqual(cache, expected)


class TestFLOPs(unittest.TestCase):

    def test_attention_core_same_for_mha_and_gqa(self):
        """Attention core FLOPs should be identical regardless of num_kv_heads."""
        B, L, d_model, h = 1, 64, 256, 8
        flops_mha = count_flops(B, L, d_model, h, h)
        flops_gqa = count_flops(B, L, d_model, h, 2)

        self.assertEqual(flops_mha["attn_qk"], flops_gqa["attn_qk"])
        self.assertEqual(flops_mha["attn_av"], flops_gqa["attn_av"])
        self.assertEqual(flops_mha["attn_softmax"], flops_gqa["attn_softmax"])
        self.assertEqual(flops_mha["attn_total"], flops_gqa["attn_total"])

    def test_projection_flops_differ(self):
        """GQA should have fewer projection FLOPs than MHA."""
        B, L, d_model, h = 1, 64, 256, 8
        flops_mha = count_flops(B, L, d_model, h, h)
        flops_gqa = count_flops(B, L, d_model, h, 2)

        self.assertEqual(flops_mha["proj_q"], flops_gqa["proj_q"])
        self.assertEqual(flops_mha["proj_o"], flops_gqa["proj_o"])
        self.assertGreater(flops_mha["proj_k"], flops_gqa["proj_k"])
        self.assertGreater(flops_mha["proj_v"], flops_gqa["proj_v"])

    def test_flops_breakdown_sums(self):
        B, L, d_model, h, h_kv = 2, 32, 128, 8, 2
        flops = count_flops(B, L, d_model, h, h_kv)
        proj = flops["proj_q"] + flops["proj_k"] + flops["proj_v"] + flops["proj_o"]
        attn = flops["attn_qk"] + flops["attn_av"] + flops["attn_softmax"]
        self.assertEqual(flops["proj_total"], proj)
        self.assertEqual(flops["attn_total"], attn)
        self.assertEqual(flops["total"], proj + attn)


class TestBackwardBeforeForward(unittest.TestCase):

    def test_raises_runtime_error(self):
        gqa = GroupedQueryAttention(64, 8, 2)
        with self.assertRaises(RuntimeError):
            gqa.backward(np.random.randn(2, 8, 64))


if __name__ == "__main__":
    unittest.main()
