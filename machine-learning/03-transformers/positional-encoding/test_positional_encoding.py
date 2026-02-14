"""Tests for positional encoding."""

import unittest
import numpy as np

from implementation import (
    sinusoidal_positional_encoding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    relative_position_matrix,
    dot_product_distance,
    encoding_statistics,
)


class TestShapeVerification(unittest.TestCase):

    def test_sinusoidal_output_shape(self):
        for L, d in [(10, 64), (1, 4), (128, 512), (50, 256)]:
            pe = sinusoidal_positional_encoding(L, d)
            self.assertEqual(pe.shape, (L, d))

    def test_batched_forward_shape_sinusoidal(self):
        enc = SinusoidalPositionalEncoding(128, 64)
        X = np.random.randn(2, 32, 64)
        out = enc.forward(X)
        self.assertEqual(out.shape, (2, 32, 64))

    def test_learned_forward_shape(self):
        enc = LearnedPositionalEncoding(128, 64)
        X = np.random.randn(2, 32, 64)
        out = enc.forward(X)
        self.assertEqual(out.shape, (2, 32, 64))

    def test_variable_batch_sizes(self):
        enc = SinusoidalPositionalEncoding(128, 64)
        for B in [1, 4, 16]:
            X = np.random.randn(B, 32, 64)
            out = enc.forward(X)
            self.assertEqual(out.shape, (B, 32, 64))

    def test_variable_sequence_lengths(self):
        enc = SinusoidalPositionalEncoding(512, 64)
        for L in [1, 16, 128, 512]:
            X = np.random.randn(2, L, 64)
            out = enc.forward(X)
            self.assertEqual(out.shape, (2, L, 64))

    def test_odd_d_model_raises_error(self):
        with self.assertRaises(ValueError):
            sinusoidal_positional_encoding(10, 7)
        with self.assertRaises(ValueError):
            SinusoidalPositionalEncoding(10, 7)


class TestCorrectnessKnownValues(unittest.TestCase):

    def test_position_zero(self):
        pe = sinusoidal_positional_encoding(1, 64)
        expected_even = np.zeros(32)
        expected_odd = np.ones(32)
        np.testing.assert_allclose(pe[0, 0::2], expected_even, atol=1e-15)
        np.testing.assert_allclose(pe[0, 1::2], expected_odd, atol=1e-15)

    def test_small_example_d4_l3(self):
        pe = sinusoidal_positional_encoding(3, 4)

        omega_0 = 1.0
        omega_1 = np.exp(-2.0 * 1.0 / 4.0 * np.log(10000.0))

        expected = np.array([
            [np.sin(0 * omega_0), np.cos(0 * omega_0), np.sin(0 * omega_1), np.cos(0 * omega_1)],
            [np.sin(1 * omega_0), np.cos(1 * omega_0), np.sin(1 * omega_1), np.cos(1 * omega_1)],
            [np.sin(2 * omega_0), np.cos(2 * omega_0), np.sin(2 * omega_1), np.cos(2 * omega_1)],
        ])

        np.testing.assert_allclose(pe, expected, atol=1e-12)

        np.testing.assert_allclose(pe[0], [0, 1, 0, 1], atol=1e-15)
        np.testing.assert_allclose(pe[1, 0], np.sin(1.0), atol=1e-10)
        np.testing.assert_allclose(pe[1, 1], np.cos(1.0), atol=1e-10)
        self.assertAlmostEqual(omega_1, 0.01, places=5)


class TestUniquenessAndBoundedness(unittest.TestCase):

    def test_unique_position_vectors(self):
        pe = sinusoidal_positional_encoding(1000, 64)
        for i in range(pe.shape[0]):
            for j in range(i + 1, pe.shape[0]):
                dist = np.linalg.norm(pe[i] - pe[j])
                self.assertGreater(dist, 0)

    def test_values_bounded(self):
        pe = sinusoidal_positional_encoding(500, 128)
        self.assertTrue(np.all(np.abs(pe) <= 1.0 + 1e-15))

    def test_deterministic(self):
        pe1 = sinusoidal_positional_encoding(100, 64)
        pe2 = sinusoidal_positional_encoding(100, 64)
        np.testing.assert_array_equal(pe1, pe2)

    def test_norms_consistent(self):
        d_model = 64
        pe = sinusoidal_positional_encoding(200, d_model)
        expected_norm = np.sqrt(d_model / 2.0)
        norms = np.linalg.norm(pe, axis=1)
        np.testing.assert_allclose(norms, expected_norm, atol=1e-12)


class TestRelativePositionProperty(unittest.TestCase):

    def test_linear_transformation_offset_1(self):
        pe = sinusoidal_positional_encoding(100, 64)
        _, max_error = relative_position_matrix(pe, 1)
        self.assertLess(max_error, 1e-10)

    def test_multiple_offsets(self):
        pe = sinusoidal_positional_encoding(200, 64)
        for k in [1, 5, 10, 50]:
            _, max_error = relative_position_matrix(pe, k)
            self.assertLess(max_error, 1e-10, f"Failed for offset k={k}")

    def test_offset_only_dependence(self):
        pe = sinusoidal_positional_encoding(200, 64)
        d_model = 64
        k = 7

        M_from_0, _ = relative_position_matrix(pe, k)

        for pos in [10, 50, 100]:
            if pos + k < 200:
                reconstructed = M_from_0 @ pe[pos]
                np.testing.assert_allclose(
                    reconstructed, pe[pos + k], atol=1e-10,
                    err_msg=f"M_k differs when computed from pos={pos}"
                )


class TestDotProductDistanceStructure(unittest.TestCase):

    def test_dot_product_symmetry(self):
        pe = sinusoidal_positional_encoding(50, 64)
        D = dot_product_distance(pe)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_self_dot_product(self):
        d_model = 64
        pe = sinusoidal_positional_encoding(50, d_model)
        D = dot_product_distance(pe)
        expected_diag = d_model / 2.0
        np.testing.assert_allclose(np.diag(D), expected_diag, atol=1e-10)

    def test_distance_dependent(self):
        pe = sinusoidal_positional_encoding(100, 64)
        D = dot_product_distance(pe)
        for k in [1, 3, 5, 10]:
            dp_from_0 = D[0, k]
            dp_from_10 = D[10, 10 + k]
            np.testing.assert_allclose(
                dp_from_0, dp_from_10, atol=1e-10,
                err_msg=f"Dot product not distance-dependent for k={k}"
            )


class TestFrequencyStructure(unittest.TestCase):

    def test_low_dimensions_change_fast(self):
        pe = sinusoidal_positional_encoding(1000, 64)
        var_dim0 = np.var(pe[:, 0])
        self.assertGreater(var_dim0, 0.3)

    def test_high_dimensions_change_slow(self):
        pe = sinusoidal_positional_encoding(1000, 64)
        var_last_sin = np.var(pe[:, -2])
        self.assertLess(var_last_sin, 0.01)

    def test_dimension_0_wavelength(self):
        L = 100
        pe = sinusoidal_positional_encoding(L, 64)
        col = pe[:, 0]
        zero_crossings = np.where(np.diff(np.sign(col)))[0]
        if len(zero_crossings) >= 2:
            half_period = np.mean(np.diff(zero_crossings))
            full_period = 2 * half_period
            np.testing.assert_allclose(full_period, 2 * np.pi, atol=0.5)

    def test_last_dimension_wavelength(self):
        d_model = 64
        omega_last = np.exp(-2.0 * (d_model // 2 - 1) / d_model * np.log(10000.0))
        wavelength = 2 * np.pi / omega_last
        expected_wavelength = 2 * np.pi * 10000 ** (2.0 * (d_model // 2 - 1) / d_model)
        np.testing.assert_allclose(wavelength, expected_wavelength, rtol=1e-10)
        self.assertGreater(wavelength, 1000)


class TestLearnedEncoding(unittest.TestCase):

    def test_initialization_statistics(self):
        np.random.seed(42)
        enc = LearnedPositionalEncoding(1000, 256)
        self.assertAlmostEqual(enc.embedding.mean(), 0.0, places=2)
        np.testing.assert_allclose(enc.embedding.std(), 0.02, atol=0.005)

    def test_sequence_length_validation(self):
        enc = LearnedPositionalEncoding(128, 64)
        X = np.random.randn(2, 256, 64)
        with self.assertRaises(ValueError):
            enc.forward(X)

    def test_backward_correctness_numerical_gradient(self):
        np.random.seed(0)
        max_seq_len, d_model = 8, 4
        B, L = 2, 5
        enc = LearnedPositionalEncoding(max_seq_len, d_model)
        X = np.random.randn(B, L, d_model)

        def loss_fn_embed(embedding_flat):
            enc.embedding = embedding_flat.reshape(max_seq_len, d_model)
            out = enc.forward(X)
            return np.sum(out ** 2)

        enc.forward(X)
        grad_output = 2.0 * enc.forward(X)
        enc.backward(grad_output)
        analytic_grad = enc.grad_embedding.flatten()

        eps = 1e-5
        embedding_flat = enc.embedding.flatten().copy()
        numerical_grad = np.zeros_like(embedding_flat)

        for idx in range(len(embedding_flat)):
            e_plus = embedding_flat.copy()
            e_plus[idx] += eps
            loss_plus = loss_fn_embed(e_plus)

            e_minus = embedding_flat.copy()
            e_minus[idx] -= eps
            loss_minus = loss_fn_embed(e_minus)

            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        enc.embedding = embedding_flat.reshape(max_seq_len, d_model)

        rel_error = np.abs(analytic_grad - numerical_grad) / (
            np.maximum(np.abs(analytic_grad) + np.abs(numerical_grad), 1e-8)
        )
        self.assertLess(rel_error.max(), 1e-5)

    def test_input_gradient_passthrough(self):
        np.random.seed(0)
        enc = LearnedPositionalEncoding(32, 16)
        X = np.random.randn(3, 10, 16)
        enc.forward(X)
        grad_output = np.random.randn(3, 10, 16)
        grad_X = enc.backward(grad_output)
        np.testing.assert_array_equal(grad_X, grad_output)

    def test_gradient_accumulation_across_batch(self):
        np.random.seed(0)
        enc = LearnedPositionalEncoding(32, 16)
        B, L, d = 4, 10, 16

        grad_single = np.random.randn(1, L, d)
        grad_batch = np.tile(grad_single, (B, 1, 1))

        X_single = np.random.randn(1, L, d)
        enc.forward(X_single)
        enc.backward(grad_single)
        grad_emb_single = enc.grad_embedding.copy()

        X_batch = np.random.randn(B, L, d)
        enc.forward(X_batch)
        enc.backward(grad_batch)
        grad_emb_batch = enc.grad_embedding.copy()

        np.testing.assert_allclose(
            grad_emb_batch[:L, :], B * grad_emb_single[:L, :], atol=1e-12
        )

    def test_zero_gradient_unused_positions(self):
        enc = LearnedPositionalEncoding(32, 16)
        L = 10
        X = np.random.randn(2, L, 16)
        enc.forward(X)
        grad_output = np.random.randn(2, L, 16)
        enc.backward(grad_output)
        np.testing.assert_array_equal(
            enc.grad_embedding[L:, :], np.zeros((32 - L, 16))
        )


class TestNumericalStability(unittest.TestCase):

    def test_large_sequence_length(self):
        pe = sinusoidal_positional_encoding(10000, 64)
        self.assertFalse(np.any(np.isnan(pe)))
        self.assertFalse(np.any(np.isinf(pe)))

    def test_large_d_model(self):
        pe = sinusoidal_positional_encoding(100, 4096)
        self.assertFalse(np.any(np.isnan(pe)))
        self.assertFalse(np.any(np.isinf(pe)))

    def test_log_space_computation_matches_direct(self):
        d_model = 512
        i = np.arange(d_model // 2, dtype=np.float64)

        omega_log = np.exp(-2.0 * i / d_model * np.log(10000.0))
        omega_direct = 1.0 / (10000.0 ** (2.0 * i / d_model))

        np.testing.assert_allclose(omega_log, omega_direct, rtol=1e-12)


class TestIntegration(unittest.TestCase):

    def test_permutation_sensitivity_with_pe(self):
        np.random.seed(42)
        d_model = 16
        L = 5
        B = 1

        W_Q = np.random.randn(d_model, d_model) * 0.1
        W_K = np.random.randn(d_model, d_model) * 0.1
        W_V = np.random.randn(d_model, d_model) * 0.1

        X = np.random.randn(B, L, d_model)
        perm = np.array([2, 0, 4, 1, 3])
        X_perm = X[:, perm, :]

        enc = SinusoidalPositionalEncoding(L, d_model)
        X_pe = enc.forward(X)
        X_perm_pe = enc.forward(X_perm)

        def simple_attn(X_in):
            Q = X_in @ W_Q
            K = X_in @ W_K
            V = X_in @ W_V
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)
            A = np.exp(scores - scores.max(axis=-1, keepdims=True))
            A = A / A.sum(axis=-1, keepdims=True)
            return A @ V

        out1 = simple_attn(X_pe)
        out2 = simple_attn(X_perm_pe)

        self.assertGreater(np.linalg.norm(out1 - out2[0:1, perm, :]), 1e-10)

    def test_permutation_insensitivity_without_pe(self):
        np.random.seed(42)
        d_model = 16
        L = 5
        B = 1

        W_Q = np.random.randn(d_model, d_model) * 0.1
        W_K = np.random.randn(d_model, d_model) * 0.1
        W_V = np.random.randn(d_model, d_model) * 0.1

        X = np.random.randn(B, L, d_model)
        perm = np.array([2, 0, 4, 1, 3])
        X_perm = X[:, perm, :]

        def simple_attn(X_in):
            Q = X_in @ W_Q
            K = X_in @ W_K
            V = X_in @ W_V
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)
            A = np.exp(scores - scores.max(axis=-1, keepdims=True))
            A = A / A.sum(axis=-1, keepdims=True)
            return A @ V

        out1 = simple_attn(X)
        out2 = simple_attn(X_perm)

        inv_perm = np.argsort(perm)
        np.testing.assert_allclose(out1, out2[:, inv_perm, :], atol=1e-12)

    def test_additive_composition_statistics(self):
        np.random.seed(42)
        d_model = 64
        L = 128
        B = 4
        X = np.random.randn(B, L, d_model)
        enc = SinusoidalPositionalEncoding(L, d_model)
        X_pe = enc.forward(X)

        self.assertFalse(np.any(np.isnan(X_pe)))
        self.assertFalse(np.any(np.isinf(X_pe)))
        ratio = X_pe.std() / X.std()
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 3.0)

    def test_forward_with_attention_order_matters(self):
        np.random.seed(42)
        d_model = 16
        L = 6
        B = 1

        W_Q = np.random.randn(d_model, d_model) * 0.1
        W_K = np.random.randn(d_model, d_model) * 0.1
        W_V = np.random.randn(d_model, d_model) * 0.1

        X = np.random.randn(B, L, d_model)
        X_swapped = X.copy()
        X_swapped[:, 0, :], X_swapped[:, 1, :] = X[:, 1, :].copy(), X[:, 0, :].copy()

        enc = SinusoidalPositionalEncoding(L, d_model)

        def attn_with_pe(X_in):
            X_pe = enc.forward(X_in)
            Q = X_pe @ W_Q
            K = X_pe @ W_K
            V = X_pe @ W_V
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)
            A = np.exp(scores - scores.max(axis=-1, keepdims=True))
            A = A / A.sum(axis=-1, keepdims=True)
            return A @ V

        out1 = attn_with_pe(X)
        out2 = attn_with_pe(X_swapped)

        self.assertGreater(np.linalg.norm(out1 - out2), 1e-10)


class TestEncodingStatistics(unittest.TestCase):

    def test_encoding_statistics_keys(self):
        pe = sinusoidal_positional_encoding(50, 64)
        stats = encoding_statistics(pe)
        self.assertIn("position_norms", stats)
        self.assertIn("mean_per_dim", stats)
        self.assertIn("var_per_dim", stats)
        self.assertIn("min_val", stats)
        self.assertIn("max_val", stats)

    def test_encoding_statistics_values(self):
        pe = sinusoidal_positional_encoding(50, 64)
        stats = encoding_statistics(pe)
        self.assertGreaterEqual(stats["min_val"], -1.0 - 1e-15)
        self.assertLessEqual(stats["max_val"], 1.0 + 1e-15)
        self.assertEqual(stats["position_norms"].shape, (50,))
        self.assertEqual(stats["mean_per_dim"].shape, (64,))
        self.assertEqual(stats["var_per_dim"].shape, (64,))


class TestGetEncoding(unittest.TestCase):

    def test_get_encoding_shape(self):
        enc = SinusoidalPositionalEncoding(128, 64)
        pe = enc.get_encoding(32)
        self.assertEqual(pe.shape, (32, 64))

    def test_get_encoding_matches_full(self):
        enc = SinusoidalPositionalEncoding(128, 64)
        pe_full = enc.get_encoding(128)
        pe_slice = enc.get_encoding(32)
        np.testing.assert_array_equal(pe_slice, pe_full[:32])


if __name__ == "__main__":
    unittest.main()
