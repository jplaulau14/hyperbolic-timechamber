"""Tests for flash attention concept."""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

import unittest
import numpy as np
from importlib import import_module

_mod = import_module("03-transformers.flash-attention-concept.implementation")
online_softmax = _mod.online_softmax
online_softmax_2d = _mod.online_softmax_2d
standard_attention = _mod.standard_attention
tiled_attention = _mod.tiled_attention
memory_analysis = _mod.memory_analysis
verify_no_full_materialization = _mod.verify_no_full_materialization


def _reference_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _reference_attention(Q, K, V):
    """Standard attention for comparison."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    S_max = np.max(S, axis=-1, keepdims=True)
    P = np.exp(S - S_max) / np.sum(np.exp(S - S_max), axis=-1, keepdims=True)
    return P @ V


class TestOnlineSoftmax(unittest.TestCase):

    def test_equivalence_random_vectors(self):
        """Online softmax matches reference for various vector sizes."""
        rng = np.random.RandomState(42)
        for N in [10, 100, 1000, 10000]:
            x = rng.randn(N)
            ref = _reference_softmax(x)
            for chunk in [1, 7, 10, 50, 100, N]:
                result, _, _ = online_softmax(x, chunk_size=chunk)
                np.testing.assert_allclose(
                    result, ref, atol=1e-10,
                    err_msg=f"N={N}, chunk_size={chunk}"
                )

    def test_numerical_stability_large_values(self):
        """Online softmax handles large values without overflow."""
        x = np.array([1000.0, 1000.0, 1000.0])
        result, m, ell = online_softmax(x, chunk_size=1)
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(result, expected, atol=1e-10)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_numerical_stability_very_negative(self):
        """Online softmax handles very negative values without underflow."""
        x = np.array([-1000.0, -1000.0, -999.0])
        result, _, _ = online_softmax(x, chunk_size=1)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertAlmostEqual(np.sum(result), 1.0, places=10)
        self.assertGreater(result[2], result[0])

    def test_single_element(self):
        result, m, ell = online_softmax(np.array([5.0]))
        np.testing.assert_allclose(result, [1.0])

    def test_two_equal_elements(self):
        result, _, _ = online_softmax(np.array([0.0, 0.0]))
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_statistics_consistency(self):
        """Running (m, ell) from chunked processing matches full-pass values."""
        rng = np.random.RandomState(99)
        x = rng.randn(200)
        _, m_full, ell_full = online_softmax(x, chunk_size=0)
        _, m_chunked, ell_chunked = online_softmax(x, chunk_size=13)
        self.assertAlmostEqual(m_full, m_chunked, places=12)
        self.assertAlmostEqual(ell_full, ell_chunked, places=8)

    def test_sums_to_one(self):
        rng = np.random.RandomState(7)
        x = rng.randn(500)
        result, _, _ = online_softmax(x, chunk_size=17)
        self.assertAlmostEqual(np.sum(result), 1.0, places=10)


class TestOnlineSoftmax2D(unittest.TestCase):

    def test_row_wise(self):
        rng = np.random.RandomState(11)
        x = rng.randn(8, 20)
        result = online_softmax_2d(x, chunk_size=5)
        self.assertEqual(result.shape, (8, 20))
        for i in range(8):
            ref = _reference_softmax(x[i])
            np.testing.assert_allclose(result[i], ref, atol=1e-10)

    def test_rows_sum_to_one(self):
        rng = np.random.RandomState(22)
        x = rng.randn(16, 50)
        result = online_softmax_2d(x, chunk_size=7)
        row_sums = np.sum(result, axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(16), atol=1e-10)


class TestStandardAttention(unittest.TestCase):

    def test_output_shape(self):
        rng = np.random.RandomState(0)
        N, d = 32, 16
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        O, P, _ = standard_attention(Q, K, V)
        self.assertEqual(O.shape, (N, d))
        self.assertEqual(P.shape, (N, N))

    def test_attention_weights_sum_to_one(self):
        rng = np.random.RandomState(1)
        N, d = 16, 8
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        _, P, _ = standard_attention(Q, K, V)
        row_sums = np.sum(P, axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(N), atol=1e-10)

    def test_peak_bytes_positive(self):
        rng = np.random.RandomState(2)
        N, d = 16, 8
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        _, _, peak = standard_attention(Q, K, V)
        self.assertGreater(peak, 0)
        self.assertEqual(peak, 2 * N * N * 8)  # two (N,N) float64 matrices


class TestTiledAttention(unittest.TestCase):

    def _assert_equivalent(self, N, d, block_q, block_kv, seed=42, atol=1e-5):
        rng = np.random.RandomState(seed)
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        O_std, _, _ = standard_attention(Q, K, V)
        O_tiled = tiled_attention(Q, K, V, block_size_q=block_q, block_size_kv=block_kv)
        np.testing.assert_allclose(
            O_tiled, O_std, atol=atol,
            err_msg=f"N={N}, d={d}, block_q={block_q}, block_kv={block_kv}"
        )

    def test_basic_64_32_16(self):
        self._assert_equivalent(N=64, d=32, block_q=16, block_kv=16)

    def test_basic_128_64_32(self):
        self._assert_equivalent(N=128, d=64, block_q=32, block_kv=32)

    def test_basic_256_128_64(self):
        self._assert_equivalent(N=256, d=128, block_q=64, block_kv=64)

    def test_non_divisible_100_32(self):
        self._assert_equivalent(N=100, d=32, block_q=32, block_kv=32)

    def test_non_divisible_65_64(self):
        self._assert_equivalent(N=65, d=32, block_q=64, block_kv=64)

    def test_various_block_sizes_same_output(self):
        rng = np.random.RandomState(77)
        N, d = 64, 32
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        O_ref, _, _ = standard_attention(Q, K, V)
        for bs in [8, 16, 32, 64]:
            O_tiled = tiled_attention(Q, K, V, block_size_q=bs, block_size_kv=bs)
            np.testing.assert_allclose(
                O_tiled, O_ref, atol=1e-5,
                err_msg=f"block_size={bs}"
            )

    def test_single_block_degenerate(self):
        """block_size >= N should exactly match standard attention."""
        rng = np.random.RandomState(33)
        N, d = 16, 8
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        O_std, _, _ = standard_attention(Q, K, V)
        O_tiled = tiled_attention(Q, K, V, block_size_q=N, block_size_kv=N)
        np.testing.assert_allclose(O_tiled, O_std, atol=1e-12)

    def test_output_shape(self):
        rng = np.random.RandomState(0)
        N, d = 48, 16
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        O = tiled_attention(Q, K, V, block_size_q=16, block_size_kv=16)
        self.assertEqual(O.shape, (N, d))

    def test_asymmetric_block_sizes(self):
        self._assert_equivalent(N=64, d=32, block_q=8, block_kv=32)
        self._assert_equivalent(N=64, d=32, block_q=32, block_kv=8)

    def test_small_block_size(self):
        self._assert_equivalent(N=64, d=16, block_q=4, block_kv=4)

    def test_large_N(self):
        self._assert_equivalent(N=2048, d=64, block_q=64, block_kv=64, atol=1e-4)

    def test_determinism(self):
        rng = np.random.RandomState(55)
        N, d = 32, 16
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        O1 = tiled_attention(Q, K, V, block_size_q=8, block_size_kv=8)
        O2 = tiled_attention(Q, K, V, block_size_q=8, block_size_kv=8)
        np.testing.assert_array_equal(O1, O2)


class TestCausalTiledAttention(unittest.TestCase):

    def test_causal_matches_masked_standard(self):
        """Causal tiled attention matches standard attention with causal mask."""
        rng = np.random.RandomState(88)
        N, d = 32, 16
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)

        S = Q @ K.T / np.sqrt(d)
        mask = np.full((N, N), -np.inf)
        mask[np.tril_indices(N)] = 0.0
        S = S + mask
        S_max = np.max(S, axis=-1, keepdims=True)
        P = np.exp(S - S_max) / np.sum(np.exp(S - S_max), axis=-1, keepdims=True)
        O_ref = P @ V

        O_causal = tiled_attention(Q, K, V, block_size_q=8, block_size_kv=8, causal=True)
        np.testing.assert_allclose(O_causal, O_ref, atol=1e-5)

    def test_causal_non_divisible(self):
        rng = np.random.RandomState(99)
        N, d = 30, 16
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)

        S = Q @ K.T / np.sqrt(d)
        mask = np.full((N, N), -np.inf)
        mask[np.tril_indices(N)] = 0.0
        S = S + mask
        S_max = np.max(S, axis=-1, keepdims=True)
        P = np.exp(S - S_max) / np.sum(np.exp(S - S_max), axis=-1, keepdims=True)
        O_ref = P @ V

        O_causal = tiled_attention(Q, K, V, block_size_q=7, block_size_kv=11, causal=True)
        np.testing.assert_allclose(O_causal, O_ref, atol=1e-5)


class TestMemoryVerification(unittest.TestCase):

    def test_no_full_materialization(self):
        """Largest intermediate should be O(block_size^2), not O(N^2)."""
        rng = np.random.RandomState(10)
        N, d, block_size = 256, 32, 32
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        O, max_elems = verify_no_full_materialization(Q, K, V, block_size)

        self.assertLessEqual(max_elems, block_size * d)
        self.assertLess(max_elems, N * N)

        O_std, _, _ = standard_attention(Q, K, V)
        np.testing.assert_allclose(O, O_std, atol=1e-5)

    def test_max_tensor_scales_with_block_not_N(self):
        """Peak intermediate should stay constant as N grows."""
        rng = np.random.RandomState(20)
        block_size = 32
        d = 16
        max_elems_list = []
        for N in [128, 256, 512]:
            Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
            _, max_elems = verify_no_full_materialization(Q, K, V, block_size)
            max_elems_list.append(max_elems)

        self.assertEqual(max_elems_list[0], max_elems_list[1])
        self.assertEqual(max_elems_list[1], max_elems_list[2])


class TestMemoryAnalysis(unittest.TestCase):

    def test_standard_quadratic(self):
        result = memory_analysis(N=1024, d=64, block_size=32, dtype="float32")
        std = result["standard"]["total_bytes"]
        self.assertGreater(std, 1024 * 1024 * 4 * 2)

    def test_tiled_much_smaller(self):
        result = memory_analysis(N=1024, d=64, block_size=32, dtype="float32")
        self.assertGreater(result["ratio"], 10)

    def test_ratio_grows_with_N(self):
        r1 = memory_analysis(N=256, d=64, block_size=32)["ratio"]
        r2 = memory_analysis(N=1024, d=64, block_size=32)["ratio"]
        r3 = memory_analysis(N=4096, d=64, block_size=32)["ratio"]
        self.assertGreater(r2, r1)
        self.assertGreater(r3, r2)

    def test_keys_present(self):
        result = memory_analysis(N=128, d=32, block_size=16)
        self.assertIn("standard", result)
        self.assertIn("tiled", result)
        self.assertIn("ratio", result)
        self.assertIn("total_bytes", result["standard"])
        self.assertIn("total_bytes", result["tiled"])


if __name__ == "__main__":
    unittest.main()
