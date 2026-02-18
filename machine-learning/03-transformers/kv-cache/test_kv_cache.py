"""Tests for KV cache."""

import sys
from pathlib import Path
import unittest
import time

import numpy as np

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from importlib import import_module

_mod = import_module("03-transformers.kv-cache.implementation")
KVCache = _mod.KVCache
block_forward_with_cache = _mod.block_forward_with_cache
generate_without_cache = _mod.generate_without_cache
generate_with_cache = _mod.generate_with_cache
memory_usage = _mod.memory_usage
flops_comparison = _mod.flops_comparison
model_kv_cache_bytes = _mod.model_kv_cache_bytes

_causal_mod = import_module("03-transformers.causal-decoding.implementation")
CausalLM = _causal_mod.CausalLM

_block_mod = import_module("03-transformers.transformer-block.implementation")
TransformerBlock = _block_mod.TransformerBlock

_gqa_mod = import_module("03-transformers.grouped-query-attention.implementation")
create_causal_mask = _gqa_mod.create_causal_mask
repeat_kv = _gqa_mod.repeat_kv
softmax = _gqa_mod.softmax

_rope_mod = import_module("03-transformers.rope.implementation")
apply_rope = _rope_mod.apply_rope


TINY = dict(
    vocab_size=100, d_model=32, num_layers=2, num_heads=4,
    num_kv_heads=2, d_ff=64, max_seq_len=64, rope_theta=10000.0,
)


def _make_model(seed=42, **overrides):
    cfg = {**TINY, **overrides}
    np.random.seed(seed)
    return CausalLM(**cfg, tie_weights=True)


# ---------------------------------------------------------------------------
# KVCache class tests
# ---------------------------------------------------------------------------

class TestKVCacheInit(unittest.TestCase):

    def test_dynamic_empty(self):
        cache = KVCache(n_layers=4, batch_size=2, n_kv_heads=8, d_k=32)
        self.assertEqual(cache.seq_len, 0)
        self.assertEqual(cache.memory_bytes(), 0)

    def test_preallocated_empty(self):
        cache = KVCache(n_layers=4, batch_size=2, n_kv_heads=8, d_k=32, max_seq_len=128)
        self.assertEqual(cache.seq_len, 0)
        self.assertEqual(cache.memory_bytes(), 0)


class TestKVCacheAppend(unittest.TestCase):

    def test_single_append_shape(self):
        cache = KVCache(n_layers=2, batch_size=2, n_kv_heads=4, d_k=16)
        k = np.random.randn(2, 4, 1, 16)
        v = np.random.randn(2, 4, 1, 16)
        K, V = cache.append(0, k, v)
        self.assertEqual(K.shape, (2, 4, 1, 16))
        self.assertEqual(V.shape, (2, 4, 1, 16))

    def test_multiple_appends_shape(self):
        cache = KVCache(n_layers=2, batch_size=2, n_kv_heads=4, d_k=16)
        for i in range(5):
            k = np.random.randn(2, 4, 1, 16)
            v = np.random.randn(2, 4, 1, 16)
            K, V = cache.append(0, k, v)
        self.assertEqual(K.shape, (2, 4, 5, 16))
        self.assertEqual(V.shape, (2, 4, 5, 16))
        self.assertEqual(cache._seq_lens[0], 5)

    def test_batch_append(self):
        """Appending multiple tokens at once (prefill)."""
        cache = KVCache(n_layers=1, batch_size=2, n_kv_heads=4, d_k=16)
        k = np.random.randn(2, 4, 10, 16)
        v = np.random.randn(2, 4, 10, 16)
        K, V = cache.append(0, k, v)
        self.assertEqual(K.shape, (2, 4, 10, 16))

    def test_preallocated_append(self):
        cache = KVCache(n_layers=1, batch_size=2, n_kv_heads=4, d_k=16, max_seq_len=128)
        k = np.random.randn(2, 4, 1, 16)
        v = np.random.randn(2, 4, 1, 16)
        K, V = cache.append(0, k, v)
        self.assertEqual(K.shape, (2, 4, 1, 16))
        np.testing.assert_array_equal(K, k)

    def test_preallocated_overflow(self):
        cache = KVCache(n_layers=1, batch_size=1, n_kv_heads=1, d_k=4, max_seq_len=3)
        for _ in range(3):
            cache.append(0, np.zeros((1, 1, 1, 4)), np.zeros((1, 1, 1, 4)))
        with self.assertRaises(ValueError):
            cache.append(0, np.zeros((1, 1, 1, 4)), np.zeros((1, 1, 1, 4)))

    def test_batch_mismatch(self):
        cache = KVCache(n_layers=1, batch_size=2, n_kv_heads=4, d_k=16)
        with self.assertRaises(ValueError):
            cache.append(0, np.zeros((3, 4, 1, 16)), np.zeros((3, 4, 1, 16)))

    def test_shape_mismatch(self):
        cache = KVCache(n_layers=1, batch_size=2, n_kv_heads=4, d_k=16)
        with self.assertRaises(ValueError):
            cache.append(0, np.zeros((2, 3, 1, 16)), np.zeros((2, 3, 1, 16)))


class TestKVCacheGetKV(unittest.TestCase):

    def test_empty_cache_shape(self):
        cache = KVCache(n_layers=2, batch_size=2, n_kv_heads=4, d_k=16)
        K, V = cache.get_kv(0)
        self.assertEqual(K.shape, (2, 4, 0, 16))
        self.assertEqual(V.shape, (2, 4, 0, 16))

    def test_get_after_append(self):
        cache = KVCache(n_layers=2, batch_size=2, n_kv_heads=4, d_k=16)
        k = np.random.randn(2, 4, 3, 16)
        v = np.random.randn(2, 4, 3, 16)
        cache.append(0, k, v)
        K, V = cache.get_kv(0)
        np.testing.assert_array_equal(K, k)
        np.testing.assert_array_equal(V, v)

    def test_multi_layer_independence(self):
        cache = KVCache(n_layers=3, batch_size=1, n_kv_heads=2, d_k=8)
        k0 = np.random.randn(1, 2, 2, 8)
        v0 = np.random.randn(1, 2, 2, 8)
        k1 = np.random.randn(1, 2, 5, 8)
        v1 = np.random.randn(1, 2, 5, 8)
        cache.append(0, k0, v0)
        cache.append(1, k1, v1)

        K0, V0 = cache.get_kv(0)
        K1, V1 = cache.get_kv(1)
        K2, V2 = cache.get_kv(2)

        self.assertEqual(K0.shape[2], 2)
        self.assertEqual(K1.shape[2], 5)
        self.assertEqual(K2.shape[2], 0)


class TestKVCacheReset(unittest.TestCase):

    def test_reset_clears(self):
        cache = KVCache(n_layers=2, batch_size=1, n_kv_heads=2, d_k=8)
        cache.append(0, np.zeros((1, 2, 3, 8)), np.zeros((1, 2, 3, 8)))
        cache.reset()
        self.assertEqual(cache.seq_len, 0)
        self.assertEqual(cache.memory_bytes(), 0)


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

class TestPreallocVsDynamic(unittest.TestCase):

    def test_both_produce_same_values(self):
        """Pre-allocated and dynamic caches should store identical values."""
        np.random.seed(0)
        B, n_kv, d_k = 2, 4, 16
        k_vals = [np.random.randn(B, n_kv, 1, d_k) for _ in range(5)]
        v_vals = [np.random.randn(B, n_kv, 1, d_k) for _ in range(5)]

        cache_dyn = KVCache(n_layers=1, batch_size=B, n_kv_heads=n_kv, d_k=d_k)
        cache_pre = KVCache(n_layers=1, batch_size=B, n_kv_heads=n_kv, d_k=d_k, max_seq_len=10)

        for k, v in zip(k_vals, v_vals):
            cache_dyn.append(0, k, v)
            cache_pre.append(0, k, v)

        K_dyn, V_dyn = cache_dyn.get_kv(0)
        K_pre, V_pre = cache_pre.get_kv(0)
        np.testing.assert_array_equal(K_dyn, K_pre)
        np.testing.assert_array_equal(V_dyn, V_pre)
        self.assertEqual(cache_dyn.memory_bytes(), cache_pre.memory_bytes())


class TestMemoryCalculation(unittest.TestCase):

    def test_known_size(self):
        """Verify memory_bytes matches manual calculation."""
        n_layers, B, n_kv, d_k = 4, 2, 8, 32
        cache = KVCache(n_layers=n_layers, batch_size=B, n_kv_heads=n_kv, d_k=d_k)
        seq_len = 10
        for layer in range(n_layers):
            k = np.random.randn(B, n_kv, seq_len, d_k)
            v = np.random.randn(B, n_kv, seq_len, d_k)
            cache.append(layer, k, v)

        element_size = 8  # float64
        expected = 2 * n_layers * B * n_kv * seq_len * d_k * element_size
        self.assertEqual(cache.memory_bytes(), expected)

    def test_linear_growth(self):
        """Memory should grow linearly with sequence length."""
        cache = KVCache(n_layers=2, batch_size=1, n_kv_heads=4, d_k=16)
        sizes = []
        for step in range(1, 11):
            cache.append(0, np.zeros((1, 4, 1, 16)), np.zeros((1, 4, 1, 16)))
            cache.append(1, np.zeros((1, 4, 1, 16)), np.zeros((1, 4, 1, 16)))
            sizes.append(cache.memory_bytes())

        diffs = [sizes[i+1] - sizes[i] for i in range(len(sizes)-1)]
        for d in diffs:
            self.assertEqual(d, diffs[0])

    def test_memory_usage_utility(self):
        cache = KVCache(n_layers=4, batch_size=2, n_kv_heads=8, d_k=32)
        for layer in range(4):
            cache.append(layer, np.zeros((2, 8, 5, 32)), np.zeros((2, 8, 5, 32)))

        info = memory_usage(cache)
        self.assertEqual(info["total_bytes"], cache.memory_bytes())
        self.assertEqual(info["seq_len"], 5)
        self.assertGreater(info["bytes_per_token"], 0)

    def test_7b_model_cache_size(self):
        """Verify ~0.5 MB per token for 7B model config (FP16)."""
        total = model_kv_cache_bytes(
            batch_size=1, seq_len=1, n_layers=32, n_heads=32,
            d_model=4096, bytes_per_element=2,
        )
        mb_per_token = total / (1024 ** 2)
        self.assertAlmostEqual(mb_per_token, 0.5, delta=0.01)

    def test_7b_4k_context(self):
        """4096-context 7B model KV cache should be ~2 GB in FP16."""
        total = model_kv_cache_bytes(
            batch_size=1, seq_len=4096, n_layers=32, n_heads=32,
            d_model=4096, bytes_per_element=2,
        )
        gb = total / (1024 ** 3)
        self.assertAlmostEqual(gb, 2.0, delta=0.01)


# ---------------------------------------------------------------------------
# Block-level correctness
# ---------------------------------------------------------------------------

class TestBlockForwardWithCache(unittest.TestCase):

    def _make_block(self, seed=42):
        np.random.seed(seed)
        return TransformerBlock(
            d_model=32, num_heads=4, num_kv_heads=2, d_ff=64,
            max_seq_len=64, rope_theta=10000.0,
        )

    def test_no_cache_matches_original(self):
        """Without cache, block_forward_with_cache should match block.forward."""
        block = self._make_block()
        B, L, d = 2, 5, 32
        np.random.seed(0)
        x = np.random.randn(B, L, d)
        positions = np.arange(L)
        mask = create_causal_mask(L)

        out_orig = block.forward(x, mask=mask, positions=positions)
        out_cached = block_forward_with_cache(block, x, positions, kv_cache=None, mask=mask)

        np.testing.assert_allclose(out_orig, out_cached, atol=1e-12)

    def test_prefill_then_decode_matches_full(self):
        """Prefill + single-token decode should match full-sequence forward."""
        block = self._make_block()
        B, d = 1, 32
        np.random.seed(0)
        x_full = np.random.randn(B, 4, d)

        positions_full = np.arange(4)
        mask_full = create_causal_mask(4)
        out_full = block.forward(x_full, mask=mask_full, positions=positions_full)

        cache = KVCache(n_layers=1, batch_size=B, n_kv_heads=2, d_k=8)

        x_prefill = x_full[:, :3, :]
        positions_prefill = np.arange(3)
        out_prefill = block_forward_with_cache(
            block, x_prefill, positions_prefill, cache, layer_idx=0,
        )

        x_decode = x_full[:, 3:4, :]
        positions_decode = np.array([3])
        out_decode = block_forward_with_cache(
            block, x_decode, positions_decode, cache, layer_idx=0,
        )

        np.testing.assert_allclose(out_prefill, out_full[:, :3, :], atol=1e-10)
        np.testing.assert_allclose(out_decode, out_full[:, 3:4, :], atol=1e-10)

    def test_incremental_consistency(self):
        """Building cache token-by-token matches batched K, V projection."""
        block = self._make_block()
        B, L, d = 1, 3, 32
        np.random.seed(0)
        x = np.random.randn(B, L, d)

        cache_inc = KVCache(n_layers=1, batch_size=B, n_kv_heads=2, d_k=8)
        for t in range(L):
            block_forward_with_cache(
                block, x[:, t:t+1, :], np.array([t]), cache_inc, layer_idx=0,
            )

        cache_batch = KVCache(n_layers=1, batch_size=B, n_kv_heads=2, d_k=8)
        block_forward_with_cache(
            block, x, np.arange(L), cache_batch, layer_idx=0,
        )

        K_inc, V_inc = cache_inc.get_kv(0)
        K_bat, V_bat = cache_batch.get_kv(0)

        np.testing.assert_allclose(K_inc, K_bat, atol=1e-12)
        np.testing.assert_allclose(V_inc, V_bat, atol=1e-12)


class TestAttentionOutputCorrectness(unittest.TestCase):

    def test_last_row_matches(self):
        """At step t, cached attention output for token t matches row t of full attention."""
        np.random.seed(42)
        block = TransformerBlock(
            d_model=32, num_heads=4, num_kv_heads=2, d_ff=64,
            max_seq_len=64, rope_theta=10000.0,
        )
        B, L, d = 1, 6, 32
        np.random.seed(0)
        x = np.random.randn(B, L, d)

        out_full = block.forward(x, mask=create_causal_mask(L), positions=np.arange(L))

        cache = KVCache(n_layers=1, batch_size=B, n_kv_heads=2, d_k=8)
        for t in range(L):
            out_t = block_forward_with_cache(
                block, x[:, t:t+1, :], np.array([t]), cache, layer_idx=0,
            )

        np.testing.assert_allclose(out_t, out_full[:, -1:, :], atol=1e-10)


# ---------------------------------------------------------------------------
# End-to-end generation correctness (THE critical test)
# ---------------------------------------------------------------------------

class TestOutputEquivalence(unittest.TestCase):

    def test_greedy_equivalence(self):
        """Generate 10 tokens with and without cache -- outputs must be identical."""
        model = _make_model()
        prompt = np.array([[1, 2, 3, 4, 5]])

        tokens_no_cache, _ = generate_without_cache(model, prompt, n_tokens=10, greedy=True, seed=0)
        tokens_cached, _ = generate_with_cache(model, prompt, n_tokens=10, greedy=True, seed=0)

        np.testing.assert_array_equal(tokens_no_cache, tokens_cached)

    def test_greedy_equivalence_longer(self):
        """Generate 20 tokens from a longer prompt."""
        model = _make_model()
        prompt = np.array([[10, 20, 30, 40, 50, 60, 70, 80]])

        tokens_no_cache, _ = generate_without_cache(model, prompt, n_tokens=20, greedy=True, seed=0)
        tokens_cached, _ = generate_with_cache(model, prompt, n_tokens=20, greedy=True, seed=0)

        np.testing.assert_array_equal(tokens_no_cache, tokens_cached)

    def test_greedy_equivalence_single_token_prompt(self):
        """Single-token prompt."""
        model = _make_model()
        prompt = np.array([[42]])

        tokens_no_cache, _ = generate_without_cache(model, prompt, n_tokens=10, greedy=True, seed=0)
        tokens_cached, _ = generate_with_cache(model, prompt, n_tokens=10, greedy=True, seed=0)

        np.testing.assert_array_equal(tokens_no_cache, tokens_cached)

    def test_greedy_equivalence_batch(self):
        """Batch of 2 prompts."""
        model = _make_model()
        prompt = np.array([[1, 2, 3], [4, 5, 6]])

        tokens_no_cache, _ = generate_without_cache(model, prompt, n_tokens=8, greedy=True, seed=0)
        tokens_cached, _ = generate_with_cache(model, prompt, n_tokens=8, greedy=True, seed=0)

        np.testing.assert_array_equal(tokens_no_cache, tokens_cached)

    def test_equivalence_deeper_model(self):
        """4-layer model to stress multi-layer cache."""
        model = _make_model(num_layers=4)
        prompt = np.array([[1, 2, 3, 4, 5]])

        tokens_no_cache, _ = generate_without_cache(model, prompt, n_tokens=10, greedy=True, seed=0)
        tokens_cached, _ = generate_with_cache(model, prompt, n_tokens=10, greedy=True, seed=0)

        np.testing.assert_array_equal(tokens_no_cache, tokens_cached)


# ---------------------------------------------------------------------------
# Prefill equivalence
# ---------------------------------------------------------------------------

class TestPrefillEquivalence(unittest.TestCase):

    def test_prefill_kv_matches_full_forward(self):
        """Running prefill through cache should produce same K, V as full forward."""
        np.random.seed(42)
        block = TransformerBlock(
            d_model=32, num_heads=4, num_kv_heads=2, d_ff=64,
            max_seq_len=64, rope_theta=10000.0,
        )
        B, L, d = 2, 6, 32
        np.random.seed(0)
        x = np.random.randn(B, L, d)

        # Full forward to get K, V via manual projection
        x_norm = block.norm1.forward(x)
        K_proj = (x_norm @ block.W_K).reshape(B, L, 2, 8).transpose(0, 2, 1, 3)
        V_proj = (x_norm @ block.W_V).reshape(B, L, 2, 8).transpose(0, 2, 1, 3)
        K_rot = apply_rope(K_proj, block.rope.cos_cache, block.rope.sin_cache, np.arange(L))

        # Prefill through cache
        cache = KVCache(n_layers=1, batch_size=B, n_kv_heads=2, d_k=8)
        block_forward_with_cache(block, x, np.arange(L), cache, layer_idx=0)
        K_cached, V_cached = cache.get_kv(0)

        np.testing.assert_allclose(K_cached, K_rot, atol=1e-12)
        np.testing.assert_allclose(V_cached, V_proj, atol=1e-12)

    def test_incremental_kv_matches_batched(self):
        """Token-by-token KV accumulation matches single-pass prefill."""
        np.random.seed(42)
        block = TransformerBlock(
            d_model=32, num_heads=4, num_kv_heads=2, d_ff=64,
            max_seq_len=64, rope_theta=10000.0,
        )
        B, L, d = 1, 5, 32
        np.random.seed(0)
        x = np.random.randn(B, L, d)

        cache_inc = KVCache(n_layers=1, batch_size=B, n_kv_heads=2, d_k=8)
        for t in range(L):
            block_forward_with_cache(
                block, x[:, t:t+1, :], np.array([t]), cache_inc, layer_idx=0,
            )

        cache_batch = KVCache(n_layers=1, batch_size=B, n_kv_heads=2, d_k=8)
        block_forward_with_cache(block, x, np.arange(L), cache_batch, layer_idx=0)

        K_inc, V_inc = cache_inc.get_kv(0)
        K_bat, V_bat = cache_batch.get_kv(0)
        np.testing.assert_allclose(K_inc, K_bat, atol=1e-12)
        np.testing.assert_allclose(V_inc, V_bat, atol=1e-12)


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

class TestShapeValidation(unittest.TestCase):

    def test_single_token_cache_shape(self):
        cache = KVCache(n_layers=1, batch_size=2, n_kv_heads=8, d_k=32)
        cache.append(0, np.zeros((2, 8, 1, 32)), np.zeros((2, 8, 1, 32)))
        K, V = cache.get_kv(0)
        self.assertEqual(K.shape, (2, 8, 1, 32))

    def test_n_appends_shape(self):
        cache = KVCache(n_layers=1, batch_size=2, n_kv_heads=8, d_k=32)
        N = 7
        for _ in range(N):
            cache.append(0, np.zeros((2, 8, 1, 32)), np.zeros((2, 8, 1, 32)))
        K, V = cache.get_kv(0)
        self.assertEqual(K.shape, (2, 8, N, 32))

    def test_batch_preserved(self):
        cache = KVCache(n_layers=1, batch_size=4, n_kv_heads=2, d_k=16)
        cache.append(0, np.random.randn(4, 2, 3, 16), np.random.randn(4, 2, 3, 16))
        K, V = cache.get_kv(0)
        self.assertEqual(K.shape[0], 4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_batch_size_1(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3]])
        tokens_nc, _ = generate_without_cache(model, prompt, n_tokens=5, greedy=True, seed=0)
        tokens_c, _ = generate_with_cache(model, prompt, n_tokens=5, greedy=True, seed=0)
        np.testing.assert_array_equal(tokens_nc, tokens_c)

    def test_1d_prompt(self):
        model = _make_model()
        prompt = np.array([1, 2, 3])
        tokens_nc, _ = generate_without_cache(model, prompt, n_tokens=5, greedy=True, seed=0)
        tokens_c, _ = generate_with_cache(model, prompt, n_tokens=5, greedy=True, seed=0)
        np.testing.assert_array_equal(tokens_nc, tokens_c)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability(unittest.TestCase):

    def test_large_activation_magnitudes(self):
        """Cache should handle large values without overflow."""
        cache = KVCache(n_layers=1, batch_size=1, n_kv_heads=2, d_k=8)
        k = np.full((1, 2, 1, 8), 1e6)
        v = np.full((1, 2, 1, 8), 1e6)
        K, V = cache.append(0, k, v)
        self.assertTrue(np.all(np.isfinite(K)))
        self.assertTrue(np.all(np.isfinite(V)))

    def test_generation_outputs_finite(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3, 4, 5]])
        tokens, _ = generate_with_cache(model, prompt, n_tokens=15, greedy=True, seed=0)
        self.assertTrue(np.all(np.isfinite(tokens)))
        self.assertTrue(np.all(tokens >= 0))
        self.assertTrue(np.all(tokens < TINY["vocab_size"]))


# ---------------------------------------------------------------------------
# FLOP comparison
# ---------------------------------------------------------------------------

class TestFLOPComparison(unittest.TestCase):

    def test_cache_fewer_projection_flops(self):
        result = flops_comparison(
            prompt_len=10, n_tokens=50, n_layers=4, d_model=256,
            num_heads=8, num_kv_heads=4,
        )
        self.assertGreater(result["without_cache"], result["with_cache"])
        self.assertGreater(result["speedup"], 1.0)

    def test_cached_projection_flops_returned(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3, 4, 5]])
        _, flops_nc = generate_without_cache(model, prompt, n_tokens=10, greedy=True, seed=0)
        _, flops_c = generate_with_cache(model, prompt, n_tokens=10, greedy=True, seed=0)
        self.assertGreater(flops_nc, flops_c)

    def test_speedup_grows_with_sequence(self):
        r1 = flops_comparison(
            prompt_len=5, n_tokens=10, n_layers=4, d_model=256,
            num_heads=8, num_kv_heads=4,
        )
        r2 = flops_comparison(
            prompt_len=5, n_tokens=100, n_layers=4, d_model=256,
            num_heads=8, num_kv_heads=4,
        )
        self.assertGreater(r2["speedup"], r1["speedup"])


# ---------------------------------------------------------------------------
# Performance test (wall-clock)
# ---------------------------------------------------------------------------

class TestPerformance(unittest.TestCase):

    def test_cached_faster_than_naive(self):
        """For seq_len >= 100, cached version should be measurably faster."""
        model = _make_model(max_seq_len=256)
        prompt = np.array([[1, 2, 3, 4, 5]])
        n_gen = 40

        t0 = time.perf_counter()
        generate_without_cache(model, prompt, n_tokens=n_gen, greedy=True, seed=0)
        t_naive = time.perf_counter() - t0

        t0 = time.perf_counter()
        generate_with_cache(model, prompt, n_tokens=n_gen, greedy=True, seed=0)
        t_cached = time.perf_counter() - t0

        self.assertLess(t_cached, t_naive)


# ---------------------------------------------------------------------------
# Multi-layer cache management
# ---------------------------------------------------------------------------

class TestMultiLayerCache(unittest.TestCase):

    def test_all_layers_populated(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3, 4, 5]])
        generate_with_cache(model, prompt, n_tokens=5, greedy=True, seed=0)
        # If we got here without error, all layers were used correctly.

    def test_layer_isolation(self):
        """Each layer's cache should contain different values."""
        np.random.seed(42)
        block0 = TransformerBlock(d_model=32, num_heads=4, num_kv_heads=2, d_ff=64, max_seq_len=64)
        np.random.seed(99)
        block1 = TransformerBlock(d_model=32, num_heads=4, num_kv_heads=2, d_ff=64, max_seq_len=64)

        B, L, d = 1, 3, 32
        np.random.seed(0)
        x = np.random.randn(B, L, d)

        cache = KVCache(n_layers=2, batch_size=B, n_kv_heads=2, d_k=8)
        out0 = block_forward_with_cache(block0, x, np.arange(L), cache, layer_idx=0)
        block_forward_with_cache(block1, out0, np.arange(L), cache, layer_idx=1)

        K0, _ = cache.get_kv(0)
        K1, _ = cache.get_kv(1)
        self.assertFalse(np.allclose(K0, K1))


if __name__ == "__main__":
    unittest.main()
