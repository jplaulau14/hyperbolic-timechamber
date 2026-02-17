"""Tests for causal decoding."""

import sys
from pathlib import Path
import unittest

import numpy as np

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from importlib import import_module

_mod = import_module("03-transformers.causal-decoding.implementation")
CausalLM = _mod.CausalLM
temperature_scale = _mod.temperature_scale
top_k_filter = _mod.top_k_filter
top_p_filter = _mod.top_p_filter
sample_token = _mod.sample_token
count_model_parameters = _mod.count_model_parameters
generation_flops = _mod.generation_flops
generation_flops_with_cache = _mod.generation_flops_with_cache
softmax = _mod.softmax


TINY = dict(
    vocab_size=100, d_model=32, num_layers=2, num_heads=4,
    num_kv_heads=2, d_ff=64, max_seq_len=64, rope_theta=10000.0,
)

TINY_PARAM_KEYS = dict(
    vocab_size=100, d_model=32, num_layers=2, num_heads=4,
    num_kv_heads=2, d_ff=64,
)


def _make_model(tie_weights=True, seed=42, **overrides):
    cfg = {**TINY, **overrides}
    np.random.seed(seed)
    return CausalLM(**cfg, tie_weights=tie_weights)


class TestOutputShape(unittest.TestCase):

    def test_basic_shape(self):
        model = _make_model()
        tokens = np.array([[1, 2, 3]])
        logits = model.forward(tokens)
        self.assertEqual(logits.shape, (1, 3, TINY["vocab_size"]))

    def test_batch_shape(self):
        model = _make_model()
        tokens = np.array([[1, 2, 3], [4, 5, 6]])
        logits = model.forward(tokens)
        self.assertEqual(logits.shape, (2, 3, TINY["vocab_size"]))

    def test_single_token(self):
        model = _make_model()
        tokens = np.array([[5]])
        logits = model.forward(tokens)
        self.assertEqual(logits.shape, (1, 1, TINY["vocab_size"]))

    def test_various_lengths(self):
        model = _make_model()
        for L in [1, 5, 10, 32]:
            tokens = np.random.randint(0, TINY["vocab_size"], (2, L))
            logits = model.forward(tokens)
            self.assertEqual(logits.shape, (2, L, TINY["vocab_size"]))


class TestEmbeddingLookup(unittest.TestCase):

    def test_embedding_correctness(self):
        model = _make_model()
        tokens = np.array([[0, 1, 2]])
        x = model.embedding[tokens]
        np.testing.assert_array_equal(x[0, 0], model.embedding[0])
        np.testing.assert_array_equal(x[0, 1], model.embedding[1])
        np.testing.assert_array_equal(x[0, 2], model.embedding[2])


class TestWeightTying(unittest.TestCase):

    def test_tied_shares_memory(self):
        model = _make_model(tie_weights=True)
        self.assertTrue(np.shares_memory(model.W_out, model.embedding))

    def test_tied_is_transpose(self):
        model = _make_model(tie_weights=True)
        np.testing.assert_array_equal(model.W_out, model.embedding.T)

    def test_untied_independent(self):
        model = _make_model(tie_weights=False)
        self.assertFalse(np.shares_memory(model.W_out, model.embedding))
        self.assertEqual(model.W_out.shape, (TINY["d_model"], TINY["vocab_size"]))

    def test_tied_forward_pass(self):
        model = _make_model(tie_weights=True)
        tokens = np.array([[1, 2, 3]])
        logits = model.forward(tokens)
        self.assertEqual(logits.shape, (1, 3, TINY["vocab_size"]))
        self.assertTrue(np.all(np.isfinite(logits)))


class TestLogitsFinite(unittest.TestCase):

    def test_no_nan_inf(self):
        model = _make_model()
        tokens = np.random.randint(0, TINY["vocab_size"], (3, 10))
        logits = model.forward(tokens)
        self.assertTrue(np.all(np.isfinite(logits)))


class TestCausalProperty(unittest.TestCase):

    def test_future_token_independence(self):
        """Two sequences sharing prefix produce identical logits at shared positions."""
        model = _make_model()
        seq1 = np.array([[1, 2, 3, 4]])
        seq2 = np.array([[1, 2, 3, 7]])
        logits1 = model.forward(seq1)
        logits2 = model.forward(seq2)
        np.testing.assert_allclose(logits1[0, :3, :], logits2[0, :3, :], atol=1e-10)
        self.assertFalse(np.allclose(logits1[0, 3, :], logits2[0, 3, :]))

    def test_single_token_vs_full_sequence(self):
        """Logit at position 0 for [5] equals position 0 for [5, 8, 3, 1]."""
        model = _make_model()
        logits_short = model.forward(np.array([[5]]))
        logits_long = model.forward(np.array([[5, 8, 3, 1]]))
        np.testing.assert_allclose(logits_short[0, 0, :], logits_long[0, 0, :], atol=1e-10)

    def test_incremental_consistency(self):
        """Logits at position 2 for [1,2,3] equal position 2 for [1,2,3,99]."""
        model = _make_model()
        logits_a = model.forward(np.array([[1, 2, 3]]))
        logits_b = model.forward(np.array([[1, 2, 3, 99]]))
        np.testing.assert_allclose(logits_a[0, 2, :], logits_b[0, 2, :], atol=1e-10)


class TestTemperatureScale(unittest.TestCase):

    def test_identity_at_1(self):
        logits = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(temperature_scale(logits, 1.0), logits)

    def test_low_temperature_sharpens(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        probs_1 = softmax(temperature_scale(logits, 1.0))
        probs_low = softmax(temperature_scale(logits, 0.1))
        self.assertGreater(probs_low[0, 2], probs_1[0, 2])

    def test_high_temperature_flattens(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        probs_1 = softmax(temperature_scale(logits, 1.0))
        probs_high = softmax(temperature_scale(logits, 10.0))
        entropy_1 = -np.sum(probs_1 * np.log(probs_1 + 1e-12))
        entropy_high = -np.sum(probs_high * np.log(probs_high + 1e-12))
        self.assertGreater(entropy_high, entropy_1)

    def test_zero_temperature_passthrough(self):
        logits = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(temperature_scale(logits, 0.0), logits)


class TestTopKFilter(unittest.TestCase):

    def test_keeps_k_tokens(self):
        logits = np.array([[1.0, 5.0, 3.0, 2.0, 4.0]])
        filtered = top_k_filter(logits, k=3)
        probs = softmax(filtered)
        non_zero = np.sum(probs[0] > 1e-10)
        self.assertEqual(non_zero, 3)

    def test_preserves_top_tokens(self):
        logits = np.array([[1.0, 5.0, 3.0, 2.0, 4.0]])
        filtered = top_k_filter(logits, k=2)
        self.assertTrue(np.isfinite(filtered[0, 1]))
        self.assertTrue(np.isfinite(filtered[0, 4]))
        self.assertEqual(filtered[0, 0], -np.inf)

    def test_k_greater_than_vocab(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        filtered = top_k_filter(logits, k=10)
        np.testing.assert_array_equal(filtered, logits)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            top_k_filter(np.array([1.0, 2.0]), k=0)
        with self.assertRaises(ValueError):
            top_k_filter(np.array([1.0, 2.0]), k=-1)


class TestTopPFilter(unittest.TestCase):

    def test_cumulative_probability(self):
        logits = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
        filtered = top_p_filter(logits, p=0.9)
        probs = softmax(filtered)
        kept = probs[0][probs[0] > 1e-10]
        self.assertGreaterEqual(np.sum(kept), 0.9 - 1e-6)

    def test_minimality(self):
        """Removing the lowest-probability kept token should drop cumulative below p."""
        logits = np.array([[0.0, 1.0, 2.0, 3.0, 10.0]])
        p = 0.8
        filtered = top_p_filter(logits, p=p)
        original_probs = softmax(logits)
        kept_mask = np.isfinite(filtered[0])
        kept_probs = original_probs[0][kept_mask]

        if len(kept_probs) > 1:
            min_kept_prob = np.min(kept_probs)
            remaining = np.sum(kept_probs) - min_kept_prob
            self.assertLess(remaining, p + 1e-6)

    def test_p_equals_1(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        filtered = top_p_filter(logits, p=1.0)
        np.testing.assert_array_equal(filtered, logits)

    def test_invalid_p(self):
        with self.assertRaises(ValueError):
            top_p_filter(np.array([1.0, 2.0]), p=0.0)
        with self.assertRaises(ValueError):
            top_p_filter(np.array([1.0, 2.0]), p=1.5)


class TestSampleToken(unittest.TestCase):

    def test_greedy_determinism(self):
        logits = np.array([[1.0, 5.0, 3.0, 2.0]])
        token1 = sample_token(logits, greedy=True)
        token2 = sample_token(logits, greedy=True)
        self.assertEqual(token1[0], token2[0])
        self.assertEqual(token1[0], 1)

    def test_greedy_equals_argmax(self):
        logits = np.array([[1.0, 5.0, 3.0, 2.0]])
        greedy_token = sample_token(logits, greedy=True)
        np.testing.assert_array_equal(greedy_token, np.argmax(logits, axis=-1))

    def test_seed_reproducibility(self):
        logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        t1 = sample_token(logits, temperature=1.0, rng=rng1)
        t2 = sample_token(logits, temperature=1.0, rng=rng2)
        self.assertEqual(t1[0], t2[0])

    def test_different_seeds_diverge(self):
        logits = np.zeros((1, 1000))
        results = set()
        for seed in range(20):
            rng = np.random.RandomState(seed)
            t = sample_token(logits, temperature=1.0, rng=rng)
            results.add(int(t[0]))
        self.assertGreater(len(results), 1)

    def test_combined_sampling_valid_distribution(self):
        logits = np.random.randn(2, 50)
        rng = np.random.RandomState(0)
        tokens = sample_token(logits, temperature=0.8, top_k=10, top_p=0.9, rng=rng)
        self.assertEqual(tokens.shape, (2,))
        self.assertTrue(np.all(tokens >= 0))
        self.assertTrue(np.all(tokens < 50))

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            sample_token(np.array([1.0, 2.0]), temperature=-1.0)

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            sample_token(np.array([1.0, 2.0]), top_k=-1)

    def test_invalid_top_p(self):
        with self.assertRaises(ValueError):
            sample_token(np.array([1.0, 2.0]), top_p=0.0)
        with self.assertRaises(ValueError):
            sample_token(np.array([1.0, 2.0]), top_p=1.5)

    def test_1d_input(self):
        logits = np.array([1.0, 5.0, 3.0])
        token = sample_token(logits, greedy=True)
        self.assertEqual(token, 1)


class TestGenerationLoop(unittest.TestCase):

    def test_output_starts_with_prompt(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3, 4, 5]])
        output = model.generate(prompt, max_new_tokens=3, greedy=True)
        np.testing.assert_array_equal(output[0, :5], prompt[0])

    def test_max_tokens_respected(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3]])
        output = model.generate(prompt, max_new_tokens=5, greedy=True)
        self.assertEqual(output.shape[1], 3 + 5)

    def test_eos_stops_generation(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3]])
        output = model.generate(prompt, max_new_tokens=50, greedy=True, eos_token_id=-999)
        self.assertLessEqual(output.shape[1], 3 + 50)

    def test_greedy_generation_determinism(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3]])
        out1 = model.generate(prompt, max_new_tokens=5, greedy=True)
        out2 = model.generate(prompt, max_new_tokens=5, greedy=True)
        np.testing.assert_array_equal(out1, out2)

    def test_sequence_length_growth(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3, 4, 5]])
        for n in [1, 3, 7]:
            output = model.generate(prompt, max_new_tokens=n, greedy=True)
            self.assertEqual(output.shape[1], 5 + n)

    def test_seed_reproducibility(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3]])
        out1 = model.generate(prompt, max_new_tokens=5, temperature=1.0, seed=123)
        out2 = model.generate(prompt, max_new_tokens=5, temperature=1.0, seed=123)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_diverge(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3]])
        out1 = model.generate(prompt, max_new_tokens=10, temperature=1.0, seed=0)
        out2 = model.generate(prompt, max_new_tokens=10, temperature=1.0, seed=999)
        self.assertFalse(np.array_equal(out1, out2))

    def test_max_seq_len_respected(self):
        model = _make_model(max_seq_len=10)
        prompt = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        output = model.generate(prompt, max_new_tokens=100, greedy=True)
        self.assertLessEqual(output.shape[1], 10)

    def test_1d_prompt(self):
        model = _make_model()
        prompt = np.array([1, 2, 3])
        output = model.generate(prompt, max_new_tokens=3, greedy=True)
        self.assertEqual(output.ndim, 2)
        self.assertEqual(output.shape[1], 6)


class TestNumericalStability(unittest.TestCase):

    def test_large_vocab_softmax(self):
        logits = np.random.uniform(-100, 100, (2, 32000))
        probs = softmax(logits)
        self.assertTrue(np.all(np.isfinite(probs)))
        np.testing.assert_allclose(np.sum(probs, axis=-1), 1.0, atol=1e-6)

    def test_no_nan_in_forward(self):
        model = _make_model()
        tokens = np.random.randint(0, TINY["vocab_size"], (2, 20))
        logits = model.forward(tokens)
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_deep_model_stability(self):
        model = _make_model(num_layers=8)
        tokens = np.random.randint(0, TINY["vocab_size"], (1, 10))
        logits = model.forward(tokens)
        self.assertTrue(np.all(np.isfinite(logits)))


class TestWeightTyingParameterCount(unittest.TestCase):

    def test_tied_fewer_params(self):
        tied = count_model_parameters(**TINY_PARAM_KEYS, tie_weights=True)
        untied = count_model_parameters(**TINY_PARAM_KEYS, tie_weights=False)
        expected_diff = TINY["d_model"] * TINY["vocab_size"]
        self.assertEqual(untied["total"] - tied["total"], expected_diff)

    def test_tied_output_proj_zero(self):
        tied = count_model_parameters(**TINY_PARAM_KEYS, tie_weights=True)
        self.assertEqual(tied["output_proj"], 0)

    def test_untied_output_proj_correct(self):
        untied = count_model_parameters(**TINY_PARAM_KEYS, tie_weights=False)
        self.assertEqual(untied["output_proj"], TINY["d_model"] * TINY["vocab_size"])


class TestConfigurationValidation(unittest.TestCase):

    def test_invalid_token_ids(self):
        model = _make_model()
        with self.assertRaises(ValueError):
            model.forward(np.array([[999]]))
        with self.assertRaises(ValueError):
            model.forward(np.array([[-1]]))

    def test_sequence_exceeds_max_seq_len(self):
        model = _make_model(max_seq_len=5)
        with self.assertRaises(ValueError):
            model.forward(np.array([[1, 2, 3, 4, 5, 6]]))


class TestAnalysisLlama2(unittest.TestCase):

    def test_llama2_7b_params(self):
        params = count_model_parameters(
            vocab_size=32000, d_model=4096, num_layers=32,
            num_heads=32, num_kv_heads=32, d_ff=11008, tie_weights=True,
        )
        total_b = params["total"] / 1e9
        self.assertAlmostEqual(total_b, 6.607, delta=0.1)

    def test_llama2_blocks_dominate(self):
        params = count_model_parameters(
            vocab_size=32000, d_model=4096, num_layers=32,
            num_heads=32, num_kv_heads=32, d_ff=11008, tie_weights=True,
        )
        self.assertGreater(params["blocks_pct"], 90.0)


class TestGenerationFlops(unittest.TestCase):

    def test_naive_more_expensive_than_cached(self):
        kwargs = dict(
            prompt_len=50, num_new_tokens=100, num_layers=32,
            d_model=4096, num_heads=32, num_kv_heads=32,
            d_ff=11008, vocab_size=32000,
        )
        naive = generation_flops(**kwargs)
        cached = generation_flops_with_cache(**kwargs)
        self.assertGreater(naive["total"], cached["total"])

    def test_per_step_flop_growth(self):
        kwargs = dict(
            prompt_len=10, num_new_tokens=20, num_layers=2,
            d_model=64, num_heads=4, num_kv_heads=2,
            d_ff=128, vocab_size=100,
        )
        result = generation_flops(**kwargs)
        steps = result["per_step"]
        for i in range(1, len(steps)):
            self.assertGreater(steps[i], steps[i - 1])

    def test_cached_has_prefill_and_decode(self):
        kwargs = dict(
            prompt_len=10, num_new_tokens=5, num_layers=2,
            d_model=64, num_heads=4, num_kv_heads=2,
            d_ff=128, vocab_size=100,
        )
        result = generation_flops_with_cache(**kwargs)
        self.assertIn("prefill", result)
        self.assertIn("decode", result)
        self.assertAlmostEqual(
            result["total"], result["prefill"] + result["decode"]
        )


class TestIntegrationEndToEnd(unittest.TestCase):

    def test_tiny_model_generate(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3, 4, 5]])
        output = model.generate(prompt, max_new_tokens=10, greedy=True)
        self.assertEqual(output.shape, (1, 15))
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output < TINY["vocab_size"]))

    def test_batch_generation(self):
        model = _make_model()
        prompt = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        output = model.generate(prompt, max_new_tokens=5, greedy=True)
        self.assertEqual(output.shape, (4, 8))

    def test_batch_independence(self):
        """Each batch element should depend only on its own prompt."""
        model = _make_model()

        prompt_a = np.array([[1, 2, 3]])
        prompt_b = np.array([[4, 5, 6]])
        out_a = model.generate(prompt_a, max_new_tokens=5, greedy=True)
        out_b = model.generate(prompt_b, max_new_tokens=5, greedy=True)

        combined = np.array([[1, 2, 3], [4, 5, 6]])
        out_combined = model.generate(combined, max_new_tokens=5, greedy=True)

        np.testing.assert_array_equal(out_combined[0], out_a[0])
        np.testing.assert_array_equal(out_combined[1], out_b[0])

    def test_forward_determinism(self):
        model = _make_model()
        tokens = np.array([[1, 2, 3, 4, 5]])
        logits1 = model.forward(tokens)
        logits2 = model.forward(tokens)
        np.testing.assert_array_equal(logits1, logits2)


if __name__ == "__main__":
    unittest.main()
