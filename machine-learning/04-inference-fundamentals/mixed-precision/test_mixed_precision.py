"""Tests for mixed precision."""

import unittest

import numpy as np

from implementation import (
    float_info,
    visualize_bits,
    extract_fp32_parts,
    extract_fp16_parts,
    bf16_simulate,
    demonstrate_overflow,
    demonstrate_underflow,
    precision_loss,
    consecutive_value_gap,
    mixed_precision_linear,
    mixed_precision_matmul,
    stable_softmax,
    detect_overflow,
    LossScaler,
    dynamic_loss_scaling_demo,
    gradient_histogram,
    count_underflow,
    effective_zero_threshold,
    model_memory_usage,
    kv_cache_memory,
    memory_bandwidth_analysis,
)


class TestFloatInfo(unittest.TestCase):

    def test_fp32_format(self):
        info = float_info(np.float32)
        self.assertEqual(info["sign_bits"], 1)
        self.assertEqual(info["exponent_bits"], 8)
        self.assertEqual(info["mantissa_bits"], 23)
        self.assertEqual(info["bias"], 127)
        self.assertAlmostEqual(info["max_value"] / 3.4028235e+38, 1.0, places=5)

    def test_fp16_format(self):
        info = float_info(np.float16)
        self.assertEqual(info["sign_bits"], 1)
        self.assertEqual(info["exponent_bits"], 5)
        self.assertEqual(info["mantissa_bits"], 10)
        self.assertEqual(info["bias"], 15)
        self.assertAlmostEqual(info["max_value"], 65504.0, places=0)

    def test_bf16_format(self):
        info = float_info("bfloat16")
        self.assertEqual(info["exponent_bits"], 8)
        self.assertEqual(info["mantissa_bits"], 7)
        self.assertEqual(info["bias"], 127)
        self.assertAlmostEqual(info["max_value"], 3.4028235e+38, places=30)

    def test_precision_digits(self):
        fp32 = float_info(np.float32)
        fp16 = float_info(np.float16)
        bf16 = float_info("bfloat16")
        self.assertGreater(fp32["precision_decimal_digits"], 6.5)
        self.assertGreater(fp16["precision_decimal_digits"], 3.0)
        self.assertGreater(bf16["precision_decimal_digits"], 2.0)
        self.assertLess(bf16["precision_decimal_digits"], fp16["precision_decimal_digits"])


class TestVisualizeBits(unittest.TestCase):

    def test_fp32_one(self):
        bits_str = visualize_bits(1.0, np.float32)
        self.assertEqual(bits_str, "0 | 01111111 | 00000000000000000000000")

    def test_fp32_negative_one(self):
        bits_str = visualize_bits(-1.0, np.float32)
        self.assertEqual(bits_str, "1 | 01111111 | 00000000000000000000000")

    def test_fp32_two(self):
        bits_str = visualize_bits(2.0, np.float32)
        self.assertEqual(bits_str, "0 | 10000000 | 00000000000000000000000")

    def test_fp32_half(self):
        bits_str = visualize_bits(0.5, np.float32)
        self.assertEqual(bits_str, "0 | 01111110 | 00000000000000000000000")

    def test_fp16_one(self):
        bits_str = visualize_bits(1.0, np.float16)
        self.assertEqual(bits_str, "0 | 01111 | 0000000000")

    def test_fp16_negative_one(self):
        bits_str = visualize_bits(-1.0, np.float16)
        self.assertEqual(bits_str, "1 | 01111 | 0000000000")


class TestExtractParts(unittest.TestCase):

    def test_fp32_one(self):
        sign, exp, mant = extract_fp32_parts(1.0)
        self.assertEqual(sign, 0)
        self.assertEqual(exp, 127)
        self.assertEqual(mant, 0)

    def test_fp32_negative(self):
        sign, exp, mant = extract_fp32_parts(-1.0)
        self.assertEqual(sign, 1)
        self.assertEqual(exp, 127)
        self.assertEqual(mant, 0)

    def test_fp32_two(self):
        sign, exp, mant = extract_fp32_parts(2.0)
        self.assertEqual(sign, 0)
        self.assertEqual(exp, 128)
        self.assertEqual(mant, 0)

    def test_fp16_one(self):
        sign, exp, mant = extract_fp16_parts(1.0)
        self.assertEqual(sign, 0)
        self.assertEqual(exp, 15)
        self.assertEqual(mant, 0)

    def test_fp16_special_inf(self):
        sign, exp, mant = extract_fp16_parts(np.inf)
        self.assertEqual(exp, 31)
        self.assertEqual(mant, 0)

    def test_fp16_special_neg_inf(self):
        sign, exp, mant = extract_fp16_parts(-np.inf)
        self.assertEqual(sign, 1)
        self.assertEqual(exp, 31)
        self.assertEqual(mant, 0)

    def test_fp16_special_nan(self):
        _, exp, mant = extract_fp16_parts(np.nan)
        self.assertEqual(exp, 31)
        self.assertNotEqual(mant, 0)

    def test_fp16_zero(self):
        sign, exp, mant = extract_fp16_parts(0.0)
        self.assertEqual(sign, 0)
        self.assertEqual(exp, 0)
        self.assertEqual(mant, 0)

    def test_fp16_negative_zero(self):
        sign, exp, mant = extract_fp16_parts(-0.0)
        self.assertEqual(sign, 1)
        self.assertEqual(exp, 0)
        self.assertEqual(mant, 0)


class TestBF16Simulate(unittest.TestCase):

    def test_exact_values(self):
        """Values exactly representable in BF16 should not change."""
        values = np.array([1.0, 2.0, -1.0, 0.0, 0.5], dtype=np.float32)
        result = bf16_simulate(values)
        np.testing.assert_array_equal(result, values)

    def test_truncation(self):
        """BF16 should truncate lower mantissa bits."""
        val = np.float32(1.0009765625)  # 1 + 2^-10, has bit in position 10
        result = bf16_simulate(np.array([val]))
        self.assertNotEqual(float(result[0]), float(val))
        # BF16 keeps only 7 mantissa bits, so 2^-10 contribution is lost
        self.assertEqual(float(result[0]), 1.0)

    def test_preserves_range(self):
        """BF16 has same range as FP32, so large values don't overflow."""
        large = np.array([65536.0, 100000.0, 1e30], dtype=np.float32)
        result = bf16_simulate(large)
        self.assertFalse(np.any(np.isinf(result)))
        for i in range(len(large)):
            self.assertGreater(float(result[i]), 0)

    def test_roundtrip_consistency(self):
        """Applying bf16_simulate twice should be idempotent."""
        x = np.random.randn(100).astype(np.float32)
        once = bf16_simulate(x)
        twice = bf16_simulate(once)
        np.testing.assert_array_equal(once, twice)

    def test_conversion_roundtrip_fp16(self):
        """FP32 -> FP16 -> FP32 roundtrip for values in valid range."""
        values = np.array([0.1, 0.5, 1.0, 100.0, -42.5], dtype=np.float32)
        fp16 = values.astype(np.float16)
        back = fp16.astype(np.float32)
        for v, b in zip(values, back):
            rel_err = abs(float(v) - float(b)) / abs(float(v)) if abs(float(v)) > 0 else 0
            self.assertLess(rel_err, 0.01)


class TestOverflow(unittest.TestCase):

    def test_fp16_max_representable(self):
        """65504 is the max finite FP16 value."""
        self.assertEqual(float(np.float16(65504)), 65504.0)
        self.assertFalse(np.isinf(np.float16(65504)))

    def test_fp16_overflow_65520(self):
        """65520 overflows to inf in FP16."""
        self.assertTrue(np.isinf(np.float16(65520)))

    def test_fp16_gradual_overflow(self):
        """Values up to 65519 round to 65504; 65520+ overflow to inf."""
        values = [65000.0, 65500.0, 65504.0, 65519.0, 65520.0, 65536.0]
        expected_inf = [False, False, False, False, True, True]
        for v, should_inf in zip(values, expected_inf):
            result = np.float16(v)
            self.assertEqual(
                bool(np.isinf(result)), should_inf,
                f"FP16({v}): expected inf={should_inf}, got {float(result)}"
            )

    def test_bf16_no_overflow(self):
        """Values that overflow FP16 are fine in BF16."""
        large_values = np.array([65505.0, 65536.0, 100000.0], dtype=np.float32)
        bf16_vals = bf16_simulate(large_values)
        self.assertFalse(np.any(np.isinf(bf16_vals)))

    def test_inf_propagation(self):
        self.assertTrue(np.isinf(np.float16(np.inf) + np.float16(1.0)))
        self.assertTrue(np.isnan(np.float16(np.inf) - np.float16(np.inf)))

    def test_demonstrate_overflow(self):
        result = demonstrate_overflow(np.float16)
        self.assertFalse(result["values"][65504.0]["is_inf"])
        self.assertTrue(result["values"][65536.0]["is_inf"])


class TestUnderflow(unittest.TestCase):

    def test_fp16_underflow_threshold(self):
        """Values below ~5.96e-8 underflow to zero in FP16."""
        self.assertEqual(float(np.float16(1e-10)), 0.0)
        self.assertEqual(float(np.float16(1e-9)), 0.0)

    def test_fp16_subnormal_range(self):
        """Values in subnormal range are non-zero but have reduced precision."""
        val = 6.0e-5
        fp16_val = np.float16(val)
        self.assertNotEqual(float(fp16_val), 0.0)

    def test_gradient_underflow(self):
        """Small gradients underflow to zero in FP16."""
        grads = np.array([1e-6, 1e-7, 1e-8], dtype=np.float32)
        fp16_grads = grads.astype(np.float16)
        # 1e-8 should underflow
        self.assertEqual(float(fp16_grads[2]), 0.0)
        # 1e-6 should survive (it's in subnormal range but still nonzero)
        self.assertNotEqual(float(fp16_grads[0]), 0.0)

    def test_accumulation_underflow(self):
        """Summing many small FP16 values loses precision vs FP32 accumulation."""
        values = np.full(1000, 0.001, dtype=np.float32)
        fp16_sum = np.float16(0)
        for v in values:
            fp16_sum = np.float16(float(fp16_sum) + float(np.float16(v)))
        fp32_sum = np.sum(values)
        self.assertAlmostEqual(float(fp32_sum), 1.0, places=5)
        # FP16 accumulation should have noticeable error
        self.assertGreater(abs(float(fp16_sum) - 1.0), abs(float(fp32_sum) - 1.0))

    def test_demonstrate_underflow(self):
        result = demonstrate_underflow(np.float16)
        self.assertTrue(result["values"][1e-10]["is_zero"])
        self.assertFalse(result["values"][1e-4]["is_zero"])


class TestPrecisionLoss(unittest.TestCase):

    def test_relative_error_fp16(self):
        rng = np.random.RandomState(0)
        x = rng.randn(100).astype(np.float32)
        errors = precision_loss(x, np.float16)
        self.assertEqual(errors.shape, (100,))
        mean_err = np.mean(errors[np.abs(x) > 0.01])
        self.assertLess(mean_err, 0.01)

    def test_large_value_precision(self):
        """At value 1000, FP16 gap is 0.5 (exponent=9, mantissa=10, gap=2^(9-10))."""
        gap = consecutive_value_gap(1000.0, np.float16)
        self.assertAlmostEqual(gap, 0.5, places=3)

    def test_bf16_less_precise(self):
        """BF16 has lower precision (more error) than FP16 for values in FP16 range."""
        x = np.array([0.1, 0.5, 1.5, 3.14], dtype=np.float32)
        err_fp16 = precision_loss(x, np.float16)
        err_bf16 = precision_loss(x, "bfloat16")
        self.assertGreater(np.mean(err_bf16), np.mean(err_fp16))

    def test_bf16_vs_fp16_range(self):
        """BF16 has same range as FP32 (no overflow for large values)."""
        x = np.array([100000.0], dtype=np.float32)
        err_bf16 = precision_loss(x, "bfloat16")
        self.assertTrue(np.isfinite(err_bf16[0]))

    def test_zero_input(self):
        x = np.array([0.0], dtype=np.float32)
        err = precision_loss(x, np.float16)
        self.assertEqual(float(err[0]), 0.0)

    def test_negative_symmetric(self):
        """Precision loss is symmetric for positive and negative values."""
        x = np.array([3.14], dtype=np.float32)
        err_pos = precision_loss(x, np.float16)
        err_neg = precision_loss(-x, np.float16)
        np.testing.assert_allclose(err_pos, err_neg, atol=1e-10)


class TestConsecutiveValueGap(unittest.TestCase):

    def test_gap_at_one_fp16(self):
        gap = consecutive_value_gap(1.0, np.float16)
        expected = 2 ** -10  # ~9.77e-4
        self.assertAlmostEqual(gap, expected, places=6)

    def test_gap_at_one_fp32(self):
        gap = consecutive_value_gap(1.0, np.float32)
        expected = 2 ** -23  # ~1.19e-7
        self.assertAlmostEqual(gap, expected, places=12)

    def test_gap_at_one_bf16(self):
        gap = consecutive_value_gap(1.0, "bfloat16")
        expected = 2 ** -7  # ~7.81e-3
        self.assertAlmostEqual(gap, expected, places=5)

    def test_gap_doubles_with_value(self):
        """Gap doubles when value doubles (same exponent region)."""
        gap_1 = consecutive_value_gap(1.0, np.float16)
        gap_2 = consecutive_value_gap(2.0, np.float16)
        self.assertAlmostEqual(gap_2 / gap_1, 2.0, places=3)

    def test_integer_representability_fp16(self):
        """All integers up to 2048 are exactly representable in FP16."""
        for i in range(1, 2049):
            fp16_val = np.float16(float(i))
            self.assertEqual(float(fp16_val), float(i), f"Integer {i} not exact in FP16")
        # 2049 should NOT be exactly representable
        self.assertNotEqual(float(np.float16(2049.0)), 2049.0)


class TestStableSoftmax(unittest.TestCase):

    def test_uniform_input(self):
        """softmax([0, 0, 0]) = [1/3, 1/3, 1/3]"""
        x = np.zeros(3, dtype=np.float32)
        result = stable_softmax(x, np.float16)
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_allclose(result.astype(np.float32), expected, atol=0.01)

    def test_single_element(self):
        """Softmax of single element is 1.0."""
        x = np.array([5.0], dtype=np.float32)
        result = stable_softmax(x, np.float16)
        self.assertAlmostEqual(float(result[0]), 1.0, places=2)

    def test_sums_to_one(self):
        x = np.random.randn(10).astype(np.float32)
        result = stable_softmax(x, np.float16)
        self.assertAlmostEqual(float(np.sum(result.astype(np.float32))), 1.0, places=2)

    def test_large_logits_no_overflow(self):
        """Large logits that would overflow naive FP16 exp are handled."""
        x = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        result = stable_softmax(x, np.float16)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        np.testing.assert_allclose(result.astype(np.float32), [1/3, 1/3, 1/3], atol=0.01)

    def test_large_logits_naive_overflows(self):
        """Without max subtraction, exp of large FP16 values overflows."""
        x = np.array([20.0], dtype=np.float16)
        # exp(20) ~ 4.85e8 >> 65504, so it overflows in FP16
        self.assertTrue(np.isinf(np.exp(x)))

    def test_batch_softmax(self):
        x = np.random.randn(4, 10).astype(np.float32)
        result = stable_softmax(x, np.float16)
        self.assertEqual(result.shape, (4, 10))
        sums = np.sum(result.astype(np.float32), axis=-1)
        np.testing.assert_allclose(sums, np.ones(4), atol=0.02)


class TestDetectOverflow(unittest.TestCase):

    def test_no_overflow(self):
        x = np.array([1.0, 2.0, 3.0])
        self.assertFalse(detect_overflow(x))

    def test_inf(self):
        x = np.array([1.0, np.inf, 3.0])
        self.assertTrue(detect_overflow(x))

    def test_nan(self):
        x = np.array([1.0, np.nan, 3.0])
        self.assertTrue(detect_overflow(x))

    def test_neg_inf(self):
        x = np.array([1.0, -np.inf, 3.0])
        self.assertTrue(detect_overflow(x))

    def test_empty(self):
        x = np.array([])
        self.assertFalse(detect_overflow(x))


class TestLossScaler(unittest.TestCase):

    def test_initial_scale(self):
        scaler = LossScaler(init_scale=1024.0)
        self.assertEqual(scaler.get_scale(), 1024.0)

    def test_scale_loss(self):
        scaler = LossScaler(init_scale=1024.0)
        loss = np.array(0.5, dtype=np.float32)
        scaled = scaler.scale_loss(loss)
        self.assertAlmostEqual(float(scaled), 512.0)

    def test_unscale_grads(self):
        scaler = LossScaler(init_scale=1024.0)
        grads = np.array([512.0, 1024.0], dtype=np.float16)
        unscaled = scaler.unscale_grads(grads)
        np.testing.assert_allclose(unscaled, [0.5, 1.0])

    def test_scale_preserves_gradients(self):
        """Gradient 1e-6 * scale 1e4 = 1e-2, survives FP16."""
        scale = 1e4
        grad = 1e-6
        scaled_grad = np.float16(grad * scale)
        self.assertNotEqual(float(scaled_grad), 0.0)
        recovered = float(scaled_grad) / scale
        self.assertAlmostEqual(recovered, grad, places=8)

    def test_overflow_reduces_scale(self):
        scaler = LossScaler(init_scale=1024.0)
        scaler.update(has_overflow=True)
        self.assertEqual(scaler.get_scale(), 512.0)

    def test_repeated_overflow(self):
        scaler = LossScaler(init_scale=1024.0, min_scale=1.0)
        for _ in range(20):
            scaler.update(has_overflow=True)
        self.assertGreaterEqual(scaler.get_scale(), 1.0)

    def test_growth_after_interval(self):
        scaler = LossScaler(init_scale=100.0, growth_interval=5, growth_factor=2.0)
        for _ in range(5):
            scaler.update(has_overflow=False)
        self.assertEqual(scaler.get_scale(), 200.0)

    def test_no_growth_before_interval(self):
        scaler = LossScaler(init_scale=100.0, growth_interval=5)
        for _ in range(4):
            scaler.update(has_overflow=False)
        self.assertEqual(scaler.get_scale(), 100.0)

    def test_overflow_resets_counter(self):
        scaler = LossScaler(init_scale=100.0, growth_interval=5)
        for _ in range(4):
            scaler.update(has_overflow=False)
        scaler.update(has_overflow=True)
        # Counter should be reset, scale halved
        self.assertEqual(scaler.get_scale(), 50.0)
        for _ in range(4):
            scaler.update(has_overflow=False)
        # Still at 50 because we haven't completed 5 clean steps
        self.assertEqual(scaler.get_scale(), 50.0)

    def test_no_nan_after_overflow_skip(self):
        """After detecting overflow and skipping, weights remain valid."""
        W = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grads = np.array([np.inf, 1.0, 2.0], dtype=np.float32)
        if detect_overflow(grads):
            pass  # skip update
        else:
            W -= 0.01 * grads
        self.assertFalse(detect_overflow(W))


class TestMixedPrecisionMatmul(unittest.TestCase):

    def test_basic_matmul(self):
        A = np.eye(3, dtype=np.float32)
        B = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        result = mixed_precision_matmul(A, B)
        np.testing.assert_allclose(result.astype(np.float32), B, atol=0.1)

    def test_fp32_accumulation_accuracy(self):
        """FP32 accumulation should match FP32 baseline closely."""
        rng = np.random.RandomState(42)
        A = rng.randn(32, 64).astype(np.float32)
        B = rng.randn(64, 32).astype(np.float32)

        fp32_result = A @ B
        mixed_result = mixed_precision_matmul(A, B, np.float16, np.float32)

        rel_err = np.mean(np.abs(mixed_result.astype(np.float32) - fp32_result)) / np.mean(np.abs(fp32_result))
        self.assertLess(rel_err, 0.01)

    def test_pure_fp16_accumulation_worse(self):
        """Pure FP16 accumulation has more error than FP32 accumulation."""
        rng = np.random.RandomState(42)
        A = rng.randn(64, 128).astype(np.float32)
        B = rng.randn(128, 64).astype(np.float32)

        fp32_ref = A @ B
        fp32_accum = mixed_precision_matmul(A, B, np.float16, np.float32)
        fp16_accum = mixed_precision_matmul(A, B, np.float16, np.float16)

        err_fp32 = np.mean(np.abs(fp32_accum.astype(np.float32) - fp32_ref))
        err_fp16 = np.mean(np.abs(fp16_accum.astype(np.float32) - fp32_ref))

        self.assertGreater(err_fp16, err_fp32)

    def test_error_grows_with_matrix_size(self):
        """Error in pure FP16 matmul grows with inner dimension."""
        rng = np.random.RandomState(42)
        errors = []
        for K in [16, 64, 256]:
            A = rng.randn(8, K).astype(np.float32)
            B = rng.randn(K, 8).astype(np.float32)
            fp32_ref = A @ B
            fp16_result = mixed_precision_matmul(A, B, np.float16, np.float16)
            err = np.mean(np.abs(fp16_result.astype(np.float32) - fp32_ref))
            errors.append(err)
        # Error should generally increase with K
        self.assertGreater(errors[-1], errors[0])

    def test_output_dtype(self):
        A = np.ones((2, 3), dtype=np.float32)
        B = np.ones((3, 4), dtype=np.float32)
        result = mixed_precision_matmul(A, B, np.float16, np.float32)
        self.assertEqual(result.dtype, np.float16)

    def test_batched_fp16_accumulation(self):
        """Pure FP16 accumulation path handles batched inputs correctly."""
        rng = np.random.RandomState(42)
        A = rng.randn(2, 3, 4).astype(np.float32)
        B = rng.randn(2, 4, 5).astype(np.float32)
        result = mixed_precision_matmul(A, B, np.float16, np.float16)
        self.assertEqual(result.shape, (2, 3, 5))
        self.assertEqual(result.dtype, np.float16)


class TestMixedPrecisionLinear(unittest.TestCase):

    def test_identity_weight(self):
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        W = np.eye(3, dtype=np.float32)
        result = mixed_precision_linear(x, W)
        np.testing.assert_allclose(result.astype(np.float32), x, atol=0.01)

    def test_with_bias(self):
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        W = np.eye(2, dtype=np.float32)
        b = np.array([10.0, 20.0], dtype=np.float32)
        result = mixed_precision_linear(x, W, b)
        np.testing.assert_allclose(result.astype(np.float32), [[11.0, 22.0]], atol=0.1)

    def test_output_dtype(self):
        x = np.ones((2, 4), dtype=np.float32)
        W = np.ones((4, 3), dtype=np.float32)
        result = mixed_precision_linear(x, W, compute_dtype=np.float16)
        self.assertEqual(result.dtype, np.float16)


class TestDynamicLossScalingDemo(unittest.TestCase):

    def test_runs_without_error(self):
        history = dynamic_loss_scaling_demo(n_steps=10)
        self.assertEqual(len(history["step"]), 10)
        self.assertEqual(len(history["loss"]), 10)
        self.assertEqual(len(history["scale"]), 10)

    def test_losses_finite(self):
        history = dynamic_loss_scaling_demo(n_steps=10)
        for loss in history["loss"]:
            self.assertTrue(np.isfinite(loss))


class TestGradientHistogram(unittest.TestCase):

    def test_basic_histogram(self):
        rng = np.random.RandomState(42)
        grads = rng.randn(1000).astype(np.float32) * 0.01
        result = gradient_histogram(grads)
        self.assertIn("bin_edges", result)
        self.assertIn("counts", result)
        self.assertEqual(np.sum(result["counts"]), np.sum(np.abs(grads.ravel()) > 0))

    def test_all_zeros(self):
        grads = np.zeros(100, dtype=np.float32)
        result = gradient_histogram(grads)
        self.assertEqual(result["num_zeros"], 100)
        self.assertEqual(len(result["bin_edges"]), 0)

    def test_typical_distribution(self):
        """Most gradients in a typical network are small."""
        rng = np.random.RandomState(42)
        grads = rng.randn(10000).astype(np.float32) * 0.001
        result = gradient_histogram(grads)
        self.assertLess(result["mean_magnitude"], 0.01)


class TestCountUnderflow(unittest.TestCase):

    def test_no_underflow(self):
        grads = np.array([0.1, 0.5, 1.0], dtype=np.float32)
        result = count_underflow(grads, np.float16)
        self.assertEqual(result["underflow_count"], 0)

    def test_some_underflow(self):
        grads = np.array([0.1, 1e-6, 1e-8], dtype=np.float32)
        result = count_underflow(grads, np.float16)
        self.assertGreater(result["underflow_count"], 0)

    def test_underflow_with_scaling(self):
        """With loss scaling, gradients that would underflow survive."""
        rng = np.random.RandomState(42)
        grads = rng.randn(1000).astype(np.float32) * 1e-6
        result_no_scale = count_underflow(grads, np.float16)
        scaled_grads = grads * 1e4
        result_scaled = count_underflow(scaled_grads, np.float16)
        self.assertLessEqual(
            result_scaled["underflow_count"],
            result_no_scale["underflow_count"],
        )


class TestEffectiveZeroThreshold(unittest.TestCase):

    def test_fp16_threshold(self):
        threshold = effective_zero_threshold(np.float16)
        self.assertGreater(threshold, 0)
        # Threshold itself survives FP16 conversion
        self.assertGreater(float(np.float16(threshold)), 0.0)
        # Half the threshold underflows to zero
        self.assertEqual(float(np.float16(threshold / 2)), 0.0)

    def test_fp32_threshold_smaller(self):
        fp16_t = effective_zero_threshold(np.float16)
        fp32_t = effective_zero_threshold(np.float32)
        self.assertLess(fp32_t, fp16_t)


class TestModelMemoryUsage(unittest.TestCase):

    def test_7b_fp16(self):
        """7B params in FP16 = 14 GB."""
        result = model_memory_usage(7_000_000_000, np.float16)
        self.assertAlmostEqual(result["gigabytes"], 14_000_000_000 / (1024**3), places=1)

    def test_7b_fp32(self):
        """7B params in FP32 = 28 GB."""
        result = model_memory_usage(7_000_000_000, np.float32)
        self.assertAlmostEqual(result["gigabytes"], 28_000_000_000 / (1024**3), places=1)

    def test_fp32_is_2x_fp16(self):
        """FP32 model is exactly 2x FP16 model size."""
        fp16 = model_memory_usage(1_000_000, np.float16)
        fp32 = model_memory_usage(1_000_000, np.float32)
        self.assertAlmostEqual(fp32["bytes"] / fp16["bytes"], 2.0)

    def test_int8(self):
        result = model_memory_usage(1_000_000, "int8")
        self.assertEqual(result["bytes"], 1_000_000)

    def test_int4(self):
        result = model_memory_usage(1_000_000, "int4")
        self.assertEqual(result["bytes"], 500_000)


class TestKVCacheMemory(unittest.TestCase):

    def test_known_architecture(self):
        """7B model (32 layers, 32 heads, 128 d_head) in FP16."""
        result = kv_cache_memory(
            n_layers=32, n_heads=32, d_head=128, seq_len=1, dtype=np.float16,
        )
        # Per token: 2 * 32 * 32 * 128 * 2 = 524288 bytes = 0.5 MB
        self.assertEqual(result["per_token_bytes"], 524288)

    def test_4k_context(self):
        result = kv_cache_memory(
            n_layers=32, n_heads=32, d_head=128, seq_len=4096, dtype=np.float16,
        )
        expected_gb = (524288 * 4096) / (1024 ** 3)
        self.assertAlmostEqual(result["gigabytes"], expected_gb, places=2)

    def test_fp32_vs_fp16(self):
        fp16 = kv_cache_memory(n_layers=32, n_heads=32, d_head=128, seq_len=100, dtype=np.float16)
        fp32 = kv_cache_memory(n_layers=32, n_heads=32, d_head=128, seq_len=100, dtype=np.float32)
        self.assertAlmostEqual(fp32["bytes"] / fp16["bytes"], 2.0)


class TestMemoryBandwidthAnalysis(unittest.TestCase):

    def test_basic_analysis(self):
        result = memory_bandwidth_analysis(batch_size=1, seq_len=1, d_model=4096)
        self.assertIn("fp32", result)
        self.assertIn("fp16", result)
        self.assertIn("memory_savings_ratio", result)

    def test_fp16_higher_intensity(self):
        """FP16 should have higher arithmetic intensity (same FLOPs, fewer bytes)."""
        result = memory_bandwidth_analysis(batch_size=1, seq_len=1, d_model=4096)
        self.assertGreater(
            result["fp16"]["arithmetic_intensity"],
            result["fp32"]["arithmetic_intensity"],
        )

    def test_savings_ratio(self):
        result = memory_bandwidth_analysis(batch_size=1, seq_len=1, d_model=4096)
        self.assertGreater(result["memory_savings_ratio"], 1.5)


class TestEdgeCases(unittest.TestCase):

    def test_zero_input_precision_loss(self):
        x = np.array([0.0], dtype=np.float32)
        err = precision_loss(x, np.float16)
        self.assertEqual(float(err[0]), 0.0)

    def test_negative_values_precision(self):
        x = np.array([-1.5, -100.0], dtype=np.float32)
        err = precision_loss(x, np.float16)
        self.assertTrue(np.all(np.isfinite(err)))

    def test_empty_array_softmax(self):
        x = np.array([], dtype=np.float32).reshape(0, 5)
        result = stable_softmax(x, np.float16)
        self.assertEqual(result.shape, (0, 5))

    def test_bf16_simulate_empty(self):
        x = np.array([], dtype=np.float32)
        result = bf16_simulate(x)
        self.assertEqual(result.shape, (0,))

    def test_mixed_matmul_empty(self):
        A = np.zeros((0, 3), dtype=np.float32)
        B = np.zeros((3, 4), dtype=np.float32)
        result = mixed_precision_matmul(A, B)
        self.assertEqual(result.shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
