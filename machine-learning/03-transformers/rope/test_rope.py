"""Tests for Rotary Position Embeddings (RoPE)."""

import unittest
import numpy as np
from implementation import (
    precompute_freqs,
    rotate_half,
    rotate_half_backward,
    apply_rope,
    apply_rope_complex,
    RoPE,
    verify_relative_position_property,
    rotation_is_orthogonal,
    compare_with_sinusoidal,
)


class TestPrecomputeFreqs(unittest.TestCase):

    def test_output_shapes(self):
        for d in [2, 4, 8, 64, 128]:
            cos, sin = precompute_freqs(d, 256)
            self.assertEqual(cos.shape, (256, d // 2))
            self.assertEqual(sin.shape, (256, d // 2))

    def test_odd_d_raises(self):
        with self.assertRaises(ValueError):
            precompute_freqs(63, 100)

    def test_variable_seq_lengths(self):
        for L in [1, 16, 128, 4096]:
            cos, sin = precompute_freqs(8, L)
            self.assertEqual(cos.shape, (L, 4))

    def test_position_zero_is_identity(self):
        cos, sin = precompute_freqs(8, 10)
        np.testing.assert_allclose(cos[0], 1.0, atol=1e-15)
        np.testing.assert_allclose(sin[0], 0.0, atol=1e-15)

    def test_geometric_progression(self):
        d = 128
        cos, sin = precompute_freqs(d, 10)
        i = np.arange(d // 2, dtype=np.float64)
        expected_freqs = np.exp(-2.0 * i / d * np.log(10000.0))
        ratio = expected_freqs[1:] / expected_freqs[:-1]
        np.testing.assert_allclose(ratio, ratio[0], atol=1e-12)

    def test_custom_theta_base(self):
        cos_10k, sin_10k = precompute_freqs(64, 100, theta_base=10000.0)
        cos_500k, sin_500k = precompute_freqs(64, 100, theta_base=500000.0)
        angles_10k = np.arccos(np.clip(cos_10k[1], -1, 1))
        angles_500k = np.arccos(np.clip(cos_500k[1], -1, 1))
        self.assertTrue(np.all(angles_500k <= angles_10k + 1e-12))

    def test_log_space_matches_direct(self):
        d = 128
        i = np.arange(d // 2, dtype=np.float64)
        log_space = np.exp(-2.0 * i / d * np.log(10000.0))
        direct = 10000.0 ** (-2.0 * i / d)
        np.testing.assert_allclose(log_space, direct, atol=1e-12)

    def test_large_seq_len_no_nan(self):
        cos, sin = precompute_freqs(128, 100000)
        self.assertFalse(np.any(np.isnan(cos)))
        self.assertFalse(np.any(np.isnan(sin)))
        self.assertFalse(np.any(np.isinf(cos)))
        self.assertFalse(np.any(np.isinf(sin)))

    def test_large_d_no_nan(self):
        cos, sin = precompute_freqs(256, 1000)
        self.assertFalse(np.any(np.isnan(cos)))
        self.assertFalse(np.any(np.isinf(cos)))


class TestRotateHalf(unittest.TestCase):

    def test_basic_2d(self):
        x = np.array([1.0, 2.0])
        result = rotate_half(x)
        np.testing.assert_array_equal(result, [-2.0, 1.0])

    def test_basic_4d(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = rotate_half(x)
        np.testing.assert_array_equal(result, [-2.0, 1.0, -4.0, 3.0])

    def test_batched(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = rotate_half(x)
        expected = np.array([[-2.0, 1.0, -4.0, 3.0], [-6.0, 5.0, -8.0, 7.0]])
        np.testing.assert_array_equal(result, expected)

    def test_4d_tensor(self):
        x = np.random.randn(2, 4, 8, 64)
        result = rotate_half(x)
        self.assertEqual(result.shape, x.shape)

    def test_rotate_half_backward_is_inverse_transpose(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        fwd = rotate_half(x)
        bwd = rotate_half_backward(x)
        # forward: (-x1, x0, -x3, x2), backward: (x1, -x0, x3, -x2)
        np.testing.assert_array_equal(fwd, [-2.0, 1.0, -4.0, 3.0])
        np.testing.assert_array_equal(bwd, [2.0, -1.0, 4.0, -3.0])

    def test_rotate_half_twice_negates(self):
        x = np.random.randn(4, 8)
        result = rotate_half(rotate_half(x))
        np.testing.assert_allclose(result, -x, atol=1e-15)


class TestApplyRope(unittest.TestCase):

    def test_output_shape(self):
        for B, H, L, d in [(1, 1, 1, 2), (2, 4, 16, 64), (1, 8, 128, 128)]:
            x = np.random.randn(B, H, L, d)
            cos, sin = precompute_freqs(d, L)
            result = apply_rope(x, cos, sin)
            self.assertEqual(result.shape, (B, H, L, d))

    def test_position_zero_identity(self):
        x = np.random.randn(2, 4, 1, 64)
        cos, sin = precompute_freqs(64, 10)
        positions = np.array([0])
        result = apply_rope(x, cos, sin, positions)
        np.testing.assert_allclose(result, x, atol=1e-14)

    def test_norm_preservation(self):
        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 64)
        cos, sin = precompute_freqs(64, 100)

        for pos in [0, 1, 50, 99]:
            positions = np.array([pos])
            x_single = x[:, :, :1, :]
            result = apply_rope(x_single, cos, sin, positions)
            orig_norms = np.linalg.norm(x_single, axis=-1)
            rot_norms = np.linalg.norm(result, axis=-1)
            np.testing.assert_allclose(rot_norms, orig_norms, atol=1e-12)

    def test_custom_positions(self):
        x = np.random.randn(1, 1, 3, 8)
        cos, sin = precompute_freqs(8, 100)
        positions = np.array([5, 10, 20])
        result = apply_rope(x, cos, sin, positions)
        self.assertEqual(result.shape, x.shape)

    def test_minimal_d2_example(self):
        """d=2, position 1: theta_0=1.0, cos(1)~0.5403, sin(1)~0.8415."""
        cos, sin = precompute_freqs(2, 3, theta_base=10000.0)

        np.testing.assert_allclose(cos[0, 0], 1.0, atol=1e-14)
        np.testing.assert_allclose(sin[0, 0], 0.0, atol=1e-14)

        x = np.array([[[[1.0, 0.0]]]])  # (1,1,1,2)
        positions = np.array([1])
        result = apply_rope(x, cos, sin, positions)

        expected_cos = np.cos(1.0)
        expected_sin = np.sin(1.0)
        np.testing.assert_allclose(result[0, 0, 0, 0], expected_cos, atol=1e-10)
        np.testing.assert_allclose(result[0, 0, 0, 1], expected_sin, atol=1e-10)

        norm_result = np.linalg.norm(result)
        np.testing.assert_allclose(norm_result, 1.0, atol=1e-12)

    def test_d4_hand_computed(self):
        """d=4, position 2: two independent rotation pairs."""
        d = 4
        cos, sin = precompute_freqs(d, 10, theta_base=10000.0)

        i = np.arange(2, dtype=np.float64)
        inv_freq = np.exp(-2.0 * i / d * np.log(10000.0))
        angles = 2.0 * inv_freq

        q = np.array([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1,1,1,4)
        positions = np.array([2])
        result = apply_rope(q, cos, sin, positions)

        expected = np.zeros(4)
        expected[0] = q[0, 0, 0, 0] * np.cos(angles[0]) - q[0, 0, 0, 1] * np.sin(angles[0])
        expected[1] = q[0, 0, 0, 0] * np.sin(angles[0]) + q[0, 0, 0, 1] * np.cos(angles[0])
        expected[2] = q[0, 0, 0, 2] * np.cos(angles[1]) - q[0, 0, 0, 3] * np.sin(angles[1])
        expected[3] = q[0, 0, 0, 2] * np.sin(angles[1]) + q[0, 0, 0, 3] * np.cos(angles[1])

        np.testing.assert_allclose(result[0, 0, 0], expected, atol=1e-12)


class TestApplyRopeComplex(unittest.TestCase):

    def test_matches_rotate_half(self):
        np.random.seed(123)
        d = 64
        B, H, L = 2, 4, 16
        x = np.random.randn(B, H, L, d)

        cos_cache, sin_cache = precompute_freqs(d, L)
        result_rh = apply_rope(x, cos_cache, sin_cache)

        i = np.arange(d // 2, dtype=np.float64)
        inv_freq = np.exp(-2.0 * i / d * np.log(10000.0))
        positions = np.arange(L, dtype=np.float64)
        angles = positions[:, np.newaxis] * inv_freq[np.newaxis, :]
        freqs_complex = np.exp(1j * angles)  # (L, d/2)
        freqs_complex = freqs_complex[np.newaxis, np.newaxis, :, :]  # (1, 1, L, d/2)

        result_cx = apply_rope_complex(x, freqs_complex)
        np.testing.assert_allclose(result_cx, result_rh, atol=1e-12)

    def test_output_shape(self):
        x = np.random.randn(2, 4, 8, 16)
        freqs = np.exp(1j * np.random.randn(1, 1, 8, 8))
        result = apply_rope_complex(x, freqs)
        self.assertEqual(result.shape, x.shape)


class TestRelativePositionProperty(unittest.TestCase):

    def test_dot_product_invariance(self):
        np.random.seed(42)
        d = 64
        rope = RoPE(d, 2048)

        q = np.random.randn(1, 1, 1, d)
        k = np.random.randn(1, 1, 1, d)

        pairs = [(5, 3), (105, 103), (505, 503), (1005, 1003)]
        dots = []
        for m, n in pairs:
            q_rot, _ = rope.forward(q, q, np.array([m]))
            _, k_rot = rope.forward(k, k, np.array([n]))
            dots.append(np.sum(q_rot * k_rot))

        for dot in dots[1:]:
            self.assertAlmostEqual(dots[0], dot, places=10)

    def test_different_relative_positions_differ(self):
        np.random.seed(42)
        d = 64
        rope = RoPE(d, 1024)

        q = np.random.randn(1, 1, 1, d)
        k = np.random.randn(1, 1, 1, d)

        q_rot5, _ = rope.forward(q, q, np.array([5]))
        _, k_rot3 = rope.forward(k, k, np.array([3]))
        dot_rel2 = np.sum(q_rot5 * k_rot3)

        q_rot5b, _ = rope.forward(q, q, np.array([5]))
        _, k_rot4 = rope.forward(k, k, np.array([4]))
        dot_rel1 = np.sum(q_rot5b * k_rot4)

        self.assertNotAlmostEqual(float(dot_rel2), float(dot_rel1), places=5)

    def test_dot_product_symmetry(self):
        np.random.seed(42)
        d = 64
        rope = RoPE(d, 512)

        q = np.random.randn(1, 1, 1, d)
        k = np.random.randn(1, 1, 1, d)

        q_rot, _ = rope.forward(q, q, np.array([10]))
        _, k_rot = rope.forward(k, k, np.array([5]))
        dot1 = np.sum(q_rot * k_rot)
        dot2 = np.sum(k_rot * q_rot)
        self.assertAlmostEqual(float(dot1), float(dot2), places=14)

    def test_same_position_gives_same_dot(self):
        """(0,0) vs (50,50) -- both relative position 0."""
        np.random.seed(42)
        d = 64
        rope = RoPE(d, 1024)

        q = np.random.randn(1, 1, 1, d)
        k = np.random.randn(1, 1, 1, d)

        q_rot0, _ = rope.forward(q, q, np.array([0]))
        _, k_rot0 = rope.forward(k, k, np.array([0]))
        dot_00 = np.sum(q_rot0 * k_rot0)

        q_rot50, _ = rope.forward(q, q, np.array([50]))
        _, k_rot50 = rope.forward(k, k, np.array([50]))
        dot_50_50 = np.sum(q_rot50 * k_rot50)

        self.assertAlmostEqual(float(dot_00), float(dot_50_50), places=10)

    def test_verify_function(self):
        np.random.seed(42)
        d = 64
        rope = RoPE(d, 2048)
        q = np.random.randn(1, 1, 1, d)
        k = np.random.randn(1, 1, 1, d)
        diffs = verify_relative_position_property(
            q, k, rope, np.array([10]), np.array([5])
        )
        self.assertTrue(np.all(diffs < 1e-10))


class TestRotationMatrixProperties(unittest.TestCase):

    def test_orthogonality(self):
        cos, sin = precompute_freqs(64, 100)
        for pos in [0, 1, 10, 50, 99]:
            R, frob_err, det = rotation_is_orthogonal(cos, sin, pos)
            self.assertLess(frob_err, 1e-10, f"Not orthogonal at pos={pos}")

    def test_determinant_one(self):
        cos, sin = precompute_freqs(64, 100)
        for pos in [0, 1, 50]:
            _, _, det = rotation_is_orthogonal(cos, sin, pos)
            self.assertAlmostEqual(det, 1.0, places=10)

    def test_composition(self):
        """R(m) @ R(n) should equal R(m+n)."""
        cos, sin = precompute_freqs(32, 200)
        m, n = 7, 13
        R_m, _, _ = rotation_is_orthogonal(cos, sin, m)
        R_n, _, _ = rotation_is_orthogonal(cos, sin, n)
        R_mn, _, _ = rotation_is_orthogonal(cos, sin, m + n)
        np.testing.assert_allclose(R_m @ R_n, R_mn, atol=1e-12)

    def test_inverse(self):
        """R(m) @ R(-m) should be identity, tested via rotate then inverse rotate."""
        np.random.seed(42)
        d = 32
        cos, sin = precompute_freqs(d, 200)
        x = np.random.randn(1, 1, 1, d)
        positions = np.array([15])

        rotated = apply_rope(x, cos, sin, positions)
        # Inverse rotation: negate sin
        unrotated = apply_rope(rotated, cos, -sin, positions)
        np.testing.assert_allclose(unrotated, x, atol=1e-12)

    def test_explicit_matrix_vs_elementwise(self):
        """Full block-diagonal R(m) @ x should match apply_rope."""
        np.random.seed(42)
        d = 16
        cos, sin = precompute_freqs(d, 100)
        pos = 7

        x = np.random.randn(d)
        R, _, _ = rotation_is_orthogonal(cos, sin, pos)
        result_matrix = R @ x

        x_4d = x.reshape(1, 1, 1, d)
        result_elem = apply_rope(x_4d, cos, sin, np.array([pos]))
        np.testing.assert_allclose(result_elem.flatten(), result_matrix, atol=1e-12)


class TestNormPreservation(unittest.TestCase):

    def test_various_positions(self):
        np.random.seed(42)
        d = 64
        cos, sin = precompute_freqs(d, 20000)

        for pos in [0, 1, 100, 10000]:
            x = np.random.randn(2, 4, 1, d)
            result = apply_rope(x, cos, sin, np.array([pos]))
            orig_norms = np.linalg.norm(x, axis=-1)
            rot_norms = np.linalg.norm(result, axis=-1)
            np.testing.assert_allclose(rot_norms, orig_norms, atol=1e-12)

    def test_large_position(self):
        d = 128
        cos, sin = precompute_freqs(d, 110000)
        x = np.random.randn(1, 1, 1, d)
        result = apply_rope(x, cos, sin, np.array([100000]))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        orig_norm = np.linalg.norm(x)
        rot_norm = np.linalg.norm(result)
        np.testing.assert_allclose(rot_norm, orig_norm, atol=1e-10)


class TestRoPEClass(unittest.TestCase):

    def test_forward_shapes(self):
        d = 64
        rope = RoPE(d, 256)
        B, H, H_kv, L = 2, 8, 2, 16
        q = np.random.randn(B, H, L, d)
        k = np.random.randn(B, H_kv, L, d)
        q_rot, k_rot = rope.forward(q, k)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_odd_d_raises(self):
        with self.assertRaises(ValueError):
            RoPE(63, 100)

    def test_gqa_compatibility(self):
        d = 64
        rope = RoPE(d, 128)
        B, H, H_kv, L = 2, 32, 8, 16
        q = np.random.randn(B, H, L, d)
        k = np.random.randn(B, H_kv, L, d)
        q_rot, k_rot = rope.forward(q, k)
        self.assertEqual(q_rot.shape, (B, H, L, d))
        self.assertEqual(k_rot.shape, (B, H_kv, L, d))

    def test_positions_parameter(self):
        d = 64
        rope = RoPE(d, 256)
        q = np.random.randn(1, 4, 3, d)
        k = np.random.randn(1, 4, 3, d)
        positions = np.array([10, 11, 12])
        q_rot, k_rot = rope.forward(q, k, positions=positions)
        self.assertEqual(q_rot.shape, q.shape)

    def test_batch_size_one(self):
        d = 32
        rope = RoPE(d, 64)
        q = np.random.randn(1, 2, 8, d)
        k = np.random.randn(1, 2, 8, d)
        q_rot, k_rot = rope.forward(q, k)
        self.assertEqual(q_rot.shape, q.shape)

    def test_single_token(self):
        d = 64
        rope = RoPE(d, 128)
        q = np.random.randn(1, 4, 1, d)
        k = np.random.randn(1, 4, 1, d)
        q_rot, k_rot = rope.forward(q, k)
        self.assertEqual(q_rot.shape, (1, 4, 1, d))


class TestBackward(unittest.TestCase):

    def test_gradient_shape(self):
        d = 64
        rope = RoPE(d, 128)
        B, H, H_kv, L = 2, 4, 2, 8
        q = np.random.randn(B, H, L, d)
        k = np.random.randn(B, H_kv, L, d)
        q_rot, k_rot = rope.forward(q, k)

        grad_q_rot = np.random.randn(*q_rot.shape)
        grad_k_rot = np.random.randn(*k_rot.shape)
        grad_q, grad_k = rope.backward(grad_q_rot, grad_k_rot)
        self.assertEqual(grad_q.shape, q.shape)
        self.assertEqual(grad_k.shape, k.shape)

    def test_inverse_rotation_recovers_original(self):
        np.random.seed(42)
        d = 32
        rope = RoPE(d, 64)
        q = np.random.randn(1, 2, 4, d)
        k = np.random.randn(1, 2, 4, d)

        q_rot, k_rot = rope.forward(q, k)
        grad_q, grad_k = rope.backward(q_rot, k_rot)
        np.testing.assert_allclose(grad_q, q, atol=1e-12)
        np.testing.assert_allclose(grad_k, k, atol=1e-12)

    def test_position_zero_passthrough(self):
        d = 32
        rope = RoPE(d, 64)
        q = np.random.randn(1, 2, 1, d)
        k = np.random.randn(1, 2, 1, d)
        positions = np.array([0])

        q_rot, k_rot = rope.forward(q, k, positions)
        grad_q_rot = np.random.randn(*q.shape)
        grad_k_rot = np.random.randn(*k.shape)
        grad_q, grad_k = rope.backward(grad_q_rot, grad_k_rot)

        np.testing.assert_allclose(grad_q, grad_q_rot, atol=1e-14)
        np.testing.assert_allclose(grad_k, grad_k_rot, atol=1e-14)

    def test_finite_difference_gradient_check(self):
        """Numerical gradient check using central differences."""
        np.random.seed(42)
        d = 8
        rope = RoPE(d, 32)
        B, H, H_kv, L = 1, 2, 1, 4
        q = np.random.randn(B, H, L, d)
        k = np.random.randn(B, H_kv, L, d)

        # Loss = sum(Q'^2) + sum(K'^2)
        q_rot, k_rot = rope.forward(q, k)
        grad_q_rot = 2.0 * q_rot
        grad_k_rot = 2.0 * k_rot
        grad_q_analytical, grad_k_analytical = rope.backward(grad_q_rot, grad_k_rot)

        eps = 1e-5
        grad_q_numerical = np.zeros_like(q)
        for idx in np.ndindex(q.shape):
            q_plus = q.copy()
            q_plus[idx] += eps
            q_rot_plus, k_rot_plus = rope.forward(q_plus, k)
            loss_plus = np.sum(q_rot_plus ** 2) + np.sum(k_rot_plus ** 2)

            q_minus = q.copy()
            q_minus[idx] -= eps
            q_rot_minus, k_rot_minus = rope.forward(q_minus, k)
            loss_minus = np.sum(q_rot_minus ** 2) + np.sum(k_rot_minus ** 2)

            grad_q_numerical[idx] = (loss_plus - loss_minus) / (2 * eps)

        rel_error = np.abs(grad_q_analytical - grad_q_numerical) / (
            np.abs(grad_q_analytical) + np.abs(grad_q_numerical) + 1e-8
        )
        self.assertTrue(np.all(rel_error < 1e-5), f"Max relative error: {rel_error.max()}")

        grad_k_numerical = np.zeros_like(k)
        for idx in np.ndindex(k.shape):
            k_plus = k.copy()
            k_plus[idx] += eps
            q_rot_plus, k_rot_plus = rope.forward(q, k_plus)
            loss_plus = np.sum(q_rot_plus ** 2) + np.sum(k_rot_plus ** 2)

            k_minus = k.copy()
            k_minus[idx] -= eps
            q_rot_minus, k_rot_minus = rope.forward(q, k_minus)
            loss_minus = np.sum(q_rot_minus ** 2) + np.sum(k_rot_minus ** 2)

            grad_k_numerical[idx] = (loss_plus - loss_minus) / (2 * eps)

        rel_error_k = np.abs(grad_k_analytical - grad_k_numerical) / (
            np.abs(grad_k_analytical) + np.abs(grad_k_numerical) + 1e-8
        )
        self.assertTrue(np.all(rel_error_k < 1e-5), f"Max relative error K: {rel_error_k.max()}")

    def test_backward_before_forward_raises(self):
        rope = RoPE(8, 32)
        with self.assertRaises(RuntimeError):
            rope.backward(np.zeros((1, 1, 1, 8)), np.zeros((1, 1, 1, 8)))


class TestImplementationEquivalence(unittest.TestCase):

    def test_precomputed_vs_on_the_fly(self):
        np.random.seed(42)
        d = 64
        L = 32
        x = np.random.randn(2, 4, L, d)

        cos_cached, sin_cached = precompute_freqs(d, 256)
        result_cached = apply_rope(x, cos_cached, sin_cached)

        cos_fresh, sin_fresh = precompute_freqs(d, L)
        result_fresh = apply_rope(x, cos_fresh, sin_fresh)

        np.testing.assert_allclose(result_fresh, result_cached, atol=1e-14)


class TestFrequencySchedule(unittest.TestCase):

    def test_matches_sinusoidal_pe(self):
        d = 128
        rope_freqs, sin_freqs = compare_with_sinusoidal(d, 100)
        np.testing.assert_allclose(rope_freqs, sin_freqs, atol=1e-15)

    def test_first_freq_is_one(self):
        cos, sin = precompute_freqs(128, 10, theta_base=10000.0)
        i0 = np.exp(-2.0 * 0 / 128 * np.log(10000.0))
        self.assertAlmostEqual(i0, 1.0, places=14)

    def test_custom_theta_500k(self):
        d = 128
        i = np.arange(d // 2, dtype=np.float64)
        freqs_10k = np.exp(-2.0 * i / d * np.log(10000.0))
        freqs_500k = np.exp(-2.0 * i / d * np.log(500000.0))
        self.assertTrue(np.all(freqs_500k <= freqs_10k + 1e-15))


class TestIntegrationWithAttention(unittest.TestCase):

    def test_full_attention_pipeline(self):
        """Wire RoPE into a simple attention computation."""
        np.random.seed(42)
        B, L, d_model, num_heads = 2, 8, 64, 4
        d_head = d_model // num_heads

        X = np.random.randn(B, L, d_model)
        W_Q = np.random.randn(d_model, d_model) * 0.02
        W_K = np.random.randn(d_model, d_model) * 0.02
        W_V = np.random.randn(d_model, d_model) * 0.02
        W_O = np.random.randn(d_model, d_model) * 0.02

        Q = (X @ W_Q).reshape(B, L, num_heads, d_head).transpose(0, 2, 1, 3)
        K = (X @ W_K).reshape(B, L, num_heads, d_head).transpose(0, 2, 1, 3)
        V = (X @ W_V).reshape(B, L, num_heads, d_head).transpose(0, 2, 1, 3)

        rope = RoPE(d_head, L)
        Q_rot, K_rot = rope.forward(Q, K)

        scores = Q_rot @ K_rot.transpose(0, 1, 3, 2) / np.sqrt(d_head)
        exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        out = attn @ V

        output = out.transpose(0, 2, 1, 3).reshape(B, L, d_model) @ W_O
        self.assertEqual(output.shape, (B, L, d_model))

    def test_causal_attention_with_rope(self):
        """RoPE applied before causal mask; masked positions get zero weight."""
        np.random.seed(42)
        B, L, d = 1, 4, 8
        num_heads = 2
        d_head = d // num_heads

        Q = np.random.randn(B, num_heads, L, d_head)
        K = np.random.randn(B, num_heads, L, d_head)

        rope = RoPE(d_head, L)
        Q_rot, K_rot = rope.forward(Q, K)

        scores = Q_rot @ K_rot.transpose(0, 1, 3, 2) / np.sqrt(d_head)

        mask = np.zeros((L, L))
        mask[np.triu_indices(L, k=1)] = -1e9
        scores = scores + mask.reshape(1, 1, L, L)

        exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        for i in range(L):
            for j in range(i + 1, L):
                self.assertAlmostEqual(attn[0, 0, i, j], 0.0, places=5)

    def test_gqa_with_rope(self):
        """Apply RoPE to Q and K with different head counts, then compute attention."""
        np.random.seed(42)
        B, L, d_head = 2, 8, 16
        H, H_kv = 8, 2
        group_size = H // H_kv

        Q = np.random.randn(B, H, L, d_head)
        K = np.random.randn(B, H_kv, L, d_head)
        V = np.random.randn(B, H_kv, L, d_head)

        rope = RoPE(d_head, L)
        Q_rot, K_rot = rope.forward(Q, K)

        K_exp = np.repeat(K_rot, group_size, axis=1)
        V_exp = np.repeat(V, group_size, axis=1)

        scores = Q_rot @ K_exp.transpose(0, 1, 3, 2) / np.sqrt(d_head)
        exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        out = attn @ V_exp

        self.assertEqual(out.shape, (B, H, L, d_head))

    def test_kv_cache_simulation(self):
        """Cached rotated keys should match full from-scratch computation."""
        np.random.seed(42)
        d = 16
        rope = RoPE(d, 128)

        Q_all = np.random.randn(1, 2, 4, d)
        K_all = np.random.randn(1, 2, 4, d)

        # Full computation: rotate all 4 positions at once
        Q_rot_full, K_rot_full = rope.forward(Q_all, K_all)
        scores_full = Q_rot_full[:, :, 3:4, :] @ K_rot_full.transpose(0, 1, 3, 2)

        # Incremental: rotate K at positions [0,1,2], cache them, then rotate Q at position 3
        K_prefix = K_all[:, :, :3, :]
        _, K_rot_cached = rope.forward(
            K_prefix, K_prefix, positions=np.array([0, 1, 2])
        )

        Q_new = Q_all[:, :, 3:4, :]
        Q_rot_new, K_rot_new = rope.forward(
            Q_new, K_all[:, :, 3:4, :], positions=np.array([3])
        )

        K_rot_combined = np.concatenate([K_rot_cached, K_rot_new], axis=2)
        scores_incr = Q_rot_new @ K_rot_combined.transpose(0, 1, 3, 2)

        np.testing.assert_allclose(scores_incr, scores_full, atol=1e-12)


class TestEdgeCases(unittest.TestCase):

    def test_single_dimension_pair(self):
        x = np.array([[[[3.0, 4.0]]]])
        cos, sin = precompute_freqs(2, 10)
        result = apply_rope(x, cos, sin, np.array([5]))
        norm_orig = np.linalg.norm(x)
        norm_rot = np.linalg.norm(result)
        np.testing.assert_allclose(norm_rot, norm_orig, atol=1e-12)

    def test_large_position_bounded(self):
        d = 128
        cos, sin = precompute_freqs(d, 110000)
        x = np.random.randn(1, 1, 1, d)
        result = apply_rope(x, cos, sin, np.array([100000]))
        norm_x = np.linalg.norm(x)
        self.assertTrue(np.all(np.abs(result) <= norm_x + 1e-10))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_very_small_frequencies(self):
        d = 128
        i = np.arange(d // 2, dtype=np.float64)
        inv_freq = np.exp(-2.0 * i / d * np.log(10000.0))
        # Lowest frequency at i=63
        self.assertAlmostEqual(inv_freq[-1], 10000.0 ** (-126.0 / 128.0), places=10)
        # At position 100000, angle ~ 100000 * 1e-4 ~ 10
        angle = 100000 * inv_freq[-1]
        self.assertTrue(np.isfinite(np.cos(angle)))
        self.assertTrue(np.isfinite(np.sin(angle)))

    def test_d_equals_2(self):
        rope = RoPE(2, 100)
        q = np.random.randn(1, 1, 5, 2)
        k = np.random.randn(1, 1, 5, 2)
        q_rot, k_rot = rope.forward(q, k)
        self.assertEqual(q_rot.shape, (1, 1, 5, 2))


class TestNumericalStability(unittest.TestCase):

    def test_large_seq_len(self):
        cos, sin = precompute_freqs(128, 100000)
        self.assertFalse(np.any(np.isnan(cos)))
        self.assertFalse(np.any(np.isnan(sin)))
        self.assertFalse(np.any(np.isinf(cos)))
        self.assertFalse(np.any(np.isinf(sin)))

    def test_large_d_head(self):
        cos, sin = precompute_freqs(256, 1000)
        self.assertFalse(np.any(np.isnan(cos)))
        self.assertFalse(np.any(np.isinf(cos)))

    def test_cos_sin_bounded(self):
        cos, sin = precompute_freqs(128, 10000)
        self.assertTrue(np.all(np.abs(cos) <= 1.0 + 1e-15))
        self.assertTrue(np.all(np.abs(sin) <= 1.0 + 1e-15))


if __name__ == "__main__":
    unittest.main()
