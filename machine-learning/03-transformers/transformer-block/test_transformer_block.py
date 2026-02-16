"""Tests for transformer_block."""

import unittest
import sys
from pathlib import Path

import numpy as np

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from importlib import import_module

_mod = import_module("03-transformers.transformer-block.implementation")
SwiGLUFFN = _mod.SwiGLUFFN
TransformerBlock = _mod.TransformerBlock
count_parameters = _mod.count_parameters
count_flops = _mod.count_flops
memory_footprint = _mod.memory_footprint
_stable_sigmoid = _mod._stable_sigmoid


def _make_block(d_model=64, num_heads=4, num_kv_heads=2, d_ff=128,
                max_seq_len=128, rope_theta=10000.0, seed=42):
    np.random.seed(seed)
    return TransformerBlock(d_model, num_heads, num_kv_heads, d_ff,
                            max_seq_len, rope_theta)


def _rel_error(a, b):
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-8)
    return float(np.max(np.abs(a - b) / denom))


# ============================================================
# Shape correctness
# ============================================================

class TestShapeCorrectness(unittest.TestCase):

    def test_output_shape_matches_input(self):
        for B, L, d in [(1, 16, 64), (2, 32, 64), (4, 8, 64)]:
            block = _make_block(d_model=d)
            x = np.random.randn(B, L, d)
            out = block.forward(x)
            self.assertEqual(out.shape, (B, L, d))

    def test_single_token(self):
        block = _make_block()
        x = np.random.randn(1, 1, 64)
        out = block.forward(x)
        self.assertEqual(out.shape, (1, 1, 64))

    def test_variable_batch_sizes(self):
        for B in [1, 2, 8]:
            block = _make_block()
            x = np.random.randn(B, 16, 64)
            out = block.forward(x)
            self.assertEqual(out.shape, (B, 16, 64))

    def test_variable_sequence_lengths(self):
        for L in [1, 16, 64, 128]:
            block = _make_block()
            x = np.random.randn(1, L, 64)
            out = block.forward(x)
            self.assertEqual(out.shape, (1, L, 64))

    def test_swiglu_shapes(self):
        np.random.seed(42)
        ffn = SwiGLUFFN(64, 128)
        x = np.random.randn(2, 8, 64)
        out = ffn.forward(x)
        self.assertEqual(out.shape, (2, 8, 64))
        hidden = ffn._cache["hidden"]
        self.assertEqual(hidden.shape, (2, 8, 128))


# ============================================================
# SwiGLU FFN correctness
# ============================================================

class TestSwiGLUCorrectness(unittest.TestCase):

    def test_known_input_output(self):
        np.random.seed(0)
        d_model, d_ff = 4, 6
        ffn = SwiGLUFFN(d_model, d_ff)
        x = np.random.randn(1, 1, d_model)

        gate = x @ ffn.W_gate
        up = x @ ffn.W_up
        sig = _stable_sigmoid(gate)
        silu_gate = gate * sig
        hidden = silu_gate * up
        expected = hidden @ ffn.W_down

        actual = ffn.forward(x)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_zero_input(self):
        np.random.seed(42)
        ffn = SwiGLUFFN(8, 16)
        x = np.zeros((1, 4, 8))
        out = ffn.forward(x)
        np.testing.assert_allclose(out, 0.0, atol=1e-15)

    def test_gating_effect(self):
        np.random.seed(42)
        ffn = SwiGLUFFN(8, 16)
        ffn.W_gate = np.full_like(ffn.W_gate, -100.0)
        # Use positive input so x @ W_gate is very negative -> SiLU ≈ 0
        x = np.abs(np.random.randn(1, 4, 8)) + 0.1
        out = ffn.forward(x)
        np.testing.assert_allclose(out, 0.0, atol=1e-30)

    def test_silu_known_values(self):
        z = np.array([0.0, 1.0, -1.0])
        sig = _stable_sigmoid(z)
        silu = z * sig
        np.testing.assert_allclose(silu[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(silu[1], 1.0 / (1.0 + np.exp(-1.0)), atol=1e-8)
        np.testing.assert_allclose(silu[2], -1.0 / (1.0 + np.exp(1.0)), atol=1e-8)

    def test_gradient_check_swiglu(self):
        np.random.seed(42)
        d_model, d_ff = 4, 6
        ffn = SwiGLUFFN(d_model, d_ff)
        x = np.random.randn(1, 2, d_model) * 0.5
        grad_output = np.random.randn(1, 2, d_model)
        h = 1e-5

        ffn.forward(x)
        dx_analytical = ffn.backward(grad_output)

        loss_fn = lambda out: np.sum(out * grad_output)

        # Check dx
        dx_num = np.zeros_like(x)
        for idx in np.ndindex(*x.shape):
            x_p, x_m = x.copy(), x.copy()
            x_p[idx] += h
            x_m[idx] -= h
            dx_num[idx] = (loss_fn(ffn.forward(x_p)) - loss_fn(ffn.forward(x_m))) / (2 * h)
        ffn.forward(x)
        ffn.backward(grad_output)
        self.assertLess(_rel_error(dx_analytical, dx_num), 1e-5)

        # Check W_gate
        dW_gate_analytical = ffn.grad_W_gate.copy()
        dW_gate_num = np.zeros_like(ffn.W_gate)
        orig = ffn.W_gate.copy()
        for idx in np.ndindex(*ffn.W_gate.shape):
            ffn.W_gate = orig.copy()
            ffn.W_gate[idx] += h
            fp = loss_fn(ffn.forward(x))
            ffn.W_gate = orig.copy()
            ffn.W_gate[idx] -= h
            fm = loss_fn(ffn.forward(x))
            dW_gate_num[idx] = (fp - fm) / (2 * h)
        ffn.W_gate = orig.copy()
        self.assertLess(_rel_error(dW_gate_analytical, dW_gate_num), 1e-5)

        # Check W_up
        ffn.forward(x)
        ffn.backward(grad_output)
        dW_up_analytical = ffn.grad_W_up.copy()
        dW_up_num = np.zeros_like(ffn.W_up)
        orig = ffn.W_up.copy()
        for idx in np.ndindex(*ffn.W_up.shape):
            ffn.W_up = orig.copy()
            ffn.W_up[idx] += h
            fp = loss_fn(ffn.forward(x))
            ffn.W_up = orig.copy()
            ffn.W_up[idx] -= h
            fm = loss_fn(ffn.forward(x))
            dW_up_num[idx] = (fp - fm) / (2 * h)
        ffn.W_up = orig.copy()
        self.assertLess(_rel_error(dW_up_analytical, dW_up_num), 1e-5)

        # Check W_down
        ffn.forward(x)
        ffn.backward(grad_output)
        dW_down_analytical = ffn.grad_W_down.copy()
        dW_down_num = np.zeros_like(ffn.W_down)
        orig = ffn.W_down.copy()
        for idx in np.ndindex(*ffn.W_down.shape):
            ffn.W_down = orig.copy()
            ffn.W_down[idx] += h
            fp = loss_fn(ffn.forward(x))
            ffn.W_down = orig.copy()
            ffn.W_down[idx] -= h
            fm = loss_fn(ffn.forward(x))
            dW_down_num[idx] = (fp - fm) / (2 * h)
        ffn.W_down = orig.copy()
        self.assertLess(_rel_error(dW_down_analytical, dW_down_num), 1e-5)


# ============================================================
# Residual connection tests
# ============================================================

class TestResidualConnections(unittest.TestCase):

    def test_identity_passthrough(self):
        block = _make_block()
        block.W_Q[:] = 0
        block.W_K[:] = 0
        block.W_V[:] = 0
        block.W_O[:] = 0
        block.ffn.W_gate[:] = 0
        block.ffn.W_up[:] = 0
        block.ffn.W_down[:] = 0
        x = np.random.randn(1, 8, 64)
        out = block.forward(x)
        np.testing.assert_allclose(out, x, atol=1e-10)

    def test_residual_gradient(self):
        block = _make_block()
        x = np.random.randn(1, 4, 64) * 0.1
        grad_output = np.random.randn(1, 4, 64)
        block.forward(x)
        grad_x = block.backward(grad_output)
        # grad_x should contain the direct residual path component
        # so ||grad_x|| >= ||grad_output|| (residual ensures no shrinkage)
        self.assertGreaterEqual(
            np.linalg.norm(grad_x) + 1e-10,
            np.linalg.norm(grad_output) * 0.5  # relaxed: sublayer can partially cancel
        )

    def test_gradient_magnitude_preservation(self):
        block = _make_block()
        # With small weights, gradient should be close to grad_output
        block.W_Q *= 0.01
        block.W_K *= 0.01
        block.W_V *= 0.01
        block.W_O *= 0.01
        block.ffn.W_gate *= 0.01
        block.ffn.W_up *= 0.01
        block.ffn.W_down *= 0.01
        x = np.random.randn(1, 4, 64) * 0.1
        grad_output = np.random.randn(1, 4, 64)
        block.forward(x)
        grad_x = block.backward(grad_output)
        # With very small sublayer weights, grad_x ≈ grad_output
        np.testing.assert_allclose(grad_x, grad_output, atol=0.5)


# ============================================================
# Pre-norm behavior
# ============================================================

class TestPreNormBehavior(unittest.TestCase):

    def test_attention_receives_normalized_input(self):
        block = _make_block()
        x = np.random.randn(2, 8, 64) * 10
        block.forward(x)
        x_norm = block._cache["x_norm"]
        rms = np.sqrt(np.mean(x_norm ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=0.01)

    def test_ffn_receives_normalized_input(self):
        block = _make_block()
        x = np.random.randn(2, 8, 64) * 10
        block.forward(x)
        h_norm = block._cache["h_norm"]
        rms = np.sqrt(np.mean(h_norm ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=0.01)

    def test_output_is_not_normalized(self):
        block = _make_block()
        x = np.random.randn(2, 8, 64) * 5
        out = block.forward(x)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        # Output should NOT be unit RMS in general
        self.assertFalse(np.allclose(rms, 1.0, atol=0.01))


# ============================================================
# RoPE integration
# ============================================================

class TestRoPEIntegration(unittest.TestCase):

    def test_position_dependent_output(self):
        """Different relative positions produce different outputs.

        RoPE preserves relative position in dot products, so we must use
        positions with different relative gaps to see a difference.
        """
        block = _make_block()
        x = np.random.randn(1, 4, 64)

        positions_a = np.array([0, 1, 2, 3])
        positions_b = np.array([0, 2, 4, 6])

        out_a = block.forward(x, positions=positions_a)
        out_b = block.forward(x, positions=positions_b)

        self.assertFalse(np.allclose(out_a, out_b, atol=1e-6))

    def test_position_zero_is_identity_rotation(self):
        np.random.seed(42)
        block = _make_block()
        x = np.random.randn(1, 1, 64)
        positions = np.array([0])
        out = block.forward(x, positions=positions)
        # At position 0, cos=1, sin=0, so RoPE is identity.
        # Output should match what we get with default positions (also [0] for L=1)
        out_default = block.forward(x)
        np.testing.assert_allclose(out, out_default, atol=1e-12)

    def test_causal_mask_with_rope(self):
        block = _make_block()
        x = np.random.randn(1, 4, 64)
        out = block.forward(x)
        A = block._cache["A"]
        # Upper triangle of attention weights should be zero (causal)
        for i in range(4):
            for j in range(i + 1, 4):
                np.testing.assert_allclose(A[0, :, i, j], 0.0, atol=1e-8)


# ============================================================
# Numerical correctness
# ============================================================

class TestNumericalCorrectness(unittest.TestCase):

    def test_gradient_check_full_block(self):
        """Finite-difference gradient check for all parameters and input."""
        np.random.seed(42)
        d_model, num_heads, num_kv_heads, d_ff = 8, 2, 1, 16
        max_seq_len = 8
        B, L = 1, 4
        h_step = 1e-5

        block = TransformerBlock(d_model, num_heads, num_kv_heads, d_ff, max_seq_len)
        x = np.random.randn(B, L, d_model) * 0.5
        grad_output = np.random.randn(B, L, d_model)

        block.forward(x)
        dx_analytical = block.backward(grad_output)

        loss_fn = lambda out: np.sum(out * grad_output)

        # Check dx
        dx_num = np.zeros_like(x)
        for idx in np.ndindex(*x.shape):
            xp, xm = x.copy(), x.copy()
            xp[idx] += h_step
            xm[idx] -= h_step
            dx_num[idx] = (loss_fn(block.forward(xp)) - loss_fn(block.forward(xm))) / (2 * h_step)
        block.forward(x)
        block.backward(grad_output)
        self.assertLess(_rel_error(dx_analytical, dx_num), 1e-4,
                        f"dx rel_error: {_rel_error(dx_analytical, dx_num)}")

        # Check W_Q
        dWQ = block.grad_W_Q.copy()
        dWQ_num = np.zeros_like(block.W_Q)
        orig = block.W_Q.copy()
        for idx in np.ndindex(*block.W_Q.shape):
            block.W_Q = orig.copy()
            block.W_Q[idx] += h_step
            fp = loss_fn(block.forward(x))
            block.W_Q = orig.copy()
            block.W_Q[idx] -= h_step
            fm = loss_fn(block.forward(x))
            dWQ_num[idx] = (fp - fm) / (2 * h_step)
        block.W_Q = orig.copy()
        self.assertLess(_rel_error(dWQ, dWQ_num), 1e-4,
                        f"W_Q rel_error: {_rel_error(dWQ, dWQ_num)}")

        # Check W_K
        block.forward(x)
        block.backward(grad_output)
        dWK = block.grad_W_K.copy()
        dWK_num = np.zeros_like(block.W_K)
        orig = block.W_K.copy()
        for idx in np.ndindex(*block.W_K.shape):
            block.W_K = orig.copy()
            block.W_K[idx] += h_step
            fp = loss_fn(block.forward(x))
            block.W_K = orig.copy()
            block.W_K[idx] -= h_step
            fm = loss_fn(block.forward(x))
            dWK_num[idx] = (fp - fm) / (2 * h_step)
        block.W_K = orig.copy()
        self.assertLess(_rel_error(dWK, dWK_num), 1e-4,
                        f"W_K rel_error: {_rel_error(dWK, dWK_num)}")

        # Check W_V
        block.forward(x)
        block.backward(grad_output)
        dWV = block.grad_W_V.copy()
        dWV_num = np.zeros_like(block.W_V)
        orig = block.W_V.copy()
        for idx in np.ndindex(*block.W_V.shape):
            block.W_V = orig.copy()
            block.W_V[idx] += h_step
            fp = loss_fn(block.forward(x))
            block.W_V = orig.copy()
            block.W_V[idx] -= h_step
            fm = loss_fn(block.forward(x))
            dWV_num[idx] = (fp - fm) / (2 * h_step)
        block.W_V = orig.copy()
        self.assertLess(_rel_error(dWV, dWV_num), 1e-4,
                        f"W_V rel_error: {_rel_error(dWV, dWV_num)}")

        # Check W_O
        block.forward(x)
        block.backward(grad_output)
        dWO = block.grad_W_O.copy()
        dWO_num = np.zeros_like(block.W_O)
        orig = block.W_O.copy()
        for idx in np.ndindex(*block.W_O.shape):
            block.W_O = orig.copy()
            block.W_O[idx] += h_step
            fp = loss_fn(block.forward(x))
            block.W_O = orig.copy()
            block.W_O[idx] -= h_step
            fm = loss_fn(block.forward(x))
            dWO_num[idx] = (fp - fm) / (2 * h_step)
        block.W_O = orig.copy()
        self.assertLess(_rel_error(dWO, dWO_num), 1e-4,
                        f"W_O rel_error: {_rel_error(dWO, dWO_num)}")

        # Check gamma1
        block.forward(x)
        block.backward(grad_output)
        dg1 = block.norm1.grad_gamma.copy()
        dg1_num = np.zeros_like(block.norm1.gamma)
        orig = block.norm1.gamma.copy()
        for idx in np.ndindex(*block.norm1.gamma.shape):
            block.norm1.gamma = orig.copy()
            block.norm1.gamma[idx] += h_step
            fp = loss_fn(block.forward(x))
            block.norm1.gamma = orig.copy()
            block.norm1.gamma[idx] -= h_step
            fm = loss_fn(block.forward(x))
            dg1_num[idx] = (fp - fm) / (2 * h_step)
        block.norm1.gamma = orig.copy()
        self.assertLess(_rel_error(dg1, dg1_num), 1e-4,
                        f"gamma1 rel_error: {_rel_error(dg1, dg1_num)}")

        # Check gamma2
        block.forward(x)
        block.backward(grad_output)
        dg2 = block.norm2.grad_gamma.copy()
        dg2_num = np.zeros_like(block.norm2.gamma)
        orig = block.norm2.gamma.copy()
        for idx in np.ndindex(*block.norm2.gamma.shape):
            block.norm2.gamma = orig.copy()
            block.norm2.gamma[idx] += h_step
            fp = loss_fn(block.forward(x))
            block.norm2.gamma = orig.copy()
            block.norm2.gamma[idx] -= h_step
            fm = loss_fn(block.forward(x))
            dg2_num[idx] = (fp - fm) / (2 * h_step)
        block.norm2.gamma = orig.copy()
        self.assertLess(_rel_error(dg2, dg2_num), 1e-4,
                        f"gamma2 rel_error: {_rel_error(dg2, dg2_num)}")

    def test_no_nan_inf(self):
        block = _make_block()
        x = np.random.uniform(-2, 2, (2, 8, 64))
        out = block.forward(x)
        self.assertTrue(np.all(np.isfinite(out)))
        grad = block.backward(np.random.randn(*out.shape))
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_forward_determinism(self):
        x = np.random.randn(1, 4, 64)
        block = _make_block()
        out1 = block.forward(x.copy())
        out2 = block.forward(x.copy())
        np.testing.assert_array_equal(out1, out2)

    def test_backward_determinism(self):
        x = np.random.randn(1, 4, 64)
        grad_output = np.random.randn(1, 4, 64)
        block = _make_block()
        block.forward(x.copy())
        g1 = block.backward(grad_output.copy())
        block.forward(x.copy())
        g2 = block.backward(grad_output.copy())
        np.testing.assert_array_equal(g1, g2)


# ============================================================
# Integration tests
# ============================================================

class TestIntegration(unittest.TestCase):

    def test_stacked_blocks(self):
        block1 = _make_block(seed=42)
        block2 = _make_block(seed=43)
        x = np.random.randn(1, 8, 64)
        h = block1.forward(x)
        out = block2.forward(h)
        self.assertEqual(out.shape, (1, 8, 64))

    def test_forward_backward_roundtrip(self):
        block = _make_block()
        x = np.random.randn(1, 4, 64)
        block.forward(x)
        grad_x = block.backward(np.random.randn(1, 4, 64))
        self.assertIsNotNone(block.grad_W_Q)
        self.assertIsNotNone(block.grad_W_K)
        self.assertIsNotNone(block.grad_W_V)
        self.assertIsNotNone(block.grad_W_O)
        self.assertIsNotNone(block.ffn.grad_W_gate)
        self.assertIsNotNone(block.ffn.grad_W_up)
        self.assertIsNotNone(block.ffn.grad_W_down)
        self.assertIsNotNone(block.norm1.grad_gamma)
        self.assertIsNotNone(block.norm2.grad_gamma)
        self.assertIsNotNone(grad_x)

    def test_gqa_configurations(self):
        configs = [
            (8, 8),   # MHA
            (8, 4),   # GQA
            (8, 1),   # MQA
        ]
        for num_heads, num_kv_heads in configs:
            np.random.seed(42)
            block = TransformerBlock(
                d_model=64, num_heads=num_heads, num_kv_heads=num_kv_heads,
                d_ff=128, max_seq_len=64,
            )
            x = np.random.randn(1, 8, 64)
            out = block.forward(x)
            self.assertEqual(out.shape, (1, 8, 64),
                             f"Failed for h={num_heads}, h_kv={num_kv_heads}")
            self.assertTrue(np.all(np.isfinite(out)))

    def test_causal_masking(self):
        block = _make_block()
        x = np.random.randn(1, 8, 64)

        out_full = block.forward(x)
        out_pos0_full = out_full[0, 0, :].copy()

        out_single = block.forward(x[:, :1, :])
        out_pos0_single = out_single[0, 0, :].copy()

        np.testing.assert_allclose(out_pos0_full, out_pos0_single, atol=1e-10)


# ============================================================
# Configuration validation
# ============================================================

class TestConfigValidation(unittest.TestCase):

    def test_d_model_not_divisible_by_heads(self):
        with self.assertRaises(ValueError):
            TransformerBlock(d_model=65, num_heads=4, num_kv_heads=2,
                             d_ff=128, max_seq_len=64)

    def test_heads_not_divisible_by_kv_heads(self):
        with self.assertRaises(ValueError):
            TransformerBlock(d_model=64, num_heads=8, num_kv_heads=3,
                             d_ff=128, max_seq_len=64)

    def test_odd_d_k_raises(self):
        # d_model=6, num_heads=2 => d_k=3 (odd), should fail for RoPE
        with self.assertRaises(ValueError):
            TransformerBlock(d_model=6, num_heads=2, num_kv_heads=1,
                             d_ff=16, max_seq_len=64)


# ============================================================
# Parameter count and FLOP tests
# ============================================================

class TestParameterCountAndFlops(unittest.TestCase):

    def test_llama_7b_params(self):
        params = count_parameters(4096, 32, 32, 11008)
        expected_wq = 4096 * 4096
        expected_wk = 4096 * 4096
        expected_wv = 4096 * 4096
        expected_wo = 4096 * 4096
        self.assertEqual(params["W_Q"], expected_wq)
        self.assertEqual(params["W_K"], expected_wk)
        self.assertEqual(params["W_V"], expected_wv)
        self.assertEqual(params["W_O"], expected_wo)
        self.assertEqual(params["W_gate"], 4096 * 11008)
        self.assertEqual(params["W_up"], 4096 * 11008)
        self.assertEqual(params["W_down"], 11008 * 4096)
        # Total ~202M
        total = params["total"]
        self.assertGreater(total, 200_000_000)
        self.assertLess(total, 210_000_000)

    def test_llama2_70b_gqa_savings(self):
        params_gqa = count_parameters(8192, 64, 8, 28672)
        params_mha = count_parameters(8192, 64, 64, 28672)
        self.assertLess(params_gqa["attn_total"], params_mha["attn_total"])
        # K and V are 8x smaller with GQA
        self.assertEqual(params_gqa["W_K"] * 8, params_mha["W_K"])
        self.assertEqual(params_gqa["W_V"] * 8, params_mha["W_V"])

    def test_ffn_dominates(self):
        configs = [
            (4096, 32, 32, 11008),    # Llama 7B
            (8192, 64, 8, 28672),     # Llama 70B
            (4096, 32, 8, 14336),     # Llama 3 8B
        ]
        for d_model, nh, nkv, dff in configs:
            params = count_parameters(d_model, nh, nkv, dff)
            self.assertGreater(params["ffn_pct"], 60.0,
                               f"FFN should dominate for config ({d_model},{nh},{nkv},{dff})")

    def test_flops_breakdown_sums(self):
        flops = count_flops(1, 128, 64, 4, 2, 128)
        component_sum = (flops["attn_proj"] + flops["attn_core"]
                         + flops["rope"] + flops["ffn_total"] + flops["norm"])
        self.assertEqual(flops["total"], component_sum)

    def test_memory_footprint_structure(self):
        mem = memory_footprint(1, 128, 64, 4, 2, 128)
        self.assertIn("param_bytes", mem)
        self.assertIn("activation_bytes", mem)
        self.assertIn("largest_tensor", mem)
        self.assertIn("total_bytes", mem)
        self.assertEqual(mem["total_bytes"],
                         mem["param_bytes"] + mem["activation_bytes"])


if __name__ == "__main__":
    unittest.main()
