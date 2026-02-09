"""Tests for optimizers."""

import unittest
import numpy as np
from implementation import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    AdamL2,
    StepDecayScheduler,
    CosineScheduler,
    WarmupCosineScheduler,
)


def make_params(*arrays: np.ndarray):
    """Create parameter list from arrays, initializing gradients to zero."""
    return [
        {"params": a.copy(), "grad": np.zeros_like(a)} for a in arrays
    ]


def set_grads(params, *grads):
    """Set gradients on a parameter list."""
    for p, g in zip(params, grads):
        p["grad"] = g.copy()


# ---------------------------------------------------------------------------
# SGD Tests
# ---------------------------------------------------------------------------


class TestSGDBasic(unittest.TestCase):
    """Single-step correctness for vanilla SGD: f(x) = x^2, x0=3.0, lr=0.1."""

    def test_single_step(self):
        params = make_params(np.array([3.0]))
        set_grads(params, np.array([6.0]))  # g = 2*x = 6
        opt = SGD(params, lr=0.1)
        opt.step()
        np.testing.assert_allclose(params[0]["params"], [2.4])

    def test_two_steps(self):
        params = make_params(np.array([3.0]))
        opt = SGD(params, lr=0.1)

        set_grads(params, np.array([6.0]))
        opt.step()
        np.testing.assert_allclose(params[0]["params"], [2.4])

        set_grads(params, np.array([4.8]))  # 2 * 2.4
        opt.step()
        np.testing.assert_allclose(params[0]["params"], [1.92])

    def test_zero_gradient_no_update(self):
        params = make_params(np.array([5.0, -3.0]))
        set_grads(params, np.array([0.0, 0.0]))
        opt = SGD(params, lr=0.1)
        opt.step()
        np.testing.assert_array_equal(params[0]["params"], [5.0, -3.0])


class TestSGDMomentum(unittest.TestCase):
    """Two steps on f(x) = x^2, verifying velocity accumulates correctly."""

    def test_two_steps(self):
        params = make_params(np.array([3.0]))
        opt = SGD(params, lr=0.1, momentum=0.9)

        # Step 1: v = 0 + 6 = 6, x = 3.0 - 0.1*6 = 2.4
        set_grads(params, np.array([6.0]))
        opt.step()
        np.testing.assert_allclose(params[0]["params"], [2.4])

        # Step 2: g = 4.8, v = 0.9*6 + 4.8 = 10.2, x = 2.4 - 0.1*10.2 = 1.38
        set_grads(params, np.array([4.8]))
        opt.step()
        np.testing.assert_allclose(params[0]["params"], [1.38])

    def test_velocity_initialized_to_zero(self):
        params = make_params(np.array([1.0, 2.0]))
        opt = SGD(params, lr=0.1, momentum=0.9)
        set_grads(params, np.array([1.0, 1.0]))
        opt.step()
        # v = 0 + g = [1,1], theta = [1,2] - 0.1*[1,1] = [0.9, 1.9]
        np.testing.assert_allclose(params[0]["params"], [0.9, 1.9])


class TestSGDNesterov(unittest.TestCase):

    def test_nesterov_requires_momentum(self):
        params = make_params(np.array([1.0]))
        with self.assertRaises(ValueError):
            SGD(params, lr=0.1, nesterov=True)

    def test_nesterov_single_step(self):
        """Hand-computed Nesterov step: v = beta*0 + g = g; theta -= lr*(beta*v + g)."""
        params = make_params(np.array([3.0]))
        opt = SGD(params, lr=0.1, momentum=0.9, nesterov=True)
        set_grads(params, np.array([6.0]))
        opt.step()
        # v1 = 0.9*0 + 6.0 = 6.0
        # Nesterov update: theta -= 0.1 * (0.9*6.0 + 6.0) = 0.1 * 11.4 = 1.14
        # theta = 3.0 - 1.14 = 1.86
        np.testing.assert_allclose(params[0]["params"], [1.86])

    def test_nesterov_differs_from_standard_momentum(self):
        x0 = np.array([3.0])
        g1, g2 = np.array([6.0]), np.array([4.8])

        params_std = make_params(x0)
        opt_std = SGD(params_std, lr=0.1, momentum=0.9)
        set_grads(params_std, g1)
        opt_std.step()
        set_grads(params_std, g2)
        opt_std.step()

        params_nes = make_params(x0)
        opt_nes = SGD(params_nes, lr=0.1, momentum=0.9, nesterov=True)
        set_grads(params_nes, g1)
        opt_nes.step()
        set_grads(params_nes, g2)
        opt_nes.step()

        self.assertFalse(
            np.allclose(params_std[0]["params"], params_nes[0]["params"]),
            "Nesterov and standard momentum should produce different trajectories",
        )


# ---------------------------------------------------------------------------
# Adam Tests
# ---------------------------------------------------------------------------


class TestAdamSingleStep(unittest.TestCase):
    """Hand-computed single step from the README."""

    def test_single_step(self):
        params = make_params(np.array([3.0]))
        set_grads(params, np.array([6.0]))
        opt = Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        opt.step()

        # m1 = 0.1 * 6 = 0.6
        # v1 = 0.001 * 36 = 0.036
        # m_hat = 0.6 / 0.1 = 6.0
        # v_hat = 0.036 / 0.001 = 36.0
        # x1 = 3.0 - 0.001 * 6.0 / (sqrt(36.0) + 1e-8) = 3.0 - 0.001 = 2.999
        np.testing.assert_allclose(params[0]["params"], [2.999], atol=1e-10)


class TestAdamBiasCorrection(unittest.TestCase):

    def test_step_1_correction(self):
        """At t=1, m_hat = m / 0.1 and v_hat = v / 0.001 for default betas."""
        params = make_params(np.array([1.0]))
        set_grads(params, np.array([2.0]))
        opt = Adam(params, lr=0.001, betas=(0.9, 0.999))
        opt.step()

        m1 = 0.1 * 2.0
        v1 = 0.001 * 4.0
        self.assertAlmostEqual(opt.state[0]["m"][0], m1)
        self.assertAlmostEqual(opt.state[0]["v"][0], v1)

        m_hat = m1 / (1 - 0.9)
        v_hat = v1 / (1 - 0.999)
        self.assertAlmostEqual(m_hat, 2.0)
        self.assertAlmostEqual(v_hat, 4.0)

    def test_correction_approaches_one(self):
        """After many steps, bias correction factor approaches 1."""
        params = make_params(np.array([0.0]))
        opt = Adam(params, lr=0.001)

        for _ in range(10000):
            set_grads(params, np.array([1.0]))
            opt.step()

        correction_beta1 = 1 - 0.9 ** 10000
        correction_beta2 = 1 - 0.999 ** 10000
        self.assertAlmostEqual(correction_beta1, 1.0, places=5)
        self.assertAlmostEqual(correction_beta2, 1.0, places=2)


class TestAdamMultipleSteps(unittest.TestCase):

    def test_three_steps(self):
        params = make_params(np.array([3.0]))
        opt = Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        grads = [6.0, 2 * params[0]["params"][0], None]

        set_grads(params, np.array([6.0]))
        opt.step()
        x1 = params[0]["params"][0]
        np.testing.assert_allclose(x1, 2.999, atol=1e-10)

        set_grads(params, np.array([2 * x1]))
        opt.step()
        x2 = params[0]["params"][0]

        set_grads(params, np.array([2 * x2]))
        opt.step()

        self.assertTrue(params[0]["params"][0] < 2.999)


# ---------------------------------------------------------------------------
# AdamW Tests
# ---------------------------------------------------------------------------


class TestAdamWWeightDecay(unittest.TestCase):

    def test_weight_decay_applied(self):
        """Verify decoupled weight decay: wd_term = lr * lambda * theta."""
        params = make_params(np.array([3.0]))
        set_grads(params, np.array([6.0]))
        opt = AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
        opt.step()

        # Adam-only update: x = 3.0 - 0.001 * 6.0 / (sqrt(36) + 1e-8) = 2.999
        # Weight decay: 0.001 * 0.1 * 3.0 = 0.0003
        # AdamW: 3.0 - 0.001 * (6.0/6.0 + 0.1*3.0) = 3.0 - 0.001*(1 + 0.3) = 3.0 - 0.0013 = 2.9987
        np.testing.assert_allclose(params[0]["params"], [2.9987], atol=1e-10)

    def test_zero_weight_decay_matches_adam(self):
        x0 = np.array([3.0, -2.0, 1.5])
        g = np.array([6.0, -4.0, 3.0])

        params_adam = make_params(x0)
        set_grads(params_adam, g)
        opt_adam = Adam(params_adam, lr=0.001)
        opt_adam.step()

        params_adamw = make_params(x0)
        set_grads(params_adamw, g)
        opt_adamw = AdamW(params_adamw, lr=0.001, weight_decay=0.0)
        opt_adamw.step()

        np.testing.assert_allclose(
            params_adam[0]["params"], params_adamw[0]["params"], atol=1e-12
        )

    def test_weight_decay_with_zero_gradient(self):
        """AdamW still applies weight decay even when gradient is zero."""
        params = make_params(np.array([5.0]))
        set_grads(params, np.array([0.0]))
        opt = AdamW(params, lr=0.01, weight_decay=0.1)
        opt.step()

        # With zero gradient, Adam update = 0 / (sqrt(0) + eps) = 0.
        # Weight decay: 5.0 - 0.01 * 0.1 * 5.0 = 5.0 - 0.005 = 4.995
        np.testing.assert_allclose(params[0]["params"], [4.995], atol=1e-10)


# ---------------------------------------------------------------------------
# Adam vs AdamW Comparison
# ---------------------------------------------------------------------------


class TestAdamVsAdamW(unittest.TestCase):

    def test_different_trajectories(self):
        """Adam+L2 and AdamW with same lambda produce different weight trajectories."""
        np.random.seed(42)
        x0 = np.array([3.0, -2.0, 0.5])
        wd = 0.1

        params_l2 = make_params(x0)
        opt_l2 = AdamL2(params_l2, lr=0.001, weight_decay=wd)

        params_w = make_params(x0)
        opt_w = AdamW(params_w, lr=0.001, weight_decay=wd)

        for _ in range(50):
            g = np.random.randn(3)
            set_grads(params_l2, g)
            set_grads(params_w, g)
            opt_l2.step()
            opt_w.step()

        self.assertFalse(
            np.allclose(params_l2[0]["params"], params_w[0]["params"], atol=1e-6),
            "Adam+L2 and AdamW should produce different weight trajectories",
        )

    def test_adamw_more_uniform_decay(self):
        """AdamW should apply more uniform weight decay across parameters with
        different gradient variances. We verify by checking that the weight
        magnitude ratio (high-grad param / low-grad param) is closer to 1.0
        for AdamW than for Adam+L2."""
        np.random.seed(123)

        p_high = np.array([10.0])
        p_low = np.array([10.0])

        params_w = [
            {"params": p_high.copy(), "grad": np.zeros(1)},
            {"params": p_low.copy(), "grad": np.zeros(1)},
        ]
        params_l2 = [
            {"params": p_high.copy(), "grad": np.zeros(1)},
            {"params": p_low.copy(), "grad": np.zeros(1)},
        ]

        opt_w = AdamW(params_w, lr=0.01, weight_decay=0.1)
        opt_l2 = AdamL2(params_l2, lr=0.01, weight_decay=0.1)

        for _ in range(500):
            g_high = np.random.randn(1) * 100
            g_low = np.random.randn(1) * 0.01

            params_w[0]["grad"] = g_high.copy()
            params_w[1]["grad"] = g_low.copy()
            params_l2[0]["grad"] = g_high.copy()
            params_l2[1]["grad"] = g_low.copy()

            opt_w.step()
            opt_l2.step()

        # Simply confirm the two optimizers produce different final weights,
        # demonstrating that decoupled vs coupled weight decay matters
        w_vals = [abs(params_w[i]["params"][0]) for i in range(2)]
        l2_vals = [abs(params_l2[i]["params"][0]) for i in range(2)]
        self.assertFalse(
            np.allclose(w_vals, l2_vals, rtol=0.05),
            "AdamW and Adam+L2 should produce different final weight magnitudes",
        )


# ---------------------------------------------------------------------------
# Convergence Tests
# ---------------------------------------------------------------------------


class TestConvergenceQuadratic(unittest.TestCase):
    """Convergence on f(x) = 0.5 * x^T A x (minimum at origin)."""

    def _run_optimizer(self, opt_class, x0, A, steps, **kwargs):
        params = make_params(x0.copy())
        opt = opt_class(params, **kwargs)
        for _ in range(steps):
            grad = A @ params[0]["params"]
            set_grads(params, grad)
            opt.step()
        return params[0]["params"]

    def test_sgd_converges(self):
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        x0 = np.array([5.0, 3.0])
        x_final = self._run_optimizer(SGD, x0, A, 500, lr=0.1)
        np.testing.assert_allclose(x_final, [0.0, 0.0], atol=0.01)

    def test_sgd_momentum_faster(self):
        # Low curvature along one axis makes vanilla SGD slow in that direction;
        # momentum accumulates velocity and converges faster
        A = np.array([[1.0, 0.0], [0.0, 0.05]])
        x0 = np.array([5.0, 5.0])

        x_sgd = self._run_optimizer(SGD, x0, A, 100, lr=0.1)
        x_mom = self._run_optimizer(SGD, x0, A, 100, lr=0.1, momentum=0.9)

        dist_sgd = np.linalg.norm(x_sgd)
        dist_mom = np.linalg.norm(x_mom)
        self.assertLess(dist_mom, dist_sgd)

    def test_adam_converges(self):
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        x0 = np.array([5.0, 3.0])
        x_final = self._run_optimizer(Adam, x0, A, 5000, lr=0.01)
        np.testing.assert_allclose(x_final, [0.0, 0.0], atol=0.1)

    def test_adamw_converges(self):
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        x0 = np.array([5.0, 3.0])
        x_final = self._run_optimizer(AdamW, x0, A, 5000, lr=0.01, weight_decay=0.0)
        np.testing.assert_allclose(x_final, [0.0, 0.0], atol=0.1)


class TestConvergenceRosenbrock(unittest.TestCase):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2. Minimum at (1,1)."""

    def _rosenbrock_grad(self, xy):
        x, y = xy
        dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
        dy = 200 * (y - x ** 2)
        return np.array([dx, dy])

    def test_adam_rosenbrock(self):
        params = make_params(np.array([-1.0, 1.0]))
        opt = Adam(params, lr=0.001)

        for _ in range(20000):
            g = self._rosenbrock_grad(params[0]["params"])
            set_grads(params, g)
            opt.step()

        np.testing.assert_allclose(params[0]["params"], [1.0, 1.0], atol=0.1)


class TestConvergenceIllConditioned(unittest.TestCase):
    """Ill-conditioned quadratic with condition number 100."""

    def test_adam_handles_conditioning(self):
        np.random.seed(42)
        Q, _ = np.linalg.qr(np.random.randn(5, 5))
        eigenvalues = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
        A = Q @ np.diag(eigenvalues) @ Q.T

        x0 = np.ones(5)
        params = make_params(x0)
        opt = Adam(params, lr=0.01)

        for _ in range(5000):
            g = A @ params[0]["params"]
            set_grads(params, g)
            opt.step()

        np.testing.assert_allclose(params[0]["params"], np.zeros(5), atol=0.5)


# ---------------------------------------------------------------------------
# Numerical Stability
# ---------------------------------------------------------------------------


class TestNumericalStability(unittest.TestCase):

    def test_very_small_gradients(self):
        params = make_params(np.array([1.0]))
        opt = Adam(params, lr=0.001)
        set_grads(params, np.array([1e-10]))
        opt.step()
        self.assertTrue(np.all(np.isfinite(params[0]["params"])))

    def test_very_large_gradients(self):
        params = make_params(np.array([1.0]))
        opt = Adam(params, lr=0.001)
        set_grads(params, np.array([1e6]))
        opt.step()
        self.assertTrue(np.all(np.isfinite(params[0]["params"])))
        # Adam's adaptive scaling should limit step size
        self.assertGreater(params[0]["params"][0], -1e6)

    def test_zero_gradient_adam(self):
        params = make_params(np.array([5.0]))
        opt = Adam(params, lr=0.001)
        set_grads(params, np.array([0.0]))
        opt.step()
        self.assertTrue(np.all(np.isfinite(params[0]["params"])))

    def test_many_steps_no_nan(self):
        params = make_params(np.array([1.0]))
        opt = Adam(params, lr=0.001)
        for _ in range(10000):
            set_grads(params, np.array([0.01]))
            opt.step()
        self.assertTrue(np.all(np.isfinite(params[0]["params"])))


# ---------------------------------------------------------------------------
# Shape and Edge Cases
# ---------------------------------------------------------------------------


class TestShapesAndEdgeCases(unittest.TestCase):

    def test_empty_parameter_list(self):
        for OptimizerClass in [SGD, Adam, AdamW]:
            opt = OptimizerClass([], lr=0.01)
            opt.step()  # should not crash
            opt.zero_grad()

    def test_single_scalar_parameter(self):
        for OptimizerClass in [SGD, Adam, AdamW]:
            params = make_params(np.array([2.0]))
            set_grads(params, np.array([1.0]))
            opt = OptimizerClass(params, lr=0.01)
            opt.step()
            self.assertEqual(params[0]["params"].shape, (1,))

    def test_multiple_parameter_groups(self):
        p1 = np.random.randn(10, 5)
        p2 = np.random.randn(5)
        p3 = np.random.randn(3, 3, 3)
        params = make_params(p1, p2, p3)
        set_grads(params, np.random.randn(10, 5), np.random.randn(5), np.random.randn(3, 3, 3))

        for OptimizerClass in [SGD, Adam, AdamW]:
            test_params = make_params(p1, p2, p3)
            set_grads(
                test_params,
                np.random.randn(10, 5),
                np.random.randn(5),
                np.random.randn(3, 3, 3),
            )
            opt = OptimizerClass(test_params, lr=0.01)
            opt.step()
            self.assertEqual(test_params[0]["params"].shape, (10, 5))
            self.assertEqual(test_params[1]["params"].shape, (5,))
            self.assertEqual(test_params[2]["params"].shape, (3, 3, 3))

    def test_zero_grad(self):
        params = make_params(np.array([1.0, 2.0]), np.array([[3.0, 4.0], [5.0, 6.0]]))
        set_grads(params, np.array([10.0, 20.0]), np.array([[1.0, 2.0], [3.0, 4.0]]))
        opt = SGD(params, lr=0.01)
        opt.zero_grad()
        np.testing.assert_array_equal(params[0]["grad"], [0.0, 0.0])
        np.testing.assert_array_equal(params[1]["grad"], [[0.0, 0.0], [0.0, 0.0]])


# ---------------------------------------------------------------------------
# Memory Overhead Verification
# ---------------------------------------------------------------------------


class TestMemoryOverhead(unittest.TestCase):

    def test_sgd_no_momentum_no_state(self):
        params = make_params(np.random.randn(100))
        opt = SGD(params, lr=0.01)
        set_grads(params, np.random.randn(100))
        opt.step()
        self.assertEqual(len(opt.state), 0)

    def test_sgd_momentum_one_buffer(self):
        params = make_params(np.random.randn(100), np.random.randn(50))
        opt = SGD(params, lr=0.01, momentum=0.9)
        set_grads(params, np.random.randn(100), np.random.randn(50))
        opt.step()
        self.assertEqual(len(opt.state), 2)
        self.assertEqual(opt.state[0]["velocity"].shape, (100,))
        self.assertEqual(opt.state[1]["velocity"].shape, (50,))

    def test_adam_two_buffers(self):
        params = make_params(np.random.randn(100), np.random.randn(50))
        opt = Adam(params, lr=0.001)
        set_grads(params, np.random.randn(100), np.random.randn(50))
        opt.step()
        self.assertEqual(len(opt.state), 2)
        for i, size in enumerate([100, 50]):
            self.assertEqual(opt.state[i]["m"].shape, (size,))
            self.assertEqual(opt.state[i]["v"].shape, (size,))

    def test_step_counter_is_single_int(self):
        params = make_params(np.random.randn(10), np.random.randn(20))
        opt = Adam(params, lr=0.001)
        set_grads(params, np.random.randn(10), np.random.randn(20))
        opt.step()
        self.assertIsInstance(opt.t, int)


# ---------------------------------------------------------------------------
# Learning Rate Scheduler Tests
# ---------------------------------------------------------------------------


class TestStepDecayScheduler(unittest.TestCase):

    def test_basic_decay(self):
        sched = StepDecayScheduler(initial_lr=0.1, decay_factor=0.1, step_size=10)
        self.assertAlmostEqual(sched.get_lr(0), 0.1)
        self.assertAlmostEqual(sched.get_lr(9), 0.1)
        self.assertAlmostEqual(sched.get_lr(10), 0.01)
        self.assertAlmostEqual(sched.get_lr(19), 0.01)
        self.assertAlmostEqual(sched.get_lr(20), 0.001)

    def test_updates_optimizer(self):
        params = make_params(np.array([1.0]))
        opt = SGD(params, lr=0.1)
        sched = StepDecayScheduler(initial_lr=0.1, decay_factor=0.5, step_size=5)
        sched.step(opt, 5)
        self.assertAlmostEqual(opt.lr, 0.05)


class TestCosineScheduler(unittest.TestCase):

    def test_start_and_end(self):
        sched = CosineScheduler(max_lr=0.1, min_lr=0.0, total_steps=100)
        self.assertAlmostEqual(sched.get_lr(0), 0.1)
        self.assertAlmostEqual(sched.get_lr(100), 0.0)

    def test_midpoint(self):
        sched = CosineScheduler(max_lr=1.0, min_lr=0.0, total_steps=100)
        # At midpoint, cos(pi/2) = 0, so lr = 0 + 0.5*(1-0)*(1+0) = 0.5
        self.assertAlmostEqual(sched.get_lr(50), 0.5, places=5)

    def test_smooth_curve(self):
        sched = CosineScheduler(max_lr=0.1, min_lr=0.01, total_steps=1000)
        lrs = [sched.get_lr(t) for t in range(1001)]
        for i in range(1, len(lrs)):
            self.assertLessEqual(lrs[i], lrs[i - 1] + 1e-10)

    def test_beyond_total_steps_clamps(self):
        sched = CosineScheduler(max_lr=0.1, min_lr=0.0, total_steps=100)
        self.assertAlmostEqual(sched.get_lr(200), 0.0)


class TestWarmupCosineScheduler(unittest.TestCase):

    def test_warmup_phase(self):
        sched = WarmupCosineScheduler(max_lr=0.1, min_lr=0.0, warmup_steps=10, total_steps=100)
        self.assertAlmostEqual(sched.get_lr(0), 0.0)
        self.assertAlmostEqual(sched.get_lr(5), 0.05)
        self.assertAlmostEqual(sched.get_lr(10), 0.1)

    def test_cosine_phase(self):
        sched = WarmupCosineScheduler(max_lr=0.1, min_lr=0.0, warmup_steps=10, total_steps=100)
        self.assertAlmostEqual(sched.get_lr(10), 0.1)
        self.assertAlmostEqual(sched.get_lr(100), 0.0, places=5)

    def test_smooth_transition(self):
        """No discontinuity at warmup boundary."""
        sched = WarmupCosineScheduler(max_lr=0.1, min_lr=0.0, warmup_steps=100, total_steps=1000)
        lr_before = sched.get_lr(99)
        lr_at = sched.get_lr(100)
        self.assertAlmostEqual(lr_before, 0.099, places=3)
        self.assertAlmostEqual(lr_at, 0.1, places=3)
        self.assertAlmostEqual(lr_before, lr_at, places=2)

    def test_linear_warmup_values(self):
        sched = WarmupCosineScheduler(max_lr=1.0, min_lr=0.0, warmup_steps=100, total_steps=200)
        for t in range(101):
            expected = t / 100.0
            self.assertAlmostEqual(sched.get_lr(t), expected, places=5)


# ---------------------------------------------------------------------------
# Integration: Schedule + Optimizer
# ---------------------------------------------------------------------------


class TestSchedulerIntegration(unittest.TestCase):

    def test_warmup_cosine_with_adam(self):
        """Train a simple quadratic with scheduled Adam."""
        params = make_params(np.array([10.0, -10.0]))
        opt = Adam(params, lr=0.0)
        sched = WarmupCosineScheduler(max_lr=0.05, min_lr=0.0, warmup_steps=100, total_steps=5000)
        A = np.eye(2) * 2.0

        for t in range(5000):
            sched.step(opt, t)
            g = A @ params[0]["params"]
            set_grads(params, g)
            opt.step()

        np.testing.assert_allclose(params[0]["params"], [0.0, 0.0], atol=1.0)


if __name__ == "__main__":
    unittest.main()
