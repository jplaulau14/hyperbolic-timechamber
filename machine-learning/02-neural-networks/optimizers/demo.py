"""
Optimizers Demo -- SGD, Momentum, Adam, AdamW with visualizations and PDF report.

Generates:
- viz/*.png -- Individual visualization files
- report.pdf -- Comprehensive PDF report
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from pathlib import Path

from implementation import (
    SGD,
    Adam,
    AdamW,
    AdamL2,
    WarmupCosineScheduler,
    CosineScheduler,
    StepDecayScheduler,
)

SEED = 42
np.random.seed(SEED)

VIZ_DIR = Path(__file__).parent / "viz"
VIZ_DIR.mkdir(exist_ok=True)

COLORS = {
    "sgd": "#e74c3c",
    "momentum": "#f39c12",
    "adam": "#3498db",
    "adamw": "#27ae60",
    "adaml2": "#9b59b6",
    "nesterov": "#e67e22",
}


# ---------------------------------------------------------------------------
# Helper: run optimizer on a function, recording trajectory
# ---------------------------------------------------------------------------

def optimize_2d(optimizer_cls, start, grad_fn, steps, **kwargs):
    """Run an optimizer on a 2D function, returning the trajectory."""
    params = [{"params": np.array(start, dtype=np.float64), "grad": np.zeros(2)}]
    opt = optimizer_cls(params, **kwargs)
    trajectory = [params[0]["params"].copy()]

    for _ in range(steps):
        params[0]["grad"] = grad_fn(params[0]["params"])
        opt.step()
        trajectory.append(params[0]["params"].copy())

    return np.array(trajectory)


def optimize_1d(optimizer_cls, start, grad_fn, steps, **kwargs):
    """Run an optimizer on a scalar function, returning parameter and loss history."""
    params = [{"params": np.array([start], dtype=np.float64), "grad": np.zeros(1)}]
    opt = optimizer_cls(params, **kwargs)
    param_hist = [start]
    loss_hist = []

    for _ in range(steps):
        x = params[0]["params"][0]
        params[0]["grad"] = grad_fn(params[0]["params"])
        opt.step()
        param_hist.append(params[0]["params"][0])

    return np.array(param_hist)


# ---------------------------------------------------------------------------
# Loss surfaces and their gradients
# ---------------------------------------------------------------------------

def beale(xy):
    x, y = xy
    t1 = (1.5 - x + x * y) ** 2
    t2 = (2.25 - x + x * y ** 2) ** 2
    t3 = (2.625 - x + x * y ** 3) ** 2
    return t1 + t2 + t3


def beale_grad(xy):
    x, y = xy
    t1 = 1.5 - x + x * y
    t2 = 2.25 - x + x * y ** 2
    t3 = 2.625 - x + x * y ** 3
    dx = 2 * t1 * (-1 + y) + 2 * t2 * (-1 + y ** 2) + 2 * t3 * (-1 + y ** 3)
    dy = 2 * t1 * x + 2 * t2 * 2 * x * y + 2 * t3 * 3 * x * y ** 2
    return np.array([dx, dy])


def quadratic_2d(xy, A):
    return 0.5 * xy @ A @ xy


def quadratic_2d_grad(xy, A):
    return A @ xy


def ravine(xy):
    """f(x,y) = x^2 + 50*y^2 -- elongated quadratic (condition number = 50)."""
    return xy[0] ** 2 + 50 * xy[1] ** 2


def ravine_grad(xy):
    return np.array([2 * xy[0], 100 * xy[1]])


# ---------------------------------------------------------------------------
# Example 1: SGD vs Momentum vs Adam on Beale function
# ---------------------------------------------------------------------------

def example_1_optimizer_trajectories():
    print("=" * 60)
    print("Example 1: Optimizer trajectories on Beale function")
    print("=" * 60)

    start = [3.0, 1.5]
    steps = 500

    traj_sgd = optimize_2d(SGD, start, beale_grad, steps, lr=0.0001)
    traj_mom = optimize_2d(SGD, start, beale_grad, steps, lr=0.0001, momentum=0.9)
    traj_adam = optimize_2d(Adam, start, beale_grad, steps, lr=0.01)

    print(f"  SGD   final: ({traj_sgd[-1][0]:.4f}, {traj_sgd[-1][1]:.4f})")
    print(f"  Mom   final: ({traj_mom[-1][0]:.4f}, {traj_mom[-1][1]:.4f})")
    print(f"  Adam  final: ({traj_adam[-1][0]:.4f}, {traj_adam[-1][1]:.4f})")
    print(f"  Beale minimum is at (3.0, 0.5)")

    x_range = np.linspace(-0.5, 4.5, 300)
    y_range = np.linspace(-0.5, 2.5, 300)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = beale(np.array([X[i, j], Y[i, j]]))

    fig, ax = plt.subplots(figsize=(10, 7))
    cs = ax.contour(X, Y, Z, levels=np.logspace(-1, 5, 30), norm=LogNorm(), cmap="coolwarm", alpha=0.6)
    ax.contourf(X, Y, Z, levels=np.logspace(-1, 5, 30), norm=LogNorm(), cmap="coolwarm", alpha=0.15)

    ax.plot(traj_sgd[:, 0], traj_sgd[:, 1], "-o", color=COLORS["sgd"],
            markersize=1.5, linewidth=1.2, label="SGD (lr=0.0001)", alpha=0.8)
    ax.plot(traj_mom[:, 0], traj_mom[:, 1], "-o", color=COLORS["momentum"],
            markersize=1.5, linewidth=1.2, label="SGD+Momentum (lr=0.0001, beta=0.9)", alpha=0.8)
    ax.plot(traj_adam[:, 0], traj_adam[:, 1], "-o", color=COLORS["adam"],
            markersize=1.5, linewidth=1.2, label="Adam (lr=0.01)", alpha=0.8)

    ax.plot(3.0, 0.5, "k*", markersize=15, zorder=5, label="Minimum (3, 0.5)")
    ax.plot(start[0], start[1], "ks", markersize=10, zorder=5, label="Start")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Optimizer Trajectories on Beale Function")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_beale_trajectories.png", dpi=150)
    plt.close(fig)

    print("  Saved: viz/01_beale_trajectories.png\n")
    return fig


# ---------------------------------------------------------------------------
# Example 2: Adam convergence on a quadratic
# ---------------------------------------------------------------------------

def example_2_adam_quadratic():
    print("=" * 60)
    print("Example 2: Adam convergence on quadratic f(x,y) = x^2 + 10*y^2")
    print("=" * 60)

    A = np.diag([2.0, 20.0])
    start = [5.0, 3.0]
    steps = 300

    loss_history = []
    params = [{"params": np.array(start, dtype=np.float64), "grad": np.zeros(2)}]
    opt = Adam(params, lr=0.1)
    trajectory = [params[0]["params"].copy()]

    for _ in range(steps):
        xy = params[0]["params"]
        loss_history.append(quadratic_2d(xy, A))
        params[0]["grad"] = quadratic_2d_grad(xy, A)
        opt.step()
        trajectory.append(params[0]["params"].copy())

    trajectory = np.array(trajectory)

    print(f"  Start:      ({start[0]:.2f}, {start[1]:.2f}), loss = {loss_history[0]:.4f}")
    print(f"  Final:      ({trajectory[-1][0]:.6f}, {trajectory[-1][1]:.6f}), loss = {loss_history[-1]:.8f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(loss_history, color=COLORS["adam"], linewidth=2)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Adam Convergence: Loss vs Step")
    axes[0].grid(True, alpha=0.3)

    x_range = np.linspace(-6, 6, 200)
    y_range = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X ** 2 + 10 * Y ** 2

    axes[1].contour(X, Y, Z, levels=30, cmap="coolwarm", alpha=0.5)
    axes[1].contourf(X, Y, Z, levels=30, cmap="coolwarm", alpha=0.1)
    axes[1].plot(trajectory[:, 0], trajectory[:, 1], "-o", color=COLORS["adam"],
                 markersize=1.5, linewidth=1.2, alpha=0.8, label="Adam path")
    axes[1].plot(0, 0, "k*", markersize=15, zorder=5)
    axes[1].plot(start[0], start[1], "ks", markersize=10, zorder=5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Adam Trajectory on Quadratic")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_adam_quadratic.png", dpi=150)
    plt.close(fig)

    print("  Saved: viz/02_adam_quadratic.png\n")
    return fig


# ---------------------------------------------------------------------------
# Example 3: Adam vs AdamW -- weight distribution comparison
# ---------------------------------------------------------------------------

def example_3_adam_vs_adamw():
    print("=" * 60)
    print("Example 3: Adam vs AdamW vs AdamL2 -- weight distributions")
    print("=" * 60)

    np.random.seed(SEED)
    n_params = 200
    steps = 1000

    def noisy_quadratic_grad(params_arr):
        return 2 * params_arr + np.random.randn(*params_arr.shape) * 0.5

    init_weights = np.random.randn(n_params) * 3.0

    results = {}
    for name, cls, kwargs in [
        ("Adam (no decay)", Adam, dict(lr=0.01)),
        ("AdamW (wd=0.1)", AdamW, dict(lr=0.01, weight_decay=0.1)),
        ("Adam+L2 (wd=0.1)", AdamL2, dict(lr=0.01, weight_decay=0.1)),
    ]:
        params = [{"params": init_weights.copy(), "grad": np.zeros(n_params)}]
        opt = cls(params, **kwargs)

        for _ in range(steps):
            params[0]["grad"] = noisy_quadratic_grad(params[0]["params"])
            opt.step()

        results[name] = params[0]["params"].copy()
        print(f"  {name:25s}  |w|_mean = {np.abs(results[name]).mean():.4f}  "
              f"|w|_std = {np.abs(results[name]).std():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = [COLORS["adam"], COLORS["adamw"], COLORS["adaml2"]]

    for ax, (name, weights), color in zip(axes, results.items(), colors):
        ax.hist(weights, bins=40, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.set_xlim(-2, 2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Weight Distributions After 1000 Steps of Noisy Gradient Descent", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_adam_vs_adamw_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  Saved: viz/03_adam_vs_adamw_weights.png\n")
    return fig


# ---------------------------------------------------------------------------
# Example 4: Learning rate scheduling -- warmup + cosine decay
# ---------------------------------------------------------------------------

def example_4_lr_scheduling():
    print("=" * 60)
    print("Example 4: Learning rate scheduling")
    print("=" * 60)

    total_steps = 1000
    warmup_steps = 100

    warmup_cosine = WarmupCosineScheduler(
        max_lr=0.001, min_lr=1e-5, warmup_steps=warmup_steps, total_steps=total_steps
    )
    cosine = CosineScheduler(max_lr=0.001, min_lr=1e-5, total_steps=total_steps)
    step_decay = StepDecayScheduler(initial_lr=0.001, decay_factor=0.5, step_size=200)

    steps_arr = np.arange(total_steps)
    wc_lrs = [warmup_cosine.get_lr(s) for s in steps_arr]
    cos_lrs = [cosine.get_lr(s) for s in steps_arr]
    sd_lrs = [step_decay.get_lr(s) for s in steps_arr]

    print(f"  Warmup+Cosine: start={wc_lrs[0]:.6f}, peak={max(wc_lrs):.6f}, end={wc_lrs[-1]:.6f}")
    print(f"  Cosine only:   start={cos_lrs[0]:.6f}, end={cos_lrs[-1]:.6f}")
    print(f"  Step decay:    start={sd_lrs[0]:.6f}, end={sd_lrs[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps_arr, wc_lrs, color=COLORS["adam"], linewidth=2, label="Warmup + Cosine")
    ax.plot(steps_arr, cos_lrs, color=COLORS["adamw"], linewidth=2, label="Cosine only", linestyle="--")
    ax.plot(steps_arr, sd_lrs, color=COLORS["sgd"], linewidth=2, label="Step decay", linestyle="-.")
    ax.axvline(warmup_steps, color="gray", linestyle=":", alpha=0.5, label=f"Warmup ends (step {warmup_steps})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedules")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_lr_schedules.png", dpi=150)
    plt.close(fig)

    print("  Saved: viz/04_lr_schedules.png\n")
    return fig


# ---------------------------------------------------------------------------
# Example 5: Momentum effect -- oscillation in a ravine
# ---------------------------------------------------------------------------

def example_5_momentum_ravine():
    print("=" * 60)
    print("Example 5: Momentum smooths oscillation in a ravine")
    print("=" * 60)

    start = [5.0, 1.0]
    steps = 200

    traj_sgd = optimize_2d(SGD, start, ravine_grad, steps, lr=0.005)
    traj_mom = optimize_2d(SGD, start, ravine_grad, steps, lr=0.005, momentum=0.9)
    traj_nest = optimize_2d(SGD, start, ravine_grad, steps, lr=0.005, momentum=0.9, nesterov=True)

    print(f"  Ravine: f(x,y) = x^2 + 50*y^2, minimum at (0,0)")
    print(f"  SGD      final: ({traj_sgd[-1][0]:.6f}, {traj_sgd[-1][1]:.6f}), "
          f"loss = {ravine(traj_sgd[-1]):.8f}")
    print(f"  Momentum final: ({traj_mom[-1][0]:.6f}, {traj_mom[-1][1]:.6f}), "
          f"loss = {ravine(traj_mom[-1]):.8f}")
    print(f"  Nesterov final: ({traj_nest[-1][0]:.6f}, {traj_nest[-1][1]:.6f}), "
          f"loss = {ravine(traj_nest[-1]):.8f}")

    x_range = np.linspace(-6, 6, 300)
    y_range = np.linspace(-1.5, 1.5, 300)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X ** 2 + 50 * Y ** 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles_data = [
        ("SGD (lr=0.005)", traj_sgd, COLORS["sgd"]),
        ("SGD + Momentum (beta=0.9)", traj_mom, COLORS["momentum"]),
        ("SGD + Nesterov (beta=0.9)", traj_nest, COLORS["nesterov"]),
    ]

    for ax, (title, traj, color) in zip(axes, titles_data):
        ax.contour(X, Y, Z, levels=30, cmap="coolwarm", alpha=0.4)
        ax.contourf(X, Y, Z, levels=30, cmap="coolwarm", alpha=0.08)
        ax.plot(traj[:80, 0], traj[:80, 1], "-o", color=color,
                markersize=2, linewidth=1, alpha=0.8)
        ax.plot(0, 0, "k*", markersize=12, zorder=5)
        ax.plot(start[0], start[1], "ks", markersize=8, zorder=5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ravine Optimization: f(x,y) = x^2 + 50y^2", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_momentum_ravine.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  Saved: viz/05_momentum_ravine.png\n")
    return fig


# ---------------------------------------------------------------------------
# Example 6: Ill-conditioned quadratic -- Adam vs SGD
# ---------------------------------------------------------------------------

def example_6_illconditioned():
    print("=" * 60)
    print("Example 6: Ill-conditioned quadratic (condition number = 100)")
    print("=" * 60)

    # Condition number = 100: eigenvalues 1 and 100
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    A = R.T @ np.diag([1.0, 100.0]) @ R

    start = [5.0, 5.0]
    steps = 500

    grad_fn = lambda xy: quadratic_2d_grad(xy, A)

    traj_sgd = optimize_2d(SGD, start, grad_fn, steps, lr=0.005)
    traj_mom = optimize_2d(SGD, start, grad_fn, steps, lr=0.005, momentum=0.9)
    traj_adam = optimize_2d(Adam, start, grad_fn, steps, lr=0.1)

    loss_sgd = [quadratic_2d(p, A) for p in traj_sgd]
    loss_mom = [quadratic_2d(p, A) for p in traj_mom]
    loss_adam = [quadratic_2d(p, A) for p in traj_adam]

    print(f"  A has eigenvalues [1, 100], condition number = 100")
    print(f"  SGD      final loss: {loss_sgd[-1]:.8f}")
    print(f"  Momentum final loss: {loss_mom[-1]:.8f}")
    print(f"  Adam     final loss: {loss_adam[-1]:.8f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].semilogy(loss_sgd[:300], color=COLORS["sgd"], linewidth=1.5, label="SGD")
    axes[0].semilogy(loss_mom[:300], color=COLORS["momentum"], linewidth=1.5, label="SGD+Momentum")
    axes[0].semilogy(loss_adam[:300], color=COLORS["adam"], linewidth=1.5, label="Adam")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Convergence on Ill-Conditioned Quadratic")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Trajectories
    x_range = np.linspace(-6, 6, 300)
    y_range = np.linspace(-6, 6, 300)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = quadratic_2d(np.array([X[i, j], Y[i, j]]), A)

    axes[1].contour(X, Y, Z, levels=40, cmap="coolwarm", alpha=0.4)
    axes[1].contourf(X, Y, Z, levels=40, cmap="coolwarm", alpha=0.08)
    n_show = 100
    axes[1].plot(traj_sgd[:n_show, 0], traj_sgd[:n_show, 1], "-o", color=COLORS["sgd"],
                 markersize=1.5, linewidth=0.8, alpha=0.7, label="SGD")
    axes[1].plot(traj_mom[:n_show, 0], traj_mom[:n_show, 1], "-o", color=COLORS["momentum"],
                 markersize=1.5, linewidth=0.8, alpha=0.7, label="SGD+Momentum")
    axes[1].plot(traj_adam[:n_show, 0], traj_adam[:n_show, 1], "-o", color=COLORS["adam"],
                 markersize=1.5, linewidth=0.8, alpha=0.7, label="Adam")
    axes[1].plot(0, 0, "k*", markersize=15, zorder=5)
    axes[1].plot(start[0], start[1], "ks", markersize=10, zorder=5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Trajectories (first 100 steps)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_illconditioned_quadratic.png", dpi=150)
    plt.close(fig)

    print("  Saved: viz/06_illconditioned_quadratic.png\n")
    return fig


# ---------------------------------------------------------------------------
# Example 7: Bias correction effect in Adam
# ---------------------------------------------------------------------------

def example_7_bias_correction():
    print("=" * 60)
    print("Example 7: Adam bias correction -- with vs without")
    print("=" * 60)

    start_val = 5.0
    steps = 50
    lr = 0.1
    beta1, beta2 = 0.9, 0.999

    def grad_fn(x):
        return 2 * x  # gradient of x^2

    # Adam WITH bias correction (standard)
    params_bc = [{"params": np.array([start_val]), "grad": np.zeros(1)}]
    opt_bc = Adam(params_bc, lr=lr, betas=(beta1, beta2))
    hist_bc = [start_val]
    m_hat_hist = []
    v_hat_hist = []

    for t in range(1, steps + 1):
        params_bc[0]["grad"] = grad_fn(params_bc[0]["params"])
        opt_bc.step()
        hist_bc.append(params_bc[0]["params"][0])
        m = opt_bc.state[0]["m"][0]
        v = opt_bc.state[0]["v"][0]
        m_hat_hist.append(m / (1 - beta1 ** t))
        v_hat_hist.append(v / (1 - beta2 ** t))

    # Adam WITHOUT bias correction (manual simulation)
    x_no_bc = start_val
    m_no, v_no = 0.0, 0.0
    hist_no_bc = [start_val]
    m_raw_hist = []
    v_raw_hist = []

    for t in range(1, steps + 1):
        g = 2 * x_no_bc
        m_no = beta1 * m_no + (1 - beta1) * g
        v_no = beta2 * v_no + (1 - beta2) * g ** 2
        # No bias correction -- use raw m, v
        x_no_bc -= lr * m_no / (np.sqrt(v_no) + 1e-8)
        hist_no_bc.append(x_no_bc)
        m_raw_hist.append(m_no)
        v_raw_hist.append(v_no)

    print(f"  After {steps} steps on f(x)=x^2, start={start_val}")
    print(f"  With bias correction:    x = {hist_bc[-1]:.8f}")
    print(f"  Without bias correction: x = {hist_no_bc[-1]:.8f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Parameter trajectory
    axes[0].plot(hist_bc, color=COLORS["adam"], linewidth=2, label="With bias correction")
    axes[0].plot(hist_no_bc, color=COLORS["sgd"], linewidth=2, linestyle="--", label="Without bias correction")
    axes[0].axhline(0, color="black", linestyle=":", alpha=0.3)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Parameter value")
    axes[0].set_title("Parameter Trajectory")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # First moment comparison
    axes[1].plot(m_hat_hist, color=COLORS["adam"], linewidth=2, label="Corrected m_hat")
    axes[1].plot(m_raw_hist, color=COLORS["sgd"], linewidth=2, linestyle="--", label="Raw m")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("First moment value")
    axes[1].set_title("First Moment (m) Estimates")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Second moment comparison
    axes[2].plot(v_hat_hist, color=COLORS["adam"], linewidth=2, label="Corrected v_hat")
    axes[2].plot(v_raw_hist, color=COLORS["sgd"], linewidth=2, linestyle="--", label="Raw v")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Second moment value")
    axes[2].set_title("Second Moment (v) Estimates")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Effect of Bias Correction in Adam (first 50 steps)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_bias_correction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Zoom into early steps
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    early = 10
    axes2[0].bar(np.arange(early) - 0.15, m_hat_hist[:early], width=0.3,
                 color=COLORS["adam"], label="Corrected m_hat", alpha=0.8)
    axes2[0].bar(np.arange(early) + 0.15, m_raw_hist[:early], width=0.3,
                 color=COLORS["sgd"], label="Raw m", alpha=0.8)
    axes2[0].set_xlabel("Step")
    axes2[0].set_ylabel("First moment")
    axes2[0].set_title("First Moment: Early Steps (bias correction effect)")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    axes2[1].bar(np.arange(early) - 0.15, v_hat_hist[:early], width=0.3,
                 color=COLORS["adam"], label="Corrected v_hat", alpha=0.8)
    axes2[1].bar(np.arange(early) + 0.15, v_raw_hist[:early], width=0.3,
                 color=COLORS["sgd"], label="Raw v", alpha=0.8)
    axes2[1].set_xlabel("Step")
    axes2[1].set_ylabel("Second moment")
    axes2[1].set_title("Second Moment: Early Steps (bias correction effect)")
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    fig2.suptitle("Bias Correction Zoomed: Steps 0-9", fontsize=13, y=1.02)
    fig2.tight_layout()
    fig2.savefig(VIZ_DIR / "07b_bias_correction_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("  Saved: viz/07_bias_correction.png")
    print("  Saved: viz/07b_bias_correction_zoom.png\n")
    return fig, fig2


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------

def generate_pdf_report():
    print("=" * 60)
    print("Generating PDF report...")
    print("=" * 60)

    pdf_path = Path(__file__).parent / "report.pdf"

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis("off")
        ax.text(0.5, 0.7, "Optimizers Demo", transform=ax.transAxes,
                ha="center", va="center", fontsize=32, fontweight="bold")
        ax.text(0.5, 0.55, "SGD, Momentum, Adam, AdamW", transform=ax.transAxes,
                ha="center", va="center", fontsize=18, color="gray")
        ax.text(0.5, 0.4, "From-scratch NumPy implementations", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
        ax.text(0.5, 0.25, f"Seed: {SEED}", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")

        summary_text = (
            "This report demonstrates optimizer behavior through 7 examples:\n\n"
            "1. Optimizer trajectories on Beale function (2D contour)\n"
            "2. Adam convergence on a quadratic (loss curve + trajectory)\n"
            "3. Adam vs AdamW vs Adam+L2 weight distributions\n"
            "4. Learning rate schedules (warmup+cosine, cosine, step decay)\n"
            "5. Momentum effect in a ravine (SGD vs Momentum vs Nesterov)\n"
            "6. Ill-conditioned quadratic (condition number = 100)\n"
            "7. Bias correction effect in early Adam steps"
        )
        ax.text(0.5, 0.05, summary_text, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Summary page
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis("off")
        ax.text(0.5, 0.95, "Key Findings", transform=ax.transAxes,
                ha="center", va="top", fontsize=24, fontweight="bold")

        findings = [
            "SGD struggles on non-convex and ill-conditioned problems due to oscillation.",
            "Momentum dampens oscillations by accumulating velocity in consistent gradient directions.",
            "Nesterov momentum provides slightly faster convergence via lookahead gradient evaluation.",
            "Adam adapts per-parameter learning rates, handling diverse gradient scales automatically.",
            "AdamW applies weight decay directly to parameters (decoupled), producing tighter weight distributions.",
            "Adam+L2 folds weight decay into gradients (coupled), resulting in non-uniform effective regularization.",
            "Bias correction is critical in early training steps -- without it, moment estimates are biased toward zero.",
            "Warmup + cosine decay is the standard LR schedule for LLM training (GPT, LLaMA, Mistral).",
        ]

        for i, finding in enumerate(findings):
            ax.text(0.05, 0.85 - i * 0.09, f"{i+1}. {finding}",
                    transform=ax.transAxes, fontsize=10, va="top", wrap=True)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Include all viz images
        viz_files = sorted(VIZ_DIR.glob("*.png"))
        for viz_file in viz_files:
            img = plt.imread(str(viz_file))
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(viz_file.stem.replace("_", " ").title(), fontsize=14, pad=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Saved: report.pdf ({len(list(VIZ_DIR.glob('*.png'))) + 2} pages)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("*" * 60)
    print("  OPTIMIZERS DEMO")
    print("  SGD | Momentum | Adam | AdamW | LR Scheduling")
    print("*" * 60)
    print()

    example_1_optimizer_trajectories()
    example_2_adam_quadratic()
    example_3_adam_vs_adamw()
    example_4_lr_scheduling()
    example_5_momentum_ravine()
    example_6_illconditioned()
    example_7_bias_correction()

    generate_pdf_report()

    print()
    print("=" * 60)
    print("All examples complete.")
    print(f"  Visualizations: {VIZ_DIR}/")
    print(f"  PDF report:     {Path(__file__).parent / 'report.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
