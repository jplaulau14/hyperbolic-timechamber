"""
Activation Functions Demo — Visualizations, comparisons, and gradient analysis.

Generates:
- viz/*.png — Individual visualization files
- report.pdf — Comprehensive PDF report
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from implementation import ReLU, LeakyReLU, Sigmoid, Tanh, GELU, SiLU, gradient_check

SEED = 42
np.random.seed(SEED)

VIZ_DIR = Path(__file__).parent / "viz"
VIZ_DIR.mkdir(exist_ok=True)
REPORT_PATH = Path(__file__).parent / "report.pdf"

COLORS = {
    "ReLU": "#e74c3c",
    "LeakyReLU": "#e67e22",
    "Sigmoid": "#3498db",
    "Tanh": "#27ae60",
    "GELU": "#9b59b6",
    "SiLU": "#f39c12",
    "GELU (exact)": "#8e44ad",
    "GELU (approx)": "#9b59b6",
}

ALL_ACTIVATIONS = [
    ("ReLU", ReLU()),
    ("LeakyReLU", LeakyReLU(alpha=0.01)),
    ("Sigmoid", Sigmoid()),
    ("Tanh", Tanh()),
    ("GELU", GELU(approximate=True)),
    ("SiLU", SiLU()),
]


def example_1_forward_functions():
    """Plot all activation forward functions side-by-side."""
    print("=" * 60)
    print("Example 1: All Activation Forward Functions")
    print("=" * 60)

    x = np.linspace(-5, 5, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (name, act) in enumerate(ALL_ACTIVATIONS):
        y = act.forward(x)
        ax = axes[idx]
        ax.plot(x, y, color=COLORS[name], linewidth=2.2)
        ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
        print(f"  {name}: f(0)={act.forward(np.array([0.0]))[0]:.6f}, "
              f"f(-3)={act.forward(np.array([-3.0]))[0]:.6f}, "
              f"f(3)={act.forward(np.array([3.0]))[0]:.6f}")

    fig.suptitle("Activation Functions — Forward Pass", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(VIZ_DIR / "01_forward_functions.png", dpi=150)
    plt.close(fig)

    fig_overlay, ax = plt.subplots(figsize=(10, 6))
    for name, act in ALL_ACTIVATIONS:
        y = act.forward(x)
        ax.plot(x, y, color=COLORS[name], linewidth=2, label=name)
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.set_title("All Activation Functions Overlaid", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 5)
    fig_overlay.tight_layout()
    fig_overlay.savefig(VIZ_DIR / "01b_forward_overlay.png", dpi=150)
    plt.close(fig_overlay)

    print()
    return [VIZ_DIR / "01_forward_functions.png", VIZ_DIR / "01b_forward_overlay.png"]


def example_2_derivatives():
    """Plot all activation derivatives side-by-side."""
    print("=" * 60)
    print("Example 2: All Activation Derivatives (Backward Pass)")
    print("=" * 60)

    x = np.linspace(-5, 5, 1000)
    grad_output = np.ones_like(x)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (name, act) in enumerate(ALL_ACTIVATIONS):
        act.forward(x)
        dy = act.backward(grad_output)
        ax = axes[idx]
        ax.plot(x, dy, color=COLORS[name], linewidth=2.2)
        ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_title(f"{name} derivative", fontsize=14, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("f'(x)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)

        act.forward(np.array([0.0]))
        grad_at_zero = act.backward(np.array([1.0]))[0]
        print(f"  {name}: f'(0) = {grad_at_zero:.6f}")

    fig.suptitle("Activation Derivatives — Backward Pass", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(VIZ_DIR / "02_derivatives.png", dpi=150)
    plt.close(fig)

    fig_overlay, ax = plt.subplots(figsize=(10, 6))
    for name, act in ALL_ACTIVATIONS:
        act.forward(x)
        dy = act.backward(grad_output)
        ax.plot(x, dy, color=COLORS[name], linewidth=2, label=name)
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f'(x)", fontsize=12)
    ax.set_title("All Activation Derivatives Overlaid", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.5, 1.5)
    fig_overlay.tight_layout()
    fig_overlay.savefig(VIZ_DIR / "02b_derivatives_overlay.png", dpi=150)
    plt.close(fig_overlay)

    print()
    return [VIZ_DIR / "02_derivatives.png", VIZ_DIR / "02b_derivatives_overlay.png"]


def example_3_dying_relu():
    """Demonstrate the dying ReLU problem vs alternatives."""
    print("=" * 60)
    print("Example 3: Dying ReLU Demonstration")
    print("=" * 60)

    np.random.seed(SEED)
    negative_inputs = np.random.uniform(-3, -0.1, size=500)
    mixed_inputs = np.concatenate([negative_inputs, np.random.uniform(0.1, 3, size=500)])
    np.random.shuffle(mixed_inputs)

    activations_to_compare = [
        ("ReLU", ReLU()),
        ("LeakyReLU (0.01)", LeakyReLU(alpha=0.01)),
        ("GELU", GELU(approximate=True)),
        ("SiLU", SiLU()),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, act) in enumerate(activations_to_compare):
        ax = axes[idx // 2][idx % 2]

        act.forward(negative_inputs)
        grads_neg = act.backward(np.ones_like(negative_inputs))

        act.forward(mixed_inputs)
        grads_mixed = act.backward(np.ones_like(mixed_inputs))

        color = COLORS.get(name.split(" ")[0], "steelblue")
        ax.hist(grads_neg, bins=50, alpha=0.6, color="#e74c3c", label="Negative inputs only", density=True)
        ax.hist(grads_mixed, bins=50, alpha=0.6, color=color, label="Mixed inputs", density=True)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Gradient value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        dead_frac_neg = np.mean(np.abs(grads_neg) < 1e-10)
        dead_frac_mixed = np.mean(np.abs(grads_mixed) < 1e-10)
        print(f"  {name}:")
        print(f"    Zero-gradient fraction (negative inputs): {dead_frac_neg:.1%}")
        print(f"    Zero-gradient fraction (mixed inputs):    {dead_frac_mixed:.1%}")

    fig.suptitle("Dying ReLU Problem — Gradient Distributions",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(VIZ_DIR / "03_dying_relu.png", dpi=150)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(10, 6))
    x_range = np.linspace(-4, 1, 500)
    grad_ones = np.ones_like(x_range)
    for name, act in activations_to_compare:
        act.forward(x_range)
        grads = act.backward(grad_ones)
        color = COLORS.get(name.split(" ")[0], "steelblue")
        ax.plot(x_range, grads, color=color, linewidth=2.2, label=name)
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("x (negative region)", fontsize=12)
    ax.set_ylabel("Gradient f'(x)", fontsize=12)
    ax.set_title("Gradient in the Negative Region — ReLU Dies, Others Survive",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(VIZ_DIR / "03b_dying_relu_gradients.png", dpi=150)
    plt.close(fig2)

    print()
    return [VIZ_DIR / "03_dying_relu.png", VIZ_DIR / "03b_dying_relu_gradients.png"]


def example_4_gelu_exact_vs_approx():
    """Compare GELU exact (erf) vs tanh approximation."""
    print("=" * 60)
    print("Example 4: GELU Exact vs Approximate Comparison")
    print("=" * 60)

    x = np.linspace(-5, 5, 2000)
    grad_output = np.ones_like(x)

    gelu_exact = GELU(approximate=False)
    gelu_approx = GELU(approximate=True)

    y_exact = gelu_exact.forward(x)
    y_approx = gelu_approx.forward(x.copy())

    gelu_exact.forward(x)
    dy_exact = gelu_exact.backward(grad_output)
    gelu_approx.forward(x.copy())
    dy_approx = gelu_approx.backward(grad_output)

    forward_diff = np.abs(y_exact - y_approx)
    backward_diff = np.abs(dy_exact - dy_approx)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.plot(x, y_exact, color="#8e44ad", linewidth=2.2, label="Exact (erf)")
    ax.plot(x, y_approx, color="#f39c12", linewidth=2.2, linestyle="--", label="Approx (tanh)")
    ax.set_title("GELU Forward: Exact vs Approximate", fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, forward_diff, color="#e74c3c", linewidth=1.5)
    ax.set_title("Forward Absolute Difference", fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("|exact - approx|")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    ax = axes[1, 0]
    ax.plot(x, dy_exact, color="#8e44ad", linewidth=2.2, label="Exact (erf)")
    ax.plot(x, dy_approx, color="#f39c12", linewidth=2.2, linestyle="--", label="Approx (tanh)")
    ax.set_title("GELU Derivative: Exact vs Approximate", fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("f'(x)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, backward_diff, color="#e74c3c", linewidth=1.5)
    ax.set_title("Derivative Absolute Difference", fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("|exact' - approx'|")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    fig.suptitle("GELU: Exact (erf) vs Tanh Approximation",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(VIZ_DIR / "04_gelu_exact_vs_approx.png", dpi=150)
    plt.close(fig)

    max_fwd_diff = np.max(forward_diff)
    max_bwd_diff = np.max(backward_diff)
    print(f"  Max forward  |exact - approx|: {max_fwd_diff:.8f}")
    print(f"  Max backward |exact - approx|: {max_bwd_diff:.8f}")
    print(f"  Forward  difference < 0.005 for x in [-5,5]: {max_fwd_diff < 0.005}")
    print(f"  Backward difference < 0.01  for x in [-5,5]: {max_bwd_diff < 0.01}")
    print()
    return [VIZ_DIR / "04_gelu_exact_vs_approx.png"]


def example_5_vanishing_gradients():
    """Compare vanishing gradient problem across activation families."""
    print("=" * 60)
    print("Example 5: Vanishing Gradient Comparison")
    print("=" * 60)

    x = np.linspace(-8, 8, 2000)
    grad_output = np.ones_like(x)

    classic = [("Sigmoid", Sigmoid()), ("Tanh", Tanh())]
    modern = [("ReLU", ReLU()), ("GELU", GELU()), ("SiLU", SiLU())]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for name, act in classic:
        act.forward(x)
        dy = act.backward(grad_output)
        ax.plot(x, dy, color=COLORS[name], linewidth=2.2, label=name)
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_title("Classic Activations — Gradients Vanish at Extremes",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("f'(x)", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, act in modern:
        act.forward(x)
        dy = act.backward(grad_output)
        ax.plot(x, dy, color=COLORS[name], linewidth=2.2, label=name)
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_title("Modern Activations — Gradients Survive for x > 0",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("f'(x)", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Vanishing Gradient: Classic vs Modern Activations",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_vanishing_gradients.png", dpi=150)
    plt.close(fig)

    test_points = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    grad_ones_small = np.ones_like(test_points)
    all_acts = classic + modern
    print(f"  {'Activation':<12} | " + " | ".join(f"x={v:5.1f}" for v in test_points))
    print("  " + "-" * 80)
    for name, act in all_acts:
        act.forward(test_points)
        grads = act.backward(grad_ones_small)
        row = " | ".join(f"{g:7.4f}" for g in grads)
        print(f"  {name:<12} | {row}")

    print()
    return [VIZ_DIR / "05_vanishing_gradients.png"]


def example_6_gradient_flow_chain():
    """Simulate gradient flow through a chain of 10 activation layers."""
    print("=" * 60)
    print("Example 6: Gradient Flow Through 10 Chained Layers")
    print("=" * 60)

    np.random.seed(SEED)
    n_layers = 10
    n_samples = 1000

    activations_to_test = [
        ("Sigmoid", lambda: Sigmoid()),
        ("Tanh", lambda: Tanh()),
        ("ReLU", lambda: ReLU()),
        ("GELU", lambda: GELU(approximate=True)),
        ("SiLU", lambda: SiLU()),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_mean = axes[0]
    ax_std = axes[1]

    for name, act_factory in activations_to_test:
        layers = [act_factory() for _ in range(n_layers)]
        x = np.random.randn(n_samples)
        intermediates = [x.copy()]

        for layer in layers:
            x = layer.forward(x)
            intermediates.append(x.copy())

        grad = np.ones(n_samples)
        grad_norms = [np.mean(np.abs(grad))]

        for layer in reversed(layers):
            grad = layer.backward(grad)
            grad_norms.append(np.mean(np.abs(grad)))

        grad_norms = list(reversed(grad_norms))

        color = COLORS.get(name, "steelblue")
        layer_indices = list(range(n_layers + 1))
        ax_mean.plot(layer_indices, grad_norms, color=color, linewidth=2.2,
                     marker="o", markersize=5, label=name)

        forward_stds = [np.std(h) for h in intermediates]
        ax_std.plot(layer_indices, forward_stds, color=color, linewidth=2.2,
                    marker="s", markersize=5, label=name)

        print(f"  {name:<10}: gradient at layer 0 = {grad_norms[0]:.6e}, "
              f"at layer {n_layers} = {grad_norms[-1]:.6e}, "
              f"ratio = {grad_norms[0] / max(grad_norms[-1], 1e-30):.4e}")

    ax_mean.set_xlabel("Layer", fontsize=12)
    ax_mean.set_ylabel("Mean |gradient|", fontsize=12)
    ax_mean.set_title("Gradient Magnitude Through Layers", fontsize=13, fontweight="bold")
    ax_mean.set_yscale("log")
    ax_mean.legend(fontsize=10)
    ax_mean.grid(True, alpha=0.3)

    ax_std.set_xlabel("Layer", fontsize=12)
    ax_std.set_ylabel("Std of activations", fontsize=12)
    ax_std.set_title("Forward Activation Spread Through Layers", fontsize=13, fontweight="bold")
    ax_std.legend(fontsize=10)
    ax_std.grid(True, alpha=0.3)

    fig.suptitle("Gradient Flow: Chaining 10 Activation Layers (no weight matrices)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_gradient_flow_chain.png", dpi=150)
    plt.close(fig)

    print()
    return [VIZ_DIR / "06_gradient_flow_chain.png"]


def example_7_temperature_scaling():
    """Show how input scaling (temperature) affects activation behavior."""
    print("=" * 60)
    print("Example 7: Temperature/Scaling Effects")
    print("=" * 60)

    x_base = np.linspace(-3, 3, 500)
    temperatures = [0.5, 1.0, 2.0, 5.0]

    activations_to_show = [
        ("Sigmoid", Sigmoid()),
        ("Tanh", Tanh()),
        ("GELU", GELU(approximate=True)),
        ("SiLU", SiLU()),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    temp_colors = ["#3498db", "#27ae60", "#e67e22", "#e74c3c"]

    for idx, (name, act) in enumerate(activations_to_show):
        ax = axes[idx]
        for t_idx, temp in enumerate(temperatures):
            x_scaled = x_base * temp
            y = act.forward(x_scaled)
            ax.plot(x_base, y, color=temp_colors[t_idx], linewidth=2,
                    label=f"scale={temp}")

        ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_title(f"{name} — Effect of Input Scaling", fontsize=13, fontweight="bold")
        ax.set_xlabel("x (before scaling)")
        ax.set_ylabel("f(scale * x)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Temperature/Scaling Effects on Activations",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(VIZ_DIR / "07_temperature_scaling.png", dpi=150)
    plt.close(fig)

    print("  Sigmoid at scale=0.5: smooth transition, nearly linear near 0")
    print("  Sigmoid at scale=5.0: sharp step function")
    print("  GELU at scale=0.5: nearly linear")
    print("  GELU at scale=5.0: approximates ReLU")
    print()
    return [VIZ_DIR / "07_temperature_scaling.png"]


def example_8_numerical_stability():
    """Demonstrate that implementations handle extreme values correctly."""
    print("=" * 60)
    print("Example 8: Numerical Stability at Extreme Values")
    print("=" * 60)

    extreme_values = np.array([-1000, -500, -100, -50, -10, -1, 0, 1, 10, 50, 100, 500, 1000],
                              dtype=np.float64)

    activations_to_test = [
        ("Sigmoid", Sigmoid()),
        ("Tanh", Tanh()),
        ("GELU (approx)", GELU(approximate=True)),
        ("GELU (exact)", GELU(approximate=False)),
        ("SiLU", SiLU()),
        ("ReLU", ReLU()),
    ]

    print(f"  {'Activation':<16} | {'Input':>8} | {'Output':>14} | {'Has NaN':>7} | {'Has Inf':>7}")
    print("  " + "-" * 65)

    all_stable = True
    stability_data = {}

    for name, act in activations_to_test:
        outputs = act.forward(extreme_values)
        act.forward(extreme_values)
        grads = act.backward(np.ones_like(extreme_values))

        has_nan = np.any(np.isnan(outputs)) or np.any(np.isnan(grads))
        has_inf = np.any(np.isinf(outputs)) or np.any(np.isinf(grads))

        stability_data[name] = {
            "outputs": outputs.copy(),
            "grads": grads.copy(),
            "has_nan": has_nan,
            "has_inf": has_inf,
        }

        if has_nan or has_inf:
            all_stable = False

        for i, v in enumerate(extreme_values):
            if abs(v) >= 100:
                flag = ""
                if np.isnan(outputs[i]):
                    flag = " [NaN!]"
                elif np.isinf(outputs[i]):
                    flag = " [Inf!]"
                print(f"  {name:<16} | {v:>8.0f} | {outputs[i]:>14.6e} | "
                      f"{'Yes' if np.isnan(outputs[i]) else 'No':>7} | "
                      f"{'Yes' if np.isinf(outputs[i]) else 'No':>7}{flag}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    x_wide = np.linspace(-20, 20, 1000)

    ax = axes[0]
    for name, act in activations_to_test:
        y = act.forward(x_wide)
        color = COLORS.get(name.split(" ")[0], "steelblue")
        linestyle = "--" if "exact" in name else "-"
        ax.plot(x_wide, y, color=color, linewidth=2, label=name, linestyle=linestyle)
    ax.set_title("Forward Values Over Extended Range [-20, 20]", fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    grad_output = np.ones_like(x_wide)
    for name, act in activations_to_test:
        act.forward(x_wide)
        dy = act.backward(grad_output)
        color = COLORS.get(name.split(" ")[0], "steelblue")
        linestyle = "--" if "exact" in name else "-"
        ax.plot(x_wide, dy, color=color, linewidth=2, label=name, linestyle=linestyle)
    ax.set_title("Gradient Values Over Extended Range [-20, 20]", fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("f'(x)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Numerical Stability — Extended Range Behavior",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(VIZ_DIR / "08_numerical_stability.png", dpi=150)
    plt.close(fig)

    print(f"\n  All activations numerically stable: {all_stable}")
    print()
    return [VIZ_DIR / "08_numerical_stability.png"]


def example_bonus_gradient_check():
    """Run gradient checks on all activations to verify backward correctness."""
    print("=" * 60)
    print("Bonus: Gradient Check — Analytical vs Numerical")
    print("=" * 60)

    test_inputs = np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0])

    for name, act in ALL_ACTIVATIONS:
        result = gradient_check(act, test_inputs)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {name:<12}: max_abs_err={result['max_abs_error']:.2e}, "
              f"max_rel_err={result['max_rel_error']:.2e}  [{status}]")

    print()


def generate_pdf_report(all_figures):
    """Generate comprehensive PDF report with all visualizations."""
    print("=" * 60)
    print("Generating PDF Report")
    print("=" * 60)

    with PdfPages(REPORT_PATH) as pdf:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis("off")
        ax.text(0.5, 0.7, "Activation Functions", fontsize=28, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.55, "Comprehensive Demo and Visualization Report",
                fontsize=16, ha="center", va="center", transform=ax.transAxes,
                color="gray")
        ax.text(0.5, 0.40,
                "ReLU | LeakyReLU | Sigmoid | Tanh | GELU | SiLU/Swish",
                fontsize=13, ha="center", va="center", transform=ax.transAxes,
                color="#555555")
        ax.text(0.5, 0.25, f"Seed: {SEED}  |  NumPy from-scratch implementation",
                fontsize=11, ha="center", va="center", transform=ax.transAxes,
                color="#888888")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis("off")
        summary_text = (
            "Summary of Key Findings\n"
            "========================\n\n"
            "1. Forward Functions: All 6 activations plotted over [-5, 5].\n"
            "   GELU and SiLU show smooth, non-monotonic transitions.\n\n"
            "2. Derivatives: Sigmoid and Tanh derivatives peak at 0.25 and 1.0\n"
            "   respectively, vanishing for large |x|. ReLU is discontinuous.\n\n"
            "3. Dying ReLU: ReLU has 100% zero gradients for negative inputs.\n"
            "   LeakyReLU, GELU, and SiLU maintain non-zero gradients.\n\n"
            "4. GELU Approximation: Tanh approximation matches exact (erf)\n"
            "   with max absolute error < 0.005 over [-5, 5].\n\n"
            "5. Vanishing Gradients: Sigmoid and Tanh saturate at extremes.\n"
            "   Modern activations (ReLU, GELU, SiLU) maintain gradient\n"
            "   flow for positive inputs.\n\n"
            "6. Gradient Flow: Through 10 chained layers, Sigmoid gradients\n"
            "   vanish exponentially. GELU/SiLU maintain better flow.\n\n"
            "7. Temperature Scaling: Higher input scaling sharpens activation\n"
            "   transitions — Sigmoid becomes step-like, GELU becomes ReLU-like.\n\n"
            "8. Numerical Stability: All implementations handle extreme values\n"
            "   (|x| up to 1000) without NaN or Inf."
        )
        ax.text(0.05, 0.95, summary_text, fontsize=11, va="top", ha="left",
                transform=ax.transAxes, family="monospace")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        titles = [
            "Example 1: Forward Functions (Individual)",
            "Example 1b: Forward Functions (Overlay)",
            "Example 2: Derivatives (Individual)",
            "Example 2b: Derivatives (Overlay)",
            "Example 3: Dying ReLU — Gradient Distributions",
            "Example 3b: Dying ReLU — Gradient in Negative Region",
            "Example 4: GELU Exact vs Approximate",
            "Example 5: Vanishing Gradient Comparison",
            "Example 6: Gradient Flow Through 10 Layers",
            "Example 7: Temperature/Scaling Effects",
            "Example 8: Numerical Stability",
        ]

        for fig_path, title in zip(all_figures, titles):
            if fig_path.exists():
                img = plt.imread(str(fig_path))
                fig, ax = plt.subplots(figsize=(11, 8))
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"  Report saved to: {REPORT_PATH}")
    print()


def main():
    print()
    print("*" * 60)
    print("  ACTIVATION FUNCTIONS — COMPREHENSIVE DEMO")
    print(f"  Seed: {SEED}")
    print("*" * 60)
    print()

    all_figures = []

    all_figures.extend(example_1_forward_functions())
    all_figures.extend(example_2_derivatives())
    all_figures.extend(example_3_dying_relu())
    all_figures.extend(example_4_gelu_exact_vs_approx())
    all_figures.extend(example_5_vanishing_gradients())
    all_figures.extend(example_6_gradient_flow_chain())
    all_figures.extend(example_7_temperature_scaling())
    all_figures.extend(example_8_numerical_stability())
    example_bonus_gradient_check()

    generate_pdf_report(all_figures)

    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"  Visualizations: {VIZ_DIR}/")
    for f in sorted(VIZ_DIR.glob("*.png")):
        print(f"    - {f.name}")
    print(f"  PDF Report:     {REPORT_PATH}")
    print()


if __name__ == "__main__":
    main()
