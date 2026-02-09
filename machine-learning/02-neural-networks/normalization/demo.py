"""
Normalization Layers Demo — LayerNorm & RMSNorm visualizations, comparisons, and PDF report.

Generates:
- viz/*.png — Individual visualization files
- report.pdf — Comprehensive PDF report
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from implementation import LayerNorm, RMSNorm

SEED = 42
np.random.seed(SEED)

VIZ_DIR = Path(__file__).parent / "viz"
VIZ_DIR.mkdir(exist_ok=True)
REPORT_PATH = Path(__file__).parent / "report.pdf"

COLORS = {
    "blue": "#3498db",
    "red": "#e74c3c",
    "green": "#27ae60",
    "orange": "#f39c12",
    "purple": "#9b59b6",
    "steel": "steelblue",
    "coral": "coral",
}

all_figures = []


def save_fig(fig, name, title=None):
    fig.savefig(VIZ_DIR / name, dpi=150, bbox_inches="tight")
    all_figures.append({"fig_path": VIZ_DIR / name, "title": title or name})
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Example 1: LayerNorm — input vs output distributions
# ─────────────────────────────────────────────────────────────


def example_1_layernorm_distribution():
    print("=" * 60)
    print("Example 1: LayerNorm — Input vs Normalized Output")
    print("=" * 60)

    D = 256
    x = np.random.randn(1000, D) * 3.0 + 5.0
    ln = LayerNorm(D)
    y = ln.forward(x)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].hist(x.flatten(), bins=80, color=COLORS["blue"], alpha=0.7, density=True)
    axes[0].set_title("Input Distribution")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].axvline(x.mean(), color=COLORS["red"], linestyle="--", label=f"mean={x.mean():.2f}")
    axes[0].axvline(x.mean() + x.std(), color=COLORS["orange"], linestyle=":", label=f"std={x.std():.2f}")
    axes[0].axvline(x.mean() - x.std(), color=COLORS["orange"], linestyle=":")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(y.flatten(), bins=80, color=COLORS["green"], alpha=0.7, density=True)
    axes[1].set_title("LayerNorm Output Distribution")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].axvline(y.mean(), color=COLORS["red"], linestyle="--", label=f"mean={y.mean():.4f}")
    axes[1].axvline(y.mean() + y.std(), color=COLORS["orange"], linestyle=":", label=f"std={y.std():.4f}")
    axes[1].axvline(y.mean() - y.std(), color=COLORS["orange"], linestyle=":")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    per_sample_means = y.mean(axis=-1)
    per_sample_vars = y.var(axis=-1)
    axes[2].scatter(per_sample_means, per_sample_vars, alpha=0.3, s=10, color=COLORS["purple"])
    axes[2].set_xlabel("Per-Sample Mean")
    axes[2].set_ylabel("Per-Sample Variance")
    axes[2].set_title("Per-Sample Statistics After LayerNorm")
    axes[2].axhline(1.0, color=COLORS["red"], linestyle="--", alpha=0.7, label="target var=1")
    axes[2].axvline(0.0, color=COLORS["green"], linestyle="--", alpha=0.7, label="target mean=0")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("LayerNorm: Normalizes to Zero Mean, Unit Variance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "01_layernorm_distribution.png", "LayerNorm Input vs Output Distributions")

    print(f"  Input  — mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"  Output — mean: {y.mean():.6f}, std: {y.std():.6f}")
    print(f"  Per-sample mean range: [{per_sample_means.min():.6f}, {per_sample_means.max():.6f}]")
    print(f"  Per-sample var  range: [{per_sample_vars.min():.6f}, {per_sample_vars.max():.6f}]")
    print()


# ─────────────────────────────────────────────────────────────
# Example 2: RMSNorm — no mean subtraction
# ─────────────────────────────────────────────────────────────


def example_2_rmsnorm_distribution():
    print("=" * 60)
    print("Example 2: RMSNorm — No Mean Subtraction")
    print("=" * 60)

    D = 256
    x = np.random.randn(1000, D) * 3.0 + 5.0
    rms = RMSNorm(D)
    y = rms.forward(x)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].hist(x.flatten(), bins=80, color=COLORS["blue"], alpha=0.7, density=True)
    axes[0].set_title("Input Distribution")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].axvline(x.mean(), color=COLORS["red"], linestyle="--", label=f"mean={x.mean():.2f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(y.flatten(), bins=80, color=COLORS["orange"], alpha=0.7, density=True)
    axes[1].set_title("RMSNorm Output Distribution")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].axvline(y.mean(), color=COLORS["red"], linestyle="--", label=f"mean={y.mean():.4f}")
    axes[1].axvline(0.0, color="gray", linestyle=":", alpha=0.5, label="zero")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    per_sample_rms = np.sqrt(np.mean(y ** 2, axis=-1))
    per_sample_means = y.mean(axis=-1)
    axes[2].scatter(per_sample_means, per_sample_rms, alpha=0.3, s=10, color=COLORS["coral"])
    axes[2].set_xlabel("Per-Sample Mean (NOT zero-centered)")
    axes[2].set_ylabel("Per-Sample RMS")
    axes[2].set_title("Per-Sample Statistics After RMSNorm")
    axes[2].axhline(1.0, color=COLORS["red"], linestyle="--", alpha=0.7, label="target RMS=1")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("RMSNorm: Rescales by Root-Mean-Square (No Mean Subtraction)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "02_rmsnorm_distribution.png", "RMSNorm Input vs Output Distributions")

    print(f"  Input  — mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"  Output — mean: {y.mean():.6f} (NOT zero — no mean subtraction)")
    print(f"  Per-sample RMS range: [{per_sample_rms.min():.6f}, {per_sample_rms.max():.6f}]")
    print()


# ─────────────────────────────────────────────────────────────
# Example 3: LayerNorm vs RMSNorm side-by-side
# ─────────────────────────────────────────────────────────────


def example_3_layernorm_vs_rmsnorm():
    print("=" * 60)
    print("Example 3: LayerNorm vs RMSNorm Comparison")
    print("=" * 60)

    D = 128
    x = np.random.randn(500, D) * 2.0 + 3.0

    ln = LayerNorm(D)
    rms = RMSNorm(D)
    y_ln = ln.forward(x)
    y_rms = rms.forward(x)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sample_idx = 0
    feature_range = np.arange(D)

    axes[0, 0].bar(feature_range, x[sample_idx], color=COLORS["blue"], alpha=0.6, width=1.0)
    axes[0, 0].set_title("Original Input (sample 0)")
    axes[0, 0].set_xlabel("Feature Index")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(feature_range, y_ln[sample_idx], color=COLORS["green"], alpha=0.6, width=1.0, label="LayerNorm")
    axes[0, 1].bar(feature_range, y_rms[sample_idx], color=COLORS["orange"], alpha=0.4, width=1.0, label="RMSNorm")
    axes[0, 1].set_title("Normalized Output (sample 0)")
    axes[0, 1].set_xlabel("Feature Index")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    diff = y_ln[sample_idx] - y_rms[sample_idx]
    axes[1, 0].bar(feature_range, diff, color=COLORS["purple"], alpha=0.7, width=1.0)
    axes[1, 0].set_title("Difference: LayerNorm - RMSNorm (sample 0)")
    axes[1, 0].set_xlabel("Feature Index")
    axes[1, 0].set_ylabel("Difference")
    axes[1, 0].axhline(0, color="black", linewidth=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    ln_means = y_ln.mean(axis=-1)
    rms_means = y_rms.mean(axis=-1)
    axes[1, 1].hist(ln_means, bins=40, color=COLORS["green"], alpha=0.6, density=True, label="LayerNorm means")
    axes[1, 1].hist(rms_means, bins=40, color=COLORS["orange"], alpha=0.6, density=True, label="RMSNorm means")
    axes[1, 1].set_title("Per-Sample Mean Distribution")
    axes[1, 1].set_xlabel("Mean")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].axvline(0, color="black", linestyle="--", alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("LayerNorm vs RMSNorm: Same Input, Different Normalization", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "03_layernorm_vs_rmsnorm.png", "LayerNorm vs RMSNorm Comparison")

    print(f"  LayerNorm output mean (overall): {y_ln.mean():.6f}")
    print(f"  RMSNorm  output mean (overall): {y_rms.mean():.6f}")
    print(f"  Max abs difference: {np.abs(diff).max():.6f}")
    corr = np.corrcoef(y_ln.flatten(), y_rms.flatten())[0, 1]
    print(f"  Correlation between outputs: {corr:.6f}")
    print()


# ─────────────────────────────────────────────────────────────
# Example 4: Effect of epsilon
# ─────────────────────────────────────────────────────────────


def example_4_epsilon_effect():
    print("=" * 60)
    print("Example 4: Effect of Epsilon on Near-Constant Input")
    print("=" * 60)

    D = 64
    near_const = np.full((4, D), 5.0)
    near_const += np.random.randn(4, D) * 1e-8

    epsilons_ln = [1e-12, 1e-8, 1e-5, 1e-2]
    epsilons_rms = [1e-12, 1e-8, 1e-6, 1e-2]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, eps in enumerate(epsilons_ln):
        ln = LayerNorm(D, eps=eps)
        y = ln.forward(near_const)
        axes[0, i].hist(y.flatten(), bins=30, color=COLORS["green"], alpha=0.7, density=True)
        axes[0, i].set_title(f"LayerNorm eps={eps:.0e}")
        axes[0, i].set_xlabel("Output Value")
        axes[0, i].grid(True, alpha=0.3)
        has_nan = np.any(np.isnan(y)) or np.any(np.isinf(y))
        status = "NaN/Inf!" if has_nan else "OK"
        axes[0, i].text(0.05, 0.95, f"range: [{y.min():.2f}, {y.max():.2f}]\n{status}",
                        transform=axes[0, i].transAxes, va="top", fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        print(f"  LayerNorm eps={eps:.0e}: range=[{y.min():.4f}, {y.max():.4f}], has_nan={has_nan}")

    for i, eps in enumerate(epsilons_rms):
        rms = RMSNorm(D, eps=eps)
        y = rms.forward(near_const)
        axes[1, i].hist(y.flatten(), bins=30, color=COLORS["orange"], alpha=0.7, density=True)
        axes[1, i].set_title(f"RMSNorm eps={eps:.0e}")
        axes[1, i].set_xlabel("Output Value")
        axes[1, i].grid(True, alpha=0.3)
        has_nan = np.any(np.isnan(y)) or np.any(np.isinf(y))
        status = "NaN/Inf!" if has_nan else "OK"
        axes[1, i].text(0.05, 0.95, f"range: [{y.min():.2f}, {y.max():.2f}]\n{status}",
                        transform=axes[1, i].transAxes, va="top", fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        print(f"  RMSNorm  eps={eps:.0e}: range=[{y.min():.4f}, {y.max():.4f}], has_nan={has_nan}")

    fig.suptitle("Effect of Epsilon on Near-Constant Input (all values ~ 5.0 +/- 1e-8)",
                 fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("LayerNorm\nDensity")
    axes[1, 0].set_ylabel("RMSNorm\nDensity")
    fig.tight_layout()
    save_fig(fig, "04_epsilon_effect.png", "Effect of Epsilon on Near-Constant Input")

    print()
    print("  Key insight: Tiny epsilon can cause instability with near-constant inputs.")
    print("  Standard: eps=1e-5 (LayerNorm), eps=1e-6 (RMSNorm)")
    print()


# ─────────────────────────────────────────────────────────────
# Example 5: Learnable parameters (gamma/beta)
# ─────────────────────────────────────────────────────────────


def example_5_learnable_parameters():
    print("=" * 60)
    print("Example 5: Learnable Parameters — Undoing Normalization")
    print("=" * 60)

    D = 64
    x = np.random.randn(200, D) * 3.0 + 7.0

    ln_default = LayerNorm(D)
    y_default = ln_default.forward(x)

    ln_undo = LayerNorm(D)
    ln_undo.gamma = np.full(D, 3.0)
    ln_undo.beta = np.full(D, 7.0)
    y_undo = ln_undo.forward(x)

    ln_selective = LayerNorm(D)
    ln_selective.gamma = np.ones(D)
    ln_selective.gamma[:D // 2] = 5.0
    ln_selective.beta = np.zeros(D)
    ln_selective.beta[:D // 2] = 10.0
    y_selective = ln_selective.forward(x)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(x.flatten(), bins=60, color=COLORS["blue"], alpha=0.7, density=True, label="Input")
    axes[0, 0].hist(y_default.flatten(), bins=60, color=COLORS["green"], alpha=0.5, density=True, label="Default (gamma=1, beta=0)")
    axes[0, 0].set_title("Default LayerNorm")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(x.flatten(), bins=60, color=COLORS["blue"], alpha=0.7, density=True, label="Input")
    axes[0, 1].hist(y_undo.flatten(), bins=60, color=COLORS["red"], alpha=0.5, density=True, label="gamma=3, beta=7")
    axes[0, 1].set_title("Gamma/Beta Approximate Original Scale")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(y_selective[:, :D // 2].flatten(), bins=60, color=COLORS["orange"], alpha=0.6,
                     density=True, label="First half (gamma=5, beta=10)")
    axes[1, 0].hist(y_selective[:, D // 2:].flatten(), bins=60, color=COLORS["purple"], alpha=0.6,
                     density=True, label="Second half (gamma=1, beta=0)")
    axes[1, 0].set_title("Selective Scaling per Feature")
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    feature_idx = np.arange(D)
    axes[1, 1].bar(feature_idx, ln_selective.gamma, color=COLORS["green"], alpha=0.6, label="gamma", width=0.9)
    ax2 = axes[1, 1].twinx()
    ax2.bar(feature_idx, ln_selective.beta, color=COLORS["coral"], alpha=0.4, label="beta", width=0.9)
    axes[1, 1].set_title("Gamma and Beta Values Per Feature")
    axes[1, 1].set_xlabel("Feature Index")
    axes[1, 1].set_ylabel("Gamma", color=COLORS["green"])
    ax2.set_ylabel("Beta", color=COLORS["coral"])
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Learnable Parameters: gamma (scale) and beta (shift) Can Undo Normalization",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "05_learnable_parameters.png", "Learnable Parameters (gamma/beta)")

    print(f"  Default output  — mean: {y_default.mean():.4f}, std: {y_default.std():.4f}")
    print(f"  Undo output     — mean: {y_undo.mean():.4f}, std: {y_undo.std():.4f}")
    print(f"  Original input  — mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"  Selective first  half — mean: {y_selective[:, :D//2].mean():.4f}")
    print(f"  Selective second half — mean: {y_selective[:, D//2:].mean():.4f}")
    print()


# ─────────────────────────────────────────────────────────────
# Example 6: Gradient flow through stacked layers
# ─────────────────────────────────────────────────────────────


def example_6_gradient_flow():
    print("=" * 60)
    print("Example 6: Gradient Flow — With vs Without Normalization")
    print("=" * 60)

    D = 64
    num_layers = 20
    batch_size = 32

    def simulate_forward_backward(use_norm, norm_type="layernorm"):
        np.random.seed(SEED)
        x = np.random.randn(batch_size, D)
        activations = [x.copy()]
        norms = []
        weights = []

        for _ in range(num_layers):
            W = np.random.randn(D, D) * 0.5 / np.sqrt(D)
            weights.append(W)
            h = x @ W

            if use_norm:
                if norm_type == "layernorm":
                    norm = LayerNorm(D)
                else:
                    norm = RMSNorm(D)
                h = norm.forward(h)
                norms.append(norm)

            x = np.tanh(h)
            activations.append(x.copy())

        grad = np.ones_like(x)
        grad_magnitudes = [np.linalg.norm(grad)]

        for i in range(num_layers - 1, -1, -1):
            grad = grad * (1 - activations[i + 1] ** 2)

            if use_norm and norms:
                grad = norms[i].backward(grad)

            grad = grad @ weights[i].T
            grad_magnitudes.append(np.linalg.norm(grad))

        grad_magnitudes.reverse()
        return activations, grad_magnitudes

    acts_no_norm, grads_no_norm = simulate_forward_backward(use_norm=False)
    acts_ln, grads_ln = simulate_forward_backward(use_norm=True, norm_type="layernorm")
    acts_rms, grads_rms = simulate_forward_backward(use_norm=True, norm_type="rmsnorm")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    layers = range(num_layers + 1)

    act_norms_no = [np.linalg.norm(a) for a in acts_no_norm]
    act_norms_ln = [np.linalg.norm(a) for a in acts_ln]
    act_norms_rms = [np.linalg.norm(a) for a in acts_rms]
    axes[0].plot(layers, act_norms_no, "o-", color=COLORS["red"], label="No Norm", markersize=4)
    axes[0].plot(layers, act_norms_ln, "s-", color=COLORS["green"], label="LayerNorm", markersize=4)
    axes[0].plot(layers, act_norms_rms, "^-", color=COLORS["orange"], label="RMSNorm", markersize=4)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Activation Norm (log scale)")
    axes[0].set_title("Forward: Activation Magnitudes")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layers, grads_no_norm, "o-", color=COLORS["red"], label="No Norm", markersize=4)
    axes[1].plot(layers, grads_ln, "s-", color=COLORS["green"], label="LayerNorm", markersize=4)
    axes[1].plot(layers, grads_rms, "^-", color=COLORS["orange"], label="RMSNorm", markersize=4)
    axes[1].set_xlabel("Layer (backward direction: output -> input)")
    axes[1].set_ylabel("Gradient Norm (log scale)")
    axes[1].set_title("Backward: Gradient Magnitudes")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    ratio_no = [grads_no_norm[i + 1] / (grads_no_norm[i] + 1e-30) for i in range(len(grads_no_norm) - 1)]
    ratio_ln = [grads_ln[i + 1] / (grads_ln[i] + 1e-30) for i in range(len(grads_ln) - 1)]
    ratio_rms = [grads_rms[i + 1] / (grads_rms[i] + 1e-30) for i in range(len(grads_rms) - 1)]
    axes[2].plot(range(num_layers), ratio_no, "o-", color=COLORS["red"], label="No Norm", markersize=4)
    axes[2].plot(range(num_layers), ratio_ln, "s-", color=COLORS["green"], label="LayerNorm", markersize=4)
    axes[2].plot(range(num_layers), ratio_rms, "^-", color=COLORS["orange"], label="RMSNorm", markersize=4)
    axes[2].axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Ideal ratio=1")
    axes[2].set_xlabel("Layer Transition")
    axes[2].set_ylabel("Gradient Ratio (layer i+1 / layer i)")
    axes[2].set_title("Gradient Ratio per Layer")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Gradient Flow Through {num_layers} Stacked Layers", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "06_gradient_flow.png", "Gradient Flow: With vs Without Normalization")

    print(f"  No Norm  — final grad norm: {grads_no_norm[-1]:.6e}, first grad norm: {grads_no_norm[0]:.6e}")
    print(f"  LayerNorm — final grad norm: {grads_ln[-1]:.6e}, first grad norm: {grads_ln[0]:.6e}")
    print(f"  RMSNorm  — final grad norm: {grads_rms[-1]:.6e}, first grad norm: {grads_rms[0]:.6e}")
    print()


# ─────────────────────────────────────────────────────────────
# Example 7: Pre-Norm vs Post-Norm architecture diagram
# ─────────────────────────────────────────────────────────────


def example_7_prenorm_vs_postnorm():
    print("=" * 60)
    print("Example 7: Pre-Norm vs Post-Norm Architecture")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    def draw_block(ax, title, steps, x_center=0.5):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.axis("off")

        n = len(steps)
        spacing = 0.8 / (n + 1)
        y_positions = [0.9 - spacing * (i + 1) for i in range(n)]

        box_w, box_h = 0.5, 0.055
        residual_left = x_center - box_w / 2 - 0.12

        for i, (label, style, residual_start) in enumerate(steps):
            y = y_positions[i]
            if style == "norm":
                color = COLORS["green"]
                text_color = "white"
            elif style == "attn":
                color = COLORS["blue"]
                text_color = "white"
            elif style == "ffn":
                color = COLORS["orange"]
                text_color = "white"
            elif style == "add":
                color = COLORS["purple"]
                text_color = "white"
            elif style == "input":
                color = "#555555"
                text_color = "white"
            elif style == "output":
                color = "#333333"
                text_color = "white"
            else:
                color = "lightgray"
                text_color = "black"

            rect = mpatches.FancyBboxPatch(
                (x_center - box_w / 2, y - box_h / 2),
                box_w, box_h,
                boxstyle="round,pad=0.01",
                facecolor=color, edgecolor="black", linewidth=1.2
            )
            ax.add_patch(rect)
            ax.text(x_center, y, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

            if i < n - 1:
                ax.annotate("", xy=(x_center, y_positions[i + 1] + box_h / 2),
                            xytext=(x_center, y - box_h / 2),
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

            if residual_start is not None:
                start_y = y_positions[residual_start]
                ax.annotate(
                    "", xy=(x_center - box_w / 2, y),
                    xytext=(residual_left, y),
                    arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2, linestyle="--")
                )
                ax.plot([residual_left, residual_left], [start_y, y],
                        color=COLORS["red"], lw=2, linestyle="--")
                ax.annotate(
                    "", xy=(residual_left, start_y),
                    xytext=(x_center - box_w / 2, start_y),
                    arrowprops=dict(arrowstyle="-", color=COLORS["red"], lw=2, linestyle="--")
                )

    postnorm_steps = [
        ("Input x", "input", None),
        ("Self-Attention", "attn", None),
        ("Add (x + attn)", "add", 0),
        ("LayerNorm", "norm", None),
        ("FFN", "ffn", None),
        ("Add (x1 + ffn)", "add", 3),
        ("LayerNorm", "norm", None),
        ("Output", "output", None),
    ]

    prenorm_steps = [
        ("Input x", "input", None),
        ("LayerNorm", "norm", None),
        ("Self-Attention", "attn", None),
        ("Add (x + attn)", "add", 0),
        ("LayerNorm", "norm", None),
        ("FFN", "ffn", None),
        ("Add (x1 + ffn)", "add", 3),
        ("Output", "output", None),
    ]

    draw_block(axes[0], "Post-Norm (Original Transformer, BERT)", postnorm_steps)
    draw_block(axes[1], "Pre-Norm (GPT-2, LLaMA, Mistral)", prenorm_steps)

    legend_items = [
        mpatches.Patch(color=COLORS["green"], label="Normalization"),
        mpatches.Patch(color=COLORS["blue"], label="Self-Attention"),
        mpatches.Patch(color=COLORS["orange"], label="Feed-Forward Network"),
        mpatches.Patch(color=COLORS["purple"], label="Residual Addition"),
        mpatches.Patch(color=COLORS["red"], label="Residual Connection (skip)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Pre-Norm vs Post-Norm: Where Normalization Goes in a Transformer Block",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "07_prenorm_vs_postnorm.png", "Pre-Norm vs Post-Norm Architecture")

    print("  Post-Norm: Norm AFTER residual addition (original transformer)")
    print("    - Gradient must pass through norm to reach residual path")
    print("    - Can cause vanishing gradients in deep networks")
    print()
    print("  Pre-Norm: Norm BEFORE sub-layer, residual bypasses norm")
    print("    - Direct gradient highway through residual connections")
    print("    - Much more stable for deep networks (GPT-2+, LLaMA)")
    print()


# ─────────────────────────────────────────────────────────────
# Example 8: 3D input (batch, seq_len, features) — transformer-like
# ─────────────────────────────────────────────────────────────


def example_8_3d_sequence_normalization():
    print("=" * 60)
    print("Example 8: 3D Sequence Input (Transformer-like)")
    print("=" * 60)

    B, L, D = 2, 8, 32
    x = np.random.randn(B, L, D) * np.linspace(0.5, 4.0, L).reshape(1, L, 1)

    ln = LayerNorm(D)
    rms = RMSNorm(D)
    y_ln = ln.forward(x)
    y_rms = rms.forward(x)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    im0 = axes[0, 0].imshow(x[0], aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0, 0].set_title("Input (batch 0)")
    axes[0, 0].set_xlabel("Feature Dim")
    axes[0, 0].set_ylabel("Sequence Position")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(y_ln[0], aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0, 1].set_title("LayerNorm Output (batch 0)")
    axes[0, 1].set_xlabel("Feature Dim")
    axes[0, 1].set_ylabel("Sequence Position")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[0, 2].imshow(y_rms[0], aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0, 2].set_title("RMSNorm Output (batch 0)")
    axes[0, 2].set_xlabel("Feature Dim")
    axes[0, 2].set_ylabel("Sequence Position")
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    input_stds = x[0].std(axis=-1)
    ln_stds = y_ln[0].std(axis=-1)
    rms_stds = y_rms[0].std(axis=-1)
    positions = np.arange(L)
    axes[1, 0].bar(positions - 0.25, input_stds, width=0.25, color=COLORS["blue"], label="Input")
    axes[1, 0].bar(positions, ln_stds, width=0.25, color=COLORS["green"], label="LayerNorm")
    axes[1, 0].bar(positions + 0.25, rms_stds, width=0.25, color=COLORS["orange"], label="RMSNorm")
    axes[1, 0].set_xlabel("Sequence Position")
    axes[1, 0].set_ylabel("Std Dev per Position")
    axes[1, 0].set_title("Per-Position Std Dev")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    input_means = x[0].mean(axis=-1)
    ln_means = y_ln[0].mean(axis=-1)
    rms_means = y_rms[0].mean(axis=-1)
    axes[1, 1].plot(positions, input_means, "o-", color=COLORS["blue"], label="Input")
    axes[1, 1].plot(positions, ln_means, "s-", color=COLORS["green"], label="LayerNorm")
    axes[1, 1].plot(positions, rms_means, "^-", color=COLORS["orange"], label="RMSNorm")
    axes[1, 1].set_xlabel("Sequence Position")
    axes[1, 1].set_ylabel("Mean per Position")
    axes[1, 1].set_title("Per-Position Mean")
    axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.3)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    input_rms_vals = np.sqrt(np.mean(x[0] ** 2, axis=-1))
    ln_rms_vals = np.sqrt(np.mean(y_ln[0] ** 2, axis=-1))
    rms_rms_vals = np.sqrt(np.mean(y_rms[0] ** 2, axis=-1))
    axes[1, 2].plot(positions, input_rms_vals, "o-", color=COLORS["blue"], label="Input")
    axes[1, 2].plot(positions, ln_rms_vals, "s-", color=COLORS["green"], label="LayerNorm")
    axes[1, 2].plot(positions, rms_rms_vals, "^-", color=COLORS["orange"], label="RMSNorm")
    axes[1, 2].set_xlabel("Sequence Position")
    axes[1, 2].set_ylabel("RMS per Position")
    axes[1, 2].set_title("Per-Position RMS")
    axes[1, 2].axhline(1.0, color="black", linestyle="--", alpha=0.3, label="target=1")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("3D Sequence Normalization: (batch=2, seq_len=8, features=32)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "08_3d_sequence.png", "3D Sequence Normalization (Transformer-like)")

    print(f"  Input shape: {x.shape}")
    print(f"  Input scale varies by position: std range [{input_stds.min():.2f}, {input_stds.max():.2f}]")
    print(f"  After LayerNorm: std range [{ln_stds.min():.4f}, {ln_stds.max():.4f}]")
    print(f"  After RMSNorm:  std range [{rms_stds.min():.4f}, {rms_stds.max():.4f}]")
    print(f"  LayerNorm per-position mean ~ 0: {np.allclose(ln_means, 0, atol=1e-10)}")
    print(f"  RMSNorm per-position RMS ~ 1:   {np.allclose(rms_rms_vals, 1, atol=0.01)}")
    print()


# ─────────────────────────────────────────────────────────────
# PDF Report
# ─────────────────────────────────────────────────────────────


def generate_pdf_report():
    print("=" * 60)
    print("Generating PDF Report")
    print("=" * 60)

    with PdfPages(REPORT_PATH) as pdf:
        # Title page
        fig = plt.figure(figsize=(10, 7.5))
        fig.text(0.5, 0.65, "Normalization Layers", ha="center", va="center",
                 fontsize=32, fontweight="bold")
        fig.text(0.5, 0.55, "LayerNorm & RMSNorm", ha="center", va="center",
                 fontsize=22, color="gray")
        fig.text(0.5, 0.40, "Comprehensive Demo with Visualizations", ha="center", va="center",
                 fontsize=16)
        fig.text(0.5, 0.30, f"Seed: {SEED}", ha="center", va="center",
                 fontsize=12, color="gray")

        summary_lines = [
            "1. LayerNorm: normalizes to zero mean, unit variance (GPT-2, BERT)",
            "2. RMSNorm: rescales by root-mean-square, no mean subtraction (LLaMA, Mistral)",
            "3. Both behave identically at training and inference time",
            "4. Epsilon prevents division by zero for near-constant inputs",
            "5. Learnable gamma/beta allow the network to undo normalization if needed",
            "6. Normalization stabilizes gradient flow through deep networks",
            "7. Pre-Norm placement (before sub-layer) is standard in modern LLMs",
        ]
        y_start = 0.22
        for i, line in enumerate(summary_lines):
            fig.text(0.1, y_start - i * 0.03, line, ha="left", va="center", fontsize=9)

        fig.patch.set_facecolor("white")
        pdf.savefig(fig)
        plt.close(fig)

        # Summary page
        fig = plt.figure(figsize=(10, 7.5))
        fig.text(0.5, 0.92, "Summary of Findings", ha="center", va="center",
                 fontsize=20, fontweight="bold")

        findings = [
            ("LayerNorm Distribution", "Normalizes each sample to mean~0, variance~1 along feature dim."),
            ("RMSNorm Distribution", "Rescales to RMS~1 but does NOT subtract mean. Simpler, fewer ops."),
            ("LayerNorm vs RMSNorm", "Highly correlated outputs. Main difference: mean centering."),
            ("Epsilon Effect", "Too small eps can cause instability with near-constant inputs."),
            ("Learnable Parameters", "gamma/beta can recover original distribution if the network learns to."),
            ("Gradient Flow", "Normalization prevents vanishing/exploding gradients through deep layers."),
            ("Pre-Norm vs Post-Norm", "Pre-Norm creates a direct gradient highway; standard in modern LLMs."),
            ("3D Sequences", "Both norms handle (B, L, D) naturally, normalizing per-position."),
        ]

        y = 0.85
        for title, desc in findings:
            fig.text(0.08, y, f"  {title}", ha="left", va="center", fontsize=11, fontweight="bold")
            fig.text(0.08, y - 0.035, f"    {desc}", ha="left", va="center", fontsize=9, color="#444444")
            y -= 0.09

        fig.text(0.5, 0.05, "LayerNorm: GPT-2, GPT-3, BERT    |    RMSNorm: LLaMA, Mistral, Gemma",
                 ha="center", va="center", fontsize=10, style="italic", color="gray")

        fig.patch.set_facecolor("white")
        pdf.savefig(fig)
        plt.close(fig)

        for entry in all_figures:
            fig = plt.figure(figsize=(10, 7.5))
            img = plt.imread(str(entry["fig_path"]))
            ax = fig.add_axes([0.02, 0.05, 0.96, 0.88])
            ax.imshow(img)
            ax.axis("off")
            if entry["title"]:
                fig.text(0.5, 0.97, entry["title"], ha="center", va="top",
                         fontsize=14, fontweight="bold")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Report saved to: {REPORT_PATH}")
    print()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main():
    print()
    print("*" * 60)
    print("  NORMALIZATION LAYERS DEMO")
    print("  LayerNorm & RMSNorm — Visualizations and Comparisons")
    print("*" * 60)
    print()

    example_1_layernorm_distribution()
    example_2_rmsnorm_distribution()
    example_3_layernorm_vs_rmsnorm()
    example_4_epsilon_effect()
    example_5_learnable_parameters()
    example_6_gradient_flow()
    example_7_prenorm_vs_postnorm()
    example_8_3d_sequence_normalization()
    generate_pdf_report()

    print("=" * 60)
    print("All examples complete!")
    print(f"  Visualizations: {VIZ_DIR}/")
    print(f"  PDF Report:     {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
