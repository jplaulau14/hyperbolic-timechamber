"""
Self-Attention Demo -- Visualizations, scaling analysis, and gradient flow.

Generates:
- viz/*.png -- Individual visualization files
- report.pdf -- Comprehensive PDF report
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pathlib import Path
import time
import sys

from implementation import (
    SelfAttention,
    create_causal_mask,
    scaled_dot_product_attention,
    softmax,
    count_flops,
    count_memory_bytes,
)

SEED = 42
np.random.seed(SEED)

VIZ_DIR = Path(__file__).parent / "viz"
VIZ_DIR.mkdir(exist_ok=True)
REPORT_PATH = Path(__file__).parent / "report.pdf"

BLUE = "#3498db"
RED = "#e74c3c"
GREEN = "#27ae60"
ORANGE = "#f39c12"
PURPLE = "#9b59b6"


def example_1_attention_heatmap():
    """Visualize the attention weight matrix for a small sequence."""
    print("=" * 60)
    print("Example 1: Attention Weights Heatmap")
    print("=" * 60)

    seq_len, d_model, d_k, d_v = 8, 32, 16, 16
    sa = SelfAttention(d_model=d_model, d_k=d_k, d_v=d_v)
    X = np.random.randn(1, seq_len, d_model)

    output = sa.forward(X)
    A = sa._cache["A"][0]  # (seq_len, seq_len)

    token_labels = [f"t{i}" for i in range(seq_len)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=A.max(), aspect="equal")
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(token_labels, fontsize=9)
    ax.set_yticklabels(token_labels, fontsize=9)
    ax.set_xlabel("Key Position (attended to)")
    ax.set_ylabel("Query Position (attending from)")
    ax.set_title(f"Attention Weights (seq_len={seq_len}, d_k={d_k})")
    for i in range(seq_len):
        for j in range(seq_len):
            color = "white" if A[i, j] > 0.5 * A.max() else "black"
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, label="Attention Weight")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_attention_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"  Input shape:  {X.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention matrix shape: {A.shape}")
    print(f"  Row sums (should be 1.0): {A.sum(axis=-1)}")
    print(f"  Max attention weight: {A.max():.4f}")
    print(f"  Min attention weight: {A.min():.4f}")
    print()
    return fig


def example_2_causal_vs_noncausal():
    """Side-by-side heatmaps showing the triangular mask effect."""
    print("=" * 60)
    print("Example 2: Causal vs Non-Causal Attention")
    print("=" * 60)

    seq_len, d_model, d_k, d_v = 8, 32, 16, 16
    sa = SelfAttention(d_model=d_model, d_k=d_k, d_v=d_v)
    X = np.random.randn(1, seq_len, d_model)

    output_nc = sa.forward(X, mask=None)
    A_noncausal = sa._cache["A"][0].copy()

    causal_mask = create_causal_mask(seq_len)
    output_c = sa.forward(X, mask=causal_mask)
    A_causal = sa._cache["A"][0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    token_labels = [f"t{i}" for i in range(seq_len)]

    for ax, A, title in [
        (axes[0], A_noncausal, "Non-Causal (Bidirectional)"),
        (axes[1], A_causal, "Causal (Autoregressive)"),
    ]:
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=max(A_noncausal.max(), A_causal.max()),
                       aspect="equal")
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(token_labels, fontsize=9)
        ax.set_yticklabels(token_labels, fontsize=9)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(title)
        for i in range(seq_len):
            for j in range(seq_len):
                color = "white" if A[i, j] > 0.5 * A.max() else "black"
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=axes, label="Attention Weight", shrink=0.8)
    fig.suptitle("Effect of Causal Masking on Attention Pattern", fontsize=14, y=1.02)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.88, bottom=0.1, wspace=0.25)
    fig.savefig(VIZ_DIR / "02_causal_vs_noncausal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Non-causal upper triangle sum: {np.triu(A_noncausal, k=1).sum():.4f}")
    print(f"  Causal upper triangle sum:     {np.triu(A_causal, k=1).sum():.6f} (should be ~0)")
    print(f"  Causal row sums: {A_causal.sum(axis=-1)}")
    print()
    return fig


def example_3_scaling_effect():
    """Compare attention weights with and without sqrt(d_k) scaling."""
    print("=" * 60)
    print("Example 3: Effect of Scaling by sqrt(d_k)")
    print("=" * 60)

    seq_len = 8
    d_k_values = [4, 16, 64, 256]

    fig, axes = plt.subplots(2, len(d_k_values), figsize=(16, 8))

    for col, d_k in enumerate(d_k_values):
        np.random.seed(SEED)
        Q = np.random.randn(1, seq_len, d_k)
        K = np.random.randn(1, seq_len, d_k)
        V = np.random.randn(1, seq_len, d_k)

        scores_raw = Q @ K.transpose(0, 2, 1)
        A_unscaled = softmax(scores_raw, axis=-1)[0]

        scores_scaled = scores_raw / np.sqrt(d_k)
        A_scaled = softmax(scores_scaled, axis=-1)[0]

        vmax = 1.0
        axes[0, col].imshow(A_unscaled, cmap="Reds", vmin=0, vmax=vmax, aspect="equal")
        axes[0, col].set_title(f"Unscaled (d_k={d_k})")
        axes[0, col].set_xlabel("Key")
        if col == 0:
            axes[0, col].set_ylabel("Query")

        axes[1, col].imshow(A_scaled, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")
        axes[1, col].set_title(f"Scaled (d_k={d_k})")
        axes[1, col].set_xlabel("Key")
        if col == 0:
            axes[1, col].set_ylabel("Query")

        entropy_unscaled = -np.sum(A_unscaled * np.log(A_unscaled + 1e-10), axis=-1).mean()
        entropy_scaled = -np.sum(A_scaled * np.log(A_scaled + 1e-10), axis=-1).mean()
        print(f"  d_k={d_k:3d} | Unscaled entropy: {entropy_unscaled:.4f} | "
              f"Scaled entropy: {entropy_scaled:.4f} | "
              f"Max uniform entropy: {np.log(seq_len):.4f}")

    fig.suptitle("Scaling Prevents Softmax Saturation\n"
                 "Top: Without scaling (saturates at large d_k) | "
                 "Bottom: With sqrt(d_k) scaling (stable)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_scaling_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print()
    return fig


def example_4_dk_sharpness():
    """Show how d_k dimension affects attention sharpness."""
    print("=" * 60)
    print("Example 4: Attention Sharpness vs d_k")
    print("=" * 60)

    seq_len = 16
    d_k_range = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    entropies_scaled = []
    entropies_unscaled = []
    max_weights_scaled = []
    max_weights_unscaled = []

    n_trials = 50
    for d_k in d_k_range:
        ent_s, ent_u, mw_s, mw_u = [], [], [], []
        for trial in range(n_trials):
            np.random.seed(SEED + trial)
            Q = np.random.randn(1, seq_len, d_k)
            K = np.random.randn(1, seq_len, d_k)

            scores = Q @ K.transpose(0, 2, 1)

            A_unscaled = softmax(scores, axis=-1)[0]
            A_scaled = softmax(scores / np.sqrt(d_k), axis=-1)[0]

            ent_u.append(-np.sum(A_unscaled * np.log(A_unscaled + 1e-10), axis=-1).mean())
            ent_s.append(-np.sum(A_scaled * np.log(A_scaled + 1e-10), axis=-1).mean())
            mw_u.append(A_unscaled.max(axis=-1).mean())
            mw_s.append(A_scaled.max(axis=-1).mean())

        entropies_unscaled.append(np.mean(ent_u))
        entropies_scaled.append(np.mean(ent_s))
        max_weights_unscaled.append(np.mean(mw_u))
        max_weights_scaled.append(np.mean(mw_s))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(d_k_range, entropies_scaled, "o-", color=BLUE, label="With scaling", linewidth=2)
    axes[0].plot(d_k_range, entropies_unscaled, "s--", color=RED, label="Without scaling", linewidth=2)
    axes[0].axhline(y=np.log(seq_len), color="gray", linestyle=":", alpha=0.7, label="Max entropy (uniform)")
    axes[0].set_xlabel("d_k (key dimension)")
    axes[0].set_ylabel("Mean Entropy of Attention Weights")
    axes[0].set_title("Attention Entropy vs d_k")
    axes[0].set_xscale("log", base=2)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(d_k_range, max_weights_scaled, "o-", color=BLUE, label="With scaling", linewidth=2)
    axes[1].plot(d_k_range, max_weights_unscaled, "s--", color=RED, label="Without scaling", linewidth=2)
    axes[1].axhline(y=1.0 / seq_len, color="gray", linestyle=":", alpha=0.7, label="Uniform weight")
    axes[1].set_xlabel("d_k (key dimension)")
    axes[1].set_ylabel("Mean Max Attention Weight")
    axes[1].set_title("Attention Sharpness vs d_k")
    axes[1].set_xscale("log", base=2)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"d_k Controls Attention Sharpness (seq_len={seq_len}, averaged over {n_trials} trials)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_dk_sharpness.png", dpi=150)
    plt.close(fig)

    for i, d_k in enumerate(d_k_range):
        print(f"  d_k={d_k:4d} | Scaled entropy: {entropies_scaled[i]:.3f} | "
              f"Unscaled entropy: {entropies_unscaled[i]:.3f} | "
              f"Scaled max_w: {max_weights_scaled[i]:.3f} | "
              f"Unscaled max_w: {max_weights_unscaled[i]:.3f}")
    print()
    return fig


def example_5_memory_scaling():
    """Plot memory usage vs sequence length showing O(n^2) growth."""
    print("=" * 60)
    print("Example 5: Quadratic Memory Scaling")
    print("=" * 60)

    seq_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    d_k, d_v, batch_size = 64, 64, 1

    mem_fp32 = []
    mem_fp16 = []
    attn_matrix_bytes = []

    for n in seq_lengths:
        m32 = count_memory_bytes(batch_size, n, d_k, d_v, dtype="float32")
        m16 = count_memory_bytes(batch_size, n, d_k, d_v, dtype="float16")
        attn_only = batch_size * n * n * 4
        mem_fp32.append(m32)
        mem_fp16.append(m16)
        attn_matrix_bytes.append(attn_only)

    mem_fp32_mb = np.array(mem_fp32) / (1024 ** 2)
    mem_fp16_mb = np.array(mem_fp16) / (1024 ** 2)
    attn_mb = np.array(attn_matrix_bytes) / (1024 ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(seq_lengths, mem_fp32_mb, "o-", color=BLUE, label="Total (FP32)", linewidth=2)
    axes[0].plot(seq_lengths, mem_fp16_mb, "s-", color=GREEN, label="Total (FP16)", linewidth=2)
    axes[0].plot(seq_lengths, attn_mb, "^--", color=RED, label="Attention matrix only (FP32)", linewidth=2)
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Memory (MB)")
    axes[0].set_title("Activation Memory vs Sequence Length")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    attn_fraction = np.array(attn_matrix_bytes) / np.array(mem_fp32)
    axes[1].plot(seq_lengths, attn_fraction * 100, "o-", color=ORANGE, linewidth=2)
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Attention Matrix as % of Total Memory")
    axes[1].set_title("Attention Matrix Dominates at Long Sequences")
    axes[1].set_xscale("log", base=2)
    axes[1].axhline(y=50, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Self-Attention Memory is O(n^2) (B={batch_size}, d_k=d_v={d_k})", fontsize=13)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_memory_scaling.png", dpi=150)
    plt.close(fig)

    for i, n in enumerate(seq_lengths):
        print(f"  n={n:5d} | FP32: {mem_fp32_mb[i]:9.2f} MB | "
              f"Attn matrix: {attn_mb[i]:9.2f} MB ({attn_fraction[i]*100:.1f}%)")
    print()
    return fig


def example_6_flop_analysis():
    """Plot compute vs sequence length showing O(n^2) growth."""
    print("=" * 60)
    print("Example 6: FLOP Analysis")
    print("=" * 60)

    seq_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    d_model, d_k, d_v, batch_size = 512, 64, 64, 1

    total_flops = []
    proj_flops = []
    attn_flops = []

    for n in seq_lengths:
        total = count_flops(batch_size, n, d_model, d_k, d_v)

        proj = 2 * batch_size * n * d_model * d_k * 3 + 2 * batch_size * n * d_v * d_model
        qk = 2 * batch_size * n * n * d_k
        av = 2 * batch_size * n * n * d_v
        sm = 5 * batch_size * n * n
        attn_core = qk + av + sm

        total_flops.append(total)
        proj_flops.append(proj)
        attn_flops.append(attn_core)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(seq_lengths, np.array(total_flops) / 1e9, "o-", color=BLUE,
                 label="Total FLOPs", linewidth=2)
    axes[0].plot(seq_lengths, np.array(proj_flops) / 1e9, "s--", color=GREEN,
                 label="Projection FLOPs (O(n))", linewidth=2)
    axes[0].plot(seq_lengths, np.array(attn_flops) / 1e9, "^--", color=RED,
                 label="Attention core FLOPs (O(n^2))", linewidth=2)
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("GFLOPs")
    axes[0].set_title("Self-Attention FLOPs Breakdown")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    attn_frac = np.array(attn_flops) / np.array(total_flops)
    axes[1].bar(range(len(seq_lengths)), attn_frac * 100, color=RED, alpha=0.7,
                label="Attention core")
    axes[1].bar(range(len(seq_lengths)), (1 - attn_frac) * 100, bottom=attn_frac * 100,
                color=GREEN, alpha=0.7, label="Projections")
    axes[1].set_xticks(range(len(seq_lengths)))
    axes[1].set_xticklabels(seq_lengths, rotation=45)
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("% of Total FLOPs")
    axes[1].set_title("Attention Core Dominates at Long Sequences")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"O(n^2) Compute Growth (d_model={d_model}, d_k=d_v={d_k})", fontsize=13)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_flop_analysis.png", dpi=150)
    plt.close(fig)

    crossover = None
    for i, n in enumerate(seq_lengths):
        af = attn_frac[i] * 100
        print(f"  n={n:4d} | Total: {total_flops[i]/1e9:8.3f} GFLOPs | "
              f"Attn core: {af:.1f}% | Projections: {100-af:.1f}%")
        if crossover is None and af > 50:
            crossover = n
    if crossover:
        print(f"  Crossover point (attention > 50%): n ~ {crossover}")
    print()
    return fig


def example_7_gradient_flow():
    """Show gradient magnitudes through the attention mechanism."""
    print("=" * 60)
    print("Example 7: Gradient Flow Visualization")
    print("=" * 60)

    seq_len, d_model, d_k, d_v = 8, 32, 16, 16
    batch_size = 4

    np.random.seed(SEED)
    sa = SelfAttention(d_model=d_model, d_k=d_k, d_v=d_v)
    X = np.random.randn(batch_size, seq_len, d_model)

    output = sa.forward(X)
    grad_output = np.ones_like(output)
    grad_X = sa.backward(grad_output)

    grad_norms = {
        "W_Q": np.linalg.norm(sa.grad_W_Q),
        "W_K": np.linalg.norm(sa.grad_W_K),
        "W_V": np.linalg.norm(sa.grad_W_V),
        "W_O": np.linalg.norm(sa.grad_W_O),
        "b_Q": np.linalg.norm(sa.grad_b_Q),
        "b_K": np.linalg.norm(sa.grad_b_K),
        "b_V": np.linalg.norm(sa.grad_b_V),
        "b_O": np.linalg.norm(sa.grad_b_O),
        "X": np.linalg.norm(grad_X),
    }

    per_position_grad = np.linalg.norm(grad_X, axis=-1)  # (B, seq_len)
    mean_per_pos = per_position_grad.mean(axis=0)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    params = list(grad_norms.keys())
    norms = list(grad_norms.values())
    colors = [BLUE] * 4 + [GREEN] * 4 + [RED]
    ax1.barh(params, norms, color=colors)
    ax1.set_xlabel("Gradient L2 Norm")
    ax1.set_title("Gradient Magnitudes per Parameter")
    ax1.grid(True, alpha=0.3, axis="x")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(seq_len), mean_per_pos, color=BLUE, alpha=0.8)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Mean Gradient Norm")
    ax2.set_title("Input Gradient by Position (Non-Causal)")
    ax2.set_xticks(range(seq_len))
    ax2.set_xticklabels([f"t{i}" for i in range(seq_len)])
    ax2.grid(True, alpha=0.3, axis="y")

    # Causal gradient flow
    causal_mask = create_causal_mask(seq_len)
    sa_c = SelfAttention(d_model=d_model, d_k=d_k, d_v=d_v)
    np.random.seed(SEED)
    sa_c.W_Q = sa.W_Q.copy()
    sa_c.W_K = sa.W_K.copy()
    sa_c.W_V = sa.W_V.copy()
    sa_c.W_O = sa.W_O.copy()
    sa_c.b_Q = sa.b_Q.copy()
    sa_c.b_K = sa.b_K.copy()
    sa_c.b_V = sa.b_V.copy()
    sa_c.b_O = sa.b_O.copy()

    output_c = sa_c.forward(X, mask=causal_mask)
    grad_X_c = sa_c.backward(np.ones_like(output_c))

    per_pos_causal = np.linalg.norm(grad_X_c, axis=-1).mean(axis=0)

    ax3 = fig.add_subplot(gs[1, 0])
    width = 0.35
    x_pos = np.arange(seq_len)
    ax3.bar(x_pos - width / 2, mean_per_pos, width, color=BLUE, alpha=0.8, label="Non-causal")
    ax3.bar(x_pos + width / 2, per_pos_causal, width, color=RED, alpha=0.8, label="Causal")
    ax3.set_xlabel("Position")
    ax3.set_ylabel("Mean Gradient Norm")
    ax3.set_title("Causal vs Non-Causal Gradient Flow")
    ax3.set_xticks(range(seq_len))
    ax3.set_xticklabels([f"t{i}" for i in range(seq_len)])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Gradient flow vs d_k
    ax4 = fig.add_subplot(gs[1, 1])
    d_k_values = [4, 8, 16, 32, 64, 128]
    grad_x_norms = []
    grad_wq_norms = []
    for dk in d_k_values:
        np.random.seed(SEED)
        sa_t = SelfAttention(d_model=d_model, d_k=dk, d_v=dk)
        X_t = np.random.randn(batch_size, seq_len, d_model)
        out_t = sa_t.forward(X_t)
        gx = sa_t.backward(np.ones_like(out_t))
        grad_x_norms.append(np.linalg.norm(gx))
        grad_wq_norms.append(np.linalg.norm(sa_t.grad_W_Q))

    ax4.plot(d_k_values, grad_x_norms, "o-", color=BLUE, label="||dL/dX||", linewidth=2)
    ax4.plot(d_k_values, grad_wq_norms, "s--", color=RED, label="||dL/dW_Q||", linewidth=2)
    ax4.set_xlabel("d_k")
    ax4.set_ylabel("Gradient L2 Norm")
    ax4.set_title("Gradient Magnitude vs d_k")
    ax4.set_xscale("log", base=2)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle("Gradient Flow Through Self-Attention", fontsize=14, y=1.01)
    fig.savefig(VIZ_DIR / "07_gradient_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  Gradient norms per parameter:")
    for name, norm in grad_norms.items():
        print(f"    {name:4s}: {norm:.6f}")
    print(f"  Per-position gradient norms (non-causal): {mean_per_pos}")
    print(f"  Per-position gradient norms (causal):     {per_pos_causal}")
    print()
    return fig


def example_extra_numerical_verification():
    """Verify backward pass with finite difference gradient checking."""
    print("=" * 60)
    print("Example (Bonus): Numerical Gradient Verification")
    print("=" * 60)

    d_model, d_k, d_v = 8, 4, 4
    seq_len, batch_size = 3, 2
    eps = 1e-5

    np.random.seed(SEED)
    sa = SelfAttention(d_model=d_model, d_k=d_k, d_v=d_v)
    X = np.random.randn(batch_size, seq_len, d_model)

    def loss_fn(sa_obj, X_in, mask=None):
        out = sa_obj.forward(X_in, mask=mask)
        return 0.5 * np.sum(out ** 2)

    loss = loss_fn(sa, X)
    sa.backward(sa.forward(X))
    grad_analytical = sa.grad_W_Q.copy()
    sa_output = sa.forward(X)
    sa.backward(sa_output)
    grad_analytical = sa.grad_W_Q.copy()

    grad_numerical = np.zeros_like(sa.W_Q)
    for i in range(sa.W_Q.shape[0]):
        for j in range(sa.W_Q.shape[1]):
            sa.W_Q[i, j] += eps
            loss_plus = loss_fn(sa, X)
            sa.W_Q[i, j] -= 2 * eps
            loss_minus = loss_fn(sa, X)
            sa.W_Q[i, j] += eps
            grad_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

    rel_error = np.abs(grad_analytical - grad_numerical) / (
        np.abs(grad_analytical) + np.abs(grad_numerical) + 1e-8
    )
    max_rel_error = rel_error.max()
    mean_rel_error = rel_error.mean()

    print(f"  W_Q gradient check:")
    print(f"    Max relative error:  {max_rel_error:.2e}")
    print(f"    Mean relative error: {mean_rel_error:.2e}")
    print(f"    Passed: {max_rel_error < 1e-4}")

    # Check dL/dX
    sa.forward(X)
    sa_out = sa.forward(X)
    grad_X_analytical = sa.backward(sa_out)

    grad_X_numerical = np.zeros_like(X)
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(d_model):
                X[b, i, j] += eps
                loss_plus = loss_fn(sa, X)
                X[b, i, j] -= 2 * eps
                loss_minus = loss_fn(sa, X)
                X[b, i, j] += eps
                grad_X_numerical[b, i, j] = (loss_plus - loss_minus) / (2 * eps)

    rel_error_x = np.abs(grad_X_analytical - grad_X_numerical) / (
        np.abs(grad_X_analytical) + np.abs(grad_X_numerical) + 1e-8
    )
    max_rel_error_x = rel_error_x.max()

    print(f"  dL/dX gradient check:")
    print(f"    Max relative error:  {max_rel_error_x:.2e}")
    print(f"    Passed: {max_rel_error_x < 1e-4}")
    print()


def generate_pdf_report(figs_data):
    """Generate comprehensive PDF report with all visualizations."""
    print("=" * 60)
    print("Generating PDF Report")
    print("=" * 60)

    with PdfPages(str(REPORT_PATH)) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.65, "Self-Attention", fontsize=36, ha="center", va="center",
                 fontweight="bold")
        fig.text(0.5, 0.55, "Scaled Dot-Product Attention Analysis", fontsize=18,
                 ha="center", va="center", color="gray")
        fig.text(0.5, 0.40, "Visualizations, Scaling Analysis, and Gradient Flow",
                 fontsize=14, ha="center", va="center")
        fig.text(0.5, 0.30,
                 "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V",
                 fontsize=13, ha="center", va="center", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#bdc3c7"))
        fig.text(0.5, 0.15, f"SEED = {SEED} | NumPy-only implementation",
                 fontsize=11, ha="center", va="center", color="gray")
        fig.text(0.5, 0.08, "From-Scratch ML Implementations", fontsize=10,
                 ha="center", va="center", color="lightgray")
        pdf.savefig(fig)
        plt.close(fig)

        # Summary page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.92, "Summary of Findings", fontsize=24, ha="center",
                 fontweight="bold")

        findings = [
            "1. Attention weights form an (n x n) matrix where each row sums to 1,",
            "   representing a probability distribution over key positions.",
            "",
            "2. Causal masking produces a lower-triangular attention pattern,",
            "   preventing positions from attending to future tokens.",
            "",
            "3. Scaling by sqrt(d_k) is essential: without it, large d_k causes",
            "   softmax saturation (near-binary weights with vanishing gradients).",
            "",
            "4. The attention matrix dominates memory at long sequences:",
            "   at n=4096, it accounts for >98% of activation memory.",
            "",
            "5. Compute is O(n^2 * d_k) for the attention core. At long sequences,",
            "   attention core FLOPs dominate over linear projection FLOPs.",
            "",
            "6. Gradient flow through attention is well-behaved with proper scaling.",
            "   Causal masking causes asymmetric gradients across positions:",
            "   earlier positions receive contributions from more downstream tokens.",
            "",
            "7. These O(n^2) costs motivate Flash Attention (tiled computation),",
            "   KV caching (avoid recomputing K/V), and GQA/MQA (share K/V heads).",
        ]
        for i, line in enumerate(findings):
            fig.text(0.08, 0.82 - i * 0.038, line, fontsize=11, family="monospace",
                     va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # Add each visualization
        viz_files = sorted(VIZ_DIR.glob("*.png"))
        titles = {
            "01": "Attention Weights Heatmap",
            "02": "Causal vs Non-Causal Attention",
            "03": "Effect of Scaling by sqrt(d_k)",
            "04": "Attention Sharpness vs d_k Dimension",
            "05": "Quadratic Memory Scaling O(n^2)",
            "06": "FLOP Analysis and O(n^2) Growth",
            "07": "Gradient Flow Through Self-Attention",
        }
        for viz_file in viz_files:
            prefix = viz_file.stem[:2]
            title = titles.get(prefix, viz_file.stem)
            fig = plt.figure(figsize=(11, 8.5))
            img = plt.imread(str(viz_file))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.5, 0.95, title, fontsize=16, ha="center", fontweight="bold")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Report saved to: {REPORT_PATH}")
    print()


def main():
    print()
    print("=" * 60)
    print("  SELF-ATTENTION DEMO")
    print("  Scaled Dot-Product Attention Analysis")
    print(f"  SEED = {SEED}")
    print("=" * 60)
    print()

    np.random.seed(SEED)

    fig1 = example_1_attention_heatmap()

    np.random.seed(SEED + 1)
    fig2 = example_2_causal_vs_noncausal()

    fig3 = example_3_scaling_effect()

    fig4 = example_4_dk_sharpness()

    np.random.seed(SEED)
    fig5 = example_5_memory_scaling()

    fig6 = example_6_flop_analysis()

    np.random.seed(SEED)
    fig7 = example_7_gradient_flow()

    example_extra_numerical_verification()

    generate_pdf_report([fig1, fig2, fig3, fig4, fig5, fig6, fig7])

    print("=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print()
    viz_files = sorted(VIZ_DIR.glob("*.png"))
    print(f"  Generated {len(viz_files)} visualizations in viz/:")
    for f in viz_files:
        print(f"    - {f.name}")
    print(f"  PDF report: report.pdf")
    print()


if __name__ == "__main__":
    main()
