"""
Positional Encoding Demo -- Sinusoidal structure, relative position property, and visualizations.

Generates:
- viz/*.png -- Individual visualization files
- report.pdf -- Comprehensive PDF report
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from implementation import (
    sinusoidal_positional_encoding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    relative_position_matrix,
    dot_product_distance,
    encoding_statistics,
)

self_attn_dir = os.path.join(os.path.dirname(__file__), "..", "self-attention")
sys.path.insert(0, self_attn_dir)
import importlib
_sa_spec = importlib.util.spec_from_file_location(
    "self_attention_impl",
    os.path.join(self_attn_dir, "implementation.py"),
)
_sa_mod = importlib.util.module_from_spec(_sa_spec)
_sa_spec.loader.exec_module(_sa_mod)
SelfAttention = _sa_mod.SelfAttention

SEED = 42
np.random.seed(SEED)

VIZ_DIR = Path(__file__).parent / "viz"
VIZ_DIR.mkdir(exist_ok=True)

COLORS = {
    "blue": "#3498db",
    "red": "#e74c3c",
    "orange": "#f39c12",
    "green": "#27ae60",
    "purple": "#9b59b6",
    "steel": "steelblue",
    "coral": "coral",
    "teal": "#1abc9c",
    "dark": "#2c3e50",
}


# ---------------------------------------------------------------------------
# Example 1: PE Matrix Heatmap Visualization
# ---------------------------------------------------------------------------
def example_1_pe_heatmap():
    """Visualize sinusoidal PE matrix and its frequency structure."""
    print("=" * 60)
    print("Example 1: PE Matrix Heatmap Visualization")
    print("=" * 60)

    d_model = 128
    seq_len = 100
    pe = sinusoidal_positional_encoding(seq_len, d_model)

    print(f"\n  Config: d_model={d_model}, seq_len={seq_len}")
    print(f"  PE shape: {pe.shape}")
    print(f"  Value range: [{pe.min():.4f}, {pe.max():.4f}]")
    print(f"  Values bounded in [-1, 1]: {np.all(np.abs(pe) <= 1.0 + 1e-10)}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: full PE heatmap
    im0 = axes[0, 0].imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                             interpolation="nearest")
    axes[0, 0].set_xlabel("Position (pos)")
    axes[0, 0].set_ylabel("Dimension (d)")
    axes[0, 0].set_title("Sinusoidal PE Matrix (d_model=128, L=100)\nLow dims oscillate fast, high dims oscillate slow",
                          fontsize=11, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Top-right: zoom into first 20 dimensions, first 50 positions
    im1 = axes[0, 1].imshow(pe[:50, :20].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                             interpolation="nearest")
    axes[0, 1].set_xlabel("Position (pos)")
    axes[0, 1].set_ylabel("Dimension (d)")
    axes[0, 1].set_title("Zoomed: First 20 dims, 50 positions\nSin/cos pairs clearly visible",
                          fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Bottom-left: individual dimension traces
    dims_to_show = [0, 1, 10, 11, 60, 61, 126, 127]
    for d in dims_to_show:
        label_type = "sin" if d % 2 == 0 else "cos"
        i = d // 2
        axes[1, 0].plot(pe[:, d], label=f"d={d} ({label_type}, i={i})", alpha=0.8, linewidth=1.5)
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_ylabel("PE value")
    axes[1, 0].set_title("Individual Dimension Traces\nGeometric progression of frequencies",
                          fontsize=11, fontweight="bold")
    axes[1, 0].legend(fontsize=7, ncol=2, loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: wavelength analysis
    d_half = d_model // 2
    i_vals = np.arange(d_half)
    omega = np.exp(-2.0 * i_vals / d_model * np.log(10000.0))
    wavelengths = 2 * np.pi / omega

    axes[1, 1].semilogy(i_vals, wavelengths, "o-", color=COLORS["blue"], markersize=3, linewidth=1.5)
    axes[1, 1].axhline(2 * np.pi, color=COLORS["red"], linestyle="--", alpha=0.6,
                        label=f"Min wavelength: 2pi = {2*np.pi:.2f}")
    axes[1, 1].axhline(2 * np.pi * 10000, color=COLORS["green"], linestyle="--", alpha=0.6,
                        label=f"Limit (d->inf): 2pi*10000 = {2*np.pi*10000:.0f}")
    axes[1, 1].set_xlabel("Dimension pair index (i)")
    axes[1, 1].set_ylabel("Wavelength (positions)")
    axes[1, 1].set_title(f"Wavelength per Dimension Pair\nGeometric progression from 2pi to {wavelengths[-1]:.0f}",
                          fontsize=11, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    print(f"\n  Frequency structure (geometric progression):")
    print(f"    omega_0 = {omega[0]:.6f}, wavelength_0 = {wavelengths[0]:.2f} (= 2*pi)")
    print(f"    omega_{d_half-1} = {omega[-1]:.6e}, wavelength_{d_half-1} = {wavelengths[-1]:.0f} (approaches 2*pi*10000 = {2*np.pi*10000:.0f} as d->inf)")
    print(f"    Ratio between consecutive wavelengths: {wavelengths[1]/wavelengths[0]:.6f}")
    print(f"    Expected ratio: 10000^(2/d_model) = {10000**(2/d_model):.6f}")

    fig.suptitle("Sinusoidal Positional Encoding: Structure and Frequencies",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_pe_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/01_pe_heatmap.png")


# ---------------------------------------------------------------------------
# Example 2: Dot Product Distance Structure
# ---------------------------------------------------------------------------
def example_2_dot_product_distance():
    """Show Toeplitz structure of PE dot products and analytical formula."""
    print("\n" + "=" * 60)
    print("Example 2: Dot Product Distance Structure")
    print("=" * 60)

    d_model = 128
    seq_len = 64
    pe = sinusoidal_positional_encoding(seq_len, d_model)

    D = dot_product_distance(pe)

    print(f"\n  Config: d_model={d_model}, seq_len={seq_len}")
    print(f"  D shape: {D.shape}")
    print(f"  D is symmetric: {np.allclose(D, D.T)}")

    # Analytical: PE[pos] . PE[pos] = sum_{i} (sin^2(omega_i*pos) + cos^2(omega_i*pos)) = d_model/2
    diag = np.diag(D)
    expected_self_dot = d_model / 2
    print(f"\n  Self-dot product analysis:")
    print(f"    Analytical: PE[pos] . PE[pos] = d_model/2 = {expected_self_dot}")
    print(f"    Observed mean: {diag.mean():.6f}")
    print(f"    Observed std: {diag.std():.2e}")
    print(f"    Max deviation: {np.max(np.abs(diag - expected_self_dot)):.2e}")
    print(f"    Proof: Each (sin, cos) pair contributes sin^2 + cos^2 = 1. There are d/2 pairs.")

    # Toeplitz check: D[i,j] depends only on |i-j|
    # Analytical: PE[i] . PE[j] = sum_{k} cos(omega_k * (i - j))
    max_dist = seq_len - 1
    toeplitz_error = 0.0
    for delta in range(1, min(20, seq_len)):
        values_at_delta = [D[i, i + delta] for i in range(seq_len - delta)]
        spread = max(values_at_delta) - min(values_at_delta)
        toeplitz_error = max(toeplitz_error, spread)

    print(f"\n  Toeplitz structure (D[i,j] depends only on |i-j|):")
    print(f"    Max variation at same relative distance: {toeplitz_error:.2e}")
    print(f"    Analytically zero -- observed deviation is accumulated floating-point error")
    print(f"    Proof: PE[i] . PE[j] = sum_k cos(omega_k * (i-j)), depends only on i-j")

    # Compute dot product as function of relative distance
    dot_by_distance = np.zeros(seq_len)
    for delta in range(seq_len):
        vals = [D[i, i + delta] for i in range(seq_len - delta)]
        dot_by_distance[delta] = np.mean(vals)

    # Analytical formula
    d_half = d_model // 2
    i_vals = np.arange(d_half)
    omega = np.exp(-2.0 * i_vals / d_model * np.log(10000.0))
    analytical_dots = np.array([np.sum(np.cos(omega * delta)) for delta in range(seq_len)])

    formula_error = np.max(np.abs(dot_by_distance - analytical_dots))
    print(f"\n  Analytical formula verification:")
    print(f"    PE[p1] . PE[p2] = sum_i cos(omega_i * (p1 - p2))")
    print(f"    Max error between observed and analytical: {formula_error:.2e}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: full dot product matrix
    im0 = axes[0, 0].imshow(D, cmap="RdBu_r", aspect="equal", interpolation="nearest")
    axes[0, 0].set_xlabel("Position j")
    axes[0, 0].set_ylabel("Position i")
    axes[0, 0].set_title(f"PE @ PE.T (d_model={d_model}, L={seq_len})\nToeplitz structure: depends only on |i-j|",
                          fontsize=11, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Top-right: zoomed diagonal bands
    D_zoomed = D[:20, :20]
    im1 = axes[0, 1].imshow(D_zoomed, cmap="RdBu_r", aspect="equal", interpolation="nearest")
    axes[0, 1].set_xlabel("Position j")
    axes[0, 1].set_ylabel("Position i")
    axes[0, 1].set_title("Zoomed: First 20x20\nConstant along diagonals = Toeplitz",
                          fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    for k in range(-3, 4):
        diag_vals = np.diag(D_zoomed, k=k)
        if len(diag_vals) > 1:
            axes[0, 1].plot([max(0, k) + i for i in range(len(diag_vals))],
                           [max(0, -k) + i for i in range(len(diag_vals))],
                           ".", color="black", markersize=1, alpha=0.3)

    # Bottom-left: dot product vs relative distance
    axes[1, 0].plot(range(seq_len), dot_by_distance, "-", color=COLORS["blue"], linewidth=2,
                    label="Observed: mean D[i, i+delta]")
    axes[1, 0].plot(range(seq_len), analytical_dots, "--", color=COLORS["red"], linewidth=1.5,
                    label="Analytical: sum_k cos(omega_k * delta)")
    axes[1, 0].axhline(expected_self_dot, color=COLORS["green"], linestyle=":", alpha=0.5,
                        label=f"Self-dot = d/2 = {expected_self_dot:.0f}")
    axes[1, 0].set_xlabel("Relative Distance (delta = |i - j|)")
    axes[1, 0].set_ylabel("Dot Product PE[i] . PE[j]")
    axes[1, 0].set_title("Dot Product vs Relative Distance\nSame curve regardless of absolute position",
                          fontsize=11, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: verify position-independence by overlaying multiple starting positions
    start_positions = [0, 10, 20, 30, 40]
    max_delta_show = 25
    for start in start_positions:
        if start + max_delta_show <= seq_len:
            dots = [D[start, start + d] for d in range(max_delta_show)]
            axes[1, 1].plot(range(max_delta_show), dots, "o-", markersize=4, alpha=0.7,
                           label=f"start={start}")
    axes[1, 1].set_xlabel("Relative Distance (delta)")
    axes[1, 1].set_ylabel("Dot Product")
    axes[1, 1].set_title("Position Independence Verification\nAll curves overlap perfectly",
                          fontsize=11, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Dot Product Distance Structure: Toeplitz Property of Sinusoidal PE",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_dot_product_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/02_dot_product_distance.png")


# ---------------------------------------------------------------------------
# Example 3: Relative Position Property
# ---------------------------------------------------------------------------
def example_3_relative_position():
    """Demonstrate PE(pos+k) = M_k @ PE(pos) via rotation matrices."""
    print("\n" + "=" * 60)
    print("Example 3: Relative Position Property")
    print("=" * 60)

    d_model = 64
    seq_len = 100
    pe = sinusoidal_positional_encoding(seq_len, d_model)

    offsets = [1, 3, 5, 10, 25, 50]
    results = {}
    for k in offsets:
        M_k, max_err = relative_position_matrix(pe, k)
        results[k] = (M_k, max_err)

    print(f"\n  Config: d_model={d_model}, seq_len={seq_len}")
    print(f"\n  Relative position property: PE(pos+k) = M_k @ PE(pos)")
    print(f"  M_k is a block-diagonal matrix with d/2 rotation blocks R_i(k)")
    print(f"  R_i(k) = [[cos(omega_i*k), sin(omega_i*k)], [-sin(omega_i*k), cos(omega_i*k)]]")
    print(f"\n  {'Offset k':>10} {'Max Recon. Error':>20} {'Machine Eps':>15}")
    print(f"  {'-'*50}")
    for k in offsets:
        _, max_err = results[k]
        print(f"  {k:>10} {max_err:>20.2e} {np.finfo(np.float64).eps:>15.2e}")

    print(f"\n  All errors are ~1e-14 (accumulated floating-point rounding over d/2 terms).")
    print(f"  The transformation is analytically EXACT -- errors are purely numerical.")
    print(f"  This is the key property enabling attention to learn relative positions.")

    # Visualize rotation matrix structure
    M_1 = results[1][0]
    M_10 = results[10][0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Top-left: M_1 full matrix (block-diagonal structure)
    im0 = axes[0, 0].imshow(np.abs(M_1), cmap="Blues", aspect="equal", interpolation="nearest")
    axes[0, 0].set_xlabel("Input dimension")
    axes[0, 0].set_ylabel("Output dimension")
    axes[0, 0].set_title("$|M_1|$: Block-Diagonal Rotation Matrix\nd_model/2 independent 2x2 blocks",
                          fontsize=11, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Top-middle: zoom into first 8x8 block of M_1
    M_1_zoom = M_1[:8, :8]
    im1 = axes[0, 1].imshow(M_1_zoom, cmap="RdBu_r", aspect="equal", interpolation="nearest",
                             vmin=-1, vmax=1)
    for i in range(8):
        for j in range(8):
            axes[0, 1].text(j, i, f"{M_1_zoom[i, j]:.3f}", ha="center", va="center",
                           fontsize=8, color="black" if abs(M_1_zoom[i, j]) < 0.5 else "white")
    axes[0, 1].set_title("$M_1$ Zoomed (first 8 dims)\n2x2 rotation blocks on diagonal",
                          fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Top-right: compare M_1 and M_10
    im2 = axes[0, 2].imshow(np.abs(M_10[:16, :16]), cmap="Blues", aspect="equal",
                             interpolation="nearest")
    axes[0, 2].set_title("$|M_{10}|$ Zoomed (first 16 dims)\nSame structure, different rotation angles",
                          fontsize=11, fontweight="bold")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Bottom-left: reconstruction error across positions for various offsets
    for k in offsets:
        M_k = results[k][0]
        errors = []
        for pos in range(seq_len - k):
            reconstructed = M_k @ pe[pos]
            err = np.linalg.norm(reconstructed - pe[pos + k])
            errors.append(err)
        axes[1, 0].plot(range(len(errors)), errors, "o", markersize=2, alpha=0.7, label=f"k={k}")

    axes[1, 0].set_xlabel("Starting Position (pos)")
    axes[1, 0].set_ylabel("Reconstruction Error ||M_k @ PE[pos] - PE[pos+k]||")
    axes[1, 0].set_title("Reconstruction Error Across All Positions\nUniformly ~1e-14 (float64 rounding)",
                          fontsize=11, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale("log")

    # Bottom-middle: show a single rotation block transformation
    d_half = d_model // 2
    i_dim = 0
    omega_0 = np.exp(-2.0 * i_dim / d_model * np.log(10000.0))

    theta_orig = np.arctan2(pe[:, 0], pe[:, 1])
    theta_shifted = np.arctan2(pe[1:, 0], pe[1:, 1])

    pos_range = np.arange(20)
    axes[1, 1].plot(pos_range, pe[:20, 0], "o-", color=COLORS["blue"], label="sin(omega_0 * pos)", markersize=5)
    axes[1, 1].plot(pos_range, pe[:20, 1], "s-", color=COLORS["red"], label="cos(omega_0 * pos)", markersize=5)

    cos_k, sin_k = np.cos(omega_0), np.sin(omega_0)
    rotated_sin = cos_k * pe[:19, 0] + sin_k * pe[:19, 1]
    rotated_cos = -sin_k * pe[:19, 0] + cos_k * pe[:19, 1]
    axes[1, 1].plot(pos_range[1:], rotated_sin, "x", color=COLORS["blue"], markersize=8,
                    label="R_0(1) applied (sin)")
    axes[1, 1].plot(pos_range[1:], rotated_cos, "+", color=COLORS["red"], markersize=8,
                    label="R_0(1) applied (cos)")
    axes[1, 1].set_xlabel("Position")
    axes[1, 1].set_ylabel("PE value")
    axes[1, 1].set_title("Rotation in Dimension Pair (0,1)\nR_0(1) maps PE[pos] to PE[pos+1]",
                          fontsize=11, fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Bottom-right: rotation angles per dimension pair for k=1
    rotation_angles = []
    for i in range(d_half):
        omega_i = np.exp(-2.0 * i / d_model * np.log(10000.0))
        rotation_angles.append(omega_i * 1.0)

    axes[1, 2].semilogy(range(d_half), rotation_angles, "o-", color=COLORS["purple"],
                        markersize=3, linewidth=1.5)
    axes[1, 2].set_xlabel("Dimension pair index (i)")
    axes[1, 2].set_ylabel("Rotation angle for k=1 (radians)")
    axes[1, 2].set_title("Per-Pair Rotation Angle (k=1)\nGeometric decay from omega_0=1 to omega_{d/2-1}",
                          fontsize=11, fontweight="bold")
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("Relative Position Property: PE(pos+k) = M_k @ PE(pos)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_relative_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/03_relative_position.png")


# ---------------------------------------------------------------------------
# Example 4: Sinusoidal vs Learned Comparison
# ---------------------------------------------------------------------------
def example_4_sinusoidal_vs_learned():
    """Compare sinusoidal and learned positional encodings on key properties."""
    print("\n" + "=" * 60)
    print("Example 4: Sinusoidal vs Learned Comparison")
    print("=" * 60)

    d_model = 64
    max_seq_len = 64
    seq_len = 64

    np.random.seed(SEED)
    sin_enc = SinusoidalPositionalEncoding(max_seq_len, d_model)
    learn_enc = LearnedPositionalEncoding(max_seq_len, d_model)

    pe_sin = sin_enc.get_encoding(seq_len)
    pe_learn = learn_enc.embedding[:seq_len]

    stats_sin = encoding_statistics(pe_sin)
    stats_learn = encoding_statistics(pe_learn)

    # Norms
    expected_norm = np.sqrt(d_model / 2)
    print(f"\n  Config: d_model={d_model}, max_seq_len={max_seq_len}")
    print(f"\n  (a) Position norms:")
    print(f"    Sinusoidal: constant = sqrt(d/2) = {expected_norm:.4f}")
    print(f"      Mean: {stats_sin['position_norms'].mean():.6f}, Std: {stats_sin['position_norms'].std():.2e}")
    print(f"    Learned (random init, N(0, 0.02)):")
    print(f"      Mean: {stats_learn['position_norms'].mean():.6f}, Std: {stats_learn['position_norms'].std():.4f}")
    print(f"    Proof: Each row has d/2 (sin,cos) pairs; sin^2 + cos^2 = 1, so ||PE||^2 = d/2.")

    # Dot product structure
    D_sin = dot_product_distance(pe_sin)
    D_learn = dot_product_distance(pe_learn)

    toeplitz_err_sin = 0.0
    toeplitz_err_learn = 0.0
    for delta in range(1, min(20, seq_len)):
        vals_sin = [D_sin[i, i + delta] for i in range(seq_len - delta)]
        vals_learn = [D_learn[i, i + delta] for i in range(seq_len - delta)]
        toeplitz_err_sin = max(toeplitz_err_sin, max(vals_sin) - min(vals_sin))
        toeplitz_err_learn = max(toeplitz_err_learn, max(vals_learn) - min(vals_learn))

    print(f"\n  (b) Dot product structure (Toeplitz test):")
    print(f"    Sinusoidal max variation at same distance: {toeplitz_err_sin:.2e} (perfectly Toeplitz)")
    print(f"    Learned max variation at same distance: {toeplitz_err_learn:.4f} (NOT Toeplitz)")

    # Extrapolation
    extrap_len = max_seq_len + 50
    pe_sin_extrap = sinusoidal_positional_encoding(extrap_len, d_model)
    extrap_norms = np.linalg.norm(pe_sin_extrap, axis=1)

    print(f"\n  (c) Extrapolation:")
    print(f"    Sinusoidal at pos={extrap_len-1}: norm = {extrap_norms[-1]:.6f} (= sqrt(d/2) = {expected_norm:.4f})")
    print(f"    Sinusoidal generates valid encodings for ANY length.")
    print(f"    Learned raises ValueError for L > max_seq_len = {max_seq_len}")
    try:
        X_test = np.random.randn(1, extrap_len, d_model)
        learn_enc.forward(X_test)
        print(f"    ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"    Correctly raised: {e}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: Sinusoidal properties
    axes[0, 0].plot(stats_sin["position_norms"], color=COLORS["blue"], linewidth=2)
    axes[0, 0].axhline(expected_norm, color=COLORS["red"], linestyle="--",
                        label=f"sqrt(d/2) = {expected_norm:.2f}")
    axes[0, 0].set_xlabel("Position")
    axes[0, 0].set_ylabel("L2 Norm")
    axes[0, 0].set_title("Sinusoidal: Position Norms\nConstant sqrt(d/2) for all positions",
                          fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(expected_norm - 0.5, expected_norm + 0.5)

    im_sin = axes[0, 1].imshow(D_sin, cmap="RdBu_r", aspect="equal", interpolation="nearest")
    axes[0, 1].set_xlabel("Position j")
    axes[0, 1].set_ylabel("Position i")
    axes[0, 1].set_title("Sinusoidal: Dot Product Matrix\nPerfect Toeplitz structure",
                          fontsize=10, fontweight="bold")
    fig.colorbar(im_sin, ax=axes[0, 1], fraction=0.046, pad=0.04)

    sin_extrap_range = np.arange(extrap_len)
    axes[0, 2].plot(sin_extrap_range, extrap_norms, color=COLORS["blue"], linewidth=1.5)
    axes[0, 2].axvline(max_seq_len, color=COLORS["red"], linestyle="--", alpha=0.7,
                        label=f"max_seq_len={max_seq_len}")
    axes[0, 2].axhline(expected_norm, color=COLORS["green"], linestyle=":", alpha=0.5)
    axes[0, 2].set_xlabel("Position")
    axes[0, 2].set_ylabel("L2 Norm")
    axes[0, 2].set_title("Sinusoidal: Extrapolation\nValid encodings beyond training length",
                          fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Learned properties
    axes[1, 0].plot(stats_learn["position_norms"], color=COLORS["orange"], linewidth=2)
    axes[1, 0].axhline(expected_norm, color=COLORS["red"], linestyle="--",
                        label=f"sqrt(d/2) = {expected_norm:.2f}")
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_ylabel("L2 Norm")
    axes[1, 0].set_title("Learned (random init): Position Norms\nVariable, not constant",
                          fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    im_learn = axes[1, 1].imshow(D_learn, cmap="RdBu_r", aspect="equal", interpolation="nearest")
    axes[1, 1].set_xlabel("Position j")
    axes[1, 1].set_ylabel("Position i")
    axes[1, 1].set_title("Learned (random init): Dot Product Matrix\nNo Toeplitz structure",
                          fontsize=10, fontweight="bold")
    fig.colorbar(im_learn, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Extrapolation comparison: show what happens at boundary
    axes[1, 2].bar(["Sinusoidal", "Learned"], [extrap_len, max_seq_len],
                   color=[COLORS["blue"], COLORS["orange"]], edgecolor="white", width=0.5)
    axes[1, 2].set_ylabel("Maximum Sequence Length")
    axes[1, 2].set_title("Extrapolation Capability\nLearned is capped at max_seq_len",
                          fontsize=10, fontweight="bold")
    axes[1, 2].grid(True, alpha=0.3, axis="y")
    axes[1, 2].text(0, extrap_len + 1, "Unlimited", ha="center", fontsize=10, fontweight="bold",
                    color=COLORS["blue"])
    axes[1, 2].text(1, max_seq_len + 1, f"max={max_seq_len}", ha="center", fontsize=10,
                    fontweight="bold", color=COLORS["orange"])

    fig.suptitle("Sinusoidal vs Learned Positional Encoding Comparison",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_sinusoidal_vs_learned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/04_sinusoidal_vs_learned.png")


# ---------------------------------------------------------------------------
# Example 5: Impact on Attention
# ---------------------------------------------------------------------------
def example_5_attention_impact():
    """Show that PE breaks permutation invariance of self-attention."""
    print("\n" + "=" * 60)
    print("Example 5: Impact on Attention")
    print("=" * 60)

    d_model = 32
    seq_len = 8
    batch_size = 1

    np.random.seed(SEED)
    X = np.random.randn(batch_size, seq_len, d_model)

    perm = np.array([3, 1, 5, 0, 7, 2, 6, 4])
    X_perm = X[:, perm, :]

    pe_enc = SinusoidalPositionalEncoding(seq_len, d_model)

    # WITHOUT PE: attention is permutation-equivariant
    np.random.seed(SEED + 100)
    attn_no_pe = SelfAttention(d_model=d_model, d_k=d_model, d_v=d_model, use_bias=False)

    out_no_pe = attn_no_pe.forward(X)
    out_no_pe_perm = attn_no_pe.forward(X_perm)

    # If attention is equivariant: out_no_pe_perm should equal out_no_pe[:, perm, :]
    out_no_pe_reperm = out_no_pe[:, perm, :]
    diff_no_pe = np.linalg.norm(out_no_pe_perm - out_no_pe_reperm)

    print(f"\n  Config: d_model={d_model}, seq_len={seq_len}")
    print(f"  Permutation: {perm.tolist()}")

    print(f"\n  WITHOUT positional encoding:")
    print(f"    ||Attn(X_perm) - Perm(Attn(X))||_F = {diff_no_pe:.2e}")
    print(f"    This should be ~0 (equivariance): permuting input just permutes output.")
    print(f"    Attention CANNOT distinguish token order without PE.")

    # WITH PE: permutation changes the result
    X_with_pe = pe_enc.forward(X)
    X_perm_with_pe = pe_enc.forward(X_perm)

    out_with_pe = attn_no_pe.forward(X_with_pe)
    out_with_pe_perm = attn_no_pe.forward(X_perm_with_pe)

    # Now equivariance is broken: out_with_pe_perm != out_with_pe[:, perm, :]
    out_with_pe_reperm = out_with_pe[:, perm, :]
    diff_with_pe = np.linalg.norm(out_with_pe_perm - out_with_pe_reperm)

    print(f"\n  WITH positional encoding:")
    print(f"    ||Attn(X_perm + PE) - Perm(Attn(X + PE))||_F = {diff_with_pe:.6f}")
    print(f"    This is NON-ZERO: PE breaks equivariance, making attention position-aware.")

    # Why this happens: X_perm + PE[:seq_len] != Perm(X + PE[:seq_len])
    # because PE is NOT permuted, so token at position 3 gets PE[0] in the permuted version
    pe_matrix = pe_enc.get_encoding(seq_len)
    X_perm_plus_pe = X_perm + pe_matrix
    perm_of_X_plus_pe = (X + pe_matrix)[:, perm, :]
    input_diff = np.linalg.norm(X_perm_plus_pe - perm_of_X_plus_pe)
    print(f"\n  Why: X_perm + PE != Perm(X + PE)")
    print(f"    ||X_perm + PE - Perm(X + PE)||_F = {input_diff:.6f}")
    print(f"    Token at position 0 in X_perm was originally at position {perm[0]},")
    print(f"    but now gets PE[0] instead of PE[{perm[0]}]. Different PE => different attention.")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Show original and permuted attention outputs
    out_orig = out_no_pe[0]
    out_p = out_no_pe_perm[0]
    out_orig_reperm = out_no_pe_reperm[0]

    im0 = axes[0, 0].imshow(out_orig, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0, 0].set_xlabel("Output dimension")
    axes[0, 0].set_ylabel("Position")
    axes[0, 0].set_title("Attn(X) -- No PE", fontsize=10, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(out_p, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0, 1].set_xlabel("Output dimension")
    axes[0, 1].set_ylabel("Position")
    axes[0, 1].set_title("Attn(X_perm) -- No PE", fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    diff_map_no_pe = np.abs(out_p - out_orig_reperm)
    im2 = axes[0, 2].imshow(diff_map_no_pe, aspect="auto", cmap="Reds", interpolation="nearest")
    axes[0, 2].set_xlabel("Output dimension")
    axes[0, 2].set_ylabel("Position")
    axes[0, 2].set_title(f"| Attn(X_perm) - Perm(Attn(X)) | -- No PE\nFrob norm = {diff_no_pe:.2e}",
                          fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # With PE
    out_pe_orig = out_with_pe[0]
    out_pe_p = out_with_pe_perm[0]
    out_pe_orig_reperm = out_with_pe_reperm[0]

    im3 = axes[1, 0].imshow(out_pe_orig, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[1, 0].set_xlabel("Output dimension")
    axes[1, 0].set_ylabel("Position")
    axes[1, 0].set_title("Attn(X + PE) -- With PE", fontsize=10, fontweight="bold")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(out_pe_p, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[1, 1].set_xlabel("Output dimension")
    axes[1, 1].set_ylabel("Position")
    axes[1, 1].set_title("Attn(X_perm + PE) -- With PE", fontsize=10, fontweight="bold")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    diff_map_with_pe = np.abs(out_pe_p - out_pe_orig_reperm)
    im5 = axes[1, 2].imshow(diff_map_with_pe, aspect="auto", cmap="Reds", interpolation="nearest")
    axes[1, 2].set_xlabel("Output dimension")
    axes[1, 2].set_ylabel("Position")
    axes[1, 2].set_title(f"| Attn(X_perm+PE) - Perm(Attn(X+PE)) | -- With PE\nFrob norm = {diff_with_pe:.4f}",
                          fontsize=10, fontweight="bold")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle("Impact of Positional Encoding on Self-Attention\n"
                 "Without PE: permutation-equivariant (top row, diff ~0). "
                 "With PE: position-aware (bottom row, diff >> 0).",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_attention_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/05_attention_impact.png")


# ---------------------------------------------------------------------------
# Example 6: Frequency Structure Analysis
# ---------------------------------------------------------------------------
def example_6_frequency_analysis():
    """Analyze wavelength structure and per-dimension variance."""
    print("\n" + "=" * 60)
    print("Example 6: Frequency Structure Analysis")
    print("=" * 60)

    d_model = 128
    seq_len = 200
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    stats = encoding_statistics(pe)

    d_half = d_model // 2
    i_vals = np.arange(d_half)
    omega = np.exp(-2.0 * i_vals / d_model * np.log(10000.0))
    wavelengths = 2 * np.pi / omega

    print(f"\n  Config: d_model={d_model}, seq_len={seq_len}")

    # Analytical variance of sin(omega*pos) over positions 0..L-1:
    # For large L and omega not too small:
    # Var[sin(omega*pos)] ~ 1/2 (since sin^2 averages to 1/2 over full periods)
    # But for very slow oscillations (high dims), the variance is lower because
    # we don't complete a full period within L positions.
    print(f"\n  Wavelength at each dimension pair:")
    print(f"    {'Pair (2i, 2i+1)':>18} {'omega_i':>12} {'Wavelength':>12} {'Periods in L={0}'.format(seq_len):>18}")
    print(f"    {'-'*65}")
    sample_indices = [0, 1, 5, 10, 20, 32, 50, 63]
    for i in sample_indices:
        periods = seq_len / wavelengths[i]
        print(f"    ({2*i:>3}, {2*i+1:>3}){omega[i]:>15.6e} {wavelengths[i]:>12.1f} {periods:>18.4f}")

    print(f"\n  Wavelength range:")
    print(f"    Minimum: 2*pi = {2*np.pi:.4f} (dimension pair 0)")
    print(f"    Maximum: {wavelengths[-1]:.0f} (dimension pair {d_half-1}; approaches 2*pi*10000 = {2*np.pi*10000:.0f} as d->inf)")
    print(f"    Geometric ratio: wavelength[i+1]/wavelength[i] = 10000^(2/d) = {10000**(2/d_model):.6f}")

    # Variance analysis
    var_per_dim = stats["var_per_dim"]
    print(f"\n  Variance across positions (per dimension):")
    print(f"    Dim 0 (fastest sin, omega=1): var = {var_per_dim[0]:.6f}")
    print(f"    Dim 1 (fastest cos, omega=1): var = {var_per_dim[1]:.6f}")
    print(f"    Dim {d_model-2} (slowest sin): var = {var_per_dim[-2]:.6e}")
    print(f"    Dim {d_model-1} (slowest cos): var = {var_per_dim[-1]:.6e}")
    print(f"    For dim 2i: Var[sin(omega_i * pos)] over L positions.")
    print(f"    Fast dims complete many periods -> var ~ 0.5 (half of amplitude^2).")
    print(f"    Slow dims barely change over L -> var ~ 0 (nearly constant).")

    # Analytical variance of sin(omega*pos) for pos in {0, ..., L-1}:
    # = 1/2 - sin(omega*L)*cos(omega*(L-1)) / (2*L*sin(omega)) for omega > 0
    # Simplified: approaches 1/2 as L * omega >> 1
    analytical_var_sin = np.zeros(d_half)
    analytical_var_cos = np.zeros(d_half)
    for i in range(d_half):
        w = omega[i]
        positions = np.arange(seq_len, dtype=np.float64)
        sin_vals = np.sin(w * positions)
        cos_vals = np.cos(w * positions)
        analytical_var_sin[i] = np.var(sin_vals)
        analytical_var_cos[i] = np.var(cos_vals)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Top-left: wavelength per dimension pair
    axes[0, 0].semilogy(i_vals, wavelengths, "o-", color=COLORS["blue"], markersize=3, linewidth=1.5)
    axes[0, 0].axhline(2 * np.pi, color=COLORS["red"], linestyle="--", alpha=0.6,
                        label=f"2*pi = {2*np.pi:.2f}")
    axes[0, 0].axhline(2 * np.pi * 10000, color=COLORS["green"], linestyle="--", alpha=0.6,
                        label=f"2*pi*10000 = {2*np.pi*10000:.0f}")
    axes[0, 0].axhline(seq_len, color=COLORS["orange"], linestyle=":", alpha=0.6,
                        label=f"seq_len = {seq_len}")
    axes[0, 0].set_xlabel("Dimension pair index (i)")
    axes[0, 0].set_ylabel("Wavelength (positions)")
    axes[0, 0].set_title("Wavelength per Dimension Pair\nGeometric progression",
                          fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-middle: variance per dimension
    axes[0, 1].plot(range(d_model), var_per_dim, "o", color=COLORS["blue"], markersize=3, alpha=0.7)
    axes[0, 1].axhline(0.5, color=COLORS["red"], linestyle="--", alpha=0.5,
                        label="Theoretical max (0.5)")
    axes[0, 1].set_xlabel("Dimension index (d)")
    axes[0, 1].set_ylabel("Variance across positions")
    axes[0, 1].set_title("Variance per Dimension\nLow dims vary a lot, high dims nearly constant",
                          fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Top-right: variance vs wavelength (log scale)
    var_by_pair = np.zeros(d_half)
    for i in range(d_half):
        var_by_pair[i] = (var_per_dim[2*i] + var_per_dim[2*i+1]) / 2
    axes[0, 2].semilogx(wavelengths, var_by_pair, "o-", color=COLORS["purple"],
                         markersize=3, linewidth=1.5, label="Observed mean var")
    axes[0, 2].axvline(seq_len, color=COLORS["orange"], linestyle="--", alpha=0.6,
                        label=f"seq_len = {seq_len}")
    axes[0, 2].axhline(0.5, color=COLORS["red"], linestyle=":", alpha=0.5,
                        label="Asymptotic (0.5)")
    axes[0, 2].set_xlabel("Wavelength")
    axes[0, 2].set_ylabel("Mean variance of (sin, cos) pair")
    axes[0, 2].set_title("Variance vs Wavelength\nVariance -> 0.5 when wavelength << seq_len",
                          fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Bottom-left: sample waveforms from different frequency bands
    sample_dims = [0, 10, 30, 50, 62]
    for i_dim in sample_dims:
        w = omega[i_dim]
        wl = wavelengths[i_dim]
        axes[1, 0].plot(np.arange(seq_len), pe[:, 2*i_dim], linewidth=1.5, alpha=0.8,
                        label=f"i={i_dim}, wl={wl:.0f}" if wl >= 10 else f"i={i_dim}, wl={wl:.1f}")
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_ylabel("sin(omega_i * pos)")
    axes[1, 0].set_title("Sample Waveforms (sin component)\nFast to slow across dimension pairs",
                          fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-middle: frequency (omega) progression
    axes[1, 1].semilogy(i_vals, omega, "o-", color=COLORS["steel"], markersize=3, linewidth=1.5)
    axes[1, 1].set_xlabel("Dimension pair index (i)")
    axes[1, 1].set_ylabel("Frequency omega_i")
    axes[1, 1].set_title("Frequency per Dimension Pair\nomega_i = 1/10000^(2i/d)",
                          fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    # Verify geometric progression
    ratios = omega[:-1] / omega[1:]
    expected_ratio = 10000 ** (2 / d_model)
    axes[1, 1].text(0.95, 0.95, f"Ratio: {ratios.mean():.4f}\nExpected: {expected_ratio:.4f}",
                    transform=axes[1, 1].transAxes, fontsize=9, va="top", ha="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Bottom-right: cumulative periods within sequence length
    periods_in_seq = seq_len / wavelengths
    axes[1, 2].semilogy(i_vals, periods_in_seq, "o-", color=COLORS["teal"],
                         markersize=3, linewidth=1.5)
    axes[1, 2].axhline(1.0, color=COLORS["red"], linestyle="--", alpha=0.6,
                        label="1 full period")
    axes[1, 2].set_xlabel("Dimension pair index (i)")
    axes[1, 2].set_ylabel(f"Periods completed in L={seq_len}")
    axes[1, 2].set_title(f"Periods Completed in seq_len={seq_len}\nLow dims: many periods; high dims: < 1 period",
                          fontsize=10, fontweight="bold")
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)

    # Find crossover dimension where wavelength = seq_len
    crossover_idx = np.searchsorted(wavelengths, seq_len)
    axes[1, 2].axvline(crossover_idx, color=COLORS["orange"], linestyle=":", alpha=0.6,
                        label=f"crossover i={crossover_idx}")
    axes[1, 2].legend(fontsize=9)

    print(f"\n  Geometric progression verification:")
    print(f"    omega[i]/omega[i+1] ratio: mean={ratios.mean():.6f}, std={ratios.std():.2e}")
    print(f"    Expected ratio: 10000^(2/d_model) = {expected_ratio:.6f}")
    print(f"    Crossover dimension (wavelength = seq_len): i = {crossover_idx}")
    print(f"    Dims i < {crossover_idx}: complete >= 1 period in L={seq_len}")
    print(f"    Dims i >= {crossover_idx}: complete < 1 period (appear nearly constant)")

    fig.suptitle("Frequency Structure Analysis of Sinusoidal Positional Encoding",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_frequency_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/06_frequency_analysis.png")


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------
def generate_pdf_report():
    """Generate comprehensive PDF report with all visualizations and analysis."""
    print("\n" + "=" * 60)
    print("Generating PDF Report")
    print("=" * 60)

    report_path = Path(__file__).parent / "report.pdf"
    viz_files = sorted(VIZ_DIR.glob("*.png"))

    with PdfPages(str(report_path)) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.78, "Positional Encoding", fontsize=28, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.68, "Sinusoidal and Learned Absolute Positional Encodings", fontsize=16,
                ha="center", va="center", transform=ax.transAxes, color="gray")
        info_text = (
            "Positional encoding injects position information into\n"
            "transformer inputs to break the permutation invariance\n"
            "of self-attention. Sinusoidal encodings use geometrically\n"
            "spaced frequencies to enable relative position learning.\n\n"
            "Key properties demonstrated:\n"
            "- PE(pos+k) = M_k @ PE(pos) via rotation matrices\n"
            "- Dot products depend only on relative distance (Toeplitz)\n"
            "- Self-dot = d_model/2 (from sin^2 + cos^2 = 1)\n"
            "- Wavelengths form geometric progression: 2pi to ~10000*2pi\n\n"
            f"Random seed: {SEED}\n"
            f"Number of visualizations: {len(viz_files)}\n"
            f"Examples: 6"
        )
        ax.text(0.5, 0.38, info_text, fontsize=12, ha="center", va="center",
                transform=ax.transAxes, linespacing=1.6)
        ax.text(0.5, 0.08, "Generated by demo.py", fontsize=10, ha="center",
                va="center", transform=ax.transAxes, style="italic", color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        # Summary page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.94, "Summary of Findings", fontsize=20, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)

        summary_items = [
            "1. PE Matrix Structure: Sinusoidal PE shows characteristic wave patterns.",
            "   Low dimensions (i=0) oscillate every ~6 positions (wavelength=2*pi).",
            "   High dimensions (i=63) have wavelength ~54000 positions (d=128).",
            "   Frequencies form a geometric progression with ratio 10000^(2/d).",
            "",
            "2. Dot Product Distance: PE @ PE.T has perfect Toeplitz structure.",
            "   D[i,j] = sum_k cos(omega_k * (i-j)), depends ONLY on relative distance.",
            "   Self-dot product = d_model/2 exactly (from sin^2 + cos^2 = 1 per pair).",
            "   Variation at same distance: ~1e-14 (floating-point rounding only).",
            "",
            "3. Relative Position Property: PE(pos+k) = M_k @ PE(pos) EXACTLY.",
            "   M_k is block-diagonal with d/2 rotation blocks R_i(k).",
            "   Reconstruction error: ~1e-14 (analytically exact, limited by float64).",
            "   This enables attention to learn relative positions via linear transforms.",
            "",
            "4. Sinusoidal vs Learned: Sinusoidal has constant norm sqrt(d/2),",
            "   perfect Toeplitz dot products, and unlimited extrapolation.",
            "   Learned (random init) has variable norms, no Toeplitz structure,",
            "   and is capped at max_seq_len.",
            "",
            "5. Attention Impact: Without PE, attention is permutation-equivariant",
            "   (diff ~0). With PE, permuting input changes output significantly,",
            "   making attention position-aware.",
            "",
            "6. Frequency Analysis: Wavelengths span 2*pi to ~10000*2*pi (exact max depends on d).",
            "   Variance drops from ~0.5 (fast dims) to ~0 (slow dims).",
            "   Crossover where wavelength = seq_len determines which dims are",
            "   informative for a given sequence length.",
        ]
        summary_text = "\n".join(summary_items)
        ax.text(0.06, 0.86, summary_text, fontsize=10, ha="left", va="top",
                transform=ax.transAxes, family="monospace", linespacing=1.3)
        pdf.savefig(fig)
        plt.close(fig)

        titles = {
            "01_pe_heatmap.png": "Example 1: PE Matrix Heatmap and Frequency Analysis",
            "02_dot_product_distance.png": "Example 2: Dot Product Distance Structure (Toeplitz Property)",
            "03_relative_position.png": "Example 3: Relative Position Property (Rotation Matrices)",
            "04_sinusoidal_vs_learned.png": "Example 4: Sinusoidal vs Learned Encoding Comparison",
            "05_attention_impact.png": "Example 5: Impact on Self-Attention (Permutation Invariance)",
            "06_frequency_analysis.png": "Example 6: Frequency Structure and Variance Analysis",
        }

        for viz_file in viz_files:
            fig = plt.figure(figsize=(11, 8.5))
            title = titles.get(viz_file.name, viz_file.stem.replace("_", " ").title())
            fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

            img = plt.imread(str(viz_file))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
            ax.imshow(img)
            ax.axis("off")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Report saved: report.pdf ({len(viz_files) + 2} pages)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Positional Encoding Demo")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print()

    example_1_pe_heatmap()
    example_2_dot_product_distance()
    example_3_relative_position()
    example_4_sinusoidal_vs_learned()
    example_5_attention_impact()
    example_6_frequency_analysis()
    generate_pdf_report()

    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print(f"Visualizations: {VIZ_DIR}/")
    print(f"Report: {Path(__file__).parent / 'report.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
