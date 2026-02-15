"""
Rotary Position Embeddings (RoPE) Demo -- Rotation visualization, relative position
property, norm preservation, sinusoidal comparison, attention impact, and context extension.

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
    precompute_freqs,
    rotate_half,
    apply_rope,
    apply_rope_complex,
    RoPE,
    rotation_is_orthogonal,
    verify_relative_position_property,
    compare_with_sinusoidal,
)

import importlib.util
_pe_spec = importlib.util.spec_from_file_location(
    "pe_impl",
    os.path.join(os.path.dirname(__file__), "..", "positional-encoding", "implementation.py"),
)
_pe_mod = importlib.util.module_from_spec(_pe_spec)
_pe_spec.loader.exec_module(_pe_mod)
sinusoidal_positional_encoding = _pe_mod.sinusoidal_positional_encoding

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
# Example 1: Rotation Visualization
# ---------------------------------------------------------------------------
def example_1_rotation_visualization():
    """For d=4, show how a vector rotates in each 2D subspace at different positions."""
    print("=" * 60)
    print("Example 1: Rotation Visualization")
    print("=" * 60)

    d = 4
    max_pos = 200
    theta_base = 10000.0

    cos_cache, sin_cache = precompute_freqs(d, max_pos, theta_base)

    np.random.seed(SEED)
    x = np.random.randn(1, 1, 1, d)
    x_norm = np.linalg.norm(x)

    print(f"\n  Config: d={d}, max_pos={max_pos}, theta_base={theta_base}")
    print(f"  Input vector: [{x[0,0,0,0]:.4f}, {x[0,0,0,1]:.4f}, {x[0,0,0,2]:.4f}, {x[0,0,0,3]:.4f}]")
    print(f"  Input norm: {x_norm:.6f}")

    freqs = np.exp(-2.0 * np.arange(d // 2, dtype=np.float64) / d * np.log(theta_base))
    print(f"\n  Frequency schedule:")
    print(f"    theta_0 = {freqs[0]:.6f}  (wavelength = {2*np.pi/freqs[0]:.2f} positions)")
    print(f"    theta_1 = {freqs[1]:.6f}  (wavelength = {2*np.pi/freqs[1]:.2f} positions)")
    print(f"  Pair (x0, x1) rotates {freqs[0]/freqs[1]:.1f}x faster than pair (x2, x3)")

    positions = np.arange(max_pos)
    pair0_x = np.zeros(max_pos)
    pair0_y = np.zeros(max_pos)
    pair1_x = np.zeros(max_pos)
    pair1_y = np.zeros(max_pos)
    norms = np.zeros(max_pos)

    for pos in positions:
        x_rot = apply_rope(x, cos_cache, sin_cache, np.array([pos]))
        pair0_x[pos] = x_rot[0, 0, 0, 0]
        pair0_y[pos] = x_rot[0, 0, 0, 1]
        pair1_x[pos] = x_rot[0, 0, 0, 2]
        pair1_y[pos] = x_rot[0, 0, 0, 3]
        norms[pos] = np.linalg.norm(x_rot)

    norm_error = np.max(np.abs(norms - x_norm))
    print(f"\n  Norm preservation: max |norm(RoPE(x,m)) - norm(x)| = {norm_error:.2e}")
    print(f"  This is guaranteed: rotation matrices are orthogonal, so ||Rx|| = ||x||.")

    pair0_r = np.sqrt(pair0_x**2 + pair0_y**2)
    pair1_r = np.sqrt(pair1_x**2 + pair1_y**2)
    expected_r0 = np.sqrt(x[0,0,0,0]**2 + x[0,0,0,1]**2)
    expected_r1 = np.sqrt(x[0,0,0,2]**2 + x[0,0,0,3]**2)
    print(f"  Pair 0 radius: expected={expected_r0:.6f}, max deviation={np.max(np.abs(pair0_r - expected_r0)):.2e}")
    print(f"  Pair 1 radius: expected={expected_r1:.6f}, max deviation={np.max(np.abs(pair1_r - expected_r1)):.2e}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Top-left: pair 0 trajectory (fast rotation)
    scatter0 = axes[0, 0].scatter(pair0_x, pair0_y, c=positions, cmap="viridis",
                                   s=15, alpha=0.8, zorder=3)
    circle0 = plt.Circle((0, 0), expected_r0, fill=False, color=COLORS["red"],
                          linestyle="--", linewidth=1.5, label=f"r = {expected_r0:.3f}")
    axes[0, 0].add_patch(circle0)
    axes[0, 0].scatter([pair0_x[0]], [pair0_y[0]], c="red", s=100, marker="*",
                        zorder=5, label="pos=0")
    axes[0, 0].set_xlabel("x0")
    axes[0, 0].set_ylabel("x1")
    axes[0, 0].set_title(f"Pair (x0, x1): theta_0 = {freqs[0]:.4f}\n"
                          f"Fast rotation, wavelength = {2*np.pi/freqs[0]:.1f} positions",
                          fontsize=10, fontweight="bold")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    fig.colorbar(scatter0, ax=axes[0, 0], label="Position", fraction=0.046, pad=0.04)

    # Top-middle: pair 1 trajectory (slow rotation)
    scatter1 = axes[0, 1].scatter(pair1_x, pair1_y, c=positions, cmap="viridis",
                                   s=15, alpha=0.8, zorder=3)
    circle1 = plt.Circle((0, 0), expected_r1, fill=False, color=COLORS["red"],
                          linestyle="--", linewidth=1.5, label=f"r = {expected_r1:.3f}")
    axes[0, 1].add_patch(circle1)
    axes[0, 1].scatter([pair1_x[0]], [pair1_y[0]], c="red", s=100, marker="*",
                        zorder=5, label="pos=0")
    axes[0, 1].set_xlabel("x2")
    axes[0, 1].set_ylabel("x3")
    axes[0, 1].set_title(f"Pair (x2, x3): theta_1 = {freqs[1]:.6f}\n"
                          f"Slow rotation, wavelength = {2*np.pi/freqs[1]:.0f} positions",
                          fontsize=10, fontweight="bold")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    fig.colorbar(scatter1, ax=axes[0, 1], label="Position", fraction=0.046, pad=0.04)

    # Top-right: dimension values vs position
    axes[0, 2].plot(positions, pair0_x, label="x0 (fast pair)", color=COLORS["blue"], linewidth=1.5)
    axes[0, 2].plot(positions, pair0_y, label="x1 (fast pair)", color=COLORS["red"], linewidth=1.5)
    axes[0, 2].plot(positions, pair1_x, label="x2 (slow pair)", color=COLORS["green"], linewidth=1.5)
    axes[0, 2].plot(positions, pair1_y, label="x3 (slow pair)", color=COLORS["orange"], linewidth=1.5)
    axes[0, 2].set_xlabel("Position")
    axes[0, 2].set_ylabel("Dimension Value")
    axes[0, 2].set_title("All Dimensions vs Position\nFast pair oscillates rapidly, slow pair barely moves",
                          fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Bottom-left: norm vs position
    axes[1, 0].plot(positions, norms, color=COLORS["blue"], linewidth=2, label="||RoPE(x, m)||")
    axes[1, 0].axhline(x_norm, color=COLORS["red"], linestyle="--", linewidth=1.5,
                        label=f"||x|| = {x_norm:.4f}")
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_ylabel("Norm")
    axes[1, 0].set_title(f"Norm Preservation (max error = {norm_error:.2e})\n"
                          f"Guaranteed by orthogonality: ||Rx|| = ||x||",
                          fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(x_norm - 0.01, x_norm + 0.01)

    # Bottom-middle: rotation angle vs position for both pairs
    angle0 = np.arctan2(pair0_y, pair0_x)
    angle1 = np.arctan2(pair1_y, pair1_x)
    initial_angle0 = np.arctan2(x[0,0,0,1], x[0,0,0,0])
    initial_angle1 = np.arctan2(x[0,0,0,3], x[0,0,0,2])
    # Unwrap for continuous plotting
    angle0_unwrap = np.unwrap(angle0)
    angle1_unwrap = np.unwrap(angle1)
    axes[1, 1].plot(positions, angle0_unwrap - angle0_unwrap[0], color=COLORS["blue"],
                    linewidth=2, label=f"Pair 0: slope = theta_0 = {freqs[0]:.4f} rad/pos")
    axes[1, 1].plot(positions, angle1_unwrap - angle1_unwrap[0], color=COLORS["green"],
                    linewidth=2, label=f"Pair 1: slope = theta_1 = {freqs[1]:.6f} rad/pos")
    axes[1, 1].plot(positions, freqs[0] * positions, "--", color=COLORS["blue"], alpha=0.5)
    axes[1, 1].plot(positions, freqs[1] * positions, "--", color=COLORS["green"], alpha=0.5)
    axes[1, 1].set_xlabel("Position")
    axes[1, 1].set_ylabel("Accumulated Rotation Angle (radians)")
    axes[1, 1].set_title("Rotation Angle = position * theta_i\nLinear accumulation, different rates per pair",
                          fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Bottom-right: frequency spectrum for larger d
    d_large = 128
    i_vals = np.arange(d_large // 2)
    freqs_large = np.exp(-2.0 * i_vals / d_large * np.log(theta_base))
    wavelengths = 2 * np.pi / freqs_large

    axes[1, 2].semilogy(i_vals, wavelengths, "o-", color=COLORS["purple"],
                         markersize=2, linewidth=1.5)
    axes[1, 2].axhline(2 * np.pi, color=COLORS["red"], linestyle="--", alpha=0.6,
                        label=f"Min wavelength: 2pi = {2*np.pi:.1f}")
    axes[1, 2].set_xlabel("Dimension Pair Index (i)")
    axes[1, 2].set_ylabel("Wavelength (positions)")
    axes[1, 2].set_title(f"Wavelength per Pair (d={d_large})\nGeometric: 2pi to {wavelengths[-1]:.0f}",
                          fontsize=10, fontweight="bold")
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("RoPE Rotation Visualization: Each Dimension Pair Traces a Circle",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_rotation_visualization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/01_rotation_visualization.png")


# ---------------------------------------------------------------------------
# Example 2: Relative Position Property (THE key property)
# ---------------------------------------------------------------------------
def example_2_relative_position():
    """The CENTERPIECE: RoPE(q,m)^T RoPE(k,n) depends only on (m-n)."""
    print("\n" + "=" * 60)
    print("Example 2: Relative Position Property")
    print("=" * 60)

    d = 64
    max_seq_len = 2048
    rope = RoPE(d, max_seq_len)

    np.random.seed(SEED)
    q = np.random.randn(1, 1, 1, d)
    k = np.random.randn(1, 1, 1, d)

    print(f"\n  Config: d={d}, max_seq_len={max_seq_len}")
    print(f"  q norm: {np.linalg.norm(q):.6f}, k norm: {np.linalg.norm(k):.6f}")

    # --- Analytical derivation ---
    print("\n  ANALYTICAL DERIVATION:")
    print("  ----------------------")
    print("  <RoPE(q,m), RoPE(k,n)> = q^T R(m)^T R(n) k = q^T R(n-m) k")
    print("  This uses: R(m)^T = R(-m) (orthogonal) and R(-m)R(n) = R(n-m) (composition)")
    print("")
    print("  Per dimension pair i:")
    print("    (q_{2i} k_{2i} + q_{2i+1} k_{2i+1}) cos((m-n) theta_i)")
    print("    + (q_{2i} k_{2i+1} - q_{2i+1} k_{2i}) sin((m-n) theta_i)")
    print("  This depends on (m-n) only, NOT on m or n individually.")

    # Verify: compute dot product at many (m,n) pairs with same relative distance
    print(f"\n  EMPIRICAL VERIFICATION:")
    relative_distances = np.arange(-50, 51)
    dot_products_by_distance = {}

    for rel_dist in relative_distances:
        dots = []
        test_positions = [0, 50, 100, 200, 500, 800]
        for m_base in test_positions:
            n = m_base
            m = m_base + rel_dist
            if 0 <= m < max_seq_len and 0 <= n < max_seq_len:
                q_rot = apply_rope(q, rope.cos_cache, rope.sin_cache, np.array([m]))
                k_rot = apply_rope(k, rope.cos_cache, rope.sin_cache, np.array([n]))
                dot = np.sum(q_rot * k_rot)
                dots.append(dot)
        if len(dots) >= 2:
            dot_products_by_distance[rel_dist] = dots

    max_variation = 0.0
    for rel_dist, dots in dot_products_by_distance.items():
        variation = max(dots) - min(dots)
        max_variation = max(max_variation, variation)

    print(f"  Tested {len(dot_products_by_distance)} relative distances, 6 absolute positions each")
    print(f"  Max variation at any fixed relative distance: {max_variation:.2e}")
    print(f"  (Should be ~1e-13: analytically zero, limited by float64)")

    # Also verify with the built-in function
    diffs = verify_relative_position_property(
        q, k, rope, np.array([10]), np.array([5])
    )
    print(f"\n  Built-in verification (m=10, n=5, deltas=[1,10,50,100,500]):")
    print(f"  Max absolute difference: {diffs.max():.2e}")

    # Compute dot product curve as function of relative distance
    mean_dots = np.zeros(len(relative_distances))
    for i, rel_dist in enumerate(relative_distances):
        if rel_dist in dot_products_by_distance:
            mean_dots[i] = np.mean(dot_products_by_distance[rel_dist])

    # Analytical formula
    freqs = rope.inv_freq
    q_flat = q[0, 0, 0]
    k_flat = k[0, 0, 0]
    analytical_dots = np.zeros(len(relative_distances))
    for i, delta in enumerate(relative_distances):
        dot_sum = 0.0
        for j in range(d // 2):
            q2i, q2ip1 = q_flat[2*j], q_flat[2*j+1]
            k2i, k2ip1 = k_flat[2*j], k_flat[2*j+1]
            cos_term = (q2i * k2i + q2ip1 * k2ip1) * np.cos(delta * freqs[j])
            sin_term = (q2i * k2ip1 - q2ip1 * k2i) * np.sin(delta * freqs[j])
            dot_sum += cos_term + sin_term
        analytical_dots[i] = dot_sum

    formula_error = np.max(np.abs(mean_dots - analytical_dots))
    print(f"\n  Analytical formula match: max error = {formula_error:.2e}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Top-left: dot product vs relative distance
    axes[0, 0].plot(relative_distances, mean_dots, "-", color=COLORS["blue"],
                    linewidth=2, label="Observed (mean over positions)")
    axes[0, 0].plot(relative_distances, analytical_dots, "--", color=COLORS["red"],
                    linewidth=1.5, alpha=0.8, label="Analytical formula")
    axes[0, 0].axvline(0, color="gray", linestyle=":", alpha=0.5)
    axes[0, 0].set_xlabel("Relative Distance (m - n)")
    axes[0, 0].set_ylabel("Dot Product <RoPE(q,m), RoPE(k,n)>")
    axes[0, 0].set_title("Dot Product Depends ONLY on Relative Distance\n"
                          "Analytical: sum_i [A_i cos(delta*theta_i) + B_i sin(delta*theta_i)]",
                          fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-middle: overlay multiple absolute positions for same relative distance
    test_rel_dists = [0, 1, 5, 10, 20]
    for rel_dist in test_rel_dists:
        if rel_dist in dot_products_by_distance:
            dots = dot_products_by_distance[rel_dist]
            axes[0, 1].plot(range(len(dots)), dots, "o-", markersize=6, alpha=0.8,
                            label=f"delta={rel_dist}")
    axes[0, 1].set_xlabel("Position Index (different absolute positions)")
    axes[0, 1].set_ylabel("Dot Product")
    axes[0, 1].set_title("Position Independence: Same Delta -> Same Dot Product\n"
                          "Each curve is flat (all points overlap)",
                          fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Top-right: variation at each relative distance
    variations = []
    dists_for_var = []
    for rel_dist in relative_distances:
        if rel_dist in dot_products_by_distance and len(dot_products_by_distance[rel_dist]) >= 2:
            dots = dot_products_by_distance[rel_dist]
            variations.append(max(dots) - min(dots))
            dists_for_var.append(rel_dist)

    axes[0, 2].semilogy(dists_for_var, variations, ".", color=COLORS["purple"],
                         markersize=4, alpha=0.7)
    axes[0, 2].axhline(1e-12, color=COLORS["red"], linestyle="--", alpha=0.5,
                        label="1e-12 (float64 noise)")
    axes[0, 2].set_xlabel("Relative Distance")
    axes[0, 2].set_ylabel("Max - Min Dot Product (across positions)")
    axes[0, 2].set_title(f"Variation is Float64 Noise (~1e-13)\n"
                          f"Max variation: {max_variation:.2e}",
                          fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Bottom-left: per-pair contribution to dot product
    delta_show = 5
    pair_contributions_cos = np.zeros(d // 2)
    pair_contributions_sin = np.zeros(d // 2)
    for j in range(d // 2):
        q2i, q2ip1 = q_flat[2*j], q_flat[2*j+1]
        k2i, k2ip1 = k_flat[2*j], k_flat[2*j+1]
        pair_contributions_cos[j] = (q2i * k2i + q2ip1 * k2ip1) * np.cos(delta_show * freqs[j])
        pair_contributions_sin[j] = (q2i * k2ip1 - q2ip1 * k2i) * np.sin(delta_show * freqs[j])

    pair_indices = np.arange(d // 2)
    axes[1, 0].bar(pair_indices - 0.15, pair_contributions_cos, 0.3,
                    label="cos term", color=COLORS["blue"], alpha=0.8)
    axes[1, 0].bar(pair_indices + 0.15, pair_contributions_sin, 0.3,
                    label="sin term", color=COLORS["red"], alpha=0.8)
    axes[1, 0].set_xlabel("Dimension Pair Index (i)")
    axes[1, 0].set_ylabel("Contribution to Dot Product")
    axes[1, 0].set_title(f"Per-Pair Breakdown at delta={delta_show}\n"
                          f"Total = sum of all bars = {np.sum(pair_contributions_cos + pair_contributions_sin):.4f}",
                          fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-middle: analytical derivation diagram
    axes[1, 1].axis("off")
    derivation = (
        "ANALYTICAL DERIVATION\n"
        "=====================\n\n"
        "Given: q' = R(m)q,  k' = R(n)k\n\n"
        "<q', k'> = (R(m)q)^T (R(n)k)\n"
        "         = q^T R(m)^T R(n) k\n"
        "         = q^T R(-m) R(n) k      [R^T = R^{-1} = R(-m)]\n"
        "         = q^T R(n-m) k           [R(a)R(b) = R(a+b)]\n\n"
        "Per pair i:\n"
        "  (q_{2i}k_{2i} + q_{2i+1}k_{2i+1}) cos((m-n) theta_i)\n"
        "  + (q_{2i}k_{2i+1} - q_{2i+1}k_{2i}) sin((m-n) theta_i)\n\n"
        "KEY: depends on (m-n), q, k only.\n"
        "NOT on m or n individually.\n\n"
        "This is why RoPE is a RELATIVE\n"
        "position encoding despite using\n"
        "ABSOLUTE positions in the rotation."
    )
    axes[1, 1].text(0.05, 0.95, derivation, fontsize=10, ha="left", va="top",
                    family="monospace", transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    # Bottom-right: long-range dot product decay
    long_dists = np.arange(0, 500)
    long_dots = np.zeros(len(long_dists))
    for i, delta in enumerate(long_dists):
        dot_sum = 0.0
        for j in range(d // 2):
            q2i, q2ip1 = q_flat[2*j], q_flat[2*j+1]
            k2i, k2ip1 = k_flat[2*j], k_flat[2*j+1]
            cos_t = (q2i * k2i + q2ip1 * k2ip1) * np.cos(delta * freqs[j])
            sin_t = (q2i * k2ip1 - q2ip1 * k2i) * np.sin(delta * freqs[j])
            dot_sum += cos_t + sin_t
        long_dots[i] = dot_sum

    axes[1, 2].plot(long_dists, long_dots, color=COLORS["teal"], linewidth=1.5)
    axes[1, 2].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[1, 2].set_xlabel("Relative Distance (delta)")
    axes[1, 2].set_ylabel("Dot Product")
    axes[1, 2].set_title("Long-Range Dot Product (for this random q, k)\n"
                          "High-freq terms oscillate rapidly; only low-freq contribute coherently",
                          fontsize=10, fontweight="bold")
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("RoPE Relative Position Property: Dot Product Depends Only on (m-n)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_relative_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/02_relative_position.png")


# ---------------------------------------------------------------------------
# Example 3: Norm Preservation & Orthogonality
# ---------------------------------------------------------------------------
def example_3_norm_orthogonality():
    """Verify ||RoPE(x,m)|| = ||x||, R^T R = I, det(R) = 1, and R(m)R(n) = R(m+n)."""
    print("\n" + "=" * 60)
    print("Example 3: Norm Preservation & Orthogonality")
    print("=" * 60)

    d = 16
    max_seq_len = 2048
    cos_cache, sin_cache = precompute_freqs(d, max_seq_len)

    # --- Norm preservation across many vectors and positions ---
    np.random.seed(SEED)
    num_vectors = 100
    X = np.random.randn(num_vectors, 1, 1, d)
    test_positions = [0, 1, 10, 100, 500, 1000, 1500, 2000]

    norm_errors = np.zeros((num_vectors, len(test_positions)))
    for j, pos in enumerate(test_positions):
        X_rot = apply_rope(X, cos_cache, sin_cache, np.array([pos]))
        original_norms = np.linalg.norm(X.reshape(num_vectors, d), axis=1)
        rotated_norms = np.linalg.norm(X_rot.reshape(num_vectors, d), axis=1)
        norm_errors[:, j] = np.abs(rotated_norms - original_norms)

    max_norm_err = norm_errors.max()
    print(f"\n  Config: d={d}, {num_vectors} random vectors, {len(test_positions)} positions")
    print(f"  Max norm error: {max_norm_err:.2e}")
    print(f"  Analytically guaranteed: R(m) is orthogonal -> ||R(m)x|| = ||x||")

    # --- Orthogonality: R^T R = I and det(R) = 1 ---
    print(f"\n  Rotation matrix properties:")
    print(f"  {'Position':>10} {'||R R^T - I||_F':>18} {'det(R)':>12} {'Orthogonal?':>14}")
    print(f"  {'-'*58}")

    ortho_positions = [0, 1, 10, 50, 100, 500, 1000]
    frob_errors = []
    determinants = []
    for pos in ortho_positions:
        R, frob_err, det_val = rotation_is_orthogonal(cos_cache, sin_cache, pos)
        frob_errors.append(frob_err)
        determinants.append(det_val)
        is_ortho = frob_err < 1e-12 and abs(det_val - 1.0) < 1e-12
        print(f"  {pos:>10} {frob_err:>18.2e} {det_val:>12.10f} {'YES' if is_ortho else 'NO':>14}")

    print(f"\n  These are analytically EXACT properties:")
    print(f"  - R^T R = I because each 2x2 block is [[cos,-sin],[sin,cos]]^T [[cos,-sin],[sin,cos]] = I")
    print(f"  - det(R) = prod of det(R_i) = prod of (cos^2 + sin^2) = prod of 1 = 1")

    # --- Composition property: R(m)R(n) = R(m+n) ---
    m, n = 17, 23
    R_m, _, _ = rotation_is_orthogonal(cos_cache, sin_cache, m)
    R_n, _, _ = rotation_is_orthogonal(cos_cache, sin_cache, n)
    R_mn, _, _ = rotation_is_orthogonal(cos_cache, sin_cache, m + n)

    composition_error = np.linalg.norm(R_m @ R_n - R_mn)
    print(f"\n  Composition property: R({m}) @ R({n}) = R({m+n})")
    print(f"  ||R({m})R({n}) - R({m+n})||_F = {composition_error:.2e}")
    print(f"  Analytically: each block R_i(m)R_i(n) = R_i(m+n) by trig addition formulas")

    # --- Inverse property: R(m)R(-m) = I ---
    # Since we can't have negative positions in the cache, construct R(-m) manually
    cos_neg = np.cos(-m * np.exp(-2.0 * np.arange(d // 2, dtype=np.float64) / d * np.log(10000.0)))
    sin_neg = np.sin(-m * np.exp(-2.0 * np.arange(d // 2, dtype=np.float64) / d * np.log(10000.0)))
    R_neg_m = np.zeros((d, d), dtype=np.float64)
    for i in range(d // 2):
        idx = 2 * i
        R_neg_m[idx, idx] = cos_neg[i]
        R_neg_m[idx, idx + 1] = -sin_neg[i]
        R_neg_m[idx + 1, idx] = sin_neg[i]
        R_neg_m[idx + 1, idx + 1] = cos_neg[i]

    inverse_error = np.linalg.norm(R_m @ R_neg_m - np.eye(d))
    print(f"\n  Inverse property: R({m}) @ R(-{m}) = I")
    print(f"  ||R({m})R(-{m}) - I||_F = {inverse_error:.2e}")
    print(f"  Analytically: R^T = R^(-1) for orthogonal matrices, R(-m) = R(m)^T")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Top-left: norm errors heatmap
    im0 = axes[0, 0].imshow(norm_errors, aspect="auto", cmap="Reds", interpolation="nearest")
    axes[0, 0].set_xlabel("Position Index")
    axes[0, 0].set_ylabel("Vector Index")
    axes[0, 0].set_xticks(range(len(test_positions)))
    axes[0, 0].set_xticklabels(test_positions, fontsize=8)
    axes[0, 0].set_title(f"||RoPE(x,m)|| - ||x|| across {num_vectors} vectors\n"
                          f"All ~0 (max = {max_norm_err:.2e})",
                          fontsize=10, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0], label="Absolute Error", fraction=0.046, pad=0.04)

    # Top-middle: R^T R - I visualization for one position
    R_example, _, _ = rotation_is_orthogonal(cos_cache, sin_cache, 42)
    RtR_err = R_example @ R_example.T - np.eye(d)
    im1 = axes[0, 1].imshow(np.abs(RtR_err), cmap="Blues", aspect="equal", interpolation="nearest")
    max_entry_err = np.max(np.abs(RtR_err))
    axes[0, 1].set_title(f"$|R(42)^T R(42) - I|$ (d={d})\nMax entry error: {max_entry_err:.1e}",
                          fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("Column")
    axes[0, 1].set_ylabel("Row")
    fig.colorbar(im1, ax=axes[0, 1], label="|Error|", fraction=0.046, pad=0.04)

    # Top-right: det(R) and frob error vs position
    ax2_twin = axes[0, 2].twinx()
    axes[0, 2].plot(ortho_positions, frob_errors, "o-", color=COLORS["blue"],
                    linewidth=2, markersize=6, label="||R R^T - I||_F")
    ax2_twin.plot(ortho_positions, [abs(det_v - 1.0) for det_v in determinants], "s-",
                  color=COLORS["red"], linewidth=2, markersize=6, label="|det(R) - 1|")
    axes[0, 2].set_xlabel("Position")
    axes[0, 2].set_ylabel("||R R^T - I||_F", color=COLORS["blue"])
    ax2_twin.set_ylabel("|det(R) - 1|", color=COLORS["red"])
    max_frob = max(frob_errors)
    max_det_err = max(abs(det_v - 1.0) for det_v in determinants)
    axes[0, 2].set_title(f"Orthogonality Verified at All Positions\n"
                          f"Max Frob err: {max_frob:.1e}, max |det-1|: {max_det_err:.1e}",
                          fontsize=10, fontweight="bold")
    axes[0, 2].legend(loc="upper left", fontsize=8)
    ax2_twin.legend(loc="upper right", fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Bottom-left: composition R(m)R(n) = R(m+n)
    R_prod = R_m @ R_n
    im3 = axes[1, 0].imshow(np.abs(R_prod - R_mn), cmap="Blues", aspect="equal",
                              interpolation="nearest")
    axes[1, 0].set_title(f"$|R({m})R({n}) - R({m+n})|$\nComposition error: {composition_error:.2e}",
                          fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Column")
    axes[1, 0].set_ylabel("Row")
    fig.colorbar(im3, ax=axes[1, 0], label="|Error|", fraction=0.046, pad=0.04)

    # Bottom-middle: block-diagonal structure of R
    R_show, _, _ = rotation_is_orthogonal(cos_cache, sin_cache, 5)
    im4 = axes[1, 1].imshow(R_show, cmap="RdBu_r", aspect="equal", interpolation="nearest",
                              vmin=-1, vmax=1)
    axes[1, 1].set_title(f"R(5) Structure (d={d})\nBlock-diagonal: {d//2} independent 2x2 rotation blocks",
                          fontsize=10, fontweight="bold")
    axes[1, 1].set_xlabel("Column")
    axes[1, 1].set_ylabel("Row")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Bottom-right: properties summary
    axes[1, 2].axis("off")
    summary = (
        "ROTATION MATRIX PROPERTIES\n"
        "==========================\n\n"
        f"R is {d}x{d} block-diagonal with {d//2} blocks\n\n"
        "Each 2x2 block R_i(m):\n"
        "  [[cos(m*theta_i), -sin(m*theta_i)],\n"
        "   [sin(m*theta_i),  cos(m*theta_i)]]\n\n"
        "Guaranteed properties:\n"
        "  1. Orthogonality: R^T R = I\n"
        f"     Verified: ||R^T R - I|| < {max(frob_errors):.0e}\n\n"
        "  2. Proper rotation: det(R) = 1\n"
        f"     Verified: |det - 1| < {max(abs(det_v-1) for det_v in determinants):.0e}\n\n"
        "  3. Composition: R(m)R(n) = R(m+n)\n"
        f"     Verified: error = {composition_error:.0e}\n\n"
        "  4. Inverse: R(m)R(-m) = I\n"
        f"     Verified: error = {inverse_error:.0e}\n\n"
        "  5. Norm preservation: ||Rx|| = ||x||\n"
        f"     Verified: max error = {max_norm_err:.0e}\n\n"
        "All from cos^2 + sin^2 = 1."
    )
    axes[1, 2].text(0.05, 0.95, summary, fontsize=10, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("RoPE Orthogonality: R^T R = I, det(R) = 1, R(m)R(n) = R(m+n), ||Rx|| = ||x||",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_norm_orthogonality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/03_norm_orthogonality.png")


# ---------------------------------------------------------------------------
# Example 4: RoPE vs Sinusoidal PE Comparison
# ---------------------------------------------------------------------------
def example_4_rope_vs_sinusoidal():
    """Both use same freq schedule, but ADDITIVE vs MULTIPLICATIVE application."""
    print("\n" + "=" * 60)
    print("Example 4: RoPE vs Sinusoidal PE Comparison")
    print("=" * 60)

    d = 64
    seq_len = 128
    max_seq_len = 256

    rope_freqs, sin_freqs = compare_with_sinusoidal(d, seq_len)
    freq_match = np.allclose(rope_freqs, sin_freqs)
    print(f"\n  Config: d={d}, seq_len={seq_len}")
    print(f"  Frequencies match: {freq_match} (both use theta_i = 10000^(-2i/d))")

    rope = RoPE(d, max_seq_len)
    pe = sinusoidal_positional_encoding(max_seq_len, d)

    # --- Key difference: absolute vs relative position in dot products ---
    np.random.seed(SEED)
    q_content = np.random.randn(d)
    k_content = np.random.randn(d)

    print(f"\n  FUNDAMENTAL DIFFERENCE:")
    print(f"  Sinusoidal PE: ADDITIVE -- q = (x + PE[m]) @ W_Q, position mixes with content")
    print(f"  RoPE:          MULTIPLICATIVE -- q' = R(m) @ (x @ W_Q), rotation after projection")
    print(f"")
    print(f"  (a) Sinusoidal: dot product depends on q, k, AND absolute positions m, n separately")
    print(f"  (b) RoPE: dot product depends on q, k, and RELATIVE position (m-n) only")
    print(f"")
    print(f"  Note: below we use q_sin = q + PE[m] (identity projection, W_Q = I).")
    print(f"  The PE[m]^T PE[n] term has the Toeplitz property (depends on m-n only),")
    print(f"  but cross-terms q^T PE[n] and PE[m]^T k still depend on absolute position.")
    print(f"  With a real W_Q projection, the structure is similar: cross-terms remain.")

    # Sinusoidal PE: q_sin(m) = q_content + PE[m], k_sin(n) = k_content + PE[n]
    # Dot product = (q + PE[m])^T (k + PE[n]) = q^T k + q^T PE[n] + PE[m]^T k + PE[m]^T PE[n]
    # The PE[m]^T PE[n] term depends on relative position only (Toeplitz property from Example 2 of PE demo).
    # But q^T PE[n] depends on absolute n and PE[m]^T k depends on absolute m.
    # So the TOTAL depends on absolute positions.

    sin_dots_same_rel = {}
    rope_dots_same_rel = {}

    test_deltas = [0, 1, 3, 5, 10, 20]
    for delta in test_deltas:
        sin_dots_same_rel[delta] = []
        rope_dots_same_rel[delta] = []
        for m_base in [0, 20, 40, 60, 80, 100]:
            m = m_base + delta
            n = m_base
            if m >= max_seq_len:
                continue

            # Sinusoidal: additive
            q_sin = q_content + pe[m]
            k_sin = k_content + pe[n]
            sin_dot = q_sin @ k_sin
            sin_dots_same_rel[delta].append(sin_dot)

            # RoPE: multiplicative
            q_rope = q_content.reshape(1, 1, 1, d)
            k_rope = k_content.reshape(1, 1, 1, d)
            q_rot = apply_rope(q_rope, rope.cos_cache, rope.sin_cache, np.array([m]))
            k_rot = apply_rope(k_rope, rope.cos_cache, rope.sin_cache, np.array([n]))
            rope_dot = np.sum(q_rot * k_rot)
            rope_dots_same_rel[delta].append(rope_dot)

    print(f"\n  {'Delta':>8} {'Sin PE Variation':>18} {'RoPE Variation':>18}")
    print(f"  {'-'*48}")
    for delta in test_deltas:
        sin_var = max(sin_dots_same_rel[delta]) - min(sin_dots_same_rel[delta])
        rope_var = max(rope_dots_same_rel[delta]) - min(rope_dots_same_rel[delta])
        print(f"  {delta:>8} {sin_var:>18.6f} {rope_var:>18.2e}")

    print(f"\n  Sinusoidal variation is LARGE because q^T PE[n] and PE[m]^T k")
    print(f"  depend on absolute positions m, n (not just m-n).")
    print(f"  RoPE variation is ~0 (float64 noise) because the dot product")
    print(f"  analytically depends ONLY on (m-n).")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Top-left: frequency comparison
    i_vals = np.arange(d // 2)
    axes[0, 0].semilogy(i_vals, rope_freqs, "o-", color=COLORS["blue"], markersize=3,
                         linewidth=1.5, label="RoPE frequencies")
    axes[0, 0].semilogy(i_vals, sin_freqs, "x", color=COLORS["red"], markersize=6,
                         alpha=0.7, label="Sinusoidal PE frequencies")
    axes[0, 0].set_xlabel("Dimension Pair Index (i)")
    axes[0, 0].set_ylabel("Frequency theta_i")
    axes[0, 0].set_title("Same Frequency Schedule\nBoth use theta_i = 10000^(-2i/d)",
                          fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-middle: sinusoidal dot product variation
    for delta in [0, 5, 10, 20]:
        if delta in sin_dots_same_rel:
            axes[0, 1].plot(range(len(sin_dots_same_rel[delta])),
                           sin_dots_same_rel[delta], "o-", markersize=6,
                           label=f"delta={delta}")
    axes[0, 1].set_xlabel("Position Pair Index")
    axes[0, 1].set_ylabel("Dot Product")
    axes[0, 1].set_title("Sinusoidal PE: NOT Position-Independent\n"
                          "(q+PE[m])^T(k+PE[n]) varies with absolute position",
                          fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Top-right: RoPE dot product invariance
    for delta in [0, 5, 10, 20]:
        if delta in rope_dots_same_rel:
            axes[0, 2].plot(range(len(rope_dots_same_rel[delta])),
                           rope_dots_same_rel[delta], "o-", markersize=6,
                           label=f"delta={delta}")
    axes[0, 2].set_xlabel("Position Pair Index")
    axes[0, 2].set_ylabel("Dot Product")
    axes[0, 2].set_title("RoPE: Perfectly Position-Independent\n"
                          "RoPE(q,m)^T RoPE(k,n) flat across absolute positions",
                          fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Bottom-left: sinusoidal PE decomposition
    # q^T k + q^T PE[n] + PE[m]^T k + PE[m]^T PE[n]
    m_test, n_test = 50, 45
    q_k = q_content @ k_content
    q_pe_n = q_content @ pe[n_test]
    pe_m_k = pe[m_test] @ k_content
    pe_m_pe_n = pe[m_test] @ pe[n_test]
    total = q_k + q_pe_n + pe_m_k + pe_m_pe_n

    terms = [q_k, q_pe_n, pe_m_k, pe_m_pe_n]
    term_labels = ["q^T k\n(content)", "q^T PE[n]\n(abs. pos n)", "PE[m]^T k\n(abs. pos m)",
                   "PE[m]^T PE[n]\n(rel. pos)"]
    bar_colors_terms = [COLORS["green"], COLORS["red"], COLORS["orange"], COLORS["blue"]]
    abs_dep = ["No", "YES (n)", "YES (m)", "No"]

    bars = axes[1, 0].bar(range(4), terms, color=bar_colors_terms, edgecolor="white", width=0.5)
    axes[1, 0].set_xticks(range(4))
    axes[1, 0].set_xticklabels(term_labels, fontsize=8)
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title(f"Sinusoidal Dot Product Decomposition (m={m_test}, n={n_test})\n"
                          f"2 of 4 terms depend on absolute position",
                          fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    for bar, dep in zip(bars, abs_dep):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"Abs. dep: {dep}", ha="center", va="bottom" if bar.get_height() >= 0 else "top",
                        fontsize=8, fontweight="bold")

    # Bottom-middle: application comparison diagram
    axes[1, 1].axis("off")
    comparison = (
        "COMPARISON: ADDITIVE vs MULTIPLICATIVE\n"
        "=======================================\n\n"
        "SINUSOIDAL PE (Additive):\n"
        "  input' = input + PE[position]\n"
        "  q = input' @ W_Q\n"
        "  k = input' @ W_K\n"
        "  score = q^T k\n"
        "        = (x+PE[m])^T W_Q^T W_K (x+PE[n])\n"
        "  --> cross-terms depend on abs. position\n\n"
        "RoPE (Multiplicative):\n"
        "  q = input @ W_Q\n"
        "  k = input @ W_K\n"
        "  q' = R(m) @ q     (rotate AFTER proj.)\n"
        "  k' = R(n) @ k\n"
        "  score = q'^T k'\n"
        "        = q^T R(m)^T R(n) k\n"
        "        = q^T R(n-m) k\n"
        "  --> depends on (n-m) only!"
    )
    axes[1, 1].text(0.05, 0.95, comparison, fontsize=10, ha="left", va="top",
                    family="monospace", transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.9))

    # Bottom-right: variation comparison bar chart
    deltas_plot = [0, 1, 5, 10, 20]
    sin_vars = [max(sin_dots_same_rel[d]) - min(sin_dots_same_rel[d]) for d in deltas_plot]
    rope_vars = [max(rope_dots_same_rel[d]) - min(rope_dots_same_rel[d]) for d in deltas_plot]

    x_pos = np.arange(len(deltas_plot))
    axes[1, 2].bar(x_pos - 0.15, sin_vars, 0.3, label="Sinusoidal PE", color=COLORS["red"],
                    edgecolor="white")
    axes[1, 2].bar(x_pos + 0.15, rope_vars, 0.3, label="RoPE", color=COLORS["blue"],
                    edgecolor="white")
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([f"delta={d}" for d in deltas_plot], fontsize=9)
    axes[1, 2].set_ylabel("Variation Across Absolute Positions")
    axes[1, 2].set_title("Dot Product Variation: Sinusoidal vs RoPE\n"
                          "RoPE has zero variation (true relative encoding)",
                          fontsize=10, fontweight="bold")
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3, axis="y")
    axes[1, 2].set_yscale("symlog", linthresh=1e-10)

    fig.suptitle("RoPE vs Sinusoidal PE: Same Frequencies, Fundamentally Different Application",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_rope_vs_sinusoidal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/04_rope_vs_sinusoidal.png")


# ---------------------------------------------------------------------------
# Example 5: Impact on Attention Patterns
# ---------------------------------------------------------------------------
def example_5_attention_impact():
    """Apply self-attention with and without RoPE; show position-dependent patterns."""
    print("\n" + "=" * 60)
    print("Example 5: Impact on Attention Patterns")
    print("=" * 60)

    d = 16
    seq_len = 12
    num_heads = 1
    batch_size = 1

    rope = RoPE(d, seq_len + 10)

    np.random.seed(SEED)
    W_Q = np.random.randn(d, d) * 0.1
    W_K = np.random.randn(d, d) * 0.1

    # Create input: ALL IDENTICAL token embeddings to isolate positional effects
    token_embed = np.random.randn(d)
    X = np.tile(token_embed, (batch_size, seq_len, 1))

    print(f"\n  Config: d={d}, seq_len={seq_len}")
    print(f"  All tokens have IDENTICAL embeddings to isolate position effects")

    Q = (X @ W_Q).reshape(batch_size, seq_len, num_heads, d).transpose(0, 2, 1, 3)
    K = (X @ W_K).reshape(batch_size, seq_len, num_heads, d).transpose(0, 2, 1, 3)
    scale = np.sqrt(d)

    # Causal mask
    causal_mask = np.zeros((seq_len, seq_len))
    causal_mask[np.triu_indices(seq_len, k=1)] = -np.inf

    # WITHOUT RoPE
    scores_no_rope = (Q @ K.transpose(0, 1, 3, 2)) / scale + causal_mask
    attn_no_rope = np.exp(scores_no_rope - np.max(scores_no_rope, axis=-1, keepdims=True))
    attn_no_rope /= attn_no_rope.sum(axis=-1, keepdims=True)

    # WITH RoPE
    Q_rot, K_rot = rope.forward(Q, K)
    scores_rope = (Q_rot @ K_rot.transpose(0, 1, 3, 2)) / scale + causal_mask
    attn_rope = np.exp(scores_rope - np.max(scores_rope, axis=-1, keepdims=True))
    attn_rope /= attn_rope.sum(axis=-1, keepdims=True)

    print(f"\n  WITHOUT RoPE (identical tokens):")
    print(f"    Attention is UNIFORM within causal mask (all q^T k products identical)")
    # Compute variance over UNMASKED positions only (not masked zeros)
    unmasked_var_no_rope = []
    for i in range(1, seq_len):
        unmasked_var_no_rope.append(np.var(attn_no_rope[0, 0, i, :i+1]))
    mean_unmasked_var_no_rope = np.mean(unmasked_var_no_rope)
    print(f"    Mean within-row variance (unmasked only): {mean_unmasked_var_no_rope:.2e}")

    print(f"\n  WITH RoPE (identical tokens):")
    print(f"    Attention becomes POSITION-DEPENDENT (weights vary by relative position)")
    unmasked_var_rope = []
    for i in range(1, seq_len):
        unmasked_var_rope.append(np.var(attn_rope[0, 0, i, :i+1]))
    mean_unmasked_var_rope = np.mean(unmasked_var_rope)
    print(f"    Mean within-row variance (unmasked only): {mean_unmasked_var_rope:.2e}")
    print(f"    (No-RoPE: {mean_unmasked_var_no_rope:.2e} vs With-RoPE: {mean_unmasked_var_rope:.2e})")

    # Shift test: shift input, attention pattern should shift correspondingly
    shift = 3
    X_shifted = np.tile(token_embed, (batch_size, seq_len, 1))
    Q_shifted = (X_shifted @ W_Q).reshape(batch_size, seq_len, num_heads, d).transpose(0, 2, 1, 3)
    K_shifted = (X_shifted @ W_K).reshape(batch_size, seq_len, num_heads, d).transpose(0, 2, 1, 3)

    shifted_positions = np.arange(shift, shift + seq_len)
    Q_rot_shifted = apply_rope(Q_shifted, rope.cos_cache, rope.sin_cache, shifted_positions)
    K_rot_shifted = apply_rope(K_shifted, rope.cos_cache, rope.sin_cache, shifted_positions)

    scores_shifted = (Q_rot_shifted @ K_rot_shifted.transpose(0, 1, 3, 2)) / scale + causal_mask
    attn_shifted = np.exp(scores_shifted - np.max(scores_shifted, axis=-1, keepdims=True))
    attn_shifted /= attn_shifted.sum(axis=-1, keepdims=True)

    # Since tokens are identical and relative positions are preserved,
    # the attention patterns should be IDENTICAL
    shift_diff = np.linalg.norm(attn_rope[0, 0] - attn_shifted[0, 0])
    print(f"\n  Position shift test (shift by {shift}):")
    print(f"    ||Attn(pos 0..{seq_len-1}) - Attn(pos {shift}..{shift+seq_len-1})|| = {shift_diff:.2e}")
    print(f"    Patterns are identical because relative positions are preserved")
    print(f"    (guaranteed by the relative position property)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    cmap = LinearSegmentedColormap.from_list("attn", ["white", "#3498db", "#1a1a2e"])

    # Top-left: attention without RoPE
    im0 = axes[0, 0].imshow(attn_no_rope[0, 0], cmap=cmap, aspect="equal",
                              interpolation="nearest", vmin=0)
    axes[0, 0].set_xlabel("Key Position")
    axes[0, 0].set_ylabel("Query Position")
    axes[0, 0].set_title("WITHOUT RoPE (identical tokens)\nUniform attention within causal mask",
                          fontsize=10, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Top-middle: attention with RoPE
    im1 = axes[0, 1].imshow(attn_rope[0, 0], cmap=cmap, aspect="equal",
                              interpolation="nearest", vmin=0)
    axes[0, 1].set_xlabel("Key Position")
    axes[0, 1].set_ylabel("Query Position")
    axes[0, 1].set_title("WITH RoPE (identical tokens)\nPosition-dependent: weights vary by relative position",
                          fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Top-right: shifted attention
    im2 = axes[0, 2].imshow(attn_shifted[0, 0], cmap=cmap, aspect="equal",
                              interpolation="nearest", vmin=0)
    axes[0, 2].set_xlabel("Key Position")
    axes[0, 2].set_ylabel("Query Position")
    axes[0, 2].set_title(f"WITH RoPE (shifted by {shift})\n"
                          f"||diff from unshifted|| = {shift_diff:.2e} (identical)",
                          fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Bottom-left: attention scores (pre-softmax) comparison
    # Show the raw scores for the last row (query at last position)
    last_row_no_rope = scores_no_rope[0, 0, -1, :].copy()
    last_row_no_rope[last_row_no_rope == -np.inf] = np.nan
    last_row_rope = scores_rope[0, 0, -1, :].copy()
    last_row_rope[last_row_rope == -np.inf] = np.nan

    axes[1, 0].plot(range(seq_len), last_row_no_rope, "o-", color=COLORS["red"],
                    linewidth=2, markersize=6, label="Without RoPE")
    axes[1, 0].plot(range(seq_len), last_row_rope, "s-", color=COLORS["blue"],
                    linewidth=2, markersize=6, label="With RoPE")
    axes[1, 0].set_xlabel("Key Position")
    axes[1, 0].set_ylabel("Attention Score (pre-softmax)")
    axes[1, 0].set_title(f"Scores from Query at Position {seq_len-1}\n"
                          f"RoPE creates position-dependent variation",
                          fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-middle: attention row profiles for different query positions
    for qpos in [3, 6, 9, 11]:
        row = attn_rope[0, 0, qpos, :]
        axes[1, 1].plot(range(seq_len), row, "o-", markersize=4, linewidth=1.5,
                        label=f"query pos {qpos}")
    axes[1, 1].set_xlabel("Key Position")
    axes[1, 1].set_ylabel("Attention Weight")
    axes[1, 1].set_title("Attention Profiles at Different Query Positions\n"
                          "Each row has unique pattern due to RoPE",
                          fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Bottom-right: difference heatmap (with vs without RoPE)
    diff = np.abs(attn_rope[0, 0] - attn_no_rope[0, 0])
    im5 = axes[1, 2].imshow(diff, cmap="Reds", aspect="equal", interpolation="nearest")
    axes[1, 2].set_xlabel("Key Position")
    axes[1, 2].set_ylabel("Query Position")
    axes[1, 2].set_title(f"|Attn(with RoPE) - Attn(without RoPE)|\n"
                          f"Max diff: {diff.max():.4f}",
                          fontsize=10, fontweight="bold")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle("Impact of RoPE on Attention: Position-Dependent Patterns from Identical Inputs",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_attention_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/05_attention_impact.png")


# ---------------------------------------------------------------------------
# Example 6: Context Extension via Theta Scaling
# ---------------------------------------------------------------------------
def example_6_context_extension():
    """Compare theta_base=10000, 500000, and NTK-aware scaling."""
    print("\n" + "=" * 60)
    print("Example 6: Context Extension via Theta Scaling")
    print("=" * 60)

    d = 128
    max_seq_len = 16384

    theta_standard = 10000.0     # Llama 1/2
    theta_llama3 = 500000.0      # Llama 3

    # NTK-aware scaling: theta' = theta * alpha^(d/(d-2))
    alpha_ntk = 4.0  # 4x context extension
    theta_ntk = theta_standard * alpha_ntk ** (d / (d - 2))

    configs = [
        ("Standard (theta=10K)", theta_standard, COLORS["blue"]),
        ("Llama 3 (theta=500K)", theta_llama3, COLORS["green"]),
        (f"NTK-aware (alpha={alpha_ntk}x)", theta_ntk, COLORS["orange"]),
    ]

    print(f"\n  Config: d={d}, max_seq_len={max_seq_len}")
    print(f"\n  Theta base values:")
    for label, theta, _ in configs:
        print(f"    {label}: theta = {theta:.1f}")

    # Frequency spectrum comparison
    i_vals = np.arange(d // 2)

    print(f"\n  Wavelength at selected dimension pairs:")
    print(f"  {'Pair':>6} {'Standard':>16} {'Llama 3':>16} {'NTK':>16}")
    print(f"  {'-'*58}")
    for idx in [0, 10, 20, 32, 50, 63]:
        wls = []
        for _, theta, _ in configs:
            freq = np.exp(-2.0 * idx / d * np.log(theta))
            wls.append(2 * np.pi / freq)
        print(f"  {idx:>6} {wls[0]:>16.1f} {wls[1]:>16.1f} {wls[2]:>16.1f}")

    # Attention score decay vs distance for each theta
    np.random.seed(SEED)
    q = np.random.randn(d)
    k = np.random.randn(d)

    distances = np.arange(0, 8192, 10)
    decay_curves = {}

    for label, theta, color in configs:
        freqs = np.exp(-2.0 * i_vals / d * np.log(theta))
        dots = np.zeros(len(distances))
        for di, delta in enumerate(distances):
            dot_sum = 0.0
            for j in range(d // 2):
                q2i, q2ip1 = q[2*j], q[2*j+1]
                k2i, k2ip1 = k[2*j], k[2*j+1]
                cos_t = (q2i * k2i + q2ip1 * k2ip1) * np.cos(delta * freqs[j])
                sin_t = (q2i * k2ip1 - q2ip1 * k2i) * np.sin(delta * freqs[j])
                dot_sum += cos_t + sin_t
            dots[di] = dot_sum
        decay_curves[label] = dots

    print(f"\n  Key insight: larger theta -> slower rotation -> less destructive")
    print(f"  interference at large distances -> extended effective context.")
    print(f"  NTK-aware preserves high-frequency resolution (dim pair 0) while")
    print(f"  stretching low-frequency components.")

    # NTK vs uniform position interpolation comparison
    print(f"\n  NTK-aware vs position interpolation:")
    freq_std = np.exp(-2.0 * i_vals / d * np.log(theta_standard))
    freq_ntk = np.exp(-2.0 * i_vals / d * np.log(theta_ntk))
    freq_pi = freq_std / alpha_ntk  # position interpolation divides all freqs

    print(f"    Pair 0 (highest freq):")
    print(f"      Standard:  {freq_std[0]:.6f}")
    print(f"      NTK-aware: {freq_ntk[0]:.6f}  (ratio: {freq_std[0]/freq_ntk[0]:.4f})")
    print(f"      Pos interp: {freq_pi[0]:.6f}  (ratio: {freq_std[0]/freq_pi[0]:.4f})")
    print(f"    Pair {d//2-1} (lowest freq):")
    print(f"      Standard:  {freq_std[-1]:.6e}")
    print(f"      NTK-aware: {freq_ntk[-1]:.6e}  (ratio: {freq_std[-1]/freq_ntk[-1]:.4f})")
    print(f"      Pos interp: {freq_pi[-1]:.6e}  (ratio: {freq_std[-1]/freq_pi[-1]:.4f})")
    print(f"    NTK barely changes high freqs but significantly stretches low freqs.")
    print(f"    Position interpolation uniformly divides all by alpha={alpha_ntk}.")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Top-left: frequency spectrum for each theta
    for label, theta, color in configs:
        freqs = np.exp(-2.0 * i_vals / d * np.log(theta))
        wavelengths = 2 * np.pi / freqs
        axes[0, 0].semilogy(i_vals, wavelengths, "o-", color=color, markersize=2,
                             linewidth=1.5, label=label)
    axes[0, 0].axhline(4096, color="gray", linestyle="--", alpha=0.5, label="4K context")
    axes[0, 0].axhline(8192, color="gray", linestyle=":", alpha=0.5, label="8K context")
    axes[0, 0].set_xlabel("Dimension Pair Index (i)")
    axes[0, 0].set_ylabel("Wavelength (positions)")
    axes[0, 0].set_title("Wavelength Spectrum per Theta Base\nLarger theta -> longer wavelengths",
                          fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-middle: frequency ratio (NTK vs standard)
    freq_ratio_ntk = freq_std / freq_ntk
    freq_ratio_pi = freq_std / freq_pi
    axes[0, 1].plot(i_vals, freq_ratio_ntk, "o-", color=COLORS["orange"], markersize=2,
                     linewidth=1.5, label=f"NTK-aware (alpha={alpha_ntk})")
    axes[0, 1].plot(i_vals, freq_ratio_pi, "s-", color=COLORS["red"], markersize=2,
                     linewidth=1.5, label=f"Pos. Interpolation (alpha={alpha_ntk})")
    axes[0, 1].axhline(alpha_ntk, color="gray", linestyle="--", alpha=0.5,
                        label=f"alpha={alpha_ntk}")
    axes[0, 1].axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    axes[0, 1].set_xlabel("Dimension Pair Index (i)")
    axes[0, 1].set_ylabel("Frequency Ratio (standard / scaled)")
    axes[0, 1].set_title("NTK Preserves High-Freq, Stretches Low-Freq\n"
                          "Pos. Interp. uniformly divides all by alpha",
                          fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Top-right: attention score decay vs distance
    for label, theta, color in configs:
        axes[0, 2].plot(distances, decay_curves[label], color=color, linewidth=1.5,
                         alpha=0.8, label=label)
    axes[0, 2].axhline(0, color="gray", linestyle=":", alpha=0.3)
    axes[0, 2].set_xlabel("Relative Distance (tokens)")
    axes[0, 2].set_ylabel("Dot Product <RoPE(q,m), RoPE(k,n)>")
    axes[0, 2].set_title("Attention Score Decay vs Distance\nLarger theta -> slower decay -> longer context",
                          fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Bottom-left: zoomed decay (first 500 positions)
    mask_short = distances < 500
    for label, theta, color in configs:
        axes[1, 0].plot(distances[mask_short], decay_curves[label][mask_short],
                         color=color, linewidth=2, label=label)
    axes[1, 0].axhline(0, color="gray", linestyle=":", alpha=0.3)
    axes[1, 0].set_xlabel("Relative Distance (tokens)")
    axes[1, 0].set_ylabel("Dot Product")
    axes[1, 0].set_title("Zoomed: Short-Range Behavior\nAll thetas similar at small distances",
                          fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-middle: effective context window analysis
    # Count how many dimension pairs have wavelength > L for each theta
    context_lengths = np.logspace(2, 5, 50)
    for label, theta, color in configs:
        freqs_cfg = np.exp(-2.0 * i_vals / d * np.log(theta))
        wavelengths_cfg = 2 * np.pi / freqs_cfg
        usable_pairs = []
        for L in context_lengths:
            n_usable = np.sum(wavelengths_cfg >= L)
            usable_pairs.append(n_usable)
        axes[1, 1].plot(context_lengths, usable_pairs, color=color, linewidth=2, label=label)

    axes[1, 1].set_xlabel("Context Length (tokens)")
    axes[1, 1].set_ylabel(f"Dim Pairs with Wavelength >= Context (out of {d//2})")
    axes[1, 1].set_title("Usable Dimension Pairs per Context Length\n"
                          "Pairs with wavelength < context contribute noise",
                          fontsize=10, fontweight="bold")
    axes[1, 1].set_xscale("log")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Bottom-right: practical recommendations
    axes[1, 2].axis("off")
    recs = (
        "CONTEXT EXTENSION METHODS\n"
        "=========================\n\n"
        f"Standard (theta={theta_standard:.0f}):\n"
        f"  Effective context: ~4K-8K tokens\n"
        f"  Used by: Llama 1/2, RoFormer\n\n"
        f"Higher base (theta={theta_llama3:.0f}):\n"
        f"  Effective context: ~128K tokens\n"
        f"  Used by: Llama 3, Qwen 2\n"
        f"  Simple but requires retraining\n\n"
        f"NTK-aware (alpha={alpha_ntk}, theta'={theta_ntk:.0f}):\n"
        f"  theta' = theta * alpha^(d/(d-2))\n"
        f"  Preserves high-freq (local) resolution\n"
        f"  Stretches low-freq (long-range) only\n"
        f"  Can be applied WITHOUT retraining\n\n"
        f"Position interpolation (alpha={alpha_ntk}):\n"
        f"  Divides ALL positions by alpha\n"
        f"  Simpler but degrades local resolution\n"
        f"  Requires fine-tuning for good results"
    )
    axes[1, 2].text(0.05, 0.95, recs, fontsize=10, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Context Extension via Theta Scaling: Standard vs Llama 3 vs NTK-Aware",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_context_extension.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/06_context_extension.png")


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
        ax.text(0.5, 0.78, "Rotary Position Embeddings (RoPE)", fontsize=28, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.68, "Comprehensive Demo and Analysis", fontsize=16,
                ha="center", va="center", transform=ax.transAxes, color="gray")
        info_text = (
            "RoPE encodes relative position directly into attention\n"
            "dot products through rotation. Each dimension pair defines\n"
            "a 2D subspace with position-dependent rotation.\n\n"
            "Key property: <RoPE(q,m), RoPE(k,n)> depends only on (m-n),\n"
            "making RoPE a true relative position encoding.\n\n"
            "Used by: Llama 1/2/3, Mistral, Qwen, Gemma, DeepSeek,\n"
            "and virtually all modern open-weight LLMs.\n\n"
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
            "1. Rotation Visualization: Each dimension pair (2i, 2i+1) traces a",
            "   circle in its 2D subspace. Pair 0 rotates fast (theta_0 = 1.0),",
            "   higher pairs rotate progressively slower. Norm is preserved",
            "   because rotations are orthogonal transformations.",
            "",
            "2. Relative Position Property (CENTERPIECE): The dot product",
            "   <RoPE(q,m), RoPE(k,n)> = q^T R(n-m) k depends ONLY on (m-n).",
            "   Proof: R(m)^T R(n) = R(-m)R(n) = R(n-m) by angle addition.",
            "   Verified empirically: max variation < 1e-12 across all positions.",
            "",
            "3. Norm & Orthogonality: ||RoPE(x,m)|| = ||x|| guaranteed by R^T R = I.",
            "   det(R) = 1 (proper rotation). R(m)R(n) = R(m+n) (composition).",
            "   R(m)R(-m) = I (inverse). All analytically exact from cos^2+sin^2=1.",
            "",
            "4. RoPE vs Sinusoidal PE: Both use theta_i = 10000^(-2i/d).",
            "   Sinusoidal is ADDITIVE (position added to content), creating cross-terms",
            "   that depend on absolute position. RoPE is MULTIPLICATIVE (q' = R(m)q),",
            "   giving a pure relative position encoding. This is the key advantage.",
            "",
            "5. Attention Impact: With identical token embeddings, RoPE creates",
            "   position-dependent attention patterns (weights vary by relative position).",
            "   Without RoPE, attention is uniform. Patterns are shift-invariant",
            "   (shifting all positions preserves relative distances).",
            "",
            "6. Context Extension: Larger theta_base stretches wavelengths,",
            "   extending effective context. Llama 3 uses theta=500K vs standard 10K.",
            "   NTK-aware scaling preserves high-freq (local) while stretching",
            "   low-freq (long-range) -- can extend context without retraining.",
        ]
        summary_text = "\n".join(summary_items)
        ax.text(0.06, 0.86, summary_text, fontsize=10, ha="left", va="top",
                transform=ax.transAxes, family="monospace", linespacing=1.3)
        pdf.savefig(fig)
        plt.close(fig)

        titles = {
            "01_rotation_visualization.png": "Example 1: Rotation Visualization -- Circles in 2D Subspaces",
            "02_relative_position.png": "Example 2: Relative Position Property -- The Key Theorem",
            "03_norm_orthogonality.png": "Example 3: Norm Preservation & Orthogonality",
            "04_rope_vs_sinusoidal.png": "Example 4: RoPE vs Sinusoidal PE -- Additive vs Multiplicative",
            "05_attention_impact.png": "Example 5: Impact on Attention Patterns",
            "06_context_extension.png": "Example 6: Context Extension via Theta Scaling",
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
    print("Rotary Position Embeddings (RoPE) Demo")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print()

    example_1_rotation_visualization()
    example_2_relative_position()
    example_3_norm_orthogonality()
    example_4_rope_vs_sinusoidal()
    example_5_attention_impact()
    example_6_context_extension()
    generate_pdf_report()

    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print(f"Visualizations: {VIZ_DIR}/")
    print(f"Report: {Path(__file__).parent / 'report.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
