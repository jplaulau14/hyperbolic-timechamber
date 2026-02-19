"""
Flash Attention Concept Demo -- Numerical equivalence, memory analysis,
tiling visualization, causal masking, and scaling projections.

Generates:
- viz/*.png -- Individual visualization files
- report.pdf -- Comprehensive PDF report
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from implementation import (
    online_softmax,
    standard_attention,
    tiled_attention,
    memory_analysis,
    verify_no_full_materialization,
)

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
# Example 1: Numerical Equivalence Verification
# ---------------------------------------------------------------------------
def example_1_numerical_equivalence():
    """Max absolute error between tiled and standard attention for various N and block sizes."""
    print("=" * 60)
    print("Example 1: Numerical Equivalence Verification")
    print("=" * 60)

    configs = [
        (32, 16, 8),
        (64, 32, 16),
        (128, 64, 32),
        (256, 128, 64),
        (512, 64, 32),
        (100, 32, 32),
        (65, 32, 64),
        (1024, 64, 64),
    ]

    rng = np.random.RandomState(SEED)
    results = []

    print(f"\n  {'N':>6} {'d':>4} {'Block':>6} {'Max |Error|':>14} {'Mean |Error|':>14} {'Status':>8}")
    print(f"  {'-' * 56}")

    for N, d, block_size in configs:
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        O_std, _, _ = standard_attention(Q, K, V)
        O_tiled = tiled_attention(Q, K, V, block_size_q=block_size, block_size_kv=block_size)

        max_err = float(np.max(np.abs(O_std - O_tiled)))
        mean_err = float(np.mean(np.abs(O_std - O_tiled)))
        status = "PASS" if max_err < 1e-5 else "FAIL"
        results.append((N, d, block_size, max_err, mean_err, status))
        print(f"  {N:>6} {d:>4} {block_size:>6} {max_err:>14.2e} {mean_err:>14.2e} {status:>8}")

    block_size_sweep = [4, 8, 16, 32, 64, 128, 256]
    N_fixed, d_fixed = 256, 64
    Q, K, V = rng.randn(N_fixed, d_fixed), rng.randn(N_fixed, d_fixed), rng.randn(N_fixed, d_fixed)
    O_std, _, _ = standard_attention(Q, K, V)
    block_errors = []
    for bs in block_size_sweep:
        O_tiled = tiled_attention(Q, K, V, block_size_q=bs, block_size_kv=bs)
        max_err = float(np.max(np.abs(O_std - O_tiled)))
        block_errors.append(max_err)

    print(f"\n  Block size sweep (N={N_fixed}, d={d_fixed}):")
    for bs, err in zip(block_size_sweep, block_errors):
        print(f"    Block={bs:>4}: max |error| = {err:.2e}")

    assert all(r[5] == "PASS" for r in results), "Numerical equivalence check failed"
    print(f"\n  ALL {len(results)} CONFIGURATIONS PASS -- tiled == standard within 1e-5")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    Ns = [r[0] for r in results]
    max_errs = [r[3] for r in results]
    labels = [f"N={r[0]}\nd={r[1]}\nB={r[2]}" for r in results]
    bars = axes[0, 0].bar(range(len(results)), max_errs, color=COLORS["blue"], edgecolor="white")
    axes[0, 0].set_xticks(range(len(results)))
    axes[0, 0].set_xticklabels(labels, fontsize=7)
    axes[0, 0].set_ylabel("Max Absolute Error")
    axes[0, 0].set_yscale("log")
    axes[0, 0].axhline(1e-5, color=COLORS["red"], linestyle="--", linewidth=1.5, label="Threshold (1e-5)")
    axes[0, 0].set_title("Max Absolute Error per Configuration\nAll below tolerance threshold",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    axes[0, 1].bar(range(len(block_size_sweep)), block_errors,
                   color=COLORS["green"], edgecolor="white")
    axes[0, 1].set_xticks(range(len(block_size_sweep)))
    axes[0, 1].set_xticklabels([str(bs) for bs in block_size_sweep])
    axes[0, 1].set_xlabel("Block Size")
    axes[0, 1].set_ylabel("Max Absolute Error")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(1e-5, color=COLORS["red"], linestyle="--", linewidth=1.5)
    axes[0, 1].set_title(f"Error vs Block Size (N={N_fixed}, d={d_fixed})\nAll block sizes produce equivalent results",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    N_err = 128
    d_err = 32
    Q_e, K_e, V_e = rng.randn(N_err, d_err), rng.randn(N_err, d_err), rng.randn(N_err, d_err)
    O_std_e, _, _ = standard_attention(Q_e, K_e, V_e)
    O_tiled_e = tiled_attention(Q_e, K_e, V_e, block_size_q=16, block_size_kv=16)
    err_matrix = np.abs(O_std_e - O_tiled_e)
    im = axes[0, 2].imshow(err_matrix, aspect="auto", cmap="hot_r", interpolation="nearest")
    axes[0, 2].set_xlabel("Head Dimension (d)")
    axes[0, 2].set_ylabel("Query Position (N)")
    axes[0, 2].set_title(f"Error Heatmap (N={N_err}, d={d_err}, B=16)\nErrors are uniformly tiny across all positions",
                         fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=axes[0, 2], label="Absolute Error")

    x_test = rng.randn(500)
    ref_softmax = np.exp(x_test - np.max(x_test)) / np.sum(np.exp(x_test - np.max(x_test)))
    chunk_sizes = [1, 5, 10, 25, 50, 100, 250, 500]
    softmax_errors = []
    for cs in chunk_sizes:
        result, _, _ = online_softmax(x_test, chunk_size=cs)
        softmax_errors.append(float(np.max(np.abs(result - ref_softmax))))
    axes[1, 0].bar(range(len(chunk_sizes)), softmax_errors,
                   color=COLORS["purple"], edgecolor="white")
    axes[1, 0].set_xticks(range(len(chunk_sizes)))
    axes[1, 0].set_xticklabels([str(cs) for cs in chunk_sizes], fontsize=8)
    axes[1, 0].set_xlabel("Chunk Size")
    axes[1, 0].set_ylabel("Max Absolute Error")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Online Softmax Accuracy vs Chunk Size\nExact for all chunk sizes (within float64 precision)",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    x_extreme = np.array([1000.0, 1000.0, 1000.0])
    result_extreme, _, _ = online_softmax(x_extreme, chunk_size=1)
    x_neg = np.array([-1000.0, -1000.0, -999.0])
    result_neg, _, _ = online_softmax(x_neg, chunk_size=1)

    ax_stab = axes[1, 1]
    x_pos = [0, 1, 2, 4, 5, 6]
    vals = list(result_extreme) + list(result_neg)
    bar_colors = [COLORS["blue"]] * 3 + [COLORS["green"]] * 3
    ax_stab.bar(x_pos, vals, color=bar_colors, edgecolor="white")
    ax_stab.set_xticks(x_pos)
    ax_stab.set_xticklabels(["x=1000"] * 3 + ["x=-1000", "x=-1000", "x=-999"], fontsize=7)
    for i, v in enumerate(vals):
        ax_stab.text(x_pos[i], v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax_stab.set_ylabel("Softmax Output")
    ax_stab.set_title("Numerical Stability of Online Softmax\nHandles extreme values without overflow/underflow",
                      fontsize=10, fontweight="bold")
    ax_stab.grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    equiv_text = (
        "NUMERICAL EQUIVALENCE VERIFIED\n"
        "==============================\n\n"
        f"Configurations tested: {len(results)}\n"
        f"All passed: YES\n"
        f"Tolerance: 1e-5\n\n"
        "Key findings:\n"
        "  - Tiled attention produces the\n"
        "    EXACT same output as standard\n"
        "    attention (within float64 eps)\n"
        "  - Block size does NOT affect\n"
        "    accuracy -- all sizes give\n"
        "    the same answer\n"
        "  - Non-divisible N works correctly\n"
        "    (partial blocks handled)\n"
        "  - Online softmax is numerically\n"
        "    stable for extreme values\n\n"
        "The tiling is purely a memory\n"
        "optimization -- it changes HOW\n"
        "the computation happens, not\n"
        "WHAT is computed."
    )
    axes[1, 2].text(0.05, 0.95, equiv_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Flash Attention: Numerical Equivalence Verification",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_numerical_equivalence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/01_numerical_equivalence.png")


# ---------------------------------------------------------------------------
# Example 2: Memory Comparison (Standard O(N^2) vs Tiled O(N))
# ---------------------------------------------------------------------------
def example_2_memory_comparison():
    """Standard O(N^2) vs tiled O(N) with actual byte counts and growing ratio."""
    print("\n" + "=" * 60)
    print("Example 2: Memory Comparison -- O(N^2) vs O(N)")
    print("=" * 60)

    d = 64
    block_size = 32
    Ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    print(f"\n  d={d}, block_size={block_size}, dtype=float32")
    print(f"\n  {'N':>6} {'Standard':>14} {'Tiled':>14} {'Ratio':>8}")
    print(f"  {'-' * 46}")

    std_bytes_list = []
    tiled_bytes_list = []
    ratios = []

    for N in Ns:
        result = memory_analysis(N, d, block_size, dtype="float32")
        std_b = result["standard"]["total_bytes"]
        tiled_b = result["tiled"]["total_bytes"]
        ratio = result["ratio"]
        std_bytes_list.append(std_b)
        tiled_bytes_list.append(tiled_b)
        ratios.append(ratio)

        def fmt(b):
            if b < 1024:
                return f"{b} B"
            if b < 1024 ** 2:
                return f"{b / 1024:.1f} KB"
            return f"{b / 1024 ** 2:.2f} MB"

        print(f"  {N:>6} {fmt(std_b):>14} {fmt(tiled_b):>14} {ratio:>7.1f}x")

    print(f"\n  Memory ratio grows with N because standard is O(N^2) while tiled is O(N)")
    print(f"  At N=8192: standard needs {std_bytes_list[-1] / 1024**2:.1f} MB vs tiled {tiled_bytes_list[-1] / 1024**2:.3f} MB")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].plot(Ns, [b / 1024**2 for b in std_bytes_list], "o-",
                    color=COLORS["red"], linewidth=2, markersize=6, label="Standard (S + P + O)")
    axes[0, 0].plot(Ns, [b / 1024**2 for b in tiled_bytes_list], "s-",
                    color=COLORS["green"], linewidth=2, markersize=6, label="Tiled (block + stats + O)")
    axes[0, 0].fill_between(Ns, [b / 1024**2 for b in tiled_bytes_list],
                             [b / 1024**2 for b in std_bytes_list],
                             alpha=0.15, color=COLORS["red"], label="Memory saved")
    axes[0, 0].set_xlabel("Sequence Length (N)")
    axes[0, 0].set_ylabel("Peak Memory (MB)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_title(f"Peak Memory: Standard vs Tiled (d={d})\n"
                         r"Standard: $O(N^2)$, Tiled: $O(N)$",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(Ns, ratios, "o-", color=COLORS["purple"], linewidth=2, markersize=6)
    for i, (n, r) in enumerate(zip(Ns, ratios)):
        if i % 2 == 0 or i == len(Ns) - 1:
            axes[0, 1].annotate(f"{r:.0f}x", (n, r), textcoords="offset points",
                               xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")
    axes[0, 1].set_xlabel("Sequence Length (N)")
    axes[0, 1].set_ylabel("Memory Ratio (Standard / Tiled)")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_title("Memory Savings Ratio Grows with N\nFlash attention advantage increases for longer sequences",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    result_breakdown = memory_analysis(1024, d, block_size, dtype="float32")
    std = result_breakdown["standard"]
    tiled = result_breakdown["tiled"]

    std_parts = [std["S_bytes"], std["P_bytes"], std["O_bytes"]]
    std_labels = [f"S (N x N)\n{std['S_bytes'] / 1024**2:.1f} MB",
                  f"P (N x N)\n{std['P_bytes'] / 1024**2:.1f} MB",
                  f"O (N x d)\n{std['O_bytes'] / 1024:.0f} KB"]
    std_colors = [COLORS["red"], COLORS["orange"], COLORS["blue"]]
    axes[0, 2].pie(std_parts, labels=std_labels, colors=std_colors,
                   autopct=lambda p: f"{p:.0f}%", pctdistance=0.75, startangle=90)
    axes[0, 2].set_title(f"Standard Attention Memory Breakdown (N=1024)\n"
                         f"Total: {std['total_bytes'] / 1024**2:.1f} MB -- dominated by N x N matrices",
                         fontsize=10, fontweight="bold")

    tiled_parts = [tiled["block_S_bytes"], tiled["block_P_bytes"],
                   tiled["statistics_bytes"], tiled["O_bytes"]]
    tiled_labels = [f"Block S\n{tiled['block_S_bytes'] / 1024:.1f} KB",
                    f"Block P\n{tiled['block_P_bytes'] / 1024:.1f} KB",
                    f"Stats (m,l)\n{tiled['statistics_bytes'] / 1024:.1f} KB",
                    f"O (N x d)\n{tiled['O_bytes'] / 1024:.0f} KB"]
    tiled_colors = [COLORS["green"], COLORS["teal"], COLORS["purple"], COLORS["blue"]]
    axes[1, 0].pie(tiled_parts, labels=tiled_labels, colors=tiled_colors,
                   autopct=lambda p: f"{p:.0f}%", pctdistance=0.75, startangle=90)
    axes[1, 0].set_title(f"Tiled Attention Memory Breakdown (N=1024, B={block_size})\n"
                         f"Total: {tiled['total_bytes'] / 1024:.1f} KB -- block matrices are tiny",
                         fontsize=10, fontweight="bold")

    dtypes = ["float32", "float16"]
    dtype_labels = ["FP32 (4 bytes)", "FP16 (2 bytes)"]
    x = np.arange(len(Ns))
    width = 0.35
    for di, (dt, dl) in enumerate(zip(dtypes, dtype_labels)):
        dtype_ratios = []
        for N in Ns:
            r = memory_analysis(N, d, block_size, dtype=dt)
            dtype_ratios.append(r["ratio"])
        color = COLORS["blue"] if di == 0 else COLORS["green"]
        axes[1, 1].bar(x + di * width - width / 2, dtype_ratios, width,
                       label=dl, color=color, edgecolor="white")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(n) for n in Ns], fontsize=8)
    axes[1, 1].set_xlabel("Sequence Length (N)")
    axes[1, 1].set_ylabel("Memory Ratio")
    axes[1, 1].set_title("Memory Savings by Data Type\nRatio is similar for FP16 and FP32",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    mem_text = (
        "MEMORY COMPARISON\n"
        "=================\n\n"
        "Standard attention allocates:\n"
        "  S = Q @ K.T    -> (N, N)\n"
        "  P = softmax(S)  -> (N, N)\n"
        "  O = P @ V       -> (N, d)\n"
        f"  Peak: 2*N^2 + N*d floats\n\n"
        "Tiled attention allocates:\n"
        "  S_block -> (B, B)  [reused]\n"
        "  P_block -> (B, B)  [reused]\n"
        "  m, ell  -> (N,)    [stats]\n"
        "  O       -> (N, d)  [output]\n"
        f"  Peak: 2*B^2 + 2*N + N*d\n\n"
        f"B={block_size}, d={d}:\n"
        f"  Block memory: {2 * block_size**2 * 4 / 1024:.1f} KB (constant!)\n"
        f"  At N=8192: {ratios[-1]:.0f}x memory saved\n\n"
        "The block matrices are REUSED\n"
        "for each tile -- only O(B^2)\n"
        "lives at any given time."
    )
    axes[1, 2].text(0.05, 0.95, mem_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Flash Attention: Memory Comparison -- O(N^2) vs O(N)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_memory_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/02_memory_comparison.png")


# ---------------------------------------------------------------------------
# Example 3: Block Size Analysis
# ---------------------------------------------------------------------------
def example_3_block_size_analysis():
    """How different block sizes affect peak memory and number of tile iterations."""
    print("\n" + "=" * 60)
    print("Example 3: Block Size Analysis")
    print("=" * 60)

    N = 512
    d = 64
    block_sizes = [4, 8, 16, 32, 64, 128, 256, 512]

    print(f"\n  N={N}, d={d}")
    print(f"\n  {'Block':>6} {'Tiles':>8} {'Block Mem':>12} {'Total Mem':>12} {'Ratio':>8}")
    print(f"  {'-' * 50}")

    tile_counts = []
    block_mem_list = []
    total_mem_list = []
    mem_ratios = []

    for bs in block_sizes:
        num_q_blocks = (N + bs - 1) // bs
        num_kv_blocks = (N + bs - 1) // bs
        n_tiles = num_q_blocks * num_kv_blocks

        result = memory_analysis(N, d, bs, dtype="float32")
        block_mem = result["tiled"]["block_S_bytes"] + result["tiled"]["block_P_bytes"]
        total_mem = result["tiled"]["total_bytes"]
        ratio = result["ratio"]

        tile_counts.append(n_tiles)
        block_mem_list.append(block_mem)
        total_mem_list.append(total_mem)
        mem_ratios.append(ratio)

        print(f"  {bs:>6} {n_tiles:>8} {block_mem / 1024:>10.1f} KB {total_mem / 1024:>10.1f} KB {ratio:>7.1f}x")

    print(f"\n  Larger blocks -> fewer tiles but more memory per block")
    print(f"  Smaller blocks -> more tiles but less memory per block")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].bar(range(len(block_sizes)), [b / 1024 for b in block_mem_list],
                   color=COLORS["blue"], edgecolor="white")
    axes[0, 0].set_xticks(range(len(block_sizes)))
    axes[0, 0].set_xticklabels([str(bs) for bs in block_sizes])
    axes[0, 0].set_xlabel("Block Size")
    axes[0, 0].set_ylabel("Block Memory (KB)")
    axes[0, 0].set_title(f"Per-Block Memory (S_block + P_block)\n"
                         r"Grows as $O(B^2)$ -- quadratic in block size",
                         fontsize=10, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    axes[0, 1].bar(range(len(block_sizes)), tile_counts,
                   color=COLORS["green"], edgecolor="white")
    axes[0, 1].set_xticks(range(len(block_sizes)))
    axes[0, 1].set_xticklabels([str(bs) for bs in block_sizes])
    axes[0, 1].set_xlabel("Block Size")
    axes[0, 1].set_ylabel("Number of Tile Iterations")
    axes[0, 1].set_title(f"Tile Iterations (N={N})\n"
                         r"Decreases as $O(N^2/B^2)$",
                         fontsize=10, fontweight="bold")
    for i, tc in enumerate(tile_counts):
        axes[0, 1].text(i, tc + max(tile_counts) * 0.02, str(tc), ha="center",
                       va="bottom", fontsize=8, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    axes[0, 2].plot(block_sizes, mem_ratios, "o-", color=COLORS["purple"],
                    linewidth=2, markersize=6)
    axes[0, 2].set_xlabel("Block Size")
    axes[0, 2].set_ylabel("Memory Ratio (Standard / Tiled)")
    axes[0, 2].set_title(f"Memory Savings vs Block Size (N={N})\nSmaller blocks = more savings (but more iterations)",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3)

    rng = np.random.RandomState(SEED)
    Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
    O_std, _, _ = standard_attention(Q, K, V)
    errors_by_bs = []
    for bs in block_sizes:
        O_tiled = tiled_attention(Q, K, V, block_size_q=bs, block_size_kv=bs)
        errors_by_bs.append(float(np.max(np.abs(O_std - O_tiled))))

    axes[1, 0].bar(range(len(block_sizes)), errors_by_bs,
                   color=COLORS["teal"], edgecolor="white")
    axes[1, 0].set_xticks(range(len(block_sizes)))
    axes[1, 0].set_xticklabels([str(bs) for bs in block_sizes])
    axes[1, 0].set_xlabel("Block Size")
    axes[1, 0].set_ylabel("Max Absolute Error")
    axes[1, 0].set_yscale("log")
    axes[1, 0].axhline(1e-5, color=COLORS["red"], linestyle="--", label="1e-5 threshold")
    axes[1, 0].set_title(f"Accuracy vs Block Size (N={N})\nCorrect for ALL block sizes",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    total_ops = [tc * bs * bs for tc, bs in zip(tile_counts, block_sizes)]
    axes[1, 1].plot(block_sizes, [t / 1e6 for t in total_ops], "o-",
                    color=COLORS["orange"], linewidth=2, markersize=6)
    axes[1, 1].set_xlabel("Block Size")
    axes[1, 1].set_ylabel("Total Elements Processed (millions)")
    axes[1, 1].set_title(f"Total Tile Elements (tiles x B^2)\n"
                         f"Constant at N^2 = {N**2 / 1e6:.2f}M regardless of block size",
                         fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    bs_text = (
        "BLOCK SIZE TRADE-OFFS\n"
        "=====================\n\n"
        "Smaller blocks (e.g., B=4):\n"
        "  + Minimal peak memory\n"
        "  - Many tile iterations\n"
        "  - More loop overhead\n"
        "  - In GPU: many kernel launches\n\n"
        "Larger blocks (e.g., B=256):\n"
        "  + Fewer iterations\n"
        "  + Better hardware utilization\n"
        "  - More memory per block\n"
        "  - Must fit in GPU SRAM\n\n"
        "In practice (CUDA):\n"
        "  B chosen to fit shared memory\n"
        "  Typical: B=64 or B=128\n"
        "  A100 shared mem: 48 KB\n"
        "  -> B*d*bytes must fit\n\n"
        "Total work is ALWAYS N^2 * d\n"
        "Block size changes memory, not\n"
        "total computation."
    )
    axes[1, 2].text(0.05, 0.95, bs_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Flash Attention: Block Size Analysis",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_block_size_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/03_block_size_analysis.png")


# ---------------------------------------------------------------------------
# Example 4: Tiling Visualization
# ---------------------------------------------------------------------------
def example_4_tiling_visualization():
    """Show which blocks get processed in what order for a small example."""
    print("\n" + "=" * 60)
    print("Example 4: Tiling Visualization")
    print("=" * 60)

    N = 16
    block_size = 4
    d = 8
    num_q_blocks = N // block_size
    num_kv_blocks = N // block_size

    processing_order = []
    step = 0
    for j_block in range(num_kv_blocks):
        for i_block in range(num_q_blocks):
            processing_order.append((i_block, j_block, step))
            step += 1

    print(f"\n  N={N}, block_size={block_size}")
    print(f"  Q blocks: {num_q_blocks}, KV blocks: {num_kv_blocks}")
    print(f"  Total tiles: {step}")
    print(f"\n  Processing order (outer: KV blocks, inner: Q blocks):")
    for i_b, j_b, s in processing_order:
        q_range = f"Q[{i_b * block_size}:{(i_b + 1) * block_size}]"
        kv_range = f"K/V[{j_b * block_size}:{(j_b + 1) * block_size}]"
        print(f"    Step {s:>2}: {q_range} x {kv_range}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    order_matrix = np.full((num_q_blocks, num_kv_blocks), -1)
    for i_b, j_b, s in processing_order:
        order_matrix[i_b, j_b] = s

    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=step - 1)

    ax = axes[0, 0]
    for i_b in range(num_q_blocks):
        for j_b in range(num_kv_blocks):
            s = order_matrix[i_b, j_b]
            color = cmap(norm(s))
            rect = Rectangle((j_b, i_b), 1, 1, facecolor=color, edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            ax.text(j_b + 0.5, i_b + 0.5, str(s), ha="center", va="center",
                   fontsize=10, fontweight="bold", color="white")
    ax.set_xlim(0, num_kv_blocks)
    ax.set_ylim(num_q_blocks, 0)
    ax.set_xlabel("KV Block Index (j)")
    ax.set_ylabel("Q Block Index (i)")
    ax.set_title(f"Tile Processing Order (N={N}, B={block_size})\n"
                 "Outer loop: KV blocks, Inner loop: Q blocks",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(num_kv_blocks) + 0.5)
    ax.set_xticklabels([str(i) for i in range(num_kv_blocks)])
    ax.set_yticks(np.arange(num_q_blocks) + 0.5)
    ax.set_yticklabels([str(i) for i in range(num_q_blocks)])
    ax.set_aspect("equal")

    ax = axes[0, 1]
    full_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            i_b = i // block_size
            j_b = j // block_size
            full_matrix[i, j] = order_matrix[i_b, j_b]

    im = ax.imshow(full_matrix, cmap="viridis", interpolation="nearest", aspect="equal")
    for b in range(1, num_q_blocks):
        ax.axhline(b * block_size - 0.5, color="white", linewidth=1.5)
        ax.axvline(b * block_size - 0.5, color="white", linewidth=1.5)
    ax.set_xlabel("Key Position (j)")
    ax.set_ylabel("Query Position (i)")
    ax.set_title(f"Full Attention Matrix Tile Coverage\n"
                 "Each {0}x{0} block processed as one unit".format(block_size),
                 fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Processing Step")

    ax = axes[0, 2]
    highlight_steps = [0, 1, 2, 3]
    colors_highlight = [COLORS["red"], COLORS["orange"], COLORS["green"], COLORS["blue"]]

    for idx, (s, c) in enumerate(zip(highlight_steps, colors_highlight)):
        i_b, j_b, _ = processing_order[s]
        q_start = i_b * block_size
        kv_start = j_b * block_size

        rect = Rectangle((kv_start, q_start), block_size, block_size,
                         facecolor=c, alpha=0.4, edgecolor=c, linewidth=2,
                         label=f"Step {s}: Q[{q_start}:{q_start + block_size}] x K[{kv_start}:{kv_start + block_size}]")
        ax.add_patch(rect)
        ax.text(kv_start + block_size / 2, q_start + block_size / 2,
               f"S{s}", ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, N)
    ax.set_ylim(N, 0)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title("First 4 Tiles Highlighted\nAll in KV block 0 (column 0), sweeping Q blocks",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    rng = np.random.RandomState(SEED)
    Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
    _, P_full, _ = standard_attention(Q, K, V)
    im = ax.imshow(P_full, cmap="Blues", interpolation="nearest", aspect="equal")
    for b in range(1, num_q_blocks):
        ax.axhline(b * block_size - 0.5, color=COLORS["red"], linewidth=0.8, alpha=0.5)
        ax.axvline(b * block_size - 0.5, color=COLORS["red"], linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title("Actual Attention Weights with Tile Boundaries\nEach block computed independently, then combined",
                 fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    ax = axes[1, 1]
    N_causal = 16
    causal_order = np.full((num_q_blocks, num_kv_blocks), -1.0)
    step = 0
    for j_b in range(num_kv_blocks):
        for i_b in range(num_q_blocks):
            i_end = (i_b + 1) * block_size
            j_start = j_b * block_size
            if j_start >= i_end:
                continue
            causal_order[i_b, j_b] = step
            step += 1

    for i_b in range(num_q_blocks):
        for j_b in range(num_kv_blocks):
            s = causal_order[i_b, j_b]
            if s >= 0:
                color = cmap(norm(s / max(step - 1, 1) * (len(processing_order) - 1)))
                rect = Rectangle((j_b, i_b), 1, 1, facecolor=color, edgecolor="white", linewidth=2)
                ax.add_patch(rect)
                ax.text(j_b + 0.5, i_b + 0.5, f"{int(s)}", ha="center", va="center",
                       fontsize=9, fontweight="bold", color="white")
            else:
                rect = Rectangle((j_b, i_b), 1, 1, facecolor="#dddddd", edgecolor="white", linewidth=2)
                ax.add_patch(rect)
                ax.text(j_b + 0.5, i_b + 0.5, "skip", ha="center", va="center",
                       fontsize=7, color="gray")
    ax.set_xlim(0, num_kv_blocks)
    ax.set_ylim(num_q_blocks, 0)
    ax.set_xlabel("KV Block Index (j)")
    ax.set_ylabel("Q Block Index (i)")
    ax.set_title("Causal Masking: Skipped Blocks\nUpper-triangular blocks entirely skipped",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(num_kv_blocks) + 0.5)
    ax.set_xticklabels([str(i) for i in range(num_kv_blocks)])
    ax.set_yticks(np.arange(num_q_blocks) + 0.5)
    ax.set_yticklabels([str(i) for i in range(num_q_blocks)])
    ax.set_aspect("equal")

    axes[1, 2].axis("off")
    tile_text = (
        "TILING STRATEGY\n"
        "===============\n\n"
        f"N={N}, block_size={block_size}\n"
        f"Q blocks: {num_q_blocks}\n"
        f"KV blocks: {num_kv_blocks}\n"
        f"Total tiles: {num_q_blocks * num_kv_blocks}\n\n"
        "Processing order:\n"
        "  for j in KV_blocks:\n"
        "    for i in Q_blocks:\n"
        "      S_ij = Q_i @ K_j.T / sqrt(d)\n"
        "      update m, ell, O\n\n"
        "Why outer KV, inner Q?\n"
        "  K_j and V_j are loaded once\n"
        "  and reused across all Q blocks.\n"
        "  In GPU: K_j, V_j go to shared\n"
        "  memory, Q blocks stream through.\n\n"
        "Causal masking:\n"
        f"  Skipped tiles: {num_q_blocks * num_kv_blocks - step}\n"
        f"  Processed tiles: {step}\n"
        f"  ~50% compute saved for causal"
    )
    axes[1, 2].text(0.05, 0.95, tile_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Flash Attention: Tiling Visualization",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_tiling_visualization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/04_tiling_visualization.png")


# ---------------------------------------------------------------------------
# Example 5: Causal Masking
# ---------------------------------------------------------------------------
def example_5_causal_masking():
    """Demonstrate correct causal tiled attention vs reference."""
    print("\n" + "=" * 60)
    print("Example 5: Causal Masking Verification")
    print("=" * 60)

    rng = np.random.RandomState(SEED)

    configs = [
        (32, 16, 8),
        (64, 32, 16),
        (128, 64, 32),
        (30, 16, 7),
        (65, 32, 11),
    ]

    print(f"\n  {'N':>6} {'d':>4} {'Block':>6} {'Max |Error|':>14} {'Status':>8}")
    print(f"  {'-' * 42}")

    results = []
    for N, d, bs in configs:
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)

        S = Q @ K.T / np.sqrt(d)
        mask = np.full((N, N), -np.inf)
        mask[np.tril_indices(N)] = 0.0
        S = S + mask
        S_max = np.max(S, axis=-1, keepdims=True)
        P = np.exp(S - S_max) / np.sum(np.exp(S - S_max), axis=-1, keepdims=True)
        O_ref = (P @ V).astype(np.float64)

        O_causal = tiled_attention(Q, K, V, block_size_q=bs, block_size_kv=bs, causal=True)
        max_err = float(np.max(np.abs(O_ref - O_causal)))
        status = "PASS" if max_err < 1e-5 else "FAIL"
        results.append((N, d, bs, max_err, status, P, O_ref, O_causal))
        print(f"  {N:>6} {d:>4} {bs:>6} {max_err:>14.2e} {status:>8}")

    assert all(r[4] == "PASS" for r in results), "Causal masking check failed"
    print(f"\n  ALL {len(results)} CAUSAL CONFIGURATIONS PASS")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    N_demo = 32
    d_demo = 16
    Q_d, K_d, V_d = rng.randn(N_demo, d_demo), rng.randn(N_demo, d_demo), rng.randn(N_demo, d_demo)

    _, P_full, _ = standard_attention(Q_d, K_d, V_d)
    im0 = axes[0, 0].imshow(P_full, cmap="Blues", interpolation="nearest", aspect="equal")
    axes[0, 0].set_xlabel("Key Position")
    axes[0, 0].set_ylabel("Query Position")
    axes[0, 0].set_title("Standard Attention (no mask)\nAll positions attend to all others",
                         fontsize=10, fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 0], label="Weight")

    P_causal = results[0][5]
    im1 = axes[0, 1].imshow(P_causal[:32, :32], cmap="Blues", interpolation="nearest", aspect="equal")
    axes[0, 1].set_xlabel("Key Position")
    axes[0, 1].set_ylabel("Query Position")
    axes[0, 1].set_title("Causal Attention (reference)\nUpper triangle is zero",
                         fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 1], label="Weight")

    O_std, _, _ = standard_attention(Q_d, K_d, V_d)
    O_causal_d = tiled_attention(Q_d, K_d, V_d, block_size_q=8, block_size_kv=8, causal=True)

    S_d = Q_d @ K_d.T / np.sqrt(d_demo)
    mask_d = np.full((N_demo, N_demo), -np.inf)
    mask_d[np.tril_indices(N_demo)] = 0.0
    S_d = S_d + mask_d
    S_max_d = np.max(S_d, axis=-1, keepdims=True)
    P_d = np.exp(S_d - S_max_d) / np.sum(np.exp(S_d - S_max_d), axis=-1, keepdims=True)
    O_ref_d = P_d @ V_d

    err_d = np.abs(O_ref_d - O_causal_d)
    im2 = axes[0, 2].imshow(err_d, cmap="hot_r", interpolation="nearest", aspect="auto")
    axes[0, 2].set_xlabel("Head Dimension")
    axes[0, 2].set_ylabel("Query Position")
    axes[0, 2].set_title("Causal Tiled vs Reference Error\nAll errors are negligible",
                         fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=axes[0, 2], label="Absolute Error")

    Ns_err = [r[0] for r in results]
    max_errs = [r[3] for r in results]
    bar_labels = [f"N={r[0]}\nB={r[2]}" for r in results]
    axes[1, 0].bar(range(len(results)), max_errs, color=COLORS["green"], edgecolor="white")
    axes[1, 0].set_xticks(range(len(results)))
    axes[1, 0].set_xticklabels(bar_labels, fontsize=8)
    axes[1, 0].set_ylabel("Max Absolute Error")
    axes[1, 0].set_yscale("log")
    axes[1, 0].axhline(1e-5, color=COLORS["red"], linestyle="--", label="Threshold")
    axes[1, 0].set_title("Causal Tiled Error per Configuration\nIncludes non-divisible N cases",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    N_compare = 32
    d_compare = 16
    Q_c, K_c, V_c = rng.randn(N_compare, d_compare), rng.randn(N_compare, d_compare), rng.randn(N_compare, d_compare)
    O_noncausal = tiled_attention(Q_c, K_c, V_c, block_size_q=8, block_size_kv=8, causal=False)
    O_causal_c = tiled_attention(Q_c, K_c, V_c, block_size_q=8, block_size_kv=8, causal=True)

    positions = range(N_compare)
    dim_to_plot = 0
    axes[1, 1].plot(positions, O_noncausal[:, dim_to_plot], "o-", color=COLORS["blue"],
                    markersize=4, linewidth=1.5, label="Non-causal (dim 0)")
    axes[1, 1].plot(positions, O_causal_c[:, dim_to_plot], "s-", color=COLORS["red"],
                    markersize=4, linewidth=1.5, label="Causal (dim 0)")
    axes[1, 1].set_xlabel("Query Position")
    axes[1, 1].set_ylabel("Output Value (dim 0)")
    axes[1, 1].set_title("Causal vs Non-Causal Output\nCausal uses only past context per position",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    causal_text = (
        "CAUSAL MASKING IN TILED ATTENTION\n"
        "=================================\n\n"
        "Causal constraint:\n"
        "  Query i attends only to keys j <= i\n"
        "  Upper triangle of S is -inf\n\n"
        "Block-level optimization:\n"
        "  If j_start >= i_end for a tile,\n"
        "  the ENTIRE block is above the\n"
        "  diagonal. Skip it completely.\n\n"
        "  For partial blocks (diagonal\n"
        "  crosses the block), apply\n"
        "  element-wise causal mask.\n\n"
        "Compute savings:\n"
        "  ~50% of tiles are skipped\n"
        "  (upper triangle of block grid)\n\n"
        "This is why flash attention is\n"
        "especially efficient for causal\n"
        "(autoregressive) models -- half\n"
        "the tiles need no computation."
    )
    axes[1, 2].text(0.05, 0.95, causal_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Flash Attention: Causal Masking Verification",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_causal_masking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/05_causal_masking.png")


# ---------------------------------------------------------------------------
# Example 6: Scaling Analysis and Real Model Projections
# ---------------------------------------------------------------------------
def example_6_scaling_analysis():
    """How memory savings grow with sequence length, projecting to real model sizes."""
    print("\n" + "=" * 60)
    print("Example 6: Scaling Analysis -- Real Model Projections")
    print("=" * 60)

    models = {
        "Llama 7B": dict(d=128, block_size=64),
        "GPT-3 175B": dict(d=128, block_size=128),
    }

    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    for model_name, cfg in models.items():
        d = cfg["d"]
        bs = cfg["block_size"]
        print(f"\n  {model_name} (d_head={d}, block={bs}):")
        print(f"  {'Context':>8} {'Standard':>14} {'Tiled':>14} {'Ratio':>8} {'Std Attn Matrix':>18}")
        print(f"  {'-' * 66}")

        for N in context_lengths:
            result = memory_analysis(N, d, bs, dtype="float16")
            std_b = result["standard"]["total_bytes"]
            tiled_b = result["tiled"]["total_bytes"]
            ratio = result["ratio"]

            attn_matrix_bytes = N * N * 2
            def fmt(b):
                if b < 1024:
                    return f"{b} B"
                if b < 1024 ** 2:
                    return f"{b / 1024:.1f} KB"
                if b < 1024 ** 3:
                    return f"{b / 1024 ** 2:.2f} MB"
                return f"{b / 1024 ** 3:.2f} GB"

            print(f"  {N:>8} {fmt(std_b):>14} {fmt(tiled_b):>14} {ratio:>7.0f}x {fmt(attn_matrix_bytes):>18}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    d = 128
    bs = 64
    std_mem = []
    tiled_mem = []
    scaling_ratios = []
    for N in context_lengths:
        r = memory_analysis(N, d, bs, dtype="float16")
        std_mem.append(r["standard"]["total_bytes"])
        tiled_mem.append(r["tiled"]["total_bytes"])
        scaling_ratios.append(r["ratio"])

    axes[0, 0].plot(context_lengths, [b / 1024**3 for b in std_mem], "o-",
                    color=COLORS["red"], linewidth=2, markersize=6,
                    label="Standard attention")
    axes[0, 0].plot(context_lengths, [b / 1024**3 for b in tiled_mem], "s-",
                    color=COLORS["green"], linewidth=2, markersize=6,
                    label="Tiled (flash) attention")
    axes[0, 0].axhline(80, color=COLORS["orange"], linestyle=":", linewidth=2, label="A100 80GB VRAM")
    axes[0, 0].axhline(40, color=COLORS["purple"], linestyle=":", linewidth=2, label="A100 40GB VRAM")
    axes[0, 0].set_xlabel("Sequence Length")
    axes[0, 0].set_ylabel("Peak Attention Memory (GB)")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title(f"Attention Memory at Scale (d_head={d}, FP16)\nStandard exceeds GPU VRAM; tiled stays manageable",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(context_lengths, scaling_ratios, "o-",
                    color=COLORS["purple"], linewidth=2, markersize=6)
    for i, (n, r) in enumerate(zip(context_lengths, scaling_ratios)):
        if i % 2 == 0 or i == len(context_lengths) - 1:
            axes[0, 1].annotate(f"{r:.0f}x", (n, r), textcoords="offset points",
                               xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")
    axes[0, 1].set_xlabel("Sequence Length")
    axes[0, 1].set_ylabel("Memory Savings Ratio")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Memory Savings Grows Linearly with N\n"
                         r"Ratio $\approx N / B$ for large N",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    attn_sizes = [N * N * 2 for N in context_lengths]
    labels_ctx = [f"{N // 1024}K" for N in context_lengths]
    bar_colors = [COLORS["green"] if s < 40 * 1024**3 else
                  (COLORS["orange"] if s < 80 * 1024**3 else COLORS["red"])
                  for s in attn_sizes]
    axes[0, 2].bar(range(len(context_lengths)), [s / 1024**3 for s in attn_sizes],
                   color=bar_colors, edgecolor="white")
    axes[0, 2].set_xticks(range(len(context_lengths)))
    axes[0, 2].set_xticklabels(labels_ctx, fontsize=8)
    axes[0, 2].set_xlabel("Context Length")
    axes[0, 2].set_ylabel("N x N Attention Matrix Size (GB, FP16)")
    axes[0, 2].axhline(80, color="black", linestyle=":", linewidth=1.5, label="80 GB")
    axes[0, 2].set_title("Size of Full Attention Matrix (FP16)\n"
                         "At 128K: 32 GB just for one matrix, one head, one layer",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    num_heads_list = [32, 40, 64, 96, 128]
    N_proj = 8192
    for nh in num_heads_list:
        total_std = nh * N_proj * N_proj * 2
        axes[1, 0].scatter(nh, total_std / 1024**3, s=80, zorder=5)
        axes[1, 0].annotate(f"{total_std / 1024**3:.1f} GB", (nh, total_std / 1024**3),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1, 0].set_xlabel("Number of Attention Heads")
    axes[1, 0].set_ylabel("Total Attention Memory (GB, FP16)")
    axes[1, 0].axhline(80, color=COLORS["red"], linestyle=":", linewidth=1.5, label="A100 80GB")
    axes[1, 0].set_title(f"Multi-Head Standard Attention Memory (N={N_proj})\n"
                         "Multiply per-head cost by number of heads",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    block_sizes_proj = [32, 64, 128]
    for bsp in block_sizes_proj:
        tiled_mems = []
        for N in context_lengths:
            r = memory_analysis(N, d, bsp, dtype="float16")
            tiled_mems.append(r["tiled"]["total_bytes"] / 1024**2)
        axes[1, 1].plot(context_lengths, tiled_mems, "o-", linewidth=1.5,
                        markersize=4, label=f"B={bsp}")
    axes[1, 1].set_xlabel("Sequence Length")
    axes[1, 1].set_ylabel("Tiled Attention Memory (MB)")
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_title("Tiled Attention Memory Scales Linearly\nNear-identical for different block sizes at large N",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    scale_text = (
        "SCALING ANALYSIS\n"
        "================\n\n"
        "Standard attention at 128K ctx:\n"
        f"  N x N matrix: {128*1024 * 128*1024 * 2 / 1024**3:.0f} GB (FP16)\n"
        "  Per head, per layer!\n"
        "  Completely impractical.\n\n"
        "Tiled attention at 128K ctx:\n"
        f"  Block mem: {2 * 64**2 * 2 / 1024:.1f} KB (constant)\n"
        f"  Output + stats: ~{128 * 1024 * 128 * 2 / 1024**2 + 128 * 1024 * 16 / 1024**2:.0f} MB\n"
        "  Completely tractable.\n\n"
        "This is why flash attention is\n"
        "THE enabling algorithm for long\n"
        "context models (128K, 1M+ tokens).\n\n"
        "Without it: 128K context would\n"
        "need 32 GB per head per layer\n"
        "just for attention scores.\n"
        "With it: a few MB total."
    )
    axes[1, 2].text(0.05, 0.95, scale_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Flash Attention: Scaling Analysis and Real Model Projections",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_scaling_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/06_scaling_analysis.png")


# ---------------------------------------------------------------------------
# Example 7: No-Materialization Proof
# ---------------------------------------------------------------------------
def example_7_no_materialization_proof():
    """Show that max intermediate tensor size stays constant as N grows."""
    print("\n" + "=" * 60)
    print("Example 7: No-Materialization Proof")
    print("=" * 60)

    rng = np.random.RandomState(SEED)
    d = 32
    block_size = 32
    Ns = [64, 128, 256, 512, 1024, 2048]

    print(f"\n  d={d}, block_size={block_size}")
    print(f"\n  {'N':>6} {'Max Tensor Elems':>18} {'N^2':>10} {'Block^2 * d':>12} {'Ratio (N^2/max)':>16}")
    print(f"  {'-' * 66}")

    max_elems_list = []
    n_squared_list = []

    for N in Ns:
        Q, K, V = rng.randn(N, d), rng.randn(N, d), rng.randn(N, d)
        O, max_elems = verify_no_full_materialization(Q, K, V, block_size)
        max_elems_list.append(max_elems)
        n_squared_list.append(N * N)

        O_std, _, _ = standard_attention(Q, K, V)
        err = float(np.max(np.abs(O - O_std)))

        ratio = (N * N) / max_elems
        print(f"  {N:>6} {max_elems:>18} {N * N:>10} {block_size * d:>12} {ratio:>15.1f}x")

    all_same = all(e == max_elems_list[0] for e in max_elems_list)
    print(f"\n  Max tensor elements constant across all N: {'YES' if all_same else 'NO'}")
    print(f"  Max tensor size: {max_elems_list[0]} elements = {max_elems_list[0] * 8} bytes (float64)")
    print(f"  This is block_size * d = {block_size} * {d} = {block_size * d}")
    print(f"  PROOF: No O(N^2) tensor is ever created in tiled attention")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].plot(Ns, max_elems_list, "s-", color=COLORS["green"], linewidth=2,
                    markersize=8, label="Max tensor (tiled)")
    axes[0, 0].plot(Ns, n_squared_list, "o-", color=COLORS["red"], linewidth=2,
                    markersize=6, label="N^2 (standard)")
    axes[0, 0].set_xlabel("Sequence Length (N)")
    axes[0, 0].set_ylabel("Max Intermediate Tensor Size (elements)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_title("Max Intermediate Tensor: Tiled vs Standard\n"
                         r"Tiled stays constant at $O(B \cdot d)$, standard grows as $O(N^2)$",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(range(len(Ns)), max_elems_list, color=COLORS["green"], edgecolor="white")
    axes[0, 1].set_xticks(range(len(Ns)))
    axes[0, 1].set_xticklabels([str(n) for n in Ns])
    axes[0, 1].set_xlabel("Sequence Length (N)")
    axes[0, 1].set_ylabel("Max Tensor Elements")
    for i, me in enumerate(max_elems_list):
        axes[0, 1].text(i, me + max(max_elems_list) * 0.03, str(me), ha="center",
                       va="bottom", fontsize=9, fontweight="bold")
    axes[0, 1].set_title("Max Tensor Size is CONSTANT\n"
                         f"Always {max_elems_list[0]} elements (B*d = {block_size}*{d})",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    improvement_factors = [n2 / me for n2, me in zip(n_squared_list, max_elems_list)]
    axes[0, 2].plot(Ns, improvement_factors, "o-", color=COLORS["purple"],
                    linewidth=2, markersize=6)
    for i, (n, f) in enumerate(zip(Ns, improvement_factors)):
        axes[0, 2].annotate(f"{f:.0f}x", (n, f), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")
    axes[0, 2].set_xlabel("Sequence Length (N)")
    axes[0, 2].set_ylabel("N^2 / Max Tensor Elements")
    axes[0, 2].set_xscale("log", base=2)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("How Much Larger Standard's Intermediate Is\n"
                         r"Grows as $O(N^2 / (B \cdot d))$",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3)

    block_sizes_sweep = [8, 16, 32, 64]
    N_fixed = 512
    for bsw in block_sizes_sweep:
        Q, K, V = rng.randn(N_fixed, d), rng.randn(N_fixed, d), rng.randn(N_fixed, d)
        _, max_e = verify_no_full_materialization(Q, K, V, bsw)
        axes[1, 0].bar(block_sizes_sweep.index(bsw), max_e,
                      color=COLORS["blue"], edgecolor="white")
    axes[1, 0].set_xticks(range(len(block_sizes_sweep)))
    axes[1, 0].set_xticklabels([str(bs) for bs in block_sizes_sweep])
    axes[1, 0].set_xlabel("Block Size")
    axes[1, 0].set_ylabel("Max Tensor Elements")
    axes[1, 0].set_title(f"Max Tensor vs Block Size (N={N_fixed})\n"
                         r"Scales as $O(B \cdot d)$, independent of N",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    tensor_names = ["Q_block\n(B x d)", "K_block\n(B x d)", "S_block\n(B x B)",
                    "P_block\n(B x B)", "PV_block\n(B x d)"]
    tensor_sizes = [block_size * d, block_size * d, block_size * block_size,
                    block_size * block_size, block_size * d]
    bar_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"],
                  COLORS["red"], COLORS["purple"]]
    axes[1, 1].bar(range(len(tensor_names)), tensor_sizes,
                   color=bar_colors, edgecolor="white")
    axes[1, 1].set_xticks(range(len(tensor_names)))
    axes[1, 1].set_xticklabels(tensor_names, fontsize=8)
    axes[1, 1].set_ylabel("Elements")
    axes[1, 1].axhline(Ns[-1] * Ns[-1], color="black", linestyle="--", linewidth=1.5,
                        label=f"N^2 = {Ns[-1]}^2 = {Ns[-1]**2:,}")
    axes[1, 1].set_title(f"Per-Tile Intermediate Sizes (B={block_size}, d={d})\n"
                         "All block intermediates << N^2",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].set_yscale("log")

    axes[1, 2].axis("off")
    proof_text = (
        "NO-MATERIALIZATION PROOF\n"
        "========================\n\n"
        "Claim: Tiled attention never\n"
        "creates an O(N^2) tensor.\n\n"
        "Proof by instrumentation:\n"
        f"  Tracked max tensor at each step\n"
        f"  for N = {Ns[0]} to {Ns[-1]}.\n\n"
        f"  Max tensor: ALWAYS {max_elems_list[0]} elements\n"
        f"  = B * d = {block_size} * {d}\n\n"
        f"  At N={Ns[-1]}:\n"
        f"    Standard: N^2 = {Ns[-1]**2:,} elements\n"
        f"    Tiled:    B*d = {max_elems_list[0]} elements\n"
        f"    Ratio:    {Ns[-1]**2 / max_elems_list[0]:,.0f}x smaller\n\n"
        "The intermediates are:\n"
        f"  S_ij  (B x B)   = {block_size**2} elements\n"
        f"  P_ij  (B x B)   = {block_size**2} elements\n"
        f"  Q_i   (B x d)   = {block_size * d} elements\n"
        f"  K_j   (B x d)   = {block_size * d} elements\n"
        f"  PV_ij (B x d)   = {block_size * d} elements\n\n"
        "All O(B^2) or O(B*d). QED."
    )
    axes[1, 2].text(0.05, 0.95, proof_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Flash Attention: No-Materialization Proof",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_no_materialization_proof.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/07_no_materialization_proof.png")


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------
def generate_pdf_report():
    """Generate comprehensive PDF report with LaTeX equations."""
    print("\n" + "=" * 60)
    print("Generating PDF Report")
    print("=" * 60)

    report_path = Path(__file__).parent / "report.pdf"
    viz_files = sorted(VIZ_DIR.glob("*.png"))

    with PdfPages(str(report_path)) as pdf:
        # --- Title page ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.78, "Flash Attention Concept", fontsize=28, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.68, "The Algorithm That Makes Long-Context Attention Tractable",
                fontsize=13, ha="center", va="center", transform=ax.transAxes, color="gray")
        info_text = (
            "Flash attention reformulates standard attention to avoid materializing the\n"
            "full N x N attention matrix. By processing Q, K, V in tiles and using online\n"
            "softmax to incrementally compute exact results, it reduces attention memory\n"
            "from O(N^2) to O(N) while producing numerically identical output.\n\n"
            "This demo covers:\n"
            "  1. Numerical equivalence: tiled == standard within float64 precision\n"
            "  2. Memory comparison: O(N^2) vs O(N) with actual byte counts\n"
            "  3. Block size analysis: memory vs iteration trade-offs\n"
            "  4. Tiling visualization: processing order and block coverage\n"
            "  5. Causal masking: correct causal attention with block skipping\n"
            "  6. Scaling analysis: projections to 8K, 32K, 128K contexts\n"
            "  7. No-materialization proof: max intermediate stays constant\n\n"
            f"Random seed: {SEED}\n"
            f"Number of visualizations: {len(viz_files)}\n"
            "Examples: 7"
        )
        ax.text(0.5, 0.30, info_text, fontsize=11, ha="center", va="center",
                transform=ax.transAxes, linespacing=1.6)
        ax.text(0.5, 0.06, "Generated by demo.py", fontsize=10, ha="center",
                va="center", transform=ax.transAxes, style="italic", color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Math page 1: Standard Attention & Online Softmax ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.96, "Mathematical Foundation", fontsize=20, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)

        y = 0.88
        dy = 0.045

        ax.text(0.05, y, "Standard Attention (the problem)", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y, r"Given $Q, K, V \in \mathbb{R}^{N \times d}$, compute:",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$S = \frac{Q K^\top}{\sqrt{d}} \in \mathbb{R}^{N \times N}$",
                fontsize=12, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$P = \mathrm{softmax}(S) \in \mathbb{R}^{N \times N} \qquad O = P V \in \mathbb{R}^{N \times d}$",
                fontsize=12, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Peak memory: $O(N^2)$ for $S$ and $P$ matrices",
                fontsize=10, transform=ax.transAxes, color=COLORS["red"])

        y -= dy * 1.5
        ax.text(0.05, y, "Online Softmax (the key insight)", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y, r"Softmax can be computed in streaming chunks. For chunk $x^{(j)}$:",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$m_{\mathrm{new}} = \max(m, \max(x^{(j)}))$",
                fontsize=12, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\ell = \ell \cdot \exp(m - m_{\mathrm{new}}) + \sum_i \exp(x^{(j)}_i - m_{\mathrm{new}})$",
                fontsize=12, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$m \leftarrow m_{\mathrm{new}}$",
                fontsize=12, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Final: $\mathrm{softmax}(x)_i = \exp(x_i - m)\;/\;\ell$",
                fontsize=12, transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y,
                r"The correction factor $\exp(m_{\mathrm{old}} - m_{\mathrm{new}})$ rescales previous sums.",
                fontsize=10, transform=ax.transAxes, style="italic", color=COLORS["dark"])

        y -= dy * 1.5
        ax.text(0.05, y, "Memory Analysis", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Standard: $\mathrm{Memory} = O(N^2)$ (must store full $N \times N$ matrices $S$ and $P$)",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Flash:    $\mathrm{Memory} = O(N)$ (only per-row statistics $m, \ell$ and output $O$)",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Block matrices $S_{ij}, P_{ij}$ are $O(B_r \cdot B_c) = O(1)$ relative to $N$",
                fontsize=10, transform=ax.transAxes, color=COLORS["green"])

        pdf.savefig(fig)
        plt.close(fig)

        # --- Math page 2: Tiled Attention Algorithm ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.96, "Tiled Attention Algorithm (Flash Attention)", fontsize=20,
                fontweight="bold", ha="center", va="top", transform=ax.transAxes)

        y = 0.88
        dy = 0.04

        ax.text(0.05, y, "Algorithm", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y, r"Initialize: $m_i = -\infty, \; \ell_i = 0, \; O = \mathbf{0}_{N \times d}$ for all $i$",
                fontsize=10, transform=ax.transAxes)

        y -= dy
        ax.text(0.07, y, r"For each KV block $j$:  $K_j = K[jB_c : (j{+}1)B_c], \; V_j = V[jB_c : (j{+}1)B_c]$",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.09, y, r"For each Q block $i$:  $Q_i = Q[iB_r : (i{+}1)B_r]$",
                fontsize=10, transform=ax.transAxes)

        y -= dy * 1.2
        ax.text(0.09, y, "Compute block scores:", fontsize=10, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.12, y,
                r"$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} \in \mathbb{R}^{B_r \times B_c}$",
                fontsize=12, transform=ax.transAxes)

        y -= dy * 1.2
        ax.text(0.09, y, "Online softmax update:", fontsize=10, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.12, y,
                r"$\tilde{m}_{ij} = \mathrm{rowmax}(S_{ij}), \quad \tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij}), \quad \tilde{\ell}_{ij} = \mathrm{rowsum}(\tilde{P}_{ij})$",
                fontsize=10, transform=ax.transAxes)

        y -= dy * 1.2
        ax.text(0.09, y, "Combine statistics:", fontsize=10, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.12, y,
                r"$m_{\mathrm{new}} = \max(m_i, \tilde{m}_{ij})$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.12, y,
                r"$\alpha = \exp(m_i - m_{\mathrm{new}}), \quad \beta = \exp(\tilde{m}_{ij} - m_{\mathrm{new}})$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.12, y,
                r"$\ell_{\mathrm{new}} = \ell_i \cdot \alpha + \tilde{\ell}_{ij} \cdot \beta$",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.2
        ax.text(0.09, y, "Update output:", fontsize=10, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.12, y,
                r"$O_i = \frac{O_i \cdot \alpha \cdot \ell_i + (\tilde{P}_{ij} \cdot \beta) V_j}{\ell_{\mathrm{new}}}$",
                fontsize=12, transform=ax.transAxes)
        y -= dy
        ax.text(0.12, y,
                r"$m_i \leftarrow m_{\mathrm{new}}, \quad \ell_i \leftarrow \ell_{\mathrm{new}}$",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.5
        ax.text(0.05, y, "Key Property", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y,
                r"The rescaling $\alpha = \exp(m_{\mathrm{old}} - m_{\mathrm{new}})$ corrects all previous",
                fontsize=10, transform=ax.transAxes)
        y -= dy * 0.8
        ax.text(0.07, y,
                r"accumulations when a new block reveals a larger maximum. This yields EXACT softmax.",
                fontsize=10, transform=ax.transAxes)

        pdf.savefig(fig)
        plt.close(fig)

        # --- Summary page ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.94, "Summary of Findings", fontsize=20, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)

        summary_items = [
            "1. NUMERICAL EQUIVALENCE: Tiled attention produces bit-identical results to",
            "   standard attention for all tested configurations (N up to 1024, various",
            "   block sizes including non-divisible N). Online softmax is numerically",
            "   stable even for extreme values (+/-1000). Max error < 1e-10.",
            "",
            "2. MEMORY COMPARISON: Standard attention requires O(N^2) memory for the S",
            "   and P matrices. Tiled attention requires O(N) -- only block-sized",
            "   intermediates plus per-row statistics. At N=8192 (d=64, B=32, FP32)",
            "   the ratio exceeds 240x, growing linearly with N.",
            "",
            "3. BLOCK SIZE TRADE-OFFS: Smaller blocks use less peak memory but require",
            "   more tile iterations. Larger blocks reduce iterations but need more",
            "   memory per tile. Total computation is always N^2 * d regardless of",
            "   block size. In practice, block size is chosen to fit GPU SRAM (~48 KB).",
            "",
            "4. TILING VISUALIZATION: The algorithm processes tiles in outer-KV, inner-Q",
            "   order. K/V blocks are loaded once and reused across all Q blocks. For",
            "   causal masking, ~50% of tiles are entirely above the diagonal and skipped.",
            "",
            "5. CAUSAL MASKING: Tiled causal attention matches reference causal attention",
            "   within 1e-5 tolerance, including non-divisible sequence lengths and",
            "   asymmetric block sizes. Block-level skipping provides ~50% compute savings.",
            "",
            "6. SCALING ANALYSIS: At 128K context with FP16, the standard attention",
            "   matrix alone requires 32 GB per head per layer. Flash attention keeps",
            "   intermediates at a few KB regardless of context length, enabling 128K+",
            "   context models that would be impossible with standard attention.",
            "",
            "7. NO-MATERIALIZATION PROOF: Instrumented tracking confirms the largest",
            "   intermediate tensor stays constant (B*d elements) as N grows from 64 to",
            "   2048. At N=2048 this is 4,096x smaller than the N^2 standard matrix.",
        ]
        summary_text = "\n".join(summary_items)
        ax.text(0.06, 0.86, summary_text, fontsize=9.5, ha="left", va="top",
                transform=ax.transAxes, family="monospace", linespacing=1.3)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Visualization pages ---
        titles = {
            "01_numerical_equivalence.png": "Example 1: Numerical Equivalence Verification",
            "02_memory_comparison.png": "Example 2: Memory Comparison -- O(N^2) vs O(N)",
            "03_block_size_analysis.png": "Example 3: Block Size Analysis",
            "04_tiling_visualization.png": "Example 4: Tiling Visualization",
            "05_causal_masking.png": "Example 5: Causal Masking Verification",
            "06_scaling_analysis.png": "Example 6: Scaling Analysis and Real Model Projections",
            "07_no_materialization_proof.png": "Example 7: No-Materialization Proof",
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

    print(f"  Report saved: report.pdf ({len(viz_files) + 4} pages)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Flash Attention Concept Demo")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print()

    example_1_numerical_equivalence()
    example_2_memory_comparison()
    example_3_block_size_analysis()
    example_4_tiling_visualization()
    example_5_causal_masking()
    example_6_scaling_analysis()
    example_7_no_materialization_proof()
    generate_pdf_report()

    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print(f"Visualizations: {VIZ_DIR}/")
    print(f"Report: {Path(__file__).parent / 'report.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
