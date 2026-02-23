"""
Mixed Precision Demo -- Bit layouts, precision analysis, loss scaling dynamics,
accumulation errors, softmax stability, memory savings, and training comparison.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from implementation import (
    float_info,
    visualize_bits,
    bf16_simulate,
    precision_loss,
    consecutive_value_gap,
    mixed_precision_matmul,
    stable_softmax,
    LossScaler,
    detect_overflow,
    model_memory_usage,
    kv_cache_memory,
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
# Example 1: Bit Layout Visualization
# ---------------------------------------------------------------------------
def example_1_bit_layout():
    """Show actual bit patterns for key values in FP32, FP16, and BF16."""
    print("=" * 60)
    print("Example 1: Bit Layout Visualization")
    print("=" * 60)

    test_values = [1.0, -1.0, 0.5, 2.0, 3.14, 0.1, 65504.0, 0.0]
    labels = ["1.0", "-1.0", "0.5", "2.0", "3.14", "0.1", "65504", "0.0"]

    fp32_bits = [visualize_bits(v, np.float32) for v in test_values]
    fp16_bits = [visualize_bits(v, np.float16) for v in test_values]

    bf16_bits = []
    for v in test_values:
        fp32_raw = visualize_bits(v, np.float32)
        parts = fp32_raw.split(" | ")
        bf16_mantissa = parts[2][:7]
        bf16_bits.append(f"{parts[0]} | {parts[1]} | {bf16_mantissa}")

    print(f"\n  {'Value':>8}  {'FP32 (1+8+23)':^40}  {'FP16 (1+5+10)':^22}  {'BF16 (1+8+7)':^22}")
    print(f"  {'-'*96}")
    for i, v in enumerate(labels):
        print(f"  {v:>8}  {fp32_bits[i]:^40}  {fp16_bits[i]:^22}  {bf16_bits[i]:^22}")

    fp32_info = float_info(np.float32)
    fp16_info = float_info(np.float16)
    bf16_info = float_info("bfloat16")

    print(f"\n  Format Properties:")
    print(f"    FP32: max={fp32_info['max_value']:.4e}, precision={fp32_info['precision_decimal_digits']:.1f} digits")
    print(f"    FP16: max={fp16_info['max_value']:.0f}, precision={fp16_info['precision_decimal_digits']:.1f} digits")
    print(f"    BF16: max={bf16_info['max_value']:.4e}, precision={bf16_info['precision_decimal_digits']:.1f} digits")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    def draw_bit_pattern(ax, bits_str, label, fmt_name, color_sign, color_exp, color_mant):
        parts = bits_str.split(" | ")
        sign_bits = parts[0]
        exp_bits = parts[1]
        mant_bits = parts[2]
        all_bits = sign_bits + exp_bits + mant_bits
        n_bits = len(all_bits)

        for i, bit in enumerate(all_bits):
            if i < len(sign_bits):
                c = color_sign
            elif i < len(sign_bits) + len(exp_bits):
                c = color_exp
            else:
                c = color_mant
            rect = plt.Rectangle((i, 0), 1, 1, facecolor=c, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(i + 0.5, 0.5, bit, ha="center", va="center", fontsize=max(6, 14 - n_bits // 5),
                    fontweight="bold", color="white")
        ax.set_xlim(0, n_bits)
        ax.set_ylim(-0.3, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{fmt_name}: {label}", fontsize=10, fontweight="bold")

    show_values = [(1.0, "1.0"), (3.14, "3.14"), (0.1, "0.1")]
    for col, (val, lbl) in enumerate(show_values):
        fp32_str = visualize_bits(val, np.float32)
        draw_bit_pattern(axes[0, col], fp32_str, lbl, "FP32",
                         COLORS["red"], COLORS["blue"], COLORS["green"])

    for col, (val, lbl) in enumerate(show_values):
        fp16_str = visualize_bits(val, np.float16)
        draw_bit_pattern(axes[1, col], fp16_str, lbl, "FP16",
                         COLORS["red"], COLORS["blue"], COLORS["green"])

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["red"], label="Sign (1 bit)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["blue"], label="Exponent"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["green"], label="Mantissa"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11,
               frameon=True, fancybox=True)

    fig.suptitle("Floating Point Bit Layout: FP32 vs FP16",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(VIZ_DIR / "01_bit_layout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/01_bit_layout.png")


# ---------------------------------------------------------------------------
# Example 2: Precision Loss Analysis
# ---------------------------------------------------------------------------
def example_2_precision_loss():
    """Relative error across the representable range for FP16 and BF16."""
    print("\n" + "=" * 60)
    print("Example 2: Precision Loss Analysis")
    print("=" * 60)

    rng = np.random.RandomState(SEED)
    values = np.concatenate([
        np.logspace(-6, 0, 200),
        np.logspace(0, 4, 200),
        rng.uniform(0.001, 100, 200),
    ]).astype(np.float32)
    values = np.sort(values)

    err_fp16 = precision_loss(values, np.float16)
    err_bf16 = precision_loss(values, "bfloat16")

    print(f"  Values tested: {len(values)}")
    print(f"  FP16 -- mean rel error: {np.mean(err_fp16):.6e}, max: {np.max(err_fp16):.6e}")
    print(f"  BF16 -- mean rel error: {np.mean(err_bf16):.6e}, max: {np.max(err_bf16):.6e}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].scatter(values, err_fp16, s=3, alpha=0.5, color=COLORS["blue"], label="FP16")
    axes[0, 0].scatter(values, err_bf16, s=3, alpha=0.5, color=COLORS["orange"], label="BF16")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Value Magnitude")
    axes[0, 0].set_ylabel("Relative Error")
    axes[0, 0].set_title("Relative Error: FP16 vs BF16\nBF16 has higher error but larger range",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    bins = np.logspace(-10, -1, 50)
    axes[0, 1].hist(err_fp16[err_fp16 > 0], bins=bins, alpha=0.6, color=COLORS["blue"], label="FP16")
    axes[0, 1].hist(err_bf16[err_bf16 > 0], bins=bins, alpha=0.6, color=COLORS["orange"], label="BF16")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Relative Error")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Error Distribution\nBF16 errors cluster at higher values",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    weight_values = rng.randn(5000).astype(np.float32) * 0.02
    err_w_fp16 = precision_loss(np.abs(weight_values[weight_values != 0]), np.float16)
    err_w_bf16 = precision_loss(np.abs(weight_values[weight_values != 0]), "bfloat16")
    axes[0, 2].hist(err_w_fp16[err_w_fp16 > 0], bins=50, alpha=0.6, color=COLORS["blue"], label="FP16")
    axes[0, 2].hist(err_w_bf16[err_w_bf16 > 0], bins=50, alpha=0.6, color=COLORS["orange"], label="BF16")
    axes[0, 2].set_xlabel("Relative Error")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title("Error on Typical Weight Distribution\nN(0, 0.02) -- where neural net weights live",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    ranges = [(0.001, 0.01), (0.1, 1.0), (1.0, 10.0), (100.0, 1000.0), (10000.0, 60000.0)]
    fp16_means = []
    bf16_means = []
    range_labels = []
    for lo, hi in ranges:
        v = np.linspace(lo, hi, 500).astype(np.float32)
        fp16_means.append(np.mean(precision_loss(v, np.float16)))
        bf16_means.append(np.mean(precision_loss(v, "bfloat16")))
        range_labels.append(f"[{lo},{hi}]")

    x = np.arange(len(ranges))
    axes[1, 0].bar(x - 0.15, fp16_means, 0.3, label="FP16", color=COLORS["blue"], edgecolor="white")
    axes[1, 0].bar(x + 0.15, bf16_means, 0.3, label="BF16", color=COLORS["orange"], edgecolor="white")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(range_labels, fontsize=8, rotation=20)
    axes[1, 0].set_ylabel("Mean Relative Error")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Mean Error by Value Range\nFP16 overflows for large values; BF16 does not",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    int_values = np.arange(1, 4097, dtype=np.float32)
    fp16_ints = int_values.astype(np.float16).astype(np.float32)
    int_errors = np.abs(int_values - fp16_ints)
    axes[1, 1].plot(int_values, int_errors, "-", color=COLORS["blue"], linewidth=0.5)
    axes[1, 1].axvline(2048, color=COLORS["red"], linestyle="--", linewidth=2,
                       label="2048 (last exact int)")
    axes[1, 1].set_xlabel("Integer Value")
    axes[1, 1].set_ylabel("Absolute Error in FP16")
    axes[1, 1].set_title("Integer Representation Error in FP16\nBeyond 2048, not all integers are exact",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    info_text = (
        "PRECISION LOSS ANALYSIS\n"
        "=======================\n\n"
        "FP16 (10-bit mantissa):\n"
        f"  Mean rel error: {np.mean(err_fp16):.2e}\n"
        f"  Max rel error:  {np.max(err_fp16):.2e}\n"
        f"  ~3.3 decimal digits\n\n"
        "BF16 (7-bit mantissa):\n"
        f"  Mean rel error: {np.mean(err_bf16):.2e}\n"
        f"  Max rel error:  {np.max(err_bf16):.2e}\n"
        f"  ~2.1 decimal digits\n\n"
        "Key insight: BF16 has worse\n"
        "precision but same range as FP32.\n"
        "Neural nets tolerate this because\n"
        "weights cluster near zero and\n"
        "learning is robust to noise."
    )
    axes[1, 2].text(0.05, 0.95, info_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Precision Loss: FP16 vs BF16 Across Value Ranges",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_precision_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/02_precision_loss.png")


# ---------------------------------------------------------------------------
# Example 3: Overflow/Underflow Boundaries
# ---------------------------------------------------------------------------
def example_3_overflow_underflow():
    """Visual showing safe vs dangerous zones for each format."""
    print("\n" + "=" * 60)
    print("Example 3: Overflow/Underflow Boundaries")
    print("=" * 60)

    fp16_max = 65504.0
    fp16_min_normal = float(np.finfo(np.float16).tiny)
    fp16_min_subnormal = float(np.finfo(np.float16).smallest_subnormal)
    fp32_max = float(np.finfo(np.float32).max)

    print(f"  FP16 max: {fp16_max}")
    print(f"  FP16 min normal: {fp16_min_normal:.4e}")
    print(f"  FP16 min subnormal: {fp16_min_subnormal:.4e}")
    print(f"  FP32 max: {fp32_max:.4e}")

    overflow_tests = [60000, 65000, 65504, 65505, 65520, 65536, 100000]
    print(f"\n  FP16 Overflow tests:")
    for v in overflow_tests:
        fp16_v = np.float16(v)
        print(f"    {v:>8} -> FP16: {float(fp16_v):>12} {'(INF!)' if np.isinf(fp16_v) else '(ok)'}")

    underflow_tests = [1e-3, 1e-4, 6.1e-5, 1e-5, 1e-6, 1e-7, 5.96e-8, 1e-8, 1e-10]
    print(f"\n  FP16 Underflow tests:")
    for v in underflow_tests:
        fp16_v = np.float16(v)
        print(f"    {v:>12.2e} -> FP16: {float(fp16_v):>12.4e} {'(ZERO!)' if float(fp16_v) == 0 else '(ok)'}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    formats = ["FP32", "FP16", "BF16"]
    maxvals = [3.4e38, 65504, 3.4e38]
    minvals = [1.18e-38, 6.1e-5, 1.18e-38]
    bar_colors = [COLORS["blue"], COLORS["coral"], COLORS["orange"]]

    axes[0, 0].barh(formats, [np.log10(m) for m in maxvals], color=bar_colors, edgecolor="white", height=0.5)
    for i, (fmt, mv) in enumerate(zip(formats, maxvals)):
        axes[0, 0].text(np.log10(mv) + 0.5, i, f"{mv:.2e}", va="center", fontsize=9, fontweight="bold")
    axes[0, 0].set_xlabel("log10(Max Value)")
    axes[0, 0].set_title("Maximum Representable Value\nFP16 caps at 65504 -- BF16 matches FP32",
                         fontsize=10, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    axes[0, 1].barh(formats, [-np.log10(m) for m in minvals], color=bar_colors, edgecolor="white", height=0.5)
    for i, (fmt, mv) in enumerate(zip(formats, minvals)):
        axes[0, 1].text(-np.log10(mv) + 0.5, i, f"{mv:.2e}", va="center", fontsize=9, fontweight="bold")
    axes[0, 1].set_xlabel("-log10(Min Positive Normal)")
    axes[0, 1].set_title("Smallest Normal Positive Value\nSmaller = better for tiny gradients",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="x")

    test_vals = np.array(overflow_tests, dtype=np.float32)
    fp16_results = test_vals.astype(np.float16).astype(np.float32)
    is_inf = np.isinf(fp16_results)
    colors_ov = [COLORS["red"] if inf else COLORS["green"] for inf in is_inf]
    axes[0, 2].bar(range(len(test_vals)), test_vals, color=colors_ov, edgecolor="white")
    axes[0, 2].axhline(65504, color="black", linestyle="--", linewidth=2, label="FP16 max (65504)")
    axes[0, 2].set_xticks(range(len(test_vals)))
    axes[0, 2].set_xticklabels([str(int(v)) for v in test_vals], fontsize=8, rotation=30)
    axes[0, 2].set_ylabel("Value")
    axes[0, 2].set_title("FP16 Overflow Test\nGreen=safe, Red=overflow to inf",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    test_small = np.array(underflow_tests, dtype=np.float32)
    fp16_small = test_small.astype(np.float16).astype(np.float32)
    is_zero = fp16_small == 0
    colors_uf = [COLORS["red"] if z else COLORS["green"] for z in is_zero]
    axes[1, 0].bar(range(len(test_small)), -np.log10(test_small), color=colors_uf, edgecolor="white")
    axes[1, 0].set_xticks(range(len(test_small)))
    axes[1, 0].set_xticklabels([f"{v:.0e}" for v in test_small], fontsize=7, rotation=40)
    axes[1, 0].set_ylabel("-log10(value)")
    axes[1, 0].set_title("FP16 Underflow Test\nGreen=survives, Red=underflows to zero",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    log_range = np.linspace(-10, 5, 1000)
    magnitudes = 10.0 ** log_range

    fp16_safe = (magnitudes >= fp16_min_subnormal) & (magnitudes <= fp16_max)
    bf16_safe = (magnitudes >= 1.18e-38) & (magnitudes <= 3.4e38)

    axes[1, 1].fill_between(log_range, 0, 1, where=fp16_safe, alpha=0.3, color=COLORS["blue"],
                            label="FP16 safe zone")
    axes[1, 1].fill_between(log_range, 1, 2, where=bf16_safe, alpha=0.3, color=COLORS["orange"],
                            label="BF16 safe zone")
    axes[1, 1].axvline(np.log10(fp16_max), color=COLORS["red"], linestyle="--", linewidth=2)
    axes[1, 1].axvline(np.log10(fp16_min_subnormal), color=COLORS["red"], linestyle="--", linewidth=1)
    axes[1, 1].text(np.log10(fp16_max), 0.5, f"FP16 max\n{fp16_max:.0f}", fontsize=8,
                    ha="right", va="center", color=COLORS["red"])
    axes[1, 1].set_xlabel("log10(value)")
    axes[1, 1].set_yticks([0.5, 1.5])
    axes[1, 1].set_yticklabels(["FP16", "BF16"])
    axes[1, 1].set_title("Safe Representable Zones\nBF16 covers the full FP32 range",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9, loc="upper left")
    axes[1, 1].grid(True, alpha=0.3, axis="x")

    axes[1, 2].axis("off")
    info_text = (
        "OVERFLOW / UNDERFLOW\n"
        "====================\n\n"
        "FP16 Overflow:\n"
        f"  Max finite: {fp16_max}\n"
        "  65520+ -> inf\n"
        "  Common trigger: loss values,\n"
        "  gradient norms, activations\n\n"
        "FP16 Underflow:\n"
        f"  Min subnormal: {fp16_min_subnormal:.2e}\n"
        "  <5.96e-8 -> zero\n"
        "  Common trigger: small gradients\n"
        "  (why loss scaling is needed)\n\n"
        "BF16 advantage:\n"
        "  Same range as FP32 (8-bit exp)\n"
        "  No overflow issues\n"
        "  No loss scaling needed\n"
        "  Trade: less precision (7-bit\n"
        "  mantissa vs FP16's 10-bit)"
    )
    axes[1, 2].text(0.05, 0.95, info_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Overflow and Underflow Boundaries: FP32, FP16, BF16",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_overflow_underflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/03_overflow_underflow.png")


# ---------------------------------------------------------------------------
# Example 4: Consecutive Value Gap
# ---------------------------------------------------------------------------
def example_4_consecutive_gap():
    """How spacing between representable numbers grows with magnitude."""
    print("\n" + "=" * 60)
    print("Example 4: Consecutive Value Gap")
    print("=" * 60)

    test_points = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    print(f"\n  {'Value':>10} {'FP32 gap':>14} {'FP16 gap':>14} {'BF16 gap':>14}")
    print(f"  {'-'*56}")
    gaps_fp32, gaps_fp16, gaps_bf16 = [], [], []
    for v in test_points:
        g32 = consecutive_value_gap(v, np.float32)
        g16 = consecutive_value_gap(v, np.float16) if v <= 65504 else float("inf")
        gbf = consecutive_value_gap(v, "bfloat16")
        gaps_fp32.append(g32)
        gaps_fp16.append(g16)
        gaps_bf16.append(gbf)
        print(f"  {v:>10.3f} {g32:>14.4e} {g16:>14.4e} {gbf:>14.4e}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    fp16_valid = [v for v in test_points if v <= 65504]
    fp16_gaps_valid = [g for v, g in zip(test_points, gaps_fp16) if v <= 65504]

    axes[0, 0].plot(test_points, gaps_fp32, "o-", color=COLORS["blue"], linewidth=2, markersize=6, label="FP32")
    axes[0, 0].plot(fp16_valid, fp16_gaps_valid, "s-", color=COLORS["coral"], linewidth=2, markersize=6, label="FP16")
    axes[0, 0].plot(test_points, gaps_bf16, "^-", color=COLORS["orange"], linewidth=2, markersize=6, label="BF16")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Gap to Next Representable Number")
    axes[0, 0].set_title("Consecutive Value Gap vs Magnitude\nGap grows proportionally to value",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    rel_gaps_fp16 = [g / v for g, v in zip(fp16_gaps_valid, fp16_valid)]
    rel_gaps_bf16 = [g / v for g, v in zip(gaps_bf16, test_points)]
    rel_gaps_fp32 = [g / v for g, v in zip(gaps_fp32, test_points)]

    axes[0, 1].plot(test_points, rel_gaps_fp32, "o-", color=COLORS["blue"], linewidth=2, label="FP32")
    axes[0, 1].plot(fp16_valid, rel_gaps_fp16, "s-", color=COLORS["coral"], linewidth=2, label="FP16")
    axes[0, 1].plot(test_points, rel_gaps_bf16, "^-", color=COLORS["orange"], linewidth=2, label="BF16")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Relative Gap (gap / value)")
    axes[0, 1].set_title("Relative Gap (Machine Epsilon)\nConstant within each format -- this IS precision",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    values_dense = np.linspace(1.0, 2.0, 10000).astype(np.float32)
    fp16_vals = np.unique(values_dense.astype(np.float16).astype(np.float32))
    bf16_vals = np.unique(bf16_simulate(values_dense))
    fp16_in_range = fp16_vals[(fp16_vals >= 1.0) & (fp16_vals <= 2.0)]
    bf16_in_range = bf16_vals[(bf16_vals >= 1.0) & (bf16_vals <= 2.0)]

    axes[0, 2].eventplot([fp16_in_range], lineoffsets=1, linewidths=0.5, colors=COLORS["coral"], label="FP16")
    axes[0, 2].eventplot([bf16_in_range], lineoffsets=2, linewidths=0.5, colors=COLORS["orange"], label="BF16")
    axes[0, 2].set_yticks([1, 2])
    axes[0, 2].set_yticklabels(["FP16", "BF16"])
    axes[0, 2].set_xlim(1.0, 2.0)
    axes[0, 2].set_xlabel("Value")
    axes[0, 2].set_title(f"Representable Values in [1.0, 2.0]\nFP16: {len(fp16_in_range)} values, "
                         f"BF16: {len(bf16_in_range)} values",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3, axis="x")

    lr_values = [1e-5, 1e-4, 1e-3, 1e-2]
    weight_magnitudes = np.logspace(-2, 2, 100)
    for lr in lr_values:
        effective_updates = []
        for w in weight_magnitudes:
            gap = consecutive_value_gap(w, np.float16) if w <= 65504 else float("inf")
            update = lr * 0.01
            effective_updates.append(1.0 if update > gap else 0.0)
        axes[1, 0].plot(weight_magnitudes, effective_updates, linewidth=2, label=f"lr={lr}")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Weight Magnitude")
    axes[1, 0].set_ylabel("Update Effective (1=yes, 0=lost)")
    axes[1, 0].set_title("Gradient Update Effectiveness in FP16\nUpdate lost when lr*grad < gap",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    values_1_8 = np.linspace(1.0, 8.0, 50000).astype(np.float32)
    fp16_mapped = values_1_8.astype(np.float16).astype(np.float32)
    gaps = np.diff(np.unique(fp16_mapped))
    unique_fp16 = np.unique(fp16_mapped)

    axes[1, 1].step(unique_fp16[:-1], gaps, where="post", color=COLORS["coral"], linewidth=1.5)
    for exp_boundary in [1.0, 2.0, 4.0, 8.0]:
        axes[1, 1].axvline(exp_boundary, color="gray", linestyle=":", alpha=0.5)
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].set_ylabel("Gap Between Consecutive FP16 Values")
    axes[1, 1].set_title("FP16 Gap Structure in [1, 8]\nGap doubles at each power of 2 (exponent change)",
                         fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    info_text = (
        "CONSECUTIVE VALUE GAPS\n"
        "======================\n\n"
        "Gap formula:\n"
        "  gap = 2^(exponent - mantissa_bits)\n\n"
        "At value 1.0:\n"
        f"  FP32: 2^(-23) = {2**-23:.2e}\n"
        f"  FP16: 2^(-10) = {2**-10:.2e}\n"
        f"  BF16: 2^(-7)  = {2**-7:.4f}\n\n"
        "At value 1000:\n"
        f"  FP32: {consecutive_value_gap(1000, np.float32):.2e}\n"
        f"  FP16: {consecutive_value_gap(1000, np.float16):.1f}\n"
        f"  BF16: {consecutive_value_gap(1000, 'bfloat16'):.1f}\n\n"
        "Implication: at value 1000,\n"
        "BF16 values are 4.0 apart.\n"
        "Any gradient update smaller\n"
        "than 4.0 is effectively zero."
    )
    axes[1, 2].text(0.05, 0.95, info_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Consecutive Value Gap: How Precision Degrades with Magnitude",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_consecutive_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/04_consecutive_gap.png")


# ---------------------------------------------------------------------------
# Example 5: Loss Scaling Dynamics
# ---------------------------------------------------------------------------
def example_5_loss_scaling():
    """Demonstrate the dynamic loss scaler over many steps."""
    print("\n" + "=" * 60)
    print("Example 5: Loss Scaling Dynamics")
    print("=" * 60)

    rng = np.random.RandomState(SEED)
    n_steps = 500
    scaler = LossScaler(init_scale=2**15, growth_interval=50, growth_factor=2.0,
                        backoff_factor=0.5, max_scale=2**24)

    scales = []
    overflows = []
    grad_survivals = []

    for step in range(n_steps):
        grads = rng.randn(64, 32).astype(np.float32) * 0.001

        if step in [50, 120, 200, 350]:
            grads[0, 0] = 1e10

        scaled_grads = (grads * scaler.get_scale()).astype(np.float16)
        unscaled = scaler.unscale_grads(scaled_grads)
        overflow = detect_overflow(unscaled)

        survived = np.sum(np.abs(grads.astype(np.float16).astype(np.float32)) > 0)
        total_nonzero = np.sum(np.abs(grads) > 0)
        survival_rate = survived / total_nonzero if total_nonzero > 0 else 1.0

        scales.append(scaler.get_scale())
        overflows.append(overflow)
        grad_survivals.append(survival_rate)
        scaler.update(overflow)

    overflow_steps = [i for i, o in enumerate(overflows) if o]
    print(f"  Total steps: {n_steps}")
    print(f"  Overflow events: {len(overflow_steps)} at steps {overflow_steps}")
    print(f"  Final scale: {scales[-1]:.0f}")
    print(f"  Scale range: [{min(scales):.0f}, {max(scales):.0f}]")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].plot(range(n_steps), scales, "-", color=COLORS["blue"], linewidth=1.5)
    for os_step in overflow_steps:
        axes[0, 0].axvline(os_step, color=COLORS["red"], alpha=0.5, linewidth=1)
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel("Loss Scale Factor")
    axes[0, 0].set_yscale("log", base=2)
    axes[0, 0].set_title("Dynamic Loss Scale Over Training\nRed lines = overflow events (scale halves)",
                         fontsize=10, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    window = 20
    smoothed = np.convolve(grad_survivals, np.ones(window)/window, mode="valid")
    axes[0, 1].plot(range(len(smoothed)), smoothed, "-", color=COLORS["green"], linewidth=1.5)
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel("Gradient Survival Rate")
    axes[0, 1].set_ylim(0.95, 1.005)
    axes[0, 1].set_title("Gradient Survival Rate (moving avg)\nFraction of gradients surviving FP16 conversion",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    test_grads = rng.randn(10000).astype(np.float32) * 0.0001
    scale_factors = [1, 10, 100, 1000, 10000, 100000]
    survival_rates = []
    for sf in scale_factors:
        scaled = (test_grads * sf).astype(np.float16)
        unscaled = scaled.astype(np.float32) / sf
        survived = np.sum(unscaled != 0)
        survival_rates.append(survived / len(test_grads))

    axes[0, 2].plot(scale_factors, survival_rates, "o-", color=COLORS["purple"], linewidth=2, markersize=8)
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_xlabel("Loss Scale Factor")
    axes[0, 2].set_ylabel("Gradient Survival Rate")
    axes[0, 2].set_title("Scale Factor vs Gradient Survival\nHigher scale = more gradients survive FP16",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3)

    grad_magnitudes = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    no_scale_survive = []
    with_scale_survive = []
    scale_for_test = 10000
    for gm in grad_magnitudes:
        g = np.float16(gm)
        no_scale_survive.append(0 if float(g) == 0 else 1)
        g_scaled = np.float16(gm * scale_for_test)
        with_scale_survive.append(0 if float(g_scaled) == 0 else 1)

    x = np.arange(len(grad_magnitudes))
    axes[1, 0].bar(x - 0.15, no_scale_survive, 0.3, label="No scaling", color=COLORS["red"], edgecolor="white")
    axes[1, 0].bar(x + 0.15, with_scale_survive, 0.3, label=f"Scale={scale_for_test}", color=COLORS["green"], edgecolor="white")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"{g:.0e}" for g in grad_magnitudes], fontsize=8, rotation=30)
    axes[1, 0].set_ylabel("Survives FP16? (1=yes)")
    axes[1, 0].set_title("Gradient Survival by Magnitude\nLoss scaling rescues small gradients",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    scaler2 = LossScaler(init_scale=2**15, growth_interval=10, growth_factor=2.0,
                         backoff_factor=0.5, max_scale=2**24)
    scales2 = [scaler2.get_scale()]
    overflow_pattern = [False]*9 + [True] + [False]*25 + [True] + [False]*50
    for has_of in overflow_pattern:
        scaler2.update(has_of)
        scales2.append(scaler2.get_scale())

    axes[1, 1].plot(range(len(scales2)), scales2, "o-", color=COLORS["blue"], linewidth=1.5, markersize=3)
    for i, has_of in enumerate(overflow_pattern):
        if has_of:
            axes[1, 1].axvline(i, color=COLORS["red"], alpha=0.7, linewidth=2, linestyle="--")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Scale Factor")
    axes[1, 1].set_yscale("log", base=2)
    axes[1, 1].set_title("Scale Factor State Machine\nGrows after N clean steps, halves on overflow",
                         fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    info_text = (
        "DYNAMIC LOSS SCALING\n"
        "====================\n\n"
        "Algorithm:\n"
        "  1. Start with large scale (2^15)\n"
        "  2. Multiply loss by scale\n"
        "  3. Compute FP16 gradients\n"
        "  4. Unscale: grads / scale\n"
        "  5. If overflow: halve scale,\n"
        "     skip weight update\n"
        "  6. After N clean steps:\n"
        "     double the scale\n\n"
        f"This run: {n_steps} steps\n"
        f"  Overflows: {len(overflow_steps)}\n"
        f"  Final scale: {scales[-1]:.0f}\n\n"
        "Why needed for FP16:\n"
        "  Small grads (< 5.96e-8)\n"
        "  underflow to zero.\n"
        "  Scaling shifts them into\n"
        "  representable range.\n\n"
        "NOT needed for BF16:\n"
        "  Same range as FP32."
    )
    axes[1, 2].text(0.05, 0.95, info_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Dynamic Loss Scaling: Keeping Gradients Alive in FP16",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_loss_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/05_loss_scaling.png")


# ---------------------------------------------------------------------------
# Example 6: Accumulation Error in Matmul
# ---------------------------------------------------------------------------
def example_6_accumulation_error():
    """Show how matmul accuracy degrades with FP16 accumulation vs FP32."""
    print("\n" + "=" * 60)
    print("Example 6: Accumulation Error in Matrix Multiply")
    print("=" * 60)

    rng = np.random.RandomState(SEED)
    k_values = [8, 16, 32, 64, 128, 256]
    errors_fp16_accum = []
    errors_fp32_accum = []

    for K in k_values:
        A = rng.randn(16, K).astype(np.float32) * 0.1
        B = rng.randn(K, 16).astype(np.float32) * 0.1
        fp32_ref = A @ B

        fp32_result = mixed_precision_matmul(A, B, np.float16, np.float32)
        fp16_result = mixed_precision_matmul(A, B, np.float16, np.float16)

        err32 = np.mean(np.abs(fp32_result.astype(np.float32) - fp32_ref)) / np.mean(np.abs(fp32_ref))
        err16 = np.mean(np.abs(fp16_result.astype(np.float32) - fp32_ref)) / np.mean(np.abs(fp32_ref))
        errors_fp32_accum.append(err32)
        errors_fp16_accum.append(err16)

        print(f"  K={K:>4}: FP32 accum err={err32:.6e}, FP16 accum err={err16:.6e}, ratio={err16/max(err32,1e-30):.1f}x")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].plot(k_values, errors_fp32_accum, "o-", color=COLORS["green"], linewidth=2, markersize=6,
                    label="FP16 compute, FP32 accum")
    axes[0, 0].plot(k_values, errors_fp16_accum, "s-", color=COLORS["red"], linewidth=2, markersize=6,
                    label="FP16 compute, FP16 accum")
    axes[0, 0].set_xlabel("Inner Dimension K")
    axes[0, 0].set_ylabel("Mean Relative Error vs FP32")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Matmul Error vs Inner Dimension\nFP16 accumulation error grows with K",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    ratios = [e16 / max(e32, 1e-30) for e16, e32 in zip(errors_fp16_accum, errors_fp32_accum)]
    axes[0, 1].bar(range(len(k_values)), ratios, color=COLORS["purple"], edgecolor="white")
    axes[0, 1].set_xticks(range(len(k_values)))
    axes[0, 1].set_xticklabels([str(k) for k in k_values])
    axes[0, 1].set_xlabel("Inner Dimension K")
    axes[0, 1].set_ylabel("Error Ratio (FP16 / FP32 accum)")
    axes[0, 1].set_title("Error Amplification from FP16 Accumulation\nHigher K = more accumulated rounding errors",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    K = 128
    A = rng.randn(32, K).astype(np.float32)
    B = rng.randn(K, 32).astype(np.float32)
    fp32_ref = A @ B
    fp32_accum = mixed_precision_matmul(A, B, np.float16, np.float32).astype(np.float32)
    fp16_accum = mixed_precision_matmul(A, B, np.float16, np.float16).astype(np.float32)

    err_map_32 = np.abs(fp32_accum - fp32_ref) / (np.abs(fp32_ref) + 1e-10)
    err_map_16 = np.abs(fp16_accum - fp32_ref) / (np.abs(fp32_ref) + 1e-10)

    im = axes[0, 2].imshow(err_map_16, cmap="Reds", aspect="auto")
    plt.colorbar(im, ax=axes[0, 2], label="Relative Error")
    axes[0, 2].set_xlabel("Column")
    axes[0, 2].set_ylabel("Row")
    axes[0, 2].set_title(f"Error Heatmap: FP16 Accum (K={K})\nElement-wise relative error vs FP32 baseline",
                         fontsize=10, fontweight="bold")

    n_values = [10, 50, 100, 500, 1000, 5000]
    fp16_sum_errors = []
    fp32_sum_errors = []
    for n in n_values:
        vals = rng.randn(n).astype(np.float32) * 0.01
        true_sum = np.sum(vals)

        fp16_sum = np.float16(0)
        for v in vals:
            fp16_sum = np.float16(float(fp16_sum) + float(np.float16(v)))

        fp32_accum_sum = np.float32(0)
        for v in vals:
            fp32_accum_sum += np.float32(np.float16(v))

        err_16 = abs(float(fp16_sum) - true_sum) / max(abs(true_sum), 1e-10)
        err_32 = abs(float(fp32_accum_sum) - true_sum) / max(abs(true_sum), 1e-10)
        fp16_sum_errors.append(err_16)
        fp32_sum_errors.append(err_32)

    axes[1, 0].plot(n_values, fp16_sum_errors, "o-", color=COLORS["red"], linewidth=2, label="FP16 accumulation")
    axes[1, 0].plot(n_values, fp32_sum_errors, "s-", color=COLORS["green"], linewidth=2, label="FP32 accumulation")
    axes[1, 0].set_xlabel("Number of Values Summed")
    axes[1, 0].set_ylabel("Relative Error vs True Sum")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Summation Error Grows with Count\nFP16 accumulation diverges; FP32 stays accurate",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    scale_factors_test = [0.01, 0.1, 1.0, 10.0]
    for sf in scale_factors_test:
        errs = []
        for K in k_values:
            A = rng.randn(16, K).astype(np.float32) * sf
            B = rng.randn(K, 16).astype(np.float32) * sf
            fp32_ref_local = A @ B
            fp16_result_local = mixed_precision_matmul(A, B, np.float16, np.float16)
            err = np.mean(np.abs(fp16_result_local.astype(np.float32) - fp32_ref_local)) / max(np.mean(np.abs(fp32_ref_local)), 1e-10)
            errs.append(err)
        axes[1, 1].plot(k_values, errs, "o-", linewidth=1.5, markersize=5, label=f"scale={sf}")

    axes[1, 1].set_xlabel("Inner Dimension K")
    axes[1, 1].set_ylabel("Mean Relative Error (FP16 accum)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("FP16 Accum Error at Different Scales\nLarger values = worse precision loss",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    info_text = (
        "ACCUMULATION PRECISION\n"
        "======================\n\n"
        "Tensor core behavior:\n"
        "  A, B in FP16 (or BF16)\n"
        "  Accumulator in FP32\n"
        "  Output cast back to FP16\n\n"
        "Why FP32 accumulation matters:\n"
        "  Each dot product sums K terms.\n"
        "  In FP16, each addition loses\n"
        "  ~0.1% precision. Over K=256\n"
        "  terms, errors compound.\n\n"
        "  FP32 accumulator has 2^23\n"
        "  mantissa values vs 2^10 for\n"
        "  FP16 -- 8192x more precise.\n\n"
        "Real hardware (A100/H100):\n"
        "  Always uses FP32 accumulators\n"
        "  for FP16/BF16 tensor core ops.\n"
        "  Pure FP16 accum is a pathological\n"
        "  case shown here for education."
    )
    axes[1, 2].text(0.05, 0.95, info_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Accumulation Precision: Why Tensor Cores Use FP32 Accumulators",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_accumulation_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/06_accumulation_error.png")


# ---------------------------------------------------------------------------
# Example 7: Softmax Stability
# ---------------------------------------------------------------------------
def example_7_softmax_stability():
    """Compare naive vs stable softmax in FP16 for large logits."""
    print("\n" + "=" * 60)
    print("Example 7: Softmax Stability in FP16")
    print("=" * 60)

    def naive_softmax_fp16(x):
        x = x.astype(np.float16)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    test_logits = [
        ("Small [1,2,3]", np.array([1.0, 2.0, 3.0])),
        ("Medium [5,10,15]", np.array([5.0, 10.0, 15.0])),
        ("Large [10,10,10]", np.array([10.0, 10.0, 10.0])),
        ("Overflow [20,20,20]", np.array([20.0, 20.0, 20.0])),
        ("Extreme [50,50,50]", np.array([50.0, 50.0, 50.0])),
    ]

    print(f"\n  {'Logits':>24} {'Naive FP16':>30} {'Stable FP16':>30} {'FP32 ref':>30}")
    print(f"  {'-'*118}")
    naive_results = []
    stable_results = []
    for name, logits in test_logits:
        with np.errstate(over="ignore", invalid="ignore"):
            naive = naive_softmax_fp16(logits)
        stable_val = stable_softmax(logits, np.float16)
        fp32_ref = stable_softmax(logits, np.float32)

        naive_str = "NaN/Inf!" if np.any(np.isnan(naive)) or np.any(np.isinf(naive)) else \
                    str(np.round(naive.astype(np.float32), 4))
        stable_str = str(np.round(stable_val.astype(np.float32), 4))
        ref_str = str(np.round(fp32_ref.astype(np.float32), 4))
        print(f"  {name:>24} {naive_str:>30} {stable_str:>30} {ref_str:>30}")
        naive_results.append(naive)
        stable_results.append(stable_val)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    max_logit_values = np.arange(1, 30, 0.5)
    naive_valid = []
    stable_valid = []
    for ml in max_logit_values:
        logits = np.array([ml, ml, ml], dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            naive = naive_softmax_fp16(logits)
        naive_ok = not (np.any(np.isnan(naive)) or np.any(np.isinf(naive)))
        stable_val = stable_softmax(logits, np.float16)
        stable_ok = not (np.any(np.isnan(stable_val)) or np.any(np.isinf(stable_val)))
        naive_valid.append(1.0 if naive_ok else 0.0)
        stable_valid.append(1.0 if stable_ok else 0.0)

    axes[0, 0].plot(max_logit_values, naive_valid, "o-", color=COLORS["red"], linewidth=2,
                    markersize=4, label="Naive softmax")
    axes[0, 0].plot(max_logit_values, stable_valid, "s-", color=COLORS["green"], linewidth=2,
                    markersize=4, label="Stable softmax")
    axes[0, 0].set_xlabel("Max Logit Value")
    axes[0, 0].set_ylabel("Valid Output (1=yes, 0=NaN/Inf)")
    axes[0, 0].set_title("Softmax Validity vs Logit Magnitude\nNaive FP16 breaks at ~11 (exp overflow)",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    exp_values = np.arange(0, 25, 0.5)
    fp16_exp = []
    for v in exp_values:
        with np.errstate(over="ignore"):
            result = np.exp(np.float16(v))
        fp16_exp.append(float(result))

    axes[0, 1].plot(exp_values, fp16_exp, "o-", color=COLORS["coral"], linewidth=1.5, markersize=3)
    axes[0, 1].axhline(65504, color=COLORS["red"], linestyle="--", linewidth=2, label="FP16 max (65504)")
    axes[0, 1].axvline(11.09, color="gray", linestyle=":", linewidth=2, label="x~11 (exp overflow)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("exp(x) in FP16")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("exp(x) in FP16\nOverflows to inf at x > 11.09",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    rng = np.random.RandomState(SEED)
    logits = rng.randn(100).astype(np.float32)
    sm_fp32 = stable_softmax(logits.reshape(1, -1), np.float32).ravel()
    sm_fp16 = stable_softmax(logits.reshape(1, -1), np.float16).astype(np.float32).ravel()
    sm_error = np.abs(sm_fp32 - sm_fp16)

    sorted_idx = np.argsort(sm_fp32)[::-1]
    axes[0, 2].bar(range(30), sm_fp32[sorted_idx[:30]], alpha=0.6, color=COLORS["blue"], label="FP32")
    axes[0, 2].bar(range(30), sm_fp16[sorted_idx[:30]], alpha=0.6, color=COLORS["orange"], label="FP16")
    axes[0, 2].set_xlabel("Class (sorted by probability)")
    axes[0, 2].set_ylabel("Probability")
    axes[0, 2].set_title("Softmax Output Comparison (top 30 classes)\nFP16 closely matches FP32 with stable impl",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    sum_fp32 = float(np.sum(sm_fp32))
    sum_fp16 = float(np.sum(sm_fp16))
    axes[1, 0].bar(["FP32", "FP16 (stable)"], [sum_fp32, sum_fp16],
                   color=[COLORS["blue"], COLORS["orange"]], edgecolor="white", width=0.5)
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_ylabel("Sum of Probabilities")
    axes[1, 0].set_ylim(0.99, 1.01)
    axes[1, 0].set_title(f"Probability Sum Check\nFP32: {sum_fp32:.6f}, FP16: {sum_fp16:.6f}",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    n_classes_list = [10, 50, 100, 500, 1000]
    max_errors = []
    mean_errors = []
    for nc in n_classes_list:
        logits_test = rng.randn(nc).astype(np.float32)
        sm32 = stable_softmax(logits_test.reshape(1, -1), np.float32).ravel()
        sm16 = stable_softmax(logits_test.reshape(1, -1), np.float16).astype(np.float32).ravel()
        err = np.abs(sm32 - sm16) / (sm32 + 1e-10)
        max_errors.append(np.max(err))
        mean_errors.append(np.mean(err))

    axes[1, 1].plot(n_classes_list, max_errors, "o-", color=COLORS["red"], linewidth=2, label="Max rel error")
    axes[1, 1].plot(n_classes_list, mean_errors, "s-", color=COLORS["green"], linewidth=2, label="Mean rel error")
    axes[1, 1].set_xlabel("Number of Classes")
    axes[1, 1].set_ylabel("Relative Error (FP16 vs FP32)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Softmax Error vs Vocabulary Size\nMore classes = slightly more accumulation error",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    info_text = (
        "SOFTMAX STABILITY\n"
        "=================\n\n"
        "The problem:\n"
        "  softmax(x)_i = exp(x_i) / sum\n"
        "  exp(x) overflows in FP16 for\n"
        "  x > 11 (exp(11) ~ 59874)\n\n"
        "The solution:\n"
        "  Subtract max before exp:\n"
        "  softmax(x-max(x))\n"
        "  Now max exponent = 0,\n"
        "  exp(0) = 1 (no overflow)\n\n"
        "Additional trick:\n"
        "  Accumulate denominator sum\n"
        "  in FP32 for precision.\n\n"
        f"Sum check (100 classes):\n"
        f"  FP32: {sum_fp32:.8f}\n"
        f"  FP16: {sum_fp16:.8f}\n\n"
        "In practice, ALL modern\n"
        "frameworks use stable softmax."
    )
    axes[1, 2].text(0.05, 0.95, info_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Softmax Numerical Stability: Naive vs Stable in FP16",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_softmax_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/07_softmax_stability.png")


# ---------------------------------------------------------------------------
# Example 8: Memory Savings
# ---------------------------------------------------------------------------
def example_8_memory_savings():
    """Model weight + KV cache memory at different precisions for 7B/13B/70B."""
    print("\n" + "=" * 60)
    print("Example 8: Memory Savings at Scale")
    print("=" * 60)

    models = {
        "7B": dict(params=7e9, n_layers=32, n_heads=32, d_head=128),
        "13B": dict(params=13e9, n_layers=40, n_heads=40, d_head=128),
        "70B": dict(params=70e9, n_layers=80, n_heads=64, d_head=128),
    }
    precisions = [np.float32, np.float16, "bfloat16", "int8", "int4"]
    precision_labels = ["FP32", "FP16", "BF16", "INT8", "INT4"]
    precision_colors = [COLORS["blue"], COLORS["coral"], COLORS["orange"], COLORS["green"], COLORS["purple"]]

    print(f"\n  Model Weight Memory (GB):")
    print(f"  {'Model':>6} {'FP32':>8} {'FP16':>8} {'BF16':>8} {'INT8':>8} {'INT4':>8}")
    print(f"  {'-'*48}")
    weight_data = {}
    for mname, mcfg in models.items():
        row = []
        for prec in precisions:
            mem = model_memory_usage(int(mcfg["params"]), prec)
            row.append(mem["gigabytes"])
        weight_data[mname] = row
        print(f"  {mname:>6} {row[0]:>8.1f} {row[1]:>8.1f} {row[2]:>8.1f} {row[3]:>8.1f} {row[4]:>8.1f}")

    print(f"\n  KV Cache Memory at 4K context (GB):")
    print(f"  {'Model':>6} {'FP32':>8} {'FP16':>8}")
    print(f"  {'-'*24}")
    kv_data = {}
    for mname, mcfg in models.items():
        fp32_kv = kv_cache_memory(mcfg["n_layers"], mcfg["n_heads"], mcfg["d_head"], 4096, np.float32)
        fp16_kv = kv_cache_memory(mcfg["n_layers"], mcfg["n_heads"], mcfg["d_head"], 4096, np.float16)
        kv_data[mname] = (fp32_kv["gigabytes"], fp16_kv["gigabytes"])
        print(f"  {mname:>6} {fp32_kv['gigabytes']:>8.1f} {fp16_kv['gigabytes']:>8.1f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    x = np.arange(len(models))
    width = 0.15
    for pi, (plabel, pcolor) in enumerate(zip(precision_labels, precision_colors)):
        vals = [weight_data[m][pi] for m in models]
        axes[0, 0].bar(x + pi * width - 2 * width, vals, width, label=plabel,
                       color=pcolor, edgecolor="white")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(list(models.keys()))
    axes[0, 0].set_ylabel("Memory (GB)")
    axes[0, 0].set_title("Model Weight Memory by Precision\nHalf precision = half memory",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=8, ncol=3)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    gpu_limits = {"A100 40GB": 40, "A100 80GB": 80, "H100 80GB": 80}
    model_names = list(models.keys())
    for pi, plabel in enumerate(precision_labels):
        vals = [weight_data[m][pi] for m in models]
        axes[0, 1].plot(model_names, vals, "o-", linewidth=2, markersize=6, label=plabel,
                        color=precision_colors[pi])
    axes[0, 1].axhline(40, color="gray", linestyle="--", linewidth=1.5, label="A100 40GB")
    axes[0, 1].axhline(80, color="black", linestyle="--", linewidth=1.5, label="A100 80GB")
    axes[0, 1].set_ylabel("Memory (GB)")
    axes[0, 1].set_title("Fits on GPU?\nPrecision determines hardware requirements",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[0, 1].grid(True, alpha=0.3)

    seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
    for mname, mcfg in models.items():
        kv_sizes = []
        for sl in seq_lens:
            kv = kv_cache_memory(mcfg["n_layers"], mcfg["n_heads"], mcfg["d_head"], sl, np.float16)
            kv_sizes.append(kv["gigabytes"])
        axes[0, 2].plot(seq_lens, kv_sizes, "o-", linewidth=2, markersize=5, label=mname)
    axes[0, 2].set_xlabel("Sequence Length")
    axes[0, 2].set_ylabel("KV Cache (GB, FP16)")
    axes[0, 2].set_xscale("log", base=2)
    axes[0, 2].set_title("KV Cache Size vs Sequence Length\nLinear growth -- dominates at long contexts",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    for idx, (mname, mcfg) in enumerate(models.items()):
        ax = axes[1, idx]
        weight_fp16 = weight_data[mname][1]
        kv_fp16 = kv_data[mname][1]
        act_est = 1.0
        sizes = [weight_fp16, kv_fp16, act_est]
        labels_pie = [f"Weights\n{weight_fp16:.1f}GB", f"KV Cache\n{kv_fp16:.1f}GB",
                      f"Activations\n~{act_est:.0f}GB"]
        colors_pie = [COLORS["blue"], COLORS["red"], COLORS["orange"]]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels_pie, colors=colors_pie,
            autopct=lambda p: f"{p:.0f}%" if p > 2 else "",
            pctdistance=0.75, startangle=90
        )
        for t in autotexts:
            t.set_fontsize(8)
        for t in texts:
            t.set_fontsize(8)
        total = sum(sizes)
        ax.set_title(f"{mname} Model (FP16, 4K ctx)\nTotal: {total:.1f} GB",
                     fontsize=10, fontweight="bold")

    fig.suptitle("Memory Analysis: Weights + KV Cache at Different Precisions",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "08_memory_savings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/08_memory_savings.png")


# ---------------------------------------------------------------------------
# Example 9: Training Comparison
# ---------------------------------------------------------------------------
def example_9_training_comparison():
    """FP32 vs FP16+scaling vs BF16 on a toy regression problem."""
    print("\n" + "=" * 60)
    print("Example 9: Training Comparison (FP32 vs FP16+scaling vs BF16)")
    print("=" * 60)

    rng = np.random.RandomState(SEED)
    n_samples = 200
    d_in, d_out = 8, 4
    X = rng.randn(n_samples, d_in).astype(np.float32) * 0.5
    W_true = rng.randn(d_in, d_out).astype(np.float32) * 0.3
    Y = X @ W_true + rng.randn(n_samples, d_out).astype(np.float32) * 0.05

    n_epochs = 150
    lr = 0.01
    batch_size = 32

    def train_fp32():
        W = rng.randn(d_in, d_out).astype(np.float32) * 0.1
        losses = []
        for epoch in range(n_epochs):
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = X[idx], Y[idx]
                pred = xb @ W
                diff = pred - yb
                loss = np.mean(diff ** 2)
                epoch_loss += loss
                grad = (2.0 / len(idx)) * xb.T @ diff
                W -= lr * grad
            losses.append(epoch_loss / (n_samples // batch_size))
        return losses, W

    def train_fp16_scaled():
        W_master = rng.randn(d_in, d_out).astype(np.float32) * 0.1
        scaler = LossScaler(init_scale=2**10, growth_interval=50)
        losses = []
        for epoch in range(n_epochs):
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = X[idx], Y[idx]

                xb_fp16 = xb.astype(np.float16)
                W_fp16 = W_master.astype(np.float16)
                pred = (xb_fp16 @ W_fp16).astype(np.float32)
                diff = pred - yb
                loss = np.mean(diff ** 2)
                epoch_loss += loss

                grad = (2.0 / len(idx)) * xb.T @ diff
                scaled_grad = (grad * scaler.get_scale()).astype(np.float16)
                unscaled_grad = scaler.unscale_grads(scaled_grad)
                overflow = detect_overflow(unscaled_grad)
                scaler.update(overflow)

                if not overflow:
                    W_master -= lr * unscaled_grad
            losses.append(epoch_loss / (n_samples // batch_size))
        return losses, W_master

    def train_bf16():
        W_master = rng.randn(d_in, d_out).astype(np.float32) * 0.1
        losses = []
        for epoch in range(n_epochs):
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = X[idx], Y[idx]

                xb_bf16 = bf16_simulate(xb)
                W_bf16 = bf16_simulate(W_master)
                pred = (xb_bf16 @ W_bf16).astype(np.float32)
                diff = pred - yb
                loss = np.mean(diff ** 2)
                epoch_loss += loss

                grad = (2.0 / len(idx)) * xb.T @ diff
                W_master -= lr * grad
            losses.append(epoch_loss / (n_samples // batch_size))
        return losses, W_master

    def train_fp16_no_scaling():
        W_master = rng.randn(d_in, d_out).astype(np.float32) * 0.1
        losses = []
        for epoch in range(n_epochs):
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = X[idx], Y[idx]

                xb_fp16 = xb.astype(np.float16)
                W_fp16 = W_master.astype(np.float16)
                pred = (xb_fp16 @ W_fp16).astype(np.float32)
                diff = pred - yb
                loss = np.mean(diff ** 2)
                epoch_loss += loss

                grad = (2.0 / len(idx)) * xb.T @ diff
                grad_fp16 = grad.astype(np.float16).astype(np.float32)
                W_master -= lr * grad_fp16
            losses.append(epoch_loss / (n_samples // batch_size))
        return losses, W_master

    rng = np.random.RandomState(SEED)
    losses_fp32, W_fp32 = train_fp32()
    rng = np.random.RandomState(SEED)
    losses_fp16s, W_fp16s = train_fp16_scaled()
    rng = np.random.RandomState(SEED)
    losses_bf16, W_bf16 = train_bf16()
    rng = np.random.RandomState(SEED)
    losses_fp16_ns, W_fp16_ns = train_fp16_no_scaling()

    print(f"\n  Final losses:")
    print(f"    FP32:              {losses_fp32[-1]:.6f}")
    print(f"    FP16 + scaling:    {losses_fp16s[-1]:.6f}")
    print(f"    BF16:              {losses_bf16[-1]:.6f}")
    print(f"    FP16 (no scaling): {losses_fp16_ns[-1]:.6f}")

    w_err_fp16s = np.mean(np.abs(W_fp16s - W_fp32)) / np.mean(np.abs(W_fp32))
    w_err_bf16 = np.mean(np.abs(W_bf16 - W_fp32)) / np.mean(np.abs(W_fp32))
    w_err_fp16_ns = np.mean(np.abs(W_fp16_ns - W_fp32)) / np.mean(np.abs(W_fp32))
    print(f"\n  Weight error vs FP32:")
    print(f"    FP16 + scaling:    {w_err_fp16s:.6f}")
    print(f"    BF16:              {w_err_bf16:.6f}")
    print(f"    FP16 (no scaling): {w_err_fp16_ns:.6f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].plot(losses_fp32, "-", color=COLORS["blue"], linewidth=2, label="FP32")
    axes[0, 0].plot(losses_fp16s, "-", color=COLORS["green"], linewidth=2, label="FP16 + scaling")
    axes[0, 0].plot(losses_bf16, "-", color=COLORS["orange"], linewidth=2, label="BF16")
    axes[0, 0].plot(losses_fp16_ns, "--", color=COLORS["red"], linewidth=2, label="FP16 (no scaling)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].set_title("Training Loss Curves\nAll methods converge; FP16 w/o scaling may lag",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(losses_fp32, "-", color=COLORS["blue"], linewidth=2, label="FP32")
    axes[0, 1].plot(losses_fp16s, "-", color=COLORS["green"], linewidth=2, label="FP16 + scaling")
    axes[0, 1].plot(losses_bf16, "-", color=COLORS["orange"], linewidth=2, label="BF16")
    axes[0, 1].plot(losses_fp16_ns, "--", color=COLORS["red"], linewidth=2, label="FP16 (no scaling)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MSE Loss (log)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Training Loss (Log Scale)\nDifferences more visible at convergence",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    methods = ["FP32", "FP16+scale", "BF16", "FP16 raw"]
    final_losses = [losses_fp32[-1], losses_fp16s[-1], losses_bf16[-1], losses_fp16_ns[-1]]
    bar_colors_final = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    axes[0, 2].bar(methods, final_losses, color=bar_colors_final, edgecolor="white", width=0.6)
    for i, fl in enumerate(final_losses):
        axes[0, 2].text(i, fl + 0.001, f"{fl:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0, 2].set_ylabel("Final MSE Loss")
    axes[0, 2].set_title("Final Loss Comparison\nAll within acceptable range for this toy problem",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    weight_errors = [0, w_err_fp16s, w_err_bf16, w_err_fp16_ns]
    axes[1, 0].bar(methods, weight_errors, color=bar_colors_final, edgecolor="white", width=0.6)
    for i, we in enumerate(weight_errors):
        axes[1, 0].text(i, we + 0.0005, f"{we:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("Mean Relative Weight Error vs FP32")
    axes[1, 0].set_title("Learned Weight Error vs FP32 Baseline\nHow close are the final weights?",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    W_diff_fp16s = np.abs(W_fp16s - W_fp32)
    W_diff_bf16 = np.abs(W_bf16 - W_fp32)
    axes[1, 1].imshow(np.concatenate([W_diff_fp16s, np.zeros((d_in, 1)), W_diff_bf16], axis=1),
                      cmap="Reds", aspect="auto")
    axes[1, 1].set_xlabel("Output dimension")
    axes[1, 1].set_ylabel("Input dimension")
    axes[1, 1].axvline(d_out + 0.5, color="white", linewidth=3)
    axes[1, 1].text(d_out / 2, -0.8, "FP16+scale", ha="center", fontsize=9, fontweight="bold")
    axes[1, 1].text(d_out + 1 + d_out / 2, -0.8, "BF16", ha="center", fontsize=9, fontweight="bold")
    axes[1, 1].set_title("Weight Error Heatmap vs FP32\nBrighter = larger deviation from FP32 weights",
                         fontsize=10, fontweight="bold")

    axes[1, 2].axis("off")
    info_text = (
        "TRAINING COMPARISON\n"
        "===================\n\n"
        f"Task: Linear regression\n"
        f"  {n_samples} samples, d_in={d_in}\n"
        f"  {n_epochs} epochs, lr={lr}\n\n"
        "Results:\n"
        f"  FP32 loss:       {losses_fp32[-1]:.6f}\n"
        f"  FP16+scale loss: {losses_fp16s[-1]:.6f}\n"
        f"  BF16 loss:       {losses_bf16[-1]:.6f}\n"
        f"  FP16 raw loss:   {losses_fp16_ns[-1]:.6f}\n\n"
        "Key observations:\n"
        "  1. FP16 + loss scaling matches\n"
        "     FP32 closely\n"
        "  2. BF16 works without scaling\n"
        "     (same exponent range as FP32)\n"
        "  3. FP16 without scaling still\n"
        "     converges here (toy problem)\n"
        "     but may fail on real nets\n"
        "     with smaller gradients"
    )
    axes[1, 2].text(0.05, 0.95, info_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Training Comparison: FP32 vs FP16+Scaling vs BF16",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "09_training_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/09_training_comparison.png")


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
        # -- Title page --
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.78, "Mixed Precision", fontsize=28, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.68, "Why Precision Is the Key to Memory Bandwidth,\n"
                "and Why Memory Bandwidth Is the Key to Inference Speed",
                fontsize=13, ha="center", va="center", transform=ax.transAxes, color="gray")
        info_text = (
            "Mixed precision computing uses lower-precision floating point formats\n"
            "(FP16, BF16) for most operations while strategically keeping certain\n"
            "values in higher precision (FP32). This roughly doubles throughput\n"
            "and halves memory usage on modern GPUs.\n\n"
            "This demo covers:\n"
            "  1. Bit layout visualization (FP32, FP16, BF16)\n"
            "  2. Precision loss analysis across value ranges\n"
            "  3. Overflow/underflow boundaries\n"
            "  4. Consecutive value gap (spacing between representable numbers)\n"
            "  5. Dynamic loss scaling dynamics\n"
            "  6. Matrix multiply accumulation error\n"
            "  7. Softmax numerical stability\n"
            "  8. Memory savings at model scale (7B/13B/70B)\n"
            "  9. Training comparison (FP32 vs FP16+scaling vs BF16)\n\n"
            f"Random seed: {SEED}\n"
            f"Number of visualizations: {len(viz_files)}\n"
            "Examples: 9"
        )
        ax.text(0.5, 0.30, info_text, fontsize=11, ha="center", va="center",
                transform=ax.transAxes, linespacing=1.6)
        ax.text(0.5, 0.06, "Generated by demo.py", fontsize=10, ha="center",
                va="center", transform=ax.transAxes, style="italic", color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        # -- Math page 1: Format Anatomy --
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.96, "Mathematical Foundation: Floating Point Formats",
                fontsize=20, fontweight="bold", ha="center", va="top", transform=ax.transAxes)

        y = 0.88
        dy = 0.042

        ax.text(0.05, y, "IEEE 754 Floating Point Representation", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y, r"Every floating point number is encoded as:",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{value} = (-1)^{s} \times 2^{e - \mathrm{bias}} \times (1 + m)$",
                fontsize=12, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$s$: sign bit, $e$: exponent field, $m$: mantissa (fraction), bias: exponent offset",
                fontsize=10, transform=ax.transAxes)

        y -= dy * 1.4
        ax.text(0.05, y, "FP32 (Single Precision) -- 32 bits", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$[1_{\mathrm{sign}}][8_{\mathrm{exp}}][23_{\mathrm{mant}}]$"
                r"$\quad \mathrm{bias}=127, \; \max \approx 3.4 \times 10^{38}, \; \sim 7.2$ decimal digits",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.4
        ax.text(0.05, y, "FP16 (Half Precision) -- 16 bits", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$[1_{\mathrm{sign}}][5_{\mathrm{exp}}][10_{\mathrm{mant}}]$"
                r"$\quad \mathrm{bias}=15, \; \max = 65504, \; \sim 3.3$ decimal digits",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.4
        ax.text(0.05, y, "BF16 (Brain Float) -- 16 bits", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$[1_{\mathrm{sign}}][8_{\mathrm{exp}}][7_{\mathrm{mant}}]$"
                r"$\quad \mathrm{bias}=127, \; \max \approx 3.4 \times 10^{38}, \; \sim 2.1$ decimal digits",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.4
        ax.text(0.05, y, "Consecutive Value Gap", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{gap}(v) = 2^{\mathrm{exponent} - m_{\mathrm{bits}}}$"
                r"$\quad$ where $m_{\mathrm{bits}}$ is the number of mantissa bits",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"At $v=1$: FP32 gap $= 2^{-23} \approx 1.19 \times 10^{-7}$, "
                r"FP16 gap $= 2^{-10} \approx 9.77 \times 10^{-4}$, "
                r"BF16 gap $= 2^{-7} \approx 7.81 \times 10^{-3}$",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"At $v=1000$: FP16 gap $= 0.5$, BF16 gap $= 4.0$ "
                r"$\Rightarrow$ gradient updates $< \mathrm{gap}$ are effectively zero",
                fontsize=10, transform=ax.transAxes)

        y -= dy * 1.4
        ax.text(0.05, y, "Precision vs Range Summary", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"FP16: high precision ($2^{10}$ mantissa values), limited range ($\leq 65504$)",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"BF16: low precision ($2^{7}$ mantissa values), full range (same as FP32)",
                fontsize=10, transform=ax.transAxes)

        pdf.savefig(fig)
        plt.close(fig)

        # -- Math page 2: Loss Scaling and Stability --
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.96, "Mathematical Foundation: Loss Scaling and Numerical Stability",
                fontsize=20, fontweight="bold", ha="center", va="top", transform=ax.transAxes)

        y = 0.88
        dy = 0.042

        ax.text(0.05, y, "Loss Scaling for FP16 Training", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y, r"Small gradients underflow to zero in FP16. Scale the loss before backprop:",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$g_{\mathrm{scaled}} = s \cdot \frac{\partial L}{\partial w}$"
                r"$\quad \Rightarrow \quad g = \frac{g_{\mathrm{scaled}}}{s}$"
                r"$\quad$ where $s$ is the scale factor",
                fontsize=12, transform=ax.transAxes)
        y -= dy * 1.2
        ax.text(0.07, y, "Dynamic scaling algorithm:", fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"1. Start with large $s$ (e.g., $2^{16}$). "
                r"2. If overflow (inf/NaN in grads): $s \leftarrow s/2$, skip update.",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"3. After $N$ clean steps: $s \leftarrow 2s$. Typical $N = 2000$.",
                fontsize=10, transform=ax.transAxes)

        y -= dy * 1.4
        ax.text(0.05, y, "Softmax Numerical Stability", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$"
                r"$\quad$ Problem: $e^x$ overflows for $x > 11$ in FP16 ($e^{11} \approx 59874 < 65504$)",
                fontsize=11, transform=ax.transAxes)
        y -= dy * 1.2
        ax.text(0.10, y,
                r"Solution: $\mathrm{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$"
                r"$\quad$ (max exponent is now 0, no overflow)",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.4
        ax.text(0.05, y, "Tensor Core Mixed Precision Matmul", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$C = A \cdot B \quad$ where $A, B \in \mathrm{FP16}$, "
                r"accumulator $\in \mathrm{FP32}$, output $C \in \mathrm{FP16}$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Pure FP16 accum error $\propto \sqrt{K}$ (inner dim); FP32 accum eliminates this",
                fontsize=10, transform=ax.transAxes)

        y -= dy * 1.4
        ax.text(0.05, y, "Memory Bandwidth Analysis", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{Arithmetic\ Intensity} = \frac{\mathrm{FLOPs}}{\mathrm{Bytes\ loaded}}$"
                r"$\quad$ GPU is compute-bound when intensity $> \frac{\mathrm{peak\ FLOPS}}{\mathrm{BW}}$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"A100: $\frac{312\ \mathrm{TFLOPS}}{2\ \mathrm{TB/s}} = 156\ \mathrm{FLOP/byte}$"
                r"$\quad$ Decode inference intensity $\approx 1\ \mathrm{FLOP/byte}$ (memory-bound!)",
                fontsize=10, transform=ax.transAxes)
        y -= dy * 1.2
        ax.text(0.10, y,
                r"Half precision $=$ half bytes $=$ $2\times$ throughput for memory-bound ops",
                fontsize=10, transform=ax.transAxes)

        y -= dy * 1.4
        ax.text(0.05, y, "Model Memory Formulas", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Weights: $\mathrm{mem} = n_{\mathrm{params}} \times \mathrm{bytes\_per\_element}$"
                r"$\quad$ 7B in FP16: $7 \times 10^9 \times 2 = 14$ GB",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"KV cache: $\mathrm{mem} = 2 \times n_{\mathrm{layers}} \times n_{\mathrm{heads}}"
                r" \times d_{\mathrm{head}} \times \mathrm{seq\_len} \times \mathrm{bytes}$",
                fontsize=10, transform=ax.transAxes)

        pdf.savefig(fig)
        plt.close(fig)

        # -- Summary page --
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.94, "Summary of Findings", fontsize=20, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)

        summary_items = [
            "1. Bit Layout: FP32 uses 8 exponent + 23 mantissa bits. FP16 uses 5+10 (limited range,",
            "   decent precision). BF16 uses 8+7 (full FP32 range, less precision). BF16 is simply",
            "   truncated FP32 -- drop the lower 16 bits.",
            "",
            "2. Precision Loss: BF16 has ~3x higher mean relative error than FP16, but handles all",
            "   magnitudes that FP32 can. For neural net weights (clustered near 0), both are adequate.",
            "",
            "3. Overflow/Underflow: FP16 overflows at 65504 and underflows below ~5.96e-8. BF16 has",
            "   the same range as FP32, making overflow/underflow a non-issue. This is why BF16 does",
            "   not need loss scaling during training.",
            "",
            "4. Value Gaps: At value 1000, consecutive FP16 numbers are 0.5 apart and BF16 numbers",
            "   are 4.0 apart. Gradient updates smaller than this gap vanish -- the staircase effect.",
            "",
            "5. Loss Scaling: Dynamic loss scaling grows the scale after N clean steps and halves on",
            "   overflow. This shifts small gradients into FP16's representable range, preventing the",
            "   underflow-to-zero problem that stalls training.",
            "",
            "6. Accumulation: Pure FP16 matmul accumulation error grows with inner dimension K.",
            "   FP32 accumulators (as in tensor cores) eliminate this. Real hardware always uses FP32",
            "   accumulation for FP16/BF16 compute.",
            "",
            "7. Softmax: exp(x) overflows in FP16 for x > 11. Stable softmax (subtract max) plus",
            "   FP32 denominator accumulation solves this. All modern frameworks do this by default.",
            "",
            "8. Memory: A 70B model in FP32 needs 280 GB (impossible on any single GPU). In FP16/BF16",
            "   it's 140 GB (2x H100). In INT4 it's 35 GB (fits on A100 40GB). Precision determines",
            "   what hardware you need.",
            "",
            "9. Training: FP32, FP16+scaling, and BF16 all converge to similar loss on toy problems.",
            "   BF16 is simplest (no scaling needed). FP16 without scaling works for easy problems",
            "   but fails when gradients are small.",
        ]
        summary_text = "\n".join(summary_items)
        ax.text(0.06, 0.86, summary_text, fontsize=9.5, ha="left", va="top",
                transform=ax.transAxes, family="monospace", linespacing=1.3)
        pdf.savefig(fig)
        plt.close(fig)

        # -- Visualization pages --
        titles = {
            "01_bit_layout.png": "Example 1: Bit Layout Visualization",
            "02_precision_loss.png": "Example 2: Precision Loss Analysis",
            "03_overflow_underflow.png": "Example 3: Overflow/Underflow Boundaries",
            "04_consecutive_gap.png": "Example 4: Consecutive Value Gap",
            "05_loss_scaling.png": "Example 5: Loss Scaling Dynamics",
            "06_accumulation_error.png": "Example 6: Accumulation Error in Matmul",
            "07_softmax_stability.png": "Example 7: Softmax Stability in FP16",
            "08_memory_savings.png": "Example 8: Memory Savings at Scale",
            "09_training_comparison.png": "Example 9: Training Comparison",
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
    print("Mixed Precision Demo")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print()

    example_1_bit_layout()
    example_2_precision_loss()
    example_3_overflow_underflow()
    example_4_consecutive_gap()
    example_5_loss_scaling()
    example_6_accumulation_error()
    example_7_softmax_stability()
    example_8_memory_savings()
    example_9_training_comparison()
    generate_pdf_report()

    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print(f"Visualizations: {VIZ_DIR}/")
    print(f"Report: {Path(__file__).parent / 'report.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
