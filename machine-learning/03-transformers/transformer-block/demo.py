"""
Transformer Block Demo -- Forward pass walkthrough, parameter analysis, SwiGLU gating,
residual connections, FLOPs breakdown, and causal masking with position sensitivity.

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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from implementation import (
    TransformerBlock,
    SwiGLUFFN,
    count_parameters,
    count_flops,
    create_causal_mask,
    _xavier,
    _stable_sigmoid,
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

MODEL_CONFIGS = {
    "Llama 2 7B": {"d_model": 4096, "num_heads": 32, "num_kv_heads": 32, "d_ff": 11008, "layers": 32},
    "Llama 2 70B": {"d_model": 8192, "num_heads": 64, "num_kv_heads": 8, "d_ff": 28672, "layers": 80},
    "Llama 3 8B": {"d_model": 4096, "num_heads": 32, "num_kv_heads": 8, "d_ff": 14336, "layers": 32},
    "Mistral 7B": {"d_model": 4096, "num_heads": 32, "num_kv_heads": 8, "d_ff": 14336, "layers": 32},
}


# ---------------------------------------------------------------------------
# Example 1: Full Forward Pass Walkthrough
# ---------------------------------------------------------------------------
def example_1_forward_pass():
    """Create a small transformer block and trace shapes at every step."""
    print("=" * 60)
    print("Example 1: Full Forward Pass Walkthrough")
    print("=" * 60)

    d_model, num_heads, num_kv_heads, d_ff = 64, 4, 2, 172
    B, L = 2, 8
    max_seq_len = 32

    np.random.seed(SEED)
    block = TransformerBlock(d_model, num_heads, num_kv_heads, d_ff, max_seq_len)
    x = np.random.randn(B, L, d_model)

    print(f"\n  Config: d_model={d_model}, h={num_heads}, h_kv={num_kv_heads}, d_ff={d_ff}")
    print(f"  Input:  x.shape = {x.shape}")

    output = block.forward(x)
    c = block._cache

    steps = [
        ("Input x", x),
        ("After RMSNorm_1 (x_norm)", c["x_norm"]),
        ("Q projection", c["Q"]),
        ("K projection", c["K"]),
        ("V projection", c["V"]),
        ("Q after RoPE (Q_rot)", c["Q_rot"]),
        ("K after RoPE (K_rot)", c["K_rot"]),
        ("K expanded (K_exp)", c["K_exp"]),
        ("V expanded (V_exp)", c["V_exp"]),
        ("Attention weights (A)", c["A"]),
        ("Attention output", c["attn_output"]),
        ("Concat (head merge)", c["concat"]),
        ("Attn out (after W_O)", c["attn_out"]),
        ("h = x + attn_out", c["h"]),
        ("After RMSNorm_2 (h_norm)", c["h_norm"]),
        ("Final output", output),
    ]

    print(f"\n  {'Step':<35} {'Shape':<25} {'Norm':>10}")
    print(f"  {'-'*72}")
    for name, tensor in steps:
        norm = np.linalg.norm(tensor)
        print(f"  {name:<35} {str(tensor.shape):<25} {norm:>10.4f}")

    assert output.shape == x.shape, "Output shape must match input shape"
    print(f"\n  Shape preserved: {x.shape} -> {output.shape}")

    residual_diff = np.linalg.norm(output - x)
    print(f"  ||output - input|| = {residual_diff:.4f} (non-zero: sublayers contribute)")

    x_norm_rms = np.sqrt(np.mean(c["x_norm"] ** 2, axis=-1))
    h_norm_rms = np.sqrt(np.mean(c["h_norm"] ** 2, axis=-1))
    output_rms = np.sqrt(np.mean(output ** 2, axis=-1))
    print(f"\n  Pre-norm architecture verification:")
    print(f"    RMS of x_norm (input to attention): mean={x_norm_rms.mean():.4f} (should be ~1.0)")
    print(f"    RMS of h_norm (input to FFN):       mean={h_norm_rms.mean():.4f} (should be ~1.0)")
    print(f"    RMS of output (unnormalized):        mean={output_rms.mean():.4f} (not ~1.0)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    step_names = [s[0] for s in steps]
    step_norms = [np.linalg.norm(s[1]) for s in steps]
    bar_colors = [COLORS["blue"]] * 3 + [COLORS["green"]] * 4 + [COLORS["red"]] * 2 + \
                 [COLORS["purple"]] * 3 + [COLORS["orange"]] * 2 + [COLORS["teal"]] * 1 + [COLORS["dark"]]
    bar_colors = bar_colors[:len(step_names)]
    axes[0, 0].barh(range(len(step_names)), step_norms, color=bar_colors[:len(step_names)], edgecolor="white")
    axes[0, 0].set_yticks(range(len(step_names)))
    axes[0, 0].set_yticklabels(step_names, fontsize=7)
    axes[0, 0].set_xlabel("Frobenius Norm")
    axes[0, 0].set_title("Tensor Norms Through the Block", fontsize=11, fontweight="bold")
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    im1 = axes[0, 1].imshow(c["A"][0, 0], cmap="Blues", aspect="equal")
    axes[0, 1].set_xlabel("Key Position")
    axes[0, 1].set_ylabel("Query Position")
    axes[0, 1].set_title("Attention Weights (B=0, Head=0)\nCausal mask visible as upper triangle zeros",
                         fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    rms_values = {
        "x_norm\n(pre-attn)": x_norm_rms.flatten(),
        "h_norm\n(pre-FFN)": h_norm_rms.flatten(),
        "output\n(post-block)": output_rms.flatten(),
    }
    positions = list(rms_values.keys())
    bp_data = [rms_values[k] for k in positions]
    bp = axes[0, 2].boxplot(bp_data, tick_labels=positions, patch_artist=True)
    box_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0, 2].axhline(1.0, color=COLORS["red"], linestyle="--", alpha=0.7, label="RMS = 1.0")
    axes[0, 2].set_ylabel("RMS Value")
    axes[0, 2].set_title("Pre-Norm: Normalized Inputs to Sublayers\nOutput is NOT normalized (residual stream)",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    x_flat = x[0, 0, :]
    xnorm_flat = c["x_norm"][0, 0, :]
    attn_flat = c["attn_out"][0, 0, :]
    h_flat = c["h"][0, 0, :]
    ffn_flat = output[0, 0, :] - c["h"][0, 0, :]
    out_flat = output[0, 0, :]

    dims = range(min(32, d_model))
    axes[1, 0].plot(list(dims), x_flat[:32], "o-", markersize=3, color=COLORS["blue"],
                    linewidth=1, label="x (input)", alpha=0.8)
    axes[1, 0].plot(list(dims), attn_flat[:32], "s-", markersize=3, color=COLORS["green"],
                    linewidth=1, label="attn_out", alpha=0.8)
    axes[1, 0].plot(list(dims), h_flat[:32], "^-", markersize=3, color=COLORS["orange"],
                    linewidth=1, label="h = x + attn_out", alpha=0.8)
    axes[1, 0].set_xlabel("Dimension")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("Residual Connection 1 (First 32 dims)\nh = x + attention_output",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(list(dims), h_flat[:32], "o-", markersize=3, color=COLORS["orange"],
                    linewidth=1, label="h", alpha=0.8)
    axes[1, 1].plot(list(dims), ffn_flat[:32], "s-", markersize=3, color=COLORS["purple"],
                    linewidth=1, label="ffn_out", alpha=0.8)
    axes[1, 1].plot(list(dims), out_flat[:32], "^-", markersize=3, color=COLORS["dark"],
                    linewidth=1, label="output = h + ffn_out", alpha=0.8)
    axes[1, 1].set_xlabel("Dimension")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title("Residual Connection 2 (First 32 dims)\noutput = h + ffn_output",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    arch_text = (
        "PRE-NORM DECODER BLOCK\n"
        "======================\n\n"
        f"d_model = {d_model}\n"
        f"num_heads = {num_heads} (h_kv = {num_kv_heads})\n"
        f"d_ff = {d_ff}\n"
        f"d_k = {d_model // num_heads}\n\n"
        "Data Flow:\n"
        "  x -> RMSNorm_1 -> Q,K,V proj\n"
        "    -> RoPE(Q,K) -> GQA -> W_O\n"
        "    -> + x (residual 1) = h\n"
        "  h -> RMSNorm_2 -> SwiGLU FFN\n"
        "    -> + h (residual 2) = output\n\n"
        "Key observations:\n"
        f"  - x_norm RMS ~ {x_norm_rms.mean():.3f} (normalized)\n"
        f"  - h_norm RMS ~ {h_norm_rms.mean():.3f} (normalized)\n"
        f"  - output RMS ~ {output_rms.mean():.3f} (not normalized)\n"
        "  - Residual stream grows unboundedly\n"
        "    (this is by design in pre-norm)"
    )
    axes[1, 2].text(0.05, 0.95, arch_text, fontsize=10, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Transformer Block: Full Forward Pass Walkthrough",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_forward_pass.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/01_forward_pass.png")


# ---------------------------------------------------------------------------
# Example 2: Parameter Distribution Analysis
# ---------------------------------------------------------------------------
def example_2_parameter_distribution():
    """Compute parameter breakdown for real LLM configs."""
    print("\n" + "=" * 60)
    print("Example 2: Parameter Distribution Analysis")
    print("=" * 60)

    results = {}
    for name, cfg in MODEL_CONFIGS.items():
        params = count_parameters(cfg["d_model"], cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"])
        params["layers"] = cfg["layers"]
        params["d_model"] = cfg["d_model"]
        params["num_heads"] = cfg["num_heads"]
        params["num_kv_heads"] = cfg["num_kv_heads"]
        params["d_ff"] = cfg["d_ff"]
        results[name] = params

    print(f"\n  {'Model':<16} {'d_model':>7} {'h':>4} {'h_kv':>5} {'d_ff':>7} "
          f"{'Attn (M)':>10} {'FFN (M)':>10} {'Norm':>8} {'Total (M)':>10} "
          f"{'Attn%':>7} {'FFN%':>7} {'x Layers':>10}")
    print(f"  {'-'*110}")
    for name, p in results.items():
        total_model = p["total"] * p["layers"]
        print(f"  {name:<16} {p['d_model']:>7} {p['num_heads']:>4} {p['num_kv_heads']:>5} {p['d_ff']:>7} "
              f"{p['attn_total']/1e6:>10.1f} {p['ffn_total']/1e6:>10.1f} {p['norm_total']:>8} "
              f"{p['total']/1e6:>10.1f} {p['attn_pct']:>6.1f}% {p['ffn_pct']:>6.1f}% "
              f"{total_model/1e9:>8.2f}B")

    llama7b = results["Llama 2 7B"]
    llama70b = results["Llama 2 70B"]
    gqa_savings = (1 - llama70b["attn_total"] / (4 * llama70b["d_model"] ** 2)) * 100
    print(f"\n  GQA savings in Llama 2 70B attention params:")
    print(f"    MHA would need: {4 * llama70b['d_model']**2 / 1e6:.1f}M (4 * d^2)")
    print(f"    GQA actual:     {llama70b['attn_total']/1e6:.1f}M")
    print(f"    Savings:        {gqa_savings:.1f}%")
    print(f"    (from reducing W_K, W_V from d_model x d_model to d_model x h_kv*d_k)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    model_names = list(results.keys())
    attn_m = [results[n]["attn_total"] / 1e6 for n in model_names]
    ffn_m = [results[n]["ffn_total"] / 1e6 for n in model_names]
    norm_m = [results[n]["norm_total"] / 1e6 for n in model_names]

    x_pos = np.arange(len(model_names))
    w = 0.25
    axes[0, 0].bar(x_pos - w, attn_m, w, label="Attention", color=COLORS["blue"], edgecolor="white")
    axes[0, 0].bar(x_pos, ffn_m, w, label="FFN (SwiGLU)", color=COLORS["red"], edgecolor="white")
    axes[0, 0].bar(x_pos + w, norm_m, w, label="Norm (x2)", color=COLORS["green"], edgecolor="white")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, fontsize=9)
    axes[0, 0].set_ylabel("Parameters (Millions)")
    axes[0, 0].set_title("Parameter Breakdown per Block\nFFN dominates: ~67% (MHA) to ~82% (GQA)",
                         fontsize=11, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    ffn_pcts = [results[n]["ffn_pct"] for n in model_names]
    attn_pcts = [results[n]["attn_pct"] for n in model_names]
    norm_pcts = [results[n]["norm_pct"] for n in model_names]
    axes[0, 1].bar(x_pos, ffn_pcts, 0.5, label="FFN %", color=COLORS["red"], edgecolor="white")
    axes[0, 1].bar(x_pos, attn_pcts, 0.5, bottom=ffn_pcts, label="Attention %",
                   color=COLORS["blue"], edgecolor="white")
    axes[0, 1].axhline(66.67, color="gray", linestyle="--", alpha=0.5, label="2/3 mark")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(model_names, fontsize=9)
    axes[0, 1].set_ylabel("% of Total Parameters")
    axes[0, 1].set_title("Attention vs FFN Share\nFFN ~67% with MHA, ~80%+ with GQA",
                         fontsize=11, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    axes[0, 1].set_ylim(0, 105)

    for i, n in enumerate(model_names):
        axes[0, 1].text(i, ffn_pcts[i] / 2, f"{ffn_pcts[i]:.1f}%", ha="center", va="center",
                       fontsize=8, fontweight="bold", color="white")
        axes[0, 1].text(i, ffn_pcts[i] + attn_pcts[i] / 2, f"{attn_pcts[i]:.1f}%",
                       ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    ref_name = "Llama 3 8B"
    ref = results[ref_name]
    components = ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]
    comp_vals = [ref[c] / 1e6 for c in components]
    comp_colors = [COLORS["blue"]] * 4 + [COLORS["red"]] * 3
    explode = [0] * 4 + [0.05] * 3
    wedges, texts, autotexts = axes[1, 0].pie(
        comp_vals, labels=components, colors=comp_colors, explode=explode,
        autopct="%1.1f%%", pctdistance=0.8, startangle=90
    )
    for t in autotexts:
        t.set_fontsize(8)
    axes[1, 0].set_title(f"Parameter Pie Chart: {ref_name}\n"
                         f"Total = {ref['total']/1e6:.1f}M per block",
                         fontsize=11, fontweight="bold")

    mha_attn = {n: 4 * results[n]["d_model"] ** 2 for n in model_names}
    actual_attn = {n: results[n]["attn_total"] for n in model_names}
    gqa_savings_list = [(1 - actual_attn[n] / mha_attn[n]) * 100 for n in model_names]

    axes[1, 1].bar(x_pos - 0.15, [mha_attn[n] / 1e6 for n in model_names], 0.3,
                   label="MHA (4 * d^2)", color=COLORS["coral"], edgecolor="white")
    axes[1, 1].bar(x_pos + 0.15, [actual_attn[n] / 1e6 for n in model_names], 0.3,
                   label="Actual (with GQA)", color=COLORS["blue"], edgecolor="white")
    for i, n in enumerate(model_names):
        if gqa_savings_list[i] > 0.1:
            axes[1, 1].annotate(f"-{gqa_savings_list[i]:.0f}%",
                               (i + 0.15, actual_attn[n] / 1e6),
                               textcoords="offset points", xytext=(0, 8),
                               ha="center", fontsize=9, fontweight="bold", color=COLORS["green"])
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(model_names, fontsize=9)
    axes[1, 1].set_ylabel("Attention Parameters (Millions)")
    axes[1, 1].set_title("GQA Reduces Attention Parameters\nLlama 2 7B uses MHA (0% savings); 70B/3 8B/Mistral use GQA",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Transformer Block: Parameter Distribution Across Real LLM Configs",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_parameter_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/02_parameter_distribution.png")


# ---------------------------------------------------------------------------
# Example 3: SwiGLU Gating Visualization
# ---------------------------------------------------------------------------
def example_3_swiglu_gating():
    """Demonstrate the SwiGLU gating mechanism and compare with standard ReLU FFN."""
    print("\n" + "=" * 60)
    print("Example 3: SwiGLU Gating Visualization")
    print("=" * 60)

    d_model, d_ff = 32, 64
    B, L = 1, 1

    np.random.seed(SEED)
    ffn = SwiGLUFFN(d_model, d_ff)
    x = np.random.randn(B, L, d_model)

    output = ffn.forward(x)
    gate_linear = ffn._cache["gate"][0, 0, :]
    sig_gate = ffn._cache["sig_gate"][0, 0, :]
    silu_gate = ffn._cache["silu_gate"][0, 0, :]
    up = ffn._cache["up"][0, 0, :]
    hidden = ffn._cache["hidden"][0, 0, :]

    print(f"\n  Config: d_model={d_model}, d_ff={d_ff}")
    print(f"  Input norm: {np.linalg.norm(x):.4f}")
    print(f"  Output norm: {np.linalg.norm(output):.4f}")

    print(f"\n  Gate signal statistics:")
    print(f"    gate (linear): mean={gate_linear.mean():.4f}, std={gate_linear.std():.4f}")
    print(f"    sigmoid(gate): mean={sig_gate.mean():.4f}, range=[{sig_gate.min():.4f}, {sig_gate.max():.4f}]")
    print(f"    SiLU(gate):    mean={silu_gate.mean():.4f}, std={silu_gate.std():.4f}")
    print(f"    up (ungated):  mean={up.mean():.4f}, std={up.std():.4f}")
    print(f"    hidden (gated): mean={hidden.mean():.4f}, std={hidden.std():.4f}")

    suppressed = np.sum(np.abs(silu_gate) < 0.1)
    total = d_ff
    print(f"\n  Gating effect: {suppressed}/{total} features have |SiLU(gate)| < 0.1")
    print(f"  ({suppressed/total*100:.1f}% of features are effectively suppressed)")

    z = np.linspace(-5, 5, 500)
    silu_z = z * _stable_sigmoid(z)
    relu_z = np.maximum(0, z)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    sorted_idx = np.argsort(np.abs(silu_gate))
    axes[0, 0].bar(range(d_ff), silu_gate[sorted_idx], color=COLORS["blue"], alpha=0.8, width=1.0)
    axes[0, 0].axhline(0, color="gray", linestyle="-", alpha=0.3)
    axes[0, 0].axhline(0.1, color=COLORS["red"], linestyle="--", alpha=0.5, label="|SiLU| = 0.1")
    axes[0, 0].axhline(-0.1, color=COLORS["red"], linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Feature Index (sorted by |SiLU|)")
    axes[0, 0].set_ylabel("SiLU(gate)")
    axes[0, 0].set_title(f"Gate Signal: SiLU(x @ W_gate)\n{suppressed}/{total} features suppressed (|val| < 0.1)",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    feature_idx = range(d_ff)
    axes[0, 1].scatter(feature_idx, up, s=15, alpha=0.6, color=COLORS["orange"], label="up = x @ W_up")
    axes[0, 1].scatter(feature_idx, hidden, s=15, alpha=0.6, color=COLORS["green"],
                       label="hidden = SiLU(gate) * up")
    axes[0, 1].set_xlabel("Feature Index")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].set_title("Ungated vs Gated Signal\nGate selectively amplifies/suppresses features",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].scatter(up, hidden, s=20, alpha=0.6, c=np.abs(silu_gate),
                       cmap="coolwarm", edgecolors="none")
    axes[0, 2].plot([-3, 3], [-3, 3], "--", color="gray", alpha=0.3, label="identity")
    axes[0, 2].plot([-3, 3], [0, 0], "-", color="gray", alpha=0.2)
    axes[0, 2].set_xlabel("Ungated (up)")
    axes[0, 2].set_ylabel("Gated (hidden)")
    axes[0, 2].set_title("Gating Effect: hidden = SiLU(gate) * up\nColor = |gate strength|",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(z, silu_z, color=COLORS["blue"], linewidth=2.5, label="SiLU(z) = z * sigma(z)")
    axes[1, 0].plot(z, relu_z, color=COLORS["red"], linewidth=2, linestyle="--", label="ReLU(z)")
    axes[1, 0].plot(z, z, color="gray", linewidth=1, linestyle=":", alpha=0.4, label="Identity")
    axes[1, 0].axhline(0, color="gray", alpha=0.3)
    axes[1, 0].axvline(0, color="gray", alpha=0.3)
    axes[1, 0].set_xlabel("z")
    axes[1, 0].set_ylabel("Activation(z)")
    axes[1, 0].set_title("SiLU vs ReLU Activation\nSiLU is smooth with a slight negative lobe",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    silu_deriv = _stable_sigmoid(z) * (1.0 + z * (1.0 - _stable_sigmoid(z)))
    relu_deriv = np.where(z > 0, 1.0, 0.0)
    axes[1, 1].plot(z, silu_deriv, color=COLORS["blue"], linewidth=2.5, label="SiLU'(z)")
    axes[1, 1].plot(z, relu_deriv, color=COLORS["red"], linewidth=2, linestyle="--", label="ReLU'(z)")
    axes[1, 1].axhline(1, color="gray", linestyle=":", alpha=0.3)
    axes[1, 1].set_xlabel("z")
    axes[1, 1].set_ylabel("Derivative")
    axes[1, 1].set_title("Derivatives: SiLU is Smooth Everywhere\nReLU has a discontinuity at z=0",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    np.random.seed(SEED + 10)
    W1 = _xavier((d_model, d_ff))
    W2 = _xavier((d_ff, d_model))

    relu_hidden = np.maximum(0, x[0, 0] @ W1)
    relu_output = relu_hidden @ W2

    relu_dead = np.sum(relu_hidden == 0)
    swiglu_near_zero = np.sum(np.abs(hidden) < 0.01)

    axes[1, 2].axis("off")
    comparison = (
        "SwiGLU vs Standard ReLU FFN\n"
        "============================\n\n"
        "Standard ReLU FFN (2 matrices):\n"
        "  output = ReLU(x @ W1) @ W2\n"
        f"  Params: 2 * d * d_ff = {2*d_model*d_ff:,}\n"
        f"  Dead neurons (ReLU=0): {relu_dead}/{d_ff} ({relu_dead/d_ff*100:.0f}%)\n\n"
        "SwiGLU FFN (3 matrices):\n"
        "  output = (SiLU(x@W_gate) * (x@W_up)) @ W_down\n"
        f"  Params: 3 * d * d_ff = {3*d_model*d_ff:,}\n"
        f"  Near-zero features: {swiglu_near_zero}/{d_ff} ({swiglu_near_zero/d_ff*100:.0f}%)\n\n"
        "Key differences:\n"
        "  - SiLU is smooth (continuous gradient)\n"
        "  - No hard zeros (no dead neurons)\n"
        "  - Gating enables selective filtering\n"
        "  - 50% more params, but better quality\n"
        "  - d_ff = 8/3 * d (not 4*d) to compensate\n\n"
        "CAVEAT: Comparisons use random weights.\n"
        "Trained SwiGLU learns meaningful gates."
    )
    axes[1, 2].text(0.05, 0.95, comparison, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("SwiGLU Gating Mechanism: Smooth, Selective Feature Filtering",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_swiglu_gating.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/03_swiglu_gating.png")


# ---------------------------------------------------------------------------
# Example 4: Residual Connection Analysis
# ---------------------------------------------------------------------------
def example_4_residual_connections():
    """Stack transformer blocks and analyze gradient flow stability."""
    print("\n" + "=" * 60)
    print("Example 4: Residual Connection Analysis")
    print("=" * 60)

    d_model, num_heads, num_kv_heads, d_ff = 32, 4, 2, 86
    max_seq_len = 32
    B, L = 1, 8
    layer_counts = [2, 4, 8, 16, 32]

    np.random.seed(SEED)
    x_base = np.random.randn(B, L, d_model) * 0.1

    print(f"\n  Config: d_model={d_model}, h={num_heads}, h_kv={num_kv_heads}, d_ff={d_ff}")
    print(f"  Input norm: {np.linalg.norm(x_base):.4f}")
    print(f"  Layer counts tested: {layer_counts}")

    output_norms = {}
    gradient_norms_per_layer = {}

    for n_layers in layer_counts:
        np.random.seed(SEED)
        blocks = [TransformerBlock(d_model, num_heads, num_kv_heads, d_ff, max_seq_len)
                  for _ in range(n_layers)]

        # Scale weights down to prevent exploding norms with many layers
        for blk in blocks:
            scale = 1.0 / np.sqrt(n_layers)
            blk.W_Q *= scale
            blk.W_K *= scale
            blk.W_V *= scale
            blk.W_O *= scale
            blk.ffn.W_gate *= scale
            blk.ffn.W_up *= scale
            blk.ffn.W_down *= scale

        h = x_base.copy()
        layer_norms_fwd = [np.linalg.norm(h)]
        for blk in blocks:
            h = blk.forward(h)
            layer_norms_fwd.append(np.linalg.norm(h))
        output_norms[n_layers] = layer_norms_fwd

        grad = np.ones_like(h) / h.size
        layer_norms_bwd = [np.linalg.norm(grad)]
        for blk in reversed(blocks):
            grad = blk.backward(grad)
            layer_norms_bwd.append(np.linalg.norm(grad))
        layer_norms_bwd.reverse()
        gradient_norms_per_layer[n_layers] = layer_norms_bwd

    print(f"\n  {'Layers':>8} {'Input Norm':>12} {'Output Norm':>12} {'Ratio':>10} {'Grad In':>12} {'Grad Out':>12}")
    print(f"  {'-'*72}")
    for n in layer_counts:
        in_n = output_norms[n][0]
        out_n = output_norms[n][-1]
        g_in = gradient_norms_per_layer[n][0]
        g_out = gradient_norms_per_layer[n][-1]
        print(f"  {n:>8} {in_n:>12.4f} {out_n:>12.4f} {out_n/in_n:>10.2f} "
              f"{g_in:>12.6f} {g_out:>12.6f}")

    print(f"\n  Observation: Output norms grow but do not explode due to 1/sqrt(N) weight scaling.")
    print(f"  Gradient norms remain stable across layers (residual 'gradient highway').")
    print(f"  CAVEAT: Random initialization; trained models exhibit better-controlled norms.")

    # Analytical pre-norm vs post-norm comparison
    # Pre-norm: grad_x = grad_output + grad_through_sublayer (direct identity path)
    # Post-norm: grad_x = LayerNorm_backward(grad_output) (no direct path)
    print(f"\n  Pre-norm gradient analysis (analytical):")
    print(f"    d(output)/d(x) = I + d(sublayer)/d(x)  [at each residual]")
    print(f"    Through N blocks: gradient includes I^N = I term (never vanishes)")
    print(f"    Post-norm: d(LN(x + f(x)))/d(x) must pass through LN backward")
    print(f"    -> no guaranteed identity path -> potential vanishing gradients")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for n in layer_counts:
        layers_range = range(len(output_norms[n]))
        axes[0, 0].plot(list(layers_range), output_norms[n], "o-", markersize=4,
                        linewidth=1.5, label=f"N={n}")
    axes[0, 0].set_xlabel("Layer Index")
    axes[0, 0].set_ylabel("Activation Norm (Frobenius)")
    axes[0, 0].set_title("Forward: Activation Norms Through Layers\nGrows moderately (1/sqrt(N) scaling applied)",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    for n in layer_counts:
        layers_range = range(len(gradient_norms_per_layer[n]))
        axes[0, 1].plot(list(layers_range), gradient_norms_per_layer[n], "s-", markersize=4,
                        linewidth=1.5, label=f"N={n}")
    axes[0, 1].set_xlabel("Layer Index (0 = input, N = output)")
    axes[0, 1].set_ylabel("Gradient Norm")
    axes[0, 1].set_title("Backward: Gradient Norms Through Layers\n'Gradient highway': residuals prevent vanishing",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    n_show = 32
    if n_show in gradient_norms_per_layer:
        gn = gradient_norms_per_layer[n_show]
        ratios = [gn[i+1] / gn[i] if gn[i] > 1e-15 else 1.0 for i in range(len(gn) - 1)]
        axes[0, 2].plot(range(len(ratios)), ratios, "o-", color=COLORS["purple"],
                        markersize=4, linewidth=1.5)
        axes[0, 2].axhline(1.0, color=COLORS["red"], linestyle="--", alpha=0.5, label="Ratio = 1.0")
        axes[0, 2].set_xlabel("Layer Transition")
        axes[0, 2].set_ylabel("||grad[i+1]|| / ||grad[i]||")
        axes[0, 2].set_title(f"Gradient Ratio Between Adjacent Layers (N={n_show})\n"
                             "Ratio near 1.0 = stable gradient flow",
                             fontsize=10, fontweight="bold")
        axes[0, 2].legend(fontsize=9)
        axes[0, 2].grid(True, alpha=0.3)

    final_norms = [output_norms[n][-1] for n in layer_counts]
    input_norm = output_norms[layer_counts[0]][0]
    axes[1, 0].plot(layer_counts, final_norms, "o-", color=COLORS["blue"], linewidth=2, markersize=8,
                    label="Output norm")
    axes[1, 0].axhline(input_norm, color=COLORS["red"], linestyle="--", alpha=0.5,
                        label=f"Input norm = {input_norm:.4f}")
    axes[1, 0].set_xlabel("Number of Layers")
    axes[1, 0].set_ylabel("Output Norm")
    axes[1, 0].set_title("Output Norm vs Depth\nGrows but does not explode (with weight scaling)",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    grad_at_input = [gradient_norms_per_layer[n][0] for n in layer_counts]
    grad_at_output = [gradient_norms_per_layer[n][-1] for n in layer_counts]
    axes[1, 1].plot(layer_counts, grad_at_input, "o-", color=COLORS["green"], linewidth=2,
                    markersize=8, label="Gradient at input (layer 0)")
    axes[1, 1].plot(layer_counts, grad_at_output, "s-", color=COLORS["orange"], linewidth=2,
                    markersize=8, label="Gradient at output (layer N)")
    axes[1, 1].set_xlabel("Number of Layers")
    axes[1, 1].set_ylabel("Gradient Norm")
    axes[1, 1].set_title("Gradient at Input vs Output Layer\nResidual connections preserve gradient magnitude",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    analysis = (
        "RESIDUAL CONNECTIONS AS\n"
        "GRADIENT HIGHWAYS\n"
        "========================\n\n"
        "Pre-norm architecture:\n"
        "  output = x + sublayer(Norm(x))\n\n"
        "Gradient at residual:\n"
        "  d(output)/d(x) = I + d(sublayer)/d(Norm(x))\n"
        "                        * d(Norm)/d(x)\n\n"
        "The identity term I ensures:\n"
        "  ||grad_x|| >= ||grad_output|| - ||sublayer grad||\n\n"
        "Through N blocks:\n"
        "  grad includes I^N = I\n"
        "  -> NEVER vanishes (gradient highway)\n\n"
        "Post-norm (for comparison):\n"
        "  output = Norm(x + sublayer(x))\n"
        "  grad must pass through Norm backward\n"
        "  -> no guaranteed I term\n"
        "  -> potential vanishing gradients\n\n"
        "This is why ALL modern LLMs\n"
        "use pre-norm (GPT-2 onwards)."
    )
    axes[1, 2].text(0.05, 0.95, analysis, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Residual Connection Analysis: Gradient Highway Effect in Pre-Norm Transformer",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_residual_connections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/04_residual_connections.png")


# ---------------------------------------------------------------------------
# Example 5: Compute Distribution (FLOPs Breakdown)
# ---------------------------------------------------------------------------
def example_5_flops_breakdown():
    """Sweep sequence length and show FLOPs crossover between FFN and attention core."""
    print("\n" + "=" * 60)
    print("Example 5: Compute Distribution (FLOPs Breakdown)")
    print("=" * 60)

    cfg = MODEL_CONFIGS["Llama 3 8B"]
    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    d_ff = cfg["d_ff"]
    d_k = d_model // num_heads
    B = 1

    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    attn_proj_flops = []
    attn_core_flops = []
    ffn_flops = []
    rope_flops_list = []
    norm_flops_list = []
    total_flops = []

    for L in seq_lens:
        f = count_flops(B, L, d_model, num_heads, num_kv_heads, d_ff)
        attn_proj_flops.append(f["attn_proj"])
        attn_core_flops.append(f["attn_core"])
        ffn_flops.append(f["ffn_total"])
        rope_flops_list.append(f["rope"])
        norm_flops_list.append(f["norm"])
        total_flops.append(f["total"])

    # Analytical crossover: attention_core = ffn_total
    # attn_core = (4*d_k + 5) * B * h * L^2   (approximately 4*B*h*L^2*d_k for large d_k)
    # ffn_total = 6 * B * L * d_model * d_ff
    # Crossover: (4*d_k + 5) * h * L^2 = 6 * L * d_model * d_ff
    # L_cross = 6 * d_model * d_ff / ((4*d_k + 5) * h)
    L_crossover = 6 * d_model * d_ff / ((4 * d_k + 5) * num_heads)

    # Also compute attn_core = attn_proj crossover
    # attn_proj = 2*B*L*(2*d^2 + 2*d*h_kv*d_k)  [per-token, linear in L]
    # attn_core = (4*d_k+5)*B*h*L^2              [quadratic in L]
    # Crossover: 2*(2*d^2 + 2*d*h_kv*d_k) = (4*d_k+5)*h*L
    attn_proj_per_token = 2 * (2 * d_model**2 + 2 * d_model * num_kv_heads * d_k)
    attn_core_per_l = (4 * d_k + 5) * num_heads
    L_proj_core_cross = attn_proj_per_token / attn_core_per_l

    print(f"\n  Config: Llama 3 8B (d={d_model}, h={num_heads}, h_kv={num_kv_heads}, d_ff={d_ff})")
    print(f"  d_k = {d_k}")

    print(f"\n  {'L':>7} {'Attn Proj':>14} {'Attn Core':>14} {'FFN':>14} {'RoPE':>10} {'Norm':>10} {'Total':>14}")
    print(f"  {'-'*90}")
    for i, L in enumerate(seq_lens):
        print(f"  {L:>7} {attn_proj_flops[i]/1e9:>13.2f}G {attn_core_flops[i]/1e9:>13.2f}G "
              f"{ffn_flops[i]/1e9:>13.2f}G {rope_flops_list[i]/1e6:>9.1f}M "
              f"{norm_flops_list[i]/1e6:>9.1f}M {total_flops[i]/1e9:>13.2f}G")

    print(f"\n  ANALYTICAL CROSSOVER POINTS:")
    print(f"    Attention core surpasses FFN at L = {L_crossover:.0f}")
    print(f"      Formula: L = 6 * d * d_ff / ((4*d_k + 5) * h)")
    print(f"      = 6 * {d_model} * {d_ff} / ((4*{d_k} + 5) * {num_heads})")
    print(f"      = {6*d_model*d_ff} / {(4*d_k+5)*num_heads} = {L_crossover:.1f}")
    print(f"    Attention core surpasses attention projections at L = {L_proj_core_cross:.0f}")
    print(f"      Formula: L = 2*(2*d^2 + 2*d*h_kv*d_k) / ((4*d_k+5)*h)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].loglog(seq_lens, attn_proj_flops, "o-", color=COLORS["blue"], linewidth=2,
                       markersize=6, label="Attn projections (O(L))")
    axes[0, 0].loglog(seq_lens, attn_core_flops, "s-", color=COLORS["red"], linewidth=2,
                       markersize=6, label="Attn core QK+AV (O(L^2))")
    axes[0, 0].loglog(seq_lens, ffn_flops, "^-", color=COLORS["green"], linewidth=2,
                       markersize=6, label="FFN SwiGLU (O(L))")
    axes[0, 0].loglog(seq_lens, rope_flops_list, "d-", color=COLORS["purple"], linewidth=1.5,
                       markersize=5, label="RoPE (O(L))")
    axes[0, 0].loglog(seq_lens, norm_flops_list, "v-", color=COLORS["orange"], linewidth=1.5,
                       markersize=5, label="Norms (O(L))")
    axes[0, 0].axvline(L_crossover, color=COLORS["red"], linestyle="--", alpha=0.5,
                        label=f"Core=FFN at L={L_crossover:.0f}")
    axes[0, 0].set_xlabel("Sequence Length (L)")
    axes[0, 0].set_ylabel("FLOPs")
    axes[0, 0].set_title("FLOPs Breakdown by Component (Log-Log)\nAttention core is O(L^2), all others O(L)",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    attn_core_pct = [100 * attn_core_flops[i] / total_flops[i] for i in range(len(seq_lens))]
    ffn_pct = [100 * ffn_flops[i] / total_flops[i] for i in range(len(seq_lens))]
    attn_proj_pct = [100 * attn_proj_flops[i] / total_flops[i] for i in range(len(seq_lens))]
    other_pct = [100 - attn_core_pct[i] - ffn_pct[i] - attn_proj_pct[i] for i in range(len(seq_lens))]

    axes[0, 1].stackplot(seq_lens, ffn_pct, attn_proj_pct, attn_core_pct, other_pct,
                          labels=["FFN", "Attn Proj", "Attn Core", "Other (RoPE+Norm)"],
                          colors=[COLORS["green"], COLORS["blue"], COLORS["red"], COLORS["purple"]],
                          alpha=0.8)
    axes[0, 1].axvline(L_crossover, color="black", linestyle="--", alpha=0.7,
                        label=f"Crossover L={L_crossover:.0f}")
    axes[0, 1].set_xlabel("Sequence Length (L)")
    axes[0, 1].set_ylabel("% of Total FLOPs")
    axes[0, 1].set_title("FLOPs Share vs Sequence Length\nFFN dominates at short L, attention core at long L",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=8, loc="center right")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 100)

    per_token = [total_flops[i] / (B * seq_lens[i]) for i in range(len(seq_lens))]
    axes[0, 2].plot(seq_lens, [p / 1e6 for p in per_token], "o-", color=COLORS["dark"],
                    linewidth=2, markersize=6)
    axes[0, 2].set_xlabel("Sequence Length (L)")
    axes[0, 2].set_ylabel("FLOPs per Token (MFLOPs)")
    axes[0, 2].set_title("Per-Token Cost Grows with L\nDue to O(L) attention per token (each attends to all prior)",
                         fontsize=10, fontweight="bold")
    axes[0, 2].set_xscale("log", base=2)
    axes[0, 2].grid(True, alpha=0.3)

    models_for_cross = {
        "Llama 2 7B": MODEL_CONFIGS["Llama 2 7B"],
        "Llama 2 70B": MODEL_CONFIGS["Llama 2 70B"],
        "Llama 3 8B": MODEL_CONFIGS["Llama 3 8B"],
        "Mistral 7B": MODEL_CONFIGS["Mistral 7B"],
    }
    cross_points = {}
    for name, mcfg in models_for_cross.items():
        dk = mcfg["d_model"] // mcfg["num_heads"]
        lc = 6 * mcfg["d_model"] * mcfg["d_ff"] / ((4 * dk + 5) * mcfg["num_heads"])
        cross_points[name] = lc

    model_names_cross = list(cross_points.keys())
    cross_vals = [cross_points[n] for n in model_names_cross]
    cross_colors = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"]]
    bars = axes[1, 0].bar(range(len(model_names_cross)), cross_vals, color=cross_colors,
                          edgecolor="white", width=0.5)
    for bar, val in zip(bars, cross_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                       f"L={val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1, 0].set_xticks(range(len(model_names_cross)))
    axes[1, 0].set_xticklabels(model_names_cross, fontsize=9)
    axes[1, 0].set_ylabel("Crossover Sequence Length")
    axes[1, 0].set_title("Attention Core = FFN Crossover Point\nL = 6*d*d_ff / ((4*d_k+5)*h)",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    L_sweep = np.array(seq_lens, dtype=float)
    attn_core_analytical = (4 * d_k + 5) * num_heads * L_sweep**2
    ffn_analytical = 6 * d_model * d_ff * L_sweep
    axes[1, 1].loglog(L_sweep, attn_core_analytical, "-", color=COLORS["red"], linewidth=2,
                       label=f"Attn core: (4*{d_k}+5)*{num_heads}*L^2")
    axes[1, 1].loglog(L_sweep, ffn_analytical, "-", color=COLORS["green"], linewidth=2,
                       label=f"FFN: 6*{d_model}*{d_ff}*L")
    axes[1, 1].loglog(L_sweep, attn_core_analytical, "s", color=COLORS["red"], markersize=6, alpha=0.5)
    axes[1, 1].loglog(L_sweep, ffn_analytical, "^", color=COLORS["green"], markersize=6, alpha=0.5)
    axes[1, 1].axvline(L_crossover, color="black", linestyle="--", alpha=0.7,
                        label=f"Crossover: L = {L_crossover:.0f}")
    axes[1, 1].set_xlabel("Sequence Length (L)")
    axes[1, 1].set_ylabel("FLOPs (analytical)")
    axes[1, 1].set_title("Analytical: Attention Core (L^2) vs FFN (L)\nCrossover is exact intersection",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    flops_summary = (
        "FLOPs FORMULAS (per block, fwd)\n"
        "================================\n\n"
        f"Config: Llama 3 8B\n"
        f"  d={d_model}, h={num_heads}, h_kv={num_kv_heads}\n"
        f"  d_k={d_k}, d_ff={d_ff}\n\n"
        "Attn projections (linear in L):\n"
        f"  Q: 2*B*L*d^2 = 2*L*{d_model}^2\n"
        f"  K: 2*B*L*d*h_kv*d_k = 2*L*{d_model}*{num_kv_heads*d_k}\n"
        f"  V: same as K\n"
        f"  O: 2*B*L*d^2\n\n"
        "Attn core (QUADRATIC in L):\n"
        f"  QK^T: 2*B*h*L^2*d_k\n"
        f"  softmax: 5*B*h*L^2\n"
        f"  AV:  2*B*h*L^2*d_k\n\n"
        "FFN SwiGLU (linear in L):\n"
        f"  gate+up+down: 6*B*L*d*d_ff\n\n"
        "CROSSOVER (core = FFN):\n"
        f"  L = 6*d*d_ff / ((4*d_k+5)*h)\n"
        f"  L = {L_crossover:.0f} tokens"
    )
    axes[1, 2].text(0.05, 0.95, flops_summary, fontsize=9, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Compute Distribution: FLOPs Breakdown and Attention/FFN Crossover",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_flops_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/05_flops_breakdown.png")


# ---------------------------------------------------------------------------
# Example 6: Causal Masking and Position Sensitivity
# ---------------------------------------------------------------------------
def example_6_causal_and_position():
    """Verify causal property and RoPE position sensitivity."""
    print("\n" + "=" * 60)
    print("Example 6: Causal Masking and Position Sensitivity")
    print("=" * 60)

    d_model, num_heads, num_kv_heads, d_ff = 64, 4, 2, 172
    max_seq_len = 32
    B, L = 1, 8

    np.random.seed(SEED)
    block = TransformerBlock(d_model, num_heads, num_kv_heads, d_ff, max_seq_len)
    x = np.random.randn(B, L, d_model)

    output_original = block.forward(x).copy()

    np.random.seed(SEED + 100)
    x_modified = x.copy()
    x_modified[:, 4:, :] = np.random.randn(B, L - 4, d_model)

    output_modified = block.forward(x_modified).copy()

    print(f"\n  Config: d_model={d_model}, h={num_heads}, L={L}")
    print(f"\n  Causal property verification:")
    print(f"  Modified tokens at positions 4..{L-1}; positions 0..3 should be unchanged")
    print(f"\n  {'Position':>10} {'||diff||':>12} {'Causal?':>10}")
    print(f"  {'-'*36}")
    causal_diffs = []
    for i in range(L):
        diff = np.linalg.norm(output_original[0, i] - output_modified[0, i])
        causal_diffs.append(diff)
        is_causal = diff < 1e-12
        status = "YES" if is_causal else "NO (expected)" if i >= 4 else "FAIL"
        print(f"  {i:>10} {diff:>12.2e} {status:>10}")

    for i in range(4):
        assert causal_diffs[i] < 1e-12, f"Position {i} should not be affected by future changes"
    for i in range(4, L):
        assert causal_diffs[i] > 1e-6, f"Position {i} should be affected by changes"

    print(f"\n  Causal masking verified: positions 0..3 unaffected by changes at 4..{L-1}")

    # RoPE position sensitivity test: use DISTINCT tokens so attention distributions
    # matter. Then show that changing positions changes the output.
    np.random.seed(SEED + 200)
    x_distinct = np.random.randn(B, L, d_model)

    output_with_rope = block.forward(x_distinct, positions=np.arange(L)).copy()

    # Remove RoPE effect by using position 0 for all tokens
    # (RoPE at position 0 is identity rotation)
    output_no_rope_effect = block.forward(x_distinct, positions=np.zeros(L, dtype=int)).copy()

    rope_diffs = np.linalg.norm(output_with_rope[0] - output_no_rope_effect[0], axis=1)
    mean_rope_diff = rope_diffs.mean()

    print(f"\n  RoPE position sensitivity (distinct tokens):")
    print(f"  Same input, positions [0..{L-1}] vs all-zeros (RoPE disabled)")
    print(f"  Per-position ||diff||:")
    for i in range(L):
        print(f"    Position {i}: {rope_diffs[i]:.6f}")
    print(f"  Mean diff: {mean_rope_diff:.6f}")
    print(f"  Position 0 diff is ~0 (RoPE at pos 0 is identity)")
    print(f"  Other positions differ because RoPE rotates Q/K differently")

    # Show that uniform shift with identical tokens preserves outputs
    # (relative position property of RoPE)
    np.random.seed(SEED)
    token_embed = np.random.randn(d_model)
    x_identical = np.tile(token_embed, (B, L, 1))

    output_default = block.forward(x_identical, positions=np.arange(L)).copy()
    output_shifted = block.forward(x_identical, positions=np.arange(5, 5 + L)).copy()
    shift_diffs = np.linalg.norm(output_default[0] - output_shifted[0], axis=1)
    mean_shift_diff = shift_diffs.mean()

    print(f"\n  Shift invariance (relative position property):")
    print(f"  Identical tokens, positions [0..{L-1}] vs [5..{5+L-1}]")
    print(f"  Mean ||diff|| per position: {mean_shift_diff:.2e}")
    print(f"  (Analytically zero: relative positions preserved)")

    x_single = x_distinct[:, :1, :].copy()
    output_single = block.forward(x_single, positions=np.array([0])).copy()
    output_first_from_full = block.forward(x_distinct, positions=np.arange(L)).copy()[:, :1, :]
    single_vs_full_diff = np.linalg.norm(output_single - output_first_from_full)
    print(f"\n  Position 0: single-token output vs full-sequence output[0]")
    print(f"    ||diff|| = {single_vs_full_diff:.2e}")
    print(f"    (Should be ~0: position 0 can only attend to itself)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    bars = axes[0, 0].bar(range(L), causal_diffs,
                          color=[COLORS["green"] if d < 1e-12 else COLORS["red"] for d in causal_diffs],
                          edgecolor="white", width=0.6)
    axes[0, 0].axhline(1e-12, color="gray", linestyle="--", alpha=0.5, label="1e-12 threshold")
    axes[0, 0].axvline(3.5, color="black", linestyle="-", alpha=0.3)
    axes[0, 0].annotate("Future tokens\nmodified here", xy=(5.5, max(causal_diffs) * 0.8),
                        fontsize=9, ha="center", color=COLORS["red"])
    axes[0, 0].annotate("Unaffected\n(causal)", xy=(1.5, max(causal_diffs) * 0.4),
                        fontsize=9, ha="center", color=COLORS["green"])
    axes[0, 0].set_xlabel("Token Position")
    axes[0, 0].set_ylabel("||output_original - output_modified||")
    axes[0, 0].set_title("Causal Property: Position i Depends Only on 0..i\nGreen = unaffected, Red = affected by future change",
                         fontsize=10, fontweight="bold")
    axes[0, 0].set_yscale("symlog", linthresh=1e-14)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    mask = create_causal_mask(L)
    mask_visual = np.where(mask.squeeze() == 0, 1.0, 0.0)
    im1 = axes[0, 1].imshow(mask_visual, cmap="Blues", aspect="equal")
    axes[0, 1].set_xlabel("Key Position")
    axes[0, 1].set_ylabel("Query Position")
    axes[0, 1].set_title("Causal Mask (1 = attend, 0 = blocked)\nLower triangle: each position sees only past",
                         fontsize=10, fontweight="bold")
    for i in range(L):
        for j in range(L):
            color = "white" if mask_visual[i, j] > 0.5 else "gray"
            axes[0, 1].text(j, i, f"{mask_visual[i,j]:.0f}", ha="center", va="center",
                           fontsize=8, color=color)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].bar(range(L), rope_diffs, color=COLORS["purple"], edgecolor="white", width=0.6)
    axes[0, 2].set_xlabel("Token Position")
    axes[0, 2].set_ylabel("||output(with RoPE) - output(no RoPE)||")
    axes[0, 2].set_title("RoPE Position Sensitivity\nPos 0 ~ 0 (identity rotation); others differ significantly",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    output_with_rope_perpos = output_with_rope[0]
    output_no_rope_perpos = output_no_rope_effect[0]
    cos_sim_with = np.zeros((L, L))
    cos_sim_without = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            n_wi = np.linalg.norm(output_with_rope_perpos[i])
            n_wj = np.linalg.norm(output_with_rope_perpos[j])
            n_ni = np.linalg.norm(output_no_rope_perpos[i])
            n_nj = np.linalg.norm(output_no_rope_perpos[j])
            if n_wi > 1e-10 and n_wj > 1e-10:
                cos_sim_with[i, j] = np.dot(output_with_rope_perpos[i], output_with_rope_perpos[j]) / (n_wi * n_wj)
            if n_ni > 1e-10 and n_nj > 1e-10:
                cos_sim_without[i, j] = np.dot(output_no_rope_perpos[i], output_no_rope_perpos[j]) / (n_ni * n_nj)

    im3 = axes[1, 0].imshow(cos_sim_with - cos_sim_without, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="equal")
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_ylabel("Position")
    axes[1, 0].set_title("Cosine Sim Difference: with RoPE - without RoPE\nRoPE reshapes inter-position similarity structure",
                         fontsize=10, fontweight="bold")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].bar(np.arange(L) - 0.15, shift_diffs, 0.3, label="Uniform shift (pos [5..12])",
                   color=COLORS["green"], edgecolor="white")
    axes[1, 1].bar(np.arange(L) + 0.15, rope_diffs, 0.3, label="No RoPE (all pos=0)",
                   color=COLORS["purple"], edgecolor="white")
    axes[1, 1].set_xlabel("Token Position")
    axes[1, 1].set_ylabel("||diff from default||")
    axes[1, 1].set_title("Shift Invariance vs RoPE Removal\nShift preserves outputs; removing RoPE changes them",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    causal_text = (
        "CAUSAL MASKING + RoPE\n"
        "======================\n\n"
        "Causal mask ensures:\n"
        "  output[i] depends only on input[0..i]\n"
        "  Verified: changing tokens at pos 4..7\n"
        "  does NOT affect output at pos 0..3\n\n"
        "RoPE position awareness:\n"
        "  RoPE at pos 0 is identity (no rotation)\n"
        "  Other positions rotate Q/K, changing\n"
        "  attention score distribution.\n"
        f"  Mean diff (RoPE vs no-RoPE): {mean_rope_diff:.4f}\n\n"
        "Shift invariance (relative pos. prop.):\n"
        "  Identical tokens + uniform shift\n"
        "  preserves all relative positions.\n"
        f"  Mean diff: {mean_shift_diff:.2e} (~0)\n\n"
        f"Single token vs full sequence:\n"
        f"  pos 0 diff = {single_vs_full_diff:.2e}\n"
        f"  (Confirms pos 0 only sees itself)\n\n"
        "CAVEAT: With random weights, position\n"
        "effects are noise-like. Trained models\n"
        "learn meaningful position-dependent\n"
        "attention patterns."
    )
    axes[1, 2].text(0.05, 0.95, causal_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Causal Masking and Position Sensitivity: Autoregressive Behavior + RoPE",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_causal_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/06_causal_position.png")


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
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.78, "Transformer Block", fontsize=28, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.68, "Pre-Norm Decoder Block: RMSNorm + GQA + RoPE + SwiGLU",
                fontsize=14, ha="center", va="center", transform=ax.transAxes, color="gray")
        info_text = (
            "The fundamental repeated unit of every modern LLM.\n"
            "Wires together RMSNorm, grouped-query attention with RoPE,\n"
            "and a SwiGLU FFN into the pre-norm architecture used by\n"
            "Llama, Mistral, and all modern open-weight models.\n\n"
            "This demo covers:\n"
            "  1. Full forward pass walkthrough with shape tracing\n"
            "  2. Parameter distribution for Llama 2/3 and Mistral configs\n"
            "  3. SwiGLU gating mechanism visualization\n"
            "  4. Residual connection gradient highway analysis\n"
            "  5. FLOPs breakdown with analytical crossover points\n"
            "  6. Causal masking and RoPE position sensitivity\n\n"
            f"Random seed: {SEED}\n"
            f"Number of visualizations: {len(viz_files)}\n"
            "Examples: 6"
        )
        ax.text(0.5, 0.35, info_text, fontsize=12, ha="center", va="center",
                transform=ax.transAxes, linespacing=1.6)
        ax.text(0.5, 0.08, "Generated by demo.py", fontsize=10, ha="center",
                va="center", transform=ax.transAxes, style="italic", color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.94, "Summary of Findings", fontsize=20, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)

        summary_items = [
            "1. Forward Pass: Output shape matches input (B, L, d_model). Pre-norm",
            "   architecture normalizes inputs to sublayers (RMS ~ 1.0) while the",
            "   residual stream remains unnormalized and grows with depth.",
            "",
            "2. Parameter Distribution: FFN (SwiGLU) dominates per-block parameters:",
            "   ~67% with MHA (Llama 2 7B), ~80%+ with GQA (70B, Llama 3, Mistral).",
            "   GQA reduces attention params (Llama 2 70B saves ~44% vs MHA by using",
            "   8 KV heads for 64 Q heads). Norms are negligible (<0.01%).",
            "",
            "3. SwiGLU Gating: The gate signal SiLU(x @ W_gate) selectively suppresses",
            "   features. SiLU is smooth (no dead neurons unlike ReLU). The gating",
            "   mechanism enables learned feature selection. 3 matrices vs 2, but",
            "   d_ff = 8/3 * d compensates for the extra parameters.",
            "",
            "4. Residual Connections: The 'gradient highway' ensures gradients never",
            "   vanish regardless of depth. Pre-norm gives d(output)/d(x) = I + ...",
            "   The identity term persists through all layers. With 1/sqrt(N) weight",
            "   scaling, norms grow moderately and gradients remain stable.",
            "",
            "5. FLOPs: Attention core is O(L^2) while FFN and projections are O(L).",
            "   For Llama 3 8B, attention core surpasses FFN at L ~ 21,296 tokens.",
            "   Formula: L_cross = 6*d*d_ff / ((4*d_k+5)*h). At short sequences,",
            "   FFN dominates; at long sequences, attention core dominates.",
            "",
            "6. Causal Masking: Position i's output depends only on positions 0..i.",
            "   Verified by showing that changing future tokens has zero effect on",
            "   past outputs. RoPE makes identical tokens position-aware through",
            "   rotation, even without any trained weights.",
        ]
        summary_text = "\n".join(summary_items)
        ax.text(0.06, 0.86, summary_text, fontsize=10, ha="left", va="top",
                transform=ax.transAxes, family="monospace", linespacing=1.3)
        pdf.savefig(fig)
        plt.close(fig)

        titles = {
            "01_forward_pass.png": "Example 1: Full Forward Pass Walkthrough",
            "02_parameter_distribution.png": "Example 2: Parameter Distribution Analysis",
            "03_swiglu_gating.png": "Example 3: SwiGLU Gating Visualization",
            "04_residual_connections.png": "Example 4: Residual Connection Gradient Highway",
            "05_flops_breakdown.png": "Example 5: FLOPs Breakdown and Crossover Analysis",
            "06_causal_position.png": "Example 6: Causal Masking and Position Sensitivity",
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
    print("Transformer Block Demo")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print()

    example_1_forward_pass()
    example_2_parameter_distribution()
    example_3_swiglu_gating()
    example_4_residual_connections()
    example_5_flops_breakdown()
    example_6_causal_and_position()
    generate_pdf_report()

    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print(f"Visualizations: {VIZ_DIR}/")
    print(f"Report: {Path(__file__).parent / 'report.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
