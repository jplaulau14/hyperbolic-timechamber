"""
KV Cache Demo -- Output equivalence verification, FLOP comparison, memory analysis,
timing benchmarks, prefill vs decode analysis, and memory bottleneck projections.

Generates:
- viz/*.png -- Individual visualization files
- report.pdf -- Comprehensive PDF report
"""

import sys
import time
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
    KVCache,
    block_forward_with_cache,
    generate_without_cache,
    generate_with_cache,
    memory_usage,
    flops_comparison,
    model_kv_cache_bytes,
    CausalLM,
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

SMALL_CFG = dict(
    vocab_size=256, d_model=64, num_layers=2, num_heads=4,
    num_kv_heads=2, d_ff=172, max_seq_len=128,
)


# ---------------------------------------------------------------------------
# Example 1: Output Equivalence Verification
# ---------------------------------------------------------------------------
def example_1_output_equivalence():
    """Prove that cached and uncached generation produce identical tokens."""
    print("=" * 60)
    print("Example 1: Output Equivalence Verification")
    print("=" * 60)

    np.random.seed(SEED)
    model = CausalLM(**SMALL_CFG)

    test_cases = [
        ("Single token prompt", np.array([[7]]), 10),
        ("Short prompt (5 tokens)", np.array([[1, 2, 3, 4, 5]]), 10),
        ("Longer prompt (8 tokens)", np.array([[10, 20, 30, 40, 50, 60, 70, 80]]), 15),
        ("Batch of 2", np.array([[1, 2, 3], [4, 5, 6]]), 8),
    ]

    all_match = True
    match_results = []

    for name, prompt, n_tokens in test_cases:
        tokens_nc, flops_nc = generate_without_cache(model, prompt, n_tokens, greedy=True, seed=0)
        tokens_c, flops_c = generate_with_cache(model, prompt, n_tokens, greedy=True, seed=0)

        match = np.array_equal(tokens_nc, tokens_c)
        all_match &= match
        match_results.append((name, match, tokens_nc, tokens_c, flops_nc, flops_c))

        status = "MATCH" if match else "MISMATCH"
        print(f"\n  {name}: {status}")
        print(f"    Prompt shape: {prompt.shape}, Generated: {n_tokens} tokens")
        print(f"    Uncached tokens: {tokens_nc[0].tolist()}")
        print(f"    Cached tokens:   {tokens_c[0].tolist()}")
        print(f"    Projection FLOPs -- uncached: {flops_nc:,}, cached: {flops_c:,}")
        print(f"    FLOP ratio: {flops_nc / flops_c:.2f}x")

    assert all_match, "Output equivalence violated"
    print(f"\n  ALL {len(test_cases)} TEST CASES MATCH -- KV cache is correct.")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    case_names = [r[0] for r in match_results]
    flops_nc_vals = [r[4] for r in match_results]
    flops_c_vals = [r[5] for r in match_results]
    ratios = [nc / c for nc, c in zip(flops_nc_vals, flops_c_vals)]

    x = np.arange(len(case_names))
    axes[0, 0].bar(x - 0.15, [f / 1e6 for f in flops_nc_vals], 0.3,
                   label="Without cache", color=COLORS["red"], edgecolor="white")
    axes[0, 0].bar(x + 0.15, [f / 1e6 for f in flops_c_vals], 0.3,
                   label="With cache", color=COLORS["green"], edgecolor="white")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([n.split("(")[0].strip() for n in case_names], fontsize=8)
    axes[0, 0].set_ylabel("Projection FLOPs (millions)")
    axes[0, 0].set_title("Projection FLOPs: Cached vs Uncached\nIdentical outputs, fewer FLOPs",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    bar_colors = [COLORS["green"]] * len(ratios)
    axes[0, 1].bar(x, ratios, 0.5, color=bar_colors, edgecolor="white")
    for i, r in enumerate(ratios):
        axes[0, 1].text(i, r + 0.1, f"{r:.1f}x", ha="center", va="bottom",
                       fontsize=10, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([n.split("(")[0].strip() for n in case_names], fontsize=8)
    axes[0, 1].set_ylabel("FLOP Ratio (uncached / cached)")
    axes[0, 1].set_title("Projection FLOP Speedup per Test Case\nMore generated tokens = larger speedup",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    _, prompt_5, n_5 = test_cases[1]
    tokens_nc_5, _ = generate_without_cache(model, prompt_5, n_5, greedy=True, seed=0)
    tokens_c_5, _ = generate_with_cache(model, prompt_5, n_5, greedy=True, seed=0)
    diff_per_pos = np.abs(tokens_nc_5[0].astype(float) - tokens_c_5[0].astype(float))
    pos_colors = [COLORS["blue"] if i < prompt_5.shape[1] else COLORS["green"]
                  for i in range(len(diff_per_pos))]
    axes[0, 2].bar(range(len(diff_per_pos)), diff_per_pos, color=pos_colors, edgecolor="white")
    axes[0, 2].set_xlabel("Position")
    axes[0, 2].set_ylabel("|uncached - cached|")
    axes[0, 2].set_title("Per-Token Absolute Difference\nAll zeros: perfect equivalence",
                         fontsize=10, fontweight="bold")
    axes[0, 2].axhline(0, color="black", linewidth=0.5)
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    prompt_8 = test_cases[2][1]
    n_gen = 15
    nc_tokens, _ = generate_without_cache(model, prompt_8, n_gen, greedy=True, seed=0)
    c_tokens, _ = generate_with_cache(model, prompt_8, n_gen, greedy=True, seed=0)

    P = prompt_8.shape[1]
    nc_generated = nc_tokens[0, P:].tolist()
    c_generated = c_tokens[0, P:].tolist()

    axes[1, 0].plot(range(n_gen), nc_generated, "o-", color=COLORS["red"],
                    markersize=6, linewidth=2, label="Without cache")
    axes[1, 0].plot(range(n_gen), c_generated, "x--", color=COLORS["green"],
                    markersize=8, linewidth=2, label="With cache")
    axes[1, 0].set_xlabel("Generation Step")
    axes[1, 0].set_ylabel("Token ID")
    axes[1, 0].set_title("Generated Token Sequence Overlay\nPerfect overlap: identical generation",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    gen_lens = [5, 10, 15, 20, 25, 30]
    nc_flops_list = []
    c_flops_list = []
    prompt_eq = np.array([[1, 2, 3, 4, 5]])
    for n in gen_lens:
        _, f_nc = generate_without_cache(model, prompt_eq, n, greedy=True, seed=0)
        _, f_c = generate_with_cache(model, prompt_eq, n, greedy=True, seed=0)
        nc_flops_list.append(f_nc)
        c_flops_list.append(f_c)

    axes[1, 1].plot(gen_lens, [f / 1e6 for f in nc_flops_list], "o-",
                    color=COLORS["red"], linewidth=2, markersize=6, label="Without cache")
    axes[1, 1].plot(gen_lens, [f / 1e6 for f in c_flops_list], "s-",
                    color=COLORS["green"], linewidth=2, markersize=6, label="With cache")
    axes[1, 1].fill_between(gen_lens, [f / 1e6 for f in c_flops_list],
                             [f / 1e6 for f in nc_flops_list],
                             alpha=0.15, color=COLORS["red"])
    axes[1, 1].set_xlabel("Tokens Generated")
    axes[1, 1].set_ylabel("Projection FLOPs (millions)")
    axes[1, 1].set_title("Actual Projection FLOPs from Generation\nShaded area = wasted computation",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    equiv_text = (
        "OUTPUT EQUIVALENCE VERIFIED\n"
        "===========================\n\n"
        f"Test cases: {len(test_cases)}\n"
        f"All match: YES\n\n"
        "What was verified:\n"
        "  - Single token prompt\n"
        "  - Short prompt (5 tokens)\n"
        "  - Longer prompt (8 tokens)\n"
        "  - Batch of 2 prompts\n\n"
        "Key insight:\n"
        "  K and V for position i depend\n"
        "  only on token i and weights W_K,\n"
        "  W_V. Future tokens don't change\n"
        "  them. Caching avoids redundant\n"
        "  recomputation without affecting\n"
        "  the output.\n\n"
        "  generate_without_cache() and\n"
        "  generate_with_cache() produce\n"
        "  BIT-IDENTICAL token sequences."
    )
    axes[1, 2].text(0.05, 0.95, equiv_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("KV Cache: Output Equivalence Verification",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_output_equivalence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/01_output_equivalence.png")


# ---------------------------------------------------------------------------
# Example 2: Theoretical FLOP Comparison
# ---------------------------------------------------------------------------
def example_2_flop_comparison():
    """Theoretical projection FLOP speedup as a function of sequence length."""
    print("\n" + "=" * 60)
    print("Example 2: Theoretical FLOP Comparison")
    print("=" * 60)

    cfg_7b = dict(n_layers=32, d_model=4096, num_heads=32, num_kv_heads=32)
    prompt_len = 512

    gen_lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    results = []

    print(f"\n  7B model config: layers={cfg_7b['n_layers']}, d_model={cfg_7b['d_model']}")
    print(f"  Prompt length: {prompt_len}")
    print(f"\n  {'Gen Tokens':>12} {'Without Cache':>18} {'With Cache':>18} {'Speedup':>10}")
    print(f"  {'-'*62}")

    for n in gen_lengths:
        r = flops_comparison(prompt_len, n, **cfg_7b)
        results.append(r)
        print(f"  {n:>12} {r['without_cache']:>18.4e} {r['with_cache']:>18.4e} {r['speedup']:>10.2f}x")

    sweep_lens = list(range(16, 4097, 16))
    sweep_speedups = []
    sweep_without = []
    sweep_with = []
    for n in sweep_lens:
        r = flops_comparison(prompt_len, n, **cfg_7b)
        sweep_speedups.append(r["speedup"])
        sweep_without.append(r["without_cache"])
        sweep_with.append(r["with_cache"])

    prompt_sweep = [32, 128, 512, 2048]
    prompt_speedup_curves = {}
    for pl in prompt_sweep:
        speedups = []
        for n in sweep_lens:
            r = flops_comparison(pl, n, **cfg_7b)
            speedups.append(r["speedup"])
        prompt_speedup_curves[pl] = speedups

    print(f"\n  At 4096 generated tokens:")
    print(f"    Speedup: {sweep_speedups[-1]:.2f}x")
    print(f"    Without cache: {sweep_without[-1]:.4e} FLOPs")
    print(f"    With cache:    {sweep_with[-1]:.4e} FLOPs")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].plot(sweep_lens, [f / 1e15 for f in sweep_without], "-",
                    color=COLORS["red"], linewidth=2, label="Without cache")
    axes[0, 0].plot(sweep_lens, [f / 1e15 for f in sweep_with], "-",
                    color=COLORS["green"], linewidth=2, label="With cache")
    axes[0, 0].fill_between(sweep_lens, [f / 1e15 for f in sweep_with],
                             [f / 1e15 for f in sweep_without],
                             alpha=0.15, color=COLORS["red"], label="Wasted projections")
    axes[0, 0].set_xlabel("Tokens Generated")
    axes[0, 0].set_ylabel("Projection FLOPs (PFLOPs)")
    axes[0, 0].set_title(f"Total Projection FLOPs (7B model, P={prompt_len})\n"
                         r"Without cache: $O(n^2 \cdot d)$, With cache: $O(n \cdot d)$",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(sweep_lens, sweep_speedups, "-", color=COLORS["purple"], linewidth=2)
    axes[0, 1].set_xlabel("Tokens Generated")
    axes[0, 1].set_ylabel("Projection FLOP Speedup")
    axes[0, 1].set_title(f"Speedup Factor (P={prompt_len})\n"
                         "Grows linearly: longer generation = larger savings",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    for pl, speedups in prompt_speedup_curves.items():
        axes[0, 2].plot(sweep_lens, speedups, "-", linewidth=1.5, label=f"P={pl}")
    axes[0, 2].set_xlabel("Tokens Generated")
    axes[0, 2].set_ylabel("Projection FLOP Speedup")
    axes[0, 2].set_title("Speedup vs Prompt Length\nLonger prompts amplify the savings",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    r_example = flops_comparison(prompt_len, 512, **cfg_7b)
    steps = range(len(r_example["without_steps"]))
    axes[1, 0].plot(list(steps), [f / 1e12 for f in r_example["without_steps"]], "-",
                    color=COLORS["red"], linewidth=1.5, label="Without cache (per step)")
    axes[1, 0].plot(list(steps), [f / 1e12 for f in r_example["decode_steps"]], "-",
                    color=COLORS["green"], linewidth=1.5, label="With cache (per decode step)")
    axes[1, 0].set_xlabel("Generation Step")
    axes[1, 0].set_ylabel("Projection FLOPs (TFLOPs)")
    axes[1, 0].set_title(f"Per-Step Projection Cost (P={prompt_len}, n=512)\n"
                         "Uncached grows linearly; cached is constant",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    cum_without = np.cumsum(r_example["without_steps"])
    cum_with = np.array([r_example["prefill_flops"]] +
                        [r_example["prefill_flops"] + sum(r_example["decode_steps"][:i+1])
                         for i in range(len(r_example["decode_steps"]))])
    axes[1, 1].plot(range(len(cum_without)), cum_without / 1e15, "-",
                    color=COLORS["red"], linewidth=2, label="Without cache (cumulative)")
    axes[1, 1].plot(range(len(cum_with)), cum_with / 1e15, "-",
                    color=COLORS["green"], linewidth=2, label="With cache (cumulative)")
    axes[1, 1].fill_between(range(len(cum_without)),
                             cum_with[:len(cum_without)] / 1e15,
                             cum_without / 1e15,
                             alpha=0.15, color=COLORS["red"])
    axes[1, 1].set_xlabel("Generation Step")
    axes[1, 1].set_ylabel("Cumulative Projection FLOPs (PFLOPs)")
    axes[1, 1].set_title(f"Cumulative Cost (P={prompt_len}, n=512)\n"
                         r"Quadratic vs linear growth",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    flop_text = (
        "PROJECTION FLOP ANALYSIS\n"
        "========================\n\n"
        "Without cache (step t):\n"
        "  Project ALL (P+t) tokens:\n"
        "  FLOPs = N * 4 * 2 * (P+t) * d^2\n"
        "  Total = N * 4 * 2 * d^2 * sum(P+i)\n"
        "        = O(n^2 * d^2)\n\n"
        "With cache (step t):\n"
        "  Project only 1 new token:\n"
        "  FLOPs = N * 4 * 2 * 1 * d^2\n"
        "  Total = prefill + n * const\n"
        "        = O(n * d^2)\n\n"
        "Speedup for large n:\n"
        "  ~ n/2 (for n >> P)\n\n"
        "Note: Attention FLOPs (QK^T, AV)\n"
        "are O(n^2) in both cases.\n"
        "The cache saves PROJECTION cost,\n"
        "which is the dominant factor\n"
        "for d_model >> seq_len."
    )
    axes[1, 2].text(0.05, 0.95, flop_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Theoretical Projection FLOP Comparison: 7B Model Scale",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_flop_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/02_flop_comparison.png")


# ---------------------------------------------------------------------------
# Example 3: Memory Analysis
# ---------------------------------------------------------------------------
def example_3_memory_analysis():
    """Cache size growth, per-token memory cost, and 7B model projections."""
    print("\n" + "=" * 60)
    print("Example 3: Memory Analysis")
    print("=" * 60)

    n_layers_7b, n_heads_7b, d_model_7b = 32, 32, 4096
    bytes_fp16 = 2

    print(f"\n  7B model: {n_layers_7b} layers, {n_heads_7b} heads, d_model={d_model_7b}, FP16")

    per_token_bytes = model_kv_cache_bytes(1, 1, n_layers_7b, n_heads_7b, d_model_7b, bytes_fp16)
    per_token_mb = per_token_bytes / (1024 ** 2)
    print(f"  Per-token KV cache cost: {per_token_bytes:,} bytes = {per_token_mb:.4f} MB")

    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    cache_sizes_gb = []
    print(f"\n  {'Seq Length':>12} {'Cache Size (GB)':>18} {'Cache Size (MB)':>18}")
    print(f"  {'-'*52}")
    for sl in seq_lens:
        total = model_kv_cache_bytes(1, sl, n_layers_7b, n_heads_7b, d_model_7b, bytes_fp16)
        gb = total / (1024 ** 3)
        mb = total / (1024 ** 2)
        cache_sizes_gb.append(gb)
        print(f"  {sl:>12} {gb:>18.3f} {mb:>18.1f}")

    model_weight_gb = 14.0  # 7B params * 2 bytes FP16
    print(f"\n  Model weights (FP16): ~{model_weight_gb:.0f} GB")
    crossover_tokens = int(model_weight_gb * 1024 / per_token_mb)
    print(f"  Cache exceeds model weights at ~{crossover_tokens:,} tokens")

    batch_sizes = [1, 4, 8, 16, 32]
    ctx_len = 4096
    print(f"\n  Batch scaling (seq_len={ctx_len}):")
    print(f"  {'Batch':>8} {'Cache (GB)':>14} {'vs Model Weights':>18}")
    print(f"  {'-'*44}")
    batch_cache_gb = []
    for bs in batch_sizes:
        total = model_kv_cache_bytes(bs, ctx_len, n_layers_7b, n_heads_7b, d_model_7b, bytes_fp16)
        gb = total / (1024 ** 3)
        batch_cache_gb.append(gb)
        ratio = gb / model_weight_gb
        print(f"  {bs:>8} {gb:>14.1f} {ratio:>17.1f}x")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].plot(seq_lens, cache_sizes_gb, "o-", color=COLORS["blue"], linewidth=2, markersize=6)
    axes[0, 0].axhline(model_weight_gb, color=COLORS["red"], linestyle="--", linewidth=2,
                        label=f"Model weights ({model_weight_gb:.0f} GB)")
    axes[0, 0].fill_between(seq_lens, cache_sizes_gb,
                             [max(c, model_weight_gb) for c in cache_sizes_gb],
                             alpha=0.0)
    above_mask = [c > model_weight_gb for c in cache_sizes_gb]
    if any(above_mask):
        first_above = next(i for i, m in enumerate(above_mask) if m)
        axes[0, 0].fill_between(seq_lens[first_above:], cache_sizes_gb[first_above:],
                                 model_weight_gb, alpha=0.2, color=COLORS["red"],
                                 label="Cache > model weights")
    axes[0, 0].set_xlabel("Sequence Length")
    axes[0, 0].set_ylabel("KV Cache Size (GB)")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_title("KV Cache Size vs Sequence Length (7B, BS=1, FP16)\n"
                         f"~{per_token_mb:.2f} MB/token, crosses model weight line at ~{crossover_tokens:,} tokens",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(range(len(batch_sizes)), batch_cache_gb, color=COLORS["blue"],
                   edgecolor="white", width=0.6)
    axes[0, 1].axhline(model_weight_gb, color=COLORS["red"], linestyle="--", linewidth=2,
                        label=f"Model weights ({model_weight_gb:.0f} GB)")
    for i, gb in enumerate(batch_cache_gb):
        axes[0, 1].text(i, gb + 0.5, f"{gb:.1f}", ha="center", va="bottom",
                       fontsize=9, fontweight="bold")
    axes[0, 1].set_xticks(range(len(batch_sizes)))
    axes[0, 1].set_xticklabels([f"BS={bs}" for bs in batch_sizes])
    axes[0, 1].set_ylabel("KV Cache Size (GB)")
    axes[0, 1].set_title(f"Cache Size vs Batch Size (seq_len={ctx_len}, FP16)\n"
                         "Batch size dominates inference memory budget",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    np.random.seed(SEED)
    cache = KVCache(n_layers=4, batch_size=1, n_kv_heads=8, d_k=32)
    steps = range(1, 51)
    mem_bytes = []
    for s in steps:
        for layer in range(4):
            k = np.random.randn(1, 8, 1, 32)
            v = np.random.randn(1, 8, 1, 32)
            cache.append(layer, k, v)
        mem_bytes.append(cache.memory_bytes())

    mem_kb = [m / 1024 for m in mem_bytes]
    axes[0, 2].plot(list(steps), mem_kb, "o-", color=COLORS["green"], linewidth=2, markersize=4)
    axes[0, 2].set_xlabel("Sequence Length (tokens)")
    axes[0, 2].set_ylabel("Cache Memory (KB)")
    axes[0, 2].set_title("Cache Memory Growth (Small Model)\n"
                         "Perfectly linear: each token adds fixed bytes",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3)

    info = memory_usage(cache)
    bytes_per_tok = info["bytes_per_token"]
    diffs = [mem_bytes[i+1] - mem_bytes[i] for i in range(len(mem_bytes)-1)]
    assert all(d == diffs[0] for d in diffs), "Memory growth is not linear"

    extended_lens = [4096, 8192, 16384, 32768, 65536, 131072]
    for bs_plot in [1, 8, 32]:
        cache_gb_extended = []
        for sl in extended_lens:
            total = model_kv_cache_bytes(bs_plot, sl, n_layers_7b, n_heads_7b, d_model_7b, bytes_fp16)
            cache_gb_extended.append(total / (1024 ** 3))
        axes[1, 0].plot(extended_lens, cache_gb_extended, "o-", markersize=5, linewidth=1.5,
                        label=f"BS={bs_plot}")
    axes[1, 0].axhline(model_weight_gb, color=COLORS["red"], linestyle="--", linewidth=2,
                        label="Model weights (14 GB)")
    axes[1, 0].axhline(80, color=COLORS["orange"], linestyle=":", linewidth=2,
                        label="A100 80GB VRAM")
    axes[1, 0].set_xlabel("Context Length")
    axes[1, 0].set_ylabel("KV Cache (GB)")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_yscale("log", base=2)
    axes[1, 0].set_title("7B Model KV Cache at Extended Context\nExceeds GPU VRAM at scale",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    ctx_for_pie = 4096
    cache_bs1 = model_kv_cache_bytes(1, ctx_for_pie, n_layers_7b, n_heads_7b, d_model_7b, bytes_fp16) / (1024**3)
    categories = ["Model Weights", "KV Cache", "Activations\n(estimate)"]
    sizes = [model_weight_gb, cache_bs1, 1.0]
    pie_colors = [COLORS["blue"], COLORS["red"], COLORS["orange"]]
    wedges, texts, autotexts = axes[1, 1].pie(
        sizes, labels=categories, colors=pie_colors,
        autopct=lambda p: f"{p:.1f}%", pctdistance=0.75, startangle=90
    )
    for t in autotexts:
        t.set_fontsize(9)
    axes[1, 1].set_title(f"GPU Memory Breakdown (7B, BS=1, ctx={ctx_for_pie})\n"
                         f"Weights={model_weight_gb:.0f}GB, Cache={cache_bs1:.1f}GB, Act~1GB",
                         fontsize=10, fontweight="bold")

    axes[1, 2].axis("off")
    mem_text = (
        "MEMORY ANALYSIS\n"
        "===============\n\n"
        "Per-token KV cache (7B, FP16):\n"
        "  2 * n_layers * n_heads * d_k * 2B\n"
        "  = 2 * 32 * 32 * 128 * 2\n"
        f"  = {per_token_bytes:,} bytes\n"
        f"  = {per_token_mb:.2f} MB/token\n\n"
        "At key context lengths:\n"
        f"  4K tokens:  {model_kv_cache_bytes(1,4096,32,32,4096,2)/(1024**3):.1f} GB\n"
        f"  32K tokens: {model_kv_cache_bytes(1,32768,32,32,4096,2)/(1024**3):.1f} GB\n"
        f"  128K tokens: {model_kv_cache_bytes(1,131072,32,32,4096,2)/(1024**3):.1f} GB\n\n"
        "With batching (ctx=4K):\n"
        f"  BS=8:  {model_kv_cache_bytes(8,4096,32,32,4096,2)/(1024**3):.1f} GB\n"
        f"  BS=32: {model_kv_cache_bytes(32,4096,32,32,4096,2)/(1024**3):.1f} GB\n\n"
        "THIS is why KV cache management\n"
        "is THE central challenge in\n"
        "LLM inference systems."
    )
    axes[1, 2].text(0.05, 0.95, mem_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("KV Cache Memory Analysis: The Inference Memory Bottleneck",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_memory_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/03_memory_analysis.png")


# ---------------------------------------------------------------------------
# Example 4: Timing Benchmark
# ---------------------------------------------------------------------------
def example_4_timing_benchmark():
    """Wall-clock speedup of cached vs uncached generation."""
    print("\n" + "=" * 60)
    print("Example 4: Timing Benchmark")
    print("=" * 60)

    np.random.seed(SEED)
    model = CausalLM(**SMALL_CFG)

    gen_counts = [5, 10, 15, 20, 25, 30]
    prompt = np.array([[1, 2, 3, 4, 5]])

    times_nc = []
    times_c = []
    speedups = []

    print(f"\n  Model: {SMALL_CFG}")
    print(f"  Prompt: {prompt[0].tolist()}")
    print(f"\n  {'Gen Tokens':>12} {'Uncached (ms)':>16} {'Cached (ms)':>14} {'Speedup':>10}")
    print(f"  {'-'*56}")

    for n in gen_counts:
        n_warmup = 1
        for _ in range(n_warmup):
            generate_without_cache(model, prompt, n_tokens=n, greedy=True, seed=0)
            generate_with_cache(model, prompt, n_tokens=n, greedy=True, seed=0)

        n_runs = 3
        t_nc_runs = []
        t_c_runs = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            generate_without_cache(model, prompt, n_tokens=n, greedy=True, seed=0)
            t_nc_runs.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            generate_with_cache(model, prompt, n_tokens=n, greedy=True, seed=0)
            t_c_runs.append(time.perf_counter() - t0)

        t_nc = np.median(t_nc_runs) * 1000
        t_c = np.median(t_c_runs) * 1000
        spd = t_nc / t_c if t_c > 0 else float("inf")

        times_nc.append(t_nc)
        times_c.append(t_c)
        speedups.append(spd)
        print(f"  {n:>12} {t_nc:>16.2f} {t_c:>14.2f} {spd:>10.2f}x")

    prefill_times = []
    decode_per_token_times = []
    for n in gen_counts:
        t0 = time.perf_counter()
        generate_with_cache(model, prompt, n_tokens=1, greedy=True, seed=0)
        t_prefill = time.perf_counter() - t0

        t0 = time.perf_counter()
        generate_with_cache(model, prompt, n_tokens=n, greedy=True, seed=0)
        t_total = time.perf_counter() - t0

        t_decode = (t_total - t_prefill) / max(n - 1, 1)
        prefill_times.append(t_prefill * 1000)
        decode_per_token_times.append(t_decode * 1000)

    print(f"\n  Prefill + Decode breakdown (cached):")
    print(f"  {'Gen Tokens':>12} {'Prefill (ms)':>14} {'Per-Decode (ms)':>16}")
    print(f"  {'-'*46}")
    for i, n in enumerate(gen_counts):
        print(f"  {n:>12} {prefill_times[i]:>14.2f} {decode_per_token_times[i]:>16.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].plot(gen_counts, times_nc, "o-", color=COLORS["red"], linewidth=2,
                    markersize=6, label="Without cache")
    axes[0, 0].plot(gen_counts, times_c, "s-", color=COLORS["green"], linewidth=2,
                    markersize=6, label="With cache")
    axes[0, 0].set_xlabel("Tokens Generated")
    axes[0, 0].set_ylabel("Wall Clock Time (ms)")
    axes[0, 0].set_title("Generation Latency: Cached vs Uncached\n"
                         "Gap widens with more tokens",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(gen_counts, speedups, "o-", color=COLORS["purple"], linewidth=2, markersize=6)
    for i, (n, s) in enumerate(zip(gen_counts, speedups)):
        axes[0, 1].annotate(f"{s:.2f}x", (n, s), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    axes[0, 1].set_xlabel("Tokens Generated")
    axes[0, 1].set_ylabel("Wall Clock Speedup")
    axes[0, 1].set_title("Measured Speedup Factor\nGrows with generation length",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    per_tok_nc = [times_nc[i] / gen_counts[i] for i in range(len(gen_counts))]
    per_tok_c = [times_c[i] / gen_counts[i] for i in range(len(gen_counts))]
    axes[0, 2].plot(gen_counts, per_tok_nc, "o-", color=COLORS["red"], linewidth=2,
                    markersize=6, label="Without cache")
    axes[0, 2].plot(gen_counts, per_tok_c, "s-", color=COLORS["green"], linewidth=2,
                    markersize=6, label="With cache")
    axes[0, 2].set_xlabel("Tokens Generated")
    axes[0, 2].set_ylabel("Time per Token (ms)")
    axes[0, 2].set_title("Per-Token Latency\nUncached grows (O(n) per step); cached stays flat",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    x = np.arange(len(gen_counts))
    axes[1, 0].bar(x - 0.15, prefill_times, 0.3, label="Prefill (1st token)",
                   color=COLORS["blue"], edgecolor="white")
    decode_totals = [decode_per_token_times[i] * max(gen_counts[i] - 1, 1) for i in range(len(gen_counts))]
    axes[1, 0].bar(x + 0.15, decode_totals, 0.3, label="Decode (remaining)",
                   color=COLORS["green"], edgecolor="white")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([str(n) for n in gen_counts])
    axes[1, 0].set_xlabel("Tokens Generated")
    axes[1, 0].set_ylabel("Time (ms)")
    axes[1, 0].set_title("Prefill vs Decode Time (Cached)\nPrefill is one-time cost; decode is per-token",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    savings_ms = [times_nc[i] - times_c[i] for i in range(len(gen_counts))]
    savings_pct = [(times_nc[i] - times_c[i]) / times_nc[i] * 100 for i in range(len(gen_counts))]
    axes[1, 1].bar(range(len(gen_counts)), savings_pct, color=COLORS["teal"], edgecolor="white")
    for i, pct in enumerate(savings_pct):
        axes[1, 1].text(i, pct + 1, f"{pct:.0f}%", ha="center", va="bottom",
                       fontsize=9, fontweight="bold")
    axes[1, 1].set_xticks(range(len(gen_counts)))
    axes[1, 1].set_xticklabels([str(n) for n in gen_counts])
    axes[1, 1].set_xlabel("Tokens Generated")
    axes[1, 1].set_ylabel("Time Saved (%)")
    axes[1, 1].set_title("Percentage of Time Saved by Caching\nApproaches theoretical maximum as n grows",
                         fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    timing_text = (
        "TIMING BENCHMARK\n"
        "================\n\n"
        f"Model: d={SMALL_CFG['d_model']}, "
        f"L={SMALL_CFG['num_layers']}, "
        f"h={SMALL_CFG['num_heads']}\n"
        f"Prompt: {prompt[0].tolist()}\n"
        f"Median of {n_runs} runs (after warmup)\n\n"
        "Observations:\n"
        "  - Cached version is consistently\n"
        "    faster for all generation lengths\n"
        "  - Speedup grows with generation\n"
        "    length (more redundancy avoided)\n"
        "  - Per-token cost is nearly constant\n"
        "    for cached (slight growth from\n"
        "    attention over longer cache)\n"
        "  - Uncached per-token cost grows\n"
        "    linearly (full recomputation)\n\n"
        "At production scale (7B model),\n"
        "the savings are even more dramatic\n"
        "because d_model^2 >> seq_len."
    )
    axes[1, 2].text(0.05, 0.95, timing_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Wall-Clock Timing Benchmark: Cached vs Uncached Generation",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_timing_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/04_timing_benchmark.png")


# ---------------------------------------------------------------------------
# Example 5: Prefill vs Decode Phase Analysis
# ---------------------------------------------------------------------------
def example_5_prefill_vs_decode():
    """Analyze the two distinct phases of KV-cached generation."""
    print("\n" + "=" * 60)
    print("Example 5: Prefill vs Decode Phase Analysis")
    print("=" * 60)

    cfg_7b = dict(n_layers=32, d_model=4096, num_heads=32, num_kv_heads=32)

    prompt_lens = [128, 256, 512, 1024, 2048, 4096]
    decode_len = 256

    print(f"\n  7B model, generating {decode_len} tokens")
    print(f"\n  {'Prompt':>8} {'Prefill FLOPs':>18} {'Decode FLOPs':>18} {'Prefill %':>12}")
    print(f"  {'-'*60}")

    prefill_flops = []
    decode_flops = []
    for pl in prompt_lens:
        r = flops_comparison(pl, decode_len, **cfg_7b)
        pf = r["prefill_flops"]
        df = r["with_cache"] - r["prefill_flops"]
        prefill_flops.append(pf)
        decode_flops.append(df)
        pf_pct = pf / r["with_cache"] * 100
        print(f"  {pl:>8} {pf:>18.4e} {df:>18.4e} {pf_pct:>11.1f}%")

    d_model = cfg_7b["d_model"]
    n_layers = cfg_7b["n_layers"]
    bytes_fp16 = 2

    print(f"\n  Prefill characteristics:")
    print(f"    - Processes all prompt tokens in parallel (batch matmul)")
    print(f"    - Compute-bound: high arithmetic intensity")
    print(f"    - Populates the KV cache for all layers")
    print(f"    - FLOPs scale linearly with prompt length")

    print(f"\n  Decode characteristics:")
    print(f"    - Processes one token at a time")
    print(f"    - Memory-bound: reads entire cache for each attention step")
    print(f"    - KV cache read per step: 2 * n_layers * seq_len * d_model * 2B")
    print(f"    - Low arithmetic intensity (few FLOPs per byte read)")

    cache_read_per_step = []
    for pl in prompt_lens:
        total_seq = pl + decode_len
        read_bytes = 2 * n_layers * total_seq * d_model * bytes_fp16
        cache_read_per_step.append(read_bytes / (1024**2))

    print(f"\n  Average cache read per decode step (MB):")
    for i, pl in enumerate(prompt_lens):
        mid_seq = pl + decode_len // 2
        mid_read = 2 * n_layers * mid_seq * d_model * bytes_fp16
        print(f"    P={pl}: ~{mid_read / (1024**2):.1f} MB")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    x = np.arange(len(prompt_lens))
    pf_tflops = [f / 1e12 for f in prefill_flops]
    df_tflops = [f / 1e12 for f in decode_flops]
    axes[0, 0].bar(x, pf_tflops, 0.6, label="Prefill", color=COLORS["blue"], edgecolor="white")
    axes[0, 0].bar(x, df_tflops, 0.6, bottom=pf_tflops, label="Decode",
                   color=COLORS["green"], edgecolor="white")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([str(pl) for pl in prompt_lens])
    axes[0, 0].set_xlabel("Prompt Length")
    axes[0, 0].set_ylabel("Projection FLOPs (TFLOPs)")
    axes[0, 0].set_title(f"Prefill vs Decode FLOPs (n={decode_len})\n"
                         "Prefill dominates for long prompts",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    prefill_pct = [pf / (pf + df) * 100 for pf, df in zip(prefill_flops, decode_flops)]
    decode_pct = [100 - p for p in prefill_pct]
    axes[0, 1].bar(x, prefill_pct, 0.6, label="Prefill %", color=COLORS["blue"], edgecolor="white")
    axes[0, 1].bar(x, decode_pct, 0.6, bottom=prefill_pct, label="Decode %",
                   color=COLORS["green"], edgecolor="white")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(pl) for pl in prompt_lens])
    axes[0, 1].set_xlabel("Prompt Length")
    axes[0, 1].set_ylabel("Percentage of Total FLOPs")
    axes[0, 1].set_title("FLOP Distribution: Prefill vs Decode\nPrefill share grows with prompt length",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    decode_lens_sweep = [64, 128, 256, 512, 1024]
    prompt_fixed = 512
    for dl in decode_lens_sweep:
        r = flops_comparison(prompt_fixed, dl, **cfg_7b)
        pf = r["prefill_flops"]
        total = r["with_cache"]
        pf_share = pf / total * 100
        axes[0, 2].scatter(dl, pf_share, s=80, zorder=5)
        axes[0, 2].annotate(f"n={dl}", (dl, pf_share), textcoords="offset points",
                           xytext=(8, 0), fontsize=8)

    dl_sweep = list(range(16, 2049, 16))
    pf_shares = []
    for dl in dl_sweep:
        r = flops_comparison(prompt_fixed, dl, **cfg_7b)
        pf_shares.append(r["prefill_flops"] / r["with_cache"] * 100)
    axes[0, 2].plot(dl_sweep, pf_shares, "-", color=COLORS["purple"], linewidth=1.5)
    axes[0, 2].set_xlabel("Decode Length (tokens generated)")
    axes[0, 2].set_ylabel("Prefill Share of Total FLOPs (%)")
    axes[0, 2].set_title(f"Prefill Share vs Decode Length (P={prompt_fixed})\n"
                         "More decoding = lower prefill share",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3)

    # Memory read per decode step
    for pl in [256, 1024, 4096]:
        read_mb = []
        for step in range(decode_len):
            seq_at_step = pl + step + 1
            read = 2 * n_layers * seq_at_step * d_model * bytes_fp16
            read_mb.append(read / (1024**2))
        axes[1, 0].plot(range(decode_len), read_mb, "-", linewidth=1.5, label=f"P={pl}")
    axes[1, 0].set_xlabel("Decode Step")
    axes[1, 0].set_ylabel("Cache Read per Step (MB)")
    axes[1, 0].set_title("Memory Bandwidth Pressure During Decode\n"
                         "Read grows linearly as cache accumulates",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Arithmetic intensity comparison
    proj_flops_per_token = n_layers * 4 * 2 * d_model * d_model
    param_bytes = n_layers * 4 * d_model * d_model * bytes_fp16

    prefill_intensities = []
    decode_intensities = []
    for pl in prompt_lens:
        pf_flops = n_layers * 4 * 2 * pl * d_model * d_model
        pf_reads = param_bytes
        pf_intensity = pf_flops / pf_reads

        mid_seq = pl + decode_len // 2
        dk_reads = 2 * n_layers * mid_seq * d_model * bytes_fp16
        dc_flops = proj_flops_per_token
        dc_intensity = dc_flops / (param_bytes + dk_reads)

        prefill_intensities.append(pf_intensity)
        decode_intensities.append(dc_intensity)

    x2 = np.arange(len(prompt_lens))
    axes[1, 1].bar(x2 - 0.15, prefill_intensities, 0.3, label="Prefill",
                   color=COLORS["blue"], edgecolor="white")
    axes[1, 1].bar(x2 + 0.15, decode_intensities, 0.3, label="Decode",
                   color=COLORS["green"], edgecolor="white")
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels([str(pl) for pl in prompt_lens])
    axes[1, 1].set_xlabel("Prompt Length")
    axes[1, 1].set_ylabel("FLOPs / Byte Read")
    axes[1, 1].set_title("Arithmetic Intensity: Prefill vs Decode\n"
                         "Prefill is compute-bound; decode is memory-bound",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].set_yscale("log")

    axes[1, 2].axis("off")
    phase_text = (
        "PREFILL vs DECODE\n"
        "=================\n\n"
        "PREFILL PHASE:\n"
        "  Input: full prompt (B, P, d)\n"
        "  Operation: batch matmul for Q,K,V\n"
        "  FLOPs: N * 4 * 2 * P * d^2\n"
        "  Bound: COMPUTE (high intensity)\n"
        "  Output: first token + full KV cache\n\n"
        "DECODE PHASE:\n"
        "  Input: single token (B, 1, d)\n"
        "  Operation: project 1 token,\n"
        "  attend over full cache\n"
        "  FLOPs: N * 4 * 2 * d^2 (proj)\n"
        "       + N * 2 * seq * d (attn)\n"
        "  Bound: MEMORY (reads full cache)\n"
        "  Output: next token\n\n"
        "Implication: Prefill can be\n"
        "parallelized across tokens;\n"
        "decode is inherently sequential.\n"
        "This is why time-to-first-token\n"
        "differs from inter-token latency."
    )
    axes[1, 2].text(0.05, 0.95, phase_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Prefill vs Decode Phase Analysis: Two Regimes of Inference",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_prefill_vs_decode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/05_prefill_vs_decode.png")


# ---------------------------------------------------------------------------
# Example 6: KV Cache as Memory Bottleneck at Scale
# ---------------------------------------------------------------------------
def example_6_memory_bottleneck():
    """Show KV cache exceeding model weight size at scale."""
    print("\n" + "=" * 60)
    print("Example 6: KV Cache as THE Memory Bottleneck")
    print("=" * 60)

    models = {
        "Llama 2 7B": dict(params_b=7, n_layers=32, n_heads=32, d_model=4096, weight_gb=14.0),
        "Llama 2 13B": dict(params_b=13, n_layers=40, n_heads=40, d_model=5120, weight_gb=26.0),
        "Llama 2 70B": dict(params_b=70, n_layers=80, n_heads=64, d_model=8192, weight_gb=140.0),
    }

    bytes_fp16 = 2
    gpu_configs = {
        "A100 40GB": 40,
        "A100 80GB": 80,
        "H100 80GB": 80,
    }

    print(f"\n  Model weight and KV cache sizes (FP16, BS=1):")
    print(f"\n  {'Model':<16} {'Weights':>10} {'Cache@4K':>12} {'Cache@32K':>12} {'Cache@128K':>14}")
    print(f"  {'-'*68}")

    for name, cfg in models.items():
        c4k = model_kv_cache_bytes(1, 4096, cfg["n_layers"], cfg["n_heads"],
                                    cfg["d_model"], bytes_fp16) / (1024**3)
        c32k = model_kv_cache_bytes(1, 32768, cfg["n_layers"], cfg["n_heads"],
                                     cfg["d_model"], bytes_fp16) / (1024**3)
        c128k = model_kv_cache_bytes(1, 131072, cfg["n_layers"], cfg["n_heads"],
                                      cfg["d_model"], bytes_fp16) / (1024**3)
        print(f"  {name:<16} {cfg['weight_gb']:>9.0f}G {c4k:>11.1f}G {c32k:>11.1f}G {c128k:>13.1f}G")

    print(f"\n  Max batch size on A100 80GB (seq_len=4096):")
    print(f"  {'Model':<16} {'Weight GB':>10} {'Cache/Seq':>12} {'Max BS':>10}")
    print(f"  {'-'*48}")
    max_bs_results = {}
    for name, cfg in models.items():
        per_seq = model_kv_cache_bytes(1, 4096, cfg["n_layers"], cfg["n_heads"],
                                        cfg["d_model"], bytes_fp16) / (1024**3)
        available = 80 - cfg["weight_gb"] - 2  # 2 GB overhead
        max_bs = int(available / per_seq) if per_seq > 0 else 0
        max_bs_results[name] = max_bs
        print(f"  {name:<16} {cfg['weight_gb']:>10.0f} {per_seq:>12.2f} {max_bs:>10}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Stacked: weights + cache at various context lengths
    model_names = list(models.keys())
    ctx_lengths = [4096, 32768, 131072]
    x = np.arange(len(model_names))
    width = 0.25

    for ci, ctx in enumerate(ctx_lengths):
        cache_gbs = []
        weight_gbs = []
        for name, cfg in models.items():
            cache_gbs.append(model_kv_cache_bytes(1, ctx, cfg["n_layers"], cfg["n_heads"],
                                                    cfg["d_model"], bytes_fp16) / (1024**3))
            weight_gbs.append(cfg["weight_gb"])

        offset = (ci - 1) * width
        axes[0, 0].bar(x + offset, weight_gbs, width, label=f"Weights" if ci == 0 else "",
                       color=COLORS["blue"], edgecolor="white", alpha=0.8)
        axes[0, 0].bar(x + offset, cache_gbs, width, bottom=weight_gbs,
                       label=f"Cache@{ctx//1024}K" if True else "",
                       color=[COLORS["green"], COLORS["orange"], COLORS["red"]][ci],
                       edgecolor="white", alpha=0.8)

    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, fontsize=9)
    axes[0, 0].set_ylabel("GPU Memory (GB)")
    axes[0, 0].set_title("Model Weights + KV Cache (BS=1, FP16)\nCache dominates at long contexts",
                         fontsize=10, fontweight="bold")
    axes[0, 0].axhline(80, color="black", linestyle=":", linewidth=1.5, label="A100 80GB")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Cache vs model weights ratio
    ctx_sweep = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    for name, cfg in models.items():
        ratios = []
        for ctx in ctx_sweep:
            cache_gb = model_kv_cache_bytes(1, ctx, cfg["n_layers"], cfg["n_heads"],
                                             cfg["d_model"], bytes_fp16) / (1024**3)
            ratios.append(cache_gb / cfg["weight_gb"])
        axes[0, 1].plot(ctx_sweep, ratios, "o-", markersize=4, linewidth=1.5, label=name)
    axes[0, 1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="Cache = Weights")
    axes[0, 1].set_xlabel("Context Length")
    axes[0, 1].set_ylabel("Cache / Model Weights Ratio")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_yscale("log", base=2)
    axes[0, 1].set_title("KV Cache / Model Weights Ratio (BS=1)\nCrossover point depends on model size",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Batch scaling: total memory
    cfg_7b = models["Llama 2 7B"]
    batch_sizes = np.arange(1, 65)
    ctx_for_batch = 4096
    total_mem = []
    for bs in batch_sizes:
        cache_gb = model_kv_cache_bytes(int(bs), ctx_for_batch, cfg_7b["n_layers"], cfg_7b["n_heads"],
                                         cfg_7b["d_model"], bytes_fp16) / (1024**3)
        total_mem.append(cfg_7b["weight_gb"] + cache_gb)
    axes[0, 2].plot(batch_sizes, total_mem, "-", color=COLORS["blue"], linewidth=2)
    axes[0, 2].axhline(40, color=COLORS["orange"], linestyle="--", linewidth=1.5, label="A100 40GB")
    axes[0, 2].axhline(80, color=COLORS["red"], linestyle="--", linewidth=1.5, label="A100 80GB")
    axes[0, 2].fill_between(batch_sizes, total_mem, 80,
                             where=[t <= 80 for t in total_mem],
                             alpha=0.1, color=COLORS["green"])
    axes[0, 2].set_xlabel("Batch Size")
    axes[0, 2].set_ylabel("Total GPU Memory (GB)")
    axes[0, 2].set_title(f"Total Memory vs Batch Size (7B, ctx={ctx_for_batch})\n"
                         "KV cache quickly fills available VRAM",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Memory composition pie at different scales
    compositions = [
        ("7B, BS=1, 4K", models["Llama 2 7B"], 1, 4096),
        ("7B, BS=16, 4K", models["Llama 2 7B"], 16, 4096),
        ("7B, BS=1, 128K", models["Llama 2 7B"], 1, 131072),
    ]

    for idx, (label, cfg, bs, ctx) in enumerate(compositions):
        ax = axes[1, idx]
        cache_gb = model_kv_cache_bytes(bs, ctx, cfg["n_layers"], cfg["n_heads"],
                                         cfg["d_model"], bytes_fp16) / (1024**3)
        act_gb = 1.0
        sizes = [cfg["weight_gb"], cache_gb, act_gb]
        labels = [f"Weights\n{cfg['weight_gb']:.0f}GB", f"KV Cache\n{cache_gb:.1f}GB",
                  f"Activations\n~{act_gb:.0f}GB"]
        colors = [COLORS["blue"], COLORS["red"], COLORS["orange"]]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct=lambda p: f"{p:.0f}%" if p > 2 else "",
            pctdistance=0.75, startangle=90
        )
        for t in autotexts:
            t.set_fontsize(8)
        for t in texts:
            t.set_fontsize(8)
        total = sum(sizes)
        ax.set_title(f"{label}\nTotal: {total:.1f} GB",
                     fontsize=10, fontweight="bold")

    fig.suptitle("KV Cache: The Memory Bottleneck That Defines Inference Systems",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_memory_bottleneck.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/06_memory_bottleneck.png")

    print(f"\n  This motivates all major inference optimizations:")
    print(f"    - PagedAttention (vLLM): manage cache like virtual memory")
    print(f"    - GQA/MQA: share K/V heads to reduce cache size")
    print(f"    - KV cache quantization: FP8/INT8 for 2x reduction")
    print(f"    - Sliding window: bound cache at O(window_size)")


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
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.78, "KV Cache", fontsize=28, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.68, "The Memory Bottleneck That Defines Inference Optimization",
                fontsize=13, ha="center", va="center", transform=ax.transAxes, color="gray")
        info_text = (
            "KV caching stores key and value tensors from previous token positions\n"
            "so they don't need to be recomputed during autoregressive generation.\n"
            "Without caching, generating n tokens requires O(n^2) projection FLOPs.\n"
            "With KV cache, the projection cost drops to O(n) -- each step projects\n"
            "only the new token, appends K and V to the cache, and attends over the\n"
            "full cached history.\n\n"
            "This demo covers:\n"
            "  1. Output equivalence: cached = uncached (bit-identical tokens)\n"
            "  2. Theoretical FLOP comparison at 7B model scale\n"
            "  3. Memory analysis: per-token cost, growth, 7B projections\n"
            "  4. Wall-clock timing benchmark\n"
            "  5. Prefill vs decode phase analysis\n"
            "  6. KV cache as THE memory bottleneck at production scale\n\n"
            f"Model config: V=256, d=64, layers=2, h=4, h_kv=2, d_ff=172\n"
            f"Random seed: {SEED}\n"
            f"Number of visualizations: {len(viz_files)}\n"
            "Examples: 6"
        )
        ax.text(0.5, 0.30, info_text, fontsize=11, ha="center", va="center",
                transform=ax.transAxes, linespacing=1.6)
        ax.text(0.5, 0.06, "Generated by demo.py", fontsize=10, ha="center",
                va="center", transform=ax.transAxes, style="italic", color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        # Math page with LaTeX equations
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.96, "Mathematical Foundation", fontsize=20, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)

        y = 0.88
        dy = 0.045

        ax.text(0.05, y, "Without KV Cache (Naive)", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y, r"At step $t$, recompute all projections for $[0, \ldots, t]$:",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$Q = X W_Q \in \mathbb{R}^{(t+1) \times d_k}, \quad K = X W_K, \quad V = X W_V$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{scores} = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{(t+1) \times (t+1)}$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Projection FLOPs: $\sum_{i=1}^{n} i \cdot 3 \cdot d \cdot d_k = O(n^2 \cdot d)$",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.5
        ax.text(0.05, y, "With KV Cache", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y, r"At step $t$, project only the new token $x_t$:",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$q_t = x_t W_Q \in \mathbb{R}^{1 \times d_k}, \quad k_t = x_t W_K, \quad v_t = x_t W_V$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$K_{c} = \mathrm{concat}(K_{c},\; k_t) \in \mathbb{R}^{(t+1) \times d_k}$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{scores} = \frac{q_t\; K_{c}^\top}{\sqrt{d_k}} \in \mathbb{R}^{1 \times (t+1)}$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Projection FLOPs: $n \cdot 3 \cdot d \cdot d_k = O(n \cdot d)$",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.5
        ax.text(0.05, y, "Memory Analysis", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{cache/layer} = 2 \cdot \mathrm{seqlen} \cdot d_k \cdot \mathrm{bytes}$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{total} = n_{\mathrm{layers}} \cdot n_{\mathrm{heads}} \cdot 2 \cdot \mathrm{seqlen} \cdot d_k \cdot \mathrm{bytes}$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"7B model (FP16): $32 \times 32 \times 2 \times S \times 128 \times 2 = 524,288 \cdot S$ bytes",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\approx 0.5$ MB/token $\Rightarrow$ 4K ctx: 2 GB, 32K ctx: 16 GB, 128K ctx: 64 GB",
                fontsize=10, transform=ax.transAxes)

        y -= dy * 1.5
        ax.text(0.05, y, "Speedup", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Projection speedup $= \frac{\sum_{i=1}^{n}(P+i)}{P + n} \approx \frac{n}{2}$ for $n \gg P$",
                fontsize=11, transform=ax.transAxes)

        pdf.savefig(fig)
        plt.close(fig)

        # Summary page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.94, "Summary of Findings", fontsize=20, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)

        summary_items = [
            "1. Output Equivalence: Cached and uncached generation produce BIT-IDENTICAL",
            "   token sequences across all tested configurations (single token, short/long",
            "   prompts, batched). K and V for position i depend only on token i and the",
            "   fixed weights -- caching avoids redundant recomputation without error.",
            "",
            "2. FLOP Comparison: At 7B scale (32 layers, d=4096), projection FLOPs drop",
            "   from O(n^2*d^2) to O(n*d^2). For 4096 generated tokens with P=512 prompt,",
            "   this yields >1000x speedup in projection cost alone.",
            "",
            "3. Memory Analysis: KV cache costs ~0.5 MB/token for a 7B model (FP16).",
            "   At 4K context: 2 GB. At 32K context: 16 GB. At 128K context: 64 GB.",
            "   With batch_size=32 and 4K context: 64 GB -- exceeding model weights (14 GB).",
            "",
            "4. Timing Benchmark: Wall-clock measurements on our small model confirm",
            "   speedup that grows with generation length. Per-token latency is nearly",
            "   constant with cache; grows linearly without cache.",
            "",
            "5. Prefill vs Decode: Prefill is compute-bound (high arithmetic intensity --",
            "   batch matmul over all prompt tokens). Decode is memory-bound (reads entire",
            "   cache for each single-token attention step). This two-regime nature drives",
            "   architecture decisions in inference systems (separate prefill/decode GPUs).",
            "",
            "6. Memory Bottleneck: For long contexts and large batches, KV cache EXCEEDS",
            "   model weight memory. A 7B model in FP16 is ~14 GB, but KV cache for",
            "   batch_size=32 at 8K context is 128 GB. This motivates PagedAttention,",
            "   GQA/MQA, KV cache quantization, and sliding window attention.",
        ]
        summary_text = "\n".join(summary_items)
        ax.text(0.06, 0.86, summary_text, fontsize=10, ha="left", va="top",
                transform=ax.transAxes, family="monospace", linespacing=1.3)
        pdf.savefig(fig)
        plt.close(fig)

        # Visualization pages
        titles = {
            "01_output_equivalence.png": "Example 1: Output Equivalence Verification",
            "02_flop_comparison.png": "Example 2: Theoretical FLOP Comparison",
            "03_memory_analysis.png": "Example 3: Memory Analysis",
            "04_timing_benchmark.png": "Example 4: Timing Benchmark",
            "05_prefill_vs_decode.png": "Example 5: Prefill vs Decode Phase Analysis",
            "06_memory_bottleneck.png": "Example 6: KV Cache as THE Memory Bottleneck",
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

    print(f"  Report saved: report.pdf ({len(viz_files) + 3} pages)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("KV Cache Demo")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print(f"Model config: {SMALL_CFG}")
    print()

    example_1_output_equivalence()
    example_2_flop_comparison()
    example_3_memory_analysis()
    example_4_timing_benchmark()
    example_5_prefill_vs_decode()
    example_6_memory_bottleneck()
    generate_pdf_report()

    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print(f"Visualizations: {VIZ_DIR}/")
    print(f"Report: {Path(__file__).parent / 'report.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
