"""
Causal Decoding Demo -- Full forward pass, causal property verification, sampling strategies,
autoregressive generation, computational cost analysis, and model parameter breakdown.

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
    CausalLM,
    sample_token,
    temperature_scale,
    top_k_filter,
    top_p_filter,
    softmax,
    count_model_parameters,
    generation_flops,
    generation_flops_with_cache,
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
# Example 1: Full Forward Pass Walkthrough
# ---------------------------------------------------------------------------
def example_1_forward_pass():
    """Create a small CausalLM and trace shapes at every stage."""
    print("=" * 60)
    print("Example 1: Full Forward Pass Walkthrough")
    print("=" * 60)

    np.random.seed(SEED)
    model = CausalLM(**SMALL_CFG)

    B, L = 1, 10
    token_ids = np.random.randint(0, SMALL_CFG["vocab_size"], (B, L))

    print(f"\n  Config: V={SMALL_CFG['vocab_size']}, d={SMALL_CFG['d_model']}, "
          f"layers={SMALL_CFG['num_layers']}, h={SMALL_CFG['num_heads']}, "
          f"h_kv={SMALL_CFG['num_kv_heads']}, d_ff={SMALL_CFG['d_ff']}")
    print(f"  Input: token_ids.shape = {token_ids.shape}")
    print(f"  Token IDs: {token_ids[0].tolist()}")

    x = model.embedding[token_ids]
    print(f"\n  Stage 1 - Embedding lookup: {token_ids.shape} -> {x.shape}")

    from implementation import create_causal_mask
    mask = create_causal_mask(L)
    positions = np.arange(L)

    h = x.copy()
    for i, block in enumerate(model.blocks):
        h = block.forward(h, mask=mask, positions=positions)
        print(f"  Stage 2.{i+1} - After block {i}: {h.shape}, norm={np.linalg.norm(h):.4f}")

    h_normed = model.final_norm.forward(h)
    print(f"  Stage 3 - After final RMSNorm: {h_normed.shape}, norm={np.linalg.norm(h_normed):.4f}")

    logits = h_normed @ model.W_out
    print(f"  Stage 4 - Output logits: {h_normed.shape} @ {model.W_out.shape} -> {logits.shape}")

    logits_check = model.forward(token_ids)
    assert logits_check.shape == (B, L, SMALL_CFG["vocab_size"])
    print(f"\n  model.forward() output shape: {logits_check.shape}")

    last_logits = logits_check[0, -1, :]
    probs = softmax(last_logits[np.newaxis, :], axis=-1)[0]
    print(f"  Last-position logits -> softmax -> probabilities:")
    print(f"    sum(probs) = {probs.sum():.10f} (should be 1.0)")
    print(f"    max prob = {probs.max():.6f} at token {np.argmax(probs)}")
    print(f"    min prob = {probs.min():.6e}")

    if model.tie_weights:
        shares_memory = np.shares_memory(model.embedding, model.W_out)
        print(f"\n  Weight tying: model.W_out shares memory with model.embedding.T: {shares_memory}")
        print(f"    embedding.shape = {model.embedding.shape}, W_out.shape = {model.W_out.shape}")
        assert shares_memory, "Weight tying should share memory"
    else:
        print(f"\n  Weight tying disabled: W_out is independent")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    stages = ["Input\nIDs", "Embed\nlookup"]
    norms = [np.linalg.norm(token_ids.astype(float)), np.linalg.norm(x)]
    h_trace = x.copy()
    for i, block in enumerate(model.blocks):
        h_trace = block.forward(h_trace, mask=mask, positions=positions)
        stages.append(f"Block {i}")
        norms.append(np.linalg.norm(h_trace))
    h_normed_trace = model.final_norm.forward(h_trace)
    stages.append("Final\nRMSNorm")
    norms.append(np.linalg.norm(h_normed_trace))
    logits_trace = h_normed_trace @ model.W_out
    stages.append("Logits")
    norms.append(np.linalg.norm(logits_trace))

    stage_colors = [COLORS["dark"]] + [COLORS["blue"]] + [COLORS["green"]] * SMALL_CFG["num_layers"] + \
                   [COLORS["orange"]] + [COLORS["red"]]
    axes[0, 0].bar(range(len(stages)), norms, color=stage_colors, edgecolor="white")
    axes[0, 0].set_xticks(range(len(stages)))
    axes[0, 0].set_xticklabels(stages, fontsize=8)
    axes[0, 0].set_ylabel("Frobenius Norm")
    axes[0, 0].set_title("Tensor Norms Through the Model\nData flows: IDs -> Embed -> Blocks -> Norm -> Logits",
                         fontsize=10, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    shapes_text = (
        f"COMPLETE SHAPE TABLE\n"
        f"====================\n\n"
        f"Token IDs:      ({B}, {L})\n"
        f"Embedding:      ({B}, {L}, {SMALL_CFG['d_model']})\n"
        f"After block i:  ({B}, {L}, {SMALL_CFG['d_model']})\n"
        f"Final RMSNorm:  ({B}, {L}, {SMALL_CFG['d_model']})\n"
        f"Logits:         ({B}, {L}, {SMALL_CFG['vocab_size']})\n\n"
        f"Next-token probs:\n"
        f"  logits[:, -1, :] -> softmax\n"
        f"  Shape: ({B}, {SMALL_CFG['vocab_size']})\n\n"
        f"Weight tying:\n"
        f"  E.shape = ({SMALL_CFG['vocab_size']}, {SMALL_CFG['d_model']})\n"
        f"  W_out = E.T = ({SMALL_CFG['d_model']}, {SMALL_CFG['vocab_size']})\n"
        f"  logits = x_norm @ E.T\n"
        f"  Shares memory: {shares_memory}"
    )
    axes[0, 1].axis("off")
    axes[0, 1].text(0.05, 0.95, shapes_text, fontsize=10, ha="left", va="top",
                    family="monospace", transform=axes[0, 1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    sorted_probs = np.sort(probs)[::-1]
    top_n = 20
    axes[0, 2].bar(range(top_n), sorted_probs[:top_n], color=COLORS["blue"], edgecolor="white")
    axes[0, 2].set_xlabel("Rank")
    axes[0, 2].set_ylabel("Probability")
    axes[0, 2].set_title(f"Next-Token Distribution (Top {top_n})\n"
                         f"sum = {probs.sum():.6f}, max = {probs.max():.6f}",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    embed_norms = np.linalg.norm(model.embedding, axis=1)
    axes[1, 0].hist(embed_norms, bins=30, color=COLORS["steel"], edgecolor="white", alpha=0.8)
    axes[1, 0].axvline(embed_norms.mean(), color=COLORS["red"], linestyle="--",
                        label=f"mean = {embed_norms.mean():.3f}")
    axes[1, 0].set_xlabel("Embedding Row Norm")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"Embedding Matrix Row Norms (Xavier init)\n"
                         f"V={SMALL_CFG['vocab_size']}, d={SMALL_CFG['d_model']}",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    logits_per_pos = logits_check[0]
    axes[1, 1].imshow(logits_per_pos[:, :50].T, aspect="auto", cmap="RdBu_r",
                       vmin=-np.abs(logits_per_pos).max() * 0.5,
                       vmax=np.abs(logits_per_pos).max() * 0.5)
    axes[1, 1].set_xlabel("Sequence Position")
    axes[1, 1].set_ylabel("Vocab Token (first 50)")
    axes[1, 1].set_title("Logits Heatmap (all positions)\nEach column is a next-token prediction",
                         fontsize=10, fontweight="bold")

    axes[1, 2].axis("off")
    arch_text = (
        "ARCHITECTURE SUMMARY\n"
        "====================\n\n"
        "CausalLM(\n"
        f"  embedding:  E[{SMALL_CFG['vocab_size']}, {SMALL_CFG['d_model']}]\n"
        f"  blocks:     {SMALL_CFG['num_layers']} x TransformerBlock(\n"
        f"                d={SMALL_CFG['d_model']}, h={SMALL_CFG['num_heads']},\n"
        f"                h_kv={SMALL_CFG['num_kv_heads']}, d_ff={SMALL_CFG['d_ff']}\n"
        f"              )\n"
        f"  final_norm: RMSNorm({SMALL_CFG['d_model']})\n"
        f"  W_out:      E.T (weight tying)\n"
        f")\n\n"
        f"Forward: E[tokens] -> blocks -> norm -> logits\n"
        f"Generate: forward -> sample -> append -> repeat\n\n"
        f"CAVEAT: Random weights produce\n"
        f"random (meaningless) distributions.\n"
        f"Trained models learn meaningful\n"
        f"next-token predictions."
    )
    axes[1, 2].text(0.05, 0.95, arch_text, fontsize=10, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Causal Decoding: Full Forward Pass Walkthrough",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_forward_pass.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/01_forward_pass.png")


# ---------------------------------------------------------------------------
# Example 2: Causal Property Verification (CENTERPIECE)
# ---------------------------------------------------------------------------
def example_2_causal_property():
    """THE key property: future tokens do not affect past logits."""
    print("\n" + "=" * 60)
    print("Example 2: Causal Property Verification (CENTERPIECE)")
    print("=" * 60)

    np.random.seed(SEED)
    model = CausalLM(**SMALL_CFG)

    seq_a = np.array([[10, 20, 30, 40]])
    seq_b = np.array([[10, 20, 30, 99]])

    logits_a = model.forward(seq_a)
    logits_b = model.forward(seq_b)

    print(f"\n  Sequence A: {seq_a[0].tolist()}")
    print(f"  Sequence B: {seq_b[0].tolist()}")
    print(f"  Shared prefix: positions 0, 1, 2 (tokens 10, 20, 30)")
    print(f"  Differ at position 3: A has 40, B has 99")

    print(f"\n  ANALYTICAL JUSTIFICATION:")
    print(f"  The causal mask inside each attention layer ensures:")
    print(f"    output[i] = f(token_0, token_1, ..., token_i)")
    print(f"  Position i's output is computed using ONLY positions 0..i.")
    print(f"  The attention weight matrix has zeros above the diagonal,")
    print(f"  so token_3 (or any later token) contributes zero to output[0..2].")

    print(f"\n  {'Position':>10} {'||logits_A - logits_B||':>25} {'Match?':>10}")
    print(f"  {'-'*50}")

    pos_diffs = []
    for i in range(4):
        diff = np.linalg.norm(logits_a[0, i] - logits_b[0, i])
        pos_diffs.append(diff)
        match = "EXACT" if diff < 1e-12 else "DIFFERS"
        print(f"  {i:>10} {diff:>25.2e} {match:>10}")

    for i in range(3):
        assert pos_diffs[i] < 1e-12, f"Position {i} should be identical"
    assert pos_diffs[3] > 1e-6, "Position 3 should differ"

    print(f"\n  Positions 0-2: IDENTICAL (future token 40 vs 99 is invisible)")
    print(f"  Position 3: DIFFERS (it sees different tokens at position 3)")
    print(f"  This is THE defining property of causal (autoregressive) models.")

    # Additional test: single token vs prefix of longer sequence
    seq_single = np.array([[10]])
    logits_single = model.forward(seq_single)
    logit_pos0_from_single = logits_single[0, 0]
    logit_pos0_from_full = logits_a[0, 0]
    single_diff = np.linalg.norm(logit_pos0_from_single - logit_pos0_from_full)

    print(f"\n  Extra verification: forward([10]) vs forward([10, 20, 30, 40])[pos 0]")
    print(f"  ||diff|| = {single_diff:.2e} (should be ~0: position 0 only sees itself)")
    assert single_diff < 1e-12

    # Test with multiple different suffixes
    suffix_tokens = [40, 99, 0, 255, 128]
    all_pos0_logits = []
    all_pos2_logits = []
    for tok in suffix_tokens:
        seq = np.array([[10, 20, 30, tok]])
        logits = model.forward(seq)
        all_pos0_logits.append(logits[0, 0].copy())
        all_pos2_logits.append(logits[0, 2].copy())

    pos0_variation = max(np.linalg.norm(all_pos0_logits[i] - all_pos0_logits[0])
                         for i in range(len(suffix_tokens)))
    pos2_variation = max(np.linalg.norm(all_pos2_logits[i] - all_pos2_logits[0])
                         for i in range(len(suffix_tokens)))

    print(f"\n  Varying position 3 token across {suffix_tokens}:")
    print(f"    Position 0 max variation: {pos0_variation:.2e}")
    print(f"    Position 2 max variation: {pos2_variation:.2e}")
    print(f"    (Both ~0: positions 0-2 never see position 3)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    bar_colors = [COLORS["green"]] * 3 + [COLORS["red"]]
    axes[0, 0].bar(range(4), pos_diffs, color=bar_colors, edgecolor="white", width=0.6)
    axes[0, 0].set_xticks(range(4))
    axes[0, 0].set_xticklabels([f"pos {i}" for i in range(4)])
    axes[0, 0].set_ylabel("||logits_A - logits_B||")
    axes[0, 0].set_yscale("symlog", linthresh=1e-14)
    axes[0, 0].axhline(1e-12, color="gray", linestyle="--", alpha=0.5, label="1e-12 threshold")
    axes[0, 0].annotate("IDENTICAL\n(future invisible)", xy=(1, max(pos_diffs) * 0.1),
                        fontsize=9, ha="center", color=COLORS["green"], fontweight="bold")
    axes[0, 0].annotate("DIFFERS\n(sees different token)", xy=(3, pos_diffs[3] * 1.5),
                        fontsize=9, ha="center", color=COLORS["red"], fontweight="bold")
    axes[0, 0].set_title("Causal Property: [10,20,30,40] vs [10,20,30,99]\n"
                         "Positions 0-2 are IDENTICAL regardless of position 3",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    from implementation import create_causal_mask
    L = 8
    mask = create_causal_mask(L)
    mask_visual = np.where(mask.squeeze() == 0, 1.0, 0.0)
    im1 = axes[0, 1].imshow(mask_visual, cmap="Blues", aspect="equal")
    axes[0, 1].set_xlabel("Key Position (what is attended to)")
    axes[0, 1].set_ylabel("Query Position (who is attending)")
    axes[0, 1].set_title("Causal Mask (L=8)\n1 = attend, 0 = blocked (future)",
                         fontsize=10, fontweight="bold")
    for i in range(L):
        for j in range(L):
            color = "white" if mask_visual[i, j] > 0.5 else "gray"
            axes[0, 1].text(j, i, f"{mask_visual[i,j]:.0f}", ha="center", va="center",
                           fontsize=7, color=color)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    probs_a = softmax(logits_a[0], axis=-1)
    probs_b = softmax(logits_b[0], axis=-1)
    prob_diff = np.abs(probs_a - probs_b)
    im2 = axes[0, 2].imshow(prob_diff.T[:50, :], aspect="auto", cmap="Reds",
                              interpolation="nearest")
    axes[0, 2].set_xlabel("Sequence Position")
    axes[0, 2].set_ylabel("Vocab Token (first 50)")
    axes[0, 2].set_title("|P(A) - P(B)| Probability Difference\nOnly position 3 differs",
                         fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    top_k_show = 15
    probs_a_last = softmax(logits_a[0, -1:], axis=-1)[0]
    probs_b_last = softmax(logits_b[0, -1:], axis=-1)[0]
    top_a = np.argsort(probs_a_last)[::-1][:top_k_show]
    top_b = np.argsort(probs_b_last)[::-1][:top_k_show]

    x_pos = np.arange(top_k_show)
    axes[1, 0].bar(x_pos - 0.15, probs_a_last[top_a], 0.3, label="Seq A (token 40)",
                   color=COLORS["blue"], edgecolor="white")
    axes[1, 0].bar(x_pos + 0.15, probs_b_last[top_a], 0.3, label="Seq B (token 99)",
                   color=COLORS["red"], edgecolor="white")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([str(t) for t in top_a], fontsize=7)
    axes[1, 0].set_xlabel("Token ID")
    axes[1, 0].set_ylabel("Probability")
    axes[1, 0].set_title("Position 3 Next-Token Distributions Differ\n"
                         "Different input at pos 3 -> different prediction for pos 4",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    pos2_probs_a = softmax(logits_a[0, 2:3], axis=-1)[0]
    pos2_probs_b = softmax(logits_b[0, 2:3], axis=-1)[0]
    top_2 = np.argsort(pos2_probs_a)[::-1][:top_k_show]
    axes[1, 1].bar(x_pos - 0.15, pos2_probs_a[top_2], 0.3, label="Seq A",
                   color=COLORS["blue"], edgecolor="white")
    axes[1, 1].bar(x_pos + 0.15, pos2_probs_b[top_2], 0.3, label="Seq B",
                   color=COLORS["red"], edgecolor="white")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([str(t) for t in top_2], fontsize=7)
    axes[1, 1].set_xlabel("Token ID")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].set_title("Position 2 Distributions are IDENTICAL\n"
                         "Bars overlap perfectly (future token invisible)",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    causal_text = (
        "THE CAUSAL PROPERTY\n"
        "===================\n\n"
        "For sequences sharing prefix [0..k-1]:\n"
        "  logits[0..k-1] are IDENTICAL\n"
        "  regardless of tokens at positions k+.\n\n"
        "WHY: The causal attention mask zeros\n"
        "out all future positions. Position i\n"
        "can only attend to positions 0..i.\n\n"
        "CONSEQUENCE: We can generate left-\n"
        "to-right, one token at a time.\n"
        "Each new token depends only on the\n"
        "prefix, never on future tokens.\n\n"
        "This enables autoregressive generation:\n"
        "  1. Forward pass on prompt\n"
        "  2. Sample next token from last logits\n"
        "  3. Append token, repeat\n\n"
        "VERIFIED:\n"
        f"  Prefix positions: max diff = {max(pos_diffs[:3]):.1e}\n"
        f"  Divergent position: diff = {pos_diffs[3]:.4f}\n"
        f"  Single vs full seq: diff = {single_diff:.1e}"
    )
    axes[1, 2].text(0.05, 0.95, causal_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Causal Property Verification: Future Tokens Are Invisible",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_causal_property.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/02_causal_property.png")


# ---------------------------------------------------------------------------
# Example 3: Sampling Strategy Comparison
# ---------------------------------------------------------------------------
def example_3_sampling_strategies():
    """Compare greedy, temperature, top-k, top-p, and combined sampling."""
    print("\n" + "=" * 60)
    print("Example 3: Sampling Strategy Comparison")
    print("=" * 60)

    np.random.seed(SEED)
    model = CausalLM(**SMALL_CFG)

    prompt = np.array([[5, 10, 15, 20, 25]])
    logits = model.forward(prompt)
    raw_logits = logits[0, -1, :].copy()

    print(f"\n  Prompt: {prompt[0].tolist()}")
    print(f"  Raw logits: shape={raw_logits.shape}, range=[{raw_logits.min():.4f}, {raw_logits.max():.4f}]")

    strategies = {
        "Greedy": dict(greedy=True),
        "Temp=0.3": dict(temperature=0.3),
        "Temp=1.0": dict(temperature=1.0),
        "Temp=1.5": dict(temperature=1.5),
        "Top-k=5": dict(top_k=5),
        "Top-p=0.9": dict(top_p=0.9),
        "Combined": dict(temperature=0.8, top_k=10, top_p=0.95),
    }

    print(f"\n  Sampling from same logits with different strategies:")
    print(f"  {'Strategy':<15} {'Token':>8} {'Max P':>10} {'Nonzero':>10} {'Entropy':>10}")
    print(f"  {'-'*58}")

    strategy_probs = {}
    strategy_tokens = {}

    for name, kwargs in strategies.items():
        logits_2d = raw_logits[np.newaxis, :].copy()

        if name == "Greedy":
            tok = np.argmax(logits_2d, axis=-1)[0]
            p = softmax(logits_2d, axis=-1)[0]
        else:
            scaled = logits_2d.copy()
            temp = kwargs.get("temperature", 1.0)
            scaled = temperature_scale(scaled, temp)

            if "top_k" in kwargs and kwargs["top_k"] > 0:
                scaled = top_k_filter(scaled, kwargs["top_k"])
            if "top_p" in kwargs and kwargs["top_p"] < 1.0:
                scaled = top_p_filter(scaled, kwargs["top_p"])

            p = softmax(scaled, axis=-1)[0]
            rng = np.random.RandomState(SEED)
            tok = rng.choice(len(p), p=p)

        nonzero = np.sum(p > 1e-10)
        entropy = -np.sum(p[p > 1e-10] * np.log(p[p > 1e-10]))
        strategy_probs[name] = p
        strategy_tokens[name] = tok
        print(f"  {name:<15} {tok:>8} {p.max():>10.6f} {nonzero:>10} {entropy:>10.4f}")

    # Generate sequences with each strategy
    gen_len = 8
    print(f"\n  Generating {gen_len} tokens from prompt {prompt[0].tolist()}:")
    gen_results = {}
    for name, kwargs in strategies.items():
        if name == "Greedy":
            gen = model.generate(prompt.copy(), max_new_tokens=gen_len, greedy=True, seed=SEED)
        else:
            gen = model.generate(prompt.copy(), max_new_tokens=gen_len, seed=SEED, **kwargs)
        gen_results[name] = gen[0].tolist()
        print(f"  {name:<15}: {gen[0].tolist()}")

    # Diversity: run non-greedy strategies 5 times with different seeds
    print(f"\n  Token diversity (5 runs per strategy, position prompt_len+0):")
    for name, kwargs in strategies.items():
        if name == "Greedy":
            continue
        tokens_seen = set()
        for s in range(5):
            rng = np.random.RandomState(SEED + s)
            logits_2d = raw_logits[np.newaxis, :].copy()
            t = sample_token(logits_2d, rng=rng, **kwargs)
            tokens_seen.add(int(np.asarray(t).flat[0]))
        print(f"    {name:<15}: {len(tokens_seen)} unique tokens from 5 samples")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Raw vs temperature-scaled distributions
    sorted_raw = np.sort(softmax(raw_logits[np.newaxis, :], axis=-1)[0])[::-1]
    for tname, temp in [("T=0.3", 0.3), ("T=1.0", 1.0), ("T=1.5", 1.5)]:
        scaled = temperature_scale(raw_logits[np.newaxis, :].copy(), temp)
        sorted_scaled = np.sort(softmax(scaled, axis=-1)[0])[::-1]
        axes[0, 0].plot(range(30), sorted_scaled[:30], "o-", markersize=4, linewidth=1.5,
                        label=tname)
    axes[0, 0].set_xlabel("Rank")
    axes[0, 0].set_ylabel("Probability")
    axes[0, 0].set_title("Temperature Effect on Distribution (Top 30)\n"
                         "Low T -> sharp (greedy-like), High T -> flat (random)",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-k filtering
    for k_val in [3, 5, 10, 20]:
        filtered = top_k_filter(raw_logits[np.newaxis, :].copy(), k_val)
        p = softmax(filtered, axis=-1)[0]
        sorted_p = np.sort(p)[::-1]
        nonz = np.sum(p > 1e-10)
        axes[0, 1].plot(range(min(25, len(sorted_p))), sorted_p[:25], "o-",
                        markersize=4, linewidth=1.5, label=f"k={k_val} ({nonz} nonzero)")
    axes[0, 1].set_xlabel("Rank")
    axes[0, 1].set_ylabel("Probability")
    axes[0, 1].set_title("Top-k Filtering: Keep Only k Highest\nRest set to -inf before softmax",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Top-p filtering
    for p_val in [0.5, 0.8, 0.9, 0.95]:
        filtered = top_p_filter(raw_logits[np.newaxis, :].copy(), p_val)
        p = softmax(filtered, axis=-1)[0]
        sorted_p = np.sort(p)[::-1]
        nonz = np.sum(p > 1e-10)
        axes[0, 2].plot(range(min(25, len(sorted_p))), sorted_p[:25], "o-",
                        markersize=4, linewidth=1.5, label=f"p={p_val} ({nonz} nonzero)")
    axes[0, 2].set_xlabel("Rank")
    axes[0, 2].set_ylabel("Probability")
    axes[0, 2].set_title("Top-p (Nucleus) Filtering\nKeep smallest set with cumsum >= p",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Entropy comparison
    strat_names = list(strategy_probs.keys())
    entropies = []
    for name in strat_names:
        p = strategy_probs[name]
        ent = -np.sum(p[p > 1e-10] * np.log(p[p > 1e-10]))
        entropies.append(ent)
    bar_colors = [COLORS["dark"], COLORS["blue"], COLORS["green"], COLORS["orange"],
                  COLORS["red"], COLORS["purple"], COLORS["teal"]]
    axes[1, 0].bar(range(len(strat_names)), entropies, color=bar_colors[:len(strat_names)],
                   edgecolor="white")
    axes[1, 0].set_xticks(range(len(strat_names)))
    axes[1, 0].set_xticklabels(strat_names, fontsize=8, rotation=30)
    axes[1, 0].set_ylabel("Entropy (nats)")
    axes[1, 0].set_title("Distribution Entropy per Strategy\nHigher = more random, Lower = more deterministic",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Effective vocabulary size (nonzero tokens)
    nonzeros = [np.sum(strategy_probs[n] > 1e-10) for n in strat_names]
    axes[1, 1].bar(range(len(strat_names)), nonzeros, color=bar_colors[:len(strat_names)],
                   edgecolor="white")
    axes[1, 1].set_xticks(range(len(strat_names)))
    axes[1, 1].set_xticklabels(strat_names, fontsize=8, rotation=30)
    axes[1, 1].set_ylabel("Nonzero Tokens")
    axes[1, 1].set_title("Effective Vocabulary Size per Strategy\nFiltering reduces the candidate set",
                         fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    sampling_text = (
        "SAMPLING PIPELINE\n"
        "=================\n\n"
        "logits -> /T -> top-k -> top-p -> softmax -> sample\n\n"
        "Temperature (T):\n"
        "  T < 1: sharper (more deterministic)\n"
        "  T > 1: flatter (more random)\n"
        "  T -> 0: equivalent to greedy\n\n"
        "Top-k:\n"
        "  Keep only k highest logits\n"
        "  Fixed candidate set size\n\n"
        "Top-p (nucleus):\n"
        "  Keep smallest set with cumsum >= p\n"
        "  Adapts to distribution shape\n\n"
        "Combined (production default):\n"
        "  T=0.8, top_k=10, top_p=0.95\n"
        "  Balances quality and diversity\n\n"
        "CAVEAT: With random weights, all\n"
        "distributions are near-uniform.\n"
        "Trained models have peaked distributions\n"
        "where these strategies matter more."
    )
    axes[1, 2].text(0.05, 0.95, sampling_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Sampling Strategies: Temperature, Top-k, Top-p, and Combined",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_sampling_strategies.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/03_sampling_strategies.png")


# ---------------------------------------------------------------------------
# Example 4: Autoregressive Generation Loop
# ---------------------------------------------------------------------------
def example_4_generation_loop():
    """Visualize the step-by-step autoregressive generation process."""
    print("\n" + "=" * 60)
    print("Example 4: Autoregressive Generation Loop")
    print("=" * 60)

    np.random.seed(SEED)
    model = CausalLM(**SMALL_CFG)

    prompt = np.array([[3, 7, 11, 15]])
    gen_tokens = 10
    P = prompt.shape[1]

    print(f"\n  Prompt: {prompt[0].tolist()} (length P={P})")
    print(f"  Generating {gen_tokens} new tokens (greedy)")
    print(f"  This is the NAIVE version: full recompute at every step")

    tokens = prompt.copy()
    step_data = []

    for step in range(gen_tokens):
        seq_len = tokens.shape[1]
        logits = model.forward(tokens)
        last_logits = logits[0, -1, :]
        probs = softmax(last_logits[np.newaxis, :], axis=-1)[0]
        next_token = np.argmax(last_logits)
        top5_idx = np.argsort(probs)[::-1][:5]
        top5_probs = probs[top5_idx]

        step_data.append({
            "step": step,
            "seq_len": seq_len,
            "next_token": int(next_token),
            "top5_idx": top5_idx.tolist(),
            "top5_probs": top5_probs.tolist(),
            "max_prob": float(probs.max()),
            "entropy": float(-np.sum(probs[probs > 1e-10] * np.log(probs[probs > 1e-10]))),
        })

        tokens = np.concatenate([tokens, np.array([[next_token]])], axis=1)

    print(f"\n  {'Step':>6} {'Seq Len':>9} {'Token':>8} {'Max P':>10} {'Entropy':>10} {'Context':>30}")
    print(f"  {'-'*78}")
    for sd in step_data:
        ctx = tokens[0, :sd["seq_len"]].tolist()
        ctx_str = str(ctx) if len(ctx) <= 8 else f"[...{len(ctx)} tokens]"
        print(f"  {sd['step']:>6} {sd['seq_len']:>9} {sd['next_token']:>8} "
              f"{sd['max_prob']:>10.6f} {sd['entropy']:>10.4f} {ctx_str:>30}")

    print(f"\n  Final sequence: {tokens[0].tolist()}")

    # O(n^2) cost analysis
    print(f"\n  COMPUTATIONAL COST (naive, no KV cache):")
    print(f"  Each step processes the ENTIRE sequence, not just the new token.")
    print(f"  {'Step':>6} {'Tokens Processed':>18} {'Cumulative':>12}")
    print(f"  {'-'*42}")
    cumulative = 0
    step_tokens = []
    for step in range(gen_tokens):
        sl = P + step  # step 0 processes P tokens, step 1 processes P+1, etc.
        cumulative += sl
        step_tokens.append(sl)
        print(f"  {step:>6} {sl:>18} {cumulative:>12}")

    total_naive = sum(step_tokens)
    total_cached = P + gen_tokens
    print(f"\n  Total token-steps (naive):  {total_naive}")
    print(f"  Total token-steps (cached): {total_cached} (prefill P + n decode steps of 1)")
    print(f"  Naive / Cached ratio: {total_naive / total_cached:.1f}x")
    print(f"  Formula: sum(P+i for i=0..n-1) = n*P + n*(n-1)/2")
    analytical = gen_tokens * P + gen_tokens * (gen_tokens - 1) // 2
    print(f"  Analytical: {gen_tokens}*{P} + {gen_tokens}*{gen_tokens-1}/2 = {analytical}")
    assert total_naive == analytical

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Growing context visualization
    for sd in step_data:
        step = sd["step"]
        sl = sd["seq_len"]
        axes[0, 0].barh(step, sl, color=COLORS["blue"], edgecolor="white", height=0.7, alpha=0.8)
        axes[0, 0].barh(step, 1, left=sl, color=COLORS["red"], edgecolor="white", height=0.7, alpha=0.8)
    axes[0, 0].set_xlabel("Sequence Length (tokens processed)")
    axes[0, 0].set_ylabel("Generation Step")
    axes[0, 0].set_title("Growing Context: Full Recompute at Each Step\n"
                         "Blue = existing context, Red = new token position",
                         fontsize=10, fontweight="bold")
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    # Per-step token cost
    axes[0, 1].plot(range(gen_tokens), step_tokens, "o-", color=COLORS["red"],
                    linewidth=2, markersize=6, label="Naive (full recompute)")
    axes[0, 1].axhline(1, color=COLORS["green"], linestyle="--", linewidth=2,
                        label="With KV cache (1 token)")
    axes[0, 1].set_xlabel("Generation Step")
    axes[0, 1].set_ylabel("Tokens Processed")
    axes[0, 1].set_title("Per-Step Cost: Naive vs KV Cache\n"
                         "Naive processes entire sequence; cache processes 1 token",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Cumulative cost
    cum_naive = np.cumsum(step_tokens)
    cum_cached = np.arange(1, gen_tokens + 1) + P
    axes[0, 2].plot(range(gen_tokens), cum_naive, "o-", color=COLORS["red"],
                    linewidth=2, markersize=6, label="Naive (O(n^2))")
    axes[0, 2].plot(range(gen_tokens), cum_cached, "s-", color=COLORS["green"],
                    linewidth=2, markersize=6, label="Cached (O(n))")
    axes[0, 2].fill_between(range(gen_tokens), cum_cached, cum_naive,
                             alpha=0.2, color=COLORS["red"], label="Wasted computation")
    axes[0, 2].set_xlabel("Generation Step")
    axes[0, 2].set_ylabel("Cumulative Tokens Processed")
    axes[0, 2].set_title("Cumulative Cost: Quadratic vs Linear\n"
                         f"After {gen_tokens} steps: naive={cum_naive[-1]}, cached={cum_cached[-1]}",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Top-5 probabilities at each step
    for rank in range(5):
        probs_at_rank = [sd["top5_probs"][rank] for sd in step_data]
        axes[1, 0].plot(range(gen_tokens), probs_at_rank, "o-", markersize=4,
                        linewidth=1.5, label=f"Rank {rank+1}")
    axes[1, 0].set_xlabel("Generation Step")
    axes[1, 0].set_ylabel("Probability")
    axes[1, 0].set_title("Top-5 Token Probabilities per Step\n"
                         "(Random weights -> near-uniform distribution)",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Entropy over steps
    entropies = [sd["entropy"] for sd in step_data]
    max_entropy = np.log(SMALL_CFG["vocab_size"])
    axes[1, 1].plot(range(gen_tokens), entropies, "o-", color=COLORS["purple"],
                    linewidth=2, markersize=6)
    axes[1, 1].axhline(max_entropy, color=COLORS["red"], linestyle="--",
                        label=f"Max entropy (uniform) = ln({SMALL_CFG['vocab_size']}) = {max_entropy:.2f}")
    axes[1, 1].set_xlabel("Generation Step")
    axes[1, 1].set_ylabel("Entropy (nats)")
    axes[1, 1].set_title("Distribution Entropy per Step\n"
                         "Near-max entropy with random weights (expected)",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    gen_text = (
        "AUTOREGRESSIVE GENERATION\n"
        "=========================\n\n"
        f"Prompt: {prompt[0].tolist()}\n"
        f"Generated: {tokens[0, P:].tolist()}\n\n"
        "Algorithm (naive, no KV cache):\n"
        "  for step in range(max_new):\n"
        "    logits = model.forward(tokens)  # FULL\n"
        "    next = sample(logits[:, -1, :]) # last\n"
        "    tokens = concat(tokens, next)\n\n"
        "Cost analysis:\n"
        f"  P={P}, n={gen_tokens}\n"
        f"  Step i processes (P+i) tokens\n"
        f"  Total = n*P + n(n-1)/2\n"
        f"        = {analytical} token-steps\n"
        f"  With KV cache: P + n = {P + gen_tokens}\n"
        f"  Ratio: {total_naive / total_cached:.1f}x\n\n"
        "For large n, naive is O(n^2),\n"
        "cached is O(n). This quadratic\n"
        "waste motivates the KV cache."
    )
    axes[1, 2].text(0.05, 0.95, gen_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Autoregressive Generation: Step-by-Step Forward Pass with Growing Context",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_generation_loop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/04_generation_loop.png")


# ---------------------------------------------------------------------------
# Example 5: Computational Cost Analysis (Naive vs KV Cache)
# ---------------------------------------------------------------------------
def example_5_computational_cost():
    """Analytical FLOP comparison: naive full recompute vs theoretical KV cache."""
    print("\n" + "=" * 60)
    print("Example 5: Computational Cost Analysis (Naive vs KV Cache)")
    print("=" * 60)

    cfg = SMALL_CFG.copy()
    prompt_len = 16
    gen_lengths = [8, 16, 32, 64]

    print(f"\n  Model config: V={cfg['vocab_size']}, d={cfg['d_model']}, "
          f"layers={cfg['num_layers']}, h={cfg['num_heads']}, d_ff={cfg['d_ff']}")
    print(f"  Prompt length: {prompt_len}")

    print(f"\n  {'Gen Tokens':>12} {'Naive FLOPs':>16} {'Cached FLOPs':>16} {'Ratio':>10} {'Savings':>10}")
    print(f"  {'-'*68}")

    ratios = []
    naive_totals = []
    cached_totals = []

    for n in gen_lengths:
        naive = generation_flops(
            prompt_len, n, cfg["num_layers"], cfg["d_model"],
            cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
        )
        cached = generation_flops_with_cache(
            prompt_len, n, cfg["num_layers"], cfg["d_model"],
            cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
        )
        ratio = naive["total"] / cached["total"]
        savings = (1 - cached["total"] / naive["total"]) * 100
        ratios.append(ratio)
        naive_totals.append(naive["total"])
        cached_totals.append(cached["total"])
        print(f"  {n:>12} {naive['total']:>16.2e} {cached['total']:>16.2e} {ratio:>10.2f}x {savings:>9.1f}%")

    # Sweep longer generation lengths for the plot
    sweep_lens = list(range(4, 105, 4))
    sweep_naive = []
    sweep_cached = []
    sweep_ratios = []

    for n in sweep_lens:
        naive = generation_flops(
            prompt_len, n, cfg["num_layers"], cfg["d_model"],
            cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
        )
        cached = generation_flops_with_cache(
            prompt_len, n, cfg["num_layers"], cfg["d_model"],
            cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
        )
        sweep_naive.append(naive["total"])
        sweep_cached.append(cached["total"])
        sweep_ratios.append(naive["total"] / cached["total"])

    print(f"\n  ANALYTICAL EXPLANATION:")
    print(f"  Naive: step i does a full forward pass over (P+i) tokens.")
    print(f"  Total FLOPs ~ sum_i cost(P+i) where cost(L) ~ C*L + A*L^2")
    print(f"    C = linear ops (projections, FFN, output proj)")
    print(f"    A = quadratic ops (attention core: QK^T, softmax, AV)")
    print(f"  Cached: prefill processes P tokens once, then each decode step")
    print(f"  projects 1 new token and attends over growing cache.")
    print(f"  The ratio grows with n because naive recomputes ALL projections")
    print(f"  for ALL previous tokens at every step.")

    # Per-step cost breakdown for one example
    n_example = 32
    naive_ex = generation_flops(
        prompt_len, n_example, cfg["num_layers"], cfg["d_model"],
        cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
    )
    cached_ex = generation_flops_with_cache(
        prompt_len, n_example, cfg["num_layers"], cfg["d_model"],
        cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
    )

    print(f"\n  Per-step breakdown (P={prompt_len}, n={n_example}):")
    print(f"  {'Step':>6} {'Naive':>14} {'Cached':>14} {'Ratio':>10}")
    print(f"  {'-'*48}")
    for i in range(0, n_example, 4):
        n_step = naive_ex["per_step"][i]
        c_step = cached_ex["per_step"][i]
        print(f"  {i:>6} {n_step:>14.2e} {c_step:>14.2e} {n_step/c_step:>10.2f}x")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Total FLOPs comparison
    axes[0, 0].plot(sweep_lens, sweep_naive, "o-", color=COLORS["red"], linewidth=2,
                    markersize=4, label="Naive (full recompute)")
    axes[0, 0].plot(sweep_lens, sweep_cached, "s-", color=COLORS["green"], linewidth=2,
                    markersize=4, label="With KV cache")
    axes[0, 0].set_xlabel("Number of Generated Tokens")
    axes[0, 0].set_ylabel("Total FLOPs")
    axes[0, 0].set_title(f"Total FLOPs: Naive vs KV Cache (P={prompt_len})\n"
                         f"Gap widens with more tokens generated",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Ratio
    axes[0, 1].plot(sweep_lens, sweep_ratios, "o-", color=COLORS["purple"], linewidth=2, markersize=4)
    axes[0, 1].set_xlabel("Number of Generated Tokens")
    axes[0, 1].set_ylabel("Naive / Cached FLOPs Ratio")
    axes[0, 1].set_title("Speedup from KV Cache\nGrows approximately linearly with generation length",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # Per-step cost
    naive_per_step = naive_ex["per_step"]
    cached_per_step = cached_ex["per_step"]
    steps = range(n_example)
    axes[0, 2].plot(list(steps), naive_per_step, "o-", color=COLORS["red"], linewidth=1.5,
                    markersize=3, label="Naive per step")
    axes[0, 2].plot(list(steps), cached_per_step, "s-", color=COLORS["green"], linewidth=1.5,
                    markersize=3, label="Cached per step")
    axes[0, 2].set_xlabel("Generation Step")
    axes[0, 2].set_ylabel("FLOPs per Step")
    axes[0, 2].set_title(f"Per-Step FLOPs (P={prompt_len}, n={n_example})\n"
                         f"Naive grows linearly; cached grows sub-linearly",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Wasted computation fraction
    wasted_frac = [(sweep_naive[i] - sweep_cached[i]) / sweep_naive[i] * 100
                   for i in range(len(sweep_lens))]
    axes[1, 0].plot(sweep_lens, wasted_frac, "o-", color=COLORS["orange"], linewidth=2, markersize=4)
    axes[1, 0].set_xlabel("Number of Generated Tokens")
    axes[1, 0].set_ylabel("Wasted FLOPs (%)")
    axes[1, 0].set_title("Fraction of Wasted Computation (Naive)\n"
                         "Approaches 100% as generation length grows",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # Varying prompt lengths
    prompt_lens_sweep = [8, 16, 32, 64]
    for pl in prompt_lens_sweep:
        pl_ratios = []
        for n in sweep_lens:
            if pl + n > SMALL_CFG["max_seq_len"]:
                break
            naive = generation_flops(
                pl, n, cfg["num_layers"], cfg["d_model"],
                cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
            )
            cached = generation_flops_with_cache(
                pl, n, cfg["num_layers"], cfg["d_model"],
                cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
            )
            pl_ratios.append(naive["total"] / cached["total"])
        axes[1, 1].plot(range(len(pl_ratios)), [sweep_lens[i] for i in range(len(pl_ratios))],
                        "o-" if len(pl_ratios) < 30 else "-", markersize=3, linewidth=1.5)
    # Re-do with correct axes
    axes[1, 1].clear()
    for pl in prompt_lens_sweep:
        pl_ratios = []
        ns = []
        for n in sweep_lens:
            if pl + n > SMALL_CFG["max_seq_len"]:
                break
            naive = generation_flops(
                pl, n, cfg["num_layers"], cfg["d_model"],
                cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
            )
            cached = generation_flops_with_cache(
                pl, n, cfg["num_layers"], cfg["d_model"],
                cfg["num_heads"], cfg["num_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
            )
            pl_ratios.append(naive["total"] / cached["total"])
            ns.append(n)
        axes[1, 1].plot(ns, pl_ratios, "o-", markersize=3, linewidth=1.5, label=f"P={pl}")
    axes[1, 1].set_xlabel("Number of Generated Tokens")
    axes[1, 1].set_ylabel("Naive / Cached Ratio")
    axes[1, 1].set_title("Speedup Ratio vs Prompt Length\nLonger prompts = even more redundant recomputation",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    cost_text = (
        "COMPUTATIONAL COST ANALYSIS\n"
        "===========================\n\n"
        "NAIVE (no KV cache):\n"
        "  Step i: forward pass over (P+i) tokens\n"
        "  All projections + attention recomputed\n"
        "  Total: sum_{i=0}^{n-1} cost(P+i)\n"
        "  Dominant term: O(n * P * d^2) + O(n^2 * d^2)\n\n"
        "WITH KV CACHE:\n"
        "  Prefill: one pass over P tokens\n"
        "  Decode: each step projects 1 token,\n"
        "  attends over growing cache\n"
        "  Total: cost(P) + n * cost(1, cache)\n\n"
        "KEY INSIGHT:\n"
        "  Naive recomputes Q,K,V projections for\n"
        "  ALL previous tokens at EVERY step.\n"
        "  KV cache stores K,V projections and\n"
        "  only computes the NEW token's Q,K,V.\n\n"
        "  This is the single most impactful\n"
        "  optimization in LLM inference."
    )
    axes[1, 2].text(0.05, 0.95, cost_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Computational Cost: Naive Full Recompute vs Theoretical KV Cache",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_computational_cost.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/05_computational_cost.png")


# ---------------------------------------------------------------------------
# Example 6: Model Parameter Analysis
# ---------------------------------------------------------------------------
def example_6_parameter_analysis():
    """Parameter breakdown for Llama 2 7B, Llama 3 8B, and Mistral 7B."""
    print("\n" + "=" * 60)
    print("Example 6: Model Parameter Analysis")
    print("=" * 60)

    # None of these production models use weight tying (tie_word_embeddings=false).
    # See: Llama 2 config, Llama 3 config, Mistral config on HuggingFace.
    # The README also notes: "Llama 2 does not use weight tying."
    MODEL_CONFIGS = {
        "Llama 2 7B": dict(vocab_size=32000, d_model=4096, num_layers=32, num_heads=32,
                           num_kv_heads=32, d_ff=11008, tie_weights=False),
        "Llama 3 8B": dict(vocab_size=128256, d_model=4096, num_layers=32, num_heads=32,
                           num_kv_heads=8, d_ff=14336, tie_weights=False),
        "Mistral 7B": dict(vocab_size=32000, d_model=4096, num_layers=32, num_heads=32,
                           num_kv_heads=8, d_ff=14336, tie_weights=False),
    }

    results = {}
    for name, cfg in MODEL_CONFIGS.items():
        params = count_model_parameters(**cfg)
        results[name] = params

    print(f"\n  {'Model':<16} {'Embedding':>12} {'Blocks':>12} {'Norm':>8} {'Out Proj':>10} "
          f"{'Total':>12} {'Total (B)':>10}")
    print(f"  {'-'*76}")
    for name, p in results.items():
        print(f"  {name:<16} {p['embedding']/1e6:>11.1f}M {p['total_blocks']/1e6:>11.1f}M "
              f"{p['final_norm']:>8} {p['output_proj']:>10} "
              f"{p['total']/1e6:>11.1f}M {p['total']/1e9:>9.2f}B")

    print(f"\n  Percentage breakdown:")
    print(f"  {'Model':<16} {'Embed%':>8} {'Blocks%':>10} {'Norm%':>8} {'OutProj%':>10}")
    print(f"  {'-'*52}")
    for name, p in results.items():
        print(f"  {name:<16} {p['embedding_pct']:>7.1f}% {p['blocks_pct']:>9.1f}% "
              f"{p['final_norm_pct']:>7.3f}% {p['output_proj_pct']:>9.1f}%")

    # Weight tying savings (hypothetical -- these models don't use it)
    print(f"\n  Hypothetical weight tying savings (none of these models use tying):")
    for name, cfg in MODEL_CONFIGS.items():
        tied = count_model_parameters(**{**cfg, "tie_weights": True})
        savings = results[name]["total"] - tied["total"]
        print(f"    {name}: would save {savings/1e6:.1f}M params "
              f"({cfg['d_model']} x {cfg['vocab_size']} = {cfg['d_model']*cfg['vocab_size']/1e6:.1f}M)")

    # Per-block breakdown for Llama 2 7B
    from implementation import _block_count_parameters
    llama2_block = _block_count_parameters(4096, 32, 32, 11008)
    print(f"\n  Llama 2 7B per-block breakdown:")
    print(f"    Attention:  {llama2_block['attn_total']/1e6:.1f}M ({llama2_block['attn_pct']:.1f}%)")
    print(f"    FFN:        {llama2_block['ffn_total']/1e6:.1f}M ({llama2_block['ffn_pct']:.1f}%)")
    print(f"    Norms:      {llama2_block['norm_total']} ({llama2_block['norm_pct']:.3f}%)")
    print(f"    Per block:  {llama2_block['total']/1e6:.1f}M")
    print(f"    x 32 layers = {llama2_block['total'] * 32 / 1e9:.2f}B")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Stacked bar chart of parameter distribution
    model_names = list(results.keys())
    embed_vals = [results[n]["embedding"] / 1e9 for n in model_names]
    block_vals = [results[n]["total_blocks"] / 1e9 for n in model_names]
    outproj_vals = [results[n]["output_proj"] / 1e9 for n in model_names]

    x_pos = np.arange(len(model_names))
    axes[0, 0].bar(x_pos, embed_vals, 0.5, label="Embedding", color=COLORS["blue"], edgecolor="white")
    bottom_blocks = embed_vals
    axes[0, 0].bar(x_pos, block_vals, 0.5, bottom=bottom_blocks, label="Blocks (N x block)",
                   color=COLORS["green"], edgecolor="white")
    bottom_outproj = [embed_vals[i] + block_vals[i] for i in range(len(model_names))]
    axes[0, 0].bar(x_pos, outproj_vals, 0.5, bottom=bottom_outproj, label="Output Proj",
                   color=COLORS["red"], edgecolor="white")
    for i, n in enumerate(model_names):
        total = results[n]["total"] / 1e9
        axes[0, 0].text(i, total + 0.05, f"{total:.2f}B", ha="center", va="bottom",
                       fontsize=9, fontweight="bold")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, fontsize=9)
    axes[0, 0].set_ylabel("Parameters (Billions)")
    axes[0, 0].set_title("Total Parameter Count (no weight tying)\n"
                         "Transformer blocks dominate (>86%)",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Pie charts for each model
    for idx, (name, p) in enumerate(results.items()):
        ax = axes[0, 1] if idx == 0 else (axes[0, 2] if idx == 1 else axes[1, 0])
        components = ["Embedding", "Blocks", "Norm"]
        values = [p["embedding"], p["total_blocks"], p["final_norm"]]
        if p["output_proj"] > 0:
            components.append("Output Proj")
            values.append(p["output_proj"])
        pie_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]

        def autopct_fn(pct):
            if pct < 0.1:
                return ""
            return f"{pct:.1f}%"

        wedges, texts, autotexts = ax.pie(
            values, labels=components, colors=pie_colors[:len(components)],
            autopct=autopct_fn, pctdistance=0.75, startangle=90
        )
        for t in autotexts:
            t.set_fontsize(8)
        for t in texts:
            t.set_fontsize(8)
        tied_label = "tied weights" if p['tie_weights'] else "untied weights"
        ax.set_title(f"{name}\nTotal = {p['total']/1e9:.2f}B ({tied_label})",
                     fontsize=10, fontweight="bold")

    # Weight tying savings comparison (hypothetical)
    actual_totals = [results[n]["total"] / 1e9 for n in model_names]
    tied_totals = []
    savings_vals = []
    for name, cfg in MODEL_CONFIGS.items():
        tied = count_model_parameters(**{**cfg, "tie_weights": True})
        tied_totals.append(tied["total"] / 1e9)
        savings_vals.append((results[name]["total"] - tied["total"]) / 1e9)

    axes[1, 1].bar(x_pos - 0.15, actual_totals, 0.3, label="Actual (no tying)",
                   color=COLORS["coral"], edgecolor="white")
    axes[1, 1].bar(x_pos + 0.15, tied_totals, 0.3, label="Hypothetical (with tying)",
                   color=COLORS["green"], edgecolor="white")
    for i in range(len(model_names)):
        axes[1, 1].annotate(f"-{savings_vals[i]*1000:.0f}M",
                           (i + 0.15, tied_totals[i]),
                           textcoords="offset points", xytext=(0, 8),
                           ha="center", fontsize=8, fontweight="bold", color=COLORS["blue"])
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(model_names, fontsize=9)
    axes[1, 1].set_ylabel("Parameters (Billions)")
    axes[1, 1].set_title("Hypothetical Weight Tying Savings\nWould save d_model x V parameters",
                         fontsize=10, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    analysis_text = (
        "MODEL PARAMETER ANALYSIS\n"
        "========================\n\n"
        "Parameter formula (with tying):\n"
        "  P = V*d + N*P_block + d\n\n"
        "Per-block (Llama 2 7B):\n"
        f"  Attn:  {llama2_block['attn_total']/1e6:.1f}M "
        f"({llama2_block['attn_pct']:.1f}%)\n"
        f"  FFN:   {llama2_block['ffn_total']/1e6:.1f}M "
        f"({llama2_block['ffn_pct']:.1f}%)\n"
        f"  Norms: {llama2_block['norm_total']} (~0%)\n\n"
        "Weight tying (hypothetical):\n"
        "  W_out = E.T (shared memory)\n"
        "  Would save V*d parameters\n"
        "  Llama 3: would save 525M (128K vocab)\n\n"
        "Key observations:\n"
        "  - Blocks are >86% of total params\n"
        "    (96% for 32K vocab, 87% for 128K)\n"
        "  - FFN is ~67% of each block (MHA)\n"
        "    (~81% with GQA due to fewer K/V)\n"
        "  - Larger vocab (Llama 3) increases\n"
        "    embedding + output proj cost\n"
        "  - These models do NOT use weight\n"
        "    tying (separate output projection)"
    )
    axes[1, 2].text(0.05, 0.95, analysis_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Model Parameter Analysis: Llama 2 7B, Llama 3 8B, Mistral 7B",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_parameter_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/06_parameter_analysis.png")


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------
def generate_pdf_report():
    """Generate comprehensive PDF report with all visualizations."""
    print("\n" + "=" * 60)
    print("Generating PDF Report")
    print("=" * 60)

    report_path = Path(__file__).parent / "report.pdf"
    viz_files = sorted(VIZ_DIR.glob("*.png"))

    with PdfPages(str(report_path)) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.78, "Causal Decoding", fontsize=28, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.68, "Complete Decoder-Only Language Model: Embeddings, Transformer Blocks,\n"
                "Output Projection, and Autoregressive Generation with Sampling",
                fontsize=13, ha="center", va="center", transform=ax.transAxes, color="gray")
        info_text = (
            "The culmination of Phase 3: a complete CausalLM that takes token IDs\n"
            "and generates token IDs, the same interface as any production LLM.\n"
            "This is the naive version -- full forward pass recompute at every\n"
            "generation step -- deliberately inefficient to motivate KV caching.\n\n"
            "This demo covers:\n"
            "  1. Full forward pass walkthrough with shape tracing\n"
            "  2. Causal property verification (THE centerpiece)\n"
            "  3. Sampling strategies: greedy, temperature, top-k, top-p\n"
            "  4. Autoregressive generation loop visualization\n"
            "  5. Computational cost: naive O(n^2) vs KV cache O(n)\n"
            "  6. Model parameter analysis for Llama/Mistral configs\n\n"
            f"Model config: V=256, d=64, layers=2, h=4, h_kv=2, d_ff=172\n"
            f"Random seed: {SEED}\n"
            f"Number of visualizations: {len(viz_files)}\n"
            "Examples: 6"
        )
        ax.text(0.5, 0.32, info_text, fontsize=11, ha="center", va="center",
                transform=ax.transAxes, linespacing=1.6)
        ax.text(0.5, 0.06, "Generated by demo.py", fontsize=10, ha="center",
                va="center", transform=ax.transAxes, style="italic", color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.94, "Summary of Findings", fontsize=20, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)

        summary_items = [
            "1. Forward Pass: Token IDs (B,L) -> embedding lookup (B,L,d) ->",
            "   N transformer blocks -> final RMSNorm -> output logits (B,L,V).",
            "   Weight tying: W_out = E.T shares memory, saving V*d parameters.",
            "   Softmax of last-position logits gives next-token probabilities.",
            "",
            "2. Causal Property (CENTERPIECE): For sequences sharing prefix [0..k-1],",
            "   logits at positions 0..k-1 are EXACTLY identical regardless of tokens",
            "   at positions k+. Verified: changing token at position 3 has ZERO effect",
            "   on positions 0-2 (diff < 1e-12). This enables autoregressive generation.",
            "",
            "3. Sampling Strategies: Temperature controls sharpness (T<1: deterministic,",
            "   T>1: random). Top-k keeps only k highest logits. Top-p (nucleus) keeps",
            "   the smallest set with cumulative probability >= p. Combined pipeline:",
            "   logits -> /T -> top-k -> top-p -> softmax -> categorical sample.",
            "",
            "4. Generation Loop: Naive version does full forward pass at each step,",
            "   processing the growing sequence (P, P+1, ..., P+n-1 tokens).",
            "   Total cost: n*P + n(n-1)/2 token-steps. With KV cache: P + n.",
            "   The redundant recomputation motivates KV caching.",
            "",
            "5. Computational Cost: Naive FLOPs grow super-linearly with generation",
            "   length. KV cache eliminates redundant K/V projections for previous",
            "   tokens. Speedup ratio grows approximately linearly with n.",
            "   This is the single most impactful optimization in LLM inference.",
            "",
            "6. Parameter Analysis: Transformer blocks are >86% of total parameters.",
            "   Llama 2 7B: ~6.74B. Llama 3 8B: ~8.03B. Mistral 7B: ~7.24B.",
            "   None use weight tying; tying would save V*d (131M-525M).",
            "   FFN dominates per-block params (~67% MHA, ~81% GQA).",
        ]
        summary_text = "\n".join(summary_items)
        ax.text(0.06, 0.86, summary_text, fontsize=10, ha="left", va="top",
                transform=ax.transAxes, family="monospace", linespacing=1.3)
        pdf.savefig(fig)
        plt.close(fig)

        titles = {
            "01_forward_pass.png": "Example 1: Full Forward Pass Walkthrough",
            "02_causal_property.png": "Example 2: Causal Property Verification (CENTERPIECE)",
            "03_sampling_strategies.png": "Example 3: Sampling Strategy Comparison",
            "04_generation_loop.png": "Example 4: Autoregressive Generation Loop",
            "05_computational_cost.png": "Example 5: Computational Cost (Naive vs KV Cache)",
            "06_parameter_analysis.png": "Example 6: Model Parameter Analysis",
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
    print("Causal Decoding Demo")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print(f"Model config: {SMALL_CFG}")
    print()

    example_1_forward_pass()
    example_2_causal_property()
    example_3_sampling_strategies()
    example_4_generation_loop()
    example_5_computational_cost()
    example_6_parameter_analysis()
    generate_pdf_report()

    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print(f"Visualizations: {VIZ_DIR}/")
    print(f"Report: {Path(__file__).parent / 'report.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
