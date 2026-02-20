"""
BPE Tokenization Demo -- Training visualization, compression analysis, byte fallback,
special tokens, roundtrip verification, and inference impact analysis.

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
from implementation import BPETrainer, BPETokenizer

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

TRAINING_CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 80,
    "Machine learning is a subset of artificial intelligence that enables systems to learn. " * 60,
    "Natural language processing allows computers to understand human language effectively. " * 60,
    "Transformers use self-attention mechanisms to process sequences in parallel efficiently. " * 60,
    "Byte pair encoding is a subword tokenization algorithm used by modern language models. " * 60,
    "The model processes input tokens through multiple layers of attention and feed-forward networks. " * 50,
    "Training large language models requires significant computational resources and data. " * 50,
    "Inference optimization reduces latency and memory usage for deployed models. " * 50,
    "def compute_attention(query, key, value):\n    scores = query @ key.T\n    return softmax(scores) @ value\n" * 40,
    "for i in range(num_layers):\n    x = attention(x) + x\n    x = feed_forward(x) + x\n" * 40,
    "Gradient descent updates the model parameters by computing partial derivatives of the loss function. " * 50,
    "Backpropagation efficiently computes gradients through the chain rule of calculus. " * 50,
    "The embedding layer maps discrete token identifiers to continuous vector representations. " * 50,
    "Positional encoding adds information about the position of each token in the sequence. " * 50,
    "Layer normalization stabilizes training by normalizing activations across the feature dimension. " * 50,
    "Residual connections allow gradients to flow directly through the network during backpropagation. " * 50,
    "The softmax function converts logits into a probability distribution over the vocabulary. " * 50,
    "Cross-entropy loss measures the difference between predicted and target probability distributions. " * 50,
    "Batch normalization reduces internal covariate shift by normalizing layer inputs per mini-batch. " * 40,
    "The learning rate scheduler adjusts the step size throughout training for better convergence. " * 40,
    "Weight decay regularization prevents overfitting by penalizing large parameter values. " * 40,
    "Dropout randomly zeroes activations during training to prevent co-adaptation of neurons. " * 40,
    "class TransformerBlock:\n    def __init__(self, d_model, num_heads, d_ff):\n        self.attention = MultiHeadAttention(d_model, num_heads)\n        self.ffn = FeedForward(d_model, d_ff)\n" * 30,
    "import numpy as np\nimport torch\nfrom typing import Optional, Tuple, List\n" * 40,
]


# ---------------------------------------------------------------------------
# Example 1: Training Process Visualization
# ---------------------------------------------------------------------------
def example_1_training_process():
    """Show merge rules being learned step by step with pair frequencies."""
    print("=" * 60)
    print("Example 1: Training Process Visualization")
    print("=" * 60)

    corpus = ["low " * 5, "lower " * 2, "newest " * 6, "widest " * 3]
    trainer = BPETrainer()

    from implementation import _get_pair_counts, _apply_merge
    import re
    from implementation import GPT2_SPLIT_PATTERN

    compiled_pattern = re.compile(GPT2_SPLIT_PATTERN)
    word_freqs = {}
    for text in corpus:
        words = compiled_pattern.findall(text)
        for word in words:
            byte_seq = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_freqs[byte_seq] = word_freqs.get(byte_seq, 0) + 1

    print(f"\n  Corpus: {[c.strip() for c in corpus]}")
    print(f"\n  Initial word frequencies:")
    for symbols, freq in sorted(word_freqs.items(), key=lambda x: -x[1]):
        readable = " ".join(s.decode("utf-8", errors="replace") for s in symbols)
        print(f"    [{readable}] x {freq}")

    merge_history = []
    top_pairs_history = []
    vocab_sizes = []
    current_vocab_size = 256

    num_merges = 12
    for step in range(num_merges):
        pair_counts = _get_pair_counts(word_freqs)
        if not pair_counts:
            break

        sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
        top_5 = sorted_pairs[:5]
        top_pairs_history.append(top_5)

        best_pair = sorted_pairs[0][0]
        best_freq = sorted_pairs[0][1]
        merged_token = best_pair[0] + best_pair[1]

        a_str = best_pair[0].decode("utf-8", errors="replace")
        b_str = best_pair[1].decode("utf-8", errors="replace")
        m_str = merged_token.decode("utf-8", errors="replace")
        merge_history.append((step + 1, a_str, b_str, m_str, best_freq))

        print(f"\n  Step {step + 1}: Merge ({a_str!r}, {b_str!r}) -> {m_str!r} (freq={best_freq})")

        word_freqs = _apply_merge(word_freqs, best_pair)
        current_vocab_size += 1
        vocab_sizes.append(current_vocab_size)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    steps = [m[0] for m in merge_history]
    freqs = [m[4] for m in merge_history]
    labels = [f"({m[1]},{m[2]})->{m[3]}" for m in merge_history]
    bar_colors = [COLORS["blue"] if f > 5 else COLORS["orange"] if f > 2 else COLORS["coral"]
                  for f in freqs]
    axes[0, 0].barh(range(len(steps)), freqs, color=bar_colors, edgecolor="white")
    axes[0, 0].set_yticks(range(len(steps)))
    axes[0, 0].set_yticklabels(labels, fontsize=8, family="monospace")
    axes[0, 0].set_xlabel("Pair Frequency")
    axes[0, 0].set_title("BPE Merge Rules (Order of Learning)\nHigher frequency pairs merged first",
                         fontsize=10, fontweight="bold")
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    n_display = min(6, len(top_pairs_history))
    for step_idx in range(n_display):
        pairs_data = top_pairs_history[step_idx]
        pair_labels = []
        pair_freqs = []
        for pair, freq in pairs_data:
            a = pair[0].decode("utf-8", errors="replace")
            b = pair[1].decode("utf-8", errors="replace")
            pair_labels.append(f"({a},{b})")
            pair_freqs.append(freq)
        x_pos = np.arange(len(pair_labels))
        color = plt.cm.viridis(step_idx / max(n_display - 1, 1))
        axes[0, 1].bar(x_pos + step_idx * 0.12 - 0.3, pair_freqs,
                       width=0.1, color=color, alpha=0.8,
                       label=f"Step {step_idx + 1}" if step_idx < 4 else "")
    axes[0, 1].set_xlabel("Top Pairs (per step)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Top-5 Pair Frequencies at Each Merge Step\nDistribution shifts as merges are applied",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    axes[0, 2].plot(steps, vocab_sizes, "o-", color=COLORS["green"], linewidth=2, markersize=6)
    axes[0, 2].set_xlabel("Merge Step")
    axes[0, 2].set_ylabel("Vocabulary Size")
    axes[0, 2].set_title("Vocabulary Growth During Training\nStarts at 256 (byte tokens) + 1 per merge",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3)

    big_corpus = TRAINING_CORPUS[:5]
    big_merges_per_step = []
    big_top_freqs = []

    big_word_freqs = {}
    big_compiled = re.compile(GPT2_SPLIT_PATTERN)
    for text in big_corpus:
        words = big_compiled.findall(text)
        for word in words:
            byte_seq = tuple(bytes([b]) for b in word.encode("utf-8"))
            big_word_freqs[byte_seq] = big_word_freqs.get(byte_seq, 0) + 1

    for step in range(50):
        pair_counts = _get_pair_counts(big_word_freqs)
        if not pair_counts:
            break
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
        big_top_freqs.append(sorted_pairs[0][1])
        best_pair = sorted_pairs[0][0]
        big_word_freqs = _apply_merge(big_word_freqs, best_pair)
        big_merges_per_step.append(step + 1)

    axes[1, 0].plot(big_merges_per_step, big_top_freqs, "o-", color=COLORS["purple"],
                    linewidth=2, markersize=3)
    axes[1, 0].set_xlabel("Merge Step")
    axes[1, 0].set_ylabel("Frequency of Best Pair")
    axes[1, 0].set_title("Best Pair Frequency Decay (Larger Corpus)\nDiminishing returns as common pairs are consumed",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    token_lengths = [len(m[3]) for m in merge_history]
    axes[1, 1].bar(range(len(token_lengths)), token_lengths, color=COLORS["teal"], edgecolor="white")
    axes[1, 1].set_xticks(range(len(token_lengths)))
    axes[1, 1].set_xticklabels([m[3] for m in merge_history], fontsize=7, family="monospace", rotation=45)
    axes[1, 1].set_xlabel("Merged Token")
    axes[1, 1].set_ylabel("Token Length (characters)")
    axes[1, 1].set_title("Length of Newly Created Tokens\nLater merges create progressively longer tokens",
                         fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    training_text = (
        "BPE TRAINING PROCESS\n"
        "====================\n\n"
        "Algorithm:\n"
        "  1. Start with 256 byte tokens\n"
        "  2. Count all adjacent pairs\n"
        "  3. Merge most frequent pair\n"
        "  4. Update corpus representation\n"
        "  5. Repeat until target vocab size\n\n"
        "Key observations:\n"
        "  - Frequent pairs merged first\n"
        "    (spaces, common bigrams)\n"
        "  - Best-pair frequency decays\n"
        "    as obvious patterns merge\n"
        "  - Token length grows over time\n"
        "    (merging previously merged)\n"
        "  - Training is deterministic:\n"
        "    same corpus = same merges\n\n"
        f"Small corpus: {len(corpus)} docs\n"
        f"Merges shown: {len(merge_history)}"
    )
    axes[1, 2].text(0.05, 0.95, training_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("BPE Training: Learning Merge Rules from Corpus",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_training_process.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/01_training_process.png")


# ---------------------------------------------------------------------------
# Example 2: Compression Ratio Analysis
# ---------------------------------------------------------------------------
def example_2_compression_analysis():
    """Characters per token as vocab size increases."""
    print("\n" + "=" * 60)
    print("Example 2: Compression Ratio Analysis")
    print("=" * 60)

    trainer = BPETrainer()
    vocab_sizes = [256, 280, 300, 350, 400, 500, 600, 800, 1000]

    test_texts = {
        "English": "The quick brown fox jumps over the lazy dog. Machine learning is fascinating.",
        "Code": "def forward(self, x):\n    return self.linear(x) + self.bias\n",
        "Mixed": "Hello world! The model uses attention: score = Q @ K.T / sqrt(d_k)",
    }

    results = {name: [] for name in test_texts}
    actual_vocab_sizes = []

    print(f"\n  Training tokenizers with vocab sizes: {vocab_sizes}")
    print(f"\n  {'Vocab':>8} {'English CPT':>14} {'Code CPT':>12} {'Mixed CPT':>12}")
    print(f"  {'-'*50}")

    for vs in vocab_sizes:
        tokenizer = trainer.train(TRAINING_CORPUS, vocab_size=vs)
        actual_vocab_sizes.append(tokenizer.vocab_size)

        for name, text in test_texts.items():
            cpt = tokenizer.characters_per_token(text)
            results[name].append(cpt)

        print(f"  {tokenizer.vocab_size:>8} "
              f"{results['English'][-1]:>14.2f} "
              f"{results['Code'][-1]:>12.2f} "
              f"{results['Mixed'][-1]:>12.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for name, cpts in results.items():
        axes[0, 0].plot(actual_vocab_sizes, cpts, "o-", linewidth=2, markersize=5, label=name)
    axes[0, 0].set_xlabel("Vocabulary Size")
    axes[0, 0].set_ylabel("Characters per Token")
    axes[0, 0].set_title("Compression vs Vocabulary Size\nLarger vocab = fewer tokens per character",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    tok_small = trainer.train(TRAINING_CORPUS, vocab_size=300)
    tok_large = trainer.train(TRAINING_CORPUS, vocab_size=800)

    sample_texts = [
        "the", "machine", "learning", "tokenization",
        "def", "self", "attention", "transformer",
    ]
    small_lens = [len(tok_small.encode(t)) for t in sample_texts]
    large_lens = [len(tok_large.encode(t)) for t in sample_texts]

    x = np.arange(len(sample_texts))
    axes[0, 1].bar(x - 0.15, small_lens, 0.3, label="V=300", color=COLORS["coral"], edgecolor="white")
    axes[0, 1].bar(x + 0.15, large_lens, 0.3, label="V=800", color=COLORS["green"], edgecolor="white")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(sample_texts, fontsize=8, rotation=30)
    axes[0, 1].set_ylabel("Number of Tokens")
    axes[0, 1].set_title("Token Count per Word: Small vs Large Vocab\nCommon words compress better with larger vocab",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    compression_ratios = []
    for vs in actual_vocab_sizes:
        idx = actual_vocab_sizes.index(vs)
        cr = results["English"][idx]
        compression_ratios.append(cr)
    axes[0, 2].bar(range(len(actual_vocab_sizes)), compression_ratios,
                   color=COLORS["blue"], edgecolor="white")
    axes[0, 2].set_xticks(range(len(actual_vocab_sizes)))
    axes[0, 2].set_xticklabels([str(v) for v in actual_vocab_sizes], fontsize=8)
    axes[0, 2].set_xlabel("Vocabulary Size")
    axes[0, 2].set_ylabel("Characters per Token (English)")
    axes[0, 2].set_title("English Text Compression by Vocab Size\nDiminishing returns at larger sizes",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    tok_800 = trainer.train(TRAINING_CORPUS, vocab_size=800)
    content_types = {
        "English prose": "The quick brown fox jumps over the lazy dog. Machine learning is a fascinating field.",
        "Python code": "def train(self, data, lr=0.01):\n    for batch in data:\n        loss = self.forward(batch)\n",
        "Numbers": "3.14159 2.71828 1.41421 0.69315 2.30259 1.09861",
        "Repeated": "aaabbbcccaaabbbcccaaabbbccc",
        "Random ASCII": "jX9#mK2&pL7!qR4%tW6@",
    }

    type_names = list(content_types.keys())
    type_cpts = [tok_800.characters_per_token(text) for text in content_types.values()]
    type_colors = [COLORS["green"], COLORS["blue"], COLORS["orange"],
                   COLORS["purple"], COLORS["red"]]

    axes[1, 0].barh(range(len(type_names)), type_cpts, color=type_colors, edgecolor="white")
    axes[1, 0].set_yticks(range(len(type_names)))
    axes[1, 0].set_yticklabels(type_names, fontsize=9)
    axes[1, 0].set_xlabel("Characters per Token")
    axes[1, 0].set_title("Compression by Content Type (V=800)\nCorpus-similar content compresses best",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="x")

    text_for_seq = "The transformer model processes input sequences through multiple layers of attention."
    seq_lengths = []
    for vs in actual_vocab_sizes:
        idx = actual_vocab_sizes.index(vs)
        tok_temp = trainer.train(TRAINING_CORPUS, vocab_size=vocab_sizes[idx])
        ids = tok_temp.encode(text_for_seq)
        seq_lengths.append(len(ids))

    axes[1, 1].plot(actual_vocab_sizes, seq_lengths, "o-", color=COLORS["red"], linewidth=2, markersize=6)
    for i, (vs, sl) in enumerate(zip(actual_vocab_sizes, seq_lengths)):
        axes[1, 1].annotate(str(sl), (vs, sl), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=8)
    axes[1, 1].set_xlabel("Vocabulary Size")
    axes[1, 1].set_ylabel("Sequence Length (tokens)")
    axes[1, 1].set_title("Sequence Length for Fixed Text\nFewer tokens = less attention compute",
                         fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    comp_text = (
        "COMPRESSION ANALYSIS\n"
        "====================\n\n"
        "Characters per Token (CPT):\n"
        "  Higher = better compression\n"
        "  = fewer tokens for same text\n"
        "  = less attention compute\n\n"
        "Key findings:\n"
        f"  V=256 (bytes only): ~1.0 CPT\n"
        f"  V=300 English: {results['English'][2]:.2f} CPT\n"
        f"  V=800 English: {results['English'][7]:.2f} CPT\n"
        f"  V=1000 English: {results['English'][-1]:.2f} CPT\n\n"
        "Content-type effects:\n"
        "  - Trained-on content: best CPT\n"
        "  - Code: moderate (training has\n"
        "    some code patterns)\n"
        "  - Random text: ~1.0 CPT\n"
        "    (no learnable patterns)\n\n"
        "Diminishing returns: doubling\n"
        "vocab from 500->1000 gives\n"
        "less gain than 256->500."
    )
    axes[1, 2].text(0.05, 0.95, comp_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("BPE Compression Ratio Analysis: Vocabulary Size vs Efficiency",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_compression_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/02_compression_analysis.png")

    return results, actual_vocab_sizes


# ---------------------------------------------------------------------------
# Example 3: Tokenization Examples
# ---------------------------------------------------------------------------
def example_3_tokenization_examples():
    """Show how various texts get tokenized with visual token boundaries."""
    print("\n" + "=" * 60)
    print("Example 3: Tokenization Examples")
    print("=" * 60)

    trainer = BPETrainer()
    tokenizer = trainer.train(TRAINING_CORPUS, vocab_size=600)

    examples = {
        "English": "The transformer architecture revolutionized natural language processing.",
        "Python code": "def attention(q, k, v):\n    return softmax(q @ k.T) @ v",
        "Unicode (CJK)": "\u4f60\u597d\u4e16\u754c",
        "Emoji": "Hello! \U0001f600\U0001f680\U0001f30d",
        "Numbers": "GPT-4 has 1.76 trillion parameters",
        "Mixed": "The \u6a21\u578b uses 32 layers for attention(\u6ce8\u610f\u529b)",
    }

    print(f"\n  Tokenizer vocab size: {tokenizer.vocab_size}")

    example_data = []
    for name, text in examples.items():
        ids = tokenizer.encode(text)
        tokens = []
        for tid in ids:
            token = tokenizer._id_to_token[tid]
            if isinstance(token, bytes):
                try:
                    tokens.append(token.decode("utf-8"))
                except UnicodeDecodeError:
                    tokens.append(f"<0x{token.hex()}>")
            else:
                tokens.append(token)

        cpt = tokenizer.characters_per_token(text)
        example_data.append((name, text, ids, tokens, cpt))

        print(f"\n  {name}:")
        print(f"    Text: {text!r}")
        print(f"    Tokens ({len(ids)}): {tokens}")
        print(f"    IDs: {ids}")
        print(f"    CPT: {cpt:.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    names = [d[0] for d in example_data]
    token_counts = [len(d[2]) for d in example_data]
    char_counts = [len(d[1]) for d in example_data]

    x = np.arange(len(names))
    axes[0, 0].bar(x - 0.15, char_counts, 0.3, label="Characters", color=COLORS["blue"], edgecolor="white")
    axes[0, 0].bar(x + 0.15, token_counts, 0.3, label="Tokens", color=COLORS["green"], edgecolor="white")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, fontsize=8, rotation=20)
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Characters vs Tokens by Input Type\nGood compression = large gap between bars",
                         fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    cpts = [d[4] for d in example_data]
    bar_colors = [COLORS["green"] if c > 2.5 else COLORS["orange"] if c > 1.5 else COLORS["red"]
                  for c in cpts]
    axes[0, 1].barh(range(len(names)), cpts, color=bar_colors, edgecolor="white")
    axes[0, 1].set_yticks(range(len(names)))
    axes[0, 1].set_yticklabels(names, fontsize=9)
    axes[0, 1].set_xlabel("Characters per Token")
    axes[0, 1].axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Byte-level (CPT=1)")
    axes[0, 1].set_title("Compression Efficiency by Input Type\nGreen = good, Red = near byte-level",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis="x")

    english_data = example_data[0]
    english_tokens = english_data[3]
    token_lens = [len(t) for t in english_tokens]
    token_colors = plt.cm.Set3(np.linspace(0, 1, len(english_tokens)))

    axes[0, 2].bar(range(len(english_tokens)), token_lens, color=token_colors, edgecolor="white")
    axes[0, 2].set_xticks(range(len(english_tokens)))
    display_tokens = [t if len(t) <= 6 else t[:4] + ".." for t in english_tokens]
    axes[0, 2].set_xticklabels(display_tokens, fontsize=6, rotation=60, family="monospace")
    axes[0, 2].set_ylabel("Token Length (chars)")
    axes[0, 2].set_title("Token Lengths: English Example\nVaried granularity from bytes to subwords",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    all_token_lens = []
    for d in example_data:
        for t in d[3]:
            all_token_lens.append(len(t))

    max_len = max(all_token_lens) if all_token_lens else 1
    bins = range(1, max_len + 2)
    axes[1, 0].hist(all_token_lens, bins=bins, color=COLORS["steel"], edgecolor="white", align="left")
    axes[1, 0].set_xlabel("Token Length (characters)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Distribution of Token Lengths (All Examples)\nMost tokens are 1-4 characters",
                         fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    code_data = example_data[1]
    code_tokens = code_data[3]
    code_ids = code_data[2]
    category_colors = []
    for t in code_tokens:
        if t.strip().isalpha():
            category_colors.append(COLORS["blue"])
        elif t.strip().isdigit():
            category_colors.append(COLORS["orange"])
        elif t.strip() in "()[]{}:.,;@":
            category_colors.append(COLORS["red"])
        elif t.strip() == "" or t in [" ", "\n", "\t"]:
            category_colors.append(COLORS["green"])
        else:
            category_colors.append(COLORS["purple"])

    axes[1, 1].bar(range(len(code_tokens)), [len(t) for t in code_tokens],
                   color=category_colors, edgecolor="white")
    axes[1, 1].set_xticks(range(len(code_tokens)))
    code_display = [repr(t)[1:-1] if t.strip() == "" else (t[:5] + ".." if len(t) > 5 else t)
                    for t in code_tokens]
    axes[1, 1].set_xticklabels(code_display, fontsize=6, rotation=60, family="monospace")
    axes[1, 1].set_ylabel("Token Length")
    axes[1, 1].set_title("Code Tokenization Detail\nBlue=alpha, Orange=digits, Red=punct, Green=space",
                         fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    tok_text = (
        "TOKENIZATION EXAMPLES\n"
        "=====================\n\n"
        f"Vocab size: {tokenizer.vocab_size}\n\n"
    )
    for name, text, ids, tokens, cpt in example_data:
        tok_text += f"{name}:\n"
        tok_text += f"  {len(ids)} tokens, CPT={cpt:.2f}\n"
    tok_text += (
        "\nKey observations:\n"
        "  - English: best compression\n"
        "    (corpus is English-heavy)\n"
        "  - CJK/emoji: poor compression\n"
        "    (falls to byte-level)\n"
        "  - Code: moderate (some code\n"
        "    patterns in training data)\n"
        "  - Numbers: depends on whether\n"
        "    number patterns were seen"
    )
    axes[1, 2].text(0.05, 0.95, tok_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("BPE Tokenization: How Different Text Types Get Tokenized",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_tokenization_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/03_tokenization_examples.png")


# ---------------------------------------------------------------------------
# Example 4: Vocabulary Size Tradeoff
# ---------------------------------------------------------------------------
def example_4_vocab_size_tradeoff():
    """Sequence length vs vocab size for fixed texts, plus embedding memory."""
    print("\n" + "=" * 60)
    print("Example 4: Vocabulary Size Tradeoff")
    print("=" * 60)

    trainer = BPETrainer()
    vocab_sizes = [256, 280, 300, 350, 400, 500, 600, 800, 1000]

    test_text = (
        "The transformer architecture uses multi-head self-attention to process "
        "input sequences in parallel. Each attention head learns different aspects "
        "of the relationships between tokens in the sequence."
    )

    seq_lengths = []
    actual_vs = []
    embedding_sizes_mb = []
    d_model = 4096

    print(f"\n  Test text: {test_text[:60]}...")
    print(f"  Text length: {len(test_text)} chars")
    print(f"\n  {'Vocab':>8} {'Seq Len':>10} {'Embed MB':>12} {'Attn Cost':>14}")
    print(f"  {'-'*48}")

    for vs in vocab_sizes:
        tok = trainer.train(TRAINING_CORPUS, vocab_size=vs)
        ids = tok.encode(test_text)
        n = len(ids)
        seq_lengths.append(n)
        actual_vs.append(tok.vocab_size)

        embed_mb = tok.vocab_size * d_model * 2 / (1024 ** 2)
        embedding_sizes_mb.append(embed_mb)
        attn_cost = n * n
        print(f"  {tok.vocab_size:>8} {n:>10} {embed_mb:>12.1f} {attn_cost:>14,}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].plot(actual_vs, seq_lengths, "o-", color=COLORS["red"], linewidth=2, markersize=6)
    for i, (vs, sl) in enumerate(zip(actual_vs, seq_lengths)):
        axes[0, 0].annotate(str(sl), (vs, sl), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=8)
    axes[0, 0].set_xlabel("Vocabulary Size")
    axes[0, 0].set_ylabel("Sequence Length (tokens)")
    axes[0, 0].set_title("Sequence Length vs Vocab Size (Fixed Text)\nFewer tokens = less attention compute",
                         fontsize=10, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(actual_vs, embedding_sizes_mb, "s-", color=COLORS["blue"], linewidth=2, markersize=6)
    axes[0, 1].set_xlabel("Vocabulary Size")
    axes[0, 1].set_ylabel("Embedding Table Size (MB, FP16)")
    axes[0, 1].set_title(f"Embedding Memory vs Vocab Size (d={d_model})\nLinear growth: larger vocab = more parameters",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    attn_costs = [n * n for n in seq_lengths]
    axes[0, 2].plot(actual_vs, [c / 1000 for c in attn_costs], "o-",
                    color=COLORS["purple"], linewidth=2, markersize=6)
    axes[0, 2].set_xlabel("Vocabulary Size")
    axes[0, 2].set_ylabel(r"Attention Cost $O(n^2)$ (thousands)")
    axes[0, 2].set_title(r"Attention Compute $\propto n^2$" + "\nQuadratic savings from shorter sequences",
                         fontsize=10, fontweight="bold")
    axes[0, 2].grid(True, alpha=0.3)

    norm_seq = [s / seq_lengths[0] for s in seq_lengths]
    norm_embed = [e / embedding_sizes_mb[0] for e in embedding_sizes_mb]
    norm_attn = [a / attn_costs[0] for a in attn_costs]

    axes[1, 0].plot(actual_vs, norm_seq, "o-", color=COLORS["red"], linewidth=2, label="Seq length")
    axes[1, 0].plot(actual_vs, norm_embed, "s-", color=COLORS["blue"], linewidth=2, label="Embed memory")
    axes[1, 0].plot(actual_vs, norm_attn, "^-", color=COLORS["purple"], linewidth=2, label="Attn cost")
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.3)
    axes[1, 0].set_xlabel("Vocabulary Size")
    axes[1, 0].set_ylabel("Normalized Value (relative to V=256)")
    axes[1, 0].set_title("All Costs Normalized to V=256 Baseline\nTradeoff: embedding grows, attention shrinks",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    n_layers = 32
    n_heads = 32
    d_head = d_model // n_heads
    bytes_fp16 = 2

    kv_cache_mb = [2 * n_layers * n_heads * sl * d_head * bytes_fp16 / (1024 ** 2)
                   for sl in seq_lengths]

    axes[1, 1].plot(actual_vs, kv_cache_mb, "o-", color=COLORS["coral"], linewidth=2, markersize=6)
    for i, (vs, mb) in enumerate(zip(actual_vs, kv_cache_mb)):
        if i % 2 == 0:
            axes[1, 1].annotate(f"{mb:.0f}", (vs, mb), textcoords="offset points",
                               xytext=(0, 8), ha="center", fontsize=8)
    axes[1, 1].set_xlabel("Vocabulary Size")
    axes[1, 1].set_ylabel("KV Cache Size (MB, 7B model FP16)")
    axes[1, 1].set_title("KV Cache Memory for Fixed Text\nDirectly proportional to sequence length",
                         fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    tradeoff_text = (
        "VOCABULARY SIZE TRADEOFF\n"
        "========================\n\n"
        "Smaller vocab (e.g. 256):\n"
        "  + Small embedding table\n"
        "  - Long sequences\n"
        "  - High attention cost O(n^2)\n"
        "  - Large KV cache\n\n"
        "Larger vocab (e.g. 50K):\n"
        "  + Short sequences\n"
        "  + Low attention cost\n"
        "  + Small KV cache\n"
        "  - Large embedding table\n"
        "  - Rare tokens under-trained\n\n"
        "Production choices:\n"
        "  GPT-2:  50,257 tokens\n"
        "  GPT-4:  100,256 tokens\n"
        "  LLaMA:  32,000 tokens\n"
        "  Mistral: 32,000 tokens\n\n"
        "Sweet spot: 32K-100K tokens\n"
        "balances all costs."
    )
    axes[1, 2].text(0.05, 0.95, tradeoff_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Vocabulary Size Tradeoff: Sequence Length, Memory, and Compute",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_vocab_size_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/04_vocab_size_tradeoff.png")


# ---------------------------------------------------------------------------
# Example 5: Byte Fallback and Special Token Handling
# ---------------------------------------------------------------------------
def example_5_byte_fallback_and_specials():
    """Demonstrate byte fallback for unseen characters and special token handling."""
    print("\n" + "=" * 60)
    print("Example 5: Byte Fallback & Special Token Handling")
    print("=" * 60)

    trainer = BPETrainer()
    specials = ["<|endoftext|>", "<|pad|>", "<|system|>"]
    tokenizer = trainer.train(TRAINING_CORPUS, vocab_size=600, special_tokens=specials)

    print(f"\n  Vocab size: {tokenizer.vocab_size}")
    print(f"  Special tokens: {specials}")

    byte_fallback_tests = [
        ("Trained English", "The quick brown fox"),
        ("Chinese (unseen)", "\u4f60\u597d\u4e16\u754c\uff01"),
        ("Arabic (unseen)", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"),
        ("Emoji (unseen)", "\U0001f600\U0001f680\U0001f30d\U0001f525"),
        ("Georgian (unseen)", "\u10e6\u10d4\u10da\u10d0\u10d7\u10d8"),
        ("Japanese (unseen)", "\u3053\u3093\u306b\u3061\u306f"),
    ]

    print(f"\n  Byte fallback demonstration:")
    fallback_data = []
    for name, text in byte_fallback_tests:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        n_tokens = len(ids)
        n_chars = len(text)
        n_bytes = len(text.encode("utf-8"))
        roundtrip_ok = decoded == text

        byte_only = sum(1 for tid in ids if tid < 256)
        byte_pct = byte_only / n_tokens * 100 if n_tokens > 0 else 0

        fallback_data.append((name, n_chars, n_bytes, n_tokens, byte_pct, roundtrip_ok))

        print(f"\n  {name}: {text!r}")
        print(f"    Chars={n_chars}, Bytes={n_bytes}, Tokens={n_tokens}")
        print(f"    Byte tokens: {byte_only}/{n_tokens} ({byte_pct:.0f}%)")
        print(f"    Roundtrip: {'PASS' if roundtrip_ok else 'FAIL'}")

    print(f"\n  Special token handling:")
    special_tests = [
        ("End of text", "<|endoftext|>"),
        ("Embedded", "Hello<|endoftext|>World"),
        ("Multiple", "<|system|>Prompt<|endoftext|>Response<|pad|>"),
        ("At boundaries", "<|endoftext|>text<|pad|>"),
    ]

    special_data = []
    for name, text in special_tests:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        roundtrip_ok = decoded == text

        special_count = sum(1 for tid in ids if tokenizer._id_to_token.get(tid) in specials)
        special_data.append((name, text, len(ids), special_count, roundtrip_ok))

        print(f"\n  {name}: {text!r}")
        print(f"    Tokens: {len(ids)}, Special: {special_count}")
        print(f"    IDs: {ids}")
        print(f"    Roundtrip: {'PASS' if roundtrip_ok else 'FAIL'}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    names_bf = [d[0] for d in fallback_data]
    byte_pcts = [d[4] for d in fallback_data]
    bf_colors = [COLORS["green"] if p < 30 else COLORS["orange"] if p < 80 else COLORS["red"]
                 for p in byte_pcts]
    axes[0, 0].barh(range(len(names_bf)), byte_pcts, color=bf_colors, edgecolor="white")
    axes[0, 0].set_yticks(range(len(names_bf)))
    axes[0, 0].set_yticklabels(names_bf, fontsize=9)
    axes[0, 0].set_xlabel("Byte Fallback Tokens (%)")
    axes[0, 0].axvline(50, color="black", linestyle="--", linewidth=1, alpha=0.3)
    axes[0, 0].set_title("Byte Fallback Usage by Script\nGreen=mostly merged, Red=mostly byte-level",
                         fontsize=10, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    x_bf = np.arange(len(names_bf))
    char_counts_bf = [d[1] for d in fallback_data]
    byte_counts_bf = [d[2] for d in fallback_data]
    token_counts_bf = [d[3] for d in fallback_data]

    axes[0, 1].bar(x_bf - 0.2, char_counts_bf, 0.2, label="Characters", color=COLORS["blue"], edgecolor="white")
    axes[0, 1].bar(x_bf, byte_counts_bf, 0.2, label="UTF-8 Bytes", color=COLORS["orange"], edgecolor="white")
    axes[0, 1].bar(x_bf + 0.2, token_counts_bf, 0.2, label="Tokens", color=COLORS["red"], edgecolor="white")
    axes[0, 1].set_xticks(x_bf)
    axes[0, 1].set_xticklabels(names_bf, fontsize=7, rotation=30)
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Characters vs Bytes vs Tokens\nMulti-byte scripts: tokens ~ bytes (no merges)",
                         fontsize=10, fontweight="bold")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    expansion = [d[3] / d[1] for d in fallback_data]
    axes[0, 2].bar(range(len(names_bf)), expansion, color=COLORS["purple"], edgecolor="white")
    axes[0, 2].set_xticks(range(len(names_bf)))
    axes[0, 2].set_xticklabels(names_bf, fontsize=7, rotation=30)
    axes[0, 2].set_ylabel("Tokens / Characters")
    axes[0, 2].axhline(1.0, color="black", linestyle="--", linewidth=1, label="1:1 ratio")
    axes[0, 2].set_title("Token Expansion Factor by Script\nMulti-byte chars expand to >1 token each",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    sp_names = [d[0] for d in special_data]
    sp_total = [d[2] for d in special_data]
    sp_special = [d[3] for d in special_data]
    sp_regular = [t - s for t, s in zip(sp_total, sp_special)]

    x_sp = np.arange(len(sp_names))
    axes[1, 0].bar(x_sp, sp_regular, 0.5, label="Regular tokens", color=COLORS["blue"], edgecolor="white")
    axes[1, 0].bar(x_sp, sp_special, 0.5, bottom=sp_regular, label="Special tokens",
                   color=COLORS["red"], edgecolor="white")
    axes[1, 0].set_xticks(x_sp)
    axes[1, 0].set_xticklabels(sp_names, fontsize=8, rotation=20)
    axes[1, 0].set_ylabel("Token Count")
    axes[1, 0].set_title("Special Token Handling\nSpecial tokens are never split by BPE merges",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    roundtrip_texts = [
        "Hello world!",
        "\u4f60\u597d\u4e16\u754c",
        "\U0001f600\U0001f680\U0001f30d",
        "def f(x): return x**2",
        "<|endoftext|>test<|pad|>",
        "a" * 500,
        "\t\n  spaces\t\n",
        "Mix: \u4f60\u597d + Hello + \U0001f600",
    ]
    roundtrip_results = []
    for text in roundtrip_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        ok = decoded == text
        roundtrip_results.append(ok)

    all_pass = all(roundtrip_results)
    result_colors = [COLORS["green"] if r else COLORS["red"] for r in roundtrip_results]
    axes[1, 1].barh(range(len(roundtrip_texts)), [1] * len(roundtrip_texts),
                    color=result_colors, edgecolor="white")
    rt_labels = [t[:35] + "..." if len(t) > 35 else t for t in roundtrip_texts]
    axes[1, 1].set_yticks(range(len(roundtrip_texts)))
    axes[1, 1].set_yticklabels([ascii(l)[1:-1] for l in rt_labels], fontsize=7, family="monospace")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_xticks([])
    status = "ALL PASS" if all_pass else "SOME FAIL"
    axes[1, 1].set_title(f"Roundtrip Verification: decode(encode(x)) == x\n{status} ({sum(roundtrip_results)}/{len(roundtrip_results)})",
                         fontsize=10, fontweight="bold")
    axes[1, 1].invert_yaxis()

    axes[1, 2].axis("off")
    fb_text = (
        "BYTE FALLBACK & SPECIALS\n"
        "========================\n\n"
        "Byte Fallback:\n"
        "  - Base vocab: 256 byte tokens\n"
        "  - ANY UTF-8 byte encodable\n"
        "  - No <unk> token needed\n"
        "  - Unseen chars: byte tokens\n"
        "  - Seen chars: merged tokens\n\n"
        "Special Tokens:\n"
        "  - Never split by BPE merges\n"
        "  - Encode to single token ID\n"
        "  - Detected before BPE encoding\n"
        "  - Preserved through roundtrip\n\n"
        "Roundtrip guarantee:\n"
        "  decode(encode(text)) == text\n"
        "  for ALL valid UTF-8 strings.\n\n"
        "This is the key advantage of\n"
        "byte-level BPE over word-level\n"
        "or character-level tokenization."
    )
    axes[1, 2].text(0.05, 0.95, fb_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Byte Fallback, Special Tokens, and Roundtrip Verification",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_byte_fallback_specials.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/05_byte_fallback_specials.png")


# ---------------------------------------------------------------------------
# Example 6: Inference Impact Analysis
# ---------------------------------------------------------------------------
def example_6_inference_impact():
    """Calculate KV cache memory and attention cost for different tokenization granularities."""
    print("\n" + "=" * 60)
    print("Example 6: Inference Impact Analysis")
    print("=" * 60)

    trainer = BPETrainer()

    n_layers = 32
    n_heads = 32
    d_model = 4096
    d_head = d_model // n_heads
    bytes_fp16 = 2

    def kv_cache_mb(seq_len):
        return 2 * n_layers * n_heads * seq_len * d_head * bytes_fp16 / (1024 ** 2)

    def attention_flops(seq_len):
        return 2 * n_layers * n_heads * seq_len * seq_len * d_head

    prompt = (
        "Explain the concept of self-attention in transformer architectures. "
        "Self-attention allows each position in the input sequence to attend to "
        "all other positions, computing a weighted sum of value vectors based on "
        "query-key compatibility scores. The mechanism enables the model to capture "
        "long-range dependencies without recurrence. Multi-head attention extends "
        "this by learning multiple attention patterns in parallel, each focusing "
        "on different aspects of the input relationships."
    )

    vocab_configs = [256, 300, 400, 600, 800, 1000]
    results = []

    print(f"\n  Prompt: {prompt[:60]}...")
    print(f"  Prompt length: {len(prompt)} chars, {len(prompt.encode('utf-8'))} bytes")
    print(f"\n  7B model: {n_layers} layers, {n_heads} heads, d_model={d_model}")
    print(f"\n  {'Vocab':>8} {'Tokens':>8} {'KV MB':>10} {'Attn GFLOPs':>14} {'Saved vs 256':>14}")
    print(f"  {'-'*58}")

    for vs in vocab_configs:
        tok = trainer.train(TRAINING_CORPUS, vocab_size=vs)
        ids = tok.encode(prompt)
        n = len(ids)
        cache_mb = kv_cache_mb(n)
        attn_gf = attention_flops(n) / 1e9

        results.append({
            "vocab_size": tok.vocab_size,
            "seq_len": n,
            "kv_cache_mb": cache_mb,
            "attn_gflops": attn_gf,
        })

        base_n = results[0]["seq_len"] if results else n
        saved = (1 - n / base_n) * 100 if base_n > 0 else 0
        print(f"  {tok.vocab_size:>8} {n:>8} {cache_mb:>10.2f} {attn_gf:>14.2f} {saved:>13.1f}%")

    batch_prompt = prompt * 5
    batch_sizes = [1, 4, 8, 16, 32]
    tok_small = trainer.train(TRAINING_CORPUS, vocab_size=300)
    tok_large = trainer.train(TRAINING_CORPUS, vocab_size=800)

    n_small = len(tok_small.encode(batch_prompt))
    n_large = len(tok_large.encode(batch_prompt))

    print(f"\n  Batch scaling with prompt x5:")
    print(f"  V=300: {n_small} tokens, V=800: {n_large} tokens")
    print(f"\n  {'Batch':>8} {'V=300 KV (GB)':>16} {'V=800 KV (GB)':>16} {'Savings (GB)':>14}")
    print(f"  {'-'*58}")

    batch_data = []
    for bs in batch_sizes:
        kv_small = kv_cache_mb(n_small) * bs / 1024
        kv_large = kv_cache_mb(n_large) * bs / 1024
        saved = kv_small - kv_large
        batch_data.append((bs, kv_small, kv_large, saved))
        print(f"  {bs:>8} {kv_small:>16.3f} {kv_large:>16.3f} {saved:>14.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    vs_list = [r["vocab_size"] for r in results]
    sl_list = [r["seq_len"] for r in results]
    kv_list = [r["kv_cache_mb"] for r in results]
    af_list = [r["attn_gflops"] for r in results]

    ax2 = axes[0, 0].twinx()
    l1 = axes[0, 0].plot(vs_list, sl_list, "o-", color=COLORS["red"], linewidth=2,
                          markersize=6, label="Sequence length")
    l2 = ax2.plot(vs_list, kv_list, "s-", color=COLORS["blue"], linewidth=2,
                   markersize=6, label="KV cache (MB)")
    axes[0, 0].set_xlabel("Vocabulary Size")
    axes[0, 0].set_ylabel("Sequence Length", color=COLORS["red"])
    ax2.set_ylabel("KV Cache (MB)", color=COLORS["blue"])
    lines = l1 + l2
    labels_leg = [l.get_label() for l in lines]
    axes[0, 0].legend(lines, labels_leg, fontsize=9, loc="center right")
    axes[0, 0].set_title("Tokens & KV Cache vs Vocab Size\nBoth decrease with larger vocabulary",
                         fontsize=10, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(vs_list, af_list, "^-", color=COLORS["purple"], linewidth=2, markersize=6)
    axes[0, 1].fill_between(vs_list, af_list, alpha=0.15, color=COLORS["purple"])
    axes[0, 1].set_xlabel("Vocabulary Size")
    axes[0, 1].set_ylabel("Attention FLOPs (GFLOPs)")
    axes[0, 1].set_title(r"Attention Cost $O(n^2)$ vs Vocab Size" +
                         "\nQuadratic reduction from shorter sequences",
                         fontsize=10, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    base_sl = results[0]["seq_len"]
    base_kv = results[0]["kv_cache_mb"]
    base_af = results[0]["attn_gflops"]
    pct_saved_sl = [(1 - r["seq_len"] / base_sl) * 100 for r in results]
    pct_saved_kv = [(1 - r["kv_cache_mb"] / base_kv) * 100 for r in results]
    pct_saved_af = [(1 - r["attn_gflops"] / base_af) * 100 for r in results]

    axes[0, 2].plot(vs_list, pct_saved_sl, "o-", color=COLORS["red"], linewidth=2, label="Seq length")
    axes[0, 2].plot(vs_list, pct_saved_kv, "s-", color=COLORS["blue"], linewidth=2, label="KV cache")
    axes[0, 2].plot(vs_list, pct_saved_af, "^-", color=COLORS["purple"], linewidth=2, label="Attn FLOPs")
    axes[0, 2].set_xlabel("Vocabulary Size")
    axes[0, 2].set_ylabel("% Saved vs V=256 Baseline")
    axes[0, 2].set_title("Savings Relative to Byte-Level Baseline\nAttn savings grow faster (quadratic effect)",
                         fontsize=10, fontweight="bold")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    bs_arr = [d[0] for d in batch_data]
    kv_small_arr = [d[1] for d in batch_data]
    kv_large_arr = [d[2] for d in batch_data]

    x_bs = np.arange(len(bs_arr))
    axes[1, 0].bar(x_bs - 0.15, kv_small_arr, 0.3, label=f"V=300 ({n_small} tok)",
                   color=COLORS["coral"], edgecolor="white")
    axes[1, 0].bar(x_bs + 0.15, kv_large_arr, 0.3, label=f"V=800 ({n_large} tok)",
                   color=COLORS["green"], edgecolor="white")
    axes[1, 0].set_xticks(x_bs)
    axes[1, 0].set_xticklabels([f"BS={bs}" for bs in bs_arr])
    axes[1, 0].set_ylabel("KV Cache (GB)")
    axes[1, 0].axhline(80, color="black", linestyle=":", linewidth=1.5, label="A100 80GB")
    axes[1, 0].set_title("KV Cache: V=300 vs V=800 (Batched)\nBetter tokenization = more requests per GPU",
                         fontsize=10, fontweight="bold")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    saved_gb = [d[3] for d in batch_data]
    axes[1, 1].bar(range(len(bs_arr)), saved_gb, color=COLORS["teal"], edgecolor="white")
    for i, s in enumerate(saved_gb):
        axes[1, 1].text(i, s + 0.01, f"{s:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1, 1].set_xticks(range(len(bs_arr)))
    axes[1, 1].set_xticklabels([f"BS={bs}" for bs in bs_arr])
    axes[1, 1].set_ylabel("KV Cache Savings (GB)")
    axes[1, 1].set_title("Memory Saved by Better Tokenization\nSavings scale linearly with batch size",
                         fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    impact_text = (
        "INFERENCE IMPACT\n"
        "================\n\n"
        "Token count determines:\n"
        "  1. Attention cost: O(n^2)\n"
        "  2. KV cache memory: O(n)\n"
        "  3. Generation latency\n"
        "  4. Max concurrent requests\n\n"
        f"This prompt ({len(prompt)} chars):\n"
        f"  V=256: {results[0]['seq_len']} tokens\n"
        f"  V=1000: {results[-1]['seq_len']} tokens\n"
        f"  Attn savings: {pct_saved_af[-1]:.0f}%\n"
        f"  KV savings: {pct_saved_kv[-1]:.0f}%\n\n"
        "Real-world implication:\n"
        "  Better tokenization means\n"
        "  more users per GPU, lower\n"
        "  latency, and lower cost.\n\n"
        "  This is why tokenizer choice\n"
        "  is a critical design decision\n"
        "  for production LLM systems."
    )
    axes[1, 2].text(0.05, 0.95, impact_text, fontsize=9.5, ha="left", va="top",
                    family="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Tokenization Impact on Inference: Memory, Compute, and Cost",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_inference_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: viz/06_inference_impact.png")

    return results


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
        ax.text(0.5, 0.78, "Byte Pair Encoding (BPE) Tokenization",
                fontsize=24, fontweight="bold", ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.68, "The Input Pipeline That Determines Everything Downstream",
                fontsize=13, ha="center", va="center", transform=ax.transAxes, color="gray")
        info_text = (
            "Byte Pair Encoding is the subword tokenization algorithm used by GPT-2,\n"
            "GPT-4, LLaMA, Mistral, and virtually every modern LLM. Starting from a\n"
            "base vocabulary of 256 byte tokens, BPE iteratively merges the most\n"
            "frequent adjacent pairs to build a vocabulary of subword units.\n\n"
            "This demo covers:\n"
            "  1. Training process visualization: merge rules learned step by step\n"
            "  2. Compression ratio analysis: characters per token vs vocab size\n"
            "  3. Tokenization examples: English, code, Unicode, emoji\n"
            "  4. Vocabulary size tradeoff: sequence length, memory, compute\n"
            "  5. Byte fallback & special tokens: handling any UTF-8 input\n"
            "  6. Inference impact: KV cache and attention cost differences\n\n"
            f"Training corpus: {len(TRAINING_CORPUS)} documents\n"
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

        # -- Math page with LaTeX --
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.96, "Mathematical Foundation", fontsize=20, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)

        y = 0.88
        dy = 0.042

        ax.text(0.05, y, "BPE Training Algorithm", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y, r"Given corpus $C$ and target vocabulary size $|V|$:",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$V_0 = \{b_i : i \in [0, 255]\}$ (base byte vocabulary)",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"For $k = 1, \ldots, |V| - 256$:",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.12, y,
                r"$(a^*, b^*) = \arg\max_{(a,b)} \sum_{w \in C} \mathrm{count}((a, b) \text{ in } w) \cdot \mathrm{freq}(w)$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.12, y,
                r"$V_k = V_{k-1} \cup \{a^* \| b^*\}$, replace all $(a^*, b^*)$ with $a^* \| b^*$ in $C$",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.5
        ax.text(0.05, y, "Encoding (Applying Merge Rules)", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y, r"Given text $t$ and ordered merge rules $M = [(a_1, b_1), \ldots, (a_m, b_m)]$:",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{tokens}_0 = [c_1, c_2, \ldots, c_n]$ where $c_i$ are byte tokens of $t$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"For each $(a_k, b_k) \in M$: replace all adjacent $(a_k, b_k)$ with $a_k \| b_k$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{IDs} = [\mathrm{vocab}[\mathrm{tok}] \text{ for tok in final tokens}]$",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.5
        ax.text(0.05, y, "Inference Cost Analysis", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.07, y, r"Let $n$ = sequence length (tokens), $L$ = layers, $h$ = heads, $d_h$ = head dim:",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Attention FLOPs $= 2 \cdot L \cdot h \cdot n^2 \cdot d_h \quad \Rightarrow \quad O(n^2)$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"KV cache memory $= 2 \cdot L \cdot h \cdot n \cdot d_h \cdot b$ bytes $\quad \Rightarrow \quad O(n)$",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Embedding memory $= |V| \cdot d_{\mathrm{model}} \cdot b$ bytes $\quad \Rightarrow \quad O(|V|)$",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.5
        ax.text(0.05, y, "Compression Ratio", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{CPT} = \frac{|\mathrm{text}|}{|\mathrm{encode}(\mathrm{text})|}$ (characters per token, higher = better)",
                fontsize=11, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"$\mathrm{CR} = \frac{|\mathrm{text}|_{\mathrm{bytes}}}{|\mathrm{encode}(\mathrm{text})|}$ (compression ratio)",
                fontsize=11, transform=ax.transAxes)

        y -= dy * 1.5
        ax.text(0.05, y, "Vocabulary Size Tradeoff", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Smaller $|V|$: longer $n$ $\Rightarrow$ attention $O(n^2) \uparrow$, KV cache $O(n) \uparrow$, embedding $O(|V|) \downarrow$",
                fontsize=10, transform=ax.transAxes)
        y -= dy
        ax.text(0.10, y,
                r"Larger $|V|$: shorter $n$ $\Rightarrow$ attention $O(n^2) \downarrow$, KV cache $O(n) \downarrow$, embedding $O(|V|) \uparrow$",
                fontsize=10, transform=ax.transAxes)

        pdf.savefig(fig)
        plt.close(fig)

        # -- Summary page --
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.94, "Summary of Findings", fontsize=20, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)

        summary_items = [
            "1. Training Process: BPE greedily merges the most frequent adjacent byte pairs.",
            "   The best-pair frequency decays as obvious patterns are consumed. Later merges",
            "   create progressively longer tokens by combining previously merged tokens.",
            "",
            "2. Compression: Characters per token (CPT) improves with vocabulary size, with",
            "   diminishing returns. English text trained on English corpus achieves best CPT.",
            "   Random/unseen text compresses poorly (~1.0 CPT, byte-level fallback).",
            "",
            "3. Tokenization: English text compresses well (high CPT). Code is moderate.",
            "   CJK, Arabic, and emoji fall back to byte-level encoding (multiple tokens per",
            "   character) since the training corpus is English-heavy.",
            "",
            "4. Vocab Size Tradeoff: Larger vocabulary produces shorter sequences, reducing",
            "   attention compute (O(n^2)) and KV cache memory (O(n)), but increases embedding",
            "   table size (O(|V|)). Sweet spot: 32K-100K for production LLMs.",
            "",
            "5. Byte Fallback: The 256 base byte tokens guarantee encoding of ANY UTF-8 input.",
            "   No <unk> token needed. Special tokens (e.g., <|endoftext|>) are protected from",
            "   BPE splitting and always encode to a single token ID. Roundtrip correctness",
            "   is guaranteed: decode(encode(text)) == text for all valid UTF-8.",
            "",
            "6. Inference Impact: Better tokenization directly reduces inference cost. Shorter",
            "   sequences mean less attention compute, smaller KV cache, and more concurrent",
            "   requests per GPU. At batch scale, the memory savings from even modest CPT",
            "   improvements translate to significant GPU memory freed for serving.",
        ]
        summary_text = "\n".join(summary_items)
        ax.text(0.06, 0.86, summary_text, fontsize=10, ha="left", va="top",
                transform=ax.transAxes, family="monospace", linespacing=1.3)
        pdf.savefig(fig)
        plt.close(fig)

        # -- Visualization pages --
        titles = {
            "01_training_process.png": "Example 1: BPE Training Process",
            "02_compression_analysis.png": "Example 2: Compression Ratio Analysis",
            "03_tokenization_examples.png": "Example 3: Tokenization Examples",
            "04_vocab_size_tradeoff.png": "Example 4: Vocabulary Size Tradeoff",
            "05_byte_fallback_specials.png": "Example 5: Byte Fallback & Special Tokens",
            "06_inference_impact.png": "Example 6: Inference Impact Analysis",
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
    print("BPE Tokenization Demo")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print(f"Training corpus: {len(TRAINING_CORPUS)} documents")
    print()

    example_1_training_process()
    example_2_compression_analysis()
    example_3_tokenization_examples()
    example_4_vocab_size_tradeoff()
    example_5_byte_fallback_and_specials()
    example_6_inference_impact()
    generate_pdf_report()

    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print(f"Visualizations: {VIZ_DIR}/")
    print(f"Report: {Path(__file__).parent / 'report.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
