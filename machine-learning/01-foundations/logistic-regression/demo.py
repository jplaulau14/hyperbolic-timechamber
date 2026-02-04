"""
Logistic Regression Demo — Examples, sklearn comparison, and visualizations.

Generates:
- viz/*.png — Individual visualization files
- report.pdf — Comprehensive PDF report
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from implementation import LogisticRegression, sigmoid, binary_cross_entropy

SEED = 42
np.random.seed(SEED)

VIZ_DIR = Path(__file__).parent / "viz"
VIZ_DIR.mkdir(exist_ok=True)

COLORS = {
    "class_0": "steelblue",
    "class_1": "coral",
    "decision": "#27ae60",
    "sklearn": "#e74c3c",
    "ours": "#3498db",
    "lr_1": "#3498db",
    "lr_2": "#e74c3c",
    "lr_3": "#27ae60",
    "lr_4": "#f39c12",
}


def example_1_2d_classification():
    """Simple 2D classification with decision boundary visualization."""
    print("=" * 60)
    print("Example 1: Simple 2D Classification with Decision Boundary")
    print("=" * 60)

    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=SEED,
    )

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    accuracy = model.score(X, y)
    print(f"Training accuracy: {accuracy:.4f}")
    print(f"Weights: {model.w}")
    print(f"Bias: {model.b:.6f}")
    print(f"Converged in {len(model.history)} iterations")

    fig, ax = plt.subplots(figsize=(10, 8))

    mask_0 = y == 0
    mask_1 = y == 1
    ax.scatter(X[mask_0, 0], X[mask_0, 1], c=COLORS["class_0"], label="Class 0", alpha=0.7, edgecolors="k", s=60)
    ax.scatter(X[mask_1, 0], X[mask_1, 1], c=COLORS["class_1"], label="Class 1", alpha=0.7, edgecolors="k", s=60)

    slope, intercept = model.decision_boundary_params()
    x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    decision_line = slope * x_range + intercept
    ax.plot(x_range, decision_line, c=COLORS["decision"], linewidth=2, linestyle="--", label="Decision Boundary")

    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title(f"Logistic Regression: 2D Classification (Accuracy: {accuracy:.2%})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_2d_classification.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '01_2d_classification.png'}")
    return {"fig_data": (X, y, model, accuracy), "accuracy": accuracy}


def example_2_convergence():
    """Convergence / loss over training visualization."""
    print("\n" + "=" * 60)
    print("Example 2: Convergence Analysis")
    print("=" * 60)

    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_redundant=0,
        n_informative=5,
        n_clusters_per_class=1,
        random_state=SEED,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(learning_rate=0.5, n_iterations=500, tolerance=1e-8)
    model.fit(X_scaled, y)

    print(f"Initial loss: {model.history[0]:.6f}")
    print(f"Final loss: {model.history[-1]:.6f}")
    print(f"Converged in {len(model.history)} iterations")
    print(f"Final accuracy: {model.score(X_scaled, y):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(model.history, color=COLORS["ours"], linewidth=2)
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
    ax1.set_title("Training Loss Over Iterations", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(model.history, color=COLORS["ours"], linewidth=2)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Binary Cross-Entropy Loss (log scale)", fontsize=12)
    ax2.set_title("Training Loss (Log Scale)", fontsize=14)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_convergence.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '02_convergence.png'}")
    return {"history": model.history}


def example_3_learning_rate_comparison():
    """Compare different learning rates."""
    print("\n" + "=" * 60)
    print("Example 3: Learning Rate Comparison")
    print("=" * 60)

    X, y = make_classification(
        n_samples=200,
        n_features=3,
        n_redundant=0,
        n_informative=3,
        n_clusters_per_class=1,
        random_state=SEED,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    learning_rates = [0.01, 0.1, 0.5, 1.0]
    histories = {}
    final_accuracies = {}

    for lr in learning_rates:
        model = LogisticRegression(learning_rate=lr, n_iterations=200, tolerance=1e-10)
        model.fit(X_scaled, y)
        histories[lr] = model.history
        final_accuracies[lr] = model.score(X_scaled, y)
        print(f"LR={lr}: {len(model.history)} iters, final loss={model.history[-1]:.6f}, accuracy={final_accuracies[lr]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS["lr_1"], COLORS["lr_2"], COLORS["lr_3"], COLORS["lr_4"]]
    for (lr, history), color in zip(histories.items(), colors):
        ax.plot(history, label=f"LR={lr}", color=color, linewidth=2)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
    ax.set_title("Learning Rate Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_learning_rate_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '03_learning_rate_comparison.png'}")
    return {"histories": histories, "accuracies": final_accuracies}


def example_4_sklearn_comparison():
    """Validate against sklearn implementation."""
    print("\n" + "=" * 60)
    print("Example 4: Sklearn Comparison")
    print("=" * 60)

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_redundant=2,
        n_informative=8,
        n_clusters_per_class=2,
        random_state=SEED,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    our_model = LogisticRegression(learning_rate=0.5, n_iterations=1000, tolerance=1e-8)
    our_model.fit(X_train_scaled, y_train)

    sklearn_model = SklearnLogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
    sklearn_model.fit(X_train_scaled, y_train)

    our_train_acc = our_model.score(X_train_scaled, y_train)
    our_test_acc = our_model.score(X_test_scaled, y_test)
    sklearn_train_acc = sklearn_model.score(X_train_scaled, y_train)
    sklearn_test_acc = sklearn_model.score(X_test_scaled, y_test)

    print(f"Our Implementation:")
    print(f"  Train accuracy: {our_train_acc:.4f}")
    print(f"  Test accuracy:  {our_test_acc:.4f}")
    print(f"\nSklearn Implementation:")
    print(f"  Train accuracy: {sklearn_train_acc:.4f}")
    print(f"  Test accuracy:  {sklearn_test_acc:.4f}")

    our_proba = our_model.predict_proba(X_test_scaled)
    sklearn_proba = sklearn_model.predict_proba(X_test_scaled)[:, 1]

    proba_diff = np.abs(our_proba - sklearn_proba)
    print(f"\nProbability difference (mean): {proba_diff.mean():.6f}")
    print(f"Probability difference (max):  {proba_diff.max():.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    metrics = ["Train Acc", "Test Acc"]
    our_scores = [our_train_acc, our_test_acc]
    sklearn_scores = [sklearn_train_acc, sklearn_test_acc]
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, our_scores, width, label="Our Implementation", color=COLORS["ours"])
    bars2 = ax1.bar(x + width / 2, sklearn_scores, width, label="Sklearn", color=COLORS["sklearn"])

    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Accuracy Comparison: Our vs Sklearn", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax1.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    for bar in bars2:
        ax1.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

    ax2 = axes[1]
    ax2.scatter(sklearn_proba, our_proba, alpha=0.5, c=COLORS["ours"], edgecolors="k", s=40)
    ax2.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Agreement")
    ax2.set_xlabel("Sklearn Predicted Probability", fontsize=12)
    ax2.set_ylabel("Our Predicted Probability", fontsize=12)
    ax2.set_title("Probability Agreement", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_sklearn_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '04_sklearn_comparison.png'}")
    return {
        "our_acc": (our_train_acc, our_test_acc),
        "sklearn_acc": (sklearn_train_acc, sklearn_test_acc),
        "proba_diff_mean": proba_diff.mean(),
    }


def example_5_breast_cancer():
    """Real dataset: Breast Cancer classification."""
    print("\n" + "=" * 60)
    print("Example 5: Breast Cancer Dataset (Real Data)")
    print("=" * 60)

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {data.target_names}")
    print(f"Class distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(learning_rate=0.5, n_iterations=2000, tolerance=1e-8)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Converged in {len(model.history)} iterations")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(model.history, color=COLORS["ours"], linewidth=2)
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
    ax1.set_title("Breast Cancer: Training Loss", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    top_k = 10
    sorted_idx = np.argsort(np.abs(model.w))[::-1][:top_k]
    top_weights = model.w[sorted_idx]
    top_features = [feature_names[i] for i in sorted_idx]

    colors = [COLORS["class_1"] if w > 0 else COLORS["class_0"] for w in top_weights]
    bars = ax2.barh(range(top_k), top_weights, color=colors, edgecolor="k")
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(top_features, fontsize=9)
    ax2.set_xlabel("Weight", fontsize=12)
    ax2.set_title(f"Top {top_k} Feature Weights (Breast Cancer)", fontsize=14)
    ax2.axvline(x=0, color="k", linewidth=1)
    ax2.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_breast_cancer.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '05_breast_cancer.png'}")
    return {"train_acc": train_acc, "test_acc": test_acc, "n_iterations": len(model.history)}


def example_6_probability_calibration():
    """Probability calibration visualization (sigmoid curve with data points)."""
    print("\n" + "=" * 60)
    print("Example 6: Probability Calibration (Sigmoid Visualization)")
    print("=" * 60)

    np.random.seed(SEED)
    n_samples = 100
    X = np.random.randn(n_samples, 1) * 2
    prob_true = sigmoid(X.ravel())
    y = (np.random.rand(n_samples) < prob_true).astype(int)

    model = LogisticRegression(learning_rate=1.0, n_iterations=1000)
    model.fit(X, y)

    print(f"True weight: 1.0, Learned weight: {model.w[0]:.4f}")
    print(f"True bias: 0.0, Learned bias: {model.b:.4f}")
    print(f"Training accuracy: {model.score(X, y):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    x_range = np.linspace(-5, 5, 200)
    y_sigmoid = sigmoid(x_range)
    ax1.plot(x_range, y_sigmoid, color=COLORS["ours"], linewidth=3, label="Sigmoid: $\\sigma(z) = 1/(1+e^{-z})$")
    ax1.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Decision threshold (0.5)")
    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax1.fill_between(x_range, 0, y_sigmoid, alpha=0.1, color=COLORS["ours"])
    ax1.set_xlabel("z (logit)", fontsize=12)
    ax1.set_ylabel("$\\sigma(z)$ (probability)", fontsize=12)
    ax1.set_title("Sigmoid Function", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-0.05, 1.05)

    ax2 = axes[1]
    X_sorted_idx = np.argsort(X.ravel())
    X_sorted = X[X_sorted_idx]
    y_sorted = y[X_sorted_idx]
    proba_sorted = model.predict_proba(X_sorted)

    jitter = np.random.uniform(-0.02, 0.02, len(y_sorted))
    mask_0 = y_sorted == 0
    mask_1 = y_sorted == 1
    ax2.scatter(X_sorted[mask_0], y_sorted[mask_0] + jitter[mask_0], c=COLORS["class_0"],
                alpha=0.6, label="Class 0", edgecolors="k", s=50)
    ax2.scatter(X_sorted[mask_1], y_sorted[mask_1] + jitter[mask_1], c=COLORS["class_1"],
                alpha=0.6, label="Class 1", edgecolors="k", s=50)

    ax2.plot(X_sorted, proba_sorted, color=COLORS["decision"], linewidth=3, label="Predicted P(y=1|X)")

    ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Threshold")
    ax2.set_xlabel("Feature X", fontsize=12)
    ax2.set_ylabel("Probability / Class Label", fontsize=12)
    ax2.set_title("Probability Calibration: Data Points + Sigmoid Fit", fontsize=14)
    ax2.legend(fontsize=10, loc="center right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_probability_calibration.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '06_probability_calibration.png'}")
    return {"learned_w": model.w[0], "learned_b": model.b}


def generate_pdf_report(results):
    """Generate comprehensive PDF report."""
    print("\n" + "=" * 60)
    print("Generating PDF Report")
    print("=" * 60)

    pdf_path = Path(__file__).parent / "report.pdf"

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        title_text = "Logistic Regression\nComprehensive Demo Report"
        ax.text(0.5, 0.7, title_text, transform=ax.transAxes, fontsize=28,
                ha="center", va="center", fontweight="bold")

        description = (
            "Binary classification via the sigmoid function.\n"
            "Transforms linear output into probabilities using sigmoid activation,\n"
            "then applies binary cross-entropy loss for training.\n\n"
            "This is the building block for neural network output layers."
        )
        ax.text(0.5, 0.45, description, transform=ax.transAxes, fontsize=14,
                ha="center", va="center", style="italic")

        ax.text(0.5, 0.2, f"Random Seed: {SEED}", transform=ax.transAxes, fontsize=12,
                ha="center", va="center")

        ax.text(0.5, 0.1, "Generated with NumPy-only implementation", transform=ax.transAxes,
                fontsize=10, ha="center", va="center", color="gray")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        ax.text(0.5, 0.95, "Summary of Results", transform=ax.transAxes, fontsize=20,
                ha="center", va="top", fontweight="bold")

        summary_text = f"""
Example 1: 2D Classification
    - Training accuracy: {results['ex1']['accuracy']:.2%}
    - Successfully visualized decision boundary

Example 2: Convergence Analysis
    - Initial loss: {results['ex2']['history'][0]:.6f}
    - Final loss: {results['ex2']['history'][-1]:.6f}
    - Iterations: {len(results['ex2']['history'])}

Example 3: Learning Rate Comparison
    - LR=0.01: Final accuracy {results['ex3']['accuracies'][0.01]:.2%}
    - LR=0.1:  Final accuracy {results['ex3']['accuracies'][0.1]:.2%}
    - LR=0.5:  Final accuracy {results['ex3']['accuracies'][0.5]:.2%}
    - LR=1.0:  Final accuracy {results['ex3']['accuracies'][1.0]:.2%}

Example 4: Sklearn Comparison
    - Our test accuracy: {results['ex4']['our_acc'][1]:.2%}
    - Sklearn test accuracy: {results['ex4']['sklearn_acc'][1]:.2%}
    - Mean probability difference: {results['ex4']['proba_diff_mean']:.6f}

Example 5: Breast Cancer Dataset
    - Training accuracy: {results['ex5']['train_acc']:.2%}
    - Test accuracy: {results['ex5']['test_acc']:.2%}
    - Converged in {results['ex5']['n_iterations']} iterations

Example 6: Probability Calibration
    - Learned weight: {results['ex6']['learned_w']:.4f} (true: 1.0)
    - Learned bias: {results['ex6']['learned_b']:.4f} (true: 0.0)
"""
        ax.text(0.1, 0.85, summary_text, transform=ax.transAxes, fontsize=11,
                ha="left", va="top", family="monospace")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        viz_files = sorted(VIZ_DIR.glob("*.png"))
        for viz_file in viz_files:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            img = plt.imread(viz_file)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(viz_file.stem.replace("_", " ").title(), fontsize=14, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {pdf_path}")


def main():
    print("Logistic Regression Demo")
    print("=" * 60)
    print(f"Random seed: {SEED}")
    print(f"Output directory: {VIZ_DIR}")
    print()

    results = {}

    results["ex1"] = example_1_2d_classification()
    results["ex2"] = example_2_convergence()
    results["ex3"] = example_3_learning_rate_comparison()
    results["ex4"] = example_4_sklearn_comparison()
    results["ex5"] = example_5_breast_cancer()
    results["ex6"] = example_6_probability_calibration()

    generate_pdf_report(results)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"Visualizations saved to: {VIZ_DIR}")
    print(f"PDF report saved to: {Path(__file__).parent / 'report.pdf'}")


if __name__ == "__main__":
    main()
