"""
Multilayer Perceptron Demo -- Examples, sklearn comparison, and visualizations.

Generates:
- viz/*.png -- Individual visualization files
- report.pdf -- Comprehensive PDF report
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.datasets import load_breast_cancer, make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import importlib.util

_impl_path = Path(__file__).parent / "implementation.py"
_spec = importlib.util.spec_from_file_location("mlp_module", _impl_path)
_mlp_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mlp_mod)

MLP = _mlp_mod.MLP
Layer = _mlp_mod.Layer
ReLU = _mlp_mod.ReLU
Tanh = _mlp_mod.Tanh
Sigmoid = _mlp_mod.Sigmoid
GELU = _mlp_mod.GELU
SiLU = _mlp_mod.SiLU
LeakyReLU = _mlp_mod.LeakyReLU
one_hot_encode = _mlp_mod.one_hot_encode

SEED = 42
np.random.seed(SEED)

VIZ_DIR = Path(__file__).parent / "viz"
VIZ_DIR.mkdir(exist_ok=True)

COLORS = {
    "blue": "steelblue",
    "coral": "coral",
    "green": "#27ae60",
    "red": "#e74c3c",
    "orange": "#f39c12",
    "purple": "#8e44ad",
    "teal": "#3498db",
}

all_figures = []


def save_fig(fig, name, title=None):
    fig.savefig(VIZ_DIR / name, dpi=150, bbox_inches="tight")
    all_figures.append({"fig_path": VIZ_DIR / name, "title": title or name})
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Example 1: XOR Problem
# ──────────────────────────────────────────────────────────────────────────────
def example_1_xor():
    print("=" * 60)
    print("Example 1: XOR Problem")
    print("=" * 60)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.float64)

    np.random.seed(SEED)
    mlp = MLP(
        layer_sizes=[2, 8, 2],
        activations=[ReLU(), "softmax"],
        init_method="he",
    )
    history = mlp.fit(X, y, epochs=500, learning_rate=0.1, verbose=False)

    predictions = mlp.predict(X)
    accuracy = mlp.score(X, y)
    print(f"  Predictions: {predictions}")
    print(f"  True labels: {y.astype(int)}")
    print(f"  Accuracy:    {accuracy:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history, color=COLORS["blue"], linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("XOR Training Loss")
    axes[0].grid(True, alpha=0.3)

    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = mlp.predict_proba(grid)[:, 1].reshape(xx.shape)

    axes[1].contourf(xx, yy, proba, levels=50, cmap="RdYlBu_r", alpha=0.8)
    axes[1].contour(xx, yy, proba, levels=[0.5], colors="black", linewidths=2)
    scatter_colors = [COLORS["red"] if yi == 0 else COLORS["blue"] for yi in y]
    axes[1].scatter(X[:, 0], X[:, 1], c=scatter_colors, s=200, edgecolors="black",
                    linewidths=2, zorder=5)
    for i, (xi, yi) in enumerate(zip(X, y)):
        axes[1].annotate(f"({int(xi[0])},{int(xi[1])})={int(yi)}", xy=(xi[0], xi[1]),
                         xytext=(10, 10), textcoords="offset points", fontsize=9,
                         fontweight="bold")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].set_title("XOR Decision Boundary")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("XOR Problem -- Classic Nonlinear Classification", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "01_xor_problem.png", "XOR Problem -- Classic Nonlinear Classification")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 2: Decision Boundary Visualization (Moons & Circles)
# ──────────────────────────────────────────────────────────────────────────────
def example_2_decision_boundaries():
    print("=" * 60)
    print("Example 2: Decision Boundary Visualization")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    datasets = [
        ("Moons", make_moons(n_samples=300, noise=0.2, random_state=SEED)),
        ("Circles", make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=SEED)),
    ]
    architectures = [
        ("Small (2-4-2)", [2, 4, 2]),
        ("Large (2-32-16-2)", [2, 32, 16, 2]),
    ]

    for row, (ds_name, (X, y)) in enumerate(datasets):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for col, (arch_name, sizes) in enumerate(architectures):
            np.random.seed(SEED)
            acts = [ReLU() for _ in range(len(sizes) - 2)] + ["softmax"]
            mlp = MLP(layer_sizes=sizes, activations=acts, init_method="he")
            mlp.fit(X_scaled, y, epochs=300, learning_rate=0.05, batch_size=32)

            acc = mlp.score(X_scaled, y)
            print(f"  {ds_name} + {arch_name}: accuracy = {acc:.4f}")

            ax = axes[row, col]
            xx, yy = np.meshgrid(
                np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 200),
                np.linspace(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5, 200),
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = mlp.predict(grid).reshape(xx.shape)

            ax.contourf(xx, yy, Z, levels=1, colors=[COLORS["teal"], COLORS["coral"]], alpha=0.3)
            ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
            ax.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], c=COLORS["teal"],
                       s=15, alpha=0.7, label="Class 0")
            ax.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], c=COLORS["coral"],
                       s=15, alpha=0.7, label="Class 1")
            ax.set_title(f"{ds_name} -- {arch_name}\nAccuracy: {acc:.2%}", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Decision Boundaries -- Dataset vs Architecture Complexity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "02_decision_boundaries.png", "Decision Boundaries -- Dataset vs Architecture Complexity")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 3: Training Convergence -- Loss Curves
# ──────────────────────────────────────────────────────────────────────────────
def example_3_convergence():
    print("=" * 60)
    print("Example 3: Training Convergence")
    print("=" * 60)

    X, y = make_moons(n_samples=500, noise=0.2, random_state=SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    learning_rates = [0.001, 0.01, 0.05, 0.1]
    lr_colors = [COLORS["teal"], COLORS["green"], COLORS["orange"], COLORS["red"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for lr, color in zip(learning_rates, lr_colors):
        np.random.seed(SEED)
        mlp = MLP(layer_sizes=[2, 16, 2], activations=[ReLU(), "softmax"], init_method="he")
        history = mlp.fit(X_scaled, y, epochs=200, learning_rate=lr, batch_size=32)
        final_acc = mlp.score(X_scaled, y)
        axes[0].plot(history, color=color, linewidth=1.5, label=f"lr={lr} (acc={final_acc:.2%})")
        print(f"  lr={lr}: final loss={history[-1]:.6f}, accuracy={final_acc:.4f}")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Effect of Learning Rate on Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    arch_configs = [
        ("2-4-2", [2, 4, 2]),
        ("2-16-2", [2, 16, 2]),
        ("2-32-16-2", [2, 32, 16, 2]),
        ("2-64-32-16-2", [2, 64, 32, 16, 2]),
    ]
    arch_colors = [COLORS["teal"], COLORS["green"], COLORS["orange"], COLORS["red"]]

    for (name, sizes), color in zip(arch_configs, arch_colors):
        np.random.seed(SEED)
        acts = [ReLU() for _ in range(len(sizes) - 2)] + ["softmax"]
        mlp = MLP(layer_sizes=sizes, activations=acts, init_method="he")
        history = mlp.fit(X_scaled, y, epochs=200, learning_rate=0.05, batch_size=32)
        final_acc = mlp.score(X_scaled, y)
        axes[1].plot(history, color=color, linewidth=1.5, label=f"{name} (acc={final_acc:.2%})")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Effect of Architecture Depth on Convergence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training Convergence Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "03_convergence.png", "Training Convergence Analysis")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 4: Weight Initialization Comparison
# ──────────────────────────────────────────────────────────────────────────────
def example_4_weight_init():
    print("=" * 60)
    print("Example 4: Weight Initialization Comparison")
    print("=" * 60)

    X, y = make_moons(n_samples=500, noise=0.2, random_state=SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # -- Activation variance across layers for different inits --
    layer_sizes_deep = [2, 64, 64, 64, 64, 64, 2]
    X_probe = np.random.randn(200, 2)

    init_configs = [
        ("He Init", "he"),
        ("Xavier Init", "xavier"),
    ]

    for init_name, init_method in init_configs:
        np.random.seed(SEED)
        acts = [ReLU() for _ in range(5)] + ["softmax"]
        mlp = MLP(layer_sizes=layer_sizes_deep, activations=acts, init_method=init_method)
        mlp.forward(X_probe)

        variances = []
        for layer in mlp.layers[:-1]:
            variances.append(np.var(layer.a))

        color = COLORS["blue"] if init_method == "he" else COLORS["orange"]
        axes[0].plot(range(1, len(variances) + 1), variances, "o-", color=color,
                     linewidth=2, markersize=8, label=init_name)

    # Bad init (large scale)
    np.random.seed(SEED)
    acts_bad = [ReLU() for _ in range(5)] + ["softmax"]
    mlp_bad = MLP(layer_sizes=layer_sizes_deep, activations=acts_bad, init_method="he")
    for layer in mlp_bad.layers:
        layer.W = np.random.randn(*layer.W.shape) * 1.0
    mlp_bad.forward(X_probe)
    variances_bad = [np.var(layer.a) for layer in mlp_bad.layers[:-1]]
    axes[0].plot(range(1, len(variances_bad) + 1), variances_bad, "o--", color=COLORS["red"],
                 linewidth=2, markersize=8, label="Bad Init (std=1.0)")

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Activation Variance")
    axes[0].set_title("Activation Variance Across Layers")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # -- Training curves comparison --
    histories = {}
    for init_name, init_method in [("He", "he"), ("Xavier", "xavier")]:
        np.random.seed(SEED)
        mlp = MLP(
            layer_sizes=[2, 32, 16, 2],
            activations=[ReLU(), ReLU(), "softmax"],
            init_method=init_method,
        )
        hist = mlp.fit(X_scaled, y, epochs=200, learning_rate=0.05, batch_size=32)
        acc = mlp.score(X_scaled, y)
        histories[init_name] = (hist, acc)
        print(f"  {init_name} Init: final loss={hist[-1]:.6f}, accuracy={acc:.4f}")

    axes[1].plot(histories["He"][0], color=COLORS["blue"], linewidth=1.5,
                 label=f"He (acc={histories['He'][1]:.2%})")
    axes[1].plot(histories["Xavier"][0], color=COLORS["orange"], linewidth=1.5,
                 label=f"Xavier (acc={histories['Xavier'][1]:.2%})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss: He vs Xavier (ReLU network)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # -- Weight distribution histograms --
    np.random.seed(SEED)
    mlp_he = MLP(layer_sizes=[2, 64, 64, 2], activations=[ReLU(), ReLU(), "softmax"], init_method="he")
    np.random.seed(SEED)
    mlp_xavier = MLP(layer_sizes=[2, 64, 64, 2], activations=[ReLU(), ReLU(), "softmax"], init_method="xavier")

    he_weights = np.concatenate([l.W.ravel() for l in mlp_he.layers])
    xavier_weights = np.concatenate([l.W.ravel() for l in mlp_xavier.layers])

    axes[2].hist(he_weights, bins=50, alpha=0.6, color=COLORS["blue"], label="He", density=True)
    axes[2].hist(xavier_weights, bins=50, alpha=0.6, color=COLORS["orange"], label="Xavier", density=True)
    axes[2].set_xlabel("Weight Value")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Initial Weight Distributions")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Weight Initialization Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "04_weight_init.png", "Weight Initialization Comparison")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 5: Sklearn Comparison
# ──────────────────────────────────────────────────────────────────────────────
def example_5_sklearn_comparison():
    print("=" * 60)
    print("Example 5: Sklearn Comparison")
    print("=" * 60)

    X, y = make_moons(n_samples=500, noise=0.25, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    np.random.seed(SEED)
    ours = MLP(
        layer_sizes=[2, 32, 16, 2],
        activations=[ReLU(), ReLU(), "softmax"],
        init_method="he",
    )
    ours_history = ours.fit(X_train_s, y_train, epochs=300, learning_rate=0.05, batch_size=32)
    ours_train_acc = ours.score(X_train_s, y_train)
    ours_test_acc = ours.score(X_test_s, y_test)

    sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.05,
        batch_size=32,
        max_iter=300,
        random_state=SEED,
    )
    sklearn_mlp.fit(X_train_s, y_train)
    sk_train_acc = sklearn_mlp.score(X_train_s, y_train)
    sk_test_acc = sklearn_mlp.score(X_test_s, y_test)

    print(f"  Our MLP     -- Train: {ours_train_acc:.4f}, Test: {ours_test_acc:.4f}")
    print(f"  Sklearn MLP -- Train: {sk_train_acc:.4f}, Test: {sk_test_acc:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss curve (ours only; sklearn doesn't expose per-epoch loss easily with SGD)
    axes[0].plot(ours_history, color=COLORS["blue"], linewidth=1.5, label="Our MLP")
    if hasattr(sklearn_mlp, "loss_curve_"):
        axes[0].plot(sklearn_mlp.loss_curve_, color=COLORS["coral"], linewidth=1.5,
                     label="Sklearn MLP", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Decision boundaries side by side
    for idx, (model, name, acc) in enumerate([
        (ours, "Our MLP", ours_test_acc),
        (None, "Sklearn MLP", sk_test_acc),
    ]):
        ax = axes[idx + 1]
        xx, yy = np.meshgrid(
            np.linspace(X_test_s[:, 0].min() - 1, X_test_s[:, 0].max() + 1, 200),
            np.linspace(X_test_s[:, 1].min() - 1, X_test_s[:, 1].max() + 1, 200),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        if model is not None:
            Z = model.predict(grid).reshape(xx.shape)
        else:
            Z = sklearn_mlp.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=1, colors=[COLORS["teal"], COLORS["coral"]], alpha=0.3)
        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
        ax.scatter(X_test_s[y_test == 0, 0], X_test_s[y_test == 0, 1],
                   c=COLORS["teal"], s=20, alpha=0.7, label="Class 0")
        ax.scatter(X_test_s[y_test == 1, 0], X_test_s[y_test == 1, 1],
                   c=COLORS["coral"], s=20, alpha=0.7, label="Class 1")
        ax.set_title(f"{name}\nTest Accuracy: {acc:.2%}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Our MLP vs Sklearn MLPClassifier", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "05_sklearn_comparison.png", "Our MLP vs Sklearn MLPClassifier")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 6: Real Dataset -- Breast Cancer
# ──────────────────────────────────────────────────────────────────────────────
def example_6_real_dataset():
    print("=" * 60)
    print("Example 6: Real Dataset -- Breast Cancer")
    print("=" * 60)

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_features = X_train_s.shape[1]
    architectures = {
        "Small (30-16-2)": [n_features, 16, 2],
        "Medium (30-64-32-2)": [n_features, 64, 32, 2],
        "Large (30-128-64-32-2)": [n_features, 128, 64, 32, 2],
    }

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_list = [COLORS["teal"], COLORS["green"], COLORS["orange"]]

    for (name, sizes), color in zip(architectures.items(), colors_list):
        np.random.seed(SEED)
        acts = [ReLU() for _ in range(len(sizes) - 2)] + ["softmax"]
        mlp = MLP(layer_sizes=sizes, activations=acts, init_method="he")
        history = mlp.fit(X_train_s, y_train, epochs=200, learning_rate=0.01, batch_size=32)
        train_acc = mlp.score(X_train_s, y_train)
        test_acc = mlp.score(X_test_s, y_test)
        results[name] = (train_acc, test_acc, history)
        print(f"  {name}: Train={train_acc:.4f}, Test={test_acc:.4f}")

        axes[0].plot(history, color=color, linewidth=1.5, label=f"{name}")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss by Architecture")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    arch_names = list(results.keys())
    train_accs = [results[n][0] for n in arch_names]
    test_accs = [results[n][1] for n in arch_names]
    x_pos = np.arange(len(arch_names))
    width = 0.35

    bars1 = axes[1].bar(x_pos - width / 2, train_accs, width, label="Train",
                        color=COLORS["blue"], alpha=0.8)
    bars2 = axes[1].bar(x_pos + width / 2, test_accs, width, label="Test",
                        color=COLORS["coral"], alpha=0.8)

    for bar in bars1:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(["Small", "Medium", "Large"], fontsize=9)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Train vs Test Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0.9, 1.02)

    fig.suptitle("Breast Cancer Classification (30 features)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "06_breast_cancer.png", "Breast Cancer Classification (30 features)")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 7: Activation Function Comparison
# ──────────────────────────────────────────────────────────────────────────────
def example_7_activation_comparison():
    print("=" * 60)
    print("Example 7: Activation Function Comparison")
    print("=" * 60)

    X, y = make_moons(n_samples=500, noise=0.2, random_state=SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    activation_configs = [
        ("ReLU", lambda: ReLU()),
        ("GELU", lambda: GELU()),
        ("SiLU", lambda: SiLU()),
        ("Tanh", lambda: Tanh()),
    ]
    act_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    results = {}

    for col, ((act_name, act_fn), color) in enumerate(zip(activation_configs, act_colors)):
        np.random.seed(SEED)
        mlp = MLP(
            layer_sizes=[2, 32, 16, 2],
            activations=[act_fn(), act_fn(), "softmax"],
            init_method="he" if act_name in ("ReLU", "GELU", "SiLU") else "xavier",
        )
        history = mlp.fit(X_scaled, y, epochs=200, learning_rate=0.05, batch_size=32)
        acc = mlp.score(X_scaled, y)
        results[act_name] = (history, acc)
        print(f"  {act_name}: final loss={history[-1]:.6f}, accuracy={acc:.4f}")

        axes[0, col].plot(history, color=color, linewidth=1.5)
        axes[0, col].set_title(f"{act_name}\nAcc: {acc:.2%}", fontsize=11)
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].set_ylabel("Loss")
        axes[0, col].grid(True, alpha=0.3)

        xx, yy = np.meshgrid(
            np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 200),
            np.linspace(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5, 200),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = mlp.predict(grid).reshape(xx.shape)

        axes[1, col].contourf(xx, yy, Z, levels=1, colors=[COLORS["teal"], COLORS["coral"]], alpha=0.3)
        axes[1, col].contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
        axes[1, col].scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1],
                             c=COLORS["teal"], s=10, alpha=0.5)
        axes[1, col].scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1],
                             c=COLORS["coral"], s=10, alpha=0.5)
        axes[1, col].set_title(f"{act_name} Boundary", fontsize=11)
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle("Activation Function Comparison -- Same Architecture, Different Activations",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "07_activation_comparison.png",
             "Activation Function Comparison -- Same Architecture, Different Activations")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 8: Mini-batch vs Full-batch
# ──────────────────────────────────────────────────────────────────────────────
def example_8_batch_comparison():
    print("=" * 60)
    print("Example 8: Mini-batch vs Full-batch Convergence")
    print("=" * 60)

    X, y = make_moons(n_samples=500, noise=0.2, random_state=SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    batch_configs = [
        ("Full Batch (n=500)", None),
        ("Batch Size 128", 128),
        ("Batch Size 32", 32),
        ("SGD (Batch Size 1)", 1),
    ]
    batch_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for (name, bs), color in zip(batch_configs, batch_colors):
        np.random.seed(SEED)
        mlp = MLP(
            layer_sizes=[2, 32, 16, 2],
            activations=[ReLU(), ReLU(), "softmax"],
            init_method="he",
        )
        history = mlp.fit(X_scaled, y, epochs=100, learning_rate=0.05, batch_size=bs)
        acc = mlp.score(X_scaled, y)
        print(f"  {name}: final loss={history[-1]:.6f}, accuracy={acc:.4f}")

        axes[0].plot(history, color=color, linewidth=1.5 if bs != 1 else 0.8,
                     alpha=0.9 if bs != 1 else 0.6, label=f"{name} (acc={acc:.2%})")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Convergence: Mini-batch vs Full-batch")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Smoothed comparison for noisy SGD
    window = 10
    for (name, bs), color in zip(batch_configs, batch_colors):
        np.random.seed(SEED)
        mlp = MLP(
            layer_sizes=[2, 32, 16, 2],
            activations=[ReLU(), ReLU(), "softmax"],
            init_method="he",
        )
        history = mlp.fit(X_scaled, y, epochs=100, learning_rate=0.05, batch_size=bs)

        if len(history) >= window:
            smoothed = np.convolve(history, np.ones(window) / window, mode="valid")
            axes[1].plot(smoothed, color=color, linewidth=1.5, label=name)
        else:
            axes[1].plot(history, color=color, linewidth=1.5, label=name)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (smoothed)")
    axes[1].set_title(f"Smoothed Loss (window={window})")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Mini-batch vs Full-batch Training", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "08_batch_comparison.png", "Mini-batch vs Full-batch Training")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PDF Report
# ──────────────────────────────────────────────────────────────────────────────
def generate_pdf_report():
    print("=" * 60)
    print("Generating PDF Report")
    print("=" * 60)

    report_path = Path(__file__).parent / "report.pdf"
    with PdfPages(str(report_path)) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.65, "Multilayer Perceptron", ha="center", va="center",
                 fontsize=32, fontweight="bold")
        fig.text(0.5, 0.55, "Comprehensive Demo and Analysis", ha="center", va="center",
                 fontsize=18, color="gray")
        fig.text(0.5, 0.40,
                 "From-scratch NumPy implementation with backpropagation,\n"
                 "mini-batch gradient descent, and multiple activation functions.",
                 ha="center", va="center", fontsize=13, style="italic", color="#555555")
        fig.text(0.5, 0.25, f"Seed: {SEED}  |  NumPy only  |  Validated against sklearn",
                 ha="center", va="center", fontsize=11, color="#888888")
        fig.text(0.5, 0.10, "Generated by demo.py", ha="center", va="center",
                 fontsize=10, color="#aaaaaa")
        pdf.savefig(fig)
        plt.close(fig)

        # Summary page
        fig = plt.figure(figsize=(11, 8.5))
        summary_text = (
            "SUMMARY OF EXAMPLES\n\n"
            "1. XOR Problem\n"
            "   The classic nonlinear classification problem that proves hidden layers\n"
            "   can learn representations where classes become linearly separable.\n\n"
            "2. Decision Boundary Visualization\n"
            "   Shows how MLP complexity (width and depth) affects the ability to\n"
            "   carve nonlinear decision boundaries on moons and circles datasets.\n\n"
            "3. Training Convergence\n"
            "   Demonstrates the effect of learning rate and architecture depth\n"
            "   on convergence speed and final loss.\n\n"
            "4. Weight Initialization Comparison\n"
            "   He vs Xavier initialization -- activation variance stability across\n"
            "   layers and impact on training dynamics.\n\n"
            "5. Sklearn Comparison\n"
            "   Validates our from-scratch MLP against sklearn's MLPClassifier,\n"
            "   showing comparable accuracy on the moons dataset.\n\n"
            "6. Real Dataset -- Breast Cancer\n"
            "   30-feature medical dataset with multiple architecture sizes.\n"
            "   Demonstrates practical classification performance.\n\n"
            "7. Activation Function Comparison\n"
            "   ReLU vs GELU vs SiLU vs Tanh -- same architecture, different\n"
            "   activations, comparing convergence and decision boundaries.\n\n"
            "8. Mini-batch vs Full-batch\n"
            "   Compares SGD (batch=1), mini-batch (32, 128), and full-batch\n"
            "   training to show convergence noise vs speed tradeoffs."
        )
        fig.text(0.08, 0.95, summary_text, ha="left", va="top", fontsize=11,
                 fontfamily="monospace", linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)

        # All visualization pages
        for entry in all_figures:
            fig = plt.figure(figsize=(11, 8.5))
            img = plt.imread(str(entry["fig_path"]))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.90])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.5, 0.96, entry["title"], ha="center", va="top",
                     fontsize=14, fontweight="bold")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Report saved to: {report_path}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("*" * 60)
    print("  MULTILAYER PERCEPTRON -- COMPREHENSIVE DEMO")
    print(f"  Seed: {SEED}")
    print("*" * 60)
    print()

    example_1_xor()
    example_2_decision_boundaries()
    example_3_convergence()
    example_4_weight_init()
    example_5_sklearn_comparison()
    example_6_real_dataset()
    example_7_activation_comparison()
    example_8_batch_comparison()
    generate_pdf_report()

    print("=" * 60)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 60)
    print(f"  Visualizations: {VIZ_DIR}/")
    for entry in all_figures:
        print(f"    - {entry['fig_path'].name}")
    print(f"  PDF Report: {Path(__file__).parent / 'report.pdf'}")
    print()


if __name__ == "__main__":
    main()
