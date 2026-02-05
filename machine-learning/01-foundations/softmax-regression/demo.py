"""
Softmax Regression Demo - Examples, sklearn comparison, and visualizations.

Generates:
- viz/*.png - Individual visualization files
- report.pdf - Comprehensive PDF report
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from pathlib import Path
from sklearn.datasets import load_iris, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from implementation import SoftmaxRegression, softmax, cross_entropy_loss, one_hot_encode

SEED = 42
np.random.seed(SEED)

VIZ_DIR = Path(__file__).parent / "viz"
VIZ_DIR.mkdir(exist_ok=True)


def example_1_three_class_2d():
    """3-class 2D classification with decision boundary visualization."""
    print("=" * 60)
    print("Example 1: 3-Class 2D Classification with Decision Boundaries")
    print("=" * 60)

    X, y = make_blobs(n_samples=300, centers=3, n_features=2,
                      cluster_std=1.0, random_state=SEED)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SoftmaxRegression(num_features=2, num_classes=3, learning_rate=0.5)
    model.fit(X_scaled, y, epochs=500)

    accuracy = model.score(X_scaled, y)
    print(f"Training Accuracy: {accuracy:.4f}")
    print(f"Final Loss: {model.history[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(10, 8))

    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid_points).reshape(xx.shape)

    colors = ['#e74c3c', '#3498db', '#27ae60']
    cmap_light = ListedColormap(['#fadbd8', '#d4e6f1', '#d5f5e3'])
    cmap_bold = ListedColormap(colors)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)

    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold,
                        edgecolors='white', s=50, linewidths=0.5)

    ax.set_xlabel("Feature 1 (standardized)")
    ax.set_ylabel("Feature 2 (standardized)")
    ax.set_title(f"3-Class Softmax Regression Decision Boundaries\nAccuracy: {accuracy:.2%}")
    ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_decision_boundaries.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '01_decision_boundaries.png'}")

    return {"title": "3-Class Decision Boundaries", "fig_path": VIZ_DIR / "01_decision_boundaries.png",
            "accuracy": accuracy, "final_loss": model.history[-1]}


def example_2_convergence():
    """Convergence / loss over training."""
    print("\n" + "=" * 60)
    print("Example 2: Training Convergence Analysis")
    print("=" * 60)

    X, y = make_blobs(n_samples=200, centers=3, n_features=2,
                      cluster_std=1.2, random_state=SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    learning_rates = [0.01, 0.1, 0.5, 1.0]
    histories = {}

    for lr in learning_rates:
        np.random.seed(SEED)
        model = SoftmaxRegression(num_features=2, num_classes=3, learning_rate=lr)
        model.fit(X_scaled, y, epochs=500)
        histories[lr] = model.history
        print(f"LR={lr}: Final loss = {model.history[-1]:.6f}, Accuracy = {model.score(X_scaled, y):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
    for (lr, history), color in zip(histories.items(), colors):
        axes[0].plot(history, label=f'LR={lr}', color=color, linewidth=1.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 500)

    for (lr, history), color in zip(histories.items(), colors):
        axes[1].plot(history[:100], label=f'LR={lr}', color=color, linewidth=1.5)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("Training Loss (First 100 Epochs)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_convergence.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '02_convergence.png'}")

    return {"title": "Training Convergence", "fig_path": VIZ_DIR / "02_convergence.png",
            "learning_rates": learning_rates, "final_losses": {lr: h[-1] for lr, h in histories.items()}}


def example_3_softmax_visualization():
    """Visualize how softmax transforms logits to probabilities."""
    print("\n" + "=" * 60)
    print("Example 3: Softmax Function Visualization")
    print("=" * 60)

    z = np.linspace(-5, 5, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    logits_3class = np.column_stack([z, np.zeros_like(z), np.zeros_like(z)])
    probs = softmax(logits_3class)

    axes[0, 0].plot(z, probs[:, 0], 'b-', linewidth=2, label='P(class 0)')
    axes[0, 0].plot(z, probs[:, 1], 'r--', linewidth=2, label='P(class 1)')
    axes[0, 0].plot(z, probs[:, 2], 'g:', linewidth=2, label='P(class 2)')
    axes[0, 0].axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='1/3')
    axes[0, 0].set_xlabel("z_0 (logit for class 0)")
    axes[0, 0].set_ylabel("Probability")
    axes[0, 0].set_title("Softmax: Varying z_0 with z_1=z_2=0")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    logits_examples = [
        [-2, 0, 2],
        [0, 0, 0],
        [5, 1, 1],
        [1, 1, 1],
    ]
    bar_width = 0.2
    x = np.arange(3)
    colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12']

    for i, logits in enumerate(logits_examples):
        probs = softmax(np.array(logits))
        offset = (i - 1.5) * bar_width
        axes[0, 1].bar(x + offset, probs, bar_width, label=f'z={logits}', color=colors[i], alpha=0.8)

    axes[0, 1].set_xlabel("Class")
    axes[0, 1].set_ylabel("Probability")
    axes[0, 1].set_title("Softmax Output for Different Logit Vectors")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    extreme_logits = [
        [1000, 1000, 1000],
        [1000, 0, 0],
        [-1000, -999, -998],
    ]
    labels = ['[1000, 1000, 1000]', '[1000, 0, 0]', '[-1000, -999, -998]']

    for i, (logits, label) in enumerate(zip(extreme_logits, labels)):
        probs = softmax(np.array(logits))
        offset = (i - 1) * bar_width * 1.5
        axes[1, 0].bar(x + offset, probs, bar_width * 1.2, label=f'z={label}', color=colors[i], alpha=0.8)
        print(f"softmax({label}) = [{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]")

    axes[1, 0].set_xlabel("Class")
    axes[1, 0].set_ylabel("Probability")
    axes[1, 0].set_title("Numerical Stability: Extreme Logits")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    c_values = [-5, 0, 5, 100]
    base_logits = np.array([1.0, 2.0, 3.0])

    bar_positions = []
    bar_heights = []
    bar_colors_list = []
    bar_labels = []

    for i, c in enumerate(c_values):
        shifted = base_logits + c
        probs = softmax(shifted)
        bar_positions.extend(x + (i - 1.5) * bar_width)
        bar_heights.extend(probs)
        bar_colors_list.extend([colors[i]] * 3)
        if i == 0:
            bar_labels.append(f'c={c}')
        else:
            bar_labels.append(f'c={c}')

    for i, c in enumerate(c_values):
        shifted = base_logits + c
        probs = softmax(shifted)
        offset = (i - 1.5) * bar_width
        axes[1, 1].bar(x + offset, probs, bar_width, label=f'c={c}', color=colors[i], alpha=0.8)

    axes[1, 1].set_xlabel("Class")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].set_title("Shift Invariance: softmax(z) = softmax(z + c)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_softmax_visualization.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '03_softmax_visualization.png'}")

    return {"title": "Softmax Function Visualization", "fig_path": VIZ_DIR / "03_softmax_visualization.png"}


def example_4_temperature_scaling():
    """Temperature scaling effect (T=0.5, T=1, T=2)."""
    print("\n" + "=" * 60)
    print("Example 4: Temperature Scaling Effect")
    print("=" * 60)

    def softmax_with_temperature(z, temperature=1.0):
        return softmax(z / temperature)

    logits = np.array([2.0, 1.0, 0.5])
    temperatures = [0.5, 1.0, 2.0, 5.0]

    print(f"Base logits: {logits}")
    print("-" * 40)
    for T in temperatures:
        probs = softmax_with_temperature(logits, T)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        print(f"T={T}: probs=[{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}], entropy={entropy:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x = np.arange(3)
    bar_width = 0.18
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']

    for i, T in enumerate(temperatures):
        probs = softmax_with_temperature(logits, T)
        offset = (i - 1.5) * bar_width
        axes[0, 0].bar(x + offset, probs, bar_width, label=f'T={T}', color=colors[i], alpha=0.8)

    axes[0, 0].set_xlabel("Class")
    axes[0, 0].set_ylabel("Probability")
    axes[0, 0].set_title(f"Temperature Scaling Effect\nlogits = {list(logits)}")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    T_range = np.linspace(0.1, 5, 100)
    entropies = []
    max_probs = []

    for T in T_range:
        probs = softmax_with_temperature(logits, T)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)
        max_probs.append(np.max(probs))

    axes[0, 1].plot(T_range, entropies, 'b-', linewidth=2)
    axes[0, 1].axhline(y=np.log(3), color='gray', linestyle='--', alpha=0.7, label='Max entropy (uniform)')
    axes[0, 1].set_xlabel("Temperature (T)")
    axes[0, 1].set_ylabel("Entropy")
    axes[0, 1].set_title("Distribution Entropy vs Temperature")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(T_range, max_probs, 'r-', linewidth=2)
    axes[1, 0].axhline(y=1/3, color='gray', linestyle='--', alpha=0.7, label='Uniform (1/3)')
    axes[1, 0].set_xlabel("Temperature (T)")
    axes[1, 0].set_ylabel("Max Probability")
    axes[1, 0].set_title("Confidence (Max Prob) vs Temperature")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    z_vary = np.linspace(-3, 5, 100)
    for T, color in zip([0.5, 1.0, 2.0], colors[:3]):
        logits_vary = np.column_stack([z_vary, np.ones_like(z_vary), np.zeros_like(z_vary)])
        probs_class0 = []
        for logit_row in logits_vary:
            p = softmax_with_temperature(logit_row, T)
            probs_class0.append(p[0])
        axes[1, 1].plot(z_vary, probs_class0, color=color, linewidth=2, label=f'T={T}')

    axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel("z_0 (logit for class 0)")
    axes[1, 1].set_ylabel("P(class 0)")
    axes[1, 1].set_title("Softmax Response Sharpness")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_temperature_scaling.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '04_temperature_scaling.png'}")

    return {"title": "Temperature Scaling", "fig_path": VIZ_DIR / "04_temperature_scaling.png",
            "temperatures": temperatures}


def example_5_sklearn_comparison():
    """Sklearn comparison (validate correctness)."""
    print("\n" + "=" * 60)
    print("Example 5: Sklearn Comparison")
    print("=" * 60)

    X, y = make_blobs(n_samples=500, centers=4, n_features=3,
                      cluster_std=1.5, random_state=SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    our_model = SoftmaxRegression(num_features=3, num_classes=4, learning_rate=0.5)
    our_model.fit(X_train_scaled, y_train, epochs=1000)

    our_train_acc = our_model.score(X_train_scaled, y_train)
    our_test_acc = our_model.score(X_test_scaled, y_test)
    our_probs = our_model.predict_proba(X_test_scaled)

    sklearn_model = LogisticRegression(
        solver='lbfgs', max_iter=1000, random_state=SEED,
        penalty=None,  # No regularization to match our implementation
    )
    sklearn_model.fit(X_train_scaled, y_train)

    sklearn_train_acc = sklearn_model.score(X_train_scaled, y_train)
    sklearn_test_acc = sklearn_model.score(X_test_scaled, y_test)
    sklearn_probs = sklearn_model.predict_proba(X_test_scaled)

    print(f"Our Implementation:")
    print(f"  Train Accuracy: {our_train_acc:.4f}")
    print(f"  Test Accuracy:  {our_test_acc:.4f}")
    print(f"\nSklearn LogisticRegression:")
    print(f"  Train Accuracy: {sklearn_train_acc:.4f}")
    print(f"  Test Accuracy:  {sklearn_test_acc:.4f}")

    print(f"\nAccuracy Difference: {abs(our_test_acc - sklearn_test_acc):.4f}")

    prob_diff = np.abs(our_probs - sklearn_probs)
    print(f"\nProbability Difference Statistics:")
    print(f"  Mean: {prob_diff.mean():.6f}")
    print(f"  Max:  {prob_diff.max():.6f}")
    print(f"  Std:  {prob_diff.std():.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    models = ['Our Implementation', 'Sklearn']
    train_accs = [our_train_acc, sklearn_train_acc]
    test_accs = [our_test_acc, sklearn_test_acc]

    x = np.arange(2)
    width = 0.35

    bars1 = axes[0, 0].bar(x - width/2, train_accs, width, label='Train', color='steelblue', alpha=0.8)
    bars2 = axes[0, 0].bar(x + width/2, test_accs, width, label='Test', color='coral', alpha=0.8)

    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0.8, 1.0)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    for bar in bars1 + bars2:
        height = bar.get_height()
        axes[0, 0].annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    sample_idx = 0
    our_sample_probs = our_probs[sample_idx]
    sklearn_sample_probs = sklearn_probs[sample_idx]

    x_classes = np.arange(4)
    axes[0, 1].bar(x_classes - 0.2, our_sample_probs, 0.4, label='Our Model', color='steelblue', alpha=0.8)
    axes[0, 1].bar(x_classes + 0.2, sklearn_sample_probs, 0.4, label='Sklearn', color='coral', alpha=0.8)
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_title(f'Predicted Probabilities (Sample {sample_idx})')
    axes[0, 1].set_xticks(x_classes)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    axes[1, 0].scatter(sklearn_probs.ravel(), our_probs.ravel(), alpha=0.5, s=10, color='steelblue')
    axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect agreement')
    axes[1, 0].set_xlabel('Sklearn Probability')
    axes[1, 0].set_ylabel('Our Probability')
    axes[1, 0].set_title('Probability Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(-0.05, 1.05)
    axes[1, 0].set_ylim(-0.05, 1.05)

    axes[1, 1].hist(prob_diff.ravel(), bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    axes[1, 1].axvline(x=prob_diff.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean = {prob_diff.mean():.4f}')
    axes[1, 1].set_xlabel('Absolute Probability Difference')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Probability Differences')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_sklearn_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '05_sklearn_comparison.png'}")

    return {"title": "Sklearn Comparison", "fig_path": VIZ_DIR / "05_sklearn_comparison.png",
            "our_test_acc": our_test_acc, "sklearn_test_acc": sklearn_test_acc,
            "mean_prob_diff": prob_diff.mean()}


def example_6_iris_dataset():
    """Real dataset (use sklearn's Iris dataset - classic 3-class problem)."""
    print("\n" + "=" * 60)
    print("Example 6: Iris Dataset (Classic 3-Class Problem)")
    print("=" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(target_names)} classes")
    print(f"Features: {feature_names}")
    print(f"Classes: {list(target_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SoftmaxRegression(num_features=4, num_classes=3, learning_rate=0.5)
    model.fit(X_train_scaled, y_train, epochs=1000)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Final Loss:     {model.history[-1]:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = ['#e74c3c', '#3498db', '#27ae60']
    cmap_bold = ListedColormap(colors)

    X_scaled = scaler.transform(X)

    feature_pairs = [(0, 1), (2, 3)]
    for idx, (f1, f2) in enumerate(feature_pairs):
        ax = axes[0, idx]

        X_2d = X_scaled[:, [f1, f2]]
        model_2d = SoftmaxRegression(num_features=2, num_classes=3, learning_rate=0.5)
        model_2d.fit(X_2d, y, epochs=500)

        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                             np.linspace(y_min, y_max, 150))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        Z = model_2d.predict(grid_points).reshape(xx.shape)

        cmap_light = ListedColormap(['#fadbd8', '#d4e6f1', '#d5f5e3'])
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
        ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)

        for class_idx, (color, name) in enumerate(zip(colors, target_names)):
            mask = y == class_idx
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color,
                      label=name, edgecolors='white', s=50, linewidths=0.5)

        acc_2d = model_2d.score(X_2d, y)
        ax.set_xlabel(feature_names[f1])
        ax.set_ylabel(feature_names[f2])
        ax.set_title(f'Features {f1+1} & {f2+1} (Acc: {acc_2d:.2%})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[1, 0].plot(model.history, 'b-', linewidth=1.5)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Cross-Entropy Loss")
    axes[1, 0].set_title("Training Loss on Iris Dataset")
    axes[1, 0].grid(True, alpha=0.3)

    y_pred = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title(f'Confusion Matrix (Test Acc: {test_acc:.2%})')
    plt.colorbar(im, ax=axes[1, 1])

    axes[1, 1].set_xticks(np.arange(3))
    axes[1, 1].set_yticks(np.arange(3))
    axes[1, 1].set_xticklabels(target_names)
    axes[1, 1].set_yticklabels(target_names)
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('True')

    for i in range(3):
        for j in range(3):
            text = axes[1, 1].text(j, i, cm[i, j], ha="center", va="center",
                                   color="white" if cm[i, j] > cm.max()/2 else "black")

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_iris_dataset.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {VIZ_DIR / '06_iris_dataset.png'}")

    return {"title": "Iris Dataset", "fig_path": VIZ_DIR / "06_iris_dataset.png",
            "train_acc": train_acc, "test_acc": test_acc}


def generate_pdf_report(results):
    """Generate comprehensive PDF report."""
    print("\n" + "=" * 60)
    print("Generating PDF Report")
    print("=" * 60)

    report_path = Path(__file__).parent / "report.pdf"

    with PdfPages(report_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        title_text = "Softmax Regression\nComprehensive Demo Report"
        ax.text(0.5, 0.7, title_text, transform=ax.transAxes, fontsize=24,
                ha='center', va='center', fontweight='bold')

        subtitle = (
            "Multiclass classification with softmax function\n\n"
            "Topics Covered:\n"
            "- Decision boundary visualization\n"
            "- Training convergence analysis\n"
            "- Softmax function properties\n"
            "- Temperature scaling effects\n"
            "- Sklearn comparison and validation\n"
            "- Real-world dataset (Iris)"
        )
        ax.text(0.5, 0.4, subtitle, transform=ax.transAxes, fontsize=12,
                ha='center', va='center')

        footer = f"SEED = {SEED} for reproducibility"
        ax.text(0.5, 0.1, footer, transform=ax.transAxes, fontsize=10,
                ha='center', va='center', style='italic', color='gray')

        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        ax.text(0.5, 0.95, "Summary of Results", transform=ax.transAxes, fontsize=18,
                ha='center', va='top', fontweight='bold')

        summary_lines = []

        if 'example_1' in results:
            r = results['example_1']
            summary_lines.append(f"1. 3-Class Decision Boundaries")
            summary_lines.append(f"   - Training Accuracy: {r['accuracy']:.2%}")
            summary_lines.append(f"   - Final Loss: {r['final_loss']:.6f}")
            summary_lines.append("")

        if 'example_2' in results:
            r = results['example_2']
            summary_lines.append(f"2. Convergence Analysis")
            for lr, loss in r['final_losses'].items():
                summary_lines.append(f"   - LR={lr}: Final Loss = {loss:.6f}")
            summary_lines.append("")

        if 'example_4' in results:
            summary_lines.append(f"3. Temperature Scaling")
            summary_lines.append(f"   - Temperatures tested: {results['example_4']['temperatures']}")
            summary_lines.append(f"   - Lower T -> sharper distribution")
            summary_lines.append(f"   - Higher T -> softer distribution")
            summary_lines.append("")

        if 'example_5' in results:
            r = results['example_5']
            summary_lines.append(f"4. Sklearn Comparison")
            summary_lines.append(f"   - Our Test Accuracy: {r['our_test_acc']:.4f}")
            summary_lines.append(f"   - Sklearn Test Accuracy: {r['sklearn_test_acc']:.4f}")
            summary_lines.append(f"   - Mean Probability Difference: {r['mean_prob_diff']:.6f}")
            summary_lines.append("")

        if 'example_6' in results:
            r = results['example_6']
            summary_lines.append(f"5. Iris Dataset")
            summary_lines.append(f"   - Train Accuracy: {r['train_acc']:.4f}")
            summary_lines.append(f"   - Test Accuracy: {r['test_acc']:.4f}")

        summary_text = "\n".join(summary_lines)
        ax.text(0.1, 0.85, summary_text, transform=ax.transAxes, fontsize=11,
                ha='left', va='top', family='monospace')

        pdf.savefig(fig)
        plt.close(fig)

        for key in ['example_1', 'example_2', 'example_3', 'example_4', 'example_5', 'example_6']:
            if key in results:
                r = results[key]
                img = plt.imread(r['fig_path'])

                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(r['title'], fontsize=14, fontweight='bold', pad=10)

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved: {report_path}")
    return report_path


def main():
    print("=" * 60)
    print("SOFTMAX REGRESSION - COMPREHENSIVE DEMO")
    print("=" * 60)
    print(f"SEED = {SEED}")
    print(f"Output directory: {VIZ_DIR}")
    print()

    results = {}

    results['example_1'] = example_1_three_class_2d()
    results['example_2'] = example_2_convergence()
    results['example_3'] = example_3_softmax_visualization()
    results['example_4'] = example_4_temperature_scaling()
    results['example_5'] = example_5_sklearn_comparison()
    results['example_6'] = example_6_iris_dataset()

    report_path = generate_pdf_report(results)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files:")
    for png_file in sorted(VIZ_DIR.glob("*.png")):
        print(f"  - {png_file}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
