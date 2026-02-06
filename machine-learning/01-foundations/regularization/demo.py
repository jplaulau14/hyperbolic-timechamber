"""
Regularization Demo -- Examples, sklearn comparison, and visualizations.

Generates:
- viz/*.png -- Individual visualization files
- report.pdf -- Comprehensive PDF report
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from implementation import (
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression,
    l1_penalty,
    l2_penalty,
    l1_gradient,
    l2_gradient,
    sgd_with_weight_decay,
    _mse_loss,
)

SEED = 42
np.random.seed(SEED)

VIZ_DIR = Path(__file__).parent / "viz"
VIZ_DIR.mkdir(exist_ok=True)

ALL_FIGURES = []


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(VIZ_DIR / name, dpi=150)
    ALL_FIGURES.append((name, fig))
    plt.close(fig)


# =========================================================================
# Example 1: Overfitting demonstration with polynomial features
# =========================================================================
def example_1_overfitting():
    print("=" * 60)
    print("Example 1: Overfitting Demonstration (Polynomial Features)")
    print("=" * 60)

    np.random.seed(SEED)
    n_train, n_test = 20, 200
    X_train = np.sort(np.random.uniform(-3, 3, n_train))
    y_train = np.sin(X_train) + np.random.normal(0, 0.3, n_train)
    X_test = np.linspace(-3, 3, n_test)
    y_test_true = np.sin(X_test)

    degree = 12
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly.transform(X_test.reshape(-1, 1))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    no_reg = RidgeRegression(lambda_=0.0, learning_rate=0.01, n_iterations=5000)
    no_reg.fit(X_train_scaled, y_train)
    y_pred_noreg = no_reg.predict(X_test_scaled)

    ridge = RidgeRegression(lambda_=0.1, learning_rate=0.01, n_iterations=5000)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)

    lasso = LassoRegression(lambda_=0.05, learning_rate=0.01, n_iterations=5000)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)

    mse_noreg = np.mean((y_test_true - y_pred_noreg) ** 2)
    mse_ridge = np.mean((y_test_true - y_pred_ridge) ** 2)
    mse_lasso = np.mean((y_test_true - y_pred_lasso) ** 2)

    print(f"  Degree-{degree} polynomial, {n_train} training points")
    print(f"  No Regularization  MSE: {mse_noreg:.6f}")
    print(f"  Ridge (L2=0.1)     MSE: {mse_ridge:.6f}")
    print(f"  Lasso (L1=0.05)    MSE: {mse_lasso:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax in axes:
        ax.scatter(X_train, y_train, color="steelblue", s=40, zorder=5,
                   label="Training data", alpha=0.8)
        ax.plot(X_test, y_test_true, "k--", alpha=0.5, label="True function")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.3)

    axes[0].plot(X_test, y_pred_noreg, color="#e74c3c", linewidth=2,
                 label=f"No reg (MSE={mse_noreg:.3f})")
    axes[0].set_title("No Regularization (Overfitting)")
    axes[0].legend(fontsize=8)

    axes[1].plot(X_test, y_pred_ridge, color="#27ae60", linewidth=2,
                 label=f"Ridge L2=0.1 (MSE={mse_ridge:.3f})")
    axes[1].set_title("Ridge (L2) Regularization")
    axes[1].legend(fontsize=8)

    axes[2].plot(X_test, y_pred_lasso, color="#f39c12", linewidth=2,
                 label=f"Lasso L1=0.05 (MSE={mse_lasso:.3f})")
    axes[2].set_title("Lasso (L1) Regularization")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Overfitting Demo: Degree-{degree} Polynomial, {n_train} Points",
                 fontsize=14, y=1.02)
    save_fig(fig, "01_overfitting_demo.png")

    w_noreg, _ = no_reg.get_weights()
    w_ridge, _ = ridge.get_weights()
    w_lasso, _ = lasso.get_weights()

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(w_noreg))
    width = 0.25
    ax.bar(x_pos - width, np.abs(w_noreg), width, label="No Reg", color="#e74c3c", alpha=0.8)
    ax.bar(x_pos, np.abs(w_ridge), width, label="Ridge", color="#27ae60", alpha=0.8)
    ax.bar(x_pos + width, np.abs(w_lasso), width, label="Lasso", color="#f39c12", alpha=0.8)
    ax.set_xlabel("Feature Index (Polynomial Degree)")
    ax.set_ylabel("|Weight|")
    ax.set_title("Weight Magnitudes: No Reg vs Ridge vs Lasso")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"x^{i+1}" for i in range(len(w_noreg))], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(fig, "01b_weight_magnitudes.png")

    return {
        "mse_noreg": mse_noreg,
        "mse_ridge": mse_ridge,
        "mse_lasso": mse_lasso,
    }


# =========================================================================
# Example 2: L1 vs L2 weight comparison (sparsity)
# =========================================================================
def example_2_l1_vs_l2_weights():
    print()
    print("=" * 60)
    print("Example 2: L1 vs L2 Weight Comparison (Sparsity)")
    print("=" * 60)

    np.random.seed(SEED)
    n_samples, n_features, n_informative = 200, 50, 8
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, noise=10.0, random_state=SEED
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    lambda_val = 0.5

    ridge = RidgeRegression(lambda_=lambda_val, learning_rate=0.005,
                            n_iterations=5000, tolerance=1e-9)
    ridge.fit(X, y)
    w_ridge, _ = ridge.get_weights()

    lasso = LassoRegression(lambda_=lambda_val, learning_rate=0.005,
                            n_iterations=5000, tolerance=1e-9)
    lasso.fit(X, y)
    w_lasso, _ = lasso.get_weights()

    threshold = 0.1
    ridge_near_zero = np.sum(np.abs(w_ridge) < threshold)
    lasso_near_zero = np.sum(np.abs(w_lasso) < threshold)

    print(f"  {n_features} features, {n_informative} informative, lambda={lambda_val}")
    print(f"  Ridge weights near zero (|w| < {threshold}): {ridge_near_zero}/{n_features}")
    print(f"  Lasso weights near zero (|w| < {threshold}): {lasso_near_zero}/{n_features}")
    print(f"  Ridge L2 norm: {np.linalg.norm(w_ridge):.4f}")
    print(f"  Lasso L2 norm: {np.linalg.norm(w_lasso):.4f}")
    print(f"  Ridge L1 norm: {np.sum(np.abs(w_ridge)):.4f}")
    print(f"  Lasso L1 norm: {np.sum(np.abs(w_lasso)):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sorted_ridge = np.sort(np.abs(w_ridge))[::-1]
    sorted_lasso = np.sort(np.abs(w_lasso))[::-1]
    x_pos = np.arange(n_features)

    axes[0].bar(x_pos, sorted_ridge, color="#3498db", alpha=0.8)
    axes[0].axhline(y=threshold, color="#e74c3c", linestyle="--", alpha=0.7,
                     label=f"Threshold={threshold}")
    axes[0].set_title(f"Ridge (L2): {ridge_near_zero} near-zero weights")
    axes[0].set_xlabel("Weight Index (sorted by magnitude)")
    axes[0].set_ylabel("|Weight|")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x_pos, sorted_lasso, color="#f39c12", alpha=0.8)
    axes[1].axhline(y=threshold, color="#e74c3c", linestyle="--", alpha=0.7,
                     label=f"Threshold={threshold}")
    axes[1].set_title(f"Lasso (L1): {lasso_near_zero} near-zero weights")
    axes[1].set_xlabel("Weight Index (sorted by magnitude)")
    axes[1].set_ylabel("|Weight|")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"L1 vs L2: Weight Distributions (lambda={lambda_val}, "
                 f"{n_features} features, {n_informative} informative)", fontsize=13)
    save_fig(fig, "02_l1_vs_l2_weights.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(w_ridge, bins=30, color="#3498db", alpha=0.7, edgecolor="black")
    axes[0].set_title("Ridge Weight Distribution")
    axes[0].set_xlabel("Weight Value")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(w_lasso, bins=30, color="#f39c12", alpha=0.7, edgecolor="black")
    axes[1].set_title("Lasso Weight Distribution")
    axes[1].set_xlabel("Weight Value")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Weight Value Histograms: Ridge vs Lasso", fontsize=13)
    save_fig(fig, "02b_weight_histograms.png")

    return {
        "ridge_near_zero": ridge_near_zero,
        "lasso_near_zero": lasso_near_zero,
    }


# =========================================================================
# Example 3: Lambda sweep -- train/val loss curves
# =========================================================================
def example_3_lambda_sweep():
    print()
    print("=" * 60)
    print("Example 3: Lambda Sweep (Train/Val Loss Curves)")
    print("=" * 60)

    np.random.seed(SEED)
    X, y = make_regression(
        n_samples=300, n_features=30, n_informative=10,
        noise=15.0, random_state=SEED
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=SEED
    )

    lambdas = np.logspace(-4, 1.5, 30)
    ridge_train_losses = []
    ridge_val_losses = []
    lasso_train_losses = []
    lasso_val_losses = []

    for lam in lambdas:
        ridge = RidgeRegression(lambda_=lam, learning_rate=0.01,
                                n_iterations=3000, tolerance=1e-9)
        ridge.fit(X_train, y_train)
        ridge_train_losses.append(np.mean((y_train - ridge.predict(X_train)) ** 2))
        ridge_val_losses.append(np.mean((y_val - ridge.predict(X_val)) ** 2))

        lasso = LassoRegression(lambda_=lam, learning_rate=0.01,
                                n_iterations=3000, tolerance=1e-9)
        lasso.fit(X_train, y_train)
        lasso_train_losses.append(np.mean((y_train - lasso.predict(X_train)) ** 2))
        lasso_val_losses.append(np.mean((y_val - lasso.predict(X_val)) ** 2))

    best_ridge_idx = np.argmin(ridge_val_losses)
    best_lasso_idx = np.argmin(lasso_val_losses)

    print(f"  Ridge best lambda: {lambdas[best_ridge_idx]:.6f} "
          f"(val MSE: {ridge_val_losses[best_ridge_idx]:.2f})")
    print(f"  Lasso best lambda: {lambdas[best_lasso_idx]:.6f} "
          f"(val MSE: {lasso_val_losses[best_lasso_idx]:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogx(lambdas, ridge_train_losses, "o-", color="#3498db",
                     label="Train MSE", markersize=3)
    axes[0].semilogx(lambdas, ridge_val_losses, "s-", color="#e74c3c",
                     label="Val MSE", markersize=3)
    axes[0].axvline(x=lambdas[best_ridge_idx], color="#27ae60", linestyle="--",
                     alpha=0.7, label=f"Best lambda={lambdas[best_ridge_idx]:.4f}")
    axes[0].set_xlabel("Lambda (log scale)")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Ridge: Lambda Sweep")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(lambdas, lasso_train_losses, "o-", color="#3498db",
                     label="Train MSE", markersize=3)
    axes[1].semilogx(lambdas, lasso_val_losses, "s-", color="#e74c3c",
                     label="Val MSE", markersize=3)
    axes[1].axvline(x=lambdas[best_lasso_idx], color="#27ae60", linestyle="--",
                     alpha=0.7, label=f"Best lambda={lambdas[best_lasso_idx]:.4f}")
    axes[1].set_xlabel("Lambda (log scale)")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Lasso: Lambda Sweep")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Lambda Sweep: Finding the Bias-Variance Sweet Spot", fontsize=14)
    save_fig(fig, "03_lambda_sweep.png")

    return {
        "best_ridge_lambda": lambdas[best_ridge_idx],
        "best_lasso_lambda": lambdas[best_lasso_idx],
        "best_ridge_val_mse": ridge_val_losses[best_ridge_idx],
        "best_lasso_val_mse": lasso_val_losses[best_lasso_idx],
    }


# =========================================================================
# Example 4: Sparsity visualization -- near-zero weights vs lambda
# =========================================================================
def example_4_sparsity_vs_lambda():
    print()
    print("=" * 60)
    print("Example 4: Sparsity vs Lambda (L1 vs L2)")
    print("=" * 60)

    np.random.seed(SEED)
    n_features = 40
    X, y = make_regression(
        n_samples=200, n_features=n_features, n_informative=8,
        noise=10.0, random_state=SEED
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    lambdas = np.logspace(-3, 1, 25)
    threshold = 0.1

    l1_sparsity = []
    l2_sparsity = []
    l1_norms = []
    l2_norms = []

    for lam in lambdas:
        ridge = RidgeRegression(lambda_=lam, learning_rate=0.005,
                                n_iterations=3000, tolerance=1e-9)
        ridge.fit(X, y)
        w_r, _ = ridge.get_weights()
        l2_sparsity.append(np.sum(np.abs(w_r) < threshold))
        l2_norms.append(np.linalg.norm(w_r))

        lasso = LassoRegression(lambda_=lam, learning_rate=0.005,
                                n_iterations=3000, tolerance=1e-9)
        lasso.fit(X, y)
        w_l, _ = lasso.get_weights()
        l1_sparsity.append(np.sum(np.abs(w_l) < threshold))
        l1_norms.append(np.linalg.norm(w_l))

    print(f"  At lambda=0.001: L1 sparse={l1_sparsity[0]}, L2 sparse={l2_sparsity[0]}")
    print(f"  At lambda=10.0:  L1 sparse={l1_sparsity[-1]}, L2 sparse={l2_sparsity[-1]}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogx(lambdas, l1_sparsity, "o-", color="#f39c12",
                     label="Lasso (L1)", markersize=4, linewidth=2)
    axes[0].semilogx(lambdas, l2_sparsity, "s-", color="#3498db",
                     label="Ridge (L2)", markersize=4, linewidth=2)
    axes[0].set_xlabel("Lambda (log scale)")
    axes[0].set_ylabel(f"Near-Zero Weights (|w| < {threshold})")
    axes[0].set_title("Sparsity: Number of Near-Zero Weights vs Lambda")
    axes[0].axhline(y=n_features, color="gray", linestyle=":", alpha=0.5,
                     label=f"Total features={n_features}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(lambdas, l1_norms, "o-", color="#f39c12",
                     label="Lasso (L1)", markersize=4, linewidth=2)
    axes[1].semilogx(lambdas, l2_norms, "s-", color="#3498db",
                     label="Ridge (L2)", markersize=4, linewidth=2)
    axes[1].set_xlabel("Lambda (log scale)")
    axes[1].set_ylabel("L2 Norm of Weights")
    axes[1].set_title("Weight Norm Shrinkage vs Lambda")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Sparsity and Shrinkage: L1 vs L2 Across Lambda Values", fontsize=14)
    save_fig(fig, "04_sparsity_vs_lambda.png")

    return {
        "l1_sparsity_range": (l1_sparsity[0], l1_sparsity[-1]),
        "l2_sparsity_range": (l2_sparsity[0], l2_sparsity[-1]),
    }


# =========================================================================
# Example 5: Sklearn comparison (Ridge and Lasso)
# =========================================================================
def example_5_sklearn_comparison():
    print()
    print("=" * 60)
    print("Example 5: Sklearn Comparison (Ridge & Lasso)")
    print("=" * 60)

    np.random.seed(SEED)
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=10,
        noise=5.0, random_state=SEED
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y - np.mean(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED
    )

    # ---- Ridge comparison ----
    # sklearn Ridge minimizes: (1/2n)||Xw - y||^2 + alpha/2 * ||w||^2
    # Our implementation uses the same convention with lambda_ = alpha
    alpha_ridge = 1.0

    our_ridge = RidgeRegression(
        lambda_=alpha_ridge, learning_rate=0.01,
        n_iterations=10000, tolerance=1e-10
    )
    our_ridge.fit(X_train, y_train)

    our_ridge_cf = RidgeRegression(lambda_=alpha_ridge)
    our_ridge_cf.fit_closed_form(X_train, y_train)

    # sklearn Ridge: minimizes ||y - Xw||^2 / (2*n_samples) + alpha/2 * ||w||^2
    # when we set alpha = lambda_ * n_samples (because sklearn uses 1/(2n) internally? No.)
    # Actually sklearn Ridge minimizes: ||y - Xw||^2_2 + alpha * ||w||^2_2
    # Note: sklearn does NOT divide by n_samples. So their alpha != our lambda_.
    #
    # Our loss: (1/2n) * ||y - Xw||^2 + (lambda/2) * ||w||^2
    # sklearn:  (1/2n) * ||y - Xw||^2 + (alpha/2) * ||w||^2   (with fit_intercept=True)
    # Actually sklearn uses: ||y - Xw||^2 / (2*n) + alpha * ||w||^2
    # Hmm, let's check. sklearn.linear_model.Ridge minimizes:
    # ||y - Xw||^2_2 + alpha * ||w||^2_2  (no 1/n, no 1/2 on the alpha term)
    #
    # Our GD loss: (1/(2n)) sum(y-Xw)^2 + (lambda/2) ||w||^2
    # Our closed form matches GD: (X^TX + n*lambda*I)w = X^Ty
    # sklearn closed form: (X^TX + alpha*I)w = X^Ty  (where alpha is their parameter)
    # So: n * lambda = alpha_sklearn  =>  alpha_sklearn = n_train * lambda_ours

    n_train = X_train.shape[0]
    sklearn_alpha_ridge = alpha_ridge * n_train

    sk_ridge = Ridge(alpha=sklearn_alpha_ridge, fit_intercept=True)
    sk_ridge.fit(X_train, y_train)

    our_w, our_b = our_ridge_cf.get_weights()
    sk_w, sk_b = sk_ridge.coef_, sk_ridge.intercept_

    w_diff_ridge = np.max(np.abs(our_w - sk_w))
    b_diff_ridge = abs(our_b - sk_b)

    our_r2_ridge = our_ridge_cf.score(X_test, y_test)
    sk_r2_ridge = sk_ridge.score(X_test, y_test)

    print(f"  --- Ridge (our lambda={alpha_ridge}, sklearn alpha={sklearn_alpha_ridge}) ---")
    print(f"  Max weight diff (closed-form vs sklearn): {w_diff_ridge:.8f}")
    print(f"  Bias diff: {b_diff_ridge:.8f}")
    print(f"  Our R^2 (closed-form): {our_r2_ridge:.6f}")
    print(f"  Sklearn R^2:           {sk_r2_ridge:.6f}")

    # ---- Lasso comparison ----
    # sklearn Lasso minimizes: (1/(2n)) * ||y - Xw||^2 + alpha * ||w||_1
    # Our Lasso:               (1/(2n)) * ||y - Xw||^2 + lambda * ||w||_1
    # So alpha_sklearn = lambda_ours (same convention!)
    alpha_lasso = 0.5

    our_lasso = LassoRegression(
        lambda_=alpha_lasso, learning_rate=0.005,
        n_iterations=20000, tolerance=1e-10
    )
    our_lasso.fit(X_train, y_train)

    sk_lasso = Lasso(alpha=alpha_lasso, fit_intercept=True, max_iter=20000, tol=1e-10)
    sk_lasso.fit(X_train, y_train)

    our_w_l, our_b_l = our_lasso.get_weights()
    sk_w_l, sk_b_l = sk_lasso.coef_, sk_lasso.intercept_

    our_r2_lasso = our_lasso.score(X_test, y_test)
    sk_r2_lasso = sk_lasso.score(X_test, y_test)

    our_sparsity = np.sum(np.abs(our_w_l) < 0.1)
    sk_sparsity = np.sum(np.abs(sk_w_l) < 0.1)

    print(f"\n  --- Lasso (alpha={alpha_lasso}) ---")
    print(f"  Our R^2:    {our_r2_lasso:.6f}")
    print(f"  Sklearn R^2: {sk_r2_lasso:.6f}")
    print(f"  Our sparse weights (|w|<0.1):    {our_sparsity}/10")
    print(f"  Sklearn sparse weights (|w|<0.1): {sk_sparsity}/10")
    print(f"  Note: Lasso uses subgradient descent (ours) vs coordinate descent (sklearn),")
    print(f"  so weights may differ but both achieve sparsity.")

    # ---- Visualizations ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    feature_idx = np.arange(10)

    axes[0].bar(feature_idx - 0.15, our_w, 0.3, label="Ours (closed-form)",
                color="#3498db", alpha=0.8)
    axes[0].bar(feature_idx + 0.15, sk_w, 0.3, label="Sklearn Ridge",
                color="#e74c3c", alpha=0.8)
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Weight Value")
    axes[0].set_title(f"Ridge Weights (our lambda={alpha_ridge}, "
                      f"sklearn alpha={sklearn_alpha_ridge})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_xticks(feature_idx)

    axes[1].bar(feature_idx - 0.15, our_w_l, 0.3, label="Ours (subgradient)",
                color="#3498db", alpha=0.8)
    axes[1].bar(feature_idx + 0.15, sk_w_l, 0.3, label="Sklearn Lasso",
                color="#e74c3c", alpha=0.8)
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("Weight Value")
    axes[1].set_title(f"Lasso Weights (alpha={alpha_lasso})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_xticks(feature_idx)

    fig.suptitle("Sklearn Comparison: Our Implementation vs Sklearn", fontsize=14)
    save_fig(fig, "05_sklearn_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(our_w, sk_w, color="#3498db", s=60, label="Ridge weights", zorder=5)
    lims = [min(our_w.min(), sk_w.min()) - 1, max(our_w.max(), sk_w.max()) + 1]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect agreement")
    ax.set_xlabel("Our Weights (closed-form)")
    ax.set_ylabel("Sklearn Weights")
    ax.set_title(f"Ridge Weight Agreement (max diff={w_diff_ridge:.2e})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    save_fig(fig, "05b_ridge_weight_agreement.png")

    return {
        "ridge_w_diff": w_diff_ridge,
        "ridge_b_diff": b_diff_ridge,
        "our_r2_ridge": our_r2_ridge,
        "sk_r2_ridge": sk_r2_ridge,
        "our_r2_lasso": our_r2_lasso,
        "sk_r2_lasso": sk_r2_lasso,
    }


# =========================================================================
# Example 6: Ridge closed-form vs gradient descent
# =========================================================================
def example_6_closed_form_vs_gd():
    print()
    print("=" * 60)
    print("Example 6: Ridge Closed-Form vs Gradient Descent")
    print("=" * 60)

    np.random.seed(SEED)
    X, y = make_regression(
        n_samples=150, n_features=8, n_informative=8,
        noise=5.0, random_state=SEED
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    lambdas = [0.01, 0.1, 1.0, 5.0]
    results = []

    for lam in lambdas:
        ridge_cf = RidgeRegression(lambda_=lam)
        ridge_cf.fit_closed_form(X, y)
        w_cf, b_cf = ridge_cf.get_weights()

        ridge_gd = RidgeRegression(
            lambda_=lam, learning_rate=0.01,
            n_iterations=10000, tolerance=1e-10
        )
        ridge_gd.fit(X, y)
        w_gd, b_gd = ridge_gd.get_weights()

        max_w_diff = np.max(np.abs(w_cf - w_gd))
        b_diff = abs(b_cf - b_gd)
        r2_cf = ridge_cf.score(X, y)
        r2_gd = ridge_gd.score(X, y)

        results.append({
            "lambda": lam,
            "w_cf": w_cf, "w_gd": w_gd,
            "max_w_diff": max_w_diff, "b_diff": b_diff,
            "r2_cf": r2_cf, "r2_gd": r2_gd,
            "gd_iterations": len(ridge_gd.history),
        })

        print(f"  lambda={lam:.2f}: max |w_cf - w_gd| = {max_w_diff:.2e}, "
              f"|b_cf - b_gd| = {b_diff:.2e}, "
              f"R2_cf={r2_cf:.6f}, R2_gd={r2_gd:.6f}, "
              f"GD iters={len(ridge_gd.history)}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        feat_idx = np.arange(len(res["w_cf"]))
        ax.bar(feat_idx - 0.15, res["w_cf"], 0.3, label="Closed-form",
               color="#3498db", alpha=0.8)
        ax.bar(feat_idx + 0.15, res["w_gd"], 0.3, label="Gradient Descent",
               color="#e74c3c", alpha=0.8)
        ax.set_title(f"lambda={res['lambda']:.2f}, "
                     f"max diff={res['max_w_diff']:.2e}")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Weight Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(feat_idx)

    fig.suptitle("Ridge: Closed-Form vs Gradient Descent Weights", fontsize=14)
    save_fig(fig, "06_closed_form_vs_gd.png")

    return results


# =========================================================================
# Example 7: Weight decay vs L2 equivalence for SGD
# =========================================================================
def example_7_weight_decay_equivalence():
    print()
    print("=" * 60)
    print("Example 7: Weight Decay vs L2 Equivalence (SGD)")
    print("=" * 60)

    np.random.seed(SEED)
    n_samples, n_features = 100, 5
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=5, noise=3.0, random_state=SEED
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    lr = 0.01
    lambda_ = 0.1
    n_iters = 500

    w_l2 = np.zeros(n_features)
    b_l2 = 0.0
    w_wd = np.zeros(n_features)
    b_wd = 0.0

    l2_trajectory = [w_l2.copy()]
    wd_trajectory = [w_wd.copy()]
    diff_trajectory = [0.0]

    for _ in range(n_iters):
        # L2 regularization path
        y_pred_l2 = X @ w_l2 + b_l2
        error_l2 = y_pred_l2 - y
        dw_data_l2 = X.T @ error_l2 / n_samples
        db_l2 = np.mean(error_l2)

        dw_total_l2 = dw_data_l2 + l2_gradient(w_l2, lambda_)
        w_l2 = w_l2 - lr * dw_total_l2
        b_l2 = b_l2 - lr * db_l2

        # Weight decay path
        y_pred_wd = X @ w_wd + b_wd
        error_wd = y_pred_wd - y
        dw_data_wd = X.T @ error_wd / n_samples
        db_wd = np.mean(error_wd)

        w_wd = sgd_with_weight_decay(w_wd, dw_data_wd, lr, lambda_)
        b_wd = b_wd - lr * db_wd

        l2_trajectory.append(w_l2.copy())
        wd_trajectory.append(w_wd.copy())
        diff_trajectory.append(np.max(np.abs(w_l2 - w_wd)))

    l2_trajectory = np.array(l2_trajectory)
    wd_trajectory = np.array(wd_trajectory)

    final_diff = np.max(np.abs(w_l2 - w_wd))
    print(f"  lr={lr}, lambda={lambda_}, n_iters={n_iters}")
    print(f"  Final max |w_l2 - w_wd|: {final_diff:.2e}")
    print(f"  L2 final weights:          {w_l2}")
    print(f"  Weight decay final weights: {w_wd}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6"]
    for i in range(n_features):
        axes[0].plot(l2_trajectory[:, i], color=colors[i], linewidth=1.5,
                     label=f"w[{i}] L2", alpha=0.8)
        axes[0].plot(wd_trajectory[:, i], "--", color=colors[i], linewidth=1.5,
                     label=f"w[{i}] WD", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Weight Value")
    axes[0].set_title("Weight Trajectories: L2 (solid) vs WD (dashed)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(diff_trajectory, color="#e74c3c", linewidth=2)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Max |w_L2 - w_WD|")
    axes[1].set_title("Difference Between L2 and Weight Decay")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    final_idx = np.arange(n_features)
    axes[2].bar(final_idx - 0.15, w_l2, 0.3, label="L2 Reg",
                color="#3498db", alpha=0.8)
    axes[2].bar(final_idx + 0.15, w_wd, 0.3, label="Weight Decay",
                color="#e74c3c", alpha=0.8)
    axes[2].set_xlabel("Feature Index")
    axes[2].set_ylabel("Final Weight Value")
    axes[2].set_title(f"Final Weights (max diff={final_diff:.2e})")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].set_xticks(final_idx)

    fig.suptitle("Weight Decay = L2 Regularization for Vanilla SGD", fontsize=14)
    save_fig(fig, "07_weight_decay_equivalence.png")

    return {"final_max_diff": final_diff}


# =========================================================================
# PDF Report Generation
# =========================================================================
def generate_pdf_report(results):
    pdf_path = Path(__file__).parent / "report.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.65, "Regularization", fontsize=36, ha="center",
                 fontweight="bold", family="serif")
        fig.text(0.5, 0.55, "L1 (Lasso), L2 (Ridge), Elastic Net & Weight Decay",
                 fontsize=18, ha="center", family="serif", color="gray")
        fig.text(0.5, 0.40, "Comprehensive Demo with Sklearn Comparison",
                 fontsize=14, ha="center", family="serif")
        fig.text(0.5, 0.30, f"Seed: {SEED}", fontsize=12, ha="center",
                 family="monospace", color="#555555")
        fig.text(0.5, 0.15,
                 "From-scratch NumPy implementation\n"
                 "Phase 1: Foundations",
                 fontsize=11, ha="center", family="serif", color="#777777")
        pdf.savefig(fig)
        plt.close(fig)

        # Summary page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.92, "Summary of Results", fontsize=24, ha="center",
                 fontweight="bold", family="serif")

        summary_lines = [
            "Example 1: Overfitting Demonstration",
            f"  - No regularization MSE: {results['ex1']['mse_noreg']:.4f}",
            f"  - Ridge (L2=0.1) MSE: {results['ex1']['mse_ridge']:.4f}",
            f"  - Lasso (L1=0.05) MSE: {results['ex1']['mse_lasso']:.4f}",
            "",
            "Example 2: L1 vs L2 Weight Sparsity",
            f"  - Ridge near-zero weights: {results['ex2']['ridge_near_zero']}/50",
            f"  - Lasso near-zero weights: {results['ex2']['lasso_near_zero']}/50",
            "",
            "Example 3: Lambda Sweep",
            f"  - Best Ridge lambda: {results['ex3']['best_ridge_lambda']:.4f}"
            f" (val MSE: {results['ex3']['best_ridge_val_mse']:.2f})",
            f"  - Best Lasso lambda: {results['ex3']['best_lasso_lambda']:.4f}"
            f" (val MSE: {results['ex3']['best_lasso_val_mse']:.2f})",
            "",
            "Example 5: Sklearn Comparison (Ridge)",
            f"  - Max weight diff (closed-form): {results['ex5']['ridge_w_diff']:.2e}",
            f"  - Our R2: {results['ex5']['our_r2_ridge']:.6f}",
            f"  - Sklearn R2: {results['ex5']['sk_r2_ridge']:.6f}",
            "",
            "Example 7: Weight Decay vs L2 Equivalence",
            f"  - Final max diff: {results['ex7']['final_max_diff']:.2e}",
            "  - Confirms identical trajectories for vanilla SGD",
        ]

        y_pos = 0.85
        for line in summary_lines:
            fontsize = 11 if line.startswith("Example") else 10
            weight = "bold" if line.startswith("Example") else "normal"
            fig.text(0.1, y_pos, line, fontsize=fontsize, fontweight=weight,
                     family="monospace", verticalalignment="top")
            y_pos -= 0.035

        pdf.savefig(fig)
        plt.close(fig)

        # One page per visualization
        for name, fig_obj in ALL_FIGURES:
            fig_copy = plt.figure(figsize=(11, 8.5))

            img = plt.imread(str(VIZ_DIR / name))
            ax = fig_copy.add_axes([0.02, 0.05, 0.96, 0.88])
            ax.imshow(img)
            ax.axis("off")

            title = name.replace(".png", "").replace("_", " ").title()
            fig_copy.text(0.5, 0.97, title, fontsize=14, ha="center",
                         fontweight="bold", family="serif")
            pdf.savefig(fig_copy)
            plt.close(fig_copy)

    print(f"\n  PDF report saved to: {pdf_path}")


# =========================================================================
# Main
# =========================================================================
def main():
    print()
    print("*" * 60)
    print("  REGULARIZATION DEMO")
    print("  L1 (Lasso), L2 (Ridge), Elastic Net & Weight Decay")
    print(f"  Seed: {SEED}")
    print("*" * 60)
    print()

    results = {}

    results["ex1"] = example_1_overfitting()
    results["ex2"] = example_2_l1_vs_l2_weights()
    results["ex3"] = example_3_lambda_sweep()
    results["ex4"] = example_4_sparsity_vs_lambda()
    results["ex5"] = example_5_sklearn_comparison()
    results["ex6"] = example_6_closed_form_vs_gd()
    results["ex7"] = example_7_weight_decay_equivalence()

    print()
    print("=" * 60)
    print("Generating PDF Report")
    print("=" * 60)
    generate_pdf_report(results)

    print()
    print("=" * 60)
    print("All visualizations saved to viz/")
    print("=" * 60)
    for name, _ in ALL_FIGURES:
        print(f"  {name}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
