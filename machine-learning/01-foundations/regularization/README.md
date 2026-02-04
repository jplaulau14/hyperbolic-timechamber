# Regularization

**Phase 1 · Topic 4** — Preventing overfitting through weight constraints, and the foundation for weight decay in AdamW.

## What it is

Regularization adds a penalty term to the loss function that discourages the model from learning overly complex solutions. Without regularization, models with sufficient capacity will memorize training data, including noise, leading to poor generalization on unseen data. This is overfitting.

The core idea is simple: add a term to the loss that penalizes large weights. This forces the optimizer to find solutions that not only minimize the data loss but also keep weights small. Smaller weights mean smoother decision boundaries and less sensitivity to individual training examples.

There are three main regularization techniques you need to understand:

- **L2 (Ridge):** Penalizes the squared magnitude of weights. Pushes weights toward zero but rarely makes them exactly zero. Results in smooth, distributed weight vectors.
- **L1 (Lasso):** Penalizes the absolute value of weights. Pushes weights all the way to zero, inducing sparsity. Useful for feature selection.
- **Elastic Net:** Combines L1 and L2, getting the benefits of both: sparsity from L1 and stability from L2.

## The math

### L2 Regularization (Ridge)

The L2-regularized loss adds the squared L2 norm of the weight vector:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \|w\|_2^2 = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \sum_i w_i^2$$

**Gradient contribution from the penalty:**

$$\frac{\partial}{\partial w} \left[ \frac{\lambda}{2} \sum_i w_i^2 \right] = \lambda w$$

**Full gradient update:**

$$w \leftarrow w - \alpha \left( \frac{\partial \mathcal{L}_{\text{data}}}{\partial w} + \lambda w \right)$$

The $\lambda w$ term pulls each weight toward zero proportionally to its current value. Large weights get penalized more, small weights less. This is multiplicative shrinkage.

**Shape definitions:**
- $w \in \mathbb{R}^{d}$ or $w \in \mathbb{R}^{d \times k}$ weight matrix
- $\lambda \in \mathbb{R}$ scalar regularization strength (typically $10^{-4}$ to $0.1$)
- $\alpha \in \mathbb{R}$ scalar learning rate

### L1 Regularization (Lasso)

The L1-regularized loss adds the L1 norm (sum of absolute values):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \|w\|_1 = \mathcal{L}_{\text{data}} + \lambda \sum_i |w_i|$$

**Gradient contribution from the penalty:**

$$\frac{\partial}{\partial w} \left[ \lambda \sum_i |w_i| \right] = \lambda \cdot \text{sign}(w)$$

Where $\text{sign}(w)$ is:
- $+1$ if $w_i > 0$
- $-1$ if $w_i < 0$
- $0$ if $w_i = 0$ (subgradient at the kink)

**Full gradient update:**

$$w \leftarrow w - \alpha \left( \frac{\partial \mathcal{L}_{\text{data}}}{\partial w} + \lambda \cdot \text{sign}(w) \right)$$

The key difference: L1 applies a constant push toward zero regardless of weight magnitude. A weight at 0.01 gets the same push as a weight at 100. This drives small weights all the way to exactly zero, creating sparse weight vectors.

**Technical note:** The L1 norm is not differentiable at zero. In practice, we use the subgradient (returning 0 when $w_i = 0$) or proximal operators for more sophisticated implementations.

### Elastic Net (L1 + L2 Combined)

Elastic Net combines both penalties with a mixing parameter:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \left( \rho \|w\|_1 + \frac{1 - \rho}{2} \|w\|_2^2 \right)$$

Where:
- $\rho$ (or `l1_ratio`): mixing parameter in $[0, 1]$
  - $\rho = 1$: pure L1 (Lasso)
  - $\rho = 0$: pure L2 (Ridge)
  - $\rho = 0.5$: equal mix

**Gradient contribution:**

$$\frac{\partial}{\partial w} [\text{penalty}] = \lambda \left( \rho \cdot \text{sign}(w) + (1 - \rho) w \right)$$

Elastic Net gets you sparsity from L1 while the L2 term stabilizes the solution when features are correlated.

### Weight Decay vs. L2 Regularization

For vanilla SGD, L2 regularization and weight decay are mathematically equivalent:

$$w \leftarrow w - \alpha \left( \frac{\partial \mathcal{L}_{\text{data}}}{\partial w} + \lambda w \right) = w - \alpha \frac{\partial \mathcal{L}_{\text{data}}}{\partial w} - \alpha \lambda w$$

**Weight decay:**

$$w \leftarrow w - \alpha \frac{\partial \mathcal{L}_{\text{data}}}{\partial w} - \alpha \lambda_{\text{wd}} w$$

Set $\lambda = \lambda_{\text{wd}}$ and they're identical.

**But for Adam/AdamW, they differ.** In Adam, gradients are normalized by the second moment:

```python
# L2 in Adam (incorrect for regularization)
g = dL_data_dw + lambda_ * w
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g**2
w = w - alpha * m_hat / (np.sqrt(v_hat) + eps)
```

The weight penalty $\lambda w$ gets scaled by the adaptive learning rate, which is wrong. Large weights have large gradients, so they have large $v$, so the effective penalty is *reduced*.

```python
# Decoupled weight decay in AdamW (correct)
g = dL_data_dw  # no penalty in gradient
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g**2
w = w - alpha * (m_hat / (np.sqrt(v_hat) + eps) + lambda_ * w)
```

Weight decay is applied directly to the weights, bypassing the adaptive scaling. This is why AdamW works better than Adam with L2 regularization for training large models.

## Why it matters for inference

### Weight Decay in AdamW

Every modern LLM (GPT, Llama, Mistral, etc.) is trained with AdamW. The "W" stands for decoupled weight decay. Typical values are $\lambda = 0.01$ to $0.1$. Understanding why weight decay is decoupled from the gradient computation helps you reason about training dynamics and hyperparameter tuning.

### L1 Regularization and Sparsity

L1 regularization produces sparse weight matrices where many weights are exactly zero. This connects directly to:

- **Pruning:** Remove zero or near-zero weights to reduce model size
- **Sparse inference kernels:** Skip computation for zero weights (e.g., sparse matrix multiply)
- **Feature selection:** In linear models, non-zero weights indicate which features matter

Modern pruning techniques often start from L1-regularized training or use magnitude-based pruning (which L1 naturally encourages).

### Effect on Model Capacity

Regularization constrains the model's hypothesis space. Stronger regularization (higher $\lambda$) means the model can fit less complex functions. This is the bias-variance tradeoff:

- High regularization: higher bias, lower variance (underfitting risk)
- Low regularization: lower bias, higher variance (overfitting risk)

For LLMs, regularization helps prevent memorization of training data, which is critical for generalization.

### Connection to Bayesian Interpretation

L2 regularization is equivalent to placing a Gaussian prior on weights (mean 0, variance proportional to $1/\lambda$). L1 regularization corresponds to a Laplace prior. This Bayesian view helps explain why regularization works: we're encoding prior knowledge that weights should be small.

## What to implement

### Core Functions

- [ ] `l2_penalty(w, lambda_)`: compute $\frac{\lambda}{2} \sum_i w_i^2$
- [ ] `l2_gradient(w, lambda_)`: compute $\lambda w$
- [ ] `l1_penalty(w, lambda_)`: compute $\lambda \sum_i |w_i|$
- [ ] `l1_gradient(w, lambda_)`: compute $\lambda \cdot \text{sign}(w)$ with proper handling at zero
- [ ] `elastic_net_penalty(w, lambda_, l1_ratio)`: combined penalty
- [ ] `elastic_net_gradient(w, lambda_, l1_ratio)`: combined gradient

### Regularized Regression Classes

- [ ] `RidgeRegression` class:
  - `__init__(self, lambda_: float = 1.0, learning_rate: float = 0.01, n_iterations: int = 1000)`
  - `fit(X, y)`: train with L2-regularized gradient descent
  - `predict(X)`: return predictions
  - `get_weights()`: return learned weights for inspection

- [ ] `LassoRegression` class:
  - Same interface as Ridge
  - Should produce sparse weights for appropriate $\lambda$

- [ ] `ElasticNetRegression` class:
  - Additional parameter `l1_ratio: float = 0.5`
  - Same interface otherwise

### Weight Decay Demonstration

- [ ] `sgd_with_weight_decay(w, gradient, lr, lambda_)`: single update step
- [ ] Demonstrate equivalence of L2 regularization and weight decay for SGD
- [ ] Show how L2-in-Adam differs from weight-decay-in-AdamW (qualitative comparison)

### Analytical Solutions (Optional but Educational)

- [ ] Ridge closed-form: $w = (X^\top X + \lambda I)^{-1} X^\top y$
  - Compare to gradient descent solution
  - Show how $\lambda$ affects condition number

## Test cases to cover

### Correctness Tests

- [ ] **Penalty value computation:**
  - $w = [1, 2, 3]$, $\lambda = 0.1$: L2 penalty = $\frac{0.1}{2} \cdot (1 + 4 + 9) = 0.7$
  - $w = [1, -2, 3]$, $\lambda = 0.1$: L1 penalty = $0.1 \cdot (1 + 2 + 3) = 0.6$

- [ ] **Gradient computation:**
  - $w = [1, 2, 3]$, $\lambda = 0.1$: L2 gradient = $[0.1, 0.2, 0.3]$
  - $w = [1, -2, 0]$, $\lambda = 0.1$: L1 gradient = $[0.1, -0.1, 0.0]$

- [ ] **Elastic Net interpolation:**
  - `l1_ratio = 1.0` should equal pure L1
  - `l1_ratio = 0.0` should equal pure L2

### Sparsity Tests (L1)

- [ ] **L1 induces sparsity:** Train Lasso on data with redundant features. Verify that some weights become exactly (or very close to) zero.

- [ ] **Sparsity increases with lambda:** Higher $\lambda$ should result in more zero weights.

- [ ] **Compare weight distributions:**
  - L1: many weights near zero, some large (sparse)
  - L2: weights distributed more evenly, none exactly zero

### Shrinkage Tests (L2)

- [ ] **L2 shrinks weights:** Compare weight magnitudes with and without regularization. Regularized weights should have smaller L2 norm.

- [ ] **No sparsity:** L2 should not produce exactly zero weights (only very small ones).

- [ ] **Analytical vs gradient descent:** Ridge closed-form solution should match gradient descent solution (within tolerance).

### Hyperparameter Sensitivity

- [ ] **Lambda sweep:** Train models with $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$.
  - Plot training loss vs validation loss
  - Find the optimal $\lambda$ that minimizes validation loss (bias-variance sweet spot)

- [ ] **Underfitting detection:** Very high $\lambda$ should cause high training error (model is too constrained).

- [ ] **Overfitting detection:** Very low $\lambda$ should cause low training error but high validation error.

### Edge Cases

- [ ] **Zero weights:** L1 gradient should return 0 for weights that are exactly 0.

- [ ] **Large weights:** Regularization should aggressively penalize outlier weights.

- [ ] **Single feature:** Regularization should still work with `n_features = 1`.

- [ ] **Zero lambda:** Should reduce to unregularized regression.

### Weight Decay Equivalence

- [ ] **SGD equivalence:** For SGD, verify that L2 regularization and weight decay produce identical weight trajectories (within numerical precision).

- [ ] **Gradient computation:** Verify that $\frac{\partial \mathcal{L}_{\text{total}}}{\partial w} = \frac{\partial \mathcal{L}_{\text{data}}}{\partial w} + \lambda w$ for L2-regularized loss.

### Numerical Stability

- [ ] **Condition number:** For Ridge, verify that $(X^\top X + \lambda I)$ is better conditioned than $(X^\top X)$ alone (relevant for analytical solution).

- [ ] **Gradient magnitude:** Regularization gradients should be bounded and not cause gradient explosion.
