# Linear Regression

**Phase 1 · Topic 1** — Your first training loop: forward, loss, backward, update.

## What it is

Linear regression models the relationship between input features and a continuous output as a linear function. Given input features $X$ and target values $y$, it finds weights $w$ and bias $b$ such that the predicted output $\hat{y} = Xw + b$ is as close as possible to $y$.

This is the simplest supervised learning algorithm, but don't underestimate its importance. The training pattern you learn here — forward pass to compute predictions, loss function to measure error, backward pass to compute gradients, update step to adjust parameters — is identical in transformers with billions of parameters. The only differences are scale and the depth of the computation graph.

Linear regression also has a closed-form solution (the normal equation), which makes it unique among ML algorithms. You'll implement both gradient descent and the analytical solution, giving you a reference point for understanding when iterative optimization is necessary (spoiler: almost always in deep learning).

## The math

### Model

$$\hat{y} = Xw + b$$

Where:
- $X \in \mathbb{R}^{n \times d}$ is the input matrix (n_samples, n_features)
- $w \in \mathbb{R}^{d}$ is the weight vector
- $b \in \mathbb{R}$ is the bias scalar
- $\hat{y} \in \mathbb{R}^{n}$ is the prediction vector

### Loss function (Mean Squared Error)

$$\mathcal{L} = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

The $\frac{1}{2}$ factor is a convenience — it cancels with the exponent during differentiation. Some implementations use $\frac{1}{n}$ or $\frac{1}{2n}$; just be consistent.

In matrix form:

$$\mathcal{L} = \frac{1}{2n} \| Xw + b - y \|_2^2$$

### Gradients (for gradient descent)

Taking the derivative of $\mathcal{L}$ with respect to $w$ and $b$:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^\top (\hat{y} - y)$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

**Derivation for $\frac{\partial \mathcal{L}}{\partial w}$:**

1. Let $e = \hat{y} - y$ (the error vector)
2. $\mathcal{L} = \frac{1}{2n} e^\top e$
3. $\frac{\partial \mathcal{L}}{\partial e} = \frac{1}{n} e$
4. $\frac{\partial e}{\partial w} = X$ (since $\hat{y} = Xw + b$)
5. By chain rule: $\frac{\partial \mathcal{L}}{\partial w} = X^\top \frac{\partial \mathcal{L}}{\partial e} = \frac{1}{n} X^\top (\hat{y} - y)$

**Shape check:**
- $X^\top$ is $(d, n)$
- $(\hat{y} - y)$ is $(n,)$
- Result $\frac{\partial \mathcal{L}}{\partial w}$ is $(d,)$ — same shape as $w$ ✓

### Gradient descent update

$$w \leftarrow w - \alpha \frac{\partial \mathcal{L}}{\partial w}$$

$$b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}$$

Where $\alpha$ is the learning rate. Repeat until convergence (loss stops decreasing significantly) or max iterations reached.

### Analytical solution (Normal Equation)

For linear regression specifically, we can solve for the optimal weights directly:

$$w^* = (X^\top X)^{-1} X^\top y$$

**Derivation:**

1. Set $\frac{\partial \mathcal{L}}{\partial w} = 0$ (necessary condition for minimum)
2. $X^\top (Xw - y) = 0$
3. $X^\top X w = X^\top y$
4. $w = (X^\top X)^{-1} X^\top y$

**With bias term:** Add a column of ones to $X$ (design matrix augmentation):

```python
X_aug = np.column_stack([np.ones(n_samples), X])
w_aug = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
b = w_aug[0]
w = w_aug[1:]
```

**When to use the normal equation:**
- Small datasets ($d < 10{,}000$)
- When you need the exact solution
- When you don't want to tune learning rate

**When NOT to use:**
- Large feature dimensions (matrix inversion is $O(d^3)$)
- Singular or near-singular $X^\top X$ (use gradient descent or regularization)
- When you want to add regularization (requires modified equation)

## Why it matters for inference

Linear regression introduces the training loop that scales to transformers:

```python
for epoch in range(n_epochs):
    # Forward pass
    y_hat = X @ w + b

    # Loss computation
    loss = mse(y_hat, y)

    # Backward pass (compute gradients)
    dw = X.T @ (y_hat - y) / n
    db = (y_hat - y).mean()

    # Update parameters
    w -= lr * dw
    b -= lr * db
```

This exact pattern appears in every neural network. In transformers:
- Forward pass is hundreds of matrix multiplications and nonlinearities
- Loss is cross-entropy over vocabulary
- Backward pass uses automatic differentiation (but the math is identical)
- Update uses AdamW instead of vanilla gradient descent

**Compute characteristics:**
- Forward pass: matrix-vector multiplication $Xw$ is $O(n \cdot d)$
- Gradient computation: $X^\top e$ is the same complexity
- In neural networks, these become matrix-matrix multiplications (GEMMs), which dominate compute time

**Memory characteristics:**
- Must store $X$ in memory during training (for gradient computation)
- Gradient descent is memory-efficient: $O(d)$ for weights
- Normal equation requires $O(d^2)$ for $X^\top X$

Understanding these tradeoffs prepares you for analyzing attention mechanisms, where the choice between memory and compute is critical.

## What to implement

### Core class

- [ ] `LinearRegression` class with `fit(X, y)` and `predict(X)` methods
- [ ] Constructor accepts `method='gradient_descent'` or `method='normal_equation'`
- [ ] Constructor accepts `learning_rate`, `n_iterations`, `tolerance` for gradient descent

### Training methods

- [ ] `_fit_gradient_descent(X, y)` — iterative optimization
- [ ] `_fit_normal_equation(X, y)` — closed-form solution using `np.linalg.solve` (not `np.linalg.inv`)
- [ ] Store training history: loss at each iteration

### Loss and gradients

- [ ] `mse_loss(y_true, y_pred)` — compute mean squared error
- [ ] `_compute_gradients(X, y, y_pred)` — return (dw, db)

### Utilities

- [ ] `score(X, y)` — return R² coefficient of determination
- [ ] Handle bias term properly (either augment X or track separately)
- [ ] Support both 1D and 2D input arrays

### Numerical stability

- [ ] Use `np.linalg.solve` instead of explicit matrix inversion
- [ ] Check for convergence: stop if $|\mathcal{L}_{t-1} - \mathcal{L}_t| < \epsilon$
- [ ] Initialize weights to small random values or zeros

## Test cases to cover

### Basic correctness

- [ ] **Perfect linear fit:** Generate $y = 2x + 3$, verify weights recover $[2]$ and bias $3$
- [ ] **Multiple features:** Generate $y = x_1 + 2x_2 + 3x_3 + 4$, verify all coefficients
- [ ] **Gradient descent vs normal equation:** Both methods produce same weights (within tolerance)
- [ ] **Prediction shape:** `predict(X)` returns correct shape for both 1D and 2D inputs

### Edge cases

- [ ] **Single sample:** Fit with $n=1$ (should work but with warning or special handling)
- [ ] **Single feature:** 1D feature vector input
- [ ] **Zero variance feature:** Column of identical values (should handle gracefully)
- [ ] **Large feature values:** $X$ with values in range $[10^6, 10^7]$ (test numerical stability)
- [ ] **Negative targets:** $y$ with negative values

### Convergence and training

- [ ] **Convergence check:** Loss decreases monotonically with appropriate learning rate
- [ ] **Learning rate too high:** Loss diverges (verify detection/handling)
- [ ] **Early stopping:** Converges before max iterations when tolerance is set
- [ ] **Training history:** Loss history has correct length and decreasing trend

### Numerical precision

- [ ] **Near-singular matrix:** $X^\top X$ is poorly conditioned (normal equation should warn or handle)
- [ ] **Float32 vs Float64:** Results are consistent across dtypes
- [ ] **Noise tolerance:** With Gaussian noise added to $y$, R² is reasonable (e.g., > 0.9 for low noise)

### Against known values

- [ ] **Scikit-learn comparison (for testing only):** Your implementation matches `sklearn.linear_model.LinearRegression` on the same data
- [ ] **Hand-computed example:** For a 2×2 system, manually compute expected weights and verify

### R² metric

- [ ] **Perfect fit:** $R^2 = 1.0$ when predictions exactly match targets
- [ ] **Mean prediction:** $R^2 = 0.0$ when model predicts mean of $y$ for all inputs
- [ ] **Worse than mean:** $R^2 < 0$ is possible (verify formula handles this)

## Implementation notes

### Weight initialization

For gradient descent, initialize weights to zeros or small random values:

```python
self.w = np.zeros(n_features)
self.b = 0.0
```

### Using np.linalg.solve

Never compute `np.linalg.inv(X.T @ X)` explicitly. Instead:

```python
# Bad (numerically unstable)
w = np.linalg.inv(X.T @ X) @ X.T @ y

# Good (numerically stable)
w = np.linalg.solve(X.T @ X, X.T @ y)
```

### Vectorized gradient computation

Avoid loops over samples. The gradient computation should be fully vectorized:

```python
# Compute all predictions at once
y_pred = X @ self.w + self.b

# Compute gradients (no loops)
error = y_pred - y
dw = X.T @ error / n_samples
db = np.mean(error)
```

### R² formula

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

Where:
- $SS_{res}$ is the residual sum of squares
- $SS_{tot}$ is the total sum of squares
- $\bar{y}$ is the mean of $y$
