# Regularization (L1, L2, Elastic Net) -- Deep Dive

## The Intuition

### What Problem Are We Solving?

Imagine you have a dataset of house prices with 20 features: square footage, number of bedrooms, distance to school, whether the house number is even, the day of the week the listing was posted, and so on. Some of these features genuinely predict price. Others are noise -- pure coincidence in your training data.

A model with enough capacity will learn to exploit *all* of these features, including the noise. It will memorize that houses listed on Tuesdays in your training set happened to cost more, even though that is meaningless. On training data, this model looks brilliant. On new data, it falls apart.

This is **overfitting**: the model has learned the noise in the training data instead of the underlying signal. The gap between training performance and test performance is the hallmark of overfitting.

Regularization is the antidote. It adds a cost for complexity, forcing the model to justify every weight it uses. A feature must provide enough predictive value to overcome its regularization penalty. Noise features cannot clear this bar, so their weights get pushed toward (or exactly to) zero.

### The Key Insight

The core insight is beautifully simple: **smaller weights mean simpler models**. A weight vector like $[3.0, -1.0, 0.5, 0.0, 0.0]$ is simpler than $[3.2, -0.8, 0.6, 0.3, -0.2]$, even though both might fit the training data almost equally well. By penalizing the *size* of the weight vector, we nudge the optimizer toward solutions that are simple enough to generalize.

The surprising part is *how* you measure "size" completely changes the behavior:
- Measure with the **sum of squares** (L2) and you get smooth shrinkage -- all weights get smaller, none hit zero.
- Measure with the **sum of absolute values** (L1) and you get sparsity -- some weights are driven to *exactly* zero, performing automatic feature selection.

### Real-World Analogy

Think of packing for a trip with a weight limit on your luggage. Every item you bring incurs a cost (weight in your bag). Essential items -- passport, phone, medication -- are worth their weight. But that "just in case" third pair of shoes? The weight penalty makes you reconsider. With a strict weight limit (strong regularization), you bring only the essentials. With a generous limit (weak regularization), you might pack some extras.

L2 regularization is like a weight limit that makes heavier items proportionally more expensive -- you bring everything but in smaller quantities. L1 regularization is like a flat fee per item regardless of weight -- you either bring the item or you leave it behind entirely, which is why some items (features) get cut to exactly zero.

---

## The Math, Step by Step

### Building Up to the Formulas

**Step 1 -- Unregularized Loss.** Start with ordinary linear regression. We minimize the mean squared error:

$$
\mathcal{L}_{\text{data}} = \frac{1}{2n} \sum_{i} (\hat{y}_i - y_i)^2
$$

This loss only cares about fitting the data. It has no opinion about how large the weights are. With enough features relative to samples, the model can achieve near-zero training loss by memorizing every data point.

**Step 2 -- Add a Penalty.** We modify the objective to include a regularization term:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \text{Penalty}(w)
$$

Now the optimizer must balance two competing goals: fit the data well *and* keep the penalty small. The regularization strength $\lambda$ controls this tradeoff. Larger $\lambda$ means "I care more about small weights than fitting the data."

**Step 3 -- Choose Your Penalty.** The choice of penalty function gives fundamentally different behavior.

### The Core Equations

#### L2 Regularization (Ridge)

$$
\text{Penalty:} \quad \frac{\lambda}{2} \|w\|_2^2 = \frac{\lambda}{2} \sum_{i} w_i^2
$$

$$
\text{Gradient:} \quad \lambda w
$$

Where:
- $w$: weight vector, shape $(d,)$ or weight matrix, shape $(d, k)$
- $\lambda$: regularization strength, scalar $> 0$
- The $\frac{1}{2}$ factor is a convenience so the gradient is clean ($\lambda w$, not $2\lambda w$)

The L2 gradient is **proportional to the weight itself**. A weight of $10.0$ gets a gradient push of $\lambda \cdot 10.0$, while a weight of $0.01$ gets only $\lambda \cdot 0.01$. Large weights are penalized heavily; small weights are barely touched. This is **multiplicative shrinkage** -- every weight shrinks by a fraction of its current value each step, so weights approach zero asymptotically but never reach it.

#### L1 Regularization (Lasso)

$$
\text{Penalty:} \quad \lambda \|w\|_1 = \lambda \sum_{i} |w_i|
$$

$$
\text{Gradient:} \quad \lambda \, \text{sign}(w)
$$

Where:

$$
\text{sign}(w_i) = \begin{cases} +1 & \text{if } w_i > 0 \\ -1 & \text{if } w_i < 0 \\ 0 & \text{if } w_i = 0 \quad \text{(subgradient convention)} \end{cases}
$$

The L1 gradient is a **constant push** toward zero: $+\lambda$ or $-\lambda$, regardless of how large or small the weight is. A weight of $100.0$ gets the same push as a weight of $0.001$. This means small weights can be pushed all the way to *exactly* zero, creating sparse weight vectors.

**Technical note:** The absolute value function $|w|$ has a kink at zero where it is not differentiable. We use the **subgradient** convention, choosing $\text{sign}(0) = 0$. NumPy's `np.sign` follows this convention, which is why the implementation is a single line.

#### Elastic Net (Combined)

$$
\text{Penalty:} \quad \lambda \left( \rho \|w\|_1 + \frac{1 - \rho}{2} \|w\|_2^2 \right)
$$

$$
\text{Gradient:} \quad \lambda \left( \rho \, \text{sign}(w) + (1 - \rho) \, w \right)
$$

Where:
- $\rho$ (`l1_ratio`): mixing parameter in $[0, 1]$
  - $\rho = 1.0$: pure L1 (Lasso)
  - $\rho = 0.0$: pure L2 (Ridge)
  - $\rho = 0.5$: equal mix

Elastic Net gets sparsity from the L1 component while the L2 component provides stability, especially when features are correlated.

### Full Gradient Update Rules

For all three methods, the gradient descent update follows the same pattern:

$$
w \leftarrow w - \alpha \left( \frac{\partial \mathcal{L}_{\text{data}}}{\partial w} + \text{regularization gradient} \right)
$$

Expanded for each:

$$
\text{Ridge:} \quad w \leftarrow w - \alpha \left( \frac{X^\top(Xw - y)}{n} + \lambda w \right)
$$

$$
\text{Lasso:} \quad w \leftarrow w - \alpha \left( \frac{X^\top(Xw - y)}{n} + \lambda \, \text{sign}(w) \right)
$$

$$
\text{Elastic Net:} \quad w \leftarrow w - \alpha \left( \frac{X^\top(Xw - y)}{n} + \lambda \left( \rho \, \text{sign}(w) + (1 - \rho) \, w \right) \right)
$$

The bias $b$ is **not regularized** in any of these. We only penalize feature weights, not the intercept, because the intercept just shifts the prediction and does not contribute to model complexity.

---

## The L1 vs L2 Geometry -- Why L1 Produces Sparsity

This is one of the most important insights in machine learning. Why does changing from squared weights to absolute-value weights suddenly produce exact zeros?

### The Constraint View

Penalizing weights in the loss is mathematically equivalent to constraining weights to a region. Specifically:

$$
\min_w \mathcal{L}_{\text{data}}(w) \quad \text{subject to} \quad \|w\|_p \leq C
$$

is equivalent to:

$$
\min_w \mathcal{L}_{\text{data}}(w) + \lambda \|w\|_p
$$

for some relationship between $C$ and $\lambda$. The constraint region defines the set of "allowed" weight vectors.

### L2: The Ball (Sphere)

For L2, the constraint $\|w\|_2 \leq C$ is a **ball** (circle in 2D, sphere in 3D, hypersphere in higher dimensions):

```
         L2 constraint region (2D)

              . . .
          .           .
        .               .
       .                 .
       .        O        .      O = origin
       .                 .
        .               .
          .           .
              . . .
```

The loss function has elliptical contours. The optimal constrained solution is where an elliptical contour *just touches* the ball. Because the ball is smooth and round, this tangent point can happen anywhere -- there is no reason for it to land on an axis.

### L1: The Diamond

For L1, the constraint $\|w\|_1 \leq C$ is a **diamond** (rotated square in 2D, cross-polytope in higher dimensions):

```
         L1 constraint region (2D)

              .
            / | \
          /   |   \
        /     |     \
       .------O------.       O = origin
        \     |     /
          \   |   /
            \ | /
              .
```

The diamond has **corners on the axes**. When an elliptical loss contour approaches this shape, it is far more likely to touch at a corner than along a flat edge. At a corner, one or more coordinates are exactly zero. In higher dimensions, the diamond has exponentially more corners relative to its surface area, making axis-aligned contact even more probable.

### Why This Matters in 2D

```
     w2
      |
      |       Loss contours (ellipses)
      |      .----.
      |     /      \             L2 ball
      |    |   *    |          .---.
      |     \      /         /     \
      |      '----'         |   O   |
      |                      \     /
      +--------*-------------'---'---------> w1
               ^                   tangent point is NOT on axis
               optimal w1 is non-zero

     w2
      |
      |       Loss contours (ellipses)
      |      .----.
      |     /      \          L1 diamond
      |    |   *    |         /\
      |     \      /         /  \
      |      '----'         O    .
      |                      \  /
      +--------*--------------\/-----------> w1
               ^              tangent hits the CORNER
               optimal w1 IS zero, w2 IS zero
```

The loss contour almost always first touches the L1 diamond at one of its corners. At a corner, at least one coordinate is zero. This is the geometric reason L1 produces sparsity.

### The Gradient Perspective

There is an equivalent way to see this through gradients. Consider a weight $w_i$ that is small but positive, say $w_i = 0.01$.

**L2 gradient at $w_i = 0.01$:**

$$
\lambda \cdot w_i = \lambda \cdot 0.01 = \text{very small push toward zero}
$$

The push vanishes as the weight approaches zero. The weight gets asymptotically close but never reaches it.

**L1 gradient at $w_i = 0.01$:**

$$
\lambda \cdot \text{sign}(0.01) = \lambda \cdot 1 = \text{full push toward zero}
$$

The push is the same whether the weight is $100$ or $0.01$. It keeps pushing with full force, overshooting zero and getting clipped back, effectively locking the weight at zero once the data gradient can no longer justify keeping it non-zero.

---

## Worked Example: L1 Driving Weights to Zero

Let us trace through a concrete example to see L1 sparsity in action.

### Setup

$$
\text{Data: 4 samples, 2 features}
$$

$$
X = \begin{bmatrix} 1 & 0.1 \\ 2 & 0.2 \\ 3 & 0.1 \\ 4 & 0.3 \end{bmatrix}, \quad y = \begin{bmatrix} 2.1 \\ 4.0 \\ 6.1 \\ 7.9 \end{bmatrix}
$$

True relationship: $y \approx 2 x_1$ (feature 2 is noise)

$\lambda = 0.5$, $\alpha = 0.1$ (learning rate)

Initial: $w = [0, 0]$, $b = 0$

### Iteration 1

$$
\hat{y} = X \begin{bmatrix} 0 \\ 0 \end{bmatrix} + 0 = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}
$$

$$
\text{error} = \hat{y} - y = \begin{bmatrix} -2.1 \\ -4.0 \\ -6.1 \\ -7.9 \end{bmatrix}
$$

Data gradient for $w$:

$$
\frac{\partial w}{\partial \mathcal{L}} = \frac{X^\top \cdot \text{error}}{4} = \frac{1}{4} \begin{bmatrix} 1 & 2 & 3 & 4 \\ 0.1 & 0.2 & 0.1 & 0.3 \end{bmatrix} \begin{bmatrix} -2.1 \\ -4.0 \\ -6.1 \\ -7.9 \end{bmatrix} = \begin{bmatrix} -15.0 \\ -0.9975 \end{bmatrix}
$$

L1 gradient: $\lambda \cdot \text{sign}([0, 0]) = 0.5 \cdot [0, 0] = [0, 0]$ (since $\text{sign}(0) = 0$ by subgradient convention)

Total gradient: $dw = [-15.0, -0.9975] + [0, 0] = [-15.0, -0.9975]$

Update: $w = [0, 0] - 0.1 \cdot [-15.0, -0.9975] = [1.5, 0.09975]$

Both weights move away from zero because the data gradient dominates.

### After Many Iterations (Converging)

Suppose after 500 iterations we have $w = [1.95, 0.08]$. Feature 1's weight is close to the true value of $2.0$. Feature 2's weight is small because it is noise.

Data gradient for $w_2$ (the noise feature): approximately $-0.02$ (small, because the feature barely helps)

L1 gradient for $w_2$: $\lambda \cdot \text{sign}(0.08) = 0.5 \cdot 1 = 0.5$

Total gradient for $w_2$: $-0.02 + 0.5 = 0.48$ (the L1 penalty **dominates** the data gradient)

Update for $w_2$: $0.08 - 0.1 \cdot 0.48 = 0.08 - 0.048 = 0.032$

The L1 penalty ($0.5$) overwhelms the data gradient ($-0.02$). The weight is being pushed firmly toward zero. After a few more iterations, $w_2$ reaches zero and stays there. Once at zero, $\text{sign}(0) = 0$, so the L1 penalty stops pushing, and the tiny data gradient alone is not enough to move it away.

**Contrast with L2:** At $w_2 = 0.08$, the L2 gradient would be $\lambda \cdot 0.08 = 0.5 \cdot 0.08 = 0.04$. The total gradient would be $-0.02 + 0.04 = 0.02$, a gentle push toward zero. But as the weight gets smaller, so does the L2 push. At $w_2 = 0.001$, the L2 gradient is only $0.0005$, balancing against the data gradient. The weight stabilizes at some small but non-zero value.

### Key Takeaway

L1's constant-magnitude push overwhelms the data gradient for unimportant features, driving their weights to zero. L2's proportional push weakens as weights shrink, creating a stable equilibrium at a small but non-zero value.

---

## From Math to Code

### The Data Structures

Each regularized regression model maintains:

| State | Type | Purpose |
|-------|------|---------|
| `w` | `np.ndarray`, shape `(d,)` | Feature weights |
| `b` | `float` | Bias (intercept), not regularized |
| `history` | `List[float]` | Total loss at each iteration (for diagnostics) |
| `lambda_` | `float` | Regularization strength |

The trailing underscore on `lambda_` avoids shadowing Python's `lambda` keyword.

### Implementation Walkthrough: Core Functions

```python
def l2_penalty(w: np.ndarray, lambda_: float) -> float:
    return 0.5 * lambda_ * np.sum(w ** 2)
```

This computes $\frac{\lambda}{2} \|w\|_2^2$. The `np.sum(w ** 2)` works for any shape -- 1D vectors or 2D weight matrices -- because it sums all elements. The $\frac{1}{2}$ factor is not arbitrary: it makes the gradient cleaner ($\lambda w$ instead of $2\lambda w$), the same reason we use $\frac{1}{2n}$ in MSE.

```python
def l2_gradient(w: np.ndarray, lambda_: float) -> np.ndarray:
    return lambda_ * w
```

The derivative of $\frac{\lambda}{2} w_i^2$ with respect to $w_i$ is $\lambda w_i$. Broadcasting handles all shapes.

```python
def l1_penalty(w: np.ndarray, lambda_: float) -> float:
    return lambda_ * np.sum(np.abs(w))
```

`np.abs(w)` takes elementwise absolute values, then `np.sum` adds them all up. This is the $\ell_1$ norm.

```python
def l1_gradient(w: np.ndarray, lambda_: float) -> np.ndarray:
    return lambda_ * np.sign(w)
```

`np.sign` returns $+1$, $-1$, or $0$ elementwise. The $0$ for zero-valued weights is the subgradient convention -- it means "stop pushing once the weight reaches zero."

```python
def elastic_net_penalty(w: np.ndarray, lambda_: float, l1_ratio: float) -> float:
    l1_term = l1_ratio * np.sum(np.abs(w))
    l2_term = 0.5 * (1.0 - l1_ratio) * np.sum(w ** 2)
    return lambda_ * (l1_term + l2_term)
```

A weighted combination. When `l1_ratio = 1.0`, `l2_term` vanishes and we get pure L1. When `l1_ratio = 0.0`, `l1_term` vanishes and we get pure L2. The tests verify these boundary conditions explicitly.

```python
def elastic_net_gradient(w: np.ndarray, lambda_: float, l1_ratio: float) -> np.ndarray:
    return lambda_ * (l1_ratio * np.sign(w) + (1.0 - l1_ratio) * w)
```

Each component contributes its own gradient term, scaled by the mixing ratio.

### Implementation Walkthrough: RidgeRegression.fit()

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
    if X.ndim == 1:
        X = X.reshape(-1, 1)                          # (1)

    n_samples, n_features = X.shape
    self.w = np.zeros(n_features)                      # (2)
    self.b = 0.0
    self.history = []

    prev_loss = float("inf")

    for _ in range(self.n_iterations):
        y_pred = X @ self.w + self.b                   # (3)
        data_loss = _mse_loss(y, y_pred)               # (4)
        total_loss = data_loss + l2_penalty(self.w, self.lambda_)  # (5)
        self.history.append(total_loss)

        if abs(prev_loss - total_loss) < self.tolerance:  # (6)
            break
        prev_loss = total_loss

        error = y_pred - y                             # (7)
        dw = X.T @ error / n_samples + l2_gradient(self.w, self.lambda_)  # (8)
        db = np.mean(error)                            # (9)

        self.w -= self.learning_rate * dw              # (10)
        self.b -= self.learning_rate * db
```

**Line-by-line:**

- **(1)** Handle 1D input by reshaping to a column vector. This ensures `X.shape` is always $(n, d)$.
- **(2)** Initialize weights to zero. Bias also starts at zero. Zero initialization is safe for linear models (unlike neural networks where it causes symmetry problems).
- **(3)** Forward pass: $\hat{y} = Xw + b$. Matrix multiply $(n, d) \times (d,) \to (n,)$, then add scalar bias.
- **(4)** MSE with $\frac{1}{2n}$ factor. The $\frac{1}{n}$ normalizes across samples; the $\frac{1}{2}$ cancels the 2 in the derivative.
- **(5)** Total loss = data loss + regularization penalty. This is what we are actually minimizing.
- **(6)** Early stopping: if the loss barely changed, we have converged. This prevents wasting iterations.
- **(7)** Residual vector, shape $(n,)$.
- **(8)** The critical line. $X^\top \cdot \text{error} / n$ is the data gradient ($\frac{\partial \mathcal{L}_{\text{data}}}{\partial w}$), shape $(d, n) \times (n,) / n \to (d,)$. Then we add the L2 regularization gradient $\lambda w$. The two gradients are summed because $\frac{\partial}{\partial w}(\mathcal{L}_{\text{data}} + \text{penalty}) = \frac{\partial \mathcal{L}_{\text{data}}}{\partial w} + \frac{\partial(\text{penalty})}{\partial w}$.
- **(9)** Bias gradient is just the mean error. No regularization on the bias.
- **(10)** Standard gradient descent update.

### Implementation Walkthrough: Ridge Closed-Form Solution

```python
def fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape
    X_aug = np.column_stack([np.ones(n_samples), X])      # (1)

    reg_matrix = n_samples * self.lambda_ * np.eye(n_features + 1)  # (2)
    reg_matrix[0, 0] = 0.0                                # (3)

    A = X_aug.T @ X_aug + reg_matrix                      # (4)
    b_vec = X_aug.T @ y                                    # (5)
    w_aug = np.linalg.solve(A, b_vec)                      # (6)

    self.b = w_aug[0]                                      # (7)
    self.w = w_aug[1:]
```

**Line-by-line:**

- **(1)** Augment $X$ with a column of ones to absorb the bias into the weight vector: $X_{\text{aug}} = [\mathbf{1} \mid X]$, shape $(n, d+1)$.
- **(2)** Create the regularization matrix. The factor $n \cdot \lambda$ is needed because the data loss gradient uses $\frac{1}{n}$ normalization. The closed-form must match the same objective that gradient descent is minimizing.
- **(3)** Zero out the $(0,0)$ entry so the bias term is **not** regularized.
- **(4)** The normal equation's left-hand side: $X_{\text{aug}}^\top X_{\text{aug}} + n\lambda I$.
- **(5)** The normal equation's right-hand side: $X_{\text{aug}}^\top y$.
- **(6)** Solve the linear system $A \cdot w_{\text{aug}} = b_{\text{vec}}$. This is numerically better than computing the inverse explicitly.
- **(7)** Extract bias (first element) and weights (remaining elements).

### The Tricky Parts

**Why $n \cdot \lambda$ in the closed-form?** The gradient descent version computes the data gradient as $X^\top \cdot \text{error} / n$, which introduces a $\frac{1}{n}$ factor. The full GD objective is:

$$
\frac{1}{2n} \|Xw - y\|^2 + \frac{\lambda}{2} \|w\|^2
$$

Taking derivatives and setting to zero:

$$
\frac{1}{n} X^\top(Xw - y) + \lambda w = 0
$$

$$
\frac{X^\top X w}{n} + \lambda w = \frac{X^\top y}{n}
$$

$$
X^\top X w + n\lambda w = X^\top y
$$

$$
(X^\top X + n\lambda I) w = X^\top y
$$

That is where the $n\lambda$ comes from. The test `test_closed_form_matches_gd` verifies this correspondence.

**Why does $\text{sign}(0) = 0$ matter?** If $\text{sign}(0)$ returned $+1$ or $-1$, the L1 gradient would keep pushing a zero-valued weight away from zero, causing oscillation. The subgradient convention $\text{sign}(0) = 0$ means: once a weight reaches zero and the data gradient is not strong enough to move it, it stays at zero. This is what makes L1 sparsity stable.

**Why is the bias not regularized?** The bias shifts all predictions by a constant. Regularizing it would penalize the model for predicting values far from zero, which is arbitrary. If the true mean of $y$ is 1000, we still want $b \approx 1000$ without penalty.

---

## Deriving the Ridge Closed-Form Solution

This derivation shows how adding L2 regularization transforms the ordinary least squares normal equation into a always-solvable system.

### Starting Point: The Ridge Objective

$$
\mathcal{L}(w) = \frac{1}{2n} \|Xw - y\|^2 + \frac{\lambda}{2} \|w\|^2
$$

Expand the squared norm:

$$
\mathcal{L}(w) = \frac{1}{2n} (Xw - y)^\top (Xw - y) + \frac{\lambda}{2} w^\top w
$$

$$
= \frac{1}{2n} \left( w^\top X^\top X w - 2 w^\top X^\top y + y^\top y \right) + \frac{\lambda}{2} w^\top w
$$

### Take the Gradient

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} \left( X^\top X w - X^\top y \right) + \lambda w
$$

### Set to Zero and Solve

$$
\frac{1}{n} X^\top X w - \frac{1}{n} X^\top y + \lambda w = 0
$$

$$
\frac{1}{n} X^\top X w + \lambda w = \frac{1}{n} X^\top y
$$

$$
\left( \frac{1}{n} X^\top X + \lambda I \right) w = \frac{1}{n} X^\top y
$$

Multiply both sides by $n$:

$$
(X^\top X + n\lambda I) \, w = X^\top y
$$

Therefore:

$$
w^* = (X^\top X + n\lambda I)^{-1} X^\top y
$$

### Why This Always Has a Solution

Ordinary least squares requires $X^\top X$ to be invertible, which fails when features are collinear or when $d > n$. Adding $n\lambda I$ to $X^\top X$ shifts all eigenvalues by $n\lambda$:

If eigenvalues of $X^\top X$ are $[\sigma_1, \sigma_2, \ldots, \sigma_d]$, then eigenvalues of $X^\top X + n\lambda I$ are $[\sigma_1 + n\lambda, \sigma_2 + n\lambda, \ldots, \sigma_d + n\lambda]$.

Since $n\lambda > 0$, all eigenvalues are now strictly positive, so the matrix is always positive definite and always invertible. This is why Ridge regression is sometimes called "Tikhonov regularization" -- it is a standard technique for stabilizing ill-conditioned linear systems.

The test `test_condition_number_improvement` verifies this directly: the condition number of $X^\top X + \lambda I$ is always less than the condition number of $X^\top X$.

---

## Weight Decay vs L2 Regularization

### For SGD: They Are Identical

The implementation provides `sgd_with_weight_decay` to demonstrate this equivalence:

```python
def sgd_with_weight_decay(w, gradient, lr, lambda_):
    return w - lr * gradient - lr * lambda_ * w
```

Compare to L2-regularized SGD:

$$
w \leftarrow w - \alpha (\nabla + \lambda w) = w - \alpha \nabla - \alpha \lambda w
$$

They are identical after distributing $\alpha$. The test `test_equivalence_to_l2_sgd` verifies this to $10^{-15}$ precision, and `test_multi_step_trajectory` confirms the equivalence holds over 100 steps.

### For Adam: They Are NOT Identical

This is the critical distinction. In Adam, gradients are normalized by the second moment estimate:

```python
# L2 regularization in Adam (WRONG approach)
g = data_gradient + lambda_ * w           # regularization mixed into gradient
m = beta1 * m + (1 - beta1) * g           # first moment includes penalty
v = beta2 * v + (1 - beta2) * g**2        # second moment includes penalty
w = w - lr * m_hat / (sqrt(v_hat) + eps)  # penalty is scaled by adaptive rate
```

The problem: the $\lambda w$ term gets absorbed into $v$, inflating the second moment for large weights. A large weight has a large penalty gradient, which increases $v$, which *decreases* the effective learning rate for that weight. The adaptive scaling is counteracting the regularization -- exactly the opposite of what we want.

```python
# Decoupled weight decay in AdamW (CORRECT approach)
g = data_gradient                          # NO regularization in gradient
m = beta1 * m + (1 - beta1) * g           # clean first moment
v = beta2 * v + (1 - beta2) * g**2        # clean second moment
w = w - lr * (m_hat / (sqrt(v_hat) + eps) + lambda_ * w)  # weight decay applied DIRECTLY
```

In AdamW, the weight decay term $\lambda w$ bypasses the adaptive scaling entirely. Every weight shrinks by the same fraction $\alpha \lambda$ of its current value, regardless of gradient history. This is what "decoupled" means: the decay is decoupled from the gradient normalization.

### Why This Matters

Every modern large language model (GPT, Llama, Mistral, Gemma) is trained with AdamW, not Adam with L2 regularization. The typical weight decay value is $0.01$ to $0.1$. The "W" in AdamW literally stands for "decoupled weight decay," and the paper that introduced it (Loshchilov & Hutter, 2019) showed that the decoupled version generalizes significantly better.

Understanding this distinction means understanding why AdamW exists at all.

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Why |
|-----------|------------|-----|
| `l2_penalty` | $O(d)$ | One pass over all $d$ weights to square and sum |
| `l2_gradient` | $O(d)$ | Elementwise multiply: $\lambda w$ |
| `l1_penalty` | $O(d)$ | One pass: absolute value and sum |
| `l1_gradient` | $O(d)$ | Elementwise sign operation |
| `elastic_net_*` | $O(d)$ | Two passes (L1 + L2) combined |
| `Ridge.fit` (GD) | $O(T \cdot n \cdot d)$ | $T$ iterations, each with an $(n, d) \times (d,)$ matmul |
| `Ridge.fit_closed_form` | $O(nd^2 + d^3)$ | Form $X^\top X$ is $O(nd^2)$, solve is $O(d^3)$ |
| `Lasso.fit` (GD) | $O(T \cdot n \cdot d)$ | Same as Ridge GD |

The regularization functions themselves are $O(d)$ and negligible compared to the matrix operations in each training iteration. The $X^\top \cdot \text{error}$ computation dominates at $O(nd)$ per iteration.

### Space Complexity

| What | Size | Why |
|------|------|-----|
| Weights $w$ | $O(d)$ | One weight per feature |
| Loss history | $O(T)$ | One float per iteration |
| Intermediate $\hat{y}$ | $O(n)$ | Predictions for all samples |
| Intermediate error | $O(n)$ | Residuals |
| Closed-form $X_{\text{aug}}^\top X_{\text{aug}}$ | $O(d^2)$ | The Gram matrix |

### The Bottleneck

For gradient descent, the bottleneck is the matrix-vector products $Xw$ and $X^\top \cdot \text{error}$ at $O(nd)$ each, repeated for $T$ iterations. The regularization gradient is $O(d)$, which is negligible.

For the closed-form solution, the bottleneck is forming $X^\top X$ at $O(nd^2)$ and solving the $d \times d$ system at $O(d^3)$. When $n \gg d$, formation dominates. When $d \gg n$, the solve dominates (and the closed-form becomes impractical).

---

## Common Pitfalls

### Pitfall 1: Regularizing the Bias

**The mistake:**

```python
# Wrong: penalizing the bias term
dw = X.T @ error / n + lambda_ * w
db = np.mean(error) + lambda_ * b  # <-- BUG: bias should not be regularized
```

**Why it is wrong:** The bias shifts all predictions by a constant. Regularizing it penalizes the model for predicting values far from zero. If the target mean is 100, the model would be penalized for having $b \approx 100$, forcing worse predictions. The bias does not contribute to model complexity.

**The fix:**

```python
# Correct: only regularize feature weights
dw = X.T @ error / n + lambda_ * w
db = np.mean(error)  # no regularization on bias
```

In the implementation, $b$ is always updated with $db = \text{mean}(\text{error})$ and no penalty term.

### Pitfall 2: Forgetting the $\frac{1}{2}$ Factor in L2

**The mistake:**

```python
# Wrong: penalty and gradient are inconsistent
def l2_penalty(w, lambda_):
    return lambda_ * np.sum(w ** 2)  # missing 1/2

def l2_gradient(w, lambda_):
    return lambda_ * w  # this is the gradient of (lambda/2)*||w||^2, not lambda*||w||^2
```

**Why it is wrong:** If the penalty is $\lambda \|w\|^2$, the gradient should be $2\lambda w$. The mismatch means the gradient does not correspond to the objective being minimized, causing the loss history to be inconsistent. The optimizer might converge to a different point than the true minimum.

**The fix:**

```python
# Correct: consistent convention
def l2_penalty(w, lambda_):
    return 0.5 * lambda_ * np.sum(w ** 2)  # (lambda/2) * ||w||^2

def l2_gradient(w, lambda_):
    return lambda_ * w  # d/dw[(lambda/2) * w^2] = lambda * w
```

### Pitfall 3: Using L2 Regularization Instead of Weight Decay with Adam

**The mistake:**

```python
# Wrong: L2 penalty in gradient, then fed to Adam
total_gradient = data_gradient + lambda_ * w
# ... Adam updates using total_gradient ...
```

**Why it is wrong:** As discussed above, the adaptive scaling in Adam counteracts the regularization effect. The penalty gradient inflates the second moment, reducing the effective penalty on large weights.

**The fix:**

```python
# Correct: decoupled weight decay (AdamW)
# ... Adam updates using only data_gradient ...
w = w - lr * (adam_update + lambda_ * w)  # weight decay applied separately
```

### Pitfall 4: Wrong Sign in the Gradient Update

**The mistake:**

```python
# Wrong: adding gradient instead of subtracting
self.w += self.learning_rate * dw  # <-- BUG: should be -=
```

**Why it is wrong:** Gradient descent *descends* -- it moves in the *negative* gradient direction. Adding the gradient moves uphill, increasing the loss.

**The fix:**

```python
# Correct: subtract the gradient
self.w -= self.learning_rate * dw
```

---

## Connection to Inference Optimization

### Pruning and Sparse Inference

L1 regularization produces sparse weight matrices where many weights are exactly zero. This connects directly to model compression for inference:

```
Dense weight matrix (unpruned):      Sparse weight matrix (L1-trained):
[0.5  0.3  0.1  0.4]                [0.5  0.0  0.0  0.4]
[0.2  0.6  0.3  0.1]                [0.0  0.6  0.0  0.0]
[0.1  0.2  0.7  0.3]                [0.0  0.0  0.7  0.3]

12 multiplications                   5 multiplications (58% reduction)
```

Modern inference engines exploit sparsity in several ways:

1. **Magnitude-based pruning**: After training, remove weights below a threshold. L1 regularization naturally pushes unimportant weights toward zero, making magnitude-based pruning more effective.

2. **Structured pruning**: Remove entire neurons/channels where all weights are near zero. L1-regularized models have natural candidates.

3. **Sparse matrix formats**: Store only non-zero values (CSR, CSC formats), reducing memory and enabling specialized sparse matrix multiplication kernels.

4. **N:M sparsity**: NVIDIA Ampere GPUs support 2:4 structured sparsity natively, where 2 out of every 4 consecutive weights are zero. L1-style regularization during training helps achieve this pattern.

### AdamW in LLM Training

Every major LLM training pipeline uses AdamW with decoupled weight decay:

| Model | Weight Decay Value |
|-------|--------------------|
| GPT-3 | $0.1$ |
| LLaMA | $0.1$ |
| Mistral | $0.1$ |
| Gemma | $0.1$ |

The $0.1$ value is remarkably consistent across architectures. This means at every training step, each weight is multiplied by $(1 - \alpha \cdot 0.1)$. For a typical learning rate of $3 \times 10^{-4}$, each weight shrinks by a factor of $0.99997$ per step. Over hundreds of thousands of steps, this prevents any single weight from growing unchecked while allowing the data gradient to maintain weights that are truly important.

### From Naive to Optimized

| Naive (what we implemented) | Optimized (production) |
|----------------------------|------------------------|
| Dense weight matrix | Sparse/pruned weights after L1 training |
| Full matrix multiply | Sparse matrix multiply, skip zero blocks |
| Store all weights in memory | Store only non-zero values (CSR format) |
| Adam + L2 regularization | AdamW (decoupled weight decay) |
| Single $\lambda$ for all layers | Per-layer regularization strength |
| L1 via subgradient descent | Proximal gradient methods for sharper sparsity |

Understanding the naive version is essential because:
- You need to know *why* weights become sparse to reason about pruning
- You need to understand the L2-vs-weight-decay distinction to tune AdamW properly
- You need to see the closed-form solution to understand why Ridge always has a unique solution

---

## Testing Your Understanding

### Quick Checks

1. **What would happen if you set $\lambda = 0$?** The regularization term vanishes entirely, and all three methods reduce to ordinary least squares. The test `test_zero_lambda_reduces_to_unregularized` verifies that Ridge and Lasso produce identical weights when $\lambda = 0$.

2. **Why does the closed-form solution multiply $\lambda$ by $n$?** Because the data loss gradient uses $\frac{1}{n}$ normalization ($X^\top \cdot \text{error} / n$). To match the same objective, the closed-form must scale $\lambda$ by $n$. Without this, the closed-form would have effectively weaker regularization than the GD version.

3. **If $w = [5.0, 0.001, -3.0]$ and $\lambda = 0.1$, what are the L1 and L2 gradients?**
   - L2: $0.1 \cdot [5.0, 0.001, -3.0] = [0.5, 0.0001, -0.3]$ -- proportional to weight magnitude
   - L1: $0.1 \cdot [1, 1, -1] = [0.1, 0.1, -0.1]$ -- same magnitude for all non-zero weights

4. **Why does L2 improve the condition number of $X^\top X$?** Adding $\lambda I$ shifts all eigenvalues by $\lambda$. If the smallest eigenvalue was near zero (ill-conditioned), it becomes $\lambda$ (well-conditioned). The ratio of largest to smallest eigenvalue decreases.

### Exercises

1. **Easy**: Modify the `l1_gradient` function to use a "soft" version where $\text{sign}(w)$ is replaced by $\frac{w}{|w| + \epsilon}$ for some small $\epsilon$. How does this change the sparsity behavior?

2. **Medium**: Implement a `GroupLasso` regularization that penalizes groups of weights together: $\lambda \sum_g \|w_g\|_2$. This drives entire *groups* of weights to zero simultaneously (e.g., all weights associated with a single input feature across multiple outputs).

3. **Hard**: Implement proximal gradient descent for L1 regularization. Instead of the subgradient method used here, the proximal operator applies a soft-thresholding step:

$$
w \leftarrow \text{soft\_threshold}(w - \alpha \nabla_{\text{data}}, \, \alpha \lambda)
$$

$$
\text{soft\_threshold}(x, t) = \text{sign}(x) \cdot \max(|x| - t, \, 0)
$$

Compare convergence speed to the subgradient method.

---

## Summary

### Key Takeaways

- **Regularization prevents overfitting** by adding a penalty for large weights to the loss function, forcing the optimizer to find simpler solutions that generalize better.
- **L2 (Ridge) shrinks weights smoothly** toward zero but never reaches it. The penalty gradient $\lambda w$ is proportional to the weight, so small weights experience negligible pressure. Geometrically, the constraint region is a smooth ball with no corners.
- **L1 (Lasso) drives weights to exactly zero**, performing automatic feature selection. The penalty gradient $\lambda \, \text{sign}(w)$ is constant-magnitude regardless of weight size, overwhelming the data gradient for unimportant features. Geometrically, the diamond-shaped constraint region has corners on the axes.
- **Elastic Net combines both**, getting sparsity from L1 and stability from L2, controlled by the $\rho$ (`l1_ratio`) mixing parameter.
- **Weight decay and L2 regularization are equivalent for SGD** but **differ for Adam**. AdamW applies weight decay directly to weights, bypassing the adaptive learning rate, which is critical for proper regularization in modern LLM training.
- **The Ridge closed-form solution** $w = (X^\top X + n\lambda I)^{-1} X^\top y$ always exists because adding $\lambda I$ makes the system positive definite, even when ordinary least squares would fail.

### Quick Reference

```
Regularization
|
+-- L2 (Ridge)
|   +-- Penalty:  (lambda/2) * sum(w_i^2)
|   +-- Gradient: lambda * w
|   +-- Effect:   Smooth shrinkage, no exact zeros
|   +-- Closed-form: w = (X^T X + n*lambda*I)^{-1} X^T y
|
+-- L1 (Lasso)
|   +-- Penalty:  lambda * sum(|w_i|)
|   +-- Gradient: lambda * sign(w)
|   +-- Effect:   Sparsity, drives weights to exactly zero
|   +-- No closed-form solution
|
+-- Elastic Net (L1 + L2)
|   +-- Penalty:  lambda * (rho * ||w||_1 + (1-rho)/2 * ||w||_2^2)
|   +-- Gradient: lambda * (rho * sign(w) + (1-rho) * w)
|   +-- Effect:   Sparsity + stability
|
+-- Weight Decay
    +-- SGD: equivalent to L2 regularization
    +-- Adam: NOT equivalent -- use AdamW (decoupled weight decay)
    +-- Every modern LLM uses AdamW with weight_decay ~ 0.1

Complexity (all penalty/gradient functions):
  Time:  O(d)  -- single pass over weights
  Space: O(d)  -- gradient same size as weights

Complexity (regularized regression training):
  GD:          O(T * n * d) time, O(n + d) space
  Closed-form: O(n*d^2 + d^3) time, O(d^2) space

Optimized by: pruning, sparse kernels, N:M sparsity, AdamW
```
