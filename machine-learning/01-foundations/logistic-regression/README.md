# Logistic Regression

**Phase 1 - Topic 2** — Binary classification via the sigmoid function. Introduces nonlinearity, decision boundaries, and the loss function used in neural network output layers.

## What it is

Logistic regression transforms a linear model into a binary classifier by passing the linear output through the sigmoid function. Despite its name, it is a classification algorithm, not regression. The model outputs a probability between 0 and 1, interpreted as $P(y=1|x)$.

The key insight is that while linear regression can produce any real number, classification requires bounded outputs. The sigmoid function "squashes" any input into the $(0, 1)$ range, giving us a valid probability. The decision boundary is the hyperplane where $P(y=1|x) = 0.5$, which occurs when the linear part equals zero.

This is your first encounter with a nonlinear activation function. The pattern here — linear transformation followed by nonlinearity — is the fundamental building block of neural networks. Every hidden layer in a neural network follows this pattern, and the sigmoid specifically appears in binary classification heads, attention gates, and LSTM/GRU architectures.

## The math

### Sigmoid function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**
- Output range: $(0, 1)$
- $\sigma(0) = 0.5$
- $\sigma(-z) = 1 - \sigma(z)$ (symmetry)
- Derivative: $\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$

**Shapes:**
- Input $z \in \mathbb{R}^{n}$ or $\mathbb{R}^{n \times 1}$
- Output: same shape as input

### Model

$$z = Xw + b$$
$$\hat{y} = \sigma(z)$$

Where $z$ is the linear part (logits) and $\hat{y}$ is the predicted probabilities.

**Shapes:**
- $X \in \mathbb{R}^{n \times d}$ (n_samples, n_features)
- $w \in \mathbb{R}^{d}$ or $\mathbb{R}^{d \times 1}$
- $b \in \mathbb{R}$ (scalar or $(1,)$)
- $z \in \mathbb{R}^{n}$ or $\mathbb{R}^{n \times 1}$
- $\hat{y} \in \mathbb{R}^{n}$ or $\mathbb{R}^{n \times 1}$

### Binary Cross-Entropy Loss (BCE)

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

**Why this loss?** It's derived from maximum likelihood estimation for Bernoulli-distributed outcomes. When $y=1$, we want to maximize $\log(\hat{y})$. When $y=0$, we want to maximize $\log(1-\hat{y})$.

**Numerical stability issue:** $\log(0) = -\infty$. When $\hat{y}$ is exactly 0 or 1 (due to floating point saturation of sigmoid), the loss becomes undefined. Solution: clip $\hat{y}$ to $[\epsilon, 1-\epsilon]$ before taking the log.

$$\epsilon = 10^{-15}$$
$$\hat{y}_{\text{clipped}} = \text{clip}(\hat{y}, \epsilon, 1 - \epsilon)$$
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_{\text{clipped},i}) + (1 - y_i) \log(1 - \hat{y}_{\text{clipped},i}) \right]$$

### Gradient derivation

Starting from the loss, we need $\frac{\partial \mathcal{L}}{\partial w}$ and $\frac{\partial \mathcal{L}}{\partial b}$ for gradient descent.

**Step 1: Chain rule setup**

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**Step 2: Compute each term**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\left(\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}\right)$$

$$\frac{\partial \hat{y}}{\partial z} = \sigma(z) \cdot (1 - \sigma(z)) = \hat{y} \cdot (1 - \hat{y})$$

$$\frac{\partial z}{\partial w} = X$$

**Step 3: Simplify (the magic cancellation)**

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$

$$= -\left(\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y} \cdot (1 - \hat{y})$$

$$= -\left(y(1 - \hat{y}) - (1 - y)\hat{y}\right)$$

$$= -\left(y - y\hat{y} - \hat{y} + y\hat{y}\right)$$

$$= -(y - \hat{y})$$

$$= \hat{y} - y$$

**Final gradients:**

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^\top (\hat{y} - y)$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

**Shapes:**
- $(\hat{y} - y) \in \mathbb{R}^{n}$
- $\frac{\partial \mathcal{L}}{\partial w} \in \mathbb{R}^{d}$ — same shape as $w$
- $\frac{\partial \mathcal{L}}{\partial b} \in \mathbb{R}$ — same shape as $b$

Note: The gradient has the identical form to linear regression! This is because the sigmoid and log-loss derivatives cancel out elegantly. This is not a coincidence — BCE loss was designed this way.

### Gradient descent update

$$w \leftarrow w - \alpha \frac{\partial \mathcal{L}}{\partial w}$$

$$b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}$$

### Decision boundary

The decision boundary is where $P(y=1|x) = 0.5$, which means $\sigma(z) = 0.5$, which means $z = 0$.

$$Xw + b = 0$$

For 2D features $(x_1, x_2)$:

$$w_1 x_1 + w_2 x_2 + b = 0$$

$$x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$

This is a line with slope $-\frac{w_1}{w_2}$ and intercept $-\frac{b}{w_2}$.

## Why it matters for inference

### Foundation for neural network outputs

The sigmoid function is how neural networks produce binary classification outputs. The final layer of any binary classifier is:

```python
logits = final_hidden @ W_out + b_out
probabilities = sigmoid(logits)
```

Understanding sigmoid saturation (outputs near 0 or 1) is crucial for understanding vanishing gradients during training.

### Binary cross-entropy everywhere

BCE loss appears in:
- Binary classification heads
- Multi-label classification (sigmoid per class instead of softmax)
- Variational autoencoders (reconstruction loss for binary data)
- GANs (discriminator loss)

### Sigmoid in attention mechanisms

The sigmoid function appears in gating mechanisms:
- LSTM: forget gate, input gate, output gate all use sigmoid
- GRU: reset gate, update gate use sigmoid
- Mixture of Experts: gating networks often use sigmoid

### Numerical stability patterns

The epsilon-clipping pattern for $\log(0)$ prevention is universal. You will see this in:
- Cross-entropy loss implementations
- KL divergence computations
- Any log-probability calculation

## What to implement

### Core components

- [ ] `sigmoid(z)` — Sigmoid activation function
- [ ] `sigmoid_derivative(z)` or compute from sigmoid output — For backprop understanding
- [ ] `binary_cross_entropy(y_true, y_pred)` — Loss with numerical stability (epsilon clipping)

### LogisticRegression class

- [ ] `__init__(learning_rate, n_iterations, tolerance)` — Hyperparameters
- [ ] `fit(X, y)` — Training with gradient descent, store loss history
- [ ] `predict_proba(X)` — Return probabilities
- [ ] `predict(X, threshold=0.5)` — Return binary predictions
- [ ] `score(X, y)` — Return accuracy

### Training loop internals

- [ ] Forward pass: compute $z$, then $\hat{y}$
- [ ] Loss computation with clipping
- [ ] Backward pass: compute gradients
- [ ] Parameter update
- [ ] Convergence check (loss change $< \epsilon$)
- [ ] Loss history tracking for plotting

### Utilities

- [ ] `decision_boundary_params()` — Return slope and intercept for 2D visualization
- [ ] Weight initialization (zeros or small random values)

## Test cases to cover

### Basic correctness

- [ ] **Perfect separation:** Two clusters that are linearly separable. Model should achieve 100% accuracy after training.
- [ ] **Known weights:** Create data with known $w$ and $b$, verify model recovers approximately correct parameters.
- [ ] **Single sample:** $X$ shape $(1, d)$, $y$ shape $(1,)$. Should not crash.
- [ ] **Single feature:** $X$ shape $(n, 1)$. Verify shapes propagate correctly.

### Sigmoid function

- [ ] $\sigma(0) = 0.5$
- [ ] $\sigma(\text{large positive}) \approx 1.0$ (e.g., $\sigma(100)$ very close to 1)
- [ ] $\sigma(\text{large negative}) \approx 0.0$ (e.g., $\sigma(-100)$ very close to 0)
- [ ] $\sigma(-z) = 1 - \sigma(z)$ for random $z$ values
- [ ] **Vectorized:** sigmoid on array returns array of same shape

### Numerical stability

- [ ] **BCE with $\hat{y}=0$:** Should not produce inf or nan (epsilon prevents $\log(0)$)
- [ ] **BCE with $\hat{y}=1$:** Should not produce inf or nan
- [ ] **Extreme logits:** $\sigma(1000)$ should return $1.0$ (not overflow), $\sigma(-1000)$ should return $0.0$
- [ ] **Gradient with saturated sigmoid:** When $\hat{y}$ is very close to 0 or 1, gradients should still be finite

### Gradient correctness

- [ ] **Numerical gradient check:** Compare analytical gradient to $\frac{\mathcal{L}(w+h) - \mathcal{L}(w-h)}{2h}$ for small $h$ (e.g., $10^{-5}$)
- [ ] **Gradient shape:** $\frac{\partial \mathcal{L}}{\partial w}$ should have same shape as $w$
- [ ] **Zero gradient at optimum:** After convergence on separable data, gradient magnitude should be small

### Convergence

- [ ] **Loss decreases:** $\mathcal{L}_{i+1} \leq \mathcal{L}_i$ for reasonable learning rate
- [ ] **Early stopping:** With $\epsilon=10^{-6}$, training stops when loss change is small
- [ ] **Learning rate sensitivity:** $\alpha=0.01$ converges, $\alpha=100$ may diverge (loss increases)

### Edge cases

- [ ] **All same class:** $y = [1, 1, 1, 1]$. Model should predict all 1s (or close to it).
- [ ] **Balanced vs imbalanced:** 50-50 split vs 90-10 split. Both should train without error.
- [ ] **Zero features:** All $X$ values are 0. Model should still work (predictions based on bias only).
- [ ] **Collinear features:** Two features that are identical. Should not crash (though weights may be arbitrary).

### Shape validation

- [ ] Input $X$ must be 2D: $(n, d)$
- [ ] Input $y$ must be 1D: $(n,)$ or 2D: $(n, 1)$
- [ ] $y$ values must be 0 or 1 (or raise error)
- [ ] `predict_proba` output shape matches number of samples
- [ ] `predict` output shape matches number of samples

### Comparison with known implementation

- [ ] **Sklearn comparison (for validation only):** Train both on same data, verify predictions are similar (within tolerance). This is for testing correctness, not for the implementation itself.

## Memory analysis (bonus)

For a dataset with $n$ samples and $d$ features:

| Component | Size (floats) |
|-----------|---------------|
| $X$ | $n \cdot d$ |
| $y$ | $n$ |
| $w$ | $d$ |
| $b$ | $1$ |
| $z$ (intermediate) | $n$ |
| $\hat{y}$ (intermediate) | $n$ |
| gradient $\frac{\partial \mathcal{L}}{\partial w}$ | $d$ |

Total memory: $O(n \cdot d)$ dominated by the data matrix.

This is trivial compared to neural networks, but the pattern of tracking intermediate activations for backprop is the same pattern that makes transformer training memory-intensive.
