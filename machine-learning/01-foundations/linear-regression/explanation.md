# Linear Regression — Deep Dive

## The Intuition

### What Problem Are We Solving?

Imagine you are a real estate agent trying to predict house prices. You notice that larger houses tend to sell for more money. You have data on past sales: square footage and sale price. Can you find a rule that predicts price from size?

This is the essence of linear regression: **finding the best straight line through your data**.

```
Price ($)
    ^
400k|                              *
    |                         *
300k|                    *   *
    |               *  *
200k|          *  *
    |      * *
100k|   *
    |________________________> Square Feet
        1000  1500  2000  2500
```

You want to find: `price = w * square_feet + b`

Where:
- `w` (weight) is how much each extra square foot adds to the price
- `b` (bias) is the base price (what a 0 sq ft house would cost, theoretically)

### The Key Insight

The "best" line is the one that minimizes the total error between your predictions and the actual values. We measure error using the **mean squared error** — the average of the squared differences between predictions and reality.

Why squared? Two reasons:
1. It treats positive and negative errors equally (an overestimate by $10k is as bad as an underestimate by $10k)
2. It penalizes large errors more than small ones (being wrong by $100k is much worse than being wrong by $10k ten times)

### Real-World Analogy

Think of it like adjusting the angle and position of a ruler on a scatter plot. You slide and rotate the ruler until it passes through the "middle" of the points in a way that minimizes the total distance from each point to the ruler.

---

## The Math, Step by Step

### Building Up to the Formula

**Start simple: One feature, one sample**

For a single house with 1500 sq ft that sold for $250k:
- Prediction: `y_hat = w * 1500 + b`
- Error: `y_hat - 250000`
- Squared error: `(y_hat - 250000)^2`

**Add more samples**

With 100 houses, we want to minimize the total squared error:

```
Total error = (pred_1 - actual_1)^2 + (pred_2 - actual_2)^2 + ... + (pred_100 - actual_100)^2
```

**Add more features**

What if price depends on square footage AND number of bedrooms? Now we have multiple weights:

```
price = w_1 * square_feet + w_2 * bedrooms + b
```

### The Core Equations

**Model (Forward Pass)**

```
y_hat = X @ w + b

Where:
- X: Input features         (n_samples, n_features)
- w: Weight vector          (n_features,)
- b: Bias scalar            scalar
- y_hat: Predictions        (n_samples,)
```

The `@` symbol means matrix multiplication. For each sample, we compute the dot product of its features with the weights, then add the bias.

**Loss Function (Mean Squared Error)**

```
L = (1 / 2n) * sum((y_hat - y)^2)

Where:
- n: Number of samples
- y: True target values     (n_samples,)
- The 1/2 is for convenience — it cancels with the 2 from the derivative
```

**Gradients (How to improve)**

The gradient tells us: "If I nudge this weight up a tiny bit, how much does the loss change?"

```
dL/dw = (1/n) * X.T @ (y_hat - y)

Shape derivation:
- X.T is (n_features, n_samples)
- (y_hat - y) is (n_samples,)
- Result is (n_features,) — same shape as w!

dL/db = (1/n) * sum(y_hat - y) = mean(error)

This is just the average error — a scalar.
```

**Update Rule (Gradient Descent)**

```
w = w - learning_rate * dL/dw
b = b - learning_rate * dL/db
```

We move in the opposite direction of the gradient because the gradient points uphill (toward higher loss), and we want to go downhill (toward lower loss).

### Worked Example

Let's trace through a tiny example by hand.

**Setup:**
- 3 samples, 1 feature
- X = [[1], [2], [3]]
- y = [2, 4, 6]  (this is y = 2x, so we expect w=2, b=0)
- Initialize: w = [0], b = 0

**Iteration 1:**

```
Step 1: Forward pass
y_hat = X @ w + b = [1, 2, 3] @ [0] + 0 = [0, 0, 0]

Step 2: Compute loss
error = y_hat - y = [0, 0, 0] - [2, 4, 6] = [-2, -4, -6]
squared_error = [4, 16, 36]
loss = sum([4, 16, 36]) / (2 * 3) = 56 / 6 = 9.33

Step 3: Compute gradients
dw = X.T @ error / n = [1, 2, 3] @ [-2, -4, -6] / 3
   = (-2 - 8 - 18) / 3 = -28 / 3 = -9.33

db = mean(error) = (-2 - 4 - 6) / 3 = -4

Step 4: Update (learning_rate = 0.1)
w = 0 - 0.1 * (-9.33) = 0.933
b = 0 - 0.1 * (-4) = 0.4
```

**Iteration 2:**

```
Step 1: Forward pass
y_hat = [1, 2, 3] @ [0.933] + 0.4 = [1.33, 2.27, 3.20]

Step 2: Loss
error = [1.33-2, 2.27-4, 3.20-6] = [-0.67, -1.73, -2.80]
loss = (0.45 + 3.0 + 7.84) / 6 = 1.88 (down from 9.33!)

...and so on until convergence.
```

After many iterations, w approaches 2.0 and b approaches 0.0.

---

## From Math to Code

### The Data Structures

Our `LinearRegression` class maintains:

```python
self.w: np.ndarray   # Weight vector, shape (n_features,)
self.b: float        # Bias term, scalar
self.history: list   # Loss at each iteration (for gradient descent)
```

### Implementation Walkthrough

**The MSE Loss Function:**

```python
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = y_true.shape[0]
    return np.sum((y_pred - y_true) ** 2) / (2 * n)
```

Line-by-line:
- `y_pred - y_true`: Element-wise subtraction gives error vector
- `** 2`: Square each error
- `np.sum(...)`: Add up all squared errors
- `/ (2 * n)`: Divide by 2n for the 1/2n factor

**The Gradient Computation:**

```python
def _compute_gradients(self, X, y, y_pred):
    n = X.shape[0]
    error = y_pred - y
    dw = X.T @ error / n
    db = np.mean(error)
    return dw, db
```

Line-by-line:
- `error = y_pred - y`: Difference between predictions and truth
- `X.T @ error`: This is the key! Multiplying transposed features by errors
- `/ n`: Average over samples
- `np.mean(error)`: Bias gradient is just the mean error

**The Gradient Descent Loop:**

```python
def _fit_gradient_descent(self, X, y):
    n_samples, n_features = X.shape

    # Initialize to zeros
    self.w = np.zeros(n_features)
    self.b = 0.0
    self.history = []

    prev_loss = float("inf")

    for _ in range(self.n_iterations):
        # Forward pass
        y_pred = X @ self.w + self.b

        # Compute and record loss
        loss = mse_loss(y, y_pred)
        self.history.append(loss)

        # Check for convergence
        if abs(prev_loss - loss) < self.tolerance:
            break
        prev_loss = loss

        # Backward pass
        dw, db = self._compute_gradients(X, y, y_pred)

        # Update parameters
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
```

This is the **training loop** that appears in every ML model. The only things that change in more complex models are:
1. The forward pass (becomes deeper with more layers)
2. The loss function (cross-entropy for classification, etc.)
3. The backward pass (automatic differentiation handles this)
4. The update rule (Adam instead of vanilla gradient descent)

### The Tricky Parts

**Why `X.T @ error` and not `error @ X`?**

Shapes must match for matrix multiplication. We want the result to have shape (n_features,):
- X.T has shape (n_features, n_samples)
- error has shape (n_samples,)
- X.T @ error gives (n_features,) as needed

**Why initialize weights to zero?**

For linear regression, zero is fine. For neural networks with multiple layers, zero initialization causes problems (all neurons learn the same thing). You will see random initialization later.

**Why use `np.linalg.lstsq` instead of `np.linalg.solve`?**

The implementation uses `lstsq` (least squares) which is more robust:
- Handles underdetermined systems (more features than samples)
- Handles singular matrices gracefully
- Returns the minimum-norm solution when multiple solutions exist

---

## Normal Equation vs Gradient Descent

### The Normal Equation (Closed-Form Solution)

Linear regression is special — we can solve for the optimal weights directly using calculus.

Setting the gradient to zero and solving:

```
X.T @ X @ w = X.T @ y
w = (X.T @ X)^(-1) @ X.T @ y
```

The implementation uses the augmented matrix approach to handle the bias:

```python
def _fit_normal_equation(self, X, y):
    n_samples = X.shape[0]
    # Add column of ones for bias term
    X_aug = np.column_stack([np.ones(n_samples), X])

    # Solve the system (more stable than explicit inverse)
    w_aug, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

    # Extract bias and weights
    self.b = w_aug[0]
    self.w = w_aug[1:]
```

**Why `np.linalg.lstsq` instead of explicit inverse?**

```python
# Bad (numerically unstable)
w = np.linalg.inv(X.T @ X) @ X.T @ y

# Good (numerically stable)
w = np.linalg.solve(X.T @ X, X.T @ y)

# Better (handles edge cases)
w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
```

Matrix inversion is numerically unstable when X.T @ X is nearly singular. `lstsq` uses SVD internally, which is more robust.

### When to Use Each

```
                    Normal Equation          Gradient Descent
                    ===============          ================
Complexity          O(d^3) for inversion     O(n*d*iterations)
Memory              O(d^2) for X.T @ X       O(d) for weights only
Tuning required     None                     Learning rate, iterations
Best when           d < 10,000               d is large, or data streams in
Exact solution      Yes                      Approximate (but can be very close)
Works for           Linear regression only   Any differentiable model
```

**Rule of thumb:**
- Small dataset (< 10k features)? Use normal equation.
- Large dataset, or planning to extend to neural networks? Use gradient descent.

---

## Complexity Analysis

### Time Complexity

| Operation | Normal Equation | Gradient Descent |
|-----------|-----------------|------------------|
| Fit       | O(nd^2 + d^3)   | O(nd * iterations) |
| Predict   | O(nd)           | O(nd) |

**Normal equation breakdown:**
- X.T @ X: O(nd^2) — n samples, each contributing a d x d outer product
- Solving/inverting: O(d^3) — matrix operations are cubic in dimension
- X.T @ y: O(nd) — matrix-vector product

**Gradient descent breakdown:**
- Each iteration: O(nd) for forward pass + O(nd) for gradient
- Total: O(nd * iterations)

**When gradient descent wins:**
If d = 100,000 features, the d^3 term is 10^15 operations!
But gradient descent with 1000 iterations and 10,000 samples is only 10^12 operations.

### Space Complexity

| Operation | Normal Equation | Gradient Descent |
|-----------|-----------------|------------------|
| During fit| O(d^2 + nd)     | O(nd) |
| Storage   | O(d)            | O(d) |

**Normal equation:** Must store X.T @ X matrix, which is d x d.

**Gradient descent:** Only needs to store the gradients (size d) at each step. However, X must be in memory for both.

### The Bottleneck

For linear regression:
- **Compute bound:** The matrix multiplications X @ w and X.T @ error
- **Memory bound:** Storing the full dataset X in memory

In practice, for large datasets:
1. Use gradient descent (avoids O(d^3) inversion)
2. Use minibatches (only load part of X at a time)
3. Use streaming/online learning (update after each sample)

---

## Common Pitfalls

### Pitfall 1: Learning Rate Too High

**The mistake:**

```python
model = LinearRegression(learning_rate=10.0)  # Way too high!
model.fit(X, y)
# Loss: 1.5, 15.2, 152.8, 1528.4, inf, nan...
```

**Why it is wrong:** The gradient tells you the direction to move, but a huge step can overshoot the minimum completely. The loss oscillates and diverges.

```
Loss
  ^
  |    *
  |   / \
  |  /   \  *
  | *     \/  \
  |           * \
  +-------------->  iterations
      (diverging)
```

**The fix:**

```python
model = LinearRegression(learning_rate=0.01)  # Much safer
# Or use a learning rate finder
```

### Pitfall 2: Using Explicit Matrix Inverse

**The mistake:**

```python
# Tempting but dangerous
w = np.linalg.inv(X.T @ X) @ X.T @ y
```

**Why it is wrong:** If X.T @ X is nearly singular (has very small eigenvalues), the inverse amplifies numerical errors. You get garbage weights.

**The fix:**

```python
# Use solve (LU decomposition)
w = np.linalg.solve(X.T @ X, X.T @ y)

# Or even better, use lstsq (SVD-based)
w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
```

### Pitfall 3: Forgetting to Handle 1D Input

**The mistake:**

```python
X = np.array([1, 2, 3, 4, 5])  # 1D array, shape (5,)
y_pred = X @ self.w  # ERROR! Shape mismatch
```

**Why it is wrong:** Matrix multiplication expects X to be 2D: (n_samples, n_features).

**The fix:**

```python
if X.ndim == 1:
    X = X.reshape(-1, 1)  # Shape becomes (5, 1)
```

### Pitfall 4: Not Checking for Unfitted Model

**The mistake:**

```python
model = LinearRegression()
predictions = model.predict(X)  # Forgot to call fit()!
# TypeError or wrong results
```

**The fix:**

```python
def predict(self, X):
    if self.w is None:
        raise ValueError("Model has not been fitted. Call fit() first.")
    ...
```

---

## Connection to Deep Learning

### The Pattern That Scales

The training loop you implemented here is identical to what runs in GPT-4:

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ FORWARD  │ → │   LOSS   │ → │ BACKWARD │ → │  UPDATE  │ │
│  │   PASS   │   │          │   │   PASS   │   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                             │
│  Linear Reg:    MSE          Manual         SGD            │
│  y = Xw + b                  gradients      w -= lr * dw   │
│                                                             │
│  Transformer:   Cross-       Automatic      AdamW          │
│  100+ layers    Entropy      Differentiation + schedulers  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What Changes in Neural Networks

| Aspect | Linear Regression | Neural Networks |
|--------|-------------------|-----------------|
| Forward pass | One matrix multiply | Many matrix multiplies + nonlinearities |
| Loss | MSE | Cross-entropy, etc. |
| Gradients | Hand-derived | Automatic differentiation |
| Update | w -= lr * dw | Adam, weight decay, momentum |
| Parameters | d weights | Millions/billions |

### The Same Math, Deeper

A neural network is just many linear regressions stacked with nonlinearities:

```
Layer 1: h1 = activation(X @ W1 + b1)
Layer 2: h2 = activation(h1 @ W2 + b2)
...
Output:  y = hN @ W_out + b_out
```

The gradient computation uses the chain rule to propagate errors backward through all layers — but the fundamental operation (`X.T @ error`) appears everywhere.

### Why This Matters for Inference Optimization

Understanding linear regression helps you understand:

1. **Matrix multiplication is the bottleneck** — This is why GPUs (which excel at parallel matrix ops) dominate ML.

2. **Memory vs compute tradeoffs** — Normal equation needs O(d^2) memory; gradient descent needs O(d). In transformers, this becomes the KV cache question.

3. **Numerical stability matters** — Using `lstsq` vs `inv` seems minor here but becomes critical with 16-bit and 8-bit quantization.

4. **The training/inference asymmetry** — Training needs gradients (backward pass); inference only needs the forward pass. This is why inference can be optimized differently.

---

## Testing Your Understanding

### Quick Checks

1. **What would happen if we removed the 1/n factor from the gradients?**

   The gradients would be n times larger, requiring a learning rate n times smaller. The algorithm still works but becomes sensitive to batch size.

2. **Why do we need the bias term b?**

   Without b, the line must pass through the origin. With b, we can fit y = 2x + 100 instead of only y = 2x.

3. **What is the output shape if input is (32, 5)?**

   32 samples, 5 features. Output is (32,) — one prediction per sample.

### Exercises

1. **Easy:** Modify the implementation to support L2 regularization (Ridge regression). Add `lambda * sum(w^2)` to the loss and `2 * lambda * w` to the weight gradient.

2. **Medium:** Implement minibatch gradient descent. Instead of using all samples each iteration, randomly sample a subset.

3. **Hard:** Implement learning rate scheduling. Start with a high learning rate and decay it over time (e.g., halve it every 100 iterations).

---

## Summary

### Key Takeaways

- **Linear regression is the foundation** — The forward-loss-backward-update loop scales to transformers with billions of parameters.

- **Two solutions exist** — Normal equation (exact, O(d^3)) and gradient descent (iterative, O(nd * iterations)). Use gradient descent for large problems.

- **Numerical stability matters** — Use `np.linalg.lstsq` instead of `np.linalg.inv`. This becomes critical at scale.

- **Everything is matrix multiplication** — Understanding X @ w prepares you for understanding attention mechanisms, where Q @ K.T is the core operation.

### Quick Reference

```
Linear Regression
├── Forward: O(nd) — y_hat = X @ w + b
├── Loss: O(n) — MSE = (1/2n) * sum((y_hat - y)^2)
├── Backward: O(nd) — dw = X.T @ error / n, db = mean(error)
├── Update: O(d) — w -= lr * dw
└── Fit complexity:
    ├── Normal equation: O(nd^2 + d^3)
    └── Gradient descent: O(nd * iterations)

Memory: O(d) for weights, O(nd) for data

When d < 10,000: Use normal equation
When d > 10,000: Use gradient descent
When building neural networks: Always gradient descent
```
