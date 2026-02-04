# Softmax Regression

**Phase 1 · Topic 3** — Multiclass extension of logistic regression, and the foundation for every transformer output layer.

## What it is

Softmax regression (also called multinomial logistic regression) generalizes binary logistic regression to handle K classes simultaneously. Instead of predicting a single probability, it produces a probability distribution over all classes.

The key insight is the softmax function: it takes a vector of arbitrary real numbers (logits) and converts them into a valid probability distribution where all values are positive and sum to 1. This transformation is differentiable, which means we can train the model end-to-end with gradient descent.

Where logistic regression uses sigmoid to squeeze a single value into [0, 1], softmax regression uses the softmax function to squeeze K values into a K-dimensional probability simplex. The decision boundary between any two classes is still linear (a hyperplane), but now we have $K(K-1)/2$ pairwise boundaries.

## The math

### Notation and shapes

- $X \in \mathbb{R}^{n \times d}$ — input data (n samples, d features)
- $W \in \mathbb{R}^{d \times K}$ — weight matrix (d features, K classes)
- $b \in \mathbb{R}^{K}$ — bias vector (one bias per class)
- $Z \in \mathbb{R}^{n \times K}$ — logits (raw scores before softmax)
- $P \in \mathbb{R}^{n \times K}$ — probabilities (softmax output)
- $Y \in \mathbb{R}^{n \times K}$ — one-hot labels (ground truth)

### Forward pass

**Step 1: Compute logits**

$$Z = XW + b$$

Each row of $Z$ contains $K$ raw scores, one per class.

**Step 2: Apply softmax**

For a single sample with logit vector $z = [z_1, z_2, \ldots, z_K]$:

$$\text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

This produces probabilities $p = [p_1, p_2, \ldots, p_K]$ where $\sum_i p_i = 1$.

### Numerical stability (the subtract-max trick)

The naive softmax is numerically unstable. If $z_i$ is large (e.g., 1000), $\exp(z_i)$ overflows to infinity. If $z_i$ is very negative (e.g., -1000), $\exp(z_i)$ underflows to 0.

**Solution:** Subtract the maximum value before exponentiating:

$$\text{softmax}(z)_i = \frac{\exp(z_i - \max(z))}{\sum_j \exp(z_j - \max(z))}$$

This is mathematically equivalent (the max cancels in numerator and denominator) but now the largest exponent is $\exp(0) = 1$, preventing overflow. This trick is essential and appears in every production softmax implementation.

**Proof of equivalence:**

$$\frac{\exp(z_i - c)}{\sum_j \exp(z_j - c)} = \frac{\exp(z_i) \cdot \exp(-c)}{\sum_j \exp(z_j) \cdot \exp(-c)} = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

### Cross-entropy loss

For one-hot encoded labels $Y$, the cross-entropy loss is:

$$\mathcal{L} = -\frac{1}{n} \sum_i \sum_k Y_{ik} \log(P_{ik})$$

Since $Y$ is one-hot, only one term per sample is non-zero. If sample $i$ has true class $c$:

$$\mathcal{L}_i = -\log(P_{ic})$$

This is the negative log probability of the correct class. When the model is confident and correct, $P_{ic}$ is close to 1, so $-\log(P_{ic})$ is close to 0. When the model assigns low probability to the correct class, the loss is high.

### Gradient derivation

This is one of the most elegant results in ML. The gradient of cross-entropy loss with respect to logits simplifies to:

$$\frac{\partial \mathcal{L}}{\partial Z} = \frac{1}{n} (P - Y)$$

**Why this matters:** You don't need to backpropagate through softmax and cross-entropy separately. The combined gradient is just "predictions minus targets", identical in form to linear and logistic regression.

**Full derivation:**

Let's derive for a single sample. We want $\frac{\partial \mathcal{L}}{\partial z_i}$ where $\mathcal{L} = -\log(p_c)$ and $c$ is the true class.

First, the softmax Jacobian. For the diagonal ($i = j$):

$$\frac{\partial p_i}{\partial z_i} = p_i (1 - p_i)$$

For off-diagonal ($i \neq j$):

$$\frac{\partial p_i}{\partial z_j} = -p_i p_j$$

Now apply the chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \sum_k \frac{\partial \mathcal{L}}{\partial p_k} \cdot \frac{\partial p_k}{\partial z_i}$$

Since $\mathcal{L} = -\log(p_c)$, we have $\frac{\partial \mathcal{L}}{\partial p_k} = -\frac{1}{p_c}$ if $k = c$, else 0.

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{-1}{p_c} \cdot \frac{\partial p_c}{\partial z_i}$$

**Case 1: $i = c$ (gradient for the true class)**

$$\frac{\partial \mathcal{L}}{\partial z_c} = \frac{-1}{p_c} \cdot p_c (1 - p_c) = -(1 - p_c) = p_c - 1$$

**Case 2: $i \neq c$ (gradient for other classes)**

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{-1}{p_c} \cdot (-p_c \cdot p_i) = p_i$$

Combining both cases with one-hot encoding where $y_c = 1$ and $y_i = 0$ for $i \neq c$:

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$

In matrix form for all samples:

$$\frac{\partial \mathcal{L}}{\partial Z} = \frac{1}{n} (P - Y)$$

### Weight and bias gradients

Using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n} X^\top (P - Y) \quad \text{shape: } (d, K)$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{\text{rows}} (P - Y) \quad \text{shape: } (K,)$$

### Gradient descent update

$$W \leftarrow W - \alpha \frac{\partial \mathcal{L}}{\partial W}$$

$$b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}$$

## Why it matters for inference

Softmax is everywhere in modern ML:

**1. Transformer output layer:** Every language model ends with a linear projection to vocabulary size followed by softmax. For GPT-style models with 50K+ vocabulary, this is a significant computation.

**2. Attention weights:** The attention mechanism uses $\text{softmax}(QK^\top / \sqrt{d_k})$ to convert attention scores into weights. Every attention layer in every transformer runs softmax.

**3. Temperature scaling:** At inference time, dividing logits by temperature $T$ before softmax controls output sharpness:
- $T < 1$: sharper distribution (more confident)
- $T > 1$: softer distribution (more uniform)

**4. Numerical stability is critical:** The subtract-max trick isn't optional. Production inference engines (vLLM, TensorRT-LLM) implement numerically stable softmax, and you'll write a CUDA kernel for this exact operation.

**5. Online softmax for flash attention:** Flash attention computes softmax without materializing the full attention matrix by using the "online softmax" algorithm, which requires understanding how softmax can be computed incrementally.

**Compute characteristics:**
- Softmax is memory-bound, not compute-bound
- You read the full vector to find max, then read again to compute exp and sum
- This is 3 passes over the data in the naive implementation
- Fused implementations reduce this to 1-2 passes

## What to implement

### Core components

- [ ] `softmax(z)` — Numerically stable softmax function
  - Input: logits array of shape $(n, K)$ or $(K,)$
  - Output: probabilities of same shape, rows sum to 1
  - Must handle both 1D and 2D inputs

- [ ] `cross_entropy_loss(P, Y)` — Cross-entropy loss computation
  - Input: probabilities $P$ $(n, K)$, one-hot labels $Y$ $(n, K)$
  - Output: scalar loss value
  - Add small epsilon ($10^{-15}$) to log argument for stability

- [ ] `one_hot_encode(y, num_classes)` — Convert class indices to one-hot
  - Input: class indices $y$ $(n,)$, number of classes $K$
  - Output: one-hot matrix $(n, K)$

### SoftmaxRegression class

- [ ] `__init__(self, num_features, num_classes, learning_rate=0.01)`
  - Initialize $W$ with small random values (e.g., randn * 0.01)
  - Initialize $b$ with zeros

- [ ] `forward(self, X)` — Compute logits and probabilities
  - Return both $Z$ and $P$ for use in backward pass

- [ ] `backward(self, X, P, Y)` — Compute gradients
  - Return $\frac{\partial \mathcal{L}}{\partial W}$ and $\frac{\partial \mathcal{L}}{\partial b}$

- [ ] `fit(self, X, y, epochs=1000, verbose=False)`
  - Full training loop
  - Convert $y$ to one-hot internally
  - Track loss history

- [ ] `predict(self, X)` — Return class predictions
  - Return argmax of probabilities

- [ ] `predict_proba(self, X)` — Return probability distributions
  - Return full softmax output

### Optional extensions

- [ ] Mini-batch gradient descent
- [ ] Learning rate scheduling
- [ ] Early stopping based on loss plateau

## Test cases to cover

### Numerical stability tests

- [ ] **Large logits:** softmax([1000, 1000, 1000]) should not overflow, should return [0.333, 0.333, 0.333]
- [ ] **Mixed extreme logits:** softmax([1000, 0, 0]) should return [1, 0, 0] (approximately)
- [ ] **Negative logits:** softmax([-1000, -999, -998]) should work without underflow
- [ ] **Zero logits:** softmax([0, 0, 0]) should return [0.333, 0.333, 0.333]

### Softmax properties

- [ ] **Sum to one:** All rows of softmax output sum to 1.0 (within floating point tolerance)
- [ ] **Positive outputs:** All softmax outputs are strictly positive
- [ ] **Monotonicity:** Larger logits produce larger probabilities
- [ ] **Invariance to constant shift:** $\text{softmax}(z) = \text{softmax}(z + c)$ for any constant $c$

### Cross-entropy loss tests

- [ ] **Perfect prediction:** If $P$ matches $Y$ exactly, loss should be close to 0
- [ ] **Confident wrong prediction:** If $P$ puts 0.99 on wrong class, loss should be high (~4.6)
- [ ] **Uniform prediction:** $P = [0.333, 0.333, 0.333]$ with any one-hot $Y$ gives loss $= \log(3) \approx 1.1$

### Gradient verification

- [ ] **Numerical gradient check:** Compare analytical gradient to finite differences
  - For each weight $w_{ij}$, compute $(\mathcal{L}(w + \epsilon) - \mathcal{L}(w - \epsilon)) / (2\epsilon)$
  - Should match analytical gradient within $10^{-5}$ relative error

- [ ] **Gradient shape:** $\frac{\partial \mathcal{L}}{\partial W}$ has shape $(d, K)$, $\frac{\partial \mathcal{L}}{\partial b}$ has shape $(K,)$

### Training tests

- [ ] **Linearly separable data:** Should achieve >95% accuracy on simple 3-class problem
- [ ] **Loss decreases:** Loss should monotonically decrease (approximately) during training
- [ ] **Convergence:** Should converge to stable loss value

### Edge cases

- [ ] **Single sample:** Training with $n=1$ should work
- [ ] **Two classes:** Should behave like logistic regression
- [ ] **Single feature:** $d=1$ should work
- [ ] **Unbalanced classes:** Should still learn reasonable decision boundaries

### Integration tests

- [ ] **Iris dataset:** Classic 3-class, 4-feature dataset (implement by hand or generate similar)
- [ ] **predict vs predict_proba:** argmax(predict_proba(X)) should equal predict(X)
