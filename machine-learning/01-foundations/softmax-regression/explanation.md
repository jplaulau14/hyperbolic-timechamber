# Softmax Regression — Deep Dive

## The Intuition

### What Problem Are We Solving?

Imagine you're building a system to classify handwritten digits (0-9). With binary logistic regression, you could only answer questions like "Is this a 7 or not?" But that's awkward — you'd need 10 separate classifiers, and their outputs wouldn't even form a coherent probability distribution.

What we really want is a model that looks at an image and says: "I'm 85% sure this is a 7, 10% sure it's a 1, 3% sure it's a 9, and 2% spread across everything else." That's a **probability distribution** over all classes, and it's exactly what softmax regression provides.

The pain point is clear: binary classification doesn't scale to $K$ classes in a principled way. We need a function that takes $K$ raw scores and converts them into $K$ probabilities that:
1. Are all positive
2. Sum to exactly 1
3. Preserve the relative ordering (higher score = higher probability)

### The Key Insight

The exponential function is the hero of this story. If you have scores $[2, 1, 0]$ and want probabilities, you could try just normalizing: $[2/3, 1/3, 0]$. But that fails immediately — what about negative scores like $[-1, 0, 1]$? Normalization gives $[\text{negative}, 0, \text{positive}]$, which isn't a valid probability.

The insight is: **exponentiate first, then normalize**. Since $e^x > 0$ for all $x$, you guarantee positivity. Then dividing by the sum guarantees the probabilities sum to 1. This is softmax:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### Real-World Analogy

Think of softmax as a "voting with enthusiasm" system. Each class casts a vote, but not all votes are equal — the vote's weight is exponential in how confident the model is. A class with score 10 doesn't just get 2x the vote of a class with score 5; it gets $e^{10}/e^{5} = e^{5} \approx 148$ times the vote. This exponential scaling means small differences in confidence become large differences in probability, creating a "soft" version of argmax (hence the name).

---

## The Softmax Function

### Why Exponential? A Geometric Perspective

The softmax function has deep connections to geometry and information theory. Consider what properties we need:

1. **Positivity**: Probabilities must be non-negative
2. **Normalization**: Probabilities must sum to 1
3. **Monotonicity**: Higher logits should give higher probabilities
4. **Differentiability**: We need gradients for learning

The exponential function $e^x$ satisfies all of these:
- Always positive (maps $\mathbb{R}$ to $\mathbb{R}^+$)
- Smooth and differentiable everywhere
- Strictly increasing
- Has a special property: $\frac{d}{dx} e^x = e^x$ (derivative equals itself)

But there's a deeper reason. Softmax is the **unique** function that maximizes entropy subject to a linear constraint on expected values. In other words, it's the "least biased" way to convert scores to probabilities while respecting the relative ordering.

### The Probability Simplex

The output of softmax lives on the **probability simplex** — the set of all valid probability distributions. For 3 classes, this is a triangle in 3D space:

```
        (0, 0, 1)
           /\
          /  \
         /    \
        /      \
       /   .P   \
      /          \
     /____________\
(1,0,0)        (0,1,0)

The probability simplex for K=3:
- Vertices are one-hot vectors (certain predictions)
- Interior points are uncertain predictions
- Center (1/3, 1/3, 1/3) is maximum uncertainty
```

Softmax maps any point in $\mathbb{R}^K$ to a point on this simplex. The mapping is smooth and covers the entire interior of the simplex (but never quite reaches the vertices — you'd need infinite logits for that).

### The Formula in Detail

For a logit vector $z = [z_1, z_2, \ldots, z_K]$:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Let's trace through with $z = [2, 1, 0]$:

```
exp(2) = 7.389
exp(1) = 2.718
exp(0) = 1.000
--------------
sum    = 11.107

softmax([2, 1, 0]) = [7.389/11.107, 2.718/11.107, 1.000/11.107]
                   = [0.665, 0.245, 0.090]
```

Notice: the class with the highest logit (2) gets most of the probability mass (66.5%), but the other classes still get some probability.

---

## The Subtract-Max Trick

### The Problem: Numerical Overflow

Consider what happens with large logits:

```python
z = [1000, 1001, 1002]
exp(1000) = inf  # Overflow!
exp(1001) = inf
exp(1002) = inf

inf / inf = NaN  # Disaster
```

The exponential function grows incredibly fast. $e^{710}$ is already larger than the largest float64 can represent.

### The Solution: Shift Before Exponentiating

Here's the trick: subtract the maximum value before taking exponentials:

```python
z = [1000, 1001, 1002]
max(z) = 1002
z_shifted = [1000-1002, 1001-1002, 1002-1002] = [-2, -1, 0]

exp(-2) = 0.135
exp(-1) = 0.368
exp(0)  = 1.000
```

Now the largest exponent is $e^0 = 1$ — no overflow possible!

### Proof of Equivalence

Why does this give the same answer? Let $c = \max(z)$:

$$
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_j e^{z_j} \cdot e^{-c}} = \frac{e^{z_i} \cdot e^{-c}}{e^{-c} \cdot \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

The $e^{-c}$ factor appears in both numerator and denominator, so it cancels. The result is identical to the original formula, but computed without overflow.

### Why This Also Prevents Underflow

What about very negative logits like $[-1000, -1000, -1000]$?

Without the trick:
```
exp(-1000) = 0.0  # Underflows to zero
0 / 0 = NaN       # Disaster
```

With the trick:
```
max([-1000, -1000, -1000]) = -1000
shifted = [0, 0, 0]
exp(0) = 1
softmax = [1/3, 1/3, 1/3]  # Correct!
```

The subtract-max trick handles both overflow AND underflow.

### Implementation

```python
def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)

    if z.ndim == 1:
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)

    # 2D case: subtract max per row
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

The `keepdims=True` is crucial for broadcasting. Without it, the shapes wouldn't align for the subtraction and division.

---

## Cross-Entropy Loss

### Information Theory Connection

Cross-entropy comes from information theory. The **entropy** of a distribution $P$ measures uncertainty:

$$
H(P) = -\sum_i P_i \log(P_i)
$$

The **cross-entropy** between the true distribution $Y$ and predicted distribution $P$ is:

$$
H(Y, P) = -\sum_i Y_i \log(P_i)
$$

For one-hot labels ($Y$ is 1 for the true class, 0 elsewhere), this simplifies beautifully:

$$
H(Y, P) = -\log(P_c) \quad \text{where } c \text{ is the true class}
$$

This is exactly "negative log probability of the correct class" — a natural measure of how well the model predicts.

### Why Cross-Entropy?

Consider alternatives:
- **Mean squared error**: $(P - Y)^2$ — Doesn't penalize confidently wrong predictions enough
- **0-1 loss**: 1 if wrong, 0 if right — Not differentiable

Cross-entropy has the perfect gradient behavior:
- When you're confidently wrong, the gradient is large (strong correction)
- When you're correctly confident, the gradient is small (don't mess with success)
- The gradient is always well-defined and smooth

### The Loss Function

For a batch of $n$ samples:

$$
L = -\frac{1}{n} \sum_i \sum_k Y_{ik} \log(P_{ik})
$$

Since $Y$ is one-hot, only one term survives per sample:

$$
L = -\frac{1}{n} \sum_i \log(P_{i, c_i})
$$

where $c_i$ is the true class for sample $i$.

### Numerical Stability: The Epsilon Trick

What if the model predicts $P = 0$ for the correct class? Then $\log(0) = -\infty$. We prevent this by clipping:

```python
def cross_entropy_loss(P: np.ndarray, Y: np.ndarray, eps: float = 1e-15) -> float:
    P_clipped = np.clip(P, eps, 1.0 - eps)
    n = Y.shape[0]
    return float(-np.sum(Y * np.log(P_clipped)) / n)
```

The tiny epsilon ($10^{-15}$) prevents $\log(0)$ while having negligible effect on the actual loss value.

---

## The Elegant Gradient: $P - Y$

### The Most Beautiful Result in ML

The gradient of cross-entropy loss with respect to logits is simply:

$$
\frac{\partial L}{\partial Z} = \frac{1}{n}(P - Y)
$$

That's it. Predictions minus targets. This elegant formula appears in linear regression, logistic regression, AND softmax regression. It's not a coincidence — it emerges from the mathematical structure of exponential families.

### Full Derivation

Let's derive this step by step for a single sample. We have:
- Logits: $z = [z_1, \ldots, z_K]$
- Probabilities: $p = \text{softmax}(z)$
- One-hot label: $y$ (with $y_c = 1$ for true class $c$)
- Loss: $L = -\log(p_c)$

**Step 1: Softmax Jacobian**

First, we need the derivatives of softmax outputs with respect to logits.

For the diagonal case ($i = j$):

$$
\frac{\partial p_i}{\partial z_i} = \frac{\partial}{\partial z_i} \left[\frac{e^{z_i}}{\sum_k e^{z_k}}\right]
$$

Using the quotient rule:

$$
= \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S} \cdot \left(1 - \frac{e^{z_i}}{S}\right) = p_i(1 - p_i)
$$

where $S = \sum_k e^{z_k}$.

For off-diagonal ($i \neq j$):

$$
\frac{\partial p_i}{\partial z_j} = \frac{e^{z_i} \cdot (-1/S^2) \cdot e^{z_j}}{1} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_j}}{S} = -p_i \, p_j
$$

Compactly: $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$

where $\delta_{ij}$ is 1 if $i = j$, 0 otherwise.

**Step 2: Chain Rule**

$$
\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial p_k} \cdot \frac{\partial p_k}{\partial z_i}
$$

Since $L = -\log(p_c)$, we have $\frac{\partial L}{\partial p_k} = -1/p_c$ if $k = c$, else $0$.

$$
\frac{\partial L}{\partial z_i} = \frac{-1}{p_c} \cdot \frac{\partial p_c}{\partial z_i}
$$

**Step 3: Two Cases**

*Case 1: $i = c$ (gradient for the true class)*

$$
\frac{\partial L}{\partial z_c} = \frac{-1}{p_c} \cdot p_c(1 - p_c) = -(1 - p_c) = p_c - 1
$$

*Case 2: $i \neq c$ (gradient for other classes)*

$$
\frac{\partial L}{\partial z_i} = \frac{-1}{p_c} \cdot (-p_c \, p_i) = p_i
$$

**Step 4: Unify with One-Hot**

Since $y_c = 1$ and $y_i = 0$ for $i \neq c$:
- When $i = c$: $\frac{\partial L}{\partial z_c} = p_c - 1 = p_c - y_c$
- When $i \neq c$: $\frac{\partial L}{\partial z_i} = p_i = p_i - y_i$ (since $y_i = 0$)

Both cases: $\frac{\partial L}{\partial z_i} = p_i - y_i$

In vector form: $\frac{\partial L}{\partial z} = p - y$

**Step 5: Extend to Batches**

For $n$ samples, average the gradients:

$$
\frac{\partial L}{\partial Z} = \frac{1}{n}(P - Y)
$$

### Why Is This So Clean?

The simplicity isn't accidental. Softmax + cross-entropy is a natural pairing from the exponential family of distributions. The "sufficient statistics" of the multinomial distribution are exactly the one-hot encoded labels, and the log-normalizer (log-sum-exp) has a gradient that is the expected value of the sufficient statistics (softmax probabilities). The gradient being $P - Y$ follows from this structure.

---

## Worked Example: 3-Class Classification

Let's trace through one complete forward-backward pass with actual numbers.

### Setup

```
Inputs X (2 samples, 3 features):
    [[1.0, 0.5, -0.5],
     [0.2, -0.3, 0.8]]

Weights W (3 features, 3 classes):
    [[0.1, 0.2, 0.3],
     [0.4, -0.1, 0.2],
     [-0.2, 0.3, 0.1]]

Bias b (3 classes):
    [0.0, 0.1, -0.1]

True labels y: [0, 2]  (first sample is class 0, second is class 2)
```

### Forward Pass

**Step 1: Compute Logits $Z = XW + b$**

For sample 1: $x = [1.0, 0.5, -0.5]$
```
z_1 = 1.0*0.1 + 0.5*0.4 + (-0.5)*(-0.2) + 0.0 = 0.1 + 0.2 + 0.1 = 0.4
z_2 = 1.0*0.2 + 0.5*(-0.1) + (-0.5)*0.3 + 0.1 = 0.2 - 0.05 - 0.15 + 0.1 = 0.1
z_3 = 1.0*0.3 + 0.5*0.2 + (-0.5)*0.1 + (-0.1) = 0.3 + 0.1 - 0.05 - 0.1 = 0.25
```

For sample 2: $x = [0.2, -0.3, 0.8]$
```
z_1 = 0.2*0.1 + (-0.3)*0.4 + 0.8*(-0.2) + 0.0 = 0.02 - 0.12 - 0.16 = -0.26
z_2 = 0.2*0.2 + (-0.3)*(-0.1) + 0.8*0.3 + 0.1 = 0.04 + 0.03 + 0.24 + 0.1 = 0.41
z_3 = 0.2*0.3 + (-0.3)*0.2 + 0.8*0.1 + (-0.1) = 0.06 - 0.06 + 0.08 - 0.1 = -0.02
```

```
Z = [[0.40, 0.10, 0.25],
     [-0.26, 0.41, -0.02]]
```

**Step 2: Apply Softmax**

For sample 1: $z = [0.40, 0.10, 0.25]$
```
max(z) = 0.40
z_shifted = [0.00, -0.30, -0.15]
exp(z_shifted) = [1.000, 0.741, 0.861]
sum = 2.602
p = [0.384, 0.285, 0.331]
```

For sample 2: $z = [-0.26, 0.41, -0.02]$
```
max(z) = 0.41
z_shifted = [-0.67, 0.00, -0.43]
exp(z_shifted) = [0.512, 1.000, 0.651]
sum = 2.163
p = [0.237, 0.462, 0.301]
```

```
P = [[0.384, 0.285, 0.331],
     [0.237, 0.462, 0.301]]
```

**Step 3: One-Hot Encode Labels**

```
y = [0, 2]
Y = [[1, 0, 0],
     [0, 0, 1]]
```

**Step 4: Compute Loss**

$$
L = -\frac{1}{2}\bigl[\log(P_{0,0}) + \log(P_{1,2})\bigr] = -\frac{1}{2}\bigl[\log(0.384) + \log(0.301)\bigr]
$$

$$
= -\frac{1}{2}\bigl[-0.957 + (-1.201)\bigr] = -\frac{1}{2}(-2.158) = 1.079
$$

### Backward Pass

**Step 1: Compute $P - Y$**

$$
P - Y = \begin{bmatrix} 0.384 & 0.285 & 0.331 \\ 0.237 & 0.462 & 0.301 \end{bmatrix} - \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} -0.616 & 0.285 & 0.331 \\ 0.237 & 0.462 & -0.699 \end{bmatrix}
$$

**Step 2: Compute $dW = \frac{1}{n} X^\top (P - Y)$**

$$
X^\top = \begin{bmatrix} 1.0 & 0.2 \\ 0.5 & -0.3 \\ -0.5 & 0.8 \end{bmatrix}
$$

$$
dW = \frac{1}{2} X^\top (P - Y)
$$

```
Row 1: [1.0*(-0.616) + 0.2*0.237, 1.0*0.285 + 0.2*0.462, 1.0*0.331 + 0.2*(-0.699)] / 2
     = [-0.569, 0.377, 0.191] / 2 = [-0.284, 0.189, 0.096]

Row 2: [0.5*(-0.616) + (-0.3)*0.237, ...] / 2
     = [-0.379, 0.004, 0.376] / 2 = [-0.190, 0.002, 0.188]

Row 3: [(-0.5)*(-0.616) + 0.8*0.237, ...] / 2
     = [0.498, 0.227, -0.725] / 2 = [0.249, 0.114, -0.362]
```

$$
dW = \begin{bmatrix} -0.284 & 0.189 & 0.096 \\ -0.190 & 0.002 & 0.188 \\ 0.249 & 0.114 & -0.362 \end{bmatrix}
$$

**Step 3: Compute $db = \text{mean}(P - Y, \text{axis}=0)$**

$$
db = \left[\frac{-0.616 + 0.237}{2},\; \frac{0.285 + 0.462}{2},\; \frac{0.331 - 0.699}{2}\right] = [-0.190,\; 0.374,\; -0.184]
$$

**Step 4: Update Parameters**

With learning rate $\alpha = 0.1$:

$$
W_{\text{new}} = W - \alpha \cdot dW
$$

$$
b_{\text{new}} = b - \alpha \cdot db
$$

---

## Math-to-Code Mapping

### The Forward Pass

**Math:**

$$
Z = XW + b
$$

$$
P = \text{softmax}(Z)
$$

**Code:**
```python
def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # (n, d) @ (d, K) -> (n, K)
    Z = X @ self.W + self.b
    P = softmax(Z)
    return Z, P
```

### The Backward Pass

**Math:**

$$
\frac{\partial L}{\partial W} = \frac{1}{n} X^\top (P - Y)
$$

$$
\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{\text{rows}} (P - Y)
$$

**Code:**
```python
def backward(self, X: np.ndarray, P: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    error = P - Y
    # (d, n) @ (n, K) -> (d, K)
    dW = X.T @ error / n
    # (K,)
    db = np.mean(error, axis=0)
    return dW, db
```

### Shape Annotations

```
X:     (n, d)  — n samples, d features
W:     (d, K)  — d features, K classes
b:     (K,)    — one bias per class
Z:     (n, K)  — logits
P:     (n, K)  — probabilities
Y:     (n, K)  — one-hot labels
P - Y: (n, K)  — error
dW:    (d, K)  — weight gradient
db:    (K,)    — bias gradient
```

### Key Broadcasting Operations

**Adding bias:**
```python
Z = X @ self.W + self.b  # (n, K) + (K,) broadcasts to (n, K)
```

**Softmax normalization:**
```python
z_shifted = z - np.max(z, axis=1, keepdims=True)  # (n, K) - (n, 1) = (n, K)
exp_z / np.sum(exp_z, axis=1, keepdims=True)      # (n, K) / (n, 1) = (n, K)
```

The `keepdims=True` maintains the shape for proper broadcasting.

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Why |
|-----------|------------|-----|
| Forward ($Z = XW + b$) | $O(n \cdot d \cdot K)$ | Matrix multiplication |
| Softmax | $O(n \cdot K)$ | Three passes: max, exp, sum (per row) |
| Cross-entropy | $O(n \cdot K)$ | Element-wise multiply and sum |
| Backward ($dW$) | $O(n \cdot d \cdot K)$ | Matrix multiplication $X^\top \cdot \text{error}$ |
| Backward ($db$) | $O(n \cdot K)$ | Sum over samples |

**Total per iteration:** $O(n \cdot d \cdot K)$

The matrix multiplications dominate. For large vocabulary ($K = 50{,}000$ in LLMs), this becomes expensive.

### Space Complexity

| What | Size | Why |
|------|------|-----|
| Weights $W$ | $O(d \cdot K)$ | Stored for inference and training |
| Bias $b$ | $O(K)$ | One per class |
| Logits $Z$ | $O(n \cdot K)$ | Intermediate (can be computed on-the-fly) |
| Probabilities $P$ | $O(n \cdot K)$ | Needed for loss and gradients |
| One-hot $Y$ | $O(n \cdot K)$ | Can use sparse representation |

**For a language model with 50K vocabulary and batch size 1024:**
- $P$ alone is $1024 \times 50{,}000 \times 4$ bytes $= 200$ MB (in float32)
- This is why LLMs often use mixed precision

### The Bottleneck

The softmax computation itself is **memory-bound**, not compute-bound:
1. Read all $K$ logits to find max
2. Read all $K$ logits again to compute exp and sum
3. Read all $K$ values again to normalize

Three passes over the data! This is why fused softmax kernels that do everything in one pass are so valuable.

---

## Common Pitfalls

### Pitfall 1: Forgetting the Subtract-Max Trick

**The mistake:**
```python
def softmax_wrong(z):
    exp_z = np.exp(z)  # BOOM for large z
    return exp_z / np.sum(exp_z)
```

**Why it's wrong:** $e^{1000} = \infty$, leading to $\infty / \infty = \text{NaN}$.

**The fix:**
```python
def softmax(z):
    z_shifted = z - np.max(z)  # Now max is 0
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)
```

### Pitfall 2: Wrong Axis for 2D Softmax

**The mistake:**
```python
def softmax_wrong(z):
    z_shifted = z - np.max(z)  # Takes global max, not per-row
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)  # Sums entire matrix
```

**Why it's wrong:** Each row should be a separate probability distribution. The global max/sum mixes samples together.

**The fix:**
```python
def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

### Pitfall 3: Missing Epsilon in Cross-Entropy

**The mistake:**
```python
def cross_entropy_wrong(P, Y):
    return -np.mean(np.sum(Y * np.log(P), axis=1))
```

**Why it's wrong:** If the model is very confident but wrong, $P$ for the correct class might be $\approx 0$, and $\log(0) = -\infty$.

**The fix:**
```python
def cross_entropy(P, Y, eps=1e-15):
    P_clipped = np.clip(P, eps, 1.0 - eps)
    return -np.mean(np.sum(Y * np.log(P_clipped), axis=1))
```

### Pitfall 4: Forgetting to Average the Gradient

**The mistake:**
```python
def backward_wrong(X, P, Y):
    dW = X.T @ (P - Y)  # Missing /n
    return dW
```

**Why it's wrong:** The gradient scales with batch size. Training becomes unstable as batch size changes.

**The fix:**
```python
def backward(X, P, Y):
    n = X.shape[0]
    dW = X.T @ (P - Y) / n
    return dW
```

---

## Connection to Transformers

### Attention Weights

Every transformer attention layer uses softmax:

```python
# Attention mechanism
scores = Q @ K.T / sqrt(d_k)  # (seq, seq)
weights = softmax(scores)     # (seq, seq) — each row sums to 1
output = weights @ V          # (seq, d_v)
```

The softmax converts raw attention scores into a probability distribution over positions. This is why understanding softmax is essential for understanding attention.

```
Attention Flow:

    Q (seq, d)     K^T (d, seq)       Scores (seq, seq)
    ┌─────────┐    ┌─────────┐        ┌─────────────┐
    │ query 1 │    │ k k k k │        │ 2.1 0.5 -0.3│
    │ query 2 │  @ │ e e e e │   =    │ 0.3 3.2 1.1 │
    │ query 3 │    │ y y y y │        │ 1.1 0.8 2.5 │
    └─────────┘    └─────────┘        └─────────────┘
                                             │
                                        softmax (row-wise)
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │ 0.6 0.3 0.1 │  <- weights sum to 1
                                      │ 0.1 0.7 0.2 │
                                      │ 0.2 0.2 0.6 │
                                      └─────────────┘
```

### Output Layer

Every language model ends with:
```python
logits = hidden @ W_vocab + b_vocab  # (batch, vocab_size)
probs = softmax(logits)              # Probability over vocabulary
next_token = sample(probs)           # Or argmax for greedy decoding
```

For GPT-3 with 50,257 vocabulary tokens, this softmax runs over 50K+ values for every token generated.

### Why Softmax is a Bottleneck

In transformers, softmax appears:
1. **In every attention head** — $O(\text{seq\_length}^2)$ operations for self-attention
2. **In the output layer** — $O(\text{vocab\_size})$ operations per token

For long sequences, the attention softmax is quadratic in sequence length. This is the main motivation for efficient attention methods like Flash Attention.

---

## Temperature Scaling

### How Temperature Works

Temperature divides the logits before softmax:

```python
def softmax_with_temperature(z, temperature=1.0):
    z_scaled = z / temperature
    return softmax(z_scaled)
```

### The Effect

Consider $z = [2, 1, 0]$:

| Temperature | Softmax Output | Character |
|-------------|----------------|-----------|
| $T = 0.5$ | $[0.84, 0.11, 0.05]$ | Sharp (confident) |
| $T = 1.0$ | $[0.67, 0.24, 0.09]$ | Normal |
| $T = 2.0$ | $[0.51, 0.31, 0.18]$ | Soft (uncertain) |
| $T \to 0$ | $[1.0, 0.0, 0.0]$ | Argmax (one-hot) |
| $T \to \infty$ | $[0.33, 0.33, 0.33]$ | Uniform |

```
Temperature Effect on Distribution:

T = 0.1 (very sharp)    T = 1.0 (normal)      T = 5.0 (soft)
     ___                     ___                  ___
    |   |                   |   |                |   |
    |   |                   |   |_              _|   |_
    |   |_               ___|   | |_           | |   | |
____|   | |_         ___|   |   | | |___   ____|_|   | |____
   A  B  C             A  B  C             A  B  C
```

### Mathematical Intuition

Dividing logits by $T$ is equivalent to raising probabilities to the power $1/T$:

$$
\text{softmax}(z/T)_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}} \propto \bigl[\text{softmax}(z)_i\bigr]^{1/T}
$$

- $T < 1$: Sharpens the distribution (raises probabilities to power $> 1$)
- $T > 1$: Flattens the distribution (raises probabilities to power $< 1$)

### Use in LLM Inference

Temperature is the primary creativity control:
- **$T = 0$** (or very small): Deterministic, always picks most likely token
- **$T = 0.7$**: Good balance for coherent but varied text
- **$T = 1.0$**: Sample from the model's actual distribution
- **$T > 1.0$**: More random, creative, but potentially incoherent

```python
# Typical LLM sampling
logits = model(input_ids)
scaled_logits = logits / temperature
probs = softmax(scaled_logits)
next_token = np.random.choice(vocab_size, p=probs)
```

---

## Testing Your Understanding

### Quick Checks

1. **Why can't we use $z / \sum(z)$ instead of softmax?**
   - Negative logits would give negative "probabilities"
   - We need $\exp(\cdot)$ to guarantee positivity

2. **Why subtract the max instead of, say, the mean?**
   - The max gives the tightest bound on the largest exponent
   - Subtracting the mean could still leave large positive values that overflow

3. **What shape is the softmax Jacobian for a single sample with $K$ classes?**
   - $(K, K)$ — derivatives of $K$ outputs with respect to $K$ inputs

4. **Why is the gradient of softmax + cross-entropy simpler than the product of their individual gradients?**
   - Cancellation: the $1/p$ term from cross-entropy gradient cancels with the $p$ term from softmax Jacobian

### Exercises

1. **Easy**: Implement softmax for 3D input (batch, sequence, classes). Handle the axis correctly.

2. **Medium**: Implement temperature scaling and verify that $T \to 0$ gives argmax behavior.

3. **Hard**: Implement the "online softmax" algorithm that computes softmax in a single pass (used in Flash Attention).

---

## Summary

### Key Takeaways

- **Softmax converts logits to probabilities** using exponentiation and normalization, mapping $\mathbb{R}^K$ to the probability simplex

- **The subtract-max trick is essential** for numerical stability, preventing both overflow and underflow without changing the result

- **Cross-entropy loss measures prediction quality** as negative log probability of the correct class, with information-theoretic roots

- **The gradient $P - Y$ is beautifully simple** because softmax and cross-entropy are naturally paired from the exponential family

- **Softmax appears everywhere in transformers** — in attention weights and output layers, making it a critical operation for inference optimization

- **Temperature controls distribution sharpness** — lower temperature = more confident/deterministic, higher = more uniform/random

### Quick Reference

```
Softmax Regression
├── Forward: O(n * d * K) — matrix multiply + softmax
├── Backward: O(n * d * K) — gradient is (P - Y), then matrix multiply
├── Memory: O(d * K + K) — weights and biases
└── Key insight: gradient = predictions - targets

Numerical Stability:
├── Softmax: subtract max before exp
└── Cross-entropy: clip probabilities before log

Softmax in Transformers:
├── Attention: softmax(QK^T / sqrt(d_k)) — over positions
├── Output: softmax(logits) — over vocabulary
└── Optimization: fused kernels, flash attention, online softmax
```
