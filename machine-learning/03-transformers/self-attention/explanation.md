# Self-Attention -- Deep Dive

## The Intuition

### What Problem Are We Solving?

Consider translating the sentence "The cat sat on the mat because **it** was tired." What does "it" refer to? A human reader instantly knows "it" means "the cat," not "the mat." But for a model processing this sequence token by token, there is no built-in mechanism to connect "it" at position 8 back to "cat" at position 2. Recurrent networks attempted this with hidden states passed forward step by step, but information decayed over long distances and computation could not be parallelized.

Self-attention solves this by letting every position in a sequence directly examine every other position in a single operation. Position 8 ("it") can look at position 2 ("cat"), compute a relevance score, and pull in that information -- all without the bottleneck of sequential processing.

### The Key Insight

The core idea is **content-based addressing**. Instead of hard-coding which positions to look at (like a convolutional filter with a fixed window), self-attention computes relevance scores dynamically from the content itself. A token asks "who is relevant to me?" and every other token answers with a score. These scores become weights in a weighted average over the sequence.

The decomposition into three separate projections -- queries ($Q$), keys ($K$), and values ($V$) -- is what makes this work. It separates the notion of "what I am looking for" from "what I am advertising" from "what I actually provide."

### Real-World Analogy: The Library Lookup

Imagine you are in a library with a specific question (your **query**). Every book on the shelf has a title on its spine (its **key**). You scan the titles, and some match your question better than others -- that is the dot product between your query and each key, producing a relevance score. But you do not read the title itself; you open the book and read its content (the **value**). The answer you walk away with is a blend of the contents from all the books, weighted by how relevant each title was to your question.

In a hash table, you get exactly one match. In attention, every "book" contributes to your answer -- the highly relevant ones contribute a lot, and the irrelevant ones contribute almost nothing. This soft selection is what makes attention differentiable and trainable.

---

## The Math, Step by Step

### Building Up to the Formula

**Step 1 -- The simplest version.** If we just want to measure similarity between positions, we could compute the dot product of every pair of input vectors:

$$S = XX^\top \quad \text{shape: } (n, n)$$

Each element $S_{i,j}$ measures how similar position $i$ is to position $j$. Normalize each row with softmax and we have attention weights. But this is too simple -- the same vector serves as both the "question" and the "answer," so there is no way to learn different roles.

**Step 2 -- Separate projections.** Project the input into different spaces:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

Now queries and keys can learn to encode different aspects of the input. The scores become $S = QK^\top$, and the output is a weighted sum of values rather than of raw inputs. This separation is what gives attention its expressive power.

**Step 3 -- Scaling.** As we will derive below, the dot products in $QK^\top$ have variance proportional to $d_k$. Without correction, large dimensions cause softmax to saturate, producing near-binary attention weights with vanishing gradients. Dividing by $\sqrt{d_k}$ normalizes the variance back to 1.

**Step 4 -- Masking.** For autoregressive models (like GPT), position $i$ must not see positions $j > i$. An additive mask of $-\infty$ for future positions forces their attention weights to zero after softmax.

### The Core Equations

**The complete attention formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Q/K/V projections:**

$$Q = XW_Q + b_Q \quad \in \mathbb{R}^{B \times n \times d_k}$$

$$K = XW_K + b_K \quad \in \mathbb{R}^{B \times n \times d_k}$$

$$V = XW_V + b_V \quad \in \mathbb{R}^{B \times n \times d_v}$$

Where:
- $X \in \mathbb{R}^{B \times n \times d_{\text{model}}}$: input sequence ($B$ = batch, $n$ = sequence length)
- $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$: query projection matrix
- $W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$: key projection matrix
- $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$: value projection matrix
- $b_Q, b_K \in \mathbb{R}^{d_k}$, $b_V \in \mathbb{R}^{d_v}$: optional bias vectors

**Output projection:**

$$\text{output} = OW_O + b_O \quad \in \mathbb{R}^{B \times n \times d_{\text{model}}}$$

Where $O$ is the attention output, $W_O \in \mathbb{R}^{d_v \times d_{\text{model}}}$, and $b_O \in \mathbb{R}^{d_{\text{model}}}$.

### Scaled Dot-Product Attention -- Shape by Shape

Here is the full pipeline with shapes annotated at every step:

```
Step 1: Compute raw scores
    Q             K^T              S
(B, n, d_k) @ (B, d_k, n)  =  (B, n, n)

    S[b,i,j] = sum over k of Q[b,i,k] * K[b,j,k]
    "How much should position i attend to position j?"

Step 2: Scale
    S = S / sqrt(d_k)
    Shape unchanged: (B, n, n)

Step 3: Mask (optional)
    S = S + M           where M[i,j] = -inf for masked positions
    Shape unchanged: (B, n, n)

Step 4: Softmax (row-wise)
    A = softmax(S, axis=-1)
    Shape: (B, n, n)
    Each row A[b,i,:] sums to 1

Step 5: Weighted sum of values
    A          V             O
(B, n, n) @ (B, n, d_v) = (B, n, d_v)

    O[b,i,:] = sum over j of A[b,i,j] * V[b,j,:]
    "Output at position i is a weighted blend of all value vectors"
```

### Why Three Separate Projections?

This question deserves a dedicated answer. Why not just use $X$ directly for queries, keys, and values?

Using a single shared representation forces each token to serve all three roles with the same vector. But what a token is "looking for" (query), what it "advertises about itself" (key), and what "information it provides" (value) are conceptually different functions. Separate projections let the model learn independent representations for each role.

Consider: the word "bank" in "river bank" vs. "bank account" has the same token embedding, but its query ("what context am I in?"), key ("I could be about nature or finance"), and value ("here is the information I provide in this context") should all be different. Separate $W_Q$, $W_K$, $W_V$ matrices enable this.

---

## Why Scale by $\sqrt{d_k}$ -- The Variance Argument

This is one of the most important details to understand. Here is the full derivation.

### Setup

Assume $Q$ and $K$ have elements independently drawn from a distribution with mean 0 and variance 1 (a reasonable assumption after proper initialization and layer normalization). Consider a single dot product between a query vector $q \in \mathbb{R}^{d_k}$ and a key vector $k \in \mathbb{R}^{d_k}$:

$$s = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

### Computing the Variance

Each term $q_i k_i$ is a product of two independent zero-mean, unit-variance random variables:

$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \cdot \mathbb{E}[k_i] = 0 \cdot 0 = 0$$

$$\text{Var}[q_i k_i] = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \cdot \mathbb{E}[k_i^2] - 0 = 1 \cdot 1 = 1$$

Since the $d_k$ terms are independent, the variance of the sum is the sum of variances:

$$\text{Var}[s] = \text{Var}\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} \text{Var}[q_i k_i] = d_k$$

So the standard deviation of the dot product is $\sqrt{d_k}$.

### The Problem Without Scaling

For $d_k = 64$ (a typical head dimension), the dot products have standard deviation $\sqrt{64} = 8$. In a row of scores, some values will be around $+16$ while others are around $-16$. When softmax sees inputs that differ by more than about 10, it saturates: the output becomes essentially one-hot.

```
Example without scaling (d_k = 64):
    scores = [12.3, -8.7, 15.1, -3.2]
    softmax = [0.06, 0.00, 0.94, 0.00]    <-- near-binary, gradients vanish
```

### The Fix

Dividing by $\sqrt{d_k}$ normalizes the variance:

$$\text{Var}\left[\frac{s}{\sqrt{d_k}}\right] = \frac{\text{Var}[s]}{d_k} = \frac{d_k}{d_k} = 1$$

Now scores have standard deviation 1 regardless of the head dimension, keeping softmax in its useful regime:

```
Example with scaling (d_k = 64, dividing by 8):
    scores = [1.54, -1.09, 1.89, -0.40]
    softmax = [0.28, 0.06, 0.40, 0.10]    <-- meaningful distribution, healthy gradients
```

This is verified empirically in the test `test_large_dk_without_scaling_saturates`: with $d_k = 512$ and no scaling, the maximum attention weight per row averages above 0.9 (near one-hot), while with proper scaling it drops below 0.5 (a meaningful distribution).

---

## Causal Masking

### Why Autoregressive Models Need It

In a language model like GPT, the task is to predict the next token given all previous tokens. During training, we process the entire sequence at once for efficiency, but the model at position $i$ must only use information from positions $\leq i$. If position 3 could "see" position 4, the model would learn to cheat -- just copy the answer from the future rather than actually learning to predict.

### The $-\infty$ Trick

We need attention weights to be exactly zero for future positions. Multiplying the attention weights by zero after softmax would break normalization (rows would no longer sum to 1). Instead, we add $-\infty$ to the scores *before* softmax:

$$M_{i,j} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

After adding the mask: $e^{-\infty} = 0$, so softmax naturally redistributes probability mass only among the unmasked (past and present) positions.

### What the Mask Looks Like

For a 4-token sequence:

```
Mask M:                        Attention weights A:
[ 0   -inf  -inf  -inf]       [1.00  0.00  0.00  0.00]
[ 0    0    -inf  -inf]       [0.45  0.55  0.00  0.00]
[ 0    0     0    -inf]       [0.20  0.35  0.45  0.00]
[ 0    0     0     0  ]       [0.15  0.25  0.30  0.30]
                                Each row sums to 1
```

Token 0 can only attend to itself (weight = 1.0). Token 1 splits attention between positions 0 and 1. Token 3 can attend to all four positions.

### Implementation

```python
def create_causal_mask(seq_len: int) -> np.ndarray:
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask
```

`np.triu_indices(n, k=1)` returns the indices of the strictly upper-triangular portion of an $n \times n$ matrix. Setting these to $-\infty$ produces the additive causal mask.

---

## The Attention Matrix -- What Patterns Mean

The attention weight matrix $A \in \mathbb{R}^{n \times n}$ is one of the most interpretable artifacts in deep learning. Each row $A[i, :]$ is a probability distribution showing where position $i$ is looking.

### Common Patterns

```
Diagonal (self-attending):      Vertical stripe:              Lower-triangular (causal):
[0.8 0.1 0.1 0.0]              [0.1 0.7 0.1 0.1]            [1.0  0   0   0 ]
[0.1 0.7 0.1 0.1]              [0.1 0.6 0.2 0.1]            [0.4 0.6  0   0 ]
[0.1 0.1 0.8 0.0]              [0.1 0.7 0.1 0.1]            [0.2 0.3 0.5  0 ]
[0.0 0.1 0.1 0.8]              [0.2 0.5 0.2 0.1]            [0.1 0.2 0.3 0.4]
"Each token focuses on itself"  "Everyone looks at token 1"   "Autoregressive pattern"
```

- **Diagonal-heavy**: the model is relying on local/positional information
- **Column-heavy (vertical stripes)**: one token (e.g., [CLS] or a key noun) is globally important
- **Diffuse/uniform rows**: the model is averaging broadly, perhaps for context aggregation
- **Block structure**: the model has learned to segment the sequence

---

## Worked Example

Let us trace through self-attention with actual numbers. We use 2 tokens and $d_k = d_v = 3$, with identity-like projections (skip projections to focus on the attention mechanism itself).

### Input

$$Q = \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 0.0 & 1.0 & 0.0 \end{bmatrix}, \quad K = \begin{bmatrix} 1.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 1.0 \end{bmatrix}, \quad V = \begin{bmatrix} 10.0 & 20.0 & 30.0 \\ 40.0 & 50.0 & 60.0 \end{bmatrix}$$

Shapes: $Q, K \in \mathbb{R}^{1 \times 2 \times 3}$ (batch=1), $V \in \mathbb{R}^{1 \times 2 \times 3}$.

### Step 1: Raw Scores $S = QK^\top$

$$S = QK^\top = \begin{bmatrix} 1 \cdot 1 + 0 \cdot 0 + 1 \cdot 0 & 1 \cdot 0 + 0 \cdot 1 + 1 \cdot 1 \\ 0 \cdot 1 + 1 \cdot 0 + 0 \cdot 0 & 0 \cdot 0 + 1 \cdot 1 + 0 \cdot 1 \end{bmatrix} = \begin{bmatrix} 1.0 & 1.0 \\ 0.0 & 1.0 \end{bmatrix}$$

Shape: $(1, 2, 2)$.

### Step 2: Scale by $\sqrt{d_k} = \sqrt{3} \approx 1.732$

$$S_{\text{scaled}} = \frac{S}{\sqrt{3}} = \begin{bmatrix} 0.577 & 0.577 \\ 0.000 & 0.577 \end{bmatrix}$$

### Step 3: No Mask (bidirectional attention)

$S_{\text{masked}} = S_{\text{scaled}}$ (unchanged).

### Step 4: Softmax (row-wise)

**Row 0**: inputs are $[0.577, 0.577]$

$$\text{softmax}([0.577, 0.577]) = \left[\frac{e^{0.577}}{e^{0.577} + e^{0.577}}, \frac{e^{0.577}}{e^{0.577} + e^{0.577}}\right] = [0.500, 0.500]$$

**Row 1**: inputs are $[0.000, 0.577]$

$$e^{0.000} = 1.000, \quad e^{0.577} = 1.781$$

$$\text{softmax}([0.000, 0.577]) = \left[\frac{1.000}{1.000 + 1.781}, \frac{1.781}{1.000 + 1.781}\right] = [0.360, 0.640]$$

$$A = \begin{bmatrix} 0.500 & 0.500 \\ 0.360 & 0.640 \end{bmatrix}$$

### Step 5: Weighted Sum $O = AV$

$$O = AV = \begin{bmatrix} 0.500 \cdot 10 + 0.500 \cdot 40 & 0.500 \cdot 20 + 0.500 \cdot 50 & 0.500 \cdot 30 + 0.500 \cdot 60 \\ 0.360 \cdot 10 + 0.640 \cdot 40 & 0.360 \cdot 20 + 0.640 \cdot 50 & 0.360 \cdot 30 + 0.640 \cdot 60 \end{bmatrix}$$

$$O = \begin{bmatrix} 25.0 & 35.0 & 45.0 \\ 29.2 & 39.2 & 49.2 \end{bmatrix}$$

**Interpretation**: Token 0 attends equally to both tokens (0.5/0.5), so its output is the midpoint of the two value vectors. Token 1 attends more to token 1 (0.64 vs 0.36), so its output is shifted toward the second value vector.

---

## From Math to Code

### The Data Structures

The `SelfAttention` class maintains:

**Parameters (learned):**
- `W_Q` $(d_{\text{model}}, d_k)$, `W_K` $(d_{\text{model}}, d_k)$, `W_V` $(d_{\text{model}}, d_v)$, `W_O` $(d_v, d_{\text{model}})$: projection matrices
- `b_Q` $(d_k)$, `b_K` $(d_k)$, `b_V` $(d_v)$, `b_O` $(d_{\text{model}})$: optional biases

**Cache (saved during forward for backward):**
- `X`, `Q`, `K`, `V`, `A`, `O`, `mask`: all intermediates needed to compute gradients without re-running the forward pass

**Gradients (computed during backward):**
- `grad_W_Q`, `grad_W_K`, `grad_W_V`, `grad_W_O`: parameter gradients
- `grad_b_Q`, `grad_b_K`, `grad_b_V`, `grad_b_O`: bias gradients

### Implementation Walkthrough: Forward Pass

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    A = softmax(scores, axis=-1)
    output = A @ V
    return output, A
```

**Line-by-line:**

- `d_k = Q.shape[-1]`: Extract the query/key dimension from the last axis of $Q$. This is the dimension we sum over in the dot product, and the value we scale by.

- `Q @ K.transpose(0, 2, 1)`: This is $QK^\top$. The `.transpose(0, 2, 1)` swaps the last two axes of $K$ (from $(B, n, d_k)$ to $(B, d_k, n)$) while leaving the batch axis alone. The `@` operator performs batched matrix multiplication. Result shape: $(B, n, n)$.

- `/ np.sqrt(d_k)`: The scaling factor, applied element-wise to all scores. This is critical for numerical stability -- see the variance argument above.

- `scores = scores + mask`: Additive masking. The mask is broadcastable to $(B, n, n)$. For causal masks, the shape is $(n, n)$ and broadcasting adds the batch dimension. For padding masks, the shape is $(B, 1, n)$ and broadcasting adds the query dimension.

- `A = softmax(scores, axis=-1)`: Row-wise softmax. Each row becomes a probability distribution over keys.

- `output = A @ V`: Weighted sum of values. Shape $(B, n, n) \times (B, n, d_v) = (B, n, d_v)$. Each output vector is a convex combination of all value vectors, weighted by the attention distribution.

### Implementation Walkthrough: Full SelfAttention.forward

```python
def forward(self, X, mask=None):
    X = np.asarray(X, dtype=np.float64)

    Q = X @ self.W_Q          # (B, n, d_model) @ (d_model, d_k) -> (B, n, d_k)
    K = X @ self.W_K
    V = X @ self.W_V          # -> (B, n, d_v)

    if self.use_bias:
        Q = Q + self.b_Q      # broadcast: (B, n, d_k) + (d_k,)
        K = K + self.b_K
        V = V + self.b_V

    O, A = scaled_dot_product_attention(Q, K, V, mask)

    output = O @ self.W_O     # (B, n, d_v) @ (d_v, d_out) -> (B, n, d_out)
    if self.use_bias:
        output = output + self.b_O

    self._cache = {"X": X, "Q": Q, "K": K, "V": V, "A": A, "O": O, "mask": mask}
    return output
```

Note that every intermediate is cached. The cache is essential for the backward pass -- without it, we would need to re-run the entire forward pass during backpropagation.

### The Tricky Parts

**Why `.transpose(0, 2, 1)` and not just `.T`?** With a 3D tensor of shape $(B, n, d_k)$, `.T` would reverse all axes to $(d_k, n, B)$, which is wrong -- we want to transpose only the last two axes (the matrix dimensions) while keeping the batch dimension in place. `.transpose(0, 2, 1)` does exactly this.

**Why `np.float64` everywhere?** The implementation casts to float64 in the forward pass. This is not what production code does (production uses float16 or bfloat16), but for gradient checking with finite differences, float64 precision is necessary. The numerical gradient formula $(f(x+h) - f(x-h)) / 2h$ accumulates rounding errors that would cause gradient checks to fail in float32.

**Why cache `mask`?** The mask itself is not directly used in the backward pass implementation shown here, but it is stored for completeness and for potential use in debugging or more complex backward passes. The gradient through the mask is handled implicitly: softmax backward produces zero gradients for masked positions because the attention weights at those positions are already zero.

**Why `np.einsum` for weight gradients?** In the backward pass, computing $\frac{\partial \mathcal{L}}{\partial W_Q} = X^\top \cdot \frac{\partial \mathcal{L}}{\partial Q}$ requires summing over the batch dimension. The expression `np.einsum("bnd,bnk->dk", X, grad_Q)` performs the batched matrix multiplication and batch reduction in a single, readable operation. It says: "contract over batch ($b$) and sequence ($n$) dimensions, producing a result indexed by $d$ and $k$."

---

## Backward Pass -- The 6 Steps

The backward pass propagates the loss gradient through every operation in reverse order. Let $\frac{\partial \mathcal{L}}{\partial \text{out}}$ be the upstream gradient with shape $(B, n, d_{\text{model}})$.

### Step 1: Through Output Projection

Forward: $\text{out} = OW_O + b_O$

This is a linear layer, so the backward pass follows the standard matrix calculus rules:

$$\frac{\partial \mathcal{L}}{\partial O} = \frac{\partial \mathcal{L}}{\partial \text{out}} \cdot W_O^\top \quad \in \mathbb{R}^{B \times n \times d_v}$$

$$\frac{\partial \mathcal{L}}{\partial W_O} = O^\top \cdot \frac{\partial \mathcal{L}}{\partial \text{out}} \quad \in \mathbb{R}^{d_v \times d_{\text{model}}}$$

$$\frac{\partial \mathcal{L}}{\partial b_O} = \sum_{b,i} \frac{\partial \mathcal{L}}{\partial \text{out}} \quad \in \mathbb{R}^{d_{\text{model}}}$$

```python
grad_O = grad_output @ self.W_O.T
self.grad_W_O = np.einsum("biv,bio->vo", O, grad_output)
self.grad_b_O = grad_output.sum(axis=(0, 1))
```

The `einsum` notation `"biv,bio->vo"` sums over both batch ($b$) and sequence ($i$) dimensions, implementing the batched $O^\top \cdot \text{grad\_output}$.

### Step 2: Through Value Weighting

Forward: $O = AV$

This is a matrix multiplication, so:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial O} \cdot V^\top \quad \in \mathbb{R}^{B \times n \times n}$$

$$\frac{\partial \mathcal{L}}{\partial V} = A^\top \cdot \frac{\partial \mathcal{L}}{\partial O} \quad \in \mathbb{R}^{B \times n \times d_v}$$

```python
grad_A = grad_O @ V.transpose(0, 2, 1)
grad_V = A.transpose(0, 2, 1) @ grad_O
```

### Step 3: Through Softmax (the hard part)

Forward: $A = \text{softmax}(S)$ applied row-wise.

The Jacobian of softmax for a single row is:

$$\frac{\partial A_{i,j}}{\partial S_{i,k}} = A_{i,j}(\delta_{jk} - A_{i,k})$$

where $\delta_{jk}$ is the Kronecker delta (1 if $j = k$, 0 otherwise).

The vector-Jacobian product (what we actually need) is derived by contracting the upstream gradient $g = \frac{\partial \mathcal{L}}{\partial A}$ with this Jacobian:

$$\frac{\partial \mathcal{L}}{\partial S_{i,k}} = \sum_j g_{i,j} \cdot A_{i,j}(\delta_{jk} - A_{i,k})$$

$$= g_{i,k} \cdot A_{i,k} - A_{i,k} \sum_j g_{i,j} \cdot A_{i,j}$$

$$= A_{i,k}\left(g_{i,k} - \sum_j g_{i,j} A_{i,j}\right)$$

In matrix form:

$$\frac{\partial \mathcal{L}}{\partial S} = A \odot \left(\frac{\partial \mathcal{L}}{\partial A} - \text{rowsum}\left(\frac{\partial \mathcal{L}}{\partial A} \odot A\right)\right)$$

```python
def softmax_backward(grad_output, softmax_output):
    dot = np.sum(grad_output * softmax_output, axis=-1, keepdims=True)  # rowsum(g * A)
    return softmax_output * (grad_output - dot)                          # A * (g - rowsum)
```

**Why this formula is elegant:** We only need the cached softmax output $A$ (not the input scores $S$). The `keepdims=True` in the sum produces shape $(B, n, 1)$, which broadcasts against $(B, n, n)$ when subtracted from `grad_output`.

### Step 4: Through Masking and Scaling

The mask is additive ($S_{\text{masked}} = S_{\text{scaled}} + M$), so gradients pass through unchanged. The scaling ($S_{\text{scaled}} = S_{\text{raw}} / \sqrt{d_k}$) is element-wise, so:

$$\frac{\partial \mathcal{L}}{\partial S_{\text{raw}}} = \frac{1}{\sqrt{d_k}} \cdot \frac{\partial \mathcal{L}}{\partial S}$$

```python
grad_raw_scores = grad_scores / scale   # scale = np.sqrt(d_k)
```

### Step 5: Through $QK^\top$

Forward: $S_{\text{raw}} = QK^\top$

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial S_{\text{raw}}} \cdot K \quad \in \mathbb{R}^{B \times n \times d_k}$$

$$\frac{\partial \mathcal{L}}{\partial K} = \left(\frac{\partial \mathcal{L}}{\partial S_{\text{raw}}}\right)^\top \cdot Q \quad \in \mathbb{R}^{B \times n \times d_k}$$

```python
grad_Q = grad_raw_scores @ K
grad_K = grad_raw_scores.transpose(0, 2, 1) @ Q
```

The shape logic: $\frac{\partial \mathcal{L}}{\partial S_{\text{raw}}}$ is $(B, n, n)$, $K$ is $(B, n, d_k)$, so the product is $(B, n, d_k)$ -- exactly the shape of $Q$.

### Step 6: Through Q/K/V Projections

Forward: $Q = XW_Q + b_Q$, $K = XW_K + b_K$, $V = XW_V + b_V$

Weight gradients (summed over batch):

$$\frac{\partial \mathcal{L}}{\partial W_Q} = X^\top \cdot \frac{\partial \mathcal{L}}{\partial Q}, \quad \frac{\partial \mathcal{L}}{\partial W_K} = X^\top \cdot \frac{\partial \mathcal{L}}{\partial K}, \quad \frac{\partial \mathcal{L}}{\partial W_V} = X^\top \cdot \frac{\partial \mathcal{L}}{\partial V}$$

Input gradient (accumulated from all three paths):

$$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Q} \cdot W_Q^\top + \frac{\partial \mathcal{L}}{\partial K} \cdot W_K^\top + \frac{\partial \mathcal{L}}{\partial V} \cdot W_V^\top$$

```python
self.grad_W_Q = np.einsum("bnd,bnk->dk", X, grad_Q)
self.grad_W_K = np.einsum("bnd,bnk->dk", X, grad_K)
self.grad_W_V = np.einsum("bnd,bnv->dv", X, grad_V)

grad_X = grad_Q @ self.W_Q.T + grad_K @ self.W_K.T + grad_V @ self.W_V.T
```

**Why does $\frac{\partial \mathcal{L}}{\partial X}$ accumulate three terms?** Because $X$ is used three times in the forward pass -- once for each projection. By the multivariate chain rule, when a variable feeds into multiple branches of the computation graph, its total gradient is the sum of gradients from each branch. This is the most important structural insight of the backward pass.

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Why |
|-----------|------------|-----|
| Q/K/V Projections | $O(B \cdot n \cdot d_{\text{model}} \cdot d_k)$ | Three matrix multiplications of $(n, d_{\text{model}}) \times (d_{\text{model}}, d_k)$ |
| $QK^\top$ scores | $O(B \cdot n^2 \cdot d_k)$ | Every query dot-producted with every key |
| Softmax | $O(B \cdot n^2)$ | Exp, sum, divide for each of $n^2$ elements |
| $AV$ weighting | $O(B \cdot n^2 \cdot d_v)$ | Each output position weighted-sums all values |
| Output projection | $O(B \cdot n \cdot d_v \cdot d_{\text{model}})$ | One matrix multiplication |
| **Forward total** | $O(B \cdot n^2 \cdot d_k + B \cdot n \cdot d_{\text{model}} \cdot d_k)$ | Quadratic in $n$, linear in $d$ |
| **Backward total** | Same as forward | Each forward op has a corresponding backward op with matching complexity |

The quadratic $n^2$ terms ($QK^\top$ and $AV$) dominate for long sequences. For $n = 4096$ and $d_k = 64$: the attention core is $\sim 4 \times 10^9$ FLOPs, while projections are $\sim 4 \times 10^6$ FLOPs.

### Space Complexity

| What | Size | Why |
|------|------|-----|
| Attention matrix $A$ | $O(B \cdot n^2)$ | Every position pair has a weight |
| Q, K, V | $O(B \cdot n \cdot d_k)$ | Linear in sequence length |
| Forward cache (for backward) | $O(B \cdot n^2 + B \cdot n \cdot d_k)$ | Must store $A$, $Q$, $K$, $V$, $O$, $X$ |

**The attention matrix dominates.** For $n = 4096$, $B = 1$, float32: $A$ alone is $4096^2 \times 4 = 64$ MB per head. With 32 heads: 2 GB just for attention matrices. With 128 heads (as in some large models): 8 GB.

### The Bottleneck

The $O(n^2)$ attention matrix is the bottleneck in both time and space. It must be:
1. **Written** to memory when computing $QK^\top$
2. **Read** for scaling and masking
3. **Written** again after softmax
4. **Read** again for the $AV$ multiplication

Each of these memory round-trips is expensive on modern hardware where compute is fast but memory bandwidth is limited.

### FLOPs Counting in the Implementation

```python
def count_flops(batch_size, seq_len, d_model, d_k, d_v):
    B, n = batch_size, seq_len

    proj_q = 2 * B * n * d_model * d_k     # Q = X @ W_Q (multiply + accumulate = 2 FLOPs per element)
    proj_k = 2 * B * n * d_model * d_k
    proj_v = 2 * B * n * d_model * d_v
    proj_o = 2 * B * n * d_v * d_model

    qk = 2 * B * n * n * d_k               # S = Q @ K^T
    av = 2 * B * n * n * d_v               # O = A @ V

    sm = 5 * B * n * n                      # exp + sum + div ~ 5 ops per element

    return proj_q + proj_k + proj_v + proj_o + qk + av + sm
```

The factor of 2 in matrix multiplication FLOPs comes from counting each multiply-accumulate as two operations (one multiply, one add). The factor of 5 for softmax accounts for: subtract max, exponentiate, sum, divide, and one auxiliary operation per element.

---

## Common Pitfalls

### Pitfall 1: Wrong Transpose Axis

**The mistake:**
```python
# Wrong: .T reverses ALL axes for 3D tensors
scores = Q @ K.T
```

**Why it is wrong:** For a 3D tensor of shape $(B, n, d_k)$, `.T` produces shape $(d_k, n, B)$. The batch dimension ends up in the wrong position, and the matmul produces garbage (or a shape error if you are lucky).

**The fix:**
```python
# Correct: transpose only the last two axes
scores = Q @ K.transpose(0, 2, 1)   # (B, n, d_k) @ (B, d_k, n) -> (B, n, n)
```

### Pitfall 2: Forgetting the Scaling Factor

**The mistake:**
```python
# Wrong: no scaling
scores = Q @ K.transpose(0, 2, 1)
A = softmax(scores)
```

**Why it is wrong:** Without dividing by $\sqrt{d_k}$, the dot products have variance $d_k$. For $d_k = 64$, scores have standard deviation 8, pushing softmax into saturation with near-binary outputs and vanishing gradients. The model will train extremely slowly or not at all.

**The fix:**
```python
# Correct: scale by sqrt(d_k)
scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
A = softmax(scores)
```

### Pitfall 3: Applying Mask After Softmax

**The mistake:**
```python
# Wrong: mask after softmax
A = softmax(scores)
A = A * (1 - upper_triangular_mask)   # zero out future positions
```

**Why it is wrong:** After multiplying, the rows of $A$ no longer sum to 1. The attention weights are no longer a valid probability distribution. This means the output at each position is artificially scaled down -- positions early in the sequence (which have few unmasked entries) get much weaker outputs than positions later in the sequence.

**The fix:**
```python
# Correct: mask before softmax
scores = scores + causal_mask           # -inf for future positions
A = softmax(scores)                     # e^(-inf) = 0, rows still sum to 1
```

### Pitfall 4: Forgetting to Accumulate Input Gradients

**The mistake:**
```python
# Wrong: only use gradient from Q path
grad_X = grad_Q @ self.W_Q.T
```

**Why it is wrong:** The input $X$ is used three times: for $Q$, $K$, and $V$ projections. The multivariate chain rule requires summing the gradient contributions from all three paths. Using only one path loses two-thirds of the gradient signal.

**The fix:**
```python
# Correct: accumulate from all three paths
grad_X = grad_Q @ self.W_Q.T + grad_K @ self.W_K.T + grad_V @ self.W_V.T
```

### Pitfall 5: Naive Softmax Without the Max Trick

**The mistake:**
```python
# Wrong: naive softmax overflows for large inputs
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

**Why it is wrong:** `np.exp(800)` overflows to `inf` in float64 (and `np.exp(89)` overflows in float32). The attention scores can easily reach these magnitudes, especially early in training or with large inputs. The result is `inf / inf = NaN`.

**The fix:**
```python
# Correct: subtract max before exp (mathematically equivalent, numerically stable)
def softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

This works because $\text{softmax}(x - c) = \text{softmax}(x)$ for any constant $c$. Subtracting the row-wise maximum ensures the largest exponent is $e^0 = 1$.

---

## The $O(n^2)$ Problem

### Why Attention is Quadratic

The attention matrix $A \in \mathbb{R}^{n \times n}$ has $n^2$ elements because every query position computes a score against every key position. There is no way around this in the naive algorithm -- the softmax normalization requires knowing all scores in a row before any output can be computed.

```
Sequence length:  128     512    2048    4096     8192
Attention matrix: 16K    262K    4.2M    16.8M   67.1M   elements per head
Memory (FP32):    64 KB   1 MB   16 MB   64 MB   256 MB  per head
With 32 heads:    2 MB   32 MB  512 MB   2 GB     8 GB
```

This quadratic scaling is why transformer context lengths were historically limited to 512 or 2048 tokens. Doubling the sequence length quadruples the memory and compute for the attention portion.

### Memory is the Bottleneck, Not Compute

On modern GPUs, the arithmetic intensity of attention (FLOPs per byte of memory traffic) is below the hardware's "ridge point." This means the GPU's compute units are waiting for data to arrive from memory, not the other way around.

For $n = 4096$, $d_k = 64$:

$$\text{Arithmetic intensity} = \frac{n(4d_k + 5)}{4(4d_k + n)} \approx \frac{4096 \times 261}{4 \times 4352} \approx 61 \text{ FLOPs/byte}$$

An A100 GPU's ridge point is roughly 156 FLOPs/byte. Self-attention at this sequence length operates at 61 FLOPs/byte -- firmly in the **memory-bound** regime. Making attention faster requires reducing memory traffic, not reducing FLOPs.

### The Memory Access Pattern Problem

Here is the naive algorithm's memory access pattern:

```
HBM (slow, large)                  SRAM (fast, small)
+-------------------+
| Q (B, n, d_k)     |----read---->  Compute Q @ K^T
| K (B, n, d_k)     |----read---->
|                   |<---write----  S (B, n, n)  <-- FULL matrix written to HBM
+-------------------+
| S (B, n, n)       |----read---->  S / sqrt(d_k)
|                   |<---write----  S_scaled     <-- read and write
+-------------------+
| S_scaled          |----read---->  S + mask
|                   |<---write----  S_masked     <-- read and write
+-------------------+
| S_masked          |----read---->  softmax(S)
|                   |<---write----  A (B, n, n)  <-- read and write
+-------------------+
| A (B, n, n)       |----read---->  A @ V
| V (B, n, d_v)     |----read---->
|                   |<---write----  O (B, n, d_v)
+-------------------+
```

The attention matrix $(B, n, n)$ is read and written **four times** between HBM and SRAM. For $n = 4096$ with 32 heads in float32, that is $4 \times 2$ GB $= 8$ GB of memory traffic just for intermediate attention matrices.

---

## Connection to Flash Attention

### What Gets Optimized

Flash Attention (Dao et al., 2022) observes that the naive algorithm is memory-bound and that the entire $n \times n$ attention matrix never needs to exist in memory at once. The key insight: we can compute attention in **tiles**, processing small blocks of rows and columns at a time, keeping everything in fast SRAM.

### The Core Idea

Instead of:
1. Compute ALL scores (write $n \times n$ to HBM)
2. Apply softmax to ALL scores (read and write $n \times n$)
3. Multiply ALL weights by $V$ (read $n \times n$ from HBM)

Flash Attention does:
1. For each tile of $Q$ and each tile of $K$, $V$:
   - Compute a tile of scores in SRAM
   - Update a running softmax using the **online softmax** algorithm
   - Accumulate the partial output
   - Never write the attention tile back to HBM

### From Naive to Optimized

| Aspect | Naive (what we implemented) | Flash Attention |
|--------|---------------------------|-----------------|
| Attention matrix | Fully materialized in HBM | Never materialized; computed in tiles in SRAM |
| Memory | $O(n^2)$ | $O(n)$ |
| HBM reads/writes | $O(n^2)$ (multiple passes) | $O(n^2 d / M)$ where $M$ is SRAM size |
| FLOPs | Same | Same (it is not a FLOPs optimization) |
| Softmax | Standard (needs full row) | Online softmax (incremental, no full row needed) |
| Speedup | Baseline | 2-4x wall-clock on A100 |

The critical prerequisite for understanding Flash Attention is understanding *exactly* why the naive version is slow -- and that means understanding the memory access pattern we implemented. Every read and write of the $(B, n, n)$ attention matrix in our implementation is a round-trip to slow HBM that Flash Attention eliminates.

### Why Understanding Naive Attention First is Essential

You cannot appreciate what Flash Attention optimizes without first understanding:
1. That the attention matrix is $O(n^2)$ and dominates memory
2. That it is read/written multiple times (score, scale, mask, softmax, multiply)
3. That softmax requires the full row to normalize (which is why tiling is hard)
4. That the operation is memory-bound, not compute-bound (so reducing memory traffic is the key)

Our implementation makes all of this explicit and measurable through `count_flops` and `count_memory_bytes`.

---

## Testing Your Understanding

### Quick Checks

1. **What would happen if we removed the scaling factor?** The dot products would have variance $d_k$ instead of 1. For $d_k = 64$, scores would have standard deviation 8, causing softmax to produce near-one-hot distributions with vanishing gradients. Training would stall.

2. **Why do we need three separate projections instead of just using $X$ for everything?** Separate projections allow the model to learn different representations for "what I am looking for" (query), "what I advertise" (key), and "what I provide" (value). Using $X$ directly forces a single vector to serve all three roles.

3. **What is the output shape if input is $(4, 128, 512)$ with $d_k = 64$, $d_v = 64$?** The attention output $O$ is $(4, 128, 64)$, and after the output projection $W_O \in \mathbb{R}^{64 \times 512}$, the final output is $(4, 128, 512)$.

4. **Why does the input gradient $\frac{\partial \mathcal{L}}{\partial X}$ have three terms?** Because $X$ is used three times in the forward pass (once each for $Q$, $K$, $V$ projections). The multivariate chain rule says the total gradient is the sum of partial gradients from each usage.

5. **Why is the causal mask added before softmax rather than applied as a multiplicative mask after softmax?** Multiplying attention weights by zero after softmax would break normalization -- rows would no longer sum to 1. Adding $-\infty$ before softmax lets $e^{-\infty} = 0$ naturally exclude future positions while maintaining a valid probability distribution.

### Exercises

1. **Easy**: Modify `scaled_dot_product_attention` to return the pre-softmax scores as well (useful for debugging). Verify that rows of scores before masking have variance approximately equal to 1 (after scaling) for random inputs.

2. **Medium**: Implement a `combined_mask` function that takes both a causal mask and a padding mask and produces a single additive mask. Test it on a batch where different sequences have different lengths.

3. **Hard**: Implement a simplified version of the online softmax algorithm. Instead of computing all scores, applying softmax, and then multiplying by $V$, process the keys one at a time, maintaining running statistics for the softmax normalization. Verify the output matches our standard implementation.

---

## Summary

### Key Takeaways

- Self-attention computes a weighted sum of value vectors where weights come from query-key dot products. This enables every position to directly access information from every other position.

- The scaling factor $1/\sqrt{d_k}$ is not optional -- without it, softmax saturates for any reasonable $d_k$, killing gradients and preventing learning.

- The $O(n^2)$ attention matrix is the central bottleneck of transformer inference. It dominates both memory and compute for long sequences, and every major inference optimization (Flash Attention, KV caching, GQA/MQA) exists to address this.

- The backward pass requires caching the full attention matrix $A$ (among other intermediates), which doubles the memory cost during training. This is exactly what memory-efficient attention algorithms avoid.

- Understanding the naive implementation completely -- especially its memory access patterns -- is a prerequisite for understanding why Flash Attention works and why it provides speedups despite performing the same number of FLOPs.

### Quick Reference

```
Self-Attention (single head)
├── Forward:  O(B * n^2 * d_k)  -- QK^T and AV dominate
├── Backward: O(B * n^2 * d_k)  -- symmetric with forward
├── Memory:   O(B * n^2)        -- attention matrix dominates
│
├── Parameters: W_Q, W_K, W_V (d_model, d_k), W_O (d_v, d_model)
├── Cache:      X, Q, K, V, A, O  (A at B*n*n is the big one)
│
├── Key formula:  Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
├── Scaling:      Normalizes dot-product variance from d_k to 1
├── Causal mask:  -inf above diagonal, added before softmax
│
└── Optimized by: Flash Attention (tiled SRAM computation, O(n) memory)
                  KV Cache (avoid recomputing K,V for past tokens)
                  GQA/MQA (share K,V heads across query heads)
```
