# Positional Encoding -- Deep Dive

## The Intuition

### What Problem Are We Solving?

Consider these two sentences:

- "The dog bit the man."
- "The man bit the dog."

Same words, completely different meanings. The difference is *order*. Now consider how self-attention processes these sentences. For each token, attention computes pairwise similarity scores with all other tokens via $QK^\top$. If the same token embeddings are used, swapping the order of tokens simply produces the same computation in a different order -- the outputs get permuted, but no new information is created. Formally, if you apply a permutation matrix $P$ to the input, the attention output is just $P$ applied to the original output. Self-attention is **permutation-equivariant**: it treats its input as a *set*, not a *sequence*.

This is not a bug -- it is what makes attention parallelizable (unlike RNNs, which must process tokens sequentially). But language has order, so we must inject positional information from outside.

### The Key Insight

Add a unique, position-dependent vector to each token embedding *before* it enters the attention layers. Position 0 gets one vector, position 1 gets a different vector, position 2 gets yet another, and so on. These vectors are constructed so that:

1. Every position gets a **unique** encoding (the model can distinguish positions).
2. The encoding captures **relative distance** -- the mathematical relationship between the vectors at position $p$ and position $p + k$ depends only on $k$, not on $p$.
3. The values are **bounded** (they do not blow up the scale of the embeddings they are added to).

The original Transformer achieves all three properties using sine and cosine functions at geometrically spaced frequencies. Each dimension of the encoding oscillates at a different rate: low dimensions change rapidly from position to position (capturing fine-grained local position), while high dimensions change slowly (capturing coarse global position). Together, they form a unique "fingerprint" for each position.

### Real-World Analogy

Think of how a clock tells time. A clock has multiple hands: the second hand spins fast (one full rotation per minute), the minute hand spins slower (one rotation per hour), and the hour hand spins slowest (one rotation per 12 hours). No two moments have the same configuration of all three hands, even though each individual hand revisits the same positions. The combination of oscillations at different frequencies creates a unique encoding for every point in time.

Sinusoidal positional encoding works the same way. Each pair of dimensions is like one clock hand: low-index dimensions spin fast (high frequency), high-index dimensions spin slow (low frequency). The combination of all these "hands" uniquely identifies every position in the sequence.

---

## The Math, Step by Step

### Why Attention Needs Positional Information

Self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$. If we permute the input by applying a permutation matrix $P$:

$$Q' = PXW_Q = PQ, \quad K' = PK, \quad V' = PV$$

$$Q'K'^\top = PQK^\top P^\top$$

The attention output becomes $P \cdot \text{Attention}(Q, K, V)$ -- the same result, just permuted. The model treats "dog bit man" identically to "man bit dog" (same tokens, permuted). Positional encoding breaks this symmetry:

$$X' = X + PE$$

Now $Q = (X + PE)W_Q$ depends on position. Different orderings produce different $Q$, $K$, $V$, and therefore different attention patterns.

### Building Up to the Sinusoidal Formula

**Step 1 -- The simplest idea: integer position.**

We could just add the position index itself: $PE_{pos} = pos$. But this fails immediately -- position 1000 would add 1000 to the embedding values, completely dominating the content information. We need bounded values.

**Step 2 -- A single sine wave.**

Use $PE_{pos} = \sin(pos)$. This is bounded in $[-1, 1]$, but it repeats every $2\pi \approx 6.3$ positions, so positions 0 and 6 get nearly the same encoding. A single frequency cannot distinguish all positions.

**Step 3 -- Multiple frequencies (the full solution).**

Use many sine and cosine waves at different frequencies, each assigned to a different dimension of the encoding vector. Low frequencies distinguish distant positions; high frequencies distinguish nearby positions. The combination is unique for every position.

### The Core Equations

For position $pos \in \{0, 1, \ldots, L-1\}$ and dimension pair index $i \in \{0, 1, \ldots, d_{model}/2 - 1\}$:

$$PE_{(pos, 2i)} = \sin(pos \cdot \omega_i)$$

$$PE_{(pos, 2i+1)} = \cos(pos \cdot \omega_i)$$

where the frequency for each dimension pair is:

$$\omega_i = \frac{1}{10000^{2i/d_{model}}}$$

The frequencies form a geometric progression from $\omega_0 = 1$ (fastest) to $\omega_{d_{model}/2 - 1} \approx 1/10000$ (slowest).

**Shapes:**
- $pos \in \mathbb{R}^{L \times 1}$ -- position indices as a column vector
- $\omega \in \mathbb{R}^{1 \times d_{model}/2}$ -- frequencies as a row vector
- $\theta = pos \cdot \omega \in \mathbb{R}^{L \times d_{model}/2}$ -- angle matrix (outer product)
- $PE \in \mathbb{R}^{L \times d_{model}}$ -- final encoding with sin/cos interleaved

### Log-Space Computation

Computing $10000^{2i/d_{model}}$ directly can overflow for large $d_{model}$. The implementation uses the mathematically equivalent log-space form:

$$\omega_i = \exp\!\left(-\frac{2i}{d_{model}} \cdot \ln(10000)\right)$$

This is numerically identical because $\exp(-x \ln(a)) = a^{-x} = 1/a^x$, but avoids computing $10000^{2i/d_{model}}$ as an intermediate value.

---

## Worked Example: $d_{model} = 4$, $L = 3$

Let us compute the full PE matrix by hand.

### Step 1: Compute Frequencies

With $d_{model} = 4$, we have $d_{model}/2 = 2$ dimension pairs, so two frequencies:

$$\omega_0 = 10000^{-0/4} = 10000^0 = 1.0$$

$$\omega_1 = 10000^{-2/4} = 10000^{-0.5} = \frac{1}{\sqrt{10000}} = \frac{1}{100} = 0.01$$

### Step 2: Compute Angle Matrix

$$\theta = \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix} \times \begin{bmatrix} 1.0 & 0.01 \end{bmatrix} = \begin{bmatrix} 0.0 & 0.00 \\ 1.0 & 0.01 \\ 2.0 & 0.02 \end{bmatrix}$$

### Step 3: Apply Sin and Cos, Interleave

$$PE = \begin{bmatrix} \sin(0) & \cos(0) & \sin(0) & \cos(0) \\ \sin(1) & \cos(1) & \sin(0.01) & \cos(0.01) \\ \sin(2) & \cos(2) & \sin(0.02) & \cos(0.02) \end{bmatrix}$$

Computing the values:

$$PE = \begin{bmatrix} 0.0000 & 1.0000 & 0.0000 & 1.0000 \\ 0.8415 & 0.5403 & 0.0100 & 0.99995 \\ 0.9093 & -0.4161 & 0.0200 & 0.9998 \end{bmatrix}$$

### What to Notice

**Position 0** is always $[0, 1, 0, 1, \ldots, 0, 1]$ because $\sin(0) = 0$ and $\cos(0) = 1$ for every frequency. The test suite verifies this:

```python
def test_position_zero(self):
    pe = sinusoidal_positional_encoding(1, 64)
    np.testing.assert_allclose(pe[0, 0::2], np.zeros(32), atol=1e-15)   # sin(0) = 0
    np.testing.assert_allclose(pe[0, 1::2], np.ones(32), atol=1e-15)    # cos(0) = 1
```

**Columns 0-1** (frequency $\omega_0 = 1.0$) change rapidly: $0.0 \to 0.84 \to 0.91$ for the sin column, $1.0 \to 0.54 \to -0.42$ for the cos column. These high-frequency dimensions capture fine-grained position.

**Columns 2-3** (frequency $\omega_1 = 0.01$) change slowly: $0.0 \to 0.01 \to 0.02$ for sin, $1.0 \to 0.99995 \to 0.9998$ for cos. These low-frequency dimensions barely move between adjacent positions -- they only distinguish positions that are far apart.

### Visualizing the Frequency Structure

```
Position     dim 0 (sin, w=1.0)     dim 2 (sin, w=0.01)
  0          |                      |
  1          |========              |
  2          |=========             |
  3          |=====                 |
  4          |                      |
  5          |                      |=
  6          |                      |=
  ...        (period ~ 6.3)         (period ~ 628)
             ^^ fast oscillation    ^^ barely changes
```

```
Frequency spectrum across dimension pairs:

dim pair 0:  omega = 1.0       wavelength = 2*pi ~ 6.3 positions
dim pair 1:  omega = 0.01      wavelength = 628 positions
   ...
dim pair d/2-1: omega ~ 0.0001  wavelength ~ 62,832 positions

Low dims -----> fast oscillation, local position info
High dims ----> slow oscillation, global position info
```

### Norm Verification

Each row of PE has norm $\sqrt{d_{model}/2}$. For $d_{model} = 4$:

$$\|PE_0\|_2 = \sqrt{0^2 + 1^2 + 0^2 + 1^2} = \sqrt{2}$$

This holds for every position because each (sin, cos) pair contributes $\sin^2(\theta) + \cos^2(\theta) = 1$, and there are $d_{model}/2$ pairs, giving $\|PE_{pos}\|_2 = \sqrt{d_{model}/2}$.

---

## The Relative Position Property

### Why It Matters

The most mathematically elegant property of sinusoidal encodings: for any fixed offset $k$, there exists a **linear transformation** $M_k$ that maps $PE_{pos}$ to $PE_{pos+k}$, regardless of $pos$. This means the model can learn to detect relative position through linear operations in its attention weights.

### The Rotation Matrix Derivation

For a single frequency $\omega_i$, we want to express $PE_{pos+k}$ in terms of $PE_{pos}$. Using the angle addition identities:

$$\sin(\omega_i(pos + k)) = \sin(\omega_i \cdot pos)\cos(\omega_i k) + \cos(\omega_i \cdot pos)\sin(\omega_i k)$$

$$\cos(\omega_i(pos + k)) = \cos(\omega_i \cdot pos)\cos(\omega_i k) - \sin(\omega_i \cdot pos)\sin(\omega_i k)$$

In matrix form:

$$\begin{bmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{bmatrix} = \underbrace{\begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix}}_{R_i(k)} \begin{bmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{bmatrix}$$

This is a **2D rotation matrix** that depends only on $k$ and $\omega_i$, not on $pos$. The full transformation $M_k$ is block-diagonal with $d_{model}/2$ such rotation blocks:

$$M_k = \text{diag}(R_0(k), R_1(k), \ldots, R_{d_{model}/2-1}(k))$$

### Visualizing the Block-Diagonal Structure

```
M_k (d_model x d_model):

+-------+-------+-------+-------+-------+
| R_0(k)|   0   |   0   |  ...  |   0   |
+-------+-------+-------+-------+-------+
|   0   | R_1(k)|   0   |  ...  |   0   |
+-------+-------+-------+-------+-------+
|   0   |   0   | R_2(k)|  ...  |   0   |
+-------+-------+-------+-------+-------+
|  ...  |  ...  |  ...  |  ...  |  ...  |
+-------+-------+-------+-------+-------+
|   0   |   0   |   0   |  ...  |R_{d/2-1}(k)|
+-------+-------+-------+-------+-------+

Each R_i(k) is a 2x2 rotation:
[ cos(w_i*k)   sin(w_i*k) ]
[ -sin(w_i*k)  cos(w_i*k) ]
```

### Numerical Verification

The implementation verifies this property directly. For $L = 100$, $d_{model} = 64$, and offset $k$:

```python
def relative_position_matrix(pe, offset):
    # Build M_k from the known rotation formula
    for i in range(d_half):
        omega_i = np.exp(-2.0 * i / d_model * np.log(10000.0))
        cos_val = np.cos(omega_i * offset)
        sin_val = np.sin(omega_i * offset)
        M_k[2*i, 2*i] = cos_val
        M_k[2*i, 2*i+1] = sin_val
        M_k[2*i+1, 2*i] = -sin_val
        M_k[2*i+1, 2*i+1] = cos_val

    # Verify: M_k @ PE[pos] == PE[pos + k] for ALL valid positions
    for pos in range(L - offset):
        error = np.linalg.norm(M_k @ pe[pos] - pe[pos + offset])
        # error < 1e-10 for all positions
```

The test suite checks offsets $k = 1, 5, 10, 50$ and confirms reconstruction error below $10^{-10}$ for every position.

### Connection to RoPE

This rotation property is the mathematical precursor to RoPE (Rotary Position Embeddings, Topic 13). The key difference:

- **Sinusoidal PE**: Adds the rotation to the input. The model must *learn* to use the rotation structure through its attention weights.
- **RoPE**: Applies the rotation *directly* to $Q$ and $K$ inside the attention computation, so that the dot product $q_i^\top k_j$ depends only on relative distance $i - j$ by construction.

Understanding the rotation matrix here is essential for understanding why RoPE works.

---

## Dot Product Distance Properties

### Distance-Dependent Dot Products

When sinusoidal encodings are used, the dot product between two PE vectors depends only on the position *difference*, not the absolute positions:

$$PE_{p_1}^\top PE_{p_2} = \sum_{i=0}^{d_{model}/2 - 1} \left[\sin(\omega_i p_1)\sin(\omega_i p_2) + \cos(\omega_i p_1)\cos(\omega_i p_2)\right]$$

Using the product-to-sum identity $\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$:

$$PE_{p_1}^\top PE_{p_2} = \sum_{i=0}^{d_{model}/2 - 1} \cos(\omega_i(p_1 - p_2))$$

This depends only on $p_1 - p_2$. The test verifies:

```python
def test_distance_dependent(self):
    pe = sinusoidal_positional_encoding(100, 64)
    D = dot_product_distance(pe)   # D = pe @ pe.T
    for k in [1, 3, 5, 10]:
        # D[0, k] should equal D[10, 10+k] -- same offset k
        np.testing.assert_allclose(D[0, k], D[10, 10 + k], atol=1e-10)
```

### Self-Dot Product

The diagonal of the dot product matrix is $PE_{pos}^\top PE_{pos} = d_{model}/2$ for all positions (since $\sin^2 + \cos^2 = 1$ for each of the $d_{model}/2$ dimension pairs).

### Toeplitz-Like Structure

The dot product matrix $D$ has approximate Toeplitz structure: the value at $(i, j)$ depends mainly on $|i - j|$. This means the PE vectors are organized so that the "similarity" between positions is a function of distance, which is exactly what the model needs to learn position-dependent attention patterns.

```
D (dot product matrix, L=8, d_model=64):

          pos 0   pos 1   pos 2   pos 3   ...
pos 0   [ 32.0    31.2    29.1    26.0   ... ]
pos 1   [ 31.2    32.0    31.2    29.1   ... ]
pos 2   [ 29.1    31.2    32.0    31.2   ... ]
pos 3   [ 26.0    29.1    31.2    32.0   ... ]
  ...      ...     ...     ...     ...

Each diagonal has the same value (Toeplitz property).
Diagonal = d_model/2 = 32. Off-diagonals decrease with distance.
```

---

## From Math to Code

### The Data Structures

The implementation has three main components:

1. **`sinusoidal_positional_encoding(seq_len, d_model)`** -- Standalone function that generates the PE matrix.
2. **`SinusoidalPositionalEncoding`** -- Class that precomputes and caches the PE matrix, with a `forward` method that adds it to input embeddings.
3. **`LearnedPositionalEncoding`** -- Trainable embedding table with forward and backward passes.

### Implementation Walkthrough: Sinusoidal Encoding

```python
def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    # (d_model/2,)
    i = np.arange(d_model // 2, dtype=np.float64)
    omega = np.exp(-2.0 * i / d_model * np.log(10000.0))

    # (L, 1) * (1, d_model/2) -> (L, d_model/2)
    pos = np.arange(seq_len, dtype=np.float64)[:, np.newaxis]
    angles = pos * omega[np.newaxis, :]

    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe
```

**Line-by-line:**

- `i = np.arange(d_model // 2)`: Creates the dimension pair indices $[0, 1, 2, \ldots, d_{model}/2 - 1]$. Each index $i$ corresponds to one (sin, cos) pair occupying columns $2i$ and $2i+1$.

- `omega = np.exp(-2.0 * i / d_model * np.log(10000.0))`: Computes frequencies in log-space. This is equivalent to $\omega_i = 1/10000^{2i/d_{model}}$ but avoids computing $10000^{large}$ as an intermediate. Shape: $(d_{model}/2,)$.

- `pos = np.arange(seq_len)[:, np.newaxis]`: Creates position indices as a column vector of shape $(L, 1)$. The `[:, np.newaxis]` adds a dimension for broadcasting.

- `angles = pos * omega[np.newaxis, :]`: Outer product of positions and frequencies. Broadcasting: $(L, 1) \times (1, d_{model}/2) \to (L, d_{model}/2)$. Each element is $pos \times \omega_i$.

- `pe[:, 0::2] = np.sin(angles)`: Fills even columns (0, 2, 4, ...) with sine values.

- `pe[:, 1::2] = np.cos(angles)`: Fills odd columns (1, 3, 5, ...) with cosine values.

The interleaving means column 0 is $\sin(\omega_0 \cdot pos)$, column 1 is $\cos(\omega_0 \cdot pos)$, column 2 is $\sin(\omega_1 \cdot pos)$, column 3 is $\cos(\omega_1 \cdot pos)$, and so on. This matches the original Transformer paper.

### Implementation Walkthrough: SinusoidalPositionalEncoding Class

```python
class SinusoidalPositionalEncoding:
    def __init__(self, max_seq_len: int, d_model: int):
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pe = sinusoidal_positional_encoding(max_seq_len, d_model)

    def forward(self, X: np.ndarray) -> np.ndarray:
        L = X.shape[1]
        return X + self.pe[:L, :]

    def get_encoding(self, seq_len: int) -> np.ndarray:
        return self.pe[:seq_len, :]
```

**Key design decisions:**

- **Precomputation**: The PE matrix is computed once in `__init__` and reused. No redundant sin/cos calls during forward passes.

- **Slicing**: `self.pe[:L, :]` selects the first $L$ rows when the input sequence is shorter than `max_seq_len`. This means you can allocate a large PE buffer once and handle variable-length inputs.

- **Broadcasting in `X + self.pe[:L, :]`**: $X$ has shape $(B, L, d_{model})$ and `self.pe[:L, :]` has shape $(L, d_{model})$. NumPy broadcasts the PE across the batch dimension automatically: the same positional encoding is added to every item in the batch.

- **No backward pass**: Sinusoidal encodings are fixed (non-trainable). There are no parameters, so no gradients to compute.

### Implementation Walkthrough: Learned Positional Encoding

```python
class LearnedPositionalEncoding:
    def __init__(self, max_seq_len: int, d_model: int):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = np.random.randn(max_seq_len, d_model).astype(np.float64) * 0.02
        self._cache: dict = {}
        self.grad_embedding: np.ndarray = np.zeros_like(self.embedding)

    def forward(self, X: np.ndarray) -> np.ndarray:
        L = X.shape[1]
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}"
            )
        self._cache = {"X": X, "L": L}
        return X + self.embedding[:L, :]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        L = self._cache["L"]
        self.grad_embedding = np.zeros_like(self.embedding)
        self.grad_embedding[:L, :] = grad_output.sum(axis=0)
        return grad_output.copy()
```

**Line-by-line for the backward pass:**

- `self.grad_embedding = np.zeros_like(self.embedding)`: Initializes gradient for the full embedding table $(L_{max}, d_{model})$ to zeros.

- `self.grad_embedding[:L, :] = grad_output.sum(axis=0)`: The forward pass was $X' = X + PE[:L, :]$. Since $PE[:L, :]$ has shape $(L, d_{model})$ and is broadcast over the batch dimension $B$, the gradient with respect to $PE[:L, :]$ is the sum of the upstream gradients across the batch: $\sum_{b=0}^{B-1} \frac{\partial \mathcal{L}}{\partial X'_b}$. The `sum(axis=0)` collapses the batch dimension. Positions $L$ through $L_{max}-1$ keep their zero gradients since they were not used.

- `return grad_output.copy()`: The gradient with respect to $X$ is the upstream gradient passed through unchanged, because $\frac{\partial(X + PE)}{\partial X} = I$ (addition is a pass-through). The `.copy()` ensures the returned array is independent of `grad_output`.

### The Tricky Parts

**Why `d_model` must be even.** Each dimension pair (sin, cos) needs exactly two columns. With odd $d_{model}$, one dimension would be unpaired, and the interleaving `pe[:, 0::2]` and `pe[:, 1::2]` would produce arrays of different lengths.

**Why `float64` throughout.** The test suite verifies reconstruction errors below $10^{-10}$. Using float32 would introduce rounding errors on the order of $10^{-7}$, which would fail the relative position property tests.

**Why `self.pe[:L, :]` and not `self.pe[:L]`.** Both work in NumPy, but the explicit `:` in the second dimension makes the 2D nature of the slice clearer. This is a readability choice, not a functional one.

---

## Sinusoidal vs Learned Encodings

### The Tradeoffs

| Property | Sinusoidal | Learned |
|----------|------------|---------|
| Parameters | 0 (fixed, deterministic) | $L_{max} \times d_{model}$ (trainable) |
| Extrapolation | Can generate PE for any position | Cannot exceed $L_{max}$ |
| Relative position | Dot product depends on $p_1 - p_2$ | No structural guarantee |
| Expressiveness | Constrained to sinusoidal patterns | Can learn arbitrary patterns |
| Computation | Precompute sin/cos once | Table lookup (index select) |
| Used in | Original Transformer (2017) | BERT (2018), GPT-2 (2019) |
| Backward pass | None (frozen) | Sum of upstream gradients across batch |

### When to Use Which

**Sinusoidal** is the right choice when:
- You need to handle sequences longer than anything seen during training (extrapolation).
- You want zero additional parameters.
- The mathematical guarantees (bounded values, distance-dependent dot products, rotation property) matter for your application.

**Learned** is the right choice when:
- You have a fixed maximum sequence length and enough training data.
- You want maximum expressiveness -- the model can learn position-dependent patterns that sinusoids cannot capture.
- Empirically, learned embeddings often slightly outperform sinusoidal at the cost of $L_{max} \times d_{model}$ extra parameters.

### Why Modern LLMs Use Neither

Both sinusoidal and learned encodings are *absolute* -- they encode the position itself, not the distance between positions. Position 5 always gets the same encoding regardless of context. This creates two problems:

1. **Extrapolation**: Learned embeddings fail entirely beyond $L_{max}$. Sinusoidal encodings produce valid vectors but attention patterns trained on positions 0-2048 may not generalize to position 5000.

2. **Absolute vs relative information**: For attention, what matters is "how far apart are these tokens?" not "what is the absolute position of this token?" Absolute encodings mix both concerns.

RoPE (Topic 13) solves both by rotating $Q$ and $K$ vectors by position-dependent angles so that the dot product $q_i^\top k_j$ depends only on relative distance $i - j$. The progression:

```
2017  Sinusoidal absolute     Good math properties, limited extrapolation
2018  Learned absolute        More expressive, no extrapolation
2023+ RoPE (relative)         Direct relative position, better extrapolation
```

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Why |
|-----------|------------|-----|
| Generate PE matrix | $O(L \times d_{model})$ | Compute sin/cos for each of $L \times d_{model}/2$ angle values |
| Forward (sinusoidal) | $O(B \times L \times d_{model})$ | Elementwise addition with broadcasting |
| Forward (learned) | $O(B \times L \times d_{model})$ | Same: elementwise addition |
| Backward (learned) | $O(B \times L \times d_{model})$ | Sum across batch dimension |

### Space Complexity

- **Sinusoidal PE buffer**: $O(L_{max} \times d_{model})$ -- precomputed once, read-only.
- **Learned embedding table**: $O(L_{max} \times d_{model})$ -- trainable parameter.
- **Working memory during forward**: $O(B \times L \times d_{model})$ -- the output tensor.

### The Bottleneck (Hint: It Is Not Here)

Positional encoding is computationally trivial compared to attention and feed-forward layers. For $L = 4096$, $d_{model} = 4096$, FP16:

- **PE addition**: $4096 \times 4096 \times 2 = 32$ MB read + 32 MB write. One memory-bound kernel.
- **Attention core** ($QK^\top$): $O(L^2 d_{model})$ FLOPs $\approx 10^{11}$.
- **FFN layer**: $O(L \times d_{model}^2)$ FLOPs $\approx 10^{11}$.

PE is roughly 1000x cheaper than a single attention layer. In optimized inference engines, the PE addition is typically fused with the token embedding lookup into a single kernel:

```
token_ids --> embedding_table[token_ids] + PE[:L] --> X'
```

One kernel, one memory pass, negligible cost.

---

## Common Pitfalls

### Pitfall 1: Computing Frequencies Without Log-Space

**The mistake:**

```python
# Wrong: direct computation overflows for large d_model
omega = 1.0 / (10000.0 ** (2.0 * i / d_model))
```

**Why it is problematic:** For $d_{model} = 4096$ and $i$ near $d_{model}/2$, the term $10000^{2i/d_{model}}$ can be as large as $10000^{1.0} = 10000$. This is fine in float64, but in float32 with larger bases or dimensions, intermediate overflow is possible. More importantly, the log-space version is the standard implementation pattern and is numerically equivalent:

```python
# Correct: log-space avoids overflow risk
omega = np.exp(-2.0 * i / d_model * np.log(10000.0))
```

The test suite verifies these are identical to machine precision:

```python
def test_log_space_computation_matches_direct(self):
    omega_log = np.exp(-2.0 * i / d_model * np.log(10000.0))
    omega_direct = 1.0 / (10000.0 ** (2.0 * i / d_model))
    np.testing.assert_allclose(omega_log, omega_direct, rtol=1e-12)
```

### Pitfall 2: Concatenating Instead of Interleaving Sin and Cos

**The mistake:**

```python
# Wrong: concatenation puts all sins first, then all cosines
pe = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
```

**Why it is wrong:** This places sin values in columns $[0, d_{model}/2)$ and cos values in columns $[d_{model}/2, d_{model})$. The rotation matrix property assumes column $2i$ is sin and column $2i+1$ is cos. With concatenation, the (sin, cos) pairs are not adjacent, and the block-diagonal rotation structure breaks. The `relative_position_matrix` function would fail verification.

**The fix:**

```python
# Correct: interleave sin and cos at adjacent columns
pe = np.zeros((seq_len, d_model), dtype=np.float64)
pe[:, 0::2] = np.sin(angles)   # even columns: sin
pe[:, 1::2] = np.cos(angles)   # odd columns: cos
```

### Pitfall 3: Adding PE After Projection (Wrong Order)

**The mistake:**

```python
# Wrong: adding PE after Q/K/V projections
Q = X @ W_Q
Q = Q + PE[:L, :]   # PE is added too late
```

**Why it is wrong:** Positional encoding must be added to the input embeddings *before* the linear projections. When PE is added before projection, the resulting $Q$ contains cross-terms between content ($X$) and position ($PE$) through the weight matrix:

$$Q = (X + PE) W_Q = XW_Q + PE \cdot W_Q$$

Adding PE after projection means the position information bypasses the projection entirely and is not transformed by the learned weights, severely limiting the model's ability to use position information in its attention computation.

**The fix:**

```python
# Correct: add PE to input, then project
X_pe = X + PE[:L, :]
Q = X_pe @ W_Q
K = X_pe @ W_K
V = X_pe @ W_V
```

### Pitfall 4: Forgetting to Sum Gradients Across Batch (Learned Encoding)

**The mistake:**

```python
# Wrong: taking only the first batch element's gradient
self.grad_embedding[:L, :] = grad_output[0]
```

**Why it is wrong:** The embedding table $PE[:L, :]$ is shared across all $B$ batch elements in the forward pass. By the chain rule, the gradient for a parameter used by multiple paths is the sum of gradients from each path. Dropping $B - 1$ batch elements loses most of the gradient signal.

**The fix:**

```python
# Correct: sum across the batch dimension
self.grad_embedding[:L, :] = grad_output.sum(axis=0)   # (B, L, d) -> (L, d)
```

The test suite verifies that the batch gradient is $B$ times the single-element gradient:

```python
def test_gradient_accumulation_across_batch(self):
    # With B identical upstream gradients, grad_embedding should be B times
    # what a single-element batch produces
    np.testing.assert_allclose(
        grad_emb_batch[:L, :], B * grad_emb_single[:L, :], atol=1e-12
    )
```

---

## The PE Matrix as a Visual Pattern

### Heatmap Structure

If you plotted the PE matrix as a heatmap (rows = positions, columns = dimensions), you would see:

```
     dim 0  dim 1  dim 2  dim 3  ...  dim d-2  dim d-1
     (sin)  (cos)  (sin)  (cos)       (sin)    (cos)
pos 0  [  0    1.0    0    1.0   ...    0       1.0   ]
pos 1  [rapid oscillation         ...  barely changing ]
pos 2  [   |    |     |    |     ...    |        |    ]
pos 3  [   v    v     |    |     ...    |        |    ]
  ...  [              v    v     ...    |        |    ]
  ...  [                         ...    v        v    ]
pos L  [                         ...                  ]

Left columns:   rapid vertical stripes (high frequency)
Right columns:  smooth vertical gradients (low frequency)
```

The leftmost columns show rapid alternation between positive and negative values (short wavelength). Moving right, the pattern becomes smoother -- gradual transitions that take hundreds of positions to complete one cycle. This multi-scale structure is what gives each position a unique "fingerprint."

---

## Connection to Inference Optimization

### Computational Cost at Inference

Positional encoding is negligible at inference time:

**Sinusoidal**: Precompute the PE matrix once during model initialization. At inference, it is a single elementwise addition of shape $(L, d_{model})$, broadcast over the batch. Zero trainable parameters, zero gradient computation.

**Learned**: A single index-select (gather $L$ rows from a table). Marginally faster than sinusoidal since no sin/cos computation is needed, but the difference is irrelevant in practice since both are dwarfed by attention and FFN costs.

### The Real Inference Implication: Sequence Length Limits

The critical inference concern is not speed but **length extrapolation**:

- **Learned embeddings cannot extrapolate.** If $L_{max} = 2048$, the model fails at position 2049. There is no embedding vector for it. Options: truncate input, fine-tune with longer sequences, or switch encoding scheme.

- **Sinusoidal encodings produce valid vectors for any position.** But "valid vectors" does not mean "good performance." Attention patterns learned during training on positions 0-2048 may not generalize to position 5000 because the model has never seen those attention score distributions.

This limitation drove the field toward RoPE and related relative position schemes, which encode position differences rather than absolute positions and demonstrate better length generalization (especially combined with techniques like NTK-aware scaling and YaRN).

### From Naive to Optimized

| Naive (what we implemented) | Optimized (production) |
|----------------------------|------------------------|
| Separate PE addition kernel | Fused with embedding lookup |
| Full PE matrix stored in float64 | PE computed on-the-fly or cached in float16 |
| Absolute position encoding | RoPE: relative position in attention |
| Fixed max length | Position interpolation / NTK scaling |
| Applied once before first layer | (In RoPE: applied at every layer) |

Understanding absolute positional encoding is essential because it establishes the baseline that RoPE improves upon. Every advantage of RoPE -- direct relative position encoding, better extrapolation, rotation-based structure -- is best understood in contrast to what absolute encoding does and where it falls short.

---

## Analysis Functions

The implementation includes three analysis functions that verify the mathematical properties discussed above.

### `relative_position_matrix(pe, offset)`

Constructs the block-diagonal rotation matrix $M_k$ and verifies that $M_k \cdot PE_{pos} \approx PE_{pos+k}$ for all valid positions. Returns the matrix and the maximum reconstruction error.

### `dot_product_distance(pe)`

Computes $D = PE \cdot PE^\top \in \mathbb{R}^{L \times L}$. For sinusoidal encodings, this matrix has Toeplitz structure: $D_{ij}$ depends only on $|i - j|$.

### `encoding_statistics(pe)`

Returns per-position norms (all equal to $\sqrt{d_{model}/2}$), per-dimension mean and variance, and global min/max values (bounded in $[-1, 1]$).

---

## Testing Your Understanding

### Quick Checks

1. **What is $PE_{(0, :)}$ for any $d_{model}$?** It is $[0, 1, 0, 1, \ldots, 0, 1]$ because $\sin(0) = 0$ and $\cos(0) = 1$ for every frequency.

2. **Why does each position vector have norm $\sqrt{d_{model}/2}$?** Each (sin, cos) pair contributes $\sin^2(\theta) + \cos^2(\theta) = 1$ to the squared norm. There are $d_{model}/2$ pairs, so $\|PE_{pos}\|^2 = d_{model}/2$.

3. **If $d_{model} = 512$ and $L = 100$, what is the shape of the PE matrix?** $(100, 512)$. After adding to input $X$ of shape $(B, 100, 512)$, the output is $(B, 100, 512)$.

4. **Why can sinusoidal encodings extrapolate but learned encodings cannot?** Sinusoidal encodings are defined by a formula: given any position, you compute $\sin(pos \cdot \omega_i)$ and $\cos(pos \cdot \omega_i)$. Learned encodings are a lookup table: there is no entry for positions beyond $L_{max}$.

5. **What happens if you use the same PE vector for two different positions?** Those two positions become indistinguishable to the model. Attention treats them identically, losing the ability to use order information. This is why uniqueness is critical and verified in the tests.

### Exercises

1. **Easy**: Modify the `sinusoidal_positional_encoding` function to use concatenation instead of interleaving (first half sin, second half cos). Run the tests and observe which ones fail and why.

2. **Medium**: Implement a function `pe_similarity_by_distance(pe)` that computes the average dot product for each distance $k$ (averaging over all position pairs $(p, p+k)$). Plot the result and verify it decreases with distance.

3. **Hard**: Implement a version of `LearnedPositionalEncoding` that initializes the embedding table to match sinusoidal encodings (instead of random $\mathcal{N}(0, 0.02)$). This was actually explored in the original Transformer paper -- they found no significant difference between sinusoidal and learned encodings initialized this way.

---

## Summary

### Key Takeaways

- Self-attention is permutation-equivariant: without positional information, it treats input as a set, not a sequence. Positional encoding adds position-dependent vectors to break this symmetry.

- Sinusoidal encodings use sine and cosine functions at geometrically spaced frequencies. Low dimensions oscillate fast (local position), high dimensions oscillate slow (global position). The combination uniquely identifies every position.

- The critical mathematical property: a fixed offset $k$ between positions corresponds to a rotation matrix $M_k$ that is independent of absolute position. This enables the model to learn relative position through its attention weights.

- Learned encodings are a trainable lookup table -- more expressive but cannot extrapolate beyond the maximum training length. The backward pass sums upstream gradients across the batch for each position row.

- Both are absolute position encodings, which is why modern LLMs have moved to RoPE (relative position, applied directly in attention). Understanding absolute encoding is essential context for understanding why RoPE exists.

### Quick Reference

```
Positional Encoding
|-- Sinusoidal (Fixed)
|   |-- Generate: O(L * d_model) -- compute sin/cos for angle matrix
|   |-- Forward:  O(B * L * d_model) -- elementwise add, broadcast over batch
|   |-- Memory:   O(L_max * d_model) -- precomputed buffer
|   |-- Params:   0
|   |-- Extrapolation: Yes (formula works for any position)
|
|-- Learned (Trainable)
|   |-- Forward:  O(B * L * d_model) -- index select + add
|   |-- Backward: O(B * L * d_model) -- sum gradients across batch
|   |-- Memory:   O(L_max * d_model) -- embedding table
|   |-- Params:   L_max * d_model
|   |-- Extrapolation: No (bounded by L_max)
|
|-- Mathematical Properties (Sinusoidal):
|   |-- Values bounded in [-1, 1]
|   |-- Position norms: sqrt(d_model / 2) for all positions
|   |-- Relative position: PE[pos+k] = M_k @ PE[pos] (rotation)
|   |-- Dot product: PE_i . PE_j depends only on |i - j|
|
|-- Superseded by: RoPE (rotary position embeddings)
|   |-- Encodes relative position directly in Q @ K^T
|   |-- Better length extrapolation
|   |-- Used in LLaMA, Mistral, Phi, and all modern LLMs
```
