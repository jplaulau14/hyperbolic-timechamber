# Multi-Head Attention -- Deep Dive

## The Intuition

### What Problem Are We Solving?

Single-head self-attention computes one attention pattern: for each token, it produces one set of weights over all other tokens. But language is not that simple. When processing the sentence "The cat sat on the mat because it was tired," the word "it" needs to simultaneously track multiple relationships:

- **Syntactic**: "it" is the subject of "was" (structural dependency)
- **Semantic**: "it" refers to "cat" (coreference)
- **Positional**: "it" relates to the immediately preceding clause boundary

A single attention head must compress all of these relationships into one distribution over positions. It computes a single weighted average, which means it must compromise -- it cannot give full weight to the syntactic neighbor and full weight to the semantic antecedent at the same time.

### The Key Insight

Instead of one attention operation in a 512-dimensional space, run eight independent attention operations in 64-dimensional subspaces. Each head learns its own projection matrices $W_Q^i$, $W_K^i$, $W_V^i$, so each head can specialize: one head might learn "attend to the previous token," another "attend to the subject of this clause," another "attend to tokens with similar semantic role." After computing these eight independent perspectives, concatenate the results and project back to the original dimension.

The surprising result: this does not increase computational cost. The total FLOPs for eight heads of dimension 64 equals the FLOPs for one head of dimension 512. You get more representational power for free -- the only difference is *how* the computation is organized.

### Real-World Analogy

Think of a courtroom trial where a judge must make a decision. A single judge (single-head attention) hears all the evidence and produces one verdict. Multi-head attention is like a panel of specialized judges: one focuses on procedural law, one on precedent, one on witness credibility, one on physical evidence. Each judge independently reviews the case from their own angle, then they combine their findings into a unified decision. Each individual judge sees the same case but attends to different aspects -- and the combined verdict is richer than what any single judge could produce alone.

---

## The Math, Step by Step

### Building Up to the Full Formula

**Step 1 -- Start with single-head attention** (what we built in Topic 9):

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q, K \in \mathbb{R}^{B \times L \times d_k}$ and $V \in \mathbb{R}^{B \times L \times d_v}$. This produces one attention pattern of shape $(B, L, L)$.

**Step 2 -- Add multiple heads** (conceptual view):

Each head $i \in \{1, \ldots, h\}$ has its own projection matrices:

$$W_Q^i \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_K^i \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_V^i \in \mathbb{R}^{d_{\text{model}} \times d_v}$$

where $d_k = d_v = d_{\text{model}} / h$. Each head computes its own attention:

$$\text{head}_i = \text{Attention}(X W_Q^i,\; X W_K^i,\; X W_V^i) \quad \in \mathbb{R}^{B \times L \times d_v}$$

**Step 3 -- Concatenate and project** (the full formula):

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

where $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$. The concatenation produces a $(B, L, h \cdot d_v) = (B, L, d_{\text{model}})$ tensor, and $W^O$ mixes information across heads.

### The Core Equations

**Equation 1: Q/K/V Projections (fused)**
$$Q = X \cdot W^Q, \quad K = X \cdot W^K, \quad V = X \cdot W^V$$

Where:
- $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$: input tensor
- $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$: fused projection matrices (each contains all $h$ per-head matrices concatenated)
- $Q, K, V \in \mathbb{R}^{B \times L \times d_{\text{model}}}$: projected queries, keys, values

**Equation 2: Head Split (reshape + transpose)**
$$Q \leftarrow Q.\text{reshape}(B, L, h, d_k).\text{transpose}(0, 2, 1, 3) \quad \rightarrow (B, h, L, d_k)$$

**Equation 3: Scaled Dot-Product Attention**
$$\text{scores} = \frac{Q \cdot K^\top}{\sqrt{d_k}} \quad \in \mathbb{R}^{B \times h \times L \times L}$$

$$A = \text{softmax}(\text{scores}, \text{axis}=-1) \quad \in \mathbb{R}^{B \times h \times L \times L}$$

$$\text{attn\_output} = A \cdot V \quad \in \mathbb{R}^{B \times h \times L \times d_v}$$

**Equation 4: Head Merge (transpose + reshape)**
$$\text{concat} = \text{attn\_output}.\text{transpose}(0, 2, 1, 3).\text{reshape}(B, L, d_{\text{model}})$$

**Equation 5: Output Projection**
$$\text{output} = \text{concat} \cdot W^O \quad \in \mathbb{R}^{B \times L \times d_{\text{model}}}$$

---

## The Reshape Trick -- How It Actually Works

### Why Not Loop Over Heads?

The naive approach would be:

```python
# SLOW: h separate small matmuls
for i in range(h):
    Q_i = X @ W_Q_per_head[i]   # (B, L, d_model) @ (d_model, d_k) -> (B, L, d_k)
    K_i = X @ W_K_per_head[i]
    V_i = X @ W_V_per_head[i]
    # ... attention for head i ...
```

This has two problems: (1) $h$ small matrix multiplications are much slower than one large one on GPUs (poor utilization of streaming multiprocessors), and (2) each per-head weight matrix requires a separate memory allocation.

### The Fused Approach

Stack all per-head weight matrices into one big matrix:

$$W^Q = [W_Q^1 \;|\; W_Q^2 \;|\; \cdots \;|\; W_Q^h] \quad \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

One large GEMM produces all heads simultaneously:

$$Q_{\text{fused}} = X \cdot W^Q \quad \in \mathbb{R}^{B \times L \times d_{\text{model}}}$$

Now the trick: the last dimension $d_{\text{model}}$ is actually $h$ groups of $d_k$ values. We reinterpret this via reshape:

```
Memory layout (one token's Q values):
[q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7]
 \___ head 0 ___/  \___ head 1 ___/
       d_k=4              d_k=4
```

### Memory Layout in Detail

After the projection, $Q$ has shape $(B, L, d_{\text{model}})$ and is contiguous in memory -- elements are laid out as:

```
q[b, l, 0], q[b, l, 1], ..., q[b, l, d_model-1]   (for each b, l)
```

**Reshape to $(B, L, h, d_k)$**: This is a zero-cost reinterpretation. No data is copied. NumPy simply changes the shape metadata. The element at position `[b, l, i, j]` is the same memory location as `[b, l, i*d_k + j]`.

**Transpose to $(B, h, L, d_k)$**: This swaps axes 1 and 2. Now elements within a single head across sequence positions are "nearby" in the logical layout. However, this creates a **non-contiguous** view -- the strides change from $(L \cdot d_{\text{model}},\; d_{\text{model}},\; d_k,\; 1)$ to $(L \cdot d_{\text{model}},\; d_k,\; d_{\text{model}},\; 1)$.

```
Before transpose (B, L, h, d_k) -- contiguous:
Strides: (L*d_model, d_model, d_k, 1)

     Axis 0 (B)      Axis 1 (L)    Axis 2 (h)   Axis 3 (d_k)
     jump d_model*L   jump d_model   jump d_k     jump 1

After transpose (B, h, L, d_k) -- non-contiguous:
Strides: (L*d_model, d_k, d_model, 1)

     Axis 0 (B)      Axis 1 (h)    Axis 2 (L)   Axis 3 (d_k)
     jump d_model*L   jump d_k      jump d_model  jump 1
```

After transpose, stepping along the sequence dimension (axis 2) requires jumping by $d_{\text{model}}$ elements instead of $d_k$. In production code, calling `.contiguous()` (PyTorch) or `np.ascontiguousarray()` copies data into a new layout for better cache performance.

---

## Full Forward Pass -- Step by Step with Shapes

The implementation follows this data flow:

```
    Input X
    (B, L, d_model)
         |
    +----+----+----+
    |         |         |
    v         v         v
  X @ W_Q   X @ W_K   X @ W_V         Three fused GEMMs
  (B,L,dm)  (B,L,dm)  (B,L,dm)
    |         |         |
    v         v         v
  reshape   reshape   reshape          Zero-cost reinterpretation
  (B,L,h,dk)(B,L,h,dk)(B,L,h,dk)
    |         |         |
    v         v         v
  transpose transpose transpose        Creates strided view
  (B,h,L,dk)(B,h,L,dk)(B,h,L,dk)
    |         |         |
    +----+----+         |
         |              |
         v              |
    Q @ K^T / sqrt(dk)  |              Batched over B and h
    (B, h, L, L)        |
         |              |
         v              |
    + causal mask       |              Broadcasting (1,1,L,L)
    (B, h, L, L)        |
         |              |
         v              |
    softmax(axis=-1)    |
    (B, h, L, L)        |
         |              |
         +-------+------+
                 |
                 v
            A @ V                       Batched matmul
            (B, h, L, dk)
                 |
                 v
            transpose(0,2,1,3)          Reverse the split
            (B, L, h, dk)
                 |
                 v
            reshape(B, L, d_model)      Merge heads
            (B, L, d_model)
                 |
                 v
            concat @ W_O                Output projection
            (B, L, d_model)
```

### Implementation Walkthrough

```python
def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    B, L, _ = X.shape

    # Step 1: Three fused GEMMs -- (B, L, d_model) @ (d_model, d_model)
    Q = X @ self.W_Q     # -> (B, L, d_model)
    K = X @ self.W_K
    V = X @ self.W_V

    if self.use_bias:
        Q = Q + self.b_Q  # Broadcasting: (d_model,) broadcasts over (B, L)
        K = K + self.b_K
        V = V + self.b_V

    # Step 2: Split heads -- reshape then transpose
    Q = self._split_heads(Q)  # (B, L, d_model) -> (B, h, L, d_k)
    K = self._split_heads(K)
    V = self._split_heads(V)

    # Step 3: Scaled dot-product attention, batched over B and h
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)  # -> (B, h, L, L)

    if mask is not None:
        scores = scores + mask  # Additive mask, broadcasts from (1, 1, L, L)

    A = softmax(scores, axis=-1)  # -> (B, h, L, L)

    # Step 4: Weighted sum of values
    attn_output = A @ V  # (B, h, L, L) @ (B, h, L, d_k) -> (B, h, L, d_k)

    # Step 5: Merge heads -- transpose then reshape
    concat = self._merge_heads(attn_output)  # -> (B, L, d_model)

    # Step 6: Output projection
    output = concat @ self.W_O  # -> (B, L, d_model)
    if self.use_bias:
        output = output + self.b_O

    # Cache everything needed for backward
    self._cache = {
        "X": X, "Q": Q, "K": K, "V": V, "A": A,
        "attn_output": attn_output, "concat": concat, "mask": mask,
    }
    return output
```

**The `_split_heads` method:**

```python
def _split_heads(self, x: np.ndarray) -> np.ndarray:
    B, L, _ = x.shape
    return x.reshape(B, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
```

This is two operations: `.reshape(B, L, h, d_k)` reinterprets the last dimension as two dimensions (zero-copy), then `.transpose(0, 2, 1, 3)` swaps the sequence and head axes so that the head dimension becomes a batch dimension for the attention matmul.

**The `_merge_heads` method:**

```python
def _merge_heads(self, x: np.ndarray) -> np.ndarray:
    B, _, L, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
```

The exact inverse: transpose puts sequence before heads, then reshape collapses the last two dimensions back into $d_{\text{model}}$.

**Why `K.transpose(0, 1, 3, 2)` and not `K.transpose(0, 2, 1)`?**

In single-head attention, $K$ has shape $(B, L, d_k)$ and we swap the last two axes: `.transpose(0, 2, 1)`. In multi-head attention, $K$ has shape $(B, h, L, d_k)$ -- four dimensions. We only want to transpose the last two (the "matrix" part of the batched matmul), keeping $B$ and $h$ as batch dimensions: `.transpose(0, 1, 3, 2)`.

---

## Full Backward Pass -- The 7-Step Chain Rule

Let $g_{\text{out}} = \frac{\partial \mathcal{L}}{\partial \text{output}} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$.

### Step 1: Gradient Through Output Projection

Forward: $\text{output} = \text{concat} \cdot W^O$

This is a standard linear layer backward:

$$\frac{\partial \mathcal{L}}{\partial \text{concat}} = g_{\text{out}} \cdot W^{O\top} \quad \in \mathbb{R}^{B \times L \times d_{\text{model}}}$$

$$\frac{\partial \mathcal{L}}{\partial W^O} = \text{concat}^\top \cdot g_{\text{out}} = \sum_b \text{concat}_b^\top \cdot g_{\text{out},b} \quad \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

In code: `np.einsum("blm,bln->mn", concat, grad_output)` sums over batch and sequence.

```python
grad_concat = grad_output @ self.W_O.T
self.grad_W_O = np.einsum("blm,bln->mn", concat, grad_output)
```

### Step 2: Gradient Through Head Merge

Forward: transpose$(0, 2, 1, 3)$ then reshape$(B, L, d_{\text{model}})$.

Backward: undo in reverse order -- reshape then transpose:

$$g_{\text{attn\_out}} = g_{\text{concat}}.\text{reshape}(B, L, h, d_v).\text{transpose}(0, 2, 1, 3) \quad \in \mathbb{R}^{B \times h \times L \times d_v}$$

Reshape and transpose are linear operations with orthogonal Jacobians. The gradient simply gets rearranged -- no multiplication, no scaling.

```python
grad_attn_output = grad_concat.reshape(B, L, self.num_heads, self.d_v).transpose(0, 2, 1, 3)
```

### Step 3: Gradient Through Value Weighting

Forward: $\text{attn\_output} = A \cdot V$ where $A \in \mathbb{R}^{B \times h \times L \times L}$, $V \in \mathbb{R}^{B \times h \times L \times d_v}$.

Standard batched matmul backward (batched over both $B$ and $h$):

$$\frac{\partial \mathcal{L}}{\partial A} = g_{\text{attn\_out}} \cdot V^\top \quad \in \mathbb{R}^{B \times h \times L \times L}$$

$$\frac{\partial \mathcal{L}}{\partial V} = A^\top \cdot g_{\text{attn\_out}} \quad \in \mathbb{R}^{B \times h \times L \times d_v}$$

The transposes here are on the last two dimensions only -- `.transpose(0, 1, 3, 2)`.

```python
grad_A = grad_attn_output @ V.transpose(0, 1, 3, 2)
grad_V = A.transpose(0, 1, 3, 2) @ grad_attn_output
```

### Step 4: Gradient Through Softmax

The softmax backward formula, applied independently to each $(b, h)$ pair:

$$\frac{\partial \mathcal{L}}{\partial \text{scores}} = A \odot \left(g_A - \sum_{j} (g_A \odot A)\right)$$

where $\sum_j$ sums along the last axis (key dimension) with keepdims. Expanded:

$$\text{dot}_{b,h,i} = \sum_j g_{A_{b,h,i,j}} \cdot A_{b,h,i,j}$$

$$\frac{\partial \mathcal{L}}{\partial \text{scores}_{b,h,i,j}} = A_{b,h,i,j} \cdot \left(g_{A_{b,h,i,j}} - \text{dot}_{b,h,i}\right)$$

The causal mask needs no special treatment: masked positions have $A_{i,j} = 0$, so $\frac{\partial \mathcal{L}}{\partial \text{scores}_{i,j}} = 0 \cdot (\ldots) = 0$ automatically.

```python
grad_scores = softmax_backward(grad_A, A)
```

### Step 5: Gradient Through Scaling and $QK^\top$

Forward: $\text{scores} = \frac{Q \cdot K^\top}{\sqrt{d_k}}$

First, undo the scaling:

$$g_{\text{raw}} = \frac{g_{\text{scores}}}{\sqrt{d_k}}$$

Then, batched matmul backward for $\text{raw} = Q \cdot K^\top$:

$$\frac{\partial \mathcal{L}}{\partial Q} = g_{\text{raw}} \cdot K \quad \in \mathbb{R}^{B \times h \times L \times d_k}$$

$$\frac{\partial \mathcal{L}}{\partial K} = g_{\text{raw}}^\top \cdot Q \quad \in \mathbb{R}^{B \times h \times L \times d_k}$$

The transpose in $g_{\text{raw}}^\top$ is on the last two dimensions: $(B, h, L, L) \to (B, h, L, L)$, which transposes the $L \times L$ attention pattern.

```python
grad_raw = grad_scores / scale
grad_Q = grad_raw @ K                        # (B,h,L,L) @ (B,h,L,dk) -> (B,h,L,dk)
grad_K = grad_raw.transpose(0, 1, 3, 2) @ Q  # (B,h,L,L) @ (B,h,L,dk) -> (B,h,L,dk)
```

### Step 6: Gradient Through Head Split

Forward: reshape$(B, L, h, d_k)$ then transpose$(0, 2, 1, 3)$.

Backward: undo in reverse -- transpose$(0, 2, 1, 3)$ then reshape$(B, L, d_{\text{model}})$:

$$g_{Q_{\text{flat}}} = \frac{\partial \mathcal{L}}{\partial Q}.\text{transpose}(0, 2, 1, 3).\text{reshape}(B, L, d_{\text{model}})$$

Same for $K$ and $V$.

```python
grad_Q_flat = grad_Q.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
grad_K_flat = grad_K.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
grad_V_flat = grad_V.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
```

### Step 7: Gradient Through Q/K/V Projections

Forward: $Q_{\text{flat}} = X \cdot W^Q$ (and similarly for $K$, $V$).

Weight gradients:

$$\frac{\partial \mathcal{L}}{\partial W^Q} = \sum_{b,l} X_{b,l}^\top \cdot g_{Q_{\text{flat}},b,l} \quad \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

Input gradient -- **this is the key insight of the backward pass**:

$$\frac{\partial \mathcal{L}}{\partial X} = g_{Q_{\text{flat}}} \cdot W^{Q\top} + g_{K_{\text{flat}}} \cdot W^{K\top} + g_{V_{\text{flat}}} \cdot W^{V\top}$$

### Why $\nabla X$ Accumulates From Three Branches

The input $X$ is used three times in the forward pass: once for the $Q$ projection, once for $K$, once for $V$. By the multivariate chain rule, the total gradient is the sum of the gradients from each use:

$$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Q_{\text{flat}}} \cdot \frac{\partial Q_{\text{flat}}}{\partial X} + \frac{\partial \mathcal{L}}{\partial K_{\text{flat}}} \cdot \frac{\partial K_{\text{flat}}}{\partial X} + \frac{\partial \mathcal{L}}{\partial V_{\text{flat}}} \cdot \frac{\partial V_{\text{flat}}}{\partial X}$$

Since $\frac{\partial Q_{\text{flat}}}{\partial X} = W^{Q\top}$ (and similarly for $K$, $V$), we get the three-way sum above. This is a standard application of the chain rule for functions with a shared input. Missing any one of these three terms would produce an incorrect gradient.

```python
self.grad_W_Q = np.einsum("blm,bld->md", X, grad_Q_flat)
self.grad_W_K = np.einsum("blm,bld->md", X, grad_K_flat)
self.grad_W_V = np.einsum("blm,bld->md", X, grad_V_flat)

grad_X = grad_Q_flat @ self.W_Q.T + grad_K_flat @ self.W_K.T + grad_V_flat @ self.W_V.T
```

### What Gets Cached and Why

| Cached Tensor | Shape | Used in Backward Step |
|---------------|-------|-----------------------|
| $X$ | $(B, L, d_{\text{model}})$ | Step 7: compute $\nabla W^Q$, $\nabla W^K$, $\nabla W^V$ |
| $Q$ | $(B, h, L, d_k)$ | Step 5: compute $\nabla K$ via $g_{\text{raw}}^\top \cdot Q$ |
| $K$ | $(B, h, L, d_k)$ | Step 5: compute $\nabla Q$ via $g_{\text{raw}} \cdot K$ |
| $V$ | $(B, h, L, d_v)$ | Step 3: compute $\nabla A$ via $g_{\text{attn\_out}} \cdot V^\top$ |
| $A$ | $(B, h, L, L)$ | Step 3: compute $\nabla V$; Step 4: softmax backward |
| concat | $(B, L, d_{\text{model}})$ | Step 1: compute $\nabla W^O$ |

The attention matrix $A$ at shape $(B, h, L, L)$ dominates memory for long sequences. With $L = 4096$ and $h = 32$, that is $32 \times 4096^2 \approx 537M$ elements -- just for one layer. This is exactly what Flash Attention avoids materializing.

---

## Worked Example

### Setup: $d_{\text{model}} = 4$, $h = 2$, $L = 2$, $B = 1$

With $d_k = d_v = d_{\text{model}} / h = 2$.

**Input:**

$$X = \begin{bmatrix} 1.0 & 0.0 & -1.0 & 0.5 \\ 0.5 & 1.0 & 0.0 & -0.5 \end{bmatrix} \quad \in \mathbb{R}^{1 \times 2 \times 4}$$

**Weights** (using simple values for clarity, no bias):

$$W^Q = \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \end{bmatrix}, \quad W^K = \begin{bmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}$$

$$W^V = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix}, \quad W^O = I_4 \text{ (identity, for simplicity)}$$

### Step 1: Project Q, K, V

$$Q = X \cdot W^Q = \begin{bmatrix} 1(1)+0(0)+(-1)(0)+0.5(1) & 1(0)+0(1)+(-1)(0)+0.5(0) & 1(0)+0(1)+(-1)(1)+0.5(0) & 1(1)+0(0)+(-1)(0)+0.5(0) \\ 0.5(1)+1(0)+0(0)+(-0.5)(1) & 0.5(0)+1(1)+0(0)+(-0.5)(0) & 0.5(0)+1(1)+0(1)+(-0.5)(0) & 0.5(1)+1(0)+0(0)+(-0.5)(0) \end{bmatrix}$$

$$Q = \begin{bmatrix} 1.5 & 0.0 & -1.0 & 1.0 \\ 0.0 & 1.0 & 1.0 & 0.5 \end{bmatrix}$$

$$K = X \cdot W^K = \begin{bmatrix} 0+0+0+0 & 1+0-1+0 & 1+0+0+0.5 & 0+0+0+0.5 \\ 0+1+0+0 & 0.5+0+0+0 & 0.5+0+0-0.5 & 0+1+0-0.5 \end{bmatrix}$$

$$K = \begin{bmatrix} 0.0 & 0.0 & 1.5 & 0.5 \\ 1.0 & 0.5 & 0.0 & 0.5 \end{bmatrix}$$

$$V = X \cdot W^V = \begin{bmatrix} 1+0+0+0.5 & 0+0+0+0.5 & 0+0+0+0 & 0+0-1+0 \\ 0.5+0+0-0.5 & 0+1+0-0.5 & 0+1+0+0 & 0+0+0+0 \end{bmatrix}$$

$$V = \begin{bmatrix} 1.5 & 0.5 & 0.0 & -1.0 \\ 0.0 & 0.5 & 1.0 & 0.0 \end{bmatrix}$$

### Step 2: Split Heads

Reshape $(1, 2, 4) \to (1, 2, 2, 2)$ then transpose $(0, 2, 1, 3) \to (1, 2, 2, 2)$:

**$Q$ split into heads:**

$$Q_{\text{head 0}} = \begin{bmatrix} 1.5 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad Q_{\text{head 1}} = \begin{bmatrix} -1.0 & 1.0 \\ 1.0 & 0.5 \end{bmatrix}$$

**$K$ split into heads:**

$$K_{\text{head 0}} = \begin{bmatrix} 0.0 & 0.0 \\ 1.0 & 0.5 \end{bmatrix}, \quad K_{\text{head 1}} = \begin{bmatrix} 1.5 & 0.5 \\ 0.0 & 0.5 \end{bmatrix}$$

**$V$ split into heads:**

$$V_{\text{head 0}} = \begin{bmatrix} 1.5 & 0.5 \\ 0.0 & 0.5 \end{bmatrix}, \quad V_{\text{head 1}} = \begin{bmatrix} 0.0 & -1.0 \\ 1.0 & 0.0 \end{bmatrix}$$

### Step 3: Attention Per Head

**Head 0:** $d_k = 2$, so $\sqrt{d_k} = \sqrt{2} \approx 1.414$.

$$\text{scores}_0 = \frac{Q_0 \cdot K_0^\top}{\sqrt{2}} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1.5(0)+0(0) & 1.5(1)+0(0.5) \\ 0(0)+1(0) & 0(1)+1(0.5) \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 0 & 1.5 \\ 0 & 0.5 \end{bmatrix}$$

$$= \begin{bmatrix} 0.000 & 1.061 \\ 0.000 & 0.354 \end{bmatrix}$$

$$A_0 = \text{softmax}(\text{scores}_0) = \begin{bmatrix} \frac{1}{1+e^{1.061}} & \frac{e^{1.061}}{1+e^{1.061}} \\ \frac{1}{1+e^{0.354}} & \frac{e^{0.354}}{1+e^{0.354}} \end{bmatrix} \approx \begin{bmatrix} 0.257 & 0.743 \\ 0.412 & 0.588 \end{bmatrix}$$

$$\text{out}_0 = A_0 \cdot V_0 = \begin{bmatrix} 0.257(1.5)+0.743(0) & 0.257(0.5)+0.743(0.5) \\ 0.412(1.5)+0.588(0) & 0.412(0.5)+0.588(0.5) \end{bmatrix} = \begin{bmatrix} 0.386 & 0.500 \\ 0.618 & 0.500 \end{bmatrix}$$

**Head 1:**

$$\text{scores}_1 = \frac{Q_1 \cdot K_1^\top}{\sqrt{2}} = \frac{1}{\sqrt{2}} \begin{bmatrix} (-1)(1.5)+1(0.5) & (-1)(0)+1(0.5) \\ 1(1.5)+0.5(0.5) & 1(0)+0.5(0.5) \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} -1.0 & 0.5 \\ 1.75 & 0.25 \end{bmatrix}$$

$$= \begin{bmatrix} -0.707 & 0.354 \\ 1.237 & 0.177 \end{bmatrix}$$

$$A_1 = \text{softmax}(\text{scores}_1) \approx \begin{bmatrix} 0.258 & 0.742 \\ 0.742 & 0.258 \end{bmatrix}$$

Wait -- let me recompute more carefully. $e^{-0.707} \approx 0.493$, $e^{0.354} \approx 1.425$. Row 0 sum: $0.493 + 1.425 = 1.918$.

$$A_1[0, :] = [0.493/1.918,\; 1.425/1.918] = [0.257,\; 0.743]$$

$e^{1.237} \approx 3.445$, $e^{0.177} \approx 1.194$. Row 1 sum: $3.445 + 1.194 = 4.639$.

$$A_1[1, :] = [3.445/4.639,\; 1.194/4.639] = [0.743,\; 0.257]$$

$$\text{out}_1 = A_1 \cdot V_1 = \begin{bmatrix} 0.257(0)+0.743(1) & 0.257(-1)+0.743(0) \\ 0.743(0)+0.257(1) & 0.743(-1)+0.257(0) \end{bmatrix} = \begin{bmatrix} 0.743 & -0.257 \\ 0.257 & -0.743 \end{bmatrix}$$

### Step 4: Merge Heads

Transpose back and concatenate along the last dimension:

$$\text{concat} = \begin{bmatrix} 0.386 & 0.500 & 0.743 & -0.257 \\ 0.618 & 0.500 & 0.257 & -0.743 \end{bmatrix}$$

### Step 5: Output Projection

With $W^O = I_4$: $\text{output} = \text{concat}$.

Notice how the two heads produced different attention patterns: Head 0 had both tokens attending more to token 1 (weights $\approx 0.74$ and $0.59$), while Head 1 showed token 0 attending to token 1 and token 1 attending to token 0 -- capturing a *different* relationship in the same data.

---

## Why FLOPs Are the Same

This is a crucial result that surprises many people: splitting attention into $h$ heads does not change the total FLOPs.

### Projection GEMMs -- Identical

Both single-head and multi-head use the same weight matrices:

$$W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}, \quad W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

Projection cost: $4 \times 2BLd_{\text{model}}^2 = 8BLd_{\text{model}}^2$ FLOPs regardless of $h$.

### Attention Core -- The Cancellation

**Single-head** ($h = 1$, $d_k = d_{\text{model}}$):

$$QK^\top: \quad 2BL^2 d_{\text{model}} \text{ FLOPs}$$
$$AV: \quad 2BL^2 d_{\text{model}} \text{ FLOPs}$$

**Multi-head** ($h$ heads, $d_k = d_{\text{model}} / h$):

$$QK^\top: \quad h \times 2BL^2 d_k = h \times 2BL^2 \times \frac{d_{\text{model}}}{h} = 2BL^2 d_{\text{model}} \text{ FLOPs}$$
$$AV: \quad h \times 2BL^2 d_k = 2BL^2 d_{\text{model}} \text{ FLOPs}$$

The $h$ cancels with $d_k = d_{\text{model}} / h$. You do $h$ times as many operations, but each is $h$ times smaller.

### What *Does* Change: Softmax

Softmax is applied to each head's $L \times L$ score matrix independently:

- Single-head: $5BL^2$ FLOPs (one $L \times L$ matrix)
- Multi-head: $5BhL^2$ FLOPs ($h$ separate $L \times L$ matrices)

But softmax is a tiny fraction of total compute. For $d_{\text{model}} = 4096$ and $L = 2048$, the matmul FLOPs are $\sim 10^{12}$ while the softmax FLOPs are $\sim 10^9$ -- three orders of magnitude smaller.

### Verification in Code

```python
def count_flops(batch_size, seq_len, d_model, n_heads):
    B, L, h = batch_size, seq_len, n_heads
    d_k = d_model // h

    proj_qkv = 3 * 2 * B * L * d_model * d_model   # Same regardless of h
    proj_o = 2 * B * L * d_model * d_model           # Same regardless of h
    qk = 2 * B * h * L * L * d_k      # h * d_k = d_model, so same
    av = 2 * B * h * L * L * d_k      # h * d_k = d_model, so same
    sm = 5 * B * h * L * L            # This DOES scale with h

    return proj_qkv + proj_o + qk + av + sm
```

The test suite verifies this directly:

```python
def test_multi_head_same_matmul_flops_as_single_head(self):
    flops_1h = count_flops(B, L, d_model, 1)
    flops_4h = count_flops(B, L, d_model, 4)

    # Subtract softmax FLOPs (the only part that differs)
    matmul_1h = flops_1h - 5 * B * 1 * L * L
    matmul_4h = flops_4h - 5 * B * 4 * L * L

    assert matmul_1h == matmul_4h  # Identical!
```

---

## Causal Masking -- Broadcasting $(1, 1, L, L)$

### How the Mask Works

The causal mask ensures that position $i$ can only attend to positions $j \leq i$ (autoregressive constraint). It is an additive mask applied before softmax:

$$M_{i,j} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

For $L = 4$:

$$M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

After adding to scores and applying softmax, $e^{-\infty} = 0$, so future positions get zero weight.

### Why Shape $(1, 1, L, L)$?

The attention scores have shape $(B, h, L, L)$. The mask needs to broadcast across batch and heads:

```
Scores:   (B, h, L, L)       e.g., (32, 8, 512, 512)
Mask:     (1, 1, L, L)       e.g., ( 1, 1, 512, 512)
                                     ^  ^
                                     |  |
                            broadcasts over B and h
```

Every batch element and every head sees the **same** causal constraint. This is correct because causality is not head-specific -- no head should attend to future tokens regardless of what pattern it has learned.

### Implementation

```python
def create_causal_mask(seq_len: int) -> np.ndarray:
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask.reshape(1, 1, seq_len, seq_len)
```

`np.triu_indices(seq_len, k=1)` returns indices of the strict upper triangle (above the diagonal). Setting these to $-\infty$ and everything else to $0$ creates exactly the mask we need.

### Backward Through the Mask

The mask is a constant added to scores. Its gradient is simply the upstream gradient passed through unchanged (derivative of addition is 1). But positions where $A_{i,j} = 0$ (masked positions) automatically get zero gradient through softmax backward:

$$\frac{\partial \mathcal{L}}{\partial \text{scores}_{i,j}} = A_{i,j} \cdot (\ldots)$$

Since $A_{i,j} = 0$ for masked positions, their gradient is zero with no extra logic needed.

---

## Single-Head Equivalence -- Proof That $h = 1$ Reduces to Self-Attention

When $h = 1$ and $d_k = d_{\text{model}}$, multi-head attention becomes self-attention. Here is why.

### Forward Pass Equivalence

**Projections**: Identical. Both compute $Q = XW^Q$, $K = XW^K$, $V = XW^V$ with the same weight shapes.

**Head split** with $h = 1$:

```python
# reshape(B, L, 1, d_model) -- adds a trivial dimension
# transpose(0, 2, 1, 3) -- swaps L and 1
# Result: (B, 1, L, d_model)
```

The extra dimension of size 1 has no effect on the matmul. NumPy's `@` operator treats the first two dimensions as batch dimensions:

$$Q \cdot K^\top: \quad (B, 1, L, d_{\text{model}}) \times (B, 1, d_{\text{model}}, L) \to (B, 1, L, L)$$

This is identical to the 3D version $(B, L, d_{\text{model}}) \times (B, d_{\text{model}}, L) \to (B, L, L)$ with a size-1 dimension inserted.

**Head merge** with $h = 1$:

```python
# transpose(0, 2, 1, 3): (B, 1, L, d_model) -> (B, L, 1, d_model)
# reshape(B, L, d_model): removes the trivial dimension
```

The output is $(B, L, d_{\text{model}})$, exactly what self-attention produces.

### Test Verification

The test suite explicitly checks this:

```python
def test_single_head_matches_self_attention(self):
    mha = MultiHeadAttention(d_model=8, num_heads=1, use_bias=True)
    sa = SelfAttention(d_model=8, d_k=8, d_v=8, d_out=8, use_bias=True)

    # Copy identical weights
    sa.W_Q = mha.W_Q.copy()
    sa.W_K = mha.W_K.copy()
    sa.W_V = mha.W_V.copy()
    sa.W_O = mha.W_O.copy()
    sa.b_Q = mha.b_Q.copy()
    # ... (all biases)

    X = np.random.randn(2, 5, 8)
    out_mha = mha.forward(X)
    out_sa = sa.forward(X)

    np.testing.assert_allclose(out_mha, out_sa, atol=1e-12)
```

This test also runs with causal masking, confirming the equivalence holds for both masked and unmasked attention.

---

## Complexity Analysis

### Time Complexity

| Operation | FLOPs | Why |
|-----------|-------|-----|
| Q/K/V projections | $6BLd_{\text{model}}^2$ | Three $(B, L, d_{\text{model}}) \times (d_{\text{model}}, d_{\text{model}})$ matmuls |
| $QK^\top$ (all heads) | $2BL^2 d_{\text{model}}$ | $h$ matmuls of $(B, L, d_k) \times (d_k, L)$, with $h \cdot d_k = d_{\text{model}}$ |
| Softmax | $5BhL^2$ | Per-element exp, sum, div across $h$ heads |
| $AV$ (all heads) | $2BL^2 d_{\text{model}}$ | $h$ matmuls of $(B, L, L) \times (L, d_v)$ |
| Output projection | $2BLd_{\text{model}}^2$ | One $(B, L, d_{\text{model}}) \times (d_{\text{model}}, d_{\text{model}})$ matmul |
| **Total** | $8BLd_{\text{model}}^2 + 4BL^2 d_{\text{model}} + 5BhL^2$ | |

The backward pass is approximately $2\text{--}3\times$ the forward pass FLOPs (each matmul has two backward terms).

### Space Complexity

| Tensor | Elements | Note |
|--------|----------|------|
| $Q, K, V$ (after split) | $3 \times BhLd_k = 3BLd_{\text{model}}$ | Same total as before split |
| Attention matrices $A$ | $BhL^2$ | **Dominant for long sequences** |
| Attention output | $BhLd_v = BLd_{\text{model}}$ | Same as Q/K/V total / 3 |
| Concat | $BLd_{\text{model}}$ | After merge |

Total intermediate memory: $O(BLd_{\text{model}} + BhL^2)$.

### The Bottleneck

For typical LLM dimensions ($d_{\text{model}} = 4096$, $L = 2048$, $h = 32$):

- Projection FLOPs: $8 \times B \times 2048 \times 4096^2 \approx 2.7 \times 10^{12} B$
- Attention core FLOPs: $4 \times B \times 2048^2 \times 4096 \approx 6.9 \times 10^{10} B$

Projections dominate by $\sim 40\times$. But for very long sequences ($L > d_{\text{model}}$), the $O(L^2)$ attention core becomes the bottleneck -- this is why Flash Attention targets the attention computation, not the projections.

Memory-wise, the attention matrix $A$ stores $BhL^2 = B \times 32 \times 2048^2 \approx 134M \times B$ elements per layer. At float16, that is $\sim 256$ MB per batch element per layer.

---

## Common Pitfalls

### Pitfall 1: Wrong Transpose Axes in 4D

**The mistake:**

```python
# Wrong: using 3D transpose axes on a 4D tensor
K_transposed = K.transpose(0, 2, 1)  # ERROR or wrong result
scores = Q @ K_transposed
```

**Why it is wrong:** After head splitting, $K$ has shape $(B, h, L, d_k)$ -- four dimensions. Using `.transpose(0, 2, 1)` is either an error (wrong number of axes) or swaps the wrong dimensions. We need to transpose only the last two dimensions ($L$ and $d_k$), keeping batch $B$ and head $h$ as batch dims.

**The fix:**

```python
# Correct: transpose only the last two dimensions
K_transposed = K.transpose(0, 1, 3, 2)  # (B, h, d_k, L)
scores = Q @ K_transposed  # (B, h, L, L) -- batched over B and h
```

### Pitfall 2: Forgetting to Accumulate $\nabla X$ From All Three Paths

**The mistake:**

```python
# Wrong: only backpropagate through Q projection
grad_X = grad_Q_flat @ self.W_Q.T
```

**Why it is wrong:** $X$ feeds into three projections ($Q$, $K$, $V$). By the multivariate chain rule, you must sum the gradients from all three paths. Omitting $K$ and $V$ paths means two-thirds of the gradient signal is lost.

**The fix:**

```python
# Correct: sum gradients from all three projection paths
grad_X = (grad_Q_flat @ self.W_Q.T
        + grad_K_flat @ self.W_K.T
        + grad_V_flat @ self.W_V.T)
```

### Pitfall 3: Reshaping Before Transposing in Head Merge

**The mistake:**

```python
# Wrong: reshape first, then transpose (destroys head grouping)
concat = attn_output.reshape(B, L, d_model).transpose(...)
```

**Why it is wrong:** The attention output has shape $(B, h, L, d_v)$. If you reshape directly to $(B, h \cdot L, d_v)$ or $(B, L, d_{\text{model}})$, you interleave tokens from different heads incorrectly. You must transpose first to put $L$ before $h$, *then* reshape.

**The fix:**

```python
# Correct: transpose THEN reshape
concat = attn_output.transpose(0, 2, 1, 3)  # (B, L, h, d_v)
concat = concat.reshape(B, L, d_model)       # (B, L, d_model)
```

### Pitfall 4: Scaling by $\sqrt{d_{\text{model}}}$ Instead of $\sqrt{d_k}$

**The mistake:**

```python
# Wrong: scaling by full model dimension
scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_model)
```

**Why it is wrong:** Each head operates in $d_k$-dimensional space, not $d_{\text{model}}$-dimensional space. The scaling factor $1/\sqrt{d_k}$ prevents dot products from growing with $d_k$. Using $d_{\text{model}}$ over-scales by $\sqrt{h}$, making attention weights too uniform (softmax inputs too close to zero).

**The fix:**

```python
# Correct: scale by per-head dimension
scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
```

---

## Connection to GQA -- What We Optimize Next

### The KV Cache Problem

During autoregressive generation, we cache $K$ and $V$ tensors from previous tokens to avoid recomputation. With standard MHA, the KV cache stores:

$$\text{KV cache per layer} = 2 \times B \times h \times L \times d_k \times \text{bytes\_per\_element}$$

For Llama 2 70B ($h = 64$, $d_k = 128$, 80 layers), generating a 4096-token sequence at float16:

$$2 \times 1 \times 64 \times 4096 \times 128 \times 2 \times 80 \approx 10.7 \text{ GB}$$

That is 10.7 GB *just for the KV cache* of one sequence.

### Grouped-Query Attention (GQA)

GQA reduces KV cache by sharing $K$ and $V$ heads across groups of $Q$ heads:

| Variant | Q heads | K/V heads | KV cache reduction |
|---------|---------|-----------|---------------------|
| MHA | $h$ | $h$ | $1\times$ (baseline) |
| GQA | $h$ | $g$ | $h/g \times$ |
| MQA | $h$ | $1$ | $h\times$ |

With GQA, $h/g$ query heads share each K/V head. The modification to our implementation would be:

```
MHA:  Q (B, h, L, d_k)   K (B, h, L, d_k)   V (B, h, L, d_k)
GQA:  Q (B, h, L, d_k)   K (B, g, L, d_k)   V (B, g, L, d_k)
                               ^                   ^
                          fewer heads          fewer heads
```

Each group of $h/g$ query heads attends to the same K/V head. Understanding the head dimension from MHA makes this modification straightforward: instead of projecting to $h$ K/V heads, project to $g < h$ heads and broadcast.

---

## Connection to Inference -- Tensor Parallelism

### Why Heads Are the Natural Parallelization Axis

Multi-head attention is embarrassingly parallel across heads. Each head:
1. Uses a slice of $W^Q$, $W^K$, $W^V$ (columns $[i \cdot d_k : (i+1) \cdot d_k]$)
2. Computes its own attention independently
3. Produces output that gets concatenated with other heads

This maps directly to tensor parallelism (Megatron-LM style):

```
                    Input X (replicated on all GPUs)
                              |
            +--------+--------+--------+--------+
            |        |        |        |        |
          GPU 0    GPU 1    GPU 2    GPU 3
        heads 0-3  heads 4-7 heads 8-11 heads 12-15
            |        |        |        |
            |   Column slices of W_Q, W_K, W_V
            |   Each GPU: (d_model, d_model/4)
            |        |        |        |
            v        v        v        v
        Local Q,K,V  Local    Local    Local
        attention    attn     attn     attn
            |        |        |        |
            +--------+--------+--------+--------+
                              |
                         All-Reduce
                              |
                        Output (B, L, d_model)
```

**Column-parallel**: Each GPU holds columns of $W^Q$, $W^K$, $W^V$ for its heads. The input projection becomes a column-parallel matmul.

**Row-parallel**: Each GPU holds rows of $W^O$ corresponding to its heads. The output projection becomes a row-parallel matmul followed by an all-reduce to sum partial results.

### What Gets Optimized vs. What We Implemented

| Our Implementation (Naive) | Production (Optimized) |
|---------------------------|------------------------|
| Full $(B, h, L, L)$ attention matrix materialized | Flash Attention: tile-based, $O(L)$ memory |
| Single-device computation | Tensor parallelism: heads split across GPUs |
| Standard MHA: $h$ K/V heads | GQA: $g < h$ K/V heads, reduced KV cache |
| Full recomputation on each token | KV cache: store and reuse $K$, $V$ |
| float64 for correctness | float16/bfloat16/int8 for throughput |

Understanding the naive version is essential because every optimization is a *targeted modification* to one part of this pipeline. Flash Attention changes how Steps 3-4 execute (tiled softmax, no materialized $A$). GQA changes the K/V projection in Step 1. KV caching changes how $K$ and $V$ are constructed across decoding steps. Tensor parallelism splits Step 2 across devices. Without understanding the full naive pipeline, you cannot reason about where each optimization applies or what tradeoffs it makes.

---

## Testing Your Understanding

### Quick Checks

1. If we used $\sqrt{d_{\text{model}}}$ instead of $\sqrt{d_k}$ as the scaling factor, what would happen to the attention weights? (They would become more uniform -- the softmax inputs would be scaled too small, pushing the distribution toward uniform.)

2. Why does the causal mask have shape $(1, 1, L, L)$ instead of $(B, h, L, L)$? (The causal constraint is identical for all batch elements and all heads. Broadcasting avoids allocating $B \times h$ copies of the same mask.)

3. If input shape is $(4, 128, 512)$ with $h = 8$, what is the shape of the attention matrix $A$? ($(4, 8, 128, 128)$ -- four batch elements, eight heads, $128 \times 128$ attention pattern per head.)

4. Why must $d_{\text{model}}$ be divisible by $h$? (Each head gets $d_k = d_{\text{model}} / h$ dimensions. Non-integer $d_k$ would mean the reshape from $(B, L, d_{\text{model}})$ to $(B, L, h, d_k)$ is impossible.)

### Exercises

1. **Easy**: Modify the forward pass to return the per-head attention weights $A$ as a list of $h$ matrices of shape $(B, L, L)$, for visualization.

2. **Medium**: Implement Multi-Query Attention (MQA) by modifying the class so that $W^K$ and $W^V$ have shape $(d_{\text{model}}, d_k)$ instead of $(d_{\text{model}}, d_{\text{model}})$, and the single K/V head is broadcast to all query heads.

3. **Hard**: Implement a "memory-efficient" forward pass that computes attention one head at a time in a loop (trading compute for memory). Verify the output matches the fused version and measure the memory reduction.

---

## Summary

### Key Takeaways

- Multi-head attention runs $h$ independent attention operations in $d_k$-dimensional subspaces, enabling different heads to learn different relationship types (syntactic, semantic, positional).
- The reshape trick (`reshape` + `transpose`) replaces a loop over heads with a single fused GEMM plus a zero-copy memory reinterpretation, making the implementation both mathematically equivalent and dramatically faster on GPUs.
- Total FLOPs are identical to single-head attention with the same $d_{\text{model}}$ -- the $h$ factor in "number of heads" exactly cancels with the $1/h$ factor in "dimension per head."
- The backward pass requires the multivariate chain rule: $\nabla X$ accumulates from three branches ($Q$, $K$, $V$) because $X$ is used three times in the forward pass.
- The head dimension is the natural axis for tensor parallelism (splitting across GPUs) and for KV cache reduction (GQA/MQA share K/V heads).

### Quick Reference

```
Multi-Head Attention
|-- Forward:  O(8BLd^2 + 4BL^2 d) -- projections + attention core
|-- Backward: O(16BLd^2 + 8BL^2 d) -- ~2x forward (two grads per matmul)
|-- Memory:   O(BLd + BhL^2) -- intermediates + attention matrices
|
|-- Projections: 4 weight matrices, each (d_model, d_model)
|-- Parameters:  4 * d_model^2 (+ 4 * d_model if biased)
|
|-- Key shapes:
|   |-- After split:  (B, h, L, d_k)
|   |-- Scores/A:     (B, h, L, L)
|   |-- After merge:  (B, L, d_model)
|
|-- Optimized by:
    |-- Flash Attention: avoids materializing (B, h, L, L) attention matrix
    |-- Tensor Parallelism: splits heads across GPUs (Megatron-LM)
    |-- GQA/MQA: shares K/V heads to reduce KV cache memory
    |-- KV Cache: reuses K/V from previous tokens during generation
```
