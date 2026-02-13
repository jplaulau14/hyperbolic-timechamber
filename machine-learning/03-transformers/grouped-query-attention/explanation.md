# Grouped-Query Attention -- Deep Dive

## The Intuition

### What Problem Are We Solving?

You have built Multi-Head Attention. It works beautifully: $h$ independent heads each learn different attention patterns, and the model gets rich, diverse representations. But there is a problem hiding in the deployment story.

During autoregressive generation (the token-by-token loop that produces text), the model must store the key and value tensors from every previous token so that each new token can attend to the full history. This storage is called the **KV cache**, and it grows linearly with sequence length, number of heads, and head dimension:

$$\text{KV cache per layer} = 2 \times B \times h \times L \times d_k \times \text{bytes\_per\_element}$$

For Llama 2 70B with standard MHA ($h = 64$, $d_k = 128$, 80 layers, FP16), generating a 4096-token sequence requires:

$$80 \times 2 \times 64 \times 4096 \times 128 \times 2 \approx 107 \text{ GB}$$

That is 107 GB *just for the KV cache of a single request*. This exceeds the memory of even an 80 GB A100 GPU. The model weights themselves take another 140 GB. Serving this model becomes a multi-GPU engineering nightmare where most of the GPU memory is consumed by cached keys and values rather than doing useful computation.

The question is: do all 64 key-value heads need to be independent, or can we share some of them without destroying quality?

### The Key Insight

Not all query heads need their own private key-value heads. Multiple query heads can **share** a single key-value head. Each query head still computes its own unique attention pattern (because $Q$ differs per head), but the keys and values they attend over are the same within a group.

Think about it this way: in standard MHA with 64 heads, heads 0 through 7 might all be learning variations of "attend to nearby tokens." Their query projections differ, but the key-value information they need is largely redundant. By giving them a single shared KV head, we eliminate that redundancy without losing much expressive power. The query heads can still compute *different attention weights* over the shared keys -- they just cannot extract *different information* from the values.

This creates a spectrum controlled by a single parameter, $h_{kv}$ (the number of KV heads):

```
MHA  (h_kv = h)     GQA (1 < h_kv < h)     MQA (h_kv = 1)
No sharing           Groups share            All share
Full memory          Reduced memory          Minimal memory
Full quality         Near-full quality       Some quality loss
```

### Real-World Analogy

Imagine a newsroom with 32 journalists (query heads) and a pool of photographers (KV heads). In standard MHA, every journalist has their own dedicated photographer -- 32 photographers total. Each photographer goes to different locations and captures different scenes. Each journalist then writes their story based on what their photographer captured.

In GQA, the newsroom realizes that groups of 4 journalists covering related beats (politics, foreign affairs, economy, social policy) can share a single photographer. Now 8 photographers serve 32 journalists. Each journalist still writes a unique article (different query), but groups of 4 draw from the same visual evidence (shared keys and values). The newsroom saves 75% on photography costs while barely affecting article quality -- because the journalists in each group needed similar source material anyway.

In MQA, the newsroom goes to the extreme: one photographer, 32 journalists. Every article is based on the same photos. This saves the most resources but risks homogeneity -- if the single photographer missed something, all 32 articles suffer.

---

## The Math, Step by Step

### Building Up to the Full Formula

**Step 1 -- Start with standard MHA** (what we built in the previous topic).

In Multi-Head Attention, we have $h$ query heads, $h$ key heads, and $h$ value heads. The projection matrices all have shape $(d_{model}, d_{model})$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

After reshaping and transposing, $Q$, $K$, and $V$ all have shape $(B, h, L, d_k)$ where $d_k = d_{model} / h$. Each query head $i$ has its own key head $i$ and value head $i$.

**Step 2 -- The redundancy observation.**

Research showed that many KV heads learn similar representations. Shazeer (2019) proposed Multi-Query Attention (MQA): use just one KV head for all query heads. This gave massive memory savings but sometimes hurt quality, because forcing a single KV representation to serve all $h$ query heads is too constraining.

**Step 3 -- The GQA compromise.**

Ainslie et al. (2023) proposed the middle ground: use $h_{kv}$ KV heads where $1 < h_{kv} < h$. Group every $g = h / h_{kv}$ query heads to share one KV head. The projection matrices for K and V *shrink*:

$$W_K \in \mathbb{R}^{d_{model} \times (h_{kv} \cdot d_k)}, \quad W_V \in \mathbb{R}^{d_{model} \times (h_{kv} \cdot d_v)}$$

The only new operation is expanding the $h_{kv}$ KV heads to $h$ heads (by repeating each $g$ times) before computing attention. Everything else is identical to MHA.

### Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| $d_{model}$ | Model dimension | 4096, 8192 |
| $h$ | Number of query heads | 32, 64 |
| $h_{kv}$ | Number of key/value heads | 8 |
| $g = h / h_{kv}$ | Group size (query heads per KV head) | 4, 8 |
| $d_k = d_{model} / h$ | Per-head dimension | 128 |
| $B$ | Batch size | 1, 4 |
| $L$ | Sequence length | 2048, 4096 |

### Projection Weight Shapes

This is where GQA differs from MHA. The Q and output projections remain full-size, but K and V projections shrink:

$$W_Q \in \mathbb{R}^{d_{model} \times d_{model}}, \quad W_O \in \mathbb{R}^{d_{model} \times d_{model}}$$

$$W_K \in \mathbb{R}^{d_{model} \times (h_{kv} \cdot d_k)}, \quad W_V \in \mathbb{R}^{d_{model} \times (h_{kv} \cdot d_v)}$$

When $h_{kv} = h$: $W_K$ is $(d_{model}, d_{model})$ -- standard MHA, no savings.
When $h_{kv} = 8$ and $h = 64$: $W_K$ is $(d_{model}, 8 \cdot d_k)$ -- eight times smaller.
When $h_{kv} = 1$: $W_K$ is $(d_{model}, d_k)$ -- minimal size, MQA.

### The Core Equations

**Equation 1: Q/K/V Projections**

$$Q = X W_Q + b_Q \quad \in \mathbb{R}^{B \times L \times (h \cdot d_k)}$$

$$K = X W_K + b_K \quad \in \mathbb{R}^{B \times L \times (h_{kv} \cdot d_k)}$$

$$V = X W_V + b_V \quad \in \mathbb{R}^{B \times L \times (h_{kv} \cdot d_v)}$$

Note the asymmetry: $Q$ projects to $h \cdot d_k = d_{model}$ features, but $K$ and $V$ project to only $h_{kv} \cdot d_k$ features.

**Equation 2: Head Split**

$$Q \rightarrow (B, L, h, d_k) \xrightarrow{\text{transpose}} (B, h, L, d_k)$$

$$K \rightarrow (B, L, h_{kv}, d_k) \xrightarrow{\text{transpose}} (B, h_{kv}, L, d_k)$$

$$V \rightarrow (B, L, h_{kv}, d_v) \xrightarrow{\text{transpose}} (B, h_{kv}, L, d_v)$$

After this step, $Q$ has $h$ heads but $K$ and $V$ only have $h_{kv}$ heads. We cannot yet compute $Q K^\top$ because the head dimensions do not match.

**Equation 3: KV Head Expansion (the GQA-specific step)**

$$K_{exp} = \text{repeat\_kv}(K, g) \quad \in \mathbb{R}^{B \times h \times L \times d_k}$$

$$V_{exp} = \text{repeat\_kv}(V, g) \quad \in \mathbb{R}^{B \times h \times L \times d_v}$$

Each of the $h_{kv}$ KV heads is copied $g$ times so that the head dimension matches $h$.

**Equation 4: Scaled Dot-Product Attention (identical to standard MHA)**

$$\text{scores} = \frac{Q \cdot K_{exp}^\top}{\sqrt{d_k}} \quad \in \mathbb{R}^{B \times h \times L \times L}$$

$$A = \text{softmax}(\text{scores} + M, \text{axis}=-1) \quad \in \mathbb{R}^{B \times h \times L \times L}$$

$$O = A \cdot V_{exp} \quad \in \mathbb{R}^{B \times h \times L \times d_v}$$

**Equation 5: Head Merge and Output Projection**

$$\text{concat} = O.\text{transpose}(0, 2, 1, 3).\text{reshape}(B, L, d_{model})$$

$$\text{output} = \text{concat} \cdot W_O + b_O \quad \in \mathbb{R}^{B \times L \times d_{model}}$$

### The Three Special Cases

The entire spectrum is controlled by $h_{kv}$:

| Variant | $h_{kv}$ | $g$ | repeat\_kv behavior | $W_K$ shape |
|---------|-----------|-----|---------------------|-------------|
| **MHA** | $h$ | 1 | No-op (returns input unchanged) | $(d_{model}, d_{model})$ |
| **GQA** | $h / r$ for $r > 1$ | $r$ | Each KV head copied $r$ times | $(d_{model}, h_{kv} \cdot d_k)$ |
| **MQA** | 1 | $h$ | Single KV head copied $h$ times | $(d_{model}, d_k)$ |

When $h_{kv} = h$, the `repeat_kv` function returns its input unmodified ($g = 1$), and GQA reduces **exactly** to standard MHA. This is a key correctness test verified in the test suite.

---

## How `repeat_kv` Works -- The Heart of GQA

### Visualizing the Expansion

This is the single operation that distinguishes GQA from MHA. Suppose we have $h = 8$ query heads and $h_{kv} = 2$ KV heads, giving $g = 4$.

Before expansion, K has shape $(B, 2, L, d_k)$ -- just 2 KV heads:

```
K heads:  [  KV_0  |  KV_1  ]
           head 0    head 1
```

After `repeat_kv(K, g=4)`, K_exp has shape $(B, 8, L, d_k)$ -- 8 heads:

```
K_exp:    [  KV_0  |  KV_0  |  KV_0  |  KV_0  |  KV_1  |  KV_1  |  KV_1  |  KV_1  ]
           head 0    head 1    head 2    head 3    head 4    head 5    head 6    head 7
           \_____________ group 0 _____________/  \_____________ group 1 _____________/
```

Now query heads 0-3 all compute attention against the same keys (KV_0), and query heads 4-7 all compute attention against the same keys (KV_1). But each query head still has its own unique $Q$ vectors, so they produce *different attention patterns* over the shared keys.

### Worked Example: repeat_kv with Numbers

Let us trace through a concrete case with $B = 1$, $h_{kv} = 2$, $L = 2$, $d_k = 3$, and $g = 2$.

**Input K:** shape $(1, 2, 2, 3)$

```
KV head 0:                    KV head 1:
  token 0: [1.0, 2.0, 3.0]     token 0: [7.0, 8.0, 9.0]
  token 1: [4.0, 5.0, 6.0]     token 1: [10., 11., 12.]
```

**After `repeat_kv(K, num_repeats=2)`:** shape $(1, 4, 2, 3)$

```
Head 0 (copy of KV_0):       Head 1 (copy of KV_0):
  token 0: [1.0, 2.0, 3.0]     token 0: [1.0, 2.0, 3.0]
  token 1: [4.0, 5.0, 6.0]     token 1: [4.0, 5.0, 6.0]

Head 2 (copy of KV_1):       Head 3 (copy of KV_1):
  token 0: [7.0, 8.0, 9.0]     token 0: [7.0, 8.0, 9.0]
  token 1: [10., 11., 12.]     token 1: [10., 11., 12.]
```

Query heads 0 and 1 will attend against the same key vectors $[1,2,3]$ and $[4,5,6]$. Query heads 2 and 3 will attend against $[7,8,9]$ and $[10,11,12]$. Each query head can still produce a different attention distribution because its $Q$ vectors are unique.

### The NumPy Implementation

```python
def repeat_kv(x: np.ndarray, num_repeats: int) -> np.ndarray:
    """
    Expand KV heads to match query heads.

    Args:
        x: KV tensor, shape (B, h_kv, L, d)
        num_repeats: Group size g = num_heads // num_kv_heads

    Returns:
        Expanded tensor, shape (B, h_kv * num_repeats, L, d)
    """
    if num_repeats == 1:
        return x
    return np.repeat(x, repeats=num_repeats, axis=1)
```

The early return for `num_repeats == 1` is not just an optimization -- it is the path that makes GQA reduce exactly to MHA. When $h_{kv} = h$, $g = 1$, and K and V pass through unchanged.

`np.repeat(x, repeats=g, axis=1)` duplicates each element along axis 1 (the head dimension) $g$ times. If the input has heads $[A, B]$ and $g = 3$, the output is $[A, A, A, B, B, B]$. This interleaving pattern means consecutive query heads share the same KV head.

---

## The Full Forward Pass -- Data Flow with Shapes

```
    Input X
    (B, L, d_model)
         |
    +----+-----+-----+
    |          |          |
    v          v          v
  X @ W_Q    X @ W_K    X @ W_V              Three GEMMs (K, V are smaller)
  (B,L,d_m)  (B,L,h_kv*dk) (B,L,h_kv*dv)
    |          |          |
    v          v          v
  reshape    reshape    reshape
  (B,L,h,dk) (B,L,h_kv,dk) (B,L,h_kv,dv)
    |          |          |
    v          v          v
  transpose  transpose  transpose
  (B,h,L,dk) (B,h_kv,L,dk) (B,h_kv,L,dv)
    |          |          |
    |          v          v
    |       repeat_kv  repeat_kv             *** GQA-specific step ***
    |       (B,h,L,dk) (B,h,L,dv)
    |          |          |
    +----+-----+          |
         |                |
         v                |
    Q @ K_exp^T / sqrt(dk)|                  Batched over B and h
    (B, h, L, L)          |
         |                |
         v                |
    + causal mask         |                  Broadcasting (1,1,L,L)
    (B, h, L, L)          |
         |                |
         v                |
    softmax(axis=-1)      |
    (B, h, L, L)          |
         |                |
         +--------+-------+
                  |
                  v
             A @ V_exp                       Batched matmul
             (B, h, L, dv)
                  |
                  v
             transpose(0,2,1,3)
             (B, L, h, dv)
                  |
                  v
             reshape(B, L, d_model)          Merge heads
             (B, L, d_model)
                  |
                  v
             concat @ W_O + b_O             Output projection
             (B, L, d_model)
```

The diagram is almost identical to standard MHA. The only structural difference is the `repeat_kv` step inserted between the head split and the attention computation. The K and V projections also produce smaller outputs ($h_{kv} \cdot d_k$ instead of $h \cdot d_k$).

### Complete Shape Table

| Step | Operation | Shape |
|------|-----------|-------|
| Input | $X$ | $(B, L, d_{model})$ |
| Q projection | $X W_Q + b_Q$ | $(B, L, h \cdot d_k) = (B, L, d_{model})$ |
| K projection | $X W_K + b_K$ | $(B, L, h_{kv} \cdot d_k)$ |
| V projection | $X W_V + b_V$ | $(B, L, h_{kv} \cdot d_v)$ |
| Reshape Q | | $(B, L, h, d_k)$ |
| Transpose Q | | $(B, h, L, d_k)$ |
| Reshape K | | $(B, L, h_{kv}, d_k)$ |
| Transpose K | | $(B, h_{kv}, L, d_k)$ |
| **Repeat K** | `repeat_kv(K, g)` | $(B, h, L, d_k)$ |
| **Repeat V** | `repeat_kv(V, g)` | $(B, h, L, d_v)$ |
| Scores | $Q K_{exp}^\top / \sqrt{d_k}$ | $(B, h, L, L)$ |
| Mask | scores $+ M$ | $(B, h, L, L)$ |
| Attention weights | $\text{softmax}(\text{scores})$ | $(B, h, L, L)$ |
| Value weighting | $A \cdot V_{exp}$ | $(B, h, L, d_v)$ |
| Transpose back | | $(B, L, h, d_v)$ |
| Reshape (merge) | | $(B, L, d_{model})$ |
| Output projection | $\text{concat} \cdot W_O + b_O$ | $(B, L, d_{model})$ |

---

## From Math to Code

### Implementation Walkthrough: Forward Pass

```python
def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    B, L, _ = X.shape

    # Step 1: Three GEMMs -- Q is full-size, K and V are smaller
    Q = X @ self.W_Q + self.b_Q       # (B, L, d_model) @ (d_model, d_model) -> (B, L, d_model)
    K = X @ self.W_K + self.b_K       # (B, L, d_model) @ (d_model, h_kv*d_k) -> (B, L, h_kv*d_k)
    V = X @ self.W_V + self.b_V       # (B, L, d_model) @ (d_model, h_kv*d_v) -> (B, L, h_kv*d_v)

    # Step 2: Reshape and transpose to separate heads
    Q = Q.reshape(B, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)     # (B, h, L, d_k)
    K = K.reshape(B, L, self.num_kv_heads, self.d_k).transpose(0, 2, 1, 3)  # (B, h_kv, L, d_k)
    V = V.reshape(B, L, self.num_kv_heads, self.d_v).transpose(0, 2, 1, 3)  # (B, h_kv, L, d_v)

    # Step 3: Expand KV heads to match query heads -- THE GQA STEP
    K_exp = repeat_kv(K, self.group_size)   # (B, h_kv, L, d_k) -> (B, h, L, d_k)
    V_exp = repeat_kv(V, self.group_size)   # (B, h_kv, L, d_v) -> (B, h, L, d_v)

    # Step 4: Standard scaled dot-product attention (identical to MHA from here)
    scores = Q @ K_exp.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)  # (B, h, L, L)

    if mask is not None:
        scores = scores + mask

    A = softmax(scores, axis=-1)                # (B, h, L, L)
    attn_output = A @ V_exp                     # (B, h, L, d_v)

    # Step 5: Merge heads and output projection
    concat = attn_output.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)  # (B, L, d_model)
    output = concat @ self.W_O + self.b_O       # (B, L, d_model)

    # Cache everything needed for backward
    self._cache = {
        "X": X, "Q": Q, "K_exp": K_exp, "V_exp": V_exp,
        "A": A, "attn_output": attn_output, "concat": concat, "mask": mask,
    }
    return output
```

**Line-by-line commentary:**

- `X @ self.W_K + self.b_K`: This is the K projection. The key difference from MHA is that `self.W_K` has shape $(d_{model}, h_{kv} \cdot d_k)$, which is *smaller* than $(d_{model}, d_{model})$ when $h_{kv} < h$. This is where the parameter savings occur.

- `K.reshape(B, L, self.num_kv_heads, self.d_k)`: We reshape K using `num_kv_heads`, not `num_heads`. The last dimension $h_{kv} \cdot d_k$ splits into $h_{kv}$ groups of $d_k$.

- `K_exp = repeat_kv(K, self.group_size)`: This is the entire GQA mechanism. It expands $(B, h_{kv}, L, d_k)$ to $(B, h, L, d_k)$ by repeating each KV head $g$ times. After this line, $K_{exp}$ has the same shape as $Q$, and the rest of the computation is standard MHA.

- `self._cache` stores `K_exp` and `V_exp` (the expanded versions), not the original $K$ and $V$. The backward pass needs the expanded versions for computing $\nabla Q$ and $\nabla A$.

### The Tricky Parts

**Why does the cache store `K_exp` instead of `K`?**

In the backward pass, we need $K_{exp}$ to compute $\nabla Q = \frac{g_{\text{raw}}}{\sqrt{d_k}} \cdot K_{exp}$. We could reconstruct `K_exp` from `K` during backward, but that would mean running `repeat_kv` again. It is simpler and more explicit to cache the expanded version. The tradeoff is extra memory ($h$ heads stored instead of $h_{kv}$).

**Why `.transpose(0, 1, 3, 2)` on `K_exp`?**

$K_{exp}$ has shape $(B, h, L, d_k)$. For the batched matmul $Q K_{exp}^\top$, we need to transpose only the last two axes ($L$ and $d_k$), keeping $B$ and $h$ as batch dimensions. `.transpose(0, 1, 3, 2)` swaps axes 2 and 3 while leaving 0 and 1 in place.

**Why add the mask rather than multiply?**

Adding $-10^9$ before softmax gives $e^{-10^9} \approx 0$, which naturally zeros out future positions while keeping the attention weights summing to 1. A multiplicative mask after softmax would break normalization.

---

## The Backward Pass -- Step by Step

The backward pass follows MHA exactly, with one critical addition: the gradient through `repeat_kv` must accumulate gradients from all query heads in a group back to the shared KV head.

Let $g_{\text{out}} = \frac{\partial \mathcal{L}}{\partial \text{output}} \in \mathbb{R}^{B \times L \times d_{model}}$.

### Step 1: Gradient Through Output Projection

Forward: $\text{output} = \text{concat} \cdot W_O + b_O$

$$\frac{\partial \mathcal{L}}{\partial \text{concat}} = g_{\text{out}} \cdot W_O^\top \quad \in \mathbb{R}^{B \times L \times d_{model}}$$

$$\frac{\partial \mathcal{L}}{\partial W_O} = \sum_{b,l} \text{concat}_{b,l}^\top \cdot g_{\text{out},b,l} \quad \in \mathbb{R}^{d_{model} \times d_{model}}$$

$$\frac{\partial \mathcal{L}}{\partial b_O} = \sum_{b,l} g_{\text{out},b,l} \quad \in \mathbb{R}^{d_{model}}$$

```python
grad_concat = grad_output @ self.W_O.T
self.grad_W_O = np.einsum("blm,bln->mn", concat, grad_output)
self.grad_b_O = grad_output.sum(axis=(0, 1))
```

### Step 2: Gradient Through Head Merge

Forward: transpose$(0, 2, 1, 3)$ then reshape$(B, L, d_{model})$.

Backward reverses both operations:

$$g_{\text{attn\_out}} = g_{\text{concat}}.\text{reshape}(B, L, h, d_v).\text{transpose}(0, 2, 1, 3) \quad \in \mathbb{R}^{B \times h \times L \times d_v}$$

```python
grad_attn_output = grad_concat.reshape(B, L, self.num_heads, self.d_v).transpose(0, 2, 1, 3)
```

### Step 3: Gradient Through Value Weighting

Forward: $\text{attn\_output} = A \cdot V_{exp}$

$$\frac{\partial \mathcal{L}}{\partial A} = g_{\text{attn\_out}} \cdot V_{exp}^\top \quad \in \mathbb{R}^{B \times h \times L \times L}$$

$$\frac{\partial \mathcal{L}}{\partial V_{exp}} = A^\top \cdot g_{\text{attn\_out}} \quad \in \mathbb{R}^{B \times h \times L \times d_v}$$

```python
grad_A = grad_attn_output @ V_exp.transpose(0, 1, 3, 2)
grad_V_exp = A.transpose(0, 1, 3, 2) @ grad_attn_output
```

Note: $\nabla V_{exp}$ has shape $(B, h, L, d_v)$ -- it has $h$ heads, not $h_{kv}$. This is the gradient with respect to the *expanded* V, not the original.

### Step 4: Gradient Through Softmax

$$\frac{\partial \mathcal{L}}{\partial \text{scores}} = A \odot \left(g_A - \sum_j g_{A} \odot A \right)$$

```python
grad_scores = softmax_backward(grad_A, A)
```

### Step 5: Gradient Through Scaling and $QK_{exp}^\top$

Forward: $\text{scores} = Q \cdot K_{exp}^\top / \sqrt{d_k}$

$$g_{\text{raw}} = \frac{g_{\text{scores}}}{\sqrt{d_k}}$$

$$\frac{\partial \mathcal{L}}{\partial Q} = g_{\text{raw}} \cdot K_{exp} \quad \in \mathbb{R}^{B \times h \times L \times d_k}$$

$$\frac{\partial \mathcal{L}}{\partial K_{exp}} = g_{\text{raw}}^\top \cdot Q \quad \in \mathbb{R}^{B \times h \times L \times d_k}$$

```python
grad_raw = grad_scores / scale
grad_Q = grad_raw @ K_exp                        # (B, h, L, L) @ (B, h, L, d_k)
grad_K_exp = grad_raw.transpose(0, 1, 3, 2) @ Q  # (B, h, L, L) @ (B, h, L, d_k)
```

At this point, $\nabla K_{exp}$ has shape $(B, h, L, d_k)$ -- it has $h$ heads. But the original $K$ only has $h_{kv}$ heads. We need to "undo" the repeat operation.

### Step 6: Gradient Through repeat_kv (The GQA-Specific Step)

This is the key difference from the MHA backward pass. During the forward pass, `repeat_kv` duplicated each KV head $g$ times:

$$K_{exp}[:, i, :, :] = K[:, \lfloor i/g \rfloor, :, :] \quad \text{for } i = 0, \ldots, h-1$$

The reverse of duplication is **summation**. If one KV head was used by $g$ query heads, the gradient for that KV head is the sum of the gradients from all $g$ query heads that used it:

$$\frac{\partial \mathcal{L}}{\partial K}[:, j, :, :] = \sum_{i=jg}^{(j+1)g - 1} \frac{\partial \mathcal{L}}{\partial K_{exp}}[:, i, :, :]$$

The implementation achieves this with a reshape and sum:

```python
def reduce_kv_grad(grad_expanded, num_kv_heads, group_size):
    if group_size == 1:
        return grad_expanded
    B, _, L, d = grad_expanded.shape
    return grad_expanded.reshape(B, num_kv_heads, group_size, L, d).sum(axis=2)
```

**Why does reshaping work here?** The `repeat_kv` function placed heads in the order $[KV_0, KV_0, \ldots, KV_0, KV_1, KV_1, \ldots]$ -- each KV head repeated $g$ times contiguously. So reshaping $(B, h, L, d)$ into $(B, h_{kv}, g, L, d)$ groups the $g$ copies of each KV head together along axis 2. Summing over axis 2 collapses each group back into a single gradient.

### Visualizing the Gradient Accumulation

For $h = 8$, $h_{kv} = 2$, $g = 4$:

```
grad_K_exp (B, 8, L, d_k):  [g0  g1  g2  g3 | g4  g5  g6  g7]
                              \__ group 0 __/   \__ group 1 __/

reshape to (B, 2, 4, L, d_k):
  group 0: [g0, g1, g2, g3]     group 1: [g4, g5, g6, g7]

sum over axis 2:
  grad_K (B, 2, L, d_k):   [g0+g1+g2+g3 | g4+g5+g6+g7]
                             \_ KV head 0_/ \_ KV head 1_/
```

Each KV head receives the **sum** of gradients from all the query heads that shared it.

```python
grad_K = reduce_kv_grad(grad_K_exp, self.num_kv_heads, self.group_size)
grad_V = reduce_kv_grad(grad_V_exp, self.num_kv_heads, self.group_size)
```

### Step 7: Gradient Through Head Split and Projections

After reducing to $h_{kv}$ heads, reverse the transpose and reshape:

$$\nabla K_{\text{flat}} = \nabla K.\text{transpose}(0, 2, 1, 3).\text{reshape}(B, L, h_{kv} \cdot d_k)$$

Note the reshape target is $h_{kv} \cdot d_k$, not $d_{model}$. This matches the shape of the K projection output.

```python
grad_Q_flat = grad_Q.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
grad_K_flat = grad_K.transpose(0, 2, 1, 3).reshape(B, L, self.num_kv_heads * self.d_k)
grad_V_flat = grad_V.transpose(0, 2, 1, 3).reshape(B, L, self.num_kv_heads * self.d_v)
```

Weight gradients:

$$\frac{\partial \mathcal{L}}{\partial W_Q} = \sum_{b,l} X_{b,l}^\top \cdot \nabla Q_{\text{flat},b,l} \quad \in \mathbb{R}^{d_{model} \times d_{model}}$$

$$\frac{\partial \mathcal{L}}{\partial W_K} = \sum_{b,l} X_{b,l}^\top \cdot \nabla K_{\text{flat},b,l} \quad \in \mathbb{R}^{d_{model} \times (h_{kv} \cdot d_k)}$$

```python
self.grad_W_Q = np.einsum("blm,bld->md", X, grad_Q_flat)
self.grad_W_K = np.einsum("blm,bld->md", X, grad_K_flat)
self.grad_W_V = np.einsum("blm,bld->md", X, grad_V_flat)
```

Input gradient (three-way accumulation, same as MHA):

$$\frac{\partial \mathcal{L}}{\partial X} = \nabla Q_{\text{flat}} \cdot W_Q^\top + \nabla K_{\text{flat}} \cdot W_K^\top + \nabla V_{\text{flat}} \cdot W_V^\top$$

```python
grad_X = grad_Q_flat @ self.W_Q.T + grad_K_flat @ self.W_K.T + grad_V_flat @ self.W_V.T
```

Note that $\nabla K_{\text{flat}} \cdot W_K^\top$ has shape $(B, L, h_{kv} \cdot d_k) \times (h_{kv} \cdot d_k, d_{model}) = (B, L, d_{model})$. The dimension reduction from $h$ to $h_{kv}$ was already handled by `reduce_kv_grad`, so the shapes work out correctly.

### What Gets Cached and Why

| Cached Tensor | Shape | Used in Backward Step |
|---------------|-------|-----------------------|
| $X$ | $(B, L, d_{model})$ | Step 7: $\nabla W_Q$, $\nabla W_K$, $\nabla W_V$ |
| $Q$ | $(B, h, L, d_k)$ | Step 5: $\nabla K_{exp} = g_{\text{raw}}^\top \cdot Q$ |
| $K_{exp}$ | $(B, h, L, d_k)$ | Step 5: $\nabla Q = g_{\text{raw}} \cdot K_{exp}$ |
| $V_{exp}$ | $(B, h, L, d_v)$ | Step 3: $\nabla A = g_{\text{attn\_out}} \cdot V_{exp}^\top$ |
| $A$ | $(B, h, L, L)$ | Step 3: $\nabla V_{exp}$; Step 4: softmax backward |
| concat | $(B, L, d_{model})$ | Step 1: $\nabla W_O$ |

---

## Worked Example: Full GQA Forward Pass

### Setup

$d_{model} = 8$, $h = 4$, $h_{kv} = 2$, $g = 2$, $L = 3$, $B = 1$, $d_k = d_{model} / h = 2$.

**Input:** $X = \mathbf{1}_{1 \times 3 \times 8}$ (all ones, for simplicity).

**Weights** (simplified for tractability):

$$W_Q = 0.1 \cdot I_8, \quad W_K = 0.05 \cdot \mathbf{1}_{8 \times 4}, \quad W_V = 0.05 \cdot \mathbf{1}_{8 \times 4}, \quad W_O = 0.1 \cdot I_8$$

All biases are zero. This matches the `test_known_small_config` test case.

### Step 1: Projections

$$Q = X \cdot W_Q = \mathbf{1}_{1 \times 3 \times 8} \cdot (0.1 \cdot I_8) = 0.1 \cdot \mathbf{1}_{1 \times 3 \times 8}$$

Every element of $Q$ is 0.1.

$$K = X \cdot W_K = \mathbf{1}_{1 \times 3 \times 8} \cdot (0.05 \cdot \mathbf{1}_{8 \times 4})$$

Each element of $K$: $\sum_{i=1}^{8} 1.0 \times 0.05 = 0.4$. So $K$ is $0.4 \cdot \mathbf{1}_{1 \times 3 \times 4}$.

Similarly, $V = 0.4 \cdot \mathbf{1}_{1 \times 3 \times 4}$.

### Step 2: Head Split

$Q$: reshape $(1, 3, 8) \rightarrow (1, 3, 4, 2)$, transpose $\rightarrow (1, 4, 3, 2)$. Each head's $d_k = 2$ values are $[0.1, 0.1]$.

$K$: reshape $(1, 3, 4) \rightarrow (1, 3, 2, 2)$, transpose $\rightarrow (1, 2, 3, 2)$. Each KV head's values are $[0.4, 0.4]$.

$V$: same as $K$: $(1, 2, 3, 2)$ with all values 0.4.

### Step 3: repeat_kv

$K_{exp} = \text{repeat\_kv}(K, g=2)$: $(1, 2, 3, 2) \rightarrow (1, 4, 3, 2)$

KV head 0 is copied to heads 0, 1. KV head 1 is copied to heads 2, 3. All values remain 0.4.

$V_{exp}$: same expansion, all values 0.4.

### Step 4: Attention

For each of the 4 query heads and each query position:

$$\text{score}_{i,j} = \frac{Q_i \cdot K_j^\top}{\sqrt{2}} = \frac{[0.1, 0.1] \cdot [0.4, 0.4]^\top}{\sqrt{2}} = \frac{0.04 + 0.04}{\sqrt{2}} = \frac{0.08}{1.414} \approx 0.0566$$

Since all input tokens are identical, every score is 0.0566. The score matrix for each head is:

$$\text{scores} = \begin{bmatrix} 0.0566 & 0.0566 & 0.0566 \\ 0.0566 & 0.0566 & 0.0566 \\ 0.0566 & 0.0566 & 0.0566 \end{bmatrix}$$

After softmax (all inputs equal): $A_{i,j} = 1/3$ for all $i, j$.

### Step 5: Value Weighting

$$O_i = \sum_j A_{i,j} \cdot V_j = \frac{1}{3}(V_0 + V_1 + V_2) = \frac{1}{3}(3 \times [0.4, 0.4]) = [0.4, 0.4]$$

Every head, every position produces $[0.4, 0.4]$.

### Step 6: Merge and Output

Merge: $(1, 4, 3, 2) \rightarrow (1, 3, 8)$, all values 0.4.

Output: $0.4 \cdot \mathbf{1}_{1 \times 3 \times 8} \cdot (0.1 \cdot I_8) = 0.04 \cdot \mathbf{1}_{1 \times 3 \times 8}$

**Key verification:** All three token positions produce the same output (since all inputs are identical). The test checks `np.allclose(out[0, 0], out[0, 1])`.

---

## Complexity Analysis

### Time Complexity

| Operation | MHA FLOPs | GQA FLOPs | Why |
|-----------|-----------|-----------|-----|
| Q projection | $2BL d_{model}^2$ | $2BL d_{model}^2$ | Same: full $Q$ always needed |
| K projection | $2BL d_{model}^2$ | $2BL d_{model} (h_{kv} d_k)$ | GQA: smaller $W_K$ |
| V projection | $2BL d_{model}^2$ | $2BL d_{model} (h_{kv} d_v)$ | GQA: smaller $W_V$ |
| O projection | $2BL d_{model}^2$ | $2BL d_{model}^2$ | Same: full output projection |
| $QK^\top$ | $2BhL^2 d_k$ | $2BhL^2 d_k$ | **Same**: $h$ query heads regardless |
| Softmax | $5BhL^2$ | $5BhL^2$ | **Same**: applied to all $h$ heads |
| $AV$ | $2BhL^2 d_v$ | $2BhL^2 d_v$ | **Same**: $h$ attention patterns |
| repeat\_kv | $O(Bh_{kv}gLd_k)$ | - | Memory copy, not matmul; negligible |

The attention core FLOPs ($QK^\top$, softmax, $AV$) are **identical** for MHA and GQA. Only the projection FLOPs differ, because $W_K$ and $W_V$ are smaller.

### Space Complexity

| Tensor | MHA | GQA |
|--------|-----|-----|
| $W_K$, $W_V$ parameters | $2 d_{model}^2$ | $2 d_{model} \cdot h_{kv} \cdot d_k$ |
| K, V after projection | $2BL \cdot d_{model}$ | $2BL \cdot h_{kv} \cdot d_k$ |
| K\_exp, V\_exp (expanded) | $2BhLd_k$ | $2BhLd_k$ (same after expansion) |
| Attention matrix $A$ | $BhL^2$ | $BhL^2$ (same: $h$ query heads) |
| **KV cache** (inference) | $2BhLd_k$ | $2Bh_{kv}Ld_k$ |

The KV cache savings are the whole point: a factor of $g = h / h_{kv}$ reduction.

### Comparing MHA vs GQA vs MQA

For Llama 2 70B dimensions ($d_{model} = 8192$, $h = 64$, $d_k = 128$, $L = 4096$):

| Metric | MHA ($h_{kv}=64$) | GQA ($h_{kv}=8$) | MQA ($h_{kv}=1$) |
|--------|-------------------|-------------------|-------------------|
| $W_K$ parameters | 67M | 8.4M | 1.0M |
| $W_V$ parameters | 67M | 8.4M | 1.0M |
| KV cache/layer (FP16) | 134 MB | 16.8 MB | 2.1 MB |
| KV cache total (80 layers) | 10.5 GB | 1.3 GB | 164 MB |
| Attention FLOPs | $4BhL^2 d_k$ | $4BhL^2 d_k$ | $4BhL^2 d_k$ |
| Projection FLOPs (K+V) | $4BLd^2$ | $0.5BLd^2$ | $0.0625BLd^2$ |

The attention core FLOPs are identical across all three -- the savings come entirely from the smaller K/V projections and the reduced KV cache.

### The Bottleneck

During **prefill** (processing the prompt in parallel), the computation is compute-bound -- large batch matmuls keep the GPU busy. GQA provides modest savings from smaller $W_K$ and $W_V$ projections.

During **decode** (generating tokens one at a time), the computation is memory-bandwidth-bound. Each new token requires reading the entire KV cache from GPU memory. With GQA, reading $h_{kv}$ KV heads instead of $h$ reduces memory bandwidth by factor $g$. Since decode is bandwidth-limited, this translates almost linearly to throughput improvement.

---

## Real-World Configurations

### Llama 2 70B

| Parameter | Value |
|-----------|-------|
| $d_{model}$ | 8192 |
| $h$ (query heads) | 64 |
| $h_{kv}$ (KV heads) | 8 |
| $g$ (group size) | 8 |
| $d_k$ | 128 |
| Layers | 80 |
| Context length | 4096 |

```python
# KV cache with GQA (actual):
>>> kv_cache_size_model(1, 4096, 80, 8, 128, "float16")
13,421,772,800  # ~13.4 GB

# KV cache if MHA had been used (hypothetical):
>>> kv_cache_size_model(1, 4096, 80, 64, 128, "float16")
107,374,182,400  # ~107 GB

# 8x reduction: the difference between 2 GPUs and 1 GPU for the KV cache
```

### Mistral 7B

| Parameter | Value |
|-----------|-------|
| $d_{model}$ | 4096 |
| $h$ (query heads) | 32 |
| $h_{kv}$ (KV heads) | 8 |
| $g$ (group size) | 4 |
| $d_k$ | 128 |
| Layers | 32 |
| Context length | 8192 (with sliding window) |

```python
# KV cache with GQA:
>>> kv_cache_size_model(1, 8192, 32, 8, 128, "float16")
4,294,967,296  # ~4.3 GB

# 4x reduction compared to MHA
```

### Llama 2 7B (uses standard MHA)

Interestingly, the smaller Llama 2 7B uses standard MHA ($h_{kv} = 32 = h$). The KV cache is manageable at this scale, and full MHA preserves maximum model quality. GQA was only applied to the 34B and 70B variants where the KV cache would otherwise be prohibitive.

---

## Common Pitfalls

### Pitfall 1: Forgetting to Sum Gradients Across the Group

**The mistake:**

```python
# Wrong: just reshape, forgetting to sum over the group dimension
grad_K = grad_K_exp.reshape(B, num_kv_heads, group_size, L, d_k)[:, :, 0, :, :]
```

**Why it is wrong:** This takes only the gradient from the first query head in each group, discarding the contributions from the other $g - 1$ query heads. Since all $g$ query heads in a group used the same K during the forward pass, the chain rule requires summing all $g$ gradient contributions. Dropping them means the KV heads receive only $1/g$ of their true gradient, causing slow and incorrect training.

**The fix:**

```python
# Correct: reshape AND sum over group dimension
grad_K = grad_K_exp.reshape(B, num_kv_heads, group_size, L, d_k).sum(axis=2)
```

### Pitfall 2: Reshaping grad_K_flat to the Wrong Size

**The mistake:**

```python
# Wrong: reshape to d_model instead of h_kv * d_k
grad_K_flat = grad_K.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
```

**Why it is wrong:** After `reduce_kv_grad`, $\nabla K$ has shape $(B, h_{kv}, L, d_k)$. Reshaping the last two dimensions gives $h_{kv} \cdot d_k$, which equals $d_{model}$ only when $h_{kv} = h$. For GQA with $h_{kv} < h$, $h_{kv} \cdot d_k < d_{model}$, and this reshape would produce the wrong shape or silently flatten more dimensions than intended.

**The fix:**

```python
# Correct: use the actual KV projection dimension
grad_K_flat = grad_K.transpose(0, 2, 1, 3).reshape(B, L, self.num_kv_heads * self.d_k)
```

### Pitfall 3: Using num_heads Instead of num_kv_heads for K/V Reshape

**The mistake:**

```python
# Wrong: using h instead of h_kv for the K reshape
K = K.reshape(B, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
```

**Why it is wrong:** K was projected to $h_{kv} \cdot d_k$ features, not $h \cdot d_k$. Using `num_heads` in the reshape would either fail (if $h_{kv} \cdot d_k \ne h \cdot d_k$) or silently produce a tensor with incorrect head assignments. This is the most common copy-paste error when adapting MHA code to GQA.

**The fix:**

```python
# Correct: K has h_kv heads, not h
K = K.reshape(B, L, self.num_kv_heads, self.d_k).transpose(0, 2, 1, 3)
```

### Pitfall 4: Not Handling the g = 1 Case in repeat_kv

**The mistake:**

```python
# Wasteful: always repeat, even when g = 1
def repeat_kv(x, num_repeats):
    return np.repeat(x, repeats=num_repeats, axis=1)
```

**Why it is problematic:** When $h_{kv} = h$ (standard MHA), $g = 1$ and `np.repeat(x, 1, axis=1)` returns a *copy* of the input rather than the input itself. This wastes memory (doubling K and V storage) and breaks the exact numerical equivalence with MHA. The implementation would still be mathematically correct, but the unnecessary copy affects performance and complicates gradient verification.

**The fix:**

```python
# Correct: early return avoids unnecessary copy
def repeat_kv(x, num_repeats):
    if num_repeats == 1:
        return x
    return np.repeat(x, repeats=num_repeats, axis=1)
```

---

## Connection to Inference Optimization

### KV Cache: The Primary Motivation

GQA exists because of the KV cache. During autoregressive generation, each new token must attend to all previous tokens. Without caching, this requires recomputing all K and V projections for the entire sequence at every step -- $O(L)$ work per token, $O(L^2)$ total. The KV cache stores the K and V tensors so each token only needs one new K and one new V computation.

The KV cache stores K and V in their *pre-expansion* form -- shape $(B, h_{kv}, L, d_k)$, not $(B, h, L, d_k)$. The `repeat_kv` expansion happens at attention computation time, and in optimized implementations, it is done via broadcasting without materializing the expanded tensor.

```
Standard MHA KV Cache:              GQA KV Cache:

  Layer 0: K (B, h, L, d_k)          Layer 0: K (B, h_kv, L, d_k)
           V (B, h, L, d_v)                   V (B, h_kv, L, d_v)
  Layer 1: K (B, h, L, d_k)          Layer 1: K (B, h_kv, L, d_k)
           V (B, h, L, d_v)                   V (B, h_kv, L, d_v)
  ...                                 ...
  Layer N: K (B, h, L, d_k)          Layer N: K (B, h_kv, L, d_k)
           V (B, h, L, d_v)                   V (B, h_kv, L, d_v)

  Total: 2 * N * B * h * L * d_k      Total: 2 * N * B * h_kv * L * d_k
                                       Savings: h / h_kv = g times smaller
```

### From Naive to Optimized

| Aspect | Naive (what we implemented) | Production (optimized) |
|--------|---------------------------|------------------------|
| `repeat_kv` | `np.repeat` creates full copy | Broadcasting: no materialized expansion |
| KV storage | Cache `K_exp` (expanded) | Cache K (compact, $h_{kv}$ heads) |
| Attention matrix | Full $(B, h, L, L)$ materialized | Flash Attention: tiled, $O(L)$ memory |
| Precision | float64 (for gradient checking) | float16 / bfloat16 / int8 |
| Devices | Single-device | Tensor parallelism: heads split across GPUs |

### Broadcasting vs. Materializing

In our NumPy implementation, `repeat_kv` physically copies data:

```python
K_exp = np.repeat(K, repeats=g, axis=1)  # Allocates new array of size B*h*L*d_k
```

In PyTorch, the equivalent can be done without copying:

```python
# PyTorch: expand does NOT allocate memory -- uses stride tricks
K_exp = K[:, :, None, :, :].expand(B, h_kv, g, L, d_k).reshape(B, h, L, d_k)
```

This is important because the expanded K is $g$ times larger than the original. For Llama 2 70B with $g = 8$, materializing the expansion would consume 8x more memory for the K and V intermediates.

### Interaction with Other Optimizations

| Optimization | How GQA Interacts |
|-------------|-------------------|
| **KV Cache** | Stores $(B, h_{kv}, L, d_k)$ instead of $(B, h, L, d_k)$. $g$x memory savings. |
| **PagedAttention (vLLM)** | Smaller KV per token means more tokens per page, less fragmentation. |
| **Flash Attention** | Unchanged -- operates per head. Just fewer unique KV heads to process. |
| **Tensor Parallelism** | KV heads may be fewer than GPU count. With 8 KV heads and 8 GPUs, each GPU gets 1 KV head but 4-8 query heads. |
| **Speculative Decoding** | Smaller KV cache means cheaper verification of draft tokens. |
| **Quantization** | KV cache quantization (e.g., INT8 KV) compounds with GQA: $g$x from GQA $\times$ 2x from INT8 = $2g$x total reduction. |

### Why Not Always Use MQA?

MQA ($h_{kv} = 1$) gives maximum memory savings -- $h$x reduction. But empirical results show quality degradation, especially on tasks requiring diverse attention patterns. With a single KV head, all query heads attend over the same key-value representation. They can compute *different attention weights* (because $Q$ differs), but the information they can *extract* from V is constrained to one shared representation.

GQA with $h_{kv} = 8$ preserves 8 independent KV representations. Each KV head can specialize (e.g., one for local patterns, one for syntactic relations, one for semantic similarity). The quality-efficiency tradeoff is empirically favorable: Llama 2 70B with GQA matches or exceeds MHA baselines while using 8x less KV cache.

---

## Analysis Functions Walkthrough

The implementation includes utility functions for computing parameter counts, KV cache sizes, and FLOPs. These connect the math directly to deployment decisions.

### Parameter Counting

```python
def count_parameters(d_model: int, num_heads: int, num_kv_heads: int) -> Dict[str, int]:
    d_k = d_model // num_heads
    wq = d_model * d_model                    # W_Q is always full-size
    wk = d_model * (num_kv_heads * d_k)       # W_K shrinks with fewer KV heads
    wv = d_model * (num_kv_heads * d_k)       # W_V shrinks with fewer KV heads
    wo = d_model * d_model                    # W_O is always full-size
    # ... biases and totals
```

For Llama 2 70B ($d_{model} = 8192$, $h = 64$, $h_{kv} = 8$, $d_k = 128$):
- $W_Q$: $8192 \times 8192 = 67$M parameters
- $W_K$: $8192 \times (8 \times 128) = 8192 \times 1024 = 8.4$M parameters
- $W_V$: $8.4$M parameters
- $W_O$: $67$M parameters

Total weight savings: $(67 + 8.4 + 8.4 + 67)$M vs $(67 + 67 + 67 + 67)$M $= 150.8$M vs $268$M $\approx 44\%$ fewer parameters in the attention layer.

### FLOPs Breakdown

```python
def count_flops(batch_size, seq_len, d_model, num_heads, num_kv_heads):
    # ...
    attn_qk = 2 * B * h * L * L * d_k     # Uses num_heads (h), not num_kv_heads
    attn_av = 2 * B * h * L * L * d_k     # Same: h query heads attend
    # ...
```

The attention core FLOPs use $h$ (query heads), not $h_{kv}$. This is because every query head computes its own attention scores and value-weighted sum, regardless of whether the keys and values are shared. The FLOPs are identical for MHA and GQA -- the test suite verifies this:

```python
def test_attention_core_same_for_mha_and_gqa(self):
    flops_mha = count_flops(B, L, d_model, h, h)
    flops_gqa = count_flops(B, L, d_model, h, 2)
    self.assertEqual(flops_mha["attn_qk"], flops_gqa["attn_qk"])
    self.assertEqual(flops_mha["attn_av"], flops_gqa["attn_av"])
```

---

## Testing Your Understanding

### Quick Checks

1. **If $h_{kv} = h$, what does `repeat_kv` return?** It returns the input unchanged (early return for $g = 1$). The entire GQA module reduces to standard MHA.

2. **Why does $\nabla K$ require summing over the group dimension, but $\nabla Q$ does not?** Because each K head was *shared* by $g$ query heads during the forward pass. By the chain rule, the gradient for a variable that feeds into multiple consumers is the sum of gradients from each consumer. Each Q head is used exactly once, so no summation is needed.

3. **What is the attention matrix shape for GQA with $h = 32$, $h_{kv} = 8$?** Still $(B, 32, L, L)$. Every query head produces its own attention pattern, even though groups of 4 share the same keys and values.

4. **Why are the attention core FLOPs identical for MHA and GQA?** The $QK^\top$ and $AV$ matmuls involve $h$ query heads regardless. Each query head computes a $(L, d_k) \times (d_k, L)$ matmul, and there are $h$ such matmuls. The fact that some heads share K/V does not reduce the number of matmuls -- it only reduces the amount of *unique* K/V data.

5. **What is the KV cache memory ratio between Llama 2 70B's actual GQA design and a hypothetical MHA version?** $h / h_{kv} = 64 / 8 = 8$x. GQA uses 8x less KV cache memory.

### Exercises

1. **Easy**: Verify that `repeat_kv` followed by `reduce_kv_grad` recovers the original tensor scaled by $g$. Explain why the scaling factor is $g$ (hint: each element appears in $g$ copies, so summing gives $g$ times the original).

2. **Medium**: Modify the `GroupedQueryAttention` class to accept a `head_groups` parameter that specifies *non-uniform* group sizes (e.g., heads 0-5 share KV head 0, heads 6-7 share KV head 1). Implement the custom grouping in both forward and backward passes.

3. **Hard**: Implement a version of `repeat_kv` that uses NumPy broadcasting (via `np.broadcast_to` and reshaping) instead of `np.repeat`, so that no data is actually copied. Then adapt the backward pass to work without `reduce_kv_grad` by using the broadcasting structure directly. Compare memory usage.

---

## Summary

### Key Takeaways

- Grouped-Query Attention is a single-parameter modification to Multi-Head Attention: reduce the number of KV heads from $h$ to $h_{kv}$, where $h \bmod h_{kv} = 0$. The only new operation is `repeat_kv`, which expands the $h_{kv}$ KV heads to $h$ by repeating each $g = h / h_{kv}$ times.

- The backward pass through `repeat_kv` requires summing gradients from all $g$ query heads in a group back to the shared KV head. This is the only GQA-specific gradient computation: `grad_K_exp.reshape(B, h_kv, g, L, d_k).sum(axis=2)`.

- GQA reduces KV cache memory by a factor of $g$ without changing the attention core FLOPs. The savings come from smaller $W_K$/$W_V$ projections and, critically, from storing fewer KV heads in the cache during inference.

- GQA unifies three attention variants: MHA ($h_{kv} = h$), GQA ($1 < h_{kv} < h$), and MQA ($h_{kv} = 1$). Every production LLM (Llama 2 70B, Mistral 7B, Llama 3) uses GQA because the quality-efficiency tradeoff is strongly favorable.

### Quick Reference

```
Grouped-Query Attention
|-- Forward:  O(8BLd^2_model + 4BhL^2 d_k) -- projections + attention core
|             (K/V projections are smaller: d_model * h_kv * d_k instead of d_model^2)
|-- Backward: ~2x forward (+ reduce_kv_grad: reshape + sum over group axis)
|-- Memory:   O(BLd + BhL^2) -- intermediates + attention matrices
|
|-- Projections:
|   |-- W_Q: (d_model, d_model)             -- full size
|   |-- W_K: (d_model, h_kv * d_k)          -- reduced
|   |-- W_V: (d_model, h_kv * d_v)          -- reduced
|   |-- W_O: (d_model, d_model)             -- full size
|
|-- Key shapes:
|   |-- Q after split:   (B, h, L, d_k)
|   |-- K after split:   (B, h_kv, L, d_k)   <-- fewer heads
|   |-- K after expand:  (B, h, L, d_k)       <-- repeat_kv
|   |-- Attention A:     (B, h, L, L)         <-- h query heads
|
|-- KV cache per layer: 2 * B * h_kv * L * d_k * bytes
|-- KV cache reduction: h / h_kv = g times vs MHA
|
|-- Special cases:
|   |-- h_kv = h:   MHA (repeat_kv is no-op)
|   |-- h_kv = 1:   MQA (all queries share one KV head)
|
|-- Optimized by:
    |-- Broadcasting: avoid materializing repeat_kv expansion
    |-- Flash Attention: tiled attention, O(L) memory
    |-- KV cache: store h_kv heads, not h
    |-- Tensor Parallelism: split heads across GPUs
```
