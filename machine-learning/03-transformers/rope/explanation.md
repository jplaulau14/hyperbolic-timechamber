# Rotary Position Embeddings (RoPE) -- Deep Dive

## The Intuition

### What Problem Are We Solving?

In the previous topic we built sinusoidal positional encoding -- a clever scheme that adds position-dependent vectors to token embeddings before they enter attention. It works, but it has a fundamental limitation: position information is *indirect*.

Here is the problem. After adding sinusoidal PE to the input, we compute:

$$Q = (X + PE)W_Q, \quad K = (X + PE)W_K$$

The attention score between positions $m$ and $n$ is:

$$Q_m^\top K_n = (X_m + PE_m)^\top W_Q^\top W_K (X_n + PE_n)$$

Expanding this product gives four cross-terms: content-content, content-position, position-content, and position-position. The model must *learn* -- through the weight matrices $W_Q$ and $W_K$ -- to extract relative position information from this tangled mixture. It can be done, but it is indirect and fragile. The position signal passes through a learned linear transformation that mixes it with content, and the model has no structural guarantee that it will learn to use relative position rather than absolute position.

What if we could build a position encoding where the attention score $Q_m^\top K_n$ depends on relative distance $(m - n)$ *by construction*, not by learning? Where the math guarantees that shifting both positions by the same amount leaves the attention score unchanged?

That is exactly what RoPE does.

### The Key Insight

Instead of *adding* a position vector to the embedding, *rotate* the query and key vectors by a position-dependent angle. Pair up dimensions -- $(d_0, d_1)$, $(d_2, d_3)$, and so on -- and apply a 2D rotation to each pair. The rotation angle depends on the token's position and the dimension pair index.

The critical mathematical property: when you take the dot product of a rotated query at position $m$ with a rotated key at position $n$, the rotation matrices compose, and the result depends only on $(m - n)$. This is not an approximation or a learned behavior -- it is a theorem that follows from the properties of rotation matrices.

### Real-World Analogy

Imagine two clock towers in a city. Each tower has multiple clock faces, each ticking at a different speed. Tower A is at block $m$ and Tower B is at block $n$. To figure out how far apart they are, you do not need to know the absolute block numbers -- you just compare the differences in their clock readings. If you moved both towers 100 blocks east, all the absolute readings would change, but every difference between corresponding clock faces would remain exactly the same. The "distance signal" is encoded in the *relative* angles, not the absolute ones.

RoPE works the same way. Each dimension pair is like one clock face. Rotating $Q$ and $K$ by position-dependent angles means their dot product sees only the angle *differences* -- which encode relative distance.

---

## The Math, Step by Step

### Starting Simple: Rotation in 2D

Before tackling the full $d$-dimensional formula, let us understand what rotation means for a single pair of dimensions.

A 2D rotation by angle $\alpha$ transforms a vector $(x_0, x_1)$ into:

$$\begin{bmatrix} x'_0 \\ x'_1 \end{bmatrix} = \begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \end{bmatrix}$$

Three key properties of this rotation matrix $R(\alpha)$:

1. **Orthogonal**: $R(\alpha)^\top R(\alpha) = I$ -- the matrix preserves vector norms
2. **Composition**: $R(\alpha) R(\beta) = R(\alpha + \beta)$ -- rotations add
3. **Inverse**: $R(\alpha)^\top = R(-\alpha)$ -- transpose is inverse rotation

These three properties are what make the entire RoPE construction work.

### Adding Complexity: Multiple Dimension Pairs

The head dimension $d$ is split into $d/2$ pairs: $(0, 1)$, $(2, 3)$, ..., $(d-2, d-1)$. Each pair gets its own rotation frequency $\theta_i$, and each is rotated independently. At position $m$, pair $i$ is rotated by angle $m \cdot \theta_i$:

$$\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$$

The frequencies follow the same geometric progression as sinusoidal positional encoding:

$$\theta_i = \Theta^{-2i/d} = \frac{1}{\Theta^{2i/d}}$$

where $\Theta = 10000$ (original RoFormer, LLaMA 1/2) or $\Theta = 500000$ (LLaMA 3).

The full rotation matrix is block-diagonal:

$$R(m) = \begin{bmatrix} R_0(m) & & \\ & \ddots & \\ & & R_{d/2-1}(m) \end{bmatrix}$$

### The Core Equations

**Equation 1: Frequency schedule**

$$\theta_i = \exp\!\left(-\frac{2i}{d} \ln \Theta\right) \quad \text{for } i \in \{0, 1, \ldots, d/2 - 1\}$$

Where:
- $\theta_i$: rotation frequency for dimension pair $i$ (scalar)
- $d$: head dimension (must be even)
- $\Theta$: base frequency (10000 or 500000)

**Equation 2: Rotation application**

$$q'_m = R(m) \cdot q_m, \quad k'_n = R(n) \cdot k_n$$

Where:
- $q_m \in \mathbb{R}^d$: query vector at position $m$
- $R(m) \in \mathbb{R}^{d \times d}$: block-diagonal rotation matrix
- $q'_m$: rotated query, same shape as $q_m$

**Equation 3: Efficient element-wise form**

$$x' = x \odot \cos(m\theta) + \text{rotate\_half}(x) \odot \sin(m\theta)$$

Where $\odot$ is element-wise multiplication and $\text{rotate\_half}$ swaps and negates dimension pairs:

$$(x_0, x_1, x_2, x_3, \ldots) \xrightarrow{\text{rotate\_half}} (-x_1, x_0, -x_3, x_2, \ldots)$$

### The Relative Position Property (The Key Theorem)

This is the reason RoPE exists. The dot product between rotated query and key:

$$\langle q'_m, k'_n \rangle = (R(m) q_m)^\top (R(n) k_n) = q_m^\top R(m)^\top R(n) k_n = q_m^\top R(n - m) k_n$$

The last step uses: $R(m)^\top = R(-m)$ (orthogonality) and $R(-m) R(n) = R(n - m)$ (composition). The result depends only on the relative distance $(n - m)$, not on the absolute positions $m$ and $n$ individually.

Expanding for a single dimension pair $i$:

$$q'_{m,2i} k'_{n,2i} + q'_{m,2i+1} k'_{n,2i+1} = (q_{2i}k_{2i} + q_{2i+1}k_{2i+1})\cos((m-n)\theta_i) + (q_{2i}k_{2i+1} - q_{2i+1}k_{2i})\sin((m-n)\theta_i)$$

This depends on $q$, $k$ (content) and $(m - n)$ (relative position). Not on $m$ or $n$ individually.

The full dot product:

$$\langle q'_m, k'_n \rangle = \sum_{i=0}^{d/2-1} \left[ (q_{2i}k_{2i} + q_{2i+1}k_{2i+1})\cos((m-n)\theta_i) + (q_{2i}k_{2i+1} - q_{2i+1}k_{2i})\sin((m-n)\theta_i) \right]$$

### The Complex Number Formulation

There is an elegant alternative way to see RoPE: treat each dimension pair as a complex number.

View pair $(x_{2i}, x_{2i+1})$ as the complex number $\tilde{x}_i = x_{2i} + j \cdot x_{2i+1}$.

Rotation becomes complex multiplication:

$$\tilde{x}'_i = \tilde{x}_i \cdot e^{j \cdot m \cdot \theta_i}$$

Using Euler's formula $e^{j\alpha} = \cos\alpha + j\sin\alpha$:

$$\text{Re}(\tilde{x}'_i) = x_{2i}\cos(m\theta_i) - x_{2i+1}\sin(m\theta_i) \quad \checkmark$$
$$\text{Im}(\tilde{x}'_i) = x_{2i}\sin(m\theta_i) + x_{2i+1}\cos(m\theta_i) \quad \checkmark$$

This matches the rotation formula exactly. The relative position property also becomes obvious in complex form:

$$\tilde{q}'_{m,i} \cdot \overline{\tilde{k}'_{n,i}} = \tilde{q}_i e^{jm\theta_i} \cdot \overline{\tilde{k}_i e^{jn\theta_i}} = \tilde{q}_i \overline{\tilde{k}_i} \cdot e^{j(m-n)\theta_i}$$

The result depends only on $(m - n)$. This complex number viewpoint is not just elegant -- it maps directly to the `apply_rope_complex` function in the implementation, which serves as a correctness check.

---

## Worked Example: $d = 4$, Position $m = 2$

Let us trace through a complete example by hand.

### Step 1: Compute Frequencies

With $d = 4$ and $\Theta = 10000$:

$$\theta_0 = 10000^{-0/4} = 1.0$$
$$\theta_1 = 10000^{-2/4} = 10000^{-0.5} = 0.01$$

Pair 0 rotates fast (angle increases by 1.0 radian per position). Pair 1 rotates slowly (angle increases by 0.01 radian per position).

### Step 2: Compute Angles at Position 2

$$\text{angle}_0 = m \cdot \theta_0 = 2 \times 1.0 = 2.0$$
$$\text{angle}_1 = m \cdot \theta_1 = 2 \times 0.01 = 0.02$$

### Step 3: Compute Cos and Sin

$$\cos(2.0) \approx -0.4161, \quad \sin(2.0) \approx 0.9093$$
$$\cos(0.02) \approx 0.9998, \quad \sin(0.02) \approx 0.02000$$

### Step 4: Apply Rotation to $q = [1, 2, 3, 4]$

**Pair 0** ($q_0 = 1$, $q_1 = 2$, angle $= 2.0$):

$$q'_0 = 1 \cdot \cos(2.0) - 2 \cdot \sin(2.0) = 1 \cdot (-0.4161) - 2 \cdot (0.9093) = -2.2347$$
$$q'_1 = 1 \cdot \sin(2.0) + 2 \cdot \cos(2.0) = 1 \cdot (0.9093) + 2 \cdot (-0.4161) = 0.0771$$

**Pair 1** ($q_2 = 3$, $q_3 = 4$, angle $= 0.02$):

$$q'_2 = 3 \cdot \cos(0.02) - 4 \cdot \sin(0.02) = 3 \cdot (0.9998) - 4 \cdot (0.02000) = 2.9194$$
$$q'_3 = 3 \cdot \sin(0.02) + 4 \cdot \cos(0.02) = 3 \cdot (0.02000) + 4 \cdot (0.9998) = 4.0592$$

**Result:** $q' = [-2.2347, 0.0771, 2.9194, 4.0592]$

### Step 5: Verify Norm Preservation

$$\|q\|^2 = 1^2 + 2^2 + 3^2 + 4^2 = 30$$
$$\|q'\|^2 = (-2.2347)^2 + (0.0771)^2 + (2.9194)^2 + (4.0592)^2 \approx 4.9939 + 0.0059 + 8.5229 + 16.4773 = 30.0000$$

Norm is preserved exactly -- rotation is an orthogonal transformation.

### Step 6: Verify the Relative Position Property

Take $q = [1, 2, 3, 4]$ and $k = [5, 6, 7, 8]$. Rotate $q$ at position 2 and $k$ at position 0:

$k'$ at position 0 is just $k$ itself (since $\cos(0) = 1$, $\sin(0) = 0$): $k' = [5, 6, 7, 8]$.

Dot product: $q' \cdot k' = (-2.2347)(5) + (0.0771)(6) + (2.9194)(7) + (4.0592)(8) = -11.174 + 0.463 + 20.436 + 32.474 = 42.199$

Now shift both positions by 100: rotate $q$ at 102, $k$ at 100. The relative distance is still 2. Due to the theorem $R(m)^\top R(n) = R(n - m)$, this dot product will be identical: $42.199$.

The test suite verifies this to tolerance $< 10^{-10}$:

```python
def test_dot_product_invariance(self):
    pairs = [(5, 3), (105, 103), (505, 503), (1005, 1003)]
    # All have relative position 2 -- all dot products match
```

---

## Visualizing the Rotation

### 2D Rotation of a Single Dimension Pair

```
           y (x_{2i+1})
           ^
           |         . (x'_{2i}, x'_{2i+1})
           |       /
           |     /  angle = m * theta_i
           |   /
           | /____________. (x_{2i}, x_{2i+1})
           |
    -------+-----------------------> x (x_{2i})
           |
           |  The vector is rotated counterclockwise
           |  by angle m * theta_i.
           |  Its length (norm) is preserved.
```

### Dimension Pairing Structure

```
Head dimension d = 8:

  [ x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7 ]
    \___/      \___/      \___/      \___/
   pair 0     pair 1     pair 2     pair 3
  theta_0    theta_1    theta_2    theta_3
   FAST        |          |        SLOW
   rotation    |          |        rotation
               v          v
         medium speed   medium speed
```

### Frequency Spectrum

```
Dimension pair     theta_i (base=10000, d=128)    Wavelength
    0              1.0                              ~6.3 positions
    16             0.1                              ~63 positions
    32             0.01                             ~628 positions
    48             0.001                            ~6,283 positions
    63             ~0.0001                          ~62,832 positions

Low indices -----> high frequency, captures local position
High indices ----> low frequency, captures global position
```

### Before vs After RoPE Application

```
Before RoPE (position has no effect):

  Q at pos 5:   [0.3, -0.7, 1.2, 0.4, ...]     content only
  Q at pos 100: [0.3, -0.7, 1.2, 0.4, ...]     same content = same Q

After RoPE (position is baked in):

  Q at pos 5:   [0.8, -0.2, 1.2, 0.4, ...]     rotated by 5 * theta
  Q at pos 100: [-0.6, 0.5, 1.1, 0.6, ...]     rotated by 100 * theta

  Different positions --> different rotations --> different dot products
  But: dot(Q'_5, K'_3) == dot(Q'_105, K'_103)   (same relative distance 2)
```

---

## From Math to Code

### The Data Structures

The implementation has these components:

1. **`precompute_freqs(d, max_seq_len, theta_base)`** -- Computes cos/sin caches once
2. **`rotate_half(x)`** -- The core swap-and-negate operation
3. **`rotate_half_backward(x)`** -- The transpose rotation for backward pass
4. **`apply_rope(x, cos_cache, sin_cache)`** -- Applies rotation using element-wise form
5. **`apply_rope_complex(x, freqs)`** -- Correctness check via complex multiplication
6. **`RoPE` class** -- Bundles precomputation, forward, and backward

### Implementation Walkthrough: `precompute_freqs`

```python
def precompute_freqs(
    d: int, max_seq_len: int, theta_base: float = 10000.0
) -> Tuple[np.ndarray, np.ndarray]:
    if d % 2 != 0:
        raise ValueError(f"d must be even, got {d}")

    # (d/2,) -- log-space for numerical stability
    i = np.arange(d // 2, dtype=np.float64)
    inv_freq = np.exp(-2.0 * i / d * np.log(theta_base))

    # (max_seq_len, d/2) via outer product
    positions = np.arange(max_seq_len, dtype=np.float64)
    angles = positions[:, np.newaxis] * inv_freq[np.newaxis, :]

    return np.cos(angles), np.sin(angles)
```

**Line-by-line:**

- `i = np.arange(d // 2)`: Creates indices $[0, 1, \ldots, d/2 - 1]$, one per dimension pair. Shape: $(d/2,)$.

- `inv_freq = np.exp(-2.0 * i / d * np.log(theta_base))`: Computes $\theta_i = \Theta^{-2i/d}$ in log-space. This is equivalent to `theta_base ** (-2.0 * i / d)` but avoids computing large intermediate powers. For $d = 128$ and $\Theta = 10000$, this gives frequencies from $1.0$ (pair 0) down to $\sim 10^{-4}$ (pair 63). Shape: $(d/2,)$.

- `angles = positions[:, np.newaxis] * inv_freq[np.newaxis, :]`: Outer product of positions $(L,1)$ and frequencies $(1, d/2)$. Each element is $m \cdot \theta_i$ -- the rotation angle for position $m$, pair $i$. Shape: $(L, d/2)$.

- `return np.cos(angles), np.sin(angles)`: Each cache has shape $(L, d/2)$. These are precomputed once and reused for every forward pass.

### Implementation Walkthrough: `rotate_half`

```python
def rotate_half(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2)
    x = np.stack([-x[..., 1], x[..., 0]], axis=-1)
    return x.reshape(*x.shape[:-2], d)
```

This is the heart of the efficient RoPE computation. Let us trace it with $x = [1, 2, 3, 4]$ ($d = 4$):

1. `x.reshape(..., 2, 2)` groups into pairs: $[[1, 2], [3, 4]]$
2. `np.stack([-x[..., 1], x[..., 0]], axis=-1)`:
   - For pair 0: $[-2, 1]$
   - For pair 1: $[-4, 3]$
   - Result: $[[-2, 1], [-4, 3]]$
3. `reshape(..., 4)`: Flatten back to $[-2, 1, -4, 3]$

This produces the output $(-x_1, x_0, -x_3, x_2)$ for every pair. Why does this work? Let us verify against the rotation formula.

The element-wise formula says $x' = x \odot \cos + \text{rotate\_half}(x) \odot \sin$. For pair $i$:

$$x'_{2i} = x_{2i} \cos(m\theta_i) + (-x_{2i+1}) \sin(m\theta_i) = x_{2i}\cos(m\theta_i) - x_{2i+1}\sin(m\theta_i) \quad \checkmark$$

$$x'_{2i+1} = x_{2i+1} \cos(m\theta_i) + x_{2i} \sin(m\theta_i) \quad \checkmark$$

### Implementation Walkthrough: `rotate_half_backward`

```python
def rotate_half_backward(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2)
    x = np.stack([x[..., 1], -x[..., 0]], axis=-1)
    return x.reshape(*x.shape[:-2], d)
```

This is the transpose rotation: $(x_0, x_1) \to (x_1, -x_0)$. Compare with `rotate_half`'s $(x_0, x_1) \to (-x_1, x_0)$. They are negatives of each other when composed: `rotate_half_backward(rotate_half(x)) = -x` and `rotate_half(rotate_half_backward(x)) = -x`. The test verifies: applying `rotate_half` twice gives $-x$.

### Implementation Walkthrough: `_broadcast_cos_sin`

```python
def _broadcast_cos_sin(cos_cache, sin_cache, seq_len, d, positions=None):
    if positions is not None:
        cos = cos_cache[positions]
        sin = sin_cache[positions]
    else:
        cos = cos_cache[:seq_len]
        sin = sin_cache[:seq_len]

    # (L, d/2) -> (L, d) by repeating each freq for both dims in pair
    cos = np.repeat(cos, 2, axis=-1)
    sin = np.repeat(sin, 2, axis=-1)

    # (1, 1, L, d) for broadcasting over batch and heads
    return cos[np.newaxis, np.newaxis, :, :], sin[np.newaxis, np.newaxis, :, :]
```

**Why `np.repeat(cos, 2, axis=-1)`?** The cos cache has shape $(L, d/2)$ -- one value per dimension pair. But the input $x$ has shape $(B, H, L, d)$ -- two elements per pair. We need $[\cos(\theta_0), \cos(\theta_0), \cos(\theta_1), \cos(\theta_1), \ldots]$ so that both elements in a pair are multiplied by the same cosine value. `np.repeat(..., 2, axis=-1)` achieves this: $(L, d/2) \to (L, d)$.

**Why `[np.newaxis, np.newaxis, :, :]`?** This adds two leading dimensions of size 1, giving shape $(1, 1, L, d)$. NumPy broadcasting then stretches this across the batch ($B$) and head ($H$) dimensions of the input.

**Why `positions` parameter?** During KV cache inference, new tokens arrive at non-contiguous positions. If we are generating token at position 100, we need $\cos(100 \cdot \theta)$ and $\sin(100 \cdot \theta)$, not the values for position 0. The `positions` array allows indexing into the cache at arbitrary positions.

### Implementation Walkthrough: `apply_rope`

```python
def apply_rope(x, cos_cache, sin_cache, positions=None):
    seq_len = x.shape[2]
    d = x.shape[-1]
    cos, sin = _broadcast_cos_sin(cos_cache, sin_cache, seq_len, d, positions)
    return x * cos + rotate_half(x) * sin
```

This is the entire forward computation in one line: $x' = x \odot \cos + \text{rotate\_half}(x) \odot \sin$. The shapes:

```
x:              (B, H, L, d)
cos, sin:       (1, 1, L, d)   -- broadcast over B and H
rotate_half(x): (B, H, L, d)
result:         (B, H, L, d)
```

### Implementation Walkthrough: `apply_rope_complex`

```python
def apply_rope_complex(x, freqs):
    d = x.shape[-1]
    x_pairs = x.reshape(*x.shape[:-1], d // 2, 2)
    x_complex = x_pairs[..., 0] + 1j * x_pairs[..., 1]
    x_rotated = x_complex * freqs
    result = np.stack([x_rotated.real, x_rotated.imag], axis=-1)
    return result.reshape(*x.shape)
```

This is the complex number implementation -- mathematically equivalent to `apply_rope` but using a completely different code path. It serves as a correctness check. The test verifies both implementations agree to $< 10^{-12}$:

```python
def test_matches_rotate_half(self):
    result_cx = apply_rope_complex(x, freqs_complex)
    np.testing.assert_allclose(result_cx, result_rh, atol=1e-12)
```

### Implementation Walkthrough: `RoPE.backward`

```python
def backward(self, grad_q_rot, grad_k_rot):
    cos = self._cache["cos"]
    sin = self._cache["sin"]

    grad_q = grad_q_rot * cos + rotate_half_backward(grad_q_rot) * sin
    grad_k = grad_k_rot * cos + rotate_half_backward(grad_k_rot) * sin

    return grad_q, grad_k
```

The backward pass applies the *inverse* rotation $R(-m)$ to the upstream gradient. Since $R(m)$ is orthogonal, $R(-m) = R(m)^\top$. In the element-wise form, the inverse rotation uses `rotate_half_backward` (which produces $(x_1, -x_0, x_3, -x_2, \ldots)$) instead of `rotate_half` (which produces $(-x_1, x_0, -x_3, x_2, \ldots)$).

The structure is identical to the forward pass but with `rotate_half_backward` replacing `rotate_half`. This makes sense: the Jacobian of the rotation is the rotation matrix itself, and the backward pass multiplies by the transpose of the Jacobian.

### The Tricky Parts

**Why does `rotate_half` reshape to `(..., d//2, 2)` instead of simply indexing even/odd elements?**

Indexing even and odd elements (`x[..., 0::2]` and `x[..., 1::2]`) creates non-contiguous views in NumPy. The reshape approach keeps memory access patterns cleaner and generalizes to arbitrary leading dimensions. Both approaches produce the same result, but reshape is the standard pattern used in production implementations.

**Why `np.repeat(cos, 2, axis=-1)` and not `np.tile`?**

`np.repeat` duplicates each element: $[a, b, c] \to [a, a, b, b, c, c]$. `np.tile` repeats the entire array: $[a, b, c] \to [a, b, c, a, b, c]$. We need the first pattern because each frequency covers a pair of *adjacent* dimensions. Using `tile` would assign the wrong frequencies to dimensions.

**Why does the backward pass use the same `sin` values (not negated) with `rotate_half_backward`?**

This is subtle. The inverse rotation negates the sine in the $2 \times 2$ rotation matrix:

$$R(-m) = \begin{bmatrix} \cos(m\theta) & \sin(m\theta) \\ -\sin(m\theta) & \cos(m\theta) \end{bmatrix}$$

But the element-wise form absorbs this negation into `rotate_half_backward`. Forward uses `rotate_half` with $+\sin$; backward uses `rotate_half_backward` with $+\sin$. The sign flip happens inside `rotate_half_backward`, which negates the *opposite* element compared to `rotate_half`.

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Why |
|-----------|------------|-----|
| `precompute_freqs` | $O(L_{max} \cdot d)$ | Outer product + cos/sin for each angle |
| `rotate_half` | $O(B \cdot H \cdot L \cdot d)$ | Reshape + stack over all elements |
| `apply_rope` | $O(B \cdot H \cdot L \cdot d)$ | Two element-wise multiplies + one add |
| `forward` (Q + K) | $O(B \cdot (H + H_{kv}) \cdot L \cdot d)$ | Apply rotation to both Q and K |
| `backward` | $O(B \cdot (H + H_{kv}) \cdot L \cdot d)$ | Same as forward (inverse rotation) |

### Space Complexity

- **Cos/sin cache**: $O(L_{max} \cdot d)$ -- precomputed once, read-only during forward/backward
- **Forward working memory**: $O(B \cdot H \cdot L \cdot d)$ -- for `rotate_half` intermediate and output
- **Backward cache**: $O(L \cdot d)$ -- stores cos/sin slices used during forward (the Q and K inputs are also cached for reference)

### The Bottleneck

RoPE is **memory-bandwidth bound**, not compute bound. Each element of Q and K is read, multiplied by a cached value, and written -- just a few arithmetic operations per memory access. The total FLOPs per attention head are approximately $6Ld$ for the forward pass. Compare this to:

- Q/K projection: $2BLd_{model}^2$ FLOPs (compute bound)
- Attention scores $QK^\top$: $2BHL^2d$ FLOPs (compute or memory bound depending on $L$)

RoPE is roughly $L \cdot d_{model} / (6Ld) = d_{model} / (6d) \approx h/6$ times cheaper than a single attention head's score computation, and thousands of times cheaper than the Q/K projections. It is negligible in the total computational budget.

---

## Common Pitfalls

### Pitfall 1: Applying RoPE to V (Not Just Q and K)

**The mistake:**

```python
# Wrong: rotating the value tensor
q_rot = apply_rope(q, cos, sin)
k_rot = apply_rope(k, cos, sin)
v_rot = apply_rope(v, cos, sin)  # <-- This is wrong
```

**Why it is wrong:** RoPE encodes relative position into the *attention score* via the dot product $Q^\top K$. The value tensor $V$ carries content that gets weighted and aggregated by the attention weights. Rotating $V$ would corrupt the content representation without providing any positional benefit, since $V$ does not participate in the dot product that determines attention weights.

**The fix:**

```python
# Correct: rotate only Q and K
q_rot = apply_rope(q, cos, sin)
k_rot = apply_rope(k, cos, sin)
# V is used as-is
```

### Pitfall 2: Using `np.tile` Instead of `np.repeat` for Frequency Expansion

**The mistake:**

```python
# Wrong: tile repeats the whole array, giving wrong frequency assignment
cos = np.tile(cos_cache[:seq_len], (1, 2))  # [t0, t1, ..., t0, t1, ...]
```

**Why it is wrong:** With `tile`, dimension 0 gets $\theta_0$, dimension 1 gets $\theta_1$, dimension 2 gets $\theta_0$ again, dimension 3 gets $\theta_1$ again. But dimensions 0 and 1 should *both* get $\theta_0$ (they form a pair), and dimensions 2 and 3 should *both* get $\theta_1$. Using `tile` means each element in a pair gets a *different* frequency, breaking the 2D rotation structure entirely.

**The fix:**

```python
# Correct: repeat each frequency for its pair of dimensions
cos = np.repeat(cos_cache[:seq_len], 2, axis=-1)  # [t0, t0, t1, t1, ...]
```

### Pitfall 3: Computing Frequencies Without Log-Space

**The mistake:**

```python
# Potentially unstable for large d or large theta_base
inv_freq = 1.0 / (theta_base ** (2.0 * i / d))
```

**Why it is problematic:** For large $\Theta$ (like 500000 for LLaMA 3) and large $d$, the intermediate $\Theta^{2i/d}$ can overflow or lose precision in float32. The log-space form avoids this:

```python
# Correct: numerically stable
inv_freq = np.exp(-2.0 * i / d * np.log(theta_base))
```

Both are mathematically identical ($e^{-x \ln a} = a^{-x}$), but the log-space version keeps intermediate values small.

### Pitfall 4: Forgetting to Handle Non-Contiguous Positions for KV Cache

**The mistake:**

```python
# Wrong: always slice from position 0
cos = cos_cache[:seq_len]
sin = sin_cache[:seq_len]
```

**Why it is wrong:** During incremental decoding with a KV cache, new tokens arrive at specific positions (e.g., position 100, not position 0). Using `cos_cache[:1]` gives the cosine values for position 0, not position 100. The rotated key would encode the wrong position, and the relative position property would break.

**The fix:**

```python
# Correct: index by actual position
if positions is not None:
    cos = cos_cache[positions]
    sin = sin_cache[positions]
else:
    cos = cos_cache[:seq_len]
```

The test suite verifies this explicitly with a KV cache simulation:

```python
def test_kv_cache_simulation(self):
    # Incremental computation with explicit positions must match
    # full from-scratch computation
    np.testing.assert_allclose(scores_incr, scores_full, atol=1e-12)
```

---

## Comparison with Sinusoidal Positional Encoding

### Side-by-Side

| Property | Sinusoidal PE | RoPE |
|----------|--------------|------|
| **Application** | Added to input *before* projection: $X' = X + PE$ | Applied to Q, K *after* projection: $Q' = R(m) \cdot Q$ |
| **Mechanism** | Additive | Multiplicative (rotation) |
| **Where in the pipeline** | Once, before the first layer | At every layer, inside attention |
| **Relative position** | Approximate -- model must learn to extract it from cross-terms | Exact -- dot product depends only on $(m-n)$ by construction |
| **Affected by $W_Q$, $W_K$** | Yes -- position info is mixed with content by projection | No -- rotation is applied *after* projection |
| **Frequency schedule** | $\omega_i = 10000^{-2i/d}$ | $\theta_i = 10000^{-2i/d}$ (identical) |
| **KV cache** | Invalidated if position scheme changes | Keys store rotation; always valid |
| **Used by** | Original Transformer (2017) | LLaMA, Mistral, Qwen, and all modern LLMs |

### The Frequency Connection

Both sinusoidal PE and RoPE use exactly the same frequency schedule. The test verifies this:

```python
def test_matches_sinusoidal_pe(self):
    rope_freqs, sin_freqs = compare_with_sinusoidal(d, 100)
    np.testing.assert_allclose(rope_freqs, sin_freqs, atol=1e-15)
```

RoPE takes the rotation property that sinusoidal PE *has* (the block-diagonal $M_k$ matrix from Topic 12) and makes it the *entire* position encoding mechanism. Instead of adding a position vector and hoping the model learns to exploit the rotation structure, RoPE applies the rotation directly where it matters: in the $QK^\top$ dot product.

### Why RoPE Won

1. **Exact relative position**: The dot product property holds by construction, not by learning. This gives a stronger inductive bias for language modeling.

2. **KV cache compatibility**: Rotated keys contain their position information intrinsically. Cached keys remain valid regardless of what future positions are generated. With additive PE, position is baked into the *input*, so changing the position scheme invalidates everything downstream.

3. **Context length extension**: The frequency-based formulation enables principled techniques like NTK-aware scaling and YaRN to extend context length by modifying $\Theta$. This is harder with additive encodings.

4. **Per-layer application**: RoPE is applied at every attention layer, giving each layer fresh position information. Additive PE is applied once at the input and must survive through many layers of transformation.

---

## Connection to Inference Optimization

### KV Cache: Why RoPE Matters

With additive positional encoding, the key at position $n$ is:

$$K_n = (X_n + PE_n) W_K$$

Position is mixed into the content before projection. If you cache $K_n$ and later want to change the positional encoding scheme, every cached key is wrong.

With RoPE, the key at position $n$ is:

$$K'_n = R(n) \cdot (X_n W_K)$$

Position is applied as a rotation *after* projection. The cached rotated key $K'_n$ at position $n$ is correct forever. When the model generates a new token at position $m$:

$$Q'_m \cdot K'_n = (R(m) Q_m)^\top (R(n) K_n) = Q_m^\top R(n-m) K_n$$

The relative position $(n - m)$ is automatically encoded. No recomputation of cached keys is needed.

```
RoPE in the Attention Pipeline with KV Cache:

   New token at position m
           |
     X_m @ W_Q = Q_m              X_m @ W_K = K_m
           |                            |
     R(m) * Q_m = Q'_m            R(m) * K_m = K'_m  --> append to KV cache
           |                            |
           |                  KV Cache: [K'_0, K'_1, ..., K'_{m-1}, K'_m]
           |                            |
           +----------+  +--------------+
                      |  |
                 Q'_m @ [K'_0, ..., K'_m]^T  / sqrt(d)
                      |
                   softmax
                      |
                   output

   Key point: K'_0 through K'_{m-1} were computed and cached
   in previous steps. They are NEVER recomputed.
```

### Context Length Extension

The base frequency $\Theta$ controls the effective context window. The rotation angle at position $m$ for pair $i$ is $m \cdot \theta_i = m \cdot \Theta^{-2i/d}$. Increasing $\Theta$ makes all rotation angles smaller, which means the model can represent more positions before the angles "wrap around."

| Model | $\Theta$ | Context Length |
|-------|----------|----------------|
| LLaMA 1/2 | 10,000 | 4,096 |
| LLaMA 3 | 500,000 | 8,192+ |

Advanced techniques like NTK-aware scaling modify $\Theta$ to extend context without fine-tuning: $\Theta' = \Theta \cdot \alpha^{d/(d-2)}$. This preserves high-frequency components (local position resolution) while stretching low-frequency components (long-range). Understanding the base RoPE implementation is essential for implementing and debugging these extensions.

### Computational Cost at Inference

RoPE is negligible in the total inference budget:

| Operation | FLOPs per token | Category |
|-----------|----------------|----------|
| RoPE application | $\sim 6 \cdot d_{model}$ | Memory-bandwidth bound |
| Q/K projection | $2 \cdot d_{model}^2$ | Compute bound |
| Attention scores ($QK^\top$) | $2 \cdot H \cdot L \cdot d$ | Memory-bandwidth bound (decode) |

For $d_{model} = 4096$, RoPE costs $\sim 24K$ FLOPs per token, while Q/K projection costs $\sim 33M$ FLOPs -- over 1000x more expensive. RoPE is a prime candidate for kernel fusion with the Q/K projection in optimized inference engines.

### From Naive to Optimized

| Naive (what we implemented) | Optimized (production) |
|----------------------------|------------------------|
| Full cos/sin cache in float64 | On-the-fly computation in float16/bfloat16 |
| `np.repeat` materializes expanded cos/sin | Fused kernel: compute cos/sin inline |
| Separate `rotate_half` allocation | In-place rotation within fused Q/K kernel |
| Fixed $\Theta = 10000$ | Dynamic $\Theta$ for context extension |
| All positions precomputed | Only compute for current position (decode) |

---

## Backward Pass Intuition

### Why the Backward Pass is Just Inverse Rotation

The forward pass for a single element is:

$$x' = R(m) \cdot x$$

where $R(m)$ is the block-diagonal rotation matrix. Since $R(m)$ is a constant (it depends on position, not on $x$), the Jacobian of this operation is simply $R(m)$ itself:

$$\frac{\partial x'}{\partial x} = R(m)$$

The backward pass multiplies the upstream gradient by the transpose of the Jacobian:

$$\frac{\partial \mathcal{L}}{\partial x} = R(m)^\top \frac{\partial \mathcal{L}}{\partial x'}$$

Since rotation matrices are orthogonal, $R(m)^\top = R(-m)$ -- the transpose is the inverse rotation. The backward pass simply "un-rotates" the gradient.

### Verification

The test confirms that applying forward then backward recovers the original:

```python
def test_inverse_rotation_recovers_original(self):
    q_rot, k_rot = rope.forward(q, k)
    grad_q, grad_k = rope.backward(q_rot, k_rot)
    np.testing.assert_allclose(grad_q, q, atol=1e-12)
```

If we forward-rotate $q$ to get $q' = R(m)q$, then pass $q'$ as the "upstream gradient" to backward, we get $R(-m) \cdot R(m) \cdot q = I \cdot q = q$. Rotation followed by inverse rotation is the identity.

At position 0, both forward and backward are the identity (since $\cos(0) = 1$, $\sin(0) = 0$):

```python
def test_position_zero_passthrough(self):
    # Backward at position 0: gradient passes through unchanged
    np.testing.assert_allclose(grad_q, grad_q_rot, atol=1e-14)
```

---

## Testing Your Understanding

### Quick Checks

1. **What happens if you remove the scaling factor $1/\sqrt{d_k}$ from attention?** The scaling factor prevents dot products from growing with $d_k$, keeping softmax inputs in a reasonable range. It has nothing to do with RoPE itself -- RoPE modifies Q and K before the dot product, and the scaling is applied to the dot product result. Removing it would make attention weights too peaked (saturated softmax).

2. **Why must $d$ be even?** Each dimension pair requires exactly two elements. With odd $d$, one dimension would be unpaired, and the rotation formula would not apply. The implementation raises `ValueError` for odd $d`.

3. **If input is $(B, H, L, d)$, what is the output shape of `apply_rope`?** Exactly $(B, H, L, d)$. Rotation preserves shape -- it only modifies the values along the last dimension, not the shape.

4. **Why does RoPE apply to Q and K but not V?** The relative position property lives in the dot product $Q^\top K$. The value tensor $V$ carries content information that gets aggregated by attention weights. Position information in $V$ would be meaningless because $V$ does not participate in computing which tokens to attend to.

5. **If you double $\Theta$ from 10000 to 20000, what happens?** All rotation frequencies decrease (rotate more slowly). The effective context window extends because angles grow more slowly with position. The model can distinguish more positions before angles wrap around. But it also means the model has coarser local position resolution.

### Exercises

1. **Easy**: Verify by hand that `rotate_half(rotate_half(x))` gives $-x$ for $x = [a, b, c, d]$. First application: $[-b, a, -d, c]$. Second application: $[-a, -b, -c, -d] = -x$.

2. **Medium**: Implement RoPE using the explicit block-diagonal rotation matrix (construct the full $d \times d$ matrix and multiply). Compare the output with `apply_rope` for random inputs. This is much slower but helps build intuition.

3. **Hard**: Implement NTK-aware scaling by modifying `precompute_freqs` to accept a scaling factor $\alpha$ and compute $\Theta' = \Theta \cdot \alpha^{d/(d-2)}$. Verify that high-frequency components are relatively preserved while low-frequency components are stretched.

---

## Summary

### Key Takeaways

- RoPE encodes relative position directly into the attention dot product by rotating Q and K vectors. The dot product $\langle R(m)q, R(n)k \rangle$ depends only on $(m-n)$, not on absolute positions -- this is guaranteed by the composition property of rotation matrices.

- The efficient implementation uses element-wise operations: $x' = x \odot \cos + \text{rotate\_half}(x) \odot \sin$. No rotation matrices are ever constructed. The `rotate_half` function (swap and negate dimension pairs) is the key building block.

- RoPE uses the same frequency schedule as sinusoidal PE ($\theta_i = \Theta^{-2i/d}$) but applies it multiplicatively (rotation) instead of additively. This makes position encoding exact, KV-cache-friendly, and extensible to longer contexts via theta scaling.

### Quick Reference

```
Rotary Position Embeddings (RoPE)
|-- Precompute: O(L_max * d) -- cos/sin cache, done once
|-- Forward:    O(B * H * L * d) -- element-wise rotation of Q and K
|-- Backward:   O(B * H * L * d) -- inverse rotation (transpose of orthogonal matrix)
|-- Memory:     O(L_max * d) cache + O(B * H * L * d) working
|
|-- Key property: dot(RoPE(q,m), RoPE(k,n)) = f(q, k, m-n)
|   Only relative position matters.
|
|-- Frequency schedule: theta_i = base^{-2i/d}
|   |-- base = 10000 (LLaMA 1/2), 500000 (LLaMA 3)
|   |-- Low i: fast rotation (local position)
|   |-- High i: slow rotation (global position)
|
|-- Integration with attention:
|   |-- Applied AFTER Q/K projection, BEFORE dot product
|   |-- V is NOT rotated
|   |-- KV cache stores ROTATED keys (position baked in)
|   |-- GQA-compatible: applied per head independently
|
|-- vs Sinusoidal PE:
|   |-- Same frequencies, different application
|   |-- Additive vs multiplicative
|   |-- Approximate vs exact relative position
|   |-- Applied once vs at every layer
|
|-- Optimized by: kernel fusion with Q/K projection
|   Context extension via theta scaling (NTK-aware, YaRN)
```
