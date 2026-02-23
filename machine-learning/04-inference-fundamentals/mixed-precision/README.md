# Mixed Precision

**Phase 4 · Topic 19** — Why precision is the key to memory bandwidth, and why memory bandwidth is the key to inference speed.

## What it is

Mixed precision computing uses lower-precision floating point formats (FP16, BF16) for most operations while strategically keeping certain values in higher precision (FP32). This technique roughly doubles throughput and halves memory usage on modern GPUs because tensor cores operate natively on FP16/BF16, and memory bandwidth is almost always the bottleneck for inference.

Understanding floating point precision is foundational for inference engineering. When you see that a model "runs in FP16" or "uses BF16 weights," you need to know exactly what that means: the bit layout, the representable range, the precision tradeoffs, and why certain operations (like loss accumulation or softmax denominators) must stay in FP32 to avoid numerical disasters.

The key insight is that neural network weights and activations don't need 32-bit precision. Most values cluster near zero, gradients are small, and the network learns to be robust to noise. This slack in precision requirements is what makes quantization (INT8, INT4) possible later. Mixed precision is your first step into this territory.

## The math

### Floating point format anatomy

Every IEEE floating point number is represented as:

$$\text{value} = (-1)^{s} \times 2^{e - \text{bias}} \times (1 + m)$$

Where:
- $s$ is the sign bit (0 = positive, 1 = negative)
- $e$ is the exponent field, determining the magnitude (range)
- $m$ is the mantissa (also called significand or fraction), determining precision
- $\text{bias}$ is the offset to allow negative exponents (e.g., 127 for FP32, 15 for FP16)

### FP32 (single precision) — 32 bits

$$\underbrace{[1]}_{\text{sign}} \underbrace{[8]}_{\text{exponent}} \underbrace{[23]}_{\text{mantissa}}$$

- **Exponent bias:** 127
- **Range:** $\sim 1.18 \times 10^{-38}$ to $\sim 3.4 \times 10^{38}$
- **Precision:** $\sim 7.2$ decimal digits ($2^{23} = 8{,}388{,}608$ distinct mantissa values)
- **Special values:**
  - Exponent all 0s, mantissa 0: zero (signed)
  - Exponent all 0s, mantissa $\neq 0$: subnormal (denormalized) numbers
  - Exponent all 1s, mantissa 0: infinity (signed)
  - Exponent all 1s, mantissa $\neq 0$: NaN

### FP16 (half precision) — 16 bits

$$\underbrace{[1]}_{\text{sign}} \underbrace{[5]}_{\text{exponent}} \underbrace{[10]}_{\text{mantissa}}$$

- **Exponent bias:** 15
- **Range:** $\sim 5.96 \times 10^{-8}$ to $\sim 65{,}504$
- **Precision:** $\sim 3.3$ decimal digits ($2^{10} = 1{,}024$ distinct mantissa values)
- **Max representable integer:** 2048 (beyond this, not all integers are representable)
- **Smallest positive normal:** $2^{-14} \approx 6.1 \times 10^{-5}$
- **Smallest positive subnormal:** $2^{-24} \approx 5.96 \times 10^{-8}$

**Critical limitation:** The maximum value is 65,504. Any value larger overflows to infinity. This is why FP16 requires careful loss scaling during training.

### BF16 (bfloat16 / brain float) — 16 bits

$$\underbrace{[1]}_{\text{sign}} \underbrace{[8]}_{\text{exponent}} \underbrace{[7]}_{\text{mantissa}}$$

- **Exponent bias:** 127 (same as FP32)
- **Range:** Same as FP32 ($\sim 1.18 \times 10^{-38}$ to $\sim 3.4 \times 10^{38}$)
- **Precision:** $\sim 2.4$ decimal digits ($2^{7} = 128$ distinct mantissa values)
- **Key property:** Truncated FP32 — you can convert FP32 to BF16 by dropping the lower 16 bits

**Why BF16 exists:** Google designed BF16 specifically for deep learning. Neural networks care more about dynamic range than precision. By keeping the FP32 exponent, BF16 avoids overflow/underflow issues while still halving memory.

### Comparison table

| Format | Bits | Sign | Exponent | Mantissa | Range | Precision |
|--------|------|------|----------|----------|-------|-----------|
| FP32   | 32   | 1    | 8        | 23       | $\sim 10^{38}$ | $\sim 7$ digits |
| FP16   | 16   | 1    | 5        | 10       | $\sim 65{,}504$ | $\sim 3$ digits |
| BF16   | 16   | 1    | 8        | 7        | $\sim 10^{38}$ | $\sim 2$ digits |

### Precision vs range tradeoff

**Precision** determines how many distinct values exist between any two powers of 2:
- FP32: $2^{23} = 8.4$ million values between 1.0 and 2.0
- FP16: $2^{10} = 1{,}024$ values between 1.0 and 2.0
- BF16: $2^{7} = 128$ values between 1.0 and 2.0

**Range** determines the largest/smallest representable magnitudes:
- FP16 overflows at $\sim 65{,}504$ — loss values, gradient norms, and some activations can exceed this
- BF16 has the same range as FP32 — overflow is rarely a problem

**Spacing between consecutive values:**

For a value $v$, the gap to the next representable value is:

$$\text{gap} = 2^{e - m_{\text{bits}}}$$

where $e$ is the exponent and $m_{\text{bits}}$ is the number of mantissa bits.

At $v = 1.0$:

$$\text{FP32 gap} = 2^{-23} \approx 1.19 \times 10^{-7}$$

$$\text{FP16 gap} = 2^{-10} \approx 9.77 \times 10^{-4}$$

$$\text{BF16 gap} = 2^{-7} \approx 7.81 \times 10^{-3}$$

At $v = 1000.0$ (exponent = 9):

$$\text{FP32 gap} = 2^{9-23} \approx 6.1 \times 10^{-5}$$

$$\text{FP16 gap} = 2^{9-10} = 0.5$$

$$\text{BF16 gap} = 2^{9-7} = 4.0$$

This means at value 1000, consecutive FP16 numbers are 0.5 apart, and BF16 numbers are 4.0 apart. Gradient updates smaller than this gap are effectively zero.

### Conversion formulas

**FP32 to FP16:**

```python
def fp32_to_fp16(x):
    """
    NumPy handles this, but conceptually:
    1. If |x| > 65504: return +/- inf
    2. If |x| < 2^-24: return 0 (underflow to zero)
    3. If 2^-24 <= |x| < 2^-14: subnormal (reduced precision)
    4. Otherwise: round mantissa from 23 to 10 bits
    """
    return x.astype(np.float16)
```

**FP32 to BF16:**

```python
def fp32_to_bf16_simulate(x):
    """
    BF16 is literally truncated FP32.
    Drop the lower 16 bits of the mantissa.
    NumPy doesn't support BF16 natively, so we simulate it.
    """
    x32 = x.astype(np.float32)
    # View as uint32, mask off lower 16 bits, view back as float32
    x_int = x32.view(np.uint32)
    x_int = x_int & 0xFFFF0000  # Zero out lower 16 bits
    return x_int.view(np.float32)
```

### Loss scaling for FP16 training

Small gradients underflow to zero in FP16. Loss scaling multiplies the loss by a large factor before backprop, then divides gradients by the same factor after:

$$g_{\text{scaled}} = \frac{\partial (L \cdot s)}{\partial w} = s \cdot \frac{\partial L}{\partial w}$$

$$g = \frac{g_{\text{scaled}}}{s}$$

where $s$ is the scale factor.

```python
# Problem: gradient = 1e-6 becomes 0 in FP16

# Solution: loss scaling
scale = 1024  # or dynamically adjusted

# Forward pass (FP16)
y_pred = model(x)
loss = compute_loss(y_pred, y)

# Scale loss before backward
scaled_loss = loss * scale

# Backward pass (FP16 gradients are now scale * true_gradient)
scaled_grads = backward(scaled_loss)

# Unscale gradients (back to FP32)
grads = scaled_grads.astype(np.float32) / scale

# Check for inf/nan
if has_inf_or_nan(grads):
    scale = scale / 2  # Reduce scale factor
    # Skip this update
else:
    # Update weights (FP32 master copy)
    weights = weights - lr * grads
    scale = min(scale * 2, max_scale)  # Increase scale factor
```

**Dynamic loss scaling algorithm:**
1. Start with a large scale factor (e.g., $2^{16} = 65536$)
2. If gradients overflow (inf/nan), halve the scale and skip the update
3. If $N$ consecutive steps succeed without overflow, double the scale
4. Typical $N = 2000$ steps

### Numerical stability concerns

**Softmax in FP16:**

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Problem: $e^x$ overflows for $x > 11$ in FP16 ($e^{11} \approx 59{,}874 < 65{,}504$, but $e^{12} = \infty$)

Solution: Subtract max before exp

$$\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

**Accumulation precision:**

```python
# Bad: accumulating in FP16 loses precision
sum_fp16 = np.float16(0)
for x in values_fp16:
    sum_fp16 += x  # Each addition loses precision

# Good: accumulate in FP32, convert result
sum_fp32 = np.float32(0)
for x in values_fp16:
    sum_fp32 += np.float32(x)
result = np.float16(sum_fp32)
```

**Matrix multiplication:**

Modern tensor cores compute:

$$C = A \cdot B$$

where $A, B \in \text{FP16}$, accumulator $\in \text{FP32}$, output $C \in \text{FP16}$.

This gives you the memory benefits of FP16 with FP32 accumulation precision.

## Why it matters for inference

### Memory bandwidth is THE bottleneck

For transformer inference (especially at large batch sizes or long sequences), you are almost always **memory-bandwidth bound**, not compute-bound.

**A100 GPU specs:**
- Memory bandwidth: 2 TB/s
- FP16 tensor core throughput: 312 TFLOPS

For a matrix multiply of shape $(M, K) \times (K, N)$:
- Data to load: $(M \cdot K + K \cdot N) \times \text{bytes\_per\_element}$
- Compute: $2 \cdot M \cdot K \cdot N$ FLOPs

$$\text{Arithmetic intensity} = \frac{\text{FLOPs}}{\text{Bytes}}$$

$$\text{FP32 intensity} = \frac{2 \cdot M \cdot K \cdot N}{(M \cdot K + K \cdot N) \times 4}$$

$$\text{FP16 intensity} = \frac{2 \cdot M \cdot K \cdot N}{(M \cdot K + K \cdot N) \times 2} \quad \text{(2x higher!)}$$

The GPU is compute-bound when:

$$\text{intensity} > \frac{\text{peak FLOPS}}{\text{memory bandwidth}} = \frac{312 \text{ TFLOPS}}{2 \text{ TB/s}} = 156 \text{ FLOP/byte}$$

For small batch inference ($M=1$, a single sequence):

$$\text{intensity}_{\text{FP16}} = \frac{2 \cdot K \cdot N}{(K + K \cdot N) \times 2} = \frac{N}{1 + N} \approx 1 \text{ (for large } N \text{)}$$

This is WAY below 156, so inference is memory-bound. Half precision = half the bytes = 2x throughput.

### Throughput multipliers

| Precision | Memory per weight | Relative throughput |
|-----------|------------------|---------------------|
| FP32      | 4 bytes          | 1x (baseline)       |
| FP16/BF16 | 2 bytes          | $\sim 2\times$      |
| INT8      | 1 byte           | $\sim 4\times$ (with quality loss) |
| INT4      | 0.5 bytes        | $\sim 8\times$ (with quality loss) |

### Why BF16 is preferred for LLMs

1. **No overflow issues:** The FP32 exponent range means you never hit the 65504 ceiling
2. **Simple conversion:** Truncate FP32 to BF16 by dropping bits (no complex rounding)
3. **Loss scaling unnecessary:** During training, BF16 doesn't need loss scaling
4. **Hardware support:** A100, H100, TPUs all have native BF16 tensor cores

**When FP16 is still used:**
- Older hardware (V100) has FP16 but not BF16 tensor cores
- Inference with known value ranges (post-training, values are bounded)
- When you need slightly more precision than BF16

### KV cache memory impact

For a 7B parameter model:

$$\text{KV cache per token} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes}$$

$$= 2 \times 32 \times 32 \times 128 \times \text{bytes}$$

| Precision | Bytes | Per Token | At 4K Context |
|-----------|-------|-----------|---------------|
| FP32      | 4     | 1 MB      | 4 GB          |
| FP16      | 2     | 0.5 MB    | 2 GB          |

Half precision KV cache doubles the context length you can serve on the same hardware.

### Model weight memory

For model weights:

$$\text{Memory} = n_{\text{params}} \times \text{bytes\_per\_element}$$

**7B model weights:**

| Precision | Calculation | Memory |
|-----------|-------------|--------|
| FP32      | $7\text{B} \times 4$ | 28 GB |
| FP16      | $7\text{B} \times 2$ | 14 GB |
| INT8      | $7\text{B} \times 1$ | 7 GB  |
| INT4      | $7\text{B} \times 0.5$ | 3.5 GB |

**70B model weights:**

| Precision | Memory | Hardware Requirement |
|-----------|--------|---------------------|
| FP32      | 280 GB | Doesn't fit on any single GPU |
| FP16      | 140 GB | Requires 2x H100 80GB |
| INT8      | 70 GB  | Fits on 1x H100 80GB |
| INT4      | 35 GB  | Fits on 1x A100 40GB |

Understanding precision is the first step toward quantization.

## What to implement

### Precision format utilities

- [ ] `float_info(dtype)` — return dict with sign_bits, exponent_bits, mantissa_bits, bias, max_value, min_positive, precision_decimal_digits
- [ ] `visualize_bits(value, dtype)` — return string showing sign/exponent/mantissa bit breakdown
- [ ] `bf16_simulate(x)` — simulate BF16 by truncating FP32 mantissa (since NumPy lacks native BF16)

### Range and precision analysis

- [ ] `demonstrate_overflow(dtype)` — show values that overflow to inf in FP16 but not FP32/BF16
- [ ] `demonstrate_underflow(dtype)` — show small values that underflow to zero
- [ ] `precision_loss(x, dtype)` — compute relative error when converting FP32 to lower precision
- [ ] `consecutive_value_gap(value, dtype)` — compute the gap to next representable number

### Mixed precision forward pass

- [ ] `mixed_precision_linear(x, W, b, compute_dtype, accumulate_dtype)` — linear layer with configurable precision
- [ ] `mixed_precision_matmul(A, B, compute_dtype, accumulate_dtype)` — matmul simulating tensor core behavior
- [ ] `stable_softmax(x, dtype)` — softmax with max subtraction for numerical stability in low precision

### Loss scaling

- [ ] `LossScaler` class with:
  - `scale_loss(loss)` — multiply loss by current scale factor
  - `unscale_grads(grads)` — divide gradients by scale factor
  - `update(has_overflow)` — adjust scale factor based on overflow detection
  - `get_scale()` — return current scale factor
- [ ] `detect_overflow(tensor)` — check for inf or nan values
- [ ] `dynamic_loss_scaling_demo(model, data, n_steps)` — demonstrate full training loop with dynamic scaling

### Gradient analysis

- [ ] `gradient_histogram(grads)` — show distribution of gradient magnitudes
- [ ] `count_underflow(grads, dtype)` — count gradients that would underflow to zero
- [ ] `effective_zero_threshold(dtype)` — compute the smallest gradient that survives conversion

### Memory analysis

- [ ] `model_memory_usage(n_params, dtype)` — compute bytes for model weights
- [ ] `kv_cache_memory(n_layers, n_heads, d_head, seq_len, dtype)` — compute KV cache size
- [ ] `memory_bandwidth_analysis(batch_size, seq_len, d_model)` — compare FP32 vs FP16 memory requirements

## Test cases to cover

### Bit layout verification

- [ ] **FP32 bit extraction:** Verify sign/exponent/mantissa extraction for known values (1.0, -1.0, 0.5, 2.0)
- [ ] **FP16 special values:** Confirm inf, -inf, nan, 0, -0 are correctly represented
- [ ] **BF16 simulation:** Verify bf16_simulate produces same result as FP32 truncation
- [ ] **Conversion roundtrip:** FP32 -> FP16 -> FP32 for values in valid FP16 range

### Overflow behavior

- [ ] **FP16 overflow threshold:** Verify 65504 is representable, 65520 overflows to inf
- [ ] **Gradual overflow:** Test values 65000, 65500, 65504, 65505, 65536 in FP16
- [ ] **BF16 no overflow:** Same large values that overflow FP16 are fine in BF16
- [ ] **Inf propagation:** Verify inf + finite = inf, inf - inf = nan

### Underflow behavior

- [ ] **FP16 underflow threshold:** Values below $\sim 5.96 \times 10^{-8}$ underflow to zero
- [ ] **Subnormal range:** Test values in FP16 subnormal range ($5.96 \times 10^{-8}$ to $6.1 \times 10^{-5}$)
- [ ] **Gradient underflow:** Simulate small gradients ($10^{-6}$, $10^{-7}$, $10^{-8}$) converting to FP16
- [ ] **Accumulation underflow:** Sum of many small FP16 values vs FP32 accumulation

### Precision loss measurement

- [ ] **Relative error:** Compute relative error for random FP32 values converted to FP16/BF16
- [ ] **Large value precision:** At value 1000, verify FP16 rounds to nearest 0.5
- [ ] **Small value precision:** At value 0.001, verify FP16 has higher relative error
- [ ] **BF16 vs FP16:** Verify BF16 has lower precision but same range as FP32

### Consecutive value gaps

- [ ] **Gap at 1.0:** FP16 gap is $\sim 0.001$, FP32 gap is $\sim 10^{-7}$
- [ ] **Gap scaling:** Gap doubles when value doubles (within same exponent range)
- [ ] **Integer representability:** Verify which integers are exactly representable in FP16 (up to 2048)

### Softmax stability

- [ ] **Large logits FP16:** softmax([100, 100, 100]) without max subtraction overflows
- [ ] **Stable softmax:** Same input with max subtraction produces valid probabilities
- [ ] **Softmax sum:** Output probabilities sum to 1.0 (within precision tolerance)
- [ ] **Known output:** softmax([0, 0, 0]) = [1/3, 1/3, 1/3]

### Loss scaling

- [ ] **Scale factor effect:** Gradient $10^{-6} \times$ scale $10^{4} = 10^{-2}$ (survives FP16)
- [ ] **Unscaling correctness:** Scaled gradients / scale = original gradients
- [ ] **Overflow detection:** Gradient inf triggers scale reduction
- [ ] **Dynamic scaling:** Scale increases after $N$ successful steps, decreases on overflow
- [ ] **No NaN propagation:** After overflow detection and skipping, weights remain valid

### Matrix multiply accumulation

- [ ] **FP16 accumulation error:** Large matrix multiply accumulates error in FP16
- [ ] **FP32 accumulation:** Same multiply with FP32 accumulator matches FP32 baseline
- [ ] **Relative error vs matrix size:** Error grows with matrix dimensions in pure FP16

### Memory calculations

- [ ] **Model size:** 7B params in FP16 = 14 GB (exactly)
- [ ] **KV cache size:** Known architecture produces expected cache size
- [ ] **Precision comparison:** FP32 model is exactly 2x FP16 model size

### Gradient histogram analysis

- [ ] **Typical gradient distribution:** Most gradients are small (< 0.01)
- [ ] **Underflow percentage:** Compute what fraction of gradients underflow without scaling
- [ ] **With scaling:** Same gradients with scale factor have near-zero underflow

### End-to-end training comparison

- [ ] **FP32 baseline:** Train simple model in FP32, record final loss
- [ ] **FP16 with scaling:** Same model/data with FP16 and loss scaling achieves similar loss
- [ ] **FP16 without scaling:** Verify training diverges or stalls without loss scaling
- [ ] **BF16 without scaling:** Training succeeds without loss scaling (BF16 doesn't need it)

### Edge cases

- [ ] **Zero input:** All operations handle zero correctly
- [ ] **Negative values:** Precision loss is symmetric for positive and negative
- [ ] **Empty arrays:** Functions handle empty inputs gracefully
- [ ] **Single element:** Softmax of single element is 1.0

### Against known values

- [ ] **IEEE 754 compliance:** Specific bit patterns produce expected values
- [ ] **Hardware consistency:** Results match what GPU hardware produces (if testable)
- [ ] **Framework comparison:** Results match PyTorch/JAX half precision (for test validation only)

## Implementation notes

### Simulating BF16 in NumPy

NumPy doesn't have native BF16 support. Simulate it by truncating FP32:

```python
def to_bf16(x):
    """Simulate BF16 by zeroing lower 16 bits of FP32 mantissa."""
    x = np.asarray(x, dtype=np.float32)
    # View as uint32, mask, view back
    x_int = x.view(np.uint32)
    x_int = x_int & np.uint32(0xFFFF0000)
    return x_int.view(np.float32)
```

### Extracting bit fields

```python
def extract_fp32_parts(x):
    """Extract sign, exponent, mantissa from FP32."""
    x = np.float32(x)
    bits = x.view(np.uint32)
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return sign, exponent, mantissa
```

### Accumulation with higher precision

```python
def matmul_mixed(A, B, compute_dtype=np.float16, accum_dtype=np.float32):
    """Matrix multiply with low-precision compute, high-precision accumulation."""
    A_low = A.astype(compute_dtype)
    B_low = B.astype(compute_dtype)

    # Simulate tensor core behavior: compute in low precision, accumulate in high
    M, K = A_low.shape
    K2, N = B_low.shape

    result = np.zeros((M, N), dtype=accum_dtype)
    for i in range(M):
        for j in range(N):
            # Each dot product accumulates in high precision
            result[i, j] = np.dot(
                A_low[i, :].astype(accum_dtype),
                B_low[:, j].astype(accum_dtype)
            )
    return result.astype(compute_dtype)
```

Note: In practice, use NumPy's built-in matmul with appropriate dtypes. The loop above is for illustration only.

### Numerical stability in softmax

```python
def stable_softmax_fp16(x):
    """Softmax that works in FP16 without overflow."""
    x = x.astype(np.float16)
    x_max = np.max(x, axis=-1, keepdims=True)
    x_shifted = x - x_max  # Now max is 0, exp(0) = 1
    exp_x = np.exp(x_shifted)
    # Accumulate sum in FP32 for precision
    sum_exp = np.sum(exp_x.astype(np.float32), axis=-1, keepdims=True)
    return (exp_x / sum_exp.astype(np.float16)).astype(np.float16)
```
