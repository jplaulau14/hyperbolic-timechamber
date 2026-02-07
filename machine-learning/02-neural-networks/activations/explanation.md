# Activation Functions -- Deep Dive

## The Intuition

### What Problem Are We Solving?

Imagine you stack ten linear transformations:

```
y = W_10 * (W_9 * (... (W_1 * x + b_1) ...) + b_9) + b_10
```

No matter how many layers you stack, the entire thing collapses into a single matrix multiply:

```
y = W' * x + b'
```

where `W' = W_10 * W_9 * ... * W_1` and `b'` is some combined bias. Ten layers, a hundred layers, a thousand -- they all produce the same expressive power as one layer. Your deep network is a fraud. It can only learn lines, planes, and hyperplanes.

This is the fundamental crisis that activation functions solve. By inserting a nonlinear function between every linear layer, you break the collapse. Each layer can now bend, twist, and fold the input space in ways that a linear transformation cannot. Stack enough of these nonlinear layers, and you get a **universal approximator** -- a network that can approximate any continuous function to arbitrary precision.

### The Key Insight

An activation function is deceptively simple: it is an element-wise operation applied independently to every number in a tensor. No matrix multiplies, no dot products, no interactions between elements. Just `f(x)` applied to each scalar. Yet this tiny per-element nonlinearity is the difference between a network that can only draw straight lines and one that can learn to recognize faces, translate languages, or generate code.

### Real-World Analogy

Think of a linear layer as a rubber sheet that can only be stretched, compressed, rotated, or shifted -- but never folded. An activation function is the fold. Between each rubber-sheet transformation, you fold the sheet. After many stretch-fold-stretch-fold cycles, you can crumple the sheet into any shape you want. Without the folds, no amount of stretching changes the fundamental flatness.

---

## The Evolution: Sigmoid --> ReLU --> GELU/SiLU

Before diving into each function, it helps to understand why the field kept inventing new activations. This is not academic churn -- each transition solved a concrete training failure mode.

```
Timeline of Dominance:

1990s-2012        2012-2017         2017-present
Sigmoid/Tanh  --> ReLU/LeakyReLU --> GELU/SiLU
   |                  |                  |
   |                  |                  |
   Problem:           Problem:           Why they win:
   Vanishing          Dying neurons,     Smooth, no dead
   gradients in       non-differentiable neurons, non-
   deep nets          at zero            monotonic,
                                         self-gating
```

**Sigmoid era (1990s-2012):** Sigmoid and tanh were the standard activations. They worked for shallow networks but catastrophically failed in deep ones. The gradient of sigmoid is at most 0.25 (at x=0) and quickly approaches 0 for large |x|. Chain ten of these together via backpropagation and the gradient shrinks by a factor of ~0.25^10 = 9.5e-7. The weights in early layers barely update. This is the **vanishing gradient problem**, and it limited practical networks to a few layers for decades.

**ReLU revolution (2012-2017):** ReLU's gradient is either 0 or 1 -- never a fractional value that causes shrinkage. For positive inputs, gradients flow through unchanged no matter how deep the network. This single property unlocked the training of deep networks (AlexNet, VGG, ResNet) and launched the modern deep learning era. But ReLU introduced a new problem: **dying neurons**. If a neuron's input is always negative (perhaps from an unlucky weight initialization or a too-large learning rate), its gradient is permanently zero and it never recovers.

**GELU/SiLU era (2017-present):** Modern activations combine the best properties. They are smooth everywhere (no gradient discontinuity at zero), they never have zero gradient for finite inputs (no dead neurons), and they have a subtle non-monotonic dip in the negative region that provides richer gradient signals. The field converged on GELU for encoder models (BERT, GPT-2/3) and SiLU for decoder models (LLaMA, Mistral).

---

## ReLU: The Breakthrough

### The Math

The simplest idea that could possibly work: keep positive values, zero out negative ones.

**Forward:**
```
ReLU(x) = max(0, x)
```

**Backward:**
```
ReLU'(x) = 1   if x > 0
            0   if x <= 0
```

At x = 0, the derivative is technically undefined (it is a corner). By convention, we use 0 (the subgradient).

### Worked Example

```
Input:       x = [-2.0, -1.0, 0.0, 1.0, 2.0]

Forward:
  max(0, -2.0) = 0.0
  max(0, -1.0) = 0.0
  max(0,  0.0) = 0.0
  max(0,  1.0) = 1.0
  max(0,  2.0) = 2.0

Output:      [0.0, 0.0, 0.0, 1.0, 2.0]

Backward (with grad_output = [1, 1, 1, 1, 1]):
  x = -2.0 <= 0, so gradient = 0.0
  x = -1.0 <= 0, so gradient = 0.0
  x =  0.0 <= 0, so gradient = 0.0   <-- convention: 0 at zero
  x =  1.0 >  0, so gradient = 1.0
  x =  2.0 >  0, so gradient = 1.0

Grad input:  [0.0, 0.0, 0.0, 1.0, 1.0]
```

### Implementation Walkthrough

```python
class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self._cache = {"x": x.copy()}
        return np.maximum(0.0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        x = self._cache["x"]
        return grad_output * (x > 0).astype(np.float64)
```

**Line-by-line:**
- `x.copy()`: Critical. We cache a copy of the input so that if the caller mutates their array after calling `forward`, our cached values are unaffected. The test `TestCacheIndependence` specifically verifies this.
- `np.maximum(0.0, x)`: Element-wise maximum. This is NOT `np.max()` (which finds the single largest element). `np.maximum` compares element-wise against 0.0.
- `(x > 0).astype(np.float64)`: Creates a boolean mask (True where x > 0, False elsewhere), then casts to float (1.0 or 0.0). Note that `x > 0` is False at x = 0, giving us the gradient = 0 convention.
- `grad_output * ...`: Element-wise multiplication applies the chain rule. Each position's incoming gradient is either passed through (multiplied by 1) or blocked (multiplied by 0).

### The Dying ReLU Problem

This is not a theoretical concern -- it happens in practice:

```
Scenario: A neuron with pre-activation always negative

  x values over time: [-0.3, -1.2, -0.7, -2.1, ...]
  ReLU output:        [ 0.0,  0.0,  0.0,  0.0, ...]
  Gradient:           [ 0.0,  0.0,  0.0,  0.0, ...]
                                                  ^
                                            Forever zero.
                                            Weights never update.
                                            Neuron is dead.
```

Once a neuron dies, it cannot recover because zero gradient means zero weight update. In a deep network with an aggressive learning rate, 10-40% of neurons can die during training. The test `TestVanishingGradients.test_dying_relu` demonstrates this directly: feeding all-negative inputs to ReLU produces an all-zero gradient.

---

## Leaky ReLU: The Simple Fix

### The Math

Instead of zeroing out negative values, scale them by a small constant alpha (typically 0.01).

**Forward:**
```
LeakyReLU(x) = x         if x > 0
               alpha * x  if x <= 0
```

**Backward:**
```
LeakyReLU'(x) = 1      if x > 0
                alpha   if x <= 0
```

### Worked Example

```
Input:       x = [-2.0, -1.0, 0.0, 1.0, 2.0],  alpha = 0.01

Forward:
  x = -2.0 <= 0: 0.01 * (-2.0) = -0.02
  x = -1.0 <= 0: 0.01 * (-1.0) = -0.01
  x =  0.0 <= 0: 0.01 *   0.0  =  0.00
  x =  1.0 >  0:                   1.00
  x =  2.0 >  0:                   2.00

Output:      [-0.02, -0.01, 0.0, 1.0, 2.0]

Backward (with grad_output = [1, 1, 1, 1, 1]):
  x = -2.0 <= 0: gradient = 0.01
  x = -1.0 <= 0: gradient = 0.01
  x =  0.0 <= 0: gradient = 0.01  <-- non-zero! Neuron can recover
  x =  1.0 >  0: gradient = 1.00
  x =  2.0 >  0: gradient = 1.00

Grad input:  [0.01, 0.01, 0.01, 1.0, 1.0]
```

### Implementation Walkthrough

```python
class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self._cache = {"x": x.copy()}
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        x = self._cache["x"]
        return grad_output * np.where(x > 0, 1.0, self.alpha)
```

**Key observations:**
- `np.where(x > 0, x, self.alpha * x)`: This is a vectorized if/else. Where the condition is True, pick from the second argument; where False, pick from the third.
- The `alpha` parameter is configurable. The tests verify two special cases: `alpha=0` degenerates to standard ReLU, and `alpha=1` becomes the identity function (`f(x) = x` everywhere). These are useful sanity checks.
- Unlike ReLU, the gradient at x = 0 is `alpha` (not zero), which means every neuron always receives some gradient signal. The test `test_leaky_relu_survives` confirms non-zero gradients for all-negative inputs.

---

## Sigmoid: The Classic

### The Math

Maps any real number to the range (0, 1). Used for centuries in statistics (as the logistic function), and the original neural network activation.

**Forward:**
```
sigmoid(x) = 1 / (1 + exp(-x))
```

**Backward:**
```
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            = s * (1 - s)
```

The derivative has an elegant form: it only depends on the output `s`, not the original input `x`. This means we can cache the forward output and reuse it in backward without recomputing the expensive exponential.

### Numerical Stability

The naive formula `1 / (1 + exp(-x))` fails for large negative x. When x = -1000, `exp(-(-1000)) = exp(1000)` overflows to infinity, producing `1 / inf = NaN` or `0/0`.

The fix is to branch on the sign of x:

```python
def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    pos = x >= 0
    neg = ~pos
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))    # exp(-x) is small, safe
    exp_x = np.exp(x[neg])                           # exp(x) is small when x < 0
    result[neg] = exp_x / (1.0 + exp_x)              # equivalent formula
    return result
```

**Why this works:** For x >= 0, we compute `exp(-x)` which is in (0, 1] -- safe. For x < 0, we compute `exp(x)` which is also in (0, 1) -- safe. We never compute an exponential of a large positive number.

The two formulas are mathematically identical:
```
1 / (1 + exp(-x)) = exp(x) / (exp(x) + 1)     [multiply top and bottom by exp(x)]
```

### Worked Example

```
Input:       x = [-2.0, 0.0, 2.0]

Forward:
  sigmoid(-2.0): x < 0, so exp(-2) / (1 + exp(-2))
               = 0.1353 / (1 + 0.1353)
               = 0.1353 / 1.1353
               = 0.1192

  sigmoid(0.0):  x >= 0, so 1 / (1 + exp(0))
               = 1 / (1 + 1)
               = 0.5

  sigmoid(2.0):  x >= 0, so 1 / (1 + exp(-2))
               = 1 / (1 + 0.1353)
               = 1 / 1.1353
               = 0.8808

Output:      [0.1192, 0.5, 0.8808]

Backward (with grad_output = [1, 1, 1]):
  s(-2.0) = 0.1192: 0.1192 * (1 - 0.1192) = 0.1192 * 0.8808 = 0.1050
  s( 0.0) = 0.5:    0.5    * (1 - 0.5)    = 0.5    * 0.5    = 0.2500  <-- maximum!
  s( 2.0) = 0.8808: 0.8808 * (1 - 0.8808) = 0.8808 * 0.1192 = 0.1050

Grad input:  [0.1050, 0.2500, 0.1050]
```

Notice the gradient peaks at 0.25 when x = 0 and quickly shrinks. At x = 10, sigmoid(10) = 0.99995, and the gradient is 0.99995 * 0.00005 = 0.0000045. This is the vanishing gradient problem. The test `test_sigmoid_gradient_shrinks` verifies that the gradient monotonically decreases as |x| increases.

### The Vanishing Gradient Problem Visualized

```
         Sigmoid Output                  Sigmoid Gradient
    1.0 |          ________            0.25 |       *
        |        /                          |      * *
    0.5 |------*-----------            0.12 |     *   *
        |    /                              |    *     *
    0.0 |___/                           0.0 |***       ***
        -5   0   5                          -5   0   5

  Saturates near 0 and 1           Gradient nearly zero for |x| > 3
  for large |x|                    Chain 10 layers: 0.25^10 ~ 10^-6
```

When you backpropagate through many sigmoid layers, these tiny gradients multiply together. After 10 layers, the gradient reaching the first layer is roughly 0.25^10 ~ one millionth of the output gradient. The early layers learn almost nothing. This is why sigmoid is no longer used in hidden layers of deep networks.

### Where Sigmoid Is Still Used

Despite its flaws in hidden layers, sigmoid remains essential:
1. **Output layer for binary classification** -- maps logits to probabilities in (0, 1)
2. **Gating mechanisms** -- in LSTMs, GRUs, and as a building block for SiLU
3. **Attention masks** -- converting raw scores to probabilities

---

## Tanh: The Zero-Centered Sigmoid

### The Math

**Forward:**
```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

Equivalently: `tanh(x) = 2 * sigmoid(2x) - 1`

**Backward:**
```
tanh'(x) = 1 - tanh(x)^2 = 1 - t^2
```

Like sigmoid, the derivative depends only on the output, so we cache it.

### Worked Example

```
Input:       x = [-1.0, 0.0, 1.0]

Forward:
  tanh(-1.0) = (0.3679 - 2.7183) / (0.3679 + 2.7183)
             = -2.3504 / 3.0862
             = -0.7616

  tanh(0.0)  = (1 - 1) / (1 + 1)
             = 0.0

  tanh(1.0)  = (2.7183 - 0.3679) / (2.7183 + 0.3679)
             = 2.3504 / 3.0862
             = 0.7616

Output:      [-0.7616, 0.0, 0.7616]

Backward (with grad_output = [1, 1, 1]):
  1 - (-0.7616)^2 = 1 - 0.5800 = 0.4200
  1 - (0.0)^2     = 1 - 0      = 1.0000   <-- maximum!
  1 - (0.7616)^2  = 1 - 0.5800 = 0.4200

Grad input:  [0.4200, 1.0000, 0.4200]
```

**Comparison with sigmoid:** Tanh's gradient peaks at 1.0 (vs sigmoid's 0.25), giving 4x stronger gradients near zero. It is also zero-centered, which means its outputs have a mean near zero -- helpful because the next layer receives inputs that are not systematically biased positive. However, tanh still saturates for large |x| and still causes vanishing gradients in deep networks.

### Implementation

```python
class Tanh(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        t = np.tanh(x)
        self._cache = {"t": t}
        return t

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        t = self._cache["t"]
        return grad_output * (1.0 - t ** 2)
```

The implementation uses `np.tanh()` directly, which handles numerical stability internally. Writing tanh from scratch with raw exponentials would require careful handling of overflow for large inputs.

---

## GELU: The Probabilistic Activation

### The Key Insight

GELU asks a question: "If x were a draw from a standard Gaussian distribution, how likely is it to be positive?" Then it scales x by that probability.

```
GELU(x) = x * Phi(x)
```

where `Phi(x) = P(X <= x)` for X ~ N(0, 1) -- the Gaussian CDF.

- For large positive x: Phi(x) is near 1, so GELU(x) is near x (pass through)
- For large negative x: Phi(x) is near 0, so GELU(x) is near 0 (suppress)
- Near zero: smooth, gradual transition

This is a probabilistic, smooth version of ReLU. Where ReLU makes a hard binary decision at zero, GELU makes a soft probabilistic one.

### The Math

**Exact forward (erf-based):**
```
GELU(x) = x * Phi(x)
         = x * 0.5 * (1 + erf(x / sqrt(2)))
```

**Tanh approximation (commonly used):**
```
c = sqrt(2/pi) ~ 0.7978845608

inner = c * (x + 0.044715 * x^3)
GELU(x) ~ 0.5 * x * (1 + tanh(inner))
```

**Exact backward:**
```
GELU'(x) = Phi(x) + x * phi(x)

where phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)  is the Gaussian PDF
```

**Tanh approximation backward:**
```
Let inner = c * (x + 0.044715 * x^3)
Let t = tanh(inner)

d(inner)/dx = c * (1 + 3 * 0.044715 * x^2)
            = c * (1 + 0.134145 * x^2)

GELU'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * d(inner)/dx
```

### Worked Example (Tanh Approximation)

```
Input:       x = [-1.0, 0.0, 1.0]
c = sqrt(2/pi) = 0.7979

--- x = -1.0 ---
inner = 0.7979 * (-1.0 + 0.044715 * (-1)^3)
      = 0.7979 * (-1.0 - 0.044715)
      = 0.7979 * (-1.044715)
      = -0.8336
t = tanh(-0.8336) = -0.6837

GELU(-1.0) = 0.5 * (-1.0) * (1 + (-0.6837))
           = 0.5 * (-1.0) * 0.3163
           = -0.1582

--- x = 0.0 ---
inner = 0.7979 * (0 + 0) = 0
t = tanh(0) = 0

GELU(0.0) = 0.5 * 0 * (1 + 0) = 0.0

--- x = 1.0 ---
inner = 0.7979 * (1.0 + 0.044715 * 1.0)
      = 0.7979 * 1.044715
      = 0.8336
t = tanh(0.8336) = 0.6837

GELU(1.0) = 0.5 * 1.0 * (1 + 0.6837)
          = 0.5 * 1.6837
          = 0.8419

Output:      [-0.1582, 0.0, 0.8419]
```

Notice the **asymmetry**: GELU(-1) = -0.1582 but GELU(1) = 0.8419. Negative inputs are strongly suppressed but not completely zeroed. The small negative dip at x ~ -0.17 gives GELU its characteristic non-monotonic shape.

### Backward Worked Example (Tanh Approximation)

```
Continuing from above, for x = 1.0:
  t = 0.6837
  sech^2 = 1 - 0.6837^2 = 1 - 0.4674 = 0.5326
  d(inner)/dx = 0.7979 * (1 + 0.134145 * 1.0^2)
              = 0.7979 * 1.134145
              = 0.9049

  GELU'(1.0) = 0.5 * (1 + 0.6837) + 0.5 * 1.0 * 0.5326 * 0.9049
             = 0.5 * 1.6837 + 0.5 * 0.4820
             = 0.8419 + 0.2410
             = 1.0829

For x = 0.0:
  t = 0
  sech^2 = 1
  d(inner)/dx = 0.7979 * 1.0 = 0.7979

  GELU'(0.0) = 0.5 * (1 + 0) + 0.5 * 0 * 1 * 0.7979
             = 0.5 + 0
             = 0.5
```

The gradient at x = 0 is 0.5 -- significantly stronger than sigmoid's 0.25. And unlike ReLU, the gradient transitions smoothly through zero rather than having a discontinuity.

### Implementation Walkthrough

```python
class GELU(Activation):
    SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)  # ~0.7978845608

    def __init__(self, approximate: bool = True):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self._cache = {"x": x.copy()}

        if self.approximate:
            inner = self.SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)
            t = np.tanh(inner)
            self._cache["t"] = t
            return 0.5 * x * (1.0 + t)
        else:
            phi = 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
            self._cache["phi"] = phi
            return x * phi

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        x = self._cache["x"]

        if self.approximate:
            t = self._cache["t"]
            sech2 = 1.0 - t ** 2
            d_inner = self.SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x ** 2)
            local_grad = 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
        else:
            phi = self._cache["phi"]
            pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
            local_grad = phi + x * pdf

        return grad_output * local_grad
```

**The tricky parts:**

- **`_erf` wrapper:** NumPy does not expose a vectorized erf function directly. The implementation uses `np.vectorize(math.erf)` to create one. This is slower than a native vectorized implementation but mathematically correct. In production, you would use `scipy.special.erf` or a CUDA intrinsic.

- **Why cache `t` but not `inner`?** The backward pass needs `t` (to compute `sech^2 = 1 - t^2`) and `x` (to compute `d_inner`). It does NOT need `inner` itself because tanh's derivative is expressed purely in terms of tanh's output. This is the same pattern as the standalone Tanh class.

- **`0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner`:** This is the product rule applied to `GELU(x) = 0.5 * x * (1 + t)`. The first term `0.5 * (1 + t)` is the derivative of the `0.5 * x` part (holding `(1+t)` constant). The second term is the derivative of `0.5 * (1 + t)` part (holding `x` constant), expanded through tanh's chain rule.

- **Approximation accuracy:** The test `test_approx_vs_exact` verifies that the tanh approximation stays within 0.005 of exact GELU across [-10, 10]. The constant 0.044715 in the cubic term was fit to minimize this error.

### Why GPT and BERT Use GELU

GELU's non-monotonic shape creates a richer loss landscape than ReLU. The small negative dip acts as a form of implicit regularization -- near-zero negative values produce slightly negative outputs before being suppressed. Empirically, this leads to faster convergence and better final accuracy in transformer training. The probabilistic interpretation also provides a principled motivation: the network learns which inputs are "likely enough" (under a Gaussian prior) to pass through.

---

## SiLU/Swish: The Self-Gating Activation

### The Key Insight

SiLU is beautifully simple: the input gates itself.

```
SiLU(x) = x * sigmoid(x)
```

The sigmoid of x acts as a gate that controls how much of x passes through. For large positive x, sigmoid(x) ~ 1, so the full value passes. For large negative x, sigmoid(x) ~ 0, so the value is suppressed. But unlike ReLU, the transition is smooth and the gate is learned implicitly from the data.

### The Math

**Forward:**
```
SiLU(x) = x * sigmoid(x)
```

**Backward:**
```
Let s = sigmoid(x)

SiLU'(x) = s + x * s * (1 - s)
          = s * (1 + x * (1 - s))
          = s * (1 + x - x * s)
```

This is the product rule: `d/dx [x * s] = s + x * ds/dx = s + x * s * (1-s)`.

### Worked Example

```
Input:       x = [-2.0, 0.0, 1.0]

Forward:
  s(-2.0) = sigmoid(-2.0) = exp(-2) / (1 + exp(-2))
          = 0.1353 / 1.1353 = 0.1192
  SiLU(-2.0) = -2.0 * 0.1192 = -0.2384

  s(0.0) = sigmoid(0.0) = 0.5
  SiLU(0.0) = 0.0 * 0.5 = 0.0

  s(1.0) = sigmoid(1.0) = 1 / (1 + exp(-1))
         = 1 / 1.3679 = 0.7311
  SiLU(1.0) = 1.0 * 0.7311 = 0.7311

Output:      [-0.2384, 0.0, 0.7311]

Backward (with grad_output = [1, 1, 1]):
  x = -2.0, s = 0.1192:
    s * (1 + x * (1 - s))
    = 0.1192 * (1 + (-2.0) * (1 - 0.1192))
    = 0.1192 * (1 + (-2.0) * 0.8808)
    = 0.1192 * (1 - 1.7616)
    = 0.1192 * (-0.7616)
    = -0.0908

  x = 0.0, s = 0.5:
    0.5 * (1 + 0 * 0.5) = 0.5

  x = 1.0, s = 0.7311:
    0.7311 * (1 + 1.0 * (1 - 0.7311))
    = 0.7311 * (1 + 0.2689)
    = 0.7311 * 1.2689
    = 0.9277

Grad input:  [-0.0908, 0.5, 0.9277]
```

Key observation: at x = -2.0, the gradient is -0.0908. It is **negative**, meaning the function is still decreasing at that point. This is the non-monotonic behavior -- SiLU dips below zero before approaching zero from below. The minimum of SiLU is approximately -0.278 at x ~ -1.28.

### Implementation Walkthrough

```python
class SiLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        s = _stable_sigmoid(x)
        self._cache = {"x": x.copy(), "s": s}
        return x * s

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._check_cache()
        x = self._cache["x"]
        s = self._cache["s"]
        local_grad = s * (1.0 + x * (1.0 - s))
        return grad_output * local_grad
```

**Key observations:**
- SiLU caches both `x` and `s = sigmoid(x)`. The backward needs both.
- `_stable_sigmoid` is reused from the top-level function, ensuring consistent numerical stability.
- The `Swish = SiLU` alias at module level makes the two names interchangeable. The test `test_swish_alias` verifies `Swish is SiLU` (identity, not just equality).

### Why LLaMA and Mistral Use SiLU

LLaMA-family models use SiLU in the **SwiGLU** feedforward network pattern:

```
Standard FFN (GPT-2 style):
  FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

SwiGLU FFN (LLaMA style):
  FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
```

In SwiGLU, there are two parallel projections from the input: one goes through SiLU (the "gate"), the other passes through unchanged (the "up projection"). They are multiplied element-wise, then projected back down. This gated architecture consistently outperforms the standard FFN in ablation studies, and SiLU's self-gating property complements the explicit gating structure.

```
SwiGLU Architecture:

     Input x
       |
   ----+----
   |       |
   v       v
 x @ W_g  x @ W_u       Two parallel linear projections
   |       |
   v       |
  SiLU     |             Gate activated by SiLU
   |       |
   +---*---+             Element-wise multiply (gating)
       |
       v
   result @ W_d           Down-project back to model dim
       |
       v
     Output
```

---

## The Base Class and Gradient Checking

### The Activation Interface

```python
class Activation(ABC):
    def __init__(self):
        self._cache: Optional[dict] = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray: ...

    def _check_cache(self) -> None:
        if self._cache is None:
            raise RuntimeError(
                "backward() called before forward(). Run forward() first."
            )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
```

This design enforces a protocol:
1. Call `forward(x)` first -- it computes the output AND caches intermediate values.
2. Call `backward(grad_output)` to get the gradient -- it uses cached values.
3. Calling `backward` without a prior `forward` raises `RuntimeError`.

The `__call__` method allows using activation instances as functions: `relu(x)` instead of `relu.forward(x)`.

### Gradient Checking

```python
def gradient_check(activation, x, h=1e-5):
    x = np.asarray(x, dtype=np.float64)
    activation.forward(x)
    grad_output = np.ones_like(x)
    analytical = activation.backward(grad_output)

    numerical = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += h
        x_minus[idx] -= h
        f_plus = activation.forward(x_plus)
        f_minus = activation.forward(x_minus)
        numerical[idx] = (f_plus[idx] - f_minus[idx]) / (2.0 * h)
        it.iternext()

    activation.forward(x)  # Restore cache

    abs_error = np.abs(analytical - numerical)
    max_abs_error = float(np.max(abs_error)) if abs_error.size > 0 else 0.0
    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-8)
    rel_error = abs_error / denom
    max_rel_error = float(np.max(rel_error)) if rel_error.size > 0 else 0.0

    return {"max_abs_error": max_abs_error, "max_rel_error": max_rel_error,
            "passed": max_rel_error < 1e-5}
```

This uses **central finite differences**: `f'(x) ~ (f(x+h) - f(x-h)) / (2h)`. This is O(h^2) accurate, much better than the forward difference `(f(x+h) - f(x)) / h` which is only O(h) accurate.

**Why iterate element by element?** For element-wise activations, this is technically unnecessary since each output depends only on the corresponding input. But the `np.nditer` approach is general -- it works for any function, including future implementations where outputs depend on multiple inputs.

**The restore step** (`activation.forward(x)` at the end) puts the cache back to the original input, since the numerical computation runs `forward` with many different perturbed inputs.

---

## Gradient Flow Comparison

This is where the practical impact of activation choice becomes visible. Consider the gradient magnitude as a function of x for each activation:

```
Gradient Magnitude vs. Input Value

ReLU:
  grad |  1 |         _______________
       |    |        |
       |  0 |________|
       |    +----|----+---->
       |       -2  0  2     x
       Dead zone: all x < 0

Leaky ReLU (alpha=0.01):
  grad |  1 |         _______________
       |    |        |
  0.01 |....|........|                 (small but non-zero)
       |    +----|----+---->
       |       -2  0  2     x
       No dead zone

Sigmoid:
  grad | .25|       *
       |    |      * *
       | .12|     *   *
       |    |    *     *
       |  0 |***       ***
       |    +----|----+---->
       |       -2  0  2     x
       Maximum 0.25, vanishes quickly

Tanh:
  grad |  1 |       *
       |    |      * *
       | .5 |     *   *
       |    |    *     *
       |  0 |***       ***
       |    +----|----+---->
       |       -2  0  2     x
       Maximum 1.0, 4x better than sigmoid

GELU:
  grad | 1.1|          *
       |    |       * *
       | .5 |      *
       |    |   *
       |  0 |  *
       | -.1| *
       |    +----|----+---->
       |       -2  0  2     x
       Smooth, slightly exceeds 1.0

SiLU:
  grad | 1.1|           *
       |  .5|       * *
       |    |      *
       |  0 |---*--
       | -.1|  *
       |    +----|----+---->
       |       -2  0  2     x
       Smooth, non-monotonic, slightly exceeds 1.0
```

**What to notice:**
- ReLU has a hard discontinuity at zero. Left of zero: nothing. Right of zero: everything.
- Sigmoid and Tanh vanish for large |x|. Sigmoid's max gradient (0.25) is four times weaker than Tanh's (1.0).
- GELU and SiLU have gradients that slightly exceed 1.0 for moderate positive x, then settle toward 1.0 for large x. They never go to zero for finite inputs. Their non-monotonic dip in the negative region provides gradient signal even for slightly negative inputs.

### Gradient Flow Through Multiple Layers

The real impact shows when you chain layers. After L layers, the gradient reaching the first layer is roughly:

```
                 L
 grad_1  ~  PI  f'(x_i)    (product of local gradients)
               i=1

Activation        Gradient range     After 10 layers
---------------------------------------------------------
Sigmoid           (0, 0.25]          Worst case: 0.25^10 ~ 10^-6
Tanh              (0, 1.0]           Worst case: still shrinks
ReLU              {0, 1}             Binary: either 0 or 1^10 = 1
GELU              ~(-0.17, 1.08)     Stays near 1 for positive path
SiLU              ~(-0.10, 1.10)     Stays near 1 for positive path
```

ReLU solves vanishing gradients by having gradient exactly 1 in the active region. GELU and SiLU solve it similarly but with smooth transitions, eliminating the dead neuron problem that ReLU introduces.

---

## From Math to Code: The Data Structures

Every activation class maintains a simple data structure:

```
Activation instance
  |
  +-- _cache: Optional[dict]
        |
        +-- None  (before forward, or after reset)
        |
        +-- {"x": np.ndarray, ...}  (after forward)
             |
             +-- "x"   : copy of input (ReLU, LeakyReLU, GELU, SiLU)
             +-- "s"   : sigmoid output (Sigmoid, SiLU)
             +-- "t"   : tanh output (Tanh, GELU approximate)
             +-- "phi" : Gaussian CDF values (GELU exact)
```

**What each activation caches:**

| Activation | Caches | Why |
|-----------|--------|-----|
| ReLU | `x` | Need sign of x for backward mask |
| LeakyReLU | `x` | Need sign of x for backward mask |
| Sigmoid | `s = sigmoid(x)` | Backward is `s * (1 - s)`, no need for x |
| Tanh | `t = tanh(x)` | Backward is `1 - t^2`, no need for x |
| GELU (approx) | `x`, `t = tanh(inner)` | Need both for the product-rule backward |
| GELU (exact) | `x`, `phi = CDF(x)` | Need both for `phi + x * pdf` |
| SiLU | `x`, `s = sigmoid(x)` | Need both for `s * (1 + x * (1 - s))` |

Sigmoid and Tanh are especially elegant: their backward passes use only the forward output, not the original input. This means they cache less data. In contrast, GELU and SiLU need both the input and an intermediate value, requiring more memory.

---

## Complexity Analysis

### Time Complexity

| Activation | Forward | Backward | Dominant Operation |
|-----------|---------|----------|--------------------|
| ReLU | O(n) | O(n) | Comparison + copy |
| LeakyReLU | O(n) | O(n) | Comparison + multiply |
| Sigmoid | O(n) | O(n) | exp() per element |
| Tanh | O(n) | O(n) | exp() per element (inside np.tanh) |
| GELU (approx) | O(n) | O(n) | tanh() per element (which computes exp) |
| GELU (exact) | O(n) | O(n) | erf() per element + exp() in backward |
| SiLU | O(n) | O(n) | exp() per element (inside sigmoid) |

where n is the total number of elements in the input tensor.

All activations are O(n) because they are element-wise. The constant factor differs significantly: ReLU involves a comparison and conditional copy (extremely cheap), while GELU involves a cubic polynomial, a tanh (which computes two exponentials internally), and several multiplies. In practice, ReLU is roughly 3-5x faster than GELU/SiLU when measured in isolation.

### Space Complexity

| Activation | Cache Size | Why |
|-----------|-----------|-----|
| ReLU | O(n) | Stores copy of x |
| LeakyReLU | O(n) | Stores copy of x |
| Sigmoid | O(n) | Stores s (same size as x) |
| Tanh | O(n) | Stores t (same size as x) |
| GELU (approx) | O(2n) | Stores both x and t |
| GELU (exact) | O(2n) | Stores both x and phi |
| SiLU | O(2n) | Stores both x and s |

GELU and SiLU use twice the cache memory of simpler activations. For a transformer with hidden dim 4096, batch size 32, and sequence length 2048, that is `32 * 2048 * 4096 * 8 bytes * 2 = 4 GB` of cached activation values per layer (in float64). This is one reason gradient checkpointing exists -- you can discard these caches and recompute them during backward at the cost of extra compute.

### The Bottleneck

For activations running on GPU, the bottleneck is **memory bandwidth, not compute**. Each element requires only a few FLOPs (even GELU is under 20 FLOPs), but reading and writing the tensor from/to global memory takes far longer. The arithmetic intensity (FLOPs per byte) is well below 1, making activations memory-bound.

This is why kernel fusion is the primary optimization -- not because the activation itself is slow, but because eliminating the memory round-trip for the activation tensor saves more time than the activation compute itself costs.

---

## Common Pitfalls

### Pitfall 1: Unstable Sigmoid

**The mistake:**
```python
# Wrong -- overflows for large negative x
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

sigmoid(-1000)  # exp(1000) = inf -> 1/inf = NaN or 0/0
```

**Why it is wrong:** `np.exp(-(-1000)) = np.exp(1000)` overflows float64 (max ~ 1.8e308), producing `inf`. Then `1 / (1 + inf)` produces 0.0 in some implementations but can produce NaN in edge cases, and more importantly, any downstream computation with the overflow intermediate can produce garbage.

**The fix:**
```python
# Correct -- branch on sign
def _stable_sigmoid(x):
    result = np.zeros_like(x)
    pos = x >= 0
    neg = ~pos
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))  # exp of negative: safe
    exp_x = np.exp(x[neg])                         # exp of negative: safe
    result[neg] = exp_x / (1.0 + exp_x)
    return result
```

### Pitfall 2: Forgetting to Copy Cached Input

**The mistake:**
```python
# Wrong -- cache holds a reference, not a copy
def forward(self, x):
    self._cache = {"x": x}  # reference to caller's array
    return np.maximum(0.0, x)
```

**Why it is wrong:** If the caller modifies `x` after calling `forward`, the cached value changes too. When `backward` runs later, it sees the mutated data, not the original input.

```python
act = ReLU()
x = np.array([1.0, -1.0, 2.0])
act.forward(x)
x[:] = -5.0  # mutate original
grad = act.backward(np.ones(3))
# Expected: [1.0, 0.0, 1.0] (based on original x)
# Got:      [0.0, 0.0, 0.0] (based on mutated x = all -5.0)
```

**The fix:**
```python
# Correct -- cache a copy
def forward(self, x):
    self._cache = {"x": x.copy()}
    return np.maximum(0.0, x)
```

### Pitfall 3: Using np.max Instead of np.maximum

**The mistake:**
```python
# Wrong -- np.max returns a single scalar
def forward(self, x):
    return np.max(0.0, x)  # TypeError or wrong behavior
```

**Why it is wrong:** `np.max(array)` returns the maximum element of the array. `np.maximum(a, b)` computes element-wise maximum. These are completely different operations.

**The fix:**
```python
# Correct -- element-wise maximum
def forward(self, x):
    return np.maximum(0.0, x)
```

### Pitfall 4: Wrong GELU Derivative

**The mistake:**
```python
# Wrong -- forgetting the chain rule through tanh
def backward(self, grad_output):
    t = self._cache["t"]
    return grad_output * 0.5 * (1 + t)  # Missing the second term!
```

**Why it is wrong:** `GELU(x) = 0.5 * x * (1 + t(x))` where `t` depends on `x`. The product rule gives TWO terms: `d/dx [x * g(x)] = g(x) + x * g'(x)`. The code above only has the first term.

**The fix:**
```python
# Correct -- full product rule
def backward(self, grad_output):
    x = self._cache["x"]
    t = self._cache["t"]
    sech2 = 1.0 - t ** 2
    d_inner = self.SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x ** 2)
    local_grad = 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
    return grad_output * local_grad
```

---

## Connection to Inference Optimization

### What Gets Optimized

Activations are **memory-bandwidth-bound** operations. The arithmetic intensity for GELU is roughly:

```
~15 FLOPs per element / 16 bytes per element (read + write float64) ~ 1 FLOP/byte
```

Modern GPUs can do ~300 TFLOPS but only move ~3 TB/s. To saturate the compute, you need ~100 FLOPs/byte. Activations are 100x below that threshold. The GPU spends almost all its time waiting for memory, not computing.

### Kernel Fusion -- The Primary Optimization

The key insight: if the activation is fused into the preceding matrix multiply, you never write the intermediate tensor to memory at all.

```
Unfused (3 kernel launches, 3 memory round-trips):

  Kernel 1: h = x @ W1 + b1     Read: x, W1, b1    Write: h
  Kernel 2: h = gelu(h)         Read: h             Write: h    <-- WASTED
  Kernel 3: out = h @ W2 + b2   Read: h, W2, b2     Write: out

Fused (2 kernel launches, 2 memory round-trips):

  Kernel 1: h = gelu(x @ W1 + b1)   Read: x, W1, b1    Write: h
  Kernel 2: out = h @ W2 + b2       Read: h, W2, b2     Write: out

Savings: eliminated one full read+write of h (size: batch * seq * hidden)
```

For a typical transformer layer with hidden dim 4096 and batch*seq = 65536, the intermediate tensor `h` is 65536 * 4096 * 4 bytes = 1 GB (in float32). Eliminating one round-trip of this tensor at 3 TB/s bandwidth saves ~0.33 ms. Across all layers and all forward passes, this adds up.

### From Naive to Optimized

| Aspect | Naive (our implementation) | Optimized (production) |
|--------|---------------------------|------------------------|
| Backend | NumPy on CPU | CUDA/Triton on GPU |
| Fusion | Separate function call | Fused into GEMM epilogue |
| Memory | Reads/writes full tensor | Computes inline, no extra write |
| Precision | float64 | float16/bfloat16 (2-4x less memory traffic) |
| Caching | Python dict with full copy | Register-level reuse in fused kernel |
| Throughput | ~1 GFLOP/s | ~100 TFLOP/s |

### How Each Activation Maps to CUDA

```
cuBLAS epilogue fusion options:
  CUBLASLT_EPILOGUE_RELU        -- ReLU fused after matmul
  CUBLASLT_EPILOGUE_GELU        -- GELU fused after matmul
  CUBLASLT_EPILOGUE_GELU_AUX    -- GELU with aux output for backward
  CUBLASLT_EPILOGUE_SWISH       -- SiLU/Swish fused after matmul

Triton example (GELU fused with matmul):
  @triton.jit
  def fused_matmul_gelu(x_ptr, w_ptr, out_ptr, ...):
      # Load tiles, compute matmul in registers
      acc = tl.dot(x_tile, w_tile)
      # Apply GELU inline -- no memory write between matmul and activation
      inner = 0.7979 * (acc + 0.044715 * acc * acc * acc)
      out = 0.5 * acc * (1.0 + tl.math.tanh(inner))
      tl.store(out_ptr, out)
```

Understanding the naive NumPy version -- the exact formula, the derivative, the numerical stability requirements -- is prerequisite to writing these fused kernels. You need to know what to inline.

### SwiGLU Fusion

For LLaMA-style models, the fusion opportunity is even richer:

```
Unfused SwiGLU (4 kernels):
  gate = x @ W_gate           # GEMM
  gate = silu(gate)            # Activation
  up   = x @ W_up             # GEMM
  h    = gate * up             # Element-wise multiply

Fused SwiGLU (2 kernels or even 1):
  gate, up = fused_dual_gemm(x, W_gate, W_up)   # Two GEMMs, one read of x
  h = silu(gate) * up                            # Fused activation + multiply
```

The SiLU activation and the element-wise multiply are fused into a single kernel, eliminating two memory round-trips.

---

## Testing Your Understanding

### Quick Checks

1. **What happens if you remove all activation functions from a 10-layer network?** The entire network collapses into a single linear transformation `y = W'x + b'`. Training it is equivalent to training one layer, regardless of depth.

2. **Why does sigmoid's backward pass not need the original input x?** Because `sigmoid'(x) = s * (1 - s)` where `s = sigmoid(x)`. The derivative is expressed purely in terms of the output. This is a special property of the logistic function.

3. **Why does GELU need to cache x but Sigmoid does not?** GELU's backward uses the product rule on `0.5 * x * (1 + t)`, which requires `x` explicitly. Sigmoid's backward only needs `s = sigmoid(x)`.

4. **What is the maximum gradient of sigmoid, and at what input?** Maximum is 0.25, at x = 0. This is why sigmoid causes vanishing gradients -- even at its best, it attenuates gradients by 4x.

5. **What is the output shape of any activation function?** Identical to the input shape. All activations are element-wise.

### Exercises

1. **Easy**: Modify the `LeakyReLU` class to implement **PReLU** (Parametric ReLU), where alpha is a learnable parameter with its own gradient. Add a `backward_alpha` method that returns the gradient with respect to alpha.

2. **Medium**: Implement **ELU** (Exponential Linear Unit): `f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0`. Include both forward and backward, and verify with `gradient_check`.

3. **Hard**: Implement a **fused** GELU-Linear operation where `forward(x, W, b)` computes `GELU(x @ W + b)` in a single pass over the output elements (no intermediate tensor for the pre-activation). The backward should return gradients for x, W, and b.

---

## Summary

### Key Takeaways

- Without nonlinear activations, deep networks collapse to a single linear layer. Activations are what make depth useful.
- Sigmoid and Tanh suffer from vanishing gradients: their derivatives approach zero for large |x|, killing gradient flow in deep networks.
- ReLU fixed vanishing gradients with a gradient of exactly 0 or 1, but introduced the dying neuron problem where neurons with persistently negative inputs stop learning forever.
- GELU and SiLU are smooth, non-monotonic, and never have zero gradient for finite inputs. They dominate modern transformers: GELU in GPT/BERT, SiLU in LLaMA/Mistral.
- Activations are memory-bound, not compute-bound. The primary optimization is kernel fusion -- inlining the activation into the preceding matmul to eliminate a memory round-trip. Understanding the forward and backward formulas is prerequisite to writing these fused kernels.

### Quick Reference

```
Activation Functions
|
+-- ReLU:        max(0, x)
|   Forward:  O(n) -- comparison + copy
|   Backward: O(n) -- mask multiply
|   Gradient: {0, 1} -- binary, dead zone for x <= 0
|
+-- LeakyReLU:   max(alpha*x, x)
|   Forward:  O(n) -- comparison + multiply
|   Backward: O(n) -- mask multiply
|   Gradient: {alpha, 1} -- no dead zone
|
+-- Sigmoid:     1 / (1 + exp(-x))
|   Forward:  O(n) -- exp per element
|   Backward: O(n) -- s * (1 - s), reuses cached s
|   Gradient: (0, 0.25] -- vanishes for large |x|
|
+-- Tanh:        (exp(x) - exp(-x)) / (exp(x) + exp(-x))
|   Forward:  O(n) -- exp per element
|   Backward: O(n) -- 1 - t^2, reuses cached t
|   Gradient: (0, 1.0] -- vanishes for large |x|, 4x better than sigmoid
|
+-- GELU:        x * Phi(x) ~ 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))
|   Forward:  O(n) -- cubic + tanh per element
|   Backward: O(n) -- product rule with sech^2
|   Gradient: ~(-0.17, 1.08) -- smooth, non-monotonic
|   Used by:  GPT-2, GPT-3, BERT
|
+-- SiLU/Swish:  x * sigmoid(x)
    Forward:  O(n) -- exp per element
    Backward: O(n) -- s * (1 + x * (1 - s)), reuses cached s
    Gradient: ~(-0.10, 1.10) -- smooth, non-monotonic, self-gating
    Used by:  LLaMA, Mistral, Gemma (in SwiGLU FFN)

Optimized by: GEMM epilogue fusion (cuBLAS, Triton), half-precision compute
```
