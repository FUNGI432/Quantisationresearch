# Mathematical Framing (Section E)

This document works out the formal content to drop into the paper's Methodology
and Discussion sections. It is written so results.json data can be plugged
straight into the derivations once Section A/B runs complete.

## 1. QAT training memory model

Let P be parameter count, and consider mixed-precision QAT with an 8-bit
optimizer, training the model in FP32 (as our pipeline does -- the "master
weight" and "working weight" are the same FP32 tensor, since FakeQuant nodes
simulate INT8 in the forward/backward pass but the underlying parameter stays
full precision for the STE gradient update).

Fixed memory terms (bytes):
  - Weights (FP32):              4P
  - Gradients (FP32):            4P
  - Optimizer state (8-bit Adam, 2 moments quantized to INT8 + FP32 scale):
                                  ~2P  (bitsandbytes block-wise quantization)
  Total fixed:                   ~10P bytes

Variable term: activation memory A(L, B, S, H) where L=layers, B=batch,
S=sequence length, H=hidden size -- reduced by ~70% under gradient
checkpointing (recomputation trades this for +-20% compute time).

  V_peak = 10P + A_checkpointed(L, B, S, H) + V_cuda_context

with V_cuda_context ~= 0.5-0.8 GB.

### Empirical validation (fill in after Section A completes)

| Model | P (M) | Predicted fixed (10P) | Measured peak (training) | Residual (activations+ctx) |
|---|---|---|---|---|
| OPT-350M | 331 | 3.31 GB | 3.6-3.9 GB (from prior run) | ~0.3-0.6 GB |
| Pythia-410M | 405 | 4.05 GB | TBD | TBD |
| Qwen2.5-0.5B | 494 | 4.94 GB | TBD | TBD |

Fit a line `measured = a * P + b` across the matrix models; report `a` (should
be close to 10) and `b` (activation + context overhead, should be roughly
constant across models at fixed batch/seq-len). This is the citable
"memory-scaling law" the paper can present as Fig. X, and it's what makes the
frontier-model prediction in Section 2 a falsifiable claim rather than an
assertion.

## 2. The QAT-feasibility crossover point

Given the fitted line `V_peak(P) = a*P + b` and a hardware VRAM budget `V_max`
(6 GB here, minus CUDA context reservation), the maximum feasible parameter
count under full QAT is:

  P_max = (V_max - b) / a

With a ~= 10, b ~= 0.5-0.8 GB, V_max = 5.5 GB usable: P_max ~= 0.47-0.5B.
This matches the "danger zone" observed empirically for Qwen2.5-0.5B and
predicts that TinyLlama-1.1B (or any >=1B model) will exceed budget by roughly
a factor of 2x -- which Section B's frontier experiment tests directly. Report
both the predicted P_max and the measured OOM point (or lack thereof) side by
side as the validation of this formula.

## 3. Why the Straight-Through Estimator treats weights and activations asymmetrically

The STE approximates the gradient of the (non-differentiable) rounding
function as identity within the quantization range:

  quantize(x) = round(x / s) * s.       d/dx quantize(x) := 1   (STE)

For a weight tensor, this rounding is applied to a **static, per-channel**
parameter -- the effective quantization noise `e_w = quantize(w) - w` is
bounded by `s/2` and, since `s` is calibrated from the moving-average of a
distribution that only slowly drifts during training (a parameter, not an
activation), the noise is roughly i.i.d. and small relative to the range of
`w`. This is precisely the setting in which **weight-noise injection acts as
an explicit regularizer** on the loss landscape -- it is mathematically
equivalent to an approximate form of the perturbation used in flat-minima /
noise-injection regularization results (e.g. the classical link between
additive weight noise during training and an implicit penalty on the
sharpness of the loss around the trained parameters). This gives a principled
grounding for the empirical "weight-only QAT beats the FP16 control" result,
rather than leaving it as a post-hoc anecdote.

For an **activation** tensor, the same rounding is applied to a value that is
a function of the *current input* -- its distribution is highly
non-stationary within a single forward pass (per-token, per-position) and
subject to the outlier phenomenon documented in the literature (Wei et al.
2022; see docs/PLAN.md refs). Two consequences:

  1. The moving-average observer calibrating `s` for activations is chasing a
     much higher-variance, heavier-tailed target than the weight observer,
     so `s` is a systematically worse fit -> larger `e_a` on average.
  2. Because activation error at layer `l` propagates multiplicatively into
     layer `l+1`'s input, activation quantization noise **compounds across
     depth**, unlike weight noise, which is bounded per-layer and does not
     interact with the noise injected at other layers in the same way.

This is the formal argument for why "activations only" QAT degrades
performance while "weights only" QAT helps, and why "both" is worse than
either individually (the two noise sources compound rather than average out).
Cite Wei et al. (2022) and the SmoothQuant/AWQ outlier literature already in
the paper's Related Work to support point (1); point (2) can be stated as a
straightforward error-propagation argument and, if time allows, verified
empirically by measuring per-layer activation quantization error norms
across training (an easy addition to fakequant.py: log `(fake_quant(x) -
x).norm() / x.norm()` per layer, per step).

## 4. What to actually compute once Section A data lands

1. Fit `V_peak = a*P + b` across the 3 matrix models -> Table + Fig.
2. Compute `P_max` from the fit, compare to the measured Section B outcome.
3. Report per-layer activation quantization error norms (optional but strong)
   to substantiate the compounding-error argument in Section 3 empirically,
   not just theoretically.
