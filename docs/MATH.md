# Mathematical Framing (Section E)

This document works out the formal content to drop into the paper's Methodology
and Discussion sections. It is written so results.json data can be plugged
straight into the derivations once Section A/B runs complete.

**Status (updated after Day 2, 2 of 3 Section-A models complete):** Section 1's
memory-scaling fit now uses real measured data from OPT-350M and Pythia-410M
(Qwen2.5-0.5B still needed for a 3-point fit / to check linearity holds).
Section 3's error-asymmetry argument is now backed by real per-run
`mean_weight_rel_error` / `mean_activation_rel_error` numbers logged by
`fakequant.py`, not just theory -- see the updated §3 for a genuinely
interesting wrinkle this surfaced (similar relative error ratio on both
models, wildly different downstream perplexity impact).

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

### Empirical validation (real data, 2 of 3 models; Qwen2.5-0.5B pending)

Measured `peak_train_vram_mib` from `results/<model>.json`, averaged across
all 4 QAT configs per model (strategy barely moves this number -- OPT-350M's
4 configs ranged only 3633-3644 MiB, Pythia-410M's ranged only 4350-4368 MiB
-- confirming the fixed-cost term dominates over the strategy-dependent
FakeQuantize buffer overhead, as the model predicts):

| Model | P (M) | Predicted fixed (10P) | Measured peak (mean across 4 configs) | Residual (activations+ctx) |
|---|---|---|---|---|
| OPT-350M | 331.2 | 3.31 GB | 3.554 GB | 0.244 GB |
| Pythia-410M | 405.3 | 4.05 GB | 4.257 GB | 0.207 GB |
| Qwen2.5-0.5B | 494 | 4.94 GB | *(pending)* | |

Fitting `measured = a * P + b` on these 2 points (P in billions, V in GB):

  a = (4.257 - 3.554) / (0.4053 - 0.3312) = 0.703 / 0.0741 ≈ **9.49**
  b = 3.554 - 9.49 * 0.3312 ≈ **0.41 GB**

`a ≈ 9.49` lands almost exactly on the predicted `10` from the byte-counting
argument in §1 -- strong validation that the fixed-cost model (weights +
gradients + 8-bit optimizer state, all in the training-precision terms
assumed above) is the right accounting, not just a coincidence of scale.
`b ≈ 0.41 GB` is the activation + CUDA-context overhead at this pipeline's
fixed batch size (1), gradient accumulation (8), and sequence length (512) --
lower than the original informal 0.5-0.8 GB estimate, plausibly because
gradient checkpointing is doing more work than that estimate assumed. This is
a genuine, falsifiable fitted line now (not an assumption) -- the paper can
present it as Fig. X with these two points and the residual band, and it
should get a third point from Qwen2.5-0.5B to confirm the line stays straight
across a third architecture (RoPE + GQA) before treating it as a general law
rather than a two-point fit.

## 2. The QAT-feasibility crossover point

Given the fitted line `V_peak(P) = a*P + b` above and a hardware VRAM budget
`V_max`, the maximum feasible parameter count under full QAT is:

  P_max = (V_max - b) / a

Using the real fitted values (a=9.49, b=0.41 GB) against this machine's
6,141 MiB (5.996 GB) total VRAM:

  P_max = (5.996 - 0.41) / 9.49 ≈ **0.589 B**

(Using a more conservative usable budget that reserves ~0.5 GB for the CUDA
context and OS overhead observed in practice, V_max=5.5 GB: P_max ≈ 0.536 B.)

This lands close to, but meaningfully higher than, the original informal
estimate (~0.47-0.5B) -- the refined 2-point fit pushes the predicted ceiling
up slightly because the real `b` (activation overhead) came in lower than
assumed. This predicts Qwen2.5-0.5B (0.494B) should still fit comfortably
(consistent with it being selected as the "safe" third matrix model rather
than the frontier model), and that TinyLlama-1.1B or larger (Section B's
frontier model, ≥1.1B) should exceed the 6GB budget by roughly 1.9-2x --
Section B's frontier experiment tests this directly. Report both this
predicted `P_max` and the measured OOM point (or lack thereof) side by side
as the validation of this formula once Section B runs.

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
the paper's Related Work to support point (1); point (2) is stated as a
straightforward error-propagation argument, now with real supporting
measurements below rather than left purely theoretical.

### Real measured quantization error (point 1, verified)

`fakequant.py` logs `mean_weight_rel_error` and `mean_activation_rel_error`
per run (`||fake_quant(x) - x|| / ||x||`, averaged over every quantized layer
and every training step). Real numbers from the completed runs:

| Model | Mean weight rel. error | Mean activation rel. error | Ratio (act/weight) |
|---|---|---|---|
| OPT-350M | 0.78% | 3.9% | ~5.0x |
| Pythia-410M | 0.86% | 4.9% | ~5.7x |

This directly confirms point (1): activation fake-quantization introduces
roughly **5-6x more relative error** than weight fake-quantization on both
architectures tested, consistent with the claim that the moving-average
observer fits activations' higher-variance, heavier-tailed distribution
systematically worse than it fits the comparatively stable weight
distribution.

### A puzzle this surfaces, and why point (2) is the likely answer

The relative-error ratio above is strikingly *similar* across the two models
(~5.0x vs ~5.7x) -- but the downstream perplexity damage is wildly
*different*: activation-only QAT costs OPT-350M +3.3% PPL but costs
Pythia-410M +106% PPL (see `docs/reports/2026-08-24_session-2.md`, §2.1).
**Similar per-layer relative error, very different end-to-end degradation.**
This is exactly what point (2) -- compounding error across depth -- would
predict if the two architectures propagate a similarly-sized per-layer error
very differently: Pythia-410M's fused QKV projection concentrates three
downstream uses (query, key, value) into one quantized tensor, so a given
relative error there feeds into three separate attention computations
simultaneously, whereas OPT's three separate projections each carry
independent, smaller-blast-radius error. This is a plausible mechanism, not
yet proven -- confirming it would require measuring how a fixed-size
perturbation to a fused QKV output propagates through attention compared to
separate q/k/v perturbations, which is a good candidate for a small
additional experiment before the paper asserts this as a explanation rather
than a hypothesis.

## 4. What to actually compute once Section A data lands

1. ~~Fit `V_peak = a*P + b`~~ -- done with 2 of 3 models (§1); add
   Qwen2.5-0.5B's point once its matrix runs to confirm linearity holds
   across a third, structurally different architecture (RoPE + GQA).
2. Compute `P_max` from the fit, compare to the measured Section B outcome --
   formula gives P_max ≈ 0.54-0.59B (§2); not yet validated against Section
   B's actual frontier-model attempt (not run yet).
3. ~~Report per-layer activation quantization error norms~~ -- done (§3);
   the resulting puzzle (similar error ratio, very different PPL impact)
   is now the more interesting open question -- see the mechanism hypothesis
   above.
