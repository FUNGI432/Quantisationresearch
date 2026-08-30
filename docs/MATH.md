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

### Empirical validation (real data, all 3 matrix models) -- the params-only model breaks down on Qwen

Measured `peak_train_vram_mib` from `results/<model>.json`, averaged across
all 4 QAT configs per model for OPT-350M/Pythia-410M (strategy barely moves
this number for them -- OPT-350M's 4 configs ranged only 3633-3644 MiB,
Pythia-410M's ranged only 4350-4368 MiB -- confirming the fixed-cost term
dominates over the strategy-dependent FakeQuantize buffer overhead, as the
model predicts):

| Model | P (M) | Vocab size | Predicted fixed (10P) | Measured peak (training) | Residual |
|---|---|---|---|---|---|
| OPT-350M | 331.2 | ~50,272 | 3.31 GB | 3.554 GB (mean of 4 configs) | 0.244 GB |
| Pythia-410M | 405.3 | ~50,304 | 4.05 GB | 4.257 GB (mean of 4 configs) | 0.207 GB |
| Qwen2.5-0.5B | 494.0 | **151,936** | 4.94 GB | **6.62 GB** (clean isolated repro, `none` strategy; see below) | **1.68 GB** |

Qwen's residual (1.68 GB) is **7-8x larger** than OPT's or Pythia's
(~0.2-0.24 GB) relative to the same `10P` fixed-cost prediction. The
2-point fit from OPT/Pythia alone (`a=9.49, b=0.41 GB`, see below) predicts
5.10 GB for Qwen -- still off by 1.52 GB from the measured 6.62 GB. **The
params-only linear model does not hold for Qwen2.5-0.5B.** The leading
explanation, found via direct profiling on Day 5
(`docs/reports/2026-08-30_session-5.md` §5) rather than assumed: Qwen's
vocabulary is ~3x larger than OPT's/Pythia's (151,936 vs ~50,300), which
inflates the final logits/loss tensor (shape `[batch, seq_len, vocab_size]`)
disproportionately to parameter count -- a large vocabulary contributes
substantial *activation* memory (via the embedding and LM head, and their
gradients) that scales with `vocab_size`, not just with total `P`, and this
model's `10P` term only accounts for weights+gradients+optimizer state, not
this vocabulary-driven activation cost.

**Practical note on the two Qwen numbers in the table**: the real production
run's *reported* training peak was actually 9.46 GB, not 6.62 GB -- the
6.62 GB figure is from a clean, isolated synthetic reproduction (same
batch=1, 8-step grad accumulation, no resume-checkpoint loading involved)
run specifically to separate "genuine architectural cost" from "possible
resume-related overhead." The real run's higher number may include overhead
from loading a mid-training resume checkpoint (untested hypothesis) on top
of the same underlying vocabulary-driven cost. Both numbers exceed the
2-point fit's prediction substantially either way -- the qualitative
conclusion (the linear model needs a vocabulary term) holds regardless of
which Qwen number is used.

**Revised model** (not yet fitted with real data -- flagged for future
work, not resolved this session):

  V_peak ≈ 10P + c * V + A_checkpointed(L, B, S, H) + V_cuda_context

where `V` is vocabulary size and `c` is a small per-token-per-vocab-entry
constant capturing the logits/loss tensor's contribution. With only one
large-vocabulary data point (Qwen), `c` cannot be fitted yet -- this would
need at least one more large-vocabulary model in the matrix to separate the
`P` and `V` terms properly, since for OPT/Pythia the two are confounded
(their vocab sizes are nearly identical to each other).

### Prior 2-point fit (OPT-350M, Pythia-410M only -- superseded by the above for cross-model prediction)

Fitting `measured = a * P + b` on OPT-350M and Pythia-410M alone (P in
billions, V in GB):

  a = (4.257 - 3.554) / (0.4053 - 0.3312) = 0.703 / 0.0741 ≈ **9.49**
  b = 3.554 - 9.49 * 0.3312 ≈ **0.41 GB**

`a ≈ 9.49` lands almost exactly on the predicted `10` from the byte-counting
argument in §1 -- strong validation that the fixed-cost model (weights +
gradients + 8-bit optimizer state) is the right accounting for
similar-vocabulary architectures. `b ≈ 0.41 GB` is the activation +
CUDA-context overhead at this pipeline's fixed batch size (1), gradient
accumulation (8), and sequence length (512). This fit remains valid *within*
the OPT/Pythia-like (small-vocabulary) architecture family, but should NOT
be used to predict VRAM for a large-vocabulary model like Qwen2.5-0.5B or
any future frontier model -- see the vocabulary-size discussion above.

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
assumed.

**This prediction was tested against real Qwen2.5-0.5B data on Day 5 and did
NOT hold** (`docs/reports/2026-08-30_session-5.md` §5): P_max here predicts
Qwen (0.494B) should fit comfortably, but its actual measured training peak
was 6.62-9.46 GB, well over this machine's 6 GB budget -- see the
vocabulary-size discussion in §1. **`P_max` computed this way is only valid
for OPT/Pythia-like (small-vocabulary) architectures**; it does not
generalize across architecture families the way a true parameter-count-only
law would need to. This is itself a genuine, reportable finding for the
paper -- the "how big can a model be under N GB" question has at least two
independent variables (parameter count AND vocabulary size), not one, which
is a more interesting and more defensible claim than the original
single-variable law would have been.

A second, separate discovery from the same investigation: **eval-time peak
VRAM depends on eval batch size independently of training peak VRAM**, and
was the dominant factor in a real multi-hour slowdown incident on Qwen
(eval peaked at 11.47 GB with batch_size=4; dropping to batch_size=1 cut
this to 1.27 GB, a 9x reduction, with zero effect on the perplexity result
since it's a token-count-weighted global average and therefore exactly
batch-size invariant). Any VRAM budget analysis for this pipeline should
track training-peak and eval-peak as two separate quantities, not one --
the frontier experiment (Section B) should measure and report both.

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
