# Quantization Sensitivity Varies by Architecture: A Multi-Model Study of Selective Quantization-Aware Training Under a Consumer-GPU Memory Budget

**Status: INCOMPLETE DRAFT, first full pass — 2026-09-18.** This is the first
time the project's accumulated results and mathematical framing have been
assembled into paper form, immediately after Section A's 3-model QAT matrix
reached 36/36 completed training runs. It follows the structure of
`docs/PAPER_OUTLINE.md` (post-review-round-1 revision). Sections are written
in full where real data exists; sections with no data yet are marked
**[NOT YET RUN]** rather than filled with placeholder numbers. Treat every
number below as sourced directly from `results/*.json` in this repository
(commit `6cdf383`) unless stated otherwise — nothing here is invented or
extrapolated beyond what is explicitly labeled as a fit or estimate.

**What is done, going into this draft:** Section A (36/36 QAT training runs
across 3 models, all PTQ baselines, all controls). **What is not:** Section
B (memory-frontier/QLoRA), Section C (downstream lm-eval-harness), Section
D's significance testing (implemented, not yet executed on real data). This
draft exists to make that boundary concrete and to surface what the
completed data actually says, so the next-steps conversation has a real
document to react to rather than a status table.

---

## Abstract

We study selective quantization-aware training (QAT) — quantizing only
weights, only activations, or both — across three architecturally distinct
small language models (OPT-350M, Pythia-410M, Qwen2.5-0.5B; 331-494M
parameters) under a hard 6GB consumer-laptop-GPU memory budget, with a
properly isolated FP16 fine-tuned control, three random seeds per trained
configuration, and PTQ baselines (INT8 dynamic, SmoothQuant) for context.
We find that activation-quantization damage varies dramatically across
architecture families at matched scale: a mild +3.9% perplexity increase
on OPT-350M versus a severe +106.2% increase on Pythia-410M and +100.4% on
Qwen2.5-0.5B (all relative to each model's own control; absolute terms,
bits/token, tell the same story: +0.055 vs. +1.044 vs. +1.003). This
variation survives being reported in both relative and absolute terms and
is far larger than seed-to-seed noise for OPT and Pythia. Separately, we
show that a claimed "QAT-as-regularizer" effect (weight-only QAT beating an
unquantized control), visible in this project's original single-seed
predecessor study, does not survive proper 3-seed evaluation on OPT-350M
(control 21.02 vs. weights-only 21.09 — a marginal *increase*, not a
decrease). We also show that a memory-scaling relationship fit on two
architectures (`V_peak ≈ 9.49P + 0.41` GB) fails on a third: Qwen2.5-0.5B's
~3x larger vocabulary inflates measured peak training VRAM to 6.62-9.46 GB
against a 5.10 GB prediction, motivating a vocabulary-size term the current
2-point fit cannot estimate. A standard SmoothQuant implementation silently
fails (PPL 256.5 and 88.2 against expected ~29-30 and ~21-22) on 2 of 3
tested architectures, both non-pre-norm relative to the one architecture
where it works correctly — root cause not yet fully isolated. All
experiments ran on a single consumer laptop GPU; this shaped the study's
breadth-over-depth design rather than being a headline result. Sections
covering a memory-frontier QAT-vs-QLoRA comparison and downstream zero-shot
task evaluation are designed and implemented but **not yet run** — this is
stated plainly rather than described as complete.

---

## I. Introduction

Deploying language models under a fixed, small memory budget — a consumer
GPU, an edge device — increasingly means choosing not just *whether* to
quantize but *what* to quantize: weights, activations, or both, and at
what point in training. Selective quantization-aware training (QAT) lets a
practitioner make that choice, but the literature's evidence for how well
a given choice generalizes is thin in one specific way: studies (including
this project's own single-model predecessor) typically test one
architecture and, implicitly or explicitly, treat that architecture's
sensitivity profile as representative of "small language models" as a
class. This is a different question from *scale*-dependent quantization
difficulty, which is already well documented (Dettmers et al. 2022,
LLM.int8() — outlier-driven degradation grows with model size *within* an
architecture family). We ask instead whether the *same* selective-QAT
treatment, at matched, fixed, small scale, causes comparable damage
*across* architecture families — and find that it does not.

**Contributions**, split per the reviewer feedback that shaped this
revision into scientific claims and engineering artifacts, so the two are
not conflated:

**Scientific findings:**

1. Activation-quantization sensitivity varies substantially across
   architecture families at matched, fixed scale (331-494M parameters) —
   from +3.9% to +106.2% relative perplexity increase over a matched
   control — reported in both relative and absolute (bits/token) terms,
   with the absolute-terms comparison telling the same story as the
   relative one (Section V.E).
2. A methodological demonstration that a claimed "QAT-as-regularizer"
   effect, visible under single-seed evaluation in this project's own
   predecessor study, does not survive proper multi-seed, controlled
   re-evaluation (Section V.B) — a caution against trusting single-seed
   QAT claims generally.
3. A memory-scaling relationship for QAT VRAM requirements, fit on two
   architectures, shown to be *insufficient* (not merely extendable) when
   tested against a third — presented as a candidate relationship
   requiring a vocabulary-size term, with the real incident and root-cause
   analysis that revealed the gap (Section V.F).
4. A silent failure mode in a standard SmoothQuant implementation,
   root-caused on one architecture and observed to recur, unresolved, on a
   second — an explicitly open problem, not a solved one (Section V.A,
   Discussion).

**Engineering / reproducibility artifacts** (enabling infrastructure, not
headline claims):

5. A memory-optimized selective-QAT pipeline (gradient checkpointing,
   8-bit AdamW, gradient accumulation) with mid-training/eval checkpointing
   verified against a real, unplanned hard-kill scenario, making
   long-running, interruptible research tractable on a single
   non-dedicated laptop GPU (Section III.F, Appendix XI).
6. A fully open, reproducible pipeline: code, per-seed results, per-example
   evaluation data, and a complete session-by-session engineering log
   (`docs/reports/`, ten sessions as of this draft) documenting every bug
   found and fixed along the way, including the ones introduced by our own
   fixes.

---

## II. Related Work

**Post-training quantization (PTQ).** SmoothQuant (Xiao et al. 2023)
migrates activation outlier magnitude into weights via a per-channel
smoothing factor computed offline, enabling W8A8 quantization without
retraining; it and comparable methods (GPTQ, Frantar et al. 2023; AWQ, Lin
et al. 2024) are calibrated and validated predominantly on pre-norm,
separate-QKV transformer architectures. Section V.A reports a case where a
direct reimplementation of the SmoothQuant idea fails silently (produces a
plausible-looking but badly degraded number, not a crash) on two
architectures outside that common assumption set.

**Quantization-aware training and its memory cost.** QAT trains through a
fake-quantization forward/backward pass so the model adapts to quantization
noise, typically outperforming PTQ at the cost of requiring gradient
computation and optimizer state at training time. 8-bit optimizers
(Dettmers et al. 2021) make QAT's optimizer-state memory cost tractable on
constrained hardware — a precondition for this study's 6GB budget being
viable at all. QAT for transformer-scale models remains comparatively
understudied relative to CNN quantization, in part because of exactly this
memory cost.

**Mixed-precision and extreme quantization.** QLoRA (Dettmers et al. 2023a)
freezes a 4-bit-quantized base model and trains small full-precision LoRA
adapters, trading full-model adaptability for a dramatically smaller
trainable/optimizer footprint — the natural comparison point once a model
is too large for full QAT to fit a fixed memory budget (Section V.G,
designed, not yet run). SpQR (Dettmers et al. 2023b) and sub-2-bit methods
(Ma et al. 2024, 1.58-bit LLMs) represent further points on the same
precision-vs-capacity tradeoff curve, at a scale beyond this study's
hardware budget.

**Activation outliers and scale-dependence.** LLM.int8() (Dettmers et al.
2022) documents that activation outlier magnitude — and the quantization
difficulty it causes — grows with model scale *within* an architecture
family, motivating mixed-precision decomposition at scale. Wei et al.
(2022) characterize the outlier phenomenon that motivates SmoothQuant-style
smoothing. **This paper's contribution is explicitly narrower and
orthogonal to this line of work**: we hold scale roughly fixed
(331-494M parameters, a <50% spread) and vary architecture *family*
instead, showing the same nominal treatment (activation QAT) produces
wildly different damage without any change in scale. This is a different
axis of variation, not a re-demonstration of scale-dependent outlier
growth, and we are not aware of prior work that holds architecture family
as the controlled variable at fixed small scale with matched
seeds/controls/significance testing across the comparison.

---

## III. Methodology

**A. Pipeline architecture.** For each model: FP16 zero-shot baseline →
PTQ family (INT8 dynamic via `bitsandbytes`, SmoothQuant; AWQ attempted,
unavailable in this environment — `awq` package not installable, recorded
as `skipped` in results rather than silently omitted) → FP16 fine-tuned
control (3 seeds) → selective QAT (weights-only / activations-only / both;
3 seeds each).

**B. The domain-adaptation control, and why it exists.** This project's
single-seed predecessor study reported that weight-only QAT *beat* an
unquantized baseline on OPT-350M, framed as evidence of a
quantization-as-regularizer effect. That comparison conflated two distinct
effects: the model adapting to *quantization noise* versus the model simply
adapting to the *fine-tuning domain* (WikiText-2) that the original
unquantized baseline never saw. A matched FP16 fine-tuned control — same
data, same steps, same optimizer, no quantization — isolates the
quantization effect specifically. Under this proper control, the effect
does not clearly survive (Section V.B) — a confound a single-baseline QAT
study would not have caught, reported here as a methodological lesson
rather than a footnote.

**C. Memory optimization under a 6GB ceiling.** Gradient checkpointing
(trading ~20% compute time for reduced activation memory), 8-bit AdamW
(`bitsandbytes`), and micro-batching with gradient accumulation
(batch size 1, 8 accumulation steps, effective batch 8) together bring
peak training VRAM for a ~0.4B-parameter model under fake-quantization to
roughly 3.5-6.6 GB depending on architecture (Section V.F) — without these
three techniques together, QAT at this parameter scale would not fit this
study's hardware at all.

**D. Selective FakeQuant injection.** A wrapper module (`FakeQuantLinear`)
replaces `nn.Linear` layers, preserving the original weight `nn.Parameter`
and inserting `torch.ao.quantization` FakeQuantize submodules on the
weight path, the activation path, or both, per the selected strategy.
Weight quantization uses a per-channel symmetric INT8 scheme
(`MovingAveragePerChannelMinMaxObserver`); activation quantization uses a
per-tensor affine INT8 scheme (`MovingAverageMinMaxObserver`). Gradients
flow through the rounding operation via the standard Straight-Through
Estimator. Per-layer relative quantization error
(`‖fake_quant(x) − x‖ / ‖x‖`) is logged once per training step (batched to
avoid a per-microbatch blocking CUDA sync — see Appendix XI) for the
error-asymmetry analysis in Section V.F / Discussion. 168 Linear layers are
replaced per model (`lm_head`/`embed_out` excluded from quantization in all
three architectures).

**E. Statistical methodology — [PARTIALLY COMPLETE].** Three seeds per
trained configuration; mean ± standard deviation reported for every
configuration (Section V.C). A per-example paired Wilcoxon signed-rank
test on held-out NLL (`stats.py::wilcoxon_per_seed()`, backed by
`common.evaluate_perplexity(return_per_example=True)`) is implemented and
was validated in a dry run, but **has not yet been executed on the full
completed dataset** — it requires backfilling per-example NLL for OPT/
Pythia runs recorded before per-example logging existed
(`backfill_per_example_nll.py`, dry-run verified, not yet run at scale) and
a dedicated GPU-idle window. No p-values are reported in this draft; where
Section V describes an effect as "large" or "small," that is a
mean-and-standard-deviation comparison, not yet a statistically tested
claim, and this document says so explicitly rather than implying
significance testing has occurred. A multiple-comparisons correction
(e.g., Benjamini-Hochberg) is planned for whenever the test battery
actually runs, given the number of pairwise comparisons across 3 models ×
3 strategies.

**F. Reproducibility infrastructure.** Every result records exact
hyperparameters, seed, and git commit hash (`common.run_metadata()`).
Training is checkpointed every 10 steps (`config.CHECKPOINT_EVERY_STEPS`)
plus immediately on a caught interrupt signal, and evaluation is
checkpointed per batch — both verified against real, not simulated,
interruption scenarios over the course of this project, including a hard
process kill (not a graceful signal the interrupt handler could catch)
during the final Qwen QAT run: the periodic checkpoint recovered training
at step 250/500 against an actual last-completed step of 253, a loss of
three steps (~410s) rather than a restart from zero (full account in
Appendix XI).

---

## IV. Experimental Setup

**Models:**

| Model | Params (M) | Position encoding | Attention | QKV projection | Vocab size | Normalization |
|---|---|---|---|---|---|---|
| OPT-350M | 331.2 | Learned absolute | MHA | Separate Q/K/V | ~50,272 | Post-norm (non-standard for its own OPT family — see note below) |
| Pythia-410M | 405.3 | Learned absolute | MHA | **Fused QKV** | ~50,304 | Pre-norm |
| Qwen2.5-0.5B | 494.0 | RoPE | GQA | Separate Q/K/V | **151,936** | Pre-norm (RMSNorm) |

**Explicit confound statement**: these three models differ along at least
four architectural axes simultaneously (position encoding, QKV fusion,
vocabulary size, normalization placement/type). This study can establish
that cross-architecture variation in quantization sensitivity exists and
is large; it cannot, by itself, attribute that variation to any single
axis. Section VIII specifies the controlled ablation that would be needed
to do so.

**Known architectural note on OPT-350M**: OPT-350M is documented to depart
from its own model family's usual scaling conventions (a normalization
placement difference relative to other OPT checkpoints), so it should be
read as one data point rather than a clean "representative post-norm
architecture" — this strengthens, rather than weakens, the honesty of
treating this as an n=3 existence-claim comparison rather than a clean
2x2 factorial design.

**Dataset**: WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`), 10,000
pre-tokenized training samples, 512-token sequence length; a fixed
1,500-example held-out subset used for every seed-level evaluation in the
matrix (`n_tokens` = 209,921 per eval run, identical across all Qwen
configs, confirming a consistent eval set). **[NOT YET DONE]**: the
outline calls for validating this 1,500-example subset against the full
10,000-example set on a sample of runs to confirm the subset is
representative; this validation has not been run and is not assumed —
flagged here rather than silently dropped.

**Hardware**: a single NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB
(~6.0 GB) total VRAM, Windows 11. This ceiling is load-bearing for the
memory-scaling analysis (Section V.F) and is treated as a first-class
experimental parameter, not an incidental detail.

---

## V. Results

### A. PTQ baselines, all 3 models

| Model | FP16 zero-shot | INT8 dynamic | SmoothQuant (own impl.) | AWQ |
|---|---|---|---|---|
| OPT-350M | 41.28 | 41.39 | 43.67 | not available (dependency missing) |
| Pythia-410M | 29.54 | 29.72 | **256.50** (~8.7x expected) | not available |
| Qwen2.5-0.5B | 21.61 | 21.84 | **88.17** (~4x expected) | not available |

SmoothQuant behaves correctly on OPT-350M (43.67, a modest, expected
degradation over the 41.28 zero-shot baseline) but fails badly on both
Pythia-410M and Qwen2.5-0.5B — the only architecture where it works
correctly is the one with an atypical normalization placement (Section IV
note), and the two where it fails are otherwise different from each other
in every other respect (fused vs. separate QKV, learned vs. RoPE position
encoding, ~50K vs. ~152K vocabulary). The common thread has not been
isolated; see Discussion and Limitations. This is reported as an open,
unresolved problem, not attributed to a specific cause.

### B. Domain-adaptation isolation: the regularizer effect does not survive proper controls

| | OPT-350M (this study, 3-seed mean) |
|---|---|
| FP16 fine-tuned control | **21.02** |
| QAT weights-only | **21.09** (+0.3%) |

This project's single-seed predecessor study reported weight-only QAT
*beating* an FP16 baseline (interpreted as a regularization effect). Under
a properly matched 3-seed control (Section III.B), the effect reverses in
direction: weights-only QAT is marginally *worse* than the control, not
better, though the gap (0.3%) is small enough that it may not survive
formal significance testing once Section D's test battery is actually run
— this draft does not claim the gap is statistically meaningful, only that
the direction of the original single-seed claim does not hold up.

### C. Full selective-QAT matrix, all 3 models

Perplexity, mean ± std across 3 seeds (control and all three QAT
strategies):

| Model | Control | Weights-only | Activations-only | Both |
|---|---|---|---|---|
| OPT-350M | 21.02 ± 0.02 | 21.09 ± 0.05 | 21.84 ± 0.07 | 21.92 ± 0.07 |
| Pythia-410M | 16.75 ± 0.09 | 16.82 ± 0.14 | 34.54 ± 0.88 | 34.58 ± 0.80 |
| Qwen2.5-0.5B | 15.41 ± 2.25 | 21.94 ± 6.04 | 30.90 ± 1.88 | 31.64 ± 1.99 |

**Qwen2.5-0.5B's variance is unusual and is called out explicitly rather
than averaged over.** Per-seed detail:

| Configuration | seed=42 | seed=1337 | seed=2024 |
|---|---|---|---|
| Control | 18.01 | 14.02 | 14.21 |
| Weights-only | 18.69 | **28.91** | 18.23 |
| Activations-only | 28.74 | 31.80 | 32.15 |
| Both | 31.81 | 29.58 | 33.55 |

Two things stand out. First, Qwen's control itself has much higher
seed-to-seed spread (std 2.25) than OPT's (0.02) or Pythia's (0.09) —
seed=42's control run (18.01) is an outlier relative to the other two
(14.02, 14.21), for reasons not yet investigated. Second, and more
consequentially, weights-only QAT's seed=1337 run (28.91) is dramatically
higher than its two sibling seeds (18.69, 18.23) — this exact discrepancy
was flagged as an open, unexplained question on Day 7 of this project
(`docs/PLAN.md`) and **remains unresolved as of this draft**. It single-
handedly inflates Qwen's weights-only mean to 21.94 (nominally a +42.3%
hit vs. control) in a way that one outlier seed, not a uniform
architectural effect, is driving. Reporting Qwen's weights-only result as
a clean "+42.3%" architecture effect without this caveat would misrepresent
what the data shows — the honest reading is "two of three seeds land close
to the control (18.69, 18.23 vs. 15.41-mean control, itself skewed by its
own seed=42 outlier); one seed is a clear outlier and the cause is
unidentified." Activations-only and both, by contrast, show tight,
consistent 3-seed clustering on Qwen (std 1.88 and 1.99, comparable to
Pythia's), suggesting whatever is driving the weights-only and control
variance is specific to those two configurations, not a general Qwen
instability.

### D. Significance testing — **[NOT YET RUN]**

See Methodology III.E. No p-values are reported in this draft.

### E. The headline cross-architecture comparison, relative and absolute terms

Activation-involving strategies only (weights-only excluded here given the
Qwen variance caveat in V.C; both models' + Qwen's weights-only clean-seed
subset (18.69/18.23) versus control is consistent with OPT/Pythia's small
weights-only effect, for what it's worth as a secondary reading):

| Model | Strategy | % change vs. control | Bits/token (abs.) | Δ bits/token vs. control |
|---|---|---|---|---|
| OPT-350M | Activations-only | **+3.9%** | 4.449 | +0.055 |
| OPT-350M | Both | +4.3% | 4.454 | +0.060 |
| Pythia-410M | Activations-only | **+106.2%** | 5.110 | +1.044 |
| Pythia-410M | Both | +106.5% | 5.112 | +1.046 |
| Qwen2.5-0.5B | Activations-only | **+100.4%** | 4.949 | +1.003 |
| Qwen2.5-0.5B | Both | +105.3% | 4.984 | +1.038 |

The finding is unchanged whether read in relative (%) or absolute
(bits/token) terms, which was a specific concern raised in review (that a
percentage framing alone can be misleading when baseline perplexities
differ across models): OPT-350M's activation-QAT cost is roughly
**19x smaller in absolute bits/token** than Pythia's or Qwen's
(0.055-0.060 vs. ~1.0-1.05 bits/token), not just smaller by a raw
percentage that could be an artifact of a different baseline scale.
Pythia-410M and Qwen2.5-0.5B land within ~4% of each other in relative
terms and within ~0.04 bits/token of each other in absolute terms, despite
having almost nothing else architecturally in common (fused vs. separate
QKV, learned-absolute vs. RoPE position encoding, ~50K vs. ~152K
vocabulary) — this is itself worth noting as a coincidence, or a hint
that whatever mechanism causes the damage is present in both despite their
other differences (Discussion).

### F. Memory-scaling relationship

Peak training VRAM (mean across each model's 4 completed QAT
configurations):

| Model | P (M) | Vocab size | 10P prediction | Measured peak | Residual |
|---|---|---|---|---|---|
| OPT-350M | 331.2 | ~50,272 | 3.31 GB | 3.554 GB | 0.244 GB |
| Pythia-410M | 405.3 | ~50,304 | 4.05 GB | 4.257 GB | 0.207 GB |
| Qwen2.5-0.5B | 494.0 | **151,936** | 4.94 GB | **6.62-9.46 GB** | **1.68-4.52 GB** |

A 2-point linear fit on OPT/Pythia alone (`V_peak = 9.49·P + 0.41` GB,
P in billions) predicts Qwen's peak at 5.10 GB — a substantial
underestimate against its measured 6.62 GB (clean isolated repro) to
9.46 GB (real production run, possibly including resume-checkpoint
overhead, untested). The params-only linear model does not hold across
architecture families; the residual is 7-8x larger for Qwen relative to
the same fixed-cost prediction that fits OPT/Pythia well. The leading
explanation (direct profiling, not assumption): Qwen's ~3x larger
vocabulary inflates the final logits/loss tensor
(shape `[batch, seq_len, vocab_size]`) disproportionately to parameter
count — a term the `10P` weights+gradients+optimizer-state accounting does
not capture. A revised model,
`V_peak ≈ 10P + c·V + A_checkpointed(L,B,S,H) + V_cuda_context`, is
proposed but **not yet fitted** — one large-vocabulary data point cannot
separate the `P` and `V` terms; this needs at least one more
large-vocabulary model in the matrix (Section VIII).

A separate, independent finding from the same investigation: eval-time
peak VRAM is governed by eval batch size independently of training peak
VRAM (11.47 GB at eval batch size 4 vs. 1.27 GB at batch size 1 on Qwen — a
9x reduction with zero effect on the perplexity result, since it is a
token-count-weighted average and therefore exactly batch-size invariant).
Any VRAM budget analysis for this pipeline must track training-peak and
eval-peak as two separate quantities.

### F-2. A new wrinkle: per-layer quantization error ratio does not predict downstream damage monotonically

Extending the weight/activation relative-quantization-error analysis
(Section III.D) to all three models, now that Qwen's data exists:

| Model | Mean weight rel. error | Mean activation rel. error | Ratio (act/weight) | Downstream activations-only PPL damage |
|---|---|---|---|---|
| OPT-350M | 0.79% | 3.96% | **5.02x** | +3.9% |
| Pythia-410M | 0.86% | 4.93% | **5.74x** | +106.2% |
| Qwen2.5-0.5B | 1.26% | 2.94% | **2.33x** | +100.4% |

This is a genuinely new result as of this draft (Qwen's error data did not
exist when `docs/MATH.md` §3 was last written) and it complicates the
existing "compounding error across depth" hypothesis rather than confirming
it cleanly: Qwen has the **lowest** activation/weight error ratio of the
three architectures, yet the **second-highest** downstream perplexity
damage, comparable to Pythia's. If a higher per-layer relative error ratio
alone predicted more downstream damage, Qwen (2.33x) should look more like
OPT (5.02x → +3.9%) than like Pythia (5.74x → +106.2%); instead it looks
like Pythia despite having the *smallest* error ratio of the three. This
suggests per-layer relative error magnitude is not sufficient on its own to
predict downstream damage — architectural propagation structure (GQA's
shared KV heads, RoPE's position-dependent rotation applied after
quantization, or some interaction not yet identified) plausibly matters as
much as or more than raw per-layer error size. This is flagged as an open
question, not resolved here (Section VIII).

### G. Memory-frontier study (QAT vs. QLoRA) — **[NOT YET RUN]**

Designed (Section II of `docs/PLAN.md`) but not started. The original
params-only crossover estimate (P_max ≈ 0.536-0.589B under this machine's
budget) is now known to be invalid for large-vocabulary architectures
(Section V.F) and would need re-deriving with a vocabulary-aware model, or
explicit acknowledgment that the frontier model choice is not yet
justified by a validated formula.

### H. Downstream zero-shot evaluation (LAMBADA/PIQA/HellaSwag) — **[NOT YET RUN]**

Pipeline built, smoke-tested only (per `docs/PLAN.md` Day 4 note). No
results exist.

---

## VI. Discussion

The central result — activation-quantization damage ranging from +3.9% to
+106.2% across three architectures at matched scale — should be read
strictly as an **existence claim**: cross-architecture variation in
quantization sensitivity is real, large, and (for the OPT vs. Pythia/Qwen
comparison specifically) far too large to plausibly be seed noise, even
before Section D's formal test runs. It should **not** be read as having
identified *which* architectural feature causes it. Section V.F-2's new
finding sharpens rather than resolves the mechanism question: the
fused-QKV/error-concentration hypothesis proposed earlier in this project
(a fused QKV projection feeding three downstream uses from one quantized
tensor) remains a plausible explanation for *Pythia* specifically, but
Qwen shows comparable damage via a different architecture (separate QKV,
GQA, RoPE) and, if anything, a *smaller* per-layer error signal — so either
more than one mechanism is at work, or the true mechanism is something
Pythia and Qwen share that OPT does not (both are pre-norm; OPT is not —
Section IV's architectural note). This normalization-placement coincidence
is noted here as a candidate alternative hypothesis, generated by this
draft's own analysis, not previously considered in this project's docs —
it is exactly as unconfirmed as the fused-QKV hypothesis and should be
weighed the same way pending the controlled ablation in Section VIII.

The regularizer-effect fragility (Section V.B) is offered as a
field-general caution: a real effect reported under single-seed evaluation
reversed direction under proper controls and 3 seeds, on the very same
model and configuration. This is a methodological point independent of
this paper's architecture-comparison thesis, and is reported because it
happened during this project's own work, not because it is being singled
out as unusual for the field.

The memory-scaling finding (Section V.F) is a practical warning:
parameter count alone is an unreliable proxy for QAT memory cost once
vocabulary size varies substantially, and the field's rule-of-thumb
formulas (this project's own included) should be checked against
vocabulary size before being applied to a large-vocabulary model. Three
data points, one of them large-vocabulary, cannot yet support a validated
two-variable law — this is stated plainly rather than the fit being
presented as more settled than it is.

The SmoothQuant tooling failure (Section V.A) is evidence that a
standard PTQ technique's implicit architectural assumptions do not hold
universally, and that a run completing without an error or crash is not
evidence of correctness — both failures here produced plausible-looking
numbers, not obvious garbage, and were only caught by comparing against an
expected range from the zero-shot/INT8-dynamic baselines. This recurrence
on 2 of 3 tested architectures, still unresolved, argues for treating any
single-architecture PTQ validation in the literature with some caution
when applying it elsewhere.

Any mixed-precision or selective-QAT recommendation drawn from this data
("weight-only quantization is close to free; activation quantization can
be catastrophic") should be read with the architecture-dependence finding
as an explicit caveat — the *direction* of the finding (activation
quantization is riskier than weight quantization) is consistent across
all three architectures tested, but the *magnitude* is not, and this
paper cannot yet say which architectural property predicts the magnitude
for an architecture not yet tested.

---

## VII. Limitations

- **The core confound, stated directly**: three models differing on 4+
  architectural axes simultaneously (position encoding, QKV fusion,
  vocabulary size, normalization placement/type) means this study
  demonstrates that architecture-dependent variation exists, not which
  specific feature causes it.
- Model scale capped at ~500M parameters by the 6GB hardware budget —
  quantified via the (candidate, not validated) memory relation rather
  than left as an unexplained ceiling.
- The SmoothQuant failure on 2 of 3 architectures is documented but not
  fully root-caused — an explicitly open problem.
- The memory-scaling relationship rests on 3 data points (one
  large-vocabulary) and has not been validated out-of-sample.
- Single dataset (WikiText-2); cross-domain generalization untested.
- The VRAM-oversubscription slowdown behavior documented in this project's
  engineering log is specific to this Windows/consumer-GPU setup and may
  not reproduce identically elsewhere.
- **Qwen2.5-0.5B's control and weights-only results show substantially
  higher seed-to-seed variance than the other two models, for reasons not
  yet investigated** (Section V.C) — any single-seed reading of Qwen's
  weights-only result specifically should be treated with more caution
  than the equivalent OPT/Pythia numbers.
- Section B (memory-frontier/QLoRA), Section C (downstream lm-eval-harness
  tasks), and Section D (formal significance testing) are designed and,
  for B and D, partially implemented, but contain no results as of this
  draft.

---

## VIII. Future Work

- **Highest priority**: a controlled ablation isolating exactly one
  architectural axis — e.g., a matched pair of models identical in
  parameter count, vocabulary, and training data, differing only in fused-
  vs-separate QKV projection, or only in normalization placement/type
  (motivated by the new normalization-placement coincidence noted in
  Section VI) — to convert the architecture-dependence finding from an
  existence claim into a mechanistic one.
- Investigate the unresolved Qwen weights-only seed=1337 outlier
  (Section V.C) directly — rerun with additional seeds or instrument the
  run for a divergence signature, rather than leaving it as an
  unexplained data point.
- Investigate the non-monotonic relationship between per-layer
  weight/activation error ratio and downstream damage (Section V.F-2) —
  possibly via a targeted perturbation experiment on GQA/RoPE-specific
  propagation, analogous to the fused-QKV perturbation experiment already
  proposed for Pythia.
- Out-of-sample validation of the memory-scaling relationship on a fourth,
  independently chosen large-vocabulary architecture not used to fit or
  patch the equation.
- Root-cause the SmoothQuant failure recurrence beyond the two
  architectures where it is currently only documented, not explained.
- Run Section D's implemented significance-testing pipeline on the now-
  complete Section A dataset (backfill older per-example NLL, apply a
  multiple-comparisons correction).
- Complete Section B (memory-frontier) and Section C (downstream
  evaluation), and extend this draft's confound-isolation and
  variance-reporting discipline to both once they produce data.

---

## IX. Conclusion

Under matched, rigorous methodology — a proper isolated control, three
seeds per configuration, PTQ baselines for context, and both relative and
absolute effect-size reporting — the same selective-QAT treatment
(activation quantization) causes wildly different damage across three
small language models at fixed scale: from a mild +3.9% perplexity
increase to a severe +106.2% one. This variation is an existence claim,
not a mechanistic one: this study cannot yet say which of the several
architectural differences between these models is responsible. A
methodological caution accompanies the headline result: a real regularizer
effect reported under single-seed evaluation in this project's own earlier
work did not survive proper controlled re-evaluation. A memory-scaling
relationship fit on two architectures required revision, not just
extension, once tested against a third with a substantially larger
vocabulary — parameter count alone is an unreliable memory proxy across
architecture families. Every experiment in this draft ran on a single
consumer laptop GPU under a 6GB budget; this shaped a breadth-over-depth
study design deliberately, and is reported as a methodological detail
rather than the paper's selling point. Sections covering a memory-frontier
QAT-vs-QLoRA comparison and downstream zero-shot evaluation remain
unstarted as of this draft and are named as such rather than implied
complete.

---

## X. Broader Impact / Societal Considerations

Better-understood, more predictable quantization lowers the barrier to
running language models on-device and at the edge, which has genuine
dual-use texture: it can improve privacy (less data leaving a device to a
server) but also removes server-side moderation and safety layers that
many deployments rely on, and can lower the cost of deploying models for
harmful purposes as easily as beneficial ones. This study does not
evaluate bias, privacy, or safety properties of any quantized model
directly and makes no claim about them; its scope is limited to
perplexity-based language-modeling quality and memory cost.

---

## XI. Reproducibility Appendix

**Full hyperparameters** (all runs unless noted): learning rate 1e-5,
batch size 1, gradient accumulation 8 steps (effective batch 8), sequence
length 512, 500 training steps, optimizer `bitsandbytes.optim.AdamW8bit`,
gradient checkpointing enabled, `GRAD_CLIP_NORM=1.0` (added Day 8 of this
project after a real gradient-explosion divergence, described below; not
present in the original single-seed predecessor study). Weight
quantization: per-channel symmetric INT8,
`MovingAveragePerChannelMinMaxObserver`. Activation quantization:
per-tensor affine INT8, `MovingAverageMinMaxObserver`. Eval batch size 1
(reduced from 4 after the Day 5 VRAM incident below; perplexity is exactly
batch-size invariant, verified).

**Mid-training/eval checkpointing, verified against real interruptions.**
Training checkpoints every 10 steps plus immediately on a caught interrupt
signal; eval checkpoints per batch. This was exercised for real, not
simulated, multiple times over the project, most concretely: the final
Qwen `both`/seed=2024 run was deliberately stopped via a hard process kill
(not a graceful signal the training loop's interrupt handler could catch)
at what turned out to be step 253/500; the periodic checkpoint recovered
training at step 250/500 on the next launch, a loss of 3 steps (~410
seconds) rather than a restart from step 0 — confirmed by loading the
checkpoint file directly and reading back its recorded step number rather
than assumed.

**The VRAM-oversubscription incident** (`docs/reports/2026-08-30_session-5.md`,
`docs/MATH.md` §1-2). A multi-hour Qwen2.5-0.5B slowdown was traced to
measured peak VRAM (6.62-9.46 GB training, 11.47 GB eval at batch size 4)
exceeding this machine's 6.14 GB card, causing Windows to silently fall
back to system RAM — high GPU utilization, no error, no crash, just
glacial progress indistinguishable from a stuck process without directly
measuring VRAM. Diagnosis method: direct VRAM measurement, not inference
from symptoms. Fix: `EVAL_BATCH_SIZE=1`, cutting eval peak VRAM to 1.27 GB
(9x reduction) with zero effect on the perplexity result.

**Known bugs found and fixed during this project** (full detail in
`docs/reports/`, one entry per session): a per-layer quantization-error
logging path calling a blocking CUDA sync 1,344 times per training step,
turning one step into ~54 minutes (Day 7); a resume-checkpoint load
leaving a redundant ~2.78GB copy in VRAM (Day 7); `evaluate_perplexity()`
never using `torch.no_grad()`, building an unused autograd graph on every
eval batch for the project's entire life until caught (Day 6); an
unclipped-gradient-explosion divergence on activation-QAT, traced
step-by-step from loss 661 to ~3×10^18 before overflowing to NaN, fixed
with standard gradient clipping (Day 8) and confirmed working in
production, including a self-caught reporting error where clip frequency
was initially misreported as declining when it was in fact a constant
100% (Day 9).

**Code and data availability**: this repository
(`https://github.com/FUNGI432/Quantisationresearch`, commit `6cdf383` as
of this draft) contains all training/eval code, every result as
`results/<model>.json` (full hyperparameters + git commit hash per run),
per-example NLL arrays for paired significance testing
(`results/per_example/`), and a complete session-by-session engineering
log (`docs/reports/2026-08-23_session-1.md` through
`docs/reports/2026-09-17_session-10.md` as of this draft) documenting
every bug found and fixed, including mistakes in this project's own
reporting, in the order they actually happened.

---

## References

Gholami et al. 2022 (quantization survey); Xiao et al. 2023 (SmoothQuant);
Dettmers et al. 2021 (8-bit optimizers); Dettmers et al. 2022 (LLM.int8());
Frantar et al. 2023 (GPTQ); Lin et al. 2024 (AWQ); Dettmers et al. 2023a
(QLoRA); Dettmers et al. 2023b (SpQR); Ma et al. 2024 (1.58-bit LLMs); Wei
et al. 2022 (outlier suppression); OPT model card/appendix (supporting the
OPT-350M architectural note in Section IV). **[TODO before submission]**:
full bibliographic entries (venue, year, exact title) need to be verified
and formatted — this list currently records the citation keys used
throughout this project's docs, not a complete reference list.
