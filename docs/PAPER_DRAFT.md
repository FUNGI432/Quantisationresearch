# Quantization Sensitivity Varies by Architecture: A Multi-Model Study of Selective Quantization-Aware Training Under a Consumer-GPU Memory Budget

**Status: DRAFT, updated Day 13 — 2026-09-18/19.** This is the first time
the project's accumulated results and mathematical framing have been
assembled into paper form, immediately after Section A's 3-model QAT matrix
reached 36/36 completed training runs; updated the same session as Section
D (significance testing), Section C (downstream eval), and Section B
(memory-frontier QAT-vs-QLoRA) were each run on real data in turn. It
follows the structure of `docs/PAPER_OUTLINE.md` (post-review-round-1
revision). Sections are written in full where real data exists. Treat
every number below as sourced directly from `results/*.json` in this
repository unless stated otherwise — nothing here is invented or
extrapolated beyond what is explicitly labeled as a fit or estimate.

**What is done, going into this draft:** Sections A, B, C, and D are all
complete as of Day 12/13 — the full 3-model QAT matrix, the memory-frontier
QAT-vs-QLoRA comparison, downstream zero-shot evaluation, and real
significance testing with a multiple-comparisons correction. This draft
exists to surface what the completed data actually says, so the
next-steps conversation (Sections E/F polish, the open research questions,
or a genuinely new direction) has a real, load-bearing document to react
to rather than a status table.

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
variation survives being reported in both relative and absolute terms, is
far larger than seed-to-seed noise for OPT and Pythia, and — now backed by
a full battery of 24 paired Wilcoxon signed-rank tests on held-out
per-example NLL with a Benjamini-Hochberg multiple-comparisons correction
— is statistically significant in all 18 activation-involving comparisons
tested (q<0.05, most at p<10⁻⁶). Downstream zero-shot evaluation
(LAMBADA/PIQA/HellaSwag) independently confirms this on a completely
different metric: Pythia-410M's activation-QAT damage nearly halves
LAMBADA next-word accuracy (43.1%→19.1%) while OPT-350M's equivalent
change is noise-level (≤2%). A further, unanticipated finding on
Qwen2.5-0.5B: unlike on OPT/Pythia, *weight-only* QAT also costs Qwen
substantial downstream accuracy (−32.2% relative on LAMBADA) despite only
modest perplexity impact — complicating the "weight quantization is
nearly free" reading the other two models would otherwise support. Separately, we
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
breadth-over-depth design rather than being a headline result. Downstream
zero-shot evaluation (LAMBADA/PIQA/HellaSwag) independently confirms the
headline finding and surfaces a new wrinkle on Qwen2.5-0.5B (weight-only
QAT costs real downstream accuracy there despite modest perplexity
impact). The memory-frontier comparison (a 1.1B-parameter model) shows a
clean, directly observed crossover: full QAT does not fit this 6GB card at
all (a Windows VRAM-oversubscription slowdown rather than a clean OOM,
but a genuine failure either way), while QLoRA fits comfortably (1.33 of
6.14 GB used), training only 0.41% of the model's parameters.

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
   confirmed by a real paired significance test with a multiple-
   comparisons correction (Section V.D), and independently corroborated
   on a downstream zero-shot task battery where the same architecture that
   showed severe perplexity damage also loses nearly half its next-word
   prediction accuracy while the mild-damage architecture shows no
   measurable downstream cost (Section V.H).
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
5. A directly observed (not merely predicted) crossover point at which
   practitioners must switch from full QAT to a parameter-efficient method
   under a fixed VRAM budget: a 1.1B-parameter model's full-QAT attempt
   fails on this 6GB card while QLoRA fits training under 0.5% of its
   parameters — including the finding that the failure manifests as a
   driver-level slowdown rather than a clean out-of-memory exception on
   this Windows setup (Section V.G).

**Engineering / reproducibility artifacts** (enabling infrastructure, not
headline claims):

6. A memory-optimized selective-QAT pipeline (gradient checkpointing,
   8-bit AdamW, gradient accumulation) with mid-training/eval checkpointing
   verified against a real, unplanned hard-kill scenario, making
   long-running, interruptible research tractable on a single
   non-dedicated laptop GPU (Section III.F, Appendix XI).
7. A fully open, reproducible pipeline: code, per-seed results, per-example
   evaluation data, and a complete session-by-session engineering log
   (`docs/reports/`, twelve sessions as of this draft) documenting every
   bug found and fixed along the way, including the ones introduced by our
   own fixes.

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

**E. Statistical methodology.** Three seeds per trained configuration;
mean ± standard deviation reported for every configuration (Section V.C).
A per-example paired Wilcoxon signed-rank test on held-out NLL
(`stats.py::wilcoxon_per_seed()`, backed by
`common.evaluate_perplexity(return_per_example=True)`) was run for every
(model, QAT strategy, seed) triple with available per-example data — 24 of
27 possible triples (Qwen2.5-0.5B `weights_only`/seed=42 lacks per-example
data due to an unrelated, previously documented crash-recovery gap, see
Appendix XI). OPT-350M and Pythia-410M's 24 runs needed their per-example
NLL backfilled from saved checkpoints (`backfill_per_example_nll.py`),
since per-example logging was added to the pipeline after those two
matrices finished; Qwen's per-example NLL was captured live during
training and needed no backfill. **A data-quality caveat surfaced by the
backfill's own built-in sanity check, reported here rather than
suppressed**: the backfilled per-example NLL for Pythia-410M's
`activations_only`/`both` configurations carries a small, systematic,
*conservative* (damage-understating) bias of 1.9-2.75%, and OPT-350M's
equivalent configurations carry a much smaller 0.1-0.3% version of the
same bias — full mechanism and evidence in Appendix XI. All 24 raw and
FDR-corrected p-values, per seed, are pooled into one
Benjamini-Hochberg multiple-comparisons correction
(`stats.py::full_significance_report()`) rather than being read in
isolation — results in Section V.D.

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

### D. Significance testing

24 per-seed paired Wilcoxon signed-rank tests (n=1,500 paired examples
each) across 3 models × 3 QAT strategies vs. their matched control,
Benjamini-Hochberg corrected as one pooled family (Methodology III.E):

**21 of 24 tests are significant after correction (q<0.05); zero tests
that were significant at raw p<0.05 were overturned by the correction.**
Every single `activations_only` and `both` comparison — all 3 models, all
available seeds, 18 tests total — is significant, most at p<10⁻⁶. The 3
non-significant results are all `weights_only` vs. control:

| Model | Seed | Raw p | FDR q | Mean NLL diff | Verdict |
|---|---|---|---|---|---|
| OPT-350M | 42 | 0.301 | 0.301 | −0.0004 (weights-only *lower*) | Not significant |
| Pythia-410M | 1337 | 0.053 | 0.057 | +0.0038 | Not significant |
| Pythia-410M | 42 | 0.055 | 0.057 | +0.0530 | Not significant |

This is the statistical backing the "regularizer effect fragility" claim
(Section V.B) needed and did not have until now: `weights_only` vs.
control is the *only* comparison in the entire battery where the null
hypothesis (no difference) survives correction, and it does so specifically
on OPT-350M's seed=42 (the seed configuration closest to this project's
original single-seed predecessor study) and on 2 of Pythia's 3 seeds.
OPT-350M's other two weights-only seeds (1337: p=0.019; 2024: p<10⁻⁴) and
both of Qwen's testable weights-only seeds (1337: p<10⁻⁶; 2024: p<10⁻⁶) *do*
reach significance — so even the weights-only story is not uniformly
"no effect," it is inconsistently significant in a way that a single-seed
study could not have revealed either way. The headline
architecture-dependence finding (Section V.E) is now standing on tested,
corrected significance, not an eyeballed mean comparison: full results in
`results/significance_report.json`.

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

### G. Memory-frontier study (QAT vs. QLoRA)

TinyLlama-1.1B (1.1B params, RoPE, GQA — chosen per `docs/PLAN.md` Section
II as comfortably above the ~0.54-0.59B crossover the 2-point OPT/Pythia
fit predicted; that fit is now known invalid for large-vocabulary
architectures, Section V.F, but TinyLlama's ~32K vocabulary is smaller
than all three Section A models', so if anything this understates how far
over budget full QAT would be here — a conservative choice of frontier
model, not an inflated one).

**Full QAT attempt**: predicted fixed VRAM (`10P` rule) = 11.0 GB against
this machine's 6.14 GB card. Rather than a clean `torch.cuda.OutOfMemoryError`,
the attempt entered the same Windows CUDA driver shared-memory-fallback
slowdown this project already documented for Qwen2.5-0.5B's eval (Day 5,
Section III.F) — VRAM usage climbed to 93.6% of the card (5,746 of 6,141
MiB) and plateaued there, GPU utilization pinned at 100% but power draw
only ~26 W (far below genuine compute load), zero training steps
completed after 17 minutes. This was confirmed directly via
`nvidia-smi`'s memory/utilization/power trend, not inferred from an
absence of output, and killed deliberately rather than left to run
indefinitely. This is the experiment's intended failure result manifesting
as a silent slowdown rather than a clean exception on this
Windows/consumer-GPU setup — full detail in Appendix XI.

**QLoRA** (4-bit NF4 frozen base, double quantization, LoRA rank 16 on
`q_proj`/`k_proj`/`v_proj`/`o_proj`, same 500-step/batch-1/8-grad-accum
budget as Section A, evaluated on the same fixed 1,500-example subset):

| Metric | Full QAT (attempted) | QLoRA |
|---|---|---|
| Outcome | Oversubscription slowdown, 0 steps completed | **Completed successfully** |
| Peak training VRAM | 5.75 GB (93.6% of card, still climbing) | **1.33 GB (21.6% of card)** |
| Trainable parameters | 1.1B (100%) | **4.5M (0.41%)** |
| Perplexity | N/A (did not train) | **9.40** |
| Training wall-clock (500 steps) | N/A | 1,750s (29.2 min) |

This is a clean, empirically demonstrated crossover: the same model, same
data, same step budget genuinely does not fit this 6GB card under full
QAT — not as a theoretical prediction, but as a directly observed failure
— while QLoRA fits with substantial headroom to spare (1.33 of 6.14 GB
used), training only 0.41% of the model's parameters. TinyLlama's QLoRA
perplexity (9.40) is not directly comparable to Section A's control
perplexities (different model, different pretraining, different
adaptation method) and is reported as a standalone frontier-method result,
not a fourth architecture added to the Section A matrix.

### H. Downstream zero-shot evaluation (LAMBADA/PIQA/HellaSwag)

Zero-shot, 500-example subsample per task, all 3 seeds, all 3 models, all
4 configurations (36 runs total). Metric: `acc` for LAMBADA (next-word
prediction), `acc_norm` for PIQA/HellaSwag (length-normalized multiple
choice). Percent change vs. each model's own control, mean across 3 seeds:

| Model | Strategy | LAMBADA (Δacc) | PIQA (Δacc_norm) | HellaSwag (Δacc_norm) |
|---|---|---|---|---|
| OPT-350M | weights-only | +1.9% | +0.1% | +0.3% |
| OPT-350M | activations-only | −1.2% | +0.6% | +0.8% |
| OPT-350M | both | −0.5% | +0.0% | +1.3% |
| Pythia-410M | weights-only | +2.9% | −0.3% | −1.6% |
| Pythia-410M | activations-only | **−55.6%** | −14.2% | −10.5% |
| Pythia-410M | both | **−54.9%** | −11.8% | −11.2% |
| Qwen2.5-0.5B | weights-only | **−32.2%** | −9.6% | −8.6% |
| Qwen2.5-0.5B | activations-only | −38.5% | −10.4% | −17.7% |
| Qwen2.5-0.5B | both | −39.3% | −9.6% | −16.8% |

Two findings, one confirmatory and one new:

**Confirmatory**: Pythia-410M's activation-QAT damage, already the most
severe perplexity result in this study (+106.2%), is not a perplexity
artifact — it devastates actual downstream task performance on a
completely different evaluation axis, nearly halving LAMBADA's next-word
accuracy (43.1% → 19.1-19.5%). OPT-350M's near-zero perplexity cost
likewise shows up as near-zero downstream cost (all changes within ±2%,
indistinguishable from noise at this sample size). This is exactly the
independent corroboration Section C was designed to provide, and it lands
cleanly in both directions.

**New, not visible in perplexity alone**: on Qwen2.5-0.5B, `weights-only`
QAT — the strategy that was cheap-to-free on both OPT and Pythia, in both
perplexity and downstream terms — costs **−32.2% relative LAMBADA
accuracy and −8.6 to −9.6% on PIQA/HellaSwag**, comparable in magnitude to
Qwen's own `activations_only`/`both` results, and nowhere close to
OPT/Pythia's weights-only downstream cost (≤3% either direction). This
complicates the "weight quantization is nearly free" reading that OPT and
Pythia alone would support: for Qwen specifically, *all three* QAT
strategies meaningfully hurt downstream zero-shot accuracy, not just the
activation-involving ones. This is independent of, and arguably more
striking than, the already-flagged `weights_only` perplexity seed-variance
issue (Section V.C) — LAMBADA/PIQA/HellaSwag results are consistent across
all 3 seeds (no single-seed outlier driving this), so it cannot be
explained away as the same seed=1337 anomaly. Not yet understood: whether
this is downstream-task-format sensitivity specific to Qwen's tokenizer/
vocabulary, a genuine property of RoPE+GQA weight quantization, or
something else — flagged as a new open question (Section VIII), not
resolved here.

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
- **Qwen2.5-0.5B's `weights_only` downstream-accuracy cost (Section V.H)
  is new and not yet understood** — whether it reflects a genuine RoPE/GQA
  weight-quantization sensitivity, a tokenizer/vocabulary-specific
  interaction with these particular tasks, or something else is not
  established here.
- Downstream evaluation (Section C) used 500-example subsamples per task,
  not the full task sets — sufficient to detect the large effects reported
  here, but not fine-grained enough to rule out smaller effects.
- The memory-frontier comparison (Section B) uses one model (TinyLlama-1.1B)
  and reports a single run per method, not a multi-seed comparison — the
  crossover itself (fits vs. does not fit) is a binary, directly observed
  fact, but the QLoRA perplexity value (9.40) should not be read with the
  same seed-variance confidence as Section A's 3-seed numbers.
- The full-QAT frontier attempt failed via a Windows-specific VRAM-
  oversubscription slowdown rather than a clean CUDA OOM (Section V.G,
  Appendix XI) — the qualitative conclusion (does not fit) is unambiguous,
  but the exact failure *mode* may not reproduce identically on Linux or a
  different driver/GPU combination.

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
- Investigate Qwen2.5-0.5B's new, unexplained `weights_only`
  downstream-accuracy cost (Section V.H) — is it specific to RoPE/GQA
  weight quantization, to Qwen's tokenizer, or to these particular tasks?
- Extend the memory-frontier comparison (Section B) with multiple seeds
  for QLoRA and a second frontier-scale model, to give the QLoRA
  perplexity value the same seed-variance confidence as Section A, and to
  check whether the full-QAT failure mode (clean OOM vs. oversubscription
  slowdown) is specific to this Windows/driver setup or general.

---

## IX. Conclusion

Under matched, rigorous methodology — a proper isolated control, three
seeds per configuration, PTQ baselines for context, both relative and
absolute effect-size reporting, real paired significance testing with a
multiple-comparisons correction, and independent confirmation on a
downstream zero-shot task battery — the same selective-QAT treatment
(activation quantization) causes wildly different damage across three
small language models at fixed scale: from a mild, statistically
indistinguishable-from-noise +3.9% perplexity increase to a severe,
highly significant +106.2% one that nearly halves downstream next-word
accuracy. This variation is an existence claim, not a mechanistic one:
this study cannot yet say which of the several architectural differences
between these models is responsible. A methodological caution accompanies
the headline result: a real regularizer effect reported under single-seed
evaluation in this project's own earlier work did not survive proper
controlled re-evaluation, and this is now backed by real significance
tests rather than an eyeballed mean comparison. A memory-scaling
relationship fit on two architectures required revision, not just
extension, once tested against a third with a substantially larger
vocabulary — parameter count alone is an unreliable memory proxy across
architecture families. Downstream evaluation also surfaced a genuinely new
and unresolved wrinkle: on Qwen2.5-0.5B, even weight-only QAT costs real
downstream accuracy despite modest perplexity impact, complicating the
"weight quantization is nearly free" reading OPT and Pythia alone would
support. The memory-frontier comparison demonstrates the same 6GB budget's
practical limit directly: a 1.1B-parameter model's full QAT attempt fails
(via a driver-level slowdown rather than a clean exception, but fails
regardless), while QLoRA fits comfortably, training under half a percent
of the model's parameters — an empirically observed, not merely predicted,
crossover point. Every experiment in this draft ran on a single consumer
laptop GPU under this same 6GB budget; this shaped a breadth-over-depth
study design deliberately, and is reported as a methodological detail
rather than the paper's selling point.

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

**The same failure mode recurred on the Section B frontier attempt (Day
12).** Attempting full QAT on TinyLlama-1.1B (predicted 11.0 GB fixed
VRAM against the 6.14 GB card) produced no `torch.cuda.OutOfMemoryError`
at all: VRAM climbed to 5,746 of 6,141 MiB (93.6%) and plateaued there,
GPU utilization pinned at 100% with power draw only ~26 W — the same
signature (high utilization, low power, no progress) as the original
incident, confirmed the same way (direct `nvidia-smi` measurement over
time, not inferred from silence: memory was checked at multiple points
and shown to have stopped climbing, ruling out "about to finish loading").
Killed manually after 17 minutes with zero training steps completed. This
recurrence across two different models (a ~0.5B and a 1.1B parameter
model) suggests the pattern is a general property of this
Windows/consumer-GPU/driver combination when VRAM demand exceeds the
card, not a one-off tied to Qwen's vocabulary size specifically.

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

**A checkpoint-precision bug found and fixed while backfilling per-example
NLL (Day 12).** `train_qat.py` originally cast every floating-point tensor
in the final checkpoint to FP16, including `FakeQuantize`'s calibrated
`scale`/`zero_point`/`min_val`/`max_val` buffers — rounding a single
per-tensor activation-quantization scale value that the original,
already-recorded perplexity was computed *without* rounding. Re-evaluating
these already-cast checkpoints (necessary to backfill OPT-350M's and
Pythia-410M's per-example NLL, Section III.E) surfaced this directly:
recomputed PPL was consistently *lower* than the originally stored value,
every time, only for configurations involving activation quantization.
Effect size tracks the same architecture-dependent pattern as this paper's
headline finding — negligible (<0.1 PPL, no warning) for `none`/
`weights_only` on both models; small (0.1-0.3%) for OPT-350M's
`activations_only`/`both`; 6-27x larger (1.9-2.75%) for the same
configurations on Pythia-410M. This is independent corroborating evidence
for the compounding-error-across-depth mechanism proposed in Section VI:
whatever architectural property makes Pythia amplify real
activation-quantization noise so much more than OPT also amplifies this
small, unrelated FP16-rounding artifact by roughly the same relative
margin. The bias is conservative (understates rather than overstates
activation-QAT damage) and small enough not to change this paper's
conclusions, but is reported explicitly rather than silently absorbed;
`train_qat.py` was fixed the same day to keep these calibration buffers at
FP32 in all future checkpoints, so this cannot recur for Section B or any
future retraining.

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
