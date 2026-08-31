# Paper Outline (v2 -- revised after blind review round 1, score 5/10)

**Working title**: *Quantization Sensitivity Varies by Architecture: A Multi-Model Study of Selective Quantization-Aware Training Under a Consumer-GPU Memory Budget*

**Core thesis** (existence claim, explicitly not a mechanistic claim):
Under matched, rigorous methodology (proper control, 3 seeds, real
significance testing), the amount of damage the *same* selective-QAT
treatment (activations-only) causes varies dramatically across three
small, similarly-sized language models -- from a mild effect to a severe
one. This variation is large enough, and survives strict statistical
scrutiny well enough, that it cannot be attributed to noise. We do not
claim to have isolated *which* architectural feature causes it (position
encoding, attention structure, vocabulary size, and normalization
placement all differ simultaneously across our three models) -- we
propose one candidate mechanism, state it explicitly as an untested
hypothesis, and specify the controlled experiment that would test it as
concrete future work. Separately, we show that a memory-scaling law fit
on two of these architectures needs revision (not just extension) once
tested against a third, and we are explicit that three data points support
a *candidate* relationship, not a validated law.

**Positioning against existing work (added per review feedback)**: it is
already known that quantization difficulty is not architecture/scale
invariant -- LLM.int8() (Dettmers et al. 2022) documents outlier-driven
degradation that grows with model *scale within a family*. Our contribution
is narrower and different in kind: we hold scale roughly fixed (~350-500M
params) and vary architecture *family*, showing the same nominal treatment
produces wildly different damage even without scaling up. This is a
different axis of variation than prior outlier-scaling work, not a
re-demonstration of it -- the paper must say this explicitly rather than
let a reader assume it's already been shown.

---

## Abstract

State: the constraint (single 6GB consumer laptop GPU, no institutional
compute); the method (selective QAT vs. a properly isolated FP16
fine-tuned control and PTQ baselines, across 3 architecturally distinct
small language models, 3 seeds each, real per-example paired significance
testing); the headline finding, stated as an existence claim with both
relative AND absolute effect sizes reported (avoiding the
percentage-perplexity scale-invariance trap flagged in review); the
memory-scaling finding, stated as a *candidate* relationship requiring
out-of-sample validation, not a validated law; the practical upshot,
appropriately hedged (mixed-precision recommendations may need
architecture-specific caveats, pending confirmation of a causal
mechanism). State plainly that every experiment ran on a single consumer
laptop, but do not lean on this as the paper's primary selling point --
it is a methodological detail that enabled the study design, not a
finding in itself.

## I. Introduction

- Motivate the memory wall / edge deployment problem briefly.
- State the actual gap: quantization studies (including this project's own
  predecessor) typically test one model and implicitly assume findings
  generalize across architecture *families* at fixed scale. Distinguish
  this explicitly from known scale-dependent outlier work (LLM.int8.) --
  see Related Work.
- **Contributions, split into two tiers (per review feedback -- do not
  dilute scientific claims with engineering ones):**

  **Scientific findings:**
  1. Evidence that activation-quantization sensitivity varies substantially
     across architecture families at matched, fixed scale -- reported with
     both relative (%) and absolute (NLL/bits-per-token) effect sizes, and
     validated with a real per-example paired significance test, not just
     an aggregate comparison.
  2. A methodological demonstration that a claimed "QAT-as-regularizer"
     effect (weight-only QAT beating an unquantized control), which
     appeared under single-seed evaluation, weakens substantially under
     proper multi-seed, controlled evaluation with multiple-comparisons
     awareness -- a caution against trusting single-seed QAT claims.
  3. A memory-scaling relationship for QAT VRAM requirements fit on two
     architectures, explicitly shown to be insufficient (not just
     "extendable") when tested against a third -- presented as a candidate
     relationship pending out-of-sample validation, with the real incident
     and root-cause analysis that revealed the gap.
  4. Identification of a silent failure in a standard SmoothQuant
     implementation on non-pre-norm architectures (root-caused and fixed
     with an architecture-agnostic reimplementation), plus the open,
     explicitly unresolved observation that the same failure mode recurs
     on 2 of 3 tested architectures.

  **Engineering / reproducibility artifacts** (presented as enabling
  infrastructure, not headline contributions):
  5. A memory-optimized selective-QAT pipeline with mid-training/eval
     checkpointing verified against a real hard-kill scenario, making
     long, interruptible research tractable on a single non-dedicated GPU.
  6. A fully open, reproducible pipeline (code, per-seed results,
     per-example evaluation data, full session-by-session engineering log).

## II. Related Work

- Post-Training Quantization: SmoothQuant, GPTQ, AWQ -- and where each
  assumes an architecture family (pre-norm, separate QKV) shown here not
  to hold universally.
- Quantization-Aware Training: memory cost, 8-bit optimizers (Dettmers et
  al. 2021), why QAT is understudied for transformers relative to CNNs.
- Mixed-precision / extreme quantization: QLoRA, SpQR, 1.58-bit LLMs,
  MatQuant.
- **Activation outlier and scale-dependence literature -- direct
  engagement, not just a citation**: LLM.int8() (Dettmers et al. 2022)
  shows outlier-driven quantization difficulty emerges and grows *with
  model scale within a family*. This paper's contribution is explicitly
  differentiated: holding scale roughly fixed and varying architecture
  *family* instead. State plainly that this is a narrower, complementary
  finding, not a bigger or competing claim.
- Explicit gap statement: no prior work we are aware of holds architecture
  family as the controlled variable at fixed small scale, with matched
  seeds/controls/significance testing across the comparison.

## III. Methodology

A. **Pipeline architecture**: FP16 zero-shot baseline -> PTQ family (INT8
   dynamic, SmoothQuant, AWQ where feasible) -> FP16 fine-tuned control ->
   selective QAT (weights-only / activations-only / both).

B. **The domain-adaptation control, and why it exists**: an early QAT run
   produced an implausibly good perplexity by fine-tuning on the same data
   used for evaluation, conflating "learned the domain" with "learned to
   compensate for quantization." A matched FP16 fine-tuned control isolates
   the quantization effect from the fine-tuning effect -- framed as a
   methodological lesson (a confound a single-baseline QAT study would not
   catch), not just a fix.

C. **Memory optimization under a 6GB ceiling**: gradient checkpointing,
   8-bit AdamW, micro-batching with gradient accumulation -- presented
   alongside the memory-scaling analysis (Section V.F) as its direct
   product, with the analysis explicitly scoped as a candidate relationship
   (see Core Thesis positioning above).

D. **Selective FakeQuant injection**: wrapper preserving original
   `nn.Parameter` objects, FakeQuantize submodules for weights and/or
   activations, native Straight-Through Estimator. Per-layer relative
   quantization error logged during training for the error-ratio analysis
   in Discussion.

E. **Statistical methodology (strengthened per review)**: 3 seeds per
   trained configuration; a real per-example paired Wilcoxon signed-rank
   test on held-out NLL (same example, same seed, two configs) rather than
   a t-test on 3 aggregate perplexity numbers. **Explicitly address
   multiple comparisons**: state the total number of significance tests run
   across models/configs and apply a correction (e.g. Benjamini-Hochberg
   FDR) rather than reading each p-value in isolation. Report effect sizes
   (mean per-example NLL difference) alongside p-values, in both relative
   and absolute terms.

F. **Reproducibility infrastructure**: every result records exact
   hyperparameters and git commit; training and evaluation are both
   resumable at a fine grain, verified against a real hard-kill scenario --
   presented in Methodology as enabling infrastructure, cross-referenced to
   the Reproducibility Appendix rather than claimed as a headline result.

## IV. Experimental Setup

- **Models** (table: params, position encoding, attention type, vocab size,
  normalization placement): OPT-350M (learned pos. emb., MHA, post-norm,
  ~50K vocab), Pythia-410M (learned pos. emb., MHA, fused QKV, ~50K vocab),
  Qwen2.5-0.5B (RoPE, GQA, ~152K vocab). **Explicitly flag the confound**:
  these three models differ along at least four architectural axes
  simultaneously; this study can establish that cross-architecture
  variation exists and is large, but cannot by itself attribute it to any
  single axis (see Limitations and Future Work).
- **Known architectural note on OPT-350M**: OPT-350M is documented to
  depart from its own model family's scaling conventions (different
  normalization placement than other OPT checkpoints) -- flagged
  explicitly as a reason to treat it as one data point rather than a clean
  "representative post-norm architecture," strengthening (not weakening)
  the paper's honesty about the n=3 comparison's limits.
- **Dataset**: WikiText-2, 10,000 pre-tokenized samples, 512-token
  sequence length; a fixed 1,500-example held-out subset used for the
  seed-level matrix, validated against the full 10k-example set for a
  subset of runs (negligible difference observed -- report the specific
  comparison).
- **Hardware**: stated exactly and treated as a first-class experimental
  parameter -- the 6GB ceiling is load-bearing for the memory-scaling
  analysis.

## V. Results

A. PTQ baselines, all 3 models (table).
B. Domain-adaptation isolation: naive QAT vs. proper control, with numbers.
C. Selective-QAT results, all 3 models x 4 configs x 3 seeds, mean ± std.
D. **Significance testing, with multiple-comparisons correction applied**
   (per Methodology E) -- report which effects survive correction and
   which do not; this directly operationalizes the "regularizer effect"
   fragility claim from the Introduction rather than asserting it.
E. **The headline cross-architecture comparison, reported in BOTH relative
   and absolute terms** (% PPL change AND absolute NLL/bits-per-token
   shift) -- explicitly to preempt the scale-invariance critique; if the
   absolute-terms comparison tells a different story than the percentage
   framing, report that honestly rather than lead with whichever framing
   looks more dramatic.
F. Memory-scaling relationship: the two-architecture fit, its failure on
   the third, and the vocabulary-size argument -- explicitly labeled a
   candidate relationship with only 3 data points and framed as motivating
   future out-of-sample validation, not as a validated law.
G. *(if completed by submission)* Memory-frontier result (QAT vs. QLoRA
   crossover), reporting both training-peak and eval-peak VRAM as
   independent quantities (per the Day 5 finding that they are not the
   same thing).
H. *(if completed by submission)* Downstream zero-shot task results
   (LAMBADA/PIQA/HellaSwag).

## VI. Discussion

- Interpret the architecture-dependence result strictly as an existence
  claim. Propose the fused-QKV / error-concentration mechanism as one
  candidate explanation among possibly several (position encoding and
  vocabulary size are equally plausible given the confound), explicitly
  unconfirmed.
- **State the controlled ablation that would test this directly, as a
  specific, well-defined next experiment** (see Future Work) -- do not
  leave the mechanism question purely rhetorical.
- Interpret the fragility of the "regularization" effect under proper
  seeding and multiple-comparisons correction as a field-general caution.
- Interpret the memory-scaling finding as a practical warning (parameter
  count alone is an unreliable proxy for large-vocabulary models) while
  being explicit that 3 points cannot yet support a general law.
- Discuss the SmoothQuant tooling failure as evidence that standard PTQ
  tooling encodes architectural assumptions that do not hold universally,
  and that a successful run with no error is not proof of correctness.
- Revisit mixed-precision recommendations from prior work as needing
  architecture-specific caveats -- explicitly hedge this on the unconfirmed
  mechanism, not state it as settled.

## VII. Limitations

- **The core confound, stated directly (not just implied)**: three models
  differing on 4+ architectural axes simultaneously means this study
  demonstrates that architecture-dependent variation exists, not which
  specific feature causes it.
- Model scale capped at ~500M parameters by the hardware budget -- reframed
  as a quantified, predictable boundary via the (candidate) memory relation
  rather than an unexplained ceiling.
- The SmoothQuant failure on 2 of 3 architectures is documented but not
  fully root-caused -- an explicitly open problem.
- The memory-scaling relationship rests on 3 data points and has not been
  validated out-of-sample -- state this plainly rather than calling it
  "validated."
- Single dataset (WikiText-2); cross-domain generalization untested beyond
  whatever downstream tasks are completed by submission.
- The VRAM-oversubscription slowdown behavior is specific to this
  Windows/consumer-GPU setup and may not reproduce identically elsewhere.

## VIII. Future Work (elevated to its own section per review feedback --
     not buried as a throwaway line in Discussion)

- **The single highest-priority follow-up**: a controlled ablation
  isolating exactly one architectural axis (e.g., a matched pair of models
  identical in parameter count, vocabulary, and training data, differing
  only in fused-vs-separate QKV projections, or only in normalization
  placement) to convert the architecture-dependence finding from an
  existence claim into a mechanistic one. Specify what such a matched pair
  would need to look like and how it would be trained/sourced.
- Out-of-sample validation of the memory-scaling relationship on a fourth,
  independently-chosen architecture not used to fit or patch the equation.
- Root-causing the SmoothQuant failure recurrence beyond the two
  architectures where it's currently just documented.
- Extending the confound-isolation logic to the memory-frontier and
  downstream-evaluation results if those are completed after this
  submission.

## IX. Conclusion

Restate the core finding as an existence claim (architecture-dependent
variation is real, large, and survives significance testing), the
methodological caution (seed variance and multiple comparisons can erase a
headline claim), the appropriately-hedged mathematical contribution
(parameter count alone is an unreliable memory proxy; a fuller model needs
more data to validate), and the honest scope statement: this was achieved
on a single consumer laptop, which shaped the study's breadth-over-depth
design choice deliberately, not as a constraint apologized for.

## X. Broader Impact / Societal Considerations

Short, direct: easier edge deployment of LMs via better-understood
quantization has dual-use texture; briefly note bias/privacy considerations
of on-device models operating without server-side moderation, and that
this study does not evaluate those dimensions directly.

## XI. Reproducibility Appendix

- Full hyperparameter table.
- Mid-training/eval checkpointing design and hard-kill verification.
- The VRAM-oversubscription incident, written up as a standalone
  methodological note (root cause, diagnosis method, fix, verification).
- Code and data availability statement.

## References

Existing list (Gholami et al. 2022; Xiao et al. 2023 SmoothQuant; Dettmers
et al. 2021 8-bit optimizers; **Dettmers et al. 2022 LLM.int8() -- added,
now load-bearing for the novelty positioning above**; Frantar et al. 2023
GPTQ; Lin et al. 2024 AWQ; Dettmers et al. 2023a QLoRA; Dettmers et al.
2023b SpQR; Ma et al. 2024 1.58-bit LLMs; Wei et al. 2022 outlier
suppression) plus the OPT architectural-family documentation (OPT paper /
appendix) supporting the OPT-350M oddity note in Section IV.
