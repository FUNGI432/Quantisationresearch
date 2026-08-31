# Expanded QAT Research Plan (v2)

Origin: ACL Submission #173 reviewer feedback. Goal: turn the single-model OPT-350M
study into a multi-model, statistically rigorous, publication-grade paper suitable
for a journal (venue not yet decided).

## Status (updated after Day 6)

| Section | Status |
|---|---|
| A: Selective-QAT matrix | 🔄 2 of 3 models complete (OPT-350M ✅, Pythia-410M ✅, Qwen2.5-0.5B baselines done / QAT matrix at **3/12** complete -- all 3 `none`/control seeds, see session-6 report) -- 27/36 training runs done |
| B: Memory-frontier (QLoRA) | **Not started at all** -- and its planning assumptions need revisiting per the Day 5 vocabulary-size finding (see below) |
| C: Downstream evaluation | **Not started at all** (pipeline built, only ever smoke-tested) -- flagged Day 4 |
| D: Statistics | 🔄 Real Wilcoxon signed-rank test implemented (Day 4), not yet exercised on real data -- needs a GPU-idle window to backfill the 24 completed OPT/Pythia runs plus Qwen's seed=42 control (`backfill_per_example_nll.py`, dry-run verified) before a paired test can run on Qwen's control seeds as a full set |
| E: Mathematical framing | 🔄 In progress -- now backed by real 3-model data, and a real finding that the params-only memory model breaks down for large-vocabulary architectures (Day 5, `docs/MATH.md` §1-2) |
| F: Reproducibility | 🔄 Ongoing -- code + results pushed to GitHub after each session; mid-training/eval resumability and per-step visibility added Day 5; two real eval-path bugs (missing `torch.no_grad()`, optimizer/cache not freed before eval) found and fixed Day 6 |

**Day 6 finding (see `docs/reports/2026-08-31_session-6.md`):** Day 5's
eval-speed fix (`EVAL_BATCH_SIZE=1`) did not hold up in production the way
its isolated benchmark suggested -- both real evals this session ran
20-45x slower than the ~80s benchmark figure, with peak VRAM (~5.65 GB)
still sitting right at the 6.14 GB card ceiling. Root-caused to two
compounding bugs: `evaluate_perplexity()` never used `torch.no_grad()`
(building an unused autograd graph on every eval batch, project-wide, since
the function was first written), and the optimizer/training CUDA cache was
never freed before eval started in the same process. Both fixed in
`src/train_qat.py` / `src/common.py`; neither could take effect on the
runs already in flight (`run_matrix.py` is one continuous process per
model), so this needs confirming on the very next fresh launch, not
assumed fixed.

**Day 5 finding (see `docs/MATH.md` §1-2 and
`docs/reports/2026-08-30_session-5.md`):** a real multi-hour slowdown on
Qwen2.5-0.5B was root-caused (not just worked around) to VRAM exceeding the
6GB card during both training (6.62-9.46 GB measured vs. ~5.1 GB predicted)
and especially eval (11.47 GB with batch_size=4, fixed to 1.27 GB with
batch_size=1 -- a 9x reduction, verified empirically, with zero effect on
results since perplexity is batch-size invariant). Root cause: Qwen's ~3x
larger vocabulary inflates memory beyond what parameter count alone
predicts -- the memory-scaling law needs a vocabulary-size term, which is a
stronger, more defensible finding for the paper than the original
single-variable law would have been.

**Honest note (Day 4):** completing Section A's matrix answers only 1 of
the 6 original reviewer critiques (architectural diversity). Sections B and
C remain fully unstarted; Section D moved from unstarted to "built,
awaiting data" this session. See `docs/reports/2026-08-29_session-4.md` and
`docs/JOURNAL.md` for the fuller reasoning behind flagging this explicitly.

Headline finding so far (see `docs/MATH.md` §3 and
`docs/reports/2026-08-24_session-2.md` §2.1): weights-only QAT behaves
almost identically on OPT-350M and Pythia-410M (~+0.3-0.5% PPL either way),
but activation-only QAT is a mild +3.3% hit on OPT-350M versus a severe
+106% hit on Pythia-410M -- a genuine architecture-dependence result Section
A was specifically designed to be able to surface. Whether Qwen2.5-0.5B
(RoPE, GQA) lands closer to one extreme or the other is still open.

## Reviewer weakness -> plan item

| # | Weakness (ACL review) | Plan item |
|---|---|---|
| 1 | Single model / single dataset | 3-model selective-QAT matrix (Sec. A) + 1-model frontier study (Sec. B) |
| 2 | No seeds / variance | 3 seeds per trained configuration, report mean +/- std, paired significance test |
| 3 | No QLoRA / SmoothQuant / AWQ comparison | SmoothQuant + AWQ as PTQ baselines (Sec. A); QLoRA as the comparison method in the frontier study (Sec. B) |
| 4 | Perplexity-only eval | Zero-shot downstream accuracy via lm-eval-harness (LAMBADA, PIQA, HellaSwag, subsampled) |
| 5 | Missing repro details | Full hyperparameter appendix + code release (this repo) |
| 6 | No societal impact section | Added to final paper draft |

## Section A: Selective-QAT architectural-diversity matrix

Models (all fit the validated ~10P-byte QAT memory rule under 6GB):
- facebook/opt-350m       (learned pos. embeddings, MHA)        -- P=0.331B -- DONE
- EleutherAI/pythia-410m  (learned pos. embeddings, fused QKV)  -- P=0.405B -- DONE
- Qwen/Qwen2.5-0.5B       (RoPE, GQA, modern tokenizer)         -- P=0.494B -- baselines done, QAT paused at 0/12

Per model:
- FP16 zero-shot baseline (1 run, deterministic)
- INT8 PTQ baseline: dynamic (bitsandbytes), SmoothQuant, AWQ (1 run each, calibration-seeded)
- FP16 fine-tuned control (3 seeds)
- QAT weights_only (3 seeds)
- QAT activations_only (3 seeds)
- QAT both (3 seeds)

= 4 trained configs x 3 seeds x 3 models = 36 training runs, + 9 PTQ-family runs.

## Section B: Memory-frontier study (QAT vs. QLoRA)

**Superseded (Day 5, see `docs/MATH.md` §1-2 and
`docs/reports/2026-08-30_session-5.md` §5): the params-only rule below
(`9.49 * P + 0.41`) does NOT hold across architecture families.** It
predicted Qwen2.5-0.5B would fit comfortably (~5.1 GB); real measured
training peak was 6.62-9.46 GB, causing a genuine multi-hour slowdown
incident (Windows' shared-GPU-memory fallback under VRAM pressure). Root
cause: Qwen's ~3x larger vocabulary (152K vs ~50K tokens) inflates the
final logits/loss tensor beyond what parameter count alone predicts. The
frontier model below should be chosen and its expected VRAM estimated with
vocabulary size as a second variable, not parameter count alone -- and its
eval batch size should be set conservatively (batch_size=1, per the Day 5
fix in `config.EVAL_BATCH_SIZE`) regardless of what the training-side
memory estimate suggests, since eval and training peak VRAM are independent
quantities that must each be checked separately.

Original (still useful within the OPT/Pythia-like small-vocabulary family):
peak QAT training VRAM ≈ 9.49 * P + 0.41 (GB), predicting QAT becomes
infeasible under 6GB once P > ~0.54-0.59B. Pick one model at/above this
crossover for the frontier study:

- TinyLlama-1.1B or SmolLM2-1.7B -- P >= 1.1B, full QAT provably exceeds 6GB.

Experiment:
1. Attempt full QAT (expected to OOM or require sequence length collapse) -- document the failure quantitatively (measure actual OOM point vs. formula prediction).
2. Run QLoRA (4-bit frozen base + 16-bit LoRA adapters) as the comparison method that *does* fit.
3. Report: perplexity, peak VRAM, trainable-parameter count, wall-clock time for both.

This converts "we didn't compare to QLoRA" into a genuine finding: an empirically
validated crossover point at which practitioners must switch from QAT to PEFT
under a fixed VRAM budget.

## Section C: Downstream evaluation

lm-eval-harness, zero-shot, subsampled for laptop time budget:
- LAMBADA (next-word prediction accuracy -- tests whether QAT strategies preserve
  long-range coherence, not just local perplexity)
- PIQA (physical commonsense, short-context sanity check)
- HellaSwag (commonsense completion, subsampled to ~500 examples per config to
  keep runtime bounded)

Run on every trained checkpoint from Section A (best seed by dev perplexity, or
all 3 seeds if time allows) + both models in Section B.

## Section D: Statistics

- Mean +/- std perplexity across 3 seeds per configuration -- implemented,
  `stats.py::summarize()`.
- Paired comparison (same seeds, same eval set) between QAT weights_only and
  FP16 fine-tuned control per model -- Wilcoxon signed-rank test on
  per-example NLL (more robust than a t-test on 3 aggregate PPL numbers) --
  **implemented Day 4** (`stats.py::wilcoxon_per_seed()`, backed by
  `common.evaluate_perplexity(return_per_example=True)`), not yet run on
  real multi-seed data. Also now covers activations_only and both vs.
  control, not just weights_only.
- Report effect size, not just p-value -- mean per-example NLL diff is
  reported alongside the p-value; a more formal effect-size statistic
  (e.g. rank-biserial correlation) is not yet added.

## Section E: Mathematical framing (addresses "make it a proper mathematical paper")

- Formal derivation of the QAT memory formula (~10P bytes), validated against
  measured peak VRAM across all models in Section A -- turns an engineering
  anecdote into a citable equation with empirical validation (Table + fitted line).
- Formal statement of the Straight-Through Estimator gradient approximation used
  by FakeQuantLinear, and why it introduces bias in the activation path
  specifically (motivating the empirical asymmetry between weight and activation
  QAT).
- Grounded theoretical framing of the "weight quantization as regularizer" claim:
  connect INT8 weight quantization noise to known noise-injection / flat-minima
  regularization results in the literature, rather than asserting it purely from
  the PPL numbers.

## Section F: Reproducibility

- All code in this repo (`qat-research/`), pinned `requirements.txt`.
- Every run writes full hyperparameters + seed + git commit hash into its
  results.json entry (`common.run_metadata()` -- this was claimed but not
  actually true until Day 3; see the session-3 report §1).
- Mid-training checkpointing (Day 3, `config.CHECKPOINT_EVERY_STEPS`) means a
  run can be safely paused (even via a hard kill) and resumed without
  restarting from step 0 -- verified end-to-end, see session-3 report §5.
- Final paper appendix: full hyperparameter table.

## Rough compute budget

- Section A: ~3-5 hrs GPU time (training) + ~1-2 hrs (SmoothQuant/AWQ calibration)
  -- **actual observed: significantly higher**, partly due to intermittent
  eval-time anomalies on Pythia-410M (some runs' eval took up to ~40x longer
  than normal with no code-path difference or correctness issue -- see
  `docs/reports/2026-08-24_session-2.md` §2.2, still unresolved). Treat this
  budget as a lower bound, not an estimate to plan sessions around.
- Section B: ~2-3 hrs GPU time
- Section C: several hours, parallelizable / can run overnight, subsampled
- Total: ~2-3 days of laptop time across sessions (revised: likely more,
  given the above)
