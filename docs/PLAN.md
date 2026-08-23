# Expanded QAT Research Plan (v2)

Origin: ACL Submission #173 reviewer feedback. Goal: turn the single-model OPT-350M
study into a multi-model, statistically rigorous, publication-grade paper suitable
for a journal (venue not yet decided).

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
- facebook/opt-350m       (learned pos. embeddings, MHA)   -- P=0.331B
- EleutherAI/pythia-410m  (learned pos. embeddings, MHA)   -- P=0.405B
- Qwen/Qwen2.5-0.5B       (RoPE, modern tokenizer)         -- P=0.494B

Per model:
- FP16 zero-shot baseline (1 run, deterministic)
- INT8 PTQ baseline: dynamic (bitsandbytes), SmoothQuant, AWQ (1 run each, calibration-seeded)
- FP16 fine-tuned control (3 seeds)
- QAT weights_only (3 seeds)
- QAT activations_only (3 seeds)
- QAT both (3 seeds)

= 4 trained configs x 3 seeds x 3 models = 36 training runs, + 9 PTQ-family runs.

## Section B: Memory-frontier study (QAT vs. QLoRA)

Validated rule from Section-A data: peak QAT training VRAM ~= 10 * P (GB).
This predicts QAT becomes infeasible under 6GB once P > ~0.5-0.55B (matches
observed "danger zone" for Qwen2.5-0.5B). Pick one model at/above this crossover:

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

- Mean +/- std perplexity across 3 seeds per configuration.
- Paired comparison (same seeds, same eval set) between QAT weights_only and
  FP16 fine-tuned control per model -- Wilcoxon signed-rank test on per-example
  NLL (more robust than a t-test on 3 aggregate PPL numbers).
- Report effect size, not just p-value.

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
  results.json entry.
- Final paper appendix: full hyperparameter table.

## Rough compute budget

- Section A: ~3-5 hrs GPU time (training) + ~1-2 hrs (SmoothQuant/AWQ calibration)
- Section B: ~2-3 hrs GPU time
- Section C: several hours, parallelizable / can run overnight, subsampled
- Total: ~2-3 days of laptop time across sessions
