# Selective Quantization-Aware Training on Small Language Models

**A hardware-constrained study of which tensors actually need QAT: weights,
activations, or both — and where full QAT stops fitting a 6 GB GPU at all.**

This is an active undergraduate research project (Bennett University, SCSET)
being expanded from an ACL Submission (#173) that came back with reviewer
feedback. This README explains what the project is, what's been done, and
exactly where it stands right now — see [`docs/reports/`](docs/reports/) for
detailed session-by-session logs if you want the full story including bugs
found and fixed along the way, and [`docs/JOURNAL.md`](docs/JOURNAL.md) for
the narrative, "how it actually went" version of the same story.

## The research question

Quantization-Aware Training (QAT) recovers accuracy lost to compressing a
model's weights and activations to lower precision (e.g. INT8), by training
the model to compensate for the precision loss. But QAT is expensive — it can
need 4-6x the VRAM of the base model — and it's not obvious which tensors
(weights, activations, or both) actually benefit from that expense. This
project answers that question empirically, entirely on a single
consumer-grade laptop GPU.

## Background: why this exists

The original version of this study (single model — `facebook/opt-350m`,
single random seed) was submitted to ACL and reviewed. The review raised six
concrete points: no seed variance, only one model/dataset tested, no
comparison against established methods (QLoRA, SmoothQuant, AWQ), perplexity
was the only metric, some reproducibility details were missing, and there was
no discussion of broader impact. Separately, the original project's source
code was lost in a laptop reset — only the PDF write-ups, final results, and
a chat transcript survived. This repository is a from-scratch rebuild of that
pipeline, now being expanded into a proper multi-model, multi-seed study that
answers all six review points. The full expansion plan, with each reviewer
point mapped to a specific experiment, is in [`docs/PLAN.md`](docs/PLAN.md).

## Hardware this was built and run on

Every number in this repository was produced on one specific machine, under a
strict, deliberate VRAM ceiling — that ceiling is itself part of the research
question (can this work at all on hardware this constrained?):

| Component | Spec |
|---|---|
| CPU | 13th Gen Intel Core i5-13450HX |
| RAM | 24 GB |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU — **6,141 MiB VRAM** (the hard ceiling for every experiment here) |
| OS | Windows 11 |
| Python / PyTorch | 3.12.10 / 2.11.0+cu128 |

Full software version list in [`docs/reports/2026-08-23_session-1.md`](docs/reports/2026-08-23_session-1.md).

## Method, in short

1. Take a small causal language model (under ~500M params, so full QAT fits
   under 6 GB — see the memory-scaling math in
   [`docs/MATH.md`](docs/MATH.md)).
2. Establish baselines: zero-shot FP16, INT8 post-training quantization
   (PTQ), and SmoothQuant PTQ.
3. Establish a **fine-tuned FP16 control** — the same model, fine-tuned for
   the same 500 steps, but with no quantization at all. This exists because
   an earlier run showed that comparing a *fine-tuned* QAT model against a
   *zero-shot* baseline makes QAT look far better than it actually is (the
   model is also just learning the dataset, not only learning to handle
   quantization noise). The control isolates the quantization effect from
   the fine-tuning effect.
4. Run selective QAT: inject fake-quantization into **weights only**,
   **activations only**, or **both**, and compare all three against the
   control.
5. Repeat every trained configuration with **3 random seeds** and report the
   spread, not just one number.
6. Do this across multiple architecturally different models, not just one.

## Current status

**2 of 3 models in the architectural-diversity matrix (Section A of the plan)
are fully complete. Qwen2.5-0.5B's PTQ baselines are done; 11 of 12 QAT
runs are confirmed good, 1 in progress** (`none` 3/3, `weights_only` 3/3,
`activations_only` 3/3). `activations_only` was run, then deliberately
cleared back to 0/3 on Day 8 after 2 of its 3 seeds diverged numerically
(one to PPL~1.2e15, one to outright NaN) — a real, root-caused finding
(unclipped gradient explosion), not a bug in the usual sense. Gradient
clipping (`GRAD_CLIP_NORM=1.0`) was added and confirmed working in
production on the redo (Day 9). `both` is now 2/3 done (seed=42 PPL 31.81,
seed=1337 PPL 29.58); seed=2024 is mid-training (stopped cleanly at step
250/500 on Day 10, resumable) — see Known Issues.

**Honest framing (added Day 4, `docs/reports/2026-08-29_session-4.md`):**
Section A (the matrix) answers only 1 of the 6 original ACL reviewer
critiques — architectural diversity. The other 5 (seed variance now
partially addressed; QLoRA/SmoothQuant/AWQ comparisons partially addressed
via SmoothQuant + INT8; downstream evaluation, real significance testing,
and societal impact) are either partial or **not started at all**. Finishing
Qwen's matrix is not the same as finishing this project's response to the
reviewers — see `docs/JOURNAL.md` for the fuller reflection on this.

| Model | PTQ baselines | QAT matrix (4 configs x 3 seeds = 12 runs) |
|---|---|---|
| **OPT-350M** (learned position embeddings, MHA) | ✅ done | ✅ **12/12 done**, eval-set-consistency fixed on Day 3 |
| **Pythia-410M** (learned position embeddings, MHA, fused QKV) | ✅ done (SmoothQuant result anomalous — flagged, not yet debugged) | ✅ **12/12 done** |
| **Qwen2.5-0.5B** (RoPE, GQA) | ✅ done (SmoothQuant also anomalous here — see Known Issues) | 🔄 **11/12 confirmed good, 1 in progress** — `none` 3/3 (PPL 18.01 / 14.02 / 14.21), `weights_only` 3/3 (PPL 18.69 / 28.91 / 18.23), `activations_only` 3/3 redone with gradient clipping (PPL 28.74 / 31.80 / 32.15); `both` 2/3 (PPL 31.81 / 29.58), seed=2024 mid-training (step 250/500, stopped cleanly Day 10) |

**Overall: 35 of 36 planned QAT training runs complete or in progress (34 confirmed-good, 1 resumable mid-run).**

### Results so far

**OPT-350M** (fully complete, 3 seeds each):

| Configuration | Perplexity |
|---|---|
| FP16 zero-shot | 41.28 |
| INT8 PTQ | 41.39 |
| SmoothQuant PTQ | 43.67 |
| FP16 fine-tuned control | **21.02** |
| QAT weights-only | 21.09 |
| QAT activations-only | 21.84 |
| QAT both | 21.92 |

*(Revised on Day 3 after fixing an internal eval-set inconsistency — 5 of
OPT-350M's 12 runs were originally scored on the full 10k-example set before
the fast-eval fix existed, while the rest used the 1,500-example subset. All
12 are now consistently on the subset; see
[`docs/reports/2026-08-26_session-3.md`](docs/reports/2026-08-26_session-3.md).
The numbers shifted slightly but the finding didn't change.)*

**Pythia-410M** (fully complete, 3 seeds each):

| Configuration | Perplexity |
|---|---|
| FP16 zero-shot | 29.54 |
| INT8 PTQ | 29.72 |
| SmoothQuant PTQ | 256.50 *(broken — see Known Issues)* |
| FP16 fine-tuned control | **16.75** |
| QAT weights-only | 16.82 |
| QAT activations-only | 34.54 |
| QAT both | 34.58 |

**Qwen2.5-0.5B** (PTQ baselines + all 3 control seeds done; QAT matrix
11/12, `both`/seed=2024 mid-training):

| Configuration | Perplexity |
|---|---|
| FP16 zero-shot | 21.61 |
| INT8 PTQ | 21.84 |
| SmoothQuant PTQ | 88.17 *(broken — see Known Issues; also confirms this isn't Pythia-specific)* |
| FP16 fine-tuned control | **18.01 / 14.02 / 14.21** (seeds 42 / 1337 / 2024 — mean 15.41, stdev 2.25) |
| QAT weights-only | **18.69 / 28.91 / 18.23** (seeds 42 / 1337 / 2024) |
| QAT activations-only | **28.74 / 31.80 / 32.15** (seeds 42 / 1337 / 2024 — redone with gradient clipping after the Day 8 divergence, see Known Issues) |
| QAT both | **31.81 / 29.58 /** *(seed=2024 in progress, step 250/500)* (seeds 42 / 1337 / 2024) |

Notably, Qwen2.5-0.5B's zero-shot FP16 perplexity (21.61) is already close to
where OPT-350M and Pythia-410M land only *after* 500 steps of fine-tuning —
a modern, better-pretrained small model needs far less adaptation to the
target domain. Whether QAT still moves the needle meaningfully at that
starting point is one more open question the completed matrix will answer.

The control's seed-to-seed spread (~14.6% coefficient of variation) is
notably wider than OPT-350M or Pythia-410M ever showed (well under 1%).
Seed=42 (18.01) is the outlier against seeds 1337/2024 (~14.1 average), and
it's also the seed whose training run hit the Day 5 VRAM-thrashing incident
— worth treating as an open question for the formal significance test
rather than either dismissing or over-interpreting on 3 seeds alone (see
`docs/reports/2026-08-31_session-6.md` §6).

**`weights_only`'s two completed seeds show an even larger, unexplained
spread**: seed=42 lands close to its own control (18.69 vs. 18.01, ~4%
worse), while seed=1337 is dramatically worse than its control (28.91 vs.
14.02, ~106% worse) — the opposite pattern from OPT-350M/Pythia-410M, where
`weights_only` consistently tracked control within ~0.5% regardless of
seed. Not yet explained (genuine seed variance, training instability
specific to that seed, or an artifact of that run's higher peak VRAM) —
see `docs/reports/2026-09-07_session-7.md` §4.

### What the pattern looks like so far

On both models tested, **activations-only and both QAT strategies land worse
than the control**, consistent with the hypothesis that dynamic activation
outliers are the harder thing to quantize (weights are static and quantize
more predictably). **Weights-only QAT lands essentially tied with the
control on both models** (within ~0.5%) rather than clearly beating it — this
is notably different from the original single-seed study's headline claim
that weights-only QAT beats the baseline. That's being reported honestly
here: it looks like the original "QAT as regularizer" finding may not have
survived contact with seed variance, which is exactly the kind of thing the
ACL review's seed-variance critique was meant to catch.

The more striking finding is how differently the two architectures react to
**activation** quantization: on OPT-350M it's a mild +3.3% hit
(21.02 -> 21.84), but on Pythia-410M it's a **+106% hit** (16.75 -> 34.54) —
more than double. Pythia's GPT-NeoX-style fused QKV projection appears to be
dramatically more sensitive to activation quantization than OPT's separate
q/k/v projections. This is exactly the kind of architecture-dependence the
single-model v1 study had no way to see, and is a stronger, more general
result than anything in the original submission. Whether Qwen2.5-0.5B (RoPE,
GQA — a third, different design) lands closer to OPT's mild degradation or
Pythia's severe one is the open question the last model in the matrix will
answer. A formal significance test across all 3 models (once complete) will
also settle the weights-only-vs-control question properly — see
[`docs/PLAN.md`](docs/PLAN.md), Section D.

## Known issues (being tracked, not hidden)

- **SmoothQuant gives a broken (much too high) number on 2 of 3 models —
  Pythia-410M (256 PPL vs. ~29-30 expected) and Qwen2.5-0.5B (88 PPL vs.
  ~21-22 expected).** Only OPT-350M's SmoothQuant number looks right (43.67,
  a modest +5.8% over FP16). An earlier, more severe version of this bug on
  OPT-350M was already found and fixed (its post-norm architecture broke the
  standard SmoothQuant fold-into-LayerNorm trick entirely — see
  `docs/reports/2026-08-23_session-1.md`) with a from-scratch,
  architecture-agnostic reimplementation in `src/smoothquant_manual.py`.
  Originally assumed Pythia's fused QKV projection (GPT-NeoX style) was
  triggering a distinct edge case in that same code path — but Qwen2.5-0.5B
  uses separate q/k/v/o projections like OPT and *still* shows large
  degradation, which rules that explanation out. More likely culprit:
  something in the calibration itself (sample count, or `alpha=0.5` not
  being well-tuned across architectures) — not yet root-caused, needs real
  investigation before these numbers go in the paper.
- **AWQ is unavailable on this machine.** `autoawq` requires `triton`, which
  has no compatible Windows wheel for this Python/CUDA combination.
  INT8 dynamic + SmoothQuant stand in as the PTQ baseline family for now.
- **~~Intermittent, large slowdowns spanning multiple models and both
  training and eval phases~~ — root-caused and fixed for Qwen on Day 5.**
  First seen on Pythia-410M's eval (e.g. `both`/seed=1337: 7,528s vs. a
  normal ~150-200s), then far more severely on Qwen2.5-0.5B, where a single
  eval pass ran ~7 hours (`none`/seed=42, `docs/reports/2026-08-30_session-5.md`
  §5). Root cause found via direct GPU profiling rather than left as a
  mystery: **peak VRAM was exceeding the 6.14 GB card** (training measured
  6.62-9.46 GB; eval measured 11.47 GB), almost certainly triggering
  Windows' silent fallback to slow shared (system RAM) memory — which
  explains why GPU utilization stayed high the whole time despite no real
  progress being visible. Cause: Qwen's ~3x larger vocabulary (152K vs ~50K
  tokens) inflates memory beyond what parameter count alone predicts (see
  `docs/MATH.md` §1). **Fixed and verified**: `config.EVAL_BATCH_SIZE`
  lowered from 4 to 1, cutting eval's peak VRAM from 11.47 GB to 1.27 GB (a
  9x reduction) with zero effect on results (perplexity is exactly
  batch-size invariant). Training's smaller ~1.7 GB overshoot above the
  fitted prediction was not separately fixed — it's much milder and wasn't
  the dominant cause. Pythia's original (milder) instance of this pattern
  is presumably explained by the same general mechanism (VRAM pressure) but
  wasn't separately re-diagnosed.
- **Day 5's eval-speed fix (`EVAL_BATCH_SIZE=1`) didn't hold up in
  production — root-caused further and fixed on Day 6.** Day 5 verified a
  1.27 GB peak / ~80s eval in an isolated fresh-process benchmark; two real
  Qwen evals this session instead took ~27 min and ~62.5 min, with peak VRAM
  (~5.65 GB) still sitting right at the 6.14 GB card ceiling. Two
  compounding bugs found: (1) `common.evaluate_perplexity()` never wrapped
  its forward passes in `torch.no_grad()` — every eval this entire project
  has run built an unused autograd graph, wasting memory and compute (this
  also means Day 5's own benchmark number already included this waste, so
  the true safety margin was smaller than believed); (2) `train_qat.py`
  never freed the optimizer (8-bit AdamW state) or training's CUDA cache
  before calling eval in the same process, so eval was inheriting
  training's VRAM footprint on top of its own. Both fixed
  (`torch.no_grad()` added; `del optimizer; gc.collect();
  torch.cuda.empty_cache()` now runs before eval starts). **Neither run
  this session could benefit** — `run_matrix.py` is one continuous process
  per model, so a source-code fix only takes effect on the next fresh
  launch; this needs confirming on the next run, not assumed. See
  `docs/reports/2026-08-31_session-6.md` §2-3.
- **`activations_only` diverged numerically on 2 of 3 Qwen seeds — root-caused
  and fixed with gradient clipping on Day 8.** Never seen on `none` or
  `weights_only` (max loss there was 5.59 across all seeds). Seed 42 showed 3
  distinct loss spikes (up to 232.6x its baseline) that never fully
  recovered, landing at PPL=94.75 (~426% over control — far beyond even
  Pythia-410M's "severe" activation-sensitivity case of +106%). Seed 1337
  effectively collapsed (PPL≈1.23×10^15). Seed 2024 fully diverged, tracked
  precisely step-by-step: loss climbed from 661 to ~3×10^18 over 17 steps
  before overflowing to NaN at step 287, then stayed NaN for the rest of
  training (NaN is permanent once it enters Adam's momentum/variance state —
  verified directly, not assumed). **Fix**: standard gradient clipping
  (`config.GRAD_CLIP_NORM=1.0`, `torch.nn.utils.clip_grad_norm_` before each
  optimizer step, logged whenever it actually engages) — its absence was the
  real anomaly here, not its addition; this is standard practice in
  essentially all LLM/QAT training. All 3 `activations_only` seeds were
  cleared from results and queued for a redo with clipping (including
  seed=42, since it also showed spikes and should be trained under identical
  conditions to the other two). `none`/`weights_only` results were
  deliberately left as-is — neither ever showed a spike, and clipping is a
  no-op when the gradient norm is already under the threshold. See
  `docs/reports/2026-09-09_session-8.md`. **Day 9 update**: the fix is
  confirmed working in production, not just compiling -- the
  `activations_only`/seed=42 redo held a normal, bounded loss (3.2-4.5)
  through 115 steps with zero spikes and zero NaN, versus the original
  unclipped run which had already spiked 3 times by that point. One
  precise correction made along the way: gradient clipping is engaging on
  **100% of steps observed so far** (114/114), not tapering off as
  initially (incorrectly) reported in chat — every window's clip count
  just happened to equal that window's step count. This means clipping
  is acting as a continuous rescaling of every update for this
  configuration, not an occasional catch of rare outliers. Not itself a
  red flag (clipping preserves gradient direction, so it can't be the
  mechanism that corrupts training the way the original explosion did),
  but a real, more precise characterization worth keeping accurate — see
  `docs/reports/2026-09-13_session-9.md`. **Day 10 update**: `both`/seed=42
  and seed=1337 completed cleanly (PPL 31.81 / 29.58) with clipping engaged
  throughout and no spikes; seed=2024 (the 12th and final Qwen QAT run) was
  stopped mid-training by request at step 250/500, loss holding normal
  (3.2-4.8 range) with no spikes or NaN. Stop was a hard kill (not a
  graceful Ctrl+C/SIGTERM the training loop's interrupt handler could
  catch), so the periodic `CHECKPOINT_EVERY_STEPS=10` resume checkpoint was
  used rather than an immediate one — verified directly by loading the
  checkpoint (`step: 250`), confirming only 3 steps (~410s) of progress
  were lost, not the whole in-between window. Will resume from step 250 to
  finish the matrix.
- **Two more real VRAM/sync bugs found and fixed on Day 7, both on the very
  first run that could expose them.** (1) `FakeQuantLinear._log_error()`
  called `.item()` (a blocking CUDA sync) on every fake-quantized layer's
  error, every microbatch — 168 layers x 8 grad-accum steps = 1,344
  syncs/step for `weights_only` alone, versus zero for the `none` control
  runs (`track_error=False` there). Turned the very first QAT-strategy
  training step into ~54 minutes. Fixed by storing the error as a GPU
  tensor and flushing to plain floats once per step
  (`FakeQuantLinear.flush_error_log()`) instead of once per layer per
  microbatch. (2) Resuming from a checkpoint
  (`torch.load(resume_path, map_location=DEVICE)`) left the *entire*
  loaded checkpoint (~2.78 GB) sitting in VRAM for the whole run even after
  its values were copied into the live model/optimizer — `ckpt` was never
  freed. VRAM after setup: 5,696 MiB on a resume vs. 1,885 MiB on a fresh
  start of the same run. Fixed with `del ckpt; gc.collect();
  torch.cuda.empty_cache()`. Both verified against real runs (not just
  `py_compile`) before being trusted. See
  `docs/reports/2026-09-07_session-7.md` §1-2.
- **A bug introduced while fixing the above, caught the same session**:
  the post-training error-summary code still called `.item()` on values
  the new per-step flush had already converted to plain floats, crashing
  `weights_only`/seed=42 right after its 500 training steps finished (all
  training work preserved — only the post-processing crashed). Fixed with
  an `isinstance` guard. Side effect: because the crash happened after
  training but the resume checkpoint was saved at exactly step 500,
  resuming skipped the training loop entirely, so
  `FakeQuantLinear.error_log` never got repopulated — **seed=42's
  `mean_weight_rel_error` is permanently `null`** (PPL and per-example NLL,
  which come from eval, are unaffected). See
  `docs/reports/2026-09-07_session-7.md` §3.
- **~~No mid-training checkpointing~~ — fixed on Day 3, extended to eval on
  Day 5.** `train_qat.py` saves a full resumable checkpoint (model +
  FakeQuantize buffers + optimizer state + cumulative elapsed time) every
  `config.CHECKPOINT_EVERY_STEPS` (lowered from 50 to **10** on Day 5, once
  a single step was observed taking minutes rather than seconds — see the
  slowdown item above), plus immediately on Ctrl+C/SIGTERM. Verified against
  a hard kill via `TaskStop` (Day 3) — resume picked up at the correct step
  with optimizer state intact. **Day 5**: eval itself was previously *not*
  resumable at all — a run that finished training but got killed mid-eval
  had to redo the entire (potentially 10+ hour) training run from scratch.
  `common.evaluate_perplexity` now accepts a `resume_path` and checkpoints
  every `config.EVAL_CHECKPOINT_EVERY_BATCHES` (10) batches; the
  training-phase checkpoint is now kept until eval + results are both fully
  saved (previously deleted right after training, which was the actual gap).
  **Caveat**: the eval-resume path is syntax-verified but has not had a
  dedicated hard-kill test the way the Day 3 training checkpoint did — a
  genuine test of it is still owed.
- **~~stdout was silently full-buffered~~ — fixed Day 5.** For most of this
  project, progress prints didn't appear in real time when piped to a log
  file (Python's default when stdout isn't a terminal) — explaining a
  pattern seen repeatedly ("Loading weights: 100%" then silence, then
  everything at once at completion). Fixed with
  `sys.stdout.reconfigure(line_buffering=True)` in every entry-point
  script. Training now also prints every single step (was every 50) and
  eval prints every 10 batches with a running perplexity estimate — GPU
  utilization is no longer the only available signal that a run is
  progressing.
- **~~The Wilcoxon signed-rank test described in `docs/PLAN.md` Section D
  isn't implemented yet~~ — implemented Day 4, not yet exercised on real
  data.** `common.evaluate_perplexity(return_per_example=True)` now logs
  one NLL value per held-out example; `src/stats.py` runs a real paired
  Wilcoxon test per seed on that data, falling back to the old caveated
  t-test only for runs that predate this (all 24 completed OPT-350M/
  Pythia-410M runs, for now). `src/backfill_per_example_nll.py` can add
  this retroactively from saved checkpoints (no retraining) — dry-run
  verified, not yet actually run (needs a GPU-idle window; Qwen's matrix
  was using the GPU when this was built).
- **~~`docs/PLAN.md` claimed every result recorded its git commit hash and
  full hyperparameters~~ — fixed on Day 3.** It didn't, for any of the 24
  completed runs. Closed via `common.run_metadata()`, now wired into every
  result-logging path. Old runs aren't retroactively stamped (not needed —
  every run used identical settings from version-controlled `config.py`,
  fully recoverable from git history), but nothing new will have this gap.

## Setup

```bash
cd qat-research
python -m venv .venv
./.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
./.venv/Scripts/pip install -r requirements.txt
```

Verify CUDA:
```bash
./.venv/Scripts/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Running the pipeline (from `src/`)

1. **Tokenize data** (once per model):
   ```bash
   python prepare_data.py --model opt-350m
   python prepare_data.py --model pythia-410m
   python prepare_data.py --model qwen2.5-0.5b
   python prepare_data.py --model frontier
   ```

2. **PTQ baselines** (FP16 zero-shot, INT8 dynamic, SmoothQuant, AWQ):
   ```bash
   python eval_baseline.py --model opt-350m
   ```

3. **Full selective-QAT matrix** (resumable — skips any seed/strategy already
   completed as a real 500-step run):
   ```bash
   python run_matrix.py --model opt-350m
   python run_matrix.py --model pythia-410m
   python run_matrix.py --model qwen2.5-0.5b
   ```

4. **Memory-frontier study** (QAT-infeasibility probe + QLoRA comparison on a
   1.1B model — not yet run):
   ```bash
   python qlora_frontier.py --mode both
   ```

5. **Downstream zero-shot eval** (on trained checkpoints — not yet run):
   ```bash
   python eval_downstream.py --model opt-350m --strategy weights_only --seed 42
   ```

6. **Aggregate stats** (mean +/- std, significance test):
   ```bash
   python stats.py --model all
   ```

## Repository layout

- `src/` — all pipeline code (see [`docs/PLAN.md`](docs/PLAN.md) for what
  each script does and why).
- `src/config.py` — the model registry, hyperparameters, and seeds; the
  single source of truth every script reads from.
- `src/results/<model>.json` — every run's results, keyed by configuration
  name, one list entry per seed, each entry carrying its own hyperparameters
  inline (no separate config log to go stale).
- `docs/PLAN.md` — the full expansion plan, mapping each ACL reviewer point
  to a specific experiment.
- `docs/MATH.md` — the mathematical framing: a validated QAT memory-scaling
  formula, the derived QAT-feasibility crossover point, and the theoretical
  argument for why weight and activation quantization behave asymmetrically.
- `docs/reports/` — dated session reports with full detail on what was done,
  what was found, and every bug hit and fixed along the way.
- `checkpoints/`, `data/` — not committed (regenerable from code; checkpoints
  alone run into multiple GB per model).

## Next steps

1. Finish the Qwen2.5-0.5B QAT matrix (1 of 12 training runs remain:
   `both`/seed=2024, stopped mid-training at step 250/500, resumable from
   checkpoint). Once complete, evaluate current standing and decide next
   steps deliberately rather than rolling straight into Section B/C — see
   `docs/JOURNAL.md`.
2. Debug the SmoothQuant anomaly — now confirmed on 2 of 3 models, not
   Pythia-specific.
3. Run the memory-frontier (QLoRA) experiment on the 1.1B model.
4. Run downstream zero-shot evaluation (`lm-eval-harness`) on all trained
   checkpoints.
5. Run the formal cross-model statistics and significance tests (needs
   per-example NLL logging added first for the real Wilcoxon test).
6. Rewrite the paper with the honest multi-seed findings, add the missing
   societal-impact section, and settle on a target venue.
