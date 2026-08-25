# Selective Quantization-Aware Training on Small Language Models

**A hardware-constrained study of which tensors actually need QAT: weights,
activations, or both — and where full QAT stops fitting a 6 GB GPU at all.**

This is an active undergraduate research project (Bennett University, SCSET)
being expanded from an ACL Submission (#173) that came back with reviewer
feedback. This README explains what the project is, what's been done, and
exactly where it stands right now — see [`docs/reports/`](docs/reports/) for
detailed session-by-session logs if you want the full story including bugs
found and fixed along the way.

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
are now fully complete.** This is the core experiment — everything else (the
memory-frontier/QLoRA study, downstream task evaluation, final statistics)
builds on top of it and hasn't started yet.

| Model | PTQ baselines | QAT matrix (4 configs x 3 seeds = 12 runs) |
|---|---|---|
| **OPT-350M** (learned position embeddings, MHA) | ✅ done | ✅ **12/12 done** |
| **Pythia-410M** (learned position embeddings, MHA, fused QKV) | ✅ done (SmoothQuant result anomalous — flagged, not yet debugged) | ✅ **12/12 done** |
| **Qwen2.5-0.5B** (RoPE, GQA) | not started | not started |

**Overall: 24 of 36 planned QAT training runs complete.**

### Results so far

**OPT-350M** (fully complete, 3 seeds each):

| Configuration | Perplexity |
|---|---|
| FP16 zero-shot | 41.28 |
| INT8 PTQ | 41.39 |
| SmoothQuant PTQ | 43.67 |
| FP16 fine-tuned control | **21.12** |
| QAT weights-only | 21.18 |
| QAT activations-only | 21.82 |
| QAT both | 21.92 |

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
(21.12 -> 21.82), but on Pythia-410M it's a **+106% hit** (16.75 -> 34.54) —
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

- **SmoothQuant gives a broken number on Pythia-410M** (256 PPL vs. an
  expected ~29-30). A similar bug was already found and fixed for OPT-350M
  (its post-norm architecture broke the standard SmoothQuant implementation
  — full writeup in the session report) using a from-scratch,
  architecture-agnostic reimplementation in `src/smoothquant_manual.py`.
  Pythia's fused QKV projection (GPT-NeoX style) appears to be triggering a
  different edge case in the same code path — not yet root-caused.
- **AWQ is unavailable on this machine.** `autoawq` requires `triton`, which
  has no compatible Windows wheel for this Python/CUDA combination.
  INT8 dynamic + SmoothQuant stand in as the PTQ baseline family for now.
- **Intermittent, large eval-time slowdowns on Pythia-410M.** A handful of
  runs (e.g. `both`/seed=1337: 7,528s eval vs. a normal ~150-200s) took far
  longer than every other run of the identical configuration, with no
  difference in code path or result correctness (perplexity landed in the
  expected range each time). Doesn't appear to correlate with a specific
  strategy — some `both` runs were fast, some slow. Likely background system
  contention (disk I/O, OS activity) on this laptop rather than a pipeline
  bug, but not confirmed. Doesn't affect correctness, only wall-clock time.

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

1. Run the Qwen2.5-0.5B matrix (12 training runs + baselines — the last model).
2. Debug the Pythia SmoothQuant anomaly.
3. Run the memory-frontier (QLoRA) experiment on the 1.1B model.
4. Run downstream zero-shot evaluation (`lm-eval-harness`) on all trained
   checkpoints.
5. Run the formal cross-model statistics and significance tests.
6. Rewrite the paper with the honest multi-seed findings, add the missing
   societal-impact section, and settle on a target venue.
