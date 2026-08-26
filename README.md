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
are fully complete. Qwen2.5-0.5B's PTQ baselines are done; its QAT matrix
hasn't produced a result yet** — the first attempted run ran 2.5+ hours
without finishing (an unresolved slowdown anomaly, not an error — see Known
Issues) and was deliberately stopped rather than left running indefinitely.
Session paused here; resuming is safe (see below).

| Model | PTQ baselines | QAT matrix (4 configs x 3 seeds = 12 runs) |
|---|---|---|
| **OPT-350M** (learned position embeddings, MHA) | ✅ done | ✅ **12/12 done**, eval-set-consistency fixed on Day 3 |
| **Pythia-410M** (learned position embeddings, MHA, fused QKV) | ✅ done (SmoothQuant result anomalous — flagged, not yet debugged) | ✅ **12/12 done** |
| **Qwen2.5-0.5B** (RoPE, GQA) | ✅ done (SmoothQuant also anomalous here — see Known Issues) | ⏸️ **0/12** — paused mid-first-run |

**Overall: 24 of 36 planned QAT training runs complete.**

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

**Qwen2.5-0.5B** (PTQ baselines only — QAT matrix paused before its first
run completed):

| Configuration | Perplexity |
|---|---|
| FP16 zero-shot | 21.61 |
| INT8 PTQ | 21.84 |
| SmoothQuant PTQ | 88.17 *(broken — see Known Issues; also confirms this isn't Pythia-specific)* |
| FP16 fine-tuned control | *(pending)* |
| QAT weights-only | *(pending)* |
| QAT activations-only | *(pending)* |
| QAT both | *(pending)* |

Notably, Qwen2.5-0.5B's zero-shot FP16 perplexity (21.61) is already close to
where OPT-350M and Pythia-410M land only *after* 500 steps of fine-tuning —
a modern, better-pretrained small model needs far less adaptation to the
target domain. Whether QAT still moves the needle meaningfully at that
starting point is one more open question the completed matrix will answer.

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
- **Intermittent, large slowdowns that now span multiple models and both
  training and eval phases.** First seen on Pythia-410M's eval (e.g.
  `both`/seed=1337: 7,528s eval vs. a normal ~150-200s). Then seen on
  Qwen2.5-0.5B's *training* phase — its first QAT run (`none`/seed=42) ran
  over 2.5 hours without completing 500 steps, versus the usual 20-45
  minutes, and was eventually killed deliberately rather than left running
  indefinitely (see `docs/reports/2026-08-26_session-3.md` §4). GPU stayed
  near 100% utilization throughout — this is not a hang, just unpredictably
  slow. No correlation found yet with strategy, model, or training-vs-eval
  phase. Given it's now recurred three times across two models and both
  phases of the pipeline, this should be treated as a real, unresolved
  infrastructure issue rather than isolated noise.
- **No mid-training checkpointing.** `train_qat.py` only saves a checkpoint
  after all 500 steps *and* evaluation complete. Combined with the slowdown
  issue above, this means an interrupted run loses all of its progress with
  no way to resume partway through. Recommended fix (not yet implemented):
  periodic checkpointing (e.g. every 100 steps) with resume-from-partial
  logic.
- **~~The Wilcoxon signed-rank test described in `docs/PLAN.md` Section D
  isn't implemented yet~~** — `src/stats.py` currently falls back to a
  Welch's t-test on 3 aggregate per-seed PPL values, explicitly caveated in
  its own output as too small a sample to trust. The real per-example paired
  test needs per-example NLL logging added to the eval loop first.
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

1. (Recommended before resuming long runs) Add mid-training checkpointing +
   resume-from-partial logic to `train_qat.py`, given the recurring
   slowdown issue below.
2. Finish the Qwen2.5-0.5B QAT matrix (12 training runs — baselines are done).
3. Debug the SmoothQuant anomaly — now confirmed on 2 of 3 models, not
   Pythia-specific.
4. Run the memory-frontier (QLoRA) experiment on the 1.1B model.
5. Run downstream zero-shot evaluation (`lm-eval-harness`) on all trained
   checkpoints.
6. Run the formal cross-model statistics and significance tests (needs
   per-example NLL logging added first for the real Wilcoxon test).
7. Rewrite the paper with the honest multi-seed findings, add the missing
   societal-impact section, and settle on a target venue.
