"""Central model registry and shared hyperparameters for the QAT study."""

# --- Section A: selective-QAT architectural-diversity matrix -----------------
MATRIX_MODELS = {
    "opt-350m": {
        "hf_id": "facebook/opt-350m",
        "params_b": 0.331,
        "pos_embedding": "learned",
        "attention": "MHA",
    },
    "pythia-410m": {
        "hf_id": "EleutherAI/pythia-410m",
        "params_b": 0.405,
        "pos_embedding": "learned",
        "attention": "MHA",
    },
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "params_b": 0.494,
        "pos_embedding": "RoPE",
        "attention": "GQA",
    },
}

# --- Section B: memory-frontier study (QAT vs. QLoRA) -------------------------
FRONTIER_MODEL = {
    "hf_id": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "params_b": 1.1,
    "pos_embedding": "RoPE",
    "attention": "GQA",
}

# --- Shared training hyperparameters -----------------------------------------
TRAIN_STEPS = 500
# Save a resumable mid-training checkpoint every N steps, so a run can be
# killed (to free the GPU for something urgent, or because it's stuck) and
# continued later without losing more than a few minutes of progress. Also
# saved immediately on Ctrl+C / SIGTERM regardless of this interval. See
# train_qat.py. Lowered from 50 to 10 after a run where 50 steps took over
# 2 hours during an unexplained slowdown -- 50 was found to be far too
# coarse a safety margin when a single step can take minutes instead of
# seconds. The checkpoint write itself (a few GB) takes low single-digit
# seconds on this machine's disk, so 5x more frequent saves cost negligible
# overhead relative to the interval length even in the slow case.
CHECKPOINT_EVERY_STEPS = 10
# Same reasoning applied to eval, which previously had NO checkpointing at
# all -- see common.evaluate_perplexity's resume_path parameter.
EVAL_CHECKPOINT_EVERY_BATCHES = 10
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
MAX_SEQ_LEN = 512
LEARNING_RATE = 1e-5
# Added Day 8 after activations_only diverged on 2 of 3 Qwen seeds (one to
# PPL~1.2e15, one to outright NaN) -- large, sporadic gradient spikes tied
# to activation FakeQuantize's per-tensor scale reacting to an outlier
# activation. Standard practice in LLM training regardless; its absence
# was the real anomaly. 1.0 is the common default, not tuned to this data.
GRAD_CLIP_NORM = 1.0
SEEDS = [42, 1337, 2024]
TOKENIZE_SAMPLES = 10_000

# Full 10k-example perplexity eval takes ~30 min per run under FakeQuant
# overhead -- fine for the one-off PTQ baselines, but re-running it after
# every one of the 36 QAT training runs would add 15-20+ hours of eval time
# alone. Use a fixed, non-shuffled subset for the per-seed matrix sweep
# instead: perplexity estimates stabilize well before 10k examples, and using
# the SAME fixed subset across every strategy/seed keeps comparisons
# apples-to-apples (a random subset per run would add unwanted variance on
# top of the seed variance we're already trying to measure).
EVAL_SUBSET_SIZE = 1_500

# Lowered from 4 to 1 after discovering (Day 5) that Qwen2.5-0.5B's eval
# peaked at 11.47 GB -- nearly double the 6.14 GB card -- causing Windows'
# CUDA driver to silently fall back to slow shared (system RAM) memory,
# which is the real explanation for a 7-hour eval that should take ~3
# minutes (high GPU utilization throughout, since the driver keeps
# retrying, is NOT a reliable signal that this isn't happening). Root
# cause: Qwen's ~152K-token vocabulary (vs ~50K for OPT/Pythia) makes the
# final logits/loss tensor much larger per example than parameter count
# alone predicts, and eval's batch_size=4 multiplied that by 4x. Perplexity
# is batch-size invariant (it's a token-count-weighted global average), so
# this only costs some eval wall-clock time on the two models that didn't
# need it -- it does not change any result.
EVAL_BATCH_SIZE = 1

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

RESULTS_DIR = "results"
DATA_DIR = "data"
LOG_DIR = "logs"

VRAM_LIMIT_MIB = 6140

# Note on SmoothQuant: see smoothquant_manual.py docstring. We implement it
# ourselves (architecture-agnostic, smooths each Linear's own input directly)
# rather than using llmcompressor's SmoothQuantModifier, whose LN-fold mapping
# assumes a pre-norm block and silently corrupts post-norm architectures like
# facebook/opt-350m (do_layer_norm_before=False) instead of erroring.
