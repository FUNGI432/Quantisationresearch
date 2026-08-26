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
# killed (to free the GPU for something urgent) and continued later without
# losing more than a few minutes of progress. Also saved immediately on
# Ctrl+C / SIGTERM regardless of this interval. See train_qat.py.
CHECKPOINT_EVERY_STEPS = 50
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
MAX_SEQ_LEN = 512
LEARNING_RATE = 1e-5
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
