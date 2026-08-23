"""Orchestrates the full Section-A run for one model: all trained configs x all seeds.

Run once per model in MATRIX_MODELS. Safe to interrupt and resume -- append_result
just adds to the JSON list, and results already present aren't re-run unless
--force is passed (checked by count, so a partial 3-seed run resumes correctly).
"""

import argparse
import json
import os

from config import MATRIX_MODELS, RESULTS_DIR, SEEDS, TRAIN_STEPS

# The per-seed matrix eval uses config.EVAL_SUBSET_SIZE (1,500 examples ->
# ~200k tokens after padding); every smoke test run during development used a
# tiny subset (tens of examples, a few thousand tokens). This threshold sits
# well below the real per-seed figure so it only catches smoke tests, not
# legitimate variation in real runs. (Some already-completed runs above used
# the full 10k-example set from before this was introduced -- those have far
# more tokens than this floor, so they still correctly count as done.)
MIN_REAL_EVAL_TOKENS = 100_000
from fakequant import STRATEGIES
from train_qat import train_one

TRAINED_CONFIGS = ["none", "weights_only", "activations_only", "both"]


def already_done(model_key: str, strategy: str, seed: int) -> bool:
    """Only counts a config as done if it was a REAL full run (correct step
    count and evaluated against the full tokenized set) -- a short smoke-test
    run (e.g. --steps 5 during development) must never be mistaken for the
    real thing and silently skipped. This was a real bug: a 5-step debug
    checkpoint from an earlier smoke test caused a genuine 500-step run to be
    skipped because the naive check only looked for "a seed entry exists".
    """
    path = os.path.join(RESULTS_DIR, f"{model_key}.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    key = "fp16_finetuned_control" if strategy == "none" else f"qat_{strategy}"
    entries = data.get(key, [])
    if isinstance(entries, dict):
        entries = [entries]
    return any(
        e.get("seed") == seed and e.get("steps") == TRAIN_STEPS and e.get("n_tokens", 0) >= MIN_REAL_EVAL_TOKENS
        for e in entries
    )


def main(model_key: str, force: bool = False):
    assert model_key in MATRIX_MODELS, f"{model_key} not in MATRIX_MODELS"
    for strategy in TRAINED_CONFIGS:
        for seed in SEEDS:
            if not force and already_done(model_key, strategy, seed):
                print(f"SKIP (already done): {model_key} / {strategy} / seed={seed}")
                continue
            train_one(model_key, strategy, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MATRIX_MODELS.keys()))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.model, args.force)
