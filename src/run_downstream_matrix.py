"""Orchestrates Section C (downstream zero-shot eval) across the full
Section-A matrix: all 3 models x 4 trained configs x 3 seeds, mirroring
run_matrix.py's skip-if-done + resumable pattern so this can be safely
interrupted and continued.

All 3 seeds are run per config (not just the "best seed" option PLAN.md
allows) to stay consistent with this project's established stance against
single-seed/cherry-picked-seed evaluation (see docs/PAPER_DRAFT.md V.B).
"""

import argparse
import json
import os

from config import MATRIX_MODELS, RESULTS_DIR, SEEDS
from eval_downstream import DEFAULT_TASKS, run

TRAINED_CONFIGS = ["none", "weights_only", "activations_only", "both"]

# A smoke test (e.g. --limit 5 to verify the pipeline works) must never be
# mistaken for a real run and silently skipped -- same failure mode
# run_matrix.py's MIN_REAL_EVAL_TOKENS check was built to prevent. limit=None
# means the full task set, which is always "real" regardless of size.
MIN_REAL_LIMIT = 100


def already_done(model_key: str, strategy: str, seed: int, tasks: list) -> bool:
    path = os.path.join(RESULTS_DIR, f"{model_key}.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    entries = data.get(f"downstream_{strategy}", [])
    if isinstance(entries, dict):
        entries = [entries]
    return any(
        e.get("seed") == seed and set(e.get("tasks", [])) >= set(tasks) and not e.get("skipped")
        and (e.get("limit") is None or e.get("limit", 0) >= MIN_REAL_LIMIT)
        for e in entries
    )


def main(model_key: str, tasks: list, limit: int, force: bool = False):
    assert model_key in MATRIX_MODELS, f"{model_key} not in MATRIX_MODELS"
    for strategy in TRAINED_CONFIGS:
        for seed in SEEDS:
            if not force and already_done(model_key, strategy, seed, tasks):
                print(f"SKIP (already done): {model_key} / downstream_{strategy} / seed={seed}")
                continue
            print(f"\n{'='*60}\nDownstream eval: {model_key} / {strategy} / seed={seed}\n{'='*60}")
            run(model_key, strategy, tasks, limit, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MATRIX_MODELS.keys()))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--limit", type=int, default=500, help="subsample per task; 0 = full set")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.model, args.tasks, args.limit or None, args.force)
