"""Backfill per-example NLL arrays for runs completed before that logging
existed (all of OPT-350M and Pythia-410M's 24 runs -- see common.py's
evaluate_perplexity(return_per_example=True), added after those matrices
finished). Reuses each run's saved FP16 checkpoint -- no retraining, just a
fresh evaluation pass on the same fixed eval subset.

Do NOT run this while another GPU-bound run is in progress (it needs the
same ~2-4GB of VRAM headroom as any other eval pass) -- run it between
matrix runs or once a model's matrix is fully paused/complete.

Sanity check included: recomputed perplexity is compared against the
stored value from the original run and flagged if they disagree by more
than a small tolerance, since a mismatch would mean the checkpoint doesn't
actually match what produced that PPL (e.g. wrong strategy/seed pairing).
"""

import argparse
import json
import os
import sys

import torch

from common import evaluate_perplexity, get_dataloader, load_tokenized_dataset, save_per_example_nll
from config import EVAL_BATCH_SIZE, EVAL_SUBSET_SIZE, RESULTS_DIR
from eval_downstream import load_trained_checkpoint

# See train_qat.py's docstring: without this, progress prints don't appear in
# real time when stdout isn't a TTY (e.g. piped through `tail`), making a
# genuinely-running process look hung.
sys.stdout.reconfigure(line_buffering=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PPL_MISMATCH_TOLERANCE = 0.01  # relative


def backfill_model(model_key: str, dry_run: bool = False):
    path = os.path.join(RESULTS_DIR, f"{model_key}.json")
    with open(path) as f:
        data = json.load(f)

    ds = load_tokenized_dataset(model_key)
    eval_loader = get_dataloader(ds.select(range(min(EVAL_SUBSET_SIZE, len(ds)))), batch_size=EVAL_BATCH_SIZE)

    trained_keys = {"fp16_finetuned_control": "none", "qat_weights_only": "weights_only",
                    "qat_activations_only": "activations_only", "qat_both": "both"}

    for key, strategy in trained_keys.items():
        entries = data.get(key, [])
        for entry in entries:
            if entry.get("per_example_nll_path") and os.path.exists(entry["per_example_nll_path"]):
                continue  # already backfilled
            seed = entry["seed"]
            print(f"{model_key} / {key} / seed={seed}: re-evaluating from checkpoint for per-example NLL ...")
            if dry_run:
                continue
            model = load_trained_checkpoint(model_key, strategy, seed)
            result = evaluate_perplexity(model, eval_loader, DEVICE, desc=f"backfill_{strategy}_seed{seed}",
                                          return_per_example=True)
            rel_diff = abs(result["perplexity"] - entry["perplexity"]) / entry["perplexity"]
            if rel_diff > PPL_MISMATCH_TOLERANCE:
                print(f"  WARNING: recomputed PPL {result['perplexity']:.4f} disagrees with stored "
                      f"{entry['perplexity']:.4f} (rel diff {rel_diff:.3%}) -- investigate before trusting this pair")
            per_example_path = save_per_example_nll(model_key, key, seed, result.pop("per_example_nll"))
            entry["per_example_nll_path"] = per_example_path
            print(f"  -> {per_example_path}")
            del model
            torch.cuda.empty_cache()

    if not dry_run:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Updated {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dry-run", action="store_true", help="list what would be backfilled without running eval")
    args = parser.parse_args()
    backfill_model(args.model, args.dry_run)
