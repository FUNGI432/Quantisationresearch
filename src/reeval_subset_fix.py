"""One-off fix: re-evaluate OPT-350M runs that were originally scored on the
full 10,000-example set before config.EVAL_SUBSET_SIZE existed, so every run
in results/opt-350m.json is scored on the SAME held-out subset as everything
else (Pythia-410M was entirely trained after the fix, so it never had this
problem). No retraining -- loads the already-saved checkpoint and only
re-runs evaluation. See docs/reports/2026-08-26_session-3.md for why this
was needed.
"""

import json
import os

import torch

from common import append_result, evaluate_perplexity, get_dataloader, load_tokenized_dataset, run_metadata
from config import EVAL_SUBSET_SIZE, RESULTS_DIR
from eval_downstream import load_trained_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_KEY = "opt-350m"

# (strategy, seed) pairs that were scored on the full 10k set instead of the subset
AFFECTED = [
    ("none", 42),
    ("none", 1337),
    ("none", 2024),
    ("weights_only", 42),
    ("weights_only", 1337),
]


def main():
    ds = load_tokenized_dataset(MODEL_KEY)
    eval_loader = get_dataloader(ds.select(range(min(EVAL_SUBSET_SIZE, len(ds)))), batch_size=4)

    path = os.path.join(RESULTS_DIR, f"{MODEL_KEY}.json")
    with open(path) as f:
        data = json.load(f)

    for strategy, seed in AFFECTED:
        key = "fp16_finetuned_control" if strategy == "none" else f"qat_{strategy}"
        entries = data[key]
        idx = next(i for i, e in enumerate(entries) if e["seed"] == seed)
        old = entries[idx]
        print(f"\n{key} / seed={seed}: old PPL={old['perplexity']:.4f} (n_tokens={old['n_tokens']}, full set)")

        model = load_trained_checkpoint(MODEL_KEY, strategy, seed)
        result = evaluate_perplexity(model, eval_loader, DEVICE, desc=f"reeval_{strategy}_seed{seed}")
        print(f"  new PPL={result['perplexity']:.4f} (n_tokens={result['n_tokens']}, subset -- consistent with rest of table)")

        # Preserve training-side fields (they were never wrong -- only eval-side fields change)
        updated = dict(old)
        updated.update(result)
        updated["reevaluated_for_eval_set_consistency"] = True
        updated.update(run_metadata())
        entries[idx] = updated

        del model
        torch.cuda.empty_cache()

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
