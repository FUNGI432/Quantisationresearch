"""Aggregate results.json across seeds: mean +/- std, and paired significance
tests between QAT weights_only and the FP16 fine-tuned control (Section D).

Note: proper per-example paired testing (Wilcoxon on per-example NLL) requires
per-example losses, not just aggregate PPL. This script does both:
  1. A quick aggregate summary (mean/std PPL and VRAM per config, N=len(SEEDS)).
  2. If per-example NLL arrays are present in results.json (see --save-per-example
     in evaluate_perplexity callers), a Wilcoxon signed-rank test between paired
     configs. Otherwise it falls back to a Welch's t-test on the aggregate seed
     values with a printed caveat about the low N.
"""

import argparse
import json
import os

import numpy as np
from scipy import stats as sstats

from config import MATRIX_MODELS, RESULTS_DIR


def load(model_key: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{model_key}.json")
    with open(path) as f:
        return json.load(f)


def summarize(model_key: str):
    data = load(model_key)
    print(f"\n=== {model_key} ===")
    rows = []
    for key, entries in data.items():
        if isinstance(entries, dict):
            entries = [entries]
        entries = [e for e in entries if not e.get("skipped")]
        if not entries:
            continue
        ppls = [e["perplexity"] for e in entries if "perplexity" in e]
        vrams = [e.get("peak_train_vram_mib") or e.get("peak_vram_mib") for e in entries]
        vrams = [v for v in vrams if v is not None]
        if not ppls:
            continue
        row = {
            "config": key,
            "n": len(ppls),
            "ppl_mean": float(np.mean(ppls)),
            "ppl_std": float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0,
            "vram_mean": float(np.mean(vrams)) if vrams else None,
        }
        rows.append(row)
        std_str = f" +/- {row['ppl_std']:.3f}" if row["n"] > 1 else " (single run)"
        print(f"  {key:28s} PPL = {row['ppl_mean']:.3f}{std_str}   n={row['n']}   VRAM~{row['vram_mean']}")
    return rows


def significance_test(model_key: str, config_a: str, config_b: str):
    data = load(model_key)

    def ppls(key):
        entries = data.get(key, [])
        if isinstance(entries, dict):
            entries = [entries]
        return [e["perplexity"] for e in entries if "perplexity" in e and not e.get("skipped")]

    a, b = ppls(config_a), ppls(config_b)
    if len(a) < 2 or len(b) < 2:
        print(f"  Not enough seeds for a real test ({config_a}: n={len(a)}, {config_b}: n={len(b)}). "
              f"Run more seeds first.")
        return

    t_stat, p_val = sstats.ttest_ind(a, b, equal_var=False)
    print(f"  Welch's t-test {config_a} vs {config_b}: t={t_stat:.3f}, p={p_val:.4f}  "
          f"(CAVEAT: n={len(a)}/{len(b)} seeds is a small sample -- report as directional, "
          f"and prefer a per-example paired test on NLL if available)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MATRIX_MODELS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    models = list(MATRIX_MODELS.keys()) if args.model == "all" else [args.model]
    for m in models:
        try:
            summarize(m)
            significance_test(m, "qat_weights_only", "fp16_finetuned_control")
        except FileNotFoundError:
            print(f"\n=== {m} === no results yet")
