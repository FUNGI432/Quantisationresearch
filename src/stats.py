"""Aggregate results.json across seeds: mean +/- std, and paired significance
tests between QAT weights_only and the FP16 fine-tuned control (Section D).

Two significance tests, in order of preference:
  1. Real per-example paired Wilcoxon signed-rank test, per seed, using the
     per_example_nll_path .npy files saved by evaluate_perplexity(
     return_per_example=True) since the reproducibility work on Day 5 --
     this pairs the SAME held-out example's NLL under both configs (same
     seed, same eval set), giving a real sample size (the eval subset size,
     ~1500) instead of just 3 aggregate points. Reported per seed rather than
     pooled across seeds, since pooling would mix different training runs'
     examples as if independent, which they aren't.
  2. Fallback: Welch's t-test on the 3 aggregate per-seed PPL values, for any
     run predating the per-example logging (or if a per_example_nll_path is
     missing/unreadable for some other reason) -- explicitly caveated as too
     small a sample to trust on its own.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats as sstats

from config import MATRIX_MODELS, RESULTS_DIR

# See train_qat.py's docstring: without this, progress prints don't appear in
# real time when stdout isn't a TTY (e.g. piped through `tail`).
sys.stdout.reconfigure(line_buffering=True)


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


def _entries_by_seed(data: dict, key: str) -> dict:
    entries = data.get(key, [])
    if isinstance(entries, dict):
        entries = [entries]
    return {e["seed"]: e for e in entries if "seed" in e and not e.get("skipped")}


def wilcoxon_per_seed(model_key: str, config_a: str, config_b: str, records: list = None):
    """The real test: per seed, pairs config_a's and config_b's per-example
    NLL (same held-out examples, same seed) and runs a paired Wilcoxon
    signed-rank test. Returns True if it ran for at least one seed (so the
    caller knows whether to fall back to the t-test). If `records` is
    passed, appends one dict per seed actually tested -- used by
    full_significance_report() to pool every test's p-value for a
    multiple-comparisons correction, rather than reading each in isolation."""
    data = load(model_key)
    by_seed_a = _entries_by_seed(data, config_a)
    by_seed_b = _entries_by_seed(data, config_b)
    common_seeds = sorted(set(by_seed_a) & set(by_seed_b))

    ran_any = False
    verdicts = []
    for seed in common_seeds:
        path_a = by_seed_a[seed].get("per_example_nll_path")
        path_b = by_seed_b[seed].get("per_example_nll_path")
        if not path_a or not path_b or not os.path.exists(path_a) or not os.path.exists(path_b):
            continue  # this seed predates per-example logging -- skip, not fail
        nll_a = np.load(path_a)
        nll_b = np.load(path_b)
        if nll_a.shape != nll_b.shape:
            print(f"  seed={seed}: SKIPPED -- per-example array shape mismatch ({nll_a.shape} vs {nll_b.shape}, "
                  f"likely different eval_subset_size between when these two runs were produced)")
            continue

        diff = nll_a - nll_b
        if np.allclose(diff, 0):
            print(f"  seed={seed}: identical NLL arrays -- nothing to test (are these the same run twice?)")
            continue

        stat, p_val = sstats.wilcoxon(nll_a, nll_b)
        mean_diff = float(diff.mean())
        direction = f"{config_a} {'lower NLL (better)' if mean_diff < 0 else 'higher NLL (worse)'} than {config_b}"
        print(f"  seed={seed}: Wilcoxon signed-rank on {len(nll_a)} paired examples: "
              f"stat={stat:.1f}, p={p_val:.6f}  ({direction}, mean per-example NLL diff={mean_diff:+.5f})")
        verdicts.append(p_val < 0.05)
        ran_any = True
        if records is not None:
            records.append({
                "model": model_key, "config_a": config_a, "config_b": config_b, "seed": seed,
                "n": len(nll_a), "p_val": p_val, "mean_diff": mean_diff, "direction": direction,
            })

    if ran_any:
        n_sig = sum(verdicts)
        print(f"  -> significant (p<0.05) in {n_sig}/{len(verdicts)} seeds tested")
    return ran_any


def benjamini_hochberg(records: list, alpha: float = 0.05) -> list:
    """Benjamini-Hochberg FDR correction across every per-seed test pooled
    together (PLAN.md Section D / paper outline Methodology E: "explicitly
    address multiple comparisons" rather than reading each p-value alone).
    Adds 'q_val' and 'significant_fdr' to each record in place, and returns
    the same list sorted by raw p-value (ascending)."""
    ordered = sorted(records, key=lambda r: r["p_val"])
    m = len(ordered)
    prev_q = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q = ordered[i]["p_val"] * m / rank
        prev_q = min(prev_q, q)  # step-up procedure: q-values must be monotone non-decreasing with rank
        ordered[i]["q_val"] = min(prev_q, 1.0)
    for r in ordered:
        r["significant_fdr"] = bool(r["q_val"] < alpha)
    return ordered


def full_significance_report(model_keys: list, alpha: float = 0.05):
    """Runs every (model, strategy) vs. control Wilcoxon test across the
    given models, pools all resulting per-seed p-values, and applies one
    Benjamini-Hochberg correction across the whole set -- the total number
    of tests run is stated explicitly rather than left implicit."""
    records = []
    for m in model_keys:
        for strategy_key in ("qat_weights_only", "qat_activations_only", "qat_both"):
            wilcoxon_per_seed(m, strategy_key, "fp16_finetuned_control", records=records)

    print(f"\n=== Multiple-comparisons correction (Benjamini-Hochberg FDR, alpha={alpha}) ===")
    print(f"Total per-seed Wilcoxon tests pooled: {len(records)}")
    corrected = benjamini_hochberg(records, alpha=alpha)
    n_sig_raw = sum(1 for r in corrected if r["p_val"] < alpha)
    n_sig_fdr = sum(1 for r in corrected if r["significant_fdr"])
    print(f"Significant at raw p<{alpha}: {n_sig_raw}/{len(records)}   "
          f"Significant after FDR correction: {n_sig_fdr}/{len(records)}")
    print(f"\n{'model':14s} {'config_a':22s} seed  n     p_val      q_val     fdr_sig  mean_diff  direction")
    for r in corrected:
        print(f"{r['model']:14s} {r['config_a']:22s} {r['seed']:<5d} {r['n']:<5d} "
              f"{r['p_val']:<10.6f} {r['q_val']:<9.6f} {str(r['significant_fdr']):<8s} "
              f"{r['mean_diff']:+.5f}  {r['direction']}")
    return corrected


def significance_test(model_key: str, config_a: str, config_b: str):
    print(f"\n-- {config_a} vs {config_b} --")
    if wilcoxon_per_seed(model_key, config_a, config_b):
        return  # real test ran for at least one seed -- don't also print the weaker fallback

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
    print(f"  No per-example NLL data found for these runs (predates that logging) -- "
          f"falling back to Welch's t-test on aggregate PPL: t={t_stat:.3f}, p={p_val:.4f}  "
          f"(CAVEAT: n={len(a)}/{len(b)} seeds is a small sample -- report as directional only)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MATRIX_MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--full-report", action="store_true",
                         help="pool every per-seed Wilcoxon test across the given model(s) and apply a single "
                              "Benjamini-Hochberg FDR correction, saving the result to results/significance_report.json")
    args = parser.parse_args()

    models = list(MATRIX_MODELS.keys()) if args.model == "all" else [args.model]

    if args.full_report:
        corrected = full_significance_report(models)
        out_path = os.path.join(RESULTS_DIR, "significance_report.json")
        with open(out_path, "w") as f:
            json.dump(corrected, f, indent=2)
        print(f"\nSaved -> {out_path}")
        raise SystemExit(0)

    for m in models:
        try:
            summarize(m)
            significance_test(m, "qat_weights_only", "fp16_finetuned_control")
            significance_test(m, "qat_activations_only", "fp16_finetuned_control")
            significance_test(m, "qat_both", "fp16_finetuned_control")
        except FileNotFoundError:
            print(f"\n=== {m} === no results yet")
