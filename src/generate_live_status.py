"""Generates docs/LIVE_STATUS.md -- an auto-refreshed cross-model comparison
table, regenerated every ~10 minutes (from the same heartbeat loop that
posts training progress in chat) while the Qwen matrix is running.

Not part of the research pipeline itself -- purely a status view, built
from the same results/*.json files every other script reads/writes. Safe
to regenerate at any time; never touches training state.

Usage: python generate_live_status.py [path/to/current/training/log]
"""
import datetime
import json
import os
import statistics
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "LIVE_STATUS.md")

MODELS = ["opt-350m", "pythia-410m", "qwen2.5-0.5b"]
MODEL_LABELS = {"opt-350m": "OPT-350M", "pythia-410m": "Pythia-410M", "qwen2.5-0.5b": "Qwen2.5-0.5B"}

CONFIGS = [
    "fp16_zero_shot", "int8_dynamic", "smoothquant_w8a8_manual", "awq_int4",
    "fp16_finetuned_control", "qat_weights_only", "qat_activations_only", "qat_both",
]
CONFIG_LABELS = {
    "fp16_zero_shot": "FP16 zero-shot",
    "int8_dynamic": "INT8 PTQ",
    "smoothquant_w8a8_manual": "SmoothQuant PTQ",
    "awq_int4": "AWQ PTQ",
    "fp16_finetuned_control": "FP16 fine-tuned control",
    "qat_weights_only": "QAT weights-only",
    "qat_activations_only": "QAT activations-only",
    "qat_both": "QAT both",
}
TRAINED_CONFIGS = ["fp16_finetuned_control", "qat_weights_only", "qat_activations_only", "qat_both"]
QAT_CONFIGS = ["qat_weights_only", "qat_activations_only", "qat_both"]
SEEDS = [42, 1337, 2024]


def load_results():
    data = {}
    for m in MODELS:
        path = os.path.join(RESULTS_DIR, f"{m}.json")
        data[m] = json.load(open(path)) if os.path.exists(path) else {}
    return data


def summarize(entries):
    """Returns (display_string, list_of_ppls_or_None)."""
    if not entries:
        return "pending", None
    if entries[0].get("skipped"):
        return "skipped", None
    ppls = [e["perplexity"] for e in entries if "perplexity" in e]
    if not ppls:
        return "pending", None
    if len(ppls) == 1:
        return f"{ppls[0]:.2f}", ppls
    mean = statistics.mean(ppls)
    std = statistics.stdev(ppls)
    return f"{mean:.2f} ± {std:.2f} (n={len(ppls)})", ppls


def pct_change(control_ppls, config_ppls):
    if not control_ppls or not config_ppls:
        return None
    if len(control_ppls) == len(config_ppls):
        diffs = [(c - ctrl) / ctrl * 100 for ctrl, c in zip(control_ppls, config_ppls)]
        return statistics.mean(diffs)
    return (statistics.mean(config_ppls) - statistics.mean(control_ppls)) / statistics.mean(control_ppls) * 100


def tail_log(log_path, n=6):
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, "r", errors="ignore") as f:
        lines = f.readlines()
    return [l.rstrip("\n") for l in lines[-n:] if l.strip()]


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_results()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out = []
    out.append("# Live Cross-Model Matrix Status")
    out.append("")
    out.append(f"_Auto-generated (not hand-written) every ~10 minutes while the pipeline "
                f"runs, from the same `results/*.json` files every other script uses. "
                f"Last updated: {now}. Not a substitute for the narrative status in "
                f"`README.md` / `docs/PLAN.md` -- this is a live number dump._")
    out.append("")

    out.append("## Current live run")
    out.append("")
    tail = tail_log(log_path)
    if tail:
        out.append("```")
        out.extend(tail)
        out.append("```")
    else:
        out.append("_No active run log found._")
    out.append("")

    out.append("## QAT matrix completion (3 seeds x 4 trained configs = 12 runs per model)")
    out.append("")
    out.append("| Model | Runs complete |")
    out.append("|---|---|")
    for m in MODELS:
        count = sum(1 for c in TRAINED_CONFIGS for e in data[m].get(c, []) if "perplexity" in e)
        out.append(f"| {MODEL_LABELS[m]} | {count}/12 |")
    out.append("")

    out.append("## Cross-model comparison (perplexity, mean ± std across seeds where trained)")
    out.append("")
    out.append("| Configuration | " + " | ".join(MODEL_LABELS[m] for m in MODELS) + " |")
    out.append("|---|" + "---|" * len(MODELS))
    summaries = {m: {} for m in MODELS}
    for cfg in CONFIGS:
        row = [CONFIG_LABELS[cfg]]
        for m in MODELS:
            label, ppls = summarize(data[m].get(cfg, []))
            row.append(label)
            summaries[m][cfg] = ppls
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    out.append("## % change vs. FP16 fine-tuned control (architecture-sensitivity comparison)")
    out.append("")
    out.append("| Strategy | " + " | ".join(MODEL_LABELS[m] for m in MODELS) + " |")
    out.append("|---|" + "---|" * len(MODELS))
    for cfg in QAT_CONFIGS:
        row = [CONFIG_LABELS[cfg].replace("QAT ", "")]
        for m in MODELS:
            pct = pct_change(summaries[m].get("fp16_finetuned_control"), summaries[m].get(cfg))
            row.append(f"{pct:+.1f}%" if pct is not None else "pending")
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    out.append("## Qwen2.5-0.5B per-seed detail (the in-progress model)")
    out.append("")
    out.append("| Strategy | Seed 42 | Seed 1337 | Seed 2024 |")
    out.append("|---|---|---|---|")
    qd = data["qwen2.5-0.5b"]
    for cfg in TRAINED_CONFIGS:
        row = [CONFIG_LABELS[cfg]]
        seed_map = {e.get("seed"): e.get("perplexity") for e in qd.get(cfg, [])}
        for s in SEEDS:
            v = seed_map.get(s)
            row.append(f"{v:.2f}" if v is not None else "pending")
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"[live-status] wrote {OUT_PATH} at {now}", flush=True)


if __name__ == "__main__":
    main()
