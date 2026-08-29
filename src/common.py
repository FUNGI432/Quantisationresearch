"""Shared utilities: dataset loading, perplexity evaluation, VRAM tracking, result logging."""

import json
import os
import random
import subprocess
import time

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from config import DATA_DIR, RESULTS_DIR


def git_commit_hash() -> str:
    """Best-effort short commit hash for reproducibility -- 'unknown' (not a
    hard failure) if git isn't available or there are no commits yet, since
    this must never block a training run from completing."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return "unknown"


def run_metadata() -> dict:
    """Static settings that don't vary per-seed but must still be recorded
    for reproducibility (PLAN.md Section F / ACL reviewer critique #5) --
    previously claimed as captured but not actually written to results.json;
    this closes that gap."""
    from config import DATASET_CONFIG, DATASET_NAME, EVAL_SUBSET_SIZE, MAX_SEQ_LEN, TOKENIZE_SAMPLES

    return {
        "git_commit": git_commit_hash(),
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "max_seq_len": MAX_SEQ_LEN,
        "tokenize_samples": TOKENIZE_SAMPLES,
        "eval_subset_size": EVAL_SUBSET_SIZE,
        "optimizer": "bitsandbytes.optim.AdamW8bit",
        "gradient_checkpointing": True,
        "fakequant_weight_scheme": "per_channel_symmetric, qint8, MovingAveragePerChannelMinMaxObserver",
        "fakequant_activation_scheme": "per_tensor_affine, qint8, MovingAverageMinMaxObserver",
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenized_dataset(model_key: str):
    path = os.path.join(DATA_DIR, f"tokenized_{model_key}")
    ds = load_from_disk(path)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds


def get_dataloader(ds, batch_size: int, shuffle: bool = False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device, desc: str = "eval", return_per_example: bool = False):
    """Global (token-count-weighted) perplexity so padding doesn't bias short batches.

    return_per_example=True additionally computes one mean-NLL-per-example
    value for every example in the dataloader, in dataloader iteration order.
    This only matters (and costs anything) when the caller needs a per-example
    paired comparison -- see PLAN.md Section D / stats.py's Wilcoxon test,
    which needs the SAME example's NLL under two different configs, not just
    an aggregate perplexity. Requires the dataloader to be unshuffled (as
    every eval_loader in this codebase already is) so example index i means
    the same underlying example across two different runs/configs.

    Computed via manual reduction='none' cross-entropy rather than the
    model's internal `labels=` loss, which only returns a single batch-mean
    scalar -- not enough to recover individual examples' NLL from a
    batch_size>1 loader.
    """
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    total_nll = 0.0
    total_tokens = 0
    per_example_nll = [] if return_per_example else None
    start = time.time()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        if return_per_example:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            per_token_nll = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="none"
            ).view(shift_labels.shape)
            valid = (shift_labels != -100)
            n_tokens_per_example = valid.sum(dim=1)
            nll_sum_per_example = per_token_nll.sum(dim=1)
            for nll_sum, n_tok in zip(nll_sum_per_example.tolist(), n_tokens_per_example.tolist()):
                per_example_nll.append(nll_sum / n_tok if n_tok > 0 else float("nan"))
            n_tokens = int(n_tokens_per_example.sum().item())
            total_nll += float(nll_sum_per_example.sum().item())
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            n_tokens = (labels != -100).sum().item()
            total_nll += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    elapsed = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    ppl = float(np.exp(total_nll / total_tokens))
    print(f"  [{desc}] PPL={ppl:.4f}  peak_vram={peak_vram:.1f}MiB  time={elapsed:.1f}s")
    result = {"perplexity": ppl, "peak_vram_mib": peak_vram, "eval_time_s": elapsed, "n_tokens": total_tokens}
    if return_per_example:
        result["per_example_nll"] = per_example_nll
    return result


PER_EXAMPLE_DIR = os.path.join(RESULTS_DIR, "per_example")


def save_per_example_nll(model_key: str, config_key: str, seed: int, per_example_nll: list) -> str:
    """Saves the per-example NLL array from evaluate_perplexity(return_per_example=True)
    to a compact .npy file rather than inlining ~1500 floats into results.json.
    Returns the path, which the caller stores in the result dict as
    'per_example_nll_path' so stats.py can find and pair it with another
    config's array for the real Wilcoxon significance test."""
    os.makedirs(PER_EXAMPLE_DIR, exist_ok=True)
    path = os.path.join(PER_EXAMPLE_DIR, f"{model_key}_{config_key}_seed{seed}.npy")
    np.save(path, np.array(per_example_nll, dtype=np.float64))
    return path


def append_result(model_key: str, key: str, payload: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{model_key}.json")
    data = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    data.setdefault(key, [])
    if isinstance(data[key], dict):
        data[key] = [data[key]]  # migrate old single-run format
    data[key].append(payload)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  -> appended to {path}[{key}] (n={len(data[key])})")
