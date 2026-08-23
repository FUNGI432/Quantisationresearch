"""Shared utilities: dataset loading, perplexity evaluation, VRAM tracking, result logging."""

import json
import os
import random
import time

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from config import DATA_DIR, RESULTS_DIR


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
def evaluate_perplexity(model, dataloader, device, desc: str = "eval"):
    """Global (token-count-weighted) perplexity so padding doesn't bias short batches."""
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    total_nll = 0.0
    total_tokens = 0
    start = time.time()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_nll += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    elapsed = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    ppl = float(np.exp(total_nll / total_tokens))
    print(f"  [{desc}] PPL={ppl:.4f}  peak_vram={peak_vram:.1f}MiB  time={elapsed:.1f}s")
    return {"perplexity": ppl, "peak_vram_mib": peak_vram, "eval_time_s": elapsed, "n_tokens": total_tokens}


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
