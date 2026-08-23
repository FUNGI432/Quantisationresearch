"""Pre-tokenize WikiText-2 for a given model's tokenizer and cache to disk.

Static tokenization (padding to max_length, fixed truncation) so training never
tokenizes on the fly -- avoids CPU-bottlenecking the laptop during QAT runs.
"""

import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer

from config import DATASET_NAME, DATASET_CONFIG, MAX_SEQ_LEN, TOKENIZE_SAMPLES, DATA_DIR, MATRIX_MODELS, FRONTIER_MODEL


def resolve_hf_id(model_key: str) -> str:
    if model_key in MATRIX_MODELS:
        return MATRIX_MODELS[model_key]["hf_id"]
    if model_key == "frontier":
        return FRONTIER_MODEL["hf_id"]
    return model_key  # allow passing a raw HF id directly


def prepare(model_key: str, max_samples: int = TOKENIZE_SAMPLES, max_length: int = MAX_SEQ_LEN):
    hf_id = resolve_hf_id(model_key)
    out_dir = os.path.join(DATA_DIR, f"tokenized_{model_key}")

    print(f"Tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Downloading {DATASET_NAME}/{DATASET_CONFIG} ...")
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)
    combined = raw["train"] if "train" in raw else list(raw.values())[0]

    def is_nonempty(example):
        text = example["text"].strip()
        return len(text) > 0 and not text.startswith("=")

    filtered = combined.filter(is_nonempty)
    if max_samples > 0:
        filtered = filtered.select(range(min(max_samples, len(filtered))))

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = filtered.map(tokenize_fn, batched=True, remove_columns=filtered.column_names)
    tokenized.save_to_disk(out_dir)

    print(f"Saved {len(tokenized)} examples x {max_length} tokens -> {out_dir}")
    print(f"Columns: {tokenized.column_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="key from MATRIX_MODELS, 'frontier', or a raw HF model id")
    parser.add_argument("--max-samples", type=int, default=TOKENIZE_SAMPLES)
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LEN)
    args = parser.parse_args()
    prepare(args.model, args.max_samples, args.max_length)
