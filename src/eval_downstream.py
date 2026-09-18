"""Section C: zero-shot downstream evaluation via lm-eval-harness.

Loads a trained checkpoint saved by train_qat.py, wraps it in lm_eval's HFLM
adapter, and runs zero-shot tasks against it. Subsampled task sets keep
runtime bounded on laptop hardware -- see --limit.
"""

import argparse
import json
import os

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import append_result, run_metadata
from fakequant import STRATEGIES, inject_fake_quant
from prepare_data import resolve_hf_id
from train_qat import checkpoint_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_TASKS = ["lambada_openai", "piqa", "hellaswag"]


def load_trained_checkpoint(model_key: str, strategy: str, seed: int):
    """Reconstructs the architecture, injects FakeQuant to match the trained
    structure, then loads the actual fine-tuned weights saved by train_qat.py.
    Evaluating a freshly initialized base model here would silently measure
    the wrong thing -- it must be the exact checkpoint the perplexity numbers
    came from.
    """
    ckpt_path = checkpoint_path(model_key, strategy, seed)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run train_qat.py --model {model_key} "
            f"--strategy {strategy} --seed {seed} first."
        )
    hf_id = resolve_hf_id(model_key)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float16).to(DEVICE)
    inject_fake_quant(model, strategy, DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"  WARNING: state_dict mismatch -- missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def run(model_key: str, strategy: str, tasks: list, limit: int, seed: int):
    tokenizer = AutoTokenizer.from_pretrained(resolve_hf_id(model_key))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_trained_checkpoint(model_key, strategy, seed)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=str(DEVICE), batch_size=4)

    results = simple_evaluate(model=lm, tasks=tasks, limit=limit, random_seed=seed, numpy_random_seed=seed)

    metrics = {task: res for task, res in results["results"].items()}
    payload = {
        "seed": seed,
        "strategy": strategy,
        "tasks": tasks,
        "limit": limit,
        "metrics": metrics,
        **run_metadata(),
    }
    key = f"downstream_{strategy}"
    append_result(model_key, key, payload)
    del model
    torch.cuda.empty_cache()
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--limit", type=int, default=500, help="subsample per task; 0 = full set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.model, args.strategy, args.tasks, args.limit or None, args.seed)
