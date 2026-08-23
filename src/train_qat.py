"""Selective QAT training, generalized across the model matrix and multiple seeds.

Strategy 'none' is the FP16 fine-tuned control (isolates domain adaptation from
quantization -- see docs/PLAN.md, Section D). All strategies share identical
memory-optimization settings so peak-VRAM comparisons are apples-to-apples:
batch size 1, 8 gradient-accumulation steps, bitsandbytes AdamW8bit, gradient
checkpointing.
"""

import argparse
import gc
import os
import time

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM

from common import append_result, evaluate_perplexity, get_dataloader, load_tokenized_dataset, set_seed
from config import BATCH_SIZE, EVAL_SUBSET_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE, RESULTS_DIR, TRAIN_STEPS
from fakequant import STRATEGIES, inject_fake_quant
from prepare_data import resolve_hf_id

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"


def checkpoint_path(model_key: str, strategy: str, seed: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{model_key}_{strategy}_seed{seed}.pt")


def train_one(model_key: str, strategy: str, seed: int, steps: int = TRAIN_STEPS):
    set_seed(seed)
    hf_id = resolve_hf_id(model_key)
    print(f"\n{'=' * 60}\nModel={model_key}  strategy={strategy}  seed={seed}  steps={steps}\n{'=' * 60}")

    ds = load_tokenized_dataset(model_key)
    dataloader = get_dataloader(ds, batch_size=BATCH_SIZE, shuffle=True)
    # Fixed, non-shuffled subset shared across every strategy/seed for this
    # model -- see config.EVAL_SUBSET_SIZE for why this isn't the full 10k set.
    eval_loader = get_dataloader(ds.select(range(min(EVAL_SUBSET_SIZE, len(ds)))), batch_size=4)

    print(f"  Loading {hf_id} in FP32 for QAT ...")
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.1f}M")

    n_replaced = inject_fake_quant(model, strategy, DEVICE)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    from fakequant import FakeQuantLinear
    FakeQuantLinear.error_log = {}
    FakeQuantLinear.track_error = strategy != "none"

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)

    print(f"  VRAM after setup: {torch.cuda.memory_allocated(DEVICE) / (1024 ** 2):.1f}MiB")
    torch.cuda.reset_peak_memory_stats(DEVICE)

    model.train()
    data_iter = iter(dataloader)
    start = time.time()
    running_loss = 0.0

    for step in range(steps):
        optimizer.zero_grad()
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += loss.item()

        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"  step {step + 1}/{steps}  loss={running_loss / 50:.4f}")
            running_loss = 0.0

    train_time = time.time() - start
    peak_train_vram = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    print(f"  Training complete: {steps} steps in {train_time:.1f}s, peak VRAM {peak_train_vram:.1f}MiB")

    FakeQuantLinear.track_error = False  # don't pollute error log with eval-time noise
    weight_errors = [v for k, vs in FakeQuantLinear.error_log.items() if k.endswith(".weight") for v in vs]
    act_errors = [v for k, vs in FakeQuantLinear.error_log.items() if k.endswith(".activation") for v in vs]
    error_summary = {
        "mean_weight_rel_error": (sum(weight_errors) / len(weight_errors)) if weight_errors else None,
        "mean_activation_rel_error": (sum(act_errors) / len(act_errors)) if act_errors else None,
    }

    eval_result = evaluate_perplexity(model, eval_loader, DEVICE, desc=f"{strategy}_seed{seed}")
    eval_result.update(error_summary)
    eval_result.update({
        "strategy": strategy,
        "seed": seed,
        "n_params": n_params,
        "n_layers_replaced": n_replaced,
        "train_time_s": train_time,
        "peak_train_vram_mib": peak_train_vram,
        "steps": steps,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
    })

    key = "fp16_finetuned_control" if strategy == "none" else f"qat_{strategy}"
    append_result(model_key, key, eval_result)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = checkpoint_path(model_key, strategy, seed)
    cpu_state = {
        k: (v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu())
        for k, v in model.state_dict().items()
    }
    torch.save({"state_dict": cpu_state, "strategy": strategy, "seed": seed}, ckpt_path)
    print(f"  Saved checkpoint (FP16) -> {ckpt_path}")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="key from MATRIX_MODELS or 'frontier'")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    args = parser.parse_args()
    train_one(args.model, args.strategy, args.seed, args.steps)
