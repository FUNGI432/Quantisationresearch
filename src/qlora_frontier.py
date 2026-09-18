"""Section B: memory-frontier study.

1. Attempt full QAT on the >=1.1B frontier model and document the VRAM failure
   quantitatively against the validated ~10*P(GB) rule from the matrix models.
2. Run QLoRA (4-bit frozen base + 16-bit LoRA adapters) as the method that
   *does* fit, on the same model/dataset/step budget, for a direct comparison.
"""

import argparse
import gc
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from common import append_result, evaluate_perplexity, get_dataloader, load_tokenized_dataset, set_seed
from config import BATCH_SIZE, EVAL_BATCH_SIZE, EVAL_SUBSET_SIZE, FRONTIER_MODEL, GRAD_ACCUM_STEPS, LEARNING_RATE, TRAIN_STEPS
from fakequant import inject_fake_quant

# See train_qat.py's docstring: without this, progress prints don't appear in
# real time when stdout isn't a TTY (e.g. piped through `tail`) -- this
# script's full_qat_attempt has no per-step logging at all, so this matters
# even more here: it's the difference between "still loading" and "silently
# hung" being distinguishable at all from the log alone.
sys.stdout.reconfigure(line_buffering=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_KEY = "frontier"


def predicted_qat_vram_gb() -> float:
    return 10 * FRONTIER_MODEL["params_b"]


def attempt_full_qat(steps: int = TRAIN_STEPS):
    """Expected to OOM (or come very close) -- that failure IS the result."""
    print(f"\n== Attempting full QAT on {FRONTIER_MODEL['hf_id']} ({FRONTIER_MODEL['params_b']}B params) ==")
    print(f"   Predicted fixed VRAM (10*P rule): {predicted_qat_vram_gb():.2f} GB  (limit: 6.14 GB)")

    ds = load_tokenized_dataset(MODEL_KEY)
    dataloader = get_dataloader(ds, batch_size=BATCH_SIZE, shuffle=True)

    result = {"predicted_fixed_vram_gb": predicted_qat_vram_gb(), "outcome": None}
    try:
        model = AutoModelForCausalLM.from_pretrained(FRONTIER_MODEL["hf_id"], dtype=torch.float32).to(DEVICE)
        inject_fake_quant(model, "both", DEVICE)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)

        torch.cuda.reset_peak_memory_stats(DEVICE)
        model.train()
        data_iter = iter(dataloader)
        print(f"   VRAM after setup: {torch.cuda.memory_allocated(DEVICE) / (1024 ** 2):.1f}MiB")
        for step in range(min(steps, 20)):  # short probe -- OOM shows up fast if it's going to
            optimizer.zero_grad()
            for micro in range(GRAD_ACCUM_STEPS):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                (loss / GRAD_ACCUM_STEPS).backward()
                print(f"   step {step + 1}/{min(steps, 20)} microbatch {micro + 1}/{GRAD_ACCUM_STEPS}: "
                      f"loss={loss.item():.4f}  VRAM={torch.cuda.memory_allocated(DEVICE) / (1024 ** 2):.1f}MiB")
            optimizer.step()

        peak = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
        result.update({"outcome": "fit", "actual_peak_vram_gb": peak, "steps_completed": min(steps, 20)})
        print(f"   Unexpectedly fit: peak {peak:.2f} GB (formula may be conservative for this arch)")
        del model, optimizer
    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
        result.update({"outcome": "oom", "peak_vram_before_oom_gb": peak, "error": str(e)[:300]})
        print(f"   OOM as predicted. Peak before OOM: {peak:.2f} GB")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    append_result(MODEL_KEY, "full_qat_frontier_attempt", result)
    return result


def run_qlora(steps: int = TRAIN_STEPS, lora_r: int = 16):
    print(f"\n== QLoRA on {FRONTIER_MODEL['hf_id']} ==")
    set_seed(42)
    ds = load_tokenized_dataset(MODEL_KEY)
    dataloader = get_dataloader(ds, batch_size=BATCH_SIZE, shuffle=True)
    # Same fixed, non-shuffled EVAL_SUBSET_SIZE subset train_qat.py uses for
    # every Section-A run (config.py) -- NOT a held-out split (train_qat.py's
    # own eval subset overlaps the shuffled training data too), but using the
    # same fixed subset size keeps this directly comparable to Section A's
    # numbers and avoids the ~10k-example, ~20+ minute full-set eval this
    # originally ran (mirroring the exact reasoning in config.EVAL_SUBSET_SIZE).
    eval_loader = get_dataloader(ds.select(range(min(EVAL_SUBSET_SIZE, len(ds)))), batch_size=EVAL_BATCH_SIZE)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(FRONTIER_MODEL["hf_id"], quantization_config=bnb_config, device_map="auto")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    model.train()
    data_iter = iter(dataloader)
    start = time.time()

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
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            (loss / GRAD_ACCUM_STEPS).backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step + 1}/{steps}")

    train_time = time.time() - start
    peak_train_vram = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

    eval_result = evaluate_perplexity(model, eval_loader, DEVICE, desc="qlora")
    eval_result.update({
        "trainable_params": trainable,
        "total_params": total,
        "train_time_s": train_time,
        "peak_train_vram_mib": peak_train_vram,
        "lora_r": lora_r,
        "steps": steps,
    })
    append_result(MODEL_KEY, "qlora", eval_result)
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full_qat_attempt", "qlora", "both"], default="both")
    args = parser.parse_args()
    if args.mode in ("full_qat_attempt", "both"):
        attempt_full_qat()
    if args.mode in ("qlora", "both"):
        run_qlora()
