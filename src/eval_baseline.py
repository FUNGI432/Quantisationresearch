"""FP16 zero-shot baseline + PTQ baseline family (dynamic INT8, SmoothQuant, AWQ).

Each PTQ method is attempted independently and gracefully skipped (with a note
in results.json) if its library isn't available or the model isn't supported --
this keeps the sweep runnable across a heterogeneous model roster without one
missing dependency blocking every other run.
"""

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import append_result, evaluate_perplexity, get_dataloader, load_tokenized_dataset, run_metadata, set_seed
from prepare_data import resolve_hf_id

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_fp16_zero_shot(hf_id: str, dataloader):
    print("\n== FP16 zero-shot ==")
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, device_map="auto")
    result = evaluate_perplexity(model, dataloader, DEVICE, desc="fp16_zero_shot")
    _free(model)
    return result


def run_int8_dynamic(hf_id: str, dataloader):
    print("\n== INT8 PTQ (bitsandbytes dynamic) ==")
    try:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(hf_id, quantization_config=bnb_config, device_map="auto")
        result = evaluate_perplexity(model, dataloader, DEVICE, desc="int8_dynamic")
        _free(model)
        return result
    except Exception as e:
        print(f"  SKIPPED (int8_dynamic): {e}")
        return {"skipped": True, "reason": str(e)}


def run_smoothquant(model_key: str, max_eval_batches: int):
    """Uses our own architecture-agnostic SmoothQuant implementation
    (smoothquant_manual.py) rather than llmcompressor's SmoothQuantModifier --
    see that module's docstring for why: llmcompressor's LN-fold mapping is
    invalid for post-norm architectures like OPT-350M and silently corrupts
    the model rather than erroring, which was caught by isolating smoothing
    alone (no quantization) and finding it already broke perplexity."""
    print("\n== SmoothQuant W8A8 (manual, architecture-agnostic) ==")
    try:
        from smoothquant_manual import run as run_smoothquant_manual

        return run_smoothquant_manual(model_key, max_eval_batches=max_eval_batches)
    except Exception as e:
        print(f"  SKIPPED (smoothquant): {e}")
        return {"skipped": True, "reason": str(e)}


def run_awq(hf_id: str, dataloader):
    print("\n== AWQ (autoawq) ==")
    try:
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_pretrained(hf_id)
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
        model.quantize(tokenizer, quant_config=quant_config)
        result = evaluate_perplexity(model.model.to(DEVICE), dataloader, DEVICE, desc="awq_int4")
        _free(model)
        return result
    except Exception as e:
        print(f"  SKIPPED (awq): {e}")
        return {"skipped": True, "reason": str(e)}


def main(model_key: str, max_batches: int = 0, seed: int = 42):
    set_seed(seed)
    hf_id = resolve_hf_id(model_key)
    ds = load_tokenized_dataset(model_key)
    if max_batches > 0:
        ds = ds.select(range(min(max_batches, len(ds))))
    dataloader = get_dataloader(ds, batch_size=4)

    print(f"Model: {hf_id}  |  device: {DEVICE}  |  eval examples: {len(ds)}")
    meta = run_metadata()

    append_result(model_key, "fp16_zero_shot", {**run_fp16_zero_shot(hf_id, dataloader), **meta})
    append_result(model_key, "int8_dynamic", {**run_int8_dynamic(hf_id, dataloader), **meta})
    run_smoothquant(model_key, max_batches // 4 if max_batches > 0 else 0)  # appends its own result key
    append_result(model_key, "awq_int4", {**run_awq(hf_id, dataloader), **meta})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = full eval set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.model, args.max_batches, args.seed)
