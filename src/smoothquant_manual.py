"""Hand-rolled SmoothQuant (Xiao et al. 2023), implemented directly from the
published formula, generalized to be architecture-agnostic.

Why not use llmcompressor's SmoothQuantModifier, and why not the textbook
"fold into the preceding LayerNorm" implementation:

  1. llmcompressor 0.13.0's SmoothQuantModifier produces catastrophically
     broken perplexity (tens of thousands) on OPT-350M even in FP32 with
     correct, manually-specified layer mappings -- isolated by testing
     smoothing alone (no quantization). Root cause found below.
  2. The textbook SmoothQuant implementation folds the inverse smoothing
     scale into the affine parameters of the LayerNorm immediately preceding
     the target Linear -- valid only for pre-norm architectures where that
     LayerNorm's output IS the Linear's input. facebook/opt-350m sets
     `do_layer_norm_before=False`: self_attn_layer_norm runs AFTER
     attention+residual and never feeds q/k/v_proj; q/k/v_proj instead
     consume the raw residual stream directly. There is no LayerNorm output
     to fold into for this checkpoint, so the "find the preceding LN by name
     prefix" mapping (used both by llmcompressor and an earlier version of
     this script) is architecturally invalid for OPT-350M specifically --
     it silently produces a mapping that looks plausible by naming
     convention but scales the wrong tensor, corrupting the model. This is
     the actual explanation for (1), not a library bug per se.

Generalization used here: rather than requiring an upstream affine layer to
fold into, smooth each target Linear's OWN input directly via an explicit
per-input-channel division inserted right before that Linear (the SmoothQuant
paper's Appendix notes this fallback for the case with no adjacent
normalization to fold into). This is architecture-agnostic -- it works
identically for pre-norm and post-norm blocks -- at the cost of one small
elementwise multiply per forward pass instead of a free fold.

  1. Calibrate: per target Linear, hook its *own* input and record
     per-channel max(|activation|) over a calibration set.
  2. w_max_j = max(|W[:, j]|) (per input channel j, that Linear's own weight).
  3. s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha).
  4. Permanently fold s into the Linear's weight (W[:, j] *= s_j) and store
     1/s as a runtime buffer applied to the input: y = (x / s) @ (W*s)^T,
     which equals x @ W^T exactly in exact arithmetic -- unchanged FP32
     output, only the quantizer's view of the dynamic range changes.
  5. Then apply standard W8A8 fake-quantization (reusing fakequant.py's
     FakeQuantLinear machinery) and evaluate zero-shot -- the PTQ SmoothQuant
     baseline (one-shot calibration + quantization, no gradient training).
"""

import argparse

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from common import append_result, evaluate_perplexity, get_dataloader, load_tokenized_dataset, set_seed
from fakequant import _make_activation_fake_quant, _make_weight_fake_quant
from prepare_data import resolve_hf_id

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.5
EXCLUDE_SUBSTRINGS = ["lm_head", "embed_out"]


class SmoothedFakeQuantLinear(nn.Module):
    """Linear with a permanently-folded SmoothQuant weight scale, an explicit
    per-input-channel input division (the smoothing step itself), and
    optional W8A8 fake-quantization on top."""

    def __init__(self, linear: nn.Linear, smooth_scale: torch.Tensor, device):
        super().__init__()
        with torch.no_grad():
            linear.weight.mul_(smooth_scale.unsqueeze(0))  # fold s into weight (per input-channel column)
        self.linear = linear
        self.register_buffer("inv_scale", (1.0 / smooth_scale).to(device))
        self.weight_fake_quant = _make_weight_fake_quant().to(device)
        self.act_fake_quant = _make_activation_fake_quant().to(device)

    def forward(self, x):
        x = x * self.inv_scale
        weight = self.weight_fake_quant(self.linear.weight)
        out = nn.functional.linear(x, weight, self.linear.bias)
        return self.act_fake_quant(out)


@torch.no_grad()
def calibrate_activations(model, target_names, calibration_loader, max_batches: int = 8):
    absmax = {n: None for n in target_names}
    hooks = []

    def make_hook(name):
        def hook(module, inp):
            x = inp[0].detach()
            cur = x.abs().amax(dim=tuple(range(x.dim() - 1)))
            prev = absmax[name]
            absmax[name] = cur if prev is None else torch.maximum(prev, cur)
        return hook

    for name in target_names:
        hooks.append(model.get_submodule(name).register_forward_pre_hook(make_hook(name)))

    model.eval()
    for i, batch in enumerate(calibration_loader):
        if i >= max_batches:
            break
        model(input_ids=batch["input_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE))

    for h in hooks:
        h.remove()
    return absmax


def apply_smoothquant_w8a8(model, calibration_loader, alpha: float = ALPHA, max_batches: int = 8):
    target_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)
                    and not any(s in n for s in EXCLUDE_SUBSTRINGS)]

    act_absmax = calibrate_activations(model, target_names, calibration_loader, max_batches)

    n_smoothed = 0
    for name in target_names:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        original = getattr(parent, attr)

        x_max = act_absmax[name].clamp_min(1e-5)
        w_max = original.weight.detach().abs().amax(dim=0).clamp_min(1e-5)  # per input-channel
        scale = (x_max.pow(alpha) / w_max.pow(1 - alpha)).clamp_min(1e-5)

        wrapped = SmoothedFakeQuantLinear(original, scale, DEVICE)
        setattr(parent, attr, wrapped)
        n_smoothed += 1

    print(f"  [smoothquant_manual] smoothed + W8A8-wrapped {n_smoothed} Linear layers (alpha={alpha})")
    return n_smoothed


def run(model_key: str, max_calibration_batches: int = 8, max_eval_batches: int = 0, seed: int = 42):
    set_seed(seed)
    hf_id = resolve_hf_id(model_key)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32).to(DEVICE)

    ds = load_tokenized_dataset(model_key)
    calib_ds = ds.select(range(min(max_calibration_batches * 8, len(ds))))
    calib_loader = get_dataloader(calib_ds, batch_size=8)

    apply_smoothquant_w8a8(model, calib_loader, max_batches=max_calibration_batches)

    # Standard PTQ protocol for the FakeQuantize submodules: observer ON /
    # fake_quant OFF while calibrating scale/zero_point, then freeze and flip
    # for eval. FakeQuantize defaults BOTH on simultaneously, which for a
    # zero-shot PTQ pass (no training to adapt through it) would quantize with
    # the default scale=1.0 on the very first forward pass and corrupt the
    # calibration stats themselves (train_qat.py wants both on from step 1
    # since the model trains through the noise; a calibrate-only pass does not).
    from torch.ao.quantization import disable_fake_quant, disable_observer, enable_fake_quant, enable_observer

    model.apply(disable_fake_quant)
    model.apply(enable_observer)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if i >= max_calibration_batches:
                break
            model(input_ids=batch["input_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE))
    model.apply(disable_observer)
    model.apply(enable_fake_quant)

    eval_ds = ds if max_eval_batches == 0 else ds.select(range(min(max_eval_batches * 4, len(ds))))
    eval_loader = get_dataloader(eval_ds, batch_size=4)
    result = evaluate_perplexity(model, eval_loader, DEVICE, desc="smoothquant_manual_w8a8")
    append_result(model_key, "smoothquant_w8a8_manual", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-calibration-batches", type=int, default=8)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.model, args.max_calibration_batches, args.max_eval_batches, args.seed)
