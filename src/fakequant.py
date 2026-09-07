"""Selective FakeQuant injection for QAT, generalized across model architectures.

Wraps existing nn.Linear layers rather than replacing them, so the optimizer's
parameter references and state_dict stay intact. FakeQuantize submodules are
registered as children (device-aware buffers, included in state_dict). The
Straight-Through Estimator is provided natively by
torch.fake_quantize_per_channel_affine / per_tensor_affine.
"""

import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

STRATEGIES = ["none", "weights_only", "activations_only", "both"]

# Layer names to exclude from injection: output projection / LM head must stay
# high precision or generation quality collapses immediately.
EXCLUDE_SUBSTRINGS = ["lm_head", "embed_out"]


def _make_weight_fake_quant():
    return FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )()


def _make_activation_fake_quant():
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
    )()


class FakeQuantLinear(nn.Module):
    """Wraps an nn.Linear, optionally fake-quantizing its weight and/or output."""

    #: class-level toggle for the per-layer relative quantization-error logging
    #: used to empirically support the compounding-error argument in docs/MATH.md
    #: Section 3. Off by default (adds a norm() call per fake-quantized tensor).
    track_error = False
    #: populated when track_error is True: {layer_name: [rel_error, ...]}
    error_log: dict = {}

    def __init__(self, linear: nn.Linear, quantize_weight: bool, quantize_activation: bool, device, name: str = ""):
        super().__init__()
        self.linear = linear  # preserves original nn.Parameter objects
        self.quantize_weight = quantize_weight
        self.quantize_activation = quantize_activation
        self.layer_name = name
        if quantize_weight:
            self.weight_fake_quant = _make_weight_fake_quant().to(device)
        if quantize_activation:
            self.act_fake_quant = _make_activation_fake_quant().to(device)

    def _log_error(self, key: str, original: torch.Tensor, quantized: torch.Tensor):
        with torch.no_grad():
            rel_error = (quantized - original).norm() / original.norm().clamp_min(1e-8)
            # Kept as a GPU tensor rather than calling .item() here: .item() forces a
            # blocking CUDA sync, and this runs once per fake-quantized layer per
            # microbatch (168 layers x 8 grad-accum steps = 1344 syncs/step for
            # weights_only alone). Combined with training already sitting at the
            # edge of the 6GB card, those syncs turned one step into ~54 minutes
            # (vs. a normal ~45-80s) the first time a QAT strategy actually enabled
            # error tracking. Converted to floats once, in bulk, after training
            # finishes (train_qat.py) instead.
            FakeQuantLinear.error_log.setdefault(key, []).append(rel_error.detach())

    @classmethod
    def flush_error_log(cls):
        """Converts any GPU-tensor entries in error_log to plain floats in place.

        Call once per training step (not once per training run) -- deferring
        ALL the way to the end of a 500-step run would keep every logged
        tensor alive simultaneously (up to ~672,000 for weights_only, given
        PyTorch's CUDA allocator rounds even a 4-byte scalar up to its minimum
        block size), adding real VRAM pressure back on a card already sitting
        at its ceiling. Flushing every step bounds the live count to one
        step's worth (~168-336) and still lets a full step's forward/backward
        work queue up before any blocking sync happens, instead of forcing
        one every single layer-microbatch.
        """
        for values in cls.error_log.values():
            for i, v in enumerate(values):
                if isinstance(v, torch.Tensor):
                    values[i] = v.item()

    def forward(self, x):
        weight = self.linear.weight
        if self.quantize_weight:
            q_weight = self.weight_fake_quant(weight)
            if FakeQuantLinear.track_error:
                self._log_error(f"{self.layer_name}.weight", weight, q_weight)
            weight = q_weight
        out = nn.functional.linear(x, weight, self.linear.bias)
        if self.quantize_activation:
            q_out = self.act_fake_quant(out)
            if FakeQuantLinear.track_error:
                self._log_error(f"{self.layer_name}.activation", out, q_out)
            out = q_out
        return out


def _is_excluded(name: str) -> bool:
    return any(s in name for s in EXCLUDE_SUBSTRINGS)


def inject_fake_quant(model: nn.Module, strategy: str, device) -> int:
    """Selectively replaces nn.Linear layers with FakeQuantLinear in-place.

    Returns the number of layers replaced. Collects targets before mutating
    the module tree since iterating and replacing simultaneously is unsafe.
    """
    assert strategy in STRATEGIES, f"unknown strategy: {strategy}"
    if strategy == "none":
        return 0

    quantize_weight = strategy in ("weights_only", "both")
    quantize_activation = strategy in ("activations_only", "both")

    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not _is_excluded(name):
            targets.append(name)

    replaced = 0
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        original = getattr(parent, attr)
        wrapped = FakeQuantLinear(original, quantize_weight, quantize_activation, device, name=name)
        setattr(parent, attr, wrapped)
        replaced += 1

    print(f"  [inject_fake_quant] strategy='{strategy}': replaced {replaced} Linear layers "
          f"(excluded: {EXCLUDE_SUBSTRINGS})")
    return replaced
