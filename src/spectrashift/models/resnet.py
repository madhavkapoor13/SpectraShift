from __future__ import annotations

from pathlib import Path

import torch


def build_resnet18(input_channels: int, outputs: int | None = 19):
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Identity() if outputs is None else nn.Linear(model.fc.in_features, outputs)
    return model


def build_imagenet_resnet18_core10(weights_path: str | Path, outputs: int = 19):
    """Build the declared M1 ten-band ImageNet initialization.

    RGB kernels map to core10 B04/B03/B02 indices 2/1/0. Remaining
    channels receive the RGB mean, and the complete stem is scaled by 3/10.
    The classifier is always newly initialized.
    """
    from torchvision.models import resnet18

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "conv1.weight" not in state:
        raise ValueError("ImageNet checkpoint lacks torchvision ResNet-18 conv1 weights")
    rgb = state["conv1.weight"]
    if tuple(rgb.shape) != (64, 3, 7, 7):
        raise ValueError(f"Unexpected ImageNet conv1 shape: {tuple(rgb.shape)}")
    state = {key: value for key, value in state.items() if not key.startswith("fc.")}
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, outputs)
    model.conv1 = torch.nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
    converted = rgb.mean(dim=1, keepdim=True).repeat(1, 10, 1, 1)
    converted[:, 2] = rgb[:, 0]
    converted[:, 1] = rgb[:, 1]
    converted[:, 0] = rgb[:, 2]
    state["conv1.weight"] = converted * (3.0 / 10.0)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise ValueError(f"Invalid ImageNet conversion: missing={missing}, unexpected={unexpected}")
    return model


def build_imagenet_resnet18_rgb(weights_path: str | Path, outputs: int = 19):
    """Build M1RGB with the original three-channel ImageNet stem unchanged."""
    from torchvision.models import resnet18

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    rgb = state.get("conv1.weight") if isinstance(state, dict) else None
    if not isinstance(rgb, torch.Tensor) or tuple(rgb.shape) != (64, 3, 7, 7):
        raise ValueError("ImageNet checkpoint lacks the standard ResNet-18 RGB stem")
    state = {key: value for key, value in state.items() if not key.startswith("fc.")}
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, outputs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise ValueError(f"Invalid ImageNet RGB initialization: missing={missing}, unexpected={unexpected}")
    return model


def build_resnet18_from_encoder_export(
    export_path: str | Path,
    expected_model_id: str,
    expected_seed: int,
    outputs: int = 19,
):
    payload = torch.load(export_path, map_location="cpu", weights_only=False)
    if payload.get("model_id") != expected_model_id or int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError("Encoder export identity does not match the downstream run")
    adapter = str(payload.get("adapter"))
    channels = 3 if adapter == "rgb" else 10 if adapter == "core10" else -1
    if channels < 0:
        raise ValueError(f"Unsupported Week 4 encoder adapter: {adapter}")
    model = build_resnet18(channels, outputs=outputs)
    missing, unexpected = model.load_state_dict(payload["encoder"], strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise ValueError(f"Invalid encoder export: missing={missing}, unexpected={unexpected}")
    return model, payload
