from __future__ import annotations


def build_resnet18(input_channels: int, outputs: int | None = 19):
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Identity() if outputs is None else nn.Linear(model.fc.in_features, outputs)
    return model
