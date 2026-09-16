from __future__ import annotations

import torch
import torch.nn as nn

from .resnet import build_resnet18


class VICRegProjector(nn.Sequential):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 256,
    ) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )


class VICRegModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        projector_hidden_dim: int = 1024,
        projector_output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = build_resnet18(input_channels, outputs=None)
        self.projector = VICRegProjector(512, projector_hidden_dim, projector_output_dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        return features, self.projector(features)

    def encoder_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.encoder.parameters())

    def projector_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.projector.parameters())
