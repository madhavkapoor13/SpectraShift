from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square covariance matrix")
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


@dataclass(frozen=True)
class VICRegTerms:
    total: torch.Tensor
    invariance: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    mean_std: torch.Tensor


class VICRegLoss(nn.Module):
    """VICReg with explicit reductions matching the SpectraShift protocol."""

    def __init__(
        self,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        variance_target: float = 1.0,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.invariance_weight = float(invariance_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.variance_target = float(variance_target)
        self.epsilon = float(epsilon)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> VICRegTerms:
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("VICReg inputs must share shape [batch, dimensions]")
        if first.shape[0] < 2:
            raise ValueError("VICReg covariance requires at least two physical samples")
        first = first.float()
        second = second.float()
        invariance = F.mse_loss(first, second)
        first_std = torch.sqrt(first.var(dim=0, unbiased=True) + self.epsilon)
        second_std = torch.sqrt(second.var(dim=0, unbiased=True) + self.epsilon)
        variance = F.relu(self.variance_target - first_std).mean() + F.relu(
            self.variance_target - second_std
        ).mean()
        first_centered = first - first.mean(dim=0)
        second_centered = second - second.mean(dim=0)
        denominator = first.shape[0] - 1
        first_covariance = first_centered.T @ first_centered / denominator
        second_covariance = second_centered.T @ second_centered / denominator
        dimensions = first.shape[1]
        covariance = (
            off_diagonal(first_covariance).square().sum()
            + off_diagonal(second_covariance).square().sum()
        ) / dimensions
        total = (
            self.invariance_weight * invariance
            + self.variance_weight * variance
            + self.covariance_weight * covariance
        )
        return VICRegTerms(
            total=total,
            invariance=invariance,
            variance=variance,
            covariance=covariance,
            mean_std=(first_std.mean() + second_std.mean()) / 2,
        )


def effective_rank(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Effective rank requires [batch, dimensions] with batch >= 2")
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values.square()
    probabilities = probabilities / probabilities.sum().clamp_min(torch.finfo(probabilities.dtype).eps)
    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).eps).log()).sum()
    return entropy.exp()
