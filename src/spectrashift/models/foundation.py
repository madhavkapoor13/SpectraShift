from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for candidate in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = candidate.relative_to(root).as_posix()
        if relative.startswith(".git/") or "__pycache__" in candidate.parts:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class DINOv2Classifier(torch.nn.Module):
    """DINOv2 ViT-S/14 with global mean patch-token pooling."""

    feature_dimension = 384

    def __init__(self, encoder: torch.nn.Module, outputs: int = 19) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(self.feature_dimension, outputs)

    def features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder.forward_features(batch["image"])
        if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
            raise ValueError("DINOv2 encoder did not return normalized patch tokens")
        tokens = output["x_norm_patchtokens"]
        if tokens.ndim != 3 or tokens.shape[1:] != (81, self.feature_dimension):
            raise ValueError(f"Unexpected DINOv2 patch-token shape: {tuple(tokens.shape)}")
        return tokens.mean(dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.features(batch))


class OlmoEarthClassifier(torch.nn.Module):
    """OlmoEarth v1.1 Tiny encoder with validity-aware token pooling."""

    feature_dimension = 192

    def __init__(self, encoder: torch.nn.Module, sample_type, outputs: int = 19) -> None:
        super().__init__()
        self.encoder = encoder
        self.sample_type = sample_type
        self.head = torch.nn.Linear(self.feature_dimension, outputs)

    @staticmethod
    def pool_tokens(tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        flat_tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        flat_masks = masks.reshape(masks.shape[0], -1)
        available = flat_masks.eq(0)
        counts = available.sum(dim=1, keepdim=True)
        if (counts == 0).any():
            raise ValueError("OlmoEarth sample has no valid spatial tokens")
        return (flat_tokens * available.unsqueeze(-1)).sum(dim=1) / counts

    def features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["image"].permute(0, 2, 3, 1).unsqueeze(3)
        valid = batch["valid"].permute(0, 2, 3, 1).unsqueeze(3)
        mask = torch.where(
            valid,
            torch.zeros((), dtype=torch.long, device=valid.device),
            torch.full((), 3, dtype=torch.long, device=valid.device),
        )
        sample = self.sample_type(
            timestamps=batch["timestamp"].unsqueeze(1),
            sentinel2_l2a=image,
            sentinel2_l2a_mask=mask,
        )
        output = self.encoder(sample, patch_size=8, input_res=10, fast_pass=False)
        tokens_and_masks = output["tokens_and_masks"]
        return self.pool_tokens(
            tokens_and_masks.sentinel2_l2a,
            tokens_and_masks.sentinel2_l2a_mask,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.features(batch))


def build_dinov2_classifier(
    source_dir: str | Path,
    weights_path: str | Path,
    expected_source_sha256: str,
    expected_weights_sha256: str,
    outputs: int = 19,
) -> DINOv2Classifier:
    source_dir, weights_path = Path(source_dir), Path(weights_path)
    if _file_sha256(weights_path) != expected_weights_sha256:
        raise ValueError("DINOv2 weight hash differs from the frozen contract")
    if not (source_dir / "hubconf.py").is_file():
        raise FileNotFoundError("Pinned DINOv2 source does not contain hubconf.py")
    if _directory_sha256(source_dir) != expected_source_sha256:
        raise ValueError("DINOv2 source tree differs from the frozen contract")
    encoder = torch.hub.load(
        str(source_dir), "dinov2_vits14", source="local", pretrained=False,
        trust_repo=True,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    return DINOv2Classifier(encoder, outputs)


def build_olmoearth_classifier(
    source_dir: str | Path,
    model_dir: str | Path,
    expected_source_sha256: str,
    expected_config_sha256: str,
    expected_weights_sha256: str,
    outputs: int = 19,
) -> OlmoEarthClassifier:
    source_dir, model_dir = Path(source_dir), Path(model_dir)
    if _directory_sha256(source_dir) != expected_source_sha256:
        raise ValueError("OlmoEarth minimal source differs from the frozen contract")
    if _file_sha256(model_dir / "config.json") != expected_config_sha256:
        raise ValueError("OlmoEarth configuration hash differs from the frozen contract")
    if _file_sha256(model_dir / "weights.pth") != expected_weights_sha256:
        raise ValueError("OlmoEarth weights hash differs from the frozen contract")
    sys.path.insert(0, str(source_dir))
    try:
        from olmoearth_pretrain_minimal import load_model_from_path
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )

        pretrained = load_model_from_path(model_dir, load_weights=True)
    finally:
        if sys.path and sys.path[0] == str(source_dir):
            sys.path.pop(0)
    return OlmoEarthClassifier(pretrained.encoder, MaskedOlmoEarthSample, outputs)
