from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset

from spectrashift.data.bands import BAND_ADAPTERS
from spectrashift.data.dataset import SpectraShiftDataset
from spectrashift.data.labels import CANONICAL_LABELS
from spectrashift.eval.metrics import multilabel_metrics
from spectrashift.models.vicreg import VICRegModel

from .common import atomic_torch_save, file_sha256, seed_everything, select_device


def encode_labels(values) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(CANONICAL_LABELS)}
    result = np.zeros(len(CANONICAL_LABELS), dtype=np.float32)
    for value in values:
        result[lookup[str(value)]] = 1
    return result


class _FeatureInput(Dataset):
    def __init__(self, base: SpectraShiftDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        if item["labels"] is None:
            raise ValueError("Linear probe requires visible D/V labels")
        return torch.from_numpy(item["image"]), torch.from_numpy(encode_labels(item["labels"]))


def _extract(encoder, base, device, batch_size: int, num_workers: int):
    loader = DataLoader(
        _FeatureInput(base), batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features, targets = [], []
    encoder.eval()
    with torch.inference_mode():
        for images, labels in loader:
            features.append(encoder(images.to(device, non_blocking=True)).cpu())
            targets.append(labels)
    return torch.cat(features), torch.cat(targets)


def probe_ssl(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_override: str | Path | None = None,
) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    probe = config["probe"]
    data = config["data"]
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = select_device(bool(config["training"].get("require_cuda", False)))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_channels = len(BAND_ADAPTERS[data["adapter"]])
    model = VICRegModel(
        input_channels,
        int(config["model"].get("projector_hidden_dim", 1024)),
        int(config["model"].get("projector_output_dim", 256)),
    )
    model.load_state_dict(checkpoint["model"])
    encoder = model.encoder.to(device)
    bases = {
        partition: SpectraShiftDataset(
            data["manifest_path"], data["staged_root"], data["normalization_path"],
            partition, data["adapter"], int(data.get("shard_size", 512)),
            int(data.get("height", 120)), int(data.get("width", 120)),
        )
        for partition in ("D", "V")
    }
    train_features, train_targets = _extract(
        encoder, bases["D"], device, int(probe["extraction_batch_size"]), int(probe["num_workers"])
    )
    validation_features, validation_targets = _extract(
        encoder, bases["V"], device, int(probe["extraction_batch_size"]), int(probe["num_workers"])
    )
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_features = (train_features - mean) / std
    validation_features = (validation_features - mean) / std
    head = torch.nn.Linear(train_features.shape[1], len(CANONICAL_LABELS)).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(probe["learning_rate"]), weight_decay=float(probe["weight_decay"])
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_features, train_targets), batch_size=int(probe["batch_size"]),
        shuffle=True, generator=generator,
    )
    best_map = float("-inf")
    best_epoch = -1
    best_state = None
    started = time.perf_counter()
    for epoch in range(int(probe["epochs"])):
        head.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(features.to(device))
            loss = loss_function(logits, targets.to(device))
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            probabilities = head(validation_features.to(device)).sigmoid().cpu().numpy()
        metrics = multilabel_metrics(validation_targets.numpy(), probabilities)
        validation_map = float(metrics["macro_average_precision"])
        if validation_map > best_map:
            best_map = validation_map
            best_epoch = epoch
            best_state = {name: value.detach().cpu() for name, value in head.state_dict().items()}
    output_dir = Path(output_override) if output_override else Path(config["run"]["output_dir"]) / "probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / "linear_probe.pt"
    atomic_torch_save({
        "head": best_state, "feature_mean": mean, "feature_std": std,
        "checkpoint_sha256": file_sha256(checkpoint_path), "best_epoch": best_epoch,
        "validation_map": best_map,
    }, head_path)
    result = {
        "run_id": config["run"]["id"],
        "fitted_on": "D",
        "selected_on": "V",
        "train_patches": len(bases["D"]),
        "validation_patches": len(bases["V"]),
        "feature_dimension": train_features.shape[1],
        "best_epoch": best_epoch,
        "validation_macro_average_precision": best_map,
        "elapsed_seconds": time.perf_counter() - started,
        "head_path": str(head_path),
        "head_sha256": file_sha256(head_path),
    }
    (output_dir / "probe_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
