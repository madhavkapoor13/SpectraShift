from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .dataset import SpectraShiftDataset
from .labels import CANONICAL_LABELS


def validate_manifest(config: dict) -> dict[str, object]:
    staging = config["staging"]
    manifest = pd.read_parquet(Path(staging["final_manifest_dir"]) / "partitions.parquet")
    if manifest["patch_id"].duplicated().any() or manifest["location_key"].duplicated().any():
        raise ValueError("Duplicate patch IDs or locations in frozen manifest")
    hidden = manifest["partition"].isin(["U", "I", "T-FI", "T-PT"])
    if not manifest.loc[hidden, "labels"].isna().all():
        raise ValueError("A sealed partition exposes labels")
    visible = manifest["partition"].isin(["D", "V"])
    if manifest.loc[visible, "labels"].isna().any():
        raise ValueError("A development partition is missing labels")
    expected = {name: int(value) for name, value in config["split"]["caps"].items()}
    observed = manifest.groupby("partition").size().to_dict()
    if observed != expected:
        raise ValueError(f"Partition counts differ: expected {expected}, found {observed}")
    return {"partition_counts": observed, "label_isolation": True}


def _targets(values) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(CANONICAL_LABELS)}
    result = np.zeros(len(CANONICAL_LABELS), dtype=np.float32)
    for value in values:
        result[lookup[str(value)]] = 1
    return result


def run_smoke(config_path: str | Path, max_steps: int = 500) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from spectrashift.models.resnet import build_resnet18

    config = yaml.safe_load(Path(config_path).read_text())
    integrity = validate_manifest(config)
    staging = config["staging"]
    dataset = config["dataset"]
    normalization = Path(staging["final_manifest_dir"]) / "normalization.json"
    d_data = SpectraShiftDataset(
        Path(staging["final_manifest_dir"]) / "partitions.parquet",
        staging["output_dir"],
        normalization,
        "D",
        "core10",
        int(staging["shard_size"]),
        int(dataset["target_height"]),
        int(dataset["target_width"]),
    )

    class TorchView(Dataset):
        def __len__(self):
            return min(32, len(d_data))

        def __getitem__(self, index):
            item = d_data[index]
            return torch.from_numpy(item["image"]), torch.from_numpy(_targets(item["labels"]))

    loader = DataLoader(TorchView(), batch_size=min(32, len(TorchView())), shuffle=False)
    images, labels = next(iter(loader))
    if images.shape[1:] != (10, 120, 120) or not torch.isfinite(images).all():
        raise ValueError("Invalid core10 smoke batch")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        device_arch = f"sm_{major}{minor}"
        supported_arches = torch.cuda.get_arch_list()
        if supported_arches and device_arch not in supported_arches:
            raise RuntimeError(
                f"{torch.cuda.get_device_name(0)} ({device_arch}) is unsupported by this "
                f"PyTorch build ({supported_arches}); select a T4 GPU"
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(10).to(device)
    images, labels = images.to(device), labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    started = time.perf_counter()
    micro_f1 = 0.0
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            prediction = logits.sigmoid() >= 0.5
            truth = labels.bool()
            tp = (prediction & truth).sum().item()
            fp = (prediction & ~truth).sum().item()
            fn = (~prediction & truth).sum().item()
            micro_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if micro_f1 >= 0.95:
            break
    result = {
        **integrity,
        "device": str(device),
        "steps": step + 1,
        "elapsed_seconds": time.perf_counter() - started,
        "training_micro_f1": micro_f1,
        "overfit_gate": micro_f1 >= 0.95,
    }
    if not result["overfit_gate"]:
        raise RuntimeError(f"Tiny-overfit gate failed: {micro_f1:.4f}")
    report_dir = Path(staging["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "smoke_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
