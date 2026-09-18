from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from .bands import BAND_ADAPTERS
from .labels import CANONICAL_LABELS
from .rgb import SpectraShiftImageNetRGBDataset
from .shards import ShardReader
from .supervised import supervised_spatial_transform
from .views import deterministic_view_seed


OLMO_BAND_ORDER = tuple(BAND_ADAPTERS["olmo12"])


class DINOv2RGBDataset:
    """Frozen U-percentile RGB preprocessing followed by a 126 px resize."""

    def __init__(
        self,
        manifest_path: str | Path,
        staged_root: str | Path,
        normalization_path: str | Path,
        rgb_contract_path: str | Path,
        partition: str,
        shard_size: int = 512,
        height: int = 120,
        width: int = 120,
        output_size: int = 126,
    ) -> None:
        self.base = SpectraShiftImageNetRGBDataset(
            manifest_path, staged_root, normalization_path, rgb_contract_path,
            partition, shard_size, height, width,
        )
        self.frame = self.base.frame
        self.partition = partition
        self.output_size = int(output_size)
        if self.output_size != 126 or self.output_size % 14:
            raise ValueError("DINOv2 ViT-S/14 requires the frozen 126 px input")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = dict(self.base[index])
        image = torch.from_numpy(item["image"]).unsqueeze(0)
        image = functional.interpolate(
            image, size=(self.output_size, self.output_size), mode="bilinear",
            align_corners=False, antialias=True,
        )[0]
        item["image"] = image.numpy().astype(np.float32)
        return item


class OlmoEarthS2Dataset:
    """Official OlmoEarth 12-band input in raw Sentinel-2 L2A DN units."""

    def __init__(
        self,
        manifest_path: str | Path,
        staged_root: str | Path,
        olmo_contract_path: str | Path,
        partition: str,
        shard_size: int = 512,
        height: int = 120,
        width: int = 120,
    ) -> None:
        manifest = pd.read_parquet(manifest_path)
        self.frame = manifest[manifest["partition"].eq(partition)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Partition has no rows: {partition}")
        if "timestamp" not in self.frame:
            raise ValueError("OlmoEarth requires real acquisition timestamps")
        self.partition = partition
        self.reader = ShardReader(staged_root, shard_size, height, width)
        contract = json.loads(Path(olmo_contract_path).read_text())
        if contract.get("status") != "frozen" or contract.get("raw_units") != "sentinel2-l2a-dn":
            raise ValueError("OlmoEarth requires its frozen raw-DN normalization contract")
        if tuple(contract.get("band_order", ())) != OLMO_BAND_ORDER:
            raise ValueError("OlmoEarth band order differs from the official model order")
        means = np.asarray(contract["normalizer_mean"], dtype=np.float32)
        stds = np.asarray(contract["normalizer_std"], dtype=np.float32)
        multiplier = float(contract["std_multiplier"])
        if means.shape != (12,) or stds.shape != (12,) or np.any(stds <= 0):
            raise ValueError("Invalid OlmoEarth normalizer statistics")
        self.fill = means[:, None, None]
        self.low = (means - multiplier * stds)[:, None, None]
        self.high = (means + multiplier * stds)[:, None, None]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        pixels, valid = self.reader.read(int(row["storage_index"]))
        raw = pixels.astype(np.float32)
        valid = valid.astype(bool)
        raw = np.where(valid, raw, self.fill)
        normalized = (raw - self.low) / (self.high - self.low)
        timestamp = pd.Timestamp(row["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        labels = row.get("labels") if self.partition in {"D", "V"} else None
        return {
            "image": normalized.astype(np.float32),
            "valid": valid,
            "timestamp": np.asarray(
                [timestamp.day, timestamp.month - 1, timestamp.year], dtype=np.int64
            ),
            "labels": labels,
            "patch_id": str(row["patch_id"]),
        }


class FoundationDownstreamDataset:
    """Subset and deterministic augmentation wrapper for foundation inputs."""

    def __init__(
        self,
        base,
        patch_ids: list[str] | None,
        seed: int,
        training: bool,
    ) -> None:
        self.base = base
        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0
        if patch_ids is None:
            self.indices = list(range(len(base)))
        else:
            lookup = {str(value): index for index, value in enumerate(base.frame["patch_id"])}
            missing = set(map(str, patch_ids)) - set(lookup)
            if missing:
                raise ValueError(f"Subset contains unknown D patch IDs: {sorted(missing)[:3]}")
            self.indices = [lookup[str(value)] for value in patch_ids]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.base[self.indices[index]]
        if item["labels"] is None:
            raise ValueError("Foundation downstream loader requires visible D/V labels")
        image = torch.from_numpy(item["image"])
        valid = torch.from_numpy(item["valid"].astype(np.uint8)).bool()
        if self.training:
            view_seed = deterministic_view_seed(
                self.seed, self.epoch, str(item["patch_id"])
            )
            image = supervised_spatial_transform(image, view_seed)
            valid = supervised_spatial_transform(valid, view_seed)
        label_lookup = {name: position for position, name in enumerate(CANONICAL_LABELS)}
        labels = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float32)
        for value in item["labels"]:
            labels[label_lookup[str(value)]] = 1.0
        return {
            "image": image,
            "valid": valid,
            "timestamp": torch.as_tensor(
                item.get("timestamp", np.asarray([1, 0, 2000])), dtype=torch.long
            ),
            "labels": labels,
            "patch_id": str(item["patch_id"]),
        }
