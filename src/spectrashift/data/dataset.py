from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bands import BAND_ADAPTERS, CANONICAL_BANDS, adapt_bands, band_indices
from .shards import ShardReader


class SpectraShiftDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        staged_root: str | Path,
        normalization_path: str | Path,
        partition: str,
        adapter: str,
        shard_size: int = 512,
        height: int = 120,
        width: int = 120,
    ) -> None:
        manifest = pd.read_parquet(manifest_path)
        self.frame = manifest[manifest["partition"].eq(partition)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Partition has no rows: {partition}")
        self.partition = partition
        self.adapter = adapter
        self.reader = ShardReader(staged_root, shard_size, height, width)
        stats = json.loads(Path(normalization_path).read_text())
        indices = band_indices(BAND_ADAPTERS[adapter])
        self.scale = np.asarray(stats["effective_scale"], dtype=np.float32)[list(indices), None, None]
        self.offset = np.asarray(stats["effective_offset"], dtype=np.float32)[list(indices), None, None]
        self.mean = np.asarray(stats["mean"], dtype=np.float32)[list(indices), None, None]
        self.std = np.asarray(stats["std"], dtype=np.float32)[list(indices), None, None]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        pixels, valid = self.reader.read(int(row["storage_index"]))
        pixels = adapt_bands(pixels, self.adapter).astype(np.float32)
        valid = adapt_bands(valid, self.adapter)
        reflectance = pixels * self.scale + self.offset
        reflectance = np.where(valid, reflectance, self.mean)
        normalized = (reflectance - self.mean) / self.std
        labels = row.get("labels")
        if self.partition not in {"D", "V"}:
            labels = None
        return {
            "image": normalized.astype(np.float32),
            "labels": labels,
            "patch_id": row["patch_id"],
            "valid": valid,
        }

