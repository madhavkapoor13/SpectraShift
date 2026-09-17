from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bands import BAND_ADAPTERS, band_indices
from .shards import ShardReader


IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


def histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    """Return NumPy's linear quantile exactly from an integer histogram."""
    histogram = np.asarray(histogram, dtype=np.int64)
    if histogram.ndim != 1 or histogram.sum() <= 0:
        raise ValueError("Histogram must contain at least one observation")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("Quantile must be in [0, 1]")
    count = int(histogram.sum())
    position = float(quantile) * (count - 1)
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    cumulative = np.cumsum(histogram, dtype=np.int64)
    lower = int(np.searchsorted(cumulative, lower_rank + 1, side="left"))
    upper = int(np.searchsorted(cumulative, upper_rank + 1, side="left"))
    return float(lower + (position - lower_rank) * (upper - lower))


def compute_rgb_percentile_contract(
    manifest_path: str | Path,
    staged_root: str | Path,
    normalization_path: str | Path,
    manifest_contract_sha256: str,
    normalization_sha256: str,
    shard_size: int = 512,
    height: int = 120,
    width: int = 120,
) -> dict[str, object]:
    """Fit exact B04/B03/B02 p02/p98 values from valid U pixels only."""
    manifest = pd.read_parquet(manifest_path)
    hidden = ~manifest["partition"].isin({"D", "V"})
    if manifest.loc[hidden, "labels"].notna().any():
        raise ValueError("Evaluation labels are forbidden while fitting the RGB contract")
    u_rows = manifest[manifest["partition"].eq("U")].sort_values("patch_id", kind="stable")
    if len(u_rows) != 20_000:
        raise ValueError(f"Expected 20,000 U patches, found {len(u_rows)}")

    indices = band_indices(BAND_ADAPTERS["rgb"])
    histograms = np.zeros((3, 65_536), dtype=np.int64)
    staged_root = Path(staged_root)
    storage_indices = u_rows["storage_index"].to_numpy(dtype=np.int64)
    shard_indices = storage_indices // int(shard_size)
    for shard_index in sorted(np.unique(shard_indices)):
        rows = storage_indices[shard_indices == shard_index] % int(shard_size)
        pixels = np.load(staged_root / f"shard-{shard_index:05d}-pixels.npy", mmap_mode="r")
        packed = np.load(staged_root / f"shard-{shard_index:05d}-validity.npy", mmap_mode="r")
        selected_pixels = np.asarray(pixels[rows][:, list(indices)])
        selected_packed = np.asarray(packed[rows][:, list(indices)])
        valid = np.unpackbits(
            selected_packed, axis=2, count=height * width, bitorder="little"
        ).reshape(len(rows), 3, height, width).astype(bool)
        for output_band in range(3):
            values = selected_pixels[:, output_band][valid[:, output_band]]
            if values.size:
                histograms[output_band] += np.bincount(values, minlength=65_536)

    normalization = json.loads(Path(normalization_path).read_text())
    if normalization.get("sha256") != normalization_sha256:
        raise ValueError("Week 2 normalization hash differs from the frozen contract")
    scale = np.asarray(normalization["effective_scale"], dtype=np.float64)[list(indices)]
    offset = np.asarray(normalization["effective_offset"], dtype=np.float64)[list(indices)]
    means = np.asarray(normalization["mean"], dtype=np.float64)[list(indices)]
    raw_low = np.asarray([histogram_quantile(row, 0.02) for row in histograms])
    raw_high = np.asarray([histogram_quantile(row, 0.98) for row in histograms])
    reflectance_low = raw_low * scale + offset
    reflectance_high = raw_high * scale + offset
    if np.any(reflectance_high <= reflectance_low):
        raise ValueError("RGB percentile ranges must be strictly positive")

    return {
        "status": "frozen",
        "method": "exact-streaming-uint16-histogram-linear-quantile",
        "fit_partition": "U",
        "fit_patch_count": int(len(u_rows)),
        "band_order": list(BAND_ADAPTERS["rgb"]),
        "quantiles": [0.02, 0.98],
        "valid_pixel_counts": [int(row.sum()) for row in histograms],
        "raw_uint16_percentile_02": raw_low.tolist(),
        "raw_uint16_percentile_98": raw_high.tolist(),
        "reflectance_percentile_02": reflectance_low.tolist(),
        "reflectance_percentile_98": reflectance_high.tolist(),
        "invalid_fill_u_mean": means.tolist(),
        "effective_scale": scale.tolist(),
        "effective_offset": offset.tolist(),
        "imagenet_mean": list(IMAGENET_RGB_MEAN),
        "imagenet_std": list(IMAGENET_RGB_STD),
        "manifest_contract_sha256": manifest_contract_sha256,
        "normalization_sha256": normalization_sha256,
        "evaluation_labels_loaded": False,
    }


class SpectraShiftImageNetRGBDataset:
    """Three-band dataset implementing the frozen M1RGB preprocessing order."""

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
    ) -> None:
        manifest = pd.read_parquet(manifest_path)
        self.frame = manifest[manifest["partition"].eq(partition)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Partition has no rows: {partition}")
        self.partition = partition
        self.adapter = "rgb"
        self.reader = ShardReader(staged_root, shard_size, height, width)
        normalization = json.loads(Path(normalization_path).read_text())
        contract = json.loads(Path(rgb_contract_path).read_text())
        if contract.get("status") != "frozen" or contract.get("fit_partition") != "U":
            raise ValueError("M1RGB requires a frozen U-only percentile contract")
        if contract.get("band_order") != list(BAND_ADAPTERS["rgb"]):
            raise ValueError("M1RGB percentile contract has the wrong band order")
        if contract.get("normalization_sha256") != normalization.get("sha256"):
            raise ValueError("M1RGB percentile and normalization contracts disagree")
        indices = band_indices(BAND_ADAPTERS["rgb"])
        self.indices = indices
        self.scale = np.asarray(normalization["effective_scale"], dtype=np.float32)[list(indices), None, None]
        self.offset = np.asarray(normalization["effective_offset"], dtype=np.float32)[list(indices), None, None]
        self.fill = np.asarray(contract["invalid_fill_u_mean"], dtype=np.float32)[:, None, None]
        self.low = np.asarray(contract["reflectance_percentile_02"], dtype=np.float32)[:, None, None]
        self.high = np.asarray(contract["reflectance_percentile_98"], dtype=np.float32)[:, None, None]
        self.imagenet_mean = np.asarray(contract["imagenet_mean"], dtype=np.float32)[:, None, None]
        self.imagenet_std = np.asarray(contract["imagenet_std"], dtype=np.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        pixels, valid = self.reader.read(int(row["storage_index"]))
        pixels = pixels[list(self.indices)].astype(np.float32)
        valid = valid[list(self.indices)]
        reflectance = pixels * self.scale + self.offset
        reflectance = np.where(valid, reflectance, self.fill)
        unit = np.clip(reflectance, self.low, self.high)
        unit = (unit - self.low) / (self.high - self.low)
        image = (unit - self.imagenet_mean) / self.imagenet_std
        labels = row.get("labels") if self.partition in {"D", "V"} else None
        return {
            "image": image.astype(np.float32),
            "labels": labels,
            "patch_id": row["patch_id"],
            "valid": valid,
        }
