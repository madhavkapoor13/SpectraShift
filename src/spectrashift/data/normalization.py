from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .bands import CANONICAL_BANDS
from .shards import ShardReader


def compute_normalization(config_path: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    dataset = config["dataset"]
    staging = config["staging"]
    manifest = pd.read_parquet(Path(staging["final_manifest_dir"]) / "partitions.parquet")
    source = manifest[manifest["partition"].eq("U")].sort_values("patch_id")
    reader = ShardReader(
        staging["output_dir"],
        int(staging["shard_size"]),
        int(dataset["target_height"]),
        int(dataset["target_width"]),
    )
    sums = np.zeros(len(CANONICAL_BANDS), dtype=np.float64)
    squares = np.zeros(len(CANONICAL_BANDS), dtype=np.float64)
    counts = np.zeros(len(CANONICAL_BANDS), dtype=np.int64)
    samples: list[list[np.ndarray]] = [[] for _ in CANONICAL_BANDS]
    for row in source.itertuples():
        pixels, valid = reader.read(row.storage_index)
        for band_index in range(len(CANONICAL_BANDS)):
            values = pixels[band_index][valid[band_index]].astype(np.float64)
            sums[band_index] += values.sum()
            squares[band_index] += np.square(values).sum()
            counts[band_index] += len(values)
            if len(values):
                step = max(1, len(values) // 32)
                samples[band_index].append(values[::step][:32])
    raw_mean = sums / counts
    raw_variance = np.maximum(squares / counts - np.square(raw_mean), 0)
    raw_std = np.sqrt(raw_variance)
    raw_quantiles = [
        np.quantile(np.concatenate(values), [0.01, 0.5, 0.99]).tolist() for values in samples
    ]
    staging_summary = json.loads((Path(staging["output_dir"]) / "staging_summary.json").read_text())
    effective_scale = []
    effective_offset = []
    scale_source = []
    for band, quantiles in zip(CANONICAL_BANDS, raw_quantiles, strict=True):
        observed = staging_summary["radiometry"][band]
        scale_offsets = {(float(row[0]), float(row[1])) for row in observed}
        if len(scale_offsets) != 1:
            raise ValueError(f"Inconsistent GeoTIFF scale/offset metadata for {band}: {scale_offsets}")
        scale, offset = next(iter(scale_offsets))
        if scale == 1.0 and offset == 0.0 and quantiles[2] > 2:
            scale = 1e-4
            scale_source.append("BigEarthNet integer reflectance fallback")
        else:
            scale_source.append("GeoTIFF metadata")
        effective_scale.append(scale)
        effective_offset.append(offset)
    effective_scale = np.asarray(effective_scale, dtype=np.float64)
    effective_offset = np.asarray(effective_offset, dtype=np.float64)
    mean = raw_mean * effective_scale + effective_offset
    std = raw_std * effective_scale
    if np.any(std <= 0) or np.any(np.array([q[2] for q in raw_quantiles]) * effective_scale > 2.0):
        raise ValueError("Radiometry review failed: implausible normalized statistics")
    stats = {
        "source_partition": "U",
        "source_patches": len(source),
        "bands": list(CANONICAL_BANDS),
        "count": counts.tolist(),
        "raw_mean": raw_mean.tolist(),
        "raw_std": raw_std.tolist(),
        "raw_quantiles_01_50_99": raw_quantiles,
        "effective_scale": effective_scale.tolist(),
        "effective_offset": effective_offset.tolist(),
        "scale_source": scale_source,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    payload = json.dumps(stats, sort_keys=True).encode()
    stats["sha256"] = hashlib.sha256(payload).hexdigest()
    output = Path(staging["final_manifest_dir"]) / "normalization.json"
    output.write_text(json.dumps(stats, indent=2) + "\n")
    return stats
