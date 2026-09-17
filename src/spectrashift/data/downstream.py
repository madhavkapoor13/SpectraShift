from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .labels import CANONICAL_LABELS


WEEK5_SEEDS = (17, 29, 43)
FRACTION_COUNTS = {
    "01": 120,
    "05": 600,
    "10": 1_200,
    "25": 3_000,
    "50": 6_000,
    "100": 12_000,
}
UNSUPPORTED_SOURCE_CLASSES = {
    "Agro-forestry areas",
    "Beaches, dunes, sands",
    "Coastal wetlands",
}


def encode_label_matrix(values: pd.Series) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(CANONICAL_LABELS)}
    result = np.zeros((len(values), len(CANONICAL_LABELS)), dtype=np.float32)
    for row, labels in enumerate(values):
        for label in labels:
            if str(label) not in lookup:
                raise ValueError(f"Unknown downstream label: {label}")
            result[row, lookup[str(label)]] = 1.0
    return result


def supported_source_indices(label_matrix: np.ndarray) -> list[int]:
    positives = label_matrix.sum(axis=0)
    negatives = len(label_matrix) - positives
    result = [
        index for index, label in enumerate(CANONICAL_LABELS)
        if positives[index] >= 50 and negatives[index] >= 50
    ]
    names = {CANONICAL_LABELS[index] for index in result}
    expected = set(CANONICAL_LABELS) - UNSUPPORTED_SOURCE_CLASSES
    if names != expected:
        raise ValueError(
            "Frozen C_source differs from the declared 16-class support contract: "
            f"observed={sorted(names)}"
        )
    return result


def nested_balanced_order(frame: pd.DataFrame, seed: int, batch_size: int = 64) -> np.ndarray:
    """Return a deterministic nested ordering balanced across labels and MGRS tiles.

    The sampler greedily fills label and tile deficits for each declared prefix. It
    recomputes deficits after small batches to keep the 12k-patch construction fast
    while retaining the intended iterative behavior.
    """
    required = {"patch_id", "labels", "mgrs_tile"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"D manifest lacks downstream fields: {sorted(missing)}")
    if len(frame) != FRACTION_COUNTS["100"]:
        raise ValueError(f"Expected 12,000 D patches, found {len(frame)}")
    canonical = frame.sort_values("patch_id", kind="stable").reset_index(drop=True)
    labels = encode_label_matrix(canonical["labels"])
    tile_names = sorted(canonical["mgrs_tile"].astype(str).unique())
    tile_lookup = {name: index for index, name in enumerate(tile_names)}
    tile_index = canonical["mgrs_tile"].astype(str).map(tile_lookup).to_numpy(dtype=np.int64)
    total_labels = labels.sum(axis=0)
    total_tiles = np.bincount(tile_index, minlength=len(tile_names)).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    jitter = rng.random(len(canonical)) * 1e-9
    remaining = np.ones(len(canonical), dtype=bool)
    selected: list[int] = []
    label_counts = np.zeros(labels.shape[1], dtype=np.float64)
    tile_counts = np.zeros(len(tile_names), dtype=np.float64)

    for target in list(FRACTION_COUNTS.values())[:-1]:
        target_labels = total_labels * (target / len(canonical))
        target_tiles = total_tiles * (target / len(canonical))
        while len(selected) < target:
            candidates = np.flatnonzero(remaining)
            label_deficit = np.maximum(target_labels - label_counts, 0.0)
            tile_deficit = np.maximum(target_tiles - tile_counts, 0.0)
            label_weight = label_deficit / np.maximum(target_labels, 1.0)
            tile_weight = tile_deficit / np.maximum(target_tiles, 1.0)
            scores = labels[candidates] @ label_weight
            scores += 0.5 * tile_weight[tile_index[candidates]]
            scores += jitter[candidates]
            take = min(batch_size, target - len(selected), len(candidates))
            chosen_local = np.argpartition(scores, -take)[-take:]
            chosen = candidates[chosen_local[np.argsort(-scores[chosen_local], kind="stable")]]
            for index in chosen.tolist():
                remaining[index] = False
                selected.append(index)
                label_counts += labels[index]
                tile_counts[tile_index[index]] += 1

    tail = np.flatnonzero(remaining)
    tail = tail[np.argsort(jitter[tail], kind="stable")]
    selected.extend(tail.tolist())
    if len(selected) != len(canonical) or len(set(selected)) != len(canonical):
        raise RuntimeError("Nested downstream sampler did not produce one complete ordering")
    return canonical.iloc[selected]["patch_id"].to_numpy()


def build_subset_records(frame: pd.DataFrame, seeds: tuple[int, ...] = WEEK5_SEEDS) -> tuple[pd.DataFrame, dict[str, object]]:
    d_frame = frame[frame["partition"].eq("D")].copy()
    if d_frame["labels"].isna().any():
        raise ValueError("D labels must be visible when freezing downstream subsets")
    canonical = d_frame.sort_values("patch_id", kind="stable").reset_index(drop=True)
    label_matrix = encode_label_matrix(canonical["labels"])
    supported = supported_source_indices(label_matrix)
    by_id = canonical.set_index("patch_id", drop=False)
    rows = []
    summaries: dict[str, object] = {}
    for seed in seeds:
        order = nested_balanced_order(canonical, int(seed))
        ordered = by_id.loc[order]
        for rank, row in enumerate(ordered.itertuples(index=False)):
            rows.append({
                "patch_id": str(row.patch_id),
                "downstream_seed": int(seed),
                "subset_rank": int(rank),
                "mgrs_tile": str(row.mgrs_tile),
            })
        fraction_summary = {}
        for code, count in FRACTION_COUNTS.items():
            subset = ordered.iloc[:count]
            matrix = encode_label_matrix(subset["labels"])
            fraction_summary[code] = {
                "count": count,
                "label_positives": {
                    label: int(matrix[:, index].sum())
                    for index, label in enumerate(CANONICAL_LABELS)
                },
                "mgrs_tile_counts": {
                    str(name): int(value)
                    for name, value in subset["mgrs_tile"].value_counts().sort_index().items()
                },
            }
        summaries[str(seed)] = fraction_summary
    records = pd.DataFrame(rows).sort_values(
        ["downstream_seed", "subset_rank"], kind="stable"
    ).reset_index(drop=True)
    digest = hashlib.sha256(
        records[["patch_id", "downstream_seed", "subset_rank"]]
        .to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    summary = {
        "status": "frozen",
        "seeds": list(seeds),
        "fraction_counts": FRACTION_COUNTS,
        "records": len(records),
        "subset_manifest_sha256": digest,
        "supported_class_indices": supported,
        "supported_class_names": [CANONICAL_LABELS[index] for index in supported],
        "unsupported_class_names": sorted(UNSUPPORTED_SOURCE_CLASSES),
        "fractions": summaries,
        "evaluation_labels_loaded": False,
    }
    return records, summary


def freeze_downstream_subsets(config_path: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    data = config["data"]
    manifest_path = Path(data["manifest_path"])
    output_dir = Path(config["contracts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(manifest_path)
    hidden = ~frame["partition"].isin({"D", "V"})
    if frame.loc[hidden, "labels"].notna().any():
        raise ValueError("Evaluation labels are forbidden in the development manifest")
    records, summary = build_subset_records(frame)
    records.to_parquet(output_dir / "downstream_subsets.parquet", index=False)
    summary["manifest_file_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (output_dir / "downstream_contract.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
