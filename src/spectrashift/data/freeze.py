from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from shapely import from_wkb

from .geometry import keep_outside_buffer
from .labels import CANONICAL_LABELS


DEVELOPMENT_LABEL_PARTITIONS = {"D", "V"}
def _select(
    frame: pd.DataFrame,
    partition: str,
    count: int,
    exclusions: list,
    buffer_m: float,
) -> pd.DataFrame:
    subset = frame[
        frame["partition"].eq(partition)
        & frame["band_complete"]
        & frame["quality_accepted"]
    ].sort_values("candidate_rank")
    geometries = [from_wkb(value) for value in subset["geometry_wkb_3035"]]
    keep = keep_outside_buffer(geometries, exclusions, buffer_m)
    selected = subset.loc[keep].head(count).copy()
    if len(selected) != count:
        raise RuntimeError(
            f"Partition {partition} has {len(selected)} eligible candidates after geometry checks; needs {count}"
        )
    return selected


def _manifest_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["partition", "patch_id"])
    digest = hashlib.sha256()
    for row in ordered.itertuples():
        digest.update(str(row.partition).encode())
        digest.update(b"\0")
        digest.update(str(row.patch_id).encode())
        digest.update(b"\0")
        digest.update(bytes(row.geometry_wkb_3035))
        digest.update(b"\n")
    return digest.hexdigest()


def _support(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in CANONICAL_LABELS:
        positives = int(frame["labels"].map(lambda values: label in values).sum())
        rows.append({"label": label, "positives": positives, "negatives": len(frame) - positives})
    return pd.DataFrame(rows)


def freeze_split(config_path: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    staging = config["staging"]
    staged_path = Path(staging["output_dir"]) / "candidates_staged.parquet"
    candidates = pd.read_parquet(staged_path)
    candidates["quality_accepted"] = (
        candidates["invalid_fraction_max_core"] <= float(config["quality"]["maximum_invalid_fraction"])
    )
    if candidates["geometry_wkb_3035"].isna().any():
        raise ValueError("Candidate geometry is incomplete")
    caps = {name: int(value) for name, value in config["split"]["caps"].items()}
    buffer_m = float(config["geometry"]["exclusion_buffer_m"])
    selected: dict[str, pd.DataFrame] = {}
    selected["T-FI"] = _select(candidates, "T-FI", caps["T-FI"], [], buffer_m)
    selected["T-PT"] = _select(candidates, "T-PT", caps["T-PT"], [], buffer_m)
    target_geometries = [
        from_wkb(value)
        for name in ("T-FI", "T-PT")
        for value in selected[name]["geometry_wkb_3035"]
    ]
    selected["I"] = _select(candidates, "I", caps["I"], target_geometries, buffer_m)
    i_geometries = [from_wkb(value) for value in selected["I"]["geometry_wkb_3035"]]
    selected["V"] = _select(
        candidates, "V", caps["V"], target_geometries + i_geometries, buffer_m
    )
    evaluation_geometries = target_geometries + i_geometries + [
        from_wkb(value) for value in selected["V"]["geometry_wkb_3035"]
    ]
    selected["U"] = _select(candidates, "U", caps["U"], evaluation_geometries, buffer_m)
    selected["D"] = _select(candidates, "D", caps["D"], evaluation_geometries, buffer_m)
    final = pd.concat([selected[name] for name in ("U", "D", "V", "I", "T-FI", "T-PT")])
    if final["patch_id"].duplicated().any() or final["location_key"].duplicated().any():
        raise ValueError("Frozen manifest contains duplicate patch IDs or locations")
    visible = final.copy()
    hidden = ~visible["partition"].isin(DEVELOPMENT_LABEL_PARTITIONS)
    visible.loc[hidden, "labels"] = None
    output_dir = Path(staging["final_manifest_dir"])
    report_dir = Path(staging["report_dir"])
    for directory in (output_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    visible.to_parquet(output_dir / "partitions.parquet", index=False)
    support = _support(selected["D"])
    support.to_csv(report_dir / "source_label_support.csv", index=False)
    min_positive = int(config["split"]["class_min_positives"])
    min_negative = int(config["split"]["class_min_negatives"])
    supported = support[(support["positives"] >= min_positive) & (support["negatives"] >= min_negative)]
    block_counts = final.groupby("partition")["block_12km"].nunique().to_dict()
    minimum_blocks = int(config["geometry"]["minimum_evaluation_blocks"])
    evaluation_block_gate = all(block_counts[name] >= minimum_blocks for name in ("V", "I", "T-FI", "T-PT"))
    summary = {
        "status": "frozen" if evaluation_block_gate else "blocked",
        "split_name": config["split"]["name"],
        "manifest_sha256": _manifest_hash(visible),
        "partition_counts": final.groupby("partition").size().to_dict(),
        "block_counts": block_counts,
        "supported_classes": len(supported),
        "support_gate": len(supported) >= int(config["split"]["minimum_supported_classes"]),
        "evaluation_block_gate": evaluation_block_gate,
        "evaluation_labels_loaded": False,
        "training_approved": bool(
            evaluation_block_gate
            and len(supported) >= int(config["split"]["minimum_supported_classes"])
        ),
    }
    (report_dir / "freeze_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
