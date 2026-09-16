from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from .metadata import enrich_metadata
from .split import allocate_source_groups, balanced_location_sample


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    payload = frame[columns].sort_values(columns).to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_coverage_map(
    frame: pd.DataFrame,
    allocation: dict[str, str],
    target_tiles: set[str],
    target_countries: set[str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import mgrs

    converter = mgrs.MGRS()
    rows = []
    for tile, group in frame.groupby("mgrs_tile", sort=True):
        latitude, longitude = converter.toLatLon(tile.removeprefix("T") + "5000050000")
        countries = set(group["country"])
        if tile in target_tiles:
            target = sorted(countries & target_countries)
            role = "target_" + (target[0].lower() if target else "boundary")
        else:
            role = allocation[tile]
        rows.append(
            {
                "tile": tile,
                "latitude": latitude,
                "longitude": longitude,
                "role": role,
                "patches": len(group),
            }
        )
    tile_frame = pd.DataFrame(rows)
    palette = {
        "source_train": "#426b8a",
        "source_validation": "#dfa33a",
        "source_test": "#a35b85",
        "target_finland": "#3d8f65",
        "target_portugal": "#bb4d3e",
        "target_boundary": "#777777",
    }
    labels = {
        "source_train": "Source train",
        "source_validation": "Source validation",
        "source_test": "Source ID test",
        "target_finland": "Finland target",
        "target_portugal": "Portugal target",
        "target_boundary": "Target-touching boundary",
    }
    figure, axis = plt.subplots(figsize=(10, 7))
    for role, subset in tile_frame.groupby("role", sort=True):
        axis.scatter(
            subset["longitude"],
            subset["latitude"],
            s=30 + 90 * (subset["patches"] / tile_frame["patches"].max()) ** 0.5,
            color=palette[role],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.6,
            label=labels[role],
        )
        for row in subset.itertuples():
            axis.annotate(
                row.tile.removeprefix("T"),
                (row.longitude, row.latitude),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
            )
    axis.set_title(
        "BENv2-SpectraShift draft geographic coverage\n"
        "MGRS 100 km tile centres; marker area reflects clean patch count"
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, format="svg")
    figure.savefig(output_path.with_suffix(".png"), dpi=160)
    plt.close(figure)


def _label_counts(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        positives = int(frame["labels"].map(lambda values: label in values).sum())
        rows.append({"label": label, "positives": positives, "negatives": len(frame) - positives})
    return pd.DataFrame(rows)


def run_audit(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    metadata_path = Path(config["dataset"]["metadata_path"])
    actual_md5 = file_md5(metadata_path)
    expected_md5 = config["dataset"]["expected_md5"]
    if actual_md5 != expected_md5:
        raise ValueError(f"Metadata MD5 mismatch: expected {expected_md5}, found {actual_md5}")
    frame = enrich_metadata(pd.read_parquet(metadata_path))
    targets = set(config["split"]["target_countries"])
    target_tiles = set(frame.loc[frame["country"].isin(targets), "mgrs_tile"])
    target_rows = frame[frame["country"].isin(targets)].copy()
    boundary_excluded = frame[frame["mgrs_tile"].isin(target_tiles) & ~frame["country"].isin(targets)].copy()
    source = frame[~frame["mgrs_tile"].isin(target_tiles)].copy()
    allocation = allocate_source_groups(
        source,
        config["split"]["ratios"],
        int(config["split"]["seed"]),
        int(config["split"]["trials"]),
    )
    source["draft_domain"] = source["mgrs_tile"].map(allocation.group_to_split)
    caps = config["split"]["caps"]
    candidate_caps = config["split"].get("candidate_caps", caps)
    seed = int(config["split"]["seed"])
    source_train = source[source["draft_domain"] == "source_train"].copy()
    u_candidates = balanced_location_sample(source_train, int(candidate_caps["U"]), seed + 1)
    remaining = source_train[~source_train["location_key"].isin(set(u_candidates["location_key"]))]
    d_candidates = balanced_location_sample(remaining, int(candidate_caps["D"]), seed + 2)
    v_candidates = balanced_location_sample(
        source[source["draft_domain"] == "source_validation"], int(candidate_caps["V"]), seed + 3
    )
    i_candidates = balanced_location_sample(
        source[source["draft_domain"] == "source_test"], int(candidate_caps["I"]), seed + 4
    )
    fi_candidates = balanced_location_sample(
        target_rows[target_rows["country"] == "Finland"], int(candidate_caps["T-FI"]), seed + 5
    )
    pt_candidates = balanced_location_sample(
        target_rows[target_rows["country"] == "Portugal"], int(candidate_caps["T-PT"]), seed + 6
    )
    candidates = {
        "U": u_candidates,
        "D": d_candidates,
        "V": v_candidates,
        "I": i_candidates,
        "T-FI": fi_candidates,
        "T-PT": pt_candidates,
    }
    partitions = {name: subset.iloc[: int(caps[name])].copy() for name, subset in candidates.items()}
    manifest_dir = Path(config["output"]["manifest_dir"])
    sealed_dir = Path(config["output"]["sealed_dir"])
    report_dir = Path(config["output"]["report_dir"])
    for directory in (manifest_dir, sealed_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    public_rows = []
    keep = ["patch_id", "country", "mgrs_tile", "location_key", "timestamp", "orbit", "h_order", "v_order"]
    for partition, subset in partitions.items():
        public = subset[keep].copy()
        public.insert(0, "partition", partition)
        if partition in {"D", "V"}:
            public["labels"] = subset["labels"].map(list)
        else:
            public["labels"] = None
        public_rows.append(public)
    public_manifest = pd.concat(public_rows, ignore_index=True)
    public_manifest.to_parquet(manifest_dir / "partitions.parquet", index=False)
    candidate_rows = []
    sealed_candidate_rows = []
    for partition, subset in candidates.items():
        candidate = subset[keep].copy()
        candidate.insert(0, "partition", partition)
        candidate.insert(1, "candidate_rank", range(len(candidate)))
        candidate["is_primary"] = candidate["candidate_rank"] < int(caps[partition])
        candidate["labels"] = subset["labels"].map(list) if partition in {"D", "V"} else None
        candidate_rows.append(candidate)
        if partition in {"I", "T-FI", "T-PT"}:
            sealed = subset[["patch_id", "labels"]].copy()
            sealed.insert(0, "partition", partition)
            sealed["labels"] = sealed["labels"].map(list)
            sealed_candidate_rows.append(sealed)
    candidate_manifest = pd.concat(candidate_rows, ignore_index=True)
    candidate_manifest.to_parquet(manifest_dir / "candidates.parquet", index=False)
    pd.concat(sealed_candidate_rows, ignore_index=True).to_parquet(
        sealed_dir / "evaluation_candidate_labels.parquet", index=False
    )
    d_support = _label_counts(partitions["D"], list(config["dataset"]["label_classes"]))
    d_support.to_csv(report_dir / "source_label_support.csv", index=False)
    country_tiles = (
        frame.groupby(["country", "mgrs_tile"], as_index=False)
        .size()
        .sort_values(["country", "size"], ascending=[True, False])
    )
    country_tiles.to_csv(report_dir / "country_tile_counts.csv", index=False)
    acquisition_coverage = (
        frame.assign(acquisition_month=frame["timestamp"].dt.strftime("%Y-%m"))
        .groupby(["country", "acquisition_month"], as_index=False)
        .size()
        .sort_values(["country", "acquisition_month"])
    )
    acquisition_coverage.to_csv(report_dir / "acquisition_coverage.csv", index=False)
    partition_summary = pd.DataFrame(
        [
            {
                "partition": name,
                "patches": len(subset),
                "locations": subset["location_key"].nunique(),
                "mgrs_tiles": subset["mgrs_tile"].nunique(),
                "countries": subset["country"].nunique(),
            }
            for name, subset in partitions.items()
        ]
    )
    partition_summary.to_csv(report_dir / "partition_summary.csv", index=False)
    partition_country_counts = (
        public_manifest.groupby(["partition", "country"], as_index=False)
        .size()
        .sort_values(["partition", "country"])
    )
    partition_country_counts.to_csv(report_dir / "partition_country_counts.csv", index=False)
    source_assignments = pd.DataFrame(
        sorted(allocation.group_to_split.items()), columns=["mgrs_tile", "draft_domain"]
    )
    source_assignments.to_csv(report_dir / "source_tile_assignments.csv", index=False)
    _write_coverage_map(
        frame,
        allocation.group_to_split,
        target_tiles,
        targets,
        report_dir / "coverage_map.svg",
    )
    supported = d_support[
        (d_support["positives"] >= int(config["split"]["class_min_positives"]))
        & (d_support["negatives"] >= int(config["split"]["class_min_negatives"]))
    ]
    minimum_groups = int(config["split"]["minimum_evaluation_groups"])
    evaluation_groups_ok = bool(
        partition_summary[partition_summary["partition"].isin(["V", "I", "T-FI", "T-PT"])]["mgrs_tiles"].ge(minimum_groups).all()
    )
    summary = {
        "status": "draft",
        "dataset": config["dataset"]["name"],
        "metadata_rows": len(frame),
        "metadata_md5": actual_md5,
        "countries": sorted(frame["country"].unique().tolist()),
        "mgrs_tiles": int(frame["mgrs_tile"].nunique()),
        "target_tiles_reserved": len(target_tiles),
        "boundary_rows_excluded": len(boundary_excluded),
        "source_supported_classes": len(supported),
        "source_support_gate": len(supported) >= int(config["split"]["minimum_supported_classes"]),
        "evaluation_group_gate": evaluation_groups_ok,
        "evaluation_group_resolution": (
            "Week 2 will derive fixed 12 km EPSG:3035 blocks for target-domain uncertainty; "
            "Finland and Portugal remain entirely held out."
            if not evaluation_groups_ok
            else "MGRS tile groups satisfy the provisional minimum."
        ),
        "partition_manifest_sha256": stable_frame_hash(public_manifest, ["partition", "patch_id"]),
        "limitations": [
            "Exact raster footprints and the 2.4 km buffer are not available from metadata alone.",
            "Evaluation labels were sealed mechanically and were not aggregated in this audit.",
            "The manifest is not approved for training until the Week 2 footprint audit passes.",
        ],
    }
    (report_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
