from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from spectrashift.data.bands import BAND_ADAPTERS
from spectrashift.data.dataset import SpectraShiftDataset
from spectrashift.data.foundation import DINOv2RGBDataset, OlmoEarthS2Dataset
from spectrashift.data.labels import CANONICAL_LABELS
from spectrashift.data.rgb import SpectraShiftImageNetRGBDataset
from spectrashift.eval.metrics import multilabel_metrics
from spectrashift.models.resnet import build_resnet18
from spectrashift.train.common import (
    directory_sha256,
    file_sha256,
    git_commit,
    hardware_record,
    object_sha256,
    select_device,
    source_tree_sha256,
)
from spectrashift.train.downstream import MODEL_ADAPTERS, verify_downstream_checkpoint
from spectrashift.train.foundation import _build_model as build_foundation_model
from spectrashift.train.foundation import verify_foundation_checkpoint


EVALUATION_PARTITIONS = ("I", "T-FI", "T-PT")
DOMAIN_NAMES = {"I": "I", "T-FI": "Finland", "T-PT": "Portugal"}
EXPECTED_PARTITION_COUNTS = {"I": 3000, "T-FI": 4000, "T-PT": 4000}
WEEK8_SEEDS = (17, 29, 43)
FRACTION_COUNTS = {"01": 120, "05": 600, "10": 1200, "25": 3000, "50": 6000, "100": 12000}
CONTROLLED_MODELS = ("M0", "M1", "M2", "M3", "M4")
FOUNDATION_MODELS = ("M5", "M6")
EXPECTED_RUNS_PER_SEED = 37
EXPECTED_LEDGER_ROWS = 111
EXPECTED_PREDICTION_DOMAINS = 333
HYPOTHESES = {"H1": ("M3", "M0"), "H2": ("M3", "M2"), "H3": ("M4", "M3")}


def _json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=True, default=str) + "\n")


def _canonical_table_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    payload = frame.loc[:, list(columns)].copy()
    for column in payload.columns:
        if column == "labels":
            payload[column] = payload[column].map(
                lambda values: "|".join(sorted(map(str, values))) if values is not None else ""
            )
        elif pd.api.types.is_datetime64_any_dtype(payload[column]):
            payload[column] = payload[column].map(lambda value: pd.Timestamp(value).isoformat())
    return hashlib.sha256(
        payload.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


def validate_week7_approval(
    summary_path: str | Path,
    ledger_path: str | Path,
    expected_summary_sha256: str | None = None,
    expected_ledger_sha256: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    summary_path, ledger_path = Path(summary_path), Path(ledger_path)
    if expected_summary_sha256 and file_sha256(summary_path) != expected_summary_sha256:
        raise ValueError("Week 7 summary hash differs from the frozen pre-label artifact")
    if expected_ledger_sha256 and file_sha256(ledger_path) != expected_ledger_sha256:
        raise ValueError("Week 8 checkpoint-ledger hash differs from the frozen pre-label artifact")
    summary = _json(summary_path)
    if (
        not summary.get("week7_complete")
        or not summary.get("week8_approved")
        or summary.get("week8_checkpoint_ledger_count") != EXPECTED_LEDGER_ROWS
        or summary.get("evaluation_labels_loaded") is not False
    ):
        raise ValueError("Week 7 aggregate does not approve sealed Week 8 evaluation")
    ledger = pd.read_csv(ledger_path, dtype={"fraction_code": str})
    required = {
        "family", "run_id", "model_id", "fraction_code", "seed",
        "checkpoint_sha256", "validation_macro_average_precision",
    }
    if set(ledger.columns) != required or len(ledger) != EXPECTED_LEDGER_ROWS:
        raise ValueError("Week 8 checkpoint ledger does not contain the exact 111-row schema")
    ledger["fraction_code"] = ledger["fraction_code"].str.zfill(2)
    if ledger["run_id"].duplicated().any() or ledger["checkpoint_sha256"].duplicated().any():
        raise ValueError("Week 8 checkpoint ledger contains duplicate runs or checkpoints")
    if set(ledger["seed"].astype(int)) != set(WEEK8_SEEDS):
        raise ValueError("Week 8 checkpoint ledger must cover seeds 17, 29, and 43")
    expected_per_seed = {seed: EXPECTED_RUNS_PER_SEED for seed in WEEK8_SEEDS}
    if ledger.groupby(ledger["seed"].astype(int)).size().to_dict() != expected_per_seed:
        raise ValueError("Every Week 8 seed must contain exactly 37 frozen runs")
    embedded = summary.get("week8_checkpoint_ledger")
    if not isinstance(embedded, list) or len(embedded) != EXPECTED_LEDGER_ROWS:
        raise ValueError("Week 7 summary does not embed the exact checkpoint ledger")
    embedded_by_id = {str(row["run_id"]): row for row in embedded}
    for row in ledger.itertuples(index=False):
        frozen = embedded_by_id.get(str(row.run_id))
        if frozen is None or str(frozen["checkpoint_sha256"]) != str(row.checkpoint_sha256):
            raise ValueError(f"Ledger and Week 7 summary disagree for {row.run_id}")
    return summary, ledger.sort_values("run_id", kind="stable").reset_index(drop=True)


def _encode_labels(values: Iterable[Iterable[str]]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(CANONICAL_LABELS)}
    rows = list(values)
    matrix = np.zeros((len(rows), len(CANONICAL_LABELS)), dtype=np.uint8)
    for row_index, labels in enumerate(rows):
        for label in labels:
            if str(label) not in lookup:
                raise ValueError(f"Unknown evaluation label: {label}")
            matrix[row_index, lookup[str(label)]] = 1
    return matrix


def _support_indices(matrix: np.ndarray) -> list[int]:
    positives = matrix.sum(axis=0)
    negatives = len(matrix) - positives
    return np.flatnonzero((positives > 0) & (negatives > 0)).astype(int).tolist()


def _block_ids(frame: pd.DataFrame) -> pd.Series:
    if not {"mgrs_tile", "h_order", "v_order"}.issubset(frame.columns):
        raise ValueError("Final manifest lacks MGRS coordinates required for 12 km blocks")
    # A BigEarthNet patch spans 1.2 km at 10 m, so a 10x10 index cell is 12 km.
    return (
        frame["mgrs_tile"].astype(str)
        + "-b"
        + (frame["h_order"].astype(int) // 10).astype(str)
        + "_"
        + (frame["v_order"].astype(int) // 10).astype(str)
    )


def freeze_week8_evaluation(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    paths, frozen = config["paths"], config["frozen"]
    week7, ledger = validate_week7_approval(
        paths["week7_summary_path"], paths["checkpoint_ledger_path"],
        frozen["week7_summary_sha256"], frozen["checkpoint_ledger_sha256"],
    )
    manifest_path = Path(paths["manifest_path"])
    normalization_path = Path(paths["normalization_path"])
    freeze_path = Path(paths["freeze_summary_path"])
    week5_contracts_path = Path(paths["week5_contracts_path"])
    manifest = pd.read_parquet(manifest_path)
    required_manifest = {
        "partition", "patch_id", "country", "mgrs_tile", "timestamp", "h_order", "v_order", "labels"
    }
    if not required_manifest.issubset(manifest.columns):
        raise ValueError(f"Final manifest lacks columns: {sorted(required_manifest - set(manifest.columns))}")
    public_eval = manifest[manifest["partition"].isin(EVALUATION_PARTITIONS)].copy()
    if public_eval["labels"].notna().any():
        raise ValueError("Public final manifest exposes sealed evaluation labels")
    counts = public_eval.groupby("partition").size().to_dict()
    if counts != EXPECTED_PARTITION_COUNTS:
        raise ValueError(f"Final evaluation partition counts differ: {counts}")
    if public_eval["patch_id"].duplicated().any():
        raise ValueError("Final evaluation manifest contains duplicate patch IDs")

    freeze = _json(freeze_path)
    normalization = _json(normalization_path)
    week5 = _json(week5_contracts_path)
    if (
        not freeze.get("training_approved")
        or freeze.get("manifest_sha256") != frozen["manifest_contract_sha256"]
        or normalization.get("sha256") != frozen["normalization_sha256"]
        or not week5.get("week5_contracts_complete")
        or week5.get("evaluation_labels_loaded") is not False
    ):
        raise ValueError("Week 2/5 frozen data contracts are inconsistent")
    source_indices = [int(value) for value in week5["supported_class_indices"]]
    if len(source_indices) != 16:
        raise ValueError("C_source must contain exactly 16 frozen classes")

    candidate_path = Path(paths["candidate_labels_path"])
    candidates = pd.read_parquet(candidate_path)
    required_candidate = {"partition", "patch_id", "labels"}
    if set(candidates.columns) != required_candidate:
        raise ValueError("Candidate label artifact has an unexpected schema")
    if candidates.duplicated(["partition", "patch_id"]).any():
        raise ValueError("Candidate label artifact contains duplicate keys")
    if not set(candidates["partition"]).issubset(set(EVALUATION_PARTITIONS)):
        raise ValueError("Candidate label artifact contains an unexpected partition")
    id_partition = public_eval.set_index("patch_id")["partition"].astype(str).to_dict()
    candidate_final = candidates[candidates["patch_id"].isin(id_partition)].copy()
    mismatched = candidate_final[
        candidate_final.apply(lambda row: id_partition[str(row["patch_id"])] != str(row["partition"]), axis=1)
    ]
    if not mismatched.empty:
        raise ValueError("Candidate labels assign final patch IDs to the wrong partition")
    sealed = public_eval.drop(columns=["labels"]).merge(
        candidates, on=["partition", "patch_id"], how="left", validate="one_to_one"
    )
    if len(sealed) != sum(EXPECTED_PARTITION_COUNTS.values()) or sealed["labels"].isna().any():
        raise ValueError("Candidate label artifact is missing final evaluation labels")
    if sealed["patch_id"].duplicated().any():
        raise ValueError("Sealed label join produced duplicate patch IDs")
    sealed["mgrs_group"] = sealed["mgrs_tile"].astype(str)
    sealed["block_12km"] = _block_ids(sealed)
    sealed["timestamp"] = pd.to_datetime(sealed["timestamp"], utc=True)
    sealed["quarter"] = sealed["timestamp"].dt.quarter.astype(int)
    sealed = sealed.sort_values(["partition", "patch_id"], kind="stable").reset_index(drop=True)

    domain_support: dict[str, object] = {}
    common = set(range(len(CANONICAL_LABELS)))
    for partition in EVALUATION_PARTITIONS:
        subset = sealed[sealed["partition"].eq(partition)]
        indices = _support_indices(_encode_labels(subset["labels"]))
        common &= set(indices)
        domain_support[partition] = {
            "indices": indices,
            "names": [CANONICAL_LABELS[index] for index in indices],
        }
    common_indices = sorted(common)
    common_source = sorted(common.intersection(source_indices))
    support_contract = {
        "status": "frozen",
        "all_class_names": list(CANONICAL_LABELS),
        "source_supported_indices": source_indices,
        "source_supported_names": [CANONICAL_LABELS[index] for index in source_indices],
        "domain_support": domain_support,
        "common_id_ood_indices": common_indices,
        "common_id_ood_names": [CANONICAL_LABELS[index] for index in common_indices],
        "common_source_indices": common_source,
        "common_source_names": [CANONICAL_LABELS[index] for index in common_source],
        "missing_class_policy": "AP is NA when a domain has no positive or no negative examples",
    }
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "evaluation_labels.parquet"
    sealed.to_parquet(labels_path, index=False)
    support_path = output_dir / "support_contract.json"
    _write_json(support_path, support_contract)
    contract = {
        "status": "frozen",
        "model_selection_frozen_before_label_access": True,
        "week7_complete": True,
        "week8_approved": True,
        "checkpoint_ledger_count": len(ledger),
        "partition_counts": EXPECTED_PARTITION_COUNTS,
        "bootstrap_seed": int(config["uncertainty"]["bootstrap_seed"]),
        "bootstrap_replicates": int(config["uncertainty"]["bootstrap_replicates"]),
        "primary_group": "mgrs_group",
        "sensitivity_group": "block_12km",
        "hashes": {
            "week7_summary_sha256": file_sha256(paths["week7_summary_path"]),
            "checkpoint_ledger_sha256": file_sha256(paths["checkpoint_ledger_path"]),
            "manifest_file_sha256": file_sha256(manifest_path),
            "manifest_contract_sha256": frozen["manifest_contract_sha256"],
            "normalization_file_sha256": file_sha256(normalization_path),
            "normalization_sha256": frozen["normalization_sha256"],
            "candidate_labels_sha256": file_sha256(candidate_path),
            "final_labels_sha256": file_sha256(labels_path),
            "final_labels_table_sha256": _canonical_table_sha256(
                sealed, ["partition", "patch_id", "labels", "mgrs_group", "block_12km"]
            ),
            "support_contract_sha256": file_sha256(support_path),
            "source_tree_sha256": source_tree_sha256(),
            "week8_config_sha256": file_sha256(config_path),
        },
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(sealed)),
        "excluded_candidate_rows": int(len(candidates) - len(sealed)),
        "all_class_count": len(CANONICAL_LABELS),
        "source_supported_class_count": len(source_indices),
        "common_source_class_count": len(common_source),
        "evaluation_labels_loaded": True,
        "no_post_seal_selection": True,
    }
    contract_path = output_dir / "evaluation_contract.json"
    _write_json(contract_path, contract)
    summary = {
        "week8_sealing_complete": True,
        "week7_summary_sha256": file_sha256(paths["week7_summary_path"]),
        "checkpoint_ledger_sha256": file_sha256(paths["checkpoint_ledger_path"]),
        "evaluation_contract_sha256": file_sha256(contract_path),
        "support_contract_sha256": file_sha256(support_path),
        "evaluation_labels_sha256": file_sha256(labels_path),
        "partition_counts": EXPECTED_PARTITION_COUNTS,
        "checkpoint_ledger_count": len(ledger),
        "model_selection_frozen_before_label_access": True,
        "evaluation_labels_loaded": True,
    }
    _write_json(output_dir / "week8_sealing_summary.json", summary)
    return summary


class _ControlledEvaluationDataset(Dataset):
    def __init__(self, base, labels: dict[str, np.ndarray]) -> None:
        self.base, self.labels = base, labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        patch_id = str(item["patch_id"])
        return torch.from_numpy(item["image"]), torch.from_numpy(self.labels[patch_id]), patch_id


class _FoundationEvaluationDataset(Dataset):
    def __init__(self, base, labels: dict[str, np.ndarray]) -> None:
        self.base, self.labels = base, labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.base[index]
        patch_id = str(item["patch_id"])
        return {
            "image": torch.from_numpy(item["image"]),
            "valid": torch.from_numpy(item["valid"].astype(np.uint8)).bool(),
            "timestamp": torch.as_tensor(item.get("timestamp", [1, 0, 2000]), dtype=torch.long),
            "labels": torch.from_numpy(self.labels[patch_id]),
            "patch_id": patch_id,
        }


def _find_run_file(roots: Sequence[str | Path], run_id: str, name: str, digest: str | None = None) -> Path:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            path for path in Path(root).rglob(name) if path.parent.name == run_id
        )
    if digest is not None:
        candidates = [path for path in candidates if file_sha256(path) == digest]
    unique = {str(path.resolve()): path for path in candidates}
    if len(unique) != 1:
        raise ValueError(f"Expected one {name} for {run_id}, found {list(unique.values())}")
    return next(iter(unique.values()))


def _find_unique_named(roots: Sequence[str | Path], name: str) -> Path:
    candidates: dict[str, Path] = {}
    for root in roots:
        for path in Path(root).rglob(name):
            candidates.setdefault(file_sha256(path), path)
    if len(candidates) != 1:
        raise ValueError(f"Expected one unique {name}, found {list(candidates.values())}")
    return next(iter(candidates.values()))


def _load_run_artifacts(roots: Sequence[str | Path], ledger_row) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    checkpoint = _find_run_file(
        roots, str(ledger_row.run_id), "best-model.pt", str(ledger_row.checkpoint_sha256)
    )
    summary_name = "foundation_summary.json" if str(ledger_row.family) == "foundation" else "downstream_summary.json"
    summary_path = _find_run_file(roots, str(ledger_row.run_id), summary_name)
    config_path = _find_run_file(roots, str(ledger_row.run_id), "resolved_config.yaml")
    summary = _json(summary_path)
    resolved = yaml.safe_load(config_path.read_text())
    if (
        summary.get("run_id") != ledger_row.run_id
        or summary.get("best_checkpoint_sha256") != ledger_row.checkpoint_sha256
        or int(summary.get("seed", -1)) != int(ledger_row.seed)
        or str(summary.get("model_id")) != str(ledger_row.model_id)
        or str(summary.get("fraction_code")).zfill(2) != str(ledger_row.fraction_code).zfill(2)
        or not summary.get("completion_gate")
        or summary.get("evaluation_labels_loaded") is not False
    ):
        raise ValueError(f"Frozen run summary does not match the ledger for {ledger_row.run_id}")
    checkpoint_payload = (
        verify_foundation_checkpoint(checkpoint, str(ledger_row.run_id))
        if str(ledger_row.family) == "foundation"
        else verify_downstream_checkpoint(checkpoint, str(ledger_row.run_id))
    )
    if object_sha256(resolved) != checkpoint_payload["config_sha256"]:
        raise ValueError(f"Resolved configuration changed after training for {ledger_row.run_id}")
    return checkpoint, summary_path, summary, resolved


def _relocate_foundation_config(
    resolved: dict[str, object], week7_contracts_path: str | Path
) -> dict[str, object]:
    contracts_path = Path(week7_contracts_path)
    contracts = _json(contracts_path)
    model_id = str(resolved["run"]["model_id"])
    asset = contracts["dinov2" if model_id == "M5" else "olmoearth"]
    initialization = dict(resolved["initialization"])
    initialization["source_dir"] = str((contracts_path.parent / asset["source_dir"]).resolve())
    if model_id == "M5":
        initialization["weights_path"] = str((contracts_path.parent / asset["weights_file"]).resolve())
    else:
        initialization["model_dir"] = str((contracts_path.parent / asset["model_dir"]).resolve())
    relocated = dict(resolved)
    relocated["initialization"] = initialization
    return relocated


def _controlled_base(config: dict[str, object], model_id: str, partition: str):
    paths = config["paths"]
    common = (
        paths["manifest_path"], paths["staged_root"], paths["normalization_path"], partition
    )
    if model_id == "M1RGB":
        return SpectraShiftImageNetRGBDataset(
            paths["manifest_path"], paths["staged_root"], paths["normalization_path"],
            paths["rgb_contract_path"], partition, int(config["evaluation"]["shard_size"]),
            int(config["evaluation"]["height"]), int(config["evaluation"]["width"]),
        )
    return SpectraShiftDataset(
        *common, MODEL_ADAPTERS[model_id], int(config["evaluation"]["shard_size"]),
        int(config["evaluation"]["height"]), int(config["evaluation"]["width"]),
    )


def _foundation_base(config: dict[str, object], model_id: str, partition: str):
    paths, evaluation = config["paths"], config["evaluation"]
    if model_id == "M5":
        return DINOv2RGBDataset(
            paths["manifest_path"], paths["staged_root"], paths["normalization_path"],
            paths["rgb_contract_path"], partition, int(evaluation["shard_size"]),
            int(evaluation["height"]), int(evaluation["width"]), 126,
        )
    return OlmoEarthS2Dataset(
        paths["manifest_path"], paths["staged_root"], paths["olmo_contract_path"],
        partition, int(evaluation["shard_size"]), int(evaluation["height"]),
        int(evaluation["width"]),
    )


def _predict_controlled(model, dataset, device, batch_size: int, workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    logits, targets, patch_ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, labels, ids in loader:
            logits.append(model(images.to(device, non_blocking=True)).float().cpu().numpy())
            targets.append(labels.numpy())
            patch_ids.extend(map(str, ids))
    return np.concatenate(logits), np.concatenate(targets), patch_ids


def _predict_foundation(model, dataset, device, batch_size: int, workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    logits, targets, patch_ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            patch_ids.extend(map(str, batch.pop("patch_id")))
            targets.append(batch.pop("labels").numpy())
            moved = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            logits.append(model(moved).float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(targets), patch_ids


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))


def _metric_payload(
    targets: np.ndarray, logits: np.ndarray, threshold: float, supported: Sequence[int]
) -> dict[str, object]:
    probabilities = _probabilities(logits)
    half = multilabel_metrics(targets, probabilities, thresholds=0.5, supported_indices=supported)
    selected = multilabel_metrics(
        targets, probabilities, thresholds=float(threshold), supported_indices=supported
    )
    return {"metrics_at_0_5": half, "metrics_at_source_v_threshold": selected}


def _save_prediction(path: Path, logits: np.ndarray, patch_ids: Sequence[str]) -> None:
    if logits.ndim != 2 or logits.shape[1] != len(CANONICAL_LABELS) or not np.isfinite(logits).all():
        raise ValueError("Evaluation logits must be finite with shape [N,19]")
    if len(patch_ids) != len(logits) or len(set(patch_ids)) != len(patch_ids):
        raise ValueError("Evaluation patch IDs are missing or duplicated")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, logits=logits.astype(np.float32), patch_ids=np.asarray(patch_ids, dtype="U"))
    os.replace(temporary, path)


def _preprocessing_record(model_id: str, config: dict[str, object]) -> dict[str, object]:
    paths = config["paths"]
    record = {
        "manifest_contract_sha256": config["frozen"]["manifest_contract_sha256"],
        "normalization_sha256": config["frozen"]["normalization_sha256"],
    }
    if model_id in {"M0", "M1", "M2", "M3", "M4"}:
        record.update({"adapter": MODEL_ADAPTERS[model_id], "band_order": list(BAND_ADAPTERS[MODEL_ADAPTERS[model_id]])})
    elif model_id == "M1RGB":
        record.update({"adapter": "rgb", "band_order": list(BAND_ADAPTERS["rgb"]),
                       "rgb_contract_sha256": file_sha256(paths["rgb_contract_path"])})
    elif model_id == "M5":
        record.update({"adapter": "dinov2-rgb", "band_order": list(BAND_ADAPTERS["rgb"]),
                       "rgb_contract_sha256": file_sha256(paths["rgb_contract_path"]), "resolution": 126})
    else:
        record.update({"adapter": "olmo12", "band_order": list(BAND_ADAPTERS["olmo12"]),
                       "olmo_contract_sha256": file_sha256(paths["olmo_contract_path"]), "resolution": 120})
    record["sha256"] = object_sha256(record)
    return record


def evaluate_week8_seed(
    config_path: str | Path, seed: int, input_roots: Sequence[str | Path], output_dir: str | Path
) -> dict[str, object]:
    if int(seed) not in WEEK8_SEEDS:
        raise ValueError("Week 8 seed must be 17, 29, or 43")
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    paths, frozen = config["paths"], config["frozen"]
    _, ledger = validate_week7_approval(
        paths["week7_summary_path"], paths["checkpoint_ledger_path"],
        frozen["week7_summary_sha256"], frozen["checkpoint_ledger_sha256"],
    )
    contract = _json(paths["evaluation_contract_path"])
    support = _json(paths["support_contract_path"])
    if (
        contract.get("status") != "frozen"
        or not contract.get("model_selection_frozen_before_label_access")
        or contract["hashes"]["checkpoint_ledger_sha256"] != file_sha256(paths["checkpoint_ledger_path"])
        or contract["hashes"]["final_labels_sha256"] != file_sha256(paths["evaluation_labels_path"])
    ):
        raise ValueError("Week 8 evaluation contract is incomplete or changed")
    labels_frame = pd.read_parquet(paths["evaluation_labels_path"])
    if labels_frame.groupby("partition").size().to_dict() != EXPECTED_PARTITION_COUNTS:
        raise ValueError("Sealed evaluation label counts changed")
    label_lookup = {
        str(row.patch_id): matrix
        for row, matrix in zip(labels_frame.itertuples(index=False), _encode_labels(labels_frame["labels"]), strict=True)
    }
    seed_ledger = ledger[ledger["seed"].astype(int).eq(int(seed))].copy()
    if len(seed_ledger) != EXPECTED_RUNS_PER_SEED:
        raise ValueError("Seed ledger does not contain exactly 37 runs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(bool(config["evaluation"].get("require_cuda", True)))
    runs: list[dict[str, object]] = []
    seed_started = time.perf_counter()
    for ledger_row in seed_ledger.itertuples(index=False):
        run_started = time.perf_counter()
        checkpoint_path, summary_path, summary, resolved = _load_run_artifacts(input_roots, ledger_row)
        model_id, run_id = str(ledger_row.model_id), str(ledger_row.run_id)
        if str(ledger_row.family) == "foundation":
            relocated = _relocate_foundation_config(resolved, paths["week7_contracts_path"])
            model, _ = build_foundation_model(relocated)
            payload = verify_foundation_checkpoint(checkpoint_path, run_id)
        else:
            channels = 3 if model_id in {"M1RGB", "M2"} else 10
            model = build_resnet18(channels, outputs=19)
            payload = verify_downstream_checkpoint(checkpoint_path, run_id)
        model.load_state_dict(payload["model"], strict=True)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError(f"Checkpoint contains non-finite values: {run_id}")
        model = model.to(device).eval()
        threshold_record = summary.get("selected_global_threshold")
        if not isinstance(threshold_record, dict) or threshold_record.get("fitted_on") != "V":
            raise ValueError(f"Run lacks a frozen source-V threshold: {run_id}")
        threshold = float(threshold_record["threshold"])
        preprocessing = _preprocessing_record(model_id, config)
        predictions, metrics = {}, {}
        for partition in EVALUATION_PARTITIONS:
            if model_id in FOUNDATION_MODELS:
                base = _foundation_base(config, model_id, partition)
                dataset = _FoundationEvaluationDataset(base, label_lookup)
                logits, targets, patch_ids = _predict_foundation(
                    model, dataset, device, int(config["evaluation"]["foundation_batch_size"]),
                    int(config["evaluation"]["num_workers"]),
                )
            else:
                base = _controlled_base(config, model_id, partition)
                dataset = _ControlledEvaluationDataset(base, label_lookup)
                logits, targets, patch_ids = _predict_controlled(
                    model, dataset, device, int(config["evaluation"]["controlled_batch_size"]),
                    int(config["evaluation"]["num_workers"]),
                )
            if len(patch_ids) != EXPECTED_PARTITION_COUNTS[partition] or set(patch_ids) != set(
                labels_frame.loc[labels_frame["partition"].eq(partition), "patch_id"].astype(str)
            ):
                raise ValueError(f"Prediction IDs differ from the sealed {partition} contract")
            prediction_path = output_dir / run_id / f"predictions-{partition.lower()}.npz"
            _save_prediction(prediction_path, logits, patch_ids)
            predictions[partition] = {
                "path": str(prediction_path), "sha256": file_sha256(prediction_path),
                "rows": len(patch_ids), "logit_columns": logits.shape[1],
            }
            metric = _metric_payload(targets, logits, threshold, support["source_supported_indices"])
            metadata = labels_frame.set_index("patch_id").loc[patch_ids]
            metric["counts"] = {
                "samples": len(patch_ids),
                "positives": targets.sum(axis=0).astype(int).tolist(),
                "negatives": (len(targets) - targets.sum(axis=0)).astype(int).tolist(),
                "mgrs_groups": int(metadata["mgrs_group"].nunique()),
                "blocks_12km": int(metadata["block_12km"].nunique()),
                "countries": metadata["country"].value_counts().sort_index().astype(int).to_dict(),
                "quarters": {str(key): int(value) for key, value in metadata["quarter"].value_counts().sort_index().items()},
            }
            metrics[partition] = metric
        run_result = {
            "run_id": run_id, "family": str(ledger_row.family), "model_id": model_id,
            "fraction_code": str(ledger_row.fraction_code).zfill(2), "seed": int(seed),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "frozen_checkpoint_sha256": str(ledger_row.checkpoint_sha256),
            "source_v_threshold": threshold_record,
            "source_v_threshold_sha256": object_sha256(threshold_record),
            "preprocessing": preprocessing,
            "predictions": predictions, "metrics": metrics,
            "batch_size": int(config["evaluation"]["foundation_batch_size"] if model_id in FOUNDATION_MODELS else config["evaluation"]["controlled_batch_size"]),
            "runtime_seconds": time.perf_counter() - run_started,
            "hardware": hardware_record(device),
            "model_selection_after_label_access": False,
            "parameter_updates_after_label_access": False,
            "evaluation_labels_loaded": True,
        }
        _write_json(output_dir / run_id / "evaluation_summary.json", run_result)
        runs.append(run_result)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(runs) != EXPECTED_RUNS_PER_SEED or len({run["run_id"] for run in runs}) != EXPECTED_RUNS_PER_SEED:
        raise ValueError("Seed evaluation did not produce 37 unique runs")
    result = {
        "week8_seed_evaluation_complete": True, "seed": int(seed),
        "run_count": len(runs), "prediction_domain_count": len(runs) * 3,
        "runs": runs, "elapsed_seconds": time.perf_counter() - seed_started,
        "week7_summary_sha256": file_sha256(paths["week7_summary_path"]),
        "checkpoint_ledger_sha256": file_sha256(paths["checkpoint_ledger_path"]),
        "evaluation_contract_sha256": file_sha256(paths["evaluation_contract_path"]),
        "support_contract_sha256": file_sha256(paths["support_contract_path"]),
        "source_tree_sha256": source_tree_sha256(), "git_commit": git_commit(),
        "model_selection_frozen_before_label_access": True,
        "model_selection_after_label_access": False,
        "parameter_updates_after_label_access": False,
        "evaluation_labels_loaded": True,
    }
    _write_json(output_dir / f"week8_seed{seed}_summary.json", result)
    return result


def equal_country_ood(finland: float, portugal: float) -> float:
    return (float(finland) + float(portugal)) / 2.0


def normalized_log_aulc(label_counts: Sequence[int], values: Sequence[float]) -> float:
    counts = np.asarray(label_counts, dtype=np.float64)
    scores = np.asarray(values, dtype=np.float64)
    if len(counts) < 2 or counts.shape != scores.shape or np.any(counts <= 0) or not np.isfinite(scores).all():
        raise ValueError("AULC requires aligned finite values at two or more positive label counts")
    order = np.argsort(counts)
    x, y = np.log10(counts[order]), scores[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("AULC label counts must be unique")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _safe_ap(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=bool)
    positives = int(targets.sum())
    if positives == 0 or positives == len(targets):
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ranked = targets[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def _supported_map(targets: np.ndarray, probabilities: np.ndarray, indices: Sequence[int]) -> float:
    values = np.asarray([_safe_ap(targets[:, index], probabilities[:, index]) for index in indices])
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _prediction_path(seed_summary_path: Path, run: dict[str, object], partition: str) -> Path:
    return seed_summary_path.parent / str(run["run_id"]) / Path(str(run["predictions"][partition]["path"])).name


def _load_predictions(seed_summary_path: Path, run: dict[str, object], partition: str) -> tuple[np.ndarray, np.ndarray]:
    path = _prediction_path(seed_summary_path, run, partition)
    expected = run["predictions"][partition]
    if file_sha256(path) != expected["sha256"]:
        raise ValueError(f"Prediction hash changed for {run['run_id']} {partition}")
    with np.load(path, allow_pickle=False) as payload:
        logits, patch_ids = payload["logits"], payload["patch_ids"].astype(str)
    if logits.shape != (EXPECTED_PARTITION_COUNTS[partition], 19) or len(set(patch_ids)) != len(patch_ids):
        raise ValueError(f"Prediction shape or identity failure for {run['run_id']} {partition}")
    if not np.isfinite(logits).all():
        raise ValueError(f"Non-finite prediction for {run['run_id']} {partition}")
    return logits, patch_ids


def _bootstrap_draws(groups: np.ndarray, replicates: int, seed: int) -> list[np.ndarray]:
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Geographic bootstrap requires at least two groups")
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws.append(np.concatenate([members[group] for group in sampled]))
    return draws


def _paired_bootstrap_difference(
    targets: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    draws: Sequence[np.ndarray],
    indices: Sequence[int],
) -> np.ndarray:
    result = np.empty(len(draws), dtype=np.float64)
    for position, selected in enumerate(draws):
        result[position] = _supported_map(targets[selected], left[selected], indices) - _supported_map(
            targets[selected], right[selected], indices
        )
    return result


def _metrics_rows(
    run: dict[str, object], partition: str, targets: np.ndarray, logits: np.ndarray,
    source_indices: Sequence[int], metadata: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    threshold = float(run["source_v_threshold"]["threshold"])
    probabilities = _probabilities(logits)
    metric_half = multilabel_metrics(targets, probabilities, 0.5, source_indices)
    metric_selected = multilabel_metrics(targets, probabilities, threshold, source_indices)
    domain = DOMAIN_NAMES[partition]
    row = {
        "run_id": run["run_id"], "model_id": run["model_id"],
        "fraction_code": run["fraction_code"], "seed": run["seed"], "domain": domain,
        "sample_count": len(targets), "supported_map": metric_half["macro_average_precision"],
        "all_class_map": metric_half["all_class_macro_average_precision"],
        "macro_f1_at_0_5": metric_half["macro_f1"], "micro_f1_at_0_5": metric_half["micro_f1"],
        "macro_f1_at_source_v_threshold": metric_selected["macro_f1"],
        "micro_f1_at_source_v_threshold": metric_selected["micro_f1"],
        "source_v_threshold": threshold,
        "binary_brier_score": metric_half["binary_brier_score"],
        "macro_classwise_ece_15bin": metric_half["macro_classwise_expected_calibration_error"],
        "mgrs_group_count": int(metadata["mgrs_group"].nunique()),
        "block_12km_count": int(metadata["block_12km"].nunique()),
        "country_count": int(metadata["country"].nunique()),
        "quarter_count": int(metadata["quarter"].nunique()),
    }
    class_rows = []
    positives = targets.sum(axis=0).astype(int)
    for index, name in enumerate(CANONICAL_LABELS):
        class_rows.append({
            "run_id": run["run_id"], "model_id": run["model_id"],
            "fraction_code": run["fraction_code"], "seed": run["seed"], "domain": domain,
            "class_index": index, "class_name": name,
            "positive_count": int(positives[index]), "negative_count": int(len(targets) - positives[index]),
            "average_precision": _safe_ap(targets[:, index], probabilities[:, index]),
            "recall_at_0_5": metric_half["per_class_recall"][index],
            "recall_at_source_v_threshold": metric_selected["per_class_recall"][index],
            "f1_at_0_5": metric_half["per_class_f1"][index],
            "f1_at_source_v_threshold": metric_selected["per_class_f1"][index],
            "ece_15bin": metric_half["per_class_expected_calibration_error"][index],
            "in_c_source": index in source_indices,
        })
    return row, class_rows


def aggregate_week8(
    config_path: str | Path, seed_summary_paths: Sequence[str | Path], output_dir: str | Path
) -> dict[str, object]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    paths, frozen = config["paths"], config["frozen"]
    _, ledger = validate_week7_approval(
        paths["week7_summary_path"], paths["checkpoint_ledger_path"],
        frozen["week7_summary_sha256"], frozen["checkpoint_ledger_sha256"],
    )
    contract = _json(paths["evaluation_contract_path"])
    support = _json(paths["support_contract_path"])
    labels = pd.read_parquet(paths["evaluation_labels_path"])
    if contract["hashes"]["final_labels_sha256"] != file_sha256(paths["evaluation_labels_path"]):
        raise ValueError("Final evaluation labels changed after sealing")
    if len(seed_summary_paths) != 3:
        raise ValueError("Week 8 aggregation requires exactly three seed summaries")
    summary_paths = [Path(path) for path in seed_summary_paths]
    summaries = [_json(path) for path in summary_paths]
    if {int(summary["seed"]) for summary in summaries} != set(WEEK8_SEEDS):
        raise ValueError("Week 8 seed summaries must cover 17, 29, and 43")
    if any(
        not summary.get("week8_seed_evaluation_complete")
        or summary.get("run_count") != EXPECTED_RUNS_PER_SEED
        or summary.get("prediction_domain_count") != EXPECTED_RUNS_PER_SEED * 3
        or summary.get("model_selection_after_label_access") is not False
        or summary.get("parameter_updates_after_label_access") is not False
        for summary in summaries
    ):
        raise ValueError("A Week 8 seed output failed its immutable evaluation gate")
    runs = [run for summary in summaries for run in summary["runs"]]
    if len(runs) != EXPECTED_LEDGER_ROWS or len({run["run_id"] for run in runs}) != EXPECTED_LEDGER_ROWS:
        raise ValueError("Week 8 aggregate requires 111 unique evaluation runs")
    ledger_by_id = ledger.set_index("run_id")
    for run in runs:
        frozen_row = ledger_by_id.loc[run["run_id"]]
        if run["checkpoint_sha256"] != frozen_row["checkpoint_sha256"]:
            raise ValueError(f"Post-seal checkpoint change detected for {run['run_id']}")
    source_indices = [int(value) for value in support["source_supported_indices"]]
    common_indices = [int(value) for value in support["common_source_indices"]]
    label_by_id = labels.set_index("patch_id")
    target_by_partition = {
        partition: _encode_labels(labels.loc[labels["partition"].eq(partition), "labels"])
        for partition in EVALUATION_PARTITIONS
    }
    ids_by_partition = {
        partition: labels.loc[labels["partition"].eq(partition), "patch_id"].astype(str).to_numpy()
        for partition in EVALUATION_PARTITIONS
    }
    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    path_by_seed = {int(summary["seed"]): path for summary, path in zip(summaries, summary_paths, strict=True)}
    for run in runs:
        summary_path = path_by_seed[int(run["seed"])]
        domain_maps = {}
        for partition in EVALUATION_PARTITIONS:
            logits, patch_ids = _load_predictions(summary_path, run, partition)
            expected_ids = ids_by_partition[partition]
            position = {patch_id: index for index, patch_id in enumerate(expected_ids)}
            if set(patch_ids) != set(expected_ids):
                raise ValueError(f"Prediction IDs do not match sealed labels for {run['run_id']} {partition}")
            order = np.asarray([position[patch_id] for patch_id in patch_ids])
            targets = target_by_partition[partition][order]
            metadata = label_by_id.loc[patch_ids]
            row, per_class = _metrics_rows(run, partition, targets, logits, source_indices, metadata)
            metric_rows.append(row)
            class_rows.extend(per_class)
            domain_maps[partition] = float(row["supported_map"])
            predictions[(run["run_id"], partition)] = (_probabilities(logits), patch_ids)
        metric_rows.append({
            "run_id": run["run_id"], "model_id": run["model_id"],
            "fraction_code": run["fraction_code"], "seed": run["seed"], "domain": "OOD",
            "sample_count": 8000,
            "supported_map": equal_country_ood(domain_maps["T-FI"], domain_maps["T-PT"]),
            "all_class_map": float("nan"), "macro_f1_at_0_5": float("nan"),
            "micro_f1_at_0_5": float("nan"), "macro_f1_at_source_v_threshold": float("nan"),
            "micro_f1_at_source_v_threshold": float("nan"),
            "source_v_threshold": float(run["source_v_threshold"]["threshold"]),
            "binary_brier_score": float("nan"), "macro_classwise_ece_15bin": float("nan"),
            "mgrs_group_count": int(labels[labels["partition"].isin(["T-FI", "T-PT"])]["mgrs_group"].nunique()),
            "block_12km_count": int(labels[labels["partition"].isin(["T-FI", "T-PT"])]["block_12km"].nunique()),
            "country_count": 2, "quarter_count": int(labels[labels["partition"].isin(["T-FI", "T-PT"])]["quarter"].nunique()),
        })
    domain_metrics = pd.DataFrame(metric_rows)
    per_class_metrics = pd.DataFrame(class_rows)

    ranking_rows = []
    for (domain, fraction), group in domain_metrics.groupby(["domain", "fraction_code"], sort=True):
        means = group.groupby("model_id")["supported_map"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        for rank, (model_id, values) in enumerate(means.iterrows(), start=1):
            ranking_rows.append({
                "domain": domain, "fraction_code": str(fraction).zfill(2), "model_id": model_id,
                "mean_supported_map": float(values["mean"]),
                "sample_std_supported_map": float(values["std"]), "rank": rank,
            })
    rankings = pd.DataFrame(ranking_rows)

    aulc_rows = []
    controlled = domain_metrics[domain_metrics["model_id"].isin(CONTROLLED_MODELS)]
    for hypothesis, (left, right) in HYPOTHESES.items():
        for domain in ("I", "Finland", "Portugal", "OOD"):
            paired = []
            for seed in WEEK8_SEEDS:
                left_rows = controlled[(controlled.model_id == left) & (controlled.seed == seed) & (controlled.domain == domain)]
                right_rows = controlled[(controlled.model_id == right) & (controlled.seed == seed) & (controlled.domain == domain)]
                left_rows = left_rows.assign(count=left_rows.fraction_code.map(FRACTION_COUNTS)).sort_values("count")
                right_rows = right_rows.assign(count=right_rows.fraction_code.map(FRACTION_COUNTS)).sort_values("count")
                if len(left_rows) != 6 or len(right_rows) != 6:
                    raise ValueError(f"Missing controlled curve for {hypothesis} seed {seed}")
                paired.append(
                    normalized_log_aulc(left_rows["count"], left_rows["supported_map"])
                    - normalized_log_aulc(right_rows["count"], right_rows["supported_map"])
                )
            aulc_rows.append({
                "hypothesis": hypothesis, "left_model": left, "right_model": right, "domain": domain,
                "paired_aulc_difference_mean": float(np.mean(paired)),
                "paired_aulc_difference_sample_std": float(np.std(paired, ddof=1)),
                "seed_differences": paired,
            })

    replicates, bootstrap_seed = int(contract["bootstrap_replicates"]), int(contract["bootstrap_seed"])
    draws: dict[tuple[str, str], list[np.ndarray]] = {}
    for partition in EVALUATION_PARTITIONS:
        frame = labels[labels["partition"].eq(partition)].reset_index(drop=True)
        for group_name in ("mgrs_group", "block_12km"):
            offset = int(hashlib.sha256(f"{partition}:{group_name}".encode()).hexdigest()[:8], 16)
            draws[(partition, group_name)] = _bootstrap_draws(
                frame[group_name].astype(str).to_numpy(), replicates, bootstrap_seed + offset
            )
    bootstrap_rows = []
    run_lookup = {(run["model_id"], run["fraction_code"], int(run["seed"])): run for run in runs}
    for hypothesis, (left_model, right_model) in HYPOTHESES.items():
        for fraction in ("01", "05", "10", "25", "50", "100"):
            for group_name in ("mgrs_group", "block_12km"):
                seed_domain_replicates: dict[str, list[np.ndarray]] = {name: [] for name in ("I", "Finland", "Portugal", "OOD")}
                observed_by_domain: dict[str, list[float]] = {name: [] for name in seed_domain_replicates}
                for seed in WEEK8_SEEDS:
                    left_run, right_run = run_lookup[(left_model, fraction, seed)], run_lookup[(right_model, fraction, seed)]
                    per_seed = {}
                    for partition in EVALUATION_PARTITIONS:
                        frame = labels[labels["partition"].eq(partition)].reset_index(drop=True)
                        target = _encode_labels(frame["labels"])
                        canonical_ids = frame["patch_id"].astype(str).to_numpy()
                        def aligned(run):
                            probability, patch_ids = predictions[(run["run_id"], partition)]
                            lookup = {patch_id: index for index, patch_id in enumerate(patch_ids)}
                            return probability[[lookup[patch_id] for patch_id in canonical_ids]]
                        left_probability, right_probability = aligned(left_run), aligned(right_run)
                        values = _paired_bootstrap_difference(
                            target, left_probability, right_probability, draws[(partition, group_name)], common_indices
                        )
                        domain = DOMAIN_NAMES[partition]
                        per_seed[domain] = values
                        seed_domain_replicates[domain].append(values)
                        observed_by_domain[domain].append(
                            _supported_map(target, left_probability, common_indices)
                            - _supported_map(target, right_probability, common_indices)
                        )
                    ood_values = (per_seed["Finland"] + per_seed["Portugal"]) / 2.0
                    seed_domain_replicates["OOD"].append(ood_values)
                    observed_by_domain["OOD"].append(
                        equal_country_ood(observed_by_domain["Finland"][-1], observed_by_domain["Portugal"][-1])
                    )
                for domain in ("I", "Finland", "Portugal", "OOD"):
                    across_seeds = np.mean(np.stack(seed_domain_replicates[domain]), axis=0)
                    bootstrap_rows.append({
                        "hypothesis": hypothesis, "left_model": left_model, "right_model": right_model,
                        "fraction_code": fraction, "domain": domain,
                        "grouping": "MGRS" if group_name == "mgrs_group" else "12km-block",
                        "observed_paired_difference": float(np.mean(observed_by_domain[domain])),
                        "bootstrap_mean": float(np.nanmean(across_seeds)),
                        "percentile_2_5": float(np.nanpercentile(across_seeds, 2.5)),
                        "percentile_97_5": float(np.nanpercentile(across_seeds, 97.5)),
                        "replicates": replicates, "p_value": None,
                    })
    paired_bootstrap = pd.DataFrame(bootstrap_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_metrics.to_csv(output_dir / "domain_metrics.csv", index=False)
    per_class_metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)
    paired_bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    rankings.to_csv(output_dir / "domain_rankings.csv", index=False)
    (output_dir / "support_contract.json").write_bytes(Path(paths["support_contract_path"]).read_bytes())
    result = {
        "week8_complete": True, "week9_approved": True,
        "evaluation_run_count": len(runs),
        "prediction_domain_count": len(runs) * 3,
        "checkpoint_ledger_count": len(ledger),
        "partition_counts": EXPECTED_PARTITION_COUNTS,
        "source_supported_class_count": len(source_indices),
        "common_source_class_count": len(common_indices),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "hypothesis_aulc": aulc_rows,
        "output_hashes": {},
        "week7_summary_sha256": file_sha256(paths["week7_summary_path"]),
        "checkpoint_ledger_sha256": file_sha256(paths["checkpoint_ledger_path"]),
        "evaluation_contract_sha256": file_sha256(paths["evaluation_contract_path"]),
        "support_contract_sha256": file_sha256(output_dir / "support_contract.json"),
        "model_selection_frozen_before_label_access": True,
        "model_selection_after_label_access": False,
        "parameter_updates_after_label_access": False,
        "evaluation_labels_loaded": True,
        "week9_implemented": False,
    }
    for name in ("domain_metrics.csv", "per_class_metrics.csv", "paired_bootstrap.csv", "domain_rankings.csv", "support_contract.json"):
        result["output_hashes"][name] = file_sha256(output_dir / name)
    _write_json(output_dir / "week8_run_summary.json", result)
    return result
