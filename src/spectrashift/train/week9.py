from __future__ import annotations

import gc
import functools
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from spectrashift.data.bands import BAND_ADAPTERS
from spectrashift.data.dataset import SpectraShiftDataset
from spectrashift.data.labels import CANONICAL_LABELS
from spectrashift.eval.metrics import multilabel_metrics
from spectrashift.models.resnet import build_resnet18
from spectrashift.train.common import (
    file_sha256,
    git_commit,
    hardware_record,
    object_sha256,
    select_device,
    source_tree_sha256,
)
from spectrashift.train.downstream import model_state_sha256, verify_downstream_checkpoint
from spectrashift.train.week7 import _fit_linear_candidate, _label_matrix
from spectrashift.train.week8 import (
    DOMAIN_NAMES,
    EVALUATION_PARTITIONS,
    EXPECTED_PARTITION_COUNTS,
    _bootstrap_draws,
    _encode_labels,
    _paired_bootstrap_difference,
    _probabilities,
    _safe_ap,
    _supported_map,
    equal_country_ood,
    normalized_log_aulc,
)


WEEK9_SEEDS = (17, 29, 43)
PROBE_MODELS = ("M1", "M2", "M3", "M4", "M5", "M6")
NEW_PROBE_MODELS = ("M1", "M2", "M3", "M4")
DIAGNOSTIC_MODELS = ("M3", "M4")
PROBE_FRACTIONS = ("01", "05", "10", "25", "50", "100")
FRACTION_COUNTS = {"01": 120, "05": 600, "10": 1200, "25": 3000, "50": 6000, "100": 12000}
STRESS_CONDITIONS = (
    "clean", "missing_b08", "missing_red_edge", "missing_swir", "gain_0p9", "gain_1p1"
)
PERTURBED_CONDITIONS = STRESS_CONDITIONS[1:]
CORE10_GROUPS = {
    "missing_b08": (3,),
    "missing_red_edge": (4, 5, 6, 7),
    "missing_swir": (8, 9),
}
GAIN_CONDITIONS = {"gain_0p9": 0.9, "gain_1p1": 1.1}


def _json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True, default=str) + "\n")


def _find_unique(roots: Sequence[str | Path], name: str, digest: str | None = None) -> Path:
    candidates: dict[str, Path] = {}
    for root in roots:
        for path in Path(root).rglob(name):
            value = file_sha256(path)
            if digest is None or value == digest:
                candidates.setdefault(value, path)
    if len(candidates) != 1:
        raise ValueError(f"Expected one unique {name}, found {list(candidates.values())}")
    return next(iter(candidates.values()))


def _find_run_artifact(
    roots: Sequence[str | Path], run_id: str, name: str, digest: str | None = None
) -> Path:
    matches = []
    for root in roots:
        matches.extend(path for path in Path(root).rglob(name) if path.parent.name == run_id)
    if digest is not None:
        matches = [path for path in matches if file_sha256(path) == digest]
    unique = {str(path.resolve()): path for path in matches}
    if len(unique) != 1:
        raise ValueError(f"Expected one {name} for {run_id}, found {list(unique.values())}")
    return next(iter(unique.values()))


def _hash_rank(seed: int, partition: str, patch_id: str) -> tuple[bytes, str]:
    payload = f"{seed}:{partition}:{patch_id}".encode()
    return hashlib.blake2b(payload, digest_size=16).digest(), patch_id


def deterministic_patch_ids(
    patch_ids: Sequence[str], count: int, seed: int, partition: str
) -> list[str]:
    values = list(map(str, patch_ids))
    if len(values) != len(set(values)) or count <= 0 or count > len(values):
        raise ValueError("Deterministic sampling requires unique IDs and a valid count")
    return sorted(values, key=lambda value: _hash_rank(seed, partition, value))[:count]


def validate_week8_approval(config: dict[str, object]) -> dict[str, object]:
    paths, frozen = config["paths"], config["frozen"]
    summary_path = Path(paths["week8_summary_path"])
    if file_sha256(summary_path) != frozen["week8_summary_sha256"]:
        raise ValueError("Week 8 summary differs from the frozen approval")
    summary = _json(summary_path)
    if (
        not summary.get("week8_complete")
        or not summary.get("week9_approved")
        or summary.get("evaluation_run_count") != 111
        or summary.get("prediction_domain_count") != 333
        or summary.get("model_selection_after_label_access") is not False
        or summary.get("parameter_updates_after_label_access") is not False
        or summary.get("week9_implemented") is not False
    ):
        raise ValueError("Week 8 aggregate does not approve immutable Week 9 analysis")
    if file_sha256(paths["checkpoint_ledger_path"]) != frozen["checkpoint_ledger_sha256"]:
        raise ValueError("Checkpoint ledger changed after Week 8")
    if file_sha256(paths["evaluation_contract_path"]) != frozen["evaluation_contract_sha256"]:
        raise ValueError("Evaluation contract changed after sealing")
    if file_sha256(paths["support_contract_path"]) != frozen["support_contract_sha256"]:
        raise ValueError("Support contract changed after sealing")
    output_dir = Path(paths["week8_output_dir"])
    for name, digest in summary["output_hashes"].items():
        if file_sha256(output_dir / name) != digest:
            raise ValueError(f"Week 8 output changed after aggregation: {name}")
    return summary


def freeze_week9_contracts(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    summary = validate_week8_approval(config)
    paths, sampling = config["paths"], config["sampling"]
    manifest = pd.read_parquet(paths["manifest_path"])
    evaluation = pd.read_parquet(paths["evaluation_labels_path"])
    if evaluation.groupby("partition").size().to_dict() != EXPECTED_PARTITION_COUNTS:
        raise ValueError("Week 9 received altered evaluation labels")
    if evaluation["patch_id"].duplicated().any():
        raise ValueError("Week 9 evaluation IDs are duplicated")
    v_ids = manifest.loc[manifest["partition"].eq("V"), "patch_id"].astype(str).tolist()
    cka_ids = deterministic_patch_ids(
        v_ids, int(sampling["cka_patch_count"]), int(sampling["seed"]), "V"
    )
    query_ids = {}
    for partition, count in sampling["query_counts"].items():
        candidates = evaluation.loc[evaluation["partition"].eq(partition), "patch_id"].astype(str)
        query_ids[partition] = deterministic_patch_ids(
            candidates.tolist(), int(count), int(sampling["seed"]), str(partition)
        )
    if sum(map(len, query_ids.values())) != 100:
        raise ValueError("Week 9 nearest-neighbour query contract must contain 100 IDs")

    ledger = pd.read_csv(paths["checkpoint_ledger_path"], dtype={"fraction_code": str})
    ledger["fraction_code"] = ledger["fraction_code"].str.zfill(2)
    diagnostic = ledger[
        ledger["model_id"].isin(DIAGNOSTIC_MODELS)
        & ledger["fraction_code"].eq("10")
        & ledger["seed"].astype(int).isin(WEEK9_SEEDS)
    ].sort_values(["seed", "model_id"], kind="stable")
    if len(diagnostic) != 6 or diagnostic["run_id"].duplicated().any():
        raise ValueError("Week 9 requires exactly six frozen M3/M4 10% checkpoints")

    output_dir = Path(paths["contracts_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "week9_diagnostic_ledger.csv"
    diagnostic.to_csv(ledger_path, index=False)
    labels_path = output_dir / "evaluation_labels.parquet"
    shutil.copyfile(paths["evaluation_labels_path"], labels_path)
    support_path = output_dir / "support_contract.json"
    shutil.copyfile(paths["support_contract_path"], support_path)
    contract = {
        "status": "frozen",
        "sampling_seed": int(sampling["seed"]),
        "cka_partition": "V",
        "cka_patch_ids": cka_ids,
        "query_patch_ids": query_ids,
        "probe_contract": config["probes"],
        "diagnostic_contract": config["diagnostics"],
        "uncertainty_contract": config["uncertainty"],
        "output_schema": [
            "week9_run_summary.json", "representation_probes.csv", "probe_paired_differences.csv",
            "probe_aulc.csv", "cka_matrix.csv", "effective_rank.csv", "stress_metrics.csv",
            "stress_per_class.csv", "stress_bootstrap.csv", "spectral_sensitivity.parquet",
            "nearest_neighbors.parquet", "error_slices.csv",
        ],
        "week8_summary_sha256": file_sha256(paths["week8_summary_path"]),
        "checkpoint_ledger_sha256": file_sha256(paths["checkpoint_ledger_path"]),
        "diagnostic_ledger_sha256": file_sha256(ledger_path),
        "evaluation_contract_sha256": file_sha256(paths["evaluation_contract_path"]),
        "evaluation_labels_sha256": file_sha256(labels_path),
        "support_contract_sha256": file_sha256(support_path),
        "manifest_file_sha256": file_sha256(paths["manifest_path"]),
        "normalization_file_sha256": file_sha256(paths["normalization_path"]),
        "model_selection_after_week8": False,
        "encoder_updates_during_week9": False,
        "evaluation_labels_loaded": True,
    }
    contract_path = output_dir / "week9_contract.json"
    _write_json(contract_path, contract)
    result = {
        "week9_contracts_complete": True,
        "week8_summary_sha256": file_sha256(paths["week8_summary_path"]),
        "week9_contract_sha256": file_sha256(contract_path),
        "diagnostic_ledger_sha256": file_sha256(ledger_path),
        "evaluation_labels_sha256": file_sha256(labels_path),
        "support_contract_sha256": file_sha256(support_path),
        "cka_patch_count": len(cka_ids),
        "nearest_neighbor_query_count": sum(map(len, query_ids.values())),
        "diagnostic_checkpoint_count": len(diagnostic),
        "model_selection_after_week8": False,
        "encoder_updates_during_week9": False,
        "evaluation_labels_loaded": True,
    }
    _write_json(output_dir / "week9_contracts_summary.json", result)
    return result


def _feature_cache(roots: Sequence[str | Path], model_id: str, seed: int) -> tuple[Path, dict[str, object]]:
    stem = f"{model_id.lower()}-seed{seed}"
    summary_path = _find_unique(roots, f"{stem}.json")
    summary = _json(summary_path)
    cache_path = _find_unique(roots, f"{stem}.npz", str(summary["sha256"]))
    if (
        summary.get("model_id") != model_id
        or int(summary.get("seed", -1)) != seed
        or summary.get("d_patches") != 12000
        or summary.get("v_patches") != 2000
        or summary.get("evaluation_labels_loaded") is not False
    ):
        raise ValueError(f"Invalid frozen feature cache: {stem}")
    return cache_path, summary


def _linear_probe(
    model_id: str,
    seed: int,
    fraction: str,
    cache_path: Path,
    manifest_path: str | Path,
    subset_path: str | Path,
    subset_contract_path: str | Path,
    output_dir: Path,
    settings: dict[str, object],
) -> dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as payload:
        d_features = payload["D_features"].astype(np.float32)
        v_features = payload["V_features"].astype(np.float32)
        d_ids, v_ids = payload["D_patch_ids"].astype(str), payload["V_patch_ids"].astype(str)
    visible = pd.read_parquet(manifest_path)
    visible = visible[visible["partition"].isin(["D", "V"])]
    d_targets, v_targets = _label_matrix(visible, d_ids), _label_matrix(visible, v_ids)
    subsets = pd.read_parquet(subset_path)
    selected = subsets[subsets["downstream_seed"].eq(seed)].sort_values("subset_rank").head(
        FRACTION_COUNTS[fraction]
    )["patch_id"].astype(str)
    lookup = {value: index for index, value in enumerate(d_ids)}
    indices = np.asarray([lookup[value] for value in selected], dtype=np.int64)
    train_features, train_targets = d_features[indices], d_targets[indices]
    mean, std = train_features.mean(axis=0), train_features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_features = (train_features - mean) / std
    validation_features = (v_features - mean) / std
    subset_contract = _json(subset_contract_path)
    candidates = []
    for l2 in map(float, settings["l2_grid"]):
        score, state, logits = _fit_linear_candidate(
            train_features, train_targets, validation_features, v_targets,
            list(map(int, subset_contract["supported_class_indices"])), seed, l2,
            int(settings["linear_epochs"]), int(settings["linear_batch_size"]),
            float(settings["linear_learning_rate"]),
        )
        candidates.append((score, l2, state, logits))
    best = max(item[0] for item in candidates)
    tied = [item for item in candidates if math.isclose(item[0], best, abs_tol=1e-12, rel_tol=0)]
    preferred = [item for item in tied if math.isclose(item[1], 1e-4)]
    selected_candidate = preferred[0] if preferred else min(tied, key=lambda item: item[1])
    run_id = f"week9-{model_id.lower()}-linear-f{fraction}-seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / f"{run_id}-head.pt"
    torch.save(selected_candidate[2], head_path)
    result = {
        "run_id": run_id, "model_id": model_id, "probe_type": "linear", "seed": seed,
        "fraction_code": fraction, "sample_count": len(indices), "selected_l2": selected_candidate[1],
        "validation_macro_average_precision": selected_candidate[0],
        "candidate_scores": {str(item[1]): item[0] for item in candidates},
        "feature_cache_sha256": file_sha256(cache_path),
        "subset_manifest_sha256": subset_contract["subset_manifest_sha256"],
        "standardization_fit_partition": "D-subset", "head_file": str(head_path),
        "head_sha256": file_sha256(head_path), "validation_rows": len(v_ids),
        "validation_logit_shape": list(selected_candidate[3].shape),
        "finite_gate": bool(np.isfinite(selected_candidate[3]).all()),
        "encoder_updates": False, "evaluation_labels_loaded": False,
    }
    _write_json(output_dir / f"{run_id}.json", result)
    return result


def _knn_probe(
    model_id: str,
    seed: int,
    cache_path: Path,
    manifest_path: str | Path,
    subset_path: str | Path,
    subset_contract_path: str | Path,
    settings: dict[str, object],
) -> dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as payload:
        d_features = payload["D_features"].astype(np.float32)
        v_features = payload["V_features"].astype(np.float32)
        d_ids, v_ids = payload["D_patch_ids"].astype(str), payload["V_patch_ids"].astype(str)
    visible = pd.read_parquet(manifest_path)
    visible = visible[visible["partition"].isin(["D", "V"])]
    d_targets, v_targets = _label_matrix(visible, d_ids), _label_matrix(visible, v_ids)
    subset = pd.read_parquet(subset_path)
    selected_ids = subset[subset["downstream_seed"].eq(seed)].sort_values("subset_rank").head(1200)[
        "patch_id"
    ].astype(str).tolist()
    lookup = {value: index for index, value in enumerate(d_ids)}
    indices = np.asarray([lookup[value] for value in selected_ids], dtype=np.int64)
    train_features, train_targets = d_features[indices], d_targets[indices]
    train_ids = d_ids[indices]
    train_features /= np.maximum(np.linalg.norm(train_features, axis=1, keepdims=True), 1e-12)
    v_features /= np.maximum(np.linalg.norm(v_features, axis=1, keepdims=True), 1e-12)
    similarities = v_features @ train_features.T
    metadata = visible.set_index("patch_id")
    d_locations = metadata.loc[train_ids, "location_key"].astype(str).to_numpy()
    v_locations = metadata.loc[v_ids, "location_key"].astype(str).to_numpy()
    duplicate_mask = (v_ids[:, None] == train_ids[None, :]) | (v_locations[:, None] == d_locations[None, :])
    excluded = int(duplicate_mask.sum())
    similarities[duplicate_mask] = -np.inf
    contract = _json(subset_contract_path)
    scores_by_k = {}
    for k in map(int, settings["knn_k"]):
        neighbor_indices = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
        neighbor_similarity = np.take_along_axis(similarities, neighbor_indices, axis=1)
        if not np.isfinite(neighbor_similarity).all():
            raise ValueError("Duplicate exclusion left too few k-NN candidates")
        shifted = neighbor_similarity / float(settings["knn_temperature"])
        shifted -= shifted.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        weights /= weights.sum(axis=1, keepdims=True)
        probabilities = np.clip((train_targets[neighbor_indices] * weights[..., None]).sum(axis=1), 0, 1)
        metrics = multilabel_metrics(
            v_targets, probabilities, supported_indices=list(map(int, contract["supported_class_indices"]))
        )
        scores_by_k[k] = float(metrics["macro_average_precision"])
    best = max(scores_by_k.values())
    tied = [key for key, value in scores_by_k.items() if math.isclose(value, best, abs_tol=1e-12, rel_tol=0)]
    selected_k = 20 if 20 in tied else min(tied)
    return {
        "run_id": f"week9-{model_id.lower()}-knn-f10-seed{seed}",
        "model_id": model_id, "probe_type": "knn", "seed": seed, "fraction_code": "10",
        "sample_count": 1200, "selected_k": selected_k,
        "temperature": float(settings["knn_temperature"]),
        "validation_macro_average_precision": scores_by_k[selected_k],
        "candidate_scores": {str(key): value for key, value in scores_by_k.items()},
        "duplicate_candidates_excluded": excluded,
        "feature_cache_sha256": file_sha256(cache_path),
        "subset_manifest_sha256": contract["subset_manifest_sha256"],
        "validation_rows": len(v_ids), "finite_gate": True,
        "encoder_updates": False, "evaluation_labels_loaded": False,
    }


def run_week9_probes(
    config_path: str | Path, input_roots: Sequence[str | Path], output_dir: str | Path
) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    settings, paths = config["probes"], config["paths"]
    contracts_summary = _json(paths["week9_contracts_summary_path"])
    if not contracts_summary.get("week9_contracts_complete"):
        raise ValueError("Week 9 contracts are incomplete")
    if "week9_contract_path" in paths and (
        file_sha256(paths["week9_contract_path"]) != contracts_summary["week9_contract_sha256"]
    ):
        raise ValueError("Week 9 contract changed before probe execution")
    week7 = _json(paths["week7_summary_path"])
    probe_path = Path(paths["week7_probe_summary_path"])
    if file_sha256(probe_path) != week7["probe_summary_sha256"]:
        raise ValueError("Week 7 foundation probe summary hash changed")
    foundation = _json(probe_path)
    if foundation.get("foundation_linear_probe_count") != 36 or foundation.get("foundation_knn_run_count") != 6:
        raise ValueError("Week 7 foundation probes are incomplete")
    output_dir = Path(output_dir)
    linear_runs, knn_runs, caches = [], [], []
    for seed in WEEK9_SEEDS:
        for model_id in NEW_PROBE_MODELS:
            cache_path, cache_summary = _feature_cache(input_roots, model_id, seed)
            caches.append({**cache_summary, "path": str(cache_path)})
            for fraction in PROBE_FRACTIONS:
                linear_runs.append(_linear_probe(
                    model_id, seed, fraction, cache_path, paths["manifest_path"],
                    paths["subset_manifest_path"], paths["week5_contracts_path"],
                    output_dir / "linear", settings,
                ))
            knn_runs.append(_knn_probe(
                model_id, seed, cache_path, paths["manifest_path"], paths["subset_manifest_path"],
                paths["week5_contracts_path"], settings,
            ))
    all_linear = linear_runs + list(foundation["linear_runs"])
    all_knn = knn_runs + list(foundation["knn_runs"])
    linear_ids = {(run["model_id"], str(run["fraction_code"]).zfill(2), int(run["seed"])) for run in all_linear}
    knn_ids = {(run["model_id"], int(run["seed"])) for run in all_knn}
    expected_linear = {(model, fraction, seed) for model in PROBE_MODELS for fraction in PROBE_FRACTIONS for seed in WEEK9_SEEDS}
    expected_knn = {(model, seed) for model in PROBE_MODELS for seed in WEEK9_SEEDS}
    if linear_ids != expected_linear or knn_ids != expected_knn:
        raise ValueError("Week 9 probe registry is missing or duplicates model/fraction/seed identities")
    if any(not run.get("finite_gate") for run in all_linear + all_knn):
        raise ValueError("A Week 9 probe failed its finite gate")
    result = {
        "week9_probes_complete": True,
        "linear_probe_count": len(all_linear), "new_linear_probe_count": len(linear_runs),
        "knn_probe_count": len(all_knn), "new_knn_probe_count": len(knn_runs),
        "controlled_feature_cache_count": len(caches),
        "foundation_probe_summary_sha256": file_sha256(probe_path),
        "week9_contracts_summary_sha256": file_sha256(paths["week9_contracts_summary_path"]),
        "linear_runs": all_linear, "knn_runs": all_knn, "feature_caches": caches,
        "model_selection_after_week8": False, "encoder_updates_during_week9": False,
        "evaluation_labels_loaded": False,
    }
    _write_json(output_dir / "week9_probe_summary.json", result)
    return result


def apply_spectral_condition(
    image: np.ndarray,
    valid: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    condition: str,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if image.shape != valid.shape or image.ndim != 3 or image.shape[0] != 10:
        raise ValueError("Spectral stress expects aligned core10 [10,H,W] arrays")
    if condition == "clean":
        return image.copy()
    result = image.copy()
    if condition in CORE10_GROUPS:
        result[list(CORE10_GROUPS[condition])] = 0.0
        return result
    if condition in GAIN_CONDITIONS:
        gain = GAIN_CONDITIONS[condition]
        ratio = np.asarray(mean, dtype=np.float32).reshape(10, 1, 1) / np.asarray(
            std, dtype=np.float32
        ).reshape(10, 1, 1)
        transformed = gain * result + (gain - 1.0) * ratio
        return np.where(valid, transformed, 0.0).astype(np.float32)
    raise ValueError(f"Unknown Week 9 stress condition: {condition}")


class _StressDataset(Dataset):
    def __init__(self, base, condition: str, mean: np.ndarray, std: np.ndarray) -> None:
        self.base, self.condition, self.mean, self.std = base, condition, mean, std

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        image = apply_spectral_condition(item["image"], item["valid"], self.mean, self.std, self.condition)
        return torch.from_numpy(image), str(item["patch_id"])


def _infer(model, dataset, device: torch.device, batch_size: int, workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    head = model.fc
    model.fc = torch.nn.Identity()
    features, logits, patch_ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, ids in loader:
            values = model(images.to(device, non_blocking=True))
            features.append(values.float().cpu().numpy())
            logits.append(head(values).float().cpu().numpy())
            patch_ids.extend(map(str, ids))
    model.fc = head
    return np.concatenate(logits), np.concatenate(features), np.asarray(patch_ids, dtype=str)


def _save_prediction(path: Path, logits: np.ndarray, patch_ids: Sequence[str]) -> None:
    if logits.shape != (len(patch_ids), 19) or not np.isfinite(logits).all() or len(set(patch_ids)) != len(patch_ids):
        raise ValueError("Week 9 prediction artifact is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, logits=logits.astype(np.float32), patch_ids=np.asarray(patch_ids, dtype="U"))
    os.replace(temporary, path)


def centered_linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or len(left) != len(right) or len(left) < 2:
        raise ValueError("CKA requires aligned two-dimensional feature matrices")
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = left.T @ right
    numerator = float(np.square(cross).sum())
    denominator = math.sqrt(float(np.square(left.T @ left).sum()) * float(np.square(right.T @ right).sum()))
    if denominator <= 0:
        raise ValueError("CKA is undefined for collapsed features")
    return numerator / denominator


def covariance_effective_rank(features: np.ndarray, epsilon: float = 1e-12) -> float:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Effective rank requires a feature matrix")
    values -= values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(values, full_matrices=False, compute_uv=False)
    eigenvalues = np.square(singular) / (len(values) - 1)
    eigenvalues = eigenvalues[eigenvalues > float(epsilon)]
    if not len(eigenvalues):
        return 0.0
    probabilities = eigenvalues / eigenvalues.sum()
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def _nearest_cosine(
    query: np.ndarray, bank: np.ndarray, device: torch.device, k: int = 1,
    forbidden: Sequence[np.ndarray] | None = None, chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    q = torch.from_numpy(np.asarray(query, dtype=np.float32)).to(device)
    b = torch.from_numpy(np.asarray(bank, dtype=np.float32)).to(device)
    q = torch.nn.functional.normalize(q, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    all_values, all_indices = [], []
    for start in range(0, len(q), chunk_size):
        scores = q[start:start + chunk_size] @ b.T
        if forbidden is not None:
            for local, indices in enumerate(forbidden[start:start + len(scores)]):
                if len(indices):
                    scores[local, torch.as_tensor(indices, device=device, dtype=torch.long)] = -torch.inf
        values, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        if not torch.isfinite(values).all():
            raise ValueError("Nearest-neighbour exclusion left too few candidates")
        all_values.append(values.cpu().numpy())
        all_indices.append(indices.cpu().numpy())
    return np.concatenate(all_values), np.concatenate(all_indices)


def _label_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    first, second = set(map(str, left)), set(map(str, right))
    union = first | second
    return float(len(first & second) / len(union)) if union else 1.0


@functools.lru_cache(maxsize=None)
def _mgrs_tile_center(tile: str) -> tuple[float, float]:
    import mgrs

    latitude, longitude = mgrs.MGRS().toLatLon(str(tile).removeprefix("T") + "5000050000")
    return float(latitude), float(longitude)


def _geographic_distance_km(left, right) -> float:
    if str(left.mgrs_tile) == str(right.mgrs_tile):
        return float(1.2 * math.hypot(
            int(left.h_order) - int(right.h_order), int(left.v_order) - int(right.v_order)
        ))
    lat1, lon1 = map(math.radians, _mgrs_tile_center(str(left.mgrs_tile)))
    lat2, lon2 = map(math.radians, _mgrs_tile_center(str(right.mgrs_tile)))
    value = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return float(6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value))))


def _load_week8_clean(roots: Sequence[str | Path], seed: int, run_id: str) -> tuple[dict[str, object], Path]:
    summary_path = _find_unique(roots, f"week8_seed{seed}_summary.json")
    summary = _json(summary_path)
    matches = [run for run in summary["runs"] if run["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Week 8 output lacks clean run {run_id}")
    return matches[0], summary_path


def _copy_clean_prediction(
    roots: Sequence[str | Path], run: dict[str, object], partition: str, destination: Path
) -> tuple[np.ndarray, np.ndarray]:
    record = run["predictions"][partition]
    source = _find_run_artifact(
        roots, str(run["run_id"]), Path(str(record["path"])).name, str(record["sha256"])
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_sha256(destination) != record["sha256"]:
        raise ValueError("Copied Week 8 clean prediction changed bytes")
    with np.load(destination, allow_pickle=False) as payload:
        return payload["logits"], payload["patch_ids"].astype(str)


def _load_diagnostic_artifacts(roots: Sequence[str | Path], row):
    checkpoint = _find_run_artifact(roots, str(row.run_id), "best-model.pt", str(row.checkpoint_sha256))
    summary_path = _find_run_artifact(roots, str(row.run_id), "downstream_summary.json")
    summary = _json(summary_path)
    payload = verify_downstream_checkpoint(checkpoint, str(row.run_id))
    if summary.get("best_checkpoint_sha256") != row.checkpoint_sha256 or summary.get("evaluation_labels_loaded") is not False:
        raise ValueError(f"Frozen diagnostic run changed: {row.run_id}")
    return checkpoint, summary, payload


def _base(config: dict[str, object], partition: str):
    paths, settings = config["paths"], config["diagnostics"]
    return SpectraShiftDataset(
        paths["manifest_path"], paths["staged_root"], paths["normalization_path"], partition,
        "core10", int(settings["shard_size"]), int(settings["height"]), int(settings["width"]),
    )


def _align_features(features: np.ndarray, ids: np.ndarray, wanted: Sequence[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(map(str, ids))}
    if not set(wanted).issubset(lookup):
        raise ValueError("Feature output is missing frozen patch IDs")
    return features[[lookup[value] for value in wanted]]


def _forbidden_locations(query_frame: pd.DataFrame, bank_frame: pd.DataFrame) -> list[np.ndarray]:
    locations: dict[str, np.ndarray] = {
        str(name): indices.to_numpy(dtype=np.int64)
        for name, indices in bank_frame.groupby("location_key", sort=False).groups.items()
    }
    return [locations.get(str(value), np.empty(0, dtype=np.int64)) for value in query_frame["location_key"]]


def _make_retrieval_sheet(
    output: Path, query_ids: dict[str, list[str]], bases: dict[str, object], d_base,
    neighbors: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    selected = [value for partition in EVALUATION_PARTITIONS for value in query_ids[partition][:4]]
    q_index = {str(row.patch_id): index for partition, base in bases.items() for index, row in base.frame.iterrows()}
    q_partition = {str(row.patch_id): partition for partition, base in bases.items() for _, row in base.frame.iterrows()}
    d_index = {str(row.patch_id): index for index, row in d_base.frame.iterrows()}

    def preview(image):
        rgb = image[[2, 1, 0]].transpose(1, 2, 0)
        low, high = np.nanpercentile(rgb, [2, 98])
        return np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)

    figure, axes = plt.subplots(len(selected), 11, figsize=(16, 18), squeeze=False)
    for row_index, query_id in enumerate(selected):
        partition = q_partition[query_id]
        axes[row_index, 0].imshow(preview(bases[partition][q_index[query_id]]["image"]))
        axes[row_index, 0].set_title(f"Query {partition}", fontsize=6)
        for model_offset, model_id in enumerate(DIAGNOSTIC_MODELS):
            rows = neighbors[(neighbors.query_id == query_id) & (neighbors.model_id == model_id)].sort_values("rank")
            for rank, item in enumerate(rows.itertuples(index=False), start=1):
                column = model_offset * 5 + rank
                axes[row_index, column].imshow(preview(d_base[d_index[str(item.neighbor_id)]]["image"]))
                axes[row_index, column].set_title(f"{model_id} n{rank}", fontsize=6)
        for axis in axes[row_index]:
            axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle("Fixed seed-17 retrieval examples (illustrative only)")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def run_week9_diagnostics(
    config_path: str | Path, seed: int, input_roots: Sequence[str | Path], output_dir: str | Path
) -> dict[str, object]:
    if int(seed) not in WEEK9_SEEDS:
        raise ValueError("Week 9 seed must be 17, 29, or 43")
    config = yaml.safe_load(Path(config_path).read_text())
    paths, settings = config["paths"], config["diagnostics"]
    contract = _json(paths["week9_contract_path"])
    contracts_summary = _json(paths["week9_contracts_summary_path"])
    if (
        contract.get("status") != "frozen"
        or contract.get("model_selection_after_week8") is not False
        or file_sha256(paths["week9_contract_path"]) != contracts_summary["week9_contract_sha256"]
        or file_sha256(paths["diagnostic_ledger_path"]) != contract["diagnostic_ledger_sha256"]
        or file_sha256(paths["evaluation_labels_path"]) != contract["evaluation_labels_sha256"]
        or file_sha256(paths["support_contract_path"]) != contract["support_contract_sha256"]
        or file_sha256(paths["manifest_path"]) != contract["manifest_file_sha256"]
        or file_sha256(paths["normalization_path"]) != contract["normalization_file_sha256"]
    ):
        raise ValueError("Week 9 analysis contract is incomplete")
    ledger = pd.read_csv(paths["diagnostic_ledger_path"], dtype={"fraction_code": str})
    ledger["fraction_code"] = ledger["fraction_code"].str.zfill(2)
    rows = ledger[ledger["seed"].astype(int).eq(int(seed))]
    if len(rows) != 2 or set(rows["model_id"]) != set(DIAGNOSTIC_MODELS):
        raise ValueError("Diagnostic ledger must contain matching M3/M4 checkpoints")
    normalization = _json(paths["normalization_path"])
    band_indices = [list(BAND_ADAPTERS["core10"]).index(name) for name in BAND_ADAPTERS["core10"]]
    mean = np.asarray(normalization["mean"], dtype=np.float32)[band_indices]
    std = np.asarray(normalization["std"], dtype=np.float32)[band_indices]
    manifest = pd.read_parquet(paths["manifest_path"])
    labels = pd.read_parquet(paths["evaluation_labels_path"])
    metadata = pd.concat([manifest[manifest["partition"].isin(["D", "V"])], labels], ignore_index=True)
    metadata_by_id = metadata.set_index("patch_id", drop=False)
    query_ids = {key: list(map(str, value)) for key, value in contract["query_patch_ids"].items()}
    cka_ids = list(map(str, contract["cka_patch_ids"]))
    device = select_device(bool(settings.get("require_cuda", True)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    d_base, v_base = _base(config, "D"), _base(config, "V")
    eval_bases = {partition: _base(config, partition) for partition in EVALUATION_PARTITIONS}
    model_results, neighbor_frames, sensitivity_frames, distance_frames = [], [], [], []
    started = time.perf_counter()
    for row in rows.sort_values("model_id").itertuples(index=False):
        _, training_summary, payload = _load_diagnostic_artifacts(input_roots, row)
        model = build_resnet18(10, outputs=19)
        model.load_state_dict(payload["model"], strict=True)
        before_hash = model_state_sha256(model)
        model = model.to(device).eval()
        clean_run, _ = _load_week8_clean(input_roots, int(seed), str(row.run_id))
        if clean_run["checkpoint_sha256"] != row.checkpoint_sha256:
            raise ValueError("Week 8 clean run checkpoint does not match the diagnostic ledger")
        threshold = float(clean_run["source_v_threshold"]["threshold"])
        clean_dataset = lambda base: _StressDataset(base, "clean", mean, std)
        _, d_features, d_ids = _infer(model, clean_dataset(d_base), device, int(settings["batch_size"]), int(settings["num_workers"]))
        _, v_features, v_ids = _infer(model, clean_dataset(v_base), device, int(settings["batch_size"]), int(settings["num_workers"]))
        v1000 = _align_features(v_features, v_ids, cka_ids)
        feature_path = output_dir / str(row.run_id) / "v1000-features.npz"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        with feature_path.open("wb") as stream:
            np.savez_compressed(stream, features=v1000.astype(np.float32), patch_ids=np.asarray(cka_ids, dtype="U"))

        v_frame = metadata_by_id.loc[v_ids]
        d_frame = metadata_by_id.loc[d_ids]
        v_forbidden = _forbidden_locations(v_frame, d_frame)
        v_similarity, _ = _nearest_cosine(v_features, d_features, device, 1, v_forbidden)
        v_distances = 1.0 - v_similarity[:, 0]
        quartiles = np.quantile(v_distances, [0.25, 0.5, 0.75]).astype(float)
        predictions, cosine_records, clean_features_by_partition = {}, [], {}
        for partition in EVALUATION_PARTITIONS:
            clean_destination = output_dir / str(row.run_id) / f"predictions-{partition.lower()}-clean.npz"
            clean_logits, clean_ids = _copy_clean_prediction(input_roots, clean_run, partition, clean_destination)
            recomputed_logits, clean_features, inferred_ids = _infer(
                model, clean_dataset(eval_bases[partition]), device,
                int(settings["batch_size"]), int(settings["num_workers"]),
            )
            if set(clean_ids) != set(inferred_ids):
                raise ValueError("Clean diagnostic IDs differ from Week 8")
            order = {value: index for index, value in enumerate(inferred_ids)}
            recomputed_logits = recomputed_logits[[order[value] for value in clean_ids]]
            clean_features = clean_features[[order[value] for value in clean_ids]]
            if not np.allclose(clean_logits, recomputed_logits, rtol=1e-5, atol=1e-5):
                raise ValueError("Recomputed clean logits differ from frozen Week 8 predictions")
            clean_features_by_partition[partition] = (clean_features, clean_ids)
            predictions.setdefault("clean", {})[partition] = {
                "path": str(clean_destination), "sha256": file_sha256(clean_destination),
                "rows": len(clean_ids), "reused_week8_sha256": clean_run["predictions"][partition]["sha256"],
            }
            eval_frame = metadata_by_id.loc[clean_ids]
            forbidden = _forbidden_locations(eval_frame, d_frame)
            best_similarity, _ = _nearest_cosine(clean_features, d_features, device, 1, forbidden)
            distances = 1.0 - best_similarity[:, 0]
            bins = np.digitize(distances, quartiles, right=False) + 1
            distance_frames.append(pd.DataFrame({
                "patch_id": clean_ids, "partition": partition, "model_id": row.model_id,
                "seed": int(seed), "source_cosine_distance": distances,
                "source_distance_bin": [f"Q{value}" for value in bins],
            }))
            for condition in PERTURBED_CONDITIONS:
                logits, features, condition_ids = _infer(
                    model, _StressDataset(eval_bases[partition], condition, mean, std), device,
                    int(settings["batch_size"]), int(settings["num_workers"]),
                )
                condition_order = {value: index for index, value in enumerate(condition_ids)}
                logits = logits[[condition_order[value] for value in clean_ids]]
                features = features[[condition_order[value] for value in clean_ids]]
                path = output_dir / str(row.run_id) / f"predictions-{partition.lower()}-{condition}.npz"
                _save_prediction(path, logits, clean_ids)
                predictions.setdefault(condition, {})[partition] = {
                    "path": str(path), "sha256": file_sha256(path), "rows": len(clean_ids),
                }
                first = clean_features / np.maximum(np.linalg.norm(clean_features, axis=1, keepdims=True), 1e-12)
                second = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
                cosine_change = 1.0 - np.sum(first * second, axis=1)
                cosine_records.append(pd.DataFrame({
                    "patch_id": clean_ids, "partition": partition, "model_id": row.model_id,
                    "seed": int(seed), "condition": condition, "feature_cosine_change": cosine_change,
                }))

        query_feature_parts, query_id_order = [], []
        for partition in EVALUATION_PARTITIONS:
            features, ids = clean_features_by_partition[partition]
            query_feature_parts.append(_align_features(features, ids, query_ids[partition]))
            query_id_order.extend(query_ids[partition])
        query_features = np.concatenate(query_feature_parts)
        query_frame = metadata_by_id.loc[query_id_order]
        query_forbidden = _forbidden_locations(query_frame, d_frame)
        similarities, neighbor_indices = _nearest_cosine(
            query_features, d_features, device, int(config["sampling"]["nearest_neighbors"]), query_forbidden
        )
        nearest_rows = []
        for query_position, query_id in enumerate(query_id_order):
            query_row = metadata_by_id.loc[query_id]
            for rank, bank_index in enumerate(neighbor_indices[query_position], start=1):
                neighbor_id = str(d_ids[bank_index])
                neighbor_row = metadata_by_id.loc[neighbor_id]
                nearest_rows.append({
                    "query_id": query_id, "query_partition": str(query_row.partition),
                    "neighbor_id": neighbor_id, "rank": rank, "model_id": row.model_id,
                    "seed": int(seed), "cosine_similarity": float(similarities[query_position, rank - 1]),
                    "label_jaccard": _label_jaccard(query_row.labels, neighbor_row.labels),
                    "same_mgrs_tile": str(query_row.mgrs_tile) == str(neighbor_row.mgrs_tile),
                    "same_12km_block": str(query_row.get("block_12km", "")) == str(neighbor_row.get("block_12km", "x")),
                    "derived_geographic_distance_km": _geographic_distance_km(query_row, neighbor_row),
                })
        neighbor_frames.append(pd.DataFrame(nearest_rows))
        sensitivity_frames.extend(cosine_records)
        model = model.cpu()
        after_hash = model_state_sha256(model)
        if before_hash != after_hash:
            raise ValueError("Week 9 diagnostics modified a frozen encoder")
        model_results.append({
            "run_id": str(row.run_id), "model_id": str(row.model_id), "seed": int(seed),
            "fraction_code": "10", "checkpoint_sha256": str(row.checkpoint_sha256),
            "source_v_threshold": threshold, "feature_file": str(feature_path),
            "feature_sha256": file_sha256(feature_path), "feature_shape": list(v1000.shape),
            "source_distance_quartiles": quartiles.tolist(), "predictions": predictions,
            "encoder_state_sha256_before": before_hash, "encoder_state_sha256_after": after_hash,
            "preprocessing_sha256": clean_run["preprocessing"]["sha256"],
        })
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    neighbors = pd.concat(neighbor_frames, ignore_index=True)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    distances = pd.concat(distance_frames, ignore_index=True)
    neighbor_path = output_dir / f"nearest_neighbors_seed{seed}.parquet"
    sensitivity_path = output_dir / f"spectral_sensitivity_seed{seed}.parquet"
    distance_path = output_dir / f"source_distances_seed{seed}.parquet"
    neighbors.to_parquet(neighbor_path, index=False)
    sensitivity.to_parquet(sensitivity_path, index=False)
    distances.to_parquet(distance_path, index=False)
    retrieval_path = None
    if int(seed) == 17:
        retrieval_path = output_dir / "retrieval_examples_seed17.png"
        _make_retrieval_sheet(retrieval_path, query_ids, eval_bases, d_base, neighbors)
    result = {
        "week9_diagnostics_complete": True, "seed": int(seed),
        "diagnostic_checkpoint_count": len(model_results), "stress_condition_count": len(STRESS_CONDITIONS),
        "stress_prediction_domain_count": len(model_results) * len(STRESS_CONDITIONS) * 3,
        "new_stress_prediction_domain_count": len(model_results) * len(PERTURBED_CONDITIONS) * 3,
        "cka_patch_count": len(cka_ids), "nearest_neighbor_query_count": 100,
        "nearest_neighbor_row_count": len(neighbors), "runs": model_results,
        "nearest_neighbors_file": str(neighbor_path), "nearest_neighbors_sha256": file_sha256(neighbor_path),
        "spectral_sensitivity_file": str(sensitivity_path), "spectral_sensitivity_sha256": file_sha256(sensitivity_path),
        "source_distances_file": str(distance_path), "source_distances_sha256": file_sha256(distance_path),
        "retrieval_examples_file": str(retrieval_path) if retrieval_path else None,
        "retrieval_examples_sha256": file_sha256(retrieval_path) if retrieval_path else None,
        "elapsed_seconds": time.perf_counter() - started, "hardware": hardware_record(device),
        "source_tree_sha256": source_tree_sha256(), "git_commit": git_commit(),
        "model_selection_after_week8": False, "encoder_updates_during_week9": False,
        "evaluation_labels_loaded": True,
    }
    _write_json(output_dir / f"week9_diagnostics_seed{seed}_summary.json", result)
    return result


def _diagnostic_file(summary_path: Path, recorded: str, digest: str) -> Path:
    name = Path(recorded).name
    matches = list(summary_path.parent.rglob(name))
    valid = [path for path in matches if file_sha256(path) == digest]
    if len(valid) != 1:
        raise ValueError(f"Diagnostic artifact is missing or altered: {name}")
    return valid[0]


def _prediction_file(summary_path: Path, run: dict[str, object], condition: str, partition: str) -> Path:
    record = run["predictions"][condition][partition]
    return _diagnostic_file(summary_path, record["path"], record["sha256"])


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        logits, patch_ids = payload["logits"], payload["patch_ids"].astype(str)
    if logits.shape != (len(patch_ids), 19) or not np.isfinite(logits).all() or len(set(patch_ids)) != len(patch_ids):
        raise ValueError("Week 9 prediction failed shape, finite, or identity validation")
    return logits, patch_ids


def _metric_rows(
    run: dict[str, object], condition: str, partition: str, targets: np.ndarray, logits: np.ndarray,
    supported: Sequence[int], clean_map: float | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    probabilities = _probabilities(logits)
    threshold = float(run["source_v_threshold"])
    metrics = multilabel_metrics(targets, probabilities, threshold, supported)
    half = multilabel_metrics(targets, probabilities, 0.5, supported)
    supported_map = float(half["macro_average_precision"])
    row = {
        "run_id": run["run_id"], "model_id": run["model_id"], "seed": run["seed"],
        "condition": condition, "domain": DOMAIN_NAMES[partition], "sample_count": len(targets),
        "supported_map": supported_map,
        "degradation_from_clean": None if clean_map is None else float(clean_map - supported_map),
        "macro_f1_at_source_v_threshold": metrics["macro_f1"],
        "micro_f1_at_source_v_threshold": metrics["micro_f1"], "source_v_threshold": threshold,
    }
    class_rows = []
    positives = targets.sum(axis=0).astype(int)
    for index, name in enumerate(CANONICAL_LABELS):
        class_rows.append({
            "run_id": run["run_id"], "model_id": run["model_id"], "seed": run["seed"],
            "condition": condition, "domain": DOMAIN_NAMES[partition], "class_index": index,
            "class_name": name, "positive_count": int(positives[index]),
            "negative_count": int(len(targets) - positives[index]),
            "average_precision": _safe_ap(targets[:, index], probabilities[:, index]),
            "recall_at_source_v_threshold": metrics["per_class_recall"][index],
            "in_c_source": index in supported,
        })
    return row, class_rows


def _probe_tables(probes: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    for run in probes["linear_runs"] + probes["knn_runs"]:
        records.append({
            "run_id": run["run_id"], "model_id": run["model_id"], "probe_type": run["probe_type"],
            "fraction_code": str(run["fraction_code"]).zfill(2), "seed": int(run["seed"]),
            "sample_count": int(run["sample_count"]),
            "validation_supported_map": float(run["validation_macro_average_precision"]),
            "selected_hyperparameter": run.get("selected_l2", run.get("selected_k")),
            "feature_cache_sha256": run["feature_cache_sha256"],
        })
    frame = pd.DataFrame(records)
    comparisons = []
    contrasts = (("M3", "M1"), ("M3", "M2"), ("M4", "M3"))
    lookup = frame.set_index(["probe_type", "model_id", "fraction_code", "seed"])
    for probe_type in ("linear", "knn"):
        fractions = PROBE_FRACTIONS if probe_type == "linear" else ("10",)
        for left, right in contrasts:
            for fraction in fractions:
                values = []
                for seed in WEEK9_SEEDS:
                    values.append(float(
                        lookup.loc[(probe_type, left, fraction, seed), "validation_supported_map"]
                        - lookup.loc[(probe_type, right, fraction, seed), "validation_supported_map"]
                    ))
                comparisons.append({
                    "probe_type": probe_type, "left_model": left, "right_model": right,
                    "fraction_code": fraction, "paired_difference_mean": float(np.mean(values)),
                    "paired_difference_sample_std": float(np.std(values, ddof=1)),
                    "seed_differences": json.dumps(values),
                })
    aulc = []
    linear = frame[frame.probe_type.eq("linear")]
    for model in PROBE_MODELS:
        for seed in WEEK9_SEEDS:
            group = linear[(linear.model_id == model) & (linear.seed == seed)].copy()
            group["count"] = group.fraction_code.map(FRACTION_COUNTS)
            aulc.append({
                "record_type": "model_seed", "model_id": model, "seed": seed,
                "normalized_log_label_aulc": normalized_log_aulc(group["count"], group["validation_supported_map"]),
            })
    model_aulc = {(row["model_id"], row["seed"]): row["normalized_log_label_aulc"] for row in aulc}
    for left, right in contrasts:
        values = [model_aulc[(left, seed)] - model_aulc[(right, seed)] for seed in WEEK9_SEEDS]
        aulc.append({
            "record_type": "paired_contrast", "model_id": None, "seed": None,
            "left_model": left, "right_model": right,
            "paired_aulc_difference_mean": float(np.mean(values)),
            "paired_aulc_difference_sample_std": float(np.std(values, ddof=1)),
            "seed_differences": json.dumps(values),
        })
    return frame, pd.DataFrame(comparisons), pd.DataFrame(aulc)


def _slice_rows(
    run: dict[str, object], partition: str, targets: np.ndarray, logits: np.ndarray,
    metadata: pd.DataFrame, distances: pd.DataFrame, supported: Sequence[int],
) -> list[dict[str, object]]:
    probabilities = _probabilities(logits)
    threshold = float(run["source_v_threshold"])
    label_counts = targets.sum(axis=1)
    definitions = {
        "label_count": {
            "1": label_counts == 1, "2": label_counts == 2, "3": label_counts == 3, "4+": label_counts >= 4,
        },
        "quarter": {f"Q{value}": metadata["quarter"].to_numpy() == value for value in range(1, 5)},
        "country": {str(value): metadata["country"].astype(str).to_numpy() == str(value) for value in sorted(metadata["country"].unique())},
        "source_distance": {
            value: distances["source_distance_bin"].astype(str).to_numpy() == value for value in ("Q1", "Q2", "Q3", "Q4")
        },
    }
    rows = []
    for slice_type, values in definitions.items():
        for slice_value, mask in values.items():
            selected = np.flatnonzero(mask)
            if not len(selected):
                continue
            metric = multilabel_metrics(targets[selected], probabilities[selected], threshold, supported)
            positives = targets[selected].sum(axis=0).astype(int)
            for class_index, class_name in enumerate(CANONICAL_LABELS):
                rows.append({
                    "run_id": run["run_id"], "model_id": run["model_id"], "seed": run["seed"],
                    "domain": DOMAIN_NAMES[partition], "slice_type": slice_type, "slice_value": slice_value,
                    "sample_count": len(selected), "supported_map": metric["macro_average_precision"],
                    "class_index": class_index, "class_name": class_name,
                    "positive_count": int(positives[class_index]),
                    "negative_count": int(len(selected) - positives[class_index]),
                    "average_precision": _safe_ap(targets[selected, class_index], probabilities[selected, class_index]),
                    "recall_at_source_v_threshold": metric["per_class_recall"][class_index],
                })
    return rows


def _write_figures(
    output_dir: Path, probes: pd.DataFrame, cka: pd.DataFrame, rank: pd.DataFrame,
    stress: pd.DataFrame, stress_class: pd.DataFrame, nearest: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    root = output_dir / "figures"
    root.mkdir(exist_ok=True)
    linear = probes[probes.probe_type.eq("linear")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, group in linear.groupby("model_id"):
        summary = group.groupby("sample_count")["validation_supported_map"].agg(["mean", "std"]).sort_index()
        ax.errorbar(summary.index, summary["mean"], yerr=summary["std"], marker="o", label=model)
    ax.set_xscale("log"); ax.set_xlabel("Labeled D patches"); ax.set_ylabel("Source-V supported mAP"); ax.legend(ncol=3)
    fig.tight_layout(); fig.savefig(root / "probe_curves.png", dpi=180); plt.close(fig)

    labels = sorted(cka["left"].unique())
    matrix = cka.pivot(index="left", columns="right", values="cka").loc[labels, labels]
    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(10, 4))
    image = left_ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    left_ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); left_ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(image, ax=left_ax, fraction=.046)
    rank.sort_values(["model_id", "seed"]).plot.bar(x="checkpoint", y="effective_rank", ax=right_ax, legend=False)
    fig.tight_layout(); fig.savefig(root / "cka_effective_rank.png", dpi=180); plt.close(fig)

    perturbed = stress[stress.domain.ne("OOD") & stress.condition.ne("clean")]
    summary = perturbed.groupby(["condition", "model_id"])["degradation_from_clean"].mean().unstack()
    summary.plot.bar(figsize=(9, 4)); plt.ylabel("mAP degradation"); plt.tight_layout()
    plt.savefig(root / "stress_degradation.png", dpi=180); plt.close()

    removals = stress_class[stress_class.condition.str.startswith("missing_") & stress_class.in_c_source]
    grouped = removals.groupby(["condition", "model_id", "class_name"])["ap_degradation_from_clean"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in grouped.groupby("model_id"):
        ax.scatter(np.arange(len(group)), group["ap_degradation_from_clean"], s=8, alpha=.6, label=model)
    ax.set_ylabel("Per-class AP degradation from clean"); ax.legend(); fig.tight_layout()
    fig.savefig(root / "band_removal_class_ap.png", dpi=180); plt.close(fig)

    summary = nearest.groupby("model_id")[["label_jaccard", "derived_geographic_distance_km"]].mean(numeric_only=True)
    summary.plot.bar(subplots=True, figsize=(7, 6), legend=False); plt.tight_layout()
    plt.savefig(root / "nearest_neighbor_summary.png", dpi=180); plt.close()


def aggregate_week9(
    config_path: str | Path, probe_summary_path: str | Path,
    diagnostic_summary_paths: Sequence[str | Path], output_dir: str | Path,
) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    paths = config["paths"]
    contracts = _json(paths["week9_contracts_summary_path"])
    if not contracts.get("week9_contracts_complete"):
        raise ValueError("Week 9 contracts are incomplete")
    probes = _json(probe_summary_path)
    if (
        not probes.get("week9_probes_complete") or probes.get("linear_probe_count") != 108
        or probes.get("new_linear_probe_count") != 72 or probes.get("knn_probe_count") != 18
        or probes.get("new_knn_probe_count") != 12 or probes.get("encoder_updates_during_week9") is not False
    ):
        raise ValueError("Week 9 probe matrix is incomplete")
    if len(diagnostic_summary_paths) != 3:
        raise ValueError("Week 9 aggregation requires three diagnostic seed outputs")
    summary_paths = [Path(path) for path in diagnostic_summary_paths]
    summaries = [_json(path) for path in summary_paths]
    if {int(value["seed"]) for value in summaries} != set(WEEK9_SEEDS):
        raise ValueError("Week 9 diagnostic summaries must cover seeds 17, 29, and 43")
    if any(
        not value.get("week9_diagnostics_complete") or value.get("diagnostic_checkpoint_count") != 2
        or value.get("stress_prediction_domain_count") != 36
        or value.get("new_stress_prediction_domain_count") != 30
        or value.get("nearest_neighbor_row_count") != 1000
        or value.get("encoder_updates_during_week9") is not False
        for value in summaries
    ):
        raise ValueError("A Week 9 diagnostic output failed its immutable gate")
    labels = pd.read_parquet(paths["evaluation_labels_path"])
    support = _json(paths["support_contract_path"])
    supported = list(map(int, support["source_supported_indices"]))
    common = list(map(int, support["common_source_indices"]))
    labels_by_id = labels.set_index("patch_id", drop=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_table, probe_differences, probe_aulc = _probe_tables(probes)
    probe_table.to_csv(output_dir / "representation_probes.csv", index=False)
    probe_differences.to_csv(output_dir / "probe_paired_differences.csv", index=False)
    probe_aulc.to_csv(output_dir / "probe_aulc.csv", index=False)

    runs, path_by_seed = [], {int(summary["seed"]): path for summary, path in zip(summaries, summary_paths, strict=True)}
    feature_records = []
    nearest_frames, sensitivity_frames, distance_frames = [], [], []
    for summary, summary_path in zip(summaries, summary_paths, strict=True):
        nearest_frames.append(pd.read_parquet(_diagnostic_file(
            summary_path, summary["nearest_neighbors_file"], summary["nearest_neighbors_sha256"]
        )))
        sensitivity_frames.append(pd.read_parquet(_diagnostic_file(
            summary_path, summary["spectral_sensitivity_file"], summary["spectral_sensitivity_sha256"]
        )))
        distance_frames.append(pd.read_parquet(_diagnostic_file(
            summary_path, summary["source_distances_file"], summary["source_distances_sha256"]
        )))
        for run in summary["runs"]:
            feature_path = _diagnostic_file(summary_path, run["feature_file"], run["feature_sha256"])
            with np.load(feature_path, allow_pickle=False) as payload:
                feature_records.append((run, payload["features"].astype(np.float64), payload["patch_ids"].astype(str)))
            runs.append(run)
    if len(runs) != 6 or len({run["run_id"] for run in runs}) != 6:
        raise ValueError("Week 9 aggregate requires six unique diagnostic runs")
    canonical_ids = feature_records[0][2]
    if len(canonical_ids) != 1000 or any(not np.array_equal(ids, canonical_ids) for _, _, ids in feature_records):
        raise ValueError("CKA features do not use the same fixed 1,000 V patches")
    names = [f"{run['model_id']}-s{run['seed']}" for run, _, _ in feature_records]
    cka_rows = []
    for left_index, left_name in enumerate(names):
        for right_index, right_name in enumerate(names):
            cka_rows.append({
                "left": left_name, "right": right_name,
                "cka": centered_linear_cka(feature_records[left_index][1], feature_records[right_index][1]),
            })
    cka = pd.DataFrame(cka_rows)
    rank = pd.DataFrame([{
        "checkpoint": name, "model_id": run["model_id"], "seed": run["seed"],
        "effective_rank": covariance_effective_rank(features, float(config["diagnostics"]["effective_rank_epsilon"])),
    } for name, (run, features, _) in zip(names, feature_records, strict=True)])
    cka.to_csv(output_dir / "cka_matrix.csv", index=False)
    rank.to_csv(output_dir / "effective_rank.csv", index=False)

    metric_rows, class_rows, prediction_lookup = [], [], {}
    for run in runs:
        summary_path = path_by_seed[int(run["seed"])]
        clean_maps = {}
        for partition in EVALUATION_PARTITIONS:
            logits, patch_ids = _load_prediction(_prediction_file(summary_path, run, "clean", partition))
            target = _encode_labels(labels_by_id.loc[patch_ids, "labels"])
            row, per_class = _metric_rows(run, "clean", partition, target, logits, supported, None)
            clean_maps[partition] = row["supported_map"]
            metric_rows.append(row); class_rows.extend(per_class)
            prediction_lookup[(run["model_id"], int(run["seed"]), "clean", partition)] = (_probabilities(logits), patch_ids)
        for condition in PERTURBED_CONDITIONS:
            for partition in EVALUATION_PARTITIONS:
                logits, patch_ids = _load_prediction(_prediction_file(summary_path, run, condition, partition))
                target = _encode_labels(labels_by_id.loc[patch_ids, "labels"])
                row, per_class = _metric_rows(run, condition, partition, target, logits, supported, clean_maps[partition])
                metric_rows.append(row); class_rows.extend(per_class)
                prediction_lookup[(run["model_id"], int(run["seed"]), condition, partition)] = (_probabilities(logits), patch_ids)
    stress = pd.DataFrame(metric_rows)
    stress_class = pd.DataFrame(class_rows)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    sensitivity_summary = sensitivity.groupby(
        ["model_id", "seed", "condition", "partition"], as_index=False
    )["feature_cosine_change"].mean().rename(columns={"feature_cosine_change": "mean_feature_cosine_change"})
    sensitivity_summary["domain"] = sensitivity_summary["partition"].map(DOMAIN_NAMES)
    stress = stress.merge(
        sensitivity_summary.drop(columns="partition"),
        on=["model_id", "seed", "condition", "domain"], how="left", validate="many_to_one",
    )
    stress.loc[stress["condition"].eq("clean"), "mean_feature_cosine_change"] = 0.0
    clean_ap = stress_class[stress_class.condition.eq("clean")].set_index(
        ["run_id", "domain", "class_index"]
    )["average_precision"]
    stress_class["ap_degradation_from_clean"] = [
        float(clean_ap.loc[(row.run_id, row.domain, row.class_index)] - row.average_precision)
        if row.condition != "clean" else (0.0 if np.isfinite(row.average_precision) else float("nan"))
        for row in stress_class.itertuples(index=False)
    ]
    ood_rows = []
    for (run_id, model_id, seed, condition), group in stress.groupby(["run_id", "model_id", "seed", "condition"]):
        values = group.set_index("domain")["supported_map"]
        clean_values = group.set_index("domain")["degradation_from_clean"]
        ood_rows.append({
            "run_id": run_id, "model_id": model_id, "seed": seed, "condition": condition,
            "domain": "OOD", "sample_count": 8000,
            "supported_map": equal_country_ood(values["Finland"], values["Portugal"]),
            "degradation_from_clean": None if condition == "clean" else equal_country_ood(
                clean_values["Finland"], clean_values["Portugal"]
            ),
            "mean_feature_cosine_change": 0.0 if condition == "clean" else equal_country_ood(
                group.set_index("domain").loc["Finland", "mean_feature_cosine_change"],
                group.set_index("domain").loc["Portugal", "mean_feature_cosine_change"],
            ),
            "macro_f1_at_source_v_threshold": np.nan, "micro_f1_at_source_v_threshold": np.nan,
            "source_v_threshold": group["source_v_threshold"].iloc[0],
        })
    stress = pd.concat([stress, pd.DataFrame(ood_rows)], ignore_index=True)
    stress.to_csv(output_dir / "stress_metrics.csv", index=False)
    stress_class.to_csv(output_dir / "stress_per_class.csv", index=False)

    contract = _json(paths["week9_contract_path"])
    replicate_count, bootstrap_seed = int(config["uncertainty"]["bootstrap_replicates"]), int(config["uncertainty"]["bootstrap_seed"])
    draws = {}
    for partition in EVALUATION_PARTITIONS:
        frame = labels[labels["partition"].eq(partition)].reset_index(drop=True)
        for group_name in ("mgrs_group", "block_12km"):
            offset = int(hashlib.sha256(f"week9:{partition}:{group_name}".encode()).hexdigest()[:8], 16)
            draws[(partition, group_name)] = _bootstrap_draws(
                frame[group_name].astype(str).to_numpy(), replicate_count, bootstrap_seed + offset
            )
    bootstrap_rows = []
    for condition in STRESS_CONDITIONS:
        for group_name in ("mgrs_group", "block_12km"):
            samples_by_domain = {name: [] for name in ("I", "Finland", "Portugal", "OOD")}
            observed = {name: [] for name in samples_by_domain}
            for seed in WEEK9_SEEDS:
                per_seed = {}
                for partition in EVALUATION_PARTITIONS:
                    frame = labels[labels["partition"].eq(partition)].reset_index(drop=True)
                    target = _encode_labels(frame["labels"])
                    ids = frame["patch_id"].astype(str).to_numpy()
                    def aligned(model_id):
                        probability, patch_ids = prediction_lookup[(model_id, seed, condition, partition)]
                        lookup = {value: index for index, value in enumerate(patch_ids)}
                        return probability[[lookup[value] for value in ids]]
                    left, right = aligned("M4"), aligned("M3")
                    values = _paired_bootstrap_difference(target, left, right, draws[(partition, group_name)], common)
                    domain = DOMAIN_NAMES[partition]
                    per_seed[domain] = values; samples_by_domain[domain].append(values)
                    observed[domain].append(_supported_map(target, left, common) - _supported_map(target, right, common))
                samples_by_domain["OOD"].append((per_seed["Finland"] + per_seed["Portugal"]) / 2)
                observed["OOD"].append(equal_country_ood(observed["Finland"][-1], observed["Portugal"][-1]))
            for domain in samples_by_domain:
                values = np.mean(np.stack(samples_by_domain[domain]), axis=0)
                bootstrap_rows.append({
                    "condition": condition, "left_model": "M4", "right_model": "M3", "domain": domain,
                    "grouping": "MGRS" if group_name == "mgrs_group" else "12km-block",
                    "observed_paired_difference": float(np.mean(observed[domain])),
                    "bootstrap_mean": float(np.nanmean(values)),
                    "percentile_2_5": float(np.nanpercentile(values, 2.5)),
                    "percentile_97_5": float(np.nanpercentile(values, 97.5)),
                    "replicates": replicate_count, "p_value": None,
                })
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "stress_bootstrap.csv", index=False)

    nearest = pd.concat(nearest_frames, ignore_index=True)
    distances = pd.concat(distance_frames, ignore_index=True)
    nearest.to_parquet(output_dir / "nearest_neighbors.parquet", index=False)
    sensitivity.to_parquet(output_dir / "spectral_sensitivity.parquet", index=False)
    error_rows = []
    run_lookup = {(run["model_id"], int(run["seed"])): run for run in runs}
    for (model_id, seed, partition), distance_frame in distances.groupby(["model_id", "seed", "partition"], sort=True):
        run = run_lookup[(model_id, int(seed))]
        probabilities, prediction_ids = prediction_lookup[(model_id, int(seed), "clean", partition)]
        logits = np.log(np.clip(probabilities, 1e-12, 1 - 1e-12) / np.clip(1 - probabilities, 1e-12, 1))
        order = {value: index for index, value in enumerate(prediction_ids)}
        distance_frame = distance_frame.set_index("patch_id").loc[prediction_ids].reset_index()
        metadata = labels_by_id.loc[prediction_ids]
        target = _encode_labels(metadata["labels"])
        error_rows.extend(_slice_rows(run, partition, target, logits, metadata, distance_frame, supported))
    errors = pd.DataFrame(error_rows)
    errors.to_csv(output_dir / "error_slices.csv", index=False)
    _write_figures(output_dir, probe_table, cka, rank, stress, stress_class, nearest)

    outputs = [
        "representation_probes.csv", "probe_paired_differences.csv", "probe_aulc.csv",
        "cka_matrix.csv", "effective_rank.csv", "stress_metrics.csv", "stress_per_class.csv",
        "stress_bootstrap.csv", "spectral_sensitivity.parquet", "nearest_neighbors.parquet", "error_slices.csv",
    ]
    result = {
        "week9_complete": True, "week10_approved": True,
        "linear_probe_count": 108, "new_linear_probe_count": 72,
        "knn_probe_count": 18, "new_knn_probe_count": 12,
        "diagnostic_checkpoint_count": 6, "stress_condition_count": 6,
        "stress_prediction_domain_count": 108, "new_stress_prediction_domain_count": 90,
        "cka_patch_count": len(canonical_ids), "nearest_neighbor_query_count": 100,
        "nearest_neighbor_row_count": len(nearest), "bootstrap_replicates": replicate_count,
        "bootstrap_seed": bootstrap_seed, "output_hashes": {name: file_sha256(output_dir / name) for name in outputs},
        "week8_summary_sha256": contract["week8_summary_sha256"],
        "week9_contract_sha256": file_sha256(paths["week9_contract_path"]),
        "week9_probe_summary_sha256": file_sha256(probe_summary_path),
        "model_selection_after_week8": False, "encoder_updates_during_week9": False,
        "evaluation_labels_loaded": True, "week10_implemented": False,
    }
    _write_json(output_dir / "week9_run_summary.json", result)
    return result
