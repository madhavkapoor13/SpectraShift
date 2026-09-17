from __future__ import annotations

import json
import math
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from spectrashift.data.downstream import FRACTION_COUNTS, WEEK5_SEEDS, freeze_downstream_subsets

from .common import file_sha256, object_sha256
from .downstream import (
    MODEL_ADAPTERS,
    cache_prefinetune_features,
    expected_epochs_and_steps,
    train_downstream,
    verify_downstream_checkpoint,
)


WEEK5_MODELS = ("M0", "M1", "M2", "M3", "M4")
FINAL_FRACTIONS = ("01", "10", "100")
PILOT_MULTIPLIERS = (0.3, 1.0, 3.0)


def validate_week4_approval(path: str | Path, expected_sha256: str) -> dict[str, object]:
    path = Path(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Week 4 aggregate hash differs from the approved artifact")
    summary = json.loads(path.read_text())
    if not summary.get("week4_complete") or not summary.get("week5_approved"):
        raise ValueError("Week 4 aggregate does not approve Week 5")
    if summary.get("run_count") != 9 or len(summary.get("encoder_sha256", {})) != 9:
        raise ValueError("Week 4 aggregate does not contain nine encoders")
    if len(set(summary["encoder_sha256"].values())) != 9:
        raise ValueError("Week 4 encoders are not unique")
    if summary.get("evaluation_labels_loaded") is not False:
        raise ValueError("Week 4 aggregate does not prove evaluation-label isolation")
    return summary


def prepare_week5_contracts(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    contracts = config["contracts"]
    output_dir = Path(contracts["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    week4 = validate_week4_approval(
        contracts["week4_summary_path"], contracts["week4_summary_sha256"]
    )
    subset = freeze_downstream_subsets(config_path)
    weights = Path(contracts["imagenet_weights_path"])
    if not weights.exists():
        temporary = weights.with_suffix(weights.suffix + ".download")
        urllib.request.urlretrieve(str(contracts["imagenet_url"]), temporary)
        temporary.replace(weights)
    weights_hash = file_sha256(weights)
    if not weights_hash.startswith(str(contracts["imagenet_sha256_prefix"])):
        raise ValueError("Official ImageNet ResNet-18 checkpoint hash prefix mismatch")
    result = {
        "week5_contracts_complete": True,
        "week4_summary_sha256": file_sha256(contracts["week4_summary_path"]),
        "week4_encoder_sha256": week4["encoder_sha256"],
        "subset_manifest_sha256": subset["subset_manifest_sha256"],
        "supported_class_indices": subset["supported_class_indices"],
        "supported_class_names": subset["supported_class_names"],
        "imagenet_weights_file": weights.name,
        "imagenet_weights_sha256": weights_hash,
        "evaluation_labels_loaded": False,
    }
    (output_dir / "week5_contracts_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _multiplier_code(value: float) -> str:
    return str(value).replace(".", "p")


def _base_encoder_lr(model_id: str, training: dict[str, object]) -> float:
    return float(
        training["encoder_learning_rate_scratch"]
        if model_id == "M0" else training["encoder_learning_rate_pretrained"]
    )


def resolved_run_config(
    base: dict[str, object],
    model_id: str,
    seed: int,
    fraction_code: str,
    multiplier: float,
    output_root: str | Path,
    contracts_summary_path: str | Path,
    encoder_paths: dict[str, str | Path],
    pilot: bool,
) -> dict[str, object]:
    if model_id not in WEEK5_MODELS or seed not in WEEK5_SEEDS:
        raise ValueError("Unexpected Week 5 model or seed")
    if fraction_code not in FRACTION_COUNTS:
        raise ValueError("Unexpected Week 5 label fraction")
    if pilot and (seed != 17 or fraction_code != "10" or multiplier not in PILOT_MULTIPLIERS):
        raise ValueError("Pilot identity violates the frozen Week 5 grid")
    if not pilot and fraction_code not in FINAL_FRACTIONS:
        raise ValueError("Final Week 5 fraction is not an anchor fraction")
    contracts = json.loads(Path(contracts_summary_path).read_text())
    training_base = base["training"]
    epochs = int(base["pilot"]["epochs"] if pilot else training_base["base_epochs"])
    minimum_steps = 0 if pilot else int(training_base["minimum_optimizer_steps"])
    sample_count = int(FRACTION_COUNTS[fraction_code])
    total_epochs, _, expected_steps = expected_epochs_and_steps(
        sample_count, int(training_base["batch_size"]), epochs, minimum_steps
    )
    if pilot:
        run_id = f"week5-pilot-{model_id.lower()}-lr{_multiplier_code(multiplier)}-seed17"
    else:
        run_id = f"week5-{model_id.lower()}-f{fraction_code}-seed{seed}"
    output_dir = Path(output_root) / run_id
    data = dict(base["data"])
    data["adapter"] = MODEL_ADAPTERS[model_id]
    initialization: dict[str, object]
    if model_id == "M0":
        initialization = {"type": "random"}
    elif model_id == "M1":
        initialization = {
            "type": "imagenet1k-v1-core10",
            "path": str(Path(contracts_summary_path).parent / contracts["imagenet_weights_file"]),
            "sha256": contracts["imagenet_weights_sha256"],
            "sha256_prefix": "f37072fd",
        }
    else:
        key = f"week4-{model_id.lower()}-seed{seed}"
        if key not in encoder_paths:
            raise ValueError(f"Missing matching Week 4 encoder for {key}")
        initialization = {
            "type": "week4-encoder",
            "path": str(encoder_paths[key]),
            "sha256": contracts["week4_encoder_sha256"][key],
        }
    encoder_lr = _base_encoder_lr(model_id, training_base) * float(multiplier)
    head_lr = float(training_base["head_learning_rate"]) * float(multiplier)
    return {
        "run": {
            "id": run_id, "model_id": model_id, "seed": int(seed),
            "fraction_code": fraction_code, "sample_count": sample_count,
            "pilot": bool(pilot), "learning_rate_multiplier": float(multiplier),
            "output_dir": str(output_dir),
        },
        "data": data,
        "subsets": {
            "manifest_path": str(Path(contracts_summary_path).parent / "downstream_subsets.parquet"),
            "contract_path": str(Path(contracts_summary_path).parent / "downstream_contract.json"),
            "sha256": contracts["subset_manifest_sha256"],
        },
        "initialization": initialization,
        "training": {
            "batch_size": int(training_base["batch_size"]),
            "base_epochs": total_epochs,
            "minimum_optimizer_steps": 0,
            "expected_optimizer_steps": expected_steps,
            "encoder_learning_rate": encoder_lr,
            "head_learning_rate": head_lr,
            "weight_decay": float(training_base["weight_decay"]),
            "warmup_fraction": float(training_base["warmup_fraction"]),
            "minimum_encoder_learning_rate": min(
                float(training_base["minimum_encoder_learning_rate"]), encoder_lr
            ),
            "gradient_clip": float(training_base["gradient_clip"]),
            "amp": bool(training_base["amp"]),
            "amp_initial_scale": float(training_base["amp_initial_scale"]),
            "amp_growth_interval": int(training_base["amp_growth_interval"]),
            "fail_on_amp_overflow": bool(training_base["fail_on_amp_overflow"]),
            "num_workers": int(training_base["num_workers"]),
            "require_cuda": bool(training_base["require_cuda"]),
            "validation_batch_size": 256,
            "evaluation_every_epochs": 1 if pilot or fraction_code != "01" else 5,
            "log_every_steps": int(training_base["log_every_steps"]),
        },
    }


def _write_runtime_config(config: dict[str, object], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{config['run']['id']}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _history_map(path: Path, epoch: int) -> float:
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if int(record["epoch"]) == epoch and "validation_macro_average_precision" in record:
            return float(record["validation_macro_average_precision"])
    raise ValueError(f"Pilot log has no validation result for epoch {epoch}")


def run_week5_pilots(
    config_path: str | Path,
    output_root: str | Path,
    contracts_summary_path: str | Path,
    encoder_paths: dict[str, str | Path],
) -> dict[str, object]:
    base = yaml.safe_load(Path(config_path).read_text())
    output_root = Path(output_root)
    configs_dir = output_root / "configs"
    summaries = []
    for model_id in WEEK5_MODELS:
        for multiplier in PILOT_MULTIPLIERS:
            config = resolved_run_config(
                base, model_id, 17, "10", multiplier, output_root,
                contracts_summary_path, encoder_paths, pilot=True,
            )
            path = _write_runtime_config(config, configs_dir)
            summaries.append(train_downstream(path))
    selected = {}
    blocked = []
    tolerance = float(base["pilot"]["tie_tolerance_map"])
    for model_id in WEEK5_MODELS:
        candidates = [item for item in summaries if item["model_id"] == model_id]
        best = max(float(item["validation_macro_average_precision"]) for item in candidates)
        eligible = [item for item in candidates if best - float(item["validation_macro_average_precision"]) < tolerance]
        default = [item for item in eligible if math.isclose(float(item["run_id"].split("-lr")[1].split("-")[0].replace("p", ".")), 1.0)]
        chosen = default[0] if default else min(
            eligible,
            key=lambda item: float(item["run_id"].split("-lr")[1].split("-")[0].replace("p", ".")),
        )
        multiplier = float(chosen["run_id"].split("-lr")[1].split("-")[0].replace("p", "."))
        log_path = Path(chosen["best_checkpoint"]).parent / "training.jsonl"
        final_epoch = int(base["pilot"]["epochs"])
        reference_epoch = int(base["pilot"]["undertraining_reference_epoch"])
        gain = _history_map(log_path, final_epoch) - _history_map(log_path, reference_epoch)
        undertrained = int(chosen["best_epoch"]) == final_epoch and gain > float(
            base["pilot"]["undertraining_gain_threshold"]
        )
        selected[model_id] = {
            "run_id": chosen["run_id"], "learning_rate_multiplier": multiplier,
            "validation_macro_average_precision": chosen["validation_macro_average_precision"],
            "best_epoch": chosen["best_epoch"], "epoch_12_to_15_gain": gain,
            "undertraining_gate": not undertrained,
        }
        if undertrained:
            blocked.append(model_id)
    for summary in summaries:
        run_dir = Path(summary["best_checkpoint"]).parent
        (run_dir / "best-model.pt").unlink(missing_ok=True)
        (run_dir / "validation_predictions.npz").unlink(missing_ok=True)
        summary["pilot_weights_pruned"] = True
        (run_dir / "downstream_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    result = {
        "week5_pilots_complete": True,
        "week5_final_approved": not blocked,
        "run_count": 15,
        "selected": selected,
        "blocked_models": blocked,
        "actual_gpu_hours": sum(float(item["elapsed_seconds"]) for item in summaries) / 3600,
        "evaluation_labels_loaded": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "week5_pilot_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate_pilot_approval(path: str | Path) -> dict[str, object]:
    summary = json.loads(Path(path).read_text())
    if not summary.get("week5_pilots_complete") or not summary.get("week5_final_approved"):
        raise ValueError("Week 5 pilots do not approve the final anchor matrix")
    if set(summary.get("selected", {})) != set(WEEK5_MODELS) or summary.get("run_count") != 15:
        raise ValueError("Week 5 pilot selection is incomplete")
    if summary.get("evaluation_labels_loaded") is not False:
        raise ValueError("Week 5 pilot provenance is label-unsafe")
    return summary


def _artifact_path(summary_path: Path, recorded: str) -> Path:
    return summary_path.parent / Path(recorded).name


def validate_complete_downstream_run(summary_path: str | Path, config: dict[str, object]) -> dict[str, object]:
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    if summary.get("run_id") != config["run"]["id"] or summary.get("config_sha256") != object_sha256(config):
        raise ValueError("Downstream run identity or configuration hash mismatch")
    if summary.get("steps") != config["training"]["expected_optimizer_steps"]:
        raise ValueError("Downstream run has the wrong exact step count")
    if summary.get("amp_overflow_skips") != 0 or not summary.get("completion_gate"):
        raise ValueError("Downstream run failed completion gates")
    checkpoint = _artifact_path(summary_path, summary["best_checkpoint"])
    predictions = _artifact_path(summary_path, summary["validation_predictions"])
    if file_sha256(checkpoint) != summary["best_checkpoint_sha256"]:
        raise ValueError("Downstream checkpoint hash mismatch")
    if file_sha256(predictions) != summary["validation_predictions_sha256"]:
        raise ValueError("Downstream V-prediction hash mismatch")
    verify_downstream_checkpoint(checkpoint, summary["run_id"])
    with np.load(predictions, allow_pickle=False) as payload:
        if payload["logits"].shape != (2000, 19) or len(payload["patch_ids"]) != 2000:
            raise ValueError("Downstream V predictions have the wrong shape")
        if not np.isfinite(payload["logits"]).all():
            raise ValueError("Downstream V predictions contain non-finite logits")
    if summary.get("evaluation_labels_loaded") is not False:
        raise ValueError("Downstream run does not prove evaluation-label isolation")
    return summary


def _find_completed(roots: list[Path], run_id: str, config: dict[str, object]):
    valid = []
    for root in roots:
        for path in root.rglob(f"{run_id}/downstream_summary.json"):
            try:
                valid.append((path, validate_complete_downstream_run(path, config)))
            except (KeyError, OSError, RuntimeError, ValueError):
                continue
    if len({item[1]["best_checkpoint_sha256"] for item in valid}) > 1:
        raise ValueError(f"Conflicting completed artifacts for {run_id}")
    return valid[0] if valid else None


def _find_recovery(roots: list[Path], run_id: str, config: dict[str, object]):
    expected = object_sha256(config)
    for root in roots:
        for path in sorted(root.rglob(f"{run_id}/recovery-latest.pt"), reverse=True):
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("config_sha256") == expected:
                return path
    return None


def run_week5_seed(
    config_path: str | Path,
    seed: int,
    output_root: str | Path,
    contracts_summary_path: str | Path,
    pilot_summary_path: str | Path,
    encoder_paths: dict[str, str | Path],
    resume_roots: list[str | Path] | None = None,
) -> dict[str, object]:
    base = yaml.safe_load(Path(config_path).read_text())
    pilot = validate_pilot_approval(pilot_summary_path)
    output_root = Path(output_root)
    roots = [Path(value) for value in (resume_roots or [])]
    summaries = []
    feature_records = []
    for model_id in WEEK5_MODELS:
        multiplier = float(pilot["selected"][model_id]["learning_rate_multiplier"])
        for fraction in FINAL_FRACTIONS:
            config = resolved_run_config(
                base, model_id, int(seed), fraction, multiplier, output_root,
                contracts_summary_path, encoder_paths, pilot=False,
            )
            path = _write_runtime_config(config, output_root / "configs")
            completed = _find_completed(roots, config["run"]["id"], config)
            if completed:
                shutil.copytree(completed[0].parent, Path(config["run"]["output_dir"]), dirs_exist_ok=True)
                summary = validate_complete_downstream_run(
                    Path(config["run"]["output_dir"]) / "downstream_summary.json", config
                )
            else:
                recovery = _find_recovery(roots, config["run"]["id"], config)
                if recovery:
                    destination = Path(config["run"]["output_dir"])
                    shutil.copytree(recovery.parent, destination, dirs_exist_ok=True)
                    recovery = destination / recovery.name
                summary = train_downstream(path, resume_path=recovery)
                summary = validate_complete_downstream_run(
                    Path(config["run"]["output_dir"]) / "downstream_summary.json", config
                )
            summaries.append(summary)
        if model_id != "M0":
            feature_config = resolved_run_config(
                base, model_id, int(seed), "100", multiplier, output_root,
                contracts_summary_path, encoder_paths, pilot=False,
            )
            feature_config_path = _write_runtime_config(feature_config, output_root / "feature-configs")
            feature_records.append(cache_prefinetune_features(
                feature_config_path, output_root / "features" / f"{model_id.lower()}-seed{seed}.npz"
            ))
    result = {
        "week5_seed_complete": True, "seed": int(seed), "run_count": 15,
        "runs": summaries, "feature_caches": feature_records,
        "training_gpu_hours": sum(float(item["elapsed_seconds"]) for item in summaries) / 3600,
        "feature_gpu_hours": sum(float(item["elapsed_seconds"]) for item in feature_records) / 3600,
        "actual_gpu_hours": (
            sum(float(item["elapsed_seconds"]) for item in summaries)
            + sum(float(item["elapsed_seconds"]) for item in feature_records)
        ) / 3600,
        "pilot_summary_sha256": file_sha256(pilot_summary_path),
        "evaluation_labels_loaded": False,
    }
    (output_root / f"week5_seed{seed}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def aggregate_week5(
    seed_summary_paths: list[str | Path],
    pilot_summary_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    if len(seed_summary_paths) != 3:
        raise ValueError("Week 5 aggregation requires exactly three seed summaries")
    pilot = validate_pilot_approval(pilot_summary_path)
    paths = [Path(path) for path in seed_summary_paths]
    seeds = [json.loads(path.read_text()) for path in paths]
    if {int(item["seed"]) for item in seeds} != set(WEEK5_SEEDS):
        raise ValueError("Week 5 summaries must cover seeds 17, 29, and 43")
    if any(not item.get("week5_seed_complete") or item.get("evaluation_labels_loaded") is not False for item in seeds):
        raise ValueError("A Week 5 seed job is incomplete or label-unsafe")
    pilot_hash = file_sha256(pilot_summary_path)
    if {item.get("pilot_summary_sha256") for item in seeds} != {pilot_hash}:
        raise ValueError("Week 5 seed jobs do not share the supplied pilot decision")
    runs = [run for item in seeds for run in item["runs"]]
    expected = {(model, fraction, seed) for model in WEEK5_MODELS for fraction in FINAL_FRACTIONS for seed in WEEK5_SEEDS}
    observed = {(run["model_id"], run["fraction_code"], int(run["seed"])) for run in runs}
    if len(runs) != 45 or observed != expected:
        raise ValueError("Week 5 aggregation has missing or duplicate final runs")
    if len({run["best_checkpoint_sha256"] for run in runs}) != 45:
        raise ValueError("Week 5 final checkpoints are not unique")
    expected_steps = {"01": 300, "10": 570, "100": 5640}
    if any(
        run.get("steps") != expected_steps[run["fraction_code"]]
        or run.get("expected_optimizer_steps") != expected_steps[run["fraction_code"]]
        or run.get("amp_overflow_skips") != 0
        or not run.get("finite_gate")
        or not run.get("completion_gate")
        for run in runs
    ):
        raise ValueError("At least one Week 5 run failed its step or stability contract")
    if len({run.get("source_tree_sha256") for run in runs}) != 1:
        raise ValueError("Week 5 runs were produced by different source trees")
    for path, seed_summary in zip(paths, seeds, strict=True):
        for run in seed_summary["runs"]:
            run_dir = path.parent / run["run_id"]
            checkpoint = run_dir / Path(run["best_checkpoint"]).name
            predictions = run_dir / Path(run["validation_predictions"]).name
            if file_sha256(checkpoint) != run["best_checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash failed for {run['run_id']}")
            if file_sha256(predictions) != run["validation_predictions_sha256"]:
                raise ValueError(f"Prediction hash failed for {run['run_id']}")
            verify_downstream_checkpoint(checkpoint, run["run_id"])
            with np.load(predictions, allow_pickle=False) as payload:
                if payload["logits"].shape != (2000, 19) or not np.isfinite(payload["logits"]).all():
                    raise ValueError(f"Invalid V predictions for {run['run_id']}")
        for feature in seed_summary["feature_caches"]:
            feature_path = path.parent / "features" / Path(feature["path"]).name
            if file_sha256(feature_path) != feature["sha256"]:
                raise ValueError("Pre-fine-tuning feature cache hash mismatch")
            if feature["d_patches"] != 12000 or feature["v_patches"] != 2000:
                raise ValueError("Pre-fine-tuning feature cache has the wrong patch count")
    contracts = {
        (run["data"]["manifest_contract_sha256"], run["data"]["normalization_sha256"], run["data"]["subset_manifest_sha256"])
        for run in runs
    }
    if len(contracts) != 1:
        raise ValueError("Week 5 runs do not share one frozen data contract")
    result = {
        "week5_complete": True, "week6_approved": True, "run_count": 45,
        "models": list(WEEK5_MODELS), "fraction_codes": list(FINAL_FRACTIONS),
        "seeds": list(WEEK5_SEEDS), "selected_learning_rate_multipliers": {
            model: pilot["selected"][model]["learning_rate_multiplier"] for model in WEEK5_MODELS
        },
        "pilot_gpu_hours": float(pilot["actual_gpu_hours"]),
        "final_training_gpu_hours": sum(float(run["elapsed_seconds"]) for run in runs) / 3600,
        "feature_gpu_hours": sum(float(item["feature_gpu_hours"]) for item in seeds),
        "actual_gpu_hours": (
            float(pilot["actual_gpu_hours"])
            + sum(float(item["actual_gpu_hours"]) for item in seeds)
        ),
        "data_contract": {
            "manifest_contract_sha256": runs[0]["data"]["manifest_contract_sha256"],
            "normalization_sha256": runs[0]["data"]["normalization_sha256"],
            "subset_manifest_sha256": runs[0]["data"]["subset_manifest_sha256"],
        },
        "checkpoint_sha256": {run["run_id"]: run["best_checkpoint_sha256"] for run in runs},
        "runs": runs, "evaluation_labels_loaded": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
