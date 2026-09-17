from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import yaml

from spectrashift.data.downstream import FRACTION_COUNTS, WEEK5_SEEDS
from spectrashift.data.rgb import compute_rgb_percentile_contract

from .common import file_sha256
from .downstream import (
    MODEL_ADAPTERS,
    expected_epochs_and_steps,
    train_downstream,
    verify_downstream_checkpoint,
)
from .week5 import (
    WEEK5_MODELS,
    _find_completed,
    _find_recovery,
    _write_runtime_config,
    validate_complete_downstream_run,
    validate_pilot_approval,
)


WEEK6_FRACTIONS = ("05", "25", "50")
RGB_CONTROL_MODEL = "M1RGB"
RGB_CONTROL_FRACTION = "10"
LOCKED_MULTIPLIERS = {"M0": 1.0, "M1": 3.0, "M2": 3.0, "M3": 3.0, "M4": 3.0}
ALL_FRACTIONS = ("01", "05", "10", "25", "50", "100")


def validate_week5_approval(path: str | Path, expected_sha256: str | None = None) -> dict[str, object]:
    path = Path(path)
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError("Week 5 aggregate hash differs from the approved artifact")
    summary = json.loads(path.read_text())
    if not summary.get("week5_complete") or not summary.get("week6_approved"):
        raise ValueError("Week 5 aggregate does not approve Week 6")
    if summary.get("run_count") != 45 or summary.get("evaluation_labels_loaded") is not False:
        raise ValueError("Week 5 aggregate is incomplete or label-unsafe")
    if summary.get("selected_learning_rate_multipliers") != LOCKED_MULTIPLIERS:
        raise ValueError("Week 5 learning-rate decisions differ from the frozen Week 6 schedule")
    return summary


def freeze_week6_contracts(config_path: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    contracts = config["contracts"]
    output_dir = Path(contracts["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    week5 = validate_week5_approval(
        contracts["week5_summary_path"], contracts["week5_summary_sha256"]
    )
    week5_contracts = _week5_contracts(contracts["week5_contracts_summary_path"])
    if week5_contracts["subset_manifest_sha256"] != week5["data_contract"]["subset_manifest_sha256"]:
        raise ValueError("Week 5 subset contract differs from the approved aggregate")
    rgb = compute_rgb_percentile_contract(
        manifest_path=config["data"]["manifest_path"],
        staged_root=config["data"]["staged_root"],
        normalization_path=config["data"]["normalization_path"],
        manifest_contract_sha256=config["data"]["manifest_contract_sha256"],
        normalization_sha256=config["data"]["normalization_sha256"],
        shard_size=int(config["data"].get("shard_size", 512)),
        height=int(config["data"].get("height", 120)),
        width=int(config["data"].get("width", 120)),
    )
    rgb_path = output_dir / "rgb_percentile_contract.json"
    rgb_path.write_text(json.dumps(rgb, indent=2) + "\n")
    result = {
        "week6_contracts_complete": True,
        "week5_summary_sha256": file_sha256(contracts["week5_summary_path"]),
        "week5_contracts_summary_sha256": file_sha256(contracts["week5_contracts_summary_path"]),
        "rgb_contract_file": rgb_path.name,
        "rgb_contract_sha256": file_sha256(rgb_path),
        "rgb_band_order": rgb["band_order"],
        "rgb_fit_partition": rgb["fit_partition"],
        "rgb_fit_patch_count": rgb["fit_patch_count"],
        "data_contract": week5["data_contract"],
        "selected_learning_rate_multipliers": week5["selected_learning_rate_multipliers"],
        "evaluation_labels_loaded": False,
    }
    (output_dir / "week6_contracts_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _week5_contracts(path: str | Path) -> dict[str, object]:
    result = json.loads(Path(path).read_text())
    if not result.get("week5_contracts_complete") or result.get("evaluation_labels_loaded") is not False:
        raise ValueError("Week 5 contracts are incomplete or label-unsafe")
    return result


def resolved_week6_config(
    base: dict[str, object],
    model_id: str,
    seed: int,
    fraction_code: str,
    output_root: str | Path,
    week5_contracts_summary_path: str | Path,
    week6_contracts_summary_path: str | Path,
    encoder_paths: dict[str, str | Path],
) -> dict[str, object]:
    if seed not in WEEK5_SEEDS:
        raise ValueError("Unexpected Week 6 seed")
    main = model_id in WEEK5_MODELS and fraction_code in WEEK6_FRACTIONS
    rgb = model_id == RGB_CONTROL_MODEL and fraction_code == RGB_CONTROL_FRACTION
    if not (main or rgb):
        raise ValueError("Unexpected Week 6 model/fraction combination")

    week5_contracts = _week5_contracts(week5_contracts_summary_path)
    week6_contracts_path = Path(week6_contracts_summary_path)
    week6_contracts = json.loads(week6_contracts_path.read_text())
    if not week6_contracts.get("week6_contracts_complete"):
        raise ValueError("Week 6 RGB contracts are incomplete")
    training_base = base["training"]
    sample_count = FRACTION_COUNTS[fraction_code]
    epochs, _, expected_steps = expected_epochs_and_steps(
        sample_count, int(training_base["batch_size"]), int(training_base["base_epochs"]),
        int(training_base["minimum_optimizer_steps"]),
    )
    run_id = f"week6-{model_id.lower()}-f{fraction_code}-seed{seed}"
    data = dict(base["data"])
    data["adapter"] = MODEL_ADAPTERS[model_id]
    if model_id == RGB_CONTROL_MODEL:
        rgb_path = week6_contracts_path.parent / week6_contracts["rgb_contract_file"]
        data["preprocessing"] = {
            "mode": "imagenet_rgb_percentile",
            "contract_path": str(rgb_path),
            "contract_sha256": week6_contracts["rgb_contract_sha256"],
        }
    else:
        data["preprocessing"] = {"mode": "u_standardization"}

    if model_id == "M0":
        initialization: dict[str, object] = {"type": "random"}
    elif model_id in {"M1", RGB_CONTROL_MODEL}:
        initialization = {
            "type": "imagenet1k-v1-rgb" if model_id == RGB_CONTROL_MODEL else "imagenet1k-v1-core10",
            "path": str(Path(week5_contracts_summary_path).parent / week5_contracts["imagenet_weights_file"]),
            "sha256": week5_contracts["imagenet_weights_sha256"],
            "sha256_prefix": "f37072fd",
        }
    else:
        key = f"week4-{model_id.lower()}-seed{seed}"
        if key not in encoder_paths:
            raise ValueError(f"Missing matching Week 4 encoder for {key}")
        initialization = {
            "type": "week4-encoder",
            "path": str(encoder_paths[key]),
            "sha256": week5_contracts["week4_encoder_sha256"][key],
        }

    multiplier_model = "M1" if model_id == RGB_CONTROL_MODEL else model_id
    multiplier = float(LOCKED_MULTIPLIERS[multiplier_model])
    base_encoder_lr = float(
        training_base["encoder_learning_rate_scratch"]
        if model_id == "M0" else training_base["encoder_learning_rate_pretrained"]
    )
    encoder_lr = base_encoder_lr * multiplier
    head_lr = float(training_base["head_learning_rate"]) * multiplier
    return {
        "run": {
            "id": run_id,
            "model_id": model_id,
            "seed": int(seed),
            "fraction_code": fraction_code,
            "sample_count": sample_count,
            "pilot": False,
            "learning_rate_multiplier": multiplier,
            "output_dir": str(Path(output_root) / run_id),
        },
        "data": data,
        "subsets": {
            "manifest_path": str(Path(week5_contracts_summary_path).parent / "downstream_subsets.parquet"),
            "contract_path": str(Path(week5_contracts_summary_path).parent / "downstream_contract.json"),
            "sha256": week5_contracts["subset_manifest_sha256"],
        },
        "initialization": initialization,
        "training": {
            "batch_size": int(training_base["batch_size"]),
            "base_epochs": epochs,
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
            "evaluation_every_epochs": 1,
            "log_every_steps": int(training_base["log_every_steps"]),
        },
    }


def run_week6_seed(
    config_path: str | Path,
    seed: int,
    output_root: str | Path,
    week5_contracts_summary_path: str | Path,
    pilot_summary_path: str | Path,
    week5_summary_path: str | Path,
    week6_contracts_summary_path: str | Path,
    encoder_paths: dict[str, str | Path],
    resume_roots: list[str | Path] | None = None,
) -> dict[str, object]:
    base = yaml.safe_load(Path(config_path).read_text())
    pilot = validate_pilot_approval(pilot_summary_path)
    pilot_multipliers = {
        model: float(pilot["selected"][model]["learning_rate_multiplier"])
        for model in WEEK5_MODELS
    }
    if pilot_multipliers != LOCKED_MULTIPLIERS:
        raise ValueError("Supplied pilot summary differs from the frozen Week 6 learning rates")
    validate_week5_approval(week5_summary_path, base["contracts"]["week5_summary_sha256"])
    output_root = Path(output_root)
    roots = [Path(value) for value in (resume_roots or [])]
    identities = [
        (model, fraction) for model in WEEK5_MODELS for fraction in WEEK6_FRACTIONS
    ] + [(RGB_CONTROL_MODEL, RGB_CONTROL_FRACTION)]
    summaries = []
    for model_id, fraction in identities:
        config = resolved_week6_config(
            base, model_id, int(seed), fraction, output_root,
            week5_contracts_summary_path, week6_contracts_summary_path, encoder_paths,
        )
        config_path_runtime = _write_runtime_config(config, output_root / "configs")
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
            train_downstream(config_path_runtime, resume_path=recovery)
            summary = validate_complete_downstream_run(
                Path(config["run"]["output_dir"]) / "downstream_summary.json", config
            )
        summaries.append(summary)
    result = {
        "week6_seed_complete": True,
        "seed": int(seed),
        "run_count": 16,
        "runs": summaries,
        "actual_gpu_hours": sum(float(item["elapsed_seconds"]) for item in summaries) / 3600,
        "week5_summary_sha256": file_sha256(week5_summary_path),
        "week5_contracts_summary_sha256": file_sha256(week5_contracts_summary_path),
        "pilot_summary_sha256": file_sha256(pilot_summary_path),
        "week6_contracts_summary_sha256": file_sha256(week6_contracts_summary_path),
        "evaluation_labels_loaded": False,
    }
    (output_root / f"week6_seed{seed}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _artifact_path(summary_path: Path, run: dict[str, object], key: str) -> Path:
    return summary_path.parent / str(run["run_id"]) / Path(str(run[key])).name


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalized_log_aulc(label_counts: list[int], values: list[float]) -> float:
    x = np.log10(np.asarray(label_counts, dtype=np.float64))
    y = np.asarray(values, dtype=np.float64)
    if len(x) < 2 or len(x) != len(y) or np.any(np.diff(x) <= 0) or not np.isfinite(y).all():
        raise ValueError("AULC requires finite values at strictly increasing label counts")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def aggregate_week6(
    week5_summary_path: str | Path,
    week6_contracts_summary_path: str | Path,
    seed_summary_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    week5_path = Path(week5_summary_path)
    week5 = validate_week5_approval(week5_path)
    contracts_path = Path(week6_contracts_summary_path)
    contracts = json.loads(contracts_path.read_text())
    if not contracts.get("week6_contracts_complete") or contracts.get("evaluation_labels_loaded") is not False:
        raise ValueError("Week 6 contracts are incomplete or label-unsafe")
    if contracts.get("week5_summary_sha256") != file_sha256(week5_path):
        raise ValueError("Week 6 contracts were fitted against a different Week 5 aggregate")
    if len(seed_summary_paths) != 3:
        raise ValueError("Week 6 aggregation requires exactly three seed summaries")
    paths = [Path(path) for path in seed_summary_paths]
    seeds = [json.loads(path.read_text()) for path in paths]
    if {int(item["seed"]) for item in seeds} != set(WEEK5_SEEDS):
        raise ValueError("Week 6 summaries must cover seeds 17, 29, and 43")
    if any(
        not item.get("week6_seed_complete") or item.get("run_count") != 16
        or item.get("evaluation_labels_loaded") is not False for item in seeds
    ):
        raise ValueError("A Week 6 seed job is incomplete or label-unsafe")
    contract_hash = file_sha256(contracts_path)
    if {item.get("week6_contracts_summary_sha256") for item in seeds} != {contract_hash}:
        raise ValueError("Week 6 seed jobs do not share one RGB contract")
    if {item.get("week5_summary_sha256") for item in seeds} != {file_sha256(week5_path)}:
        raise ValueError("Week 6 seed jobs do not share one Week 5 aggregate")
    if {item.get("week5_contracts_summary_sha256") for item in seeds} != {
        contracts["week5_contracts_summary_sha256"]
    }:
        raise ValueError("Week 6 seed jobs do not share one Week 5 contract set")
    if len({item.get("pilot_summary_sha256") for item in seeds}) != 1:
        raise ValueError("Week 6 seed jobs do not share one pilot decision")

    new_runs = [run for item in seeds for run in item["runs"]]
    expected_new = {
        (model, fraction, seed)
        for model in WEEK5_MODELS for fraction in WEEK6_FRACTIONS for seed in WEEK5_SEEDS
    } | {(RGB_CONTROL_MODEL, RGB_CONTROL_FRACTION, seed) for seed in WEEK5_SEEDS}
    observed_new = {(run["model_id"], run["fraction_code"], int(run["seed"])) for run in new_runs}
    if len(new_runs) != 48 or observed_new != expected_new:
        raise ValueError("Week 6 aggregation has missing or duplicate new runs")
    expected_steps = {"05": 300, "25": 1410, "50": 2820, "10": 570}
    if any(
        run.get("steps") != expected_steps[run["fraction_code"]]
        or run.get("expected_optimizer_steps") != expected_steps[run["fraction_code"]]
        or run.get("amp_overflow_skips") != 0 or not run.get("finite_gate")
        or not run.get("completion_gate") or run.get("evaluation_labels_loaded") is not False
        for run in new_runs
    ):
        raise ValueError("At least one Week 6 run failed its step, stability, or isolation contract")
    if len({run["run_id"] for run in new_runs}) != 48 or len({run["best_checkpoint_sha256"] for run in new_runs}) != 48:
        raise ValueError("Week 6 run IDs or checkpoints are not unique")
    if len({run.get("source_tree_sha256") for run in new_runs}) != 1:
        raise ValueError("Week 6 runs were produced by different source trees")

    frozen_data = week5["data_contract"]
    for path, seed_summary in zip(paths, seeds, strict=True):
        for run in seed_summary["runs"]:
            if {
                key: run["data"][key] for key in frozen_data
            } != frozen_data:
                raise ValueError(f"Frozen data hash mismatch for {run['run_id']}")
            checkpoint = _artifact_path(path, run, "best_checkpoint")
            predictions = _artifact_path(path, run, "validation_predictions")
            if file_sha256(checkpoint) != run["best_checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash failed for {run['run_id']}")
            if file_sha256(predictions) != run["validation_predictions_sha256"]:
                raise ValueError(f"Prediction hash failed for {run['run_id']}")
            verify_downstream_checkpoint(checkpoint, run["run_id"])
            with np.load(predictions, allow_pickle=False) as payload:
                if payload["logits"].shape != (2000, 19) or len(payload["patch_ids"]) != 2000:
                    raise ValueError(f"Invalid V prediction shape for {run['run_id']}")
                if not np.isfinite(payload["logits"]).all():
                    raise ValueError(f"Non-finite V logits for {run['run_id']}")
            if run["model_id"] == RGB_CONTROL_MODEL:
                preprocessing = run.get("preprocessing", {})
                if (
                    run.get("band_order") != ["B04", "B03", "B02"]
                    or preprocessing.get("mode") != "imagenet_rgb_percentile"
                    or preprocessing.get("contract_sha256") != contracts["rgb_contract_sha256"]
                ):
                    raise ValueError("M1RGB preprocessing provenance mismatch")

    week5_runs = list(week5["runs"])
    main_runs = week5_runs + [run for run in new_runs if run["model_id"] != RGB_CONTROL_MODEL]
    expected_main = {
        (model, fraction, seed)
        for model in WEEK5_MODELS for fraction in ALL_FRACTIONS for seed in WEEK5_SEEDS
    }
    observed_main = {(run["model_id"], run["fraction_code"], int(run["seed"])) for run in main_runs}
    if len(main_runs) != 90 or observed_main != expected_main:
        raise ValueError("Merged controlled matrix is incomplete or duplicated")
    rgb_runs = [run for run in new_runs if run["model_id"] == RGB_CONTROL_MODEL]

    curve_rows = sorted([
        {
            "model_id": run["model_id"],
            "fraction_code": run["fraction_code"],
            "label_count": int(run["sample_count"]),
            "seed": int(run["seed"]),
            "validation_supported_map": float(run["validation_macro_average_precision"]),
            "run_id": run["run_id"],
            "checkpoint_sha256": run["best_checkpoint_sha256"],
        }
        for run in main_runs
    ], key=lambda row: (row["model_id"], row["seed"], row["label_count"]))
    lookup = {
        (row["model_id"], row["fraction_code"], row["seed"]): row["validation_supported_map"]
        for row in curve_rows
    }
    rgb_lookup = {int(run["seed"]): float(run["validation_macro_average_precision"]) for run in rgb_runs}
    comparisons = {"H1": ("M3", "M0"), "H2": ("M3", "M2"), "H3": ("M4", "M3")}
    difference_rows: list[dict[str, object]] = []
    for hypothesis, (left, right) in comparisons.items():
        for seed in WEEK5_SEEDS:
            for fraction in ALL_FRACTIONS:
                difference_rows.append({
                    "comparison": hypothesis,
                    "left_model": left,
                    "right_model": right,
                    "fraction_code": fraction,
                    "label_count": FRACTION_COUNTS[fraction],
                    "seed": seed,
                    "paired_map_difference": lookup[(left, fraction, seed)] - lookup[(right, fraction, seed)],
                })
    for seed in WEEK5_SEEDS:
        difference_rows.append({
            "comparison": "stem_control",
            "left_model": RGB_CONTROL_MODEL,
            "right_model": "M1",
            "fraction_code": "10",
            "label_count": FRACTION_COUNTS["10"],
            "seed": seed,
            "paired_map_difference": rgb_lookup[seed] - lookup[("M1", "10", seed)],
        })

    aulc_by_model_seed: dict[tuple[str, int], float] = {}
    for model in WEEK5_MODELS:
        for seed in WEEK5_SEEDS:
            values = [lookup[(model, fraction, seed)] for fraction in ALL_FRACTIONS]
            aulc_by_model_seed[(model, seed)] = normalized_log_aulc(
                [FRACTION_COUNTS[fraction] for fraction in ALL_FRACTIONS], values
            )
    aulc_rows: list[dict[str, object]] = [
        {
            "row_type": "model_seed",
            "name": model,
            "seed": seed,
            "normalized_log10_aulc": value,
            "mean_difference": "",
            "sample_std_difference": "",
        }
        for (model, seed), value in sorted(aulc_by_model_seed.items())
    ]
    hypothesis_aulc = {}
    for hypothesis, (left, right) in comparisons.items():
        values = [aulc_by_model_seed[(left, seed)] - aulc_by_model_seed[(right, seed)] for seed in WEEK5_SEEDS]
        hypothesis_aulc[hypothesis] = {
            "left_model": left,
            "right_model": right,
            "seed_differences": dict(zip(map(str, WEEK5_SEEDS), values, strict=True)),
            "mean_difference": float(np.mean(values)),
            "sample_std_difference": float(np.std(values, ddof=1)),
        }
        aulc_rows.append({
            "row_type": "comparison",
            "name": hypothesis,
            "seed": "",
            "normalized_log10_aulc": "",
            "mean_difference": float(np.mean(values)),
            "sample_std_difference": float(np.std(values, ddof=1)),
        })
    stem_values = [rgb_lookup[seed] - lookup[("M1", "10", seed)] for seed in WEEK5_SEEDS]
    stem_control = {
        "left_model": RGB_CONTROL_MODEL,
        "right_model": "M1",
        "fraction_code": "10",
        "seed_differences": dict(zip(map(str, WEEK5_SEEDS), stem_values, strict=True)),
        "mean_difference": float(np.mean(stem_values)),
        "sample_std_difference": float(np.std(stem_values, ddof=1)),
    }
    aulc_rows.append({
        "row_type": "comparison", "name": "stem_control", "seed": "",
        "normalized_log10_aulc": "", "mean_difference": stem_control["mean_difference"],
        "sample_std_difference": stem_control["sample_std_difference"],
    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "controlled_curves.csv", curve_rows,
        ["model_id", "fraction_code", "label_count", "seed", "validation_supported_map", "run_id", "checkpoint_sha256"],
    )
    _write_csv(
        output_dir / "paired_differences.csv", difference_rows,
        ["comparison", "left_model", "right_model", "fraction_code", "label_count", "seed", "paired_map_difference"],
    )
    _write_csv(
        output_dir / "aulc_summary.csv", aulc_rows,
        ["row_type", "name", "seed", "normalized_log10_aulc", "mean_difference", "sample_std_difference"],
    )
    result = {
        "week6_complete": True,
        "week7_approved": True,
        "new_run_count": 48,
        "controlled_run_count": 90,
        "rgb_control_run_count": 3,
        "models": list(WEEK5_MODELS),
        "fraction_codes": list(ALL_FRACTIONS),
        "seeds": list(WEEK5_SEEDS),
        "hypothesis_aulc": hypothesis_aulc,
        "stem_control": stem_control,
        "week5_summary_sha256": file_sha256(week5_path),
        "week6_contracts_summary_sha256": contract_hash,
        "new_checkpoint_sha256": {run["run_id"]: run["best_checkpoint_sha256"] for run in new_runs},
        "new_runs": new_runs,
        "evaluation_labels_loaded": False,
    }
    (output_dir / "week6_run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
