from __future__ import annotations

import csv
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from spectrashift.data.bands import BAND_ADAPTERS
from spectrashift.data.downstream import FRACTION_COUNTS, WEEK5_SEEDS
from spectrashift.data.labels import CANONICAL_LABELS
from spectrashift.eval.metrics import multilabel_metrics

from .common import directory_sha256, file_sha256, object_sha256, seed_everything
from .downstream import expected_epochs_and_steps
from .foundation import (
    FOUNDATION_FEATURE_DIMS,
    FOUNDATION_MODELS,
    _build_model,
    cache_foundation_features,
    train_foundation,
    verify_foundation_checkpoint,
)


FOUNDATION_NAMES = {"M5": "DINOv2 ViT-S/14", "M6": "OlmoEarth v1.1 Tiny"}
PILOT_MULTIPLIERS = (0.3, 1.0, 3.0)
FULL_FRACTIONS = ("01", "10", "100")
PROBE_FRACTIONS = ("01", "05", "10", "25", "50", "100")
L2_GRID = (0.0, 1e-4, 1e-2)
K_GRID = (5, 20, 50)
OLMO_CONFIG_SHA256 = "01dcb438144d8f70647ab2d11aef656a1632f3b5af1fdf9263c111127ad7bbc3"
OLMO_WEIGHTS_SHA256 = "2a3fe8132adf9ff2ca96d00c9e376b8925bfe430fda6140749b3b92764c67ae1"


def validate_week6_approval(path: str | Path, expected_sha256: str | None = None) -> dict[str, object]:
    path = Path(path)
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError("Week 6 aggregate hash differs from the approved artifact")
    summary = json.loads(path.read_text())
    if not summary.get("week6_complete") or not summary.get("week7_approved"):
        raise ValueError("Week 6 aggregate does not approve Week 7")
    if summary.get("controlled_run_count") != 90 or summary.get("rgb_control_run_count") != 3:
        raise ValueError("Week 6 aggregate is incomplete")
    if summary.get("evaluation_labels_loaded") is not False:
        raise ValueError("Week 6 approval is not label-safe")
    return summary


def _find_single(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} under {root}, found {matches}")
    return matches[0]


def freeze_week7_contracts(config_path: str | Path, assets_root: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    assets_root = Path(assets_root)
    output_dir = Path(config["contracts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    week6_path = Path(config["contracts"]["week6_summary_path"])
    week6 = validate_week6_approval(week6_path, config["contracts"]["week6_summary_sha256"])
    rgb_path = Path(config["contracts"]["rgb_contract_path"])
    rgb = json.loads(rgb_path.read_text())
    if rgb.get("fit_partition") != "U" or rgb.get("band_order") != ["B04", "B03", "B02"]:
        raise ValueError("Week 7 DINOv2 requires the approved U-only Week 6 RGB contract")

    dino_source = _find_single(assets_root, "hubconf.py").parent
    dino_weights = _find_single(assets_root, "dinov2_vits14_pretrain.pth")
    dino_archive = assets_root / "dinov2-source.zip"
    olmo_archive = assets_root / "olmoearth-minimal.zip"
    if file_sha256(dino_archive) != config["assets"]["dinov2_source_archive_sha256"]:
        raise ValueError("Pinned DINOv2 source archive hash mismatch")
    if file_sha256(dino_weights) != config["assets"]["dinov2_weights_sha256"]:
        raise ValueError("Pinned DINOv2 ViT-S/14 weight hash mismatch")
    if file_sha256(olmo_archive) != config["assets"]["olmoearth_minimal_archive_sha256"]:
        raise ValueError("Pinned OlmoEarth minimal source archive hash mismatch")
    olmo_model_dir = assets_root / "olmoearth-v1_1-tiny"
    olmo_config = olmo_model_dir / "config.json"
    olmo_weights = olmo_model_dir / "weights.pth"
    if not olmo_config.is_file() or not olmo_weights.is_file():
        raise FileNotFoundError("OlmoEarth v1.1 Tiny assets are incomplete")
    if file_sha256(olmo_config) != OLMO_CONFIG_SHA256 or file_sha256(olmo_weights) != OLMO_WEIGHTS_SHA256:
        raise ValueError("OlmoEarth v1.1 Tiny artifact differs from the Week 2 verified hashes")
    minimal_roots = [
        candidate.parent.parent for candidate in assets_root.rglob("olmoearth_pretrain_minimal/__init__.py")
    ]
    if len(minimal_roots) != 1:
        raise ValueError(f"Expected one OlmoEarth minimal source tree, found {minimal_roots}")
    olmo_source = minimal_roots[0]
    computed_path = _find_single(olmo_source, "computed.json")
    computed = json.loads(computed_path.read_text())["sentinel2_l2a"]
    band_order = list(BAND_ADAPTERS["olmo12"])
    olmo_input = {
        "status": "frozen",
        "model_id": "OLMOEARTH_V1_1_TINY",
        "raw_units": "sentinel2-l2a-dn",
        "band_order": band_order,
        "shape": ["batch", "height=120", "width=120", "time=1", "bands=12"],
        "patch_size": 8,
        "input_res_m": 10,
        "timestamp_order": ["day", "zero_indexed_month", "year"],
        "std_multiplier": 2.0,
        "normalizer_mean": [float(computed[band]["mean"]) for band in band_order],
        "normalizer_std": [float(computed[band]["std"]) for band in band_order],
        "invalid_mask_value": 3,
        "pooling": "mean-online-encoder-tokens-only",
        "normalizer_source_sha256": file_sha256(computed_path),
        "evaluation_labels_loaded": False,
    }
    olmo_input_path = output_dir / "olmoearth_input_contract.json"
    olmo_input_path.write_text(json.dumps(olmo_input, indent=2) + "\n")
    result = {
        "week7_contracts_complete": True,
        "week6_summary_sha256": file_sha256(week6_path),
        "week6_rgb_contract_sha256": file_sha256(rgb_path),
        "data_contract": {
            "manifest_contract_sha256": config["data"]["manifest_contract_sha256"],
            "normalization_sha256": config["data"]["normalization_sha256"],
            "subset_manifest_sha256": week6["new_runs"][0]["data"]["subset_manifest_sha256"],
        },
        "dinov2": {
            "display_name": "DINOv2 ViT-S/14",
            "model_name": "dinov2_vits14",
            "source_revision": config["assets"]["dinov2_source_revision"],
            "source_dir": os.path.relpath(dino_source, output_dir),
            "source_sha256": directory_sha256(dino_source),
            "weights_file": os.path.relpath(dino_weights, output_dir),
            "weights_sha256": file_sha256(dino_weights),
            "input_size": 126,
            "feature_dimension": 384,
            "register_tokens": 0,
            "pooling": "mean-normalized-patch-tokens",
        },
        "olmoearth": {
            "display_name": "OlmoEarth v1.1 Tiny",
            "model_id": "OLMOEARTH_V1_1_TINY",
            "model_revision": config["assets"]["olmoearth_model_revision"],
            "source_revision": config["assets"]["olmoearth_minimal_revision"],
            "source_dir": os.path.relpath(olmo_source, output_dir),
            "source_sha256": directory_sha256(olmo_source),
            "model_dir": os.path.relpath(olmo_model_dir, output_dir),
            "config_sha256": file_sha256(olmo_config),
            "weights_sha256": file_sha256(olmo_weights),
            "input_contract_file": olmo_input_path.name,
            "input_contract_sha256": file_sha256(olmo_input_path),
            "feature_dimension": 192,
        },
        "foundation_full_run_count": 18,
        "foundation_linear_probe_count": 36,
        "foundation_knn_run_count": 6,
        "evaluation_labels_loaded": False,
    }
    path = output_dir / "week7_contracts_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate_foundation_assets(
    config_path: str | Path, week5_contracts_path: str | Path,
    week7_contracts_path: str | Path,
) -> dict[str, object]:
    base = yaml.safe_load(Path(config_path).read_text())
    records = {}
    for model_id in FOUNDATION_MODELS:
        config = resolved_foundation_config(
            base, model_id, 17, "10", 1.0, 32, "/tmp/week7-contract-smoke",
            week5_contracts_path, week7_contracts_path, True,
        )
        model, initialization = _build_model(config)
        model.eval()
        batch = _synthetic_batch(model_id, 1, torch.device("cpu"))
        with torch.inference_mode():
            first = model.features(batch)
            second = model.features(batch)
        expected = FOUNDATION_FEATURE_DIMS[model_id]
        if first.shape != (1, expected) or not torch.isfinite(first).all():
            raise ValueError(f"{model_id} adapter produced invalid features")
        if not torch.equal(first, second):
            raise ValueError(f"{model_id} adapter is not deterministic in evaluation mode")
        model.train()
        model.zero_grad(set_to_none=True)
        loss = model(batch).square().mean()
        loss.backward()
        gradients = [value.grad for value in model.parameters() if value.grad is not None]
        if not gradients or any(not torch.isfinite(value).all() for value in gradients):
            raise ValueError(f"{model_id} adapter failed the finite gradient-flow gate")
        records[model_id] = {
            "foundation_name": FOUNDATION_NAMES[model_id],
            "feature_shape": list(first.shape),
            "parameter_count": sum(value.numel() for value in model.parameters()),
            "encoder_parameter_count": sum(value.numel() for value in model.encoder.parameters()),
            "trainable_parameter_count": sum(value.numel() for value in model.parameters() if value.requires_grad),
            "initialization": initialization,
            "deterministic_gate": True, "finite_gate": True, "gradient_gate": True,
        }
        del model
        gc.collect()
    contracts_path = Path(week7_contracts_path)
    result = {
        "week7_adapter_smoke_complete": True, "models": records,
        "evaluation_labels_loaded": False,
    }
    smoke_path = contracts_path.parent / "foundation_adapter_smoke.json"
    smoke_path.write_text(json.dumps(result, indent=2) + "\n")
    contracts = json.loads(contracts_path.read_text())
    contracts["adapter_smoke_file"] = smoke_path.name
    contracts["adapter_smoke_sha256"] = file_sha256(smoke_path)
    contracts["adapter_smoke_passed"] = True
    contracts_path.write_text(json.dumps(contracts, indent=2) + "\n")
    return result


def _contracts(path: str | Path) -> tuple[Path, dict[str, object]]:
    path = Path(path)
    result = json.loads(path.read_text())
    if not result.get("week7_contracts_complete") or result.get("evaluation_labels_loaded") is not False:
        raise ValueError("Week 7 contracts are incomplete or label-unsafe")
    return path, result


def resolved_foundation_config(
    base: dict[str, object], model_id: str, seed: int, fraction_code: str,
    multiplier: float, batch_size: int, output_root: str | Path,
    week5_contracts_summary_path: str | Path, week7_contracts_summary_path: str | Path,
    pilot: bool,
) -> dict[str, object]:
    if model_id not in FOUNDATION_MODELS or seed not in WEEK5_SEEDS or fraction_code not in FRACTION_COUNTS:
        raise ValueError("Unexpected Week 7 model, seed, or fraction")
    if batch_size not in {16, 32} or multiplier not in PILOT_MULTIPLIERS:
        raise ValueError("Unexpected Week 7 batch size or learning-rate multiplier")
    if pilot and (seed != 17 or fraction_code != "10"):
        raise ValueError("Week 7 pilots are fixed to seed 17 at 10% labels")
    contracts_path, contracts = _contracts(week7_contracts_summary_path)
    week5_path = Path(week5_contracts_summary_path)
    week5 = json.loads(week5_path.read_text())
    if not week5.get("week5_contracts_complete") or week5.get("evaluation_labels_loaded") is not False:
        raise ValueError("Week 5 subset contracts are incomplete")
    count = FRACTION_COUNTS[fraction_code]
    base_epochs, minimum_steps = (10, 0) if pilot else (20, 300)
    epochs, _, steps = expected_epochs_and_steps(count, batch_size, base_epochs, minimum_steps)
    run_id = (
        f"week7-{model_id.lower()}-pilot-lr{str(multiplier).replace('.', 'p')}-seed17"
        if pilot else f"week7-{model_id.lower()}-f{fraction_code}-seed{seed}"
    )
    asset = contracts["dinov2" if model_id == "M5" else "olmoearth"]
    if model_id == "M5":
        initialization = {
            "type": "official-dinov2-vits14",
            "source_revision": asset["source_revision"],
            "source_dir": str((contracts_path.parent / asset["source_dir"]).resolve()),
            "source_sha256": asset["source_sha256"],
            "weights_path": str((contracts_path.parent / asset["weights_file"]).resolve()),
            "weights_sha256": asset["weights_sha256"],
        }
        preprocessing = {
            "mode": "u-percentile-imagenet-rgb",
            "band_order": ["B04", "B03", "B02"], "resolution": 126,
            "rgb_contract_sha256": contracts["week6_rgb_contract_sha256"],
        }
    else:
        initialization = {
            "type": "official-olmoearth-v1_1-tiny",
            "source_revision": asset["source_revision"],
            "model_revision": asset["model_revision"],
            "source_dir": str((contracts_path.parent / asset["source_dir"]).resolve()),
            "source_sha256": asset["source_sha256"],
            "model_dir": str((contracts_path.parent / asset["model_dir"]).resolve()),
            "config_sha256": asset["config_sha256"],
            "weights_sha256": asset["weights_sha256"],
        }
        preprocessing = {
            "mode": "official-olmoearth-normalizer",
            "band_order": list(BAND_ADAPTERS["olmo12"]), "resolution": 120,
            "input_contract_sha256": asset["input_contract_sha256"],
        }
    data = dict(base["data"])
    data["rgb_contract_path"] = str(Path(base["contracts"]["rgb_contract_path"]))
    data["olmo_contract_path"] = str(contracts_path.parent / contracts["olmoearth"]["input_contract_file"])
    data["preprocessing"] = preprocessing
    return {
        "run": {
            "id": run_id, "model_id": model_id, "foundation_name": FOUNDATION_NAMES[model_id],
            "seed": seed, "fraction_code": fraction_code, "sample_count": count,
            "pilot": pilot, "learning_rate_multiplier": float(multiplier),
            "output_dir": str(Path(output_root) / run_id),
        },
        "data": data,
        "subsets": {
            "manifest_path": str(week5_path.parent / "downstream_subsets.parquet"),
            "contract_path": str(week5_path.parent / "downstream_contract.json"),
            "sha256": week5["subset_manifest_sha256"],
        },
        "initialization": initialization,
        "training": {
            "batch_size": batch_size, "base_epochs": epochs, "minimum_optimizer_steps": 0,
            "expected_optimizer_steps": steps,
            "encoder_learning_rate": float(base["training"]["encoder_learning_rate"]) * multiplier,
            "head_learning_rate": float(base["training"]["head_learning_rate"]) * multiplier,
            "weight_decay": 0.05, "warmup_fraction": 0.05,
            "minimum_encoder_learning_rate": float(base["training"]["minimum_encoder_learning_rate"]),
            "gradient_clip": 1.0, "amp": True, "amp_initial_scale": 128.0,
            "amp_growth_interval": 1_000_000, "fail_on_amp_overflow": True,
            "num_workers": int(base["training"]["num_workers"]), "require_cuda": True,
            "validation_batch_size": int(base["training"]["validation_batch_size"]),
            "evaluation_every_epochs": 1 if pilot or fraction_code != "01" else 5,
            "log_every_steps": int(base["training"]["log_every_steps"]),
        },
        "probes": dict(base["probes"]),
    }


def _write_config(config: dict[str, object], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{config['run']['id']}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _synthetic_batch(model_id: str, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    size, channels = (126, 3) if model_id == "M5" else (120, 12)
    return {
        "image": torch.zeros(batch_size, channels, size, size, device=device),
        "valid": torch.ones(batch_size, channels, size, size, dtype=torch.bool, device=device),
        "timestamp": torch.tensor([15, 5, 2019], device=device).repeat(batch_size, 1),
    }


def benchmark_foundation_batch(
    base: dict[str, object], contracts_path: str | Path, week5_contracts_path: str | Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Week 7 foundation batch benchmark requires CUDA")
    device = torch.device("cuda:0")
    errors: dict[str, str] = {}
    measurements: dict[str, object] = {}
    selected = None
    for batch_size in (32, 16):
        passed = True
        for model_id in FOUNDATION_MODELS:
            try:
                config = resolved_foundation_config(
                    base, model_id, 17, "10", 1.0, batch_size, "/tmp/week7-benchmark",
                    week5_contracts_path, contracts_path, True,
                )
                model, _ = _build_model(config)
                model = model.to(device).train()
                batch = _synthetic_batch(model_id, batch_size, device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                loss = model(batch).float().square().mean()
                loss.backward()
                torch.cuda.synchronize(device)
                measurements[f"{model_id}_batch{batch_size}"] = {
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "finite": bool(torch.isfinite(loss)),
                }
                del model, batch, loss
                gc.collect()
                torch.cuda.empty_cache()
            except RuntimeError as error:
                passed = False
                errors[f"{model_id}_batch{batch_size}"] = str(error)
                gc.collect()
                torch.cuda.empty_cache()
                break
        if passed:
            selected = batch_size
            break
    if selected is None:
        raise RuntimeError(f"Neither Week 7 physical batch passed: {errors}")
    return {"selected_batch_size": selected, "measurements": measurements, "errors": errors}


def _select_pilot(model_id: str, summaries: list[dict[str, object]], tolerance: float) -> dict[str, object]:
    candidates = [item for item in summaries if item["model_id"] == model_id]
    if len(candidates) != 3:
        raise ValueError(f"Expected three pilot candidates for {model_id}")
    best = max(float(item["validation_macro_average_precision"]) for item in candidates)
    eligible = [item for item in candidates if best - float(item["validation_macro_average_precision"]) < tolerance]
    preferred = [item for item in eligible if math.isclose(float(item["learning_rate_multiplier"]), 1.0)]
    selected = preferred[0] if preferred else min(eligible, key=lambda item: float(item["learning_rate_multiplier"]))
    history = [json.loads(line) for line in Path(selected["training_log"]).read_text().splitlines()]
    by_epoch = {int(item["epoch"]): item for item in history if "validation_macro_average_precision" in item}
    final_epoch = int(selected["epochs"])
    blocked = (
        int(selected["best_epoch"]) == final_epoch
        and final_epoch in by_epoch and 8 in by_epoch
        and float(by_epoch[final_epoch]["validation_macro_average_precision"])
        - float(by_epoch[8]["validation_macro_average_precision"]) > tolerance
    )
    return {
        "model_id": model_id,
        "learning_rate_multiplier": float(selected["learning_rate_multiplier"]),
        "validation_macro_average_precision": float(selected["validation_macro_average_precision"]),
        "best_epoch": int(selected["best_epoch"]),
        "undertraining_block": blocked,
        "candidate_scores": {
            str(item["learning_rate_multiplier"]): float(item["validation_macro_average_precision"])
            for item in candidates
        },
    }


def run_week7_pilots(
    config_path: str | Path, output_root: str | Path,
    week5_contracts_path: str | Path, week7_contracts_path: str | Path,
) -> dict[str, object]:
    base = yaml.safe_load(Path(config_path).read_text())
    output_root = Path(output_root)
    contracts_path, contracts = _contracts(week7_contracts_path)
    smoke_path = contracts_path.parent / str(contracts.get("adapter_smoke_file", ""))
    if (
        contracts.get("adapter_smoke_passed") is not True
        or not smoke_path.is_file()
        or file_sha256(smoke_path) != contracts.get("adapter_smoke_sha256")
    ):
        raise ValueError("Week 7 adapter smoke gate is absent or invalid")
    batch_gate = benchmark_foundation_batch(base, week7_contracts_path, week5_contracts_path)
    batch_size = int(batch_gate["selected_batch_size"])
    summaries = []
    for model_id in FOUNDATION_MODELS:
        for multiplier in PILOT_MULTIPLIERS:
            config = resolved_foundation_config(
                base, model_id, 17, "10", multiplier, batch_size, output_root,
                week5_contracts_path, week7_contracts_path, True,
            )
            config_file = _write_config(config, output_root / "configs")
            summary = train_foundation(config_file)
            summary["learning_rate_multiplier"] = multiplier
            summary["training_log"] = str(Path(config["run"]["output_dir"]) / "training.jsonl")
            summaries.append(summary)
    selected = {model: _select_pilot(model, summaries, float(base["pilots"]["tie_tolerance"]))
                for model in FOUNDATION_MODELS}
    approved = not any(value["undertraining_block"] for value in selected.values())
    forecast_steps = sum(
        expected_epochs_and_steps(FRACTION_COUNTS[fraction], batch_size, 20, 300)[2]
        for fraction in FULL_FRACTIONS for _ in FOUNDATION_MODELS for _ in WEEK5_SEEDS
    )
    result = {
        "week7_pilots_complete": True, "foundation_matrix_approved": approved,
        "pilot_run_count": 6, "selected_batch_size": batch_size,
        "batch_gate": batch_gate, "selected": selected,
        "forecast_full_matrix_steps": forecast_steps,
        "week7_contracts_sha256": file_sha256(week7_contracts_path),
        "week5_contracts_sha256": file_sha256(week5_contracts_path),
        "evaluation_labels_loaded": False,
    }
    (output_root / "week7_pilot_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for item in summaries:
        Path(item["best_checkpoint"]).unlink(missing_ok=True)
        Path(item["validation_predictions"]).unlink(missing_ok=True)
    return result


def validate_pilot_approval(path: str | Path) -> dict[str, object]:
    result = json.loads(Path(path).read_text())
    if not result.get("week7_pilots_complete") or not result.get("foundation_matrix_approved"):
        raise ValueError("Week 7 pilots do not approve full foundation training")
    if result.get("pilot_run_count") != 6 or set(result.get("selected", {})) != set(FOUNDATION_MODELS):
        raise ValueError("Week 7 pilot matrix is incomplete")
    if result.get("evaluation_labels_loaded") is not False:
        raise ValueError("Week 7 pilots are not label-safe")
    return result


def validate_complete_foundation_run(summary_path: str | Path, config: dict[str, object]) -> dict[str, object]:
    path = Path(summary_path)
    result = json.loads(path.read_text())
    if result.get("run_id") != config["run"]["id"] or result.get("config_sha256") != object_sha256(config):
        raise ValueError("Foundation run identity or configuration hash mismatch")
    if result.get("steps") != config["training"]["expected_optimizer_steps"]:
        raise ValueError("Foundation run has the wrong optimizer-step count")
    if result.get("amp_overflow_skips") != 0 or not result.get("completion_gate") or not result.get("finite_gate"):
        raise ValueError("Foundation run failed its stability gates")
    if result.get("evaluation_labels_loaded") is not False:
        raise ValueError("Foundation run accessed evaluation labels")
    return result


def _completed(roots: list[Path], run_id: str, config: dict[str, object]) -> Path | None:
    for root in roots:
        for path in root.rglob(f"{run_id}/foundation_summary.json"):
            try:
                validate_complete_foundation_run(path, config)
                return path
            except (ValueError, OSError, json.JSONDecodeError):
                pass
    return None


def _recovery(roots: list[Path], run_id: str, config: dict[str, object]) -> Path | None:
    for root in roots:
        for path in root.rglob(f"{run_id}/recovery-latest.pt"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("config_sha256") == object_sha256(config):
                    return path
            except (OSError, ValueError, RuntimeError):
                pass
    return None


def run_week7_seed(
    config_path: str | Path, seed: int, output_root: str | Path,
    week5_contracts_path: str | Path, week7_contracts_path: str | Path,
    pilot_summary_path: str | Path, resume_roots: list[str | Path] | None = None,
) -> dict[str, object]:
    base = yaml.safe_load(Path(config_path).read_text())
    pilots = validate_pilot_approval(pilot_summary_path)
    if pilots["week7_contracts_sha256"] != file_sha256(week7_contracts_path):
        raise ValueError("Pilot and seed jobs do not share the same Week 7 contracts")
    output_root = Path(output_root)
    roots = [Path(value) for value in (resume_roots or [])]
    summaries = []
    for model_id in FOUNDATION_MODELS:
        multiplier = float(pilots["selected"][model_id]["learning_rate_multiplier"])
        for fraction in FULL_FRACTIONS:
            config = resolved_foundation_config(
                base, model_id, int(seed), fraction, multiplier,
                int(pilots["selected_batch_size"]), output_root,
                week5_contracts_path, week7_contracts_path, False,
            )
            config_file = _write_config(config, output_root / "configs")
            found = _completed(roots, config["run"]["id"], config)
            if found:
                destination = Path(config["run"]["output_dir"])
                shutil.copytree(found.parent, destination, dirs_exist_ok=True)
                summary = validate_complete_foundation_run(destination / "foundation_summary.json", config)
            else:
                resume = _recovery(roots, config["run"]["id"], config)
                if resume:
                    destination = Path(config["run"]["output_dir"])
                    shutil.copytree(resume.parent, destination, dirs_exist_ok=True)
                    resume = destination / resume.name
                train_foundation(config_file, resume)
                summary = validate_complete_foundation_run(
                    Path(config["run"]["output_dir"]) / "foundation_summary.json", config
                )
            summaries.append(summary)
    result = {
        "week7_seed_complete": True, "seed": int(seed), "run_count": 6,
        "runs": summaries,
        "actual_gpu_hours": sum(float(item["elapsed_seconds"]) for item in summaries) / 3600,
        "week7_contracts_sha256": file_sha256(week7_contracts_path),
        "week5_contracts_sha256": file_sha256(week5_contracts_path),
        "pilot_summary_sha256": file_sha256(pilot_summary_path),
        "evaluation_labels_loaded": False,
    }
    (output_root / f"week7_seed{seed}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _label_matrix(frame: pd.DataFrame, patch_ids: np.ndarray) -> np.ndarray:
    lookup = {str(row.patch_id): list(row.labels) for row in frame.itertuples()}
    label_lookup = {name: index for index, name in enumerate(CANONICAL_LABELS)}
    matrix = np.zeros((len(patch_ids), len(CANONICAL_LABELS)), dtype=np.float32)
    for row_index, patch_id in enumerate(map(str, patch_ids)):
        if patch_id not in lookup:
            raise ValueError(f"Feature cache contains unknown patch ID: {patch_id}")
        for label in lookup[patch_id]:
            matrix[row_index, label_lookup[str(label)]] = 1.0
    return matrix


def _fit_linear_candidate(
    train_features: np.ndarray, train_targets: np.ndarray,
    validation_features: np.ndarray, validation_targets: np.ndarray,
    supported_indices: list[int], seed: int, l2: float,
    epochs: int, batch_size: int, learning_rate: float,
) -> tuple[float, dict[str, torch.Tensor], np.ndarray]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.from_numpy(train_features.astype(np.float32))
    y_train = torch.from_numpy(train_targets.astype(np.float32))
    x_validation = torch.from_numpy(validation_features.astype(np.float32)).to(device)
    model = torch.nn.Linear(train_features.shape[1], train_targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2)
    for epoch in range(epochs):
        order = torch.randperm(len(x_train), generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            features = x_train[indices].to(device)
            targets = y_train[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Week 7 linear-probe loss")
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits = model(x_validation).float().cpu().numpy()
    probabilities = 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
    metrics = multilabel_metrics(validation_targets, probabilities,
                                 supported_indices=supported_indices)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return float(metrics["macro_average_precision"]), state, logits


def run_linear_probe(
    model_id: str, seed: int, fraction: str, cache_path: str | Path,
    manifest_path: str | Path, subset_path: str | Path, subset_contract_path: str | Path,
    output_dir: str | Path, settings: dict[str, object],
) -> dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as payload:
        d_features = payload["D_features"].astype(np.float32)
        v_features = payload["V_features"].astype(np.float32)
        d_ids, v_ids = payload["D_patch_ids"], payload["V_patch_ids"]
    manifest = pd.read_parquet(manifest_path)
    visible = manifest[manifest["partition"].isin(["D", "V"])]
    d_targets, v_targets = _label_matrix(visible, d_ids), _label_matrix(visible, v_ids)
    subsets = pd.read_parquet(subset_path)
    selected_ids = subsets[subsets["downstream_seed"].eq(seed)].sort_values("subset_rank").head(
        FRACTION_COUNTS[fraction]
    )["patch_id"].astype(str).tolist()
    feature_lookup = {str(value): index for index, value in enumerate(d_ids)}
    indices = np.asarray([feature_lookup[value] for value in selected_ids], dtype=np.int64)
    train_features, train_targets = d_features[indices], d_targets[indices]
    mean, std = train_features.mean(axis=0), train_features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_features = (train_features - mean) / std
    validation_features = (v_features - mean) / std
    contract = json.loads(Path(subset_contract_path).read_text())
    candidates = []
    for l2 in L2_GRID:
        score, state, logits = _fit_linear_candidate(
            train_features, train_targets, validation_features, v_targets,
            contract["supported_class_indices"], seed, l2,
            int(settings["linear_epochs"]), int(settings["linear_batch_size"]),
            float(settings["linear_learning_rate"]),
        )
        candidates.append((score, l2, state, logits))
    best_score = max(item[0] for item in candidates)
    tied = [item for item in candidates if math.isclose(item[0], best_score, rel_tol=0, abs_tol=1e-12)]
    preferred = [item for item in tied if math.isclose(item[1], 1e-4)]
    selected = preferred[0] if preferred else min(tied, key=lambda item: item[1])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"week7-{model_id.lower()}-linear-f{fraction}-seed{seed}"
    head_path = output_dir / f"{run_id}-head.pt"
    torch.save(selected[2], head_path)
    result = {
        "run_id": run_id, "model_id": model_id, "probe_type": "linear",
        "seed": seed, "fraction_code": fraction, "sample_count": len(indices),
        "selected_l2": selected[1], "validation_macro_average_precision": selected[0],
        "candidate_scores": {str(item[1]): item[0] for item in candidates},
        "feature_cache_sha256": file_sha256(cache_path),
        "subset_manifest_sha256": contract["subset_manifest_sha256"],
        "standardization_fit_partition": "D-subset",
        "head_file": str(head_path), "head_sha256": file_sha256(head_path),
        "validation_rows": len(v_ids), "validation_logit_shape": list(selected[3].shape),
        "finite_gate": bool(np.isfinite(selected[3]).all()), "evaluation_labels_loaded": False,
    }
    (output_dir / f"{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_knn_probe(
    model_id: str, seed: int, cache_path: str | Path,
    manifest_path: str | Path, subset_path: str | Path, subset_contract_path: str | Path,
    temperature: float,
) -> dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as payload:
        d_features = payload["D_features"].astype(np.float32)
        v_features = payload["V_features"].astype(np.float32)
        d_ids, v_ids = payload["D_patch_ids"], payload["V_patch_ids"]
    visible = pd.read_parquet(manifest_path)
    visible = visible[visible["partition"].isin(["D", "V"])]
    d_targets, v_targets = _label_matrix(visible, d_ids), _label_matrix(visible, v_ids)
    subset = pd.read_parquet(subset_path)
    selected_ids = subset[subset["downstream_seed"].eq(seed)].sort_values("subset_rank").head(1200)["patch_id"].astype(str)
    lookup = {str(value): index for index, value in enumerate(d_ids)}
    indices = np.asarray([lookup[value] for value in selected_ids], dtype=np.int64)
    train_features, train_targets = d_features[indices], d_targets[indices]
    train_features /= np.maximum(np.linalg.norm(train_features, axis=1, keepdims=True), 1e-12)
    v_features /= np.maximum(np.linalg.norm(v_features, axis=1, keepdims=True), 1e-12)
    similarities = v_features @ train_features.T
    contract = json.loads(Path(subset_contract_path).read_text())
    candidate_scores, candidate_predictions = {}, {}
    for k in K_GRID:
        neighbor_indices = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
        neighbor_similarity = np.take_along_axis(similarities, neighbor_indices, axis=1)
        shifted = neighbor_similarity / temperature
        shifted -= shifted.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        weights /= weights.sum(axis=1, keepdims=True)
        scores = (train_targets[neighbor_indices] * weights[..., None]).sum(axis=1)
        # A convex combination of binary targets is a probability, but float32
        # accumulation can produce values a few ULPs outside [0, 1]. Keep the
        # numerical representation inside the probability contract consumed by
        # calibration metrics.
        scores = np.clip(scores, 0.0, 1.0)
        metrics = multilabel_metrics(v_targets, scores, supported_indices=contract["supported_class_indices"])
        candidate_scores[k] = float(metrics["macro_average_precision"])
        candidate_predictions[k] = scores
    best = max(candidate_scores.values())
    tied = [k for k, value in candidate_scores.items() if math.isclose(value, best, rel_tol=0, abs_tol=1e-12)]
    selected_k = 20 if 20 in tied else min(tied)
    return {
        "run_id": f"week7-{model_id.lower()}-knn-f10-seed{seed}",
        "model_id": model_id, "probe_type": "knn", "seed": seed,
        "fraction_code": "10", "sample_count": 1200, "selected_k": selected_k,
        "temperature": temperature, "validation_macro_average_precision": candidate_scores[selected_k],
        "candidate_scores": {str(key): value for key, value in candidate_scores.items()},
        "feature_cache_sha256": file_sha256(cache_path),
        "subset_manifest_sha256": contract["subset_manifest_sha256"],
        "validation_rows": len(v_ids),
        "finite_gate": bool(np.isfinite(candidate_predictions[selected_k]).all()),
        "evaluation_labels_loaded": False,
    }


def run_week7_probes(
    config_path: str | Path, output_root: str | Path,
    week5_contracts_path: str | Path, week7_contracts_path: str | Path,
    pilot_summary_path: str | Path,
) -> dict[str, object]:
    base = yaml.safe_load(Path(config_path).read_text())
    pilots = validate_pilot_approval(pilot_summary_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    caches, linear_runs, knn_runs = [], [], []
    for model_id in FOUNDATION_MODELS:
        config = resolved_foundation_config(
            base, model_id, 17, "100", float(pilots["selected"][model_id]["learning_rate_multiplier"]),
            int(pilots["selected_batch_size"]), output_root, week5_contracts_path,
            week7_contracts_path, False,
        )
        config_file = _write_config(config, output_root / "feature-configs")
        cache_path = output_root / "features" / f"{model_id.lower()}-d-v-features.npz"
        cache_summary = cache_foundation_features(config_file, cache_path)
        caches.append(cache_summary)
        for seed in WEEK5_SEEDS:
            for fraction in PROBE_FRACTIONS:
                linear_runs.append(run_linear_probe(
                    model_id, seed, fraction, cache_path, config["data"]["manifest_path"],
                    config["subsets"]["manifest_path"], config["subsets"]["contract_path"],
                    output_root / "linear", config["probes"],
                ))
            knn_runs.append(run_knn_probe(
                model_id, seed, cache_path, config["data"]["manifest_path"],
                config["subsets"]["manifest_path"], config["subsets"]["contract_path"],
                float(config["probes"]["knn_temperature"]),
            ))
    result = {
        "week7_probes_complete": True, "foundation_feature_cache_count": len(caches),
        "foundation_linear_probe_count": len(linear_runs), "foundation_knn_run_count": len(knn_runs),
        "feature_caches": caches, "linear_runs": linear_runs, "knn_runs": knn_runs,
        "week7_contracts_sha256": file_sha256(week7_contracts_path),
        "pilot_summary_sha256": file_sha256(pilot_summary_path),
        "evaluation_labels_loaded": False,
    }
    (output_root / "week7_probe_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _artifact_path(summary_path: Path, run: dict[str, object], key: str) -> Path:
    return summary_path.parent / str(run["run_id"]) / Path(str(run[key])).name


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def aggregate_week7(
    week6_summary_path: str | Path, controlled_curves_path: str | Path,
    week7_contracts_path: str | Path, pilot_summary_path: str | Path,
    probe_summary_path: str | Path, seed_summary_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    week6_path = Path(week6_summary_path)
    week6 = validate_week6_approval(week6_path)
    contracts_path, contracts = _contracts(week7_contracts_path)
    if contracts["week6_summary_sha256"] != file_sha256(week6_path):
        raise ValueError("Week 7 contracts use a different Week 6 aggregate")
    pilots = validate_pilot_approval(pilot_summary_path)
    if pilots["week7_contracts_sha256"] != file_sha256(contracts_path):
        raise ValueError("Week 7 pilots use different foundation contracts")
    probes_path = Path(probe_summary_path)
    probes = json.loads(probes_path.read_text())
    if (
        not probes.get("week7_probes_complete")
        or probes.get("foundation_feature_cache_count") != 2
        or probes.get("foundation_linear_probe_count") != 36
        or probes.get("foundation_knn_run_count") != 6
        or probes.get("evaluation_labels_loaded") is not False
    ):
        raise ValueError("Week 7 foundation probes are incomplete or label-unsafe")
    probe_root = probes_path.parent
    for cache in probes["feature_caches"]:
        matches = list(probe_root.rglob(Path(str(cache["path"])).name))
        if len(matches) != 1 or file_sha256(matches[0]) != cache["sha256"]:
            raise ValueError(f"Feature-cache hash failed for {cache['model_id']}")
        with np.load(matches[0], allow_pickle=False) as payload:
            if payload["D_features"].shape != (12000, int(cache["feature_dimension"])):
                raise ValueError(f"Invalid D feature cache for {cache['model_id']}")
            if payload["V_features"].shape != (2000, int(cache["feature_dimension"])):
                raise ValueError(f"Invalid V feature cache for {cache['model_id']}")
            if not np.isfinite(payload["D_features"]).all() or not np.isfinite(payload["V_features"]).all():
                raise ValueError(f"Non-finite feature cache for {cache['model_id']}")
    for run in probes["linear_runs"]:
        matches = list(probe_root.rglob(Path(str(run["head_file"])).name))
        if len(matches) != 1 or file_sha256(matches[0]) != run["head_sha256"]:
            raise ValueError(f"Linear-head hash failed for {run['run_id']}")
        state = torch.load(matches[0], map_location="cpu", weights_only=True)
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError(f"Non-finite linear head for {run['run_id']}")
    if len(seed_summary_paths) != 3:
        raise ValueError("Week 7 aggregation requires exactly three seed summaries")
    paths = [Path(value) for value in seed_summary_paths]
    seeds = [json.loads(path.read_text()) for path in paths]
    if {int(item["seed"]) for item in seeds} != set(WEEK5_SEEDS):
        raise ValueError("Week 7 seed summaries must cover 17, 29, and 43")
    if any(
        not item.get("week7_seed_complete") or item.get("run_count") != 6
        or item.get("evaluation_labels_loaded") is not False for item in seeds
    ):
        raise ValueError("A Week 7 seed bundle is incomplete or label-unsafe")
    if {item["week7_contracts_sha256"] for item in seeds} != {file_sha256(contracts_path)}:
        raise ValueError("Week 7 seed bundles use different foundation contracts")
    if {item["pilot_summary_sha256"] for item in seeds} != {file_sha256(pilot_summary_path)}:
        raise ValueError("Week 7 seed bundles use different pilot decisions")
    runs = [run for item in seeds for run in item["runs"]]
    expected = {
        (model, fraction, seed)
        for model in FOUNDATION_MODELS for fraction in FULL_FRACTIONS for seed in WEEK5_SEEDS
    }
    observed = {(run["model_id"], run["fraction_code"], int(run["seed"])) for run in runs}
    if len(runs) != 18 or observed != expected or len({run["run_id"] for run in runs}) != 18:
        raise ValueError("Week 7 full foundation matrix has missing or duplicate runs")
    batch = int(pilots["selected_batch_size"])
    expected_steps = {
        fraction: expected_epochs_and_steps(FRACTION_COUNTS[fraction], batch, 20, 300)[2]
        for fraction in FULL_FRACTIONS
    }
    if any(
        run.get("steps") != expected_steps[run["fraction_code"]]
        or run.get("expected_optimizer_steps") != expected_steps[run["fraction_code"]]
        or run.get("amp_overflow_skips") != 0 or not run.get("finite_gate")
        or not run.get("completion_gate") or run.get("evaluation_labels_loaded") is not False
        for run in runs
    ):
        raise ValueError("A Week 7 foundation run failed a completion gate")
    if len({run["best_checkpoint_sha256"] for run in runs}) != 18:
        raise ValueError("Week 7 foundation checkpoints are not unique")
    if len({run["source_tree_sha256"] for run in runs}) != 1:
        raise ValueError("Week 7 foundation runs were produced by different source trees")
    frozen_data = contracts["data_contract"]
    for run in runs:
        observed_data = {
            "manifest_contract_sha256": run["data"]["manifest_contract_sha256"],
            "normalization_sha256": run["data"]["normalization_sha256"],
            "subset_manifest_sha256": run["data"]["subset_manifest_sha256"],
        }
        if observed_data != frozen_data:
            raise ValueError(f"Frozen data hash mismatch for {run['run_id']}")
        if run["model_id"] == "M5":
            if run["foundation_name"] != "DINOv2 ViT-S/14" or "DINOv3" in json.dumps(run):
                raise ValueError("M5 must be reported explicitly as DINOv2")
            if run["initialization"]["weights_sha256"] != contracts["dinov2"]["weights_sha256"]:
                raise ValueError(f"DINOv2 initialization mismatch for {run['run_id']}")
        elif run["initialization"]["weights_sha256"] != contracts["olmoearth"]["weights_sha256"]:
            raise ValueError(f"OlmoEarth initialization mismatch for {run['run_id']}")
    checkpoint_ledger = []
    for path, seed_summary in zip(paths, seeds, strict=True):
        for run in seed_summary["runs"]:
            checkpoint = _artifact_path(path, run, "best_checkpoint")
            predictions = _artifact_path(path, run, "validation_predictions")
            if file_sha256(checkpoint) != run["best_checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash failed for {run['run_id']}")
            if file_sha256(predictions) != run["validation_predictions_sha256"]:
                raise ValueError(f"Prediction hash failed for {run['run_id']}")
            verify_foundation_checkpoint(checkpoint, run["run_id"])
            with np.load(predictions, allow_pickle=False) as payload:
                if payload["logits"].shape != (2000, 19) or len(payload["patch_ids"]) != 2000:
                    raise ValueError(f"Invalid V predictions for {run['run_id']}")
                if not np.isfinite(payload["logits"]).all():
                    raise ValueError(f"Non-finite V predictions for {run['run_id']}")
            checkpoint_ledger.append({
                "family": "foundation",
                "run_id": run["run_id"], "model_id": run["model_id"],
                "fraction_code": run["fraction_code"], "seed": run["seed"],
                "checkpoint_sha256": run["best_checkpoint_sha256"],
                "validation_macro_average_precision": run["validation_macro_average_precision"],
            })
    linear_expected = {
        (model, fraction, seed)
        for model in FOUNDATION_MODELS for fraction in PROBE_FRACTIONS for seed in WEEK5_SEEDS
    }
    linear_observed = {
        (run["model_id"], run["fraction_code"], int(run["seed"]))
        for run in probes["linear_runs"]
    }
    knn_expected = {(model, seed) for model in FOUNDATION_MODELS for seed in WEEK5_SEEDS}
    knn_observed = {(run["model_id"], int(run["seed"])) for run in probes["knn_runs"]}
    if linear_observed != linear_expected or knn_observed != knn_expected:
        raise ValueError("Week 7 probe identities are missing or duplicated")
    if any(not run.get("finite_gate") or run.get("evaluation_labels_loaded") is not False
           for run in probes["linear_runs"] + probes["knn_runs"]):
        raise ValueError("A Week 7 probe failed its finite or isolation gate")

    controlled = pd.read_csv(controlled_curves_path, dtype={"fraction_code": str})
    for row in controlled.itertuples(index=False):
        checkpoint_ledger.append({
            "family": "controlled",
            "run_id": str(row.run_id), "model_id": str(row.model_id),
            "fraction_code": str(row.fraction_code).zfill(2), "seed": int(row.seed),
            "checkpoint_sha256": str(row.checkpoint_sha256),
            "validation_macro_average_precision": float(row.validation_supported_map),
        })
    rgb_runs = [run for run in week6["new_runs"] if run["model_id"] == "M1RGB"]
    if len(rgb_runs) != 3:
        raise ValueError("Week 6 RGB controls are incomplete")
    for run in rgb_runs:
        checkpoint_ledger.append({
            "family": "rgb-control", "run_id": run["run_id"], "model_id": run["model_id"],
            "fraction_code": run["fraction_code"], "seed": int(run["seed"]),
            "checkpoint_sha256": run["best_checkpoint_sha256"],
            "validation_macro_average_precision": float(run["validation_macro_average_precision"]),
        })
    if len(checkpoint_ledger) != 111 or len({row["run_id"] for row in checkpoint_ledger}) != 111:
        raise ValueError("Week 8 checkpoint ledger must contain 111 unique controlled and foundation states")
    comparisons = []
    for model_id in FOUNDATION_MODELS:
        for fraction in FULL_FRACTIONS:
            foundation_values = [
                float(run["validation_macro_average_precision"])
                for run in runs if run["model_id"] == model_id and run["fraction_code"] == fraction
            ]
            foundation_mean, foundation_sd = _mean_sd(foundation_values)
            for baseline in ("M0", "M1", "M3"):
                baseline_values = controlled[
                    controlled["model_id"].eq(baseline)
                    & controlled["fraction_code"].astype(str).str.zfill(2).eq(fraction)
                ]["validation_supported_map"].astype(float).tolist()
                if len(baseline_values) != 3:
                    raise ValueError(f"Missing controlled baseline {baseline} at {fraction}")
                differences = [left - right for left, right in zip(foundation_values, baseline_values, strict=True)]
                difference_mean, difference_sd = _mean_sd(differences)
                comparisons.append({
                    "foundation_model": model_id, "baseline_model": baseline,
                    "fraction_code": fraction, "foundation_mean": foundation_mean,
                    "foundation_sample_std": foundation_sd,
                    "paired_difference_mean": difference_mean,
                    "paired_difference_sample_std": difference_sd,
                })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "foundation_comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    with (output_dir / "week8_checkpoint_ledger.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checkpoint_ledger[0]))
        writer.writeheader()
        writer.writerows(checkpoint_ledger)
    result = {
        "week7_complete": True, "week8_approved": True,
        "foundation_full_run_count": 18, "foundation_linear_probe_count": 36,
        "foundation_knn_run_count": 6, "foundation_feature_cache_count": 2,
        "foundation_models": {
            "M5": "DINOv2 ViT-S/14 (declared fallback; not DINOv3)",
            "M6": "OlmoEarth v1.1 Tiny",
        },
        "selected_batch_size": batch,
        "week8_checkpoint_ledger_count": 111,
        "selected_learning_rate_multipliers": {
            model: pilots["selected"][model]["learning_rate_multiplier"] for model in FOUNDATION_MODELS
        },
        "foundation_runs": runs, "foundation_comparisons": comparisons,
        "week8_checkpoint_ledger": checkpoint_ledger,
        "actual_gpu_hours": sum(float(item["actual_gpu_hours"]) for item in seeds),
        "week6_summary_sha256": file_sha256(week6_path),
        "week7_contracts_sha256": file_sha256(contracts_path),
        "pilot_summary_sha256": file_sha256(pilot_summary_path),
        "probe_summary_sha256": file_sha256(probes_path),
        "evaluation_labels_loaded": False,
    }
    (output_dir / "week7_run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
