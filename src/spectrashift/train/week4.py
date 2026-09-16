from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import torch
import yaml

from spectrashift.data.bands import BAND_ADAPTERS

from .common import append_jsonl, file_sha256, object_sha256
from .ssl import train_ssl, verify_encoder_export


WEEK4_SEEDS = (17, 29, 43)
WEEK4_MODELS = {
    "M2": {"adapter": "rgb", "spectral_dropout_probability": 0.0},
    "M3": {"adapter": "core10", "spectral_dropout_probability": 0.0},
    "M4": {"adapter": "core10", "spectral_dropout_probability": 0.25},
}
WEEK3_SUMMARY_SHA256 = "38086d2d1fe0683c5d78f27a0316496c0cb241258770ffd6d9ba21e554791995"
SELECTED_LEARNING_RATE = 1e-4
EXPECTED_U_PATCHES = 20_000
EXPECTED_BATCH_SIZE = 64
EXPECTED_EPOCHS = 60
EXPECTED_STEPS = 18_720


def expected_run_id(model_id: str, seed: int) -> str:
    return f"week4-{model_id.lower()}-seed{seed}"


def validate_week3_approval(path: str | Path) -> dict[str, object]:
    path = Path(path)
    if file_sha256(path) != WEEK3_SUMMARY_SHA256:
        raise ValueError("Week 3 summary hash does not match the approved pilot artifact")
    summary = json.loads(path.read_text())
    selection = summary.get("week4_selection", {})
    selected = selection.get("selected", {})
    if not summary.get("week3_complete") or not selection.get("week4_approved"):
        raise ValueError("Week 3 did not approve Week 4")
    if selected.get("run_id") != "week3-m3-lr1e4-seed17" or not math.isclose(
        float(selected.get("learning_rate", -1)), SELECTED_LEARNING_RATE
    ):
        raise ValueError("Week 3 summary does not select the frozen 1e-4 schedule")
    if summary.get("evaluation_labels_loaded") is not False:
        raise ValueError("Week 3 provenance does not prove evaluation-label isolation")
    return summary


def validate_week4_config(config: dict[str, object]) -> None:
    run = config["run"]
    data = config["data"]
    augmentation = config["augmentation"]
    training = config["training"]
    model_id = str(run["model_id"])
    seed = int(run["seed"])
    if model_id not in WEEK4_MODELS or seed not in WEEK4_SEEDS:
        raise ValueError(f"Unexpected Week 4 model/seed: {model_id}/{seed}")
    if run["id"] != expected_run_id(model_id, seed):
        raise ValueError("Week 4 run ID does not match its model and seed")
    contract = WEEK4_MODELS[model_id]
    if data["adapter"] != contract["adapter"]:
        raise ValueError(f"{model_id} must use {contract['adapter']} inputs")
    if not math.isclose(
        float(augmentation["spectral_dropout_probability"]),
        float(contract["spectral_dropout_probability"]),
    ):
        raise ValueError(f"{model_id} has the wrong spectral-dropout probability")
    exact = {
        "epochs": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "warmup_epochs": 5,
        "checkpoint_every_epochs": 10,
        "expected_optimizer_steps": EXPECTED_STEPS,
        "amp_growth_interval": 1_000_000,
    }
    for key, value in exact.items():
        if int(training[key]) != value:
            raise ValueError(f"Week 4 training.{key} must be {value}")
    numeric = {
        "learning_rate": SELECTED_LEARNING_RATE,
        "minimum_learning_rate": 1e-6,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "amp_initial_scale": 128.0,
    }
    for key, value in numeric.items():
        if not math.isclose(float(training[key]), value):
            raise ValueError(f"Week 4 training.{key} must be {value}")
    if not training.get("amp") or not training.get("require_cuda"):
        raise ValueError("Week 4 requires CUDA AMP")
    if not training.get("fail_on_amp_overflow"):
        raise ValueError("Week 4 must stop immediately on any AMP overflow")
    if not training.get("retain_final_checkpoint_only"):
        raise ValueError("Week 4 must prune intermediate checkpoints after success")


def _artifact_path(summary_path: Path, recorded: str) -> Path:
    return summary_path.parent / Path(recorded).name


def validate_complete_run(
    summary_path: str | Path, config: dict[str, object]
) -> dict[str, object]:
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    validate_week4_config(config)
    run = config["run"]
    if summary.get("run_id") != run["id"] or summary.get("model_id") != run["model_id"]:
        raise ValueError("Run summary identity does not match its configuration")
    if int(summary.get("seed", -1)) != int(run["seed"]):
        raise ValueError("Run summary seed does not match its configuration")
    if summary.get("config_sha256") != object_sha256(config):
        raise ValueError("Run summary configuration hash does not match")
    if int(summary.get("steps", -1)) != EXPECTED_STEPS:
        raise ValueError("Run did not complete the exact Week 4 optimizer-step count")
    if int(summary.get("amp_overflow_skips", -1)) != 0:
        raise ValueError("Run recorded AMP overflows")
    if not summary.get("stability_gate") or not summary.get("completion_gate"):
        raise ValueError("Run did not pass stability and completion gates")
    if summary.get("data", {}).get("u_patches") != EXPECTED_U_PATCHES:
        raise ValueError("Run did not use the frozen 20,000-patch U partition")
    expected_bands = list(BAND_ADAPTERS[config["data"]["adapter"]])
    if summary.get("model", {}).get("band_order") != expected_bands:
        raise ValueError("Run summary band order does not match the adapter")
    checkpoint = _artifact_path(summary_path, str(summary["checkpoint"]))
    encoder = _artifact_path(summary_path, str(summary["encoder_export"]))
    if file_sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("Final checkpoint hash verification failed")
    if file_sha256(encoder) != summary["encoder_export_sha256"]:
        raise ValueError("Encoder export hash verification failed")
    verify_encoder_export(encoder, str(summary["checkpoint_sha256"]))
    return summary


def _find_completed_run(
    roots: list[Path], run_id: str, config: dict[str, object]
) -> tuple[Path, dict[str, object]] | None:
    matches = []
    for root in roots:
        matches.extend(root.rglob(f"{run_id}/ssl_summary.json"))
    valid = []
    for path in sorted(set(matches)):
        try:
            valid.append((path, validate_complete_run(path, config)))
        except (KeyError, OSError, RuntimeError, ValueError):
            continue
    if len(valid) > 1:
        hashes = {item[1]["checkpoint_sha256"] for item in valid}
        if len(hashes) > 1:
            raise ValueError(f"Conflicting completed resume artifacts for {run_id}")
    return valid[0] if valid else None


def _find_resume_checkpoint(
    roots: list[Path], run_id: str, config: dict[str, object]
) -> Path | None:
    candidates = []
    for root in roots:
        candidates.extend(root.rglob(f"{run_id}/checkpoint-epoch-*.pt"))
    candidates.sort(key=lambda path: path.name, reverse=True)
    expected_hash = object_sha256(config)
    for path in candidates:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        matches = payload.get("config_sha256") == expected_hash
        incomplete = int(payload.get("next_epoch", 0)) < EXPECTED_EPOCHS
        del payload
        if matches and incomplete:
            return path
    return None


def run_week4_seed(
    config_paths: list[str | Path],
    output_root: str | Path,
    week3_summary_path: str | Path,
    resume_roots: list[str | Path] | None = None,
) -> dict[str, object]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    week3 = validate_week3_approval(week3_summary_path)
    configs = []
    for path in config_paths:
        config = yaml.safe_load(Path(path).read_text())
        validate_week4_config(config)
        configs.append((Path(path), config))
    seeds = {int(config["run"]["seed"]) for _, config in configs}
    models = {str(config["run"]["model_id"]) for _, config in configs}
    if len(seeds) != 1 or models != set(WEEK4_MODELS) or len(configs) != 3:
        raise ValueError("A Week 4 seed job must contain exactly M2, M3, and M4 for one seed")
    seed = seeds.pop()
    roots = [Path(path) for path in (resume_roots or [])]
    summaries = []
    for config_path, config in sorted(configs, key=lambda item: item[1]["run"]["model_id"]):
        run_id = str(config["run"]["id"])
        expected_output = output_root / run_id
        if Path(config["run"]["output_dir"]) != expected_output:
            raise ValueError(f"Runtime output directory is incorrect for {run_id}")
        completed = _find_completed_run(roots, run_id, config)
        if completed:
            source_summary, _ = completed
            shutil.copytree(source_summary.parent, expected_output, dirs_exist_ok=True)
            summary = validate_complete_run(expected_output / "ssl_summary.json", config)
            append_jsonl(output_root / "runs.jsonl", {**summary, "status": "complete", "reused": True})
        else:
            resume_path = _find_resume_checkpoint(roots, run_id, config)
            if resume_path:
                source_log = resume_path.parent / "training.jsonl"
                expected_output.mkdir(parents=True, exist_ok=True)
                if source_log.exists():
                    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
                    step_limit = int(checkpoint["global_step"])
                    del checkpoint
                    retained = []
                    for line in source_log.read_text().splitlines():
                        record = json.loads(line)
                        if int(record.get("global_step", -1)) < step_limit:
                            retained.append(line)
                    (expected_output / "training.jsonl").write_text(
                        "\n".join(retained) + ("\n" if retained else "")
                    )
            summary = train_ssl(config_path, resume_path=resume_path)
            summary = validate_complete_run(expected_output / "ssl_summary.json", config)
        summaries.append(summary)
    result = {
        "week4_seed_complete": True,
        "seed": seed,
        "models": ["M2", "M3", "M4"],
        "runs": summaries,
        "actual_gpu_hours": sum(float(item["elapsed_seconds"]) for item in summaries) / 3600,
        "week3_summary_sha256": file_sha256(week3_summary_path),
        "selected_learning_rate": float(week3["week4_selection"]["selected"]["learning_rate"]),
        "evaluation_labels_loaded": False,
    }
    path = output_root / f"week4_seed{seed}_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def aggregate_week4(
    seed_summary_paths: list[str | Path], output_path: str | Path
) -> dict[str, object]:
    if len(seed_summary_paths) != 3:
        raise ValueError("Week 4 aggregation requires exactly three seed summaries")
    summary_paths = [Path(path) for path in seed_summary_paths]
    seed_summaries = [json.loads(path.read_text()) for path in summary_paths]
    if {int(item["seed"]) for item in seed_summaries} != set(WEEK4_SEEDS):
        raise ValueError("Week 4 summaries must cover seeds 17, 29, and 43 exactly once")
    runs = [run for seed_summary in seed_summaries for run in seed_summary["runs"]]
    expected = {(model, seed) for model in WEEK4_MODELS for seed in WEEK4_SEEDS}
    observed = {(str(run["model_id"]), int(run["seed"])) for run in runs}
    if len(runs) != 9 or observed != expected:
        raise ValueError("Week 4 aggregation is missing or duplicating model-seed runs")
    if any(not item.get("week4_seed_complete") for item in seed_summaries):
        raise ValueError("At least one Week 4 seed job is incomplete")
    if {item.get("week3_summary_sha256") for item in seed_summaries} != {WEEK3_SUMMARY_SHA256}:
        raise ValueError("Week 4 seed jobs do not share the approved Week 3 decision artifact")
    if any(not run.get("stability_gate") or not run.get("completion_gate") for run in runs):
        raise ValueError("At least one Week 4 run failed its gates")
    if len({run.get("source_tree_sha256") for run in runs}) != 1:
        raise ValueError("Week 4 runs were produced by different source trees")
    if any("T4" not in str(run.get("hardware", {}).get("gpu", "")) for run in runs):
        raise ValueError("Week 4 runs must use the pilot-compatible T4 GPU")
    for seed_summary_path, seed_summary in zip(summary_paths, seed_summaries, strict=True):
        if seed_summary.get("evaluation_labels_loaded") is not False:
            raise ValueError("A Week 4 seed summary does not prove evaluation-label isolation")
        for run in seed_summary["runs"]:
            run_dir = seed_summary_path.parent / str(run["run_id"])
            checkpoint = run_dir / Path(str(run["checkpoint"])).name
            encoder = run_dir / Path(str(run["encoder_export"])).name
            if file_sha256(checkpoint) != run["checkpoint_sha256"]:
                raise ValueError(f"Final checkpoint hash failed for {run['run_id']}")
            if file_sha256(encoder) != run["encoder_export_sha256"]:
                raise ValueError(f"Encoder export hash failed for {run['run_id']}")
            verify_encoder_export(encoder, str(run["checkpoint_sha256"]))
    encoder_hashes = [str(run["encoder_export_sha256"]) for run in runs]
    if len(set(encoder_hashes)) != 9:
        raise ValueError("Week 4 must produce nine unique encoder artifacts")
    data_contracts = {
        (
            run["data"]["manifest_contract_sha256"],
            run["data"]["normalization_sha256"],
            int(run["data"]["u_patches"]),
        )
        for run in runs
    }
    if len(data_contracts) != 1:
        raise ValueError("Week 4 runs do not share one frozen data contract")
    result = {
        "week4_complete": True,
        "week5_approved": True,
        "run_count": 9,
        "models": ["M2", "M3", "M4"],
        "seeds": list(WEEK4_SEEDS),
        "learning_rate": SELECTED_LEARNING_RATE,
        "actual_gpu_hours": sum(float(run["elapsed_seconds"]) for run in runs) / 3600,
        "data_contract": {
            "manifest_contract_sha256": runs[0]["data"]["manifest_contract_sha256"],
            "normalization_sha256": runs[0]["data"]["normalization_sha256"],
            "u_patches": runs[0]["data"]["u_patches"],
        },
        "encoder_sha256": {
            str(run["run_id"]): str(run["encoder_export_sha256"]) for run in runs
        },
        "runs": runs,
        "evaluation_labels_loaded": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
