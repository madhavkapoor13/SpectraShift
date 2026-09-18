from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from spectrashift.data.downstream import FRACTION_COUNTS, WEEK5_SEEDS
from spectrashift.data.foundation import (
    DINOv2RGBDataset,
    FoundationDownstreamDataset,
    OlmoEarthS2Dataset,
)
from spectrashift.eval.metrics import fit_global_validation_threshold, multilabel_metrics
from spectrashift.models.foundation import (
    build_dinov2_classifier,
    build_olmoearth_classifier,
)

from .common import (
    append_jsonl,
    atomic_torch_save,
    capture_rng_state,
    file_sha256,
    git_commit,
    hardware_record,
    object_sha256,
    restore_rng_state,
    seed_everything,
    select_device,
    source_tree_sha256,
)
from .downstream import expected_epochs_and_steps, model_state_sha256


FOUNDATION_MODELS = ("M5", "M6")
FOUNDATION_FEATURE_DIMS = {"M5": 384, "M6": 192}


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def validate_foundation_config(config: dict[str, object]) -> None:
    run, training = config["run"], config["training"]
    model_id, seed, fraction = str(run["model_id"]), int(run["seed"]), str(run["fraction_code"])
    if model_id not in FOUNDATION_MODELS or seed not in WEEK5_SEEDS or fraction not in FRACTION_COUNTS:
        raise ValueError("Unexpected Week 7 foundation run identity")
    if int(run["sample_count"]) != FRACTION_COUNTS[fraction]:
        raise ValueError("Foundation sample count differs from the frozen subset")
    if int(training["batch_size"]) not in {16, 32}:
        raise ValueError("Week 7 physical batch must be the frozen 16 or 32 contract")
    expected = {
        "weight_decay": 0.05,
        "warmup_fraction": 0.05,
        "gradient_clip": 1.0,
        "amp_initial_scale": 128.0,
    }
    for key, value in expected.items():
        if not math.isclose(float(training[key]), value):
            raise ValueError(f"Week 7 training.{key} must be {value}")
    if not training.get("amp") or not training.get("require_cuda") or not training.get("fail_on_amp_overflow"):
        raise ValueError("Week 7 requires CUDA AMP with immediate overflow failure")
    epochs, _, steps = expected_epochs_and_steps(
        int(run["sample_count"]), int(training["batch_size"]),
        int(training["base_epochs"]), int(training["minimum_optimizer_steps"]),
    )
    if epochs != int(training["base_epochs"]) or steps != int(training["expected_optimizer_steps"]):
        raise ValueError("Week 7 exact schedule is inconsistent")
    expected_eval = 1 if run.get("pilot") or fraction != "01" else 5
    if int(training["evaluation_every_epochs"]) != expected_eval:
        raise ValueError("Week 7 validation cadence differs from the frozen protocol")


def _build_model(config: dict[str, object]) -> tuple[torch.nn.Module, dict[str, object]]:
    model_id = str(config["run"]["model_id"])
    initialization = config["initialization"]
    if model_id == "M5":
        model = build_dinov2_classifier(
            initialization["source_dir"], initialization["weights_path"],
            initialization["source_sha256"], initialization["weights_sha256"],
        )
        record = {
            "type": "official-dinov2-vits14",
            "source_revision": initialization["source_revision"],
            "source_sha256": initialization["source_sha256"],
            "weights_sha256": initialization["weights_sha256"],
            "register_tokens": 0,
            "pooling": "mean-normalized-patch-tokens",
        }
    else:
        model = build_olmoearth_classifier(
            initialization["source_dir"], initialization["model_dir"],
            initialization["source_sha256"], initialization["config_sha256"],
            initialization["weights_sha256"],
        )
        record = {
            "type": "official-olmoearth-v1_1-tiny",
            "source_revision": initialization["source_revision"],
            "model_revision": initialization["model_revision"],
            "source_sha256": initialization["source_sha256"],
            "config_sha256": initialization["config_sha256"],
            "weights_sha256": initialization["weights_sha256"],
            "pooling": "valid-spatial-token-mean",
        }
    record["model_state_sha256"] = model_state_sha256(model)
    return model, record


def _make_base(config: dict[str, object], partition: str):
    data = config["data"]
    if config["run"]["model_id"] == "M5":
        return DINOv2RGBDataset(
            data["manifest_path"], data["staged_root"], data["normalization_path"],
            data["rgb_contract_path"], partition, int(data.get("shard_size", 512)),
            int(data.get("height", 120)), int(data.get("width", 120)), 126,
        )
    return OlmoEarthS2Dataset(
        data["manifest_path"], data["staged_root"], data["olmo_contract_path"],
        partition, int(data.get("shard_size", 512)), int(data.get("height", 120)),
        int(data.get("width", 120)),
    )


def build_foundation_optimizer(
    model: torch.nn.Module, encoder_lr: float, head_lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    head_parameters = {id(value) for value in model.head.parameters()}
    groups: dict[str, list[torch.nn.Parameter]] = {
        "encoder_decay": [], "encoder_no_decay": [],
        "head_decay": [], "head_no_decay": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        prefix = "head" if id(parameter) in head_parameters else "encoder"
        suffix = "decay" if not (name.endswith("bias") or parameter.ndim <= 1) else "no_decay"
        groups[f"{prefix}_{suffix}"].append(parameter)
    return torch.optim.AdamW([
        {"params": groups["encoder_decay"], "lr": encoder_lr, "weight_decay": weight_decay, "group_name": "encoder_decay"},
        {"params": groups["encoder_no_decay"], "lr": encoder_lr, "weight_decay": 0.0, "group_name": "encoder_no_decay"},
        {"params": groups["head_decay"], "lr": head_lr, "weight_decay": weight_decay, "group_name": "head_decay"},
        {"params": groups["head_no_decay"], "lr": head_lr, "weight_decay": 0.0, "group_name": "head_no_decay"},
    ])


def _schedule(step: int, warmup: int, total: int, minimum_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def _evaluate(model, dataset, device, batch_size: int, workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    model.eval()
    logits, targets, patch_ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            patch_ids.extend(map(str, batch.pop("patch_id")))
            targets.append(batch.pop("labels"))
            logits.append(model(_move_batch(batch, device)).float().cpu())
    return torch.cat(logits).numpy(), torch.cat(targets).numpy(), patch_ids


def _save_predictions(path: Path, logits: np.ndarray, targets: np.ndarray, patch_ids: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, logits=logits.astype(np.float32),
                            targets=targets.astype(np.uint8), patch_ids=np.asarray(patch_ids, dtype="U"))
    os.replace(temporary, path)


def verify_foundation_checkpoint(path: str | Path, expected_run_id: str | None = None) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"run_id", "model_id", "seed", "fraction_code", "model", "config_sha256", "initialization"}
    if missing := required - set(payload):
        raise ValueError(f"Foundation checkpoint is missing fields: {sorted(missing)}")
    if expected_run_id and payload["run_id"] != expected_run_id:
        raise ValueError("Foundation checkpoint identity mismatch")
    if any(not torch.isfinite(value).all() for value in payload["model"].values()):
        raise ValueError("Foundation checkpoint contains non-finite parameters")
    return payload


def train_foundation(config_path: str | Path, resume_path: str | Path | None = None) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    validate_foundation_config(config)
    run, training, data = config["run"], config["training"], config["data"]
    output_dir = Path(run["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    config_hash = object_sha256(config)
    seed = int(run["seed"])
    seed_everything(seed)
    device = select_device(bool(training["require_cuda"]))
    normalization = json.loads(Path(data["normalization_path"]).read_text())
    freeze = json.loads(Path(data["freeze_summary_path"]).read_text())
    if normalization.get("sha256") != data["normalization_sha256"]:
        raise ValueError("Week 2 normalization contract mismatch")
    if not freeze.get("training_approved") or freeze.get("manifest_sha256") != data["manifest_contract_sha256"]:
        raise ValueError("Week 2 manifest does not approve foundation training")
    subset_contract = json.loads(Path(config["subsets"]["contract_path"]).read_text())
    subset_frame = pd.read_parquet(config["subsets"]["manifest_path"])
    rows = subset_frame[subset_frame["downstream_seed"].eq(seed)].sort_values("subset_rank")
    count = int(run["sample_count"])
    patch_ids = rows.head(count)["patch_id"].astype(str).tolist()
    if len(patch_ids) != count or subset_contract["subset_manifest_sha256"] != config["subsets"]["sha256"]:
        raise ValueError("Frozen downstream subset contract mismatch")
    train_dataset = FoundationDownstreamDataset(_make_base(config, "D"), patch_ids, seed, True)
    validation_dataset = FoundationDownstreamDataset(_make_base(config, "V"), None, seed, False)
    model, initialization = _build_model(config)
    model = model.to(device)
    encoder_lr, head_lr = float(training["encoder_learning_rate"]), float(training["head_learning_rate"])
    optimizer = build_foundation_optimizer(model, encoder_lr, head_lr, float(training["weight_decay"]))
    epochs, batches, expected_steps = expected_epochs_and_steps(
        count, int(training["batch_size"]), int(training["base_epochs"]),
        int(training["minimum_optimizer_steps"]),
    )
    if expected_steps != int(training["expected_optimizer_steps"]):
        raise ValueError("Foundation run has the wrong exact step count")
    warmup = max(1, round(expected_steps * float(training["warmup_fraction"])))
    minimum_ratio = float(training["minimum_encoder_learning_rate"]) / encoder_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _schedule(step, warmup, expected_steps, minimum_ratio)
    )
    amp_enabled = device.type == "cuda" and bool(training["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled,
                                 init_scale=float(training["amp_initial_scale"]),
                                 growth_interval=int(training["amp_growth_interval"]))
    criterion = torch.nn.BCEWithLogitsLoss()
    start_epoch = global_step = overflow_skips = 0
    elapsed_before = 0.0
    best_map, best_epoch = float("-inf"), -1
    history: list[dict[str, object]] = []
    best_path, recovery = output_dir / "best-model.pt", output_dir / "recovery-latest.pt"
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError("Resume checkpoint configuration differs from this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch, global_step = int(checkpoint["next_epoch"]), int(checkpoint["global_step"])
        overflow_skips, elapsed_before = int(checkpoint["amp_overflow_skips"]), float(checkpoint["elapsed_seconds"])
        best_map, best_epoch, history = float(checkpoint["best_map"]), int(checkpoint["best_epoch"]), list(checkpoint["history"])
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        loader = DataLoader(train_dataset, batch_size=int(training["batch_size"]), shuffle=True,
                            generator=torch.Generator().manual_seed(seed + epoch), drop_last=False,
                            num_workers=int(training["num_workers"]), pin_memory=device.type == "cuda")
        if len(loader) != batches:
            raise RuntimeError("Foundation batch count changed from the frozen schedule")
        model.train()
        losses = []
        for batch in loader:
            batch.pop("patch_id")
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = criterion(model(batch), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite foundation loss at step {global_step}")
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError(f"Invalid foundation gradient at step {global_step}")
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                overflow_skips += 1
                raise FloatingPointError(f"AMP overflow at foundation step {global_step}")
            scheduler.step()
            global_step += 1
            losses.append(float(loss.detach()))
        evaluate = (epoch + 1) % int(training["evaluation_every_epochs"]) == 0 or epoch + 1 == epochs
        record: dict[str, object] = {
            "epoch": epoch + 1, "global_step": global_step,
            "train_loss": float(np.mean(losses)),
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[2]["lr"],
        }
        if evaluate:
            logits, targets, validation_ids = _evaluate(
                model, validation_dataset, device, int(training["validation_batch_size"]),
                int(training["num_workers"]),
            )
            probabilities = 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
            metrics = multilabel_metrics(targets, probabilities,
                                         supported_indices=subset_contract["supported_class_indices"])
            score = float(metrics["macro_average_precision"])
            record["validation_macro_average_precision"] = score
            if score > best_map:
                best_map, best_epoch = score, epoch + 1
                atomic_torch_save({
                    "run_id": run["id"], "model_id": run["model_id"], "seed": seed,
                    "fraction_code": run["fraction_code"],
                    "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "config_sha256": config_hash, "initialization": initialization,
                    "epoch": best_epoch, "validation_macro_average_precision": best_map,
                }, best_path)
                _save_predictions(output_dir / "validation_predictions.npz", logits, targets, validation_ids)
        history.append(record)
        append_jsonl(output_dir / "training.jsonl", record)
        if evaluate:
            atomic_torch_save({
                "config_sha256": config_hash,
                "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng_state": capture_rng_state(),
                "next_epoch": epoch + 1, "global_step": global_step,
                "amp_overflow_skips": overflow_skips,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "best_map": best_map, "best_epoch": best_epoch, "history": history,
            }, recovery)
    elapsed = elapsed_before + time.perf_counter() - started
    if global_step != expected_steps or overflow_skips:
        raise ValueError("Foundation run failed its successful-step contract")
    best = verify_foundation_checkpoint(best_path, str(run["id"]))
    model.load_state_dict(best["model"])
    logits, targets, validation_ids = _evaluate(model, validation_dataset, device,
                                                 int(training["validation_batch_size"]),
                                                 int(training["num_workers"]))
    probabilities = 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
    threshold = fit_global_validation_threshold(targets, probabilities)
    metrics_half = multilabel_metrics(targets, probabilities, thresholds=0.5,
                                      supported_indices=subset_contract["supported_class_indices"])
    metrics_selected = multilabel_metrics(targets, probabilities, thresholds=float(threshold["threshold"]),
                                          supported_indices=subset_contract["supported_class_indices"])
    _save_predictions(output_dir / "validation_predictions.npz", logits, targets, validation_ids)
    result = {
        "run_id": run["id"], "model_id": run["model_id"], "foundation_name": run["foundation_name"],
        "seed": seed, "fraction_code": run["fraction_code"], "sample_count": count,
        "epochs": epochs, "steps": global_step, "expected_optimizer_steps": expected_steps,
        "best_epoch": best_epoch, "validation_macro_average_precision": best_map,
        "metrics_at_0_5": metrics_half, "selected_global_threshold": threshold,
        "metrics_at_selected_threshold": metrics_selected,
        "amp_overflow_skips": overflow_skips, "elapsed_seconds": elapsed,
        "initialization": initialization, "preprocessing": data["preprocessing"],
        "feature_dimension": FOUNDATION_FEATURE_DIMS[str(run["model_id"])],
        "config_sha256": config_hash, "source_tree_sha256": source_tree_sha256(),
        "git_commit": git_commit(), "hardware": hardware_record(device),
        "data": {
            "manifest_file_sha256": file_sha256(data["manifest_path"]),
            "manifest_contract_sha256": data["manifest_contract_sha256"],
            "normalization_sha256": data["normalization_sha256"],
            "subset_manifest_sha256": subset_contract["subset_manifest_sha256"],
            "train_patches": count, "validation_patches": len(validation_dataset),
        },
        "best_checkpoint": str(best_path), "best_checkpoint_sha256": file_sha256(best_path),
        "validation_predictions": str(output_dir / "validation_predictions.npz"),
        "validation_predictions_sha256": file_sha256(output_dir / "validation_predictions.npz"),
        "finite_gate": True, "completion_gate": True, "evaluation_labels_loaded": False,
    }
    (output_dir / "foundation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    recovery.unlink(missing_ok=True)
    return result


def cache_foundation_features(config_path: str | Path, output_path: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    validate_foundation_config(config)
    seed_everything(int(config["run"]["seed"]))
    device = select_device(bool(config["training"]["require_cuda"]))
    model, initialization = _build_model(config)
    model = model.to(device).eval()
    encoder_hash_before = model_state_sha256(model.encoder)
    arrays: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for partition in ("D", "V"):
        dataset = FoundationDownstreamDataset(_make_base(config, partition), None,
                                              int(config["run"]["seed"]), False)
        loader = DataLoader(dataset, batch_size=int(config["probes"]["feature_batch_size"]),
                            shuffle=False, num_workers=int(config["training"]["num_workers"]),
                            pin_memory=device.type == "cuda")
        features, ids = [], []
        with torch.inference_mode():
            for batch in loader:
                ids.extend(map(str, batch.pop("patch_id")))
                batch.pop("labels")
                features.append(model.features(_move_batch(batch, device)).float().cpu().numpy())
        arrays[f"{partition}_features"] = np.concatenate(features).astype(np.float16)
        arrays[f"{partition}_patch_ids"] = np.asarray(ids, dtype="U")
    if model_state_sha256(model.encoder) != encoder_hash_before:
        raise ValueError("Frozen foundation encoder changed during feature extraction")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    result = {
        "model_id": config["run"]["model_id"], "foundation_name": config["run"]["foundation_name"],
        "initialization": initialization, "encoder_state_sha256": encoder_hash_before,
        "feature_dimension": FOUNDATION_FEATURE_DIMS[str(config["run"]["model_id"])],
        "d_patches": len(arrays["D_patch_ids"]), "v_patches": len(arrays["V_patch_ids"]),
        "path": str(output_path), "sha256": file_sha256(output_path),
        "elapsed_seconds": time.perf_counter() - started, "evaluation_labels_loaded": False,
    }
    output_path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result
