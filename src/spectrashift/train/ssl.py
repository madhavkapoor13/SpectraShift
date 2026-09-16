from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from spectrashift.data.bands import BAND_ADAPTERS
from spectrashift.data.dataset import SpectraShiftDataset
from spectrashift.data.views import TwoViewTransform, ViewConfig, deterministic_view_seed
from spectrashift.losses.vicreg import VICRegLoss, effective_rank
from spectrashift.models.vicreg import VICRegModel

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


class EpochShuffleSampler(Sampler[tuple[int, int]]):
    def __init__(self, size: int, seed: int) -> None:
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        for index in torch.randperm(self.size, generator=generator).tolist():
            yield index, self.epoch

    def __len__(self) -> int:
        return self.size


class SSLPairDataset(Dataset):
    def __init__(self, base: SpectraShiftDataset, transform: TwoViewTransform, seed: int) -> None:
        self.base = base
        self.transform = transform
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, object]:
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        item = self.base[int(index)]
        if item["labels"] is not None:
            raise ValueError("SSL loader received labels")
        view_seed = deterministic_view_seed(self.seed, int(epoch), str(item["patch_id"]))
        transformed = self.transform(torch.from_numpy(item["image"]), view_seed)
        transformed["patch_id"] = str(item["patch_id"])
        return transformed


def _scheduler_lambda(step: int, warmup_steps: int, total_steps: int, minimum_ratio: float) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    remaining = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / remaining, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return minimum_ratio + (1 - minimum_ratio) * cosine


def _resolved_data(config: dict) -> tuple[SpectraShiftDataset, dict[str, object]]:
    data = config["data"]
    manifest_path = Path(data["manifest_path"])
    normalization_path = Path(data["normalization_path"])
    freeze_summary_path = Path(data["freeze_summary_path"])
    normalization = json.loads(normalization_path.read_text())
    freeze_summary = json.loads(freeze_summary_path.read_text())
    expected_normalization = data.get("normalization_sha256")
    if expected_normalization and normalization.get("sha256") != expected_normalization:
        raise ValueError("Normalization contract hash does not match the frozen Week 2 artifact")
    if not freeze_summary.get("training_approved"):
        raise ValueError("Week 2 freeze summary does not approve training")
    if freeze_summary.get("manifest_sha256") != data["manifest_contract_sha256"]:
        raise ValueError("Frozen manifest contract hash does not match the Week 2 freeze summary")
    base = SpectraShiftDataset(
        manifest_path,
        data["staged_root"],
        normalization_path,
        "U",
        data["adapter"],
        int(data.get("shard_size", 512)),
        int(data.get("height", 120)),
        int(data.get("width", 120)),
    )
    return base, {
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_contract_sha256": data["manifest_contract_sha256"],
        "freeze_summary_file_sha256": file_sha256(freeze_summary_path),
        "normalization_sha256": normalization["sha256"],
        "u_patches": len(base),
    }


def verify_encoder_export(path: str | Path, expected_checkpoint_sha256: str) -> dict[str, object]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format_version", "run_id", "model_id", "seed", "adapter", "band_order",
        "feature_dimension", "encoder_parameters", "encoder", "source_checkpoint_sha256",
        "config_sha256", "data",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Encoder export is missing fields: {sorted(missing)}")
    if payload["source_checkpoint_sha256"] != expected_checkpoint_sha256:
        raise ValueError("Encoder export references the wrong final checkpoint")
    if not isinstance(payload["encoder"], dict) or not payload["encoder"]:
        raise ValueError("Encoder export has no state dictionary")
    if any(not torch.isfinite(value).all() for value in payload["encoder"].values()):
        raise ValueError("Encoder export contains non-finite parameters or buffers")
    return payload


def checkpoint_due(completed_epoch: int, total_epochs: int, every_epochs: int) -> bool:
    if every_epochs <= 0:
        raise ValueError("every_epochs must be positive")
    return completed_epoch % every_epochs == 0 or completed_epoch == total_epochs


def prune_intermediate_checkpoints(output_dir: str | Path, final_checkpoint: str | Path) -> list[str]:
    output_dir = Path(output_dir)
    final_checkpoint = Path(final_checkpoint)
    removed = []
    for candidate in output_dir.glob("checkpoint-epoch-*.pt"):
        if candidate != final_checkpoint:
            os.unlink(candidate)
            removed.append(candidate.name)
    return sorted(removed)


def train_ssl(
    config_path: str | Path,
    seed_override: int | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    run = config["run"]
    training = config["training"]
    model_config = config["model"]
    seed = int(seed_override if seed_override is not None else run["seed"])
    output_dir = Path(run["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(run.get("ledger_path", output_dir.parent / "runs.jsonl"))
    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    seed_everything(seed)
    device = select_device(bool(training.get("require_cuda", False)))
    base, data_record = _resolved_data(config)
    view_config = ViewConfig(**config["augmentation"])
    pair_dataset = SSLPairDataset(base, TwoViewTransform(view_config), seed)
    sampler = EpochShuffleSampler(len(pair_dataset), seed)
    loader = DataLoader(
        pair_dataset,
        batch_size=int(training["batch_size"]),
        sampler=sampler,
        drop_last=True,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    if len(loader) == 0:
        raise ValueError("Physical batch exceeds the U partition")
    input_channels = len(BAND_ADAPTERS[config["data"]["adapter"]])
    model = VICRegModel(
        input_channels,
        int(model_config.get("projector_hidden_dim", 1024)),
        int(model_config.get("projector_output_dim", 256)),
    ).to(device)
    criterion = VICRegLoss(**config["loss"]).to(device)
    learning_rate = float(training["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=float(training["weight_decay"])
    )
    epochs = int(training["epochs"])
    total_steps = epochs * len(loader)
    warmup_steps = int(training["warmup_epochs"]) * len(loader)
    minimum_ratio = float(training["minimum_learning_rate"]) / learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _scheduler_lambda(step, warmup_steps, total_steps, minimum_ratio),
    )
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=float(training.get("amp_initial_scale", 128.0)),
        growth_interval=int(training.get("amp_growth_interval", 2000)),
    )
    start_epoch = 0
    global_step = 0
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["config_sha256"] != object_sha256(config):
            raise ValueError("Resume checkpoint configuration differs from this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint["global_step"])
        amp_overflow_skips = int(checkpoint.get("amp_overflow_skips", 0))
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        timed_step_seconds = list(checkpoint.get("timed_step_seconds", []))
    else:
        amp_overflow_skips = 0
        elapsed_before_resume = 0.0
        timed_step_seconds = []
    log_path = output_dir / "training.jsonl"
    epoch_summaries = []
    started = time.perf_counter()
    last_checkpoint = None
    consecutive_amp_overflows = 0
    checkpoint_every = int(training.get("checkpoint_every_epochs", 1))
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every_epochs must be positive")
    try:
        for epoch in range(start_epoch, epochs):
            sampler.set_epoch(epoch)
            model.train()
            epoch_values = {name: [] for name in ("total", "invariance", "variance", "covariance", "mean_std")}
            epoch_projector_rank = float("nan")
            epoch_encoder_rank = float("nan")
            for batch_index, batch in enumerate(loader):
                step_started = time.perf_counter()
                first = batch["view1"].to(device, non_blocking=True)
                second = batch["view2"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    first_features = model.encoder(first)
                    second_features = model.encoder(second)
                # VICReg's projector and covariance path are sensitive to fp16
                # gradient scaling. Keep this path in fp32 while retaining AMP
                # for the substantially more expensive convolutional encoder.
                with torch.autocast(device_type=device.type, enabled=False):
                    first_projection = model.projector(first_features.float())
                    second_projection = model.projector(second_features.float())
                    terms = criterion(first_projection, second_projection)
                if not torch.isfinite(terms.total):
                    raise FloatingPointError(f"Non-finite VICReg loss at step {global_step}")
                scaler.scale(terms.total).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
                if not torch.isfinite(gradient_norm):
                    if not amp_enabled:
                        raise FloatingPointError(
                            f"Invalid gradient norm at step {global_step}: {gradient_norm}"
                        )
                    previous_scale = float(scaler.get_scale())
                    scaler.step(optimizer)  # skipped internally because unscale_ found inf/nan
                    scaler.update()
                    amp_overflow_skips += 1
                    consecutive_amp_overflows += 1
                    append_jsonl(log_path, {
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "global_step": global_step,
                        "event": "amp_overflow_skip",
                        "gradient_norm_pre_clip": str(gradient_norm),
                        "previous_scale": previous_scale,
                        "new_scale": float(scaler.get_scale()),
                    })
                    if bool(training.get("fail_on_amp_overflow", False)):
                        raise FloatingPointError(
                            f"AMP overflow at step {global_step}; exact-step run aborted"
                        )
                    if consecutive_amp_overflows >= 8:
                        raise FloatingPointError(
                            "Eight consecutive AMP overflows; aborting instead of masking instability"
                        )
                    continue
                if gradient_norm <= 0:
                    raise FloatingPointError(f"Zero gradient norm at step {global_step}")
                scaler.step(optimizer)
                scaler.update()
                consecutive_amp_overflows = 0
                scheduler.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_seconds = time.perf_counter() - step_started
                timed_step_seconds.append(step_seconds)
                values = {
                    "total": float(terms.total.detach()),
                    "invariance": float(terms.invariance.detach()),
                    "variance": float(terms.variance.detach()),
                    "covariance": float(terms.covariance.detach()),
                    "mean_std": float(terms.mean_std.detach()),
                }
                for name, value in values.items():
                    epoch_values[name].append(value)
                if batch_index % int(training["log_every_steps"]) == 0 or batch_index == len(loader) - 1:
                    with torch.no_grad():
                        epoch_projector_rank = float(effective_rank(first_projection.detach()))
                        epoch_encoder_rank = float(effective_rank(first_features.detach()))
                    append_jsonl(log_path, {
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "gradient_norm_pre_clip": float(gradient_norm),
                        "projector_effective_rank": epoch_projector_rank,
                        "encoder_effective_rank": epoch_encoder_rank,
                        "step_seconds": step_seconds,
                        **values,
                    })
                global_step += 1
            epoch_summary = {
                "epoch": epoch,
                **{name: float(np.mean(values)) for name, values in epoch_values.items()},
                "median_mean_std": float(statistics.median(epoch_values["mean_std"])),
                "projector_effective_rank": epoch_projector_rank,
                "encoder_effective_rank": epoch_encoder_rank,
            }
            epoch_summaries.append(epoch_summary)
            should_checkpoint = checkpoint_due(epoch + 1, epochs, checkpoint_every)
            if should_checkpoint:
                checkpoint_payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng_state": capture_rng_state(),
                    "next_epoch": epoch + 1,
                    "global_step": global_step,
                    "amp_overflow_skips": amp_overflow_skips,
                    "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
                    "timed_step_seconds": timed_step_seconds,
                    "config": config,
                    "config_sha256": object_sha256(config),
                    "data": data_record,
                }
                last_checkpoint = output_dir / f"checkpoint-epoch-{epoch + 1:03d}.pt"
                atomic_torch_save(checkpoint_payload, last_checkpoint)
        elapsed = elapsed_before_resume + time.perf_counter() - started
        median_step = statistics.median(timed_step_seconds)
        final = epoch_summaries[-1]
        stability = {
            "finite_losses": True,
            "projector_std_gate": final["median_mean_std"] >= 0.5,
            "projector_rank_gate": final["projector_effective_rank"] >= 32,
        }
        forecast_one = 60 * len(loader) * median_step / 3600
        if last_checkpoint is None or not last_checkpoint.exists():
            raise RuntimeError("Training completed without a final checkpoint")
        checkpoint_sha256 = file_sha256(last_checkpoint)
        encoder_export = output_dir / "encoder-final.pt"
        encoder_payload = {
            "format_version": 1,
            "run_id": run["id"],
            "model_id": run["model_id"],
            "seed": seed,
            "adapter": config["data"]["adapter"],
            "band_order": list(BAND_ADAPTERS[config["data"]["adapter"]]),
            "feature_dimension": 512,
            "encoder_parameters": model.encoder_parameter_count(),
            "encoder": model.encoder.state_dict(),
            "source_checkpoint_sha256": checkpoint_sha256,
            "config_sha256": object_sha256(config),
            "data": data_record,
        }
        atomic_torch_save(encoder_payload, encoder_export)
        verify_encoder_export(encoder_export, checkpoint_sha256)
        encoder_sha256 = file_sha256(encoder_export)
        expected_steps = int(training.get("expected_optimizer_steps", total_steps))
        completion = {
            "exact_optimizer_steps": global_step == expected_steps,
            "zero_amp_overflows": amp_overflow_skips == 0,
            "final_checkpoint_verified": file_sha256(last_checkpoint) == checkpoint_sha256,
            "encoder_export_verified": file_sha256(encoder_export) == encoder_sha256,
        }
        pruned_checkpoints = []
        passed_all_run_gates = all(stability.values()) and all(completion.values())
        if bool(training.get("retain_final_checkpoint_only", False)) and passed_all_run_gates:
            pruned_checkpoints = prune_intermediate_checkpoints(output_dir, last_checkpoint)
        summary = {
            "run_id": run["id"],
            "model_id": run["model_id"],
            "hypothesis": run["hypothesis"],
            "seed": seed,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "steps": global_step,
            "expected_optimizer_steps": expected_steps,
            "config_sha256": object_sha256(config),
            "git_commit": git_commit(),
            "source_tree_sha256": source_tree_sha256(),
            "data": data_record,
            "model": {
                "input_channels": input_channels,
                "adapter": config["data"]["adapter"],
                "band_order": list(BAND_ADAPTERS[config["data"]["adapter"]]),
                "encoder_parameters": model.encoder_parameter_count(),
                "projector_parameters": model.projector_parameter_count(),
            },
            "hardware": hardware_record(device),
            "elapsed_seconds": elapsed,
            "median_step_seconds": median_step,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "amp_initial_scale": float(training.get("amp_initial_scale", 128.0)),
            "amp_overflow_skips": amp_overflow_skips,
            "forecast_hours_per_60_epoch_run": forecast_one,
            "forecast_hours_for_nine_ssl_runs": forecast_one * 9,
            "final_epoch": final,
            "stability": stability,
            "stability_gate": all(stability.values()),
            "completion": completion,
            "completion_gate": all(completion.values()),
            "week4_compute_gate": forecast_one * 9 <= 30,
            "checkpoint": str(last_checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "encoder_export": str(encoder_export),
            "encoder_export_sha256": encoder_sha256,
            "pruned_checkpoints": pruned_checkpoints,
        }
        (output_dir / "ssl_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        append_jsonl(
            ledger_path,
            {**summary, "status": "complete" if passed_all_run_gates else "failed_gates"},
        )
        return summary
    except Exception as error:
        append_jsonl(ledger_path, {
            "run_id": run["id"], "model_id": run["model_id"], "seed": seed,
            "status": "failed", "error": f"{type(error).__name__}: {error}",
            "config_sha256": object_sha256(config), "git_commit": git_commit(),
            "source_tree_sha256": source_tree_sha256(),
        })
        raise
