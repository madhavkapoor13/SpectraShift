from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import yaml

from spectrashift.data.dataset import SpectraShiftDataset


class _ImageDataset:
    def __init__(self, dataset: SpectraShiftDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        import torch

        return torch.from_numpy(self.dataset[index]["image"])


def benchmark_throughput(
    config_path: str | Path,
    batch_size: int | None = None,
    warmup_steps: int | None = None,
    timed_steps: int | None = None,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    from spectrashift.models.resnet import build_resnet18

    config = yaml.safe_load(Path(config_path).read_text())
    dataset_config = config["dataset"]
    staging = config["staging"]
    benchmark = config.get("benchmark", {})
    batch_size = int(batch_size or benchmark.get("physical_batch", 64))
    warmup_steps = int(warmup_steps if warmup_steps is not None else benchmark.get("warmup_steps", 50))
    timed_steps = int(timed_steps if timed_steps is not None else benchmark.get("timed_steps", 200))
    base = SpectraShiftDataset(
        Path(staging["final_manifest_dir"]) / "partitions.parquet",
        staging["output_dir"],
        Path(staging["final_manifest_dir"]) / "normalization.json",
        "U", "core10", int(staging["shard_size"]),
        int(dataset_config["target_height"]), int(dataset_config["target_width"]),
    )

    loader = DataLoader(
        _ImageDataset(base), batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=int(benchmark.get("num_workers", 2)), pin_memory=torch.cuda.is_available(),
    )
    if not len(loader):
        raise ValueError("Physical batch exceeds the U partition")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        device_arch = f"sm_{major}{minor}"
        supported_arches = torch.cuda.get_arch_list()
        if supported_arches and device_arch not in supported_arches:
            raise RuntimeError(
                f"{torch.cuda.get_device_name(0)} ({device_arch}) is unsupported by this "
                f"PyTorch build ({supported_arches}); select a T4 GPU"
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_resnet18(10, outputs=None).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    batches = iter(loader)

    def next_batch():
        nonlocal batches
        try:
            return next(batches)
        except StopIteration:
            batches = iter(loader)
            return next(batches)

    def step() -> None:
        images = next_batch().to(device, non_blocking=True)
        view_one = torch.flip(images, dims=[-1])
        view_two = torch.rot90(images, 1, dims=(-2, -1))
        optimizer.zero_grad(set_to_none=True)
        feature_one = model(view_one)
        feature_two = model(view_two)
        loss = (feature_one - feature_two).square().mean()
        loss.backward()
        optimizer.step()

    for _ in range(warmup_steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(timed_steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    seconds_per_step = elapsed / timed_steps
    steps_per_epoch = math.ceil(len(base) / batch_size)
    result = {
        "benchmark": "provisional two-view ResNet-18 encoder workload",
        "replace_after_week3_vicreg_implementation": True,
        "device": str(device),
        "platform": platform.platform(),
        "physical_batch": batch_size,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": seconds_per_step,
        "paired_examples_per_second": batch_size / seconds_per_step,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "u_patches": len(base),
        "forecast_hours_per_60_epoch_run": 60 * steps_per_epoch * seconds_per_step / 3600,
        "forecast_hours_for_nine_ssl_runs": 9 * 60 * steps_per_epoch * seconds_per_step / 3600,
    }
    report_dir = Path(staging["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "throughput_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
