from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


SEEDS = (17, 29, 43)
MODELS = {
    "M2": {"adapter": "rgb", "dropout": 0.0, "hypothesis": "H2-H5"},
    "M3": {"adapter": "core10", "dropout": 0.0, "hypothesis": "H1-H2-H5"},
    "M4": {"adapter": "core10", "dropout": 0.25, "hypothesis": "H3-H5"},
}


def build_config(model_id: str, seed: int) -> dict[str, object]:
    model = MODELS[model_id]
    run_id = f"week4-{model_id.lower()}-seed{seed}"
    return {
        "run": {
            "id": run_id,
            "model_id": model_id,
            "hypothesis": model["hypothesis"],
            "seed": seed,
            "output_dir": f"outputs/week4/seed{seed}/{run_id}",
            "ledger_path": f"outputs/week4/seed{seed}/runs.jsonl",
        },
        "data": {
            "manifest_path": "manifests/v1/partitions.parquet",
            "staged_root": "data/staged",
            "normalization_path": "manifests/v1/normalization.json",
            "freeze_summary_path": "reports/week2/generated/reports/freeze_summary.json",
            "manifest_contract_sha256": "0b36a0c6c55f34a8963719af725096dde5ab1dbab72968c13639e31d1099000a",
            "normalization_sha256": "3b1d191beccde5b85fdced38c81984377e6fb3171683a6df6a9254c38f2a9a05",
            "adapter": model["adapter"],
            "shard_size": 512,
            "height": 120,
            "width": 120,
        },
        "augmentation": {
            "output_size": 120,
            "minimum_crop_area": 0.8,
            "maximum_crop_area": 1.0,
            "minimum_pair_iou": 0.6,
            "spectral_dropout_probability": model["dropout"],
            "maximum_crop_attempts": 32,
        },
        "model": {"projector_hidden_dim": 1024, "projector_output_dim": 256},
        "loss": {
            "invariance_weight": 25.0,
            "variance_weight": 25.0,
            "covariance_weight": 1.0,
            "variance_target": 1.0,
            "epsilon": 0.0001,
        },
        "training": {
            "epochs": 60,
            "batch_size": 64,
            "num_workers": 2,
            "learning_rate": 0.0001,
            "minimum_learning_rate": 0.000001,
            "warmup_epochs": 5,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "amp": True,
            "amp_initial_scale": 128.0,
            # Keep the Week 3-proven scale fixed during the longer Week 4 jobs.
            # Late scale growth can otherwise cause one skipped optimizer step.
            "amp_growth_interval": 1000000,
            "fail_on_amp_overflow": True,
            "require_cuda": True,
            "log_every_steps": 50,
            "checkpoint_every_epochs": 10,
            "expected_optimizer_steps": 18720,
            "retain_final_checkpoint_only": True,
        },
    }


def write_configs(output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in SEEDS:
        for model_id in MODELS:
            path = output_dir / f"week4_{model_id.lower()}_seed{seed}.yaml"
            path.write_text(yaml.safe_dump(deepcopy(build_config(model_id, seed)), sort_keys=False))
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the immutable Week 4 SSL matrix")
    parser.add_argument("--output-dir", default="configs/ssl")
    args = parser.parse_args()
    for path in write_configs(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
