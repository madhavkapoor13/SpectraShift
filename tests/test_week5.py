from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from spectrashift.data.downstream import (
    FRACTION_COUNTS,
    UNSUPPORTED_SOURCE_CLASSES,
    build_subset_records,
)
from spectrashift.data.labels import CANONICAL_LABELS
from spectrashift.data.supervised import supervised_spatial_transform
from spectrashift.eval.metrics import fit_global_validation_threshold, multilabel_metrics
from spectrashift.models.resnet import build_imagenet_resnet18_core10
from spectrashift.train.downstream import build_optimizer, expected_epochs_and_steps
from spectrashift.train.week5 import (
    FINAL_FRACTIONS,
    PILOT_MULTIPLIERS,
    WEEK5_MODELS,
    aggregate_week5,
    resolved_run_config,
)


ROOT = Path(__file__).parents[1]


def _d_frame() -> pd.DataFrame:
    supported = [label for label in CANONICAL_LABELS if label not in UNSUPPORTED_SOURCE_CLASSES]
    rows = []
    for index in range(12_000):
        labels = [label for offset, label in enumerate(supported) if (index + offset * 7) % (offset + 3) == 0]
        if index < 12:
            labels.append("Beaches, dunes, sands")
        if index == 0:
            labels.append("Coastal wetlands")
        rows.append({
            "patch_id": f"patch-{index:05d}",
            "partition": "D",
            "labels": labels,
            "mgrs_tile": f"T{index % 12:02d}",
        })
    return pd.DataFrame(rows)


def test_nested_subsets_are_exact_reproducible_and_shared() -> None:
    records, summary = build_subset_records(_d_frame())
    assert summary["supported_class_names"] == [
        label for label in CANONICAL_LABELS if label not in UNSUPPORTED_SOURCE_CLASSES
    ]
    assert len(summary["supported_class_indices"]) == 16
    assert summary["evaluation_labels_loaded"] is False
    assert len(records) == 36_000
    for seed in (17, 29, 43):
        seeded = records[records["downstream_seed"].eq(seed)].sort_values("subset_rank")
        assert len(seeded) == 12_000
        assert seeded["patch_id"].nunique() == 12_000
        previous: set[str] = set()
        for count in FRACTION_COUNTS.values():
            current = set(seeded.head(count)["patch_id"])
            assert len(current) == count and previous <= current
            previous = current
    repeated, repeated_summary = build_subset_records(_d_frame())
    pd.testing.assert_frame_equal(records, repeated)
    assert summary["subset_manifest_sha256"] == repeated_summary["subset_manifest_sha256"]


def test_exact_anchor_step_schedules() -> None:
    assert expected_epochs_and_steps(120, 64, 30, 300) == (150, 2, 300)
    assert expected_epochs_and_steps(1_200, 64, 30, 300) == (30, 19, 570)
    assert expected_epochs_and_steps(12_000, 64, 30, 300) == (30, 188, 5_640)
    assert expected_epochs_and_steps(1_200, 64, 15, 0) == (15, 19, 285)


def test_global_threshold_ties_prefer_half_then_lower() -> None:
    targets = np.array([[1], [0]])
    scores = np.array([[0.9], [0.1]])
    result = fit_global_validation_threshold(targets, scores, np.array([0.4, 0.5, 0.6]))
    assert result["threshold"] == 0.5
    lower = fit_global_validation_threshold(targets, scores, np.array([0.4, 0.6]))
    assert lower["threshold"] == 0.4


def test_metrics_report_supported_map_brier_and_macro_ece() -> None:
    targets = np.array([[1, 0], [0, 1], [1, 1]])
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]])
    result = multilabel_metrics(targets, scores, supported_indices=[0])
    assert result["macro_average_precision"] == 1.0
    assert result["all_class_macro_average_precision"] == 1.0
    assert 0 <= result["binary_brier_score"] <= 1
    assert len(result["per_class_expected_calibration_error"]) == 2


def test_imagenet_ten_band_stem_contract(tmp_path: Path) -> None:
    source = torchvision.models.resnet18(weights=None)
    original = source.conv1.weight.detach().clone()
    checkpoint = tmp_path / "resnet18.pth"
    torch.save(source.state_dict(), checkpoint)
    model = build_imagenet_resnet18_core10(checkpoint)
    expected_mean = original.mean(dim=1)
    assert torch.allclose(model.conv1.weight[:, 2], original[:, 0] * 0.3)
    assert torch.allclose(model.conv1.weight[:, 1], original[:, 1] * 0.3)
    assert torch.allclose(model.conv1.weight[:, 0], original[:, 2] * 0.3)
    assert torch.allclose(model.conv1.weight[:, 3], expected_mean * 0.3)
    assert model.fc.out_features == 19


def test_optimizer_excludes_bias_and_norm_from_decay() -> None:
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 19)
    optimizer = build_optimizer(model, 1e-4, 1e-3, 1e-4)
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["encoder_decay"]["weight_decay"] == 1e-4
    assert groups["encoder_no_decay"]["weight_decay"] == 0
    assert groups["head_decay"]["lr"] == 1e-3
    assert groups["head_no_decay"]["weight_decay"] == 0


def test_supervised_transform_is_deterministic_and_preserves_field() -> None:
    image = torch.arange(10 * 120 * 120).reshape(10, 120, 120)
    first = supervised_spatial_transform(image, 1234)
    second = supervised_spatial_transform(image, 1234)
    assert torch.equal(first, second)
    assert first.shape == image.shape
    assert torch.equal(torch.sort(first.flatten()).values, torch.sort(image.flatten()).values)


def test_resolved_matrix_has_fresh_unique_run_contracts(tmp_path: Path) -> None:
    base = yaml.safe_load((ROOT / "configs/downstream/week5.yaml").read_text())
    contracts = {
        "subset_manifest_sha256": "subset",
        "imagenet_weights_file": "resnet18-f37072fd.pth",
        "imagenet_weights_sha256": "f37072fd" + "0" * 56,
        "week4_encoder_sha256": {
            f"week4-{model.lower()}-seed{seed}": f"{model}{seed}"
            for model in ("M2", "M3", "M4") for seed in (17, 29, 43)
        },
    }
    contract_path = tmp_path / "week5_contracts_summary.json"
    contract_path.write_text(json.dumps(contracts))
    encoders = {
        key: tmp_path / f"{key}.pt" for key in contracts["week4_encoder_sha256"]
    }
    configs = [
        resolved_run_config(base, model, seed, fraction, 1.0, tmp_path / "out", contract_path, encoders, False)
        for model in WEEK5_MODELS for seed in (17, 29, 43) for fraction in FINAL_FRACTIONS
    ]
    assert len(configs) == 45
    assert len({config["run"]["id"] for config in configs}) == 45
    assert {config["training"]["expected_optimizer_steps"] for config in configs} == {300, 570, 5640}
    pilots = [
        resolved_run_config(base, model, 17, "10", multiplier, tmp_path / "pilots", contract_path, encoders, True)
        for model in WEEK5_MODELS for multiplier in PILOT_MULTIPLIERS
    ]
    assert len(pilots) == 15 and len({config["run"]["id"] for config in pilots}) == 15


def test_week5_aggregation_rejects_missing_seed() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        aggregate_week5([], "missing.json", "output.json")


def test_week5_notebooks_are_small_output_free_and_label_isolated() -> None:
    names = [
        "07_week5_prepare.ipynb", "08_week5_lr_pilots.ipynb",
        "09a_week5_seed17.ipynb", "09b_week5_seed29.ipynb",
        "09c_week5_seed43.ipynb", "10_week5_aggregate.ipynb",
    ]
    for name in names:
        path = ROOT / "notebooks/kaggle" / name
        notebook = json.loads(path.read_text())
        assert path.stat().st_size < 1_000_000
        assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "partition='I'" not in source and "partition=\"I\"" not in source
        assert "T-FI" not in source and "T-PT" not in source
