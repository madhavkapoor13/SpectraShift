from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from spectrashift.data.raster import pack_validity
from spectrashift.data.rgb import (
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    SpectraShiftImageNetRGBDataset,
    histogram_quantile,
)
from spectrashift.models.resnet import build_imagenet_resnet18_rgb
from spectrashift.train.week6 import (
    LOCKED_MULTIPLIERS,
    RGB_CONTROL_MODEL,
    WEEK6_FRACTIONS,
    aggregate_week6,
    normalized_log_aulc,
    resolved_week6_config,
)


ROOT = Path(__file__).parents[1]


def test_histogram_quantile_matches_numpy_linear_quantile() -> None:
    values = np.array([0, 1, 1, 4, 7, 7, 7, 10], dtype=np.uint16)
    histogram = np.bincount(values, minlength=65_536)
    for quantile in (0.0, 0.02, 0.25, 0.5, 0.98, 1.0):
        assert histogram_quantile(histogram, quantile) == pytest.approx(
            np.quantile(values, quantile, method="linear")
        )


def test_imagenet_rgb_dataset_applies_frozen_order_and_preprocessing(tmp_path: Path) -> None:
    height = width = 2
    pixels = np.zeros((1, 12, height, width), dtype=np.uint16)
    pixels[0, 2] = [[20, 30], [40, 50]]  # B04 / R
    pixels[0, 1] = [[10, 20], [30, 40]]  # B03 / G
    pixels[0, 0] = [[0, 10], [20, 30]]   # B02 / B
    np.save(tmp_path / "shard-00000-pixels.npy", pixels)
    valid = np.ones((12, height, width), dtype=bool)
    valid[2, 0, 0] = False
    packed = pack_validity(valid)
    np.save(tmp_path / "shard-00000-validity.npy", packed[None])
    manifest = pd.DataFrame([{
        "patch_id": "p0", "partition": "D", "storage_index": 0, "labels": ["Urban fabric"]
    }])
    manifest_path = tmp_path / "partitions.parquet"
    manifest.to_parquet(manifest_path, index=False)
    normalization = {
        "effective_scale": [0.1] * 12,
        "effective_offset": [0.0] * 12,
        "mean": [1.0, 2.0, 3.0] + [0.0] * 9,
        "std": [1.0] * 12,
        "sha256": "normalization",
    }
    normalization_path = tmp_path / "normalization.json"
    normalization_path.write_text(json.dumps(normalization))
    contract = {
        "status": "frozen", "fit_partition": "U",
        "band_order": ["B04", "B03", "B02"],
        "normalization_sha256": "normalization",
        "invalid_fill_u_mean": [3.0, 2.0, 1.0],
        "reflectance_percentile_02": [1.0, 1.0, 0.0],
        "reflectance_percentile_98": [5.0, 4.0, 3.0],
        "imagenet_mean": list(IMAGENET_RGB_MEAN),
        "imagenet_std": list(IMAGENET_RGB_STD),
    }
    contract_path = tmp_path / "rgb.json"
    contract_path.write_text(json.dumps(contract))
    dataset = SpectraShiftImageNetRGBDataset(
        manifest_path, tmp_path, normalization_path, contract_path, "D",
        shard_size=512, height=height, width=width,
    )
    image = dataset[0]["image"]
    expected_unit = np.array([
        [[0.5, 0.5], [0.75, 1.0]],
        [[0.0, 1 / 3], [2 / 3, 1.0]],
        [[0.0, 1 / 3], [2 / 3, 1.0]],
    ], dtype=np.float32)
    expected = (
        expected_unit - np.asarray(IMAGENET_RGB_MEAN, dtype=np.float32)[:, None, None]
    ) / np.asarray(IMAGENET_RGB_STD, dtype=np.float32)[:, None, None]
    assert np.allclose(image, expected)


def test_m1rgb_keeps_original_imagenet_stem(tmp_path: Path) -> None:
    source = torchvision.models.resnet18(weights=None)
    original = source.conv1.weight.detach().clone()
    checkpoint = tmp_path / "resnet18.pth"
    torch.save(source.state_dict(), checkpoint)
    model = build_imagenet_resnet18_rgb(checkpoint)
    assert model.conv1.in_channels == 3
    assert torch.equal(model.conv1.weight, original)
    assert model.fc.out_features == 19


def _contracts(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    week5 = {
        "week5_contracts_complete": True,
        "evaluation_labels_loaded": False,
        "subset_manifest_sha256": "subset",
        "imagenet_weights_file": "resnet18-f37072fd.pth",
        "imagenet_weights_sha256": "f37072fd" + "0" * 56,
        "week4_encoder_sha256": {
            f"week4-{model.lower()}-seed{seed}": f"{model}-{seed}"
            for model in ("M2", "M3", "M4") for seed in (17, 29, 43)
        },
    }
    week5_path = tmp_path / "week5_contracts_summary.json"
    week5_path.write_text(json.dumps(week5))
    rgb_path = tmp_path / "rgb_percentile_contract.json"
    rgb_path.write_text("{}")
    week6 = {
        "week6_contracts_complete": True,
        "rgb_contract_file": rgb_path.name,
        "rgb_contract_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
    }
    week6_path = tmp_path / "week6_contracts_summary.json"
    week6_path.write_text(json.dumps(week6))
    encoders = {key: tmp_path / f"{key}.pt" for key in week5["week4_encoder_sha256"]}
    return week5_path, week6_path, encoders


def test_week6_matrix_has_48_unique_runs_and_exact_schedules(tmp_path: Path) -> None:
    base = yaml.safe_load((ROOT / "configs/downstream/week6.yaml").read_text())
    week5, week6, encoders = _contracts(tmp_path)
    configs = [
        resolved_week6_config(base, model, seed, fraction, tmp_path / "out", week5, week6, encoders)
        for seed in (17, 29, 43)
        for model in ("M0", "M1", "M2", "M3", "M4")
        for fraction in WEEK6_FRACTIONS
    ] + [
        resolved_week6_config(base, RGB_CONTROL_MODEL, seed, "10", tmp_path / "out", week5, week6, encoders)
        for seed in (17, 29, 43)
    ]
    assert len(configs) == 48
    assert len({config["run"]["id"] for config in configs}) == 48
    assert {config["run"]["learning_rate_multiplier"] for config in configs if config["run"]["model_id"] == "M0"} == {1.0}
    assert {config["run"]["learning_rate_multiplier"] for config in configs if config["run"]["model_id"] != "M0"} == {3.0}
    expected = {"05": 300, "10": 570, "25": 1410, "50": 2820}
    assert all(config["training"]["expected_optimizer_steps"] == expected[config["run"]["fraction_code"]] for config in configs)
    rgb = [config for config in configs if config["run"]["model_id"] == RGB_CONTROL_MODEL]
    assert all(config["data"]["adapter"] == "rgb" for config in rgb)
    assert all(config["data"]["preprocessing"]["mode"] == "imagenet_rgb_percentile" for config in rgb)
    assert all(config["subsets"]["manifest_path"] == configs[0]["subsets"]["manifest_path"] for config in configs)


def test_normalized_log_aulc_on_constant_and_linear_values() -> None:
    counts = [120, 600, 1200, 3000, 6000, 12000]
    assert normalized_log_aulc(counts, [0.4] * 6) == pytest.approx(0.4)
    x = np.log10(np.asarray(counts))
    scaled = ((x - x[0]) / (x[-1] - x[0])).tolist()
    assert normalized_log_aulc(counts, scaled) == pytest.approx(0.5)


def test_week6_aggregation_rejects_missing_seed() -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        aggregate_week6("missing-week5.json", "missing-contracts.json", [], "output")


def test_week6_notebooks_are_small_output_free_and_label_isolated() -> None:
    names = [
        "11_week6_prepare_rgb.ipynb", "12a_week6_seed17.ipynb",
        "12b_week6_seed29.ipynb", "12c_week6_seed43.ipynb",
        "13_week6_aggregate.ipynb",
    ]
    for name in names:
        path = ROOT / "notebooks/kaggle" / name
        notebook = json.loads(path.read_text())
        assert path.stat().st_size < 1_000_000
        assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "partition='I'" not in source and 'partition="I"' not in source
        assert "T-FI" not in source and "T-PT" not in source
