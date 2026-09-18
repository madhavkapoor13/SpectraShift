from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


torch = pytest.importorskip("torch")

from spectrashift.data.foundation import OlmoEarthS2Dataset
from spectrashift.data.raster import pack_validity
from spectrashift.models.foundation import DINOv2Classifier, OlmoEarthClassifier
from spectrashift.train.common import directory_sha256
from spectrashift.train.foundation import build_foundation_optimizer
from spectrashift.train.week7 import (
    FOUNDATION_MODELS,
    FULL_FRACTIONS,
    OLMO_CONFIG_SHA256,
    OLMO_WEIGHTS_SHA256,
    PILOT_MULTIPLIERS,
    PROBE_FRACTIONS,
    _select_pilot,
    resolved_foundation_config,
)


ROOT = Path(__file__).parents[1]


class FakeDINO(torch.nn.Module):
    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        values = images.mean(dim=(1, 2, 3), keepdim=False)
        tokens = values[:, None, None].expand(-1, 81, 384).contiguous()
        return {"x_norm_patchtokens": tokens}


def test_dinov2_adapter_uses_mean_of_81_patch_tokens() -> None:
    model = DINOv2Classifier(FakeDINO())
    batch = {"image": torch.stack([torch.ones(3, 126, 126), torch.full((3, 126, 126), 2.0)])}
    features = model.features(batch)
    assert features.shape == (2, 384)
    assert torch.equal(features[:, 0], torch.tensor([1.0, 2.0]))


def test_dinov2_adapter_rejects_register_or_wrong_grid_tokens() -> None:
    class Wrong(torch.nn.Module):
        def forward_features(self, images):
            return {"x_norm_patchtokens": torch.zeros(len(images), 82, 384)}

    with pytest.raises(ValueError, match="patch-token shape"):
        DINOv2Classifier(Wrong()).features({"image": torch.zeros(1, 3, 126, 126)})


def test_olmoearth_pool_excludes_missing_tokens() -> None:
    tokens = torch.tensor([[[1.0, 3.0], [100.0, 100.0], [5.0, 7.0]]])
    masks = torch.tensor([[0, 3, 0]])
    pooled = OlmoEarthClassifier.pool_tokens(tokens, masks)
    assert torch.equal(pooled, torch.tensor([[3.0, 5.0]]))
    with pytest.raises(ValueError, match="no valid"):
        OlmoEarthClassifier.pool_tokens(tokens, torch.full_like(masks, 3))


def test_olmoearth_dataset_uses_raw_dn_real_month_and_validity(tmp_path: Path) -> None:
    pixels = np.zeros((1, 12, 2, 2), dtype=np.uint16)
    for band in range(12):
        pixels[0, band] = 1000 + band
    np.save(tmp_path / "shard-00000-pixels.npy", pixels)
    valid = np.ones((12, 2, 2), dtype=bool)
    valid[0, 0, 0] = False
    np.save(tmp_path / "shard-00000-validity.npy", pack_validity(valid)[None])
    manifest = pd.DataFrame([{
        "patch_id": "p0", "partition": "D", "storage_index": 0,
        "timestamp": pd.Timestamp("2019-07-14", tz="UTC"),
        "labels": ["Urban fabric"],
    }])
    manifest_path = tmp_path / "partitions.parquet"
    manifest.to_parquet(manifest_path, index=False)
    contract = {
        "status": "frozen", "raw_units": "sentinel2-l2a-dn",
        "band_order": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"],
        "normalizer_mean": [1000.0] * 12, "normalizer_std": [100.0] * 12,
        "std_multiplier": 2.0,
    }
    contract_path = tmp_path / "olmo.json"
    contract_path.write_text(json.dumps(contract))
    item = OlmoEarthS2Dataset(
        manifest_path, tmp_path, contract_path, "D", height=2, width=2
    )[0]
    assert item["image"].shape == (12, 2, 2)
    assert item["timestamp"].tolist() == [14, 6, 2019]
    assert item["valid"][0, 0, 0] == 0
    assert item["image"][0, 0, 0] == pytest.approx(0.5)
    assert item["image"][1, 0, 0] == pytest.approx((1001 - 800) / 400)


def _contract_files(tmp_path: Path) -> tuple[Path, Path]:
    week5_dir = tmp_path / "week5"
    week5_dir.mkdir()
    week5 = {
        "week5_contracts_complete": True, "evaluation_labels_loaded": False,
        "subset_manifest_sha256": "subset",
    }
    week5_path = week5_dir / "week5_contracts_summary.json"
    week5_path.write_text(json.dumps(week5))
    (week5_dir / "downstream_subsets.parquet").write_bytes(b"")
    (week5_dir / "downstream_contract.json").write_text("{}")
    contract_dir = tmp_path / "week7"
    contract_dir.mkdir()
    source = contract_dir / "source"
    source.mkdir()
    (source / "x.py").write_text("x = 1\n")
    weights = contract_dir / "weights.pth"
    weights.write_bytes(b"weights")
    model_dir = contract_dir / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_bytes(b"config")
    (model_dir / "weights.pth").write_bytes(b"olmo")
    olmo_input = contract_dir / "olmoearth_input_contract.json"
    olmo_input.write_text("{}")
    week7 = {
        "week7_contracts_complete": True, "evaluation_labels_loaded": False,
        "week6_rgb_contract_sha256": "rgb",
        "dinov2": {
            "source_revision": "revision", "source_dir": "source",
            "source_sha256": directory_sha256(source), "weights_file": "weights.pth",
            "weights_sha256": hashlib.sha256(b"weights").hexdigest(),
        },
        "olmoearth": {
            "source_revision": "source-revision", "model_revision": "model-revision",
            "source_dir": "source", "source_sha256": directory_sha256(source),
            "model_dir": "model", "config_sha256": hashlib.sha256(b"config").hexdigest(),
            "weights_sha256": hashlib.sha256(b"olmo").hexdigest(),
            "input_contract_file": olmo_input.name,
            "input_contract_sha256": hashlib.sha256(b"{}").hexdigest(),
        },
    }
    week7_path = contract_dir / "week7_contracts_summary.json"
    week7_path.write_text(json.dumps(week7))
    return week5_path, week7_path


def test_week7_full_and_pilot_matrices_have_exact_schedules(tmp_path: Path) -> None:
    base = yaml.safe_load((ROOT / "configs/downstream/week7.yaml").read_text())
    week5, week7 = _contract_files(tmp_path)
    base["contracts"]["rgb_contract_path"] = str(tmp_path / "rgb.json")
    full = [
        resolved_foundation_config(base, model, seed, fraction, 1.0, 32, tmp_path / "out", week5, week7, False)
        for model in FOUNDATION_MODELS for seed in (17, 29, 43) for fraction in FULL_FRACTIONS
    ]
    pilots = [
        resolved_foundation_config(base, model, 17, "10", multiplier, 32, tmp_path / "pilots", week5, week7, True)
        for model in FOUNDATION_MODELS for multiplier in PILOT_MULTIPLIERS
    ]
    assert len(full) == 18 and len({item["run"]["id"] for item in full}) == 18
    assert len(pilots) == 6 and len({item["run"]["id"] for item in pilots}) == 6
    assert {item["training"]["expected_optimizer_steps"] for item in full} == {300, 760, 7500}
    assert {item["training"]["expected_optimizer_steps"] for item in pilots} == {380}
    assert all("DINOv2" in item["run"]["foundation_name"] for item in full if item["run"]["model_id"] == "M5")
    assert all("DINOv3" not in item["run"]["foundation_name"] for item in full)


def test_week7_batch16_fallback_regenerates_step_contracts(tmp_path: Path) -> None:
    base = yaml.safe_load((ROOT / "configs/downstream/week7.yaml").read_text())
    week5, week7 = _contract_files(tmp_path)
    base["contracts"]["rgb_contract_path"] = str(tmp_path / "rgb.json")
    steps = {
        fraction: resolved_foundation_config(
            base, "M5", 17, fraction, 1.0, 16, tmp_path, week5, week7, False
        )["training"]["expected_optimizer_steps"]
        for fraction in FULL_FRACTIONS
    }
    assert steps == {"01": 304, "10": 1500, "100": 15000}


def test_week7_pilot_tolerance_prefers_multiplier_one_and_blocks_late_gain(tmp_path: Path) -> None:
    summaries = []
    for multiplier, score in ((0.3, 0.500), (1.0, 0.498), (3.0, 0.470)):
        log = tmp_path / f"{multiplier}.jsonl"
        final_score = 0.498 if multiplier == 1.0 else score
        log.write_text(
            json.dumps({"epoch": 8, "validation_macro_average_precision": final_score - 0.006})
            + "\n"
            + json.dumps({"epoch": 10, "validation_macro_average_precision": final_score})
            + "\n"
        )
        summaries.append({
            "model_id": "M5", "learning_rate_multiplier": multiplier,
            "validation_macro_average_precision": score,
            "best_epoch": 10 if multiplier == 1.0 else 9,
            "epochs": 10, "training_log": str(log),
        })
    selected = _select_pilot("M5", summaries, 0.005)
    assert selected["learning_rate_multiplier"] == 1.0
    assert selected["undertraining_block"] is True


def test_foundation_optimizer_separates_head_and_no_decay() -> None:
    model = DINOv2Classifier(FakeDINO())
    optimizer = build_foundation_optimizer(model, 1e-5, 1e-3, 0.05)
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["encoder_decay"]["lr"] == 1e-5
    assert groups["head_decay"]["lr"] == 1e-3
    assert groups["head_no_decay"]["weight_decay"] == 0


def test_week7_registry_counts_and_pinned_olmo_hashes() -> None:
    assert len(FOUNDATION_MODELS) * len(FULL_FRACTIONS) * 3 == 18
    assert len(FOUNDATION_MODELS) * len(PROBE_FRACTIONS) * 3 == 36
    assert len(FOUNDATION_MODELS) * 3 == 6
    assert OLMO_CONFIG_SHA256.startswith("01dcb438")
    assert OLMO_WEIGHTS_SHA256.startswith("2a3fe813")


def test_week7_notebooks_are_small_output_free_and_label_isolated() -> None:
    names = [
        "14_week7_prepare_foundations.ipynb", "15_week7_lr_pilots.ipynb",
        "16a_week7_seed17.ipynb", "16b_week7_seed29.ipynb", "16c_week7_seed43.ipynb",
        "17_week7_frozen_probes.ipynb", "18_week7_aggregate.ipynb",
    ]
    for name in names:
        path = ROOT / "notebooks/kaggle" / name
        notebook = json.loads(path.read_text())
        assert path.stat().st_size < 1_000_000
        assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "partition='I'" not in source and 'partition="I"' not in source
        assert "T-FI" not in source and "T-PT" not in source
