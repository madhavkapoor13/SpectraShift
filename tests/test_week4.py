from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


torch = pytest.importorskip("torch")

from spectrashift.data.views import TwoViewTransform, ViewConfig
from spectrashift.train.common import file_sha256, object_sha256
from spectrashift.train.ssl import (
    checkpoint_due,
    prune_intermediate_checkpoints,
    verify_encoder_export,
)
from spectrashift.train.week4 import (
    EXPECTED_STEPS,
    WEEK4_MODELS,
    WEEK4_SEEDS,
    WEEK3_SUMMARY_SHA256,
    _find_resume_checkpoint,
    aggregate_week4,
    expected_run_id,
    validate_week4_config,
)


ROOT = Path(__file__).parents[1]


def _configs() -> list[tuple[Path, dict[str, object]]]:
    result = []
    for seed in WEEK4_SEEDS:
        for model_id in WEEK4_MODELS:
            path = ROOT / "configs" / "ssl" / f"week4_{model_id.lower()}_seed{seed}.yaml"
            result.append((path, yaml.safe_load(path.read_text())))
    return result


def test_week4_matrix_is_complete_and_immutable() -> None:
    configs = _configs()
    assert len(configs) == 9
    identities = set()
    for path, config in configs:
        assert path.exists()
        validate_week4_config(config)
        identity = (config["run"]["model_id"], config["run"]["seed"])
        identities.add(identity)
        assert config["run"]["id"] == expected_run_id(*identity)
        assert config["training"]["expected_optimizer_steps"] == EXPECTED_STEPS
    assert identities == {(model, seed) for model in WEEK4_MODELS for seed in WEEK4_SEEDS}


def test_week4_spatial_views_match_and_m4_alone_drops_a_group() -> None:
    image = torch.stack([torch.full((120, 120), float(index + 1)) for index in range(10)])
    seed = 91417
    m3 = TwoViewTransform(ViewConfig(spectral_dropout_probability=0.0))(image, seed)
    m4 = TwoViewTransform(ViewConfig(spectral_dropout_probability=1.0))(image, seed)
    rgb = TwoViewTransform(ViewConfig(spectral_dropout_probability=0.0))(image[[2, 1, 0]], seed)
    assert torch.equal(rgb["crop_boxes"], m3["crop_boxes"])
    assert torch.equal(m3["crop_boxes"], m4["crop_boxes"])
    assert m3["dropped_view"] == -1 and rgb["dropped_view"] == -1
    assert m4["dropped_view"] in {0, 1}
    assert m4["dropped_group"] in {"nir", "red_edge", "swir"}


def test_checkpoint_cadence_and_success_pruning(tmp_path: Path) -> None:
    assert [epoch for epoch in range(1, 61) if checkpoint_due(epoch, 60, 10)] == [10, 20, 30, 40, 50, 60]
    paths = []
    for epoch in (10, 20, 30, 40, 50, 60):
        path = tmp_path / f"checkpoint-epoch-{epoch:03d}.pt"
        path.write_bytes(str(epoch).encode())
        paths.append(path)
    removed = prune_intermediate_checkpoints(tmp_path, paths[-1])
    assert removed == [f"checkpoint-epoch-{epoch:03d}.pt" for epoch in (10, 20, 30, 40, 50)]
    assert paths[-1].exists()


def test_resume_discovery_chooses_latest_matching_incomplete_checkpoint(tmp_path: Path) -> None:
    config = _configs()[0][1]
    run_id = config["run"]["id"]
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    for epoch in (10, 20, 60):
        torch.save(
            {"config_sha256": object_sha256(config), "next_epoch": epoch},
            run_dir / f"checkpoint-epoch-{epoch:03d}.pt",
        )
    torch.save(
        {"config_sha256": "wrong", "next_epoch": 50},
        run_dir / "checkpoint-epoch-050.pt",
    )
    selected = _find_resume_checkpoint([tmp_path], run_id, config)
    assert selected is not None and selected.name == "checkpoint-epoch-020.pt"


def test_encoder_export_contract_and_finite_weights(tmp_path: Path) -> None:
    path = tmp_path / "encoder-final.pt"
    payload = {
        "format_version": 1,
        "run_id": "week4-m2-seed17",
        "model_id": "M2",
        "seed": 17,
        "adapter": "rgb",
        "band_order": ["B04", "B03", "B02"],
        "feature_dimension": 512,
        "encoder_parameters": 4,
        "encoder": {"weight": torch.ones(2, 2)},
        "source_checkpoint_sha256": "checkpoint-hash",
        "config_sha256": "config-hash",
        "data": {"u_patches": 20000},
    }
    torch.save(payload, path)
    loaded = verify_encoder_export(path, "checkpoint-hash")
    assert loaded["band_order"] == ["B04", "B03", "B02"]
    payload["encoder"]["weight"][0, 0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="non-finite"):
        verify_encoder_export(path, "checkpoint-hash")


def _fake_seed_summary(root: Path, seed: int) -> Path:
    runs = []
    for index, model_id in enumerate(WEEK4_MODELS):
        run_id = expected_run_id(model_id, seed)
        run_dir = root / f"seed{seed}" / run_id
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "checkpoint-epoch-060.pt"
        checkpoint.write_bytes(f"checkpoint:{run_id}".encode())
        checkpoint_hash = file_sha256(checkpoint)
        encoder = run_dir / "encoder-final.pt"
        torch.save(
            {
                "format_version": 1,
                "run_id": run_id,
                "model_id": model_id,
                "seed": seed,
                "adapter": WEEK4_MODELS[model_id]["adapter"],
                "band_order": [model_id],
                "feature_dimension": 512,
                "encoder_parameters": 1,
                "encoder": {"weight": torch.tensor([seed * 10 + index], dtype=torch.float32)},
                "source_checkpoint_sha256": checkpoint_hash,
                "config_sha256": run_id,
                "data": {"u_patches": 20000},
            },
            encoder,
        )
        runs.append(
            {
                "run_id": run_id,
                "model_id": model_id,
                "seed": seed,
                "steps": EXPECTED_STEPS,
                "amp_overflow_skips": 0,
                "elapsed_seconds": 1.0,
                "source_tree_sha256": "source-tree",
                "hardware": {"gpu": "Tesla T4"},
                "stability_gate": True,
                "completion_gate": True,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "encoder_export": str(encoder),
                "encoder_export_sha256": file_sha256(encoder),
                "data": {
                    "manifest_contract_sha256": "manifest",
                    "normalization_sha256": "normalization",
                    "u_patches": 20000,
                },
            }
        )
    path = root / f"seed{seed}" / f"week4_seed{seed}_summary.json"
    path.write_text(json.dumps({
        "week4_seed_complete": True,
        "seed": seed,
        "runs": runs,
        "week3_summary_sha256": WEEK3_SUMMARY_SHA256,
        "evaluation_labels_loaded": False,
    }))
    return path


def test_week4_aggregation_verifies_nine_artifacts_and_rejects_missing_seed(tmp_path: Path) -> None:
    paths = [_fake_seed_summary(tmp_path, seed) for seed in WEEK4_SEEDS]
    result = aggregate_week4(paths, tmp_path / "week4_run_summary.json")
    assert result["week4_complete"] and result["week5_approved"]
    assert result["run_count"] == 9 and len(result["encoder_sha256"]) == 9
    with pytest.raises(ValueError, match="exactly three"):
        aggregate_week4(paths[:2], tmp_path / "invalid.json")


def test_week4_notebooks_are_small_output_free_and_label_isolated() -> None:
    names = [
        "05a_week4_seed17.ipynb",
        "05b_week4_seed29.ipynb",
        "05c_week4_seed43.ipynb",
        "06_week4_aggregate.ipynb",
    ]
    for name in names:
        path = ROOT / "notebooks" / "kaggle" / name
        notebook = json.loads(path.read_text())
        assert path.stat().st_size < 1_000_000
        assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "partition='V'" not in source
        assert "partition='I'" not in source
        assert "T-FI" not in source and "T-PT" not in source
