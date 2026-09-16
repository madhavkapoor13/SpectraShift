from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from spectrashift.data.views import TwoViewTransform, ViewConfig, crop_iou, deterministic_view_seed
from spectrashift.losses.vicreg import VICRegLoss, effective_rank
from spectrashift.models.vicreg import VICRegModel
from spectrashift.train.common import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    source_tree_sha256,
)
from spectrashift.train.pilots import select_week4_pilot
from spectrashift.train.ssl import EpochShuffleSampler


def test_vicreg_constant_features_have_declared_variance_penalty() -> None:
    values = torch.zeros(4, 3, requires_grad=True)
    terms = VICRegLoss()(values, values.clone())
    assert torch.isclose(terms.invariance, torch.tensor(0.0))
    assert torch.isclose(terms.variance, torch.tensor(1.98), atol=1e-6)
    assert torch.isclose(terms.covariance, torch.tensor(0.0))
    assert torch.isclose(terms.total, torch.tensor(49.5), atol=1e-5)


def test_vicreg_covariance_has_gradient_and_effective_rank() -> None:
    base = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    values = torch.cat([base, base * 2, torch.flip(base, dims=[0])], dim=1).requires_grad_()
    terms = VICRegLoss()(values, values + 0.1)
    terms.total.backward()
    assert terms.covariance > 0
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert effective_rank(values.detach()) > 0


def test_vicreg_model_shapes_and_parameter_contract() -> None:
    model = VICRegModel(10)
    model.eval()
    with torch.inference_mode():
        features, projection = model(torch.randn(2, 10, 120, 120))
    assert features.shape == (2, 512)
    assert projection.shape == (2, 256)
    assert 11_000_000 < model.encoder_parameter_count() < 11_500_000
    assert 1_500_000 < model.projector_parameter_count() < 2_000_000


def test_two_view_transform_is_deterministic_and_respects_contract() -> None:
    image = torch.stack([torch.full((120, 120), float(index + 1)) for index in range(10)])
    transform = TwoViewTransform(ViewConfig(spectral_dropout_probability=1.0))
    first = transform(image, 1234)
    second = transform(image, 1234)
    assert torch.equal(first["view1"], second["view1"])
    assert torch.equal(first["view2"], second["view2"])
    assert torch.equal(first["crop_boxes"], second["crop_boxes"])
    boxes = [tuple(int(value) for value in row) for row in first["crop_boxes"]]
    assert crop_iou(boxes[0], boxes[1]) >= 0.6
    assert all(height * width / (120 * 120) >= 0.8 for _, _, height, width in boxes)
    assert first["dropped_view"] in {0, 1}
    assert first["dropped_group"] in {"nir", "red_edge", "swir"}
    dropped = first[f"view{first['dropped_view'] + 1}"]
    assert torch.all(dropped[:3] != 0)


def test_view_seed_and_crop_geometry_are_adapter_independent() -> None:
    seed = deterministic_view_seed(17, 4, "patch-001")
    transform = TwoViewTransform(ViewConfig())
    rgb = transform(torch.ones(3, 120, 120), seed)
    multispectral = transform(torch.ones(10, 120, 120), seed)
    assert torch.equal(rgb["crop_boxes"], multispectral["crop_boxes"])


def test_epoch_sampler_is_reproducible_and_epoch_specific() -> None:
    first = EpochShuffleSampler(20, 17)
    second = EpochShuffleSampler(20, 17)
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)


def test_rng_state_and_atomic_checkpoint_round_trip(tmp_path: Path) -> None:
    seed_everything(17)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(state)
    observed = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[0] == observed[0]
    assert expected[1] == observed[1]
    assert torch.equal(expected[2], observed[2])
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"state": state}, path)
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()


def test_source_tree_hash_tracks_paths_and_content(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    source = tmp_path / "src" / "module.py"
    source.write_text("value = 1\n")
    first = source_tree_sha256(tmp_path)
    source.write_text("value = 2\n")
    assert source_tree_sha256(tmp_path) != first


def test_pilot_selection_uses_declared_tie_break_and_blocking_rule() -> None:
    ssl = [
        {"run_id": "low", "learning_rate": 1e-4, "stability_gate": True, "week4_compute_gate": True, "forecast_hours_for_nine_ssl_runs": 12},
        {"run_id": "preferred", "learning_rate": 3e-4, "stability_gate": True, "week4_compute_gate": True, "forecast_hours_for_nine_ssl_runs": 11},
    ]
    probes = [
        {"run_id": "low", "validation_macro_average_precision": 0.501},
        {"run_id": "preferred", "validation_macro_average_precision": 0.498},
    ]
    selected = select_week4_pilot(ssl, probes)
    assert selected["week4_approved"]
    assert selected["selected"]["run_id"] == "preferred"
    blocked = select_week4_pilot([{**ssl[0], "stability_gate": False}], probes)
    assert not blocked["week4_approved"]
    assert blocked["required_diagnostic_learning_rate"] == 3e-5
