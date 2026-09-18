from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from spectrashift.train.week9 import (
    PROBE_FRACTIONS,
    PROBE_MODELS,
    WEEK9_SEEDS,
    _probe_tables,
    apply_spectral_condition,
    centered_linear_cka,
    covariance_effective_rank,
    deterministic_patch_ids,
    _forbidden_locations,
    _geographic_distance_km,
)


ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_week9_notebooks", ROOT / "scripts/build_week9_notebooks.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_NOTEBOOKS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NOTEBOOKS)


def test_deterministic_ids_ignore_input_order_and_do_not_use_labels() -> None:
    values = [f"patch-{index:04d}" for index in range(2000)]
    first = deterministic_patch_ids(values, 1000, 1729, "V")
    second = deterministic_patch_ids(list(reversed(values)), 1000, 1729, "V")
    assert first == second and len(first) == len(set(first)) == 1000
    with pytest.raises(ValueError, match="unique"):
        deterministic_patch_ids(["a", "a"], 1, 1729, "V")


def test_spectral_groups_are_exact_source_mean_replacements() -> None:
    image = np.arange(10 * 4, dtype=np.float32).reshape(10, 2, 2)
    valid = np.ones_like(image, dtype=bool)
    mean, std = np.arange(1, 11, dtype=np.float32), np.ones(10, dtype=np.float32)
    b08 = apply_spectral_condition(image, valid, mean, std, "missing_b08")
    red_edge = apply_spectral_condition(image, valid, mean, std, "missing_red_edge")
    swir = apply_spectral_condition(image, valid, mean, std, "missing_swir")
    assert np.all(b08[3] == 0) and np.array_equal(b08[:3], image[:3])
    assert np.all(red_edge[4:8] == 0) and np.array_equal(red_edge[8:], image[8:])
    assert np.all(swir[8:10] == 0) and np.array_equal(swir[:8], image[:8])


def test_reflectance_gain_happens_before_standardization_and_preserves_invalid_mean() -> None:
    image = np.ones((10, 2, 2), dtype=np.float32)
    valid = np.ones_like(image, dtype=bool)
    valid[:, 0, 0] = False
    mean = np.arange(1, 11, dtype=np.float32)
    std = np.arange(2, 12, dtype=np.float32)
    result = apply_spectral_condition(image, valid, mean, std, "gain_0p9")
    expected = 0.9 * image + (0.9 - 1.0) * (mean / std)[:, None, None]
    assert np.allclose(result[:, 1:, 1:], expected[:, 1:, 1:])
    assert np.all(result[:, 0, 0] == 0)


def test_centered_linear_cka_is_scale_and_orthogonal_invariant() -> None:
    rng = np.random.default_rng(1729)
    features = rng.normal(size=(100, 12))
    orthogonal, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    assert centered_linear_cka(features, features * 7) == pytest.approx(1.0)
    assert centered_linear_cka(features, features @ orthogonal) == pytest.approx(1.0)


def test_effective_rank_is_hand_checkable_and_detects_collapse() -> None:
    identity = np.eye(4)
    assert covariance_effective_rank(identity) == pytest.approx(3.0)
    collapsed = np.tile(np.arange(10, dtype=np.float64)[:, None], (1, 5))
    assert covariance_effective_rank(collapsed) == pytest.approx(1.0)
    assert covariance_effective_rank(np.ones((10, 5))) == 0.0


def test_duplicate_location_filter_returns_bank_positions_with_patch_id_index() -> None:
    bank = pd.DataFrame(
        {"location_key": ["tile-a", "tile-b", "tile-a"]},
        index=["S2A_patch_74_09", "S2B_patch_10_11", "S2C_patch_12_13"],
    )
    query = pd.DataFrame(
        {"location_key": ["tile-a", "tile-c"]},
        index=["query-one", "query-two"],
    )
    forbidden = _forbidden_locations(query, bank)
    assert np.array_equal(forbidden[0], np.asarray([0, 2]))
    assert forbidden[1].dtype == np.int64 and len(forbidden[1]) == 0


def test_geographic_distance_needs_no_optional_mgrs_dependency() -> None:
    class Row:
        def __init__(self, tile: str, h: int, v: int) -> None:
            self.mgrs_tile, self.h_order, self.v_order = tile, h, v

    assert _geographic_distance_km(Row("T29UPU", 10, 20), Row("T29UPU", 13, 24)) == pytest.approx(6.0)
    assert np.isnan(_geographic_distance_km(Row("T29UPU", 10, 20), Row("T30TYN", 13, 24)))


def _synthetic_probe_summary() -> dict[str, object]:
    linear, knn = [], []
    for model_index, model in enumerate(PROBE_MODELS):
        for seed_index, seed in enumerate(WEEK9_SEEDS):
            for fraction_index, fraction in enumerate(PROBE_FRACTIONS):
                linear.append({
                    "run_id": f"{model}-{fraction}-{seed}", "model_id": model,
                    "probe_type": "linear", "fraction_code": fraction, "seed": seed,
                    "sample_count": [120, 600, 1200, 3000, 6000, 12000][fraction_index],
                    "validation_macro_average_precision": .1 + model_index * .01 + fraction_index * .02,
                    "selected_l2": 1e-4, "feature_cache_sha256": f"cache-{model}-{seed}",
                })
            knn.append({
                "run_id": f"{model}-knn-{seed}", "model_id": model, "probe_type": "knn",
                "fraction_code": "10", "seed": seed, "sample_count": 1200,
                "validation_macro_average_precision": .2 + model_index * .01,
                "selected_k": 20, "feature_cache_sha256": f"cache-{model}-{seed}",
            })
    return {"linear_runs": linear, "knn_runs": knn}


def test_probe_tables_cover_exact_108_linear_and_18_knn_states() -> None:
    table, differences, aulc = _probe_tables(_synthetic_probe_summary())
    assert len(table[table.probe_type == "linear"]) == 108
    assert len(table[table.probe_type == "knn"]) == 18
    assert len(differences) == 3 * 6 + 3
    assert len(aulc) == 21


def test_week9_config_freezes_counts_conditions_and_no_optional_embedding() -> None:
    config = yaml.safe_load((ROOT / "configs/analysis/week9.yaml").read_text())
    assert config["sampling"]["cka_patch_count"] == 1000
    assert sum(config["sampling"]["query_counts"].values()) == 100
    assert config["diagnostics"]["conditions"] == [
        "clean", "missing_b08", "missing_red_edge", "missing_swir", "gain_0p9", "gain_1p1"
    ]
    payload = json.dumps(config).lower()
    assert "umap" not in payload and "t-sne" not in payload and "tsne" not in payload


def test_week9_notebooks_are_output_free_valid_and_below_kaggle_limit() -> None:
    notebooks = [
        _NOTEBOOKS.prepare(), _NOTEBOOKS.probes(), _NOTEBOOKS.diagnostics(17),
        _NOTEBOOKS.diagnostics(29), _NOTEBOOKS.diagnostics(43), _NOTEBOOKS.aggregate(),
    ]
    for notebook in notebooks:
        assert notebook["nbformat"] == 4
        assert len(json.dumps(notebook).encode()) < 1_000_000
        assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_week9_tracked_paths_do_not_include_sealed_patch_artifacts() -> None:
    ignored = (ROOT / ".gitignore").read_text()
    assert "data/sealed/" in ignored and "outputs/" in ignored
