from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from spectrashift.eval.metrics import average_precision
from spectrashift.train.common import file_sha256
ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location("build_week8_notebooks", ROOT / "scripts/build_week8_notebooks.py")
assert _SPEC is not None and _SPEC.loader is not None
_NOTEBOOKS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NOTEBOOKS)
prepare, seed_notebook, aggregate = _NOTEBOOKS.prepare, _NOTEBOOKS.seed_notebook, _NOTEBOOKS.aggregate


from spectrashift.train.week8 import (
    EXPECTED_LEDGER_ROWS,
    _bootstrap_draws,
    _save_prediction,
    equal_country_ood,
    freeze_week8_evaluation,
    normalized_log_aulc,
    validate_week7_approval,
)


def _ledger() -> pd.DataFrame:
    rows = []
    for seed in (17, 29, 43):
        for index in range(37):
            model = "M5" if index >= 31 else ("M1RGB" if index == 30 else f"M{index // 6}")
            digest = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()
            rows.append({
                "family": "foundation" if model in {"M5", "M6"} else "controlled",
                "run_id": f"run-{seed}-{index}", "model_id": model,
                "fraction_code": "10", "seed": seed, "checkpoint_sha256": digest,
                "validation_macro_average_precision": 0.5,
            })
    return pd.DataFrame(rows)


def _approval(tmp_path: Path) -> tuple[Path, Path]:
    ledger = _ledger()
    ledger_path = tmp_path / "week8_checkpoint_ledger.csv"
    ledger.to_csv(ledger_path, index=False)
    embedded = ledger.to_dict("records")
    summary = {
        "week7_complete": True, "week8_approved": True,
        "week8_checkpoint_ledger_count": EXPECTED_LEDGER_ROWS,
        "week8_checkpoint_ledger": embedded, "evaluation_labels_loaded": False,
    }
    summary_path = tmp_path / "week7_run_summary.json"
    summary_path.write_text(json.dumps(summary))
    return summary_path, ledger_path


def test_week7_approval_requires_exact_111_row_ledger_and_hashes(tmp_path: Path) -> None:
    summary, ledger = _approval(tmp_path)
    payload, frame = validate_week7_approval(
        summary, ledger, file_sha256(summary), file_sha256(ledger)
    )
    assert payload["week8_approved"] and len(frame) == 111
    altered = pd.read_csv(ledger).iloc[:-1]
    altered.to_csv(ledger, index=False)
    with pytest.raises(ValueError, match="hash|111"):
        validate_week7_approval(summary, ledger, file_sha256(summary), "0" * 64)


def test_freeze_week8_seals_exact_final_ids_and_excludes_candidate_tail(tmp_path: Path) -> None:
    summary_path, ledger_path = _approval(tmp_path)
    partitions = [("I", 3000, "Austria"), ("T-FI", 4000, "Finland"), ("T-PT", 4000, "Portugal")]
    rows, candidate_rows = [], []
    labels = ["Arable land", "Urban fabric"]
    for partition, count, country in partitions:
        for index in range(count):
            patch_id = f"{partition}-{index:05d}"
            rows.append({
                "partition": partition, "patch_id": patch_id, "country": country,
                "mgrs_tile": f"T{index % 7:02d}", "timestamp": pd.Timestamp("2019-01-01", tz="UTC") + pd.Timedelta(days=index % 365),
                "h_order": index % 100, "v_order": (index * 3) % 100, "labels": None,
            })
            candidate_rows.append({"partition": partition, "patch_id": patch_id, "labels": [labels[index % 2]]})
    for index in range(1500):
        candidate_rows.append({"partition": "I", "patch_id": f"unused-{index}", "labels": ["Arable land"]})
    manifest_path = tmp_path / "partitions.parquet"
    candidate_path = tmp_path / "evaluation_candidate_labels.parquet"
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)
    pd.DataFrame(candidate_rows).to_parquet(candidate_path, index=False)
    normalization_path = tmp_path / "normalization.json"
    normalization_path.write_text(json.dumps({"sha256": "norm"}))
    freeze_path = tmp_path / "freeze_summary.json"
    freeze_path.write_text(json.dumps({"training_approved": True, "manifest_sha256": "manifest"}))
    week5_path = tmp_path / "week5_contracts_summary.json"
    week5_path.write_text(json.dumps({
        "week5_contracts_complete": True, "evaluation_labels_loaded": False,
        "supported_class_indices": list(range(1, 17)),
    }))
    output = tmp_path / "sealed"
    config = {
        "paths": {
            "week7_summary_path": str(summary_path), "checkpoint_ledger_path": str(ledger_path),
            "manifest_path": str(manifest_path), "normalization_path": str(normalization_path),
            "freeze_summary_path": str(freeze_path), "week5_contracts_path": str(week5_path),
            "candidate_labels_path": str(candidate_path), "output_dir": str(output),
        },
        "frozen": {
            "week7_summary_sha256": file_sha256(summary_path),
            "checkpoint_ledger_sha256": file_sha256(ledger_path),
            "manifest_contract_sha256": "manifest", "normalization_sha256": "norm",
        },
        "uncertainty": {"bootstrap_seed": 1729, "bootstrap_replicates": 1000},
    }
    config_path = tmp_path / "week8.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = freeze_week8_evaluation(config_path)
    sealed = pd.read_parquet(output / "evaluation_labels.parquet")
    contract = json.loads((output / "evaluation_contract.json").read_text())
    assert result["partition_counts"] == {"I": 3000, "T-FI": 4000, "T-PT": 4000}
    assert len(sealed) == 11000 and sealed["patch_id"].is_unique
    assert contract["candidate_rows"] == 12500 and contract["excluded_candidate_rows"] == 1500
    assert {"mgrs_group", "block_12km", "quarter"}.issubset(sealed.columns)
    assert contract["model_selection_frozen_before_label_access"] is True


def test_ap_is_na_without_both_classes() -> None:
    assert np.isnan(average_precision(np.zeros(4), np.arange(4)))
    assert np.isnan(average_precision(np.ones(4), np.arange(4)))


def test_equal_country_ood_is_not_pooled() -> None:
    assert equal_country_ood(0.8, 0.2) == pytest.approx(0.5)


def test_normalized_log_aulc_is_hand_checkable() -> None:
    assert normalized_log_aulc([100, 1000, 10000], [0.2, 0.4, 0.6]) == pytest.approx(0.4)


def test_geographic_bootstrap_is_reproducible_and_grouped() -> None:
    groups = np.asarray(["a", "a", "b", "b", "c"])
    first = _bootstrap_draws(groups, 10, 1729)
    second = _bootstrap_draws(groups, 10, 1729)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert all(len(draw) >= 3 for draw in first)


def test_prediction_artifact_never_duplicates_labels(tmp_path: Path) -> None:
    path = tmp_path / "prediction.npz"
    _save_prediction(path, np.zeros((3, 19), dtype=np.float32), ["a", "b", "c"])
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {"logits", "patch_ids"}


def test_week8_notebooks_are_output_free_and_small() -> None:
    for notebook in (prepare(), seed_notebook(17), seed_notebook(29), seed_notebook(43), aggregate()):
        assert len(json.dumps(notebook).encode()) < 1_000_000
        assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_sealed_paths_are_gitignored() -> None:
    ignored = (ROOT / ".gitignore").read_text()
    assert "data/sealed/" in ignored
    assert "outputs/" in ignored
