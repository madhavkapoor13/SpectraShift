from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from spectrashift.release import (
    PUBLIC_WEEK9_OUTPUTS,
    build_final_artifacts,
    headline_values,
    private_tracked_paths,
    verify_release,
)


ROOT = Path(__file__).parents[1]
WEEK8 = ROOT / "reports/week8/generated"
WEEK9 = ROOT / "reports/week9/generated"


def test_headline_values_are_bound_to_frozen_aggregates() -> None:
    values = headline_values(WEEK8, WEEK9)
    assert values["m3_minus_m0_ood_10"] == pytest.approx(-0.0286595230)
    assert values["m3_minus_m2_ood_10"] == pytest.approx(0.0216995135)
    assert values["m4_minus_m3_clean_ood_10"] == pytest.approx(-0.0060047945)
    assert values["m4_minus_m3_missing_red_edge_ood_10"] == pytest.approx(0.0333908500)
    assert values["m4_minus_m3_missing_swir_ood_10"] == pytest.approx(0.0404172413)


def test_final_builder_reproduces_counts_figures_and_ledgers(tmp_path: Path) -> None:
    output = tmp_path / "final"
    summary = build_final_artifacts(WEEK8, WEEK9, output, ROOT)
    first_manifest_hash = summary["results_manifest_sha256"]
    summary = build_final_artifacts(WEEK8, WEEK9, output, ROOT)
    assert summary["week10_complete"] and summary["release_approved"]
    assert summary["results_manifest_sha256"] == first_manifest_hash
    assert summary["successful_run_ledger_count"] == 363
    assert summary["incident_ledger_count"] == 7
    assert len(list((output / "figures").glob("*.png"))) == 7
    ledger = pd.read_csv(output / "run_ledger.csv")
    assert ledger.groupby(["stage", "task_type"]).size().to_dict() == {
        ("week4", "ssl_pretraining"): 9,
        ("week5", "downstream_anchor"): 45,
        ("week6", "downstream_curve"): 48,
        ("week7", "foundation_finetune"): 18,
        ("week8", "frozen_evaluation"): 111,
        ("week9", "knn_probe"): 18,
        ("week9", "linear_probe"): 108,
        ("week9", "representation_diagnostics"): 6,
    }
    manifest = json.loads((output / "results_manifest.json").read_text())
    assert "results_manifest.json" not in manifest["generated_output_sha256"]
    assert "week10_run_summary.json" not in manifest["generated_output_sha256"]
    assert "reports/week4/generated/week4_run_summary.json" in manifest["source_artifact_sha256"]
    assert "reports/week5/generated/aggregate/spectrashift-week5-complete/week5_run_summary.json" in manifest["source_artifact_sha256"]


def test_public_week9_import_excludes_patch_level_artifacts() -> None:
    assert set(PUBLIC_WEEK9_OUTPUTS).issubset(path.name for path in WEEK9.iterdir())
    assert not (WEEK9 / "nearest_neighbors.parquet").exists()
    assert not (WEEK9 / "spectral_sensitivity.parquet").exists()
    summary = json.loads((WEEK9 / "week9_run_summary.json").read_text())
    assert summary["week9_complete"] and summary["week10_approved"]


def test_private_artifact_patterns_are_rejected() -> None:
    paths = [
        "README.md", "reports/week9/generated/stress_metrics.csv",
        "data/sealed/evaluation_labels.parquet", "outputs/run/predictions-i-clean.npz",
        "checkpoints/model.pt", "reports/week9/generated/nearest_neighbors.parquet",
        "reports/week9/generated/spectral_sensitivity.parquet",
    ]
    assert private_tracked_paths(paths) == sorted(paths[2:])


def test_release_documents_and_artifacts_verify_without_dirty_checkout_gate() -> None:
    result = verify_release(ROOT, require_clean=False)
    assert result["release_verified"] and result["private_artifact_violation_count"] == 0
    report = (ROOT / "REPORT.md").read_text()
    assert 3000 <= len(report.split()) <= 4000
    assert "No p-values are calculated" in report
