from __future__ import annotations

import numpy as np

from spectrashift.eval.metrics import (
    average_precision,
    expected_calibration_error,
    fit_validation_thresholds,
    multilabel_metrics,
)


def test_average_precision_is_hand_checkable() -> None:
    assert np.isclose(average_precision([1, 0, 1], [0.9, 0.8, 0.7]), (1 + 2 / 3) / 2)
    assert np.isnan(average_precision([0, 0], [0.9, 0.1]))


def test_multilabel_metrics_and_validation_threshold_provenance() -> None:
    targets = np.array([[1, 0], [0, 1], [1, 1]])
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]])
    result = multilabel_metrics(targets, scores)
    assert result["macro_average_precision"] == 1.0
    assert result["micro_f1"] == 1.0
    assert expected_calibration_error(targets, targets.astype(float)) == 0.0
    fitted = fit_validation_thresholds(targets, scores, np.array([0.5, 0.75]))
    assert fitted["fitted_on"] == "V"
    assert len(fitted["sha256"]) == 64
