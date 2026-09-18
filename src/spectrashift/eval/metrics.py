from __future__ import annotations

import hashlib
import json

import numpy as np


def _validate(targets: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets)
    scores = np.asarray(scores, dtype=np.float64)
    if targets.shape != scores.shape or targets.ndim != 2:
        raise ValueError("targets and scores must share shape [N,C]")
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("targets must be binary and scores must be finite")
    return targets.astype(bool), scores


def average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(targets.sum())
    if positives == 0 or positives == len(targets):
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = targets[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def fit_validation_thresholds(
    targets: np.ndarray,
    scores: np.ndarray,
    grid: np.ndarray | None = None,
) -> dict[str, object]:
    targets, scores = _validate(targets, scores)
    grid = np.asarray(grid if grid is not None else np.linspace(0.05, 0.95, 19))
    thresholds = []
    for class_index in range(targets.shape[1]):
        truth = targets[:, class_index]
        best = (float("-inf"), 0.5)
        for threshold in grid:
            prediction = scores[:, class_index] >= threshold
            tp = int((prediction & truth).sum())
            fp = int((prediction & ~truth).sum())
            fn = int((~prediction & truth).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            candidate = (f1, -abs(float(threshold) - 0.5), -float(threshold))
            if candidate > (best[0], -abs(best[1] - 0.5), -best[1]):
                best = (f1, float(threshold))
        thresholds.append(best[1])
    payload = {"fitted_on": "V", "thresholds": thresholds}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def fit_global_validation_threshold(
    targets: np.ndarray,
    scores: np.ndarray,
    grid: np.ndarray | None = None,
) -> dict[str, object]:
    """Select one validation threshold by micro F1.

    Ties are resolved toward 0.5 and then toward the lower threshold, as
    required by the frozen downstream protocol.
    """
    targets, scores = _validate(targets, scores)
    grid = np.asarray(grid if grid is not None else np.linspace(0.05, 0.95, 19))
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    best_f1 = float("-inf")
    for threshold in grid:
        prediction = scores >= float(threshold)
        tp = int((prediction & targets).sum())
        fp = int((prediction & ~targets).sum())
        fn = int((~prediction & targets).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        key = (f1, -abs(float(threshold) - 0.5), -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_f1 = float(f1)
    payload = {
        "fitted_on": "V",
        "threshold": best_threshold,
        "validation_micro_f1": best_f1,
        "grid": [float(value) for value in grid],
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


def expected_calibration_error(
    targets: np.ndarray, scores: np.ndarray, bins: int = 15
) -> float:
    targets, scores = _validate(targets, scores)
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("calibration scores must be probabilities")
    truth = targets.ravel()
    probability = scores.ravel()
    edges = np.linspace(0, 1, bins + 1)
    total = len(probability)
    ece = 0.0
    for index in range(bins):
        include = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if include.any():
            ece += include.mean() * abs(probability[include].mean() - truth[include].mean())
    return float(ece)


def classwise_expected_calibration_error(
    targets: np.ndarray, scores: np.ndarray, bins: int = 15
) -> tuple[float, list[float]]:
    targets, scores = _validate(targets, scores)
    values = [
        expected_calibration_error(targets[:, index:index + 1], scores[:, index:index + 1], bins)
        for index in range(targets.shape[1])
    ]
    return float(np.mean(values)), values


def multilabel_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    supported_indices: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> dict[str, object]:
    targets, scores = _validate(targets, scores)
    threshold_array = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (targets.shape[1],))
    prediction = scores >= threshold_array[None]
    tp = (prediction & targets).sum(axis=0)
    fp = (prediction & ~targets).sum(axis=0)
    fn = (~prediction & targets).sum(axis=0)
    per_class_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full_like(tp, np.nan, dtype=float), where=(2 * tp + fp + fn) > 0)
    recall = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan, dtype=float), where=(tp + fn) > 0)
    aps = np.array([average_precision(targets[:, i], scores[:, i]) for i in range(targets.shape[1])])
    total_tp, total_fp, total_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    supported = np.asarray(
        supported_indices if supported_indices is not None else np.arange(targets.shape[1]),
        dtype=int,
    )
    if supported.ndim != 1 or len(supported) == 0:
        raise ValueError("supported_indices must be a non-empty one-dimensional index list")
    if supported.min() < 0 or supported.max() >= targets.shape[1]:
        raise ValueError("supported_indices contains an out-of-range class index")
    macro_ece, per_class_ece = classwise_expected_calibration_error(targets, scores)
    brier = float(np.mean((scores - targets.astype(np.float64)) ** 2))
    return {
        "macro_average_precision": float(np.nanmean(aps[supported])),
        "all_class_macro_average_precision": float(np.nanmean(aps)),
        "micro_f1": 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1),
        "macro_f1": float(np.nanmean(per_class_f1)),
        "expected_calibration_error": expected_calibration_error(targets, scores),
        "macro_classwise_expected_calibration_error": macro_ece,
        "binary_brier_score": brier,
        "supported_class_indices": supported.tolist(),
        "per_class_average_precision": aps.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_expected_calibration_error": per_class_ece,
    }
