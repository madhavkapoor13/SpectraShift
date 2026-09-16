from __future__ import annotations


def select_week4_pilot(
    ssl_summaries: list[dict[str, object]],
    probe_summaries: list[dict[str, object]],
    tie_tolerance: float = 0.005,
    preferred_learning_rate: float = 3e-4,
) -> dict[str, object]:
    probes = {str(value["run_id"]): value for value in probe_summaries}
    candidates = []
    for summary in ssl_summaries:
        run_id = str(summary["run_id"])
        if summary.get("stability_gate") and summary.get("week4_compute_gate") and run_id in probes:
            candidates.append({
                "run_id": run_id,
                "learning_rate": float(summary["learning_rate"]),
                "validation_macro_average_precision": float(
                    probes[run_id]["validation_macro_average_precision"]
                ),
                "forecast_hours_for_nine_ssl_runs": float(
                    summary["forecast_hours_for_nine_ssl_runs"]
                ),
            })
    if not candidates:
        return {
            "week4_approved": False,
            "reason": "No pilot passed stability, compute, and probe gates",
            "required_diagnostic_learning_rate": 3e-5,
        }
    candidates.sort(key=lambda value: value["validation_macro_average_precision"], reverse=True)
    selected = candidates[0]
    if len(candidates) > 1:
        gap = candidates[0]["validation_macro_average_precision"] - candidates[1][
            "validation_macro_average_precision"
        ]
        if gap < tie_tolerance:
            preferred = [
                candidate
                for candidate in candidates
                if candidate["learning_rate"] == preferred_learning_rate
            ]
            if preferred:
                selected = preferred[0]
    return {
        "week4_approved": True,
        "selected": selected,
        "candidates": candidates,
        "tie_tolerance": tie_tolerance,
        "tie_preference_learning_rate": preferred_learning_rate,
    }
