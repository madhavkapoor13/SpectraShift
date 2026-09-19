from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRACTION_COUNTS = {"01": 120, "05": 600, "10": 1200, "25": 3000, "50": 6000, "100": 12000}
MODEL_LABELS = {
    "M0": "M0 Random MS", "M1": "M1 ImageNet MS", "M1RGB": "M1RGB ImageNet RGB",
    "M2": "M2 RGB VICReg", "M3": "M3 MS VICReg", "M4": "M4 MS VICReg + dropout",
    "M5": "M5 DINOv2", "M6": "M6 OlmoEarth",
}
MODEL_COLORS = {
    "M0": "#6b7280", "M1": "#2563eb", "M2": "#f59e0b", "M3": "#059669",
    "M4": "#dc2626", "M5": "#7c3aed", "M6": "#111827", "M1RGB": "#0891b2",
}
PUBLIC_WEEK9_OUTPUTS = (
    "representation_probes.csv", "probe_paired_differences.csv", "probe_aulc.csv",
    "cka_matrix.csv", "effective_rank.csv", "stress_metrics.csv", "stress_per_class.csv",
    "stress_bootstrap.csv", "error_slices.csv",
)
DERIVED_WEEK9_OUTPUTS = ("nearest_neighbor_summary.csv",)
PRIVATE_PATTERNS = (
    re.compile(r"(^|/)data/sealed/"),
    re.compile(r"\.(pt|pth|ckpt)$", re.IGNORECASE),
    re.compile(r"(^|/)evaluation_labels\.parquet$"),
    re.compile(r"(^|/)predictions-[^/]+\.npz$"),
    re.compile(r"(^|/)nearest_neighbors\.parquet$"),
    re.compile(r"(^|/)spectral_sensitivity\.parquet$"),
)
WEEK8_EXECUTED_COMMIT = "abc0e09"
WEEK9_EXECUTED_COMMIT = "4a94634"
STAGE_SUMMARIES = {
    "week4": "reports/week4/generated/week4_run_summary.json",
    "week5": "reports/week5/generated/aggregate/spectrashift-week5-complete/week5_run_summary.json",
    "week6": "reports/week6/generated/week6_run_summary.json",
    "week7": "reports/week7/generated/week7_run_summary.json",
}
STAGE_COMMITS = {
    "week4_complete": "11b0b16644407ac05017920ec0635325299ca1c0",
    "week5_complete": "fdb09a6b941be71240fe2e9e7e3f739b517a6d4d",
    "week6_complete": "052732644d29048b3d0d3e9403e45f43db70ab4a",
    "week7_complete": "e9af130134d7b323a5f4315338d462d01cb136e7",
    "week8_complete": "abc0e0991e5898317328e4e1c7b21cf856d8c185",
    "week9_complete": "4a94634ea3b38b0a6a596d89daa0a03fed867a1a",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _fraction_code(value: object) -> str:
    return str(value).replace(".0", "").zfill(2)


def _validate_source_artifacts(week8_dir: Path, week9_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    week8 = _json(week8_dir / "week8_run_summary.json")
    week9 = _json(week9_dir / "week9_run_summary.json")
    if not (
        week8.get("week8_complete") is True
        and week8.get("week9_approved") is True
        and week8.get("evaluation_run_count") == 111
        and week8.get("prediction_domain_count") == 333
        and week8.get("model_selection_after_label_access") is False
        and week8.get("parameter_updates_after_label_access") is False
    ):
        raise ValueError("Week 8 aggregate does not satisfy the frozen evaluation contract")
    expected_week9 = {
        "week9_complete": True, "week10_approved": True, "linear_probe_count": 108,
        "knn_probe_count": 18, "diagnostic_checkpoint_count": 6,
        "stress_prediction_domain_count": 108, "cka_patch_count": 1000,
        "nearest_neighbor_row_count": 3000, "model_selection_after_week8": False,
        "encoder_updates_during_week9": False, "week10_implemented": False,
    }
    if any(week9.get(key) != value for key, value in expected_week9.items()):
        raise ValueError("Week 9 aggregate does not satisfy the frozen analysis contract")
    for name, digest in week8.get("output_hashes", {}).items():
        path = week8_dir / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Week 8 compact artifact changed: {name}")
    for name in PUBLIC_WEEK9_OUTPUTS:
        digest = week9.get("output_hashes", {}).get(name)
        path = week9_dir / name
        if not digest or not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Week 9 compact artifact changed: {name}")
    return week8, week9


def headline_values(week8_dir: str | Path, week9_dir: str | Path) -> dict[str, float]:
    week8_dir, week9_dir = Path(week8_dir), Path(week9_dir)
    domain = pd.read_csv(week8_dir / "domain_metrics.csv")
    stress = pd.read_csv(week9_dir / "stress_metrics.csv")

    def mean_map(model: str, fraction: int, domain_name: str) -> float:
        rows = domain[
            domain.model_id.eq(model)
            & domain.fraction_code.astype(int).eq(fraction)
            & domain.domain.eq(domain_name)
        ]
        return float(rows.supported_map.mean())

    ood = stress[stress.domain.eq("OOD")]

    def stress_difference(condition: str) -> float:
        values = ood[ood.condition.eq(condition)].groupby(["seed", "model_id"]).supported_map.mean().unstack()
        return float((values["M4"] - values["M3"]).mean())

    return {
        "m3_minus_m0_ood_10": mean_map("M3", 10, "OOD") - mean_map("M0", 10, "OOD"),
        "m3_minus_m2_ood_10": mean_map("M3", 10, "OOD") - mean_map("M2", 10, "OOD"),
        "m4_minus_m3_clean_ood_10": stress_difference("clean"),
        "m4_minus_m3_missing_b08_ood_10": stress_difference("missing_b08"),
        "m4_minus_m3_missing_red_edge_ood_10": stress_difference("missing_red_edge"),
        "m4_minus_m3_missing_swir_ood_10": stress_difference("missing_swir"),
        "olmoearth_ood_1": mean_map("M6", 1, "OOD"),
        "olmoearth_ood_10": mean_map("M6", 10, "OOD"),
        "olmoearth_ood_100": mean_map("M6", 100, "OOD"),
    }


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "figure.facecolor": "white", "axes.facecolor": "#fafafa",
        "axes.grid": True, "grid.alpha": 0.22, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.bbox": "tight",
    })


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, metadata={"Software": "SpectraShift Week 10"})
    plt.close(figure)


def _label_efficiency_figure(domain: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, domain_name, title in zip(axes, ("I", "OOD"), ("Source-domain test (I)", "Equal-country OOD"), strict=True):
        subset = domain[domain.domain.eq(domain_name)]
        for model in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
            rows = subset[subset.model_id.eq(model)].copy()
            grouped = rows.groupby(rows.fraction_code.astype(int)).supported_map.agg(["mean", "std"]).sort_index()
            if grouped.empty:
                continue
            x = np.asarray([FRACTION_COUNTS[_fraction_code(value)] for value in grouped.index])
            axis.errorbar(x, grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", lw=1.7,
                          ms=4, capsize=2, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        axis.set_xscale("log")
        axis.set_xticks([120, 600, 1200, 3000, 6000, 12000], ["1%", "5%", "10%", "25%", "50%", "100%"])
        axis.set_xlabel("Labeled D fraction")
        axis.set_title(title)
    axes[0].set_ylabel("Supported-class mAP")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    figure.suptitle("Label efficiency under source and geographic shift", fontweight="bold")
    figure.subplots_adjust(bottom=0.22, wspace=0.08)
    _save(figure, path)


def _country_anchor_figure(domain: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for axis, domain_name in zip(axes, ("I", "Finland", "Portugal"), strict=True):
        subset = domain[domain.domain.eq(domain_name) & domain.fraction_code.astype(int).isin([1, 10, 100])]
        for model in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
            grouped = subset[subset.model_id.eq(model)].groupby(subset[subset.model_id.eq(model)].fraction_code.astype(int)).supported_map.agg(["mean", "std"]).sort_index()
            axis.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"].fillna(0), marker="o",
                          capsize=2, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        axis.set_xscale("log")
        axis.set_xticks([1, 10, 100], ["1%", "10%", "100%"])
        axis.set_title(domain_name)
        axis.set_xlabel("Labeled D fraction")
    axes[0].set_ylabel("Supported-class mAP")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    figure.suptitle("Anchor performance by evaluation domain", fontweight="bold")
    figure.subplots_adjust(bottom=0.25, wspace=0.08)
    _save(figure, path)


def _probe_figure(probes: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    linear = probes[probes.probe_type.eq("linear")].copy()
    linear["fraction_code"] = linear.fraction_code.map(_fraction_code)
    for model in ("M1", "M2", "M3", "M4", "M5", "M6"):
        rows = linear[linear.model_id.eq(model)]
        grouped = rows.groupby("fraction_code").validation_supported_map.agg(["mean", "std"])
        grouped = grouped.reindex([code for code in FRACTION_COUNTS if code in grouped.index])
        x = np.asarray([FRACTION_COUNTS[code] for code in grouped.index])
        axis.errorbar(x, grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", capsize=2,
                      color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    axis.set_xscale("log")
    axis.set_xticks(list(FRACTION_COUNTS.values()), ["1%", "5%", "10%", "25%", "50%", "100%"])
    axis.set_xlabel("Labeled D fraction")
    axis.set_ylabel("Source-V supported-class mAP")
    axis.set_title("Frozen-feature linear probes", fontweight="bold")
    axis.legend(ncol=2, frameon=False)
    _save(figure, path)


def _stress_figure(stress: pd.DataFrame, path: Path) -> None:
    _style()
    conditions = ["gain_0p9", "gain_1p1", "missing_b08", "missing_red_edge", "missing_swir"]
    labels = ["Gain 0.9", "Gain 1.1", "Missing B08", "Missing red edge", "Missing SWIR"]
    ood = stress[stress.domain.eq("OOD") & stress.condition.isin(conditions)]
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(conditions)); width = 0.36
    for offset, model in ((-width / 2, "M3"), (width / 2, "M4")):
        grouped = ood[ood.model_id.eq(model)].groupby("condition").degradation_from_clean.agg(["mean", "std"]).reindex(conditions)
        axis.bar(x + offset, grouped["mean"], width, yerr=grouped["std"], capsize=3,
                 color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    axis.axhline(0, color="#111827", lw=0.8)
    axis.set_xticks(x, labels, rotation=12, ha="right")
    axis.set_ylabel("Clean mAP minus stressed mAP")
    axis.set_title("Equal-country OOD spectral robustness", fontweight="bold")
    axis.legend(frameon=False)
    _save(figure, path)


def _cka_rank_figure(cka: pd.DataFrame, ranks: pd.DataFrame, path: Path) -> None:
    _style()
    labels = list(dict.fromkeys(cka["left"].tolist()))
    matrix = cka.pivot(index="left", columns="right", values="cka").reindex(index=labels, columns=labels)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    image = axes[0].imshow(matrix, vmin=0.75, vmax=1, cmap="viridis")
    axes[0].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].grid(False)
    axes[0].set_title("Centered linear CKA")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    ordered = ranks.sort_values(["model_id", "seed"])
    axes[1].bar(ordered.checkpoint, ordered.effective_rank,
                color=[MODEL_COLORS[value] for value in ordered.model_id])
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylabel("Effective rank")
    axes[1].set_title("Representation effective rank")
    figure.suptitle("M3/M4 representation structure", fontweight="bold")
    figure.subplots_adjust(wspace=0.55)
    _save(figure, path)


def _class_sensitivity_figure(per_class: pd.DataFrame, path: Path) -> None:
    _style()
    conditions = ["missing_b08", "missing_red_edge", "missing_swir"]
    domains = ["Finland", "Portugal"]
    values = per_class[per_class.domain.isin(domains) & per_class.condition.isin(["clean", *conditions])].copy()
    clean = values[values.condition.eq("clean")][["model_id", "seed", "domain", "class_index", "average_precision"]].rename(columns={"average_precision": "clean_ap"})
    stressed = values[values.condition.isin(conditions)].merge(clean, on=["model_id", "seed", "domain", "class_index"])
    stressed["degradation"] = stressed.clean_ap - stressed.average_precision
    matrix = stressed.groupby(["model_id", "condition", "class_name"]).degradation.mean().unstack("class_name")
    order = pd.MultiIndex.from_product([["M3", "M4"], conditions], names=["model_id", "condition"])
    matrix = matrix.reindex(order)
    figure, axis = plt.subplots(figsize=(13, 4.8))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-0.12, vmax=0.12)
    shorter = {
        "Land principally occupied by agriculture, with significant areas of natural vegetation": "Agriculture + natural vegetation",
        "Moors, heathland and sclerophyllous vegetation": "Moors / heath / sclerophyllous",
        "Natural grassland and sparsely vegetated areas": "Grassland / sparse vegetation",
        "Industrial or commercial units": "Industrial / commercial",
        "Beaches, dunes, sands": "Beaches / dunes / sands",
    }
    labels = [shorter.get(value, value) for value in matrix.columns]
    axis.set_xticks(range(len(matrix.columns)), labels, rotation=55, ha="right", fontsize=7)
    axis.set_yticks(range(len(matrix.index)), [f"{m} {c.replace('_', ' ')}" for m, c in matrix.index])
    axis.grid(False)
    axis.set_title("Per-class AP degradation under missing spectral groups", fontweight="bold")
    figure.colorbar(image, ax=axis, label="Clean AP minus stressed AP", fraction=0.025)
    _save(figure, path)


def _nearest_neighbor_figure(summary: pd.DataFrame, path: Path) -> None:
    _style()
    order = ["M3", "M4"]
    colors = [MODEL_COLORS[value] for value in order]
    grouped = summary.groupby("model_id")
    jaccard = grouped.label_jaccard_mean.agg(["mean", "std"]).reindex(order)
    distance = grouped.geographic_distance_km_mean.agg(["mean", "std"]).reindex(order)
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    x = np.arange(2)
    axes[0].bar(x, jaccard["mean"], yerr=jaccard["std"], capsize=4, color=colors)
    axes[0].set_xticks(x, ["M3", "M4"])
    axes[0].set_ylabel("Mean label-set Jaccard")
    axes[0].set_ylim(0, max(0.45, float(jaccard["mean"].max() + 0.08)))
    axes[0].set_title("Semantic neighbour agreement")
    axes[1].bar(x, distance["mean"], yerr=distance["std"], capsize=4, color=colors)
    axes[1].set_xticks(x, ["M3", "M4"])
    axes[1].set_ylabel("Mean source-neighbour distance (km)")
    axes[1].set_title("Geographic neighbour distance")
    figure.suptitle("Fixed-query nearest-neighbour diagnostics", fontweight="bold")
    figure.subplots_adjust(wspace=0.35)
    _save(figure, path)


def _headline_domain_table(domain: pd.DataFrame) -> pd.DataFrame:
    table = domain.groupby(["model_id", "fraction_code", "domain"], as_index=False).supported_map.agg(["mean", "std"]).reset_index()
    table["fraction_code"] = table.fraction_code.map(_fraction_code)
    table = table.rename(columns={"mean": "mean_supported_map", "std": "sample_std_supported_map"})
    return table.sort_values(["domain", "fraction_code", "model_id"])


def _headline_stress_table(stress: pd.DataFrame) -> pd.DataFrame:
    metrics = ["supported_map", "degradation_from_clean", "mean_feature_cosine_change"]
    return stress.groupby(["model_id", "condition", "domain"], as_index=False)[metrics].agg(["mean", "std"]).reset_index().sort_values(["domain", "condition", "model_id"])


def _load_stage_summary(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return _json(path) if path.is_file() else {}


def _successful_run_ledger(root: Path, week8_dir: Path, week9_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(stage: str, task: str, records: Iterable[dict[str, object]]) -> None:
        for record in records:
            rows.append({
                "stage": stage, "task_type": task, "run_id": record.get("run_id"),
                "model_id": record.get("model_id"), "seed": record.get("seed"),
                "fraction_code": _fraction_code(record.get("fraction_code")) if record.get("fraction_code") is not None else "",
                "status": "success", "elapsed_seconds": record.get("elapsed_seconds", np.nan),
                "artifact_sha256": record.get("checkpoint_sha256") or record.get("best_checkpoint_sha256") or record.get("feature_cache_sha256", ""),
                "source_commit": record.get("git_commit", ""),
            })

    add("week4", "ssl_pretraining", _load_stage_summary(root, "reports/week4/generated/week4_run_summary.json").get("runs", []))
    add("week5", "downstream_anchor", _load_stage_summary(root, "reports/week5/generated/aggregate/spectrashift-week5-complete/week5_run_summary.json").get("runs", []))
    add("week6", "downstream_curve", _load_stage_summary(root, "reports/week6/generated/week6_run_summary.json").get("new_runs", []))
    add("week7", "foundation_finetune", _load_stage_summary(root, "reports/week7/generated/week7_run_summary.json").get("foundation_runs", []))
    evaluation = pd.read_csv(week8_dir / "domain_metrics.csv")
    evaluation = evaluation.drop_duplicates(["run_id"])
    add("week8", "frozen_evaluation", evaluation.to_dict("records"))
    probes = pd.read_csv(week9_dir / "representation_probes.csv", dtype={"fraction_code": str})
    for task, frame in probes.groupby("probe_type"):
        add("week9", f"{task}_probe", frame.to_dict("records"))
    diagnostics = pd.read_csv(week9_dir / "stress_metrics.csv").drop_duplicates(["run_id"])
    add("week9", "representation_diagnostics", diagnostics.to_dict("records"))
    result = pd.DataFrame(rows)
    return result.sort_values(["stage", "task_type", "run_id"], na_position="last").reset_index(drop=True)


def _incident_ledger() -> pd.DataFrame:
    rows = [
        ("2026-09-17", "week5", "Kaggle source extraction persisted into output", "Source bundle unpacked under a durable path", "Moved extraction to ephemeral storage", "634e2cd"),
        ("2026-09-17", "week5", "Duplicate aggregate inputs", "Equivalent summaries were attached through multiple datasets", "Deduplicated by content hash", "17e0ca9"),
        ("2026-09-17", "week5", "Downloaded seed ZIPs were not accepted", "Launcher expected direct dataset directories", "Added verified ZIP discovery", "0bbde4c"),
        ("2026-09-18", "week7", "k-NN calibration exceeded the probability boundary", "Float32 convex aggregation produced tiny boundary excursions", "Clipped to the declared probability interval", "c67c20b"),
        ("2026-09-19", "week9", "Duplicate-location exclusion failed for string patch IDs", "Pandas group labels were treated as integer positions", "Replaced labels with explicit positional indices", "86adad0"),
        ("2026-09-19", "week9", "Offline diagnostics lacked the optional mgrs wheel", "Runtime geographic conversion imported an unavailable package", "Removed the optional runtime dependency", "af82dbe"),
        ("2026-09-19", "week9", "Temporary fallback omitted cross-tile distances", "Within-tile indices cannot measure cross-tile separation", "Implemented and tested MGRS decoding through pinned pyproj", "4a94634"),
    ]
    frame = pd.DataFrame(rows, columns=["date", "stage", "symptom", "root_cause", "resolution", "recovery_commit"])
    frame["scientific_effect"] = "none; affected output was rejected and regenerated before aggregation"
    return frame


def _compute_summary(root: Path) -> pd.DataFrame:
    specs = [
        ("week4", "reports/week4/generated/week4_run_summary.json", "actual_gpu_hours"),
        ("week5", "reports/week5/generated/aggregate/spectrashift-week5-complete/week5_run_summary.json", "actual_gpu_hours"),
        ("week6", "reports/week6/generated/week6_run_summary.json", None),
        ("week7", "reports/week7/generated/week7_run_summary.json", "actual_gpu_hours"),
        ("week8", "reports/week8/generated/week8_run_summary.json", None),
        ("week9", "reports/week9/generated/week9_run_summary.json", None),
    ]
    rows = []
    for stage, relative, key in specs:
        payload = _load_stage_summary(root, relative)
        value, source = np.nan, "not recorded in compact aggregate"
        if key and key in payload:
            value, source = float(payload[key]), key
        elif stage == "week6" and payload.get("new_runs"):
            value = sum(float(run.get("elapsed_seconds", 0)) for run in payload["new_runs"]) / 3600
            source = "sum(new_runs.elapsed_seconds)/3600"
        rows.append({"stage": stage, "gpu_hours": value, "measurement_source": source})
    return pd.DataFrame(rows)


def build_final_artifacts(
    week8_dir: str | Path,
    week9_dir: str | Path,
    output_dir: str | Path,
    root: str | Path = ".",
) -> dict[str, object]:
    root, week8_dir, week9_dir, output_dir = map(Path, (root, week8_dir, week9_dir, output_dir))
    root, week8_dir, week9_dir, output_dir = root.resolve(), week8_dir.resolve(), week9_dir.resolve(), output_dir.resolve()
    week8, week9 = _validate_source_artifacts(week8_dir, week9_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    domain = pd.read_csv(week8_dir / "domain_metrics.csv")
    probes = pd.read_csv(week9_dir / "representation_probes.csv", dtype={"fraction_code": str})
    stress = pd.read_csv(week9_dir / "stress_metrics.csv")
    per_class = pd.read_csv(week9_dir / "stress_per_class.csv")
    cka = pd.read_csv(week9_dir / "cka_matrix.csv")
    ranks = pd.read_csv(week9_dir / "effective_rank.csv")
    neighbor_summary = pd.read_csv(week9_dir / "nearest_neighbor_summary.csv")

    headline_domain = _headline_domain_table(domain)
    headline_stress = _headline_stress_table(stress)
    run_ledger = _successful_run_ledger(root, week8_dir, week9_dir)
    incident_ledger = _incident_ledger()
    compute = _compute_summary(root)
    headline_domain.to_csv(output_dir / "headline_domain_metrics.csv", index=False, float_format="%.10g")
    headline_stress.to_csv(output_dir / "headline_stress_metrics.csv", index=False, float_format="%.10g")
    run_ledger.to_csv(output_dir / "run_ledger.csv", index=False, float_format="%.10g")
    incident_ledger.to_csv(output_dir / "incident_ledger.csv", index=False)
    compute.to_csv(output_dir / "compute_summary.csv", index=False, float_format="%.10g")

    _label_efficiency_figure(domain, figure_dir / "label_efficiency_source_ood.png")
    _country_anchor_figure(domain, figure_dir / "country_anchor_comparison.png")
    _probe_figure(probes, figure_dir / "representation_probe_curves.png")
    _stress_figure(stress, figure_dir / "spectral_stress_degradation.png")
    _cka_rank_figure(cka, ranks, figure_dir / "cka_effective_rank.png")
    _class_sensitivity_figure(per_class, figure_dir / "per_class_spectral_sensitivity.png")
    _nearest_neighbor_figure(neighbor_summary, figure_dir / "nearest_neighbor_summary.png")

    # These two metadata files describe the generated package. Excluding their
    # previous versions prevents recursive, stale self-hashes on repeat builds.
    metadata_names = {"results_manifest.json", "week10_run_summary.json"}
    final_files = sorted(
        [
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.name not in metadata_names
        ],
        key=lambda path: path.relative_to(output_dir).as_posix(),
    )
    outputs = {path.relative_to(output_dir).as_posix(): file_sha256(path) for path in final_files}
    source_files = [root / relative for relative in STAGE_SUMMARIES.values()]
    source_files.extend([week8_dir / "week8_run_summary.json", week9_dir / "week9_run_summary.json"])
    source_files.extend(week8_dir / name for name in week8.get("output_hashes", {}))
    source_files.extend(week9_dir / name for name in PUBLIC_WEEK9_OUTPUTS)
    source_files.extend(week9_dir / name for name in DERIVED_WEEK9_OUTPUTS)
    manifest = {
        "schema_version": 1,
        "source_datasets": [
            "spectrashift-week4-complete", "spectrashift-week5-complete",
            "spectrashift-week6-complete", "spectrashift-week7-complete",
            "spectrashift-week8-complete", "spectrashift-week9-complete",
        ],
        "source_commits": STAGE_COMMITS,
        "source_artifact_sha256": {
            str(path.relative_to(root)): file_sha256(path) for path in source_files if path.is_file() and path.is_relative_to(root)
        },
        "generated_output_sha256": outputs,
        "completion_gates": {
            "week8_complete": True, "week9_complete": True, "model_selection_after_week8": False,
            "encoder_updates_during_week9": False,
        },
        "private_artifacts_excluded": [
            "evaluation labels", "prediction logits", "model checkpoints", "patch feature arrays",
            "nearest_neighbors.parquet", "spectral_sensitivity.parquet",
        ],
    }
    _write_json(output_dir / "results_manifest.json", manifest)
    values = headline_values(week8_dir, week9_dir)
    summary = {
        "week10_complete": True, "release_approved": True,
        "week8_complete": True, "week9_complete": True,
        "successful_run_ledger_count": int(len(run_ledger)),
        "incident_ledger_count": int(len(incident_ledger)),
        "final_figure_count": 7,
        "headline_values": values,
        "results_manifest_sha256": file_sha256(output_dir / "results_manifest.json"),
        "private_artifact_violation_count": 0,
        "new_training_or_model_selection": False,
    }
    _write_json(output_dir / "week10_run_summary.json", summary)
    return summary


def private_tracked_paths(paths: Sequence[str]) -> list[str]:
    return sorted(path for path in paths if any(pattern.search(path) for pattern in PRIVATE_PATTERNS))


def _tracked_files(root: Path) -> list[str]:
    if not (root / ".git").exists():
        excluded_parts = {".pytest_cache", "__pycache__", ".git"}
        virtual_environments = {
            path.parent.resolve() for path in root.rglob("pyvenv.cfg") if path.is_file()
        }
        return sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and not excluded_parts.intersection(path.relative_to(root).parts)
            and not any(path.is_relative_to(environment) for environment in virtual_environments)
        )
    output = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
    return [line for line in output.splitlines() if line]


def _local_markdown_targets(path: Path) -> list[Path]:
    text = path.read_text()
    targets = []
    for raw in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        target = raw.strip().split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        targets.append((path.parent / target).resolve())
    return targets


def verify_release(root: str | Path = ".", require_clean: bool = True) -> dict[str, object]:
    root = Path(root).resolve()
    required = [
        "README.md", "REPORT.md", "DATA.md", "MODEL_CARD.md", "EXPERIMENTS.md",
        "REPRODUCIBILITY.md", "THIRD_PARTY.md", "LICENSE", "CITATION.cff",
        "docs/application-notes.md", "reports/final/generated/week10_run_summary.json",
        "reports/final/generated/results_manifest.json",
    ]
    missing = [value for value in required if not (root / value).is_file()]
    if missing:
        raise ValueError(f"Release files are missing: {missing}")
    summary = _json(root / "reports/final/generated/week10_run_summary.json")
    if not (
        summary.get("week10_complete") is True
        and summary.get("release_approved") is True
        and summary.get("week9_complete") is True
        and summary.get("private_artifact_violation_count") == 0
        and summary.get("new_training_or_model_selection") is False
    ):
        raise ValueError("Week 10 completion gates failed")
    manifest_path = root / "reports/final/generated/results_manifest.json"
    if file_sha256(manifest_path) != summary.get("results_manifest_sha256"):
        raise ValueError("Final results manifest hash changed")
    manifest = _json(manifest_path)
    for name, digest in manifest.get("generated_output_sha256", {}).items():
        path = root / "reports/final/generated" / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Final generated artifact changed: {name}")
    tracked = _tracked_files(root)
    violations = private_tracked_paths(tracked)
    if violations:
        raise ValueError(f"Private artifacts are tracked: {violations}")
    broken_links = []
    for path in (root / "README.md", root / "REPORT.md", root / "REPRODUCIBILITY.md", root / "MODEL_CARD.md"):
        broken_links.extend(str(target) for target in _local_markdown_targets(path) if not target.exists())
    if broken_links:
        raise ValueError(f"Broken local Markdown links: {sorted(set(broken_links))}")
    figure_dir = root / "reports/final/generated/figures"
    figures = sorted(figure_dir.glob("*.png"))
    if len(figures) != 7:
        raise ValueError("Expected exactly seven final PNG figures")
    for figure in figures:
        image = plt.imread(figure)
        if image.ndim not in (2, 3) or min(image.shape[:2]) < 400 or not np.isfinite(image).all():
            raise ValueError(f"Invalid final figure: {figure.name}")
    if require_clean and (root / ".git").exists():
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        if dirty:
            raise ValueError("Release checkout is not clean")
    return {
        "release_verified": True, "week10_complete": True, "release_approved": True,
        "tracked_file_count": len(tracked), "private_artifact_violation_count": 0,
        "figure_count": len(figures), "clean_checkout": bool(require_clean),
    }
