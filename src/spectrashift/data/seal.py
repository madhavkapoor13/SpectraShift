from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


EVALUATION_PARTITIONS = {"I", "T-FI", "T-PT"}


def seal_evaluation_labels(config_path: str | Path) -> dict[str, object]:
    """Materialize final evaluation labels outside the development workflow."""
    config = yaml.safe_load(Path(config_path).read_text())
    final_path = Path(config["staging"]["final_manifest_dir"]) / "partitions.parquet"
    final = pd.read_parquet(final_path, columns=["partition", "patch_id", "labels"])
    evaluation = final[final["partition"].isin(EVALUATION_PARTITIONS)]
    if not evaluation["labels"].isna().all():
        raise ValueError("Public frozen manifest exposes evaluation labels")
    candidates = pd.read_parquet(config["dataset"]["sealed_candidate_labels"])
    sealed = evaluation[["partition", "patch_id"]].merge(
        candidates, on=["partition", "patch_id"], how="left", validate="one_to_one"
    )
    if sealed["labels"].isna().any() or len(sealed) != len(evaluation):
        raise ValueError("Evaluation labels are incomplete in the sealed candidate artifact")
    output_dir = Path(config["staging"]["sealed_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_labels.parquet"
    sealed.to_parquet(output_path, index=False)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    result = {
        "evaluation_rows": len(sealed),
        "partition_counts": sealed.groupby("partition").size().to_dict(),
        "sha256": digest,
        "output": str(output_path),
    }
    (output_dir / "sealing_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
