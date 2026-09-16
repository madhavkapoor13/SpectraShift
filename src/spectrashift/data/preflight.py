from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
import yaml


def projected_staging_bytes(patches: int, channels: int, height: int, width: int) -> int:
    pixel_bytes = patches * channels * height * width * 2
    validity_bytes = patches * channels * math.ceil(height * width / 8)
    progress_bytes = patches * (channels + channels * 4 + 256 + 32)
    return pixel_bytes + validity_bytes + progress_bytes


def _check_archive_source(value: str) -> str | None:
    if urlparse(value).scheme not in {"http", "https"}:
        return None if Path(value).exists() else "local file is missing"
    try:
        request = Request(value, headers={"Range": "bytes=0-0", "User-Agent": "SpectraShift/0.1"})
        with urlopen(request, timeout=45) as response:
            if not response.read(1):
                return "remote source returned no data"
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def run_preflight(config_path: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    dataset = config["dataset"]
    staging = config["staging"]
    archive_parts = [str(value) for value in dataset.get("archive_parts", [])]
    archive = Path(dataset["archive_path"]) if not archive_parts else None
    archive_errors = {value: error for value in archive_parts if (error := _check_archive_source(value))}
    archive_present = not archive_errors if archive_parts else bool(archive and archive.exists())
    candidate_manifest = Path(dataset["candidate_manifest"])
    if not candidate_manifest.exists():
        raise FileNotFoundError(f"Candidate manifest not found: {candidate_manifest}")
    candidates = pd.read_parquet(candidate_manifest, columns=["patch_id"])
    patches = len(candidates)
    channels = len(dataset["canonical_bands"])
    height = int(dataset["target_height"])
    width = int(dataset["target_width"])
    projected = projected_staging_bytes(patches, channels, height, width)
    configured_cap = int(staging["maximum_output_bytes"])
    output_dir = Path(staging["output_dir"])
    probe = output_dir if output_dir.exists() else output_dir.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    result = {
        "candidate_patches": patches,
        "channels": channels,
        "shape": [height, width],
        "projected_staging_bytes": projected,
        "configured_cap_bytes": configured_cap,
        "free_bytes": free,
        "archive_path": str(archive) if archive else None,
        "archive_parts": archive_parts,
        "remote_archive": any(urlparse(value).scheme in {"http", "https"} for value in archive_parts),
        "archive_present": archive_present,
        "archive_errors": archive_errors,
        "size_gate": projected <= configured_cap,
        "disk_gate": projected <= free,
        "ready": archive_present and projected <= configured_cap and projected <= free,
    }
    if not result["size_gate"]:
        raise RuntimeError(json.dumps(result, indent=2))
    return result
