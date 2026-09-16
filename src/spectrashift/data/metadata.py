from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


PATCH_RE = re.compile(
    r"^(?P<satellite>S2[AB])_MSIL2A_(?P<timestamp>\d{8}T\d{6})_"
    r"N\d{4}_R(?P<orbit>\d{3})_T(?P<mgrs>[0-9]{2}[A-Z]{3})_"
    r"(?P<h_order>\d+)_(?P<v_order>\d+)$"
)


def parse_patch_ids(patch_ids: pd.Series) -> pd.DataFrame:
    parsed = patch_ids.astype(str).str.extract(PATCH_RE)
    if parsed.isna().any(axis=1).any():
        bad = patch_ids[parsed.isna().any(axis=1)].head(5).tolist()
        raise ValueError(f"Unparseable BigEarthNet v2 patch IDs: {bad}")
    parsed["timestamp"] = pd.to_datetime(parsed["timestamp"], format="%Y%m%dT%H%M%S", utc=True)
    parsed["h_order"] = parsed["h_order"].astype(int)
    parsed["v_order"] = parsed["v_order"].astype(int)
    parsed["mgrs_tile"] = "T" + parsed.pop("mgrs")
    parsed["location_key"] = (
        parsed["mgrs_tile"]
        + "_"
        + parsed["h_order"].astype(str)
        + "_"
        + parsed["v_order"].astype(str)
    )
    return parsed


def normalize_labels(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(sorted(str(item) for item in value))
    raise TypeError(f"Unsupported labels value: {type(value)!r}")


def enrich_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"patch_id", "labels", "split", "country"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if frame["patch_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate patch_id values")
    parsed = parse_patch_ids(frame["patch_id"])
    enriched = pd.concat([frame.reset_index(drop=True), parsed], axis=1)
    enriched["labels"] = enriched["labels"].map(normalize_labels)
    return enriched

