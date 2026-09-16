from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CANONICAL_BANDS = (
    "B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"
)
BAND_ADAPTERS = {
    "rgb": ("B04", "B03", "B02"),
    "core10": CANONICAL_BANDS[:10],
    "olmo12": CANONICAL_BANDS,
}


def band_indices(requested: Sequence[str], available: Sequence[str] = CANONICAL_BANDS) -> tuple[int, ...]:
    if len(set(available)) != len(available):
        raise ValueError("Available band order contains duplicates")
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Requested unavailable bands: {sorted(unknown)}")
    return tuple(available.index(name) for name in requested)


def adapt_bands(array: np.ndarray, adapter: str) -> np.ndarray:
    if adapter not in BAND_ADAPTERS:
        raise ValueError(f"Unknown band adapter: {adapter}")
    if array.ndim not in {3, 4}:
        raise ValueError("Expected [C,H,W] or [N,C,H,W]")
    channel_axis = 0 if array.ndim == 3 else 1
    if array.shape[channel_axis] != len(CANONICAL_BANDS):
        raise ValueError(f"Expected {len(CANICAL_BANDS)} canonical bands")
    indices = band_indices(BAND_ADAPTERS[adapter])
    return np.take(array, indices, axis=channel_axis)

