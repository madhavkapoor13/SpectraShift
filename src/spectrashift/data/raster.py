from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.enums import Resampling
from rasterio.io import DatasetReader


@dataclass(frozen=True)
class ResampledBand:
    values: np.ndarray
    valid: np.ndarray
    scale: float
    offset: float
    nodata: float | None


def read_resampled_band(dataset: DatasetReader, height: int, width: int) -> ResampledBand:
    values = dataset.read(1, out_shape=(height, width), resampling=Resampling.bilinear)
    valid = dataset.read_masks(1, out_shape=(height, width), resampling=Resampling.nearest) > 0
    if values.shape != (height, width) or valid.shape != (height, width):
        raise ValueError("Unexpected resampled shape")
    scale = float(dataset.scales[0]) if dataset.scales else 1.0
    offset = float(dataset.offsets[0]) if dataset.offsets else 0.0
    return ResampledBand(values, valid, scale, offset, dataset.nodata)


def pack_validity(valid: np.ndarray) -> np.ndarray:
    if valid.ndim != 3:
        raise ValueError("Expected validity mask shaped [C,H,W]")
    return np.packbits(valid.reshape(valid.shape[0], -1), axis=1, bitorder="little")


def unpack_validity(packed: np.ndarray, height: int, width: int) -> np.ndarray:
    if packed.ndim != 2:
        raise ValueError("Expected packed validity shaped [C,bytes]")
    values = np.unpackbits(packed, axis=1, count=height * width, bitorder="little")
    return values.reshape(packed.shape[0], height, width).astype(bool)

