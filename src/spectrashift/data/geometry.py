from __future__ import annotations

import math
from dataclasses import dataclass

from pyproj import CRS, Transformer
from rasterio.io import DatasetReader
from shapely.geometry import Polygon
from shapely.ops import transform
from shapely.strtree import STRtree


@dataclass(frozen=True)
class GeometryConfig:
    metric_crs: str = "EPSG:3035"
    buffer_m: float = 2400.0
    block_size_m: float = 12000.0
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0


def footprint_from_dataset(dataset: DatasetReader, metric_crs: str = "EPSG:3035") -> Polygon:
    if dataset.crs is None:
        raise ValueError("Raster has no CRS")
    bounds = dataset.bounds
    native = Polygon(
        [
            (bounds.left, bounds.bottom),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
            (bounds.left, bounds.top),
        ]
    )
    transformer = Transformer.from_crs(CRS.from_user_input(dataset.crs), metric_crs, always_xy=True)
    projected = transform(transformer.transform, native)
    if not projected.is_valid or projected.area <= 0:
        raise ValueError("Projected raster footprint is invalid")
    return projected


def spatial_block_id(geometry: Polygon, config: GeometryConfig) -> str:
    centroid = geometry.centroid
    column = math.floor((centroid.x - config.origin_x_m) / config.block_size_m)
    row = math.floor((centroid.y - config.origin_y_m) / config.block_size_m)
    return f"b{column}_{row}"


def keep_outside_buffer(
    candidates: list[Polygon], exclusions: list[Polygon], distance_m: float
) -> list[bool]:
    if not exclusions:
        return [True] * len(candidates)
    tree = STRtree(exclusions)
    return [len(tree.query(item, predicate="dwithin", distance=distance_m)) == 0 for item in candidates]

