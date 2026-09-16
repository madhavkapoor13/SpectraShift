from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .raster import pack_validity, unpack_validity


class CandidateShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        total: int,
        channels: int,
        height: int,
        width: int,
        shard_size: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total = total
        self.channels = channels
        self.height = height
        self.width = width
        self.shard_size = shard_size
        self.mask_bytes = math.ceil(height * width / 8)
        self._pixels: dict[int, np.memmap] = {}
        self._validity: dict[int, np.memmap] = {}
        self.band_written = self._progress_array(
            "progress-band-written.npy", np.bool_, (total, channels), False
        )
        self.invalid_fraction = self._progress_array(
            "progress-invalid-fraction.npy", np.float32, (total, channels), 1.0
        )
        self.geometry_wkb = self._progress_array("progress-geometry.npy", "S256", (total,), b"")
        self.block_id = self._progress_array("progress-block.npy", "S32", (total,), b"")

    def _progress_array(self, name: str, dtype, shape: tuple[int, ...], fill_value):
        path = self.output_dir / name
        if path.exists():
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.shape != shape or array.dtype != np.dtype(dtype):
                raise ValueError(f"Existing staging progress has the wrong contract: {path}")
            return array
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        array[...] = fill_value
        array.flush()
        return array

    def _shape(self, shard_index: int) -> tuple[int, int, int, int]:
        start = shard_index * self.shard_size
        count = min(self.shard_size, self.total - start)
        return count, self.channels, self.height, self.width

    def _open(self, shard_index: int) -> tuple[np.memmap, np.memmap]:
        if shard_index not in self._pixels:
            shape = self._shape(shard_index)
            pixel_path = self.output_dir / f"shard-{shard_index:05d}-pixels.npy"
            validity_path = self.output_dir / f"shard-{shard_index:05d}-validity.npy"
            self._pixels[shard_index] = np.lib.format.open_memmap(
                pixel_path, mode="r+" if pixel_path.exists() else "w+", dtype=np.uint16, shape=shape
            )
            self._validity[shard_index] = np.lib.format.open_memmap(
                validity_path,
                mode="r+" if validity_path.exists() else "w+",
                dtype=np.uint8,
                shape=(shape[0], self.channels, self.mask_bytes),
            )
        return self._pixels[shard_index], self._validity[shard_index]

    def write_band(self, storage_index: int, band_index: int, values: np.ndarray, valid: np.ndarray) -> None:
        if values.dtype != np.uint16:
            if np.issubdtype(values.dtype, np.integer) and values.min() >= 0 and values.max() <= 65535:
                values = values.astype(np.uint16)
            else:
                raise ValueError(f"Cannot preserve raster dtype {values.dtype} as uint16")
        shard_index, row = divmod(storage_index, self.shard_size)
        pixels, validity = self._open(shard_index)
        pixels[row, band_index] = values
        validity[row, band_index] = pack_validity(valid[None])[0]
        self.band_written[storage_index, band_index] = True
        self.invalid_fraction[storage_index, band_index] = 1.0 - float(valid.mean())

    def write_geometry(self, storage_index: int, geometry_wkb: bytes, block_id: str) -> None:
        if len(geometry_wkb) > 256 or len(block_id.encode()) > 32:
            raise ValueError("Geometry progress value exceeds its fixed-width staging field")
        self.geometry_wkb[storage_index] = geometry_wkb
        self.block_id[storage_index] = block_id.encode()

    def close(self) -> None:
        for value in [*self._pixels.values(), *self._validity.values()]:
            value.flush()
        for value in (self.band_written, self.invalid_fraction, self.geometry_wkb, self.block_id):
            value.flush()
        self._pixels.clear()
        self._validity.clear()


class ShardReader:
    def __init__(self, root: str | Path, shard_size: int, height: int, width: int) -> None:
        self.root = Path(root)
        self.shard_size = shard_size
        self.height = height
        self.width = width
        self._pixels: dict[int, np.ndarray] = {}
        self._validity: dict[int, np.ndarray] = {}

    def read(self, storage_index: int) -> tuple[np.ndarray, np.ndarray]:
        shard_index, row = divmod(int(storage_index), self.shard_size)
        if shard_index not in self._pixels:
            self._pixels[shard_index] = np.load(
                self.root / f"shard-{shard_index:05d}-pixels.npy", mmap_mode="r"
            )
            self._validity[shard_index] = np.load(
                self.root / f"shard-{shard_index:05d}-validity.npy", mmap_mode="r"
            )
        pixels = np.asarray(self._pixels[shard_index][row])
        valid = unpack_validity(np.asarray(self._validity[shard_index][row]), self.height, self.width)
        return pixels, valid
