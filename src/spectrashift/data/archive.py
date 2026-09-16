from __future__ import annotations

import hashlib
import gzip
import http.client
import io
import json
import re
import tarfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import rasterio
import yaml
import zstandard
from shapely import to_wkb

from .bands import CANONICAL_BANDS
from .geometry import GeometryConfig, footprint_from_dataset, spatial_block_id
from .preflight import projected_staging_bytes
from .raster import read_resampled_band
from .shards import CandidateShardWriter


MEMBER_RE = re.compile(
    r"(?P<patch>S2[AB]_MSIL2A_\d{8}T\d{6}_N\d{4}_R\d{3}_T\d{2}[A-Z]{3}_\d+_\d+)_"
    r"(?P<band>B01|B02|B03|B04|B05|B06|B07|B08|B8A|B09|B11|B12)\.tif$"
)


class _HashingReader(io.RawIOBase):
    def __init__(self, raw, digest) -> None:
        self.raw = raw
        self.digest = digest

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        value = self.raw.read(size)
        self.digest.update(value)
        return value

    def readinto(self, target) -> int:
        value = self.raw.read(len(target))
        target[: len(value)] = value
        self.digest.update(value)
        return len(value)


def _iter_tar_members(path: Path, digest):
    with path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(_HashingReader(compressed, digest)) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    if member.isfile():
                        yield archive, member


def _is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def _source_name(value: str) -> str:
    return Path(urlparse(value).path).name if _is_url(value) else Path(value).name


class _MultiPartReader(io.RawIOBase):
    def __init__(
        self,
        sources: list[str],
        digests: list,
        byte_counts: list[int],
        expected_bytes: list[int | None],
        maximum_reconnects: int = 8,
    ) -> None:
        self.sources = sources
        self.digests = digests
        self.byte_counts = byte_counts
        self.expected_bytes = expected_bytes
        self.maximum_reconnects = maximum_reconnects
        self.reconnects = [0 for _ in sources]
        self.index = -1
        self.handle = None
        self._advance()

    def readable(self) -> bool:
        return True

    def _open_current(self) -> None:
        source = self.sources[self.index]
        offset = self.byte_counts[self.index]
        if _is_url(source):
            headers = {"User-Agent": "SpectraShift/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(source, headers=headers)
            self.handle = urlopen(request, timeout=120)
            status = getattr(self.handle, "status", None)
            if offset and status != 206:
                self.handle.close()
                self.handle = None
                raise OSError(f"Remote source ignored resume range at byte {offset}: HTTP {status}")
        else:
            self.handle = Path(source).open("rb")
            if offset:
                self.handle.seek(offset)

    def _advance(self) -> bool:
        if self.handle is not None:
            self.handle.close()
        self.index += 1
        if self.index >= len(self.sources):
            self.handle = None
        else:
            self._open_current()
        return self.handle is not None

    def _reconnect(self, error: Exception | None = None) -> None:
        if not _is_url(self.sources[self.index]):
            if error is not None:
                raise error
            raise EOFError(f"Local archive part ended early: {self.sources[self.index]}")
        self.reconnects[self.index] += 1
        if self.reconnects[self.index] > self.maximum_reconnects:
            raise OSError(
                f"Remote archive part exceeded {self.maximum_reconnects} reconnects at "
                f"byte {self.byte_counts[self.index]}"
            ) from error
        if self.handle is not None:
            self.handle.close()
        self._open_current()

    def readinto(self, target) -> int:
        total = 0
        while total < len(target) and self.handle is not None:
            try:
                value = self.handle.read(len(target) - total)
            except http.client.IncompleteRead as error:
                value = error.partial
                if not value:
                    self._reconnect(error)
                    continue
            except (TimeoutError, ConnectionError, OSError) as error:
                self._reconnect(error)
                continue
            if not value:
                expected = self.expected_bytes[self.index]
                observed = self.byte_counts[self.index]
                if expected is not None and observed < expected:
                    self._reconnect()
                elif expected is not None and observed > expected:
                    raise ValueError(
                        f"Archive part exceeded expected size: {self.sources[self.index]} "
                        f"({observed} > {expected})"
                    )
                else:
                    self._advance()
                continue
            target[total : total + len(value)] = value
            self.digests[self.index].update(value)
            self.byte_counts[self.index] += len(value)
            total += len(value)
        return total

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        super().close()


def _iter_split_gzip_members(
    sources: list[str],
    digests: list,
    byte_counts: list[int],
    expected_bytes: list[int | None],
):
    with _MultiPartReader(sources, digests, byte_counts, expected_bytes) as joined:
        with gzip.GzipFile(fileobj=joined, mode="rb") as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    if member.isfile():
                        yield archive, member
            # tarfile stops at the logical tar terminator. Drain the gzip stream so
            # transport hashes and byte counts also cover padding and trailers.
            while reader.read(1024 * 1024):
                pass


def stage_archive(config_path: str | Path, output_override: str | Path | None = None) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text())
    dataset = config["dataset"]
    staging = config["staging"]
    archive_parts = [str(value) for value in dataset.get("archive_parts", [])]
    archive_path = Path(dataset["archive_path"]) if not archive_parts else None
    if archive_parts:
        missing = [value for value in archive_parts if not _is_url(value) and not Path(value).exists()]
        if missing:
            raise FileNotFoundError(f"Archive parts not found: {missing}")
    elif archive_path is None or not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    bands = tuple(dataset["canonical_bands"])
    if bands != CANONICAL_BANDS:
        raise ValueError("Configured canonical band order does not match the code contract")
    candidates = pd.read_parquet(dataset["candidate_manifest"]).sort_values(
        ["partition", "candidate_rank"]
    ).reset_index(drop=True)
    if candidates["patch_id"].duplicated().any():
        raise ValueError("Candidate manifest contains duplicate patch IDs")
    candidates["storage_index"] = np.arange(len(candidates), dtype=np.int64)
    index_by_patch = dict(zip(candidates["patch_id"], candidates["storage_index"], strict=True))
    output_dir = Path(output_override) if output_override else Path(staging["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    height = int(dataset["target_height"])
    width = int(dataset["target_width"])
    shard_size = int(staging["shard_size"])
    projected = projected_staging_bytes(len(candidates), len(bands), height, width)
    if projected > int(staging["maximum_output_bytes"]):
        raise RuntimeError(
            f"Projected staging size {projected} exceeds configured cap "
            f"{staging['maximum_output_bytes']}"
        )
    writer = CandidateShardWriter(output_dir, len(candidates), len(bands), height, width, shard_size)
    radiometry_progress = output_dir / "progress-radiometry.json"
    if radiometry_progress.exists():
        stored_radiometry = json.loads(radiometry_progress.read_text())
        observed_radiometry = {
            band: {tuple(value) for value in stored_radiometry.get(band, [])} for band in bands
        }
    else:
        observed_radiometry = {band: set() for band in bands}
    geometry_config = GeometryConfig(
        metric_crs=config["geometry"]["metric_crs"],
        buffer_m=float(config["geometry"]["exclusion_buffer_m"]),
        block_size_m=float(config["geometry"]["block_size_m"]),
        origin_x_m=float(config["geometry"]["block_origin_x_m"]),
        origin_y_m=float(config["geometry"]["block_origin_y_m"]),
    )
    band_lookup = {band: index for index, band in enumerate(bands)}
    selected_members = 0
    resumed_members = 0
    archive_digest = hashlib.md5() if archive_path is not None else None
    part_digests = [hashlib.sha256() for _ in archive_parts]
    part_byte_counts = [0 for _ in archive_parts]
    configured_part_bytes = dataset.get("archive_part_bytes", {})
    expected_part_bytes_list = [
        int(configured_part_bytes[_source_name(source)])
        if _source_name(source) in configured_part_bytes
        else None
        for source in archive_parts
    ]
    members = (
        _iter_split_gzip_members(
            archive_parts,
            part_digests,
            part_byte_counts,
            expected_part_bytes_list,
        )
        if archive_parts
        else _iter_tar_members(archive_path, archive_digest)
    )
    try:
        for archive, member in members:
            match = MEMBER_RE.search(member.name)
            if not match:
                continue
            patch_id = match.group("patch")
            storage_index = index_by_patch.get(patch_id)
            if storage_index is None:
                continue
            band = match.group("band")
            band_index = band_lookup[band]
            geometry_complete = bool(bytes(writer.geometry_wkb[storage_index]).rstrip(b"\x00"))
            if writer.band_written[storage_index, band_index] and (band != "B02" or geometry_complete):
                resumed_members += 1
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read archive member {member.name}")
            payload = extracted.read()
            with rasterio.MemoryFile(payload) as memory_file:
                with memory_file.open() as raster:
                    result = read_resampled_band(raster, height, width)
                    writer.write_band(storage_index, band_index, result.values, result.valid)
                    radiometry = (result.scale, result.offset, str(result.nodata))
                    if radiometry not in observed_radiometry[band]:
                        observed_radiometry[band].add(radiometry)
                        radiometry_progress.write_text(json.dumps({
                            key: sorted(list(values)) for key, values in observed_radiometry.items()
                        }, indent=2) + "\n")
                    if band == "B02":
                        geometry = footprint_from_dataset(raster, geometry_config.metric_crs)
                        writer.write_geometry(
                            storage_index,
                            to_wkb(geometry),
                            spatial_block_id(geometry, geometry_config),
                        )
            selected_members += 1
    finally:
        writer.close()
    complete = writer.band_written.all(axis=1)
    actual_archive_md5 = archive_digest.hexdigest() if archive_digest is not None else None
    if archive_path is not None:
        expected_archive_md5 = dataset.get("archive_md5")
        if expected_archive_md5 and actual_archive_md5 != expected_archive_md5:
            raise ValueError(
                f"BigEarthNet-S2 archive MD5 mismatch: expected {expected_archive_md5}, found {actual_archive_md5}"
            )
    actual_part_sha256 = {
        _source_name(source): digest.hexdigest()
        for source, digest in zip(archive_parts, part_digests, strict=True)
    }
    actual_part_bytes = {
        _source_name(source): count
        for source, count in zip(archive_parts, part_byte_counts, strict=True)
    }
    expected_part_bytes = configured_part_bytes
    size_mismatches = {
        name: {"expected": int(expected_part_bytes[name]), "actual": value}
        for name, value in actual_part_bytes.items()
        if name in expected_part_bytes and value != int(expected_part_bytes[name])
    }
    if size_mismatches:
        raise ValueError(f"BigEarthNet-S2 mirror part size mismatch: {size_mismatches}")
    expected_part_sha256 = dataset.get("archive_part_sha256", {})
    mismatches = {
        name: {"expected": expected_part_sha256[name], "actual": value}
        for name, value in actual_part_sha256.items()
        if name in expected_part_sha256 and value != expected_part_sha256[name]
    }
    if mismatches:
        raise ValueError(f"BigEarthNet-S2 mirror part SHA-256 mismatch: {mismatches}")
    candidates["band_complete"] = complete
    candidates["invalid_fraction_max_core"] = writer.invalid_fraction[:, :10].max(axis=1)
    candidates["geometry_wkb_3035"] = [bytes(value).rstrip(b"\x00") or None for value in writer.geometry_wkb]
    candidates["block_12km"] = [bytes(value).rstrip(b"\x00").decode() or None for value in writer.block_id]
    candidates["array_shard"] = candidates["storage_index"] // shard_size
    candidates["array_row"] = candidates["storage_index"] % shard_size
    candidates.to_parquet(output_dir / "candidates_staged.parquet", index=False)
    output_bytes = sum(path.stat().st_size for path in output_dir.glob("*.npy"))
    if output_bytes > int(staging["maximum_output_bytes"]):
        raise RuntimeError(f"Staged arrays use {output_bytes} bytes, exceeding the configured cap")
    summary = {
        "candidate_patches": len(candidates),
        "archive_format": "split-tar-gzip" if archive_parts else "tar-zstandard",
        "archive_md5": actual_archive_md5,
        "archive_part_sha256": actual_part_sha256,
        "archive_part_bytes": actual_part_bytes,
        "selected_archive_members": selected_members,
        "resumed_archive_members": resumed_members,
        "complete_patches": int(complete.sum()),
        "incomplete_patches": int((~complete).sum()),
        "output_bytes": output_bytes,
        "radiometry": {band: sorted(list(values)) for band, values in observed_radiometry.items()},
    }
    (output_dir / "staging_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not complete.all():
        examples = candidates.loc[~complete, "patch_id"].head(10).tolist()
        raise RuntimeError(f"Archive ended with incomplete candidate patches: {examples}")
    return summary
