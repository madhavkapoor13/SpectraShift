from __future__ import annotations

import hashlib
import gzip
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import zstandard
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely import to_wkb
from shapely.geometry import box

import spectrashift.data.archive as archive_module
from spectrashift.data.archive import stage_archive
from spectrashift.data.bands import CANONICAL_BANDS, adapt_bands
from spectrashift.data.freeze import freeze_split
from spectrashift.data.geometry import GeometryConfig, footprint_from_dataset, keep_outside_buffer, spatial_block_id
from spectrashift.data.normalization import compute_normalization
from spectrashift.data.raster import pack_validity, read_resampled_band, unpack_validity
from spectrashift.data.seal import seal_evaluation_labels
from spectrashift.data.shards import CandidateShardWriter, ShardReader


def geotiff_bytes(value: int, size: int = 12) -> bytes:
    array = np.full((size, size), value, dtype=np.uint16)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=size,
            width=size,
            count=1,
            dtype="uint16",
            crs="EPSG:32632",
            transform=from_origin(500000, 5500000, 10, 10),
            nodata=0,
        ) as dataset:
            dataset.write(array, 1)
        return memory.read()


def test_band_adapters_preserve_declared_order() -> None:
    array = np.arange(12)[:, None, None] * np.ones((12, 2, 2), dtype=np.int16)
    assert adapt_bands(array, "rgb")[:, 0, 0].tolist() == [2, 1, 0]
    assert adapt_bands(array, "core10")[:, 0, 0].tolist() == list(range(10))


def test_resampling_geometry_and_validity_round_trip() -> None:
    with MemoryFile(geotiff_bytes(100)) as memory:
        with memory.open() as dataset:
            result = read_resampled_band(dataset, 120, 120)
            geometry = footprint_from_dataset(dataset)
    assert result.values.shape == (120, 120)
    assert result.valid.all()
    assert np.all(result.values == 100)
    packed = pack_validity(np.stack([result.valid, result.valid]))
    assert np.array_equal(unpack_validity(packed, 120, 120), np.stack([result.valid, result.valid]))
    assert geometry.area > 0
    assert spatial_block_id(geometry, GeometryConfig()).startswith("b")


def test_buffer_filter() -> None:
    exclusions = [box(0, 0, 100, 100)]
    candidates = [box(2501, 0, 2601, 100), box(1000, 0, 1100, 100)]
    assert keep_outside_buffer(candidates, exclusions, 2400) == [True, False]


def test_shard_round_trip(tmp_path: Path) -> None:
    writer = CandidateShardWriter(tmp_path, total=2, channels=12, height=4, width=4, shard_size=1)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    for band in range(12):
        writer.write_band(1, band, np.full((4, 4), band, dtype=np.uint16), valid)
    writer.close()
    pixels, restored = ShardReader(tmp_path, 1, 4, 4).read(1)
    assert pixels[:, 1, 1].tolist() == list(range(12))
    assert not restored[:, 0, 0].any()
    assert restored[:, 1:, 1:].all()


def test_streaming_archive_stage(tmp_path: Path) -> None:
    patch_id = "S2B_MSIL2A_20180421T100029_N9999_R122_T33TWM_00_07"
    uncompressed = io.BytesIO()
    with tarfile.open(fileobj=uncompressed, mode="w") as archive:
        for index, band in enumerate(CANONICAL_BANDS, start=1):
            payload = geotiff_bytes(index)
            info = tarfile.TarInfo(f"BigEarthNet-S2/T33TWM/{patch_id}/{patch_id}_{band}.tif")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    archive_path = tmp_path / "sample.tar.zst"
    archive_path.write_bytes(zstandard.ZstdCompressor().compress(uncompressed.getvalue()))
    candidate_path = tmp_path / "candidates.parquet"
    pd.DataFrame(
        [{
            "partition": "U", "candidate_rank": 0, "is_primary": True,
            "patch_id": patch_id, "country": "Austria", "mgrs_tile": "T33TWM",
            "location_key": "T33TWM_0_7", "timestamp": pd.Timestamp("2018-04-21", tz="UTC"),
            "orbit": "122", "h_order": 0, "v_order": 7, "labels": None,
        }]
    ).to_parquet(candidate_path, index=False)
    output = tmp_path / "staged"
    config = {
        "dataset": {
            "archive_path": str(archive_path),
            "archive_md5": hashlib.md5(archive_path.read_bytes()).hexdigest(),
            "candidate_manifest": str(candidate_path),
            "canonical_bands": list(CANONICAL_BANDS), "core_bands": list(CANONICAL_BANDS[:10]),
            "target_height": 120, "target_width": 120,
        },
        "geometry": {
            "metric_crs": "EPSG:3035", "exclusion_buffer_m": 2400,
            "block_size_m": 12000, "block_origin_x_m": 0, "block_origin_y_m": 0,
        },
        "staging": {"output_dir": str(output), "shard_size": 1, "maximum_output_bytes": 10_000_000},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    summary = stage_archive(config_path)
    assert summary["complete_patches"] == 1
    resumed = stage_archive(config_path)
    assert resumed["selected_archive_members"] == 0
    assert resumed["resumed_archive_members"] == 12
    staged = pd.read_parquet(output / "candidates_staged.parquet")
    assert staged.loc[0, "band_complete"]
    assert staged.loc[0, "block_12km"].startswith("b")

    compressed_gzip = gzip.compress(uncompressed.getvalue())
    split_at = len(compressed_gzip) // 2
    parts = [tmp_path / "sample.tar.gzaa", tmp_path / "sample.tar.gzab"]
    parts[0].write_bytes(compressed_gzip[:split_at])
    parts[1].write_bytes(compressed_gzip[split_at:])
    config["dataset"].pop("archive_path")
    config["dataset"].pop("archive_md5")
    config["dataset"]["archive_parts"] = [str(path) for path in parts]
    config["dataset"]["archive_part_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in parts
    }
    config["staging"]["output_dir"] = str(tmp_path / "staged-split-gzip")
    config_path.write_text(yaml.safe_dump(config))
    mirror_summary = stage_archive(config_path)
    assert mirror_summary["archive_format"] == "split-tar-gzip"
    assert mirror_summary["complete_patches"] == 1


def test_remote_multipart_reader_resumes_premature_eof(monkeypatch) -> None:
    payloads = {"https://example.test/a": b"abcdefgh", "https://example.test/b": b"ijkl"}
    initial_a_opened = False

    class Response(io.BytesIO):
        def __init__(self, value: bytes, status: int) -> None:
            super().__init__(value)
            self.status = status

    def fake_urlopen(request, timeout):
        nonlocal initial_a_opened
        assert timeout == 120
        source = request.full_url
        range_header = request.get_header("Range")
        if source.endswith("/a") and range_header is None and not initial_a_opened:
            initial_a_opened = True
            return Response(payloads[source][:4], 200)
        if range_header:
            offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
            return Response(payloads[source][offset:], 206)
        return Response(payloads[source], 200)

    monkeypatch.setattr(archive_module, "urlopen", fake_urlopen)
    digests = [hashlib.sha256(), hashlib.sha256()]
    counts = [0, 0]
    with archive_module._MultiPartReader(
        list(payloads), digests, counts, [len(value) for value in payloads.values()]
    ) as reader:
        assert reader.read() == b"abcdefghijkl"
        assert reader.reconnects == [1, 0]
    assert counts == [8, 4]
    assert [digest.hexdigest() for digest in digests] == [
        hashlib.sha256(value).hexdigest() for value in payloads.values()
    ]


def test_freeze_split_seals_evaluation_labels(tmp_path: Path) -> None:
    positions = {"T-FI": 0, "T-PT": 10000, "I": 20000, "V": 30000, "U": 40000, "D": 50000}
    rows = []
    sealed = []
    for partition, x in positions.items():
        count = 2 if partition in {"U", "D"} else 1
        for rank in range(count):
            patch_id = f"{partition}-{rank}"
            rows.append({
                "partition": partition, "candidate_rank": rank, "patch_id": patch_id,
                "country": "Finland" if partition == "T-FI" else "Portugal" if partition == "T-PT" else "Austria",
                "mgrs_tile": f"T00A{rank:02d}", "location_key": patch_id,
                "labels": ["Arable land"] if partition in {"D", "V"} else None,
                "band_complete": True, "invalid_fraction_max_core": 0.0,
                "geometry_wkb_3035": to_wkb(box(x + rank * 1000, 0, x + rank * 1000 + 100, 100)),
                "block_12km": f"b{x // 12000}_{rank}", "storage_index": len(rows),
                "array_shard": 0, "array_row": len(rows),
            })
            if partition in {"I", "T-FI", "T-PT"}:
                sealed.append({"partition": partition, "patch_id": patch_id, "labels": ["Arable land"]})
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    pd.DataFrame(rows).to_parquet(staged_dir / "candidates_staged.parquet", index=False)
    sealed_path = tmp_path / "sealed.parquet"
    pd.DataFrame(sealed).to_parquet(sealed_path, index=False)
    config = {
        "dataset": {"sealed_candidate_labels": str(sealed_path)},
        "quality": {"maximum_invalid_fraction": 0.01},
        "geometry": {"exclusion_buffer_m": 2400, "minimum_evaluation_blocks": 1},
        "split": {
            "name": "test-v1", "caps": {"U": 2, "D": 2, "V": 1, "I": 1, "T-FI": 1, "T-PT": 1},
            "class_min_positives": 1, "class_min_negatives": 0, "minimum_supported_classes": 1,
        },
        "staging": {
            "output_dir": str(staged_dir), "final_manifest_dir": str(tmp_path / "manifest"),
            "sealed_output_dir": str(tmp_path / "sealed-output"), "report_dir": str(tmp_path / "report"),
        },
    }
    config_path = tmp_path / "freeze.yaml"
    config_path.write_text(yaml.safe_dump(config))
    summary = freeze_split(config_path)
    assert summary["training_approved"]
    sealing = seal_evaluation_labels(config_path)
    assert sealing["evaluation_rows"] == 3
    visible = pd.read_parquet(tmp_path / "manifest" / "partitions.parquet")
    assert visible.loc[visible["partition"].isin(["U", "I", "T-FI", "T-PT"]), "labels"].isna().all()
    assert visible.loc[visible["partition"].isin(["D", "V"]), "labels"].notna().all()
    sealed_final = pd.read_parquet(tmp_path / "sealed-output" / "evaluation_labels.parquet")
    assert set(sealed_final["partition"]) == {"I", "T-FI", "T-PT"}


def test_normalization_uses_only_u_and_declared_radiometry(tmp_path: Path) -> None:
    staged = tmp_path / "staged-normalization"
    writer = CandidateShardWriter(staged, total=2, channels=12, height=2, width=2, shard_size=2)
    valid = np.ones((2, 2), dtype=bool)
    for band in range(12):
        writer.write_band(0, band, np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16), valid)
        writer.write_band(1, band, np.full((2, 2), 60000, dtype=np.uint16), valid)
    writer.close()
    (staged / "staging_summary.json").write_text(json.dumps({
        "radiometry": {band: [[1.0, 0.0, "0.0"]] for band in CANONICAL_BANDS}
    }))
    manifest_dir = tmp_path / "manifest-normalization"
    manifest_dir.mkdir()
    pd.DataFrame([
        {"partition": "U", "patch_id": "u", "storage_index": 0},
        {"partition": "D", "patch_id": "d", "storage_index": 1},
    ]).to_parquet(manifest_dir / "partitions.parquet", index=False)
    config = {
        "dataset": {"target_height": 2, "target_width": 2},
        "staging": {"output_dir": str(staged), "final_manifest_dir": str(manifest_dir), "shard_size": 2},
    }
    config_path = tmp_path / "normalization.yaml"
    config_path.write_text(yaml.safe_dump(config))
    stats = compute_normalization(config_path)
    assert np.allclose(stats["mean"], 0.25)
    assert np.allclose(stats["std"], np.std([0.1, 0.2, 0.3, 0.4]))
    assert set(stats["scale_source"]) == {"BigEarthNet integer reflectance fallback"}
