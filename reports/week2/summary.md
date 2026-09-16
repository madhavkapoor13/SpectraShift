# Week 2 implementation summary

Date: 14 September 2026  
Stage: raster pipeline and validation infrastructure  
Final split target: `BENv2-SpectraShift-v1`

## Outcome

The Week 2 pipeline and real-data Kaggle run are complete. The run streamed and verified the BigEarthNet v2 Sentinel-2 mirror, staged all 50,200 candidates, derived raster footprints, enforced geographic separation, froze the versioned manifest, computed U-only normalization statistics, measured provisional T4 throughput, and passed the real tiny-overfit gate.

The original Kaggle completion artifacts are stored under `reports/week2/generated/`; their canonical summary is `week2_run_summary.json` with SHA-256 `c85a648725429284f2be5ce385cb48aca6b108c5f4ade0a8cfcb4be6c393cc8a`. A concise run record is stored in `reports/week2/kaggle_run_evidence.json`. The 18.46 GB staged arrays and frozen manifest remain in the private Kaggle dataset `spectrashift-week2-frozen`.

## Implemented data pipeline

- Fixed 12-band storage order: `B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09`.
- RGB, core-10, and all-12 band adapters.
- Bilinear reflectance resampling and nearest-neighbor validity-mask resampling to `120 × 120`.
- Sharded `uint16` pixel arrays and bit-packed masks.
- Persistent band, quality, geometry, block, and radiometry progress for resumable staging.
- Single archive pass that simultaneously computes the official MD5 and processes selected GeoTIFF members.
- Early storage-cap preflight before array allocation.
- EPSG:3035 footprint projection and fixed 12 km centroid blocks.
- Maximum 1% invalid-pixel threshold across the ten core bands.

## Candidate and split contract

The regenerated candidate manifest contains 50,200 unique patches and unique location keys.

| Partition | Candidates | Final target | Development labels |
| --- | ---: | ---: | --- |
| U | 22,000 | 20,000 | hidden |
| D | 13,200 | 12,000 | visible |
| V | 2,500 | 2,000 | visible |
| I | 3,500 | 3,000 | sealed |
| T-FI | 4,500 | 4,000 | sealed |
| T-PT | 4,500 | 4,000 | sealed |

The reserve rows absorb invalid rasters and geometry removals without repeating label-driven sampling. The draft public partition manifest SHA-256 is:

`fdf4233d49259423256518d1a92aa834d7fbb9e8bd16d6f8b5757375516b73b9`

Final selection applies these deterministic footprint constraints:

- U and D remain at least 2.4 km from V, I, Finland, and Portugal.
- V remains at least 2.4 km from I, Finland, and Portugal.
- I remains at least 2.4 km from Finland and Portugal.
- Every evaluation partition must span at least five fixed 12 km blocks.
- Final patch IDs and repeated-acquisition location keys must remain unique.

## Label isolation

Only D and V labels appear in public development manifests. U, I, Finland, and Portugal labels are absent. I and target candidate labels are stored mechanically in ignored sealed storage and are never loaded by raster staging, split freezing, normalization, throughput measurement, or smoke testing.

Final evaluation labels can be materialized only in a separate offline environment after model selection by running:

```bash
.venv/bin/python -m scripts.seal_evaluation --config configs/data/week2.yaml
```

The obsolete target-only and draft-primary sealed files were moved into the ignored legacy directory to prevent confusion with the eventual frozen v1 artifact.

## Statistics, metrics, and model checks

- Normalization uses valid U pixels only.
- GeoTIFF scale and offset metadata are checked for consistency per band.
- The `1e-4` BigEarthNet integer-reflectance conversion is applied only when identity metadata and the observed raw range support it.
- Evaluation utilities implement per-class and macro average precision, micro/macro F1, per-class recall, expected calibration error, and source-V-only threshold fitting.
- ResNet-18 input contracts were verified for 3, 10, and 12 channels with 19 outputs.
- The real-data smoke runner validates label isolation and partition counts before attempting to overfit 32 D patches to micro-F1 of at least 0.95.
- The throughput command performs 50 warm-up and 200 timed two-view steps by default, records hardware and peak CUDA memory, and estimates the cost of nine 60-epoch SSL runs. It is explicitly provisional until Week 3 replaces the proxy loss with VICReg and the final view generator.

## External checkpoint status

OlmoEarth v1.1 Tiny was downloaded from Hugging Face at revision `74fab5714f763d6b94f8b1536bdd3300d77f45e8`. Its configuration and 107 MB weight file load successfully with PyTorch's weights-only loader and match the recorded hashes:

- Config SHA-256: `01dcb438144d8f70647ab2d11aef656a1632f3b5af1fdf9263c111127ad7bbc3`
- Weights SHA-256: `2a3fe8132adf9ff2ca96d00c9e376b8925bfe430fda6140749b3b92764c67ae1`

DINOv3 ViT-S/16 remains access-gated. Its weights can be registered only after the applicant accepts Meta's license and provides the issued private download URL. DINOv2 ViT-S/14 remains the declared fallback if access is not obtained before baseline integration.

## Verification evidence

- Test suite: 20 passed.
- Python compilation: passed for `src/`, `scripts/`, and `tests/`.
- Kaggle notebook JSON validation: passed.
- Synthetic archive staging and checksum validation: passed.
- Interrupted-stage resume behavior: passed; all 12 completed members were skipped on the second pass.
- Synthetic footprint filtering, split freezing, and offline label sealing: passed.
- U-only normalization isolation: passed.
- Synthetic throughput execution: passed on CPU.
- OlmoEarth checkpoint hashes: passed.

The 13 test warnings originate from Rasterio's internal use of the older affine multiplication operator in the synthetic GeoTIFF helper; they do not indicate failed scientific or data invariants.

## Storage and execution gate

The full 50,200-candidate cache was projected to occupy 18,450,909,600 bytes and used 18,450,935,456 bytes in the Kaggle run. The cache remains in Kaggle rather than local storage.

The launcher `notebooks/kaggle/02_stage_bigearthnet_v2.ipynb` streamed both revision-pinned TorchGeo mirror parts and resumed an interrupted transfer by byte range. Both published SHA-256 values and byte lengths matched. Because Kaggle's current PyTorch build does not support the P100's `sm_60` architecture, `notebooks/kaggle/03_complete_week2_from_staged.ipynb` completed the GPU checks against the preserved output on a Tesla T4 without repeating staging.

## Week 2 exit status

| Exit condition | Status |
| --- | --- |
| Raster pipeline and band adapters implemented | pass |
| Geometry, buffer, quality, and block logic implemented | pass |
| Metrics and model smoke infrastructure implemented | pass |
| Synthetic integration suite | pass |
| OlmoEarth checkpoint acquired and verified | pass |
| DINOv3 checkpoint access | pending applicant license acceptance |
| Mirror archive checksums observed during staging | pass |
| Final footprint-audited manifest frozen | pass |
| Real U normalization statistics | pass |
| Real GPU throughput forecast | pass |
| Real tiny-overfit gate | pass: micro-F1 0.9596 in 6 steps |
| Approved to begin Week 3 training | yes |

Week 3 model development is approved against frozen manifest `BENv2-SpectraShift-v1` with SHA-256 `0b36a0c6c55f34a8963719af725096dde5ab1dbab72968c13639e31d1099000a`. The provisional T4 benchmark forecasts about 1.10 hours per 60-epoch run and 9.93 hours for nine runs; Week 3 must replace this proxy with the implemented VICReg workload before scheduling the full sweep.
