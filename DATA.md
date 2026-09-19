# Data contract

## Primary release

- Dataset: BigEarthNet v2.0, Sentinel-2 L2A
- DOI: `10.5281/zenodo.10891137`
- Official metadata MD5: `55687065e77b6d0b0f1ff604a6e7b49c`
- Official Sentinel-2 archive MD5: `2245ed2d1a93f6ce637d839bc856396e`
- License: `CDLA-Permissive-1.0`
- Clean metadata: patches excluding seasonal snow, cloud, and cloud shadow flags
- Official archive: `BigEarthNet-S2.tar.zst`

When Zenodo's large-file endpoint is unavailable, the Kaggle launcher accepts TorchGeo's byte-split `tar.gz` mirror at repository revision `3cf3a5910a5302d449fdb8e570e5b78de24fe07f`. The two parts are streamed as one archive without creating a second 63 GB copy. If a long-lived download is interrupted, staging reopens the revision-pinned URL at the exact next byte. It enforces the published byte lengths and SHA-256 values, successful gzip/tar decoding, readable rasters, and complete bands for every selected patch.

The repository records metadata and derived patch IDs. It does not redistribute the image archive.

## Raster contract

The stored order is `B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09`. Every patch is resampled to `120 × 120`; reflectance uses bilinear resampling and validity masks use nearest-neighbor resampling. Values remain `uint16` in sharded arrays and masks are bit-packed. Invalid values are replaced by the U mean only when loading samples.

Adapters expose RGB (3 bands), core multispectral (10 bands), or all Sentinel-2 bands (12 bands). Per-band normalization is fitted from valid U pixels only. The pipeline records GeoTIFF scale/offset metadata and uses the declared BigEarthNet `1e-4` integer-reflectance conversion only when the files carry identity metadata and the raw range confirms encoded reflectance.

## Partitions

| Partition | Candidate cap | Final cap | Label access during development |
| --- | ---: | ---: | --- |
| U | 22,000 | 20,000 | hidden |
| D | 13,200 | 12,000 | visible |
| V | 2,500 | 2,000 | visible |
| I | 3,500 | 3,000 | sealed |
| T-FI | 4,500 | 4,000 | sealed |
| T-PT | 4,500 | 4,000 | sealed |

The reserve candidates absorb invalid rasters and footprint-buffer removals without resampling for favorable labels. A location key combines MGRS tile and patch row/column, so repeated acquisitions cannot cross partitions. All tiles containing Finland or Portugal samples are excluded from U, D, V, and I.

## Label boundary

Public manifests expose labels only for D and V. Week 1 writes I, Finland, and Portugal candidate labels mechanically to `data/sealed/evaluation_candidate_labels.parquet`, which is ignored by Git. Raster staging and final split freezing never load that file. After model selection, an offline evaluator runs `python -m scripts.seal_evaluation --config configs/data/week2.yaml` to join final evaluation IDs to their sealed labels.

## Week 2 gates

Raster bounds are projected to EPSG:3035. Final U and D footprints must remain at least 2.4 km from V, I, Finland, and Portugal evaluation footprints. V must remain at least 2.4 km from I and both target domains; I must remain at least 2.4 km from both target domains. No core band may exceed 1% invalid pixels. Evaluation uncertainty groups use fixed 12 km EPSG:3035 blocks with origin `(0, 0)`.

The final split is approved only when all caps are met, IDs and location keys are unique, every evaluation partition spans at least five blocks, and at least 15 source classes meet the declared D support rule.

## Final evaluation state

Week 8 opened the sealed labels only after the 111-checkpoint ledger was frozen. The final one-to-one join contains exactly 3,000 I, 4,000 Finland, and 4,000 Portugal rows. Every evaluation prediction retains patch IDs but never duplicates labels. The aggregate verified 333 domain prediction files, frozen source-V thresholds, unchanged checkpoint hashes, and no parameter updates after label access.

Sixteen source-supported classes have positive and negative support in every final domain. All 19 classes remain in `per_class_metrics.csv`; unsupported slice metrics use NA rather than silently changing the reporting set. Public Git excludes the sealed labels, candidate-label table, logits, staged rasters, patch features, and patch-level diagnostics.

The public data evidence is limited to manifests without evaluation labels, aggregate metrics, counts, hashes, and figures. See `reports/week8/generated/support_contract.json` and `reports/final/generated/results_manifest.json` for the final evidence chain.
