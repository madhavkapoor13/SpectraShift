# Week 1 audit summary

Audit date: 13 September 2026  
Dataset: BigEarthNet v2.0 Sentinel-2 clean metadata  
Draft split: `BENv2-SpectraShift-draft-v0`

## Provenance

- Clean metadata rows: 480,038
- Countries: 10
- MGRS tiles: 54
- Verified metadata MD5: `55687065e77b6d0b0f1ff604a6e7b49c`
- Draft partition manifest SHA-256: `fdf4233d49259423256518d1a92aa834d7fbb9e8bd16d6f8b5757375516b73b9`
- I, Finland, and Portugal labels: mechanically separated into ignored sealed artifacts; no evaluation-label aggregation was run

## Draft partitions

| Partition | Patches | Unique locations | MGRS tiles | Countries |
| --- | ---: | ---: | ---: | ---: |
| U | 20,000 | 20,000 | 17 | 8 |
| D | 12,000 | 12,000 | 16 | 8 |
| V | 2,000 | 2,000 | 5 | 3 |
| I | 3,000 | 3,000 | 8 | 5 |
| T-FI | 4,000 | 4,000 | 21 | 1 |
| T-PT | 4,000 | 4,000 | 3 | 1 |

All six caps are feasible with unique location keys. U and D are location-disjoint. Source pools contain no Finland or Portugal samples and no MGRS tile touching a target country.

## Source-label support

The predeclared source-support rule requires at least 50 positive and 50 negative D samples. Sixteen of the 19 canonical labels pass.

| Unsupported class | D positives | D negatives |
| --- | ---: | ---: |
| Agro-forestry areas | 0 | 12,000 |
| Beaches, dunes, sands | 12 | 11,988 |
| Coastal wetlands | 1 | 11,999 |

These classes remain in the manifest and will be reported individually. They are excluded only from the predeclared aggregate `C_source` metric unless the frozen Week 2 manifest changes their support through geometry-driven removals and resampling. The sampling is not repeated to seek favorable class counts.

## Gate results

- Metadata checksum and schema: pass
- Unique patch IDs and location keys: pass
- Target-country exclusion: pass at MGRS-tile level
- Source class-support gate: pass, 16 supported classes
- Provisional evaluation-group gate: unresolved because Portugal has three MGRS tiles
- Exact footprint and 2.4 km buffer gate: pending Week 2 by design
- Approved for training: no

Protocol amendment 001 fixes the Portugal grouping issue before training by using label-independent 12 km EPSG:3035 blocks for target uncertainty while retaining the complete country holdout.
