# Week 1 audit

Generated reports are ignored because they are reproducible from the release metadata. The audit produces:

- `audit_summary.json`
- `partition_summary.csv`
- `country_tile_counts.csv`
- `acquisition_coverage.csv`
- `partition_country_counts.csv`
- `source_tile_assignments.csv`
- `source_label_support.csv`
- `coverage_map.svg` and `coverage_map.png`

The tables describe source-label support and geographic coverage only. They intentionally omit target-label counts.

The metadata audit found that Portugal has three MGRS tiles. See `protocol-amendment-001.md` for the pre-training resolution.
