# Draft manifests

Run `python -m scripts.audit_data --config configs/data/week1.yaml` to reproduce `partitions.parquet` and the larger `candidates.parquet` reserve pool from the verified BigEarthNet v2 metadata.

The Parquet artifacts are ignored because they are generated and still lack exact footprint-buffer validation. They must not be renamed or copied into a training configuration. Week 2 produces a versioned manifest after raster geometry checks.
