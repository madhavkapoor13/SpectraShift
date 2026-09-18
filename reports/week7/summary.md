# Week 7 implementation summary

Date: 18 September 2026

Stage: DINOv2 and OlmoEarth foundation baselines

Status: implementation complete; Kaggle execution pending

M5 is the declared public DINOv2 ViT-S/14 fallback. It consumes B04/B03/B02 through the frozen U-only Week 6 percentile mapping, resizes the complete field to 126 pixels, applies ImageNet normalization, and averages the 81 normalized patch tokens. It is identified as DINOv2 in every run and report.

M6 uses OlmoEarth v1.1 Tiny through the official minimal loader. The implementation preserves the official 12-band Sentinel-2 order, raw L2A DN units, official normalization statistics, real acquisition dates with zero-indexed months, patch size 8, 10 m input resolution, and validity-aware token pooling. Local validation loaded the verified Week 2 checkpoint, produced finite 192-dimensional features, and confirmed 12,461,296 encoder parameters.

The pipeline defines six bounded LR pilots, 18 full fine-tuning anchors, two D/V feature caches, 36 linear probes, and six 10% k-NN probes. All notebooks are output-free and below 1 MB. The local suite passes 63 tests covering the adapters, preprocessing, exact matrices and step schedules, optimizer groups, pilot selection, batch fallback, notebook validity, and evaluation-label isolation.

I, Finland, and Portugal remain sealed. Week 8 stays blocked until `18_week7_aggregate.ipynb` verifies every artifact and writes `week7_complete: true` and `week8_approved: true`.
