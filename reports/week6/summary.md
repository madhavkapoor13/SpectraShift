# Week 6 implementation summary

Date: 17 September 2026  
Stage: complete label-efficiency curves and ImageNet RGB control  
Status: implemented; Kaggle execution pending

The implementation generates 48 immutable jobs: M0-M4 at 5%, 25%, and 50% labels for seeds 17, 29, and 43, plus M1RGB at 10% for the same seeds. It reuses the Week 5 subsets, learning-rate decisions, optimizer, duration, augmentation, validation, metric, checkpoint, and recovery contracts.

M1RGB uses B04/B03/B02 and the unchanged ImageNet ResNet-18 stem. Its preprocessing limits are computed with exact streaming uint16 histograms over valid U pixels only. Invalid pixels receive the U mean before clipping, scaling to `[0,1]`, and ImageNet normalization. Each run records the percentile, initialization, manifest, normalization, subset, configuration, and source hashes.

The aggregate verifies 48 new artifacts and merges them with the 45 approved Week 5 runs. It writes `controlled_curves.csv`, `paired_differences.csv`, `aulc_summary.csv`, and `week6_run_summary.json`. H1, H2, and the clean H3 diagnostic use normalized trapezoidal AULC over `log10(label count)` and report three-seed mean and sample standard deviation. The 10% M1RGB-minus-M1 stem control is reported separately. No significance tests are used.

The local suite passes 53 tests, including exact schedules, matrix identity, RGB stem preservation, histogram quantiles, preprocessing order, AULC arithmetic, notebook validity, and evaluation-label isolation. Week 7 remains blocked until Kaggle aggregation reports `week6_complete: true` and `week7_approved: true`.
