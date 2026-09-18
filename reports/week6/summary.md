# Week 6 implementation summary

Date: 18 September 2026

Stage: complete label-efficiency curves and ImageNet RGB control

Status: complete; Week 7 approved

The implementation generates 48 immutable jobs: M0-M4 at 5%, 25%, and 50% labels for seeds 17, 29, and 43, plus M1RGB at 10% for the same seeds. It reuses the Week 5 subsets, learning-rate decisions, optimizer, duration, augmentation, validation, metric, checkpoint, and recovery contracts.

M1RGB uses B04/B03/B02 and the unchanged ImageNet ResNet-18 stem. Its preprocessing limits are computed with exact streaming uint16 histograms over valid U pixels only. Invalid pixels receive the U mean before clipping, scaling to `[0,1]`, and ImageNet normalization. Each run records the percentile, initialization, manifest, normalization, subset, configuration, and source hashes.

The aggregate verifies 48 new artifacts and merges them with the 45 approved Week 5 runs. It writes `controlled_curves.csv`, `paired_differences.csv`, `aulc_summary.csv`, and `week6_run_summary.json`. H1, H2, and the clean H3 diagnostic use normalized trapezoidal AULC over `log10(label count)` and report three-seed mean and sample standard deviation. The 10% M1RGB-minus-M1 stem control is reported separately. No significance tests are used.

The local suite passes 53 tests, including exact schedules, matrix identity, RGB stem preservation, histogram quantiles, preprocessing order, AULC arithmetic, notebook validity, and evaluation-label isolation.

All 48 Kaggle runs passed their completion gates in 2.5197 reported GPU-hours. The aggregate contains exactly 90 controlled M0-M4 states and three M1RGB controls, with no evaluation-label access, and reports `week6_complete: true` and `week7_approved: true`.

Mean supported-class V mAP for M0-M4 at the new label fractions was:

| Labels | M0 | M1 | M2 | M3 | M4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5% | 0.4645 | 0.4746 | 0.3672 | 0.3908 | 0.3852 |
| 25% | 0.5602 | 0.5509 | 0.4437 | 0.5400 | 0.5397 |
| 50% | 0.5928 | 0.5879 | 0.4717 | 0.5744 | 0.5800 |

The normalized AULC differences across the complete six-point curves were M3-M0 `-0.0404 ± 0.0034`, M3-M2 `+0.0573 ± 0.0069`, and M4-M3 `-0.0047 ± 0.0072`, where dispersion is the sample standard deviation across seeds. M1RGB reached mean 10%-label mAP `0.4893 ± 0.0060`, trailing M1 by `0.0243 ± 0.0030`.

These source-V results show that multispectral SSL learns a materially stronger representation than RGB SSL, but it does not beat the random multispectral full-fine-tuning baseline across the label-efficiency curve. Spectral dropout provides no average AULC benefit over standard multispectral SSL. The ImageNet RGB control also trails the ten-band ImageNet adaptation. These are development findings; I, Finland, and Portugal remain sealed.

Canonical artifact hashes:

- `week6_run_summary.json`: `047d1a05470a55a16d156bbb60f35406cf5851e74482e98242169a4342976bd9`
- `controlled_curves.csv`: `f26d63a84faf26c7fbbf2ecd5055fe5db83c082e4e1a12d3e4e89c91b80d5beb`
- `paired_differences.csv`: `240ded12281caeaf9610b2c5311b7557cd69a6ce2a28f65665d9233110f64c52`
- `aulc_summary.csv`: `f5f18ca7dfa9b9f7d6eb71ead29089c383c5b9c5e100c0b6a6231d527382d977`
