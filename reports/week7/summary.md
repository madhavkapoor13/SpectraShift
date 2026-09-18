# Week 7 execution summary

Date: 18 September 2026

Stage: DINOv2 and OlmoEarth foundation baselines

Status: complete; Week 8 approved

M5 is the declared public DINOv2 ViT-S/14 fallback. It consumes B04/B03/B02 through the frozen U-only Week 6 percentile mapping, resizes the complete field to 126 pixels, applies ImageNet normalization, and averages the 81 normalized patch tokens. It is identified as DINOv2 throughout the artifacts.

M6 uses OlmoEarth v1.1 Tiny through the official minimal loader. The implementation preserves the official 12-band Sentinel-2 order, raw L2A DN units, official normalization statistics, real acquisition dates with zero-indexed months, patch size 8, 10 m input resolution, and validity-aware token pooling.

The six learning-rate pilots selected multiplier 1.0 for M5 and 3.0 for M6 at physical batch 32. The full matrix completed 18 fine-tuning runs across 1%, 10%, and 100% labels and seeds 17, 29, and 43. It also produced two verified D/V feature caches, 36 linear probes, and six 10% k-NN probes. A float32 boundary issue in k-NN probability aggregation was fixed by clipping the convex combinations to the declared probability interval before calibration; the regression fix is commit `c67c20b8dbf2e92a2d3fba185676fd13b34f064a`, and the full local suite passes 64 tests.

Mean supported-class validation mAP across three seeds was:

| Model | 1% | 10% | 100% |
|---|---:|---:|---:|
| M5 DINOv2 | 0.3910 | 0.5121 | 0.6275 |
| M6 OlmoEarth | 0.4766 | 0.6006 | 0.6736 |

OlmoEarth exceeded M0, M1, and M3 at every matched fraction. Its paired mean advantage over M0 was 0.1068 at 1%, 0.1009 at 10%, and 0.0625 at 100%. DINOv2 exceeded M0 and M3 at all three fractions; relative to M1 it was lower at 1%, effectively tied at 10%, and higher at 100%.

The aggregate contains 18 unique foundation checkpoints and a 111-state Week 8 ledger: 90 controlled M0-M4 states, three M1RGB controls, and 18 M5-M6 states. It reports `week7_complete: true`, `week8_approved: true`, and `evaluation_labels_loaded: false`. I, Finland, and Portugal remain sealed.

Compact verified artifacts are tracked under `generated/`:

- `week7_run_summary.json`
- `foundation_comparisons.csv`
- `week8_checkpoint_ledger.csv`
