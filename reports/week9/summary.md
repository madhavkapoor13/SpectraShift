# Week 9 execution summary

Date: 19 September 2026

Stage: representation analysis, spectral stress tests, and error inspection

Status: complete; Week 10 approved

The final aggregate verified 108 M1-M6 linear probes, 18 10%-label k-NN probes, six frozen M3/M4 diagnostic checkpoints, 108 clean/stressed model-seed-domain states, 1,000 fixed CKA patches, and 3,000 nearest-neighbour rows. Encoders remained byte-identical, and no model selection occurred after Week 8.

Frozen linear-probe AULC differences were M3−M1 `−0.0362 ± 0.0054`, M3−M2 `+0.0279 ± 0.0263`, and M4−M3 `+0.0012 ± 0.0154`. Matching-seed M3/M4 CKA was `0.8276`, `0.8946`, and `0.8896`; effective-rank effects were not consistent across seeds.

On clean 10%-label equal-country OOD, M4−M3 was `−0.0060`. Under missing B08, red-edge, and SWIR inputs, the differences became `+0.0073`, `+0.0334`, and `+0.0404`. The primary MGRS bootstrap intervals for red-edge and SWIR removal were `[0.0208, 0.0478]` and `[0.0231, 0.0548]`. These are descriptive intervals, not significance tests.

Nearest-neighbour label Jaccard and geographic distance were nearly identical for M3 and M4. Those results and the fixed retrieval sheet are qualitative support only.

The aggregate reports `week9_complete: true`, `week10_approved: true`, `model_selection_after_week8: false`, and `encoder_updates_during_week9: false`.
