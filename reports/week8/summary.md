# Week 8 execution summary

Date: 18 September 2026

Stage: frozen I/Finland/Portugal evaluation

Status: complete; Week 9 approved

The pre-label Week 7 state was frozen at commit `e9af130134d7b323a5f4315338d462d01cb136e7` and annotated tag `week7-complete`. The sealing job verified the exact Week 7 summary and 111-checkpoint ledger before label access, selected the final 11,000 evaluation IDs from the 12,500-row candidate table, and froze 3,000 I, 4,000 Finland, and 4,000 Portugal examples. All 16 source-supported classes have positive and negative support in every evaluation domain.

Three immutable GPU jobs evaluated 37 checkpoints per seed. The aggregate verified 111 unique frozen checkpoints and 333 domain prediction artifacts. No checkpoint selection, parameter update, or threshold fitting occurred after label access. All predictions used the original source-V global threshold and preserved the frozen preprocessing contracts.

Mean supported-class mAP across three seeds was:

| Model | Fraction | I | Finland | Portugal | Equal-country OOD |
|---|---:|---:|---:|---:|---:|
| M0 | 1% | 0.3476 | 0.2636 | 0.2639 | 0.2638 |
| M0 | 10% | 0.4654 | 0.3317 | 0.3050 | 0.3183 |
| M0 | 100% | 0.5870 | 0.3699 | 0.3528 | 0.3613 |
| M1 | 1% | 0.3745 | 0.2816 | 0.3098 | 0.2957 |
| M1 | 10% | 0.4843 | 0.3463 | 0.3459 | 0.3461 |
| M1 | 100% | 0.5814 | 0.3883 | 0.3719 | 0.3801 |
| M2 | 1% | 0.2910 | 0.1950 | 0.2361 | 0.2155 |
| M2 | 10% | 0.3845 | 0.2591 | 0.2768 | 0.2680 |
| M2 | 100% | 0.5130 | 0.3065 | 0.2896 | 0.2981 |
| M3 | 1% | 0.2924 | 0.2324 | 0.2497 | 0.2411 |
| M3 | 10% | 0.4156 | 0.2908 | 0.2885 | 0.2897 |
| M3 | 100% | 0.5705 | 0.3690 | 0.3487 | 0.3589 |
| M4 | 1% | 0.3036 | 0.2276 | 0.2513 | 0.2394 |
| M4 | 10% | 0.4176 | 0.2782 | 0.2892 | 0.2837 |
| M4 | 100% | 0.5629 | 0.3654 | 0.3457 | 0.3555 |
| M5 DINOv2 | 1% | 0.3705 | 0.2952 | 0.2913 | 0.2932 |
| M5 DINOv2 | 10% | 0.4941 | 0.3221 | 0.3291 | 0.3256 |
| M5 DINOv2 | 100% | 0.6242 | 0.3806 | 0.3750 | 0.3778 |
| M6 OlmoEarth | 1% | 0.4557 | 0.3600 | 0.3638 | 0.3619 |
| M6 OlmoEarth | 10% | 0.5912 | 0.4325 | 0.4169 | 0.4247 |
| M6 OlmoEarth | 100% | 0.6699 | 0.4553 | 0.4456 | 0.4505 |

OlmoEarth ranked first at every fraction it was evaluated on and in every domain. DINOv2 was strongest among the remaining models at 100% labels. The controlled-curve AULC results show that M3 remained below random multispectral M0 on I and OOD, but exceeded RGB SSL M2 consistently. M4 was effectively tied with M3 on I and slightly lower on the equal-country OOD average, so spectral dropout did not provide a clean transfer improvement.

The paired uncertainty table contains 1,000 primary MGRS bootstrap replicates and 1,000 12 km-block sensitivity replicates for every H1-H3 fraction/domain comparison. Finland and Portugal were resampled independently before equal weighting. No p-values were calculated.

The aggregate reports `week8_complete: true`, `week9_approved: true`, and `week9_implemented: false`. Compact verified artifacts are tracked under `generated/`; sealed labels, logits, and checkpoints remain private and ignored.
