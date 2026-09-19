# Application notes

## Résumé bullets

- Built a reproducible multispectral self-supervised learning pipeline on 20,000 unlabeled BigEarthNet v2 patches, trained nine VICReg encoders, and evaluated 111 frozen checkpoints across six label budgets and three final test domains.
- Quantified a spectral-invariance trade-off: ten-band VICReg improved controlled OOD label-efficiency AULC over RGB VICReg by `0.0337 ± 0.0062` mAP, while spectral dropout reduced clean 10%-label OOD mAP by `0.0060` but improved missing-red-edge and missing-SWIR robustness by `0.0334` and `0.0404` mAP.

## 100-word admissions description

SpectraShift is a controlled study of self-supervised multispectral representation learning under label scarcity and geographic shift. I built the full data, training, checkpointing, evaluation, and analysis pipeline for a frozen 45,000-patch BigEarthNet v2 subset. The study compares random, ImageNet, RGB VICReg, multispectral VICReg, spectral-dropout VICReg, DINOv2, and OlmoEarth baselines across three seeds and six label budgets. Final evaluation uses a source-domain test set plus held-out Finland and Portugal. The results are deliberately mixed: multispectral VICReg improves over RGB VICReg but not matched random/ImageNet controls, while spectral dropout trades slightly lower clean transfer for materially better missing-band robustness.

## Interview explanation

The project asks whether multispectral self-supervision creates useful representations and whether withholding spectral groups during pretraining improves transfer. I treated the split, label budgets, checkpoints, and final evaluation as immutable contracts. The strongest result was OlmoEarth, which is expected given its external EO pretraining. My own multispectral VICReg model consistently beat its RGB equivalent, but a random multispectral ResNet remained stronger after full fine-tuning. Spectral dropout did not improve clean accuracy; it did make the model much less sensitive to missing red-edge and SWIR bands. That negative-plus-mechanistic result is the main research contribution: it distinguishes clean predictive performance from robustness to spectral failure.

## Claims to avoid

- Do not describe DINOv2 as DINOv3.
- Do not call spectral dropout a novel algorithm.
- Do not claim a label-saving factor or annotation-cost reduction.
- Do not claim global geographic generalization from Finland and Portugal.
- Do not compare foundation models as if their pretraining exposure were controlled.
