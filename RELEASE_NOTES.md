# SpectraShift v1.0.0

SpectraShift is a completed multispectral representation-learning study built on
BigEarthNet v2. This release contains the authored code, frozen experiment
contracts, compact aggregate evidence, deterministic figures, and an
admissions-focused technical report.

## Headline findings

- Ten-band VICReg (M3) outperformed RGB VICReg (M2) on equal-country OOD mAP at
  10% labels by approximately **0.0217**, but did not beat the random
  multispectral baseline M0 or ImageNet-initialized M1.
- OlmoEarth was the strongest evaluated model across source-I, Finland, and
  Portugal at the anchor label fractions.
- Spectral dropout (M4) reduced clean OOD mAP by approximately **0.0060** at 10%
  labels, while improving robustness to missing B08, red-edge, and SWIR bands by
  approximately **0.0073**, **0.0334**, and **0.0404**, respectively.
- Matching-seed M3/M4 CKA values ranged from approximately **0.828–0.895**.

## Reproduce and verify

```bash
python -m pip install -c constraints/python312.txt -e '.[dev,train]'
python -m scripts.build_final_artifacts \
  --week8-dir reports/week8/generated \
  --week9-dir reports/week9/generated \
  --output-dir reports/final/generated
python -m scripts.verify_release --root .
```

See [REPORT.md](REPORT.md) for the research narrative and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete verification workflow.

Private patch-level labels, logits, predictions, feature arrays, checkpoints,
and diagnostic records are intentionally excluded from this public release.
