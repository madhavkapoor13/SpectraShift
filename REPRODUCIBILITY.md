# Reproducibility

SpectraShift separates public code and compact aggregate evidence from private or very large artifacts. The repository is sufficient to inspect the protocol, run its invariant tests, regenerate final figures, and verify the public result package. Re-running training requires the official BigEarthNet v2 archive and the private Kaggle artifacts listed below.

## Verified public environment

The release is tested with Python 3.12. Create an isolated environment from a clean checkout:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -c constraints/python312.txt -e '.[dev,train]'
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.verify_release --root .
```

The constraints file pins every direct dependency used by the verified CPU environment. Training summaries separately record the actual Kaggle container, PyTorch, CUDA, GPU, configuration, source-tree, data, and checkpoint hashes.

## Regenerating final artifacts

The final figures and compact release tables are deterministic functions of the tracked Week 8 and Week 9 aggregates:

```bash
MPLCONFIGDIR=/tmp/spectrashift-mpl \
PYTHONPATH=src .venv/bin/python -m scripts.build_final_artifacts \
  --week8-dir reports/week8/generated \
  --week9-dir reports/week9/generated \
  --output-dir reports/final/generated
```

The builder verifies the upstream aggregate hashes before writing anything. `results_manifest.json` records source and output hashes. The release verifier rejects changed files, malformed figures, broken local documentation links, and tracked private artifacts.

## Re-running the experimental pipeline

The complete sequence is documented by the numbered notebooks in `notebooks/kaggle/` and the corresponding `reports/week*/kaggle_steps.md` files. The durable private datasets are:

- `spectrashift-week2-frozen`: staged 12-band patches, split manifest, and U-only normalization.
- `spectrashift-week4-seed17/29/43`: nine SSL encoders.
- `spectrashift-week5-*` and `spectrashift-week6-*`: controlled downstream checkpoints and feature caches.
- `spectrashift-week7-*`: foundation contracts, checkpoints, probes, and frozen Week 8 ledger.
- `spectrashift-week8-*`: sealed evaluation contracts and label-free predictions.
- `spectrashift-week9-*`: frozen analysis contracts, probes, diagnostics, and final aggregate.

These datasets remain private because they contain large arrays, checkpoints, evaluation predictions, or sealed labels. Public compact outputs retain hashes that bind them to those artifacts.

## Scientific boundaries

- The data split, label subsets, thresholds, checkpoints, and hypotheses were frozen before final-label access.
- Finland and Portugal were resampled independently and averaged equally for OOD results.
- Reported uncertainty is descriptive. With three training seeds, no p-values are reported.
- DINOv2 and OlmoEarth may have unknown geographic overlap in their external pretraining corpora. They are contextual foundation baselines, not clean source-only controls.
- The release does not include raw imagery, target labels, prediction logits, model weights, patch features, or patch-level neighbour records.

See [REPORT.md](REPORT.md) for the research interpretation and [reports/final/generated/results_manifest.json](reports/final/generated/results_manifest.json) for the release evidence chain.
