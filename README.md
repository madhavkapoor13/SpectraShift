# SpectraShift

**Self-supervised multispectral representation learning under geographic and spectral shift.**

SpectraShift is a completed controlled study on BigEarthNet v2. It tests whether ten-band Sentinel-2 self-supervision improves label efficiency and held-out-country transfer, and whether spectral-group dropout trades clean accuracy for robustness when bands are missing.

[Read the technical report](REPORT.md) · [Reproduce the release](REPRODUCIBILITY.md) · [Inspect the data contract](DATA.md) · [Review the model card](MODEL_CARD.md)

![Label efficiency on source I and equal-country OOD](reports/final/generated/figures/label_efficiency_source_ood.png)

## Findings

- **Multispectral information helped within SSL.** M3 ten-band VICReg exceeded M2 RGB VICReg by `+0.0337 ± 0.0062` normalized OOD AULC.
- **The selected SSL recipe did not beat stronger controls.** M3 trailed random multispectral M0 by `−0.0243 ± 0.0095` OOD AULC and also trailed ImageNet-initialized M1.
- **OlmoEarth was the strongest practical baseline.** Its equal-country OOD mAP was `0.3619`, `0.4247`, and `0.4505` at 1%, 10%, and 100% labels.
- **Spectral dropout changed robustness rather than clean accuracy.** At 10% labels M4−M3 was `−0.0060` on clean OOD, but `+0.0334` with the red-edge group missing and `+0.0404` with SWIR missing.
- **Geographic shift remained substantial.** Every model declined from source I to held-out Finland and Portugal.

The project makes no novelty claim for VICReg or spectral dropout. Its contribution is a reproducible empirical design, matched controls, frozen geographic evaluation, and a mechanism-focused analysis of a mixed result.

## Experimental scale

| Component | Verified scale |
| --- | ---: |
| Frozen Sentinel-2 subset | 45,000 patches |
| Unlabeled source pool | 20,000 patches |
| Label budgets | 1%, 5%, 10%, 25%, 50%, 100% |
| SSL encoders | 9 |
| Frozen downstream checkpoints | 111 |
| Final domains | I, Finland, Portugal |
| Linear / k-NN probes | 108 / 18 |
| Spectral stress states | 108 |
| Successful public ledger records | 363 |

## Models

- M0: random ten-band ResNet-18
- M1: ImageNet-initialized ten-band ResNet-18
- M2: RGB VICReg ResNet-18
- M3: ten-band VICReg ResNet-18
- M4: ten-band VICReg with spectral-group dropout
- M1RGB: standard RGB ImageNet control
- M5: DINOv2 ViT-S/14
- M6: OlmoEarth v1.1 Tiny

DINOv2 is the declared public fallback and is never presented as DINOv3. Foundation results include uncontrolled external-pretraining advantages.

## Quick verification

```bash
git clone https://github.com/madhavkapoor13/SpectraShift.git
cd SpectraShift
python3.12 -m venv .venv
.venv/bin/python -m pip install -c constraints/python312.txt -e '.[dev,train]'
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.verify_release --root .
```

Regenerate the public figures and final ledgers:

```bash
MPLCONFIGDIR=/tmp/spectrashift-mpl PYTHONPATH=src \
.venv/bin/python -m scripts.build_final_artifacts \
  --week8-dir reports/week8/generated \
  --week9-dir reports/week9/generated \
  --output-dir reports/final/generated
```

## Repository map

| Path | Purpose |
| --- | --- |
| `src/spectrashift/` | Data, models, training, evaluation, and release code |
| `configs/` | Frozen data, SSL, downstream, evaluation, and analysis contracts |
| `scripts/` | Reproducible command interfaces |
| `notebooks/kaggle/` | Numbered offline execution notebooks |
| `reports/week*/generated/` | Compact verified aggregate evidence |
| `reports/final/generated/` | Final figures, ledgers, hashes, and release summary |
| `tests/` | Scientific and release invariants |

## Artifact policy

The repository intentionally excludes raw imagery, staged arrays, evaluation labels, logits, model weights, checkpoint states, cached patch features, and patch-level neighbour records. These remain in private Kaggle datasets. Public aggregates retain hashes that bind them to the frozen experiment artifacts. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [THIRD_PARTY.md](THIRD_PARTY.md).

## License and citation

SpectraShift code and authored documentation are released under the [MIT License](LICENSE). BigEarthNet, DINOv2, OlmoEarth, and other upstream assets retain their own terms. Citation metadata is provided in [CITATION.cff](CITATION.cff).
