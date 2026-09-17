# SpectraShift

Self-supervised multispectral representation learning under spectral and geographic shift.

SpectraShift studies whether representations learned from Sentinel-2 imagery improve label efficiency and transfer to held-out European countries, and whether limited spectral-group invariance helps robustness without erasing useful land-cover information.

## Current status

Weeks 1 through 4 are complete. The private Kaggle artifact `spectrashift-week2-frozen` contains 50,200 verified patches, frozen split `BENv2-SpectraShift-v1`, and U-only normalization. All geographic, support, label-isolation, throughput, and tiny-overfit gates passed.

Both Week 3 M3 VICReg pilots passed their stability and compute gates. The frozen D-to-V probe selected learning rate `1e-4` with validation macro average precision `0.4142477305`, compared with `0.3753572839` for `3e-4`. Week 4 completed all nine M2-M4 seed runs in 5.9730 measured GPU-hours. The aggregate verified nine unique encoders, exact optimizer-step counts, zero AMP overflows, and no evaluation-label access; Week 5 is approved.

The Week 5 controlled downstream pipeline is implemented for M0-M4 at the 1%, 10%, and 100% anchor fractions. It freezes nested label manifests, performs bounded source-V learning-rate selection, supports exact resumable training, preserves V logits and verified best checkpoints, and keeps all final evaluation labels inaccessible.

## Setup and verification

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,train]'
.venv/bin/python -m pytest -q
```

## Data preparation

```bash
.venv/bin/python -m scripts.audit_data --config configs/data/week1.yaml
```

The command writes the public candidate manifest, seals I/Finland/Portugal labels under ignored `data/sealed/`, and generates the Week 1 audit evidence. Then run Week 2 in order:

```bash
.venv/bin/python -m scripts.preflight --config configs/data/week2.yaml
.venv/bin/python -m scripts.stage_data --config configs/data/week2.yaml
.venv/bin/python -m scripts.freeze_split --config configs/data/week2.yaml
.venv/bin/python -m scripts.compute_normalization --config configs/data/week2.yaml
.venv/bin/python -m scripts.benchmark_throughput --config configs/data/week2.yaml
.venv/bin/python -m scripts.smoke_test --config configs/data/week2.yaml
```

Run `scripts.seal_evaluation` only in the offline evaluation environment after model selection. It is intentionally absent from the Kaggle development launcher.

## Week 3 pilots

The two fixed pilot configurations are `configs/ssl/week3_m3_lr1e4.yaml` and `configs/ssl/week3_m3_lr3e4.yaml`. They can be run directly with `scripts.train_ssl` and evaluated with `scripts.probe_ssl`; the Kaggle launcher `notebooks/kaggle/04_week3_vicreg_pilots.ipynb` performs both runs and applies the predeclared selection rule.

```bash
.venv/bin/python -m scripts.train_ssl --config configs/ssl/week3_m3_lr3e4.yaml --seed 17
.venv/bin/python -m scripts.probe_ssl --config configs/ssl/week3_m3_lr3e4.yaml --checkpoint <checkpoint.pt>
```

## Week 4 pretraining

Week 4 uses the nine immutable `week4_<model>_seed<seed>.yaml` configurations. On Kaggle, run `05a`, `05b`, and `05c` sequentially, preserve each output as a private seed dataset, then run `06_week4_aggregate.ipynb`. See `reports/week4/kaggle_steps.md` for the exact workflow.

## Week 5 downstream anchors

Week 5 uses `configs/downstream/week5.yaml` and notebooks `07` through `10`. Run the contracts job, bounded LR pilots, three sequential seed bundles, and CPU aggregate in that order. See `reports/week5/kaggle_steps.md` for exact inputs and output dataset names.

## Research controls

- Finland and Portugal are fixed geographic targets.
- Any MGRS tile touching a target country is excluded from source pools.
- Repeat acquisitions at the same tile/row/column share one location key.
- I and target-label statistics are not computed during development.
- U, I, Finland, and Portugal labels are stripped from public manifests.
- Random state and input hashes are recorded with every artifact.

See [DATA.md](DATA.md), [EXPERIMENTS.md](EXPERIMENTS.md), and [the project plan](docs/spectrashift-project-plan.pdf).
