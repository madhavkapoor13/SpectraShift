# Week 6 Kaggle execution

Run these notebooks sequentially. Keep every dataset private. Do not attach or upload `data/sealed/`.

## 1. Upload source v5

Create a private Kaggle dataset named `spectrashift-source-v5` containing only `dist/spectrashift-kaggle-source.zip`. Confirm its byte count and SHA-256 against the values printed by `scripts.build_kaggle_source`.

## 2. Freeze the RGB contract

Import `11_week6_prepare_rgb.ipynb`. Use CPU, disable Internet, and attach exactly:

- `spectrashift-source-v5`
- `spectrashift-week2-frozen`
- `spectrashift-week5-contracts`
- `spectrashift-week5-complete`

Run all cells. Require `week6_contracts_complete: true`, `rgb_band_order: [B04, B03, B02]`, `rgb_fit_partition: U`, `rgb_fit_patch_count: 20000`, and `evaluation_labels_loaded: false`. Preserve the notebook output as `spectrashift-week6-contracts`.

## 3. Run the three seed bundles

Run `12a_week6_seed17.ipynb`, `12b_week6_seed29.ipynb`, and `12c_week6_seed43.ipynb` one at a time. Select GPU T4 x2, keep Internet off, and attach:

- `spectrashift-source-v5`
- `spectrashift-week2-frozen`
- `spectrashift-week5-contracts`
- the one verified dataset containing `week5_pilot_summary.json`
- `spectrashift-week5-complete`
- `spectrashift-week6-contracts`
- the matching `spectrashift-week4-seed17`, `seed29`, or `seed43` encoder dataset

Each notebook runs 16 jobs and must end with `week6_seed_complete: true`, `run_count: 16`, and `evaluation_labels_loaded: false`. Preserve outputs as `spectrashift-week6-seed17`, `spectrashift-week6-seed29`, and `spectrashift-week6-seed43`.

If a seed notebook is interrupted, preserve its failed output as a temporary private dataset and attach it to the rerun of that same seed notebook. The launcher reuses valid completed jobs and resumes the incomplete job from its latest hash-matching recovery checkpoint.

## 4. Aggregate and approve Week 7

Import `13_week6_aggregate.ipynb`. Use CPU with Internet off and attach:

- `spectrashift-source-v5`
- `spectrashift-week5-complete`
- `spectrashift-week6-contracts`
- all three Week 6 seed datasets

Run all cells. Require:

```text
week6_complete: true
week7_approved: true
new_run_count: 48
controlled_run_count: 90
rgb_control_run_count: 3
evaluation_labels_loaded: false
```

Preserve the output as `spectrashift-week6-complete`. Download the four compact aggregate files for repository records. Do not expose I, Finland, or Portugal labels during any Week 6 step.
