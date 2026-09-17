# Week 5 Kaggle execution

## Source

Upload `dist/spectrashift-kaggle-source.zip` as the private dataset `spectrashift-source-v4`. Use the exact byte count and SHA-256 printed by `scripts.build_kaggle_source`.

## 1. Freeze contracts

Import `07_week5_prepare.ipynb`. Use CPU with Internet enabled and attach `spectrashift-source-v4`, `spectrashift-week2-frozen`, and `spectrashift-week4-complete`. Run all cells and require `week5_contracts_complete: true`, 16 supported classes, and an ImageNet checkpoint hash beginning `f37072fd`. Preserve the output as private dataset `spectrashift-week5-contracts`.

## 2. Run bounded pilots

Import `08_week5_lr_pilots.ipynb`. Select GPU T4 x2, disable Internet, and attach the source, Week 2 frozen data, Week 4 aggregate, Week 5 contracts, and `spectrashift-week4-seed17`. Run all cells and require `week5_pilots_complete: true`, `week5_final_approved: true`, and `run_count: 15`. Preserve the output as `spectrashift-week5-pilots`.

## 3. Run final seed bundles

Run `09a_week5_seed17.ipynb`, `09b_week5_seed29.ipynb`, and `09c_week5_seed43.ipynb` sequentially on GPU T4 x2 with Internet off. Attach the source, Week 2 frozen data, Week 4 aggregate, Week 5 contracts, Week 5 pilots, and the matching Week 4 seed dataset. Require `week5_seed_complete: true`, 15 runs, four feature caches, and no evaluation-label access. Preserve outputs as `spectrashift-week5-seed17`, `spectrashift-week5-seed29`, and `spectrashift-week5-seed43`.

If interrupted, create a temporary private dataset from the failed output and attach it to the rerun of the same notebook. Complete runs are reused and incomplete runs resume from `recovery-latest.pt` after hash validation.

## 4. Aggregate

Import `10_week5_aggregate.ipynb`. Use CPU with Internet off and attach the newest source, `spectrashift-week5-pilots`, and all three Week 5 seed datasets. Require `week5_complete: true`, `week6_approved: true`, `run_count: 45`, and `evaluation_labels_loaded: false`. Preserve the output as `spectrashift-week5-complete` and download only `week5_run_summary.json` for repository records.
