# Week 4 Kaggle execution

## Prerequisites

1. Preserve the successful Week 3 notebook output as the private dataset `spectrashift-week3-pilots`.
2. Upload `dist/spectrashift-kaggle-source.zip` as a new version of the private source dataset. The exact-step hotfix bundle is 85,160 bytes with SHA-256 `0790a764e0ff46f82db475df5caee44f0cc086174dc7532f9be651a7f2c1cd93`.
3. Keep `spectrashift-week2-frozen` unchanged.

## Run the three seed jobs

Run these notebooks one at a time, in order:

1. `05a_week4_seed17.ipynb`
2. `05b_week4_seed29.ipynb`
3. `05c_week4_seed43.ipynb`

For each notebook:

1. Attach the newest source dataset, `spectrashift-week2-frozen`, and `spectrashift-week3-pilots`.
2. Select GPU T4 x2 and keep Internet disabled.
3. Run through the first four cells and confirm the frozen-data, Week 3 approval, and T4 gates pass.
4. Select **Save Version -> Save & Run All**.
5. Confirm the final cell reports `week4_seed_complete: true` and three passing runs.
6. Create a private dataset directly from the notebook output named `spectrashift-week4-seed17`, `spectrashift-week4-seed29`, or `spectrashift-week4-seed43` as appropriate.

Do not download the checkpoint directories to the Mac.

## Recovery

If a seed notebook fails, create a temporary private dataset from its output and attach that dataset when rerunning the same notebook. The launcher reuses verified completed runs and resumes an incomplete run from its newest matching ten-epoch checkpoint. Do not alter seeds, model order, bands, dropout, or training settings.

## Aggregate

1. Import `06_week4_aggregate.ipynb` as a private CPU notebook.
2. Attach the newest source dataset and the three successful Week 4 seed datasets.
3. Keep Internet disabled and run all cells.
4. Confirm `week4_complete: true`, `week5_approved: true`, and `run_count: 9`.
5. Preserve the output as private dataset `spectrashift-week4-complete` and download only `week4_run_summary.json` for repository records.
