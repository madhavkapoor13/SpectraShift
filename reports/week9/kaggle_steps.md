# Week 9 Kaggle execution

Keep every dataset private. Keep Internet off for every notebook. Run the diagnostic seed notebooks sequentially.

## 1. Upload source v8

Create a private Kaggle dataset named `spectrashift-source-v8` containing only:

```text
spectrashift-kaggle-source.zip
```

Use the archive under `dist/` and confirm its size and SHA-256 against the local source-build output.

## 2. Freeze the analysis contract

Import `22_week9_prepare_analysis.ipynb`, select CPU, and attach:

- `spectrashift-source-v8`
- `spectrashift-week2-frozen`
- `spectrashift-week5-contracts`
- `spectrashift-week7-complete`
- `spectrashift-week8-evaluation-contracts`
- `spectrashift-week8-complete`

Run **Save & Run All**. Require:

```text
week9_contracts_complete: true
cka_patch_count: 1000
nearest_neighbor_query_count: 100
diagnostic_checkpoint_count: 6
model_selection_after_week8: false
```

Preserve the output as `spectrashift-week9-contracts`.

## 3. Complete the representation probes

Import `23_week9_complete_probes.ipynb`, select **GPU T4 x2**, and attach:

- `spectrashift-source-v8`
- `spectrashift-week2-frozen`
- `spectrashift-week5-contracts`
- `spectrashift-week5-seed17-full`
- `spectrashift-week5-seed29`
- `spectrashift-week5-seed43`
- `spectrashift-week7-complete`
- `spectrashift-week7-probes`
- `spectrashift-week9-contracts`

Require 108 total linear probes, 72 new linear probes, 18 total k-NN probes, 12 new k-NN probes, and `encoder_updates_during_week9: false`. Preserve the output as `spectrashift-week9-probes`.

## 4. Run seed 17 diagnostics

Import `24a_week9_diagnostics_seed17.ipynb`, select **GPU T4 x2**, and attach:

- `spectrashift-source-v8`
- `spectrashift-week2-frozen`
- `spectrashift-week5-contracts`
- `spectrashift-week5-seed17-full`
- `spectrashift-week8-seed17-eval`
- `spectrashift-week9-contracts`

Require two checkpoints, 36 total stress states, 30 new perturbation states, and 1,000 nearest-neighbour rows. Preserve the output as `spectrashift-week9-diagnostics-seed17`.

## 5. Run seeds 29 and 43

Repeat sequentially:

- `24b_week9_diagnostics_seed29.ipynb` with `spectrashift-week5-seed29` and `spectrashift-week8-seed29-eval`; save as `spectrashift-week9-diagnostics-seed29`.
- `24c_week9_diagnostics_seed43.ipynb` with `spectrashift-week5-seed43` and `spectrashift-week8-seed43-eval`; save as `spectrashift-week9-diagnostics-seed43`.

Do not alter the checkpoint list, thresholds, perturbations, query IDs, or analysis bins after a failure. Preserve logs and rerun the same immutable job.

## 6. Aggregate Week 9

Import `25_week9_aggregate.ipynb`, select CPU, and attach:

- `spectrashift-source-v8`
- `spectrashift-week8-complete`
- `spectrashift-week9-contracts`
- `spectrashift-week9-probes`
- `spectrashift-week9-diagnostics-seed17`
- `spectrashift-week9-diagnostics-seed29`
- `spectrashift-week9-diagnostics-seed43`

Require:

```text
week9_complete: true
week10_approved: true
linear_probe_count: 108
knn_probe_count: 18
diagnostic_checkpoint_count: 6
stress_prediction_domain_count: 108
new_stress_prediction_domain_count: 90
cka_patch_count: 1000
nearest_neighbor_query_count: 100
nearest_neighbor_row_count: 3000
model_selection_after_week8: false
encoder_updates_during_week9: false
week10_implemented: false
```

Preserve the output as `spectrashift-week9-complete`. Download the compact CSV files, figures, and `week9_run_summary.json` for repository records. Do not download or commit evaluation labels, logits, checkpoints, or patch-level feature files.
