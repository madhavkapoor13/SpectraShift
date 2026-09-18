# Week 8 Kaggle execution

Keep every dataset private. Use Internet off for every Week 8 notebook. Run the three seed notebooks sequentially.

## 1. Upload source v7

Create a private Kaggle dataset named `spectrashift-source-v7` containing only `dist/spectrashift-kaggle-source.zip`. Confirm its byte count and SHA-256 against the output from:

```bash
.venv/bin/python -m scripts.build_kaggle_source
```

## 2. Create the private candidate-label dataset

Create a private Kaggle dataset named `spectrashift-evaluation-candidates` containing only:

```text
evaluation_candidate_labels.parquet
```

Use the file from the ignored local directory `data/sealed/`. Do not make this dataset public and do not attach it to any development notebook.

## 3. Freeze the evaluation contract

Import `19_week8_prepare_evaluation.ipynb`. Select CPU, keep Internet off, and attach:

- `spectrashift-source-v7`
- `spectrashift-week2-frozen`
- `spectrashift-week5-contracts`
- `spectrashift-week7-complete`
- `spectrashift-evaluation-candidates`

Run **Save & Run All**. Require:

```text
week8_sealing_complete: true
partition_counts: I=3000, T-FI=4000, T-PT=4000
checkpoint_ledger_count: 111
model_selection_frozen_before_label_access: true
evaluation_labels_loaded: true
```

Preserve the output as the private dataset `spectrashift-week8-evaluation-contracts`.

## 4. Evaluate seed 17

Import `20a_week8_evaluate_seed17.ipynb`. Select **GPU T4 x2**, keep Internet off, and attach:

- `spectrashift-source-v7`
- `spectrashift-week2-frozen`
- `spectrashift-week6-contracts`
- `spectrashift-week7-contracts-v2`
- `spectrashift-week7-complete`
- `spectrashift-week8-evaluation-contracts`
- `spectrashift-week5-seed17-full`
- `spectrashift-week6-seed17`
- `spectrashift-week7-seed17`

Run **Save & Run All**. Require `week8_seed_evaluation_complete: true`, `run_count: 37`, and `prediction_domain_count: 111`. Preserve the output as `spectrashift-week8-seed17-eval`.

## 5. Evaluate seeds 29 and 43

Repeat the same process sequentially:

- `20b_week8_evaluate_seed29.ipynb` with `spectrashift-week5-seed29`, `spectrashift-week6-seed29`, and `spectrashift-week7-seed29`. Save as `spectrashift-week8-seed29-eval`.
- `20c_week8_evaluate_seed43.ipynb` with `spectrashift-week5-seed43`, `spectrashift-week6-seed43`, and `spectrashift-week7-seed43`. Save as `spectrashift-week8-seed43-eval`.

If a notebook fails, keep the logs. Do not alter thresholds, checkpoints, model configurations, support classes, or labels. Week 8 evaluation is deterministic and has no training/resume state.

## 6. Aggregate final evaluation

Import `21_week8_aggregate.ipynb`. Select CPU, keep Internet off, and attach:

- `spectrashift-source-v7`
- `spectrashift-week2-frozen`
- `spectrashift-week7-complete`
- `spectrashift-week8-evaluation-contracts`
- `spectrashift-week8-seed17-eval`
- `spectrashift-week8-seed29-eval`
- `spectrashift-week8-seed43-eval`

Run **Save & Run All**. Require:

```text
week8_complete: true
week9_approved: true
evaluation_run_count: 111
prediction_domain_count: 333
checkpoint_ledger_count: 111
evaluation_labels_loaded: true
model_selection_frozen_before_label_access: true
week9_implemented: false
```

Preserve the output as `spectrashift-week8-complete`. Download these compact files for repository records:

```text
week8_run_summary.json
domain_metrics.csv
per_class_metrics.csv
paired_bootstrap.csv
domain_rankings.csv
support_contract.json
```
