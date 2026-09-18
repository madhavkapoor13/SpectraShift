# Week 7 Kaggle execution

Run the notebooks in this order. Keep every dataset private and never attach `data/sealed/`.

## 1. Upload source v6

Create a private Kaggle dataset named `spectrashift-source-v6` containing only `dist/spectrashift-kaggle-source.zip`. Confirm its byte count and SHA-256 against the values printed by `scripts.build_kaggle_source`.

## 2. Freeze foundation contracts

Import `14_week7_prepare_foundations.ipynb`. Use CPU with Internet enabled and attach:

- `spectrashift-source-v6`
- `spectrashift-week2-frozen`
- `spectrashift-week5-contracts`
- `spectrashift-week6-contracts`
- `spectrashift-week6-complete`

The notebook downloads pinned official DINOv2 and OlmoEarth artifacts, verifies their hashes, packages offline dependencies, and runs deterministic finite-feature and gradient-flow smoke checks. Require `week7_contracts_complete: true`, `week7_adapter_smoke_complete: true`, and `evaluation_labels_loaded: false`. Preserve the output as `spectrashift-week7-contracts`.

## 3. Run the six pilots

Import `15_week7_lr_pilots.ipynb`. Select GPU T4 x2, keep Internet off, and attach:

- `spectrashift-source-v6`
- `spectrashift-week2-frozen`
- `spectrashift-week5-contracts`
- `spectrashift-week6-contracts`
- `spectrashift-week7-contracts`

The notebook tests batch 32 for both models and freezes batch 16 for both if either fails. It then runs the six 10%-label pilots. Require `week7_pilots_complete: true`, `foundation_matrix_approved: true`, `pilot_run_count: 6`, and `evaluation_labels_loaded: false`. Preserve the output as `spectrashift-week7-pilots`.

## 4. Run the three seed bundles

Run `16a_week7_seed17.ipynb`, `16b_week7_seed29.ipynb`, and `16c_week7_seed43.ipynb` sequentially on GPU T4 x2 with Internet off. Attach:

- `spectrashift-source-v6`
- `spectrashift-week2-frozen`
- `spectrashift-week5-contracts`
- `spectrashift-week6-contracts`
- `spectrashift-week7-contracts`
- `spectrashift-week7-pilots`
- any temporary failed output from the same seed, only when resuming

Each notebook runs six jobs and must end with `week7_seed_complete: true`, `run_count: 6`, and `evaluation_labels_loaded: false`. Preserve outputs as `spectrashift-week7-seed17`, `spectrashift-week7-seed29`, and `spectrashift-week7-seed43`.

If interrupted, create a temporary private dataset from the failed output and attach it to the rerun of the same seed notebook. Complete runs are reused and incomplete runs resume only from a configuration-matching recovery checkpoint.

## 5. Run frozen probes

Import `17_week7_frozen_probes.ipynb`. Use GPU T4 x2 with Internet off and attach the same five source/data/contract datasets plus `spectrashift-week7-pilots`. Run it after the seed notebooks to avoid concurrent GPU sessions. Require two feature caches, 36 linear probes, six k-NN probes, and `evaluation_labels_loaded: false`. Preserve the output as `spectrashift-week7-probes`.

## 6. Aggregate and approve Week 8

Import `18_week7_aggregate.ipynb`. Use CPU with Internet off and attach:

- `spectrashift-source-v6`
- `spectrashift-week6-complete`
- `spectrashift-week7-contracts`
- `spectrashift-week7-pilots`
- `spectrashift-week7-probes`
- `spectrashift-week7-seed17`
- `spectrashift-week7-seed29`
- `spectrashift-week7-seed43`

Require:

```text
week7_complete: true
week8_approved: true
foundation_full_run_count: 18
foundation_linear_probe_count: 36
foundation_knn_run_count: 6
foundation_feature_cache_count: 2
evaluation_labels_loaded: false
```

Preserve the output as `spectrashift-week7-complete`. Download `week7_run_summary.json`, `foundation_comparisons.csv`, and `week8_checkpoint_ledger.csv` for repository records.
