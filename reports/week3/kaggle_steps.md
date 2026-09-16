# Week 3 Kaggle pilot steps

## Inputs

1. Upload the current `dist/spectrashift-kaggle-source.zip` as a new version of the private `spectrashift-source-v3` dataset. The AMP-hotfix bundle is 70,759 bytes with SHA-256 `553f4e30cbff952cbf0c5dc5a30d27e7c2b999c4ef928139d57554a3d67d52f2`.
2. Keep the private `spectrashift-week2-frozen` dataset unchanged.
3. Import `notebooks/kaggle/04_week3_vicreg_pilots.ipynb` as a new private notebook.
4. Attach exactly these two inputs:
   - `spectrashift-source-v3`, newest version
   - `spectrashift-week2-frozen`

Do not attach the BigEarthNet archive or any sealed evaluation labels.

## Runtime

1. Select **GPU T4 x2**. The code uses GPU 0 only.
2. Keep Internet disabled.
3. Run through the GPU preflight cell first.
4. Confirm it reports `Tesla T4`, `sm_75`, 50,200 complete staged patches, and the frozen normalization and manifest hashes.
5. Select **Save Version -> Save & Run All** with output saving enabled.

The notebook runs M3 at learning rates `1e-4` and `3e-4`, ten epochs each, then fits frozen linear probes on D and selects on V. It never reads I, Finland, or Portugal labels.

## Success and preservation

The last cell must report `week3_complete: true` and `week4_approved: true`. Each SSL summary must report finite losses, `projector_std_gate: true`, `projector_rank_gate: true`, and `week4_compute_gate: true`.

Create a private Kaggle dataset from the notebook output named `spectrashift-week3-pilots`. Download these small files for repository records:

- `week3_run_summary.json`
- `runs.jsonl`
- Both `ssl_summary.json` files
- Both `probe_summary.json` files

Keep the checkpoint files in Kaggle. If the final selection is blocked, download the summaries and logs without changing the loss coefficients, batch size, spectral policy, or frozen data hashes.
