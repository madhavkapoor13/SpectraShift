# Experiment ledger

## Immutable decisions

- Research targets: Finland and Portugal
- Draft split seed: `1729`
- SSL/downstream seed bundles: `17`, `29`, `43`
- Core band order: `B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12`
- Baseline-only bands: `B01,B09`
- Primary metric: macro average precision
- Final test domains: source-domain I, Finland, Portugal

## Stage status

| Stage | Status | Evidence required |
| --- | --- | --- |
| W1 repository and provenance | implemented | Source URLs, license, checksums, reproducible command |
| W1 metadata audit | complete | Country/tile counts, source support, draft partition hashes |
| W2 pipeline implementation | complete | Streaming archive reader, band adapters, geometry filters, split freezer, U-only normalization, smoke runner |
| W2 real-data freeze | complete | Verified mirror hashes, 50,200 complete candidates, frozen manifest, T4 throughput, tiny-overfit pass |
| W3 VICReg pilots | complete | Both M3 LR pilots stable; D-to-V probe selected `1e-4`; Week 4 approved |
| W4 M2-M4 pretraining | complete | Nine complete encoder checkpoints and verified aggregate; Week 5 approved |
| W5 downstream anchors | complete | M0-M4 at 1%, 10%, and 100% across three seeds; 45 verified V-selected checkpoints; Week 6 approved |

## Change policy

Use `notebooks/kaggle/04_week3_vicreg_pilots.ipynb` with the updated source dataset and private `spectrashift-week2-frozen` input. Select T4 x2, keep Internet disabled, and preserve `/kaggle/working/spectrashift-week3-pilots` as private notebook output. Never attach `data/sealed/` to a development notebook.

Week 3 compares only M3 learning rates `1e-4` and `3e-4` at seed 17 for ten epochs. Stable pilots are ranked by full-D frozen-linear-probe mAP on V; differences below 0.005 select the predeclared `3e-4` default. The probe is engineering evidence and is not part of the final label-efficiency result.

Both pilots passed the stability and compute gates. Validation macro average precision was `0.4142477305` for `1e-4` and `0.3753572839` for `3e-4`, so the predeclared rule selects `1e-4` for Week 4. The selected schedule forecasts `3.4418` T4 GPU-hours for nine 60-epoch SSL runs.

Week 4 runs M2, M3, and M4 for seeds 17, 29, and 43 in three sequential seed jobs. Every run uses 60 epochs, batch 64, five warmup epochs, and exactly 18,720 optimizer steps. Recovery checkpoints are written every ten epochs; after success only the epoch-60 state and verified encoder export remain.

All nine Week 4 runs completed successfully in 5.9730 measured GPU-hours. The aggregate verified nine expected model-seed combinations, nine unique encoder hashes, exact optimizer-step counts, zero AMP overflows, passing stability/completion/compute gates, and no evaluation-label access. Week 5 is approved.

Week 5 freezes nested D subsets for all six eventual label fractions, runs bounded 10%-label LR pilots for M0-M4 at seed 17, and then trains the 45 anchor combinations at 1%, 10%, and 100%. Checkpoint selection uses supported-class source-V mAP only. I, Finland, and Portugal remain sealed.

All 45 Week 5 runs passed their exact-step, finiteness, checkpoint, prediction, feature-cache, and label-isolation gates. Three-seed mean source-V mAP for M0-M4 was `0.3698/0.4063/0.3106/0.3337/0.3186` at 1%, `0.4997/0.5136/0.4039/0.4526/0.4430` at 10%, and `0.6111/0.6157/0.5094/0.5957/0.5987` at 100%. The aggregate reports `week5_complete: true` and `week6_approved: true`.

The final I, Finland, and Portugal labels remain sealed through model selection. Any change to the split after pixel inspection creates a new manifest version. A target country is never swapped because of model performance.
