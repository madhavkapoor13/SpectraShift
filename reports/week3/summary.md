# Week 3 implementation summary

Date: 14 September 2026  
Stage: VICReg implementation and pilot selection  
Status: complete; Week 4 approved

## Implemented

- Exact VICReg invariance, variance, and covariance reductions with coefficients `25/25/1`.
- ResNet-18 encoder and `512 -> 1024 -> 1024 -> 256` projector.
- Deterministic paired crops retaining 80-100% area with minimum IoU 0.6.
- Shared-across-band rotations/flips and M4 asymmetric spectral-group dropout.
- U-only label-free SSL loader with patch/epoch-derived augmentation seeds.
- AMP training with FP32 VICReg statistics, gradient clipping, cosine schedule, atomic epoch checkpoints, full RNG restoration, and JSONL diagnostics.
- Frozen D-to-V linear probe and predeclared pilot-selection rule.
- T4 Kaggle launcher using the private Week 2 frozen dataset without copying its 18.46 GB arrays.

## Pilot contract

Run M3 at learning rates `1e-4` and `3e-4`, seed 17, batch 64, and ten epochs. Both use identical U order and view seeds. Stable pilots require finite losses, median projector standard deviation at least 0.5, projector effective rank at least 32, and a nine-run SSL forecast below 30 GPU-hours. Select the highest V probe mAP; within 0.005 choose `3e-4`.

## Final results

- Local suite: 29 passed.
- Synthetic end-to-end SSL training and frozen D-to-V probe: passed.
- Notebook JSON validation: passed.
- Evaluation labels are not referenced by the Week 3 code or notebook.
- Both pilots completed 3,120 optimizer steps with zero AMP overflow skips.
- `1e-4`: V macro average precision `0.4142477305`, projector standard deviation `0.9673`, projector effective rank `49.4790`.
- `3e-4`: V macro average precision `0.3753572839`, projector standard deviation `0.9764`, projector effective rank `54.6265`.
- Both stability gates and both Week 4 compute gates passed.
- The predeclared selection rule chooses M3 with learning rate `1e-4` for Week 4.
- The selected schedule forecasts `3.4418` T4 GPU-hours for nine 60-epoch SSL runs.
- The selected final checkpoint SHA-256 is `08abe41e9311d79149aaca7e22dfe72243b1eaed8c81e14f06fa0bb7bdb180fc`.

## Kaggle attempt 1

The first T4 run passed the frozen-data and hardware preflight, then stopped on the
first `1e-4` pilot at optimizer step 0 with an infinite AMP-scaled gradient norm.
No checkpoint or probe result was produced. The training path now runs the
projector and VICReg covariance calculations in FP32, starts gradient scaling at
128, backs off recoverable AMP overflows, and aborts after eight consecutive
overflows. The full local suite still passes after the change.
