# Kaggle launchers

`02_stage_bigearthnet_v2.ipynb` is the Week 2 launcher. Attach one private input:

1. A source snapshot containing this repository, excluding `data/sealed/`.
The notebook must have Internet enabled and use a T4 GPU with Kaggle's current PyTorch image. It streams the two pinned TorchGeo mirror parts directly from Hugging Face. A locally mounted official `BigEarthNet-S2.tar.zst` or both mirror parts remain supported as optional alternatives.

If staging, freezing, and normalization finish but a later GPU check fails, preserve the notebook output as a private dataset and run `03_complete_week2_from_staged.ipynb` against that output. This avoids streaming the archive again.

Enable internet only if the base image lacks a declared Python dependency. A GPU is useful only for the final tiny-overfit cell; staging is CPU and storage bound. The notebook finds both inputs, rewrites only runtime paths, runs preflight, stages the selected candidates, freezes the public split, computes U-only statistics, and executes the smoke gate.

Save `/kaggle/working/spectrashift-week2` as notebook output. A rerun with the same output files resumes completed bands through the persistent progress maps. The launcher never receives evaluation labels.

The mirror parts are consumed as one gzip stream without downloading or concatenating them in `/kaggle/working`. Interrupted HTTP transfers resume from the exact next byte. Staging checks the published byte lengths and SHA-256 values and verifies successful archive decoding plus complete raster coverage. The projected cache is 18.45 GB, below Kaggle's documented 20 GB saved-output limit but with limited headroom. The first notebook gate checks the actual free space and stops before allocating arrays if the environment cannot hold the cache.

`04_week3_vicreg_pilots.ipynb` runs the two fixed M3 VICReg learning-rate pilots and their D-to-V frozen linear probes. Attach the updated source bundle and `spectrashift-week2-frozen`, select GPU T4 x2, and keep Internet disabled. Its output contains checkpoints, JSONL training logs, probe artifacts, the run ledger, and `week3_run_summary.json`; preserve it as the private dataset `spectrashift-week3-pilots`.

Week 4 is split into `05a_week4_seed17.ipynb`, `05b_week4_seed29.ipynb`, and `05c_week4_seed43.ipynb`. Each trains M2-M4 for one seed and can resume from a prior failed output dataset. After preserving all three successful outputs, use the CPU-only `06_week4_aggregate.ipynb` to verify nine unique encoders and approve Week 5.

Week 5 starts with CPU notebook `07_week5_prepare.ipynb`, which freezes all six nested D-subset manifests and downloads the official ImageNet ResNet-18 weights. Preserve its output as `spectrashift-week5-contracts`. Run `08_week5_lr_pilots.ipynb` on T4 x2, preserve `spectrashift-week5-pilots`, then run `09a`, `09b`, and `09c` sequentially for seeds 17, 29, and 43. Each seed job can reuse complete runs or resume an incomplete run from its latest atomic recovery state. Finish with CPU notebook `10_week5_aggregate.ipynb` and preserve `spectrashift-week5-complete`. Only D/V labels are accessible throughout Week 5.
