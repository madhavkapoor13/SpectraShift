# Week 5 implementation summary

Date: 17 September 2026  
Stage: controlled downstream anchor experiments  
Status: contracts, pilots, and seeds 17/29 verified; 15 final Kaggle runs pending

The pipeline freezes deterministic nested D subsets for seeds 17, 29, and 43 at 1%, 5%, 10%, 25%, 50%, and 100%. Week 5 trains the 1%, 10%, and 100% anchors for M0-M4 after a bounded 15-run source-V learning-rate pilot.

Every final run starts with a fresh head and optimizer, uses only D for fitting, selects its checkpoint on V, saves V logits, and records exact data, subset, initialization, configuration, source, checkpoint, and prediction hashes. I, Finland, and Portugal remain sealed.

The Kaggle preparation run passed on 17 September 2026. It froze 36,000 subset-order records across the three seeds, identified the expected 16 supported source classes, verified all nine Week 4 encoder hashes, and kept evaluation labels unloaded. The downloaded ImageNet ResNet-18 checkpoint independently matched SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. The compact contract artifacts are preserved under `reports/week5/generated/contracts/`; the model checkpoint remains in the private Kaggle contracts dataset.

The 15-run Kaggle learning-rate pilot also passed on 17 September 2026. Every run completed 285 optimizer steps with finite diagnostics, zero AMP overflows, and no evaluation-label access. The selected multipliers are 1.0 for M0 and 3.0 for M1-M4; all five undertraining gates passed, so the final 45-run matrix is approved. The pilot consumed 0.273 reported GPU-hours. Compact pilot artifacts are preserved under `reports/week5/generated/pilots/`.

The seed-17 final bundle passed on 17 September 2026. All 15 model/fraction runs reached their exact optimizer-step targets, produced 2,000-by-19 finite V logits, retained 15 unique hash-verified checkpoints, and recorded zero AMP overflows. All four pre-fine-tuning feature caches contain finite 512-dimensional features for 12,000 D and 2,000 V patches. The bundle consumed 1.0192 reported GPU-hours and did not load evaluation labels.

Seed-17 supported-class V mAP for M0-M4 was `0.3797/0.4047/0.3260/0.3481/0.3343` at 1%, `0.4869/0.5172/0.3847/0.4408/0.4230` at 10%, and `0.6128/0.6172/0.5130/0.5822/0.5942` at 100%. These are development results from one seed; model comparisons remain provisional until seeds 29 and 43 and the aggregate complete.

The seed-29 final bundle also passed on 17 September 2026. Its 15 unique checkpoints and prediction files matched their hashes, all logits and four feature caches were finite with the required shapes, every run reached its exact step count with zero AMP overflows, and evaluation labels remained unloaded. The bundle consumed 1.0177 reported GPU-hours. Supported-class V mAP for M0-M4 was `0.3691/0.4121/0.3112/0.3306/0.3241` at 1%, `0.5175/0.5077/0.4212/0.4624/0.4667` at 10%, and `0.6110/0.6135/0.5035/0.6039/0.5991` at 100%. Comparisons remain provisional until seed 43 and aggregation complete.

Expected Week 5 compute is 6-10 GPU-hours. Week 6 remains blocked until the CPU aggregate verifies all 45 anchor checkpoints.
