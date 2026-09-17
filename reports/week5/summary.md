# Week 5 implementation summary

Date: 17 September 2026  
Stage: controlled downstream anchor experiments  
Status: contracts verified; Kaggle pilots and final runs pending

The pipeline freezes deterministic nested D subsets for seeds 17, 29, and 43 at 1%, 5%, 10%, 25%, 50%, and 100%. Week 5 trains the 1%, 10%, and 100% anchors for M0-M4 after a bounded 15-run source-V learning-rate pilot.

Every final run starts with a fresh head and optimizer, uses only D for fitting, selects its checkpoint on V, saves V logits, and records exact data, subset, initialization, configuration, source, checkpoint, and prediction hashes. I, Finland, and Portugal remain sealed.

The Kaggle preparation run passed on 17 September 2026. It froze 36,000 subset-order records across the three seeds, identified the expected 16 supported source classes, verified all nine Week 4 encoder hashes, and kept evaluation labels unloaded. The downloaded ImageNet ResNet-18 checkpoint independently matched SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. The compact contract artifacts are preserved under `reports/week5/generated/contracts/`; the model checkpoint remains in the private Kaggle contracts dataset.

Expected Week 5 compute is 6-10 GPU-hours. Week 6 remains blocked until the CPU aggregate verifies all 45 anchor checkpoints.
