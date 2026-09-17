# Week 5 implementation summary

Date: 17 September 2026  
Stage: controlled downstream anchor experiments  
Status: implemented; Kaggle contracts, pilots, and final runs pending

The pipeline freezes deterministic nested D subsets for seeds 17, 29, and 43 at 1%, 5%, 10%, 25%, 50%, and 100%. Week 5 trains the 1%, 10%, and 100% anchors for M0-M4 after a bounded 15-run source-V learning-rate pilot.

Every final run starts with a fresh head and optimizer, uses only D for fitting, selects its checkpoint on V, saves V logits, and records exact data, subset, initialization, configuration, source, checkpoint, and prediction hashes. I, Finland, and Portugal remain sealed.

Expected Week 5 compute is 6-10 GPU-hours. Week 6 remains blocked until the CPU aggregate verifies all 45 anchor checkpoints.
