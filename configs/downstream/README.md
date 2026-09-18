# Downstream configurations

`week5.yaml` freezes the M0-M4 anchor matrix, nested label counts, bounded learning-rate pilot, optimizer schedule, and artifact paths. Kaggle notebooks rewrite only input and output paths. Scientific settings are validated before every run.

`week6.yaml` reuses the frozen Week 5 subsets and selected learning rates. It defines the missing 5%, 25%, and 50% runs plus the 10% M1RGB ImageNet control. The RGB percentile contract is fitted from valid U pixels only; all scientific settings and input hashes are checked before training.

`week7.yaml` pins the DINOv2 ViT-S/14 fallback and OlmoEarth v1.1 Tiny. It defines the six foundation LR pilots, 18 full fine-tuning anchors, 36 frozen linear probes, six k-NN probes, offline asset hashes, and the batch-32-to-16 memory fallback gate.
