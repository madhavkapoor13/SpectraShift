# Downstream configurations

`week5.yaml` freezes the M0-M4 anchor matrix, nested label counts, bounded learning-rate pilot, optimizer schedule, and artifact paths. Kaggle notebooks rewrite only input and output paths. Scientific settings are validated before every run.

`week6.yaml` reuses the frozen Week 5 subsets and selected learning rates. It defines the missing 5%, 25%, and 50% runs plus the 10% M1RGB ImageNet control. The RGB percentile contract is fitted from valid U pixels only; all scientific settings and input hashes are checked before training.
