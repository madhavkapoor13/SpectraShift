# SSL configurations

`week3_m3_lr1e4.yaml` and `week3_m3_lr3e4.yaml` are the two predeclared M3 pilot configurations. Kaggle rewrites only artifact paths; all scientific settings remain fixed. The winning stable pilot selects the Week 4 M2-M4 learning rate, with `3e-4` preferred when validation mAP differs by less than 0.005.

The nine `week4_<model>_seed<seed>.yaml` files freeze M2-M4 across seeds 17, 29, and 43. All use the selected `1e-4` learning rate, batch 64, five warmup epochs, and 60 total epochs. M2 uses RGB, M3 uses core10, and M4 uses core10 with asymmetric spectral-group dropout at probability 0.25.
