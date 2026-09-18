# Model and checkpoint register

## Student-trained backbone

The controlled M0-M4 models use torchvision ResNet-18. RGB, core-10, and all-12 adapters replace only the input convolution; downstream heads have 19 outputs. The implemented VICReg path adds a `512 -> 1024 -> 1024 -> 256` projector with BatchNorm/ReLU after the first two layers and a linear output. Each run records actual parameter counts, PyTorch/CUDA versions, hardware, configuration and data hashes, and checkpoint hashes.

M2 uses RGB VICReg, M3 uses ten-band VICReg, and M4 applies asymmetric spectral-group dropout with probability 0.25 during pretraining only. Week 3 pilots M3 at learning rates `1e-4` and `3e-4`; M2-M4 retain random initialization and matched patch/crop seeds for the controlled Week 4 runs.

Week 5 fine-tunes M0-M4 with fresh 19-output heads at the 1%, 10%, and 100% anchor fractions. M0 is random-init core10, M1 converts ImageNet RGB kernels into the declared ten-band stem, and M2-M4 load only the matching-seed Week 4 encoder state. Every run resets its head, optimizer, scheduler, scaler, and RNG; no larger label fraction is used to initialize a smaller one.

## Generic self-supervised baseline

Week 7 selects the declared public fallback: official DINOv2 `dinov2_vits14`. It uses 126-pixel B04/B03/B02 input, the frozen U-derived display mapping, ImageNet normalization, no register tokens, and mean pooling over 81 normalized patch tokens. Every result identifies DINOv2 explicitly and is never labeled DINOv3.

## Earth-observation baseline

Pin `allenai/OlmoEarth-v1_1-Tiny`, a 12.5M-parameter ViT-Tiny artifact. Use its official Normalizer, the documented Sentinel-2 band order, one real acquisition timestamp, patch size 8, and 10 m input resolution. Record the resolved revision and file SHA-256 when downloaded. Do not attach its artifact to Git.

If the OlmoEarth loader cannot be made correct within two implementation days, use TorchGeo's SSL4EO-S12 ResNet-18 MoCo checkpoint and document the L1C-to-L2A input shift.

## Acquisition status

| Artifact | Status after Week 2 implementation | Blocking action |
| --- | --- | --- |
| torchvision ResNet-18 | verified locally | none |
| DINOv2 ViT-S/14 | selected public fallback; adapter verified locally | freeze official source and weight hashes in Week 7 contracts |
| OlmoEarth v1.1 Tiny | downloaded and verified at revision `74fab5714f763d6b94f8b1536bdd3300d77f45e8`; official minimal loader verified locally | freeze model, source, and normalizer hashes in Week 7 contracts |

Large checkpoints and credentials are ignored. Every experiment record must include the model identifier, resolved revision, checkpoint hash, input bands, preprocessing contract, and trainable parameter count.
