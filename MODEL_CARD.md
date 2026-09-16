# Model and checkpoint register

## Student-trained backbone

The controlled M0-M4 models use torchvision ResNet-18. RGB, core-10, and all-12 adapters replace only the input convolution; downstream heads have 19 outputs. The implemented VICReg path adds a `512 -> 1024 -> 1024 -> 256` projector with BatchNorm/ReLU after the first two layers and a linear output. Each run records actual parameter counts, PyTorch/CUDA versions, hardware, configuration and data hashes, and checkpoint hashes.

M2 uses RGB VICReg, M3 uses ten-band VICReg, and M4 applies asymmetric spectral-group dropout with probability 0.25 during pretraining only. Week 3 pilots M3 at learning rates `1e-4` and `3e-4`; M2-M4 retain random initialization and matched patch/crop seeds for the controlled Week 4 runs.

## Generic self-supervised baseline

Primary: Meta DINOv3 `dinov3_vits16`, ViT-S/16 distilled on LVD-1689M (21M parameters). Weight access requires the applicant to accept Meta's license and receive a private download URL. The repository cannot request or accept that license on the applicant's behalf.

Declared fallback if access is not obtained before baseline integration: official DINOv2 ViT-S/14. The fallback uses 126-pixel RGB input so the image size is divisible by 14. Any result must identify DINOv2 explicitly and must not be labeled DINOv3.

## Earth-observation baseline

Pin `allenai/OlmoEarth-v1_1-Tiny`, a 12.5M-parameter ViT-Tiny artifact. Use its official Normalizer, the documented Sentinel-2 band order, one real acquisition timestamp, patch size 8, and 10 m input resolution. Record the resolved revision and file SHA-256 when downloaded. Do not attach its artifact to Git.

If the OlmoEarth loader cannot be made correct within two implementation days, use TorchGeo's SSL4EO-S12 ResNet-18 MoCo checkpoint and document the L1C-to-L2A input shift.

## Acquisition status

| Artifact | Status after Week 2 implementation | Blocking action |
| --- | --- | --- |
| torchvision ResNet-18 | verified locally | none |
| DINOv3 ViT-S/16 | access gated | applicant accepts Meta license and supplies the issued URL |
| OlmoEarth v1.1 Tiny | downloaded and verified at revision `74fab5714f763d6b94f8b1536bdd3300d77f45e8` | integrate its official loader in Week 7 |

Large checkpoints and credentials are ignored. Every experiment record must include the model identifier, resolved revision, checkpoint hash, input bands, preprocessing contract, and trainable parameter count.
