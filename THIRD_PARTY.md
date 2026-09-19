# Third-party data, models, and software

The repository's MIT license covers code and documentation authored for SpectraShift. It does not replace the licenses of the dataset, pretrained weights, upstream source archives, or Python dependencies.

| Component | Use in SpectraShift | Upstream terms |
| --- | --- | --- |
| BigEarthNet v2 | Sentinel-2 imagery and 19-label land-cover supervision | [CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/); the imagery is not redistributed here |
| DINOv2 ViT-S/14 | M5 foundation baseline | Source revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8` and official checkpoint; see the [DINOv2 repository license](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE) |
| OlmoEarth v1.1 Tiny | M6 Earth-observation foundation baseline | Model revision `74fab5714f763d6b94f8b1536bdd3300d77f45e8`; see the [OlmoEarth Artifact License and model card](https://huggingface.co/allenai/OlmoEarth-v1_1-Tiny) |
| torchvision ResNet-18 | M0-M4 backbone and ImageNet M1 initialization | [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| VICReg | Objective definition and reference comparison | [VICReg paper](https://arxiv.org/abs/2105.04906) and [reference implementation](https://github.com/facebookresearch/vicreg) |

All downloaded model files are excluded from Git. Their exact revisions and SHA-256 hashes are recorded in `configs/downstream/week7.yaml` and the private Week 7 contracts. Dataset release checksums and acquisition details are recorded in [DATA.md](DATA.md) and [docs/provenance.md](docs/provenance.md).
