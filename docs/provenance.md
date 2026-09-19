# Dataset and method provenance

Checked 19 September 2026.

## BigEarthNet v2

- Official project: https://bigearth.net/
- Release record: https://zenodo.org/records/10891137
- DOI: https://doi.org/10.5281/zenodo.10891137
- Description: https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf
- Reproducible pipeline: https://github.com/rsim-tu-berlin/bigearthnet-pipeline
- Metadata mirror used after the Zenodo file endpoint timed out: https://huggingface.co/datasets/torchgeo/bigearthnet/resolve/main/V2/metadata.parquet
- Verified metadata MD5: `55687065e77b6d0b0f1ff604a6e7b49c`
- Dataset license: https://cdla.dev/permissive-1-0/

The mirror file matches the checksum published with the Zenodo release. The clean metadata contains 480,038 recommended patches; the full paired archive contains 549,488 patches including those separated into the snow/cloud/shadow metadata file.

## VICReg

- Paper: https://arxiv.org/abs/2105.04906
- Reference implementation: https://github.com/facebookresearch/vicreg

SpectraShift implements the declared objective directly and validates its reductions against the paper and reference code. Nine final M2-M4 encoders were trained from random initialization; no reference weights were redistributed.

## DINOv2

- Source: https://github.com/facebookresearch/dinov2
- Pinned source revision: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- Source archive SHA-256: `04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48`
- Official ViT-S/14 weight SHA-256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`
- Pinned repository license: Apache License 2.0

The project uses DINOv2 as the declared public fallback and never labels it DINOv3. The checkpoint is not redistributed.

## OlmoEarth v1.1 Tiny

- Model card: https://huggingface.co/allenai/OlmoEarth-v1_1-Tiny
- Pinned model revision: `74fab5714f763d6b94f8b1536bdd3300d77f45e8`
- Minimal-loader revision: `28eb18a852be74871a357fbcca7676f790a55c66`
- Minimal-loader archive SHA-256: `e9a875819466cd8f5b0d5693ec3fbd141fbc3cc26230e5b859542b93e4029efe`
- Terms: OlmoEarth Artifact License and Ai2 Responsible Use Guidelines

The model, official normalizer, and checkpoint remain external artifacts. SpectraShift records their hashes and adapter contract but does not redistribute them.

## Public release boundary

The SpectraShift MIT license applies only to authored code and documentation. BigEarthNet imagery, foundation checkpoints, and upstream source retain their own terms. `THIRD_PARTY.md` provides the release-facing notice, and `reports/final/generated/results_manifest.json` binds public aggregate evidence to the frozen private artifacts.
