from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .dataset import SpectraShiftDataset
from .labels import CANONICAL_LABELS
from .views import deterministic_view_seed


def supervised_spatial_transform(image: torch.Tensor, seed: int) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("Expected image [channels,height,width]")
    generator = torch.Generator().manual_seed(int(seed))
    rotations = int(torch.randint(0, 4, (), generator=generator).item())
    result = torch.rot90(image, rotations, dims=(-2, -1))
    if float(torch.rand((), generator=generator)) < 0.5:
        result = torch.flip(result, dims=(-1,))
    if float(torch.rand((), generator=generator)) < 0.5:
        result = torch.flip(result, dims=(-2,))
    return result.contiguous()


class DownstreamDataset(Dataset):
    def __init__(
        self,
        base: SpectraShiftDataset,
        patch_ids: list[str] | None,
        seed: int,
        training: bool,
    ) -> None:
        self.base = base
        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0
        if patch_ids is None:
            self.indices = list(range(len(base)))
        else:
            lookup = {str(value): index for index, value in enumerate(base.frame["patch_id"])}
            missing = set(map(str, patch_ids)) - set(lookup)
            if missing:
                raise ValueError(f"Subset contains unknown D patch IDs: {sorted(missing)[:3]}")
            self.indices = [lookup[str(value)] for value in patch_ids]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        item = self.base[self.indices[index]]
        if item["labels"] is None:
            raise ValueError("Downstream loader requires visible D/V labels")
        image = torch.from_numpy(item["image"])
        if self.training:
            seed = deterministic_view_seed(self.seed, self.epoch, str(item["patch_id"]))
            image = supervised_spatial_transform(image, seed)
        lookup = {name: position for position, name in enumerate(CANONICAL_LABELS)}
        labels = torch.zeros(len(CANONICAL_LABELS), dtype=torch.float32)
        for value in item["labels"]:
            labels[lookup[str(value)]] = 1.0
        return image, labels, str(item["patch_id"])
