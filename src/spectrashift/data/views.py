from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


SPECTRAL_GROUPS = {
    "nir": (3,),
    "red_edge": (4, 5, 6, 7),
    "swir": (8, 9),
}


def deterministic_view_seed(base_seed: int, epoch: int, patch_id: str) -> int:
    payload = f"{base_seed}:{epoch}:{patch_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)


def crop_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    top_a, left_a, height_a, width_a = first
    top_b, left_b, height_b, width_b = second
    intersection_height = max(0, min(top_a + height_a, top_b + height_b) - max(top_a, top_b))
    intersection_width = max(0, min(left_a + width_a, left_b + width_b) - max(left_a, left_b))
    intersection = intersection_height * intersection_width
    union = height_a * width_a + height_b * width_b - intersection
    return intersection / union if union else 0.0


@dataclass(frozen=True)
class ViewConfig:
    output_size: int = 120
    minimum_crop_area: float = 0.8
    maximum_crop_area: float = 1.0
    minimum_pair_iou: float = 0.6
    spectral_dropout_probability: float = 0.0
    maximum_crop_attempts: int = 32

    def __post_init__(self) -> None:
        if not 0 < self.minimum_crop_area <= self.maximum_crop_area <= 1:
            raise ValueError("Crop areas must satisfy 0 < min <= max <= 1")
        if not 0 <= self.minimum_pair_iou <= 1:
            raise ValueError("Crop IoU must lie in [0,1]")
        if not 0 <= self.spectral_dropout_probability <= 1:
            raise ValueError("Spectral dropout probability must lie in [0,1]")


class TwoViewTransform:
    def __init__(self, config: ViewConfig) -> None:
        self.config = config

    @staticmethod
    def _rand(generator: torch.Generator) -> float:
        return float(torch.rand((), generator=generator).item())

    def _sample_crop(
        self, height: int, width: int, generator: torch.Generator
    ) -> tuple[int, int, int, int]:
        fraction = self.config.minimum_crop_area + self._rand(generator) * (
            self.config.maximum_crop_area - self.config.minimum_crop_area
        )
        side = max(1, min(height, width, int(math.ceil(math.sqrt(fraction * height * width)))))
        top = int(torch.randint(0, height - side + 1, (), generator=generator).item())
        left = int(torch.randint(0, width - side + 1, (), generator=generator).item())
        return top, left, side, side

    def _sample_pair(
        self, height: int, width: int, generator: torch.Generator
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        first = self._sample_crop(height, width, generator)
        for _ in range(self.config.maximum_crop_attempts):
            second = self._sample_crop(height, width, generator)
            if crop_iou(first, second) >= self.config.minimum_pair_iou:
                return first, second
        return first, first

    def _spatial_view(
        self,
        image: torch.Tensor,
        crop: tuple[int, int, int, int],
        generator: torch.Generator,
    ) -> torch.Tensor:
        top, left, height, width = crop
        view = image[:, top : top + height, left : left + width]
        view = F.interpolate(
            view.unsqueeze(0),
            size=(self.config.output_size, self.config.output_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        rotations = int(torch.randint(0, 4, (), generator=generator).item())
        view = torch.rot90(view, rotations, dims=(-2, -1))
        if self._rand(generator) < 0.5:
            view = torch.flip(view, dims=(-1,))
        if self._rand(generator) < 0.5:
            view = torch.flip(view, dims=(-2,))
        return view.contiguous()

    def __call__(self, image: torch.Tensor, seed: int) -> dict[str, object]:
        if image.ndim != 3:
            raise ValueError("Expected image [channels,height,width]")
        generator = torch.Generator().manual_seed(int(seed))
        first_crop, second_crop = self._sample_pair(image.shape[-2], image.shape[-1], generator)
        views = [
            self._spatial_view(image, first_crop, generator),
            self._spatial_view(image, second_crop, generator),
        ]
        dropped_view = -1
        dropped_group = "none"
        if self.config.spectral_dropout_probability and self._rand(generator) < self.config.spectral_dropout_probability:
            if image.shape[0] != 10:
                raise ValueError("Spectral-group dropout requires the core10 band adapter")
            dropped_view = int(torch.randint(0, 2, (), generator=generator).item())
            names = tuple(SPECTRAL_GROUPS)
            dropped_group = names[int(torch.randint(0, len(names), (), generator=generator).item())]
            views[dropped_view][list(SPECTRAL_GROUPS[dropped_group])] = 0
        return {
            "view1": views[0],
            "view2": views[1],
            "crop_boxes": torch.tensor([first_crop, second_crop], dtype=torch.int64),
            "dropped_view": dropped_view,
            "dropped_group": dropped_group,
        }
