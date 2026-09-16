from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from spectrashift.models.resnet import build_resnet18


@pytest.mark.parametrize("channels", [3, 10, 12])
def test_resnet18_accepts_each_band_adapter(channels: int) -> None:
    model = build_resnet18(channels)
    assert model.conv1.in_channels == channels
    assert model.fc.out_features == 19
