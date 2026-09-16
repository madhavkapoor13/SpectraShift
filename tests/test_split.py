import pandas as pd

from spectrashift.data.split import allocate_source_groups, balanced_location_sample
from spectrashift.data.audit import _label_counts


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for group in range(12):
        for index in range(20):
            rows.append(
                {
                    "patch_id": f"p-{group}-{index}",
                    "mgrs_tile": f"T{group:02d}ABC",
                    "location_key": f"T{group:02d}ABC_{index}_0",
                    "country": "A" if group < 6 else "B",
                    "labels": ("forest",) if index % 2 else ("water",),
                }
            )
    return pd.DataFrame(rows)


def test_group_allocation_is_disjoint_and_deterministic() -> None:
    frame = synthetic_frame()
    ratios = {"source_train": 0.7, "source_validation": 0.15, "source_test": 0.15}
    first = allocate_source_groups(frame, ratios, seed=1729, trials=8)
    second = allocate_source_groups(frame, ratios, seed=1729, trials=8)
    assert first.group_to_split == second.group_to_split
    assert set(first.group_to_split) == set(frame["mgrs_tile"])
    assert set(first.group_to_split.values()) == set(ratios)


def test_balanced_sample_uses_unique_locations() -> None:
    frame = synthetic_frame()
    duplicated = pd.concat([frame, frame.assign(patch_id=lambda x: x.patch_id + "-repeat")])
    sample = balanced_location_sample(duplicated, count=100, seed=17)
    assert len(sample) == 100
    assert sample["location_key"].is_unique
    assert sample.groupby("mgrs_tile").size().max() - sample.groupby("mgrs_tile").size().min() <= 1


def test_support_table_retains_absent_classes() -> None:
    frame = pd.DataFrame({"labels": [("forest",), ("water",)]})
    support = _label_counts(frame, ["forest", "water", "wetlands"]).set_index("label")
    assert support.loc["wetlands", "positives"] == 0
    assert support.loc["wetlands", "negatives"] == 2
