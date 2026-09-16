from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SOURCE_SPLITS = ("source_train", "source_validation", "source_test")


@dataclass(frozen=True)
class Allocation:
    group_to_split: dict[str, str]
    score: float


def _indicator(frame: pd.DataFrame, key: str, values: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(values)}
    matrix = np.zeros((len(frame), len(values)), dtype=np.float64)
    if key == "country":
        for row, value in enumerate(frame[key]):
            matrix[row, lookup[value]] = 1.0
    else:
        for row, row_values in enumerate(frame[key]):
            for value in row_values:
                matrix[row, lookup[value]] = 1.0
    return matrix


def allocate_source_groups(
    frame: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
    trials: int,
) -> Allocation:
    if set(ratios) != set(SOURCE_SPLITS) or not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Source split ratios must define source_train/validation/test and sum to one")
    labels = sorted({label for values in frame["labels"] for label in values})
    countries = sorted(frame["country"].unique())
    label_matrix = _indicator(frame, "labels", labels)
    country_matrix = _indicator(frame, "country", countries)
    group_names = sorted(frame["mgrs_tile"].unique())
    row_group = frame["mgrs_tile"].map({g: i for i, g in enumerate(group_names)}).to_numpy()
    group_counts = np.bincount(row_group, minlength=len(group_names)).astype(float)
    group_labels = np.zeros((len(group_names), len(labels)), dtype=float)
    group_countries = np.zeros((len(group_names), len(countries)), dtype=float)
    np.add.at(group_labels, row_group, label_matrix)
    np.add.at(group_countries, row_group, country_matrix)
    total_labels = group_labels.sum(axis=0)
    total_countries = group_countries.sum(axis=0)
    total_count = group_counts.sum()
    split_names = list(SOURCE_SPLITS)
    targets = np.array([ratios[name] for name in split_names])
    rng = np.random.default_rng(seed)
    best: Allocation | None = None
    for _ in range(trials):
        order = np.lexsort((rng.random(len(group_names)), -group_counts))
        counts = np.zeros(3)
        label_counts = np.zeros((3, len(labels)))
        country_counts = np.zeros((3, len(countries)))
        assignments = np.full(len(group_names), -1, dtype=int)
        for group_index in order:
            candidate_scores = []
            for split_index in range(3):
                next_counts = counts.copy()
                next_labels = label_counts.copy()
                next_countries = country_counts.copy()
                next_counts[split_index] += group_counts[group_index]
                next_labels[split_index] += group_labels[group_index]
                next_countries[split_index] += group_countries[group_index]
                count_error = np.mean(((next_counts / total_count) - targets) ** 2)
                label_error = np.mean(
                    ((next_labels / np.maximum(total_labels, 1)) - targets[:, None]) ** 2
                )
                country_error = np.mean(
                    ((next_countries / np.maximum(total_countries, 1)) - targets[:, None]) ** 2
                )
                candidate_scores.append(count_error + label_error + country_error)
            chosen = int(np.argmin(candidate_scores))
            assignments[group_index] = chosen
            counts[chosen] += group_counts[group_index]
            label_counts[chosen] += group_labels[group_index]
            country_counts[chosen] += group_countries[group_index]
        group_penalty = sum(np.count_nonzero(assignments == i) == 0 for i in range(3)) * 1000
        score = min(candidate_scores) + group_penalty
        mapping = {group_names[i]: split_names[assignments[i]] for i in range(len(group_names))}
        candidate = Allocation(mapping, float(score))
        if best is None or candidate.score < best.score:
            best = candidate
    assert best is not None
    return best


def balanced_location_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    one_per_location = (
        frame.sample(frac=1, random_state=seed)
        .drop_duplicates("location_key", keep="first")
        .copy()
    )
    groups = {
        name: group.sample(frac=1, random_state=seed + index).reset_index(drop=True)
        for index, (name, group) in enumerate(one_per_location.groupby("mgrs_tile", sort=True))
    }
    selected: list[pd.Series] = []
    depth = 0
    while len(selected) < count:
        added = False
        for name in sorted(groups):
            group = groups[name]
            if depth < len(group):
                selected.append(group.iloc[depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    if not selected:
        return one_per_location.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)

