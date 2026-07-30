import numpy as np
import pandas as pd
import pytest

from covid_xray.dedup import (
    drop_duplicates,
    find_duplicate_groups,
    hamming_distances,
    scan_dataset,
    summarise_duplicates,
)


def test_scan_finds_every_image_and_pairs_its_mask(synthetic_dataset, fixture_counts):
    df = scan_dataset(synthetic_dataset)

    assert len(df) == sum(fixture_counts.values())
    assert set(df.columns) >= {"path", "mask_path", "class_name", "label", "md5", "phash"}
    assert df["label"].tolist() == sorted(df["label"].tolist())  # grouped by class order
    for _, row in df.iterrows():
        assert (synthetic_dataset / row["mask_path"]).exists()



def test_scan_paths_are_relative_to_root(synthetic_dataset):
    df = scan_dataset(synthetic_dataset)
    assert not any(str(p).startswith("/") for p in df["path"])


def test_labels_follow_canonical_class_order(synthetic_dataset):
    df = scan_dataset(synthetic_dataset)
    mapping = df.drop_duplicates("class_name").set_index("class_name")["label"].to_dict()
    assert mapping == {"COVID": 0, "Lung_Opacity": 1, "Normal": 2, "Viral Pneumonia": 3}


def test_hamming_distances_are_symmetric_and_zero_on_diagonal():
    hashes = np.array([[0, 0, 0, 0, 0, 0, 0, 0b1111], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    distances = hamming_distances(hashes, hashes)
    assert distances[0, 0] == 0
    assert distances[1, 1] == 0
    assert distances[0, 1] == 4
    assert distances[0, 1] == distances[1, 0]


def test_planted_exact_duplicate_is_found(synthetic_dataset):
    df = scan_dataset(synthetic_dataset)
    groups = find_duplicate_groups(df)

    paired = {frozenset(df.iloc[g]["path"].tolist()) for g in groups}
    expected = frozenset({"COVID/images/COVID-1.png", "Normal/images/Normal-8.png"})
    assert expected in paired


def test_drop_duplicates_keeps_one_member_per_group(synthetic_dataset):
    df = scan_dataset(synthetic_dataset)
    groups = find_duplicate_groups(df)

    kept, removed = drop_duplicates(df, groups)

    assert len(kept) + len(removed) == len(df)
    assert len(kept) == len(df) - sum(len(g) - 1 for g in groups)
    assert "duplicate_of" in removed.columns


def test_retained_member_is_first_by_sorted_filename(synthetic_dataset):
    """Determinism requirement from spec section 2 — not filesystem order."""
    df = scan_dataset(synthetic_dataset)
    groups = find_duplicate_groups(df)
    kept, _ = drop_duplicates(df, groups)

    # COVID/images/COVID-1.png sorts before Normal/images/Normal-8.png
    assert "COVID/images/COVID-1.png" in set(kept["path"])
    assert "Normal/images/Normal-8.png" not in set(kept["path"])


def test_summary_separates_within_class_from_cross_class(synthetic_dataset):
    df = scan_dataset(synthetic_dataset)
    groups = find_duplicate_groups(df)

    summary = summarise_duplicates(df, groups)

    assert summary["n_removed"] == sum(len(g) - 1 for g in groups)
    # The planted pair spans COVID and Normal, so at least one cross-class group.
    assert summary["n_cross_class"] >= 1
    assert summary["n_within_class"] + summary["n_cross_class"] == summary["n_groups"]
    # Chaining detector: the fixture is random noise plus one planted pair, so
    # nothing should merge beyond 2. A large value here means the threshold has
    # collapsed unrelated images into a blob.
    assert summary["max_group_size"] == 2


def _scan_and_group(root):
    df = scan_dataset(root)
    return df, find_duplicate_groups(df)


def test_dedup_is_reproducible(synthetic_dataset):
    first = drop_duplicates(*_scan_and_group(synthetic_dataset))[0]["path"].tolist()
    second = drop_duplicates(*_scan_and_group(synthetic_dataset))[0]["path"].tolist()
    assert first == second
