import pandas as pd
import pytest

from covid_xray.config import CLASS_NAMES
from covid_xray.dedup import scan_dataset
from covid_xray.splits import (
    class_distribution,
    load_manifest,
    stratified_split,
    write_manifests,
)


@pytest.fixture
def scanned(synthetic_dataset):
    return scan_dataset(synthetic_dataset).drop(columns=["phash"])


def test_split_assigns_every_row_exactly_once(scanned):
    result = stratified_split(scanned, seed=42)
    assert len(result) == len(scanned)
    assert set(result["split"]) <= {"train", "val", "test"}
    assert result["split"].notna().all()


def test_split_proportions_are_approximately_correct(scanned):
    result = stratified_split(scanned, seed=42)
    fractions = result["split"].value_counts(normalize=True)
    assert fractions["train"] == pytest.approx(0.70, abs=0.15)
    assert fractions["val"] == pytest.approx(0.15, abs=0.15)
    assert fractions["test"] == pytest.approx(0.15, abs=0.15)


def test_every_class_appears_in_every_split(scanned):
    result = stratified_split(scanned, seed=42)
    for split_name in ("train", "val", "test"):
        present = set(result.loc[result["split"] == split_name, "class_name"])
        assert present == set(CLASS_NAMES), f"{split_name} is missing classes"


def test_split_is_deterministic_for_a_fixed_seed(scanned):
    first = stratified_split(scanned, seed=42)["split"].tolist()
    second = stratified_split(scanned, seed=42)["split"].tolist()
    assert first == second


def test_different_seeds_give_different_splits(scanned):
    first = stratified_split(scanned, seed=42)["split"].tolist()
    second = stratified_split(scanned, seed=7)["split"].tolist()
    assert first != second


def test_no_path_appears_in_two_splits(scanned):
    result = stratified_split(scanned, seed=42)
    counts = result.groupby("path")["split"].nunique()
    assert (counts == 1).all()


def test_write_and_load_manifests_round_trip(scanned, tmp_path):
    result = stratified_split(scanned, seed=42)

    written = write_manifests(result, tmp_path)

    assert set(written) == {"train", "val", "test"}
    for split_name, path in written.items():
        loaded = load_manifest(path)
        assert len(loaded) == (result["split"] == split_name).sum()
        assert "phash" not in loaded.columns  # binary column must not be serialised
        assert set(loaded.columns) >= {"path", "mask_path", "class_name", "label"}


def test_class_distribution_counts_by_split(scanned):
    result = stratified_split(scanned, seed=42)
    table = class_distribution(result)
    assert table.loc["COVID"].sum() == (result["class_name"] == "COVID").sum()




def test_split_rejects_a_class_with_too_few_members(scanned):
    """Documents a real constraint rather than leaving it to be rediscovered.

    A two-stage stratified split needs every class to survive both stages.
    sklearn refuses to stratify a single-member class, and failing loudly is
    correct — silently dropping a class from val or test would corrupt every
    downstream metric without any visible symptom.
    """
    starved = pd.concat(
        [
            scanned[scanned["class_name"] != "Viral Pneumonia"],
            scanned[scanned["class_name"] == "Viral Pneumonia"].head(1),
        ]
    )

    with pytest.raises(ValueError, match="least populated"):
        stratified_split(starved, seed=42)
