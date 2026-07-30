"""Stratified split and manifest I/O.

The split is produced exactly once (Task 4) and committed. Everything
downstream reads the CSVs, so a reader who clones the repo reproduces the
exact partition without re-running the split.

Caveat carried from spec section 2: this dataset has no patient identifiers,
so the split is stratified by class only. Patient-disjointness cannot be
guaranteed and every downstream metric is an upper bound.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from covid_xray.config import CLASS_NAMES

SPLIT_NAMES = ("train", "val", "test")

# `phash` holds a numpy array per row; it exists only for de-duplication and
# must never reach a CSV.
_NON_SERIALISABLE = ("phash",)


def stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.DataFrame:
    """Add a ``split`` column with a class-stratified 70/15/15 partition."""
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError("val_fraction + test_fraction must lie strictly between 0 and 1")

    result = df.reset_index(drop=True).copy()
    positions = result.index.to_numpy()
    labels = result["label"].to_numpy()

    holdout_fraction = val_fraction + test_fraction
    train_positions, holdout_positions = train_test_split(
        positions,
        test_size=holdout_fraction,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    # Split the holdout in two, preserving the requested val:test ratio.
    val_positions, test_positions = train_test_split(
        holdout_positions,
        test_size=test_fraction / holdout_fraction,
        stratify=labels[holdout_positions],
        random_state=seed,
        shuffle=True,
    )

    result["split"] = ""
    result.loc[train_positions, "split"] = "train"
    result.loc[val_positions, "split"] = "val"
    result.loc[test_positions, "split"] = "test"
    return result


def write_manifests(df: pd.DataFrame, out_dir: str | Path) -> dict[str, Path]:
    """Write one CSV per split. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serialisable = df.drop(columns=[c for c in _NON_SERIALISABLE if c in df.columns])

    written: dict[str, Path] = {}
    for split_name in SPLIT_NAMES:
        subset = serialisable.loc[serialisable["split"] == split_name].sort_values("path")
        path = out_dir / f"{split_name}.csv"
        subset.to_csv(path, index=False)
        written[split_name] = path
    return written


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"path", "mask_path", "class_name", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"manifest {path} is missing columns: {sorted(missing)}")
    return df


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Class counts per split, in canonical class order — for the README table."""
    table = (
        df.pivot_table(index="class_name", columns="split", values="path", aggfunc="count")
        .reindex(list(CLASS_NAMES))
        .fillna(0)
        .astype(int)
    )
    return table.reindex(columns=[c for c in SPLIT_NAMES if c in table.columns])
