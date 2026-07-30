"""Exact and near-duplicate detection.

Runs before splitting, over the whole four-class pool.

Near-duplicate search is brute-force pairwise Hamming distance over 64-bit
perceptual hashes. At 21k images that is ~224M pairs, which numpy handles in
chunks in well under a minute; a BK-tree would be faster but not worth the
extra moving part for a one-off offline pass.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from covid_xray.config import CLASS_NAMES

# Precomputed popcount for every byte value. Used instead of np.bitwise_count
# so the module works on numpy 1.x as well as 2.x — Colab's numpy version is
# outside our control.
_POPCOUNT8 = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1).astype(np.uint8)

_CHUNK_ROWS = 512  # keeps the pairwise XOR buffer near 100 MB


def _md5_of_pixels(path: Path) -> str:
    """Hash decoded pixels, not the file, so re-encodings still collide."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return hashlib.md5(array.tobytes()).hexdigest()


def _phash_bytes(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = imagehash.phash(image.convert("L"), hash_size=8)
    return np.packbits(value.hash.flatten())


def scan_dataset(root: str | Path) -> pd.DataFrame:
    """Walk the dataset and hash every image.

    Expects the v5 layout: ``<root>/<class_name>/images/*.png`` with a
    same-named file under ``<root>/<class_name>/masks/``.
    """
    root = Path(root)
    records: list[dict] = []

    for label, class_name in enumerate(CLASS_NAMES):
        image_dir = root / class_name / "images"
        mask_dir = root / class_name / "masks"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"expected {image_dir} — check the dataset layout")

        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"no mask for {image_path.name} at {mask_path}")
            records.append(
                {
                    "path": image_path.relative_to(root).as_posix(),
                    "mask_path": mask_path.relative_to(root).as_posix(),
                    "class_name": class_name,
                    "label": label,
                    "md5": _md5_of_pixels(image_path),
                    "phash": _phash_bytes(image_path),
                }
            )

    return pd.DataFrame.from_records(records)


def hamming_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two stacks of packed hash bytes.

    Both arrays are ``(n, 8)`` uint8. Returns an ``(n_left, n_right)`` uint8
    matrix of bit differences.
    """
    xor = left[:, None, :] ^ right[None, :, :]
    return _POPCOUNT8[xor].sum(axis=2).astype(np.uint8)


def find_duplicate_groups(df: pd.DataFrame, max_distance: int = 1) -> list[list[int]]:
    """Group images that are byte-identical or perceptually within ``max_distance``.

    Returns positional index groups of size >= 2, each sorted ascending.

    The default of 1 is empirical, not conventional. Measured on the full
    21,165-image dataset:

        threshold   groups  removed  cross-class
        md5 exact       44       59            0
        phash <= 1     178      223            0
        phash <= 3     263      339           17
        phash <= 5     666     1529          172
        phash <= 8     355    14403          107

    Cross-class matches are the diagnostic. A genuine duplicate is almost
    always within-class — same image, same label — so cross-class hits are
    mostly false positives. At <= 5 they were visually confirmed as different
    patients: chest radiographs are structurally uniform (two lungs, ribcage,
    spine, same framing), and a 64-bit DCT hash matches "this is a frontal
    CXR" rather than "this is the same image."

    The <= 8 row shows the other failure mode. Fewer groups but 14,403 removed
    means union-find has chained A~B~C into vast blobs where A and C are
    unrelated. `summarise_duplicates` reports `max_group_size` so this is
    visible rather than silent.
    """
    n = len(df)
    parent = list(range(n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    # Exact matches first — cheap and catches re-saved copies with identical pixels.
    for _, positions in df.groupby("md5").indices.items():
        for other in positions[1:]:
            union(int(positions[0]), int(other))

    # Near matches.
    hashes = np.stack(df["phash"].to_numpy())
    for start in range(0, n, _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, n)
        distances = hamming_distances(hashes[start:stop], hashes)
        rows, cols = np.nonzero(distances <= max_distance)
        for row, col in zip(rows, cols):
            absolute_row = start + int(row)
            if absolute_row < int(col):  # upper triangle only
                union(absolute_row, int(col))

    clusters: dict[int, list[int]] = {}
    for index in range(n):
        clusters.setdefault(find(index), []).append(index)

    return [sorted(members) for members in clusters.values() if len(members) > 1]

def drop_duplicates(
    df: pd.DataFrame, groups: list[list[int]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one member per group, drop the rest.

    The retained member is the first by sorted ``path``, so the surviving set
    is identical across machines and runs regardless of filesystem ordering.
    """
    keep_positions: set[int] = set()
    removed_records: list[dict] = []

    for group in groups:
        ordered = sorted(group, key=lambda position: df.iloc[position]["path"])
        keeper = ordered[0]
        keep_positions.add(keeper)
        for position in ordered[1:]:
            record = df.iloc[position].to_dict()
            record["duplicate_of"] = df.iloc[keeper]["path"]
            removed_records.append(record)

    grouped_positions = {position for group in groups for position in group}
    kept_mask = np.array(
        [position not in grouped_positions or position in keep_positions for position in range(len(df))]
    )

    kept = df.loc[kept_mask].reset_index(drop=True)
    removed = pd.DataFrame.from_records(removed_records) if removed_records else pd.DataFrame(
        columns=[*df.columns, "duplicate_of"]
    )
    return kept, removed


def summarise_duplicates(df: pd.DataFrame, groups: list[list[int]]) -> dict:
    """Counts for the README.

    `n_cross_class` is the false-positive detector: genuine duplicates share a
    label, so cross-class hits mean either real label noise or an over-loose
    threshold. `max_group_size` is the chaining detector: union-find is
    transitive, so a loose threshold produces a few enormous blobs rather than
    many small groups. A max group size in the hundreds means the threshold has
    collapsed unrelated images together.
    """
    sizes = [len(g) for g in groups]
    within = sum(1 for g in groups if df.iloc[g]["class_name"].nunique() == 1)
    return {
        "n_groups": len(groups),
        "n_removed": sum(size - 1 for size in sizes),
        "n_within_class": within,
        "n_cross_class": len(groups) - within,
        "max_group_size": max(sizes) if sizes else 0,
    }
