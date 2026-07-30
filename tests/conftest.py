"""Shared fixtures.

`synthetic_dataset` builds a miniature stand-in for the Kaggle dataset:
same folder layout, same file naming, deterministic pixel content, and a
deliberately planted duplicate pair so de-duplication has something to find.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from covid_xray.config import CLASS_NAMES

# Small enough to keep tests fast, large enough that a 224 resize is a real
# resize and an 8x8 downsample is a real reduction.
FIXTURE_IMAGE_SIZE = 64

# Counts must survive a TWO-stage stratified split: 30% is held out, then
# halved into val/test. sklearn refuses to stratify a class with fewer than 2
# members, so the smallest class needs enough images that its 30% share is at
# least 2 — and comfortably more than that, since the allocation rounds.
# Mirrors the real imbalance (Normal largest, Viral Pneumonia smallest).
COUNTS = {"COVID": 14, "Lung_Opacity": 12, "Normal": 20, "Viral Pneumonia": 12}


def _deterministic_image(seed: int, size: int = FIXTURE_IMAGE_SIZE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size), dtype=np.uint8)


def _centre_mask(size: int = FIXTURE_IMAGE_SIZE) -> np.ndarray:
    """A mask with a known 50% coverage, so attribution ratios are checkable."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[:, : size // 2] = 255
    return mask


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "COVID-19_Radiography_Dataset"
    seed = 0
    for class_name in CLASS_NAMES:
        (root / class_name / "images").mkdir(parents=True)
        (root / class_name / "masks").mkdir(parents=True)
        for index in range(1, COUNTS[class_name] + 1):
            stem = f"{class_name}-{index}"
            Image.fromarray(_deterministic_image(seed)).save(
                root / class_name / "images" / f"{stem}.png"
            )
            Image.fromarray(_centre_mask()).save(root / class_name / "masks" / f"{stem}.png")
            seed += 1

    # Plant an exact duplicate: Normal-8 becomes a byte-identical copy of COVID-1.
    source = Image.open(root / "COVID" / "images" / "COVID-1.png")
    source.save(root / "Normal" / "images" / "Normal-8.png")

    return root


@pytest.fixture
def fixture_counts() -> dict[str, int]:
    """Per-class counts, so tests assert against the fixture rather than a
    hard-coded total that silently rots when COUNTS changes."""
    return dict(COUNTS)


@pytest.fixture
def centre_mask_array() -> np.ndarray:
    return _centre_mask()
