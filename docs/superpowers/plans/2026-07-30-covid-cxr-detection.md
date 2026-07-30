# COVID-19 CXR Detection with Confound Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 4-class chest X-ray classifier (Normal / Lung Opacity / Viral Pneumonia / COVID-19) and measure how much of its accuracy comes from lung pathology rather than from artefacts of how the dataset was assembled.

**Architecture:** A thin `src/covid_xray/` package holds all logic — de-duplication, split manifests, the `tf.data` input pipeline with four input variants, model builders, a two-stage training loop, the metric suite, and Grad-CAM. Four notebooks orchestrate: EDA/dedup/split, training, evaluation, and the confound audit. Every experiment is one YAML config against the same code path, so the raw baseline, the lung-masked model, and the two shortcut probes differ only by an `variant` string.

**Tech Stack:** Python 3.11+, TensorFlow/Keras 3, scikit-learn, ImageHash, Pillow, pandas, matplotlib, pytest. Training on Colab GPU; EDA, de-duplication and analysis on the local CPU machine.

**Spec:** [2026-07-30-covid-cxr-detection-design.md](../specs/2026-07-30-covid-cxr-detection-design.md)

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Code delivery:** implementation code is **not written to disk by the assistant**. This plan carries the full code; the engineer copies each block to the stated path. README and requirements files are the only exception and may be written directly.
- **Four classes**, canonical order (alphabetical, matching the dataset's folder names, giving stable label indices): `COVID` (0), `Lung_Opacity` (1), `Normal` (2), `Viral Pneumonia` (3). Note the space in `Viral Pneumonia`.
- **Image size 224×224**, bilinear for images, **nearest-neighbour for masks**.
- **Normalisation is `keras.applications.densenet.preprocess_input` only.** Never hand-roll ImageNet mean/std.
- **Global seed 42**, recorded in every config and set in every notebook.
- **Split 70/15/15 stratified**, produced once, committed to `data/splits/*.csv`, and never regenerated after Task 4.
- **The test split is not read** until Task 11. Tasks 5–9 use train and validation only.
- **Class imbalance is handled by per-sample weights only** — no oversampling, because stacking both distorts the calibration analysis in Task 10.
- **BatchNorm stays in inference mode during fine-tuning** (see Task 7). This is the single most common way a Keras fine-tune silently degrades.
- **No patient IDs exist in this dataset**, so patient-disjoint splitting is impossible. Near-duplicate removal is a partial mitigation. Every reported metric is an upper bound and must be labelled as such.
- **Commit after every task.** The engineer runs all `git` commands.

---

## File Structure

```
src/covid_xray/
├── __init__.py          # package marker, version
├── config.py            # RunConfig dataclass + YAML loader           [Task 1]
├── dedup.py             # exact + perceptual hashing, duplicate groups [Task 2]
├── splits.py            # stratified split, manifest read/write        [Task 3]
├── data.py              # tf.data pipeline, the four input variants    [Task 5]
├── augment.py           # augmentation layer stack                     [Task 6]
├── models.py            # DenseNet121 builder, 8×8 probe               [Task 7]
├── train.py             # two-stage loop, class weights, callbacks     [Task 8]
├── evaluate.py          # metrics, bootstrap CIs, restricted pair      [Task 10]
└── gradcam.py           # Grad-CAM, Lung Attribution Ratio            [Task 12]

tests/
├── conftest.py          # synthetic 4-class fixture dataset            [Task 1]
├── test_config.py       [Task 1]     ├── test_augment.py   [Task 6]
├── test_dedup.py        [Task 2]     ├── test_models.py    [Task 7]
├── test_splits.py       [Task 3]     ├── test_train.py     [Task 8]
├── test_data.py         [Task 5]     ├── test_evaluate.py  [Task 10]
└── test_gradcam.py      [Task 12]

configs/run1_raw.yaml, run2_masked.yaml, run3_probe8.yaml, run4_lungs_removed.yaml
notebooks/01_eda_and_dedup, 02_train, 03_evaluate, 04_gradcam_audit
```

**Deviation from spec §8, deliberate:** the spec folded de-duplication and splitting into `data.py`. They are split out here because they run **offline on CPU without TensorFlow** — the local 7.6 GB machine can run all of Tasks 2–4 before TF is ever installed, and keeping them TF-free means the EDA notebook imports in seconds. `data.py` retains one job: turning a manifest into a `tf.data.Dataset`.

---

## Task 1: Package scaffold, config, and test fixtures

**Files:**
- Create: `requirements.txt`, `src/covid_xray/__init__.py`, `src/covid_xray/config.py`, `tests/conftest.py`, `tests/test_config.py`, `pytest.ini`

**Interfaces:**
- Consumes: nothing
- Produces: `RunConfig` frozen dataclass with fields `name, variant, model, image_size, batch_size, seed, dropout, stage_a_epochs, stage_a_lr, stage_a_patience, stage_b_epochs, stage_b_lr, stage_b_patience, unfreeze_from, mixed_precision`; `load_config(path: str | Path) -> RunConfig`; module constants `CLASS_NAMES: tuple[str, ...]`, `VARIANTS: tuple[str, ...]`, `DATA_ROOT_ENV: str`. The `synthetic_dataset` pytest fixture returns a `Path` to a fake 4-class dataset with `images/` and `masks/` subfolders.

- [ ] **Step 1: Write `requirements.txt`**

```
# Core numerics — needed for every task
numpy>=1.26
pandas>=2.0
scikit-learn>=1.4
Pillow>=10.0
matplotlib>=3.8
PyYAML>=6.0
tqdm>=4.66

# De-duplication (Task 2)
ImageHash>=4.3

# Deep learning (Tasks 5-13).
# Colab already provides TensorFlow — do NOT pip install it there, it will
# break the preinstalled CUDA stack. Install locally only if you want to run
# the TF-dependent tests on CPU.
tensorflow>=2.19
keras>=3.4

# Testing
pytest>=8.0
```

- [ ] **Step 2: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = src
filterwarnings =
    ignore::DeprecationWarning
markers =
    tf: tests that require TensorFlow (deselect with '-m "not tf"')
```

- [ ] **Step 3: Write the failing test — `tests/test_config.py`**

```python
import pytest
import yaml

from covid_xray.config import CLASS_NAMES, VARIANTS, RunConfig, load_config


def test_class_names_are_canonical_and_ordered():
    assert CLASS_NAMES == ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"name": "run1_raw", "variant": "raw", "model": "densenet121"}))

    cfg = load_config(path)

    assert cfg.name == "run1_raw"
    assert cfg.variant == "raw"
    assert cfg.model == "densenet121"


def test_load_config_applies_spec_defaults(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"name": "r", "variant": "raw", "model": "densenet121"}))

    cfg = load_config(path)

    assert cfg.image_size == 224
    assert cfg.batch_size == 32
    assert cfg.seed == 42
    assert cfg.stage_a_lr == 1e-3
    assert cfg.stage_b_lr == 1e-5
    assert cfg.stage_a_patience == 3
    assert cfg.stage_b_patience == 5
    assert cfg.unfreeze_from == "conv5_block1"


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="variant"):
        RunConfig(name="bad", variant="greyscale", model="densenet121")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="model"):
        RunConfig(name="bad", variant="raw", model="resnet50")


def test_config_is_immutable():
    cfg = RunConfig(name="r", variant="raw", model="densenet121")
    with pytest.raises(Exception):
        cfg.seed = 7


def test_all_four_variants_are_declared():
    assert VARIANTS == ("raw", "masked", "lungs_removed", "downsample8")
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'covid_xray'`

- [ ] **Step 5: Write `src/covid_xray/__init__.py`**

```python
"""COVID-19 chest X-ray classification with a dataset-confound audit."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Write `src/covid_xray/config.py`**

```python
"""Run configuration.

One YAML file per experiment. All four runs share a single code path and
differ only by `variant` and `model`, which keeps the comparison honest —
no run gets an accidental advantage from a different pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

# Alphabetical, and identical to the dataset's folder names. Fixing the order
# here fixes the label indices everywhere: COVID=0, Lung_Opacity=1,
# Normal=2, Viral Pneumonia=3. Note the space in the last name.
CLASS_NAMES: tuple[str, ...] = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")

# The four input transforms. See spec section 5.
#   raw            — unmodified image (run 1, headline)
#   masked         — everything outside the lung mask zeroed (run 2)
#   lungs_removed  — everything INSIDE the lung mask zeroed (run 4, probe)
#   downsample8    — raw image at 8x8, fed to a linear model (run 3, probe)
VARIANTS: tuple[str, ...] = ("raw", "masked", "lungs_removed", "downsample8")

MODELS: tuple[str, ...] = ("densenet121", "logreg8")

# Environment variable holding the dataset root, so the same manifest CSVs
# work unchanged on the local machine and on Colab.
DATA_ROOT_ENV = "COVID_XRAY_DATA_ROOT"

# The audit's control pair: both classes are adult, so the pediatric confound
# is absent and any remaining separability is source artefact or pathology.
CONTROL_PAIR: tuple[str, str] = ("COVID", "Lung_Opacity")


@dataclass(frozen=True)
class RunConfig:
    name: str
    variant: str
    model: str
    image_size: int = 224
    batch_size: int = 32
    seed: int = 42
    dropout: float = 0.3
    stage_a_epochs: int = 8
    stage_a_lr: float = 1e-3
    stage_a_patience: int = 3
    stage_b_epochs: int = 15
    stage_b_lr: float = 1e-5
    stage_b_patience: int = 5
    unfreeze_from: str = "conv5_block1"
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; expected one of {VARIANTS}")
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; expected one of {MODELS}")
        if self.image_size < 32:
            raise ValueError(f"image_size {self.image_size} is implausibly small")

    @property
    def num_classes(self) -> int:
        return len(CLASS_NAMES)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    """Load a run config from YAML, applying spec defaults for absent keys."""
    with open(path) as handle:
        payload = yaml.safe_load(handle) or {}
    return RunConfig(**payload)
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS, 7 tests

- [ ] **Step 8: Write `tests/conftest.py`**

This fixture is used by Tasks 2, 3, 5 and 12. It mirrors the real dataset's
layout exactly — `<class>/images/*.png` and `<class>/masks/*.png` — so tests
exercise the same path-handling code the real data will.

```python
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
COUNTS = {"COVID": 6, "Lung_Opacity": 5, "Normal": 8, "Viral Pneumonia": 4}


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
def centre_mask_array() -> np.ndarray:
    return _centre_mask()
```

- [ ] **Step 9: Verify the fixture builds**

Run: `pytest tests/ -v --collect-only`
Expected: collection succeeds, 7 tests listed, no import errors.

- [ ] **Step 10: Commit**

```bash
git add requirements.txt pytest.ini src/covid_xray/__init__.py src/covid_xray/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: package scaffold, run config, and synthetic test fixture"
```

---

## Task 2: De-duplication

The dataset was aggregated from 43 publications plus SIRM, so duplicates exist.
A duplicate spanning a train/test boundary inflates every metric in the report.

**Files:**
- Create: `src/covid_xray/dedup.py`, `tests/test_dedup.py`

**Interfaces:**
- Consumes: `covid_xray.config.CLASS_NAMES`
- Produces: `scan_dataset(root: Path) -> pd.DataFrame` with columns `path, mask_path, class_name, label, md5, phash` (paths relative to `root`); `find_duplicate_groups(df: pd.DataFrame, max_distance: int = 5) -> list[list[int]]` returning positional index groups; `drop_duplicates(df: pd.DataFrame, groups: list[list[int]]) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(kept, removed)` where `removed` carries a `duplicate_of` column; `summarise_duplicates(df, groups) -> dict` with keys `n_groups, n_removed, n_within_class, n_cross_class`.

- [ ] **Step 1: Write the failing test — `tests/test_dedup.py`**

```python
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


def test_scan_finds_every_image_and_pairs_its_mask(synthetic_dataset):
    df = scan_dataset(synthetic_dataset)

    assert len(df) == 23  # 6 + 5 + 8 + 4
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
    groups = find_duplicate_groups(df, max_distance=5)

    paired = {frozenset(df.iloc[g]["path"].tolist()) for g in groups}
    expected = frozenset({"COVID/images/COVID-1.png", "Normal/images/Normal-8.png"})
    assert expected in paired


def test_drop_duplicates_keeps_one_member_per_group(synthetic_dataset):
    df = scan_dataset(synthetic_dataset)
    groups = find_duplicate_groups(df, max_distance=5)

    kept, removed = drop_duplicates(df, groups)

    assert len(kept) + len(removed) == len(df)
    assert len(kept) == len(df) - sum(len(g) - 1 for g in groups)
    assert "duplicate_of" in removed.columns


def test_retained_member_is_first_by_sorted_filename(synthetic_dataset):
    """Determinism requirement from spec section 2 — not filesystem order."""
    df = scan_dataset(synthetic_dataset)
    groups = find_duplicate_groups(df, max_distance=5)
    kept, _ = drop_duplicates(df, groups)

    # COVID/images/COVID-1.png sorts before Normal/images/Normal-8.png
    assert "COVID/images/COVID-1.png" in set(kept["path"])
    assert "Normal/images/Normal-8.png" not in set(kept["path"])


def test_summary_separates_within_class_from_cross_class(synthetic_dataset):
    df = scan_dataset(synthetic_dataset)
    groups = find_duplicate_groups(df, max_distance=5)

    summary = summarise_duplicates(df, groups)

    assert summary["n_removed"] == sum(len(g) - 1 for g in groups)
    # The planted pair spans COVID and Normal, so at least one cross-class group.
    assert summary["n_cross_class"] >= 1
    assert summary["n_within_class"] + summary["n_cross_class"] == summary["n_groups"]


def test_dedup_is_reproducible(synthetic_dataset):
    first = drop_duplicates(*_scan_and_group(synthetic_dataset))[0]["path"].tolist()
    second = drop_duplicates(*_scan_and_group(synthetic_dataset))[0]["path"].tolist()
    assert first == second


def _scan_and_group(root):
    df = scan_dataset(root)
    return df, find_duplicate_groups(df, max_distance=5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_dedup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'covid_xray.dedup'`

- [ ] **Step 3: Write `src/covid_xray/dedup.py`**

```python
"""Exact and near-duplicate detection.

Runs before splitting, over the whole four-class pool. Pure CPU, no
TensorFlow — the 21k-image scan fits comfortably on the local machine.

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


def find_duplicate_groups(df: pd.DataFrame, max_distance: int = 5) -> list[list[int]]:
    """Group images that are byte-identical or perceptually within ``max_distance``.

    Returns positional index groups of size >= 2, each sorted ascending.
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
    """Counts for the README. Cross-class groups are label noise, not just waste."""
    within = sum(1 for g in groups if df.iloc[g]["class_name"].nunique() == 1)
    return {
        "n_groups": len(groups),
        "n_removed": sum(len(g) - 1 for g in groups),
        "n_within_class": within,
        "n_cross_class": len(groups) - within,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_dedup.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add src/covid_xray/dedup.py tests/test_dedup.py
git commit -m "feat: exact and perceptual de-duplication with deterministic retention"
```

---

## Task 3: Split manifests

**Files:**
- Create: `src/covid_xray/splits.py`, `tests/test_splits.py`

**Interfaces:**
- Consumes: `covid_xray.config.CLASS_NAMES`
- Produces: `stratified_split(df: pd.DataFrame, seed: int = 42, val_fraction: float = 0.15, test_fraction: float = 0.15) -> pd.DataFrame` adding a `split` column with values `train|val|test`; `write_manifests(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]`; `load_manifest(path: str | Path) -> pd.DataFrame`; `class_distribution(df) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test — `tests/test_splits.py`**

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'covid_xray.splits'`

- [ ] **Step 3: Write `src/covid_xray/splits.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_splits.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add src/covid_xray/splits.py tests/test_splits.py
git commit -m "feat: class-stratified split with committed manifest CSVs"
```

---

## Task 4: Notebook 01 — download, EDA, de-duplication, split

This task produces the committed split manifests. **It runs once.** After the
manifests are committed, this notebook is never re-run with a different seed.

**Files:**
- Create: `notebooks/01_eda_and_dedup.ipynb`
- Produces: `data/splits/{train,val,test}.csv`, `reports/figures/class_distribution.png`, `reports/duplicates.csv`

- [ ] **Step 1: Cell 1 — environment and dataset download**

```python
# If on Colab: pip install the CPU-only extras. TF is already present.
# !pip install -q ImageHash pyyaml

import os
from pathlib import Path

# Point this at wherever the Kaggle dataset was unpacked.
# Kaggle CLI:  kaggle datasets download -d tawsifurrahman/covid19-radiography-database
#              unzip -q covid19-radiography-database.zip -d data/raw
DATA_ROOT = Path(os.environ.get("COVID_XRAY_DATA_ROOT", "data/raw/COVID-19_Radiography_Dataset"))
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

assert DATA_ROOT.exists(), f"dataset not found at {DATA_ROOT}"
sorted(p.name for p in DATA_ROOT.iterdir())
```

- [ ] **Step 2: Cell 2 — verify the layout matches the spec's assumptions**

Spec section 2 lists three things to confirm at download time. This cell checks
all three and prints the answers. **Record the output in the README.**

```python
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image
from covid_xray.config import CLASS_NAMES

for class_name in CLASS_NAMES:
    images = sorted((DATA_ROOT / class_name / "images").glob("*.png"))
    masks = sorted((DATA_ROOT / class_name / "masks").glob("*.png"))
    with Image.open(images[0]) as im, Image.open(masks[0]) as mk:
        print(f"{class_name:18s} images={len(images):6d} masks={len(masks):6d} "
              f"image_size={im.size} image_mode={im.mode} "
              f"mask_size={mk.size} mask_mode={mk.mode}")
```

Expected, approximately: COVID 3616, Lung_Opacity 6012, Normal 10192,
Viral Pneumonia 1345. Image size 299×299, mask size likely 256×256 — the
mismatch is expected and is handled by nearest-neighbour resizing in Task 5.
**If counts differ materially, stop and update the spec before continuing.**

- [ ] **Step 3: Cell 3 — scan and hash (slow: ~5-10 min for 21k images)**

```python
from covid_xray.dedup import scan_dataset

catalogue = scan_dataset(DATA_ROOT)
print(f"{len(catalogue)} images catalogued")
catalogue.groupby("class_name").size()
```

- [ ] **Step 4: Cell 4 — find and summarise duplicates**

```python
from covid_xray.dedup import find_duplicate_groups, drop_duplicates, summarise_duplicates

groups = find_duplicate_groups(catalogue, max_distance=5)
summary = summarise_duplicates(catalogue, groups)
print(summary)

kept, removed = drop_duplicates(catalogue, groups)
removed.drop(columns=["phash"]).to_csv(REPO_ROOT / "reports" / "duplicates.csv", index=False)
print(f"{len(catalogue)} -> {len(kept)} after de-duplication")
```

- [ ] **Step 5: Cell 5 — inspect the cross-class duplicates by eye**

Cross-class duplicates are label noise and belong in the README as a dataset
finding. Look at a few before believing the number.

```python
import matplotlib.pyplot as plt

cross = [g for g in groups if catalogue.iloc[g]["class_name"].nunique() > 1]
print(f"{len(cross)} cross-class duplicate groups")

for group in cross[:3]:
    rows = catalogue.iloc[group]
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4))
    axes = [axes] if len(rows) == 1 else axes
    for ax, (_, row) in zip(axes, rows.iterrows()):
        ax.imshow(plt.imread(DATA_ROOT / row["path"]), cmap="gray")
        ax.set_title(row["class_name"]); ax.axis("off")
    plt.show()
```

- [ ] **Step 6: Cell 6 — split and write manifests**

```python
from covid_xray.splits import stratified_split, write_manifests, class_distribution

SEED = 42
split_df = stratified_split(kept, seed=SEED)
written = write_manifests(split_df, REPO_ROOT / "data" / "splits")

distribution = class_distribution(split_df)
print(distribution)
distribution.to_csv(REPO_ROOT / "reports" / "class_distribution.csv")
written
```

- [ ] **Step 7: Cell 7 — class distribution figure**

```python
ax = distribution.plot(kind="bar", stacked=True, figsize=(8, 4))
ax.set_ylabel("images"); ax.set_title("Class distribution after de-duplication")
plt.tight_layout()
plt.savefig(REPO_ROOT / "reports" / "figures" / "class_distribution.png", dpi=150)
plt.show()
```

- [ ] **Step 8: Verify the manifests**

Run:
```bash
wc -l data/splits/*.csv
python -c "
import pandas as pd, itertools
frames = {s: pd.read_csv(f'data/splits/{s}.csv') for s in ['train','val','test']}
paths = {s: set(d['path']) for s, d in frames.items()}
assert not (paths['train'] & paths['test']), 'train/test overlap'
assert not (paths['train'] & paths['val']),  'train/val overlap'
assert not (paths['val']   & paths['test']), 'val/test overlap'
for s, d in frames.items():
    print(s, len(d), dict(d['class_name'].value_counts()))
"
```
Expected: no overlap, all four classes present in every split, totals summing
to the post-de-duplication count.

- [ ] **Step 9: Commit**

The figure is committed because the README references it; the raw images are not.

```bash
git add notebooks/01_eda_and_dedup.ipynb data/splits/train.csv data/splits/val.csv data/splits/test.csv
git add -f reports/figures/class_distribution.png reports/class_distribution.csv
git commit -m "feat: EDA notebook, de-duplication pass, and committed split manifests"
```

---

## Task 5: Input pipeline and the four variants

**Files:**
- Create: `src/covid_xray/data.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `RunConfig`, `CLASS_NAMES`, `load_manifest`
- Produces: `apply_variant(image: tf.Tensor, mask: tf.Tensor, variant: str) -> tf.Tensor`; `build_dataset(manifest, data_root, cfg, *, training: bool, augmenter=None, cache_path: str | None = None) -> tf.data.Dataset` yielding `(image, one_hot_label, sample_weight)` when `training=True` and `(image, one_hot_label)` otherwise; `extract_downsampled_features(manifest, data_root, size: int = 8) -> tuple[np.ndarray, np.ndarray]`; `compute_sample_weights(labels: np.ndarray, num_classes: int) -> np.ndarray`.

**Why sample weights rather than Keras `class_weight`:** targets are one-hot so
`F1Score` works, and Keras's `class_weight` is only reliably applied to integer
targets. Per-sample weights carried in the dataset are unambiguous with either.

- [ ] **Step 1: Write the failing test — `tests/test_data.py`**

```python
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from covid_xray.config import RunConfig
from covid_xray.data import (
    apply_variant,
    build_dataset,
    compute_sample_weights,
    extract_downsampled_features,
)
from covid_xray.dedup import scan_dataset
from covid_xray.splits import stratified_split


@pytest.fixture
def manifest(synthetic_dataset):
    df = scan_dataset(synthetic_dataset).drop(columns=["phash"])
    return stratified_split(df, seed=42)


def _image_and_mask():
    image = tf.ones((8, 8, 1), tf.float32) * 100.0
    mask = tf.concat([tf.ones((8, 4, 1)), tf.zeros((8, 4, 1))], axis=1)
    return image, mask


def test_raw_variant_is_identity():
    image, mask = _image_and_mask()
    assert np.allclose(apply_variant(image, mask, "raw").numpy(), image.numpy())


def test_masked_variant_zeroes_outside_the_lungs():
    image, mask = _image_and_mask()
    result = apply_variant(image, mask, "masked").numpy()
    assert np.all(result[:, :4] == 100.0)
    assert np.all(result[:, 4:] == 0.0)


def test_lungs_removed_is_the_exact_inverse_of_masked():
    image, mask = _image_and_mask()
    masked = apply_variant(image, mask, "masked").numpy()
    removed = apply_variant(image, mask, "lungs_removed").numpy()
    assert np.allclose(masked + removed, image.numpy())


def test_unknown_variant_raises():
    image, mask = _image_and_mask()
    with pytest.raises(ValueError):
        apply_variant(image, mask, "sepia")


def test_training_dataset_yields_three_element_tuples(manifest, synthetic_dataset):
    cfg = RunConfig(name="t", variant="raw", model="densenet121", batch_size=2, image_size=32)
    ds = build_dataset(manifest[manifest.split == "train"], synthetic_dataset, cfg, training=True)

    batch = next(iter(ds))
    assert len(batch) == 3
    images, labels, weights = batch
    assert images.shape == (2, 32, 32, 3)
    assert labels.shape == (2, 4)
    assert weights.shape == (2,)


def test_eval_dataset_yields_two_element_tuples(manifest, synthetic_dataset):
    cfg = RunConfig(name="t", variant="raw", model="densenet121", batch_size=2, image_size=32)
    ds = build_dataset(manifest[manifest.split == "val"], synthetic_dataset, cfg, training=False)
    assert len(next(iter(ds))) == 2


def test_eval_dataset_order_matches_the_manifest(manifest, synthetic_dataset):
    """Task 10 pairs predictions with manifest rows positionally — order must hold."""
    subset = manifest[manifest.split == "val"].reset_index(drop=True)
    cfg = RunConfig(name="t", variant="raw", model="densenet121", batch_size=2, image_size=32)
    ds = build_dataset(subset, synthetic_dataset, cfg, training=False)

    labels = np.concatenate([y.numpy().argmax(axis=1) for _, y in ds])
    assert labels.tolist() == subset["label"].tolist()


def test_images_are_three_channel_after_preprocessing(manifest, synthetic_dataset):
    cfg = RunConfig(name="t", variant="masked", model="densenet121", batch_size=2, image_size=32)
    ds = build_dataset(manifest[manifest.split == "val"], synthetic_dataset, cfg, training=False)
    images, _ = next(iter(ds))
    assert images.shape[-1] == 3
    # densenet preprocess_input is torch-mode: roughly zero-centred, not 0..255
    assert images.numpy().max() < 10.0


def test_sample_weights_are_inverse_to_frequency():
    labels = np.array([0, 0, 0, 0, 1])
    weights = compute_sample_weights(labels, num_classes=2)
    assert weights[4] > weights[0]
    assert weights.mean() == pytest.approx(1.0, abs=1e-6)


def test_downsampled_features_have_expected_shape(manifest, synthetic_dataset):
    X, y = extract_downsampled_features(manifest[manifest.split == "train"], synthetic_dataset, size=8)
    assert X.shape[1] == 64
    assert X.shape[0] == y.shape[0]
    assert set(np.unique(y)) <= {0, 1, 2, 3}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'covid_xray.data'`
(or SKIP entirely if TensorFlow is not installed locally — that is acceptable;
these tests then run on Colab in Task 9.)

- [ ] **Step 3: Write `src/covid_xray/data.py`**

```python
"""Input pipeline.

One code path serves all four experiments. The only thing that differs is
`cfg.variant`, which keeps the runs comparable — no run can accidentally
benefit from a different resize, normalisation or batching strategy.

Pipeline order matters and is deliberate:
    decode -> resize -> variant -> cache(uint8) -> batch -> augment -> preprocess
Caching as uint8 after the variant is applied keeps the cache ~1 GB for the
full dataset instead of ~4 GB as float32, and caching *before* augmentation
keeps augmentation random per epoch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from covid_xray.config import CLASS_NAMES, RunConfig

AUTOTUNE = tf.data.AUTOTUNE


def apply_variant(image: tf.Tensor, mask: tf.Tensor, variant: str) -> tf.Tensor:
    """Apply one of the four input transforms.

    `image` is float32 in 0..255 with shape (H, W, 1); `mask` is float32 in
    0..1 with the same shape.
    """
    if variant in ("raw", "downsample8"):
        return image
    if variant == "masked":
        return image * mask
    if variant == "lungs_removed":
        return image * (1.0 - mask)
    raise ValueError(f"unknown variant {variant!r}")


def compute_sample_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1.

    Mean-1 normalisation keeps the loss on the same scale as an unweighted run,
    so learning rates transfer between variants without retuning.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    per_class = counts.sum() / (num_classes * counts)
    weights = per_class[labels]
    return (weights / weights.mean()).astype(np.float32)


def _decode(path: tf.Tensor, size: int, channels: int, method: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    decoded = tf.io.decode_png(raw, channels=channels)
    decoded = tf.image.convert_image_dtype(decoded, tf.float32) * 255.0
    return tf.image.resize(decoded, (size, size), method=method)


def build_dataset(
    manifest: pd.DataFrame,
    data_root: str | Path,
    cfg: RunConfig,
    *,
    training: bool,
    augmenter: tf.keras.Model | None = None,
    cache_path: str | None = None,
) -> tf.data.Dataset:
    """Turn a manifest into a batched dataset.

    Yields `(image, one_hot, sample_weight)` when `training` is True and
    `(image, one_hot)` otherwise. Evaluation datasets preserve manifest row
    order so predictions can be paired positionally with manifest rows.
    """
    data_root = Path(data_root)
    manifest = manifest.reset_index(drop=True)

    image_paths = [str(data_root / p) for p in manifest["path"]]
    mask_paths = [str(data_root / p) for p in manifest["mask_path"]]
    labels = manifest["label"].to_numpy(dtype=np.int32)

    needs_mask = cfg.variant in ("masked", "lungs_removed")
    size = cfg.image_size

    def load(image_path, mask_path, label, weight):
        image = _decode(image_path, size, channels=1, method="bilinear")
        if needs_mask:
            # Nearest-neighbour: masks are binary and must not be interpolated
            # into intermediate values at the lung boundary.
            mask = _decode(mask_path, size, channels=1, method="nearest")
            mask = tf.cast(mask > 127.0, tf.float32)
        else:
            mask = tf.ones_like(image)
        image = apply_variant(image, mask, cfg.variant)
        return tf.cast(image, tf.uint8), label, weight

    weights = (
        compute_sample_weights(labels, cfg.num_classes)
        if training
        else np.ones(len(labels), dtype=np.float32)
    )

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths, labels, weights))
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)

    if cache_path is not None:
        ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(buffer_size=min(len(manifest), 4096), seed=cfg.seed, reshuffle_each_iteration=True)

    ds = ds.batch(cfg.batch_size, drop_remainder=False)

    def finalise(image, label, weight):
        image = tf.cast(image, tf.float32)
        if augmenter is not None:
            image = augmenter(image, training=True)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.keras.applications.densenet.preprocess_input(image)
        one_hot = tf.one_hot(label, depth=cfg.num_classes)
        return (image, one_hot, weight) if training else (image, one_hot)

    return ds.map(finalise, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def extract_downsampled_features(
    manifest: pd.DataFrame, data_root: str | Path, size: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Flattened NxN grayscale features for the linear shortcut probe (run 3).

    Deliberately bypasses tf.data: the probe is a scikit-learn model and wants
    everything in memory at once. At 8x8 the whole dataset is ~1.4 MB.
    """
    data_root = Path(data_root)
    manifest = manifest.reset_index(drop=True)

    features = np.empty((len(manifest), size * size), dtype=np.float32)
    for position, relative_path in enumerate(manifest["path"]):
        with Image.open(data_root / relative_path) as image:
            small = image.convert("L").resize((size, size), Image.BILINEAR)
        features[position] = np.asarray(small, dtype=np.float32).reshape(-1) / 255.0

    return features, manifest["label"].to_numpy(dtype=np.int64)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_data.py -v`
Expected: PASS, 11 tests (or all SKIPPED if TF is absent locally)

- [ ] **Step 5: Commit**

```bash
git add src/covid_xray/data.py tests/test_data.py
git commit -m "feat: tf.data input pipeline with raw, masked, lungs-removed and 8x8 variants"
```

---

## Task 6: Augmentation

**Files:**
- Create: `src/covid_xray/augment.py`, `tests/test_augment.py`

**Interfaces:**
- Consumes: `RunConfig`
- Produces: `build_augmenter(cfg: RunConfig) -> keras.Sequential`

Applied to batched float32 images in 0..255, **before** `preprocess_input`.
That ordering is why `RandomBrightness` gets `value_range=(0, 255)`.

- [ ] **Step 1: Write the failing test — `tests/test_augment.py`**

```python
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from covid_xray.augment import build_augmenter
from covid_xray.config import RunConfig


@pytest.fixture
def augmenter():
    return build_augmenter(RunConfig(name="t", variant="raw", model="densenet121"))


def test_augmenter_preserves_shape(augmenter):
    batch = tf.random.uniform((4, 224, 224, 1), maxval=255.0, seed=0)
    assert augmenter(batch, training=True).shape == batch.shape


def test_augmentation_changes_the_image_in_training_mode(augmenter):
    batch = tf.random.uniform((4, 224, 224, 1), maxval=255.0, seed=0)
    assert not np.allclose(augmenter(batch, training=True).numpy(), batch.numpy())


def test_augmentation_is_identity_in_inference_mode(augmenter):
    batch = tf.random.uniform((4, 224, 224, 1), maxval=255.0, seed=0)
    assert np.allclose(augmenter(batch, training=False).numpy(), batch.numpy(), atol=1e-4)


def test_no_vertical_flip_layer_is_present(augmenter):
    """Spec section 4: lungs are not vertically symmetric."""
    flips = [l for l in augmenter.layers if isinstance(l, tf.keras.layers.RandomFlip)]
    assert flips, "expected a horizontal flip layer"
    for layer in flips:
        assert layer.mode == "horizontal"


def test_layer_stack_matches_the_spec(augmenter):
    names = [type(l).__name__ for l in augmenter.layers]
    assert names == [
        "RandomFlip",
        "RandomRotation",
        "RandomZoom",
        "RandomTranslation",
        "RandomContrast",
        "RandomBrightness",
    ]


def test_output_stays_within_the_valid_pixel_range(augmenter):
    batch = tf.random.uniform((8, 64, 64, 1), maxval=255.0, seed=0)
    out = augmenter(batch, training=True).numpy()
    assert out.min() >= -1e-3
    assert out.max() <= 255.0 + 1e-3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_augment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'covid_xray.augment'`

- [ ] **Step 3: Write `src/covid_xray/augment.py`**

```python
"""Training-time augmentation.

Keras preprocessing layers rather than albumentations: these compile into the
graph and run on-GPU, whereas albumentations requires a `tf.numpy_function`
bridge that becomes the pipeline bottleneck. Revisit only if CLAHE is needed.

Brightness and contrast jitter are included on purpose. Chest radiograph pixel
values are relative and already vary with exposure settings, so this reflects
genuine acquisition variance. (Hounsfield units, which are absolute, are a CT
concept and do not apply here.)

Excluded: vertical flips (lungs are not vertically symmetric), shear, and any
hue or saturation operation.
"""

from __future__ import annotations

from tensorflow import keras

from covid_xray.config import RunConfig

# Applied before preprocess_input, so images are still in 0..255.
PIXEL_RANGE = (0.0, 255.0)


def build_augmenter(cfg: RunConfig) -> keras.Sequential:
    """The six-layer augmentation stack from spec section 4."""
    return keras.Sequential(
        [
            # Anatomically acceptable, and disrupts the L/R laterality-marker
            # shortcut that DeGrave et al. identified in this dataset family.
            keras.layers.RandomFlip("horizontal", seed=cfg.seed),
            # 0.04 turns == +/- 14.4 degrees. Patient positioning variance.
            keras.layers.RandomRotation(0.04, fill_mode="constant", fill_value=0.0, seed=cfg.seed),
            keras.layers.RandomZoom(0.10, fill_mode="constant", fill_value=0.0, seed=cfg.seed),
            keras.layers.RandomTranslation(
                0.05, 0.05, fill_mode="constant", fill_value=0.0, seed=cfg.seed
            ),
            keras.layers.RandomContrast(0.2, seed=cfg.seed),
            keras.layers.RandomBrightness(0.15, value_range=PIXEL_RANGE, seed=cfg.seed),
        ],
        name="augmenter",
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_augment.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/covid_xray/augment.py tests/test_augment.py
git commit -m "feat: GPU-side augmentation stack per spec section 4"
```

---

## Task 7: Model builders

**Files:**
- Create: `src/covid_xray/models.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `RunConfig`
- Produces: `build_densenet121(cfg: RunConfig) -> keras.Model` (base sub-model is named `densenet121`, head output named `predictions`); `set_finetune_trainable(model: keras.Model, unfreeze_from: str) -> int` returning the number of unfrozen layers; `build_probe(seed: int = 42) -> sklearn.pipeline.Pipeline`; `split_feature_and_head(model) -> tuple[keras.Model, keras.Model]` used by Task 12.

**The BatchNorm requirement, concretely:** the base is called with
`training=False` inside the functional graph, which pins every BatchNorm layer
to inference mode permanently — including during Stage B when the surrounding
layers become trainable. `set_finetune_trainable` additionally skips
BatchNormalization layers when unfreezing, so their statistics *and* affine
parameters both stay fixed. Without this, unfreezing recomputes batch
statistics at batch size 32 and quietly degrades the model.

- [ ] **Step 1: Write the failing test — `tests/test_models.py`**

```python
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from tensorflow import keras

from covid_xray.config import RunConfig
from covid_xray.models import (
    build_densenet121,
    build_probe,
    set_finetune_trainable,
    split_feature_and_head,
)


@pytest.fixture(scope="module")
def cfg():
    return RunConfig(name="t", variant="raw", model="densenet121", mixed_precision=False)


@pytest.fixture(scope="module")
def model(cfg):
    return build_densenet121(cfg)


def test_output_shape_is_four_classes(model, cfg):
    batch = tf.zeros((2, cfg.image_size, cfg.image_size, 3))
    assert model(batch, training=False).shape == (2, 4)


def test_outputs_are_a_probability_distribution(model, cfg):
    batch = tf.random.uniform((2, cfg.image_size, cfg.image_size, 3), seed=0)
    probs = model(batch, training=False).numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert (probs >= 0).all()


def test_base_is_frozen_immediately_after_build(model):
    base = model.get_layer("densenet121")
    assert base.trainable is False


def test_stage_b_unfreezes_from_the_named_block(model):
    count = set_finetune_trainable(model, "conv5_block1")
    base = model.get_layer("densenet121")

    assert count > 0
    assert base.get_layer("conv1/conv").trainable is False
    assert base.get_layer("conv5_block16_concat") is not None


def test_batchnorm_stays_frozen_after_unfreezing(model):
    """The single most common silent Keras fine-tuning bug."""
    set_finetune_trainable(model, "conv5_block1")
    base = model.get_layer("densenet121")
    bn_layers = [l for l in base.layers if isinstance(l, keras.layers.BatchNormalization)]

    assert bn_layers, "expected BatchNormalization layers in DenseNet121"
    assert all(not l.trainable for l in bn_layers)


def test_head_is_always_trainable(model):
    dense = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
    assert dense and all(l.trainable for l in dense)


def test_split_feature_and_head_reproduces_the_full_model(model, cfg):
    feature_model, head_model = split_feature_and_head(model)
    batch = tf.random.uniform((2, cfg.image_size, cfg.image_size, 3), seed=1)

    direct = model(batch, training=False).numpy()
    staged = head_model(feature_model(batch, training=False), training=False).numpy()

    assert np.allclose(direct, staged, atol=1e-4)


def test_feature_model_emits_a_spatial_map(model, cfg):
    feature_model, _ = split_feature_and_head(model)
    batch = tf.zeros((1, cfg.image_size, cfg.image_size, 3))
    features = feature_model(batch, training=False)
    assert len(features.shape) == 4
    assert features.shape[1] == features.shape[2] == cfg.image_size // 32


def test_probe_fits_and_predicts_four_classes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 64))
    y = rng.integers(0, 4, size=80)

    probe = build_probe(seed=42).fit(X, y)
    probs = probe.predict_proba(X)

    assert probs.shape == (80, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'covid_xray.models'`

- [ ] **Step 3: Write `src/covid_xray/models.py`**

```python
"""Model builders.

DenseNet121 for runs 1, 2 and 4; a multinomial logistic regression for the
8x8 shortcut probe (run 3).

DenseNet121 was chosen over a from-scratch CNN and over ResNet for the usual
medical-imaging reason — dense feature reuse works well on the low-contrast,
texture-heavy structure of chest radiographs — but the specific architecture
is not the point of this project and is not ablated.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from covid_xray.config import RunConfig

BASE_NAME = "densenet121"
# DenseNet121's final activation, shape (7, 7, 1024) at 224x224 input.
# This is the Grad-CAM target layer.
FEATURE_LAYER = "relu"


def build_densenet121(cfg: RunConfig) -> keras.Model:
    """ImageNet-pretrained DenseNet121 with a fresh 4-class head.

    The base is invoked with `training=False`, which pins every BatchNorm layer
    to inference mode for the lifetime of the model — including Stage B, when
    the convolutional blocks become trainable.
    """
    inputs = keras.Input(shape=(cfg.image_size, cfg.image_size, 3), name="image")

    base = keras.applications.DenseNet121(
        include_top=False, weights="imagenet", input_shape=(cfg.image_size, cfg.image_size, 3)
    )
    base._name = BASE_NAME
    base.trainable = False  # Stage A: head only

    features = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(features)
    x = keras.layers.Dropout(cfg.dropout, name="dropout")(x)
    # float32 output regardless of the mixed-precision policy: softmax in
    # float16 loses precision in the tails, which corrupts the calibration
    # analysis in Task 10.
    outputs = keras.layers.Dense(
        cfg.num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)

    return keras.Model(inputs, outputs, name=f"{cfg.name}_densenet121")


def set_finetune_trainable(model: keras.Model, unfreeze_from: str) -> int:
    """Stage B: unfreeze the base from `unfreeze_from` onward, BatchNorm excepted.

    Returns the number of layers made trainable.
    """
    base = model.get_layer(BASE_NAME)
    base.trainable = True

    names = [layer.name for layer in base.layers]
    matches = [i for i, name in enumerate(names) if name.startswith(unfreeze_from)]
    if not matches:
        raise ValueError(f"no layer starting with {unfreeze_from!r} in {BASE_NAME}")
    cutoff = matches[0]

    unfrozen = 0
    for index, layer in enumerate(base.layers):
        if index < cutoff or isinstance(layer, keras.layers.BatchNormalization):
            # BatchNorm is frozen everywhere. `training=False` at the call site
            # already stops statistics updates; freezing here also pins gamma
            # and beta, so the normalisation is fully fixed.
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen += 1

    return unfrozen


def split_feature_and_head(model: keras.Model) -> tuple[keras.Model, keras.Model]:
    """Split into (image -> feature map) and (feature map -> probabilities).

    Grad-CAM needs gradients of the class score with respect to the last
    convolutional feature map. Because the base is a *nested* model, its
    internal tensors are not reachable from the outer graph, so the usual
    single-model Grad-CAM recipe fails. Splitting sidesteps that.
    """
    base = model.get_layer(BASE_NAME)
    feature_model = keras.Model(base.inputs, base.get_layer(FEATURE_LAYER).output, name="features")

    feature_shape = feature_model.output.shape[1:]
    head_input = keras.Input(shape=feature_shape, name="features_in")

    x = head_input
    base_position = model.layers.index(base)
    for layer in model.layers[base_position + 1 :]:
        x = layer(x)

    return feature_model, keras.Model(head_input, x, name="head")


def build_probe(seed: int = 42) -> Pipeline:
    """Multinomial logistic regression for the 8x8 shortcut probe.

    Deliberately the weakest plausible model. If *this* separates the classes,
    the label is recoverable from global intensity structure alone and no
    amount of DenseNet accuracy means what it appears to mean.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    multi_class="multinomial",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS, 9 tests. First run downloads ImageNet weights (~33 MB).

> If `multi_class="multinomial"` emits a deprecation warning on your
> scikit-learn version, drop the argument — multinomial is the default for
> `lbfgs` in recent versions. Do not change the solver.

- [ ] **Step 5: Commit**

```bash
git add src/covid_xray/models.py tests/test_models.py
git commit -m "feat: DenseNet121 builder with frozen BatchNorm and 8x8 linear probe"
```

---

## Task 8: Two-stage training loop and run configs

**Files:**
- Create: `src/covid_xray/train.py`, `tests/test_train.py`, `configs/run1_raw.yaml`, `configs/run2_masked.yaml`, `configs/run3_probe8.yaml`, `configs/run4_lungs_removed.yaml`

**Interfaces:**
- Consumes: `RunConfig`, `build_densenet121`, `set_finetune_trainable`, `build_augmenter`, `build_dataset`
- Produces: `build_callbacks(cfg, output_dir, stage) -> list`; `train_two_stage(cfg, train_ds, val_ds, output_dir) -> tuple[keras.Model, dict]` where the dict has keys `stage_a`, `stage_b` each mapping to a history dict; `enable_mixed_precision(cfg) -> None`; `set_global_seed(seed) -> None`.

- [ ] **Step 1: Write the four configs**

`configs/run1_raw.yaml`:
```yaml
# Run 1 — headline number. Unmodified images.
name: run1_raw
variant: raw
model: densenet121
```

`configs/run2_masked.yaml`:
```yaml
# Run 2 — everything outside the lung mask zeroed. The accuracy gap against
# run 1 is the project's central finding.
name: run2_masked
variant: masked
model: densenet121
```

`configs/run3_probe8.yaml`:
```yaml
# Run 3 — shortcut probe. If a linear model on 64 pixels separates the
# classes, the label is readable from global intensity alone.
name: run3_probe8
variant: downsample8
model: logreg8
```

`configs/run4_lungs_removed.yaml`:
```yaml
# Run 4 — shortcut probe. Exact inverse of run 2: the lungs are erased and
# everything else retained. High accuracy here means the pathology was never
# necessary.
name: run4_lungs_removed
variant: lungs_removed
model: densenet121
```

- [ ] **Step 2: Write the failing test — `tests/test_train.py`**

```python
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from tensorflow import keras

from covid_xray.config import RunConfig, load_config
from covid_xray.train import build_callbacks, set_global_seed, train_two_stage


@pytest.fixture
def tiny_cfg():
    return RunConfig(
        name="tiny",
        variant="raw",
        model="densenet121",
        image_size=32,
        batch_size=4,
        stage_a_epochs=1,
        stage_b_epochs=1,
        mixed_precision=False,
    )


@pytest.fixture
def tiny_data(tiny_cfg):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 32, 32, 3)).astype(np.float32)
    y = tf.one_hot(rng.integers(0, 4, size=16), 4).numpy()
    w = np.ones(16, dtype=np.float32)
    train = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(4)
    val = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
    return train, val


def test_all_four_configs_load():
    for name in ("run1_raw", "run2_masked", "run3_probe8", "run4_lungs_removed"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg.name == name


def test_configs_cover_every_variant():
    variants = {load_config(f"configs/{n}.yaml").variant for n in
                ("run1_raw", "run2_masked", "run3_probe8", "run4_lungs_removed")}
    assert variants == {"raw", "masked", "downsample8", "lungs_removed"}


def test_callbacks_monitor_validation_macro_f1(tiny_cfg, tmp_path):
    callbacks = build_callbacks(tiny_cfg, tmp_path, stage="a")
    monitored = [c for c in callbacks if hasattr(c, "monitor")]
    assert monitored
    for callback in monitored:
        assert callback.monitor == "val_macro_f1"
        assert callback.mode == "max"


def test_callbacks_include_backup_for_colab_session_death(tiny_cfg, tmp_path):
    names = [type(c).__name__ for c in build_callbacks(tiny_cfg, tmp_path, stage="a")]
    assert "BackupAndRestore" in names
    assert "ModelCheckpoint" in names
    assert "EarlyStopping" in names
    assert "CSVLogger" in names


def test_early_stopping_patience_matches_the_stage(tiny_cfg, tmp_path):
    stop_a = [c for c in build_callbacks(tiny_cfg, tmp_path, "a")
              if isinstance(c, keras.callbacks.EarlyStopping)][0]
    stop_b = [c for c in build_callbacks(tiny_cfg, tmp_path, "b")
              if isinstance(c, keras.callbacks.EarlyStopping)][0]
    assert stop_a.patience == tiny_cfg.stage_a_patience
    assert stop_b.patience == tiny_cfg.stage_b_patience
    assert stop_a.restore_best_weights and stop_b.restore_best_weights


def test_two_stage_training_returns_both_histories(tiny_cfg, tiny_data, tmp_path):
    train_ds, val_ds = tiny_data
    model, history = train_two_stage(tiny_cfg, train_ds, val_ds, tmp_path)

    assert set(history) == {"stage_a", "stage_b"}
    assert "val_macro_f1" in history["stage_a"]
    assert "val_macro_f1" in history["stage_b"]


def test_base_is_trainable_after_stage_b(tiny_cfg, tiny_data, tmp_path):
    train_ds, val_ds = tiny_data
    model, _ = train_two_stage(tiny_cfg, train_ds, val_ds, tmp_path)
    assert model.get_layer("densenet121").trainable is True


def test_set_global_seed_makes_initialisation_reproducible():
    set_global_seed(42)
    first = tf.random.uniform((5,)).numpy()
    set_global_seed(42)
    second = tf.random.uniform((5,)).numpy()
    assert np.allclose(first, second)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'covid_xray.train'`

- [ ] **Step 4: Write `src/covid_xray/train.py`**

```python
"""Two-stage transfer-learning loop.

Stage A trains the new head against a frozen base at lr 1e-3. Stage B unfreezes
the top block at lr 1e-5. Selection is on validation macro-F1 rather than
accuracy, because accuracy is dominated by the Normal class and would happily
select a model that never predicts Viral Pneumonia.

BackupAndRestore is included because Colab sessions terminate without warning;
a killed session resumes from the last completed epoch rather than from zero.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from covid_xray.config import RunConfig
from covid_xray.models import build_densenet121, set_finetune_trainable

MONITOR = "val_macro_f1"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def enable_mixed_precision(cfg: RunConfig) -> None:
    """Halves activation memory on a T4 and roughly doubles throughput.

    Safe here because the output layer is pinned to float32 in models.py.
    """
    if cfg.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")


def _metrics(cfg: RunConfig) -> list:
    return [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.F1Score(average="macro", name="macro_f1"),
    ]


def build_callbacks(cfg: RunConfig, output_dir: str | Path, stage: str) -> list:
    """Checkpoint, early-stop, log, and back up. Stage is 'a' or 'b'."""
    output_dir = Path(output_dir)
    (output_dir / "backup").mkdir(parents=True, exist_ok=True)

    patience = cfg.stage_a_patience if stage == "a" else cfg.stage_b_patience

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"{cfg.name}_stage{stage}.keras"),
            monitor=MONITOR,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=MONITOR,
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / f"{cfg.name}_stage{stage}.csv")),
        keras.callbacks.BackupAndRestore(backup_dir=str(output_dir / "backup")),
    ]


def train_two_stage(
    cfg: RunConfig,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    output_dir: str | Path,
) -> tuple[keras.Model, dict]:
    """Run Stage A then Stage B. Returns the fitted model and both histories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(cfg.seed)

    model = build_densenet121(cfg)

    # --- Stage A: head only -------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.stage_a_lr),
        loss="categorical_crossentropy",
        metrics=_metrics(cfg),
    )
    history_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.stage_a_epochs,
        callbacks=build_callbacks(cfg, output_dir, stage="a"),
        verbose=1,
    )

    # --- Stage B: fine-tune the top block ----------------------------------
    unfrozen = set_finetune_trainable(model, cfg.unfreeze_from)
    print(f"stage B: unfroze {unfrozen} layers from {cfg.unfreeze_from!r} (BatchNorm excluded)")

    # Recompile so the optimizer state is fresh and the new trainable set is picked up.
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.stage_b_lr),
        loss="categorical_crossentropy",
        metrics=_metrics(cfg),
    )
    history_b = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.stage_b_epochs,
        callbacks=build_callbacks(cfg, output_dir, stage="b"),
        verbose=1,
    )

    model.save(output_dir / f"{cfg.name}_final.keras")

    history = {
        "stage_a": {k: [float(v) for v in vs] for k, vs in history_a.history.items()},
        "stage_b": {k: [float(v) for v in vs] for k, vs in history_b.history.items()},
    }
    with open(output_dir / f"{cfg.name}_history.json", "w") as handle:
        json.dump({"config": cfg.to_dict(), "history": history}, handle, indent=2)

    return model, history
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS, 8 tests

- [ ] **Step 6: Run the whole suite**

Run: `pytest -v`
Expected: PASS. All modules to date green.

- [ ] **Step 7: Commit**

```bash
git add src/covid_xray/train.py tests/test_train.py configs/
git commit -m "feat: two-stage training loop with macro-F1 selection and Colab backup"
```

---

## Task 9: Notebook 02 — execute the four runs

**Files:**
- Create: `notebooks/02_train.ipynb`
- Produces: `checkpoints/<run>_final.keras`, `<run>_history.json`, `run3_probe8.joblib` (all gitignored)

**Verification criterion, not a unit test.** Training is stochastic; what must
hold is that each run completes, validation macro-F1 exceeds chance (0.25), and
the histories are written.

- [ ] **Step 1: Cell 1 — Colab setup**

```python
# Colab only. Mount Drive so checkpoints survive session death.
from google.colab import drive
drive.mount('/content/drive')

!pip install -q ImageHash pyyaml
# Do NOT pip install tensorflow on Colab — it is preinstalled and matched to CUDA.

import os, sys
from pathlib import Path

REPO_ROOT = Path('/content/drive/MyDrive/covid-xray-detection')
DATA_ROOT = Path('/content/data/COVID-19_Radiography_Dataset')   # local disk: much faster than Drive
CKPT_ROOT = REPO_ROOT / 'checkpoints'
sys.path.insert(0, str(REPO_ROOT / 'src'))
CKPT_ROOT.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Cell 2 — fetch the dataset to local disk**

Reading 21k files from mounted Drive is roughly an order of magnitude slower
than from Colab's local disk. Copy once per session.

```python
os.environ['KAGGLE_CONFIG_DIR'] = str(REPO_ROOT)   # expects kaggle.json there, gitignored
!kaggle datasets download -d tawsifurrahman/covid19-radiography-database -p /content --unzip -q
assert DATA_ROOT.exists(), sorted(p.name for p in Path('/content').iterdir())
```

- [ ] **Step 3: Cell 3 — confirm the GPU**

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
!nvidia-smi --query-gpu=name,memory.total --format=csv
```
Expected: one GPU listed. If the list is empty, switch runtime type before continuing.

- [ ] **Step 4: Cell 4 — load manifests and build the training helper**

```python
from covid_xray.config import load_config
from covid_xray.splits import load_manifest
from covid_xray.data import build_dataset
from covid_xray.augment import build_augmenter
from covid_xray.train import train_two_stage, enable_mixed_precision

manifests = {s: load_manifest(REPO_ROOT / 'data' / 'splits' / f'{s}.csv') for s in ('train', 'val', 'test')}
print({s: len(d) for s, d in manifests.items()})

def run_densenet(config_name):
    cfg = load_config(REPO_ROOT / 'configs' / f'{config_name}.yaml')
    enable_mixed_precision(cfg)
    train_ds = build_dataset(manifests['train'], DATA_ROOT, cfg, training=True,
                             augmenter=build_augmenter(cfg),
                             cache_path=f'/content/cache_{cfg.name}_train')
    val_ds = build_dataset(manifests['val'], DATA_ROOT, cfg, training=False,
                           cache_path=f'/content/cache_{cfg.name}_val')
    return train_two_stage(cfg, train_ds, val_ds, CKPT_ROOT)
```

- [ ] **Step 5: Cell 5 — sanity-check one batch before committing an hour of GPU**

Look at these images. If the masked variant does not visibly show black
outside the lungs, stop and fix Task 5 rather than training on it.

```python
import matplotlib.pyplot as plt
import numpy as np

cfg_check = load_config(REPO_ROOT / 'configs' / 'run2_masked.yaml')
check_ds = build_dataset(manifests['val'].head(8), DATA_ROOT, cfg_check, training=False)
images, labels = next(iter(check_ds))

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, image, label in zip(axes.ravel(), images.numpy(), labels.numpy()):
    shown = (image - image.min()) / (image.ptp() + 1e-8)
    ax.imshow(shown); ax.set_title(f'class {label.argmax()}'); ax.axis('off')
plt.show()
```

- [ ] **Step 6: Cell 6 — run 1 (raw baseline), ~1.5-2.5 hrs**

```python
model1, history1 = run_densenet('run1_raw')
print('best val macro-F1:', max(history1['stage_b']['val_macro_f1']))
```

- [ ] **Step 7: Cell 7 — run 2 (lung-masked), ~1.5-2.5 hrs**

```python
model2, history2 = run_densenet('run2_masked')
print('best val macro-F1:', max(history2['stage_b']['val_macro_f1']))
```

- [ ] **Step 8: Cell 8 — run 4 (lungs removed), ~1.5-2.5 hrs**

```python
model4, history4 = run_densenet('run4_lungs_removed')
print('best val macro-F1:', max(history4['stage_b']['val_macro_f1']))
```

- [ ] **Step 9: Cell 9 — run 3 (8×8 linear probe), ~2 min**

```python
import joblib
from sklearn.metrics import f1_score
from covid_xray.data import extract_downsampled_features
from covid_xray.models import build_probe

X_train, y_train = extract_downsampled_features(manifests['train'], DATA_ROOT, size=8)
X_val,   y_val   = extract_downsampled_features(manifests['val'],   DATA_ROOT, size=8)

probe = build_probe(seed=42).fit(X_train, y_train)
print('probe val macro-F1:', f1_score(y_val, probe.predict(X_val), average='macro'))

# joblib serialisation is pickle-based and therefore executes code on load.
# Safe here: this artefact is written and read only by this project, never
# fetched from elsewhere. Do not load a .joblib you did not produce yourself.
joblib.dump(probe, CKPT_ROOT / 'run3_probe8.joblib')
```

**Read this number carefully.** Chance is 0.25. A probe scoring meaningfully
above chance on 64 pixels means substantial label information lives in global
intensity structure — not in lung pathology.

- [ ] **Step 10: Verify all four runs produced artefacts**

Run:
```bash
ls -la checkpoints/
python -c "
import json, glob
for path in sorted(glob.glob('checkpoints/*_history.json')):
    payload = json.load(open(path))
    best = max(payload['history']['stage_b']['val_macro_f1'])
    print(f\"{payload['config']['name']:22s} best val macro-F1 = {best:.4f}\")
"
```
Expected: three `*_final.keras`, three `*_history.json`, one `.joblib`. Every
val macro-F1 above 0.25.

- [ ] **Step 11: Commit**

Checkpoints are gitignored; only the notebook and the histories are tracked.

```bash
git add notebooks/02_train.ipynb
git add -f checkpoints/run1_raw_history.json checkpoints/run2_masked_history.json checkpoints/run4_lungs_removed_history.json
git commit -m "feat: training notebook and completed histories for all four runs"
```

---

## Task 10: Metric suite, bootstrap CIs, and the control pair

**Files:**
- Create: `src/covid_xray/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `CLASS_NAMES`, `CONTROL_PAIR`
- Produces: `predict_probabilities(model, ds) -> np.ndarray`; `per_class_metrics(y_true, y_prob, class_names) -> pd.DataFrame`; `macro_f1(y_true, y_prob) -> float`; `sensitivity_specificity(y_true, y_prob, class_index) -> tuple[float, float]`; `specificity_at_sensitivity(y_true, y_score, target=0.95) -> tuple[float, float]` returning `(specificity, threshold)`; `bootstrap_ci(y_true, y_prob, statistic, n_resamples=2000, seed=42) -> tuple[float, float]`; `expected_calibration_error(y_true, y_prob, n_bins=15) -> float`; `confusion(y_true, y_prob, class_names) -> pd.DataFrame`; `restricted_pair_metrics(y_true, y_prob, pair=CONTROL_PAIR) -> dict`; `summarise_run(name, y_true, y_prob) -> dict`.

- [ ] **Step 1: Write the failing test — `tests/test_evaluate.py`**

```python
import numpy as np
import pandas as pd
import pytest

from covid_xray.config import CLASS_NAMES
from covid_xray.evaluate import (
    bootstrap_ci,
    confusion,
    expected_calibration_error,
    macro_f1,
    per_class_metrics,
    restricted_pair_metrics,
    sensitivity_specificity,
    specificity_at_sensitivity,
    summarise_run,
)


@pytest.fixture
def perfect():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_prob = np.eye(4)[y_true] * 0.97 + 0.01
    return y_true, y_prob


@pytest.fixture
def noisy():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=400)
    logits = np.eye(4)[y_true] * 1.5 + rng.normal(size=(400, 4))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return y_true, exp / exp.sum(axis=1, keepdims=True)


def test_macro_f1_is_one_for_perfect_predictions(perfect):
    assert macro_f1(*perfect) == pytest.approx(1.0)


def test_per_class_metrics_has_a_row_per_class(noisy):
    table = per_class_metrics(*noisy, class_names=CLASS_NAMES)
    assert list(table.index) == list(CLASS_NAMES)
    assert {"precision", "recall", "f1", "support", "roc_auc", "pr_auc"} <= set(table.columns)


def test_sensitivity_and_specificity_are_perfect_on_perfect_input(perfect):
    sensitivity, specificity = sensitivity_specificity(*perfect, class_index=0)
    assert sensitivity == pytest.approx(1.0)
    assert specificity == pytest.approx(1.0)


def test_specificity_at_target_sensitivity_meets_the_target():
    rng = np.random.default_rng(1)
    y_true = np.concatenate([np.ones(100), np.zeros(300)])
    scores = np.concatenate([rng.normal(1.2, 1.0, 100), rng.normal(0.0, 1.0, 300)])

    specificity, threshold = specificity_at_sensitivity(y_true, scores, target=0.95)

    achieved = (scores[y_true == 1] >= threshold).mean()
    assert achieved >= 0.95 - 1e-9
    assert 0.0 <= specificity <= 1.0


def test_bootstrap_ci_brackets_the_point_estimate(noisy):
    y_true, y_prob = noisy
    point = macro_f1(y_true, y_prob)
    low, high = bootstrap_ci(y_true, y_prob, macro_f1, n_resamples=200, seed=42)
    assert low <= point <= high
    assert high - low > 0


def test_bootstrap_ci_is_reproducible(noisy):
    first = bootstrap_ci(*noisy, macro_f1, n_resamples=200, seed=42)
    second = bootstrap_ci(*noisy, macro_f1, n_resamples=200, seed=42)
    assert first == second


def test_confusion_matrix_is_square_and_totals_correctly(noisy):
    y_true, y_prob = noisy
    matrix = confusion(y_true, y_prob, CLASS_NAMES)
    assert matrix.shape == (4, 4)
    assert matrix.to_numpy().sum() == len(y_true)


def test_calibration_error_is_near_zero_for_confident_correct_predictions(perfect):
    assert expected_calibration_error(*perfect) < 0.10


def test_calibration_error_is_large_for_confidently_wrong_predictions():
    y_true = np.zeros(50, dtype=int)
    y_prob = np.tile([0.01, 0.97, 0.01, 0.01], (50, 1))
    assert expected_calibration_error(y_true, y_prob) > 0.5


def test_restricted_pair_uses_only_the_two_named_classes(noisy):
    y_true, y_prob = noisy
    result = restricted_pair_metrics(y_true, y_prob, pair=("COVID", "Lung_Opacity"))

    expected_n = int(((y_true == 0) | (y_true == 1)).sum())
    assert result["n"] == expected_n
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert result["pair"] == ("COVID", "Lung_Opacity")


def test_restricted_pair_ignores_the_other_classes_probability_mass():
    """Scores must be renormalised over the pair, not read off the 4-way softmax."""
    y_true = np.array([0, 1])
    # Both rows put most mass on Normal; within the pair, row 0 favours COVID.
    y_prob = np.array([[0.20, 0.05, 0.70, 0.05], [0.05, 0.20, 0.70, 0.05]])

    result = restricted_pair_metrics(y_true, y_prob, pair=("COVID", "Lung_Opacity"))

    assert result["accuracy"] == pytest.approx(1.0)


def test_summarise_run_returns_every_reported_field(noisy):
    summary = summarise_run("run1_raw", *noisy)
    assert summary["run"] == "run1_raw"
    for key in ("macro_f1", "macro_f1_ci", "covid_sensitivity", "covid_specificity",
                "covid_specificity_at_95_sensitivity", "ece", "per_class", "confusion",
                "restricted_pair"):
        assert key in summary
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'covid_xray.evaluate'`

- [ ] **Step 3: Write `src/covid_xray/evaluate.py`**

```python
"""Metric suite.

Two decisions worth stating, both from spec section 6:

1. Every headline figure carries a bootstrap confidence interval. The Viral
   Pneumonia test slice is ~200 images; a bare point estimate there implies a
   precision the data does not support.
2. Specificity is reported at a fixed 95% COVID sensitivity, not only at
   argmax. Argmax hides the trade-off that matters clinically — a false
   negative discharges an infectious patient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from covid_xray.config import CLASS_NAMES, CONTROL_PAIR


def predict_probabilities(model, dataset) -> np.ndarray:
    """Predicted class probabilities, in dataset order."""
    return np.asarray(model.predict(dataset, verbose=0), dtype=np.float64)


def macro_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(f1_score(y_true, y_prob.argmax(axis=1), average="macro", zero_division=0))


def per_class_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, class_names=CLASS_NAMES
) -> pd.DataFrame:
    """Precision, recall, F1, support, one-vs-rest ROC-AUC and PR-AUC.

    PR-AUC sits alongside ROC-AUC because ROC-AUC is optimistic under the
    ~7.6:1 imbalance in this dataset.
    """
    y_pred = y_prob.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    roc_aucs, pr_aucs = [], []
    for index in range(len(class_names)):
        binary = (y_true == index).astype(int)
        if binary.min() == binary.max():  # class absent from this split
            roc_aucs.append(float("nan"))
            pr_aucs.append(float("nan"))
        else:
            roc_aucs.append(roc_auc_score(binary, y_prob[:, index]))
            pr_aucs.append(average_precision_score(binary, y_prob[:, index]))

    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "roc_auc": roc_aucs,
            "pr_auc": pr_aucs,
        },
        index=list(class_names),
    )


def sensitivity_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, class_index: int
) -> tuple[float, float]:
    """One-vs-rest sensitivity and specificity at argmax."""
    y_pred = y_prob.argmax(axis=1)
    positives = y_true == class_index
    negatives = ~positives

    sensitivity = float((y_pred[positives] == class_index).mean()) if positives.any() else float("nan")
    specificity = float((y_pred[negatives] != class_index).mean()) if negatives.any() else float("nan")
    return sensitivity, specificity


def specificity_at_sensitivity(
    y_true_binary: np.ndarray, y_score: np.ndarray, target: float = 0.95
) -> tuple[float, float]:
    """Highest specificity achievable while holding sensitivity at or above `target`.

    Returns `(specificity, threshold)`. If the target is unreachable, returns
    the operating point with the highest sensitivity available.
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    feasible = np.nonzero(tpr >= target)[0]
    index = feasible[0] if feasible.size else int(np.argmax(tpr))
    return float(1.0 - fpr[index]), float(thresholds[index])


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    statistic,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for any statistic of (y_true, y_prob)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = np.empty(n_resamples)

    for draw in range(n_resamples):
        positions = rng.integers(0, n, size=n)
        values[draw] = statistic(y_true[positions], y_prob[positions])

    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Standard binned ECE over top-1 confidence."""
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def confusion(y_true: np.ndarray, y_prob: np.ndarray, class_names=CLASS_NAMES) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_prob.argmax(axis=1), labels=range(len(class_names)))
    return pd.DataFrame(matrix, index=list(class_names), columns=list(class_names))


def restricted_pair_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, pair: tuple[str, str] = CONTROL_PAIR
) -> dict:
    """Metrics restricted to the two-class control pair.

    Both classes are adult, so the pediatric confound is absent. Scores are
    renormalised over the pair — reading them straight off the 4-way softmax
    would let a third class's probability mass decide the comparison.

    No retraining: this reuses the same trained model's predictions.
    """
    first, second = (CLASS_NAMES.index(name) for name in pair)
    selected = (y_true == first) | (y_true == second)

    if selected.sum() == 0:
        return {"pair": pair, "n": 0, "accuracy": float("nan"), "roc_auc": float("nan")}

    truth = (y_true[selected] == first).astype(int)
    pair_prob = y_prob[selected][:, [first, second]]
    score = pair_prob[:, 0] / np.clip(pair_prob.sum(axis=1), 1e-12, None)

    accuracy = float(((score >= 0.5).astype(int) == truth).mean())
    auc = float(roc_auc_score(truth, score)) if truth.min() != truth.max() else float("nan")

    return {"pair": pair, "n": int(selected.sum()), "accuracy": accuracy, "roc_auc": auc}


def summarise_run(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Everything the README reports for one run."""
    covid_index = CLASS_NAMES.index("COVID")
    sensitivity, specificity = sensitivity_specificity(y_true, y_prob, covid_index)
    spec_at_95, threshold = specificity_at_sensitivity(
        (y_true == covid_index).astype(int), y_prob[:, covid_index], target=0.95
    )

    return {
        "run": name,
        "macro_f1": macro_f1(y_true, y_prob),
        "macro_f1_ci": bootstrap_ci(y_true, y_prob, macro_f1),
        "covid_sensitivity": sensitivity,
        "covid_specificity": specificity,
        "covid_specificity_at_95_sensitivity": spec_at_95,
        "covid_threshold_at_95_sensitivity": threshold,
        "ece": expected_calibration_error(y_true, y_prob),
        "per_class": per_class_metrics(y_true, y_prob),
        "confusion": confusion(y_true, y_prob),
        "restricted_pair": restricted_pair_metrics(y_true, y_prob),
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add src/covid_xray/evaluate.py tests/test_evaluate.py
git commit -m "feat: metric suite with bootstrap CIs and control-pair evaluation"
```

---

## Task 11: Notebook 03 — evaluation on the frozen test set

**This is the first and only time the test set is read.**

**Files:**
- Create: `notebooks/03_evaluate.ipynb`
- Produces: `reports/results.csv`, `reports/results.json`, `reports/figures/confusion_*.png`, `reports/figures/roc_*.png`, `reports/figures/calibration_*.png`

- [ ] **Step 1: Cell 1 — setup and load the trained models**

```python
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tensorflow import keras

REPO_ROOT = Path('/content/drive/MyDrive/covid-xray-detection')
DATA_ROOT = Path('/content/data/COVID-19_Radiography_Dataset')
sys.path.insert(0, str(REPO_ROOT / 'src'))

from covid_xray.config import CLASS_NAMES, load_config
from covid_xray.splits import load_manifest
from covid_xray.data import build_dataset, extract_downsampled_features
from covid_xray.evaluate import predict_probabilities, summarise_run

test_manifest = load_manifest(REPO_ROOT / 'data' / 'splits' / 'test.csv')
y_true = test_manifest['label'].to_numpy()
print(len(test_manifest), dict(test_manifest['class_name'].value_counts()))
```

- [ ] **Step 2: Cell 2 — predict with all four runs**

```python
predictions = {}

for config_name in ('run1_raw', 'run2_masked', 'run4_lungs_removed'):
    cfg = load_config(REPO_ROOT / 'configs' / f'{config_name}.yaml')
    model = keras.models.load_model(REPO_ROOT / 'checkpoints' / f'{cfg.name}_final.keras')
    test_ds = build_dataset(test_manifest, DATA_ROOT, cfg, training=False)
    predictions[config_name] = predict_probabilities(model, test_ds)
    print(config_name, predictions[config_name].shape)

# Pickle-based load — safe only because this file was produced by Task 9 in
# this same project. Never load a .joblib from an untrusted source.
probe = joblib.load(REPO_ROOT / 'checkpoints' / 'run3_probe8.joblib')
X_test, _ = extract_downsampled_features(test_manifest, DATA_ROOT, size=8)
predictions['run3_probe8'] = probe.predict_proba(X_test)
```

- [ ] **Step 3: Cell 3 — summarise every run**

```python
ORDER = ('run1_raw', 'run2_masked', 'run3_probe8', 'run4_lungs_removed')
summaries = {name: summarise_run(name, y_true, predictions[name]) for name in ORDER}

table = pd.DataFrame([{
    'run': s['run'],
    'macro_F1': round(s['macro_f1'], 4),
    'macro_F1_95CI': f"[{s['macro_f1_ci'][0]:.3f}, {s['macro_f1_ci'][1]:.3f}]",
    'COVID_sens': round(s['covid_sensitivity'], 4),
    'COVID_spec': round(s['covid_specificity'], 4),
    'COVID_spec@95sens': round(s['covid_specificity_at_95_sensitivity'], 4),
    'ECE': round(s['ece'], 4),
    'pair_acc': round(s['restricted_pair']['accuracy'], 4),
    'pair_AUC': round(s['restricted_pair']['roc_auc'], 4),
} for s in summaries.values()]).set_index('run')

table.to_csv(REPO_ROOT / 'reports' / 'results.csv')
table
```

**How to read this table.** Compare `run1_raw` against `run2_masked`: the gap
is how much accuracy depended on signal outside the lungs. Then read
`run3_probe8` and `run4_lungs_removed` against chance (macro-F1 0.25). Then
compare the `pair_*` columns against the whole-dataset columns: a probe that
scores well overall but near chance on the pair has found the pediatric
artefact specifically.

- [ ] **Step 4: Cell 4 — per-class tables**

```python
for name in ORDER:
    print(f'\n=== {name} ===')
    print(summaries[name]['per_class'].round(4))
```

- [ ] **Step 5: Cell 5 — confusion matrices, with the control pair highlighted**

```python
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, name in zip(axes, ORDER):
    matrix = summaries[name]['confusion']
    normalised = matrix.div(matrix.sum(axis=1).clip(lower=1), axis=0)
    im = ax.imshow(normalised, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_title(name); ax.set_xlabel('predicted'); ax.set_ylabel('true')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{matrix.iloc[i, j]}', ha='center', va='center',
                    color='white' if normalised.iloc[i, j] > 0.5 else 'black')
    # COVID (0) and Lung_Opacity (1) — the two cells that carry the audit.
    for i, j in ((0, 1), (1, 0)):
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor='crimson', lw=2.5))
plt.tight_layout()
plt.savefig(REPO_ROOT / 'reports' / 'figures' / 'confusion_all_runs.png', dpi=150)
plt.show()
```

- [ ] **Step 6: Cell 6 — calibration curves for the two real models**

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name in zip(axes, ('run1_raw', 'run2_masked')):
    probs = predictions[name]
    confidence, correct = probs.max(axis=1), (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, 16)
    xs, ys = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.sum() >= 5:
            xs.append(confidence[in_bin].mean()); ys.append(correct[in_bin].mean())
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect')
    ax.plot(xs, ys, 'o-', label=name)
    ax.set_xlabel('confidence'); ax.set_ylabel('accuracy')
    ax.set_title(f"{name} — ECE {summaries[name]['ece']:.3f}"); ax.legend()
plt.tight_layout()
plt.savefig(REPO_ROOT / 'reports' / 'figures' / 'calibration.png', dpi=150)
plt.show()
```

- [ ] **Step 7: Cell 7 — persist everything the README needs**

```python
serialisable = {}
for name, summary in summaries.items():
    payload = {k: v for k, v in summary.items() if k not in ('per_class', 'confusion')}
    payload['restricted_pair'] = {**summary['restricted_pair'], 'pair': list(summary['restricted_pair']['pair'])}
    payload['per_class'] = summary['per_class'].round(4).to_dict()
    payload['confusion'] = summary['confusion'].to_dict()
    serialisable[name] = payload

with open(REPO_ROOT / 'reports' / 'results.json', 'w') as handle:
    json.dump(serialisable, handle, indent=2, default=float)
print('written')
```

- [ ] **Step 8: Verify**

Run:
```bash
python -c "
import json
results = json.load(open('reports/results.json'))
assert set(results) == {'run1_raw','run2_masked','run3_probe8','run4_lungs_removed'}
for name, r in results.items():
    print(f\"{name:22s} macroF1={r['macro_f1']:.4f} pairAUC={r['restricted_pair']['roc_auc']:.4f}\")
"
```
Expected: four rows. Sanity check — `run1_raw` should have the highest macro-F1.
If `run4_lungs_removed` beats it, either the mask polarity is inverted (check
Task 5) or the dataset is even more confounded than expected; investigate
before writing conclusions.

- [ ] **Step 9: Commit**

```bash
git add notebooks/03_evaluate.ipynb
git add -f reports/results.csv reports/results.json reports/figures/confusion_all_runs.png reports/figures/calibration.png
git commit -m "feat: test-set evaluation across all four runs with CIs and control pair"
```

---

## Task 12: Grad-CAM and the Lung Attribution Ratio

**Files:**
- Create: `src/covid_xray/gradcam.py`, `tests/test_gradcam.py`

**Interfaces:**
- Consumes: `split_feature_and_head`
- Produces: `grad_cam(model, image_batch, class_index=None) -> np.ndarray` returning a `(batch, H, W)` heatmap normalised to max 1 per image; `lung_attribution_ratio(heatmap, mask) -> float`; `batch_attribution_ratios(model, dataset, masks) -> np.ndarray`.

- [ ] **Step 1: Write the failing test — `tests/test_gradcam.py`**

```python
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from covid_xray.config import RunConfig
from covid_xray.gradcam import grad_cam, lung_attribution_ratio
from covid_xray.models import build_densenet121


@pytest.fixture(scope="module")
def model():
    return build_densenet121(
        RunConfig(name="t", variant="raw", model="densenet121", image_size=64, mixed_precision=False)
    )


def test_heatmap_matches_the_input_spatial_size(model):
    batch = tf.random.uniform((2, 64, 64, 3), seed=0)
    heatmaps = grad_cam(model, batch)
    assert heatmaps.shape == (2, 64, 64)


def test_heatmap_is_non_negative_and_peaks_at_one(model):
    batch = tf.random.uniform((2, 64, 64, 3), seed=1)
    heatmaps = grad_cam(model, batch)
    assert heatmaps.min() >= 0.0
    assert np.allclose(heatmaps.max(axis=(1, 2)), 1.0, atol=1e-5)


def test_explicit_class_index_is_honoured(model):
    batch = tf.random.uniform((1, 64, 64, 3), seed=2)
    first = grad_cam(model, batch, class_index=0)
    second = grad_cam(model, batch, class_index=3)
    assert not np.allclose(first, second)


def test_attribution_ratio_is_one_when_all_mass_is_inside_the_mask():
    heatmap = np.zeros((8, 8), dtype=np.float32)
    heatmap[:, :4] = 1.0
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, :4] = 255
    assert lung_attribution_ratio(heatmap, mask) == pytest.approx(1.0)


def test_attribution_ratio_is_zero_when_all_mass_is_outside_the_mask():
    heatmap = np.zeros((8, 8), dtype=np.float32)
    heatmap[:, 4:] = 1.0
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, :4] = 255
    assert lung_attribution_ratio(heatmap, mask) == pytest.approx(0.0)


def test_attribution_ratio_is_half_for_uniform_attribution_over_a_half_mask():
    heatmap = np.ones((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, :4] = 255
    assert lung_attribution_ratio(heatmap, mask) == pytest.approx(0.5)


def test_attribution_ratio_resizes_a_mask_of_different_resolution():
    """Real masks ship at 256x256 while heatmaps come back at 224x224."""
    heatmap = np.ones((16, 16), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, :4] = 255
    assert lung_attribution_ratio(heatmap, mask) == pytest.approx(0.5, abs=0.05)


def test_attribution_ratio_of_an_empty_heatmap_is_nan():
    heatmap = np.zeros((8, 8), dtype=np.float32)
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    assert np.isnan(lung_attribution_ratio(heatmap, mask))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_gradcam.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'covid_xray.gradcam'`

- [ ] **Step 3: Write `src/covid_xray/gradcam.py`**

```python
"""Grad-CAM and the Lung Attribution Ratio.

The Lung Attribution Ratio is the point of this module. A heatmap is an
anecdote; "62% of attribution mass inside the lungs for the raw model versus
94% for the masked model" is a result. The ratio turns explainability into a
number that can carry a confidence interval and go in a table.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image

from covid_xray.models import split_feature_and_head


def grad_cam(model, image_batch, class_index: int | None = None) -> np.ndarray:
    """Gradient-weighted class activation maps for a batch.

    `class_index` defaults to each image's own predicted class. Returns a
    `(batch, height, width)` array, each map non-negative and scaled to peak 1.
    """
    feature_model, head_model = split_feature_and_head(model)
    images = tf.convert_to_tensor(image_batch)

    with tf.GradientTape() as tape:
        features = feature_model(images, training=False)
        tape.watch(features)
        predictions = head_model(features, training=False)
        if class_index is None:
            targets = tf.reduce_max(predictions, axis=1)
        else:
            targets = predictions[:, class_index]

    gradients = tape.gradient(targets, features)

    # One weight per channel: the mean gradient over spatial positions.
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    maps = tf.nn.relu(tf.reduce_sum(features * weights, axis=-1))

    maps = tf.image.resize(maps[..., None], images.shape[1:3], method="bilinear")[..., 0]
    maps = maps.numpy()

    peaks = maps.max(axis=(1, 2), keepdims=True)
    return np.divide(maps, peaks, out=np.zeros_like(maps), where=peaks > 0)


def lung_attribution_ratio(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of attribution mass falling inside the lung mask.

    The mask is resized to the heatmap's resolution with nearest-neighbour, so
    a 256x256 dataset mask and a 224x224 heatmap compare correctly. Returns NaN
    for an all-zero heatmap, which has no mass to attribute.
    """
    heatmap = np.asarray(heatmap, dtype=np.float64)
    total = heatmap.sum()
    if total <= 0:
        return float("nan")

    mask = np.asarray(mask)
    if mask.shape != heatmap.shape:
        resized = Image.fromarray(mask.astype(np.uint8)).resize(
            (heatmap.shape[1], heatmap.shape[0]), Image.NEAREST
        )
        mask = np.asarray(resized)

    inside = mask > 127
    return float(heatmap[inside].sum() / total)


def batch_attribution_ratios(model, image_batch, mask_batch) -> np.ndarray:
    """Lung Attribution Ratio for every image in a batch."""
    heatmaps = grad_cam(model, image_batch)
    return np.array(
        [lung_attribution_ratio(heat, mask) for heat, mask in zip(heatmaps, mask_batch)]
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_gradcam.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Run the whole suite**

Run: `pytest -v`
Expected: PASS across all modules.

- [ ] **Step 6: Commit**

```bash
git add src/covid_xray/gradcam.py tests/test_gradcam.py
git commit -m "feat: Grad-CAM with quantified lung attribution ratio"
```

---

## Task 13: Notebook 04 — the confound audit

**Files:**
- Create: `notebooks/04_gradcam_audit.ipynb`
- Produces: `reports/attribution_ratios.csv`, `reports/figures/gradcam_panel.png`, `reports/figures/attribution_by_class.png`, `reports/figures/failure_gallery.png`

- [ ] **Step 1: Cell 1 — setup**

```python
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras

REPO_ROOT = Path('/content/drive/MyDrive/covid-xray-detection')
DATA_ROOT = Path('/content/data/COVID-19_Radiography_Dataset')
sys.path.insert(0, str(REPO_ROOT / 'src'))

from covid_xray.config import CLASS_NAMES, load_config
from covid_xray.splits import load_manifest
from covid_xray.data import build_dataset
from covid_xray.gradcam import grad_cam, lung_attribution_ratio
from covid_xray.evaluate import bootstrap_ci

test_manifest = load_manifest(REPO_ROOT / 'data' / 'splits' / 'test.csv')
models = {
    name: keras.models.load_model(REPO_ROOT / 'checkpoints' / f'{name}_final.keras')
    for name in ('run1_raw', 'run2_masked')
}
```

- [ ] **Step 2: Cell 2 — compute the Lung Attribution Ratio across the test set**

```python
def attribution_ratios(model, config_name, manifest, batch_size=32):
    cfg = load_config(REPO_ROOT / 'configs' / f'{config_name}.yaml')
    ds = build_dataset(manifest, DATA_ROOT, cfg, training=False)

    ratios, position = [], 0
    for images, _ in ds:
        heatmaps = grad_cam(model, images)
        for heatmap in heatmaps:
            mask = np.asarray(Image.open(DATA_ROOT / manifest.iloc[position]['mask_path']))
            ratios.append(lung_attribution_ratio(heatmap, mask))
            position += 1
    return np.array(ratios)

ratio_frame = test_manifest[['path', 'class_name', 'label']].copy()
for name, model in models.items():
    ratio_frame[name] = attribution_ratios(model, name, test_manifest)

ratio_frame.to_csv(REPO_ROOT / 'reports' / 'attribution_ratios.csv', index=False)
ratio_frame.head()
```

- [ ] **Step 3: Cell 3 — the headline audit number**

```python
def mean_ci(values, seed=42, n=2000):
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    draws = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return values.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)

for name in models:
    mean, low, high = mean_ci(ratio_frame[name].to_numpy())
    print(f'{name:14s} lung attribution ratio = {mean:.3f}  95% CI [{low:.3f}, {high:.3f}]')
```

**This is the number the README leads with.** A low ratio for `run1_raw` means
the baseline model was looking outside the lungs to make its decisions.

- [ ] **Step 4: Cell 4 — attribution ratio broken out per class**

A low ratio on Viral Pneumonia predictions alongside a high ratio on COVID and
Lung Opacity would corroborate the pediatric-shortcut hypothesis from a
direction independent of the probes.

```python
summary = ratio_frame.groupby('class_name')[list(models)].agg(['mean', 'count'])
print(summary.round(3))

fig, ax = plt.subplots(figsize=(9, 4.5))
positions = np.arange(len(CLASS_NAMES))
width = 0.38
for offset, name in zip((-width/2, width/2), models):
    means = [ratio_frame.loc[ratio_frame.class_name == c, name].mean() for c in CLASS_NAMES]
    ax.bar(positions + offset, means, width, label=name)
ax.set_xticks(positions, CLASS_NAMES, rotation=20, ha='right')
ax.set_ylabel('lung attribution ratio'); ax.axhline(0.5, ls='--', c='grey', lw=1)
ax.set_title('Fraction of Grad-CAM mass inside the lungs, by true class'); ax.legend()
plt.tight_layout()
plt.savefig(REPO_ROOT / 'reports' / 'figures' / 'attribution_by_class.png', dpi=150)
plt.show()
```

- [ ] **Step 5: Cell 5 — qualitative panel, raw versus masked**

Six test images, including at least one COVID and one Lung Opacity case.

```python
picks = (test_manifest.groupby('class_name', group_keys=False)
         .head(2).reset_index(drop=True).head(6))

fig, axes = plt.subplots(3, 6, figsize=(20, 10))
for column, (_, row) in enumerate(picks.iterrows()):
    original = np.asarray(Image.open(DATA_ROOT / row['path']).convert('L').resize((224, 224)))
    axes[0, column].imshow(original, cmap='gray')
    axes[0, column].set_title(row['class_name'], fontsize=10)

    for offset, name in enumerate(models, start=1):
        cfg = load_config(REPO_ROOT / 'configs' / f'{name}.yaml')
        ds = build_dataset(picks.iloc[[column]], DATA_ROOT, cfg, training=False)
        images, _ = next(iter(ds))
        heatmap = grad_cam(models[name], images)[0]
        axes[offset, column].imshow(original, cmap='gray')
        axes[offset, column].imshow(heatmap, cmap='jet', alpha=0.45)
        axes[offset, column].set_title(f'{name}  LAR={lung_attribution_ratio(heatmap, np.asarray(Image.open(DATA_ROOT / row["mask_path"]))):.2f}', fontsize=9)

for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig(REPO_ROOT / 'reports' / 'figures' / 'gradcam_panel.png', dpi=150)
plt.show()
```

- [ ] **Step 6: Cell 6 — failure gallery**

Highest-confidence wrong predictions, where shortcut reliance shows most clearly.

```python
import json
probabilities = np.array(json.load(open(REPO_ROOT / 'reports' / 'run1_raw_probs.json'))) \
    if (REPO_ROOT / 'reports' / 'run1_raw_probs.json').exists() else None

cfg = load_config(REPO_ROOT / 'configs' / 'run1_raw.yaml')
test_ds = build_dataset(test_manifest, DATA_ROOT, cfg, training=False)
probabilities = models['run1_raw'].predict(test_ds, verbose=0)

y_true = test_manifest['label'].to_numpy()
y_pred = probabilities.argmax(axis=1)
confidence = probabilities.max(axis=1)
wrong = np.nonzero(y_pred != y_true)[0]
worst = wrong[np.argsort(-confidence[wrong])][:6]

fig, axes = plt.subplots(1, 6, figsize=(20, 4))
for ax, index in zip(axes, worst):
    row = test_manifest.iloc[index]
    ax.imshow(np.asarray(Image.open(DATA_ROOT / row['path']).convert('L')), cmap='gray')
    ax.set_title(f'true {CLASS_NAMES[y_true[index]]}\npred {CLASS_NAMES[y_pred[index]]} ({confidence[index]:.2f})', fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig(REPO_ROOT / 'reports' / 'figures' / 'failure_gallery.png', dpi=150)
plt.show()
```

- [ ] **Step 7: Verify**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_csv('reports/attribution_ratios.csv')
assert {'run1_raw','run2_masked'} <= set(df.columns)
print(df[['run1_raw','run2_masked']].describe().round(3))
assert df['run2_masked'].mean() > df['run1_raw'].mean(), \
    'masked model should attend inside the lungs more than the raw model'
"
```
Expected: the masked model's mean ratio exceeds the raw model's. If it does
not, check mask polarity in Task 5 before writing any conclusion.

- [ ] **Step 8: Commit**

```bash
git add notebooks/04_gradcam_audit.ipynb
git add -f reports/attribution_ratios.csv reports/figures/gradcam_panel.png reports/figures/attribution_by_class.png reports/figures/failure_gallery.png
git commit -m "feat: confound audit notebook with quantified lung attribution"
```

---

## Task 14: README

**Files:**
- Create: `README.md` (this file may be written directly to disk)

The README is the deliverable a reader actually sees. Lead with the audit, not
the accuracy — a 4-class macro-F1 that looks modest next to the 97% figures
littering this dataset's Kaggle notebooks is the *point*, and it must be framed
that way in the first screen, not defended in a footnote at the bottom.

- [ ] **Step 1: Write the README with this structure**

```markdown
# COVID-19 Detection in Chest X-Rays — and How Much of It Is Real

One-paragraph summary: what was built, the headline macro-F1 with its CI, and
the lung attribution ratio, in the first three sentences.

## The finding
The raw-versus-masked gap, the two probe scores against chance, and the
control-pair comparison. A table. Then two sentences interpreting it.

## Results
The full table from reports/results.csv, all four runs.
Confusion matrices figure. Calibration figure.

## Method
Dataset, de-duplication counts (including cross-class duplicates found),
split sizes, preprocessing, augmentation, the two-stage recipe. Brief —
link to the spec for detail.

## Explainability
attribution_by_class.png and gradcam_panel.png with two sentences each.

## Limitations
- No patient IDs in this dataset: patient-disjoint splitting is impossible
  and every metric is an upper bound.
- The lung masks are model-generated by the dataset authors, not drawn by
  radiologists.
- The four labels are not a clean partition of disease space; they reflect
  which datasets were merged. A pediatric viral pneumonia is also a lung
  opacity.
- Viral Pneumonia is pediatric (ages 1-5, Kermany/Guangzhou) while COVID and
  Lung Opacity are adult.
- No external validation. Generalisation to a new hospital is untested and,
  on the evidence of DeGrave et al., should not be assumed.
- **Not a medical device. Not for clinical use.**

## Reproducing
Setup, dataset download, notebook order, expected runtimes.

## References
Chowdhury et al. (dataset); DeGrave et al. 2021 (shortcut learning);
Kermany et al. (pediatric source); RSNA Pneumonia Detection Challenge.
```

- [ ] **Step 2: Fill every number from `reports/results.json` and `reports/attribution_ratios.csv`**

No placeholders. If a number is not yet computed, the notebook that computes it
has not been run.

- [ ] **Step 3: Verify the README against the artefacts**

Run:
```bash
grep -o 'reports/figures/[a-z_]*\.png' README.md | sort -u | while read -r figure; do
  test -f "$figure" && echo "ok   $figure" || echo "MISSING $figure"
done
```
Expected: every referenced figure exists.

- [ ] **Step 4: Final full-suite run**

Run: `pytest -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README with results, audit findings, and limitations"
```

---

## Self-Review

**Spec coverage.** Every section maps to a task:

| Spec section | Task(s) |
|---|---|
| §1 framing, four-class rationale | 14 (README), 11 (control-pair table) |
| §2 source, counts, mask resolution check | 4 (Step 2) |
| §2 de-duplication, deterministic retention | 2, 4 |
| §2 splitting, committed manifests, frozen test set | 3, 4, 11 |
| §2 no-patient-ID limitation | 3 (docstring), 14 (README) |
| §3 preprocessing, 224, preprocess_input, masked-not-cropped | 5 |
| §4 augmentation, six layers, exclusions | 6 |
| §5 four runs, configs | 7, 8, 9 |
| §5 training recipe, BatchNorm, sample weights, patience | 5, 7, 8 |
| §5 control-pair analysis, reported twice | 10, 11 |
| §5 infrastructure, Colab, checkpointing | 8, 9 |
| §6 full metric suite, bootstrap CIs, calibration | 10, 11 |
| §6 COVID↔Lung Opacity cells called out | 11 (Step 5) |
| §7 Grad-CAM, Lung Attribution Ratio, per class | 12, 13 |
| §7 qualitative panel, failure gallery | 13 |
| §8 repository structure | 1–13 |
| §9 risks | 14 (Limitations) |
| §10 working agreement | Global Constraints |

**Gaps found and closed during review:**
- The control-pair analysis in spec §5 initially had no home; added as
  `restricted_pair_metrics` in Task 10 with two dedicated tests, including one
  asserting scores are renormalised over the pair rather than read off the
  4-way softmax.
- Spec §2's "verify at download time" items had no verification step; added as
  Task 4 Step 2 with an explicit stop condition.
- The nested-base Grad-CAM problem would have surfaced only at Task 13; moved
  the fix forward into `split_feature_and_head` in Task 7, tested there.

**Type consistency:** `RunConfig` field names are identical across Tasks 1, 5,
6, 7 and 8. `build_dataset` returns 3-tuples when `training=True` and 2-tuples
otherwise, asserted in Task 5 and relied on in Tasks 8, 11 and 13.
`CLASS_NAMES` ordering is fixed in Task 1 and asserted in Tasks 2 and 10.
`grad_cam` returns `(batch, H, W)` in Task 12 and is consumed as such in Task 13.

**Known rough edge, deliberately left:** Task 13 Cell 5 rebuilds a one-row
dataset per image, which is slow but runs on six images. Not worth optimising.
