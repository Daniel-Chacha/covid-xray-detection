import json

import numpy as np
import pytest
from PIL import Image

from scripts.build_gallery import CLASS_NAMES, manifest_record, save_224


def test_save_224_writes_exactly_224_square(tmp_path):
    """The browser must never resize; a resize kernel mismatch is silent."""
    array = np.random.default_rng(0).integers(0, 256, (299, 299), dtype=np.uint8)
    path = tmp_path / "raw.png"

    save_224(array, path)

    with Image.open(path) as written:
        assert written.size == (224, 224)


def test_manifest_record_has_every_field_the_frontend_reads():
    record = manifest_record(
        image_id="COVID-1001",
        true_class="COVID",
        note="attention on the D marker",
        lar=0.0439,
        raw_probs=np.array([0.94, 0.03, 0.02, 0.01]),
        lungs_removed_probs=np.array([0.91, 0.05, 0.03, 0.01]),
    )

    assert set(record) == {"id", "true_class", "note", "lar", "reference"}
    assert set(record["reference"]) == {"raw", "lungs_removed"}
    assert len(record["reference"]["raw"]) == len(CLASS_NAMES)


def test_manifest_record_probabilities_sum_to_one():
    record = manifest_record(
        image_id="X", true_class="Normal", note="", lar=0.3,
        raw_probs=np.array([0.1, 0.2, 0.6, 0.1]),
        lungs_removed_probs=np.array([0.25, 0.25, 0.25, 0.25]),
    )

    for variant in ("raw", "lungs_removed"):
        assert sum(record["reference"][variant]) == pytest.approx(1.0, abs=1e-6)


def test_manifest_record_rejects_an_unknown_class():
    with pytest.raises(ValueError, match="Pneumothorax"):
        manifest_record(
            image_id="X", true_class="Pneumothorax", note="", lar=0.3,
            raw_probs=np.zeros(4), lungs_removed_probs=np.zeros(4),
        )
