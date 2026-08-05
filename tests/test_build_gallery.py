import re

import numpy as np
import pytest
from PIL import Image

from scripts.build_gallery import CLASS_NAMES, SELECTION, manifest_record, sanitize_id, save_224


def test_save_224_writes_exactly_224_square(tmp_path):
    """The browser must never resize; a resize kernel mismatch is silent."""
    array = np.random.default_rng(0).integers(0, 256, (299, 299), dtype=np.uint8)
    path = tmp_path / "raw.png"

    save_224(array, path)

    with Image.open(path) as written:
        assert written.size == (224, 224)


def test_save_224_preserves_pixels_when_already_224(tmp_path):
    """Production always hands save_224 an already-224x224 array; this is the
    path that actually runs, and it must not resize (byte-identical pixels
    are what the reference probabilities were computed on)."""
    array = np.random.default_rng(1).integers(0, 256, (224, 224), dtype=np.uint8)
    path = tmp_path / "already_224.png"

    save_224(array, path)

    with Image.open(path) as written:
        assert written.size == (224, 224)
        assert np.array_equal(np.asarray(written), array)


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


def test_manifest_record_rejects_a_probability_vector_of_the_wrong_length():
    """A length mismatch would silently misalign a probability with the wrong
    class index everywhere the frontend reads `reference` — manifest_record
    must catch it rather than write a corrupt record."""
    with pytest.raises(ValueError, match=str(len(CLASS_NAMES))):
        manifest_record(
            image_id="X", true_class="Normal", note="", lar=0.3,
            raw_probs=np.array([0.5, 0.5]),
            lungs_removed_probs=np.zeros(len(CLASS_NAMES)),
        )


def test_manifest_record_rejects_an_unknown_class():
    with pytest.raises(ValueError, match="Pneumothorax"):
        manifest_record(
            image_id="X", true_class="Pneumothorax", note="", lar=0.3,
            raw_probs=np.zeros(4), lungs_removed_probs=np.zeros(4),
        )


def test_sanitize_id_replaces_spaces_with_underscores():
    assert sanitize_id("Viral Pneumonia-1283") == "Viral_Pneumonia-1283"


def test_sanitize_id_is_idempotent_on_already_safe_ids():
    assert sanitize_id("COVID-1001") == "COVID-1001"


def test_sanitized_selection_ids_are_path_and_manifest_safe():
    """Directory names and manifest ids share one source of truth
    (sanitize_id). A raw space — or any other character deploy tooling like
    S3 sync or a tar/zip round-trip doesn't handle reliably — must never
    survive into either."""
    for image_id, _ in SELECTION:
        assert re.fullmatch(r"[A-Za-z0-9_-]+", sanitize_id(image_id))
