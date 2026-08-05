import re

import keras
import numpy as np
import pytest
from PIL import Image

from scripts.build_gallery import (
    CLASS_NAMES,
    SELECTION,
    _force_float32,
    manifest_record,
    sanitize_id,
    save_224,
)


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


def _assert_recursively_float32(layer):
    policy_name = layer.dtype_policy.name
    assert policy_name == "float32", f"{layer.name!r} kept dtype policy {policy_name!r}"
    for sub in getattr(layer, "layers", []):
        _assert_recursively_float32(sub)


def test_force_float32_recurses_into_nested_sublayers():
    """Regression test for the fix in commit 3d5457e: the DenseNet121
    checkpoints carry a `mixed_float16` dtype policy (float32 weights,
    float16 activations), and the manifest's Keras reference probabilities
    were originally computed under that policy while the TF.js export
    deliberately forces float32 (tfjs has no float16 tensor dtype). That made
    the *reference* lossy, not the exported graph, and the parity gate failed
    at 5.27e-3 until `_force_float32` was added.

    The real checkpoint's mixed precision lives on layers nested inside a
    sub-model (the DenseNet121 backbone sits inside the outer model as a
    layer with its own `.layers`), which is the actual failure mode a flat
    model would not catch: `_force_float32` only recurses through
    `getattr(layer, "layers", [])`, so a top-level-only model could pass this
    test even with a version of the fix that forgot to recurse. This builds
    a small model with exactly that nesting instead of loading the real
    64 MB checkpoints, to keep the suite fast.
    """
    original_policy = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        inner_backbone = keras.Sequential(
            [keras.layers.Dense(4, activation="relu"), keras.layers.Dense(3)],
            name="inner_backbone",
        )
        inputs = keras.Input(shape=(5,))
        features = inner_backbone(inputs)
        outputs = keras.layers.Dense(2)(features)
        model = keras.Model(inputs, outputs)

        # Confirm the fixture actually produced mixed precision on the nested
        # sublayers before asserting the fix undoes it — otherwise this test
        # would pass vacuously regardless of what _force_float32 does.
        assert inner_backbone.layers[0].dtype_policy.name == "mixed_float16"
        assert inner_backbone.layers[1].dtype_policy.name == "mixed_float16"

        for layer in model.layers:
            _force_float32(layer)

        for layer in model.layers:
            _assert_recursively_float32(layer)
    finally:
        keras.mixed_precision.set_global_policy(original_policy)
