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


def test_build_dataset_refuses_the_probe_variant(manifest, synthetic_dataset):
    """downsample8 has its own extraction path; silently serving full-resolution
    images here would turn the shortcut probe into a second raw baseline."""
    cfg = RunConfig(name="t", variant="downsample8", model="logreg8", image_size=32)
    with pytest.raises(ValueError, match="extract_downsampled_features"):
        build_dataset(manifest, synthetic_dataset, cfg, training=False)


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
