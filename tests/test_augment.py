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
