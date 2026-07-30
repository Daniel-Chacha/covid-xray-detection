import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from tensorflow import keras

from covid_xray.config import RunConfig
from covid_xray.models import (
    build_densenet121,
    build_probe,
    get_base_model,
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


def test_base_is_frozen_immediately_after_build(cfg):
    """Builds its own model — the shared fixture gets unfrozen by later tests."""
    fresh = build_densenet121(cfg)
    assert get_base_model(fresh).trainable is False


def test_stage_b_unfreezes_from_the_named_block(model):
    """Asserts the freeze boundary, not specific layer names.

    Keras 3 renamed DenseNet's early layers (conv1/conv -> conv1_conv) because
    slashes are no longer permitted. Pinning a name here would break on the
    next such change while telling us nothing about whether the cutoff works.
    """
    count = set_finetune_trainable(model, "conv5_block1")
    base = get_base_model(model)
    names = [layer.name for layer in base.layers]
    cutoff = next(i for i, name in enumerate(names) if name.startswith("conv5_block1"))

    assert count > 0
    assert not any(layer.trainable for layer in base.layers[:cutoff])
    # Not *all* of the tail: BatchNorm stays frozen throughout.
    assert any(layer.trainable for layer in base.layers[cutoff:])


def test_batchnorm_stays_frozen_after_unfreezing(model):
    """The single most common silent Keras fine-tuning bug."""
    set_finetune_trainable(model, "conv5_block1")
    base = get_base_model(model)
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
