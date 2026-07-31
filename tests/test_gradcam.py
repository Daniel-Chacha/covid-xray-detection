import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from tensorflow import keras

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
    """Wires the head deterministically so the comparison is meaningful.

    With the randomly initialised head both class maps come back all-zero.
    That is a real Grad-CAM outcome, not a bug: through the softmax,
    d(p_0)/d(feature_c) is positive for the channels class 0 reads but
    *negative* for those another likely class reads, so the map is a
    difference of comparable sums that ReLU can clip to nothing. Two
    degenerate maps are trivially equal, so the test would fail while saying
    nothing about whether class_index is honoured.

    The fix has two parts. Class 0 reads the first half of the feature
    channels and class 3 the second half, so their maps are spatially
    distinct. And bias parks almost all probability mass on class 1, which
    drives the p_0*p_3 cross-term to nearly zero and leaves each map a
    positive weighted sum. Verified: both peak at 1.0 after normalisation.
    """
    dense = next(l for l in model.layers if isinstance(l, keras.layers.Dense))
    kernel, bias = dense.get_weights()
    half = kernel.shape[0] // 2
    kernel[:] = 0.0
    kernel[:half, 0] = 1e-3
    kernel[half:, 3] = 1e-3
    bias = np.zeros_like(bias)
    bias[1] = 10.0
    dense.set_weights([kernel, bias])

    batch = tf.random.uniform((1, 64, 64, 3), seed=2)
    first = grad_cam(model, batch, class_index=0)
    second = grad_cam(model, batch, class_index=3)

    assert first.max() > 0, "class 0 map is degenerate; the test cannot discriminate"
    assert second.max() > 0, "class 3 map is degenerate; the test cannot discriminate"
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


def test_attribution_ratio_accepts_an_rgb_mask():
    """The dataset ships binary masks encoded as RGB — a PIL read gives (H, W, 3)."""
    heatmap = np.ones((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8, 3), dtype=np.uint8)
    mask[:, :4, :] = 255
    assert lung_attribution_ratio(heatmap, mask) == pytest.approx(0.5)


def test_precomputed_split_gives_identical_heatmaps(model):
    """The `split` fast path must not change the result, only the cost."""
    from covid_xray.models import split_feature_and_head

    batch = tf.random.uniform((2, 64, 64, 3), seed=3)
    rebuilt = grad_cam(model, batch)
    reused = grad_cam(model, batch, split=split_feature_and_head(model))
    assert np.allclose(rebuilt, reused, atol=1e-5)


def test_attribution_ratio_of_an_empty_heatmap_is_nan():
    heatmap = np.zeros((8, 8), dtype=np.float32)
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    assert np.isnan(lung_attribution_ratio(heatmap, mask))
