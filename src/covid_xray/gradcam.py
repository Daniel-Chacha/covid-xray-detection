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
    the dataset's 256x256 masks and a 224x224 heatmap compare correctly.
    Returns NaN for an all-zero heatmap, which has no mass to attribute.

    The shipped masks are stored as RGB despite being binary, so a PIL read
    yields (H, W, 3). Collapsing to one channel is required before indexing a
    2-D heatmap; without it this raises on the dimension mismatch.
    """
    heatmap = np.asarray(heatmap, dtype=np.float64)
    total = heatmap.sum()
    if total <= 0:
        return float("nan")

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
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
