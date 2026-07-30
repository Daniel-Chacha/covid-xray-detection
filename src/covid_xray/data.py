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
    if cfg.variant == "downsample8":
        # The 8x8 probe is a scikit-learn model wanting everything in memory.
        # Without this guard it would silently receive full-resolution raw
        # images and quietly stop being a probe.
        raise ValueError("variant 'downsample8' uses extract_downsampled_features, not build_dataset")

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
            # channels=1 collapses the dataset's RGB-encoded masks to one
            # channel. Nearest-neighbour because masks are binary and must not
            # be interpolated into intermediate values at the lung boundary.
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
        ds = ds.shuffle(
            buffer_size=min(len(manifest), 4096), seed=cfg.seed, reshuffle_each_iteration=True
        )

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
