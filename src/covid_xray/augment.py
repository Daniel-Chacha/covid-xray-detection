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
