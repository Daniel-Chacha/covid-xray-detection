"""Generate the browser demo's gallery assets.

Twelve test-set images, each rendered three ways, plus a manifest carrying the
Keras reference probabilities the frontend compares its live output against.

Every image is written at exactly 224x224. That is a correctness requirement,
not a convenience: if the browser resizes, its bilinear kernel may differ from
TensorFlow's and the pixels reaching the model drift from what it was trained
on, silently and with no error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from covid_xray.config import CLASS_NAMES
from covid_xray.data import apply_variant
from covid_xray.gradcam import grad_cam, lung_attribution_ratio
from covid_xray.models import split_feature_and_head
from covid_xray.splits import load_manifest

SIZE = 224

# (image_id, note). Hand-picked to make the argument, not to be representative
# — the page says so. Real ids from the test manifest, chosen by reading
# reports/attribution_ratios.csv and reports/run1_raw_test_probs.npy rather
# than guessing; see .superpowers/sdd/2026-08-05-browser-demo/task-3-report.md
# for the selection rationale.
SELECTION: list[tuple[str, str]] = [
    # --- COVID ---
    (
        "COVID-1001",
        "Lowest-attribution COVID case (LAR ~0.04): Grad-CAM sits on the 'D' "
        "scan marker, not the lungs, and both models misclassify it as "
        "Lung_Opacity (raw ~80%, lungs-removed ~79%) — a shortcut that is also wrong.",
    ),
    (
        "COVID-2038",
        "High-confidence error: true COVID, raw model predicts Normal at ~99% "
        "confidence even though attribution sits mostly inside the lungs (LAR ~0.46).",
    ),
    (
        "COVID-1264",
        "Highest COVID LAR in the set (~0.62): despite heavy attribution inside "
        "the lungs, the raw model still misclassifies it as Normal (~79%); the "
        "lungs-removed model, using only non-lung anatomy, gets it right (COVID ~95%).",
    ),
    # --- Lung_Opacity ---
    (
        "Lung_Opacity-449",
        "Lowest Lung_Opacity LAR in the set (~0.007): the raw model still "
        "correctly predicts Lung_Opacity (~62%) from almost no lung attribution, "
        "but the lungs-removed model gets it wrong (Normal ~56%) once that cue is erased.",
    ),
    (
        "Lung_Opacity-4106",
        "The single most confident mistake in the test set: true Lung_Opacity, "
        "both models predict Normal with near-identical confidence (raw ~99.95%, "
        "lungs-removed ~99%).",
    ),
    (
        "Lung_Opacity-5949",
        "Highest Lung_Opacity LAR in the set (~0.50): both models correctly "
        "predict Lung_Opacity, though confidence diverges (raw ~96%, lungs-removed ~63%).",
    ),
    # --- Normal ---
    (
        "Normal-9497",
        "Lowest Normal LAR in the set (~0.007): both models misclassify it as "
        "Lung_Opacity with closely matching confidence (raw ~87%, lungs-removed ~89%).",
    ),
    (
        "Normal-832",
        "High-confidence error with near-identical agreement: true Normal, both "
        "models predict Viral Pneumonia at ~99% confidence (raw and lungs-removed "
        "differ by well under 1 point).",
    ),
    (
        "Normal-3209",
        "Highest Normal LAR in the set (~0.60): both models correctly predict "
        "Normal in close agreement (raw ~83%, lungs-removed ~86%).",
    ),
    # --- Viral Pneumonia ---
    (
        "Viral Pneumonia-1283",
        "Lowest Viral Pneumonia LAR in the set (~0.075): the raw model barely "
        "guesses Normal (~54%, close to chance) while the lungs-removed model, "
        "from non-lung anatomy alone, correctly predicts Viral Pneumonia (~79%).",
    ),
    (
        "Viral Pneumonia-668",
        "True Viral Pneumonia, low LAR (~0.15): both models agree closely but "
        "share the same mistake, predicting Normal (raw ~91%, lungs-removed ~93%).",
    ),
    (
        "Viral Pneumonia-422",
        "Highest Viral Pneumonia LAR in the set (~0.54): both models correctly "
        "predict Viral Pneumonia in close agreement (raw ~99%, lungs-removed ~98%).",
    ),
]


def save_224(array: np.ndarray, path: Path) -> None:
    """Write a uint8 array as a 224x224 PNG, resizing only if needed."""
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    if image.size != (SIZE, SIZE):
        image = image.resize((SIZE, SIZE), Image.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def manifest_record(
    image_id: str,
    true_class: str,
    note: str,
    lar: float,
    raw_probs: np.ndarray,
    lungs_removed_probs: np.ndarray,
) -> dict:
    """One manifest entry. Raises if the class is not one of the canonical four."""
    if true_class not in CLASS_NAMES:
        raise ValueError(f"unknown class {true_class!r}; expected one of {CLASS_NAMES}")

    return {
        "id": image_id,
        "true_class": true_class,
        "note": note,
        "lar": float(lar),
        "reference": {
            "raw": [float(p) for p in raw_probs],
            "lungs_removed": [float(p) for p in lungs_removed_probs],
        },
    }


def _load_pair(data_root: Path, row) -> tuple[np.ndarray, np.ndarray]:
    """Return (image, mask) as float32 224x224 arrays, mask binary 0/1."""
    with Image.open(data_root / row["path"]) as handle:
        image = np.asarray(handle.convert("L").resize((SIZE, SIZE), Image.BILINEAR), np.float32)
    with Image.open(data_root / row["mask_path"]) as handle:
        # Masks ship as RGB despite being binary; NEAREST because interpolating
        # a binary mask invents intermediate values at the lung boundary.
        mask = np.asarray(handle.convert("L").resize((SIZE, SIZE), Image.NEAREST), np.float32)
    return image, (mask > 127).astype(np.float32)


def build_gallery(
    selection: list[tuple[str, str]],
    data_root: Path,
    ckpt_root: Path,
    out_dir: Path,
) -> list[dict]:
    """Write assets for every selected image and return the manifest records."""
    import tensorflow as tf
    from tensorflow import keras

    repo = Path(__file__).resolve().parent.parent
    test_manifest = load_manifest(repo / "data" / "splits" / "test.csv")
    by_stem = {Path(p).stem: row for p, row in zip(test_manifest["path"], test_manifest.to_dict("records"))}

    models = {
        "raw": keras.models.load_model(ckpt_root / "run1_raw_final.keras"),
        "lungs_removed": keras.models.load_model(ckpt_root / "run4_lungs_removed_final.keras"),
    }
    split = split_feature_and_head(models["raw"])

    records = []
    for image_id, note in selection:
        row = by_stem[image_id]
        image, mask = _load_pair(data_root, row)

        # apply_variant is the single owner of what "lungs_removed" means —
        # calling it here (rather than re-deriving `image * (1 - mask)`)
        # guarantees the demo feeds the model exactly what training fed it.
        image_hw1 = tf.convert_to_tensor(image[..., None], dtype=tf.float32)
        mask_hw1 = tf.convert_to_tensor(mask[..., None], dtype=tf.float32)
        lungs_erased = apply_variant(image_hw1, mask_hw1, "lungs_removed").numpy()[..., 0]

        variants = {
            "raw": image,
            "lungs_erased": lungs_erased,
        }
        for name, pixels in variants.items():
            save_224(pixels, out_dir / image_id / f"{name}.png")

        probs = {}
        for variant, source in (("raw", "raw"), ("lungs_removed", "lungs_erased")):
            batch = keras.applications.densenet.preprocess_input(
                np.repeat(variants[source][None, ..., None], 3, axis=-1).astype("float32")
            )
            probs[variant] = models[variant].predict(batch, verbose=0)[0]

        heatmap = grad_cam(
            models["raw"],
            keras.applications.densenet.preprocess_input(
                np.repeat(image[None, ..., None], 3, axis=-1).astype("float32")
            ),
            split=split,
        )[0]
        save_224(_overlay(image, heatmap), out_dir / image_id / "gradcam.png")

        records.append(
            manifest_record(
                image_id=image_id,
                true_class=row["class_name"],
                note=note,
                lar=lung_attribution_ratio(heatmap, mask * 255.0),
                raw_probs=probs["raw"],
                lungs_removed_probs=probs["lungs_removed"],
            )
        )

    (out_dir / "manifest.json").write_text(json.dumps(records, indent=2))
    return records


def _overlay(grayscale: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-coloured heatmap over the grayscale image."""
    import matplotlib

    base = np.repeat(grayscale[..., None], 3, axis=-1) / 255.0
    # matplotlib.cm.get_cmap was removed in Matplotlib 3.9+; this repo runs 3.11.
    colour = matplotlib.colormaps["jet"](heatmap)[..., :3]
    return np.clip((1 - alpha) * base + alpha * colour, 0, 1) * 255.0


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    build_gallery(
        selection=SELECTION,
        data_root=repo / "data" / "raw" / "COVID-19_Radiography_Dataset",
        ckpt_root=repo / "checkpoints",
        out_dir=repo / "web" / "public" / "gallery",
    )
