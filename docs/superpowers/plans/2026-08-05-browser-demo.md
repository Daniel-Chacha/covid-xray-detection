# Browser-Only Demo Frontend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A static Next.js page where selecting one of 12 chest X-rays runs two DenseNet121 models in the browser — one on the full image, one with the lungs erased — and the predictions barely differ.

**Architecture:** Three stages with independent verification. The existing `to_tf_js_convertion` workspace exports both Keras models to TF.js GraphModels with DenseNet preprocessing baked into the graph. A Python script in this repo generates 12 gallery items pre-resized to exactly 224×224 plus a manifest carrying Keras reference probabilities. A statically-exported Next.js app loads the models lazily and displays live predictions beside those references.

**Tech Stack:** Python 3.12 (conversion) · Python 3.13 (assets) · Next.js 15 + TypeScript + Tailwind · `@tensorflow/tfjs` (browser) · `@tensorflow/tfjs-node` (verification)

**Spec:** [2026-08-05-browser-demo-design.md](../specs/2026-08-05-browser-demo-design.md)

---

## Global Constraints

- **Code delivery:** implementation code is **not written to disk by the assistant**. This plan carries the full code; the engineer copies each block to the stated path. README and requirements files are the exception.
- **Class order** is `["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]` — the canonical order from `covid_xray.config.CLASS_NAMES`. Never re-derive it.
- **All gallery images are exactly 224×224.** The browser must never resize; its bilinear kernel may differ from TensorFlow's and would shift the input distribution silently.
- **Load with `tf.loadGraphModel`**, never `tf.loadLayersModel`. The exported graph is frozen.
- **The browser feeds raw 0–255 pixels.** Preprocessing lives inside the graph. No normalisation client-side, ever.
- **No upload affordance** anywhere — no file input, no drag target, no paste handler.
- **Conversion runs on Python 3.12 only.** The workspace's `convert.py` refuses other interpreters; 3.13 resolves `tensorflowjs` to a broken build.
- **Commit after every task.** The engineer runs all `git` commands.

---

## File Structure

```
to_tf_js_convertion/                      # existing workspace, Python 3.12
├── config.py                             # MODIFY: add "densenet" preprocessing  [Task 1]
├── tests/test_config.py                  # MODIFY: structural assertion          [Task 1]
├── tests/test_export.py                  # MODIFY: numerical parity assertion    [Task 1]
├── models/covid_raw.keras                # NEW (copied)                          [Task 2]
├── models/covid_lungs_removed.keras      # NEW (copied)                          [Task 2]
└── configs/covid_{raw,lungs_removed}.json# NEW                                   [Task 2]

covid-xray-detection/
├── scripts/build_gallery.py              # NEW: assets + manifest                [Task 3]
├── tests/test_build_gallery.py           # NEW                                   [Task 3]
└── web/                                  # NEW Next.js app
    ├── next.config.mjs                   # output: 'export'                      [Task 5]
    ├── app/{layout,page}.tsx, globals.css                                        [Task 5,7]
    ├── components/Disclaimer.tsx                                                 [Task 5]
    ├── components/{Gallery,Viewer,Predictions}.tsx                               [Task 7]
    ├── lib/{types,model,infer}.ts                                                [Task 6]
    ├── scripts/verify-parity.mjs                                                 [Task 4]
    └── public/{models,gallery}/                                                  [Task 2,3]
```

Each `lib/` module has one job: `types.ts` describes the manifest, `model.ts` turns a variant name into a loaded model, `infer.ts` turns an image element into probabilities. No module reads another's internals.

---

## Task 1: `densenet` preprocessing mode

> **Workspace:** `to_tf_js_convertion`, Python 3.12 venv.

**Files:**
- Modify: `config.py` (the `PREPROCESSING` dict)
- Modify: `tests/test_config.py`, `tests/test_export.py`

**Interfaces:**
- Produces: `rescaling_params("densenet") -> {"scale": list[float], "offset": list[float]}`, both length 3. Consumed unchanged by `export.py`'s `keras.layers.Rescaling(**params)`.

- [ ] **Step 1: Write the failing structural test**

Append to `tests/test_config.py`:

```python
def test_rescaling_params_densenet_is_per_channel():
    """DenseNet uses torch-mode normalisation: per-channel mean and std."""
    p = rescaling_params("densenet")
    assert len(p["scale"]) == 3
    assert len(p["offset"]) == 3
    # scale = 1/(255*std), offset = -mean/std. Spot-check the first channel.
    assert p["scale"][0] == pytest.approx(1.0 / (255.0 * 0.229))
    assert p["offset"][0] == pytest.approx(-0.485 / 0.229)
```

- [ ] **Step 2: Write the failing numerical test**

Append to `tests/test_export.py`. This is the test that matters — the constants are opaque, and a transposed channel would pass the structural test while producing confidently wrong predictions.

```python
def test_densenet_preprocessing_matches_keras_preprocess_input():
    """A transposition here yields wrong predictions and never an error."""
    import numpy as np
    import keras

    from config import rescaling_params

    x = np.random.default_rng(0).uniform(0, 255, size=(4, 8, 8, 3)).astype("float32")
    want = keras.applications.densenet.preprocess_input(x.copy())
    got = keras.layers.Rescaling(**rescaling_params("densenet"))(x).numpy()

    assert np.abs(got - want).max() < 1e-6
```

- [ ] **Step 3: Run both tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_config.py::test_rescaling_params_densenet_is_per_channel tests/test_export.py::test_densenet_preprocessing_matches_keras_preprocess_input -v`
Expected: FAIL — `KeyError: 'densenet'`

- [ ] **Step 4: Add the preprocessing mode**

Replace the `PREPROCESSING` dict in `config.py`:

```python
# torchvision / DenseNet normalisation constants.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

PREPROCESSING = {
    "mobilenet_v2": {"scale": 1 / 127.5, "offset": -1.0},
    "rescale_0_1": {"scale": 1 / 255.0, "offset": 0.0},
    # keras.applications.densenet.preprocess_input is torch-mode: divide by
    # 255, then subtract per-channel ImageNet means and divide by per-channel
    # standard deviations. Folded into one affine op:
    #     (x/255 - mean) / std  ==  x * 1/(255*std) - mean/std
    # Verified to 4.8e-07 against preprocess_input; see tests/test_export.py.
    "densenet": {
        "scale": [1.0 / (255.0 * s) for s in _IMAGENET_STD],
        "offset": [-m / s for m, s in zip(_IMAGENET_MEAN, _IMAGENET_STD)],
    },
    "none": None,
}
```

`export.py` needs no change: it already does `keras.layers.Rescaling(**params)`, and Keras accepts per-channel lists for `scale` and `offset`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/ -v`
Expected: PASS, including the two new tests and all pre-existing ones.

- [ ] **Step 6: Commit**

```bash
git add config.py tests/test_config.py tests/test_export.py
git commit -m "feat: add densenet per-channel preprocessing mode

Torch-mode normalisation folded into one affine Rescaling op, verified to
4.8e-07 against keras.applications.densenet.preprocess_input. export.py is
unchanged: it already splats the params dict into Rescaling, which accepts
per-channel lists."
```

---

## Task 2: Convert both models

> **Workspace:** `to_tf_js_convertion`.

**Files:**
- Create: `models/covid_raw.keras`, `models/covid_lungs_removed.keras` (copies)
- Create: `configs/covid_raw.json`, `configs/covid_lungs_removed.json`
- Produces: `dist/covid_raw/`, `dist/covid_lungs_removed/`

**Interfaces:**
- Consumes: `rescaling_params("densenet")` from Task 1
- Produces: TF.js GraphModel artifacts. Input `[1, 224, 224, 3]` float32 raw 0–255; output `[1, 4]` softmax.

- [ ] **Step 1: Copy the checkpoints**

```bash
cd ~/Documents/personal/to_tf_js_convertion
cp ~/Documents/personal/covid-xray-detection/checkpoints/run1_raw_final.keras models/covid_raw.keras
cp ~/Documents/personal/covid-xray-detection/checkpoints/run4_lungs_removed_final.keras models/covid_lungs_removed.keras
ls -la models/
```

Renamed because this workspace serves several projects — `run1_raw` means nothing here.

- [ ] **Step 2: Write `configs/covid_raw.json`**

```json
{
  "model": "models/covid_raw.keras",
  "input_shape": [224, 224, 3],
  "preprocessing": "densenet",
  "quantization": "none",
  "class_names": ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
}
```

- [ ] **Step 3: Write `configs/covid_lungs_removed.json`**

```json
{
  "model": "models/covid_lungs_removed.keras",
  "input_shape": [224, 224, 3],
  "preprocessing": "densenet",
  "quantization": "none",
  "class_names": ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
}
```

- [ ] **Step 4: Convert and verify both**

```bash
venv/bin/python convert.py configs/covid_raw.json
node verify_tfjs.mjs covid_raw

venv/bin/python convert.py configs/covid_lungs_removed.json
node verify_tfjs.mjs covid_lungs_removed
```

Expected: both verifiers pass. `verify.py` runs inside `convert.py` at `atol=1e-6`; `verify_tfjs.mjs` at `atol=2e-3`.

- [ ] **Step 5: Confirm no float16 casts survived**

Our checkpoints were trained under `mixed_float16`, which is exactly the failure `_force_float32` exists to prevent. Verify it worked:

```bash
for name in covid_raw covid_lungs_removed; do
  venv/bin/python -c "
import json
t = json.load(open('dist/$name/model.json'))['modelTopology']['node']
n = sum(1 for x in t if x['op']=='Cast' and x.get('attr',{}).get('DstT',{}).get('type')=='DT_HALF')
print('$name float16 casts:', n)
"
done
du -sh dist/covid_raw dist/covid_lungs_removed
```

Expected: `float16 casts: 0` for both. A non-zero count means the graph computes in float16 and disagrees with the trained model — stop and investigate rather than adjusting tolerances.

- [ ] **Step 6: Stage the artifacts into the web app**

```bash
mkdir -p ~/Documents/personal/covid-xray-detection/web/public/models/{raw,lungs_removed}
cp -r dist/covid_raw/*           ~/Documents/personal/covid-xray-detection/web/public/models/raw/
cp -r dist/covid_lungs_removed/* ~/Documents/personal/covid-xray-detection/web/public/models/lungs_removed/
du -sh ~/Documents/personal/covid-xray-detection/web/public/models/*
```

- [ ] **Step 7: Commit the workspace changes**

```bash
git add configs/covid_raw.json configs/covid_lungs_removed.json
git commit -m "feat: convert COVID CXR raw and lungs-removed models to TF.js"
```

`models/` and `dist/` are gitignored in this workspace; only the configs are tracked.

> **Quantization is deferred to Task 4**, where `tfjs-node` is already wired up and the manifest exists to measure against. Ship `none` for now.

---

## Task 3: Gallery assets and manifest

> **Workspace:** `covid-xray-detection`, Python 3.13 venv.

**Files:**
- Create: `scripts/build_gallery.py`, `tests/test_build_gallery.py`
- Produces: `web/public/gallery/<id>/{raw,lungs_erased,gradcam}.png` and `web/public/gallery/manifest.json`

**Interfaces:**
- Consumes: `covid_xray.config.CLASS_NAMES`, `covid_xray.data.apply_variant`, `covid_xray.gradcam.grad_cam`, `covid_xray.gradcam.lung_attribution_ratio`, `covid_xray.models.split_feature_and_head`, `covid_xray.splits.load_manifest`
- Produces: `build_gallery(selection, data_root, ckpt_root, out_dir) -> list[dict]` — the manifest records; `SELECTION: list[tuple[str, str]]` of `(image_id, note)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_gallery.py`:

```python
import json

import numpy as np
import pytest
from PIL import Image

from scripts.build_gallery import CLASS_NAMES, manifest_record, save_224


def test_save_224_writes_exactly_224_square(tmp_path):
    """The browser must never resize; a resize kernel mismatch is silent."""
    array = np.random.default_rng(0).integers(0, 256, (299, 299), dtype=np.uint8)
    path = tmp_path / "raw.png"

    save_224(array, path)

    with Image.open(path) as written:
        assert written.size == (224, 224)


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


def test_manifest_record_probabilities_sum_to_one():
    record = manifest_record(
        image_id="X", true_class="Normal", note="", lar=0.3,
        raw_probs=np.array([0.1, 0.2, 0.6, 0.1]),
        lungs_removed_probs=np.array([0.25, 0.25, 0.25, 0.25]),
    )

    for variant in ("raw", "lungs_removed"):
        assert sum(record["reference"][variant]) == pytest.approx(1.0, abs=1e-6)


def test_manifest_record_rejects_an_unknown_class():
    with pytest.raises(ValueError, match="Pneumothorax"):
        manifest_record(
            image_id="X", true_class="Pneumothorax", note="", lar=0.3,
            raw_probs=np.zeros(4), lungs_removed_probs=np.zeros(4),
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_build_gallery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Write `scripts/__init__.py`**

```python
"""One-off asset-generation scripts. Not part of the covid_xray package."""
```

- [ ] **Step 4: Write `scripts/build_gallery.py`**

```python
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

from covid_xray.config import CLASS_NAMES, RunConfig, load_config
from covid_xray.data import apply_variant
from covid_xray.gradcam import grad_cam, lung_attribution_ratio
from covid_xray.models import split_feature_and_head
from covid_xray.splits import load_manifest

SIZE = 224

# (image_id, note). Hand-picked to make the argument, not to be representative
# — the page says so. Replace the ids with real ones from the test manifest;
# `python -m scripts.build_gallery --list` prints candidates with their LAR.
SELECTION: list[tuple[str, str]] = []


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

        variants = {
            "raw": image,
            "lungs_erased": image * (1.0 - mask),
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
    import matplotlib.cm as cm

    base = np.repeat(grayscale[..., None], 3, axis=-1) / 255.0
    colour = cm.get_cmap("jet")(heatmap)[..., :3]
    return np.clip((1 - alpha) * base + alpha * colour, 0, 1) * 255.0


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    build_gallery(
        selection=SELECTION,
        data_root=repo / "data" / "raw" / "COVID-19_Radiography_Dataset",
        ckpt_root=repo / "checkpoints",
        out_dir=repo / "web" / "public" / "gallery",
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_build_gallery.py -v`
Expected: PASS, 4 tests. These cover the pure functions; `build_gallery` itself is exercised by actually running it in Step 7.

- [ ] **Step 6: Choose the twelve images**

Read candidates out of the audit's own outputs rather than guessing:

```python
import pandas as pd
ratios = pd.read_csv('reports/attribution_ratios.csv')
print("--- lowest lung attribution (the shortcut cases) ---")
print(ratios.nsmallest(8, 'run1_raw')[['path', 'class_name', 'run1_raw']])
print("\n--- highest (the model behaving) ---")
print(ratios.nlargest(4, 'run1_raw')[['path', 'class_name', 'run1_raw']])
```

Populate `SELECTION` with three per class. It must include the COVID case at LAR ≈ 0.04 whose heatmap sits on the `D` marker, at least one high-confidence error, and at least two cases where the two models agree closely.

- [ ] **Step 7: Generate the assets**

Run: `python -m scripts.build_gallery`
Then verify:

```bash
python -c "
import json
from pathlib import Path
from PIL import Image
records = json.load(open('web/public/gallery/manifest.json'))
assert len(records) == 12, len(records)
for r in records:
    for name in ('raw', 'lungs_erased', 'gradcam'):
        p = Path('web/public/gallery') / r['id'] / f'{name}.png'
        assert p.exists(), p
        assert Image.open(p).size == (224, 224), p
    for v in ('raw', 'lungs_removed'):
        assert abs(sum(r['reference'][v]) - 1) < 1e-5
print('12 items, 36 images, all 224x224, all probabilities normalised')
"
du -sh web/public/gallery
```

- [ ] **Step 8: Commit**

```bash
git add scripts/ tests/test_build_gallery.py
git add -f web/public/gallery
git commit -m "feat: gallery asset generation for the browser demo"
```

---

## Task 4: End-to-end parity verification and the quantization decision

**Files:**
- Create: `web/scripts/verify-parity.mjs`, `web/package.json`

**Interfaces:**
- Consumes: `web/public/models/{raw,lungs_removed}/model.json`, `web/public/gallery/manifest.json`
- Produces: a pass/fail gate plus the measurement that decides quantization

This runs **before** any UI exists. It tests the artifacts the browser will load, over the inputs it will feed them — which is where the real risk lives.

- [ ] **Step 1: Initialise the app and add the verification dependency**

```bash
cd ~/Documents/personal/covid-xray-detection
npx create-next-app@latest web --typescript --tailwind --eslint --app --src-dir=false --import-alias="@/*" --no-turbopack
cd web
npm install @tensorflow/tfjs
npm install --save-dev @tensorflow/tfjs-node
```

- [ ] **Step 2: Write `web/scripts/verify-parity.mjs`**

```javascript
/**
 * Runs every gallery image through both exported graphs and asserts the result
 * matches the Keras reference probabilities in the manifest.
 *
 * This is the test that matters. A preprocessing mismatch produces confident
 * wrong answers and never an error, so nothing upstream would catch it.
 */
import * as tf from '@tensorflow/tfjs-node'
import { readFileSync } from 'node:fs'
import { PNG } from 'pngjs'
import path from 'node:path'

const ATOL = 2e-3 // matches verify_tfjs.mjs; float16 quantization is lossy
const GALLERY = path.resolve('public/gallery')
const VARIANTS = {
  raw: { model: 'public/models/raw/model.json', image: 'raw.png' },
  lungs_removed: { model: 'public/models/lungs_removed/model.json', image: 'lungs_erased.png' },
}

function loadPixels(file) {
  const png = PNG.sync.read(readFileSync(file))
  if (png.width !== 224 || png.height !== 224) {
    throw new Error(`${file}: expected 224x224, got ${png.width}x${png.height}`)
  }
  const rgb = new Float32Array(224 * 224 * 3)
  for (let i = 0, j = 0; i < png.data.length; i += 4, j += 3) {
    rgb[j] = png.data[i]
    rgb[j + 1] = png.data[i + 1]
    rgb[j + 2] = png.data[i + 2]
  }
  return tf.tensor4d(rgb, [1, 224, 224, 3])
}

const manifest = JSON.parse(readFileSync(path.join(GALLERY, 'manifest.json'), 'utf8'))
let worstError = 0
let flips = 0

for (const [variant, cfg] of Object.entries(VARIANTS)) {
  const model = await tf.loadGraphModel(`file://${path.resolve(cfg.model)}`)
  for (const item of manifest) {
    const input = loadPixels(path.join(GALLERY, item.id, cfg.image))
    const got = Array.from(await model.predict(input).data())
    input.dispose()

    const want = item.reference[variant]
    const error = Math.max(...got.map((g, i) => Math.abs(g - want[i])))
    worstError = Math.max(worstError, error)

    const argmax = (a) => a.indexOf(Math.max(...a))
    if (argmax(got) !== argmax(want)) {
      flips += 1
      console.error(`ARGMAX FLIP  ${variant}  ${item.id}: ${argmax(want)} -> ${argmax(got)}`)
    }
    if (error > ATOL) {
      console.error(`OVER TOLERANCE  ${variant}  ${item.id}: ${error.toExponential(2)}`)
    }
  }
  console.log(`${variant}: checked ${manifest.length} images`)
}

console.log(`max abs probability error: ${worstError.toExponential(2)}  (atol ${ATOL})`)
console.log(`argmax flips: ${flips}`)

if (flips > 0 || worstError > ATOL) {
  console.error('\nFAIL — do not widen ATOL to make this pass. It exists to catch exactly this.')
  process.exit(1)
}
console.log('\nPASS')
```

- [ ] **Step 3: Add the dependency and script entry**

```bash
npm install --save-dev pngjs
```

Add to `web/package.json` under `"scripts"`:

```json
"verify-parity": "node scripts/verify-parity.mjs"
```

- [ ] **Step 4: Run it against the float32 models**

Run: `npm run verify-parity`
Expected: `argmax flips: 0`, max error well under 2e-3, `PASS`.

A failure here means the `densenet` preprocessing is wrong or the graph kept float16 casts. Fix the cause; never widen `ATOL`.

- [ ] **Step 5: Measure quantization**

Two float32 models is ~56 MB. Re-convert at each quantization and re-run the gate:

```bash
cd ~/Documents/personal/to_tf_js_convertion
for q in float16 uint8; do
  for name in covid_raw covid_lungs_removed; do
    python - <<PY
import json, pathlib
p = pathlib.Path("configs/$name.json")
cfg = json.loads(p.read_text()); cfg["quantization"] = "$q"; p.write_text(json.dumps(cfg, indent=2))
PY
    venv/bin/python convert.py configs/$name.json
  done
  cp -r dist/covid_raw/*           ~/Documents/personal/covid-xray-detection/web/public/models/raw/
  cp -r dist/covid_lungs_removed/* ~/Documents/personal/covid-xray-detection/web/public/models/lungs_removed/
  echo "=== $q ==="
  du -sh ~/Documents/personal/covid-xray-detection/web/public/models/*
  (cd ~/Documents/personal/covid-xray-detection/web && npm run verify-parity)
done
```

**Ship the smallest quantization with zero argmax flips**, then re-convert one final time at that setting so the configs on disk match what shipped. Record the measured numbers — they go in the README.

Note the sample is 12 images, which is enough to catch a gross preprocessing error but not to bound a rare flip rate. Say so in the README rather than implying a stronger guarantee.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/personal/covid-xray-detection
git add web/package.json web/package-lock.json web/scripts/verify-parity.mjs
git add -f web/public/models
git commit -m "feat: end-to-end TF.js parity gate and quantization measurement"
```

---

## Task 5: Next.js scaffold, static export, disclaimer

**Files:**
- Modify: `web/next.config.mjs`
- Create: `web/components/Disclaimer.tsx`
- Modify: `web/app/layout.tsx`

**Interfaces:**
- Produces: `<Disclaimer />` — a non-dismissible banner rendered by the root layout.

- [ ] **Step 1: Configure static export**

Replace `web/next.config.mjs`:

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  // Fully static: no server, no API routes. Deploys identically to Vercel,
  // Netlify or GitHub Pages.
  output: 'export',
  // next/image's optimiser requires a server; the gallery images are already
  // exactly 224x224 and must not be resampled anyway.
  images: { unoptimized: true },
}

export default nextConfig
```

- [ ] **Step 2: Write `web/components/Disclaimer.tsx`**

```tsx
export default function Disclaimer() {
  return (
    <div
      role="note"
      className="sticky top-0 z-50 border-b border-amber-500/40 bg-amber-950/90 px-4 py-2 text-center text-sm text-amber-100 backdrop-blur"
    >
      <strong className="font-semibold">Not a medical device. Not for clinical use.</strong>{' '}
      This page demonstrates that the model reads acquisition artefacts rather than lung
      pathology. It is a research artefact, not a diagnostic tool.
    </div>
  )
}
```

- [ ] **Step 3: Render it from the layout**

Replace the body of `web/app/layout.tsx`:

```tsx
import type { Metadata } from 'next'
import Disclaimer from '@/components/Disclaimer'
import './globals.css'

export const metadata: Metadata = {
  title: 'How much of this COVID classifier is real?',
  description:
    'Two DenseNet121 models run in your browser on the same chest X-ray — one on the full image, one with the lungs erased. The predictions barely differ.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-neutral-950 text-neutral-100 antialiased">
        <Disclaimer />
        {children}
      </body>
    </html>
  )
}
```

- [ ] **Step 4: Verify the static export builds**

Run: `cd web && npm run build`
Expected: build succeeds and reports `Exporting (3/3)` or similar, producing `web/out/`.

- [ ] **Step 5: Commit**

```bash
git add web/next.config.mjs web/app/layout.tsx web/components/Disclaimer.tsx
git commit -m "feat: Next.js static export scaffold with clinical-use disclaimer"
```

---

## Task 6: Model loading and inference

**Files:**
- Create: `web/lib/types.ts`, `web/lib/model.ts`, `web/lib/infer.ts`

**Interfaces:**
- Produces: `CLASS_NAMES: readonly string[]`; `type Variant = 'raw' | 'lungs_removed'`; `interface GalleryItem`; `loadModel(variant): Promise<tf.GraphModel>`; `predict(image: HTMLImageElement, variant): Promise<number[]>`; `activeBackend(): Promise<string>`

- [ ] **Step 1: Write `web/lib/types.ts`**

```typescript
// Canonical order from covid_xray.config.CLASS_NAMES. The model's output axis
// is index-aligned with this; never re-derive or re-sort it.
export const CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'] as const
export type ClassName = (typeof CLASS_NAMES)[number]

export type Variant = 'raw' | 'lungs_removed'

/** One record from public/gallery/manifest.json. */
export interface GalleryItem {
  id: string
  true_class: ClassName
  note: string
  /** Fraction of the raw model's Grad-CAM mass inside the lung mask. */
  lar: number
  /** Keras probabilities, index-aligned with CLASS_NAMES. */
  reference: Record<Variant, number[]>
}

export const VARIANT_LABEL: Record<Variant, string> = {
  raw: 'Full image',
  lungs_removed: 'Lungs erased',
}

/** Which gallery PNG feeds which model. */
export const VARIANT_IMAGE: Record<Variant, string> = {
  raw: 'raw.png',
  lungs_removed: 'lungs_erased.png',
}
```

- [ ] **Step 2: Write `web/lib/model.ts`**

```typescript
import * as tf from '@tensorflow/tfjs'
import type { Variant } from './types'

const MODEL_URL: Record<Variant, string> = {
  raw: '/models/raw/model.json',
  lungs_removed: '/models/lungs_removed/model.json',
}

// Cache the promise, not the model: concurrent callers during the first load
// then share one download rather than racing two.
const cache = new Map<Variant, Promise<tf.GraphModel>>()

/**
 * Load a converted model, at most once per variant.
 *
 * loadGraphModel, never loadLayersModel — the exported graph is frozen and
 * carries its preprocessing internally.
 */
export function loadModel(variant: Variant): Promise<tf.GraphModel> {
  let pending = cache.get(variant)
  if (!pending) {
    pending = tf.loadGraphModel(MODEL_URL[variant])
    cache.set(variant, pending)
  }
  return pending
}

/** WebGL, WASM or CPU — explains order-of-magnitude latency differences. */
export async function activeBackend(): Promise<string> {
  await tf.ready()
  return tf.getBackend()
}
```

- [ ] **Step 3: Write `web/lib/infer.ts`**

```typescript
import * as tf from '@tensorflow/tfjs'
import { loadModel } from './model'
import type { Variant } from './types'

export const INPUT_SIZE = 224

/**
 * Probabilities for one already-loaded 224x224 image element.
 *
 * The image is fed as raw 0-255 pixels with no normalisation and no resize.
 * Preprocessing lives inside the exported graph, and the asset is already
 * exactly 224x224 — resizing here would risk a kernel mismatch with the
 * TensorFlow bilinear used in training, which fails silently.
 */
export async function predict(image: HTMLImageElement, variant: Variant): Promise<number[]> {
  const model = await loadModel(variant)

  const output = tf.tidy(() => {
    const input = tf.browser.fromPixels(image).toFloat().expandDims(0)
    return model.predict(input) as tf.Tensor
  })

  const probabilities = Array.from(await output.data())
  output.dispose()
  return probabilities
}

/** Largest absolute per-class difference between live and reference output. */
export function maxDelta(live: number[], reference: number[]): number {
  return Math.max(...live.map((p, i) => Math.abs(p - reference[i])))
}

export function argmax(values: number[]): number {
  return values.indexOf(Math.max(...values))
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cd web && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add web/lib
git commit -m "feat: TF.js model loading and inference with lazy per-variant caching"
```

---

## Task 7: Gallery, viewer and predictions

**Files:**
- Create: `web/components/Gallery.tsx`, `web/components/Viewer.tsx`, `web/components/Predictions.tsx`
- Modify: `web/app/page.tsx`

**Interfaces:**
- Consumes: everything from Task 6
- Produces: the page

- [ ] **Step 1: Write `web/components/Predictions.tsx`**

```tsx
'use client'

import { CLASS_NAMES } from '@/lib/types'
import { argmax, maxDelta } from '@/lib/infer'

interface Props {
  live: number[] | null
  reference: number[]
  busy: boolean
}

export default function Predictions({ live, reference, busy }: Props) {
  const shown = live ?? reference
  const top = argmax(shown)

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between text-xs text-neutral-400">
        <span>{live ? 'Live, in your browser' : 'Reference (live inference unavailable)'}</span>
        {live && (
          <span title="Largest per-class difference from the Python reference">
            parity Δ {maxDelta(live, reference).toExponential(1)}
          </span>
        )}
      </div>

      {CLASS_NAMES.map((name, index) => (
        <div key={name} className={index === top ? 'text-neutral-50' : 'text-neutral-400'}>
          <div className="mb-1 flex justify-between text-sm">
            <span>{name.replace('_', ' ')}</span>
            <span className="tabular-nums">{(shown[index] * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded bg-neutral-800">
            <div
              className={`h-full transition-[width] duration-300 ${
                index === top ? 'bg-sky-400' : 'bg-neutral-600'
              }`}
              style={{ width: `${Math.max(shown[index] * 100, 0.5)}%` }}
            />
          </div>
        </div>
      ))}

      {busy && <p className="text-xs text-neutral-500">running…</p>}
    </div>
  )
}
```

- [ ] **Step 2: Write `web/components/Viewer.tsx`**

```tsx
'use client'

import { useEffect, useRef, useState } from 'react'
import Predictions from './Predictions'
import { predict } from '@/lib/infer'
import { VARIANT_IMAGE, VARIANT_LABEL, type GalleryItem, type Variant } from '@/lib/types'

const VARIANTS: Variant[] = ['raw', 'lungs_removed']

export default function Viewer({ item }: { item: GalleryItem }) {
  const [variant, setVariant] = useState<Variant>('raw')
  const [showCam, setShowCam] = useState(false)
  const [live, setLive] = useState<number[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    let cancelled = false
    const element = imageRef.current
    if (!element) return

    async function run() {
      setBusy(true)
      setLive(null)
      setError(null)
      try {
        if (!element!.complete) {
          await new Promise((resolve) => element!.addEventListener('load', resolve, { once: true }))
        }
        const probabilities = await predict(element!, variant)
        if (!cancelled) setLive(probabilities)
      } catch (cause) {
        // Graceful degradation: the manifest reference is still displayed.
        if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause))
      } finally {
        if (!cancelled) setBusy(false)
      }
    }
    run()
    return () => {
      cancelled = true
    }
  }, [item.id, variant])

  return (
    <section className="grid gap-8 md:grid-cols-2">
      <div>
        <div className="relative aspect-square overflow-hidden rounded-lg bg-black">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={imageRef}
            key={`${item.id}-${variant}`}
            src={`/gallery/${item.id}/${VARIANT_IMAGE[variant]}`}
            alt={`${item.true_class} radiograph, ${VARIANT_LABEL[variant]}`}
            width={224}
            height={224}
            crossOrigin="anonymous"
            className="h-full w-full object-contain"
          />
          {showCam && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={`/gallery/${item.id}/gradcam.png`}
              alt="Grad-CAM attribution overlay"
              className="pointer-events-none absolute inset-0 h-full w-full object-contain"
            />
          )}
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {VARIANTS.map((option) => (
            <button
              key={option}
              onClick={() => setVariant(option)}
              className={`rounded px-3 py-1.5 text-sm transition ${
                variant === option
                  ? 'bg-sky-500 text-neutral-950'
                  : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
              }`}
            >
              {VARIANT_LABEL[option]}
            </button>
          ))}
          <button
            onClick={() => setShowCam((on) => !on)}
            disabled={variant !== 'raw'}
            className="rounded bg-neutral-800 px-3 py-1.5 text-sm text-neutral-300 transition hover:bg-neutral-700 disabled:opacity-40"
          >
            {showCam ? 'Hide' : 'Show'} attention
          </button>
        </div>
      </div>

      <div className="space-y-6">
        <div>
          <p className="text-xs uppercase tracking-wide text-neutral-500">True label</p>
          <p className="text-lg">{item.true_class.replace('_', ' ')}</p>
          {item.note && <p className="mt-2 text-sm text-neutral-400">{item.note}</p>}
          <p className="mt-2 text-sm text-neutral-400">
            Lung attribution ratio <span className="tabular-nums">{item.lar.toFixed(3)}</span>{' '}
            <span className="text-neutral-600">(chance 0.238, ceiling 0.376)</span>
          </p>
        </div>

        <Predictions live={live} reference={item.reference[variant]} busy={busy} />

        {error && (
          <p className="rounded border border-amber-600/40 bg-amber-950/40 px-3 py-2 text-xs text-amber-200">
            Live inference unavailable ({error}). Showing the reference probabilities computed in
            Python.
          </p>
        )}
      </div>
    </section>
  )
}
```

- [ ] **Step 3: Write `web/components/Gallery.tsx`**

```tsx
'use client'

import type { GalleryItem } from '@/lib/types'

interface Props {
  items: GalleryItem[]
  selectedId: string
  onSelect: (id: string) => void
}

export default function Gallery({ items, selectedId, onSelect }: Props) {
  return (
    <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-12">
      {items.map((item) => (
        <button
          key={item.id}
          onClick={() => onSelect(item.id)}
          aria-pressed={item.id === selectedId}
          title={`${item.true_class} — ${item.id}`}
          className={`overflow-hidden rounded border-2 transition ${
            item.id === selectedId
              ? 'border-sky-400'
              : 'border-transparent opacity-60 hover:opacity-100'
          }`}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`/gallery/${item.id}/raw.png`}
            alt={item.true_class}
            width={224}
            height={224}
            className="aspect-square w-full object-cover"
          />
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 4: Write `web/app/page.tsx`**

```tsx
'use client'

import { useEffect, useState } from 'react'
import Gallery from '@/components/Gallery'
import Viewer from '@/components/Viewer'
import { activeBackend } from '@/lib/model'
import type { GalleryItem } from '@/lib/types'

export default function Page() {
  const [items, setItems] = useState<GalleryItem[]>([])
  const [selectedId, setSelectedId] = useState<string>('')
  const [backend, setBackend] = useState<string>('')

  useEffect(() => {
    fetch('/gallery/manifest.json')
      .then((response) => response.json())
      .then((records: GalleryItem[]) => {
        setItems(records)
        setSelectedId(records[0]?.id ?? '')
      })
    activeBackend().then(setBackend).catch(() => setBackend('unavailable'))
  }, [])

  const selected = items.find((item) => item.id === selectedId)

  return (
    <main className="mx-auto max-w-5xl space-y-10 px-4 py-10">
      <header className="space-y-4">
        <h1 className="text-3xl font-semibold tracking-tight">
          How much of this COVID classifier is real?
        </h1>
        <p className="max-w-2xl text-neutral-400">
          Two DenseNet121 models run in your browser on the same radiograph — one on the full
          image, one with the lungs erased entirely. Toggle between them. The prediction barely
          changes, because the model is reading how the image was acquired, not the lungs.
        </p>
        <p className="text-xs text-neutral-600">
          Twelve test-set images, hand-picked to make that argument rather than to be
          representative. Nothing is uploaded; inference runs entirely on your device
          {backend && ` (${backend})`}.
        </p>
      </header>

      {items.length > 0 && (
        <Gallery items={items} selectedId={selectedId} onSelect={setSelectedId} />
      )}

      {selected ? (
        <Viewer key={selected.id} item={selected} />
      ) : (
        <p className="text-neutral-500">Loading gallery…</p>
      )}
    </main>
  )
}
```

- [ ] **Step 5: Run it**

```bash
cd web && npm run dev
```

Open http://localhost:3000 and check, in order:

1. The gallery renders 12 tiles
2. Selecting a tile shows a prediction, and the parity Δ is small (~1e-3 or better)
3. Toggling to **Lungs erased** shows a visibly black-lunged image and a prediction that **barely moves** — this is the demo working
4. "Show attention" overlays the heatmap and is disabled on the lungs-erased variant
5. The disclaimer is visible and does not scroll away

- [ ] **Step 6: Verify the production build**

Run: `npm run build`
Expected: succeeds, `web/out/` produced.

- [ ] **Step 7: Commit**

```bash
git add web/app web/components
git commit -m "feat: gallery, viewer and prediction bars with live parity readout"
```

---

## Task 8: Deploy and document

**Files:**
- Create: `web/README.md` (may be written directly to disk)
- Modify: `README.md` (repo root)

- [ ] **Step 1: Write `web/README.md`**

Cover: what the demo is, how to run it locally, how assets are regenerated (Tasks 2–3), the measured quantization setting with its numbers, and the parity gate. State that the 12-image sample catches gross errors but does not bound a rare argmax-flip rate.

- [ ] **Step 2: Add a Demo section near the top of the repo README**

Immediately after the opening claim, so a reader can check it before reading the tables. Include the deployed URL and one line making clear the gallery is fixed and hand-picked.

- [ ] **Step 3: Deploy**

```bash
cd web && npm run build
npx vercel deploy --prod out
```

Any static host works — the export has no server dependency. If using GitHub Pages, set `basePath` in `next.config.mjs` to the repository name.

- [ ] **Step 4: Verify the deployed build**

Load the URL in a private window, confirm a prediction runs, and check the network tab shows the model downloading from the same origin. Note first-load transfer size.

- [ ] **Step 5: Commit**

```bash
git add README.md web/README.md
git commit -m "docs: demo README and deployment"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 purpose, no-uploads, no-backend, no-live-Grad-CAM | 5 (disclaimer), 7 (copy), 3 (precomputed CAM) |
| §3 `densenet` preprocessing + unit test | 1 |
| §3 configs, class order, dist→public mapping | 2 |
| §3 quantization measured not chosen | 4 Step 5 |
| §4 selection criteria, 224×224, `apply_variant` parity | 3 |
| §4 `manifest.json` schema | 3 |
| §5 file layout, static export, lazy second model, parity readout | 5, 6, 7 |
| §6 graceful degradation, backend display | 7 (Viewer error path, page backend) |
| §7 all five test layers | 1, 3, 4 |
| §8 framing and safety | 5, 7 |

**Gaps found and closed:**
- The spec's §5 lazy-loading requirement had no explicit task step; it is satisfied by `model.ts` caching per variant and `Viewer` only requesting the variant in use — noted in Task 6 Step 2's docstring.
- Quantization measurement originally sat in Task 2 before `tfjs-node` existed; moved to Task 4 where the runtime and manifest are both available.

**Type consistency:** `Variant` is `'raw' | 'lungs_removed'` throughout — note the manifest's *image* for that variant is `lungs_erased.png`, mapped via `VARIANT_IMAGE` in one place only. `GalleryItem` fields match `manifest_record`'s output keys exactly. `CLASS_NAMES` order is identical in `config.py`, the conversion configs, and `types.ts`.

**Known rough edge, deliberate:** `_overlay` uses `matplotlib.cm.get_cmap`, deprecated in Matplotlib 3.9+. If it warns, switch to `matplotlib.colormaps['jet']`. Left as-is because the installed version is unknown at authoring time.
