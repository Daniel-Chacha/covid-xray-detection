# Browser-Only Demo Frontend — Design

**Date:** 2026-08-05
**Status:** Approved
**Scope:** A static Next.js page running two DenseNet121 models client-side over a fixed 12-image gallery.

---

## 1. Purpose

Make the project's central claim checkable rather than assertable.

[The audit](2026-07-30-covid-cxr-detection-design.md) established that erasing the lungs costs the classifier only 3% of its macro-F1, and that on the COVID vs. Lung Opacity control pair erasing the lungs does not measurably hurt it at all. (Corrected 2026-08-07: this section previously said the lungs-erased model scores marginally *higher*. The 0.0018 gap is roughly a third of the AUC's bootstrap CI half-width, so it supports no direction.) That is a number in a table. This demo lets a reader select a radiograph, toggle the lungs off, and watch the prediction hold — the same finding, arrived at by clicking.

The headline is therefore the **agreement** between the two models, not the accuracy of either.

### Constraints that shaped the design

- **No uploads.** Inviting members of the public to submit chest X-rays to a demo of a model this project has shown to be reading acquisition artefacts would be indefensible. The gallery is fixed and drawn from the test set.
- **No backend.** Inference runs in the browser via TensorFlow.js. Images never leave the device, hosting is static and free, and there is no cold start.
- **No live Grad-CAM.** `tf.loadGraphModel` produces a frozen graph, and TensorFlow.js cannot compute gradients through it. Heatmaps are precomputed in Python with the existing `covid_xray.gradcam` module and shipped as static overlays. This is acceptable precisely because the gallery is fixed.

### Non-goals

Lung Attribution Ratio on arbitrary images (would require shipping a segmentation model) · user uploads · `run2_masked` and `run3_probe8` (the probe is a scikit-learn pipeline, not convertible to TF.js) · retraining or fine-tuning in the browser.

---

## 2. Architecture

Three stages with clean boundaries. Each can be verified independently.

| Stage | Where | Produces |
|---|---|---|
| 1. Conversion | `to_tf_js_convertion` (Python 3.12) | TF.js GraphModels for both networks |
| 2. Gallery assets | `covid-xray-detection` (Python 3.13) | 12 items × 3 PNGs + `manifest.json` |
| 3. Application | `covid-xray-detection/web` (Next.js) | Static export |

---

## 3. Stage 1 — Conversion

The existing conversion workspace supports this model with a single configuration addition. `convert.py` needs no structural change: its `_force_float32` pass already handles the `mixed_float16` policy our checkpoints were saved under, which is the exact failure its README documents.

### The `densenet` preprocessing mode

`covid_xray` normalises with `keras.applications.densenet.preprocess_input`, which is torch-mode: divide by 255, then subtract per-channel ImageNet means and divide by per-channel standard deviations. Folded into a single affine transform:

```
scale  = 1 / (255 · std)
offset = −mean / std
```

with `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`, giving a new entry in `PREPROCESSING` in `config.py`:

```python
"densenet": {
    "scale":  [0.017125, 0.017507, 0.017429],
    "offset": [-2.117904, -2.035714, -1.804444],
},
```

**Verified 2026-08-05:** a per-channel `keras.layers.Rescaling` with these values reproduces `densenet.preprocess_input` to a maximum absolute difference of **4.8e-07**, inside `verify.py`'s `atol=1e-6`. A unit test asserting this belongs in the workspace's `tests/`, because the constants are opaque and a transposition would not otherwise be caught.

### Configs

Copy the two checkpoints into the workspace under stable names — `checkpoints/run1_raw_final.keras` becomes `models/covid_raw.keras`, and `checkpoints/run4_lungs_removed_final.keras` becomes `models/covid_lungs_removed.keras`. The workspace serves several projects, so `run1_raw` would be meaningless there.

Two configs, `configs/covid_raw.json` and `configs/covid_lungs_removed.json`, identical but for the `model` field:

```json
{
  "model": "models/covid_raw.keras",
  "input_shape": [224, 224, 3],
  "preprocessing": "densenet",
  "quantization": "none",
  "class_names": ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
}
```

Class order is the canonical order fixed in `covid_xray.config.CLASS_NAMES` and must not be re-derived.

### Quantization is measured, not chosen

The conversion README is explicit that quantization is a per-model judgement against a real validation set. This project has one: `reports/run1_raw_test_probs.npy` holds reference probabilities for all 3,142 test images, produced by the Keras model.

Convert at `none`, `float16` and `uint8`; for each, run the exported graph over the test set through `@tensorflow/tfjs-node` and report **maximum absolute probability error** and **the count of argmax flips**. Ship the smallest quantization with **zero argmax flips**.

Approximate sizes per model: `none` ≈ 28 MB, `float16` ≈ 14 MB, `uint8` ≈ 7 MB. Two models are shipped, so this decision is worth between 14 MB and 56 MB of download.

`convert.py` writes to `dist/covid_raw/` and `dist/covid_lungs_removed/`; these are copied to `web/public/models/raw/` and `web/public/models/lungs_removed/` respectively.

---

## 4. Stage 2 — Gallery assets

A script in `covid-xray-detection` writing into `web/public/gallery/`.

### Selection

Twelve test-set images, three per class, chosen by hand and recorded by ID so the set is reproducible. The curation must include:

- The COVID case at **LAR 0.04** whose Grad-CAM sits on the `D` positioning marker and neck tubing — the single most persuasive image in the project
- At least one high-confidence error from the failure gallery
- At least two cases where the raw and lungs-erased predictions agree closely, since that agreement *is* the finding

Hand-picking is an argument-making choice, not a fairness claim, and the page says so.

### Per-item assets

| File | Purpose |
|---|---|
| `raw.png` | input to the `run1_raw` model |
| `lungs_erased.png` | input to the `run4_lungs_removed` model |
| `gradcam.png` | precomputed `run1_raw` attribution overlay |

There is no Grad-CAM for the lungs-erased variant. One heatmap per item keeps the asset count down, and the raw model's attribution is the one the argument rests on.

**All three are written at exactly 224×224.** This is a correctness requirement, not a convenience. If the browser resizes, its bilinear implementation may differ from TensorFlow's and the pixels reaching the model drift from the training distribution — silently, with no error. Pre-resizing reduces the browser's work to `fromPixels → toFloat → expandDims`.

`lungs_erased.png` is generated by the same `apply_variant(image, mask, "lungs_removed")` path used in training, so the demo feeds the model exactly what it was trained on.

### `manifest.json`

One array of records. Values below are illustrative — the real ones come from the generation script:

```json
{
  "id": "COVID-1001",
  "true_class": "COVID",
  "note": "Grad-CAM sits on the D marker, not the lungs",
  "lar": 0.0439,
  "reference": {
    "raw":           [0.94, 0.03, 0.02, 0.01],
    "lungs_removed": [0.91, 0.05, 0.03, 0.01]
  }
}
```

`reference` holds the Keras probabilities. It serves two purposes: the parity readout in §5, and graceful degradation in §6.

---

## 5. Stage 3 — The application

```
web/
├── app/
│   ├── layout.tsx
│   ├── page.tsx
│   └── globals.css
├── components/
│   ├── Gallery.tsx        # tile grid, selection state
│   ├── Viewer.tsx         # selected image, variant toggle, Grad-CAM overlay
│   ├── Predictions.tsx    # probability bars + parity readout
│   └── Disclaimer.tsx     # persistent not-for-clinical-use banner
├── lib/
│   ├── model.ts           # load and cache both GraphModels
│   ├── infer.ts           # pixels -> probabilities
│   ├── paths.ts           # base-path-aware asset URLs
│   └── types.ts           # manifest record types
├── public/
│   ├── models/raw/…
│   ├── models/lungs_removed/…
│   └── gallery/…
├── scripts/verify-parity.mjs
└── next.config.ts         # output: 'export'
```

Each module has one job: `model.ts` turns a name into a loaded model, `infer.ts` turns an image element into probabilities, `Predictions.tsx` turns probabilities into bars. None needs to read another's internals.

**Static export** (`output: 'export'`) so the same build deploys to Vercel, Netlify or GitHub Pages without change.

### Data flow

`manifest.json` → `Gallery` → user selects an item → `Viewer` holds the active variant → `infer.ts` reads the corresponding `<img>` → `model.predict` → probabilities → `Predictions`.

### Lazy second model

`raw` loads on mount; `lungs_removed` loads the first time the toggle is used. This halves time-to-first-prediction, and a reader who never toggles never pays for the second model.

### Parity readout

`Predictions` shows, for the **currently selected variant**, the live TF.js probability beside the manifest's Python reference, with the delta.

The conversion README's central warning is that a preprocessing mismatch produces confident wrong answers and never an error. This puts that risk on screen rather than trusting it was avoided. It also demonstrates that inference is genuinely running client-side rather than reading a lookup table.

---

## 6. Failure handling

Because `manifest.json` already carries reference probabilities, **the page degrades to a static results browser**: if TensorFlow.js fails to load, or neither backend can be initialised, the gallery remains browsable and every number is still displayed, behind a banner explaining that live inference is unavailable.

The active TF.js backend is displayed, since it explains order-of-magnitude latency differences between visitors.

**Corrected 2026-08-07.** This section originally promised a WASM fallback. `@tensorflow/tfjs` bundles **CPU and WebGL only** — WASM lives in a separate `@tensorflow/tfjs-backend-wasm` package, which was deliberately not added. The real chain is **WebGL, falling back to CPU**, and total failure is covered by the static-results degradation above, so nothing is lost.

A model that fails to load is reported inline rather than silently yielding no prediction.

---

## 7. Testing

| Layer | Test |
|---|---|
| Preprocessing constants | Unit test in the conversion workspace asserting the `densenet` entry matches `preprocess_input` to 1e-6 |
| Wrapped graph | Existing `verify.py` — wrapped model vs. original, `atol=1e-6` |
| Exported artifacts | Existing `verify_tfjs.mjs` — real `model.json` + shards through `tfjs-node` |
| Gallery script | Manifest completeness, every referenced file exists at 224×224, probability rows sum to 1 |
| End-to-end parity | `web/scripts/verify-parity.mjs` — all 12 gallery images through both exported graphs, asserted against manifest references at `atol=2e-3`, matching the existing `verify_tfjs.mjs`, with **zero tolerance for argmax disagreement** |

The last is the highest-value test: it exercises the exact artifacts the browser will load, over the exact inputs it will feed them.

---

## 8. Framing and safety

- A persistent, non-dismissible **"Not a medical device. Not for clinical use."** banner.
- **No upload affordance anywhere** — no file input, no drag target, no paste handler.
- Copy presents the page as a shortcut demonstrator. The prominent number is the *agreement* between the two models; accuracy is secondary and contextualised.
- The hand-picked nature of the gallery is stated, not implied.

---

## 9. Risks

- **Two models is 14–56 MB of download** depending on the quantization measurement. If `uint8` flips no argmax on the test set, this is comfortable; if only `none` passes, first load is slow and the lazy second model matters more.
- **A frozen graph cannot be introspected in the browser**, so the Grad-CAM overlays are static images with no guarantee, visible to the user, that they correspond to the live model. The parity readout mitigates this indirectly by showing the live model agrees with the Python reference that produced the heatmap.
- **The demo makes a model that does not work look interactive and polished.** That is the point, but it inverts the usual relationship between production value and trustworthiness, and the copy has to carry that weight. If a reader leaves thinking "impressive COVID detector", the page has failed.

---

## 10. Working agreement

Implementation code is delivered in chat for manual copying, one complete file per block labelled with its path. README and requirements files may be written directly to disk.
