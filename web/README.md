# The browser demo

A static page that runs **two** DenseNet121 classifiers on the same chest radiograph, in your
browser, through TensorFlow.js: one trained on the full image, one trained with the lung fields
erased to black. You toggle between them and watch the prediction not move.

That is the whole point. This is not a demonstration that the model works — it is a demonstration
of [the repository's finding](../README.md), that a classifier scoring 0.852 macro-F1 on this
dataset is largely reading how the image was acquired rather than what is inside the chest. The
page leads with **agreement, not accuracy**: on 9 of the 12 shipped images, erasing the lungs
entirely leaves the top answer unchanged.

The twelve images are fixed, hand-picked from the held-out test set to make that argument, and
stated on the page to be exactly that. They are not a random sample and nothing here measures
accuracy. If a reader leaves thinking "impressive COVID detector", the page has failed.

---

## Running it locally

```bash
cd web
npm install
npm run dev          # http://localhost:3000
```

For the real thing — the static export that gets deployed:

```bash
npm run build        # emits web/out/
python3 -m http.server 8000 --directory out
```

**Serve `out/` over HTTP. Do not open `out/index.html` from `file://`.** Under `file://` the
origin is opaque, so `fetch('/gallery/manifest.json')` is blocked and `tf.loadGraphModel` cannot
read the weight shards. The page renders its "gallery could not be loaded" panel and no
prediction ever runs. Any static server will do; the one above needs nothing installed.

Everything the page needs is already committed — twelve images, two converted models, and the
reference probabilities. There is no build step for the assets.

---

## No uploads, by design

There is no file picker, no drag-and-drop, no network write of any kind. The twelve radiographs
ship with the page and both models execute on the visitor's own device.

That is a deliberate limit, not a missing feature. This project's own result is that the model
keys on acquisition artefacts — scanner, processing pipeline, positioning marker, source
repository — rather than on lung pathology. Inviting the public to feed it their own chest X-rays
would attach a confident four-class output to exactly the images whose provenance the model is
secretly classifying. There is no honest way to present that output, so it is not offered.

---

## Regenerating the assets

Both steps need the dataset and the training checkpoints, which are gitignored — see
[Reproducing](../README.md#reproducing) in the repository README.

**Gallery** — twelve images × (original, lungs-erased, Grad-CAM overlay), plus
`public/gallery/manifest.json` with the Keras reference probabilities the page compares its live
output against:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.build_gallery
```

Every PNG is written at exactly 224×224, which is a correctness requirement rather than a
convenience: if the browser has to resize, its bilinear kernel may differ from TensorFlow's and
the pixels reaching the model drift off the training distribution silently, with no error. The
selection list and the reasoning behind each pick live in
[`scripts/build_gallery.py`](../scripts/build_gallery.py).

`build_gallery` forces every layer to a float32 compute policy before predicting. The checkpoints
were trained under `mixed_float16`, and the as-saved policy truncates activations to float16 at
inference; the exported graphs cannot do that, because TF.js has no float16 tensor dtype. Left
unforced, the "Keras reference" in the manifest would describe a model that never ships.

**Models** — converted in a separate workspace (`to_tf_js_convertion`), which owns the Python 3.12
+ `tensorflowjs` toolchain this repository does not depend on. One JSON config per model:

```json
{
  "model": "models/covid_raw.keras",
  "input_shape": [224, 224, 3],
  "preprocessing": "densenet",
  "quantization": "none",
  "class_names": ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
}
```

`run1_raw_final.keras` → `public/models/raw/`, `run4_lungs_removed_final.keras` →
`public/models/lungs_removed/`. The converter emits a TF.js **GraphModel** and the
`densenet` preprocessing is baked into the graph, so the page hands it raw 0–255 RGB pixels and
does no arithmetic of its own — one fewer place for a preprocessing mismatch to hide.

---

## Why the models ship unquantized

All three `tensorflowjs_converter` quantization settings were converted and measured against the
Keras references, not reasoned about:

| `quantization` | size, both models | max abs probability error | argmax flips | parity gate |
|---|---|---|---|---|
| **`none` (float32)** | **54 MB** | **2.44e-6** | **0** | **PASS** |
| `float16` | 28 MB | 4.75e-3 | 0 | FAIL — over the 2e-3 tolerance |
| `uint8` | 15 MB | 9.28e-1 | 4 | FAIL — flips and tolerance |

`uint8` is unusable outright: four of the twenty-four predictions change class.

`float16` is the honest difficulty. The plan's decision rule was "ship the smallest quantization
with **zero argmax flips**", and read literally that selects `float16` — every top-1 answer
survives, at half the download. **`none` was shipped anyway.** float16's worst error, 4.75e-3,
is more than twice the 2e-3 tolerance that `npm run verify-parity` enforces, so shipping it would
have meant either a permanently red gate or widening `ATOL` to accommodate the thing the gate
exists to detect. A project whose entire argument is that a number was not checked carefully
enough does not get to relax its own check to save 26 MB.

That is a deliberate, human-adjudicated deviation from the written rule, recorded here rather
than quietly resolved. The cost is a 54 MB first load.

---

## The parity gate

```bash
cd web
npm run verify-parity
```

Loads both exported graphs under `tfjs-node` and runs all twelve gallery images through both,
comparing against the Keras reference probabilities in the manifest — 24 predictions. It fails on
any argmax flip, or on any probability differing by more than `ATOL = 2e-3`.

This is the test that matters, because a preprocessing mismatch between Keras and TF.js does not
raise: it produces confident, wrong, entirely plausible-looking answers. Nothing else in the
pipeline would catch it. **Do not widen `ATOL` to make it pass.**

**What it does not do:** twelve images is a smoke test. It catches gross breakage — a wrong
input scale, a channel-order mistake, a corrupted shard, a bad quantization — but twelve samples
place no meaningful bound on the rate of a *rare* argmax flip. "Zero flips in 24 predictions" is
consistent with a flip rate well above zero. Read it as evidence that the conversion is not
broken, not as evidence that the browser and Keras agree everywhere.

The page carries the same check live: each prediction shows the largest absolute difference
between what your browser just computed and the stored Keras reference.

---

## WebGL costs nothing measurable

TensorFlow.js picks a backend at load time — WebGL where available, falling back to CPU — and the
page names the one it got. Driven under Chromium 148 on the twelve images × two variants:

| backend | worst parity Δ |
|---|---|
| WebGL — `ANGLE (Intel, Mesa Intel(R) UHD Graphics 620 (KBL GT2), OpenGL ES 3.2)` | 2.9e-6 |
| WebGL — `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)), SwiftShader driver)` | 3.1e-6 |
| CPU — `tfjs-node`, what `npm run verify-parity` measures | 2.44e-6 |

Zero argmax flips on either driver. GPU float32 and CPU float32 land within a rounding error of
each other, three orders of magnitude inside the 2e-3 tolerance, so the parity numbers a visitor
sees are the same numbers the gate checks.

---

## Deploying

The export is fully static — no server, no API routes, no runtime configuration — so any static
host works.

```bash
cd web
npm run build
npx vercel deploy --prod out
```

**Nothing is published until someone runs that.** After it is published, put the URL in
[the repository README](../README.md#check-the-claim-in-a-browser), which currently carries a
`REPLACE-WITH-DEPLOY-URL` placeholder.

### Hosting under a sub-path

GitHub Pages serves project sites from `/<repo>/`. Set **one** environment variable at build time:

```bash
NEXT_PUBLIC_BASE_PATH=/covid-xray-detection npm run build
```

`next.config.ts` and [`lib/paths.ts`](lib/paths.ts) both read that variable through the same
`normalizeBasePath`, deliberately: Next's `basePath` prefixes what Next emits (the `/_next/…`
script tags), while `assetPath()` prefixes everything under `public/` (the manifest, the twelve
images, the weight shards), and Next rewrites neither of those for you. Miss the first half and
the page loads no JavaScript; miss the second and it loads but shows the manifest-error panel.
The variable is inlined into the client bundle at build time, so it cannot be changed after the
fact — rebuild to change it.

`normalizeBasePath` accepts a loose value — `covid-xray-detection`, `/covid-xray-detection/` and
`/` all normalise to something Next accepts, where the raw value would hard-error on a missing
leading slash or a trailing one. **The contract is a base *path*, not an origin.** A value lacking
a leading `/` gets one prepended, so `https://cdn.example.com` silently becomes
`/https://cdn.example.com`. This is a known sharp edge, not a supported input; to serve the static
assets from a CDN origin you want Next's `assetPrefix`, which is a different lever and does not
cover `public/`.

### First load is heavy

| | |
|---|---|
| Model weights | **54 MB** — 27 MB per model, both fetched (the page runs both) |
| JavaScript for `/` | 1,686,296 bytes raw, ~447 KB gzipped |
| of which TensorFlow.js | roughly two thirds of the JavaScript |

The weights outweigh the bundle by more than thirty to one, so the bundle is not the thing to optimise
— and the 54 MB is the price of the decision above. Both models load because the demo's entire
claim is the comparison between them; deferring one until the toggle is pressed would trade the
first impression for a stall at the moment the argument lands.

---

## Layout

```
web/
├── app/            # single route; the page is a client component
├── components/     # Gallery, Viewer, Predictions, Disclaimer
├── lib/            # paths, model loading/backend, inference, types
├── scripts/        # verify-parity.mjs — the Keras/TF.js parity gate
└── public/
    ├── gallery/    # 12 images x 3 renderings + manifest.json
    └── models/     # raw/ and lungs_removed/ TF.js GraphModels
```
