# Task 4 report: end-to-end parity verification and the quantization decision

Branch `feat/browser-demo`, working directory `/home/daniel/Documents/personal/covid-xray-detection`.
Commits:

- `covid-xray-detection` `3d5457e` — `fix: force float32 precision when generating gallery reference probabilities`
- `covid-xray-detection` `ee32990` — `feat: end-to-end TF.js parity gate and quantization measurement`
- `to_tf_js_convertion` (branch `feat/covid-cxr-model`) `1433cbb` — `chore: reconvert covid configs after quantization sweep, ship float32`

**Shipped: `quantization: none` (float32), max abs probability error 2.44e-6, 54 MB total for both models, zero argmax flips.** float16 (28 MB) and uint8 (15 MB) were both measured and rejected — details below.

A real bug was found and fixed along the way: the gallery's Keras reference probabilities
(`web/public/gallery/manifest.json`) were computed under a lossy `mixed_float16` precision
path that doesn't match the pure-float32 TF.js graph actually shipped to the browser. This is
not a Task 4 scope change so much as Task 4 doing its job — "this is the test that matters"
caught a real discrepancy the earlier synthetic-input parity check (Task 2, 5.96e-8/1.79e-7)
could not see.

## Scaffolding around the existing `web/public`

`create-next-app` refuses to run in a non-empty directory, and `web/public/` already held ~56
MB of committed gallery assets plus untracked models. Scaffolded into a scratch directory
instead:

```
npx create-next-app@latest web-scaffold --typescript --tailwind --eslint --app \
  --src-dir=false --import-alias="@/*" --no-turbopack --use-npm
```

(run from the scratchpad, not the repo), then merged it into `web/` with `rsync -a` (no
`--delete`), which only adds files present in the source and never removes anything already in
the destination:

```
rsync -a --exclude='.git' <scratch>/web-scaffold/ web/
```

Verified before and after with `md5sum` over every file under `web/public/gallery` and
`web/public/models`: identical hash lists both times. `git status --short web/public` after the
merge showed only the five stock Next.js placeholder SVGs and `models/` as new — no
modifications, no deletions, under `gallery/` or `models/`. The only manual cleanup was fixing
`package.json`'s `"name"` field (the scaffold directory was named `web-scaffold`, so that's what
`create-next-app` wrote) back to `"web"`.

Dependencies added per the brief: `@tensorflow/tfjs` (runtime), `@tensorflow/tfjs-node` and
`pngjs` (dev, used only by the Node-side verification script). `@tensorflow/tfjs-node`'s native
addon (`tfjs_binding.node`, linked against `libtensorflow.so.2.9.1`) built successfully despite
an `npm warn allow-scripts` notice — that warning is about *other* packages' install scripts
being skipped (`core-js`, `unrs-resolver`), not this one; confirmed by checking
`node_modules/@tensorflow/tfjs-node/{deps,lib}` for the actual compiled artifacts.

## A detour: the float32 gate failed, and shouldn't have

Running `npm run verify-parity` against the as-shipped float32 models (before touching
quantization at all) did **not** pass cleanly:

```
OVER TOLERANCE  raw  COVID-1001: 5.27e-3
OVER TOLERANCE  raw  Viral_Pneumonia-1283: 3.38e-3
raw: checked 12 images
OVER TOLERANCE  lungs_removed  Lung_Opacity-5949: 2.44e-3
lungs_removed: checked 12 images
max abs probability error: 5.27e-3  (atol 0.002)
argmax flips: 0
FAIL — do not widen ATOL to make this pass.
```

Per the brief: "Never widen ATOL to make a failure pass. It exists to catch exactly this." So I
investigated instead of touching the constant. Systematic-debugging summary (full chain in the
commit message of `3d5457e`):

1. Confirmed the manifest reference for `COVID-1001` exactly matches Keras (`0.0` diff) when
   Keras is fed the *actual* PNG-round-tripped pixels — ruling out an 8-bit-quantization
   mismatch between the saved PNG and the reference computation. The reference generation logic
   itself was internally consistent.
2. Checked what `check_parity` (the Task-2-era 5.96e-8 measurement) actually compares: Keras
   `model.predict(...)` against `infer.predict(...)` — but by the time `check_parity` runs in
   `convert.py`, `model`'s layers have *already* been mutated to float32 by
   `build_inference_model`'s `_force_float32` call. That check can never see a dtype-policy
   problem — its own docstring says so.
3. Loaded `run1_raw_final.keras` fresh and enumerated layer `dtype_policy`: 428 layers/sublayers
   report `mixed_float16`, the rest `float32`. The checkpoint genuinely carries mixed precision.
4. Reproduced the exact discrepancy in isolation: loading the checkpoint, forcing every layer to
   `float32` **before the first `predict()` call**, and predicting on `COVID-1001`'s real pixels
   gives `[0.18562, 0.80132, ...]`; predicting on the same pixels *without* forcing float32
   (what `build_gallery.py` actually did) gives `[0.19093, 0.79612, ...]` — a 5.3e-3 gap, sign
   and magnitude matching the observed gate failure almost exactly. (Reassigning `dtype_policy`
   *after* a layer has already been built/called has no effect — Keras doesn't retroactively
   retrace already-built ops — which is why the order of operations in `build_inference_model`
   matters and why a naïve in-place fix attempt initially looked like a no-op.)
5. Ruled out the affine "densenet" preprocessing approximation as the cause: it matches
   `keras.applications.densenet.preprocess_input` to 3.6e-7 even on real image pixels, nowhere
   near enough to explain a 5e-3 probability shift.

Root cause: `scripts/build_gallery.py` loaded the checkpoints and called `.predict()` directly,
so the manifest's reference probabilities reflect the checkpoint's as-saved `mixed_float16`
compute path (float32 weights, float16 activations on ~428 of the DenseNet121's layers). The
TF.js export pipeline (`to_tf_js_convertion/export.py`) deliberately forces every layer to
`float32` before tracing — its own docstring explains why: "tfjs has no float16 tensor dtype
... measured at 3.7e-2 in probability on animal_classifier.keras, enough to flip the top class."
The manifest was the side that was wrong, not the exported graph.

**Fix** (`scripts/build_gallery.py`, commit `3d5457e`): added a local `_force_float32` helper
(mirroring `export.py`'s), called on both loaded models immediately after
`keras.models.load_model`, before `split_feature_and_head`, `.predict()`, or Grad-CAM. This
makes the reference reflect the same precision the browser will actually run.

**Blast radius, checked before committing:**
- `raw.png` / `lungs_erased.png` are untouched (byte-identical) — they come from PIL/mask
  arithmetic, never touch the model.
- Only the 12 `gradcam.png` overlays and `manifest.json` changed.
- Every probability shift is small (max observed 5.3e-3, most far smaller) and **zero argmax
  predictions changed** across all 12 images × 2 variants.
- LAR shifts are all ≤0.0011 — well inside the "~" rounding already used in the manifest's
  hand-written notes, so no narrative claim in `SELECTION` became inaccurate.
- Full test suite: `.venv/bin/python -m pytest -q` → **89 passed** in 452s, no regressions.

With the fix, `npm run verify-parity` against float32 now reports `max abs probability error:
2.44e-6 (atol 0.002)`, `argmax flips: 0`, `PASS` — reproducibly (measured twice, identical
number both times, once mid-sweep and once on the final re-shipped artifact).

I judged this in-scope for Task 4 rather than a bug to merely flag, because Task 4's whole
purpose is exactly this: "This runs before any UI exists... A preprocessing mismatch produces
confident wrong answers and never an error... this is the test that matters." A failure the gate
correctly caught, left uninvestigated, would have meant either shipping with a permanently-red
gate or (worse) quietly loosening `ATOL` to paper over a real bug — the one thing explicitly
forbidden.

## Quantization measurements

All three settings converted via `venv/bin/python convert.py configs/<name>.json` in
`to_tf_js_convertion` (Python 3.12, per the ambiguity-resolution note), copied into
`web/public/models/{raw,lungs_removed}/` (directories fully cleared before each copy — `cp -r`
alone left stale shard files from the previous setting behind the first time, since it doesn't
delete files absent from the source; caught this before drawing any conclusions from it).

### `none` (float32) — shipped

```
raw: checked 12 images
lungs_removed: checked 12 images
max abs probability error: 2.44e-6  (atol 0.002)
argmax flips: 0

PASS
```

Size: `raw` 27 MB + `lungs_removed` 27 MB = **54 MB total**.

### `float16`

```
OVER TOLERANCE  raw  COVID-1001: 3.32e-3
OVER TOLERANCE  raw  COVID-1264: 4.75e-3
OVER TOLERANCE  raw  Viral_Pneumonia-1283: 3.40e-3
raw: checked 12 images
OVER TOLERANCE  lungs_removed  COVID-1001: 4.46e-3
lungs_removed: checked 12 images
max abs probability error: 4.75e-3  (atol 0.002)
argmax flips: 0

FAIL — do not widen ATOL to make this pass. It exists to catch exactly this.
```

Size: `raw` 14 MB + `lungs_removed` 14 MB = **28 MB total**. Zero argmax flips, but the max
error is more than double `ATOL` and the script itself reports `FAIL`.

### `uint8`

```
OVER TOLERANCE  raw  COVID-1001: 1.63e-1
OVER TOLERANCE  raw  COVID-2038: 2.16e-2
OVER TOLERANCE  raw  COVID-1264: 1.53e-1
OVER TOLERANCE  raw  Lung_Opacity-449: 2.76e-1
OVER TOLERANCE  raw  Lung_Opacity-4106: 3.08e-3
OVER TOLERANCE  raw  Lung_Opacity-5949: 5.16e-2
OVER TOLERANCE  raw  Normal-9497: 2.65e-2
OVER TOLERANCE  raw  Normal-832: 4.30e-3
OVER TOLERANCE  raw  Normal-3209: 1.56e-1
ARGMAX FLIP  raw  Viral_Pneumonia-1283: 2 -> 3
OVER TOLERANCE  raw  Viral_Pneumonia-1283: 4.84e-1
OVER TOLERANCE  raw  Viral_Pneumonia-668: 5.97e-2
OVER TOLERANCE  raw  Viral_Pneumonia-422: 1.21e-2
raw: checked 12 images
OVER TOLERANCE  lungs_removed  COVID-1001: 1.71e-1
OVER TOLERANCE  lungs_removed  COVID-2038: 5.71e-2
ARGMAX FLIP  lungs_removed  COVID-1264: 0 -> 2
OVER TOLERANCE  lungs_removed  COVID-1264: 9.28e-1
ARGMAX FLIP  lungs_removed  Lung_Opacity-449: 2 -> 1
OVER TOLERANCE  lungs_removed  Lung_Opacity-449: 3.31e-1
OVER TOLERANCE  lungs_removed  Lung_Opacity-4106: 7.36e-2
ARGMAX FLIP  lungs_removed  Lung_Opacity-5949: 1 -> 2
OVER TOLERANCE  lungs_removed  Lung_Opacity-5949: 1.69e-1
OVER TOLERANCE  lungs_removed  Normal-9497: 3.71e-2
OVER TOLERANCE  lungs_removed  Normal-832: 3.88e-2
OVER TOLERANCE  lungs_removed  Normal-3209: 1.02e-1
OVER TOLERANCE  lungs_removed  Viral_Pneumonia-1283: 3.08e-1
OVER TOLERANCE  lungs_removed  Viral_Pneumonia-668: 4.16e-2
OVER TOLERANCE  lungs_removed  Viral_Pneumonia-422: 7.63e-2
lungs_removed: checked 12 images
max abs probability error: 9.28e-1  (atol 0.002)
argmax flips: 4

FAIL — do not widen ATOL to make this pass. It exists to catch exactly this.
```

Size: `raw` 7.1 MB + `lungs_removed` 7.2 MB = **15 MB total**. Unusable outright — 4 argmax
flips (one in `raw`, three in `lungs_removed`), errors up to 0.93 in probability.

## The shipping decision

| setting | size | max abs error | argmax flips | gate |
|---|---|---|---|---|
| none (float32) | 54 MB | 2.44e-6 | 0 | **PASS** |
| float16 | 28 MB | 4.75e-3 | 0 | FAIL (over tolerance) |
| uint8 | 15 MB | 9.28e-1 | 4 | FAIL (flips + over tolerance) |

`uint8` is unambiguous — 4 argmax flips rules it out under any reading of the decision rule.

`float16` is the genuinely hard case: the brief's decision rule, read most literally ("ship the
smallest quantization with zero argmax flips"), would pick it — every top-1 prediction still
matches Keras, only the confidence number moves. But the gate's own tolerance, calibrated in the
brief specifically with float16 lossiness in mind ("`ATOL = 2e-3` ... float16 quantization is
lossy"), is exceeded by more than 2×. Shipping it means `npm run verify-parity` — the exact tool
this task built to gate what ships — reports `FAIL` against the artifact actually deployed,
permanently, unless something changes. That contradicts Step 4's own stated bar ("Expected:
argmax flips 0, max error well under 2e-3, `PASS`") and the task's framing of this gate as
authoritative pre-UI proof of correctness. I chose not to ship something the gate itself flags
as broken, and did not touch `ATOL` to make it pass either.

**This was a deliberate deviation from the brief's literal rule, not a stricter reading of it.**
The brief's own wording (Step 5) is: "Ship the smallest quantization with **zero argmax
flips**." Read literally, that rule selects `float16` — 28 MB, zero argmax flips, half the size
of what actually shipped. I did not ship that. **Shipped `none` (float32)** instead — 54 MB —
because `float16` leaves `npm run verify-parity`, the gate this task exists to build as the
authoritative pre-UI proof of correctness, permanently `FAIL` against the artifact actually
deployed. Shipping an artifact that leaves the project's own parity gate permanently red defeats
the purpose of having the gate at all. The line "a correct 56 MB demo beats a wrong 14 MB one" is
not from the brief — it is from the controller's dispatch instructions for this task, and is
recorded here (not as brief text) because it is what I weighed against the brief's literal
wording when making the call. I flagged this explicitly rather than resolving it silently,
since a stricter/looser reading of the decision rule than mine is plausible; the full float16
numbers above are there to let that call be revisited if the flips-only reading was actually
intended.

**Adjudicated:** this has since been reviewed and `float32` (`none`) is confirmed as the correct
call. No artifact changes result from this fix — the correction here is to this report's
attribution only, so the record is auditable by someone who only has the repo files.

Configs in `to_tf_js_convertion` were reconverted one final time at `quantization: none` so
`dist/` and the committed config agree with what's shipped (commit `1433cbb` in that repo,
`feat/covid-cxr-model` branch) — the only diff is JSON re-serialization formatting from the
sweep script, `quantization` was already `"none"` before the sweep and is `"none"` after.

## Final state

- `web/public/models/{raw,lungs_removed}` hold the float32 GraphModels, 54 MB total, committed
  with `git add -f` (previously untracked).
- `web/public/gallery` is intact: `raw.png`/`lungs_erased.png` byte-identical throughout this
  task; `gradcam.png` (×12) and `manifest.json` changed only via the float32-precision fix,
  committed separately with full justification (`3d5457e`).
- `npm run verify-parity` (from `web/`) passes: `max abs probability error: 2.44e-6 (atol
  0.002)`, `argmax flips: 0`, `PASS`.
- Full Python test suite: 89 passed, 0 failed.

## For whoever writes the README next

The brief's Step 5 asks for a note that the 12-image gallery is "enough to catch a gross
preprocessing error but not to bound a rare flip rate" — i.e. don't imply a stronger guarantee
than 12 samples support. Task 4's file list doesn't include the README, so I haven't written it,
but flagging it here so it isn't lost: whatever task documents the quantization decision should
carry that caveat forward, along with the float16 zero-flips-but-over-tolerance nuance above (it
is not simply "float16 is worse," it is "float16 keeps every label right but the gate cannot
tell you that without knowing to check flips separately from tolerance").

## Fix report: attribution correction and `_force_float32` regression test

Two review findings came back on this task after the above was written. Both are addressed here;
no artifact (models, gallery assets, manifest) changed as part of either fix.

### Fix 1: corrected attribution in "The shipping decision"

The sentence "a correct 56 MB demo beats a wrong 14 MB one" was originally attributed to "the
brief's own fallback language." It is not brief text — grep confirms it does not appear anywhere
in `task-4-brief.md` or any other file in this repo. It came from the controller's dispatch
instructions for this task, which are not a repo artifact and so cannot be checked by someone who
only has the repo files. That paragraph has been rewritten in place (see "The shipping decision"
above) to:

- quote the brief's actual literal wording (Step 5): "Ship the smallest quantization with **zero
  argmax flips**";
- state plainly that read literally, that rule selects `float16` (zero argmax flips, 28 MB, half
  the size of what shipped) — not `none`/float32;
- attribute "a correct 56 MB demo beats a wrong 14 MB one" to controller dispatch instructions,
  not the brief;
- state without softening that shipping `none` instead of `float16` was a deliberate deviation
  from the brief's literal rule, made because shipping an artifact that leaves the project's own
  parity gate (`npm run verify-parity`) permanently `FAIL` against what's actually deployed
  defeats the purpose of having the gate;
- state that the deviation was surfaced explicitly for human adjudication in this report rather
  than made silently;
- record that it has since been adjudicated: `float32` (`none`) is confirmed correct. No artifact
  changes result — this is a correction to the report text only.

### Fix 2: regression test for `_force_float32`

The bug described above under "A detour: the float32 gate failed, and shouldn't have" — DenseNet121
checkpoints carrying a `mixed_float16` dtype policy that made the manifest's Keras reference
lossy relative to the float32 TF.js export — had no test in the 89-test suite covering the fix
that resolved it. The conversion workspace's `to_tf_js_convertion/export.py` carries a
near-identical `_force_float32` helper for a near-identical prior incident on a different model
(its docstring cites a 3.7e-2 probability shift, enough to flip a top class, on
`animal_classifier.keras`), so this is a known recurring failure class in this pipeline, not a
one-off.

Added `test_force_float32_recurses_into_nested_sublayers` to `tests/test_build_gallery.py`:

- builds a small synthetic model under a global `mixed_float16` policy: a two-layer
  `keras.Sequential` (`inner_backbone`) called as a sublayer inside an outer `keras.Model`,
  mirroring the real structure (the DenseNet121 backbone is itself a nested model inside the
  outer functional model) — a flat model would not exercise the recursion through
  `getattr(layer, "layers", [])` that `_force_float32` relies on, which is the actual failure
  mode this fix addresses;
- asserts the fixture actually produced `mixed_float16` on the nested sublayers before applying
  the fix, so the test cannot pass vacuously;
- applies `_force_float32` to each top-level layer of the outer model;
- recursively asserts (`_assert_recursively_float32`) that every layer and sublayer, at every
  nesting depth, reports a `"float32"` dtype policy;
- sets the global mixed-precision policy to `mixed_float16` at the start and restores the
  original policy (captured before mutating it) in a `finally` block, since
  `keras.mixed_precision.global_policy()` is process-wide state and leaking `mixed_float16` into
  later tests in the same session would corrupt them;
- does not load the real 64 MB checkpoints — the synthetic model is two small `Dense` layers deep,
  keeping the test fast (part of an 8s file run, not the ~7.5 minutes loading real checkpoints
  would cost).

**Confirmed the test actually detects the regression it's named for**, per the instruction not to
assume a test that passes either way is worse than no test:

1. Commented out the body of `_force_float32` in `scripts/build_gallery.py` (replaced with `pass`).
2. Ran `.venv/bin/python -m pytest -q tests/test_build_gallery.py -k force_float32` →
   **1 failed**. Failure: `AssertionError: 'input_layer' kept dtype policy 'mixed_float16'`
   (assert `'mixed_float16' == 'float32'`) — the test caught the disabled fix immediately, on the
   first layer checked.
3. Restored `_force_float32`'s body exactly (`git diff --stat scripts/build_gallery.py` empty,
   confirming a byte-identical restore).
4. Re-ran `.venv/bin/python -m pytest -q tests/test_build_gallery.py -k force_float32` →
   **1 passed**.
5. Ran the full file: `.venv/bin/python -m pytest -q tests/test_build_gallery.py` → **9 passed**.

### Full suite

`.venv/bin/python -m pytest -q` → **90 passed** (0:04:10), up from the 89 recorded above — the
one new addition is `test_force_float32_recurses_into_nested_sublayers`. No other test's outcome
changed.
