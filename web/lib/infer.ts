import * as tf from '@tensorflow/tfjs'
import { loadModel } from './model'
import { VARIANT_IMAGE } from './types'
import type { Variant } from './types'

export const INPUT_SIZE = 224

/**
 * RGB. Passed explicitly rather than relying on the default.
 *
 * `tf.browser.fromPixels` defaults to `numChannels = 3` and drops alpha, which
 * matches scripts/verify-parity.mjs stripping the 4th byte of every pixel. The
 * default is stated here so a future tfjs change cannot move it underneath us.
 */
const CHANNELS = 3

/**
 * Wait until the element actually has pixels.
 *
 * `fromPixels` on an <img> that has not finished decoding yields garbage or
 * throws, and garbage would surface as confident wrong probabilities rather
 * than as an error.
 */
async function ensureDecoded(image: HTMLImageElement): Promise<void> {
  const src = image.currentSrc || image.src

  // decode() covers both "not loaded yet" and "loaded but not yet decoded",
  // and rejects (EncodingError) on a corrupt asset instead of half-succeeding.
  if (typeof image.decode === 'function') {
    try {
      await image.decode()
    } catch (cause) {
      throw new Error(`Failed to decode image: ${src}`, { cause })
    }
    return
  }

  // Fallback for engines without HTMLImageElement.decode.
  //
  // `complete` alone is not enough — it is also true after a FAILED load, and
  // true when there is no src at all. In both of those states the load is
  // already over, so neither `load` nor `error` can ever fire again: attaching
  // listeners would leave the promise pending forever. Reject instead of
  // hanging.
  if (image.complete) {
    if (image.naturalWidth > 0) return
    throw new Error(
      `Failed to load image: ${src || '(element has no src)'} ` +
        '(load already settled with no pixels; no further load/error event can fire)',
    )
  }
  if (!src) {
    throw new Error('Image element has no src; nothing to decode')
  }

  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      image.removeEventListener('load', onLoad)
      image.removeEventListener('error', onError)
    }
    const onLoad = () => {
      cleanup()
      resolve()
    }
    const onError = () => {
      cleanup()
      reject(new Error(`Failed to load image: ${src}`))
    }
    image.addEventListener('load', onLoad)
    image.addEventListener('error', onError)
  })
}

/**
 * Copy the image into an offscreen canvas sized from its INTRINSIC dimensions.
 *
 * This exists because `tf.browser.fromPixels(imgEl)` reads `imgEl.width` and
 * `imgEl.height`, which for a rendered <img> are the LAYOUT size, not
 * `naturalWidth`/`naturalHeight` (tfjs-core/dist/ops/browser.js, and the WebGL
 * FromPixels kernel, both do `[pixels.width, pixels.height]`; the WebGL kernel
 * then rescales with `drawImage(pixels, 0, 0, width, height)`). Style a gallery
 * image at anything other than 224 CSS pixels and the tensor silently becomes
 * the wrong shape, resampled by the browser's own kernel. No exception is
 * thrown; the predictions just drift. That is precisely the failure mode this
 * project exists to expose.
 *
 * The fix works because of a difference between the two elements' IDL
 * attributes, NOT because this canvas happens to stay out of the document:
 *
 *   - `HTMLImageElement.width` reflects the image's RENDERED size while it is
 *     being rendered (falling back to the intrinsic size only when it is not).
 *     CSS therefore changes it.
 *   - `HTMLCanvasElement.width` reflects the `width` content attribute — the
 *     BITMAP's dimensions. CSS never changes it; a canvas scaled by CSS, or
 *     inserted into the document, would report exactly the same value.
 *
 * So routing through a canvas whose bitmap we size from `naturalWidth`/
 * `naturalHeight` makes the value `fromPixels` reads immune to styling,
 * wherever that canvas lives. Three-argument drawImage blits at the source's
 * intrinsic size, 1:1, with no resampling.
 */
function drawToOffscreenCanvas(image: HTMLImageElement): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = image.naturalWidth
  canvas.height = image.naturalHeight

  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (context === null) {
    throw new Error('2D canvas context unavailable; cannot read image pixels')
  }
  context.drawImage(image, 0, 0)
  return canvas
}

/**
 * Assert the element is actually showing the variant it is being scored as.
 *
 * `<img>` elements are reused: Task 7 flips one element's `src` between
 * raw.png and lungs_erased.png as the variant toggles. Both are 224x224, so
 * neither the size check nor the tensor shape check can tell them apart — an
 * element/variant mismatch would be scored and reported as the wrong variant,
 * silently, and compared against the wrong `reference` row.
 */
function assertSourceMatchesVariant(image: HTMLImageElement, variant: Variant): void {
  const src = image.currentSrc || image.src
  const expected = VARIANT_IMAGE[variant]
  const pathname = src.split(/[?#]/)[0]
  if (!pathname.endsWith(`/${expected}`)) {
    throw new Error(
      `Element/variant mismatch: scoring as '${variant}' (expects ${expected}) ` +
        `but the element is showing ${src}. Refusing to report one variant's ` +
        "pixels under the other's name.",
    )
  }
}

/**
 * Probabilities for one gallery image element.
 *
 * The image is fed as raw 0-255 float pixels with no normalisation and no
 * resize, matching scripts/verify-parity.mjs. Preprocessing lives inside the
 * exported graph, and the asset is already exactly 224x224 — resampling here
 * would risk a kernel mismatch with the TensorFlow bilinear used in training,
 * which fails silently. Hence the assertions below rather than a
 * `resizeBilinear` that would paper over the problem.
 *
 * ORDERING IS LOAD-BEARING. Everything between `ensureDecoded` and the pixel
 * snapshot is synchronous: validate the element, then copy its pixels, and only
 * then await anything. `loadModel` can take several seconds on a cold cache
 * (27 MB of weights), and the caller's <img> is live the whole time — a variant
 * toggle mid-flight would repoint it at the other PNG. Reading pixels after
 * that await would feed one variant's image to the other variant's model and
 * report it under the wrong name, with both size and shape assertions passing.
 */
export async function predict(image: HTMLImageElement, variant: Variant): Promise<number[]> {
  await ensureDecoded(image)

  // --- synchronous section: no `await` until the pixels are copied ---
  const { naturalWidth, naturalHeight } = image
  if (naturalWidth !== INPUT_SIZE || naturalHeight !== INPUT_SIZE) {
    throw new Error(
      `Gallery asset must be exactly ${INPUT_SIZE}x${INPUT_SIZE}, got ` +
        `${naturalWidth}x${naturalHeight} (${image.currentSrc || image.src}). ` +
        'The asset is wrong; do not resize here — the browser\'s resampling ' +
        'kernel differs from the TensorFlow bilinear used in training and the ' +
        'resulting drift is silent.',
    )
  }
  assertSourceMatchesVariant(image, variant)
  // The canvas is a private snapshot. Once taken, later mutations of `image`
  // cannot affect this prediction.
  const canvas = drawToOffscreenCanvas(image)
  // --- end synchronous section ---

  const model = await loadModel(variant)

  const output = tf.tidy(() => {
    const input = tf.browser.fromPixels(canvas, CHANNELS).toFloat().expandDims(0)

    // Belt and braces: whatever the source turned out to be, the tensor that
    // reaches the graph must have the training shape.
    const shape = input.shape
    const expected = [1, INPUT_SIZE, INPUT_SIZE, CHANNELS]
    if (shape.length !== expected.length || shape.some((d, i) => d !== expected[i])) {
      throw new Error(
        `Expected input tensor [${expected}], got [${shape}]. ` +
          'Refusing to predict on a mis-shaped input.',
      )
    }

    const result = model.predict(input)
    if (!(result instanceof tf.Tensor)) {
      throw new Error('Expected the graph to return a single output tensor.')
    }
    return result
  })

  try {
    return Array.from(await output.data())
  } finally {
    output.dispose()
  }
}

/** Largest absolute per-class difference between live and reference output. */
export function maxDelta(live: number[], reference: number[]): number {
  return Math.max(...live.map((p, i) => Math.abs(p - reference[i])))
}

export function argmax(values: number[]): number {
  return values.indexOf(Math.max(...values))
}
